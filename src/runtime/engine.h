#pragma once

// Engine: the runtime facade used by the CLI and the HTTP server.
//
// Owns the loaded model, thread pool, backend, KV pool and scheduler, plus a
// dedicated scheduler thread that steps continuously while there is work and
// sleeps otherwise. Clients submit requests from any thread and consume a
// RequestStream of text deltas; nothing a client does (slow reads, stalled
// sockets) can block the scheduler:
//
//   client thread ──submit──► Scheduler (handoff queue) ◄──step── scheduler thread
//   client thread ◄─RequestStream (event queue)◄── callbacks ──┘
//
// Streams may outlive the Engine; after shutdown they end with kCancelled.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "chat_template/chat_template.h"
#include "common/status.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/thread_pool.h"
#include "scheduler/scheduler.h"

namespace engine {

struct EngineOptions {
  std::string model_path;
  int threads = 0;              // 0 = physical cores
  int64_t kv_tokens = 0;        // KV capacity in tokens; 0 = min(context, 16384)
  DType kv_dtype = DType::kF16;
  int32_t max_batch_tokens = 256;
  SchedulerConfig scheduler;
};

struct GenerateParams {
  int32_t max_tokens = 256;
  std::vector<std::string> stop;  // stop strings (never emitted)
  bool stop_at_eog = true;
  int32_t priority = 0;
  int64_t timeout_ms = 0;
};

enum class StreamFinish : uint8_t { kNone, kStop, kLength, kCancelled, kError };
std::string_view stream_finish_name(StreamFinish f);

struct StreamEvent {
  std::string text;  // delta (may be empty on the final event)
  bool done = false;
  StreamFinish finish = StreamFinish::kNone;
  Status error;
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
};

// Per-request event queue: the scheduler thread pushes (never blocks), one
// consumer pops.
class RequestStream {
 public:
  uint64_t id() const { return id_; }
  // Blocks up to `timeout` for the next event; false on timeout. After the
  // final (done) event has been returned, returns false immediately.
  bool next(StreamEvent& out, std::chrono::milliseconds timeout = std::chrono::milliseconds(60000));
  // Requests cancellation; the stream then ends with kCancelled.
  void cancel();
  bool finished() const;

  // Producer side (engine internal).
  void push(StreamEvent ev);
  void set_cancel_fn(std::function<void()> fn) { cancel_fn_ = std::move(fn); }
  void set_id(uint64_t id) { id_ = id; }

 private:
  uint64_t id_ = 0;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<StreamEvent> events_;
  bool final_pushed_ = false;
  bool final_popped_ = false;
  std::function<void()> cancel_fn_;
};

class Engine {
 public:
  static Result<std::unique_ptr<Engine>> create(EngineOptions opts);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // `prompt` is tokenized as-is (special-token text parsed only when
  // `parse_special`). BOS is added per the model's metadata.
  Result<std::shared_ptr<RequestStream>> generate_text(std::string_view prompt, bool parse_special,
                                                       const GenerateParams& params);
  // Applies the model's chat template.
  Result<std::shared_ptr<RequestStream>> generate_chat(std::span<const ChatMessage> messages,
                                                       const GenerateParams& params);

  const LoadedModel& model() const { return *model_; }
  std::string backend_name() const { return std::string(backend_->name()); }
  int threads() const { return pool_->size(); }
  const KvGeometry& kv_geometry() const { return kv_->geometry(); }
  // Snapshot of scheduler counters (taken on the scheduler thread).
  SchedulerStats stats() const;

 private:
  Engine() = default;
  Result<std::shared_ptr<RequestStream>> submit(std::vector<TokenId> tokens, const GenerateParams& params);
  void loop();

  // Control block shared with streams so cancel() is safe after shutdown.
  struct Core {
    std::mutex mu;
    Scheduler* sched = nullptr;  // null after shutdown
  };

  std::unique_ptr<LoadedModel> model_;
  std::unique_ptr<ThreadPool> pool_;
  std::unique_ptr<CpuBackend> backend_;
  std::unique_ptr<KvBlockPool> kv_;
  std::unique_ptr<Transformer> transformer_;
  std::unique_ptr<Scheduler> scheduler_;
  std::shared_ptr<Core> core_;

  std::thread thread_;
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;
  bool stop_ = false;
  std::vector<std::weak_ptr<RequestStream>> live_;  // open streams (guarded by wake_mu_), ended at shutdown
  mutable std::mutex stats_mu_;
  SchedulerStats stats_;
};

}  // namespace engine

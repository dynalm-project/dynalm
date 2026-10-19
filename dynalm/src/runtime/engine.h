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

#include "dynacore/device/device_registry.h"
#include "chat_template/chat_template.h"
#include "dynacore/base/status.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "metrics/metrics.h"
#include "model/transformer.h"
#include "dynacore/execution/thread_pool.h"
#include "sampling/sampler.h"
#include "runtime/device_select.h"
#include "runtime/speculative.h"
#include "scheduler/scheduler.h"
#include "common/core.h"

namespace dynalm {

// How forward passes reach the hardware (DD-072):
//   kReference  Device ops run as the model issues them (the hand-written
//               C++ kernels; the A/B baseline).
//   kCompiled   ops are recorded as DynaCore IR, compiled (canonicalize,
//               kernel selection, fusion) at every sync point and executed;
//               any compiler failure runs the recorded ops unchanged.
enum class ExecutionMode : uint8_t { kReference, kCompiled };
std::string_view execution_mode_name(ExecutionMode m);
Result<ExecutionMode> parse_execution_mode(std::string_view s);

struct EngineOptions {
  std::string model_path;
  ExecutionMode execution = ExecutionMode::kReference;
  int threads = 0;              // 0 = physical cores
  int64_t kv_tokens = 0;        // KV capacity in tokens; 0 = auto (auto_kv_tokens)
  DeviceKind backend = DeviceKind::kCpu;
  // Pick the device: the GPU when this build has one, it initializes and it
  // runs the model (gpu_model_support); otherwise the CPU. `backend` is
  // ignored. The CLI sets this for `--backend auto` (its default).
  bool backend_auto = false;
  // With backend_auto: a GPU is required (`--gpu`). When no built-in GPU can
  // run the model, create() fails with the reason instead of using the CPU.
  bool backend_require_gpu = false;
  // With backend kCpu (including auto's CPU choice): run on ggml's CPU kernels
  // (DeviceKind::kGgmlCpu: 8-bit activations, llama.cpp-level accuracy, ~4x
  // faster) when this build has them and the model runs there; else, or if
  // that device fails to start, DynaCore's native kernels (DD-053). The
  // CLI sets it for `cpu` and `auto`; --mode exact and safe clear it.
  bool cpu_ggml = false;
  // Speculative decoding with a draft model (SchedulerConfig::spec_k > 0): a
  // small GGUF sharing the target's vocabulary, run on the same device with
  // its own KV cache (BatchDrafter). Empty = prompt lookup when spec_k > 0.
  std::string draft_model_path;
  DType kv_dtype = DType::kF16;
  // Max tokens per forward pass; 0 = the device default: max(256, both step
  // budgets + 1, the solo-prefill chunk) (256 on the CPU with default budgets).
  int32_t max_batch_tokens = 0;
  // Max rows of a matmul that uses int8 activations (DD-053); -1 = default
  // (KernelPlan, 4; kNeonInt8Rows on ARM64), 0 = never. 0 makes outputs independent of how requests
  // are batched (DD-031); the int8 path is faster for 1-4 concurrent rows.
  int32_t int8_decode_rows = -1;
  SchedulerConfig scheduler;
  // Apply the device's scheduling profile to the balanced policy (GPU: see
  // kGpuMixedPrefillBudget). False when the caller set step budgets itself.
  bool device_scheduler_defaults = true;
};

// Applies a device choice from --backend / --cpu / --gpu, `runtime.device` or
// DYNALM_BACKEND to `o`: "auto" (a GPU when one runs the model, else the
// CPU), "gpu" (any built-in GPU, never the CPU), "cpu" (ggml's CPU kernels
// when available, see cpu_ggml), "ggml-cpu" (ggml's, no fallback), or a named
// GPU ("metal", "vulkan", "cuda"). "native-cpu" is not a public backend in
// this release (kNativeCpuRetired): DynaCore's CPU kernels run only through
// --mode exact / safe and as the CPU fallback (cpu_ggml).
Status apply_backend_choice(std::string_view choice, EngineOptions& o);

// The error for `--backend native-cpu` (and `runtime.device: native-cpu` in a
// config file written for an earlier version). Never a silent redirect.
inline constexpr std::string_view kNativeCpuRetired =
    "backend 'native-cpu' is not included in this release; use --backend cpu (fast CPU kernels) or "
    "--mode exact (DD-053 accuracy, DynaCore's CPU kernels). In a config file, set runtime.device: cpu "
    "and runtime.mode: exact";

// Solo-prefill chunk on GPU devices when none is configured
// (SchedulerConfig::solo_prefill_chunk == -1). Chosen by measurement on the
// Metal backend (docs/gpu-backend.md, section 0); the CPU default stays 0.
inline constexpr int32_t kGpuSoloPrefillChunk = 256;
// GPU profile of the balanced policy (docs/gpu-backend.md, section 0): on the
// GPU a forward step pays the full weight read whatever its row count, so the
// CPU-tuned mixed-step limits (64 prompt tokens, 32 per sequence, DD-079
// target 6x / minimum 32) starve throughput. Measured: these values raise
// throughput 22-45% and cut ITL p50 2-6x at 4-16 users, at the cost of a
// higher ITL p95/p99; --policy latency keeps the original DD-079 behaviour.
inline constexpr int32_t kGpuMixedPrefillBudget = 256;
inline constexpr int32_t kGpuMaxPrefillChunk = 128;
// DD-079 on the GPU: the halving controller with target 12x (v0.1.7's profile).
// The predictive controller (decode_protect_predictive, target 4x) halved the
// ITL p99 at 4-16 users but, measured 2026-10-18 on an M4 Pro, worsened the
// ITL p50 3-4x at 16 users (1.5B: 26.6 -> 83.5 ms; 8B: 80 -> 310 ms) and cost
// 7% throughput at 8B: it pins every mixed step at the protected minimum.
// It stays available: DYNALM_DECODE_PREDICTIVE=1 DYNALM_DECODE_TARGET=4.
inline constexpr float kGpuDecodeLatencyTarget = 12.0f;
inline constexpr bool kGpuDecodePredictive = false;
inline constexpr int32_t kGpuProtectedPrefillMin = 64;
// int8 activation rows on DynaCore's ARM64 CPU kernels when not configured
// (EngineOptions::int8_decode_rows = -1); x86 keeps KernelPlan's 4.
inline constexpr int32_t kNeonInt8Rows = 64;

struct GenerateParams {
  int32_t max_tokens = 256;
  std::vector<std::string> stop;  // stop strings (never emitted)
  bool stop_at_eog = true;
  int32_t priority = 0;
  int64_t timeout_ms = 0;
  SamplingParams sampling;  // default: greedy (DD-043)
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

// AUTO KV capacity in tokens (DD-038): half of the RAM still available after
// the weights, capped at 65536 tokens, at least min(context, 2048), rounded
// down to whole 16-token blocks. `available_ram` 0 (unknown) falls back to
// min(context, 16384).
// gpu_model_support and plan_devices: runtime/device_select.h.

int64_t auto_kv_tokens(const ModelConfig& c, DType kv_dtype, int64_t weight_bytes, int64_t available_ram);

// Memory admission: whether a KV cache of `kv_bytes` (`kv_per_token` per
// token) fits beside `weight_bytes` of weights. Host memory: weights + KV must
// stay within 90% of physical RAM (`total_ram`), else the system swaps or the
// process is killed mid-run. A GPU with its own memory (`device_free` >= 0, the
// weights are copied there): weights + KV within 95% of its free memory.
// A device with a working-set cap (`kv_budget` >= 0, device_kv_budget): the KV
// cache within that budget. Unknown sizes (<= 0, kv_budget -1) are not
// checked. The error names the largest --ctx that fits and --kv q8_0.
// Bytes left for the KV cache on a device with a working-set cap (Metal's
// recommendedMaxWorkingSetSize, MemoryStats::working_set_bytes): the cap less
// a kWorkingSetMarginPct safety margin, what is already allocated against it
// (measured before the KV pool exists, so the KV is not in it), the weights
// (mapped weights count too), the graph workspace the device may grow to
// (`workspace_reserve`; kDefaultWorkspaceReserve when 0) and the per-step
// activation scratch (`scratch_bytes`). Each term once. Never negative. -1
// when the cap is unknown (working_set <= 0): no device limit.
inline constexpr int64_t kWorkingSetMarginPct = 10;
// Automatic sizing shrinks the KV cache to the working set only down to this
// many tokens (or the model's context, if smaller): below it the device is
// refused at admission instead (--backend auto then starts on the CPU, --gpu
// reports it with the largest --ctx that fits). An explicit --ctx is honoured.
inline constexpr int64_t kMinAutoKvTokens = 1024;
inline constexpr int64_t kDefaultWorkspaceReserve = int64_t{1} << 30;
int64_t device_kv_budget(int64_t working_set, int64_t allocated, int64_t weight_bytes, int64_t workspace_reserve,
                         int64_t scratch_bytes = 0);
// The most KV tokens (whole 16-token blocks) whose pool, as KvBlockPool
// allocates it (kv_geometry_for(...).total_bytes()), fits in `budget` bytes:
// the same arithmetic the admission check compares, so an automatically
// sized cache is never refused by it.
int64_t kv_tokens_within(const ModelConfig& c, DType kv_dtype, int64_t budget);

Status check_memory_fit(int64_t weight_bytes, int64_t kv_bytes, int64_t kv_per_token, int64_t total_ram,
                        int64_t device_free, int64_t kv_budget = -1);

// Observability snapshot, refreshed by the scheduler thread after each step.
struct EngineStats {
  SchedulerStats scheduler;
  PrefixCacheStats prefix;
  int32_t kv_blocks_used = 0;
  int32_t kv_blocks_total = 0;
  // Throughput over the last completed window of >= 1 s of busy steps.
  double generation_tok_s = 0;
  double prefill_tok_s = 0;
  uint64_t prefill_tokens = 0;  // prompt rows computed (excludes prefix-cache hits)
  // Prompt tokenization on the submitting threads (DD-050).
  double tokenize_ms = 0;
  uint64_t tokenized_requests = 0;
  // Filled only while profiling is on (Engine::set_profiling).
  bool profiling = false;
  // Compiled execution (DD-072); zero in reference mode.
  bool compiled = false;
  int64_t compiled_segments = 0, compiled_cache_hits = 0, compiled_cache_misses = 0, compiled_fallbacks = 0;
  int64_t compiled_device_calls = 0, compiled_recorded_calls = 0;
  ForwardProfile forward;
  ThreadPoolStats pool;
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
  std::string backend_name() const { return std::string(compiled_ ? compiled_->name() : backend_->name()); }
  // Interleaved weight copies made at load (DD-078).
  Device::PrepackStats prepack_stats() const { return backend_->prepack_stats(); }
  int threads() const { return pool_->size(); }
  // The device this engine runs on (after create()'s selection).
  DeviceKind device_kind() const { return kind_; }
  // Effective max tokens per forward pass (after device defaults).
  int32_t max_batch_tokens() const { return transformer_->max_batch_tokens(); }
  const KvGeometry& kv_geometry() const { return kv_->geometry(); }
  // Snapshot taken on the scheduler thread after its latest step.
  EngineStats stats() const;
  // Duration of each scheduler step (batched forward pass), ms.
  const metrics::Histogram& step_ms() const { return step_ms_; }
  // Bytes of the KV pool and of the memory-mapped weights.
  int64_t kv_bytes() const { return kv_->geometry().total_bytes(); }
  int64_t weight_bytes() const { return model_->weight_bytes; }
  int64_t kv_capacity_tokens() const;
  // Per-op forward timing and thread-pool accounting (benchmarks). Takes
  // effect at the next scheduler step; the counters keep accumulating.
  void set_profiling(bool on) { want_profiling_.store(on, std::memory_order_relaxed); }
  // Scheduler trace (EngineOptions::scheduler.trace); thread-safe, hands over and clears.
  // The scheduler configuration in effect (after policy overrides and clamping).
  const SchedulerConfig& scheduler_config() const { return scheduler_->config(); }
  std::vector<StepTrace> take_step_trace() { return scheduler_->take_step_trace(); }
  std::vector<RequestTrace> take_request_trace() { return scheduler_->take_request_trace(); }

 private:
  // One device, no selection (create() picks the devices and calls this).
  static Result<std::unique_ptr<Engine>> create_on(EngineOptions opts);
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
  std::unique_ptr<Device> backend_;
  DeviceKind kind_ = DeviceKind::kCpu;
  std::unique_ptr<Device> compiled_;  // RecordingDevice over backend_ (kCompiled)
  std::unique_ptr<KvBlockPool> kv_;
  std::unique_ptr<Transformer> transformer_;
  // Speculative draft model (EngineOptions::draft_model_path), on the same device.
  std::unique_ptr<LoadedModel> draft_model_;
  std::unique_ptr<KvBlockPool> draft_kv_;
  std::unique_ptr<Transformer> draft_transformer_;
  std::unique_ptr<BatchDrafter> drafter_;
  std::unique_ptr<Scheduler> scheduler_;
  std::shared_ptr<Core> core_;

  std::thread thread_;
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;
  bool stop_ = false;
  std::vector<std::weak_ptr<RequestStream>> live_;  // open streams (guarded by wake_mu_), ended at shutdown
  mutable std::mutex stats_mu_;
  EngineStats stats_;
  std::atomic<bool> want_profiling_{false};
  bool profiling_ = false;  // scheduler thread only
  std::atomic<int64_t> tokenize_ns_{0};
  std::atomic<uint64_t> tokenized_{0};
  metrics::Histogram step_ms_{metrics::latency_buckets_ms()};
};

}  // namespace dynalm

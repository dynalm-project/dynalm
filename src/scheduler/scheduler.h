#pragma once

// Continuous-batching scheduler.
//
// Owns every in-flight sequence and drives the model one iteration at a time.
// Each step():
//   1. takes newly submitted requests from the handoff queue,
//   2. admits waiting sequences while KV capacity allows,
//   3. builds one batch: decode rows first (within decode_token_budget,
//      rotating fairly), then prefill chunks by priority (within
//      prefill_token_budget),
//   4. runs a single batched forward pass,
//   5. samples for sequences whose tokens are all computed, invokes callbacks,
//      and retires finished / cancelled / failed sequences.
// Sequences join and leave between iterations; nothing waits for a "batch" to
// finish.
//
// Threading: submit() and cancel() may be called from any thread (short mutex
// on the handoff queue / atomic flag). step() runs on one scheduler thread and
// holds no lock while computing. Callbacks run on the scheduler thread and
// must be fast (they should enqueue, not block).
//
// Model-agnostic: the scheduler sees tokens, positions and KV blocks only.

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "common/status.h"
#include "kv_cache/kv_cache.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "tokenizer/tokenizer.h"

namespace engine {

struct RequestEvent {
  uint64_t request_id = 0;
  TokenId token = kNoToken;  // valid for token events
  bool finished = false;     // last event for this request
  SequenceStatus status = SequenceStatus::kWaiting;
  FinishReason reason = FinishReason::kNone;
  Status error;              // set when status == kError
};

// Called on the scheduler thread for every generated token and once at the end.
using RequestCallback = std::function<void(const RequestEvent&)>;

struct Request {
  std::vector<TokenId> prompt;
  StopParams stop;
  RequestCallback on_event;
  int32_t priority = 0;    // higher is admitted first; FCFS within a level
  int64_t timeout_ms = 0;  // 0 = none; counted from submission, fails with kDeadlineExceeded
};

// Per-step work limits. Decode rows (one per generating sequence) are cheap
// and latency-critical; prefill rows are expensive. Capping prefill rows per
// step bounds how long any step can stall decoding sequences (ITL), at the
// cost of spreading a prompt over more steps (TTFT). See DD-027.
struct SchedulerConfig {
  int32_t decode_token_budget = 64;   // max decode rows per step
  int32_t prefill_token_budget = 64;  // max prefill rows per step (chunked prefill)
  int32_t max_running = 64;           // max admitted sequences
};

struct SchedulerStats {
  uint64_t steps = 0;
  uint64_t tokens_computed = 0;  // rows through the model
  uint64_t tokens_generated = 0;
  uint64_t preemptions = 0;
  uint64_t completed = 0;
  uint64_t cancelled = 0;
  uint64_t failed = 0;
  int32_t running = 0;
  int32_t waiting = 0;
  uint64_t timed_out = 0;
  int32_t last_batch_rows = 0;
  int32_t last_batch_seqs = 0;
  int32_t last_prefill_rows = 0;
  int32_t last_decode_rows = 0;
  // Queue latency (submission -> admission), accumulated.
  double queue_ms_total = 0;
  double queue_ms_max = 0;
  uint64_t admitted = 0;
};

class Scheduler {
 public:
  Scheduler(Transformer& model, KvBlockPool& kv, const Tokenizer& tokenizer, SchedulerConfig config);
  ~Scheduler();

  // Thread-safe. Returns the request id. Invalid requests are rejected
  // through their callback on the next step (errors never throw).
  uint64_t submit(Request request);
  // Thread-safe; takes effect at the next step. Unknown ids are ignored.
  void cancel(uint64_t request_id);

  // Runs one iteration. Returns false when there is nothing left to do.
  bool step();
  // Steps until idle.
  void run_until_idle();

  bool idle() const;
  const SchedulerStats& stats() const { return stats_; }

 private:
  struct Entry {
    uint64_t id;
    std::unique_ptr<SequenceState> seq;
    RequestCallback on_event;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    uint64_t admit_order = 0;
    int32_t priority = 0;
    int64_t arrival_ns = 0;
    int64_t deadline_ns = 0;  // 0 = none
    uint64_t last_step = 0;   // last step this sequence got decode rows (fairness)
  };

  void drain_incoming();
  void expire_deadlines(int64_t now);
  void admit(int64_t now);
  bool preempt_one(const Entry* keep);
  void retire(Entry& e);
  void emit_final(Entry& e);

  Transformer& model_;
  KvBlockPool& kv_;
  const Tokenizer& tokenizer_;
  SchedulerConfig config_;
  SchedulerStats stats_;

  // Handoff from submitters (guarded by in_mu_).
  mutable std::mutex in_mu_;
  std::vector<Entry> incoming_;
  std::vector<uint64_t> cancel_requests_;
  std::atomic<uint64_t> next_id_{1};
  std::atomic<size_t> incoming_count_{0};

  // Scheduler-thread state (no lock).
  std::deque<Entry> waiting_;
  std::vector<Entry> running_;
  uint64_t admit_counter_ = 0;
  std::vector<SeqBatch> batch_;
  std::vector<Entry*> batch_owner_;
  std::vector<float> logits_;
};

}  // namespace engine

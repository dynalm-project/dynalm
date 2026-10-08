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
#include <string_view>
#include <vector>

#include "dynacore/base/status.h"
#include "kv_cache/kv_cache.h"
#include "model/transformer.h"
#include "sampling/sampler.h"
#include "prefix_cache/prefix_cache.h"
#include "runtime/sequence.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

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
  SamplingParams sampling;  // default: greedy
  RequestCallback on_event;
  int32_t priority = 0;    // higher is admitted first; FCFS within a level
  int64_t timeout_ms = 0;  // 0 = none; counted from submission, fails with kDeadlineExceeded
};

// Per-step work limits. Decode rows (one per generating sequence) are cheap
// and latency-critical; prefill rows are expensive. Capping prefill rows per
// step bounds how long any step can stall decoding sequences (ITL), at the
// cost of spreading a prompt over more steps (TTFT). See DD-027.
enum class PrefixCacheKind : uint8_t { kRadix, kHash };

// How step budgets trade time-to-first-token against inter-token latency
// (P12, DD-061). On a saturated CPU the aggregate output rate is about the
// same for all three; they differ in who waits.
enum class SchedulerPolicy : uint8_t {
  kBalanced,         // the configured budgets (default 64 prompt + 64 decode tokens per step)
  kLatencyFirst,     // small prompt budget (32): steady token streams, slower first tokens
  kThroughputFirst,  // large prompt budget (128): fast admission, burstier streams
};
std::string_view scheduler_policy_name(SchedulerPolicy p);
// "balanced" | "latency" | "throughput"; false if unknown.
bool parse_scheduler_policy(std::string_view s, SchedulerPolicy& out);

struct SchedulerConfig {
  int32_t decode_token_budget = 64;   // max decode rows per step
  int32_t prefill_token_budget = 64;  // max prefill rows per step (chunked prefill)
  int32_t max_running = 64;           // max admitted sequences
  // Max prefill rows one sequence may take per step (0 = whole prefill
  // budget). Smaller values let several prompts prefill side by side, so a
  // short prompt is not stuck behind a long one (head-of-line blocking).
  int32_t max_prefill_chunk = 32;
  SchedulerPolicy policy = SchedulerPolicy::kBalanced;
  // Decode-protected prefill (DD-079): in steps that also carry decode rows,
  // prefill rows are capped by an adaptive budget so a long prompt does not
  // stall running streams for a whole large chunk. After each mixed step the
  // budget halves (down to protected_prefill_min) when the step took longer
  // than decode_latency_target x the recent decode-only step time, and
  // doubles (up to prefill_token_budget) when it took under half of that.
  // Steps without decode rows keep the full budget. Off for kThroughputFirst;
  // kLatencyFirst uses target 4 and minimum 16. Defaults measured on
  // Qwen2.5-1.5B (DD-079): ITL p99 -38..-43% at 4-8 users for <= 1.5% tok/s.
  bool decode_protect = true;
  int32_t protected_prefill_min = 32;
  float decode_latency_target = 6.0f;
  // Reuse KV of shared prompt prefixes across requests (DD-029).
  bool enable_prefix_cache = true;
  PrefixCacheKind prefix_cache_kind = PrefixCacheKind::kRadix;
  int32_t prefix_cache_max_blocks = 0;  // 0 = bounded only by the KV pool
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

  // Where step time goes, accumulated in ms (always on: a few clock reads per
  // step). plan_ms covers cancellation, admission and batch building, and
  // includes prefix_lookup_ms and kv_reserve_ms.
  double plan_ms = 0;
  double prefix_lookup_ms = 0;
  double kv_reserve_ms = 0;
  double forward_ms = 0;
  double sample_ms = 0;         // sampling + stop/EOS checks
  double emit_ms = 0;           // per-token callbacks (stream delivery)
  double prefix_insert_ms = 0;  // offering completed blocks to the prefix cache
  // Steps by composition, with their forward time, so steady-state decode can
  // be read without prefill mixed in.
  uint64_t steps_decode_only = 0;
  uint64_t steps_prefill_only = 0;
  uint64_t steps_mixed = 0;
  double forward_decode_only_ms = 0;
  double forward_prefill_only_ms = 0;
  double forward_mixed_ms = 0;
  uint64_t decode_rows_total = 0;
  uint64_t prefill_rows_total = 0;
  uint64_t steps_split_attention = 0;  // planner chose split-K attention (DD-051)
  int32_t mixed_prefill_budget = 0;    // current decode-protected prefill budget (DD-079)
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
  // nullptr when the prefix cache is disabled.
  const PrefixCache* prefix_cache() const { return prefix_cache_.get(); }

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
    int32_t cached_blocks = 0;  // full blocks already offered to the prefix cache
    std::unique_ptr<Sampler> sampler;  // null: greedy
  };

  void drain_incoming();
  void expire_deadlines(int64_t now);
  void admit(int64_t now);
  bool preempt_one(const Entry* keep);
  void retire(Entry& e);
  void emit_final(Entry& e);

  // Decode-protected prefill state (DD-079).
  double decode_step_ema_ms_ = 0;
  int32_t mixed_prefill_budget_ = 0;

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
  std::unique_ptr<PrefixCache> prefix_cache_;
};

}  // namespace dynalm

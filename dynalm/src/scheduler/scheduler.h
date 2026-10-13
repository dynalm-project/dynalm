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

// Order of prompts within the prefill budget of a step (see SchedulerConfig::prefill_order).
enum class PrefillOrder : uint8_t { kFifo, kShortestRemaining };
std::string_view prefill_order_name(PrefillOrder o);
// "fifo" | "srpt"; false if unknown.
bool parse_prefill_order(std::string_view s, PrefillOrder& out);

// One scheduler step (SchedulerConfig::trace).
struct StepTraceRow {
  uint64_t request_id = 0;
  int32_t rows = 0;
  int32_t start_pos = 0;  // first position computed in this step
  bool decode = false;
};
struct StepTrace {
  uint64_t step = 0;
  int64_t plan_start_ns = 0, forward_start_ns = 0, forward_end_ns = 0;
  int32_t prefill_rows = 0, decode_rows = 0;
  int32_t prefill_budget = 0, prefill_chunk = 0;  // as planned (chunk 0 = up to the budget)
  int32_t prefilling = 0;                          // running sequences with prompt tokens left
  int32_t waiting = 0;                             // not yet admitted
  bool solo = false, solo_skipped = false, srpt = false;
  std::vector<StepTraceRow> rows;                  // what each scheduled sequence computed
  std::vector<uint64_t> prefill_unserved;          // prefilling sequences that got no rows this step
};
// One request's lifecycle (SchedulerConfig::trace), recorded when it retires.
struct RequestTrace {
  uint64_t id = 0;
  int32_t prompt_tokens = 0, generated = 0;
  int64_t submit_ns = 0, admit_ns = 0, first_scheduled_ns = 0, first_token_ns = 0, finish_ns = 0;
  int32_t prefill_steps = 0;        // steps in which it computed prompt rows
  int32_t prefill_unserved_steps = 0;  // steps while prefilling in which it got no rows
  double prefill_forward_ms = 0;    // forward time of the steps that computed its prompt rows
  int32_t preemptions = 0;
  SequenceStatus status = SequenceStatus::kWaiting;
};
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
  // EXPERIMENTAL (off by default): in steps with no decode rows and exactly
  // one sequence prefilling (a lone prompt), prefill uses a chunk of
  // solo_prefill_chunk rows and a budget of max(prefill_token_budget,
  // solo_prefill_chunk) instead of max_prefill_chunk and prefill_token_budget.
  // Fewer, larger prefill matmuls (-9..-11% matmul cost per token at 128 vs 32
  // rows on the i7-1255U, Phase 8). With two or more prompts prefilling the
  // step is unchanged (unless solo_prefill_split): giving the whole solo budget
  // to the oldest prompt made short prompts wait behind long ones (Phase 9 V3,
  // short-prompt TTFT +29..34%). Steps with decode rows are unchanged:
  // max_prefill_chunk, and the DD-079 budget, which stays bounded by
  // prefill_token_budget. 0 = off (every step as before).
  // `dynalm serve --solo-prefill-chunk N` (DYNALM_SOLO_PREFILL_CHUNK),
  // `dynalm benchmark --solo-prefill-chunk N`.
  int32_t solo_prefill_chunk = 0;
  // With solo_prefill_chunk on and several sequences prefilling in a step
  // with no decode rows: false = no solo chunk in that step (regular chunk and
  // budget; FIFO order as always); true = split, the solo budget shared, each
  // sequence up to max(max_prefill_chunk, solo_prefill_chunk / prefilling
  // sequences).
  // `--solo-prefill-split` (serve: DYNALM_SOLO_PREFILL_SPLIT).
  bool solo_prefill_split = false;
  // EXPERIMENTAL (default kFifo): the order in which prompts share the prefill
  // budget of a step. kFifo = priority, then admission order (as always).
  // kShortestRemaining = priority, then fewest prompt tokens still to compute,
  // so a short prompt that arrives behind a long one is served first; a prompt
  // prefilling for longer than prefill_aging_ms since admission is served
  // ahead of the others (FIFO among such prompts), so long prompts cannot
  // starve. Only the order changes: chunk caps, budgets and DD-079 do not;
  // each sequence still computes its own tokens in order, a chunk per step.
  PrefillOrder prefill_order = PrefillOrder::kFifo;
  int32_t prefill_aging_ms = 20000;
  // Record a per-step and per-request trace (take_step_trace /
  // take_request_trace). Statistics only: scheduling is unchanged.
  bool trace = false;
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
  int32_t solo_prefill_chunk = 0;      // effective SchedulerConfig::solo_prefill_chunk (0 = off)
  bool solo_prefill_split = false;     // effective SchedulerConfig::solo_prefill_split
  uint64_t steps_solo_prefill = 0;     // steps planned with the solo-prefill chunk and budget
  double forward_solo_prefill_ms = 0;    // forward time of the steps counted in steps_solo_prefill
  double solo_prefill_step_ms_max = 0;   // longest such step (since the scheduler started)
  uint64_t steps_solo_prefill_skipped = 0;  // steps without decode rows that kept the regular chunk
                                            // because several prompts were prefilling (FIFO mode)
  uint64_t steps_prefill_srpt = 0;  // steps planned in shortest-remaining order with >= 2 prompts prefilling
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
  // The configuration in effect: the requested one after policy overrides,
  // environment overrides and clamping to the batch (constructor).
  const SchedulerConfig& config() const { return config_; }
  // nullptr when the prefix cache is disabled.
  const PrefixCache* prefix_cache() const { return prefix_cache_.get(); }
  // Thread-safe: hands over (and clears) the trace recorded so far
  // (empty unless SchedulerConfig::trace).
  std::vector<StepTrace> take_step_trace();
  std::vector<RequestTrace> take_request_trace();

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
    int64_t admit_ns = 0;       // latest admission (prefill aging)
    RequestTrace trace;         // filled only with SchedulerConfig::trace
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
  // Trace (SchedulerConfig::trace), handed over by take_*_trace from any thread.
  std::mutex trace_mu_;
  std::vector<StepTrace> step_trace_;
  std::vector<RequestTrace> request_trace_;

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

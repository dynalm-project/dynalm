#pragma once

// Benchmark load generator.
//
// Drives a target (the in-process Engine, or any OpenAI-compatible HTTP
// server) with `concurrency` closed-loop clients, each sending prompts of a
// fixed token length and requesting a fixed number of output tokens, and
// measures per request: TTFT, inter-token latency, TPOT, end-to-end latency.
// Results are aggregated into P50/P90/P95/P99 (never means alone).
// Open-loop mode (arrival_rate > 0): requests arrive at Poisson times instead,
// at most `concurrency` in flight; latencies count from the scheduled arrival,
// so time spent waiting for a free slot is not hidden.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bench/analysis.h"
#include "dynacore/base/status.h"
#include "dynacore/hardware/perf_counters.h"
#include "runtime/engine.h"
#include "common/core.h"

namespace dynalm::bench {

struct RequestResult {
  bool ok = false;
  std::string error;
  double ttft_ms = 0;
  double e2e_ms = 0;
  std::vector<double> gaps_ms;  // between consecutive streamed deltas
  int32_t prompt_tokens = 0;    // as reported by the target (0 if unknown)
  int32_t completion_tokens = 0;
  int32_t requested_prompt_tokens = 0;  // the workload's length for this request (set by run_point)
  double start_delay_ms = 0;            // open loop: scheduled arrival -> request sent (included in ttft/e2e)
  int32_t index = -1;                   // request index within the point (set by run_point)
  double scheduled_ms = -1;             // open loop: scheduled arrival, ms after the point's start (-1 = closed loop)
  double dispatch_ms = 0;               // when the request was sent, ms after the point's start
  bool cap_bound = false;               // open loop: no in-flight slot was free at the scheduled arrival
};

class Target {
 public:
  virtual ~Target() = default;
  virtual std::string name() const = 0;
  // Runs one streamed request; must be callable from several threads.
  virtual RequestResult run(const std::string& prompt, int32_t max_tokens) = 0;
  // Process resource usage if the target runs in this process.
  virtual bool in_process() const { return false; }
  // The engine behind an in-process target (internal stats), else null.
  virtual Engine* engine() { return nullptr; }
};

std::unique_ptr<Target> make_engine_target(Engine& engine);
// OpenAI /v1/completions with stream=true (and ignore_eos where supported).
// Only available when built with the server (cpp-httplib).
Result<std::unique_ptr<Target>> make_http_target(const std::string& url, const std::string& model);

struct Percentiles {
  double p50 = 0, p90 = 0, p95 = 0, p99 = 0, mean = 0, max = 0;
};
Percentiles percentiles(std::vector<double> v);

struct PointConfig {
  int32_t concurrency = 1;
  int32_t prompt_tokens = 128;
  int32_t output_tokens = 128;
  int32_t requests = 0;  // 0 = 2 * concurrency (min 4)
  // Mixed workload: request i uses prompt_mix[i % size] tokens (overrides
  // prompt_tokens, which then reports the mean).
  std::vector<int32_t> prompt_mix;
  // Open loop: mean arrivals per second (Poisson, exponential gaps); 0 =
  // closed loop. `concurrency` caps the requests in flight.
  double arrival_rate = 0;
  uint32_t seed = 1;  // arrival times (the same seed gives the same schedule)
  // Full-size warm-up requests before the point is measured (uncounted, one at
  // a time, longest prompt length first); 0 = only the short paging warm-up.
  int32_t warmup_requests = 0;
  // Deterministic open loop: request i arrives at arrival_trace[i].offset_ms
  // with arrival_trace[i].prompt_tokens tokens (replaces arrival_rate,
  // prompt_tokens / prompt_mix and requests). `concurrency` caps the requests
  // in flight, as with arrival_rate.
  struct TraceEntry {
    int64_t offset_ms = 0;
    int32_t prompt_tokens = 0;
  };
  std::vector<TraceEntry> arrival_trace;
  std::string arrival_trace_path;  // reported in the JSON
};

// Reads an arrival trace: one "offset_ms,prompt_tokens" pair per line,
// non-decreasing offsets; blank lines and lines starting with '#' are skipped.
Result<std::vector<PointConfig::TraceEntry>> load_arrival_trace(const std::string& path);

// Latencies of the requests with one prompt length (mixed workloads).
struct PromptGroup {
  int32_t prompt_tokens = 0;
  int32_t completed = 0;
  Percentiles ttft_ms, itl_ms, e2e_ms;
};

// Measurement context shared by all points of a run.
struct RunContext {
  const PerfCounters* perf = nullptr;  // opened after the engine started its threads
  double peak_bw_gbs = -1;             // measure_read_bandwidth_gbs()
};

// In-process diagnostics for one point: where the time went, counters, and
// the bottleneck classification (DD-050). Durations are totals over the point.
struct Diagnostics {
  bool valid = false;
  uint64_t steps = 0, steps_decode_only = 0, steps_prefill_only = 0, steps_mixed = 0;
  uint64_t spec_drafted = 0, spec_accepted = 0;  // speculative decoding (--spec)
  uint64_t steps_solo_prefill = 0;  // steps planned with the experimental solo-prefill chunk
  uint64_t steps_solo_prefill_skipped = 0;  // steps without decode rows where several prompts kept the regular chunk
  uint64_t steps_prefill_srpt = 0;          // steps in shortest-remaining prefill order (>= 2 prompts prefilling)
  double forward_solo_prefill_ms = 0;  // forward time of the point's solo-prefill steps
  double solo_prefill_step_ms_max = 0; // longest solo-prefill step since the engine started (incl. warm-up)
  int32_t solo_prefill_chunk = 0;   // its effective setting (0 = off)
  bool solo_prefill_split = false;
  double mean_decode_rows = 0;      // decode rows per step that had any
  double decode_step_ms = 0;        // mean forward time of a decode-only step
  double plan_ms = 0, prefix_lookup_ms = 0, kv_reserve_ms = 0, forward_ms = 0;
  double forward_decode_only_ms = 0, forward_prefill_only_ms = 0, forward_mixed_ms = 0;
  double sample_ms = 0, emit_ms = 0, prefix_insert_ms = 0, tokenize_ms = 0;
  double queue_wait_mean_ms = 0;
  std::vector<std::pair<std::string, double>> op_ms;  // per forward op (profiling)
  // Matmul kernels that ran during the point (dynacore matmul registry):
  // name, calls, rows, MACs, ms (ms < 0 unless DYNACORE_MATMUL_TIMING=1).
  struct MatmulKernelUse {
    std::string name;
    double calls = 0, rows = 0, macs = 0, ms = -1;
  };
  std::vector<MatmulKernelUse> matmul_kernels;
  double pool_regions = -1, pool_region_ms = -1, pool_tail_wait_ms = -1, pool_sleeps = -1;
  // Per-thread busy time / region time over the point (caller = thread 0).
  std::vector<double> pool_thread_util;
  PerfSample perf;
  double cpu_mhz = -1;
  double est_decode_bw_gbs = -1, peak_bw_gbs = -1;
  Classification bottleneck;
};

struct PointResult {
  PointConfig cfg;
  std::string target;
  int32_t completed = 0, errors = 0;
  double wall_s = 0;
  double output_tok_s = 0;  // aggregate generated tokens / wall
  double input_tok_s = 0;   // aggregate prompt tokens / wall
  double mean_completion_tokens = 0;
  Percentiles ttft_ms, itl_ms, tpot_ms, e2e_ms;
  std::vector<PromptGroup> by_prompt;  // per prompt length, with prompt_mix only
  double start_delay_ms_mean = 0, start_delay_ms_max = 0;  // open loop only
  // Open loop (Poisson or trace): the arrival schedule and how it was met.
  std::string arrival_mode = "closed";  // closed | poisson | trace
  std::string arrival_digest;           // FNV-1a of (scheduled offset, prompt length) per request; equal = same trace
  double trace_arrival_rate = 0;        // (n - 1) / (last - first scheduled arrival), requests/s
  double achieved_arrival_rate = 0;     // (n - 1) / (last - first dispatch), requests/s
  Percentiles start_delay_ms;           // scheduled arrival -> dispatch (open loop)
  int32_t cap_bound_requests = 0;       // requests whose arrival found every in-flight slot busy (sent late)
  std::vector<RequestResult> requests;  // per request (for --request-log)
  // In-process with the scheduler trace on: the measured steps and requests (warm-up excluded).
  int64_t t0_ns = 0;                    // the point's start (now_ns clock, as the trace)
  std::vector<StepTrace> step_trace;
  std::vector<RequestTrace> request_trace;
  double rss_mb = 0;     // in-process only
  double cpu_util = 0;   // in-process: CPU seconds / wall / cores
  std::string first_error;
  Diagnostics diag;
};

// Prompt text whose tokenization (by `tokenizer`) is exactly `tokens` long,
// with a unique prefix per `request_index` (so no request reuses another's
// prefix cache).
std::string make_prompt(const Tokenizer& tokenizer, int32_t tokens, int32_t request_index);

PointResult run_point(Target& target, const Tokenizer& tokenizer, const PointConfig& cfg,
                      const RunContext& ctx = {});

// One JSON object (single line) per point (JSON lines).
std::string to_json(const PointResult& r, const std::string& model, const std::string& hardware);

// One JSON object (single line) for one request of a point (--request-log).
std::string request_to_json(const PointResult& point, const RequestResult& r);

// The point's scheduler trace as JSON lines (--step-trace): one {"type":"step"} per step and
// one {"type":"request"} per request; times in ms after the point's start (t0_ns).
std::vector<std::string> trace_to_json_lines(const PointResult& point);

}  // namespace dynalm::bench

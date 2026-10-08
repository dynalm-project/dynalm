#pragma once

// Benchmark load generator.
//
// Drives a target (the in-process Engine, or any OpenAI-compatible HTTP
// server) with `concurrency` closed-loop clients, each sending prompts of a
// fixed token length and requesting a fixed number of output tokens, and
// measures per request: TTFT, inter-token latency, TPOT, end-to-end latency.
// Results are aggregated into P50/P90/P95/P99 (never means alone).

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
  double p50 = 0, p90 = 0, p95 = 0, p99 = 0, mean = 0;
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
  double mean_decode_rows = 0;      // decode rows per step that had any
  double decode_step_ms = 0;        // mean forward time of a decode-only step
  double plan_ms = 0, prefix_lookup_ms = 0, kv_reserve_ms = 0, forward_ms = 0;
  double forward_decode_only_ms = 0, forward_prefill_only_ms = 0, forward_mixed_ms = 0;
  double sample_ms = 0, emit_ms = 0, prefix_insert_ms = 0, tokenize_ms = 0;
  double queue_wait_mean_ms = 0;
  std::vector<std::pair<std::string, double>> op_ms;  // per forward op (profiling)
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

}  // namespace dynalm::bench

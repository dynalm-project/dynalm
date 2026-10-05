#pragma once

// Benchmark analysis (DD-050): a measured memory-bandwidth ceiling, a model of
// the bytes a decode step must move, and a rule-based bottleneck classifier.
// These turn raw timings into "what limits this workload", so optimizations
// target measured bottlenecks instead of guesses.

#include <cstdint>
#include <string>
#include <vector>

#include "loader/model_loader.h"
#include "runtime/thread_pool.h"

namespace engine::bench {

// Sustained read bandwidth (GB/s, 1e9 bytes) over a buffer far larger than the
// last-level cache, using every thread of `pool`. Best of a few passes.
double measure_read_bandwidth_gbs(ThreadPool& pool, size_t buffer_bytes = size_t{256} << 20);

// Bytes of weights one decode step reads when it carries `rows` sequences:
// every layer and LM-head weight once (the input embedding table only for the
// looked-up rows, unless it doubles as the LM head), and for MoE layers only
// the experts the rows are expected to touch (1 - (1 - k/E)^rows of them).
double decode_weight_bytes_per_step(const LoadedModel& model, double rows);

// KV bytes one decoded token reads for attention at context length `ctx`.
double kv_bytes_per_token_read(int32_t num_layers, int32_t kv_heads, int32_t head_dim, int32_t head_dim_v,
                               int32_t kv_dtype_bytes, double ctx);

enum class Bottleneck : uint8_t {
  kComputeBound,
  kMemoryBound,
  kCacheBound,
  kSynchronizationBound,
  kDispatchBound,
  kLoadImbalanced,
  kIoBound,
  kMixed,
};
std::string_view bottleneck_name(Bottleneck b);

// Everything the classifier looks at. Negative = not measured.
struct ClassifierInputs {
  double wall_s = 0;
  double forward_s = 0;           // time inside the model's forward passes
  double decode_forward_s = 0;    // ... in steps with decode rows only
  double prefill_forward_s = 0;   // ... in steps with prompt rows (prefill-only + mixed)
  double host_s = 0;              // scheduler work outside forward: planning, sampling, callbacks
  double est_decode_bw_gbs = -1;  // modelled bytes / decode-only forward time
  double peak_bw_gbs = -1;        // measured ceiling
  // Thread pool (profiling on).
  double pool_region_s = -1;
  double pool_tail_wait_s = -1;
  double pool_regions = -1;
  double regions_per_step = -1;
  // OS / hardware counters.
  double major_fault_mb_s = -1;   // disk reads caused by page faults
  double ipc = -1;
  double llc_miss_ratio = -1;     // LLC misses / LLC references
};

struct Classification {
  Bottleneck primary = Bottleneck::kMixed;
  std::vector<Bottleneck> all;    // every rule that fired, in priority order
  std::vector<std::string> evidence;
};

// Thresholds and priority are documented in DD-050 and docs/performance.md.
Classification classify(const ClassifierInputs& in);

}  // namespace engine::bench

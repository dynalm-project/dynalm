#pragma once

// Per-step execution decisions handed to a backend (DD-051).
//
// The execution planner (dynalm/src/execution) decides, once per forward pass, how
// the batch should run on this hardware; the backend only reads the result.
// The type lives in the backend layer so backends never depend on the planner,
// the scheduler or model families. A backend that receives no plan uses
// KernelPlan::defaults(), which reproduces the pre-planner behavior exactly.

#include <cstdint>
#include <string_view>

namespace engine {

enum class AttentionStrategy : uint8_t {
  kAuto,    // decide per call with attention_should_split() (no plan given)
  kPerPair, // one task per (query row, head): enough pairs to fill the pool
  kSplitK,  // also split each pair's context into chunks, merged by log-sum-exp
};
std::string_view attention_strategy_name(AttentionStrategy s);

// The split-K rule: split when (row, head) pairs alone cannot occupy the
// threads and the context is long enough to amortize the merge (DD-032).
inline bool attention_should_split(int64_t pairs, int64_t longest_context, int32_t threads, int32_t chunk) {
  return pairs < 2 * static_cast<int64_t>(threads) && longest_context > 2 * static_cast<int64_t>(chunk);
}

struct KernelPlan {
  // --- matmul ---
  // Rows at or above which weights are expanded into fp32 panels and run
  // through the GEMM kernel; below it each row uses fused dequantize-dot
  // straight from the packed weights (DD-036).
  int32_t expand_min_rows = 2;
  // GEMM K-slice width (multiple of 256; 0 = no K-blocking).
  int32_t gemm_k_block = 1024;
  // Chunks per thread when a matmul's output columns are split across the
  // pool: chunks are claimed dynamically, so more chunks shorten the tail when
  // fast (P) and slow (E) cores finish unevenly, at the cost of more claims.
  int32_t matmul_chunks_per_thread = 32;  // DD-067 (was 8)
  // Matmuls with at most this many rows use int8 activations and integer
  // dot products where the backend accelerates them (DD-053); 0 = never.
  // Default 4: the measured crossover with the fp32 expand path, and within
  // the accuracy contract (perplexity <= +1%, mean KL <= 0.0025 nats).
  int32_t int8_decode_max_rows = 4;
  // Whether the FFN down projection may use int8 activations. Its input (the
  // gated activation) carries the largest outlier channels: with it, Qwen3-4B
  // perplexity rose 5.4%; without it, 0.9% (DD-053).
  bool int8_ffn_down = false;

  // --- attention ---
  // Separate decisions for full-causal and sliding-window layers: a window
  // bounds the context, so the same batch can need split-K in one and not
  // the other.
  AttentionStrategy attention_full = AttentionStrategy::kAuto;
  AttentionStrategy attention_window = AttentionStrategy::kAuto;
  int32_t attention_chunk = 256;  // split-K chunk length in tokens
  // Query heads sharing a KV head are scored and accumulated together, each
  // K/V vector loaded once for the group (DD-066). false: one head at a time
  // (the pre-DD-066 path, kept for A/B measurement and as a reference).
  bool grouped_attention = true;

  // Defaults, with the tuning overrides DYNACORE_MATMUL_EXPAND_MIN,
  // DYNACORE_GEMM_KC, DYNACORE_INT8_DECODE_ROWS, DYNACORE_ATTN_GROUPED and DYNACORE_MATMUL_CHUNKS applied (read once per process).
  static const KernelPlan& defaults();
};

// What kernel selection knows about the machine. Filled once when the model
// runtime is created.
struct HardwareProfile {
  int32_t threads = 1;         // compute threads available to kernels
  int32_t physical_cores = 1;
  int32_t performance_cores = 0;  // 0 = not a hybrid CPU
  int32_t efficiency_cores = 0;
  int64_t l2_bytes = 0;        // per core
  int64_t llc_bytes = 0;
  std::string_view isa;        // e.g. "avx2" (informational)

  // From the detected CPU, with `threads` compute threads.
  static HardwareProfile detect(int32_t threads, std::string_view isa);
};

// The numeric shape of one forward step: everything kernel selection needs,
// and nothing about sequences, requests or model families. The caller reduces
// its batch to this; plan_kernels() turns it into a KernelPlan.
struct StepShape {
  int32_t rows = 0;            // query rows through the step
  int32_t max_context = 0;     // longest attention span of any row (tokens)
  int32_t num_kv_heads = 0;
  int32_t sliding_window = 0;  // window of windowed layers (tokens)
  bool has_window_layers = false;
};

// Per-step kernel decisions (DD-051): `base` with the attention strategy and
// split-K chunking chosen for this shape on this hardware.
KernelPlan plan_kernels(const StepShape& shape, const HardwareProfile& hw, const KernelPlan& base);

}  // namespace engine

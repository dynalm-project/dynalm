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

namespace dynacore {

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
  // Default 4: the measured crossover with the fp32 expand path (AVX2), and
  // within the accuracy contract (perplexity <= +1%, mean KL <= 0.0025 nats).
  // The engine uses 64 on NEON (kNeonInt8Rows in runtime/engine.cpp), where the
  // SDOT row kernels beat the fp32 expand GEMM at prefill sizes too (M4 Pro, Qwen2.5-1.5B, 401-token prompt:
  // 2394 -> 1835 ms); the per-row arithmetic, and so the accuracy, is that of
  // decode at any row count (prefill chunks of 32/64: perplexity +0.04%, mean KL 0.0012).
  int32_t int8_decode_max_rows = 4;
  // Whether the FFN down projection may use int8 activations. Its input (the
  // gated activation) carries the largest outlier channels: with it, Qwen3-4B
  // perplexity rose 5.4%; without it, 0.9% (DD-053).
  bool int8_ffn_down = false;
  // Matmuls with at most this many rows use int16 activations where the
  // backend has kernels (DD-076); 0 = never. The model sets it only for the
  // FFN down projection (see int16_ffn_down), never globally.
  int32_t int16_decode_max_rows = 0;
  // Whether the FFN down projection, excluded from int8 (int8_ffn_down), uses
  // int16 activations up to int8_decode_max_rows rows instead of fp32
  // (DD-076). Only while int8 decode is on, so --int8-decode 0 keeps fp32
  // everywhere (batch invariance, DD-031).
  bool int16_ffn_down = true;
  // K-quant weights (Q4_K, Q6_K) on the int8 path use super-block activations
  // (one scale per 256 values, integer accumulation per super-block, DD-075)
  // for matmuls of at least this many rows; 0 = never. Off by default: +14%
  // at 4 rows, but it fails the DD-053 accuracy contract (mean KL 0.021 on
  // Qwen2.5-1.5B, 0.0067 on Qwen3-4B; limit 0.0025). Opt-in for experiments.
  int32_t int8_superblock_min_rows = 0;
  // Row limit of the super-block path, which stays ahead of the fp32 expand
  // path up to ~10 rows (bench_decode_matmul, DD-075); the per-32 path keeps
  // int8_decode_max_rows. int8_decode_max_rows == 0 still disables int8.
  int32_t int8_superblock_max_rows = 8;
  // Sub-scaled activations (DD-077): one unit scale per 256 values and an
  // integer multiplier per 32-value block, folded into the integer sub-block
  // scale. For K-quant matmuls of at least this many rows (up to
  // int8_superblock_max_rows); 0 = never. Takes precedence over super-block.
  // Default 2: within the DD-053 accuracy contract (mean KL 0.0013 on
  // Qwen2.5-1.5B and Qwen3-4B) and +16% at 4 users; one row stays on the
  // per-32 path (memory-bound either way, single-stream outputs unchanged).
  int32_t int8_subscale_min_rows = 2;
  // Q4_K matmuls of at least this many rows use the interleaved 8-row copy
  // made at load (DD-078) when one exists; 0 = never (original layout).
  int32_t q4_repack_min_rows = 2;
  // Above the decode row limits, the interleaved copy can also serve prefill
  // batches up to this many rows for matrices with at least
  // q4_repack_min_out output rows (enough 8-row groups to fill the threads).
  // It beats fp32 expand + GEMM there (up to -26% per matmul), but prefill
  // with it fails the DD-053 perplexity limit on Qwen3-4B (+2.9-3.4%, KL
  // within limit), so it is opt-in: 0 = decode shapes only (DD-078).
  int32_t q4_repack_max_rows = 0;
  int32_t q4_repack_min_out = 1024;
  // EXPERIMENTAL (off by default): matmuls of kPackedRowsMax + 1 up to this
  // many rows (prefill chunks and mixed prefill/decode steps) on a Q4_K weight
  // with an interleaved copy (or a Q6_K one, DYNACORE_Q6_PACKED=1) and at
  // least q4_repack_min_out output rows use int16 activations (the DD-076
  // accuracy class, not int8) on that copy in 8-row tiles, instead of fp32
  // expand + GEMM. Other matmuls are unchanged. 0 = off.
  // Override: DYNACORE_INT16_PREFILL_ROWS.
  int32_t int16_prefill_max_rows = 0;
  // ... and only for weights of at least this many elements (N x K): below
  // ~3M elements the int16 tiles measured slower than fp32 expand on the
  // i7-1255U (Qwen2.5-1.5B attn_q 1536 x 1536, Qwen3-4B attn_k 1024 x 2560),
  // from ~10M faster (docs/dynakernel.md, Phase 6).
  int64_t int16_prefill_min_weights = int64_t{4} << 20;
  // EXPERIMENTAL, off by default: matmuls on an interleaved Q4_K copy with
  // more than 4 rows process up to 8 rows per kernel call (one pass over the
  // packed weights) instead of 4-row tiles. Results are identical (rows are
  // independent); only speed may change. Needs the AVX2 x8 kernel (not the
  // DYNACORE_VNNI=1 one). Override: DYNACORE_PACKED_8ROWS=1.
  bool packed_int8_8rows = false;
  // DynaLM Next candidate (off by default): where the int16 prefill path above
  // applies to a Q4_K weight, run the split-byte int16 tiles instead of the
  // int16 ones. Bit-identical results (test_matmul_registry), different speed.
  // Needs AVX-VNNI. Override: DYNACORE_INT16_PREFILL_SPLIT=1.
  bool int16_prefill_split = false;
  // DynaLM Next candidate (0 = off): in decode-only steps of at most this
  // many rows, the batch planner raises int8_superblock_max_rows to it, so the
  // int8 sub-scaled (interleaved Q4_K) and int16 FFN-down decode kernels cover
  // 9..N concurrent decoders instead of fp32 expand. Those kernels compute
  // every row independently, so each decode row gets exactly the arithmetic it
  // gets at <= 8 rows today (test_matmul_registry). Prefill and mixed steps are
  // unchanged. Override: DYNACORE_DECODE_WIDE_ROWS=16 (clamped to 0..16, the
  // range test_matmul_registry covers).
  int32_t decode_wide_max_rows = 0;
  // With wide decode (more than 8 rows on the int8 interleaved kernel), use
  // its 8-row tiles: one pass over each packed weight group per 8 rows
  // instead of per 4. Identical results (test_packed_8rows); measured faster
  // at 9-16 rows (results/dynalm-release-*/profile-wide). Rows 5-8 keep the
  // 4-row tiles unless packed_int8_8rows is set. Only reachable when
  // decode_wide_max_rows is set. Override: DYNACORE_WIDE_ROWS8=0.
  bool wide_rows8 = true;
  // Matmul kernels removed from selection, as a bitmask of
  // matmul_kernel_bit(MatmulKernelId) (matmul_registry.h); selection falls
  // through to the next eligible kernel. Override: DYNACORE_MATMUL_DISABLE=
  // name,name or all-accelerated (the fp32 reference arithmetic only).
  uint32_t matmul_disabled = 0;
  // Consult kernels registered with CpuDevice::register_external_matmul
  // (DynaKernel extension point) before the built-in ones. Off by default.
  bool external_matmul = false;

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
  // DYNACORE_GEMM_KC, DYNACORE_INT8_DECODE_ROWS, DYNACORE_INT8_SUPERBLOCK,
  // DYNACORE_INT8_SUBSCALE, DYNACORE_Q4_REPACK_ROWS, DYNACORE_Q4_REPACK_MAX_ROWS,
  // DYNACORE_PACKED_8ROWS, DYNACORE_INT16_PREFILL_ROWS, DYNACORE_INT16_PREFILL_SPLIT,
  // DYNACORE_MATMUL_DISABLE, DYNACORE_INT16_FFN_DOWN, DYNACORE_ATTN_GROUPED and
  // DYNACORE_MATMUL_CHUNKS applied (read once per process).
  static const KernelPlan& defaults();

  friend bool operator==(const KernelPlan&, const KernelPlan&) = default;
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

}  // namespace dynacore

#pragma once

// Innermost CPU primitives, selected once per ISA tier.
//
// Every entry has a portable generic implementation; ISA tiers (AVX2 today)
// override the entries they accelerate. All kernels keep fp32 activations
// and fp32 accumulation, so tiers differ only in summation order (tested
// against the generic tier and the dequantize reference).
//
// SIMD code lives only in backends/cpu/<isa>/*.cpp, compiled with per-file
// ISA flags (DD-003); nothing here may be inlined into generic code.

#include <array>
#include <cstdint>

#include "dynacore/tensor/dtype.h"
#include "dynacore/hardware/isa.h"

namespace dynacore {

// Dot product of one weight row (`n` elements of the slot's dtype, packed
// blocks for quantized types) with fp32 activations.
using VecDotFn = float (*)(const void* w, const float* x, int64_t n);
// Row conversion of `n` elements of the slot's dtype to fp32.
using DequantFn = void (*)(const void* w, float* out, int64_t n);

// --- int8 activation path for decode (P3, DD-053) ---
// An activation row quantized in blocks of 32: x[i] ~= d * q[i], with q in
// [-127, 127] and `sum` = sum(q) (folds the weight formats' offsets and minima
// into one integer term).
struct ActBlockQ8 {
  float d;
  int32_t sum;
  int8_t q[32];
};
static_assert(sizeof(ActBlockQ8) == 40);
inline constexpr int kActQ8Block = 32;
// Quantizes n (multiple of 32) fp32 values into n / 32 blocks.
using QuantizeActFn = void (*)(const float* x, ActBlockQ8* out, int64_t n);
// out[r] = dot(weight row w, activation row x[r]) for r < m (1 <= m <= 4):
// each weight block is unpacked once and reused for the m rows.
inline constexpr int kDotRowsMax = 4;
using DotQ8RowsFn = void (*)(const void* w, const ActBlockQ8* const* x, int m, int64_t n, float* out);

// --- Sub-scaled int8 activations (DD-077) ---
// 256 values: x[i] ~= u * m[i / 32] * q[i], with u = amax / (127 * 128) and
// m in 1..128 the smallest multiplier that holds the block's maximum, so a
// block within 1/16 of the super-block maximum loses at most 12.5% resolution
// against its own per-32 scale. ms[b] = u * m[b] * sum(q of block b) serves
// the weight formats' minimum terms. Exactly the size of 8 ActBlockQ8.
struct ActSuperQ8 {
  int8_t q[256];
  float u;
  int16_t m[8];
  float ms[8];
  int32_t pad[3];
};
static_assert(sizeof(ActSuperQ8) == 8 * sizeof(ActBlockQ8));
inline constexpr int kActSxMaxMult = 128;

// --- int16 activation path (DD-076) ---
// For matmuls whose input carries outlier channels (the FFN down projection),
// where int8 activations break the accuracy contract: 32-value blocks of
// int16 codes (x ~= d * q, |q| <= 32767) and the code sum. madd_epi16 still
// does 16 multiply-adds per instruction against 8 for an fp32 FMA.
struct ActBlockQ16 {
  float d;
  int32_t sum;
  int16_t q[32];
};
static_assert(sizeof(ActBlockQ16) == 72);
using QuantizeAct16Fn = void (*)(const float* x, ActBlockQ16* out, int64_t n);
using DotQ16RowsFn = void (*)(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out);

struct CpuKernels {
  CpuIsa isa = CpuIsa::kGeneric;
  float (*dot_f32)(const float* a, const float* b, int64_t n) = nullptr;
  // y += a * x
  void (*axpy_f32)(float a, const float* x, float* y, int64_t n) = nullptr;
  // fp16 KV helpers used by attention.
  float (*dot_f16_f32)(const uint16_t* a, const float* b, int64_t n) = nullptr;
  void (*axpy_f16)(float a, const uint16_t* x, float* y, int64_t n) = nullptr;

  // Register-blocked GEMM panel: y[i * y_stride + r] (+)= dot(w + r*k, x + i*x_stride)
  // for r < nr (nr <= 4 weight rows, contiguous with stride k) and i < m
  // activation rows; `accumulate` adds to y instead of overwriting (K-blocking).
  // Used for prefill (many rows) after expanding weights.
  void (*gemm_panel)(const float* w, int nr, const float* x, int64_t x_stride, int64_t m, int64_t k, float* y,
                     int64_t y_stride, bool accumulate) = nullptr;

  // Per weight dtype; every supported dtype has both entries after
  // registration of the generic tier.
  std::array<VecDotFn, static_cast<size_t>(DType::kCount)> vec_dot{};
  std::array<DequantFn, static_cast<size_t>(DType::kCount)> dequant{};

  // Block-wise attention over n contiguous KV positions of one head (P7,
  // DD-056): scores[t] = scale * dot(q, k[t]); acc += sum_t w[t] * v[t].
  // k/v rows are `dim` elements apart (fp16 or fp32 KV).
  void (*attn_scores_f16)(const uint16_t* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) = nullptr;
  void (*attn_accum_f16)(const uint16_t* v, int64_t n, int32_t dim, const float* w, float* acc) = nullptr;
  void (*attn_scores_f32)(const float* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) = nullptr;
  void (*attn_accum_f32)(const float* v, int64_t n, int32_t dim, const float* w, float* acc) = nullptr;
  // Scores of nh query heads sharing one KV head (GQA, DD-066): q holds nh
  // heads `dim` apart; scores[h * s_stride + t] = scale * dot(q_h, k[t]).
  // Each k vector is loaded once for all heads.
  void (*attn_scores_heads_f32)(const float* k, int64_t n, int32_t dim, const float* q, int32_t nh, float scale,
                                float* scores, int64_t s_stride) = nullptr;
  // acc[h * dim + i] += sum_t w[h * w_stride + t] * v[t][i] for nh heads
  // sharing one KV head: each v vector is loaded once for all heads.
  void (*attn_accum_heads_f32)(const float* v, int64_t n, int32_t dim, const float* w, int32_t nh, int64_t w_stride,
                               float* acc) = nullptr;

  // int8 activation path: entries are null for types without an integer kernel
  // (the backend then uses vec_dot / dequant).
  QuantizeActFn quantize_act = nullptr;
  std::array<DotQ8RowsFn, static_cast<size_t>(DType::kCount)> dot_q8_rows{};
  // Super-block variant for K-quants (DD-075): quantize_act_sb gives the 8
  // blocks of every 256 values one shared scale (n must be a multiple of 256),
  // so dot_q8_sb_rows can sum the integer sub-block scales into one int32
  // accumulator and convert to float once per super-block instead of once
  // per 32 values. dot_q8_sb_rows requires activations from quantize_act_sb.
  QuantizeActFn quantize_act_sb = nullptr;
  std::array<DotQ8RowsFn, static_cast<size_t>(DType::kCount)> dot_q8_sb_rows{};
  // Sub-scaled variant (DD-077): quantize_act_sx writes one ActSuperQ8 per
  // 256 values (n a multiple of 256) into the space of 8 ActBlockQ8, so
  // buffers and row strides are unchanged. dot_q8_sx_rows folds the per-block
  // multipliers into the integer sub-block scales and accumulates a
  // super-block in int32. It requires activations from quantize_act_sx; the
  // ActBlockQ8 pointers it receives are reinterpreted as ActSuperQ8.
  QuantizeActFn quantize_act_sx = nullptr;
  std::array<DotQ8RowsFn, static_cast<size_t>(DType::kCount)> dot_q8_sx_rows{};
  // Interleaved Q4_K (DD-078): out[r * 8 + i] = dot(packed row i, x[r]) for
  // the 8 rows of a BlockQ4_Kx8 group (w: k / 256 consecutive groups) and
  // r < m (1 <= m <= 4), with sub-scaled activations from quantize_act_sx.
  using DotQ8X8Fn = void (*)(const void* w, const ActBlockQ8* const* x, int m, int64_t n, float* out);
  DotQ8X8Fn dot_q8_sx_x8_q4_K = nullptr;
  // Same layout with int16 activations (DD-081): out[r * 8 + i] for the 8 rows
  // of a BlockQ4_Kx8 group, activations from quantize_act16, 1 <= m <= 4.
  using DotQ16X8Fn = void (*)(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out);
  DotQ16X8Fn dot_q16_x8_q4_K = nullptr;
  DotQ16X8Fn dot_q16_x8_q6_K = nullptr;  // BlockQ6_Kx8 groups (DD-082)
  // int16 activations (DD-076); null where no tier has a kernel.
  QuantizeAct16Fn quantize_act16 = nullptr;
  std::array<DotQ16RowsFn, static_cast<size_t>(DType::kCount)> dot_q16_rows{};

  VecDotFn vec_dot_for(DType t) const { return vec_dot[static_cast<size_t>(t)]; }
  DequantFn dequant_for(DType t) const { return dequant[static_cast<size_t>(t)]; }
  DotQ8RowsFn dot_q8_rows_for(DType t) const { return dot_q8_rows[static_cast<size_t>(t)]; }
  DotQ8RowsFn dot_q8_sb_rows_for(DType t) const { return dot_q8_sb_rows[static_cast<size_t>(t)]; }
  DotQ16RowsFn dot_q16_rows_for(DType t) const { return dot_q16_rows[static_cast<size_t>(t)]; }
  DotQ8RowsFn dot_q8_sx_rows_for(DType t) const { return dot_q8_sx_rows[static_cast<size_t>(t)]; }
};

// Fills every entry with the portable implementation.
void register_generic_kernels(CpuKernels& k);
// Overrides entries with AVX2/FMA/F16C versions. Returns false (and changes
// nothing) when the AVX2 tier was not compiled in.
bool register_avx2_kernels(CpuKernels& k);
// AVX-VNNI kernels on top of the AVX2 table (DD-080); false when not compiled
// in. The caller checks the CPU feature.
bool register_avxvnni_kernels(CpuKernels& k);
// ARM64 NEON tier (Apple Silicon, Graviton, Windows on ARM); false when not compiled in.
bool register_neon_kernels(CpuKernels& k);

// Best available table for `isa` (falls back to lower tiers).
CpuKernels make_cpu_kernels(CpuIsa isa);

}  // namespace dynacore

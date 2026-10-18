// ARM64 int8 activation path (DD-053) with the dot-product instructions
// (SDOT, FEAT_DotProd: Apple M1+, Cortex-A55/A76+, Neoverse, Snapdragon X).
// Compiled with dot-product flags (engine_set_isa NEONDOT) and registered only
// when the CPU reports the feature at run time (cpu_info().features.arm_dotprod),
// so a build runs on ARM cores without it (Cortex-A72, Raspberry Pi 4).
//
// Same arithmetic as the generic reference (kernels_generic.cpp): activations
// quantized per 32 values (d = amax / 127, round to nearest even, sum of
// codes), integer sub-block dot products summed exactly in int32, then scaled
// once per sub-block. Only the fp32 accumulation order differs from the
// generic tier, as with the AVX2 tier. Each weight block is unpacked once and
// used for up to kDotRowsMax activation rows.

#include "dynacore/cpu/cpu_kernels.h"

#if ENGINE_HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

#include <algorithm>
#include <cstring>

#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

namespace dynacore {
namespace {

using namespace quant;

// One block of 32 values, as kernels_generic.cpp quantize_act: v = x * (1/d)
// clamped to +-127, rounded to nearest (ties to even, FCVTNS).
void quantize_act(const float* x, ActBlockQ8* out, int64_t n) {
  for (int64_t b = 0; b < n / kActQ8Block; ++b, x += kActQ8Block) {
    float32x4_t v[8];
    float32x4_t mx = vdupq_n_f32(0);
    for (int i = 0; i < 8; ++i) {
      v[i] = vld1q_f32(x + 4 * i);
      mx = vmaxq_f32(mx, vabsq_f32(v[i]));
    }
    const float d = vmaxvq_f32(mx) / 127.0f;
    const float id = d > 0 ? 1.0f / d : 0.0f;
    const float32x4_t lo = vdupq_n_f32(-127.0f), hi = vdupq_n_f32(127.0f);
    int32x4_t s = vdupq_n_s32(0);
    int16x8_t h[4];
    for (int i = 0; i < 4; ++i) {
      const int32x4_t a = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v[2 * i], id), lo), hi));
      const int32x4_t c = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v[2 * i + 1], id), lo), hi));
      s = vaddq_s32(s, vaddq_s32(a, c));
      h[i] = vcombine_s16(vmovn_s32(a), vmovn_s32(c));
    }
    vst1q_s8(out[b].q, vcombine_s8(vmovn_s16(h[0]), vmovn_s16(h[1])));
    vst1q_s8(out[b].q + 16, vcombine_s8(vmovn_s16(h[2]), vmovn_s16(h[3])));
    out[b].d = d;
    out[b].sum = vaddvq_s32(s);
  }
}

// Exact int32 dot of 32 weight codes (two int8x16) with an activation block.
inline int32_t idot32(int8x16_t w0, int8x16_t w1, const ActBlockQ8& a) {
  int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(a.q));
  s = vdotq_s32(s, w1, vld1q_s8(a.q + 16));
  return vaddvq_s32(s);
}

// Exact int32 dot of 16 weight codes with half `h` of an activation block.
inline int32_t idot16(int8x16_t w, const ActBlockQ8& a, int h) {
  return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(a.q + 16 * h)));
}

template <int M>
void dot_rows_q8_0_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK; ++i) {
    const int8x16_t w0 = vld1q_s8(b[i].qs), w1 = vld1q_s8(b[i].qs + 16);
    const float dw = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) acc[r] += dw * x[r][i].d * static_cast<float>(idot32(w0, w1, x[r][i]));
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

template <int M>
void dot_rows_q4_0_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);
  const int8x16_t eight = vdupq_n_s8(8);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK; ++i) {
    const uint8x16_t bytes = vld1q_u8(b[i].qs);
    // Values 0..15 are the low nibbles, 16..31 the high ones, offset by -8.
    const int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, low4)), eight);
    const int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), eight);
    const float dw = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) acc[r] += dw * x[r][i].d * static_cast<float>(idot32(w0, w1, x[r][i]));
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

// Q4_K: sub-block j (32 values) = d * sc_j * q - dmin * m_j, q in 0..15; the
// low nibbles of qs[32 * (j / 2) ..] hold even j, the high nibbles odd j.
template <int M>
void dot_rows_q4_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    for (int jp = 0; jp < 4; ++jp) {
      const uint8x16_t q0 = vld1q_u8(b[i].qs + 32 * jp), q1 = vld1q_u8(b[i].qs + 32 * jp + 16);
      const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, low4)), lo1 = vreinterpretq_s8_u8(vandq_u8(q1, low4));
      const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
      uint8_t sc0, mn0, sc1, mn1;
      get_scale_min_k4(2 * jp, b[i].scales, sc0, mn0);
      get_scale_min_k4(2 * jp + 1, b[i].scales, sc1, mn1);
      const int64_t blk = i * 8 + 2 * jp;
      for (int r = 0; r < M; ++r) {
        const ActBlockQ8& a0 = x[r][blk];
        const ActBlockQ8& a1 = x[r][blk + 1];
        acc[r] += a0.d * (d * sc0 * static_cast<float>(idot32(lo0, lo1, a0)) - dmin * mn0 * static_cast<float>(a0.sum));
        acc[r] += a1.d * (d * sc1 * static_cast<float>(idot32(hi0, hi1, a1)) - dmin * mn1 * static_cast<float>(a1.sum));
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

// Q6_K: value e of a super-block = d * scales[e / 16] * (q - 32), q in 0..63;
// each 32-value activation block spans two 16-value scales.
template <int M>
void dot_rows_q6_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const uint8x16_t low4 = vdupq_n_u8(0x0F), low2 = vdupq_n_u8(0x03);
  const int8x16_t k32 = vdupq_n_s8(32);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    // The 256 codes as 16 int8x16 vectors in value order (generic layout).
    int8x16_t q[16];
    for (int p = 0; p < 2; ++p) {
      const uint8_t* ql = b[i].ql + 64 * p;
      const uint8_t* qh = b[i].qh + 32 * p;
      for (int half = 0; half < 2; ++half) {  // l in [16 * half, 16 * half + 16)
        const uint8x16_t l0 = vld1q_u8(ql + 16 * half), l1 = vld1q_u8(ql + 32 + 16 * half);
        const uint8x16_t h = vld1q_u8(qh + 16 * half);
        const uint8x16_t c0 = vorrq_u8(vandq_u8(l0, low4), vshlq_n_u8(vandq_u8(h, low2), 4));
        const uint8x16_t c1 = vorrq_u8(vandq_u8(l1, low4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), low2), 4));
        const uint8x16_t c2 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), low2), 4));
        const uint8x16_t c3 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4));
        q[8 * p + 0 + half] = vsubq_s8(vreinterpretq_s8_u8(c0), k32);
        q[8 * p + 2 + half] = vsubq_s8(vreinterpretq_s8_u8(c1), k32);
        q[8 * p + 4 + half] = vsubq_s8(vreinterpretq_s8_u8(c2), k32);
        q[8 * p + 6 + half] = vsubq_s8(vreinterpretq_s8_u8(c3), k32);
      }
    }
    for (int g = 0; g < 8; ++g) {
      const float s0 = d * b[i].scales[2 * g], s1 = d * b[i].scales[2 * g + 1];
      for (int r = 0; r < M; ++r) {
        const ActBlockQ8& a = x[r][i * 8 + g];
        acc[r] += a.d * s0 * static_cast<float>(idot16(q[2 * g], a, 0));
        acc[r] += a.d * s1 * static_cast<float>(idot16(q[2 * g + 1], a, 1));
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

#define DYNACORE_ROWS_DISPATCH(name)                                                       \
  void name(const void* w, const ActBlockQ8* const* x, int m, int64_t n, float* out) {     \
    switch (m) {                                                                         \
      case 1: name##_m<1>(w, x, n, out); break;                                          \
      case 2: name##_m<2>(w, x, n, out); break;                                          \
      case 3: name##_m<3>(w, x, n, out); break;                                          \
      default: name##_m<4>(w, x, n, out); break;                                         \
    }                                                                                    \
  }
DYNACORE_ROWS_DISPATCH(dot_rows_q8_0)
DYNACORE_ROWS_DISPATCH(dot_rows_q4_0)
DYNACORE_ROWS_DISPATCH(dot_rows_q4_K)
DYNACORE_ROWS_DISPATCH(dot_rows_q6_K)
#undef DYNACORE_ROWS_DISPATCH

// --- int16 activation path (DD-076): the FFN down projection -----------------
// As the AVX2 tier: x ~= d * q with |q| <= 32767 per 32 values; the weight
// code times its integer sub-block scale (int16: Q4_K <= 945, Q6_K <= 4064)
// times q, summed exactly in int32 lanes of 8 products (< 1.1e9), then scaled
// once per block. SMLAL (int16 x int16 -> int32) does the multiply-adds.

void quantize_act16(const float* x, ActBlockQ16* out, int64_t n) {
  for (int64_t b = 0; b < n / kActQ8Block; ++b, x += kActQ8Block) {
    float32x4_t v[8];
    float32x4_t mx = vdupq_n_f32(0);
    for (int i = 0; i < 8; ++i) {
      v[i] = vld1q_f32(x + 4 * i);
      mx = vmaxq_f32(mx, vabsq_f32(v[i]));
    }
    const float d = vmaxvq_f32(mx) / 32767.0f;
    const float id = d > 0 ? 1.0f / d : 0.0f;
    const float32x4_t lo = vdupq_n_f32(-32767.0f), hi = vdupq_n_f32(32767.0f);
    int32x4_t s = vdupq_n_s32(0);
    for (int i = 0; i < 4; ++i) {
      const int32x4_t a = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v[2 * i], id), lo), hi));
      const int32x4_t c = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v[2 * i + 1], id), lo), hi));
      s = vaddq_s32(s, vaddq_s32(a, c));
      vst1q_s16(out[b].q + 8 * i, vcombine_s16(vmovn_s32(a), vmovn_s32(c)));
    }
    out[b].d = d;
    out[b].sum = vaddvq_s32(s);
  }
}

// Exact sum of 16 int16 weights times 16 int16 activations (two int32x4 halves).
inline int32x4_t mlal16(int32x4_t acc, int16x8_t w0, int16x8_t w1, const int16_t* a) {
  const int16x8_t a0 = vld1q_s16(a), a1 = vld1q_s16(a + 8);
  acc = vmlal_s16(acc, vget_low_s16(w0), vget_low_s16(a0));
  acc = vmlal_high_s16(acc, w0, a0);
  acc = vmlal_s16(acc, vget_low_s16(w1), vget_low_s16(a1));
  return vmlal_high_s16(acc, w1, a1);
}

// 16 int8 codes -> two int16x8, each times `scale` (exact in int16 here).
inline void widen_scaled(int8x16_t q, int16_t scale, int16x8_t& w0, int16x8_t& w1) {
  w0 = vmulq_n_s16(vmovl_s8(vget_low_s8(q)), scale);
  w1 = vmulq_n_s16(vmovl_high_s8(q), scale);
}

template <int M>
void dot_rows16_q8_0_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK; ++i) {
    int16x8_t w0, w1, w2, w3;
    widen_scaled(vld1q_s8(b[i].qs), 1, w0, w1);
    widen_scaled(vld1q_s8(b[i].qs + 16), 1, w2, w3);
    const float d = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) {
      const ActBlockQ16& a = x[r][i];
      const int32x4_t p = mlal16(mlal16(vdupq_n_s32(0), w0, w1, a.q), w2, w3, a.q + 16);
      acc[r] += d * a.d * static_cast<float>(vaddvq_s32(p));
    }
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

template <int M>
void dot_rows16_q4_K_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; ++j) get_scale_min_k4(j, b[i].scales, sc[j], mn[j]);
    for (int jp = 0; jp < 4; ++jp) {
      const uint8x16_t q0 = vld1q_u8(b[i].qs + 32 * jp), q1 = vld1q_u8(b[i].qs + 32 * jp + 16);
      int16x8_t l[4], h[4];
      widen_scaled(vreinterpretq_s8_u8(vandq_u8(q0, low4)), sc[2 * jp], l[0], l[1]);
      widen_scaled(vreinterpretq_s8_u8(vandq_u8(q1, low4)), sc[2 * jp], l[2], l[3]);
      widen_scaled(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), sc[2 * jp + 1], h[0], h[1]);
      widen_scaled(vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), sc[2 * jp + 1], h[2], h[3]);
      const int64_t blk = i * 8 + 2 * jp;
      for (int r = 0; r < M; ++r) {
        const ActBlockQ16& a0 = x[r][blk];
        const ActBlockQ16& a1 = x[r][blk + 1];
        const int32x4_t p0 = mlal16(mlal16(vdupq_n_s32(0), l[0], l[1], a0.q), l[2], l[3], a0.q + 16);
        const int32x4_t p1 = mlal16(mlal16(vdupq_n_s32(0), h[0], h[1], a1.q), h[2], h[3], a1.q + 16);
        acc[r] += d * a0.d * static_cast<float>(vaddvq_s32(p0)) + d * a1.d * static_cast<float>(vaddvq_s32(p1));
        acc[r] -= dmin * (mn[2 * jp] * a0.d * static_cast<float>(a0.sum) + mn[2 * jp + 1] * a1.d * static_cast<float>(a1.sum));
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

template <int M>
void dot_rows16_q6_K_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const uint8x16_t low4 = vdupq_n_u8(0x0F), low2 = vdupq_n_u8(0x03);
  const int8x16_t k32 = vdupq_n_s8(32);
  float acc[M] = {};
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    int8x16_t q[16];  // the 256 codes in value order, minus 32
    for (int p = 0; p < 2; ++p) {
      const uint8_t* ql = b[i].ql + 64 * p;
      const uint8_t* qh = b[i].qh + 32 * p;
      for (int half = 0; half < 2; ++half) {
        const uint8x16_t l0 = vld1q_u8(ql + 16 * half), l1 = vld1q_u8(ql + 32 + 16 * half);
        const uint8x16_t h = vld1q_u8(qh + 16 * half);
        q[8 * p + 0 + half] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, low4), vshlq_n_u8(vandq_u8(h, low2), 4))), k32);
        q[8 * p + 2 + half] =
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, low4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), low2), 4))), k32);
        q[8 * p + 4 + half] =
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), low2), 4))), k32);
        q[8 * p + 6 + half] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4))), k32);
      }
    }
    for (int g = 0; g < 8; ++g) {
      int16x8_t w0, w1, w2, w3;
      widen_scaled(q[2 * g], b[i].scales[2 * g], w0, w1);
      widen_scaled(q[2 * g + 1], b[i].scales[2 * g + 1], w2, w3);
      for (int r = 0; r < M; ++r) {
        const ActBlockQ16& a = x[r][i * 8 + g];
        const int32x4_t p = mlal16(mlal16(vdupq_n_s32(0), w0, w1, a.q), w2, w3, a.q + 16);
        acc[r] += d * a.d * static_cast<float>(vaddvq_s32(p));
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = acc[r];
}

#define DYNACORE_ROWS16_DISPATCH(name)                                                      \
  void name(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {     \
    switch (m) {                                                                          \
      case 1: name##_m<1>(w, x, n, out); break;                                           \
      case 2: name##_m<2>(w, x, n, out); break;                                           \
      case 3: name##_m<3>(w, x, n, out); break;                                           \
      default: name##_m<4>(w, x, n, out); break;                                          \
    }                                                                                     \
  }
DYNACORE_ROWS16_DISPATCH(dot_rows16_q8_0)
DYNACORE_ROWS16_DISPATCH(dot_rows16_q4_K)
DYNACORE_ROWS16_DISPATCH(dot_rows16_q6_K)
#undef DYNACORE_ROWS16_DISPATCH

}  // namespace

bool register_neon_dot_kernels(CpuKernels& k) {
  k.quantize_act16 = quantize_act16;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ8_0)] = dot_rows16_q8_0;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows16_q4_K;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows16_q6_K;
  k.quantize_act = quantize_act;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ8_0)] = dot_rows_q8_0;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ4_0)] = dot_rows_q4_0;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows_q4_K;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows_q6_K;
  k.int8_simd = true;
  return true;
}

}  // namespace dynacore

#else

namespace dynacore {
bool register_neon_dot_kernels(CpuKernels&) { return false; }
}  // namespace dynacore

#endif

// ARM64 NEON kernels (Apple Silicon, AWS Graviton, Ampere, Windows on ARM).
// NEON is mandatory on AArch64, so no per-file flags or runtime checks are
// needed beyond the architecture. Same structure as the AVX2 tier (DD-046):
// fp32 activations and accumulation; quantized blocks are unpacked to int8
// with vector bit operations and widened to fp32 4 lanes at a time.

#include "dynacore/cpu/cpu_kernels.h"

#if ENGINE_HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))

#include <arm_neon.h>

#include <cstring>

#include "dynacore/tensor/fp16.h"
#include "dynacore/quantization/quant_formats.h"

namespace engine {
namespace {

using namespace quant;

inline float hsum(float32x4_t v) { return vaddvq_f32(v); }

// 8 int8 values -> two float32x4.
inline void widen8(const int8_t* q, float32x4_t& lo, float32x4_t& hi) {
  const int16x8_t w = vmovl_s8(vld1_s8(q));
  lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
  hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
}

// Σ q[i] * x[i] over 16 / 32 int8 weights, as a 4-lane vector.
inline float32x4_t dot16_i8(const int8_t* q, const float* x) {
  float32x4_t a, b, c, d;
  widen8(q, a, b);
  widen8(q + 8, c, d);
  float32x4_t s = vmulq_f32(a, vld1q_f32(x));
  s = vfmaq_f32(s, b, vld1q_f32(x + 4));
  s = vfmaq_f32(s, c, vld1q_f32(x + 8));
  return vfmaq_f32(s, d, vld1q_f32(x + 12));
}
inline float32x4_t dot32_i8(const int8_t* q, const float* x) {
  return vaddq_f32(dot16_i8(q, x), dot16_i8(q + 16, x + 16));
}
inline float32x4_t sum32(const float* x) {
  float32x4_t s = vaddq_f32(vld1q_f32(x), vld1q_f32(x + 4));
  for (int i = 8; i < 32; i += 4) s = vaddq_f32(s, vld1q_f32(x + i));
  return s;
}

// --- float kernels ------------------------------------------------------------

float dot_f32(const float* a, const float* b, int64_t n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = hsum(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

void axpy_f32(float a, const float* x, float* y, int64_t n) {
  const float32x4_t va = vdupq_n_f32(a);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
  for (; i < n; ++i) y[i] += a * x[i];
}

inline float32x4_t load_f16x4(const uint16_t* p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }
inline float32x4_t load_bf16x4(const uint16_t* p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}

float dot_f16_f32(const uint16_t* a, const float* b, int64_t n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 = vfmaq_f32(s0, load_f16x4(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, load_f16x4(a + i + 4), vld1q_f32(b + i + 4));
  }
  float s = hsum(vaddq_f32(s0, s1));
  for (; i < n; ++i) s += fp16_to_fp32(a[i]) * b[i];
  return s;
}

void axpy_f16(float a, const uint16_t* x, float* y, int64_t n) {
  const float32x4_t va = vdupq_n_f32(a);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, load_f16x4(x + i)));
  for (; i < n; ++i) y[i] += a * fp16_to_fp32(x[i]);
}

float vec_dot_f32(const void* w, const float* x, int64_t n) { return dot_f32(static_cast<const float*>(w), x, n); }
float vec_dot_f16(const void* w, const float* x, int64_t n) {
  return dot_f16_f32(static_cast<const uint16_t*>(w), x, n);
}
float vec_dot_bf16(const void* w, const float* x, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  float32x4_t s = vdupq_n_f32(0);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) s = vfmaq_f32(s, load_bf16x4(a + i), vld1q_f32(x + i));
  float r = hsum(s);
  for (; i < n; ++i) r += bf16_to_fp32(a[i]) * x[i];
  return r;
}

void dequant_f16(const void* w, float* out, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, load_f16x4(a + i));
  for (; i < n; ++i) out[i] = fp16_to_fp32(a[i]);
}
void dequant_bf16(const void* w, float* out, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, load_bf16x4(a + i));
  for (; i < n; ++i) out[i] = bf16_to_fp32(a[i]);
}

// 4 weight rows x 2 activation rows per pass over k (8 accumulators).
void gemm_panel(const float* w, int nr, const float* x, int64_t x_stride, int64_t m, int64_t k, float* y,
                int64_t y_stride, bool accumulate) {
  auto put = [accumulate](float& dst, float v) { dst = accumulate ? dst + v : v; };
  if (nr < 4 || k % 4 != 0) {
    for (int64_t i = 0; i < m; ++i) {
      for (int r = 0; r < nr; ++r) put(y[i * y_stride + r], dot_f32(w + r * k, x + i * x_stride, k));
    }
    return;
  }
  const float* w0 = w;
  const float* w1 = w + k;
  const float* w2 = w + 2 * k;
  const float* w3 = w + 3 * k;
  int64_t i = 0;
  for (; i + 2 <= m; i += 2) {
    const float* xa = x + i * x_stride;
    const float* xb = xa + x_stride;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (int64_t kk = 0; kk < k; kk += 4) {
      const float32x4_t va = vld1q_f32(xa + kk), vb = vld1q_f32(xb + kk);
      float32x4_t vw = vld1q_f32(w0 + kk);
      a0 = vfmaq_f32(a0, vw, va);
      b0 = vfmaq_f32(b0, vw, vb);
      vw = vld1q_f32(w1 + kk);
      a1 = vfmaq_f32(a1, vw, va);
      b1 = vfmaq_f32(b1, vw, vb);
      vw = vld1q_f32(w2 + kk);
      a2 = vfmaq_f32(a2, vw, va);
      b2 = vfmaq_f32(b2, vw, vb);
      vw = vld1q_f32(w3 + kk);
      a3 = vfmaq_f32(a3, vw, va);
      b3 = vfmaq_f32(b3, vw, vb);
    }
    float* ya = y + i * y_stride;
    float* yb = ya + y_stride;
    put(ya[0], hsum(a0));
    put(ya[1], hsum(a1));
    put(ya[2], hsum(a2));
    put(ya[3], hsum(a3));
    put(yb[0], hsum(b0));
    put(yb[1], hsum(b1));
    put(yb[2], hsum(b2));
    put(yb[3], hsum(b3));
  }
  for (; i < m; ++i) {
    for (int r = 0; r < 4; ++r) put(y[i * y_stride + r], dot_f32(w + r * k, x + i * x_stride, k));
  }
}

// --- quantized: unpack to int8 -------------------------------------------------

// Q4_0 / Q4_1 nibbles: low halves are elements 0..15, high halves 16..31.
inline void unpack_q4(const uint8_t* qs, int8_t* q, int8_t bias) {
  const uint8x16_t b = vld1q_u8(qs);
  const int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(0x0F)));
  const int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
  vst1q_s8(q, vsubq_s8(lo, vdupq_n_s8(bias)));
  vst1q_s8(q + 16, vsubq_s8(hi, vdupq_n_s8(bias)));
}

inline void unpack_q5_0(const BlockQ5_0& blk, int8_t* q) {
  uint32_t qh;
  std::memcpy(&qh, blk.qh, 4);
  for (int j = 0; j < 16; ++j) {
    const int lo = (blk.qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
    const int hi = (blk.qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
    q[j] = static_cast<int8_t>(lo - 16);
    q[j + 16] = static_cast<int8_t>(hi - 16);
  }
}

float vec_dot_q8_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    acc = vfmaq_n_f32(acc, dot32_i8(b[i].qs, x), fp16_to_fp32(b[i].d));
  }
  return hsum(acc);
}

float vec_dot_q4_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  alignas(16) int8_t q[32];
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    unpack_q4(b[i].qs, q, 8);
    acc = vfmaq_n_f32(acc, dot32_i8(q, x), fp16_to_fp32(b[i].d));
  }
  return hsum(acc);
}

float vec_dot_q4_1(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_1*>(w);
  alignas(16) int8_t q[32];
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    unpack_q4(b[i].qs, q, 0);
    acc = vfmaq_n_f32(acc, dot32_i8(q, x), fp16_to_fp32(b[i].d));
    acc = vfmaq_n_f32(acc, sum32(x), fp16_to_fp32(b[i].m));
  }
  return hsum(acc);
}

float vec_dot_q5_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  alignas(16) int8_t q[32];
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    unpack_q5_0(b[i], q);
    acc = vfmaq_n_f32(acc, dot32_i8(q, x), fp16_to_fp32(b[i].d));
  }
  return hsum(acc);
}

float vec_dot_q4_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  alignas(16) int8_t lo[32], hi[32];
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const uint8_t* qs = b[i].qs;
    for (int is = 0; is < 8; is += 2, qs += 32, x += 64) {
      for (int h = 0; h < 32; h += 16) {
        const uint8x16_t bytes = vld1q_u8(qs + h);
        vst1q_s8(lo + h, vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0F))));
        vst1q_s8(hi + h, vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)));
      }
      uint8_t sc, m;
      get_scale_min_k4(is, b[i].scales, sc, m);
      acc = vfmaq_n_f32(acc, dot32_i8(lo, x), d * sc);
      acc = vfmsq_n_f32(acc, sum32(x), dmin * m);
      get_scale_min_k4(is + 1, b[i].scales, sc, m);
      acc = vfmaq_n_f32(acc, dot32_i8(hi, x + 32), d * sc);
      acc = vfmsq_n_f32(acc, sum32(x + 32), dmin * m);
    }
  }
  return hsum(acc);
}

// Q6_K: 4 groups of 32 per 128-element half; scales per 16 elements.
inline void unpack_q6_K_half(const uint8_t* ql, const uint8_t* qh, int8_t q[4][32]) {
  for (int h = 0; h < 32; h += 16) {
    const uint8x16_t l0 = vld1q_u8(ql + h), l1 = vld1q_u8(ql + 32 + h), hb = vld1q_u8(qh + h);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(0x03);
    const int8x16_t k32 = vdupq_n_s8(32);
    auto hbits = [&](int shift) { return vshlq_n_u8(vandq_u8(vshlq_u8(hb, vdupq_n_s8(static_cast<int8_t>(-shift))), m2), 4); };
    vst1q_s8(q[0] + h, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4), hbits(0))), k32));
    vst1q_s8(q[1] + h, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4), hbits(2))), k32));
    vst1q_s8(q[2] + h, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), hbits(4))), k32));
    vst1q_s8(q[3] + h, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), hbits(6))), k32));
  }
}

float vec_dot_q6_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  alignas(16) int8_t q[4][32];
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    const uint8_t* ql = b[i].ql;
    const uint8_t* qh = b[i].qh;
    const int8_t* sc = b[i].scales;
    for (int part = 0; part < 2; ++part, ql += 64, qh += 32, sc += 8, x += 128) {
      unpack_q6_K_half(ql, qh, q);
      for (int g = 0; g < 4; ++g) {
        acc = vfmaq_n_f32(acc, dot16_i8(q[g], x + 32 * g), d * sc[2 * g]);
        acc = vfmaq_n_f32(acc, dot16_i8(q[g] + 16, x + 32 * g + 16), d * sc[2 * g + 1]);
      }
    }
  }
  return hsum(acc);
}

// --- dequantization (expand path for batched matmuls) ------------------------

// out[0..n) = d * q + mn, n a multiple of 8.
inline void store_scaled(const int8_t* q, int n, float d, float mn, float* out) {
  const float32x4_t vm = vdupq_n_f32(mn);
  for (int i = 0; i < n; i += 8) {
    float32x4_t a, b;
    widen8(q + i, a, b);
    vst1q_f32(out + i, vfmaq_n_f32(vm, a, d));
    vst1q_f32(out + i + 4, vfmaq_n_f32(vm, b, d));
  }
}

void dequant_q8_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) store_scaled(b[i].qs, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
}
void dequant_q4_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  alignas(16) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    unpack_q4(b[i].qs, q, 8);
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
  }
}
void dequant_q4_1(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_1*>(w);
  alignas(16) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    unpack_q4(b[i].qs, q, 0);
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), fp16_to_fp32(b[i].m), out);
  }
}
void dequant_q5_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  alignas(16) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    unpack_q5_0(b[i], q);
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
  }
}
void dequant_q4_K(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  alignas(16) int8_t lo[32], hi[32];
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const uint8_t* qs = b[i].qs;
    for (int is = 0; is < 8; is += 2, qs += 32, out += 64) {
      for (int h = 0; h < 32; h += 16) {
        const uint8x16_t bytes = vld1q_u8(qs + h);
        vst1q_s8(lo + h, vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0F))));
        vst1q_s8(hi + h, vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)));
      }
      uint8_t sc, m;
      get_scale_min_k4(is, b[i].scales, sc, m);
      store_scaled(lo, 32, d * sc, -dmin * m, out);
      get_scale_min_k4(is + 1, b[i].scales, sc, m);
      store_scaled(hi, 32, d * sc, -dmin * m, out + 32);
    }
  }
}
void dequant_q6_K(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  alignas(16) int8_t q[4][32];
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    const uint8_t* ql = b[i].ql;
    const uint8_t* qh = b[i].qh;
    const int8_t* sc = b[i].scales;
    for (int part = 0; part < 2; ++part, ql += 64, qh += 32, sc += 8, out += 128) {
      unpack_q6_K_half(ql, qh, q);
      for (int g = 0; g < 4; ++g) {
        store_scaled(q[g], 16, d * sc[2 * g], 0.0f, out + 32 * g);
        store_scaled(q[g] + 16, 16, d * sc[2 * g + 1], 0.0f, out + 32 * g + 16);
      }
    }
  }
}

// Block-wise attention (DD-056): per position exactly this tier's dot / axpy.
void attn_scores_f16(const uint16_t* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) {
  for (int64_t t = 0; t < n; ++t) scores[t] = dot_f16_f32(k + t * dim, q, dim) * scale;
}
void attn_accum_f16(const uint16_t* v, int64_t n, int32_t dim, const float* w, float* acc) {
  for (int64_t t = 0; t < n; ++t) axpy_f16(w[t], v + t * dim, acc, dim);
}
void attn_scores_f32(const float* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) {
  for (int64_t t = 0; t < n; ++t) scores[t] = dot_f32(q, k + t * dim, dim) * scale;
}
void attn_accum_f32(const float* v, int64_t n, int32_t dim, const float* w, float* acc) {
  for (int64_t t = 0; t < n; ++t) axpy_f32(w[t], v + t * dim, acc, dim);
}

}  // namespace

bool register_neon_kernels(CpuKernels& k) {
  k.isa = CpuIsa::kNeon;
  k.dot_f32 = dot_f32;
  k.axpy_f32 = axpy_f32;
  k.dot_f16_f32 = dot_f16_f32;
  k.axpy_f16 = axpy_f16;
  k.gemm_panel = gemm_panel;
  k.attn_scores_f16 = attn_scores_f16;
  k.attn_accum_f16 = attn_accum_f16;
  k.attn_scores_f32 = attn_scores_f32;
  k.attn_accum_f32 = attn_accum_f32;
  auto set = [&](DType t, VecDotFn dot, DequantFn dq) {
    k.vec_dot[static_cast<size_t>(t)] = dot;
    if (dq) k.dequant[static_cast<size_t>(t)] = dq;
  };
  set(DType::kF32, vec_dot_f32, nullptr);
  set(DType::kF16, vec_dot_f16, dequant_f16);
  set(DType::kBF16, vec_dot_bf16, dequant_bf16);
  set(DType::kQ8_0, vec_dot_q8_0, dequant_q8_0);
  set(DType::kQ4_0, vec_dot_q4_0, dequant_q4_0);
  set(DType::kQ4_1, vec_dot_q4_1, dequant_q4_1);
  set(DType::kQ5_0, vec_dot_q5_0, dequant_q5_0);
  set(DType::kQ4_K, vec_dot_q4_K, dequant_q4_K);
  set(DType::kQ6_K, vec_dot_q6_K, dequant_q6_K);
  return true;
}

}  // namespace engine

#else

namespace engine {
bool register_neon_kernels(CpuKernels&) { return false; }
}  // namespace engine

#endif

// qmm NEON tier (dynacore/cpu/qmm.h). Included by qmm_neon_dot.cpp
// (QMM_I8MM 0: SDOT, int8 activations, gemv only) and qmm_neon_i8mm.cpp
// (QMM_I8MM 1: also USDOT for int16 activations and SMMLA/USMMLA gemm). Each
// includer defines QMM_NS, the namespace its copy lives in.
//
// Arithmetic per scale unit is the reference's (qmm_reference.cpp): exact
// int32 sub-block dots, scaled once per 32 values (16 for per-16 formats).
// Only the fp32 summation order differs.

#include <arm_neon.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "dynacore/cpu/qmm.h"
#include "qmm_unpack.h"

namespace dynacore::qmm::QMM_NS {
namespace {

using quant::BlockQ4_0;
using quant::BlockQ4_K;
using quant::BlockQ5_K;
using quant::BlockQ6_K;
using quant::BlockQ8_0;

// ---------------------------------------------------------------------------
// Activation quantization (bit-identical to quantize_row_ref).
// ---------------------------------------------------------------------------
void quantize_row(const float* x, int64_t k, ActMode mode, int8_t* hi, uint8_t* lo, int64_t cs, float* d, float* c,
                  int64_t ds) {
  const float qmax = mode == ActMode::kInt8 ? 127.0f : 32767.0f;
  const float32x4_t vlo = vdupq_n_f32(-qmax), vhi = vdupq_n_f32(qmax);
  for (int64_t b = 0; b < k / kBlock; ++b, x += kBlock) {
    float32x4_t v[8];
    float32x4_t mx = vdupq_n_f32(0);
    for (int i = 0; i < 8; ++i) {
      v[i] = vld1q_f32(x + 4 * i);
      mx = vmaxq_f32(mx, vabsq_f32(v[i]));
    }
    const float db = vmaxvq_f32(mx) / qmax;
    const float id = db > 0 ? 1.0f / db : 0.0f;
    int32x4_t q[8];
    int32x4_t s = vdupq_n_s32(0);
    for (int i = 0; i < 8; ++i) {
      q[i] = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v[i], id), vlo), vhi));
      s = vaddq_s32(s, q[i]);
    }
    for (int j = 0; j < 4; ++j) {  // chunk j: values 8j .. 8j + 7
      const int16x8_t h = vcombine_s16(vmovn_s32(q[2 * j]), vmovn_s32(q[2 * j + 1]));
      const int64_t at = (b * 4 + j) * cs;
      if (mode == ActMode::kInt8) {
        vst1_s8(hi + at, vmovn_s16(h));
      } else {
        vst1_s8(hi + at, vshrn_n_s16(h, 8));                           // arithmetic: q >> 8
        vst1_u8(lo + at, vmovn_u16(vreinterpretq_u16_s16(h)));        // q & 0xFF
      }
    }
    d[b * ds] = db;
    c[b * ds] = db * static_cast<float>(vaddvq_s32(s));
  }
}

// ---------------------------------------------------------------------------
// Weight unpacking: vectorized for the common formats, the scalar traits
// (qmm_unpack.h) otherwise. Same Group contents either way.
// ---------------------------------------------------------------------------
template <DType T>
inline void unpack(const void* row, int64_t sub0, int nsub, Group& g) {
  if constexpr (T == DType::kQ4_K) {
    const BlockQ4_K& b = static_cast<const BlockQ4_K*>(row)[sub0 / 8];
    Fmt<DType::kQ4_K>::scales(b, g);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int jp = 0; jp < 4; ++jp) {
      const uint8x16_t q0 = vld1q_u8(b.qs + 32 * jp), q1 = vld1q_u8(b.qs + 32 * jp + 16);
      vst1q_u8(reinterpret_cast<uint8_t*>(g.q) + 64 * jp, vandq_u8(q0, m4));
      vst1q_u8(reinterpret_cast<uint8_t*>(g.q) + 64 * jp + 16, vandq_u8(q1, m4));
      vst1q_u8(reinterpret_cast<uint8_t*>(g.q) + 64 * jp + 32, vshrq_n_u8(q0, 4));
      vst1q_u8(reinterpret_cast<uint8_t*>(g.q) + 64 * jp + 48, vshrq_n_u8(q1, 4));
    }
  } else if constexpr (T == DType::kQ6_K) {
    const BlockQ6_K& b = static_cast<const BlockQ6_K*>(row)[sub0 / 8];
    const float d = fp16_to_fp32(b.d);
    const int8x16_t sc8 = vld1q_s8(b.scales);
    const int16x8_t s0 = vmovl_s8(vget_low_s8(sc8)), s1 = vmovl_high_s8(sc8);
    const float32x4_t vd = vdupq_n_f32(d);
    vst1q_f32(g.s16 + 0, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0)))));
    vst1q_f32(g.s16 + 4, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s0))));
    vst1q_f32(g.s16 + 8, vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1)))));
    vst1q_f32(g.s16 + 12, vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s1))));
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(0x03);
    const int8x16_t k32 = vdupq_n_s8(32);
    for (int p = 0; p < 2; ++p) {
      const uint8_t* ql = b.ql + 64 * p;
      const uint8_t* qh = b.qh + 32 * p;
      int8_t* y = g.q + 128 * p;
      for (int half = 0; half < 2; ++half) {
        const uint8x16_t l0 = vld1q_u8(ql + 16 * half), l1 = vld1q_u8(ql + 32 + 16 * half);
        const uint8x16_t h = vld1q_u8(qh + 16 * half);
        const uint8x16_t c0 = vorrq_u8(vandq_u8(l0, m4), vshlq_n_u8(vandq_u8(h, m2), 4));
        const uint8x16_t c1 = vorrq_u8(vandq_u8(l1, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m2), 4));
        const uint8x16_t c2 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m2), 4));
        const uint8x16_t c3 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4));
        vst1q_s8(y + 16 * half + 0, vsubq_s8(vreinterpretq_s8_u8(c0), k32));
        vst1q_s8(y + 16 * half + 32, vsubq_s8(vreinterpretq_s8_u8(c1), k32));
        vst1q_s8(y + 16 * half + 64, vsubq_s8(vreinterpretq_s8_u8(c2), k32));
        vst1q_s8(y + 16 * half + 96, vsubq_s8(vreinterpretq_s8_u8(c3), k32));
      }
    }
  } else if constexpr (T == DType::kQ5_K) {
    const BlockQ5_K& b = static_cast<const BlockQ5_K*>(row)[sub0 / 8];
    const float d = fp16_to_fp32(b.d), dmin = fp16_to_fp32(b.dmin);
    for (int j = 0; j < 8; ++j) {
      uint8_t sc, mn;
      quant::get_scale_min_k4(j, b.scales, sc, mn);
      detail::set32(g, j, d * sc);
      g.mn[j] = dmin * mn;
    }
    const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
    const uint8x16_t h0 = vld1q_u8(b.qh), h1 = vld1q_u8(b.qh + 16);
    for (int jp = 0; jp < 4; ++jp) {
      const int8x16_t sl = vdupq_n_s8(static_cast<int8_t>(-2 * jp)), sh = vdupq_n_s8(static_cast<int8_t>(-2 * jp - 1));
      const uint8x16_t q0 = vld1q_u8(b.qs + 32 * jp), q1 = vld1q_u8(b.qs + 32 * jp + 16);
      auto* y = reinterpret_cast<uint8_t*>(g.q) + 64 * jp;
      vst1q_u8(y + 0, vorrq_u8(vandq_u8(q0, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h0, sl), one), 4)));
      vst1q_u8(y + 16, vorrq_u8(vandq_u8(q1, m4), vshlq_n_u8(vandq_u8(vshlq_u8(h1, sl), one), 4)));
      vst1q_u8(y + 32, vorrq_u8(vshrq_n_u8(q0, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h0, sh), one), 4)));
      vst1q_u8(y + 48, vorrq_u8(vshrq_n_u8(q1, 4), vshlq_n_u8(vandq_u8(vshlq_u8(h1, sh), one), 4)));
    }
  } else if constexpr (T == DType::kQ8_0) {
    const BlockQ8_0* b = static_cast<const BlockQ8_0*>(row) + sub0;
    for (int j = 0; j < nsub; ++j) {
      vst1q_s8(g.q + 32 * j, vld1q_s8(b[j].qs));
      vst1q_s8(g.q + 32 * j + 16, vld1q_s8(b[j].qs + 16));
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  } else if constexpr (T == DType::kQ4_0) {
    const BlockQ4_0* b = static_cast<const BlockQ4_0*>(row) + sub0;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t k8 = vdupq_n_s8(8);
    for (int j = 0; j < nsub; ++j) {
      const uint8x16_t q = vld1q_u8(b[j].qs);
      vst1q_s8(g.q + 32 * j, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4)), k8));
      vst1q_s8(g.q + 32 * j + 16, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), k8));
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  } else {
    unpack_group<T>(row, sub0, nsub, g);
  }
}

// ---------------------------------------------------------------------------
// Register-resident groups for gemv: a full group (8 sub-blocks) of one row
// as 16 code vectors (sub-block j = q[2j], q[2j + 1]) and scale vectors
// (s[0..1] per-32 scales of sub-blocks 0-3 / 4-7, or s[0..3] the 16 per-16
// scales; mn[0..1] per-32 minimum coefficients). Fully inlined, so the
// compiler keeps it in registers: no store / reload per weight block.
// ---------------------------------------------------------------------------
struct GroupV {
  int8x16_t q[16];
  float32x4_t s[4];
  float32x4_t mn[2];
};

#define QMM_INLINE inline __attribute__((always_inline))

// Q4_K / Q5_K 6-bit scales and minima of a super-block as two byte vectors
// (the layout get_scale_min_k4 decodes, all 8 at once).
QMM_INLINE void k4_scales(const uint8_t* scales, float d, float dmin, GroupV& g) {
  uint32_t u[4];
  std::memcpy(u, scales, 12);
  constexpr uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
  u[3] = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
  const uint32_t mins_lo = u[1] & km1;
  u[1] = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
  u[2] = mins_lo;
  u[0] &= km1;
  uint8x16_t b;
  std::memcpy(&b, u, 16);  // bytes 0-7: scales, 8-15: minima
  const uint16x8_t sc = vmovl_u8(vget_low_u8(b)), mn = vmovl_high_u8(b);
  const float32x4_t vd = vdupq_n_f32(d), vm = vdupq_n_f32(dmin);
  g.s[0] = vmulq_f32(vd, vcvtq_f32_u32(vmovl_u16(vget_low_u16(sc))));
  g.s[1] = vmulq_f32(vd, vcvtq_f32_u32(vmovl_high_u16(sc)));
  g.mn[0] = vmulq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_low_u16(mn))));
  g.mn[1] = vmulq_f32(vm, vcvtq_f32_u32(vmovl_high_u16(mn)));
}

// Eight fp16 block scales (stride `stride` bytes) as two float vectors.
QMM_INLINE void legacy_scales(const uint8_t* first, int64_t stride, GroupV& g) {
  uint16_t h[8];
  for (int j = 0; j < 8; ++j) std::memcpy(&h[j], first + j * stride, 2);
  const float16x8_t f = vreinterpretq_f16_u16(vld1q_u16(h));
  g.s[0] = vcvt_f32_f16(vget_low_f16(f));
  g.s[1] = vcvt_high_f32_f16(f);
}

template <DType T>
QMM_INLINE void unpack_v(const void* row, int64_t sub0, GroupV& g) {
  if constexpr (T == DType::kQ4_K) {
    const BlockQ4_K& b = static_cast<const BlockQ4_K*>(row)[sub0 / 8];
    k4_scales(b.scales, fp16_to_fp32(b.d), fp16_to_fp32(b.dmin), g);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int jp = 0; jp < 4; ++jp) {
      const uint8x16_t q0 = vld1q_u8(b.qs + 32 * jp), q1 = vld1q_u8(b.qs + 32 * jp + 16);
      g.q[4 * jp + 0] = vreinterpretq_s8_u8(vandq_u8(q0, m4));
      g.q[4 * jp + 1] = vreinterpretq_s8_u8(vandq_u8(q1, m4));
      g.q[4 * jp + 2] = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
      g.q[4 * jp + 3] = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
    }
  } else if constexpr (T == DType::kQ5_K) {
    const BlockQ5_K& b = static_cast<const BlockQ5_K*>(row)[sub0 / 8];
    k4_scales(b.scales, fp16_to_fp32(b.d), fp16_to_fp32(b.dmin), g);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), b4 = vdupq_n_u8(0x10);
    uint8x16_t h0 = vld1q_u8(b.qh), h1 = vld1q_u8(b.qh + 16);
    for (int jp = 0; jp < 4; ++jp) {
      const uint8x16_t q0 = vld1q_u8(b.qs + 32 * jp), q1 = vld1q_u8(b.qs + 32 * jp + 16);
      // Bit 2 jp of qh -> bit 4 of the low codes, bit 2 jp + 1 -> of the high ones.
      g.q[4 * jp + 0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4), vandq_u8(vshlq_n_u8(h0, 4), b4)));
      g.q[4 * jp + 1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4), vandq_u8(vshlq_n_u8(h1, 4), b4)));
      g.q[4 * jp + 2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q0, 4), vandq_u8(vshlq_n_u8(h0, 3), b4)));
      g.q[4 * jp + 3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q1, 4), vandq_u8(vshlq_n_u8(h1, 3), b4)));
      h0 = vshrq_n_u8(h0, 2);
      h1 = vshrq_n_u8(h1, 2);
    }
  } else if constexpr (T == DType::kQ6_K) {
    const BlockQ6_K& b = static_cast<const BlockQ6_K*>(row)[sub0 / 8];
    const int8x16_t sc8 = vld1q_s8(b.scales);
    const int16x8_t s0 = vmovl_s8(vget_low_s8(sc8)), s1 = vmovl_high_s8(sc8);
    const float32x4_t vd = vdupq_n_f32(fp16_to_fp32(b.d));
    g.s[0] = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))));
    g.s[1] = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s0)));
    g.s[2] = vmulq_f32(vd, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))));
    g.s[3] = vmulq_f32(vd, vcvtq_f32_s32(vmovl_high_s16(s1)));
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(0x30);
    const int8x16_t k32 = vdupq_n_s8(32);
    for (int p = 0; p < 2; ++p) {
      const uint8_t* ql = b.ql + 64 * p;
      const uint8_t* qh = b.qh + 32 * p;
      for (int half = 0; half < 2; ++half) {
        const uint8x16_t l0 = vld1q_u8(ql + 16 * half), l1 = vld1q_u8(ql + 32 + 16 * half);
        const uint8x16_t h = vld1q_u8(qh + 16 * half);
        // Values 128 p + 16 half + 32 i, i = 0..3: code vectors 8 p + 2 i + half.
        const uint8x16_t c0 = vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8(h, 4), m2));
        const uint8x16_t c1 = vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8(h, 2), m2));
        const uint8x16_t c2 = vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(h, m2));
        const uint8x16_t c3 = vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(vshrq_n_u8(h, 2), m2));
        g.q[8 * p + 0 + half] = vsubq_s8(vreinterpretq_s8_u8(c0), k32);
        g.q[8 * p + 2 + half] = vsubq_s8(vreinterpretq_s8_u8(c1), k32);
        g.q[8 * p + 4 + half] = vsubq_s8(vreinterpretq_s8_u8(c2), k32);
        g.q[8 * p + 6 + half] = vsubq_s8(vreinterpretq_s8_u8(c3), k32);
      }
    }
  } else if constexpr (T == DType::kQ8_0) {
    const BlockQ8_0* b = static_cast<const BlockQ8_0*>(row) + sub0;
    legacy_scales(reinterpret_cast<const uint8_t*>(&b[0].d), sizeof(BlockQ8_0), g);
    for (int j = 0; j < 8; ++j) {
      g.q[2 * j] = vld1q_s8(b[j].qs);
      g.q[2 * j + 1] = vld1q_s8(b[j].qs + 16);
    }
  } else if constexpr (T == DType::kQ4_0) {
    const BlockQ4_0* b = static_cast<const BlockQ4_0*>(row) + sub0;
    legacy_scales(reinterpret_cast<const uint8_t*>(&b[0].d), sizeof(BlockQ4_0), g);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t k8 = vdupq_n_s8(8);
    for (int j = 0; j < 8; ++j) {
      const uint8x16_t q = vld1q_u8(b[j].qs);
      g.q[2 * j] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4)), k8);
      g.q[2 * j + 1] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), k8);
    }
  } else {
    Group t;
    unpack_group<T>(row, sub0, 8, t);
    for (int i = 0; i < 16; ++i) g.q[i] = vld1q_s8(t.q + 16 * i);
    if constexpr (Fmt<T>::kHalf) {
      for (int i = 0; i < 4; ++i) g.s[i] = vld1q_f32(t.s16 + 4 * i);
    } else {
      g.s[0] = vld1q_f32(t.s32);
      g.s[1] = vld1q_f32(t.s32 + 4);
    }
    g.mn[0] = vld1q_f32(t.mn);
    g.mn[1] = vld1q_f32(t.mn + 4);
  }
}

// ---------------------------------------------------------------------------
// gemv
// ---------------------------------------------------------------------------

// Integer partial sums of sub-block j (two halves of 16 codes) against one
// activation row: lanes add up to the sub-block's (or half's) exact dot.
template <ActMode M>
inline int32x4_t sub_dot(int8x16_t w0, int8x16_t w1, const int8_t* ah, const uint8_t* al) {
  const int32x4_t z = vdupq_n_s32(0);
  if constexpr (M == ActMode::kInt8) {
    return vdotq_s32(vdotq_s32(z, w0, vld1q_s8(ah)), w1, vld1q_s8(ah + 16));
  } else {
#if QMM_I8MM
    const int32x4_t h = vdotq_s32(vdotq_s32(z, w0, vld1q_s8(ah)), w1, vld1q_s8(ah + 16));
    const int32x4_t l = vusdotq_s32(vusdotq_s32(z, vld1q_u8(al), w0), vld1q_u8(al + 16), w1);
    return vaddq_s32(vshlq_n_s32(h, 8), l);
#else
    (void)al;
    return z;  // not reachable: the dot tier does not offer kInt16
#endif
  }
}
template <ActMode M>
inline int32x4_t half_dot(int8x16_t w, const int8_t* ah, const uint8_t* al) {
  const int32x4_t z = vdupq_n_s32(0);
  if constexpr (M == ActMode::kInt8) {
    return vdotq_s32(z, w, vld1q_s8(ah));
  } else {
#if QMM_I8MM
    return vaddq_s32(vshlq_n_s32(vdotq_s32(z, w, vld1q_s8(ah)), 8), vusdotq_s32(z, vld1q_u8(al), w));
#else
    (void)al;
    return z;
#endif
  }
}

// [sum(a), sum(b), sum(c), sum(d)]
inline int32x4_t reduce4(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d) {
  return vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d));
}

// Accumulates one full group (8 sub-blocks, 256 values) of one row.
template <DType T, ActMode M>
inline void group_row(const Group& g, const int8_t* ah, const uint8_t* al, const float* ad, const float* ac,
                      float32x4_t& acc, float32x4_t& accm) {
  using F = Fmt<T>;
  if constexpr (!F::kHalf) {
    int32x4_t p[8];
    for (int j = 0; j < 8; ++j) {
      p[j] = sub_dot<M>(vld1q_s8(g.q + 32 * j), vld1q_s8(g.q + 32 * j + 16), ah + 32 * j, al ? al + 32 * j : nullptr);
    }
    const float32x4_t d0 = vld1q_f32(ad), d1 = vld1q_f32(ad + 4);
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[0], p[1], p[2], p[3])), vmulq_f32(vld1q_f32(g.s32), d0));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[4], p[5], p[6], p[7])), vmulq_f32(vld1q_f32(g.s32 + 4), d1));
    if constexpr (F::kMin) {
      accm = vfmaq_f32(accm, vld1q_f32(g.mn), vld1q_f32(ac));
      accm = vfmaq_f32(accm, vld1q_f32(g.mn + 4), vld1q_f32(ac + 4));
    }
  } else {
    int32x4_t p[16];
    for (int h = 0; h < 16; ++h) p[h] = half_dot<M>(vld1q_s8(g.q + 16 * h), ah + 16 * h, al ? al + 16 * h : nullptr);
    for (int q4 = 0; q4 < 4; ++q4) {  // halves 4 q4 .. 4 q4 + 3 = sub-blocks 2 q4, 2 q4 + 1
      const float32x4_t dv = vld1q_f32(ad + (q4 / 2) * 4);
      const float32x4_t dd = (q4 & 1) ? vzip2q_f32(dv, dv) : vzip1q_f32(dv, dv);
      acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[4 * q4], p[4 * q4 + 1], p[4 * q4 + 2], p[4 * q4 + 3])),
                      vmulq_f32(vld1q_f32(g.s16 + 4 * q4), dd));
    }
    if constexpr (F::kMin) {
      accm = vfmaq_f32(accm, vld1q_f32(g.mn), vld1q_f32(ac));
      accm = vfmaq_f32(accm, vld1q_f32(g.mn + 4), vld1q_f32(ac + 4));
    }
  }
}

template <DType T, ActMode M>
QMM_INLINE void group_row_v(const GroupV& g, const int8_t* ah, const uint8_t* al, const float* ad, const float* ac,
                            float32x4_t& acc, float32x4_t& accm) {
  using F = Fmt<T>;
  if constexpr (!F::kHalf) {
    int32x4_t p[8];
    for (int j = 0; j < 8; ++j) p[j] = sub_dot<M>(g.q[2 * j], g.q[2 * j + 1], ah + 32 * j, al ? al + 32 * j : nullptr);
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[0], p[1], p[2], p[3])), vmulq_f32(g.s[0], vld1q_f32(ad)));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[4], p[5], p[6], p[7])), vmulq_f32(g.s[1], vld1q_f32(ad + 4)));
  } else {
    int32x4_t p[16];
    for (int h = 0; h < 16; ++h) p[h] = half_dot<M>(g.q[h], ah + 16 * h, al ? al + 16 * h : nullptr);
    const float32x4_t d0 = vld1q_f32(ad), d1 = vld1q_f32(ad + 4);
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[0], p[1], p[2], p[3])), vmulq_f32(g.s[0], vzip1q_f32(d0, d0)));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[4], p[5], p[6], p[7])), vmulq_f32(g.s[1], vzip2q_f32(d0, d0)));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[8], p[9], p[10], p[11])), vmulq_f32(g.s[2], vzip1q_f32(d1, d1)));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(reduce4(p[12], p[13], p[14], p[15])), vmulq_f32(g.s[3], vzip2q_f32(d1, d1)));
  }
  if constexpr (F::kMin) {
    accm = vfmaq_f32(accm, g.mn[0], vld1q_f32(ac));
    accm = vfmaq_f32(accm, g.mn[1], vld1q_f32(ac + 4));
  }
}

// A partial group (legacy formats, k % 256 != 0): sub-block by sub-block.
template <DType T, ActMode M>
inline float tail_row(const Group& g, int nsub, const int8_t* ah, const uint8_t* al, const float* ad, const float* ac) {
  using F = Fmt<T>;
  static_assert(!F::kHalf, "per-16 formats are 256-value formats: no partial groups");
  float acc = 0;
  for (int j = 0; j < nsub; ++j) {
    const int32x4_t p = sub_dot<M>(vld1q_s8(g.q + 32 * j), vld1q_s8(g.q + 32 * j + 16), ah + 32 * j, al ? al + 32 * j : nullptr);
    acc += g.s32[j] * ad[j] * static_cast<float>(vaddvq_s32(p));
    if constexpr (F::kMin) acc -= g.mn[j] * ac[j];
  }
  return acc;
}

template <DType T, ActMode M, int R>
void gemv_t(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, float* y,
            int64_t y_stride, const float* bias) {
  const auto* base = static_cast<const std::byte*>(w);
  const int64_t k = a.k, kb = k / kBlock;
  const int64_t full = kb / 8 * 8;  // sub-blocks in full groups
  const int8_t* ah[R];
  const uint8_t* al[R];
  const float* ad[R];
  const float* ac[R];
  for (int r = 0; r < R; ++r) {
    ah[r] = a.hi + (r0 + r) * k;
    al[r] = a.lo ? a.lo + (r0 + r) * k : nullptr;
    ad[r] = a.d + (r0 + r) * kb;
    ac[r] = a.c + (r0 + r) * kb;
  }
  Group g;
  for (int64_t n = n0; n < n1; ++n) {
    const std::byte* row = base + n * w_row_bytes;
    float32x4_t acc[R], accm[R];
    for (int r = 0; r < R; ++r) acc[r] = accm[r] = vdupq_n_f32(0);
    for (int64_t s0 = 0; s0 < full; s0 += 8) {
      GroupV gv;
      unpack_v<T>(row, s0, gv);
      for (int r = 0; r < R; ++r) {
        group_row_v<T, M>(gv, ah[r] + s0 * 32, al[r] ? al[r] + s0 * 32 : nullptr, ad[r] + s0, ac[r] + s0, acc[r], accm[r]);
      }
    }
    float tail[R] = {};
    if constexpr (Fmt<T>::kSuper == 32) {
      if (full < kb) {
        const int nsub = static_cast<int>(kb - full);
        unpack<T>(row, full, nsub, g);
        for (int r = 0; r < R; ++r) {
          tail[r] = tail_row<T, M>(g, nsub, ah[r] + full * 32, al[r] ? al[r] + full * 32 : nullptr, ad[r] + full, ac[r] + full);
        }
      }
    }
    const float bn = bias ? bias[n] : 0.0f;
    for (int r = 0; r < R; ++r) {
      y[(r0 + r) * y_stride + n] = vaddvq_f32(acc[r]) - vaddvq_f32(accm[r]) + tail[r] + bn;
    }
  }
}

template <DType T, ActMode M>
void gemv_rows(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, int rows, float* y,
               int64_t y_stride, const float* bias) {
  switch (rows) {
    case 1: gemv_t<T, M, 1>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 2: gemv_t<T, M, 2>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 3: gemv_t<T, M, 3>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 4: gemv_t<T, M, 4>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 5: gemv_t<T, M, 5>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 6: gemv_t<T, M, 6>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    case 7: gemv_t<T, M, 7>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
    default: gemv_t<T, M, 8>(w, w_row_bytes, n0, n1, a, r0, y, y_stride, bias); break;
  }
}

// ---------------------------------------------------------------------------
// Packed Q4_K gemv (qmm.h "Packed weights"): one 16-byte vector = 4 rows x 4
// codes; SDOT by element multiplies it with 4 activation codes, giving the 4
// rows' partial sums in 4 lanes. Per 32-value sub-block the int32 sums are
// scaled once (4 rows' scales are one vector).
// ---------------------------------------------------------------------------

// acc += lane dots of 8 code vectors (4 rows x 32 codes) with 32 activation
// codes in two vectors (lanes 0..3 each).
template <ActMode M>
QMM_INLINE int32x4_t lane_dot32(const int8x16_t* c, const int8_t* ah, const uint8_t* al) {
  const int8x16_t a0 = vld1q_s8(ah), a1 = vld1q_s8(ah + 16);
  int32x4_t h = vdupq_n_s32(0);
  h = vdotq_laneq_s32(h, c[0], a0, 0);
  h = vdotq_laneq_s32(h, c[1], a0, 1);
  h = vdotq_laneq_s32(h, c[2], a0, 2);
  h = vdotq_laneq_s32(h, c[3], a0, 3);
  h = vdotq_laneq_s32(h, c[4], a1, 0);
  h = vdotq_laneq_s32(h, c[5], a1, 1);
  h = vdotq_laneq_s32(h, c[6], a1, 2);
  h = vdotq_laneq_s32(h, c[7], a1, 3);
  if constexpr (M == ActMode::kInt8) {
    (void)al;
    return h;
  } else {
#if QMM_I8MM
    const uint8x16_t l0 = vld1q_u8(al), l1 = vld1q_u8(al + 16);
    int32x4_t l = vdupq_n_s32(0);
    l = vsudotq_laneq_s32(l, c[0], l0, 0);
    l = vsudotq_laneq_s32(l, c[1], l0, 1);
    l = vsudotq_laneq_s32(l, c[2], l0, 2);
    l = vsudotq_laneq_s32(l, c[3], l0, 3);
    l = vsudotq_laneq_s32(l, c[4], l1, 0);
    l = vsudotq_laneq_s32(l, c[5], l1, 1);
    l = vsudotq_laneq_s32(l, c[6], l1, 2);
    l = vsudotq_laneq_s32(l, c[7], l1, 3);
    return vaddq_s32(vshlq_n_s32(h, 8), l);
#else
    (void)al;
    return h;  // not reachable: the dot tier does not offer kInt16
#endif
  }
}

// The 8 per-row scale vectors of a packed super-block: s[j] = d . sc[j],
// m[j] = dmin . mn[j] (4 rows each).
QMM_INLINE void packed_q4k_scales(const PackedQ4K& p, float32x4_t* s, float32x4_t* m) {
  const float32x4_t d4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p.d)));
  const float32x4_t m4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p.dmin)));
  for (int h = 0; h < 2; ++h) {
    const uint8x16_t sc = vld1q_u8(&p.sc[4 * h][0]), mn = vld1q_u8(&p.mn[4 * h][0]);
    const uint16x8_t sl = vmovl_u8(vget_low_u8(sc)), sh = vmovl_high_u8(sc);
    const uint16x8_t ml = vmovl_u8(vget_low_u8(mn)), mh = vmovl_high_u8(mn);
    s[4 * h + 0] = vmulq_f32(d4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(sl))));
    s[4 * h + 1] = vmulq_f32(d4, vcvtq_f32_u32(vmovl_high_u16(sl)));
    s[4 * h + 2] = vmulq_f32(d4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(sh))));
    s[4 * h + 3] = vmulq_f32(d4, vcvtq_f32_u32(vmovl_high_u16(sh)));
    m[4 * h + 0] = vmulq_f32(m4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(ml))));
    m[4 * h + 1] = vmulq_f32(m4, vcvtq_f32_u32(vmovl_high_u16(ml)));
    m[4 * h + 2] = vmulq_f32(m4, vcvtq_f32_u32(vmovl_u16(vget_low_u16(mh))));
    m[4 * h + 3] = vmulq_f32(m4, vcvtq_f32_u32(vmovl_high_u16(mh)));
  }
}

template <ActMode M, int R>
void gemv_packed_q4k_t(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, float* y,
                       int64_t y_stride, const float* bias) {
  const int64_t nb = k / 256, kb = k / kBlock;
  const auto* pk = static_cast<const PackedQ4K*>(p);
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  for (int64_t g = g0; g < g1; ++g) {
    float32x4_t acc[R], accm[R];
    for (int r = 0; r < R; ++r) acc[r] = accm[r] = vdupq_n_f32(0);
    for (int64_t b = 0; b < nb; ++b) {
      const PackedQ4K& P = pk[g * nb + b];
      float32x4_t sv[8], mv[8];
      packed_q4k_scales(P, sv, mv);
      for (int jp = 0; jp < 4; ++jp) {
        int8x16_t lo[8], hi[8];
        for (int i4 = 0; i4 < 8; ++i4) {
          const uint8x16_t wv = vld1q_u8(P.qs[jp][i4][0]);
          lo[i4] = vreinterpretq_s8_u8(vandq_u8(wv, m4));
          hi[i4] = vreinterpretq_s8_u8(vshrq_n_u8(wv, 4));
        }
        for (int r = 0; r < R; ++r) {
          const int64_t off = (r0 + r) * k + b * 256 + 64 * jp;
          const int8_t* ah = a.hi + off;
          const uint8_t* al = a.lo ? a.lo + off : nullptr;
          const float* ad = a.d + (r0 + r) * kb + b * 8 + 2 * jp;
          const int32x4_t dl = lane_dot32<M>(lo, ah, al);
          const int32x4_t dh = lane_dot32<M>(hi, ah + 32, al ? al + 32 : nullptr);
          acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(dl), vmulq_n_f32(sv[2 * jp], ad[0]));
          acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(dh), vmulq_n_f32(sv[2 * jp + 1], ad[1]));
        }
      }
      for (int r = 0; r < R; ++r) {
        const float* ac = a.c + (r0 + r) * kb + b * 8;
        for (int j = 0; j < 8; ++j) accm[r] = vfmaq_n_f32(accm[r], mv[j], ac[j]);
      }
    }
    const float32x4_t bv = bias ? vld1q_f32(bias + 4 * g) : vdupq_n_f32(0);
    for (int r = 0; r < R; ++r) vst1q_f32(y + (r0 + r) * y_stride + 4 * g, vaddq_f32(vsubq_f32(acc[r], accm[r]), bv));
  }
}

// --- packed Q6_K ---------------------------------------------------------

// acc += lane dots of 4 code vectors (4 rows x 16 codes) with 16 activation codes.
template <ActMode M>
QMM_INLINE int32x4_t lane_dot16(const int8x16_t* c, const int8_t* ah, const uint8_t* al) {
  const int8x16_t a0 = vld1q_s8(ah);
  int32x4_t h = vdupq_n_s32(0);
  h = vdotq_laneq_s32(h, c[0], a0, 0);
  h = vdotq_laneq_s32(h, c[1], a0, 1);
  h = vdotq_laneq_s32(h, c[2], a0, 2);
  h = vdotq_laneq_s32(h, c[3], a0, 3);
  if constexpr (M == ActMode::kInt8) {
    (void)al;
    return h;
  } else {
#if QMM_I8MM
    const uint8x16_t l0 = vld1q_u8(al);
    int32x4_t l = vdupq_n_s32(0);
    l = vsudotq_laneq_s32(l, c[0], l0, 0);
    l = vsudotq_laneq_s32(l, c[1], l0, 1);
    l = vsudotq_laneq_s32(l, c[2], l0, 2);
    l = vsudotq_laneq_s32(l, c[3], l0, 3);
    return vaddq_s32(vshlq_n_s32(h, 8), l);
#else
    (void)al;
    return h;
#endif
  }
}

// The 16 per-16-value scale vectors (4 rows each) of a packed super-block.
QMM_INLINE void packed_q6k_scales(const PackedQ6K& p, float32x4_t* sv) {
  const float32x4_t d4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p.d)));
  for (int q = 0; q < 4; ++q) {
    const int8x16_t x = vld1q_s8(&p.sc[4 * q][0]);
    const int16x8_t lo = vmovl_s8(vget_low_s8(x)), hi = vmovl_high_s8(x);
    sv[4 * q + 0] = vmulq_f32(d4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
    sv[4 * q + 1] = vmulq_f32(d4, vcvtq_f32_s32(vmovl_high_s16(lo)));
    sv[4 * q + 2] = vmulq_f32(d4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
    sv[4 * q + 3] = vmulq_f32(d4, vcvtq_f32_s32(vmovl_high_s16(hi)));
  }
}

// The 4 code vectors (qq = 0..3) of run i4 of half n: 4 rows x 4 codes each.
QMM_INLINE void q6k_codes(const PackedQ6K& p, int n, int i4, int8x16_t* c) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(0x03);
  const int8x16_t k32 = vdupq_n_s8(32);
  const uint8x16_t a = vld1q_u8(p.ql[n][0][i4][0]), b = vld1q_u8(p.ql[n][1][i4][0]), h = vld1q_u8(p.qh[n][i4][0]);
  c[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a, m4), vshlq_n_u8(vandq_u8(h, m3), 4))), k32);
  c[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m3), 4))), k32);
  c[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m3), 4))), k32);
  c[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4))), k32);
}

template <ActMode M, int R>
void gemv_packed_q6k_t(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, float* y,
                       int64_t y_stride, const float* bias) {
  const int64_t nb = k / 256, kb = k / kBlock;
  const auto* pk = static_cast<const PackedQ6K*>(p);
  for (int64_t g = g0; g < g1; ++g) {
    float32x4_t acc[R];
    for (int r = 0; r < R; ++r) acc[r] = vdupq_n_f32(0);
    for (int64_t b = 0; b < nb; ++b) {
      const PackedQ6K& P = pk[g * nb + b];
      float32x4_t sv[16];
      packed_q6k_scales(P, sv);
      for (int n = 0; n < 2; ++n) {
        for (int half = 0; half < 2; ++half) {
          int8x16_t c[4][4];  // [qq][run within the half]
          for (int il = 0; il < 4; ++il) {
            int8x16_t t[4];
            q6k_codes(P, n, 4 * half + il, t);
            for (int qq = 0; qq < 4; ++qq) c[qq][il] = t[qq];
          }
          for (int r = 0; r < R; ++r) {
            const int64_t base = (r0 + r) * k + b * 256;
            const float* ad = a.d + (r0 + r) * kb + b * 8;
            for (int qq = 0; qq < 4; ++qq) {
              const int sub = 4 * n + qq;
              const int64_t off = base + sub * 32 + 16 * half;
              const int32x4_t d = lane_dot16<M>(c[qq], a.hi + off, a.lo ? a.lo + off : nullptr);
              acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(d), vmulq_n_f32(sv[8 * n + 2 * qq + half], ad[sub]));
            }
          }
        }
      }
    }
    const float32x4_t bv = bias ? vld1q_f32(bias + 4 * g) : vdupq_n_f32(0);
    for (int r = 0; r < R; ++r) vst1q_f32(y + (r0 + r) * y_stride + 4 * g, vaddq_f32(acc[r], bv));
  }
}

template <ActMode M>
void gemv_packed_q6k_rows(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows, float* y,
                          int64_t y_stride, const float* bias) {
  switch (rows) {
    case 1: gemv_packed_q6k_t<M, 1>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 2: gemv_packed_q6k_t<M, 2>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 3: gemv_packed_q6k_t<M, 3>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 4: gemv_packed_q6k_t<M, 4>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 5: gemv_packed_q6k_t<M, 5>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 6: gemv_packed_q6k_t<M, 6>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 7: gemv_packed_q6k_t<M, 7>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    default: gemv_packed_q6k_t<M, 8>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
  }
}

template <ActMode M>
void gemv_packed_rows(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows, float* y,
                      int64_t y_stride, const float* bias) {
  switch (rows) {
    case 1: gemv_packed_q4k_t<M, 1>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 2: gemv_packed_q4k_t<M, 2>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 3: gemv_packed_q4k_t<M, 3>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 4: gemv_packed_q4k_t<M, 4>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 5: gemv_packed_q4k_t<M, 5>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 6: gemv_packed_q4k_t<M, 6>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    case 7: gemv_packed_q4k_t<M, 7>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
    default: gemv_packed_q4k_t<M, 8>(p, k, g0, g1, a, r0, y, y_stride, bias); break;
  }
}

void gemv_packed(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows, float* y,
                 int64_t y_stride, const float* bias) {
  if (wt == DType::kQ6_K) {
    if (a.mode == ActMode::kInt8) {
      gemv_packed_q6k_rows<ActMode::kInt8>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
    } else {
      gemv_packed_q6k_rows<ActMode::kInt16>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
    }
    return;
  }
  if (a.mode == ActMode::kInt8) {
    gemv_packed_rows<ActMode::kInt8>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
  } else {
    gemv_packed_rows<ActMode::kInt16>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
  }
}

void gemv(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, int rows,
          float* y, int64_t y_stride, const float* bias) {
  dispatch_format(wt, [&](auto t) {
    constexpr DType T = decltype(t)::value;
    if (a.mode == ActMode::kInt8) {
      gemv_rows<T, ActMode::kInt8>(w, w_row_bytes, n0, n1, a, r0, rows, y, y_stride, bias);
    } else {
      gemv_rows<T, ActMode::kInt16>(w, w_row_bytes, n0, n1, a, r0, rows, y, y_stride, bias);
    }
  });
}

#if QMM_I8MM
// ---------------------------------------------------------------------------
// gemm (SMMLA / USMMLA)
// ---------------------------------------------------------------------------
//
// A group of 4 weight rows is unpacked once across all of K into the scratch
// as two row pairs in 8-byte chunks (chunk j of pair p: 16 bytes, row 2p
// first) with per-unit scales stored [unit][row]. Then every pair of
// activation rows (pair layout) is multiplied against both weight pairs:
//   vmmlaq_s32(acc, A, W) -> [a0.w0, a0.w1, a1.w0, a1.w1]
// i.e. lanes (act row, weight row), act-major, the order of the output rows.

struct WGroup {
  int8_t* q;    // [2 pairs][k / 8 chunks][16]
  float* s;     // [k / 32 or k / 16 units][4 rows]
  float* mn;    // [k / 32][4 rows]
};

template <DType T>
WGroup unpack_wgroup(const std::byte* base, int64_t w_row_bytes, int64_t n, int valid, int64_t k, GemmScratch& scratch) {
  using F = Fmt<T>;
  const int64_t kb = k / kBlock;
  const int64_t units = F::kHalf ? 2 * kb : kb;
  const size_t qbytes = static_cast<size_t>(4 * k);
  const size_t need = qbytes + static_cast<size_t>(units + kb) * 4 * sizeof(float) + 128;
  if (scratch.bytes.size() < need) scratch.bytes.resize(need);
  auto* p = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(scratch.bytes.data()) + 63) & ~uintptr_t{63});
  WGroup wg{reinterpret_cast<int8_t*>(p), reinterpret_cast<float*>(p + qbytes),
            reinterpret_cast<float*>(p + qbytes + static_cast<size_t>(units) * 4 * sizeof(float))};
  Group g[4];
  for (int64_t s0 = 0; s0 < kb; s0 += 8) {
    const int nsub = static_cast<int>(std::min<int64_t>(8, kb - s0));
    for (int r = 0; r < 4; ++r) {
      // Rows past the end repeat the last valid row (computed, never stored).
      unpack<T>(base + (n + std::min(r, valid - 1)) * w_row_bytes, s0, nsub, g[r]);
    }
    for (int pr = 0; pr < 2; ++pr) {
      const Group& g0 = g[2 * pr];
      const Group& g1 = g[2 * pr + 1];
      int8_t* dst = wg.q + pr * 2 * k + s0 * 32 * 2;  // 32 values = 4 chunks = 64 bytes per pair
      for (int j = 0; j < nsub; ++j) {
        for (int h = 0; h < 2; ++h) {
          const int64x2_t x0 = vreinterpretq_s64_s8(vld1q_s8(g0.q + 32 * j + 16 * h));
          const int64x2_t x1 = vreinterpretq_s64_s8(vld1q_s8(g1.q + 32 * j + 16 * h));
          vst1q_s8(dst + 64 * j + 32 * h, vreinterpretq_s8_s64(vzip1q_s64(x0, x1)));
          vst1q_s8(dst + 64 * j + 32 * h + 16, vreinterpretq_s8_s64(vzip2q_s64(x0, x1)));
        }
      }
    }
    for (int j = 0; j < nsub; ++j) {
      for (int r = 0; r < 4; ++r) {
        if constexpr (F::kHalf) {
          wg.s[(2 * (s0 + j)) * 4 + r] = g[r].s16[2 * j];
          wg.s[(2 * (s0 + j) + 1) * 4 + r] = g[r].s16[2 * j + 1];
        } else {
          wg.s[(s0 + j) * 4 + r] = g[r].s32[j];
        }
        wg.mn[(s0 + j) * 4 + r] = g[r].mn[j];
      }
    }
  }
  return wg;
}

// Integer dot of one 8-byte chunk run [c0, c1) of a pair of activation rows
// with a pair of weight rows.
template <ActMode M>
inline int32x4_t mm_chunks(const int8_t* wq, const int8_t* ah, const uint8_t* al, int c0, int c1) {
  int32x4_t z = vdupq_n_s32(0);
  if constexpr (M == ActMode::kInt8) {
    for (int c = c0; c < c1; ++c) z = vmmlaq_s32(z, vld1q_s8(ah + 16 * c), vld1q_s8(wq + 16 * c));
    return z;
  } else {
    int32x4_t l = vdupq_n_s32(0);
    for (int c = c0; c < c1; ++c) {
      const int8x16_t wv = vld1q_s8(wq + 16 * c);
      z = vmmlaq_s32(z, vld1q_s8(ah + 16 * c), wv);
      l = vusmmlaq_s32(l, vld1q_u8(al + 16 * c), wv);
    }
    return vaddq_s32(vshlq_n_s32(z, 8), l);
  }
}

// AP activation pairs (rows 2 ap0 .. 2 (ap0 + AP) - 1) x 4 weight rows.
template <DType T, ActMode M, int AP>
inline void gemm_tile(const WGroup& wg, const Acts& a, int64_t ap0, int64_t n, int valid, float* y, int64_t y_stride,
                      const float* bias) {
  using F = Fmt<T>;
  const int64_t k = a.k, kb = k / kBlock;
  float32x4_t acc[2][AP];
  for (int p = 0; p < 2; ++p) {
    for (int q = 0; q < AP; ++q) acc[p][q] = vdupq_n_f32(0);
  }
  for (int64_t j = 0; j < kb; ++j) {
    float32x4_t da[AP], ca[AP];
    for (int q = 0; q < AP; ++q) {
      const float32x2_t dv = vld1_f32(a.d + ((ap0 + q) * kb + j) * 2);  // [d_j0, d_j1]
      da[q] = vcombine_f32(vdup_lane_f32(dv, 0), vdup_lane_f32(dv, 1));
      if constexpr (F::kMin) {
        const float32x2_t cv = vld1_f32(a.c + ((ap0 + q) * kb + j) * 2);
        ca[q] = vcombine_f32(vdup_lane_f32(cv, 0), vdup_lane_f32(cv, 1));
      }
    }
    for (int p = 0; p < 2; ++p) {
      const int8_t* wq = wg.q + p * 2 * k + j * 64;
      if constexpr (!F::kHalf) {
        const float32x2_t sv = vld1_f32(wg.s + j * 4 + 2 * p);  // [s_w0, s_w1]
        const float32x4_t sw = vcombine_f32(sv, sv);
        for (int q = 0; q < AP; ++q) {
          const int8_t* ah = a.hi + (ap0 + q) * 2 * k + j * 64;
          const uint8_t* al = a.lo ? a.lo + (ap0 + q) * 2 * k + j * 64 : nullptr;
          const int32x4_t t = mm_chunks<M>(wq, ah, al, 0, 4);
          acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(t), vmulq_f32(sw, da[q]));
        }
      } else {
        const float32x2_t sv0 = vld1_f32(wg.s + (2 * j) * 4 + 2 * p);
        const float32x2_t sv1 = vld1_f32(wg.s + (2 * j + 1) * 4 + 2 * p);
        const float32x4_t sw0 = vcombine_f32(sv0, sv0), sw1 = vcombine_f32(sv1, sv1);
        for (int q = 0; q < AP; ++q) {
          const int8_t* ah = a.hi + (ap0 + q) * 2 * k + j * 64;
          const uint8_t* al = a.lo ? a.lo + (ap0 + q) * 2 * k + j * 64 : nullptr;
          const int32x4_t t0 = mm_chunks<M>(wq, ah, al, 0, 2);
          const int32x4_t t1 = mm_chunks<M>(wq, ah, al, 2, 4);
          acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(t0), vmulq_f32(sw0, da[q]));
          acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(t1), vmulq_f32(sw1, da[q]));
        }
      }
      if constexpr (F::kMin) {
        const float32x2_t mv = vld1_f32(wg.mn + j * 4 + 2 * p);
        const float32x4_t mw = vcombine_f32(mv, mv);
        for (int q = 0; q < AP; ++q) acc[p][q] = vfmsq_f32(acc[p][q], mw, ca[q]);
      }
    }
  }
  // Lanes [a0.w0, a0.w1, a1.w0, a1.w1] of pair p: weight rows n + 2p, n + 2p + 1.
  for (int q = 0; q < AP; ++q) {
    const int64_t r = 2 * (ap0 + q);
    for (int p = 0; p < 2; ++p) {
      float out[4];
      vst1q_f32(out, acc[p][q]);
      for (int i = 0; i < 2; ++i) {
        const int wr = 2 * p + i;
        if (wr >= valid) continue;
        const float b = bias ? bias[n + wr] : 0.0f;
        y[r * y_stride + n + wr] = out[i] + b;
        if (r + 1 < a.m) y[(r + 1) * y_stride + n + wr] = out[2 + i] + b;
      }
    }
  }
}

template <DType T, ActMode M>
void gemm_t(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, float* y, int64_t y_stride,
            const float* bias, GemmScratch& scratch) {
  const auto* base = static_cast<const std::byte*>(w);
  const int64_t pairs = (a.m + 1) / 2;
  for (int64_t n = n0; n < n1; n += kGemmRowGroup) {
    const int valid = static_cast<int>(std::min<int64_t>(kGemmRowGroup, n1 - n));
    const WGroup wg = unpack_wgroup<T>(base, w_row_bytes, n, valid, a.k, scratch);
    int64_t ap = 0;
    for (; ap + 4 <= pairs; ap += 4) gemm_tile<T, M, 4>(wg, a, ap, n, valid, y, y_stride, bias);
    switch (pairs - ap) {
      case 3: gemm_tile<T, M, 3>(wg, a, ap, n, valid, y, y_stride, bias); break;
      case 2: gemm_tile<T, M, 2>(wg, a, ap, n, valid, y, y_stride, bias); break;
      case 1: gemm_tile<T, M, 1>(wg, a, ap, n, valid, y, y_stride, bias); break;
      default: break;
    }
  }
}

// A packed 4-row group (Q4_K) as a WGroup: pair layout from the lane layout
// by 32-bit zips (row 2p's 4-byte runs i4 = 2c, 2c + 1 form its 8-byte chunk c).
WGroup unpack_wgroup_packed(const PackedQ4K* grp, int64_t k, GemmScratch& scratch) {
  const int64_t kb = k / kBlock, nb = k / 256;
  const size_t qbytes = static_cast<size_t>(4 * k);
  const size_t need = qbytes + static_cast<size_t>(2 * kb) * 4 * sizeof(float) + 128;
  if (scratch.bytes.size() < need) scratch.bytes.resize(need);
  auto* base = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(scratch.bytes.data()) + 63) & ~uintptr_t{63});
  WGroup wg{reinterpret_cast<int8_t*>(base), reinterpret_cast<float*>(base + qbytes),
            reinterpret_cast<float*>(base + qbytes + static_cast<size_t>(kb) * 4 * sizeof(float))};
  const uint8x16_t m4 = vdupq_n_u8(0x0F);
  for (int64_t b = 0; b < nb; ++b) {
    const PackedQ4K& P = grp[b];
    float32x4_t sv[8], mv[8];
    packed_q4k_scales(P, sv, mv);
    for (int j = 0; j < 8; ++j) {
      vst1q_f32(wg.s + (b * 8 + j) * 4, sv[j]);
      vst1q_f32(wg.mn + (b * 8 + j) * 4, mv[j]);
    }
    for (int jp = 0; jp < 4; ++jp) {
      for (int c = 0; c < 4; ++c) {
        const uint8x16_t w0 = vld1q_u8(P.qs[jp][2 * c][0]), w1 = vld1q_u8(P.qs[jp][2 * c + 1][0]);
        const uint32x4_t l0 = vreinterpretq_u32_u8(vandq_u8(w0, m4)), l1 = vreinterpretq_u32_u8(vandq_u8(w1, m4));
        const uint32x4_t h0 = vreinterpretq_u32_u8(vshrq_n_u8(w0, 4)), h1 = vreinterpretq_u32_u8(vshrq_n_u8(w1, 4));
        const int64_t sub_lo = b * 8 + 2 * jp, sub_hi = sub_lo + 1;
        // Pair 0 = rows 0, 1 (lanes 0, 1); pair 1 = rows 2, 3 (lanes 2, 3).
        vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + sub_lo * 64 + c * 16, vreinterpretq_u8_u32(vzip1q_u32(l0, l1)));
        vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + 2 * k + sub_lo * 64 + c * 16, vreinterpretq_u8_u32(vzip2q_u32(l0, l1)));
        vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + sub_hi * 64 + c * 16, vreinterpretq_u8_u32(vzip1q_u32(h0, h1)));
        vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + 2 * k + sub_hi * 64 + c * 16, vreinterpretq_u8_u32(vzip2q_u32(h0, h1)));
      }
    }
  }
  return wg;
}

// A packed Q6_K group as a WGroup (per-16 scales [unit][row]).
WGroup unpack_wgroup_packed_q6k(const PackedQ6K* grp, int64_t k, GemmScratch& scratch) {
  const int64_t kb = k / kBlock, nb = k / 256;
  const size_t qbytes = static_cast<size_t>(4 * k);
  const size_t need = qbytes + static_cast<size_t>(3 * kb) * 4 * sizeof(float) + 128;
  if (scratch.bytes.size() < need) scratch.bytes.resize(need);
  auto* base = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(scratch.bytes.data()) + 63) & ~uintptr_t{63});
  WGroup wg{reinterpret_cast<int8_t*>(base), reinterpret_cast<float*>(base + qbytes),
            reinterpret_cast<float*>(base + qbytes + static_cast<size_t>(2 * kb) * 4 * sizeof(float))};
  for (int64_t b = 0; b < nb; ++b) {
    const PackedQ6K& P = grp[b];
    float32x4_t sv[16];
    packed_q6k_scales(P, sv);
    for (int u = 0; u < 16; ++u) vst1q_f32(wg.s + (b * 16 + u) * 4, sv[u]);
    for (int n = 0; n < 2; ++n) {
      for (int c = 0; c < 4; ++c) {
        int8x16_t c0[4], c1[4];
        q6k_codes(P, n, 2 * c, c0);
        q6k_codes(P, n, 2 * c + 1, c1);
        for (int qq = 0; qq < 4; ++qq) {
          const int64_t sub = b * 8 + 4 * n + qq;
          const uint32x4_t x0 = vreinterpretq_u32_s8(c0[qq]), x1 = vreinterpretq_u32_s8(c1[qq]);
          vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + sub * 64 + c * 16, vreinterpretq_u8_u32(vzip1q_u32(x0, x1)));
          vst1q_u8(reinterpret_cast<uint8_t*>(wg.q) + 2 * k + sub * 64 + c * 16, vreinterpretq_u8_u32(vzip2q_u32(x0, x1)));
        }
      }
    }
  }
  return wg;
}

template <DType T, ActMode M>
void gemm_packed_t(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, float* y, int64_t y_stride,
                   const float* bias, GemmScratch& scratch) {
  const int64_t nb = k / 256, pairs = (a.m + 1) / 2;
  for (int64_t g = g0; g < g1; ++g) {
    WGroup wg;
    if constexpr (T == DType::kQ4_K) {
      wg = unpack_wgroup_packed(static_cast<const PackedQ4K*>(p) + g * nb, k, scratch);
    } else {
      wg = unpack_wgroup_packed_q6k(static_cast<const PackedQ6K*>(p) + g * nb, k, scratch);
    }
    const int64_t n = 4 * g;
    int64_t ap = 0;
    for (; ap + 4 <= pairs; ap += 4) gemm_tile<T, M, 4>(wg, a, ap, n, 4, y, y_stride, bias);
    switch (pairs - ap) {
      case 3: gemm_tile<T, M, 3>(wg, a, ap, n, 4, y, y_stride, bias); break;
      case 2: gemm_tile<T, M, 2>(wg, a, ap, n, 4, y, y_stride, bias); break;
      case 1: gemm_tile<T, M, 1>(wg, a, ap, n, 4, y, y_stride, bias); break;
      default: break;
    }
  }
}

void gemm_packed(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, float* y, int64_t y_stride,
                 const float* bias, GemmScratch& scratch) {
  if (wt == DType::kQ6_K) {
    if (a.mode == ActMode::kInt8) {
      gemm_packed_t<DType::kQ6_K, ActMode::kInt8>(p, k, g0, g1, a, y, y_stride, bias, scratch);
    } else {
      gemm_packed_t<DType::kQ6_K, ActMode::kInt16>(p, k, g0, g1, a, y, y_stride, bias, scratch);
    }
    return;
  }
  if (a.mode == ActMode::kInt8) {
    gemm_packed_t<DType::kQ4_K, ActMode::kInt8>(p, k, g0, g1, a, y, y_stride, bias, scratch);
  } else {
    gemm_packed_t<DType::kQ4_K, ActMode::kInt16>(p, k, g0, g1, a, y, y_stride, bias, scratch);
  }
}

void gemm(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, float* y, int64_t y_stride,
          const float* bias, GemmScratch& scratch) {
  dispatch_format(wt, [&](auto t) {
    constexpr DType T = decltype(t)::value;
    if (a.mode == ActMode::kInt8) {
      gemm_t<T, ActMode::kInt8>(w, w_row_bytes, n0, n1, a, y, y_stride, bias, scratch);
    } else {
      gemm_t<T, ActMode::kInt16>(w, w_row_bytes, n0, n1, a, y, y_stride, bias, scratch);
    }
  });
}
#endif  // QMM_I8MM

}  // namespace
}  // namespace dynacore::qmm::QMM_NS

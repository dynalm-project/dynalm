// qmm scalar reference tier, ActBuffer and tier selection (dynacore/cpu/qmm.h).
//
// The reference is deliberately plain: exact int32 sub-block dots, then one
// fp32 multiply-add per scale unit in storage order. Optimized tiers are
// tested against it (dynacore/tests/test_qmm.cpp).

#include "dynacore/cpu/qmm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "dynacore/hardware/cpu_info.h"
#include "qmm_unpack.h"

namespace dynacore::qmm {
namespace {

void quantize_row_ref(const float* x, int64_t k, ActMode mode, int8_t* hi, uint8_t* lo, int64_t cs, float* d, float* c,
                      int64_t ds) {
  const float qmax = mode == ActMode::kInt8 ? 127.0f : 32767.0f;
  for (int64_t b = 0; b < k / kBlock; ++b, x += kBlock) {
    float amax = 0;
    for (int i = 0; i < kBlock; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float db = amax / qmax;
    const float id = db > 0 ? 1.0f / db : 0.0f;
    int32_t sum = 0;
    for (int i = 0; i < kBlock; ++i) {
      // Clamp first: a no-op for finite inputs, keeps NaN / Inf out of the cast.
      float v = x[i] * id;
      v = v > -qmax ? (v < qmax ? v : qmax) : -qmax;
      const auto q = static_cast<int32_t>(std::nearbyint(v));  // ties to even
      sum += q;
      const int64_t at = (b * 4 + i / 8) * cs + i % 8;
      if (mode == ActMode::kInt8) {
        hi[at] = static_cast<int8_t>(q);
      } else {
        hi[at] = static_cast<int8_t>(q >> 8);  // arithmetic shift: q = 256 * hi + lo
        lo[at] = static_cast<uint8_t>(q & 0xFF);
      }
    }
    d[b * ds] = db;
    c[b * ds] = db * static_cast<float>(sum);
  }
}

// Code i of activation row r, as the integer the kernels multiply.
inline int32_t act_code(const Acts& a, int64_t r, int64_t i) {
  int64_t at;
  if (a.paired) {
    at = (r / 2) * 2 * a.k + (i / 8) * 16 + (r & 1) * 8 + i % 8;
  } else {
    at = r * a.k + i;
  }
  return a.mode == ActMode::kInt8 ? a.hi[at] : 256 * a.hi[at] + a.lo[at];
}
inline int64_t act_scale_index(const Acts& a, int64_t r, int64_t b) {
  const int64_t kb = a.k / kBlock;
  return a.paired ? ((r / 2) * kb + b) * 2 + (r & 1) : r * kb + b;
}

template <DType T>
float dot_row_ref(const void* wrow, const Acts& a, int64_t r) {
  using F = Fmt<T>;
  const int64_t nsub_total = a.k / kBlock;
  Group g;
  float acc = 0;
  for (int64_t s0 = 0; s0 < nsub_total; s0 += 8) {
    const int nsub = static_cast<int>(std::min<int64_t>(8, nsub_total - s0));
    unpack_group<T>(wrow, s0, nsub, g);
    for (int j = 0; j < nsub; ++j) {
      const int64_t sb = s0 + j;
      const int64_t si = act_scale_index(a, r, sb);
      const float ad = a.d[si];
      int32_t dot[2] = {0, 0};
      for (int i = 0; i < 32; ++i) dot[i / 16] += static_cast<int32_t>(g.q[32 * j + i]) * act_code(a, r, sb * 32 + i);
      if constexpr (F::kHalf) {
        acc += g.s16[2 * j] * ad * static_cast<float>(dot[0]);
        acc += g.s16[2 * j + 1] * ad * static_cast<float>(dot[1]);
      } else {
        acc += g.s32[j] * ad * static_cast<float>(dot[0] + dot[1]);
      }
      if constexpr (F::kMin) acc -= g.mn[j] * a.c[si];
    }
  }
  return acc;
}

template <DType T>
void gemv_ref_t(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, int rows, float* y,
                int64_t y_stride, const float* bias) {
  const auto* base = static_cast<const std::byte*>(w);
  for (int64_t n = n0; n < n1; ++n) {
    for (int r = 0; r < rows; ++r) {
      y[(r0 + r) * y_stride + n] = dot_row_ref<T>(base + n * w_row_bytes, a, r0 + r) + (bias ? bias[n] : 0.0f);
    }
  }
}

void gemv_ref(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, int rows,
              float* y, int64_t y_stride, const float* bias) {
  dispatch_format(wt, [&](auto t) { gemv_ref_t<decltype(t)::value>(w, w_row_bytes, n0, n1, a, r0, rows, y, y_stride, bias); });
}

void gemm_ref(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, float* y, int64_t y_stride,
              const float* bias, GemmScratch&) {
  gemv_ref(wt, w, w_row_bytes, n0, n1, a, 0, static_cast<int>(a.m), y, y_stride, bias);
}

// Packed weights (reference): each row's Group rebuilt from the packed copy,
// then dot_row_ref's arithmetic.
template <DType T, typename P>
float dot_packed_ref(const P* grp, int r, const Acts& a, int64_t ar) {
  using F = Fmt<T>;
  const int64_t nb = a.k / 256;
  Group g;
  float acc = 0;
  for (int64_t b = 0; b < nb; ++b) {
    if constexpr (T == DType::kQ4_K) {
      unpack_packed_q4k(grp[b], r, g);
    } else {
      unpack_packed_q6k(grp[b], r, g);
    }
    for (int j = 0; j < 8; ++j) {
      const int64_t sb = b * 8 + j;
      const int64_t si = act_scale_index(a, ar, sb);
      int32_t dot[2] = {0, 0};
      for (int i = 0; i < 32; ++i) dot[i / 16] += static_cast<int32_t>(g.q[32 * j + i]) * act_code(a, ar, sb * 32 + i);
      if constexpr (F::kHalf) {
        acc += g.s16[2 * j] * a.d[si] * static_cast<float>(dot[0]);
        acc += g.s16[2 * j + 1] * a.d[si] * static_cast<float>(dot[1]);
      } else {
        acc += g.s32[j] * a.d[si] * static_cast<float>(dot[0] + dot[1]);
      }
      if constexpr (F::kMin) acc -= g.mn[j] * a.c[si];
    }
  }
  return acc;
}

template <DType T, typename P>
void gemv_packed_ref_t(const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows, float* y,
                       int64_t y_stride, const float* bias) {
  const auto* pk = static_cast<const P*>(p);
  const int64_t nb = k / 256;
  for (int64_t g = g0; g < g1; ++g) {
    for (int r = 0; r < 4; ++r) {
      const int64_t n = 4 * g + r;
      for (int t = 0; t < rows; ++t) {
        y[(r0 + t) * y_stride + n] = dot_packed_ref<T>(pk + g * nb, r, a, r0 + t) + (bias ? bias[n] : 0.0f);
      }
    }
  }
}

void gemv_packed_ref(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows, float* y,
                     int64_t y_stride, const float* bias) {
  if (wt == DType::kQ4_K) {
    gemv_packed_ref_t<DType::kQ4_K, PackedQ4K>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
  } else {
    gemv_packed_ref_t<DType::kQ6_K, PackedQ6K>(p, k, g0, g1, a, r0, rows, y, y_stride, bias);
  }
}

void gemm_packed_ref(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, float* y, int64_t y_stride,
                     const float* bias, GemmScratch&) {
  gemv_packed_ref(wt, p, k, g0, g1, a, 0, static_cast<int>(a.m), y, y_stride, bias);
}

const Kernels kReference = {
    "reference", /*gemm_matrix=*/false, /*int16=*/true, quantize_row_ref, gemv_ref, gemm_ref, gemv_packed_ref, gemm_packed_ref,
};

size_t align64(size_t n) { return (n + 63) & ~size_t{63}; }

}  // namespace

bool supports(DType wt) {
  return dispatch_format(wt, [](auto) {});
}

bool packable(DType wt, int64_t k) { return (wt == DType::kQ4_K || wt == DType::kQ6_K) && k % 256 == 0; }

size_t packed_bytes(DType wt, int64_t n, int64_t k) {
  if (!packable(wt, k)) return 0;
  const size_t unit = wt == DType::kQ4_K ? sizeof(PackedQ4K) : sizeof(PackedQ6K);
  return static_cast<size_t>(n / 4) * static_cast<size_t>(k / 256) * unit;
}

void pack_groups(DType wt, const void* w, int64_t w_row_bytes, int64_t k, int64_t g0, int64_t g1, void* out) {
  const int64_t nb = k / 256;
  const auto* base = static_cast<const std::byte*>(w);
  for (int64_t g = g0; g < g1; ++g) {
    if (wt == DType::kQ4_K) {
      const quant::BlockQ4_K* rows[4];
      for (int r = 0; r < 4; ++r) rows[r] = reinterpret_cast<const quant::BlockQ4_K*>(base + (4 * g + r) * w_row_bytes);
      for (int64_t b = 0; b < nb; ++b) pack_q4k(rows, b, static_cast<PackedQ4K*>(out)[g * nb + b]);
    } else {
      const quant::BlockQ6_K* rows[4];
      for (int r = 0; r < 4; ++r) rows[r] = reinterpret_cast<const quant::BlockQ6_K*>(base + (4 * g + r) * w_row_bytes);
      for (int64_t b = 0; b < nb; ++b) pack_q6k(rows, b, static_cast<PackedQ6K*>(out)[g * nb + b]);
    }
  }
}

const Kernels& reference_kernels() { return kReference; }

const Kernels* kernels_by_name(const char* name) {
  const std::string_view n(name);
  if (n == "reference") return &kReference;
  const CpuFeatures& f = cpu_info().features;
  if (n == "neon_dot") return f.arm_dotprod ? neon_dot_kernels() : nullptr;
  if (n == "neon_i8mm") return f.arm_dotprod && f.arm_i8mm ? neon_i8mm_kernels() : nullptr;
  return nullptr;
}

const Kernels& best_kernels() {
  static const Kernels* best = [] {
    // DYNACORE_QMM_TIER=reference|neon_dot|neon_i8mm forces a tier (when usable).
    if (const char* v = std::getenv("DYNACORE_QMM_TIER"); v != nullptr && *v != '\0') {
      if (const Kernels* k = kernels_by_name(v)) return k;
    }
    for (const char* n : {"neon_i8mm", "neon_dot"}) {
      if (const Kernels* k = kernels_by_name(n)) return k;
    }
    return &kReference;
  }();
  return *best;
}

bool accelerated() { return &best_kernels() != &kReference; }

const SmeGemm* sme_gemm() {
  static const SmeGemm* g = [] {
    // DYNACORE_SME=0 turns the SME tier off.
    if (const char* v = std::getenv("DYNACORE_SME"); v != nullptr && v[0] == '0') return static_cast<const SmeGemm*>(nullptr);
    return cpu_info().features.arm_sme ? sme_kernels() : nullptr;
  }();
  return g;
}

namespace {

// One row's dequantized values [kk0, kk0 + count) (count <= 256) from a Group.
inline void group_values(const Group& g, int count, float* out) {
  for (int i = 0; i < count; ++i) out[i] = g.s16[i / 16] * static_cast<float>(g.q[i]) - g.mn[i / 32];
}

#if defined(__ARM_NEON) && defined(__aarch64__)
// NEON panel writers (base ARMv8 NEON only): 4 rows at a time, each 16-byte
// load holding 4 rows x 4 codes is transposed to 4 values x 4 rows with one
// TBL, so every store writes 4 consecutive panel columns.
const uint8x16_t kT4 = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

// p[0..3] (4 consecutive k) x 4 rows from 16 codes [t][row] (u8), w = s * q - mn per row.
inline void store4x4_u8(uint8x16_t q, float32x4_t s, float32x4_t mn, float* p) {
  const uint16x8_t lo = vmovl_u8(vget_low_u8(q)), hi = vmovl_u8(vget_high_u8(q));
  vst1q_f32(p + 0 * kSmePanel, vfmsq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), s), mn, vdupq_n_f32(1.0f)));
  vst1q_f32(p + 1 * kSmePanel, vfmsq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), s), mn, vdupq_n_f32(1.0f)));
  vst1q_f32(p + 2 * kSmePanel, vfmsq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), s), mn, vdupq_n_f32(1.0f)));
  vst1q_f32(p + 3 * kSmePanel, vfmsq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), s), mn, vdupq_n_f32(1.0f)));
}
// The same for signed codes, no min.
inline void store4x4_s8(int8x16_t q, float32x4_t s, float* p) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
  vst1q_f32(p + 0 * kSmePanel, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s));
  vst1q_f32(p + 1 * kSmePanel, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), s));
  vst1q_f32(p + 2 * kSmePanel, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s));
  vst1q_f32(p + 3 * kSmePanel, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), s));
}

// 4 packed rows (one group) into panel columns [c, c + 4): exactly
// unpack_packed_q4k + group_values (the same fp32 products and differences).
void panel_packed_q4k(const PackedQ4K* g, int64_t nb, float* panel, int c) {
  for (int64_t b = 0; b < nb; ++b) {
    const PackedQ4K& p = g[b];
    float d[4], dm[4];
    for (int r = 0; r < 4; ++r) {
      d[r] = fp16_to_fp32(p.d[r]);
      dm[r] = fp16_to_fp32(p.dmin[r]);
    }
    const float32x4_t dv = vld1q_f32(d), dmv = vld1q_f32(dm);
    float* base = panel + b * 256 * kSmePanel + c;
    for (int jp = 0; jp < 4; ++jp) {
      const float32x4_t s0 = vmulq_f32(dv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(reinterpret_cast<const uint32_t*>(p.sc[2 * jp]))))))));
      const float32x4_t s1 = vmulq_f32(dv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(reinterpret_cast<const uint32_t*>(p.sc[2 * jp + 1]))))))));
      const float32x4_t m0 = vmulq_f32(dmv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(reinterpret_cast<const uint32_t*>(p.mn[2 * jp]))))))));
      const float32x4_t m1 = vmulq_f32(dmv, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vld1_dup_u32(reinterpret_cast<const uint32_t*>(p.mn[2 * jp + 1]))))))));
      for (int i4 = 0; i4 < 8; ++i4) {
        const uint8x16_t q = vqtbl1q_u8(vld1q_u8(&p.qs[jp][i4][0][0]), kT4);
        store4x4_u8(vandq_u8(q, vdupq_n_u8(0x0F)), s0, m0, base + (64 * jp + 4 * i4) * kSmePanel);
        store4x4_u8(vshrq_n_u8(q, 4), s1, m1, base + (64 * jp + 32 + 4 * i4) * kSmePanel);
      }
    }
  }
}

// As unpack_packed_q6k + group_values.
void panel_packed_q6k(const PackedQ6K* g, int64_t nb, float* panel, int c) {
  for (int64_t b = 0; b < nb; ++b) {
    const PackedQ6K& p = g[b];
    float d[4];
    for (int r = 0; r < 4; ++r) d[r] = fp16_to_fp32(p.d[r]);
    const float32x4_t dv = vld1q_f32(d);
    float* base = panel + b * 256 * kSmePanel + c;
    for (int n = 0; n < 2; ++n) {
      for (int i4 = 0; i4 < 8; ++i4) {
        const uint8x16_t A = vqtbl1q_u8(vld1q_u8(&p.ql[n][0][i4][0][0]), kT4);
        const uint8x16_t B = vqtbl1q_u8(vld1q_u8(&p.ql[n][1][i4][0][0]), kT4);
        const uint8x16_t H = vqtbl1q_u8(vld1q_u8(&p.qh[n][i4][0][0]), kT4);
        const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
        const uint8x16_t q[4] = {
            vorrq_u8(vandq_u8(A, m4), vshlq_n_u8(vandq_u8(H, m2), 4)),
            vorrq_u8(vandq_u8(B, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(H, 2), m2), 4)),
            vorrq_u8(vshrq_n_u8(A, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(H, 4), m2), 4)),
            vorrq_u8(vshrq_n_u8(B, 4), vshlq_n_u8(vshrq_n_u8(H, 6), 4)),
        };
        for (int qq = 0; qq < 4; ++qq) {
          const int kk = n * 128 + 32 * qq + 4 * i4;  // first of the 4 values
          const int u = kk / 16;
          const float32x4_t s = vmulq_f32(
              dv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vreinterpret_s8_u32(vld1_dup_u32(reinterpret_cast<const uint32_t*>(p.sc[u]))))))));
          store4x4_s8(vsubq_s8(vreinterpretq_s8_u8(q[qq]), vdupq_n_s8(32)), s, base + kk * kSmePanel);
        }
      }
    }
  }
}

// 4 Q8_0 rows (original layout) into panel columns [c, c + 4).
void panel_q8_0(const std::byte* const rows[4], int64_t k, float* panel, int c) {
  for (int64_t b = 0; b < k / 32; ++b) {
    const quant::BlockQ8_0* blk[4];
    float d[4];
    for (int r = 0; r < 4; ++r) {
      blk[r] = reinterpret_cast<const quant::BlockQ8_0*>(rows[r]) + b;
      d[r] = fp16_to_fp32(blk[r]->d);
    }
    const float32x4_t dv = vld1q_f32(d);
    for (int h = 0; h < 2; ++h) {
      const int8x16_t r0 = vld1q_s8(blk[0]->qs + 16 * h), r1 = vld1q_s8(blk[1]->qs + 16 * h);
      const int8x16_t r2 = vld1q_s8(blk[2]->qs + 16 * h), r3 = vld1q_s8(blk[3]->qs + 16 * h);
      // [i][row] for i = 0..15: zip rows pairwise, then the pairs.
      const int16x8_t a0 = vreinterpretq_s16_s8(vzip1q_s8(r0, r1)), a1 = vreinterpretq_s16_s8(vzip2q_s8(r0, r1));
      const int16x8_t b0 = vreinterpretq_s16_s8(vzip1q_s8(r2, r3)), b1 = vreinterpretq_s16_s8(vzip2q_s8(r2, r3));
      const int8x16_t t[4] = {vreinterpretq_s8_s16(vzip1q_s16(a0, b0)), vreinterpretq_s8_s16(vzip2q_s16(a0, b0)),
                              vreinterpretq_s8_s16(vzip1q_s16(a1, b1)), vreinterpretq_s8_s16(vzip2q_s16(a1, b1))};
      for (int qq = 0; qq < 4; ++qq) store4x4_s8(t[qq], dv, panel + (b * 32 + 16 * h + 4 * qq) * kSmePanel + c);
    }
  }
}
#endif

template <DType T>
void panel_rows(const void* w, int64_t w_row_bytes, const void* packed, int64_t n, int64_t k, int64_t n0, float* panel) {
  const int64_t packed_rows = packed != nullptr ? n / 4 * 4 : 0;
  const int64_t nb = k / 256;
  Group g;
  float v[256];
  int j0 = 0;  // first row left to the scalar path below
#if defined(__ARM_NEON) && defined(__aarch64__)
  if constexpr (T == DType::kQ4_K || T == DType::kQ6_K) {
    // Whole packed groups (n0 is a multiple of 32, so of 4).
    for (; j0 + 4 <= kSmePanel && n0 + j0 + 4 <= packed_rows; j0 += 4) {
      const int64_t grp = (n0 + j0) / 4;
      if constexpr (T == DType::kQ4_K) {
        panel_packed_q4k(static_cast<const PackedQ4K*>(packed) + grp * nb, nb, panel, j0);
      } else {
        panel_packed_q6k(static_cast<const PackedQ6K*>(packed) + grp * nb, nb, panel, j0);
      }
    }
    // Unpacked rows (packing skipped by the memory budget, or none made):
    // pack 4 rows' super-block on the fly, then the same writer.
    for (; j0 + 4 <= kSmePanel && n0 + j0 + 4 <= n; j0 += 4) {
      using Blk = std::conditional_t<T == DType::kQ4_K, quant::BlockQ4_K, quant::BlockQ6_K>;
      const Blk* rows[4];
      for (int r = 0; r < 4; ++r) {
        rows[r] = reinterpret_cast<const Blk*>(static_cast<const std::byte*>(w) + (n0 + j0 + r) * w_row_bytes);
      }
      for (int64_t b = 0; b < nb; ++b) {
        if constexpr (T == DType::kQ4_K) {
          PackedQ4K one;
          pack_q4k(rows, b, one);
          panel_packed_q4k(&one, 1, panel + b * 256 * kSmePanel, j0);
        } else {
          PackedQ6K one;
          pack_q6k(rows, b, one);
          panel_packed_q6k(&one, 1, panel + b * 256 * kSmePanel, j0);
        }
      }
    }
  } else if constexpr (T == DType::kQ8_0) {
    for (; j0 + 4 <= kSmePanel && n0 + j0 + 4 <= n; j0 += 4) {
      const std::byte* rows[4];
      for (int r = 0; r < 4; ++r) rows[r] = static_cast<const std::byte*>(w) + (n0 + j0 + r) * w_row_bytes;
      panel_q8_0(rows, k, panel, j0);
    }
  }
#endif
  for (int j = j0; j < kSmePanel; ++j) {
    const int64_t r = n0 + j;
    if (r >= n) {
      for (int64_t kk = 0; kk < k; ++kk) panel[kk * kSmePanel + j] = 0.0f;
      continue;
    }
    for (int64_t kk0 = 0; kk0 < k; kk0 += 256) {
      const int count = static_cast<int>(k - kk0 < 256 ? k - kk0 : 256);
      if constexpr (T == DType::kQ4_K || T == DType::kQ6_K) {
        if (r < packed_rows) {
          if constexpr (T == DType::kQ4_K) {
            unpack_packed_q4k(static_cast<const PackedQ4K*>(packed)[(r / 4) * nb + kk0 / 256], static_cast<int>(r % 4), g);
          } else {
            unpack_packed_q6k(static_cast<const PackedQ6K*>(packed)[(r / 4) * nb + kk0 / 256], static_cast<int>(r % 4), g);
          }
          group_values(g, count, v);
          for (int i = 0; i < count; ++i) panel[(kk0 + i) * kSmePanel + j] = v[i];
          continue;
        }
      }
      unpack_group<T>(static_cast<const std::byte*>(w) + r * w_row_bytes, kk0 / kBlock, count / kBlock, g);
      group_values(g, count, v);
      for (int i = 0; i < count; ++i) panel[(kk0 + i) * kSmePanel + j] = v[i];
    }
  }
}

}  // namespace

void weight_panel_f32(DType wt, const void* w, int64_t w_row_bytes, const void* packed, int64_t n, int64_t k, int64_t n0,
                      float* panel) {
  dispatch_format(wt, [&](auto tag) { panel_rows<decltype(tag)::value>(w, w_row_bytes, packed, n, k, n0, panel); });
}

void ActBuffer::prepare(ActMode mode, int64_t m, int64_t k, bool paired) {
  const int64_t m_pad = paired ? (m + 1) / 2 * 2 : m;
  const int64_t kb = k / kBlock;
  const size_t code_bytes = align64(static_cast<size_t>(m_pad * k));
  const size_t lo_bytes = mode == ActMode::kInt16 ? code_bytes : 0;
  const size_t scale_bytes = align64(static_cast<size_t>(m_pad * kb) * sizeof(float));
  const size_t total = code_bytes + lo_bytes + 2 * scale_bytes + 64;
  if (storage_.size() < total) storage_.resize(total);
  auto* base = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(storage_.data()) + 63) & ~uintptr_t{63});
  acts_.mode = mode;
  acts_.paired = paired;
  acts_.m = m;
  acts_.k = k;
  acts_.hi = reinterpret_cast<int8_t*>(base);
  acts_.lo = mode == ActMode::kInt16 ? reinterpret_cast<uint8_t*>(base + code_bytes) : nullptr;
  acts_.d = reinterpret_cast<float*>(base + code_bytes + lo_bytes);
  acts_.c = reinterpret_cast<float*>(base + code_bytes + lo_bytes + scale_bytes);
  if (paired && (m & 1)) {
    // The pad row of the last pair: zero codes and scales (never stored).
    const int64_t r = m;
    auto* hi = const_cast<int8_t*>(acts_.hi);
    auto* lo = const_cast<uint8_t*>(acts_.lo);
    for (int64_t j = 0; j < k / 8; ++j) {
      std::memset(hi + (r / 2) * 2 * k + j * 16 + 8, 0, 8);
      if (lo) std::memset(lo + (r / 2) * 2 * k + j * 16 + 8, 0, 8);
    }
    auto* d = const_cast<float*>(acts_.d);
    auto* c = const_cast<float*>(acts_.c);
    for (int64_t b = 0; b < kb; ++b) d[((r / 2) * kb + b) * 2 + 1] = c[((r / 2) * kb + b) * 2 + 1] = 0.0f;
  }
}

void ActBuffer::quantize_rows(const float* x, int64_t x_stride, int64_t r0, int64_t r1, const Kernels& kern) {
  const int64_t k = acts_.k, kb = k / kBlock;
  auto* hi = const_cast<int8_t*>(acts_.hi);
  auto* lo = const_cast<uint8_t*>(acts_.lo);
  auto* d = const_cast<float*>(acts_.d);
  auto* c = const_cast<float*>(acts_.c);
  for (int64_t r = r0; r < r1; ++r) {
    const float* xr = x + r * x_stride;
    if (acts_.paired) {
      const int64_t off = (r / 2) * 2 * k + (r & 1) * 8;
      const int64_t so = (r / 2) * kb * 2 + (r & 1);
      kern.quantize_row(xr, k, acts_.mode, hi + off, lo ? lo + off : nullptr, 16, d + so, c + so, 2);
    } else {
      kern.quantize_row(xr, k, acts_.mode, hi + r * k, lo ? lo + r * k : nullptr, 8, d + r * kb, c + r * kb, 1);
    }
  }
}

}  // namespace dynacore::qmm

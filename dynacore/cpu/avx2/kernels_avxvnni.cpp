// AVX-VNNI variant of the interleaved Q4_K decode kernel (DD-080).
//
// vpdpbusd multiplies 4 unsigned weight bytes by 4 signed activation bytes and
// adds the sum into an int32 lane. With the BlockQ4_Kx8 layout (4 values of 8
// weight rows per 32 bytes) each int32 lane is one weight row, so a 32-value
// sub-block is 8 vpdpbusd instead of 8 maddubs + 8 int16 adds + 1 madd. The
// integer accumulation and the half-super-block conversion are the same as in
// the AVX2 kernel, so results are identical up to the order of integer adds
// (exact): |lane| <= 32 * 15 * 127 per sub-block, times sc * m <= 8064, over
// 4 sub-blocks < 1.97e9.
//
// Built with AVX-VNNI enabled (engine_set_isa AVXVNNI) and registered only
// when the CPU reports AVX-VNNI (Alder Lake and later, Zen 5).

#include "dynacore/cpu/cpu_kernels.h"

#if ENGINE_HAS_AVX2 && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "dynacore/quantization/repack.h"

namespace dynacore {
namespace {

using quant::kQK_K;

inline __m256i dpbusd(__m256i acc, __m256i u8, __m256i s8) { return _mm256_dpbusd_avx_epi32(acc, u8, s8); }

template <int M>
void dot_x8_sx_q4_K_vnni_m(const BlockQ4_Kx8* w, const ActSuperQ8* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ4_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d), dminv = _mm256_loadu_ps(p.dmin);
    __m256i sumi[M];
    __m256 sumf[M];
    for (int r = 0; r < M; ++r) {
      sumi[r] = _mm256_setzero_si256();
      sumf[r] = _mm256_setzero_ps();
    }
    for (int jp = 0; jp < 4; ++jp) {
      __m256i wl[8], wh[8];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.qs[jp][c]));
        wl[c] = _mm256_and_si256(q, low4);
        wh[c] = _mm256_and_si256(_mm256_srli_epi16(q, 4), low4);
      }
      const int j0 = 2 * jp, j1 = 2 * jp + 1;
      // 8 rows' scales as int32 lanes; dmin * mn as the AVX2 kernel forms it.
      const __m256i s0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j0])));
      const __m256i s1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j1])));
      auto min_terms = [&](int j) {
        return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j])))),
                             dminv);
      };
      const __m256 mn0 = min_terms(j0), mn1 = min_terms(j1);
      for (int r = 0; r < M; ++r) {
        const ActSuperQ8& a = x[r][i];
        const int8_t* x0 = a.q + 32 * j0;
        const int8_t* x1 = a.q + 32 * j1;
        __m256i sa = _mm256_setzero_si256(), sb = _mm256_setzero_si256();
        for (int c = 0; c < 8; ++c) {
          int32_t b0, b1;
          std::memcpy(&b0, x0 + 4 * c, 4);
          std::memcpy(&b1, x1 + 4 * c, 4);
          sa = dpbusd(sa, wl[c], _mm256_set1_epi32(b0));
          sb = dpbusd(sb, wh[c], _mm256_set1_epi32(b1));
        }
        const __m256i pa = _mm256_mullo_epi32(sa, _mm256_mullo_epi32(s0, _mm256_set1_epi32(a.m[j0])));
        const __m256i pb = _mm256_mullo_epi32(sb, _mm256_mullo_epi32(s1, _mm256_set1_epi32(a.m[j1])));
        sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(pa, pb));
        acc[r] = _mm256_fnmadd_ps(mn0, _mm256_set1_ps(a.ms[j0]), acc[r]);
        acc[r] = _mm256_fnmadd_ps(mn1, _mm256_set1_ps(a.ms[j1]), acc[r]);
      }
      if (jp == 1 || jp == 3) {
        for (int r = 0; r < M; ++r) {
          sumf[r] = _mm256_add_ps(sumf[r], _mm256_cvtepi32_ps(sumi[r]));
          sumi[r] = _mm256_setzero_si256();
        }
      }
    }
    for (int r = 0; r < M; ++r) {
      acc[r] = _mm256_fmadd_ps(sumf[r], _mm256_mul_ps(dv, _mm256_set1_ps(x[r][i].u)), acc[r]);
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

void dot_x8_sx_q4_K_vnni(const void* w, const ActBlockQ8* const* xin, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  const ActSuperQ8* x[kDotRowsMax];
  for (int r = 0; r < m; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  switch (m) {
    case 1: dot_x8_sx_q4_K_vnni_m<1>(pw, x, n, out); break;
    case 2: dot_x8_sx_q4_K_vnni_m<2>(pw, x, n, out); break;
    case 3: dot_x8_sx_q4_K_vnni_m<3>(pw, x, n, out); break;
    default: dot_x8_sx_q4_K_vnni_m<4>(pw, x, n, out); break;
  }
}

// --- int16 x8 tiles with vpdpwssd (KernelPlan::int16_prefill_max_rows) -----
// The int16 x8 kernels of kernels_avx2.cpp (DD-081, DD-082) for up to
// kPackedRowsMax activation rows, with each madd_epi16 + add_epi32 fused into
// one vpdpwssd. vpdpwssd adds the same two exact int16 products per lane into
// the same int32 accumulator without saturation, so every result is bit for
// bit the one of the AVX2 kernel. Registered (register_avxvnni_int16_tiles)
// where the CPU reports AVX-VNNI; only the experimental prefill path uses it.
inline __m256i dpwssd(__m256i acc, __m256i a, __m256i b) { return _mm256_dpwssd_avx_epi32(acc, a, b); }

template <int M>
void dot_x8_q16_q4_K_vnni_m(const BlockQ4_Kx8* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i row_order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ4_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d), dminv = _mm256_loadu_ps(p.dmin);
    for (int jp = 0; jp < 4; ++jp) {
      __m256i wl[8][2], wh[8][2];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.qs[jp][c]));
        const __m256i lo = _mm256_and_si256(q, low4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), low4);
        wl[c][0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo));
        wl[c][1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1));
        wh[c][0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi));
        wh[c][1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1));
      }
      const int j0 = 2 * jp, j1 = 2 * jp + 1;
      auto scale_f = [&](int j) {
        return _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j])))));
      };
      auto min_f = [&](int j) {
        return _mm256_mul_ps(dminv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j])))));
      };
      const __m256 ds0 = scale_f(j0), ds1 = scale_f(j1), dm0 = min_f(j0), dm1 = min_f(j1);
      for (int r = 0; r < M; ++r) {
        const ActBlockQ16& a0 = x[r][i * 8 + j0];
        const ActBlockQ16& a1 = x[r][i * 8 + j1];
        __m256i s0a = _mm256_setzero_si256(), s0b = _mm256_setzero_si256();
        __m256i s1a = _mm256_setzero_si256(), s1b = _mm256_setzero_si256();
        for (int c = 0; c < 8; ++c) {
          int64_t b0, b1;
          std::memcpy(&b0, a0.q + 4 * c, 8);
          std::memcpy(&b1, a1.q + 4 * c, 8);
          const __m256i x0 = _mm256_set1_epi64x(b0), x1 = _mm256_set1_epi64x(b1);
          s0a = dpwssd(s0a, wl[c][0], x0);
          s0b = dpwssd(s0b, wl[c][1], x0);
          s1a = dpwssd(s1a, wh[c][0], x1);
          s1b = dpwssd(s1b, wh[c][1], x1);
        }
        const __m256i r0 = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(s0a, s0b), row_order);
        const __m256i r1 = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(s1a, s1b), row_order);
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r0), _mm256_mul_ps(ds0, _mm256_set1_ps(a0.d)), acc[r]);
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r1), _mm256_mul_ps(ds1, _mm256_set1_ps(a1.d)), acc[r]);
        acc[r] = _mm256_fnmadd_ps(dm0, _mm256_set1_ps(a0.d * static_cast<float>(a0.sum)), acc[r]);
        acc[r] = _mm256_fnmadd_ps(dm1, _mm256_set1_ps(a1.d * static_cast<float>(a1.sum)), acc[r]);
      }
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

template <int M>
void dot_x8_q16_q6_K_vnni_m(const BlockQ6_Kx8* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F), three = _mm256_set1_epi8(3), k32 = _mm256_set1_epi8(32);
  const __m128i low4h = _mm_set1_epi8(0x0F);
  const __m256i row_order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ6_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d);
    for (int pp = 0; pp < 4; ++pp) {
      __m256i wq[8][4];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.ql[pp][c]));
        const __m128i hb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p.qh[pp][c]));
        const __m256i h = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(hb, 4), low4h), _mm_and_si128(hb, low4h));
        const __m256i a = _mm256_sub_epi8(
            _mm256_or_si256(_mm256_and_si256(q, low4), _mm256_slli_epi16(_mm256_and_si256(h, three), 4)), k32);
        const __m256i b = _mm256_sub_epi8(
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), low4),
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), three), 4)),
            k32);
        wq[c][0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        wq[c][1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        wq[c][2] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
        wq[c][3] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
      }
      __m256 ds[4];
      for (int g = 0; g < 4; ++g) {
        ds[g] = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[4 * pp + g])))));
      }
      for (int r = 0; r < M; ++r) {
        for (int s = 0; s < 2; ++s) {
          const ActBlockQ16& a = x[r][i * 8 + 2 * pp + s];
          __m256i lo_a = _mm256_setzero_si256(), lo_b = _mm256_setzero_si256();
          __m256i hi_a = _mm256_setzero_si256(), hi_b = _mm256_setzero_si256();
          for (int c = 0; c < 4; ++c) {
            int64_t v0, v1;
            std::memcpy(&v0, a.q + 4 * c, 8);
            std::memcpy(&v1, a.q + 16 + 4 * c, 8);
            const __m256i x0 = _mm256_set1_epi64x(v0), x1 = _mm256_set1_epi64x(v1);
            lo_a = dpwssd(lo_a, wq[c][2 * s], x0);
            lo_b = dpwssd(lo_b, wq[c][2 * s + 1], x0);
            hi_a = dpwssd(hi_a, wq[c + 4][2 * s], x1);
            hi_b = dpwssd(hi_b, wq[c + 4][2 * s + 1], x1);
          }
          const __m256i lo = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(lo_a, lo_b), row_order);
          const __m256i hi = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(hi_a, hi_b), row_order);
          const __m256 xd = _mm256_set1_ps(a.d);
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), _mm256_mul_ps(ds[2 * s], xd), acc[r]);
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), _mm256_mul_ps(ds[2 * s + 1], xd), acc[r]);
        }
      }
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

void dot_x8_q16_q4_K_vnni_rows8(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q4_K_vnni_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q4_K_vnni_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q4_K_vnni_m<3>(pw, x, n, out); break;
    case 4: dot_x8_q16_q4_K_vnni_m<4>(pw, x, n, out); break;
    case 5: dot_x8_q16_q4_K_vnni_m<5>(pw, x, n, out); break;
    case 6: dot_x8_q16_q4_K_vnni_m<6>(pw, x, n, out); break;
    case 7: dot_x8_q16_q4_K_vnni_m<7>(pw, x, n, out); break;
    default: dot_x8_q16_q4_K_vnni_m<8>(pw, x, n, out); break;
  }
}

void dot_x8_q16_q6_K_vnni_rows8(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ6_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q6_K_vnni_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q6_K_vnni_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q6_K_vnni_m<3>(pw, x, n, out); break;
    case 4: dot_x8_q16_q6_K_vnni_m<4>(pw, x, n, out); break;
    case 5: dot_x8_q16_q6_K_vnni_m<5>(pw, x, n, out); break;
    case 6: dot_x8_q16_q6_K_vnni_m<6>(pw, x, n, out); break;
    case 7: dot_x8_q16_q6_K_vnni_m<7>(pw, x, n, out); break;
    default: dot_x8_q16_q6_K_vnni_m<8>(pw, x, n, out); break;
  }
}

// --- split-byte int16 x8 tiles (DynaLM Next) ---------------------------------
// The int16 tile above widens every weight code to int16, holds 32 widened
// vectors per 64 values (more than the 16 ymm registers, so they spill and are
// reloaded for every activation row) and folds lanes with hadd + permute. Here
// each activation code is split q = 256 * hi + lo (hi s8, lo u8). A 4-bit
// weight code w (0..15) is a valid u8 and s8 operand, so
//   sum w * q = 256 * dpbusd(w_u8, hi_s8) + dpbusd(lo_u8, w_s8),
// exactly, with one int32 lane per weight row (no fold) and the weight bytes
// used as loaded. |256 * sum w * hi| <= 256 * 32 * 15 * 128 per 32-value
// sub-block and |sum lo * w| <= 32 * 255 * 15: no overflow. The integer
// sub-block dots equal the int16 tile's, and the float steps (scale, FMA order,
// min correction) are the same, so every output is bit-identical to it.
//
// R activation rows share each weight register (R = 3: 12 int32 accumulators,
// 2 weight vectors, 1 broadcast, the nibble mask). Float accumulators live in
// `out` between 64-value steps.

inline __m256i bcast32(const void* p) {
  int32_t v;
  std::memcpy(&v, p, 4);
  return _mm256_set1_epi32(v);
}

template <int R>
void dot_x8_q16s_q4_K_vnni_r(const BlockQ4_Kx8* w, const ActBlockQ16S* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  for (int r = 0; r < R; ++r) _mm256_storeu_ps(out + 8 * r, _mm256_setzero_ps());
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ4_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d), dminv = _mm256_loadu_ps(p.dmin);
    for (int jp = 0; jp < 4; ++jp) {
      const int j0 = 2 * jp, j1 = 2 * jp + 1;
      __m256i h0[R], l0[R], h1[R], l1[R];
      for (int r = 0; r < R; ++r) h0[r] = l0[r] = h1[r] = l1[r] = _mm256_setzero_si256();
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.qs[jp][c]));
        const __m256i wl = _mm256_and_si256(q, low4);
        const __m256i wh = _mm256_and_si256(_mm256_srli_epi16(q, 4), low4);
        for (int r = 0; r < R; ++r) {
          const ActBlockQ16S& a0 = x[r][i * 8 + j0];
          const ActBlockQ16S& a1 = x[r][i * 8 + j1];
          h0[r] = dpbusd(h0[r], wl, bcast32(a0.hi + 4 * c));
          l0[r] = dpbusd(l0[r], bcast32(a0.lo + 4 * c), wl);
          h1[r] = dpbusd(h1[r], wh, bcast32(a1.hi + 4 * c));
          l1[r] = dpbusd(l1[r], bcast32(a1.lo + 4 * c), wh);
        }
      }
      const __m256 ds0 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j0])))));
      const __m256 ds1 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j1])))));
      const __m256 dm0 = _mm256_mul_ps(dminv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j0])))));
      const __m256 dm1 = _mm256_mul_ps(dminv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j1])))));
      for (int r = 0; r < R; ++r) {
        const ActBlockQ16S& a0 = x[r][i * 8 + j0];
        const ActBlockQ16S& a1 = x[r][i * 8 + j1];
        const __m256i r0 = _mm256_add_epi32(_mm256_slli_epi32(h0[r], 8), l0[r]);
        const __m256i r1 = _mm256_add_epi32(_mm256_slli_epi32(h1[r], 8), l1[r]);
        __m256 acc = _mm256_loadu_ps(out + 8 * r);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r0), _mm256_mul_ps(ds0, _mm256_set1_ps(a0.d)), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r1), _mm256_mul_ps(ds1, _mm256_set1_ps(a1.d)), acc);
        acc = _mm256_fnmadd_ps(dm0, _mm256_set1_ps(a0.d * static_cast<float>(a0.sum)), acc);
        acc = _mm256_fnmadd_ps(dm1, _mm256_set1_ps(a1.d * static_cast<float>(a1.sum)), acc);
        _mm256_storeu_ps(out + 8 * r, acc);
      }
    }
  }
}

void dot_x8_q16s_q4_K_vnni_rows8(const void* w, const ActBlockQ16S* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  int r0 = 0;
  for (; r0 + 3 <= m; r0 += 3) dot_x8_q16s_q4_K_vnni_r<3>(pw, x + r0, n, out + 8 * r0);
  if (m - r0 == 2) dot_x8_q16s_q4_K_vnni_r<2>(pw, x + r0, n, out + 8 * r0);
  if (m - r0 == 1) dot_x8_q16s_q4_K_vnni_r<1>(pw, x + r0, n, out + 8 * r0);
}

// Experimental R = 2 / R = 4 variants for the Phase 1 kernel study (not registered).
void dot_x8_q16s_q4_K_vnni_rows8_r2(const void* w, const ActBlockQ16S* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  int r0 = 0;
  for (; r0 + 2 <= m; r0 += 2) dot_x8_q16s_q4_K_vnni_r<2>(pw, x + r0, n, out + 8 * r0);
  if (m - r0 == 1) dot_x8_q16s_q4_K_vnni_r<1>(pw, x + r0, n, out + 8 * r0);
}
void dot_x8_q16s_q4_K_vnni_rows8_r4(const void* w, const ActBlockQ16S* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  int r0 = 0;
  for (; r0 + 4 <= m; r0 += 4) dot_x8_q16s_q4_K_vnni_r<4>(pw, x + r0, n, out + 8 * r0);
  for (; r0 < m; ++r0) dot_x8_q16s_q4_K_vnni_r<1>(pw, x + r0, n, out + 8 * r0);
}

}  // namespace

// Instruction-throughput probe for bench_ffn_kernels (Phase 4 study): `iters`
// rounds of 12 independent vpdpbusd (or vfmadd231ps) on ymm registers, so the
// loop is bound by the unit's throughput, not latency. Returns a value that
// depends on every accumulator (keeps the loop alive).
int64_t avxvnni_probe_dpbusd(int64_t iters) {
  __m256i a[12];
  for (int i = 0; i < 12; ++i) a[i] = _mm256_set1_epi32(i);
  const __m256i u = _mm256_set1_epi8(3), s = _mm256_set1_epi8(-2);
  for (int64_t t = 0; t < iters; ++t) {
    for (int i = 0; i < 12; ++i) a[i] = _mm256_dpbusd_avx_epi32(a[i], u, s);
  }
  __m256i r = a[0];
  for (int i = 1; i < 12; ++i) r = _mm256_add_epi32(r, a[i]);
  return _mm256_extract_epi32(r, 0);
}
int64_t avxvnni_probe_fma(int64_t iters) {
  __m256 a[12];
  for (int i = 0; i < 12; ++i) a[i] = _mm256_set1_ps(static_cast<float>(i));
  const __m256 u = _mm256_set1_ps(0.999f), s = _mm256_set1_ps(1e-7f);
  for (int64_t t = 0; t < iters; ++t) {
    for (int i = 0; i < 12; ++i) a[i] = _mm256_fmadd_ps(a[i], u, s);
  }
  __m256 r = a[0];
  for (int i = 1; i < 12; ++i) r = _mm256_add_ps(r, a[i]);
  return static_cast<int64_t>(_mm256_cvtss_f32(r));
}

// Study hooks for bench_ffn_kernels (declared there; not part of CpuKernels).
CpuKernels::DotQ16SX8Fn avxvnni_q16s_variant(int r) {
  return r == 2 ? dot_x8_q16s_q4_K_vnni_rows8_r2 : r == 4 ? dot_x8_q16s_q4_K_vnni_rows8_r4 : dot_x8_q16s_q4_K_vnni_rows8;
}

bool register_avxvnni_int16_split(CpuKernels& k) {
  k.dot_q16s_x8_q4_K_rows8 = dot_x8_q16s_q4_K_vnni_rows8;
  return true;
}

bool register_avxvnni_int16_tiles(CpuKernels& k) {
  k.dot_q16_x8_q4_K_rows8 = dot_x8_q16_q4_K_vnni_rows8;
  k.dot_q16_x8_q6_K_rows8 = dot_x8_q16_q6_K_vnni_rows8;
  return true;
}

bool register_avxvnni_kernels(CpuKernels& k) {
  k.dot_q8_sx_x8_q4_K = dot_x8_sx_q4_K_vnni;
  // No VNNI 8-row variant: never pair the AVX2 8-row kernel with the VNNI one.
  k.dot_q8_sx_x8_q4_K_rows8 = nullptr;
  return true;
}

}  // namespace dynacore

#else  // AVX2 tier not compiled

namespace dynacore {
bool register_avxvnni_kernels(CpuKernels&) { return false; }
bool register_avxvnni_int16_tiles(CpuKernels&) { return false; }
bool register_avxvnni_int16_split(CpuKernels&) { return false; }
CpuKernels::DotQ16SX8Fn avxvnni_q16s_variant(int) { return nullptr; }
int64_t avxvnni_probe_dpbusd(int64_t) { return 0; }
int64_t avxvnni_probe_fma(int64_t) { return 0; }
}  // namespace dynacore

#endif

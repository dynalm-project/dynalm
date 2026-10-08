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

}  // namespace

bool register_avxvnni_kernels(CpuKernels& k) {
  k.dot_q8_sx_x8_q4_K = dot_x8_sx_q4_K_vnni;
  return true;
}

}  // namespace dynacore

#else  // AVX2 tier not compiled

namespace dynacore {
bool register_avxvnni_kernels(CpuKernels&) { return false; }
}  // namespace dynacore

#endif

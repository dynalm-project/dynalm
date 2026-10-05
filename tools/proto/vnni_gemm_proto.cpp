// P9 feasibility prototype: fp32 (dequant panel + 4x3 FMA tile) vs int8 VNNI
// (3 weight rows x 4 activation rows) on Q8_0-shaped data, single thread.
#include <immintrin.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

constexpr int K = 896, N = 1536, M = 64, NB = K / 32;

struct WB { float d; int8_t q[32]; };  // weight block (fp32 scale for simplicity)
struct XB { float d; int8_t q[32]; };  // activation block

static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_movehdup_ps(a));
  return _mm_cvtss_f32(a);
}

int main() {
  std::mt19937 rng(1);
  std::vector<WB> w(static_cast<size_t>(N) * NB);
  std::vector<XB> xq(static_cast<size_t>(M) * NB);
  std::vector<float> xf(static_cast<size_t>(M) * K), y1(static_cast<size_t>(M) * N), y2(y1.size());
  for (auto& b : w) { b.d = 0.01f; for (auto& q : b.q) q = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127); }
  for (int i = 0; i < M; ++i)
    for (int b = 0; b < NB; ++b) {
      XB& x = xq[static_cast<size_t>(i) * NB + b];
      x.d = 0.02f;
      for (int t = 0; t < 32; ++t) {
        x.q[t] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
        xf[static_cast<size_t>(i) * K + b * 32 + t] = x.d * x.q[t];
      }
    }

  auto time = [](auto fn) {
    double best = 1e30;
    for (int r = 0; r < 7; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      fn();
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
  };

  // fp32: per 4-row panel dequantize to fp32, then 4x3 FMA tile over M rows.
  std::vector<float> panel(4 * K);
  const double t_fp = time([&] {
    for (int j0 = 0; j0 < N; j0 += 4) {
      for (int r = 0; r < 4; ++r)
        for (int b = 0; b < NB; ++b) {
          const WB& wb = w[static_cast<size_t>(j0 + r) * NB + b];
          for (int t = 0; t < 32; t += 8) {
            const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(wb.q + t))));
            _mm256_storeu_ps(&panel[static_cast<size_t>(r) * K + b * 32 + t], _mm256_mul_ps(v, _mm256_set1_ps(wb.d)));
          }
        }
      for (int i = 0; i + 3 <= M - 1; i += 3) {
        __m256 acc[12];
        for (auto& a : acc) a = _mm256_setzero_ps();
        for (int kk = 0; kk < K; kk += 8) {
          const __m256 x0 = _mm256_loadu_ps(&xf[static_cast<size_t>(i) * K + kk]);
          const __m256 x1 = _mm256_loadu_ps(&xf[static_cast<size_t>(i + 1) * K + kk]);
          const __m256 x2 = _mm256_loadu_ps(&xf[static_cast<size_t>(i + 2) * K + kk]);
          for (int r = 0; r < 4; ++r) {
            const __m256 vw = _mm256_loadu_ps(&panel[static_cast<size_t>(r) * K + kk]);
            acc[r * 3] = _mm256_fmadd_ps(vw, x0, acc[r * 3]);
            acc[r * 3 + 1] = _mm256_fmadd_ps(vw, x1, acc[r * 3 + 1]);
            acc[r * 3 + 2] = _mm256_fmadd_ps(vw, x2, acc[r * 3 + 2]);
          }
        }
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 3; ++c) y1[static_cast<size_t>(i + c) * N + j0 + r] = hsum(acc[r * 3 + c]);
      }
    }
  });

  // int8 VNNI: 3 weight rows x 4 activation rows; per block: |w| once,
  // sign(x, w) per pair, dpbusd, then one cvt + fma with the block scale.
  const double t_i8 = time([&] {
    for (int j0 = 0; j0 + 3 <= N; j0 += 3) {
      for (int i = 0; i + 4 <= M; i += 4) {
        __m256 acc[12];
        for (auto& a : acc) a = _mm256_setzero_ps();
        for (int b = 0; b < NB; ++b) {
          __m256i aw[3], wq[3];
          float dw[3];
          for (int r = 0; r < 3; ++r) {
            const WB& wb = w[static_cast<size_t>(j0 + r) * NB + b];
            wq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb.q));
            aw[r] = _mm256_sign_epi8(wq[r], wq[r]);
            dw[r] = wb.d;
          }
          for (int c = 0; c < 4; ++c) {
            const XB& xb = xq[static_cast<size_t>(i + c) * NB + b];
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.q));
            for (int r = 0; r < 3; ++r) {
              const __m256i p = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw[r], _mm256_sign_epi8(x, wq[r]));
              acc[r * 4 + c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw[r] * xb.d), acc[r * 4 + c]);
            }
          }
        }
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 4; ++c) y2[static_cast<size_t>(i + c) * N + j0 + r] = hsum(acc[r * 4 + c]);
      }
    }
  });
  const double macs = static_cast<double>(M) * N * K;
  double err = 0, mag = 0;
  for (int i = 0; i < 63; ++i)
    for (int j = 0; j < 1533; ++j) {
      err = std::max(err, static_cast<double>(std::abs(y1[i * N + j] - y2[i * N + j])));
      mag = std::max(mag, static_cast<double>(std::abs(y1[i * N + j])));
    }
  std::printf("fp32 panel+4x3: %.2f ms (%.1f GFLOP/s)\nint8 VNNI 3x4: %.2f ms (%.1f GFLOP/s)\nspeedup %.2fx  max diff %.2e of %.2e\n",
              t_fp, 2 * macs / t_fp * 1e-6, t_i8, 2 * macs / t_i8 * 1e-6, t_fp / t_i8, err, mag);
  return 0;
}

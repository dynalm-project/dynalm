// Integer-activation CPU kernels of every tier built into this binary (generic,
// AVX2, NEON with SDOT): quantize_act / dot_q8_rows (DD-053) and
// quantize_act16 / dot_q16_rows (DD-076), against an exact reference.
//
// For random weight blocks of each type and 1..4 activation rows: the
// reference is sum(dequantized weight * reconstructed activation d * q) in
// double, so it checks the integer arithmetic, the scales and the minimum
// terms; only fp32 accumulation order may differ (relative 1e-5 of the sum of
// magnitudes). Quantization itself is checked too: |x - d * q| <= d / 2 and
// sum = sum(q). No model files needed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/quantization/dequant.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;

namespace {

int g_checks = 0;
#define CHECK(c)                                                                     \
  do {                                                                               \
    ++g_checks;                                                                      \
    if (!(c)) {                                                                      \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

std::mt19937 g_rng(1234);

// One row of k values of type t: random codes, small positive fp16 scales.
std::vector<uint8_t> random_row(DType t, int64_t k) {
  std::vector<uint8_t> row(static_cast<size_t>(dtype_row_bytes(t, k)));
  for (uint8_t& b : row) b = static_cast<uint8_t>(g_rng());
  std::uniform_real_distribution<float> sd(0.002f, 0.02f);
  const auto h = [&] { return fp32_to_fp16(sd(g_rng)); };
  auto put = [&](size_t off, uint16_t v) { std::memcpy(row.data() + off, &v, 2); };
  const int64_t be = dtype_block_elems(t), bb = dtype_block_bytes(t);
  for (int64_t b = 0; b < k / be; ++b) {
    const size_t o = static_cast<size_t>(b * bb);
    switch (t) {
      case DType::kQ8_0: case DType::kQ4_0: case DType::kQ5_0: put(o, h()); break;  // d first
      case DType::kQ4_K: put(o, h()); put(o + 2, h()); break;     // d, dmin
      case DType::kQ6_K: put(o + static_cast<size_t>(bb) - 2, h()); break;  // d last
      default: break;
    }
  }
  return row;
}

double rel_err(double got, double ref, double mag) { return std::fabs(got - ref) / std::max(mag, 1e-30); }

// A multi-row kernel (DotQ8RowsNFn / DotQ16RowsNFn) on weight rows [1, 3) of
// three random rows, with a bias, against the double-precision dot, for
// 1..kDotRowsMax activation rows. Returns the number of row counts checked.
template <typename Act, typename Fn>
int check_rows_n(Fn dot_n, DType t, int64_t kk, const std::vector<std::vector<Act>>& xs, const std::vector<const Act*>& ptr) {
  const int64_t rb = dtype_row_bytes(t, kk);
  std::vector<uint8_t> w3;
  std::vector<std::vector<float>> wf3;
  for (int i = 0; i < 3; ++i) {
    const std::vector<uint8_t> wi = random_row(t, kk);
    w3.insert(w3.end(), wi.begin(), wi.end());
    wf3.emplace_back(static_cast<size_t>(kk));
    CHECK(dequantize_row(t, wi.data(), wf3.back().data(), kk));
  }
  const float bias[3] = {0.25f, -0.5f, 2.0f};
  const int max_m = static_cast<int>(std::min<size_t>(kDotRowsNMax, ptr.size()));
  for (int m = 1; m <= max_m; ++m) {
    float ys[kDotRowsNMax][2];
    float* yp[kDotRowsNMax];
    for (int r = 0; r < m; ++r) yp[r] = ys[r];
    dot_n(w3.data(), rb, 1, 3, ptr.data(), m, kk, yp, bias);
    for (int r = 0; r < m; ++r) {
      for (int i = 1; i < 3; ++i) {
        double ref = bias[i], mag = std::fabs(bias[i]);
        for (int64_t e = 0; e < kk; ++e) {
          const double xv = static_cast<double>(xs[r][e / 32].d) * xs[r][e / 32].q[e % 32];
          ref += wf3[i][e] * xv;
          mag += std::fabs(wf3[i][e] * xv);
        }
        CHECK(rel_err(ys[r][i - 1], ref, mag) <= 1e-5);
      }
    }
  }
  return max_m;
}

void check_int8(const CpuKernels& k, const char* tier) {
  if (k.quantize_act == nullptr) return;
  int tested = 0, multi = 0;
  // Every type with an int8 row kernel on some tier (Q5_0: generic only; the
  // tier's own kernel, or the generic one it inherits, is what is checked).
  for (DType t : {DType::kQ8_0, DType::kQ4_0, DType::kQ5_0, DType::kQ4_K, DType::kQ6_K}) {
    if (k.dot_q8_rows_for(t) == nullptr) continue;
    for (int64_t kk : {256, 1024, 2816}) {
      const std::vector<uint8_t> w = random_row(t, kk);
      std::vector<float> wf(static_cast<size_t>(kk));
      CHECK(dequantize_row(t, w.data(), wf.data(), kk));
      std::vector<std::vector<ActBlockQ8>> xs(kDotRowsNMax, std::vector<ActBlockQ8>(static_cast<size_t>(kk / 32)));
      std::vector<const ActBlockQ8*> ptr;
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& x : xs) {
        std::vector<float> xf(static_cast<size_t>(kk));
        for (float& v : xf) v = nd(g_rng) * (g_rng() % 8 == 0 ? 6.0f : 1.0f);  // a few outliers
        k.quantize_act(xf.data(), x.data(), kk);
        for (int64_t b = 0; b < kk / 32; ++b) {  // quantization: |x - d q| <= d / 2, sum = sum(q)
          int32_t s = 0;
          for (int i = 0; i < 32; ++i) {
            s += x[b].q[i];
            // Half a step, plus the multiply-by-reciprocal rounding (|q| * 2^-23 steps).
            CHECK(std::fabs(xf[b * 32 + i] - x[b].d * x[b].q[i]) <=
                  x[b].d * (0.5f + 3e-7f * std::abs(static_cast<float>(x[b].q[i]))) + 1e-12f);
          }
          CHECK(s == x[b].sum);
        }
        ptr.push_back(x.data());
      }
      for (int m = 1; m <= kDotRowsMax; ++m) {
        float out[kDotRowsMax];
        k.dot_q8_rows_for(t)(w.data(), ptr.data(), m, kk, out);
        for (int r = 0; r < m; ++r) {
          double ref = 0, mag = 0;
          for (int64_t i = 0; i < kk; ++i) {
            const double xv = static_cast<double>(xs[r][i / 32].d) * xs[r][i / 32].q[i % 32];
            ref += wf[i] * xv;
            mag += std::fabs(wf[i] * xv);
          }
          if (rel_err(out[r], ref, mag) > 1e-5) {
            std::fprintf(stderr, "%s int8 %s k=%lld m=%d row %d: %.9g vs %.9g\n", tier, std::string(dtype_name(t)).c_str(),
                         static_cast<long long>(kk), m, r, out[r], ref);
          }
          CHECK(rel_err(out[r], ref, mag) <= 1e-5);
        }
      }
      if (const DotQ8RowsNFn dot_n = k.dot_q8_rows_n_for(t)) multi += check_rows_n(dot_n, t, kk, xs, ptr);
      ++tested;
    }
  }
  std::printf("  %s int8 (DD-053): %d type x size cases, 1-%d rows; %d multi-row checks\n", tier, tested, kDotRowsMax, multi);
}

void check_int16(const CpuKernels& k, const char* tier) {
  if (k.quantize_act16 == nullptr) return;
  int tested = 0, multi = 0;
  for (DType t : {DType::kQ8_0, DType::kQ4_K, DType::kQ6_K}) {
    if (k.dot_q16_rows_for(t) == nullptr) continue;
    for (int64_t kk : {256, 2816}) {
      const std::vector<uint8_t> w = random_row(t, kk);
      std::vector<float> wf(static_cast<size_t>(kk));
      CHECK(dequantize_row(t, w.data(), wf.data(), kk));
      std::vector<std::vector<ActBlockQ16>> xs(kDotRowsNMax, std::vector<ActBlockQ16>(static_cast<size_t>(kk / 32)));
      std::vector<const ActBlockQ16*> ptr;
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& x : xs) {
        std::vector<float> xf(static_cast<size_t>(kk));
        for (float& v : xf) v = nd(g_rng) * (g_rng() % 8 == 0 ? 50.0f : 1.0f);
        k.quantize_act16(xf.data(), x.data(), kk);
        for (int64_t b = 0; b < kk / 32; ++b) {
          int32_t s = 0;
          for (int i = 0; i < 32; ++i) {
            s += x[b].q[i];
            // Half a step, plus the multiply-by-reciprocal rounding (|q| * 2^-23 steps).
            CHECK(std::fabs(xf[b * 32 + i] - x[b].d * x[b].q[i]) <=
                  x[b].d * (0.5f + 3e-7f * std::abs(static_cast<float>(x[b].q[i]))) + 1e-12f);
          }
          CHECK(s == x[b].sum);
        }
        ptr.push_back(x.data());
      }
      for (int m = 1; m <= kDotRowsMax; ++m) {
        float out[kDotRowsMax];
        k.dot_q16_rows_for(t)(w.data(), ptr.data(), m, kk, out);
        for (int r = 0; r < m; ++r) {
          double ref = 0, mag = 0;
          for (int64_t i = 0; i < kk; ++i) {
            const double xv = static_cast<double>(xs[r][i / 32].d) * xs[r][i / 32].q[i % 32];
            ref += wf[i] * xv;
            mag += std::fabs(wf[i] * xv);
          }
          CHECK(rel_err(out[r], ref, mag) <= 1e-5);
        }
      }
      if (const DotQ16RowsNFn dot_n = k.dot_q16_rows_n_for(t)) multi += check_rows_n(dot_n, t, kk, xs, ptr);
      ++tested;
    }
  }
  std::printf("  %s int16 (DD-076): %d type x size cases, 1-%d rows; %d multi-row checks\n", tier, tested, kDotRowsMax, multi);
}

// Multi-row sub-scaled kernels (dot_q8_sx_rows_n) and the row kernels they
// restructure (dot_q8_sx_rows), both against the double-precision dot of the
// dequantized weights and the sub-scaled activations (x = u * m[block] * q),
// relative to the sum of |terms| like the other kernel checks: K-quant dots
// difference large, nearly cancelling terms, so the summation order alone
// moves the result by more than 1e-5 of its own size. Weight rows [1, 4) of
// four, with a bias, 1..kDotRowsNMax activation rows.
void check_sx_rows_n(const CpuKernels& k, const char* tier) {
  if (k.quantize_act_sx == nullptr) return;
  int tested = 0;
  for (DType t : {DType::kQ4_K, DType::kQ6_K}) {
    const DotQ8RowsNFn dot_n = k.dot_q8_sx_rows_n_for(t);
    const DotQ8RowsFn dot = k.dot_q8_sx_rows_for(t);
    if (dot_n == nullptr || dot == nullptr) continue;
    for (int64_t kk : {256, 2816}) {
      const int64_t rb = dtype_row_bytes(t, kk);
      std::vector<uint8_t> w4;
      std::vector<std::vector<float>> wf(4, std::vector<float>(static_cast<size_t>(kk)));
      for (int i = 0; i < 4; ++i) {
        const std::vector<uint8_t> wi = random_row(t, kk);
        CHECK(dequantize_row(t, wi.data(), wf[i].data(), kk));
        w4.insert(w4.end(), wi.begin(), wi.end());
      }
      std::vector<std::vector<ActBlockQ8>> xs(kDotRowsNMax, std::vector<ActBlockQ8>(static_cast<size_t>(kk / 32)));
      std::vector<const ActBlockQ8*> ptr;
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& x : xs) {
        std::vector<float> xf(static_cast<size_t>(kk));
        for (float& v : xf) v = nd(g_rng) * (g_rng() % 8 == 0 ? 6.0f : 1.0f);
        k.quantize_act_sx(xf.data(), x.data(), kk);
        ptr.push_back(x.data());
      }
      const float bias[4] = {0.5f, 0.25f, -0.5f, 2.0f};
      for (int m = 1; m <= kDotRowsNMax; ++m) {
        float ys[kDotRowsNMax][3];
        float* yp[kDotRowsNMax];
        for (int r = 0; r < m; ++r) yp[r] = ys[r];
        dot_n(w4.data(), rb, 1, 4, ptr.data(), m, kk, yp, bias);
        for (int i = 1; i < 4; ++i) {
          float row[kDotRowsNMax];
          // The row kernel takes up to kDotRowsMax rows per call.
          for (int r0 = 0; r0 < m; r0 += kDotRowsMax) {
            dot(w4.data() + i * rb, ptr.data() + r0, std::min(kDotRowsMax, m - r0), kk, row + r0);
          }
          for (int r = 0; r < m; ++r) {
            const auto* sup = reinterpret_cast<const ActSuperQ8*>(xs[r].data());
            double ref = bias[i], mag = std::fabs(bias[i]);
            for (int64_t e = 0; e < kk; ++e) {
              const ActSuperQ8& a = sup[e / 256];
              const double xv = static_cast<double>(a.u) * a.m[(e % 256) / 32] * a.q[e % 256];
              ref += wf[i][e] * xv;
              mag += std::fabs(wf[i][e] * xv);
            }
            const double got = ys[r][i - 1], rk = static_cast<double>(row[r]) + bias[i];
            if (rel_err(got, ref, mag) > 1e-5 || rel_err(rk, ref, mag) > 1e-5) {
              std::fprintf(stderr, "%s sx %s k=%lld m=%d row %d weight %d: multi %.9g, row %.9g, ref %.9g (mag %.4g)\n", tier,
                           std::string(dtype_name(t)).c_str(), static_cast<long long>(kk), m, r, i, got, rk, ref, mag);
            }
            CHECK(rel_err(got, ref, mag) <= 1e-5);
            CHECK(rel_err(rk, ref, mag) <= 1e-5);
          }
        }
      }
      ++tested;
    }
  }
  std::printf("  %s sub-scaled multi-row (DD-077): %d type x size cases, 1-%d rows\n", tier, tested, kDotRowsNMax);
}

}  // namespace

int main() {
  const CpuIsa best = select_isa(cpu_info().features);
  std::printf("test_int_kernels (best tier: %s)\n", std::string(isa_name(best)).c_str());
  const CpuKernels gen = make_cpu_kernels(CpuIsa::kGeneric);
  check_int8(gen, "generic");
  check_int16(gen, "generic");
  if (best != CpuIsa::kGeneric) {
    const CpuKernels k = make_cpu_kernels(best);
    const std::string name(isa_name(best));
    check_int8(k, name.c_str());
    check_int16(k, name.c_str());
    check_sx_rows_n(k, name.c_str());
    std::printf("  %s: int8 kernels %s\n", name.c_str(), k.int8_simd ? "SIMD" : "scalar reference only");
  }
  std::printf("all integer kernel tests passed (%d checks)\n", g_checks);
  return 0;
}

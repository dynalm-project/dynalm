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

void check_int8(const CpuKernels& k, const char* tier) {
  if (k.quantize_act == nullptr) return;
  int tested = 0;
  // Every type with an int8 row kernel on some tier (Q5_0: generic only; the
  // tier's own kernel, or the generic one it inherits, is what is checked).
  for (DType t : {DType::kQ8_0, DType::kQ4_0, DType::kQ5_0, DType::kQ4_K, DType::kQ6_K}) {
    if (k.dot_q8_rows_for(t) == nullptr) continue;
    for (int64_t kk : {256, 1024, 2816}) {
      const std::vector<uint8_t> w = random_row(t, kk);
      std::vector<float> wf(static_cast<size_t>(kk));
      CHECK(dequantize_row(t, w.data(), wf.data(), kk));
      std::vector<std::vector<ActBlockQ8>> xs(4, std::vector<ActBlockQ8>(static_cast<size_t>(kk / 32)));
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
      ++tested;
    }
  }
  std::printf("  %s int8 (DD-053): %d type x size cases, 1-%d rows\n", tier, tested, kDotRowsMax);
}

void check_int16(const CpuKernels& k, const char* tier) {
  if (k.quantize_act16 == nullptr) return;
  int tested = 0;
  for (DType t : {DType::kQ8_0, DType::kQ4_K, DType::kQ6_K}) {
    if (k.dot_q16_rows_for(t) == nullptr) continue;
    for (int64_t kk : {256, 2816}) {
      const std::vector<uint8_t> w = random_row(t, kk);
      std::vector<float> wf(static_cast<size_t>(kk));
      CHECK(dequantize_row(t, w.data(), wf.data(), kk));
      std::vector<std::vector<ActBlockQ16>> xs(4, std::vector<ActBlockQ16>(static_cast<size_t>(kk / 32)));
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
      ++tested;
    }
  }
  std::printf("  %s int16 (DD-076): %d type x size cases, 1-%d rows\n", tier, tested, kDotRowsMax);
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
    std::printf("  %s: int8 kernels %s\n", name.c_str(), k.int8_simd ? "SIMD" : "scalar reference only");
  }
  std::printf("all integer kernel tests passed (%d checks)\n", g_checks);
  return 0;
}

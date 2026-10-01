// CPU kernel tiers vs the scalar dequantization reference.

#include "backends/cpu/cpu_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "dtype/fp16.h"
#include "platform/cpu_info.h"
#include "quant/dequant.h"

namespace engine {
namespace {

// Byte offsets of fp16 scale fields per block (GGML layouts).
std::vector<int> fp16_fields(DType t) {
  switch (t) {
    case DType::kQ4_0: case DType::kQ5_0: case DType::kQ8_0: return {0};
    case DType::kQ4_1: case DType::kQ5_1: case DType::kQ8_1: return {0, 2};
    case DType::kQ2_K: return {80, 82};
    case DType::kQ3_K: return {108};
    case DType::kQ4_K: case DType::kQ5_K: return {0, 2};
    case DType::kQ6_K: return {208};
    default: return {};
  }
}

// Random weights of `n` elements of dtype t, with finite values.
std::vector<uint8_t> random_weights(DType t, int64_t n, std::mt19937& rng) {
  std::vector<uint8_t> w(static_cast<size_t>(dtype_row_bytes(t, n)));
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  if (t == DType::kF32) {
    for (size_t i = 0; i < w.size() / 4; ++i) reinterpret_cast<float*>(w.data())[i] = u(rng);
    return w;
  }
  if (t == DType::kF16 || t == DType::kBF16) {
    for (size_t i = 0; i < w.size() / 2; ++i) {
      reinterpret_cast<uint16_t*>(w.data())[i] = t == DType::kF16 ? fp32_to_fp16(u(rng)) : fp32_to_bf16(u(rng));
    }
    return w;
  }
  for (auto& b : w) b = static_cast<uint8_t>(rng());
  const auto bb = static_cast<size_t>(dtype_block_bytes(t));
  for (size_t blk = 0; blk < w.size() / bb; ++blk) {
    for (int off : fp16_fields(t)) {
      const uint16_t h = fp32_to_fp16(u(rng) * 0.05f);
      std::memcpy(w.data() + blk * bb + static_cast<size_t>(off), &h, 2);
    }
  }
  return w;
}

std::vector<CpuKernels> tiers() {
  std::vector<CpuKernels> out;
  CpuKernels g;
  register_generic_kernels(g);
  out.push_back(g);
  if (isa_supported(CpuIsa::kAvx2, cpu_info().features)) {
    CpuKernels a = g;
    if (register_avx2_kernels(a)) out.push_back(a);
  }
  return out;
}

class Kernels : public ::testing::TestWithParam<DType> {};

TEST_P(Kernels, VecDotMatchesDequantReference) {
  const DType t = GetParam();
  std::mt19937 rng(static_cast<unsigned>(t) * 7 + 1);
  const int64_t be = dtype_block_elems(t);
  for (int64_t n : {be * 4, be * 16 + (be == 1 ? 13 : 0), int64_t{1024}}) {
    if (n % be != 0) continue;
    const auto w = random_weights(t, n, rng);
    std::vector<float> x(static_cast<size_t>(n)), wf(static_cast<size_t>(n));
    std::normal_distribution<float> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    ASSERT_TRUE(dequantize_row(t, w.data(), wf.data(), n));
    double ref = 0, mag = 0;
    for (int64_t i = 0; i < n; ++i) {
      ref += static_cast<double>(wf[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)];
      mag += std::abs(static_cast<double>(wf[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)]);
    }
    for (const CpuKernels& k : tiers()) {
      ASSERT_NE(k.vec_dot_for(t), nullptr);
      const float got = k.vec_dot_for(t)(w.data(), x.data(), n);
      EXPECT_NEAR(got, ref, 1e-5 * mag + 1e-6) << dtype_name(t) << " n=" << n << " tier " << isa_name(k.isa);
      // Row dequantization entry must agree with the reference exactly (up to fp rounding).
      std::vector<float> dq(static_cast<size_t>(n));
      k.dequant_for(t)(w.data(), dq.data(), n);
      for (int64_t i = 0; i < n; ++i) {
        ASSERT_NEAR(dq[static_cast<size_t>(i)], wf[static_cast<size_t>(i)], 1e-6f * std::abs(wf[static_cast<size_t>(i)]) + 1e-7f);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllTypes, Kernels,
                         ::testing::Values(DType::kF32, DType::kF16, DType::kBF16, DType::kQ4_0, DType::kQ4_1,
                                           DType::kQ5_0, DType::kQ5_1, DType::kQ8_0, DType::kQ2_K, DType::kQ3_K,
                                           DType::kQ4_K, DType::kQ5_K, DType::kQ6_K),
                         [](const ::testing::TestParamInfo<DType>& p) { return std::string(dtype_name(p.param)); });

TEST(Kernels, FloatHelpersAllTiers) {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0, 1);
  for (int64_t n : {1, 7, 8, 31, 64, 100, 129}) {
    std::vector<float> a(static_cast<size_t>(n)), b(static_cast<size_t>(n));
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      a[static_cast<size_t>(i)] = nd(rng);
      b[static_cast<size_t>(i)] = nd(rng);
      h[static_cast<size_t>(i)] = fp32_to_fp16(nd(rng));
    }
    double ref = 0, ref16 = 0;
    for (int64_t i = 0; i < n; ++i) {
      ref += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
      ref16 += static_cast<double>(fp16_to_fp32(h[static_cast<size_t>(i)])) * b[static_cast<size_t>(i)];
    }
    for (const CpuKernels& k : tiers()) {
      EXPECT_NEAR(k.dot_f32(a.data(), b.data(), n), ref, 1e-4) << isa_name(k.isa) << " n=" << n;
      EXPECT_NEAR(k.dot_f16_f32(h.data(), b.data(), n), ref16, 1e-4) << isa_name(k.isa);
      std::vector<float> y1 = b, y2 = b;
      k.axpy_f32(0.5f, a.data(), y1.data(), n);
      k.axpy_f16(-2.0f, h.data(), y2.data(), n);
      for (int64_t i = 0; i < n; ++i) {
        const auto s = static_cast<size_t>(i);
        ASSERT_NEAR(y1[s], b[s] + 0.5f * a[s], 1e-6f);
        ASSERT_NEAR(y2[s], b[s] - 2.0f * fp16_to_fp32(h[s]), 1e-5f);
      }
    }
  }
}

TEST(Kernels, GemmPanelMatchesDots) {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  for (int64_t k : {8, 24, 64, 13}) {
    for (int nr = 1; nr <= 4; ++nr) {
      for (int64_t m : {1, 2, 3, 7}) {
        std::vector<float> w(static_cast<size_t>(nr * k)), x(static_cast<size_t>(m * k));
        for (auto& v : w) v = nd(rng);
        for (auto& v : x) v = nd(rng);
        const int64_t y_stride = 6;  // wider than nr: results land in place
        for (const CpuKernels& kt : tiers()) {
          std::vector<float> y(static_cast<size_t>(m * y_stride), -99.0f);
          kt.gemm_panel(w.data(), nr, x.data(), k, m, k, y.data(), y_stride, false);
          for (int64_t i = 0; i < m; ++i) {
            for (int r = 0; r < nr; ++r) {
              double ref = 0;
              for (int64_t q = 0; q < k; ++q) {
                ref += static_cast<double>(w[static_cast<size_t>(r * k + q)]) * x[static_cast<size_t>(i * k + q)];
              }
              ASSERT_NEAR(y[static_cast<size_t>(i * y_stride + r)], ref, 1e-4)
                  << isa_name(kt.isa) << " k=" << k << " nr=" << nr << " m=" << m;
            }
            for (int64_t r = nr; r < y_stride; ++r) ASSERT_EQ(y[static_cast<size_t>(i * y_stride + r)], -99.0f);
          }
          // Accumulate mode adds a second identical product: results double.
          std::vector<float> y2 = y;
          kt.gemm_panel(w.data(), nr, x.data(), k, m, k, y2.data(), y_stride, true);
          for (int64_t i = 0; i < m; ++i) {
            for (int r = 0; r < nr; ++r) {
              const auto idx = static_cast<size_t>(i * y_stride + r);
              ASSERT_NEAR(y2[idx], 2.0f * y[idx], 1e-4f + 1e-6f * std::abs(y[idx]));
            }
          }
        }
      }
    }
  }
}

TEST(Kernels, Avx2TierSelectedWhenSupported) {
  const CpuKernels k = make_cpu_kernels(select_best_isa(cpu_info().features));
  if (isa_supported(CpuIsa::kAvx2, cpu_info().features) && isa_compiled(CpuIsa::kAvx2)) {
    EXPECT_EQ(k.isa, CpuIsa::kAvx2);
  } else {
    EXPECT_EQ(k.isa, CpuIsa::kGeneric);
  }
}

}  // namespace
}  // namespace engine

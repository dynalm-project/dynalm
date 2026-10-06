// Performance program P3: int8-activation decode kernels (DD-053).
//
// The integer kernels must compute exactly sum(w_dequantized * x_reconstructed)
// where x_reconstructed = d * q is the int8 activation; every tier must agree
// with that reference up to fp32 summation order. The difference to the fp32
// path is then only the activation quantization itself, which is bounded here
// and measured end to end on real models in docs/benchmarks.md.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>

#include "dynacore/cpu/cpu_backend.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/tensor/fp16.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/quantization/dequant.h"
#include "dynacore/tensor/tensor.h"

namespace engine {
namespace {

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

// Random blocks with finite fp16 scales (same scheme as test_kernels).
std::vector<uint8_t> random_weights(DType t, int64_t n, std::mt19937& rng) {
  std::vector<uint8_t> w(static_cast<size_t>(dtype_row_bytes(t, n)));
  for (auto& b : w) b = static_cast<uint8_t>(rng());
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<int> fields;
  switch (t) {
    case DType::kQ4_0: case DType::kQ5_0: case DType::kQ8_0: fields = {0}; break;
    case DType::kQ4_K: fields = {0, 2}; break;
    case DType::kQ6_K: fields = {208}; break;
    default: break;
  }
  const auto bb = static_cast<size_t>(dtype_block_bytes(t));
  for (size_t blk = 0; blk < w.size() / bb; ++blk) {
    for (int off : fields) {
      const uint16_t h = fp32_to_fp16(u(rng) * 0.05f);
      std::memcpy(w.data() + blk * bb + static_cast<size_t>(off), &h, 2);
    }
  }
  return w;
}

TEST(QuantizeAct, BoundedErrorExactSumsAndIdenticalAcrossTiers) {
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 2);
  std::vector<float> x(1024);
  for (auto& v : x) v = nd(rng);
  x[5] = 0;
  std::fill(x.begin() + 64, x.begin() + 96, 0.0f);  // an all-zero block
  std::vector<std::vector<ActBlockQ8>> per_tier;
  for (const CpuKernels& k : tiers()) {
    ASSERT_NE(k.quantize_act, nullptr);
    std::vector<ActBlockQ8> q(x.size() / 32);
    k.quantize_act(x.data(), q.data(), static_cast<int64_t>(x.size()));
    for (size_t b = 0; b < q.size(); ++b) {
      int32_t sum = 0;
      for (int i = 0; i < 32; ++i) {
        sum += q[b].q[i];
        EXPECT_LE(std::abs(x[b * 32 + i] - q[b].d * q[b].q[i]), 0.5f * q[b].d + 1e-6f) << isa_name(k.isa);
      }
      EXPECT_EQ(sum, q[b].sum);
    }
    EXPECT_EQ(q[2].d, 0.0f);  // zero block: zero scale, zero codes
    per_tier.push_back(std::move(q));
  }
  for (size_t t = 1; t < per_tier.size(); ++t) {
    ASSERT_EQ(std::memcmp(per_tier[0].data(), per_tier[t].data(), per_tier[0].size() * sizeof(ActBlockQ8)), 0)
        << "tier " << t << " quantizes differently from the generic reference";
  }
}

class DotRows : public ::testing::TestWithParam<DType> {};

TEST_P(DotRows, MatchesDequantizedReferenceForOneToFourRows) {
  const DType t = GetParam();
  std::mt19937 rng(static_cast<unsigned>(t) + 3);
  std::normal_distribution<float> nd(0, 1);
  for (int64_t n : {int64_t{256}, int64_t{1024}, int64_t{2304}}) {
    const auto w = random_weights(t, n, rng);
    std::vector<float> wf(static_cast<size_t>(n));
    ASSERT_TRUE(dequantize_row(t, w.data(), wf.data(), n));
    for (const CpuKernels& k : tiers()) {
      ASSERT_NE(k.dot_q8_rows_for(t), nullptr) << dtype_name(t);
      for (int m = 1; m <= kDotRowsMax; ++m) {
        std::vector<std::vector<ActBlockQ8>> act(static_cast<size_t>(m));
        std::vector<const ActBlockQ8*> ptr;
        std::vector<double> ref(static_cast<size_t>(m), 0.0), mag(static_cast<size_t>(m), 0.0);
        for (int r = 0; r < m; ++r) {
          std::vector<float> x(static_cast<size_t>(n));
          for (auto& v : x) v = nd(rng);
          act[r].resize(static_cast<size_t>(n / 32));
          k.quantize_act(x.data(), act[r].data(), n);
          ptr.push_back(act[r].data());
          for (int64_t i = 0; i < n; ++i) {
            const ActBlockQ8& a = act[r][static_cast<size_t>(i / 32)];
            const double xr = static_cast<double>(a.d) * a.q[i % 32];
            ref[r] += wf[static_cast<size_t>(i)] * xr;
            mag[r] += std::abs(wf[static_cast<size_t>(i)] * xr);
          }
        }
        std::vector<float> out(static_cast<size_t>(m));
        k.dot_q8_rows_for(t)(w.data(), ptr.data(), m, n, out.data());
        for (int r = 0; r < m; ++r) {
          EXPECT_NEAR(out[r], ref[r], 2e-5 * mag[r] + 1e-6)
              << dtype_name(t) << " n=" << n << " m=" << m << " row " << r << " tier " << isa_name(k.isa);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Types, DotRows,
                         ::testing::Values(DType::kQ8_0, DType::kQ4_0, DType::kQ5_0, DType::kQ4_K, DType::kQ6_K),
                         [](const ::testing::TestParamInfo<DType>& p) { return std::string(dtype_name(p.param)); });

// The backend path: quantize M rows once, all weight rows in parallel,
// passes of four rows, bias. Checked against fp32 matmul within the
// activation-quantization error.
TEST(Int8Matmul, BackendPathMatchesFp32WithinQuantizationError) {
  if (!isa_supported(CpuIsa::kAvx2, cpu_info().features) || !isa_compiled(CpuIsa::kAvx2)) {
    GTEST_SKIP() << "int8 decode is accelerated only on the AVX2 tier";
  }
  ThreadPool pool(4);
  CpuBackend be(pool, CpuIsa::kAvx2);
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0, 1);
  const int64_t n = 67, k = 512;  // odd n: uneven parallel chunks
  for (DType t : {DType::kQ4_K, DType::kQ6_K, DType::kQ8_0}) {
    const auto wbytes = random_weights(t, n * k, rng);
    auto w = Tensor::empty(t, {n, k});
    std::memcpy(w->data(), wbytes.data(), wbytes.size());
    auto bias = Tensor::empty(DType::kF32, {n});
    for (int64_t j = 0; j < n; ++j) bias->data_as<float>()[j] = nd(rng);
    for (int64_t m : {1, 3, 5, 8}) {
      auto x = Tensor::empty(DType::kF32, {m, k});
      for (int64_t i = 0; i < m * k; ++i) x->data_as<float>()[i] = nd(rng);
      auto y_fp = Tensor::empty(DType::kF32, {m, n});
      auto y_i8 = Tensor::empty(DType::kF32, {m, n});
      KernelPlan kp = KernelPlan::defaults();
      kp.int8_decode_max_rows = 0;
      be.set_kernel_plan(kp);
      be.matmul(*x, *w, &bias->view(), *y_fp);
      kp.int8_decode_max_rows = 8;
      be.set_kernel_plan(kp);
      be.matmul(*x, *w, &bias->view(), *y_i8);
      double err = 0, scale = 0;
      for (int64_t i = 0; i < m * n; ++i) {
        err = std::max(err, std::abs(double{y_fp->data_as<float>()[i]} - y_i8->data_as<float>()[i]));
        scale = std::max(scale, std::abs(double{y_fp->data_as<float>()[i]}));
      }
      // int8 activations carry ~0.4% relative error per element; summed over
      // k random products the output error stays a small fraction of the range.
      EXPECT_LT(err, 0.02 * scale) << dtype_name(t) << " m=" << m;
    }
  }
}

// P11 (DD-060): grouped matmul_many runs jobs of different sizes (int8 rows,
// fused single rows, 4-row panels) in one region, with strided inputs and
// outputs like MoE slices. Each job must match its own matmul.
TEST(GroupedMatmul, EveryPathMatchesIndividualMatmuls) {
  ThreadPool pool(4);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  std::mt19937 rng(21);
  std::normal_distribution<float> nd(0, 1);
  const int64_t k = 256, n = 40, wide = 2 * k;  // x rows are slices of a wider buffer
  auto wbytes = random_weights(DType::kQ4_K, n * k, rng);
  auto w = Tensor::empty(DType::kQ4_K, {n, k});
  std::memcpy(w->data(), wbytes.data(), wbytes.size());
  for (int int8 : {0, 4}) {
    KernelPlan kp = KernelPlan::defaults();
    kp.int8_decode_max_rows = int8;
    be.set_kernel_plan(kp);
    std::vector<int64_t> rows = {1, 3, 5, 9};
    std::vector<Tensor> xs, ys, refs;
    std::vector<Backend::MatmulJob> jobs;
    for (int64_t m : rows) {
      auto xbuf = Tensor::empty(DType::kF32, {m, wide});
      for (int64_t i = 0; i < m * wide; ++i) xbuf->data_as<float>()[i] = nd(rng);
      auto ybuf = Tensor::zeros(DType::kF32, {m, 2 * n});
      auto ref = Tensor::zeros(DType::kF32, {m, n});
      xs.push_back(std::move(*xbuf));
      ys.push_back(std::move(*ybuf));
      refs.push_back(std::move(*ref));
    }
    for (size_t t = 0; t < rows.size(); ++t) {
      const TensorView x = *xs[t].view().slice(1, 0, k);  // strided rows
      const TensorView y = *ys[t].view().slice(1, n, n);  // strided output columns
      jobs.push_back({x, *w, y});
      be.matmul(x, *w, nullptr, refs[t]);
    }
    be.matmul_many(jobs);
    for (size_t t = 0; t < rows.size(); ++t) {
      for (int64_t i = 0; i < rows[t]; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          const float got = ys[t].data_as<float>()[i * 2 * n + n + j];
          const float want = refs[t].data_as<float>()[i * n + j];
          ASSERT_NEAR(got, want, 1e-4f + 1e-4f * std::abs(want)) << "int8=" << int8 << " job rows " << rows[t];
        }
        EXPECT_EQ(ys[t].data_as<float>()[i * 2 * n], 0.0f);  // columns outside the view untouched
      }
    }
  }
}

}  // namespace
}  // namespace engine

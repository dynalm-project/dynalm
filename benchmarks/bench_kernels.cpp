// Phase 17 benchmark: fused dequantize-dot throughput per weight type,
// generic vs AVX2 tier (one 4096-element row, the decode matmul inner loop).

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "backends/cpu/cpu_kernels.h"
#include "bench_harness.h"
#include "dtype/fp16.h"
#include "platform/cpu_info.h"

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  CpuKernels generic;
  register_generic_kernels(generic);
  CpuKernels avx2 = generic;
  const bool have_avx2 = isa_supported(CpuIsa::kAvx2, cpu_info().features) && register_avx2_kernels(avx2);

  constexpr int64_t kN = 4096;
  std::mt19937 rng(1);
  std::vector<float> x(kN);
  for (auto& v : x) v = static_cast<float>(rng() % 1000) / 1000.0f - 0.5f;
  std::printf("%-6s %14s %14s %9s %12s\n", "type", "generic ns", "avx2 ns", "speedup", "avx2 GB/s");
  for (DType t : {DType::kF32, DType::kF16, DType::kBF16, DType::kQ8_0, DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ4_K,
                  DType::kQ5_K, DType::kQ6_K}) {
    const int64_t bytes = dtype_row_bytes(t, kN);
    std::vector<uint8_t> w(static_cast<size_t>(bytes));
    // Quantized: small random bytes keep fp16 scale fields finite. Float
    // types need real values: tiny byte patterns would be denormals, which
    // are pathologically slow and would misrepresent the kernel.
    for (auto& b : w) b = static_cast<uint8_t>(rng() & 0x3F);
    for (int64_t i = 0; i < kN && !dtype_is_quantized(t); ++i) {
      const float v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      if (t == DType::kF32) reinterpret_cast<float*>(w.data())[i] = v;
      if (t == DType::kF16) reinterpret_cast<uint16_t*>(w.data())[i] = fp32_to_fp16(v);
      if (t == DType::kBF16) reinterpret_cast<uint16_t*>(w.data())[i] = fp32_to_bf16(v);
    }
    float sink = 0;
    const bench::Options opt{.warmup_samples = 5, .samples = 200, .batch = 20};
    const auto g = bench::run([&] { sink += generic.vec_dot_for(t)(w.data(), x.data(), kN); }, opt);
    double a_p50 = 0;
    if (have_avx2) {
      const auto a = bench::run([&] { sink += avx2.vec_dot_for(t)(w.data(), x.data(), kN); }, opt);
      a_p50 = a.p50;
    }
    bench::do_not_optimize(sink);
    std::printf("%-6s %14.0f %14.0f %8.1fx %12.2f\n", std::string(dtype_name(t)).c_str(), g.p50, a_p50,
                a_p50 > 0 ? g.p50 / a_p50 : 0.0, a_p50 > 0 ? bytes / a_p50 : 0.0);
  }

  // Row dequantization (the batched-matmul expand path), one 4096-element row.
  std::printf("\ndequant row (%lld elements):\n%-6s %14s %14s %9s\n", static_cast<long long>(kN), "type",
              "generic ns", "avx2 ns", "speedup");
  for (DType t : {DType::kF16, DType::kBF16, DType::kQ8_0, DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ4_K, DType::kQ6_K}) {
    std::vector<uint8_t> w(static_cast<size_t>(dtype_row_bytes(t, kN)));
    for (auto& b : w) b = static_cast<uint8_t>(rng() & 0x3F);
    std::vector<float> out(kN);
    const bench::Options opt{.warmup_samples = 5, .samples = 200, .batch = 20};
    const auto g = bench::run([&] { generic.dequant_for(t)(w.data(), out.data(), kN); }, opt);
    const auto a = bench::run([&] { avx2.dequant_for(t)(w.data(), out.data(), kN); }, opt);
    bench::do_not_optimize(out[0]);
    std::printf("%-6s %14.0f %14.0f %8.1fx\n", std::string(dtype_name(t)).c_str(), g.p50, a.p50, g.p50 / a.p50);
  }

  // GEMM panel microkernel, single thread: 4 weight rows x m activation rows.
  std::printf("\ngemm_panel (single thread, nr=4):\n");
  for (int64_t m : {2, 64, 256}) {
    for (int64_t k : {896, 4864}) {
      std::vector<float> w(static_cast<size_t>(4 * k)), xs(static_cast<size_t>(m * k)), y(static_cast<size_t>(m * 4));
      for (auto& v : w) v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      for (auto& v : xs) v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      const auto& kk = have_avx2 ? avx2 : generic;
      const auto s = bench::run([&] { kk.gemm_panel(w.data(), 4, xs.data(), k, m, k, y.data(), 4, false); },
                                {.warmup_samples = 3, .samples = 50, .batch = 5});
      const double flops = 2.0 * 4 * static_cast<double>(m) * static_cast<double>(k);
      std::printf("  m=%-4lld k=%-5lld p50 %10.0f ns  %6.1f GFLOPS\n", static_cast<long long>(m),
                  static_cast<long long>(k), s.p50, flops / s.p50);
    }
  }
  return 0;
}

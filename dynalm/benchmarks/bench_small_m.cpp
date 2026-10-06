// Small-M prefill decomposition: per 4-row panel, how much time is Q4_K
// dequantization vs the 4x3 FMA tile, single thread, real weights.
//   bench_small_m <model.gguf>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "dynacore/cpu/cpu_kernels.h"
#include "loader/model_loader.h"
#include "dynacore/tensor/dtype.h"
#include "common/core.h"

using namespace dynalm;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_small_m <model.gguf>\n");
    return 2;
  }
  auto lm = load_model(argv[1]);
  if (!lm.ok()) {
    std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
    return 1;
  }
  const Tensor* w = (*lm)->weights.find(TensorRole::kFfnGate, 0);
  if (!w) return 1;
  const DType wt = w->dtype();
  const int64_t n = w->shape()[0], k = w->shape()[1];
  CpuKernels kern;
  register_generic_kernels(kern);
  register_avx2_kernels(kern);
  const DequantFn dq = kern.dequant_for(wt);
  const auto* base = static_cast<const std::byte*>(w->data());
  const int64_t row_bytes = dtype_row_bytes(wt, k);
  // One thread covers 1/10 of the panels, as it would with 10 workers.
  const int64_t panels = n / 4 / 10;

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  std::printf("%s %s [%lld, %lld], %lld panels per thread\n", argv[1], dtype_name(wt).data(),
              static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(panels));
  std::printf("   M | dequant ms | tile ms | total ms | tile ms/row\n");
  for (int64_t m : {1, 3, 6, 12, 24, 48}) {
    std::vector<float> x(static_cast<size_t>(m * k)), y(static_cast<size_t>(m * n));
    for (auto& v : x) v = u(rng);
    std::vector<float> panel(static_cast<size_t>(4 * k));
    auto best = [](auto fn) {
      double b = 1e30;
      for (int r = 0; r < 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        b = std::min(b, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      return b;
    };
    const double t_dq = best([&] {
      for (int64_t p = 0; p < panels; ++p)
        for (int r = 0; r < 4; ++r) dq(base + (p * 4 + r) * row_bytes, panel.data() + r * k, k);
    });
    const double t_tile = best([&] {
      for (int64_t p = 0; p < panels; ++p) kern.gemm_panel(panel.data(), 4, x.data(), k, m, k, y.data() + p * 4, n, false);
    });
    const double t_all = best([&] {
      for (int64_t p = 0; p < panels; ++p) {
        for (int r = 0; r < 4; ++r) dq(base + (p * 4 + r) * row_bytes, panel.data() + r * k, k);
        kern.gemm_panel(panel.data(), 4, x.data(), k, m, k, y.data() + p * 4, n, false);
      }
    });
    std::printf("%4lld | %10.3f | %7.3f | %8.3f | %.4f\n", static_cast<long long>(m), t_dq, t_tile, t_all,
                t_tile / static_cast<double>(m));
  }
  return 0;
}

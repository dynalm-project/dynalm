// Phase 18: CpuBackend::matmul throughput vs thread count, prefill-shaped
// (m=256 activation rows) and decode-shaped (m=1), Q8_0 and f32 weights.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "bench_harness.h"
#include "dtype/fp16.h"
#include "platform/cpu_info.h"
#include "quant/quant_formats.h"

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  // Shapes: (k=896, n=4864) like Qwen2.5-0.5B gate/up; (k=4864, n=896) like down.
  const int64_t kK = std::getenv("BENCH_DOWN") ? 4864 : 896, kN = std::getenv("BENCH_DOWN") ? 896 : 4864;
  std::mt19937 rng(5);
  // Q8_0 weights with sane scales.
  auto wq = Tensor::empty(DType::kQ8_0, {kN, kK});
  auto* blocks = static_cast<quant::BlockQ8_0*>(wq->data());
  for (int64_t b = 0; b < kN * kK / 32; ++b) {
    blocks[b].d = fp32_to_fp16(0.01f);
    for (auto& q : blocks[b].qs) q = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
  }
  auto wf = Tensor::empty(DType::kF32, {kN, kK});
  for (int64_t i = 0; i < kN * kK; ++i) wf->data_as<float>()[i] = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;

  std::printf("%8s %6s %5s %12s %10s\n", "weights", "m", "thr", "p50 ms", "GFLOPS");
  for (const Tensor* w : {&*wq, &*wf}) {
    for (int64_t m : {256, 1}) {
      auto x = Tensor::empty(DType::kF32, {m, kK});
      auto y = Tensor::empty(DType::kF32, {m, kN});
      for (int64_t i = 0; i < m * kK; ++i) x->data_as<float>()[i] = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      for (int threads : {1, 2, 4, 10}) {
        ThreadPool pool(threads);
        CpuBackend be(pool, select_best_isa(cpu_info().features));
        const auto s = bench::run([&] { be.matmul(*x, *w, nullptr, *y); },
                                  {.warmup_samples = 2, .samples = m == 1 ? 50 : 10, .batch = 1});
        const double flops = 2.0 * static_cast<double>(m) * kN * kK;
        std::printf("%8s %6lld %5d %12.2f %10.1f\n", std::string(dtype_name(w->dtype())).c_str(),
                    static_cast<long long>(m), threads, s.p50 * 1e-6, flops / s.p50);
      }
    }
  }
  return 0;
}

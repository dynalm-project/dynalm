// Elementwise device ops at decode shapes (1 row) and prefill shapes:
// act_mul, rms_norm, add. Small ops are latency-bound; this separates kernel
// time from parallel-region dispatch.
//
//   bench_elementwise [threads]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench_harness.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

using namespace dynacore;

namespace {

TensorView view(std::vector<float>& v, int64_t rows, int64_t cols) {
  auto layout = TensorLayout::contiguous(DType::kF32, TensorShape{rows, cols});
  return TensorView(v.data(), *layout);
}

template <typename Fn>
double p50_us(Fn&& fn, int iters = 200) {
  std::vector<double> t;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : cpu_info().physical_cores;
  ThreadPool pool(threads);
  CpuDevice dev(pool, select_best_isa(cpu_info().features));
  std::printf("%-28s %10s %10s\n", "op [rows x cols]", "p50 us", "GB/s");
  for (int64_t m : {1, 4, 64}) {
    for (int64_t d : {1536, 8960}) {
      std::vector<float> g(static_cast<size_t>(m * d), 0.3f), u(g), o(g), w(static_cast<size_t>(d), 1.0f);
      const TensorView gv = view(g, m, d), uv = view(u, m, d), ov = view(o, m, d);
      auto wl = TensorLayout::contiguous(DType::kF32, TensorShape{d});
      const TensorView wv(w.data(), *wl);
      const double bytes = 3.0 * static_cast<double>(m * d) * 4;
      double us = p50_us([&] { dev.act_mul(Activation::kSilu, gv, uv, ov); });
      std::printf("act_mul silu [%lld x %lld] %10.1f %10.1f\n", static_cast<long long>(m), static_cast<long long>(d), us,
                  bytes / us / 1e3);
      us = p50_us([&] { dev.rms_norm(gv, wv, 1e-6f, ov); });
      std::printf("rms_norm     [%lld x %lld] %10.1f %10.1f\n", static_cast<long long>(m), static_cast<long long>(d), us,
                  2.0 * static_cast<double>(m * d) * 4 / us / 1e3);
      us = p50_us([&] { dev.add(gv, uv, ov); });
      std::printf("add          [%lld x %lld] %10.1f %10.1f\n", static_cast<long long>(m), static_cast<long long>(d), us,
                  bytes / us / 1e3);
    }
  }
  return 0;
}

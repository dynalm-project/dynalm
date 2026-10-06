// Phase 1 microbenchmarks.
//   - fp16/bf16 scalar conversion: generic-backend cost per element
//   - host allocation: why hot paths must use pools instead of host_alloc
//   - view creation: slicing/selecting must be cheap (no allocation)

#include <cstdio>
#include <vector>

#include "bench_harness.h"
#include "dynacore/tensor/fp16.h"
#include "dynacore/memory/host_memory.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/tensor/tensor.h"

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());

  constexpr int kN = 4096;
  std::vector<uint16_t> h(kN);
  std::vector<float> f(kN);
  for (int i = 0; i < kN; ++i) {
    f[i] = (i - kN / 2) * 0.001f;
    h[i] = fp32_to_fp16(f[i]);
  }

  bench::Options conv{.warmup_samples = 5, .samples = 200, .batch = 10};
  bench::print_header();

  auto s = bench::run([&] {
    for (int i = 0; i < kN; ++i) f[i] = fp16_to_fp32(h[i]);
    bench::do_not_optimize(f[0]);
  }, conv);
  bench::print_row("fp16->fp32 x4096", s);
  std::printf("  = %.2f ns/elem (p50)\n", s.p50 / kN);

  s = bench::run([&] {
    for (int i = 0; i < kN; ++i) h[i] = fp32_to_fp16(f[i]);
    bench::do_not_optimize(h[0]);
  }, conv);
  bench::print_row("fp32->fp16 x4096", s);
  std::printf("  = %.2f ns/elem (p50)\n", s.p50 / kN);

  s = bench::run([&] {
    for (int i = 0; i < kN; ++i) f[i] = bf16_to_fp32(h[i]);
    bench::do_not_optimize(f[0]);
  }, conv);
  bench::print_row("bf16->fp32 x4096", s);

  bench::print_row("host_alloc+free 4 KiB", bench::run([] {
                     void* p = host_alloc(4096);
                     bench::do_not_optimize(p);
                     host_free(p, 4096);
                   }));
  bench::print_row("host_alloc+free 1 MiB", bench::run([] {
                     void* p = host_alloc(1 << 20);
                     bench::do_not_optimize(p);
                     host_free(p, 1 << 20);
                   }, {.warmup_samples = 5, .samples = 200, .batch = 50}));

  auto t = Tensor::zeros(DType::kF32, {32, 128, 64});
  if (!t.ok()) return 1;
  const TensorView v = *t;
  int64_t i = 0;
  bench::print_row("TensorView::select+slice", bench::run([&] {
                     auto head = v.select(0, i++ & 31);
                     auto rows = head->slice(0, 0, 64);
                     bench::do_not_optimize(rows);
                   }));
  return 0;
}

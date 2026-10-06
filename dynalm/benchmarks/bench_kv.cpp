// Phase 10 benchmark: paged KV pool operations.
//   - allocate+release latency, uncontended and with 4 contending threads
//   - clone of a 256-block table (fork cost)
//   - copy-on-write of one block (SmolLM2-135M geometry, f16)

#include <atomic>
#include <cstdio>
#include <thread>

#include "dynacore/cpu/cpu_backend.h"
#include "bench_harness.h"
#include "kv_cache/kv_cache.h"
#include "dynacore/hardware/cpu_info.h"

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  ThreadPool tp(1);
  CpuBackend be(tp, CpuIsa::kGeneric);
  // SmolLM2-135M: 30 layers, 3 kv heads, head_dim 64; 16-token blocks, f16.
  auto pool = KvBlockPool::create(KvGeometry{30, 3, 64, 64, 16, 1024, DType::kF16}, be);
  if (!pool.ok()) return 1;
  KvBlockPool& p = **pool;
  bench::print_header();

  bench::print_row("allocate+release (1 thread)", bench::run([&] {
                     auto b = p.allocate();
                     p.release(*b);
                   }));

  // 4 threads hammering the pool; report per-op latency seen by thread 0.
  std::atomic<bool> stop{false};
  std::vector<std::thread> noise;
  for (int t = 0; t < 3; ++t) {
    noise.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        auto b = p.allocate();
        if (b.ok()) p.release(*b);
      }
    });
  }
  bench::print_row("allocate+release (4 threads)", bench::run([&] {
                     auto b = p.allocate();
                     p.release(*b);
                   }));
  stop = true;
  for (auto& t : noise) t.join();

  KvBlockTable big(p);
  (void)big.reserve(256 * 16);
  bench::print_row("clone 256-block table + release", bench::run([&] {
                     KvBlockTable c = big.clone();
                     bench::do_not_optimize(c);
                   }, {.warmup_samples = 5, .samples = 200, .batch = 20}));

  KvBlockTable a(p);
  (void)a.reserve(16);
  bench::print_row("copy-on-write 1 block (30 layers)", bench::run([&] {
                     KvBlockTable c = a.clone();
                     (void)c.make_writable(0, 1);
                   }, {.warmup_samples = 5, .samples = 200, .batch = 20}));
  const double block_kib = static_cast<double>(p.geometry().bytes_per_layer()) / p.num_blocks() *
                           p.geometry().num_layers / 1024.0;
  std::printf("  (one block across all layers = %.0f KiB)\n", block_kib);
  return 0;
}

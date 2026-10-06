// Performance program P6: cost of one parallel region (fork + join) of the
// kernel thread pool, the fixed overhead every op of a decode step pays.
//
//   bench_thread_pool [threads]
//
// Measures back-to-back regions (workers still spinning, as inside a forward
// pass) and regions after an idle gap (workers asleep, as at the start of a
// step), for an empty body and for a tiny body.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "bench_harness.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/thread_qos.h"
#include "dynacore/execution/thread_pool.h"

int main(int argc, char** argv) {
  using namespace dynacore;
  const int threads = argc > 1 ? std::atoi(argv[1]) : cpu_info().physical_cores;
  request_full_speed_process();
  request_full_speed_thread();
  ThreadPool pool(threads);
  std::atomic<int64_t> sink{0};
  std::printf("cpu: %s, %d threads\n\n", cpu_info().brand.c_str(), threads);
  std::printf("%-38s %10s %10s %10s\n", "case", "p50 us", "p90 us", "p99 us");
  auto report = [](const char* name, const bench::Stats& s) {
    std::printf("%-38s %10.2f %10.2f %10.2f\n", name, s.p50 * 1e-3, s.p90 * 1e-3, s.p99 * 1e-3);
  };
  // Back to back: the forward-pass case (first pass also warms the pool up).
  report("empty region, back to back (1st pass)", bench::run([&] { pool.parallel_for(threads * 4, 1, [](size_t, size_t) {}); },
                                                   {.warmup_samples = 100, .samples = 2000, .batch = 1}));
  report("empty region, back to back", bench::run([&] { pool.parallel_for(threads * 4, 1, [](size_t, size_t) {}); },
                                                   {.warmup_samples = 100, .samples = 2000, .batch = 1}));
  report("tiny body (64 adds per chunk)", bench::run([&] {
           pool.parallel_for(threads * 4, 1, [&](size_t b, size_t e) {
             int64_t s = 0;
             for (size_t i = b; i < e; ++i)
               for (int k = 0; k < 64; ++k) s += static_cast<int64_t>(i) * k;
             sink.fetch_add(s, std::memory_order_relaxed);
           });
         },
                                                     {.warmup_samples = 100, .samples = 2000, .batch = 1}));
  // After an idle gap long enough for workers to fall asleep.
  report("empty region after 2 ms idle (wake-up)", bench::run([&] {
           std::this_thread::sleep_for(std::chrono::milliseconds(2));
           const auto t0 = std::chrono::steady_clock::now();
           pool.parallel_for(threads * 4, 1, [](size_t, size_t) {});
           const auto t1 = std::chrono::steady_clock::now();
           sink.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                          std::memory_order_relaxed);
         },
                                                              {.warmup_samples = 5, .samples = 200, .batch = 1}));
  std::printf("\n(the wake-up row includes the 2 ms sleep; subtract it)\n");
  return sink.load() == 42 ? 1 : 0;
}

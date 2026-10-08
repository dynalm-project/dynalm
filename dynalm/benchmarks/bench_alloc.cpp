// Forensic analysis: heap allocations in the decode hot path.
//
//   bench_alloc <model.gguf> [threads]
//
// Replaces the global operator new/delete with counting versions, runs the
// real Scheduler (sampling with temperature/top-k/top-p, streaming callbacks)
// for N = 1, 4, 8 concurrent requests, and reports allocations per step
// separately for decode-only steps, once every request is past its prompt.
// The target is zero allocations per decode step.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "dynacore/cpu/cpu_device.h"
#include "loader/model_loader.h"
#include "runtime/sequence.h"
#include "scheduler/scheduler.h"
#include "common/core.h"

namespace {
std::atomic<uint64_t> g_allocs{0}, g_bytes{0}, g_frees{0};
}  // namespace

void* operator new(std::size_t n) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  g_bytes.fetch_add(n, std::memory_order_relaxed);
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  g_bytes.fetch_add(n, std::memory_order_relaxed);
#ifdef _WIN32
  if (void* p = _aligned_malloc(n == 0 ? 1 : n, static_cast<size_t>(a))) return p;
#else
  if (void* p = std::aligned_alloc(static_cast<size_t>(a), (n + static_cast<size_t>(a) - 1) / static_cast<size_t>(a) * static_cast<size_t>(a))) return p;
#endif
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p) noexcept {
  if (p) g_frees.fetch_add(1, std::memory_order_relaxed);
  std::free(p);
}
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { operator delete(p); }
void operator delete(void* p, std::align_val_t) noexcept {
  if (p) g_frees.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
  _aligned_free(p);
#else
  std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t a) noexcept { operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { operator delete(p, a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { operator delete(p, a); }

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_alloc <model.gguf> [threads]\n");
    return 2;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  constexpr int kPrompt = 32, kGen = 48;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, 16 * (kPrompt + kGen + 16)), be);
  auto tf = Transformer::create(c, (*m)->weights, be, 256);
  if (!kv.ok() || !tf.ok()) return 1;

  std::printf("%5s | %14s %14s %14s | %10s\n", "reqs", "allocs/step", "bytes/step", "frees/step", "steps");
  for (int n : {1, 4, 8}) {
    Scheduler sched(**tf, **kv, *(*m)->tokenizer, SchedulerConfig{});
    for (int i = 0; i < n; ++i) {
      Request r;
      r.prompt.assign(kPrompt, static_cast<TokenId>(100 + i));
      r.stop = StopParams{kGen, false};
      r.sampling.temperature = 0.8f;
      r.sampling.top_k = 40;
      r.sampling.top_p = 0.9f;
      r.sampling.seed = static_cast<uint64_t>(i + 1);
      r.sampling.has_seed = true;
      r.on_event = [](const RequestEvent&) {};
      sched.submit(std::move(r));
    }
    // Run past every prompt (plus a few warm-up decode steps), then count.
    while (sched.stats().prefill_rows_total < static_cast<uint64_t>(n * kPrompt)) sched.step();
    for (int i = 0; i < 4; ++i) sched.step();
    const uint64_t a0 = g_allocs.load(), b0 = g_bytes.load(), f0 = g_frees.load();
    const uint64_t s0 = sched.stats().steps_decode_only;
    for (int i = 0; i < 24; ++i) sched.step();
    const uint64_t steps = sched.stats().steps_decode_only - s0;
    const double k = steps > 0 ? 1.0 / static_cast<double>(steps) : 0.0;
    std::printf("%5d | %14.1f %14.0f %14.1f | %10llu\n", n, static_cast<double>(g_allocs.load() - a0) * k,
                static_cast<double>(g_bytes.load() - b0) * k, static_cast<double>(g_frees.load() - f0) * k,
                static_cast<unsigned long long>(steps));
    sched.run_until_idle();
  }
  return 0;
}

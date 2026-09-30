// Phase 15 benchmark: prefix reuse for a shared system prompt.
//
// One request warms the cache with a 512-token system prompt; then 15
// requests sharing that prompt (+16 unique tokens, 16 generated) arrive at
// once. Compares TTFT / wall time / rows computed with the cache on and off.
//
// Usage: bench_prefix <model.gguf> [threads]

#include <cstdio>
#include <cstdlib>
#include <map>

#include "backends/cpu/cpu_backend.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "runtime/sequence.h"
#include "scheduler/scheduler.h"

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_prefix <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  constexpr int kSystem = 512, kUnique = 16, kGen = 16, kReqs = 15;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, (kReqs + 2) * (kSystem + kUnique + kGen + 16)), be);
  auto tf = Transformer::create(c, (*m)->weights, be, 256);
  if (!kv.ok() || !tf.ok()) return 1;

  std::printf("model: %s  threads %d  system %d  unique %d  gen %d  requests 1+%d\n\n", argv[1], threads, kSystem,
              kUnique, kGen, kReqs);
  std::printf("%-8s %8s %12s %10s %10s %10s %9s\n", "cache", "wall s", "rows", "TTFT p50", "TTFT p90", "TTFT p99",
              "hit rate");
  std::vector<TokenId> system(kSystem);
  for (int i = 0; i < kSystem; ++i) system[static_cast<size_t>(i)] = static_cast<TokenId>(1000 + (i * 37) % 5000);

  for (bool enabled : {false, true}) {
    Scheduler sched(**tf, **kv, *(*m)->tokenizer,
                    SchedulerConfig{.decode_token_budget = 64, .prefill_token_budget = 128, .max_running = 32,
                                    .max_prefill_chunk = 128, .enable_prefix_cache = enabled});
    auto make = [&](int i, std::vector<double>* ttft, int64_t t0) {
      Request r;
      r.prompt = system;
      for (int u = 0; u < kUnique; ++u) r.prompt.push_back(static_cast<TokenId>(200 + i * kUnique + u));
      r.stop = StopParams{kGen, false};
      auto first = std::make_shared<bool>(true);
      r.on_event = [ttft, t0, first](const RequestEvent& ev) {
        if (!ev.finished && *first && ttft) {
          ttft->push_back(static_cast<double>(now_ns() - t0) * 1e-6);
          *first = false;
        }
      };
      return r;
    };
    sched.submit(make(-1, nullptr, 0));  // warm-up request
    sched.run_until_idle();
    const uint64_t rows_before = sched.stats().tokens_computed;
    std::vector<double> ttft;
    const int64_t t0 = now_ns();
    for (int i = 0; i < kReqs; ++i) sched.submit(make(i, &ttft, t0));
    sched.run_until_idle();
    const double wall = static_cast<double>(now_ns() - t0) * 1e-9;
    const auto st = bench::summarize(ttft);
    const double hit = enabled ? sched.prefix_cache()->stats().hit_rate() : 0.0;
    std::printf("%-8s %8.2f %12llu %10.0f %10.0f %10.0f %8.0f%%\n", enabled ? "on" : "off", wall,
                static_cast<unsigned long long>(sched.stats().tokens_computed - rows_before), st.p50, st.p90, st.p99,
                hit * 100);
  }
  return 0;
}

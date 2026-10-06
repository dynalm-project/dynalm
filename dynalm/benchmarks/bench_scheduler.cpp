// Phase 12 benchmark: continuous batching end to end through the Scheduler.
//
// N requests (64-token prompts, 64 generated tokens each) are submitted at
// once; the scheduler runs until idle. Reports aggregate throughput and the
// per-request TTFT / ITL distributions (P50/P90/P99).
//
// Usage: bench_scheduler <model.gguf> [threads] [prefill_token_budget]

#include <cstdio>
#include <cstdlib>
#include <map>

#include "dynacore/cpu/cpu_device.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "runtime/sequence.h"
#include "scheduler/scheduler.h"
#include "common/core.h"

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_scheduler <model.gguf> [threads] [prefill_token_budget]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  const int prefill_budget = argc > 3 ? std::atoi(argv[3]) : 64;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  constexpr int kPrompt = 64, kGen = 64;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, 32 * (kPrompt + kGen + 16)), be);
  auto tf = Transformer::create(c, (*m)->weights, be, 256);
  if (!kv.ok() || !tf.ok()) return 1;

  std::printf("model: %s  threads %d  prompt %d  gen %d  prefill budget %d\n\n", argv[1], threads, kPrompt, kGen,
              prefill_budget);
  std::printf("%5s %9s %9s | %8s %8s %8s | %8s %8s %8s\n", "reqs", "wall s", "tok/s", "TTFT p50", "p90", "p99",
              "ITL p50", "p90", "p99");
  for (int n : {1, 4, 8, 16, 32}) {
    Scheduler sched(**tf, **kv, *(*m)->tokenizer,
                    SchedulerConfig{.decode_token_budget = 64, .prefill_token_budget = prefill_budget, .max_running = 64});
    std::map<uint64_t, int64_t> last_ns;
    std::vector<double> ttft, itl;
    const int64_t t0 = now_ns();
    for (int i = 0; i < n; ++i) {
      Request r;
      r.prompt.assign(kPrompt, static_cast<TokenId>(100 + i));
      r.stop = StopParams{kGen, false};
      r.on_event = [&](const RequestEvent& ev) {
        if (ev.finished) return;
        const int64_t now = now_ns();
        auto it = last_ns.find(ev.request_id);
        if (it == last_ns.end()) {
          ttft.push_back(static_cast<double>(now - t0) * 1e-6);
          last_ns[ev.request_id] = now;
        } else {
          itl.push_back(static_cast<double>(now - it->second) * 1e-6);
          it->second = now;
        }
      };
      sched.submit(std::move(r));
    }
    sched.run_until_idle();
    const double wall = static_cast<double>(now_ns() - t0) * 1e-9;
    const auto tt = bench::summarize(ttft), it = bench::summarize(itl);
    std::printf("%5d %9.2f %9.1f | %8.0f %8.0f %8.0f | %8.1f %8.1f %8.1f\n", n, wall,
                static_cast<double>(sched.stats().tokens_generated) / wall, tt.p50, tt.p90, tt.p99, it.p50, it.p90,
                it.p99);
  }
  return 0;
}

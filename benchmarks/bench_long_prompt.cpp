// Phase 14 benchmark: a long prompt arriving while other requests decode.
//
// 8 requests are decoding when a 2048-token prompt arrives. Measures the
// decoders' inter-token latency while the long prompt is prefilled, and the
// long request's TTFT, for different prefill budgets / chunk caps.
//
// Usage: bench_long_prompt <model.gguf> [threads]

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
    std::fprintf(stderr, "usage: bench_long_prompt <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  constexpr int kLong = 2048, kDecoders = 8;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, kLong + kDecoders * 512), be);
  auto tf = Transformer::create(c, (*m)->weights, be, kLong + 64);
  if (!kv.ok() || !tf.ok()) return 1;

  std::printf("model: %s  threads %d  long prompt %d  decoders %d\n\n", argv[1], threads, kLong, kDecoders);
  std::printf("%-22s | %9s %9s %9s | %11s\n", "config", "ITL p50", "ITL p90", "ITL p99", "long TTFT s");
  struct Cfg {
    const char* name;
    int32_t budget, chunk;
  };
  for (const Cfg& cfg : {Cfg{"unchunked (2048)", kLong, 0}, Cfg{"budget 256 chunk 256", 256, 256},
                         Cfg{"budget 64 chunk 64", 64, 64}, Cfg{"budget 32 chunk 32", 32, 32}}) {
    Scheduler sched(**tf, **kv, *(*m)->tokenizer,
                    SchedulerConfig{.decode_token_budget = 64, .prefill_token_budget = cfg.budget, .max_running = 16,
                                    .max_prefill_chunk = cfg.chunk});
    std::map<uint64_t, int64_t> last;
    std::vector<double> itl;
    bool long_started = false, long_done = false;
    int64_t long_submit = 0, long_first = 0;
    uint64_t long_id = 0;
    for (int i = 0; i < kDecoders; ++i) {
      Request r;
      r.prompt.assign(32, static_cast<TokenId>(100 + i));
      r.stop = StopParams{400, false};
      r.on_event = [&](const RequestEvent& ev) {
        if (ev.finished) return;
        const int64_t now = now_ns();
        auto it = last.find(ev.request_id);
        // Record decoder ITL only while the long prompt is being prefilled.
        if (it != last.end() && long_started && !long_done) itl.push_back(static_cast<double>(now - it->second) * 1e-6);
        last[ev.request_id] = now;
      };
      sched.submit(std::move(r));
    }
    for (int s = 0; s < 8; ++s) sched.step();  // decoders are running
    Request big;
    big.prompt.assign(kLong, 7);
    big.stop = StopParams{1, false};
    big.on_event = [&](const RequestEvent& ev) {
      if (!ev.finished && !long_done) {
        long_first = now_ns();
        long_done = true;
      }
    };
    long_submit = now_ns();
    long_started = true;
    long_id = sched.submit(std::move(big));
    while (!long_done) sched.step();
    (void)long_id;
    const auto st = bench::summarize(itl);
    std::printf("%-22s | %9.1f %9.1f %9.1f | %11.2f\n", cfg.name, st.p50, st.p90, st.p99,
                static_cast<double>(long_first - long_submit) * 1e-9);
    // Drain the remaining decoders (they cancel to keep the run short).
  }
  return 0;
}

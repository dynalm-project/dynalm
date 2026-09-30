// Phase 9 benchmark: decode latency vs context length.
//
// Fills the KV cache to N tokens, then times single-token decode steps at
// that context. With KV appended in place (never copied), per-token cost
// should be flat apart from the attention reads, which grow linearly in N.
//
// Usage: bench_decode_context <model.gguf> [threads]

#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "backends/cpu/cpu_backend.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/sequence.h"

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_decode_context <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;

  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  constexpr int kMaxCtx = 4096 + 64;
  auto cache = KvCache::create(kv_geometry_for(c, DType::kF16, 16, kMaxCtx), be);
  auto t = Transformer::create(c, (*m)->weights, be, 256);
  if (!cache.ok() || !t.ok()) return 1;

  std::printf("model: %s  threads %d  kv f16\n\n", argv[1], threads);
  std::printf("%8s %10s %10s %10s %10s\n", "context", "p50 ms", "p90 ms", "p99 ms", "tok/s");
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  for (int ctx : {64, 256, 1024, 2048, 4096}) {
    KvSequence seq(**cache);
    if (!seq.reserve(ctx + 32).ok()) return 1;
    // Fill positions [0, ctx) in chunks.
    std::vector<TokenId> toks(256, 100);
    std::vector<int32_t> pos(256);
    for (int off = 0; off < ctx; off += 256) {
      const int n = std::min(256, ctx - off);
      std::iota(pos.begin(), pos.begin() + n, off);
      if (!(*t)->forward({toks.data(), static_cast<size_t>(n)}, {pos.data(), static_cast<size_t>(n)}, **cache,
                         seq.block_table(), logits).ok()) {
        return 1;
      }
    }
    std::vector<double> ms;
    for (int step = 0; step < 24; ++step) {
      const TokenId tok = 100;
      const int32_t p = ctx + step;
      const Stopwatch sw;
      if (!(*t)->forward({&tok, 1}, {&p, 1}, **cache, seq.block_table(), logits).ok()) return 1;
      if (step >= 4) ms.push_back(sw.elapsed_ms());  // skip warm-up steps
    }
    const auto s = bench::summarize(ms);
    std::printf("%8d %10.2f %10.2f %10.2f %10.1f\n", ctx, s.p50, s.p90, s.p99, 1000.0 / s.p50);
  }
  return 0;
}

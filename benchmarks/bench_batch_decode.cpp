// Phase 11 benchmark: aggregate decode throughput with N concurrent sequences.
//
// Each sequence first gets a 128-token context; then one step decodes one
// token for every sequence in a single batched forward pass. Weights are read
// once per step regardless of N, so aggregate tok/s should grow with N until
// compute (not weight bandwidth / conversion) dominates.
//
// Usage: bench_batch_decode <model.gguf> [threads]

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
    std::fprintf(stderr, "usage: bench_batch_decode <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  constexpr int kCtx = 128, kMaxSeqs = 32, kSteps = 12;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, (kCtx + 32) * kMaxSeqs), be);
  auto tf = Transformer::create(c, (*m)->weights, be, 256);
  if (!kv.ok() || !tf.ok()) return 1;

  std::printf("model: %s  threads %d  context %d\n\n", argv[1], threads, kCtx);
  std::printf("%6s %12s %12s %12s %14s\n", "seqs", "step p50 ms", "step p99 ms", "ITL p50 ms", "aggregate tok/s");
  const auto vocab = static_cast<size_t>(c.vocab_size);
  for (int n : {1, 2, 4, 8, 16, 32}) {
    std::vector<KvBlockTable> tables;
    std::vector<float> logits(vocab * static_cast<size_t>(n));
    std::vector<TokenId> ctx_tokens(kCtx, 100);
    std::vector<int32_t> pos(kCtx);
    std::iota(pos.begin(), pos.end(), 0);
    for (int s = 0; s < n; ++s) {
      tables.emplace_back(**kv);
      if (!tables.back().reserve(kCtx + kSteps + 1).ok()) return 1;
      if (!(*tf)->forward(ctx_tokens, pos, **kv, tables.back().block_table(), {logits.data(), vocab}).ok()) return 1;
    }
    std::vector<TokenId> step_tok(static_cast<size_t>(n), 200);
    std::vector<double> ms;
    for (int step = 0; step < kSteps; ++step) {
      std::vector<SeqBatch> batch;
      for (int s = 0; s < n; ++s) {
        batch.push_back({{&step_tok[static_cast<size_t>(s)], 1}, kCtx + step, tables[static_cast<size_t>(s)].block_table(),
                         true});
      }
      const Stopwatch sw;
      if (!(*tf)->forward_batch(batch, **kv, logits).ok()) return 1;
      if (step >= 2) ms.push_back(sw.elapsed_ms());
    }
    const auto st = bench::summarize(ms);
    std::printf("%6d %12.2f %12.2f %12.2f %14.1f\n", n, st.p50, st.p99, st.p50, n * 1000.0 / st.p50);
  }
  return 0;
}

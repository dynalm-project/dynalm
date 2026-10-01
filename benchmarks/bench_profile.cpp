// Phase 18: where does forward time go? Per-op breakdown for a 256-token
// prefill and for single-token decode steps at ~300 tokens of context.
//
// Usage: bench_profile <model.gguf> [threads]

#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "backends/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/sequence.h"

namespace {

void print(const char* title, const engine::ForwardProfile& p) {
  using namespace engine;
  const double total = static_cast<double>(p.total_ns());
  std::printf("%s: %llu calls, %llu rows, %.1f ms total, %.3f ms/row\n", title,
              static_cast<unsigned long long>(p.calls), static_cast<unsigned long long>(p.rows), total * 1e-6,
              total * 1e-6 / static_cast<double>(p.rows));
  for (size_t i = 0; i < p.ns.size(); ++i) {
    const double ms = static_cast<double>(p.ns[i]) * 1e-6;
    std::printf("  %-12s %9.2f ms  %5.1f%%\n", std::string(forward_op_name(static_cast<ForwardOp>(i))).c_str(), ms,
                100.0 * static_cast<double>(p.ns[i]) / total);
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_profile <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, 1024), be);
  auto tf = Transformer::create(c, (*m)->weights, be, 256);
  if (!kv.ok() || !tf.ok()) return 1;
  std::printf("model: %s  threads %d  backend %s\n\n", argv[1], threads, std::string(be.name()).c_str());

  KvBlockTable seq(**kv);
  if (!seq.reserve(512).ok()) return 1;
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  std::vector<TokenId> toks(256);
  for (size_t i = 0; i < toks.size(); ++i) toks[i] = static_cast<TokenId>(100 + i % 500);
  std::vector<int32_t> pos(256);
  std::iota(pos.begin(), pos.end(), 0);

  // Warm-up (page in the mmapped weights), then measure.
  (void)(*tf)->forward(toks, pos, **kv, seq.block_table(), logits);
  (*tf)->set_profiling(true);
  (*tf)->reset_profile();
  if (!(*tf)->forward(toks, pos, **kv, seq.block_table(), logits).ok()) return 1;
  print("prefill (256 tokens)", (*tf)->profile());

  (*tf)->reset_profile();
  for (int s = 0; s < 32; ++s) {
    const TokenId t = 100;
    const int32_t p = 256 + s;
    if (!(*tf)->forward({&t, 1}, {&p, 1}, **kv, seq.block_table(), logits).ok()) return 1;
  }
  std::printf("\n");
  print("decode (32 steps)", (*tf)->profile());
  return 0;
}

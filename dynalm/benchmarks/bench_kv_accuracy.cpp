// KV cache precision (DD-074): f16 KV vs q8_0 KV at a given context length.
//
//   bench_kv_accuracy <model.gguf> [context=1024] [threads]
//
// Both caches are filled with the same text (a paragraph repeated to the
// context length) by chunked prefill, so every stored K/V went through its
// cache's precision. Then the last 256 positions are fed one token at a time
// (decode shape) and the next-token distributions compared: perplexity,
// mean/max KL(f16 || q8), top-1 agreement. Finally both generate 64 tokens
// greedily and the first divergence is reported.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "common/core.h"

namespace {

constexpr const char* kText =
    "The river ran quietly through the valley, carrying the last of the autumn leaves toward the sea. "
    "On its banks the old mill still stood, its wheel long silent, its windows dark. Children from the "
    "village would sometimes dare each other to walk inside, but none of them stayed for long. The miller "
    "had left decades ago, and nobody remembered exactly why. Some said he had gone to the city to make "
    "his fortune; others insisted he had simply walked into the forest one winter morning and never come "
    "back. Whatever the truth, the mill had become part of the landscape, as natural as the hills and the "
    "water. In spring the meadow around it filled with flowers, and in summer the swallows nested under "
    "its roof. Every evening the light turned the stones gold, and for a few minutes the place looked "
    "almost alive again, as if the wheel might begin to turn and the miller might step out to greet the "
    "travelers on the road. ";

std::vector<double> log_softmax(const float* l, int64_t n) {
  double mx = -INFINITY;
  for (int64_t i = 0; i < n; ++i) mx = std::max(mx, static_cast<double>(l[i]));
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += std::exp(l[i] - mx);
  const double lse = mx + std::log(s);
  std::vector<double> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = l[i] - lse;
  return out;
}

int64_t argmax(const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); }

}  // namespace

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_kv_accuracy <model.gguf> [context=1024] [threads]\n");
    return 1;
  }
  const int ctx = argc > 2 ? std::max(300, std::atoi(argv[2])) : 1024;
  const int threads = argc > 3 ? std::atoi(argv[3]) : cpu_info().physical_cores;
  auto lm = load_model(argv[1]);
  if (!lm.ok()) return std::fprintf(stderr, "%s\n", lm.status().to_string().c_str()), 1;
  const ModelConfig& c = (*lm)->config;
  ThreadPool pool(threads);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  auto tf = Transformer::create(c, (*lm)->weights, be, 256);
  if (!tf.ok()) return std::fprintf(stderr, "%s\n", tf.status().to_string().c_str()), 1;

  std::string text;
  while (text.size() < static_cast<size_t>(ctx) * 8) text += kText;
  std::vector<TokenId> toks = (*lm)->tokenizer->encode(text, true, false);
  if (static_cast<int>(toks.size()) < ctx + 1) return std::fprintf(stderr, "text too short\n"), 1;
  toks.resize(static_cast<size_t>(ctx) + 1);
  const int64_t vocab = c.vocab_size;
  constexpr int kTail = 256, kGen = 64;
  const int prefill = ctx - kTail;

  struct Path {
    std::unique_ptr<KvBlockPool> kv;
    std::unique_ptr<KvBlockTable> table;
  };
  auto make = [&](DType dt) {
    Path p;
    auto kv = KvBlockPool::create(kv_geometry_for(c, dt, 16, ctx + kGen + 32), be);
    if (!kv.ok()) {
      std::fprintf(stderr, "%s\n", kv.status().to_string().c_str());
      std::exit(1);
    }
    p.kv = std::move(*kv);
    p.table = std::make_unique<KvBlockTable>(*p.kv);
    if (!p.table->reserve(ctx + kGen + 1).ok()) std::exit(1);
    return p;
  };
  std::vector<float> logits(static_cast<size_t>(vocab) * 256);
  auto run = [&](Path& p, std::span<const TokenId> t, int32_t pos0, std::vector<float>& out) {
    std::vector<int32_t> pos(t.size());
    for (size_t i = 0; i < t.size(); ++i) pos[i] = pos0 + static_cast<int32_t>(i);
    out.resize(static_cast<size_t>(vocab));
    if (!(*tf)->forward(t, pos, *p.kv, p.table->block_table(), out).ok()) std::exit(1);
  };

  Path f16 = make(DType::kF16), q8 = make(DType::kQ8_0);
  std::vector<float> lf, lq;
  for (int p0 = 0; p0 < prefill; p0 += 256) {
    const int len = std::min(256, prefill - p0);
    const std::span<const TokenId> chunk(toks.data() + p0, static_cast<size_t>(len));
    run(f16, chunk, p0, lf);
    run(q8, chunk, p0, lq);
  }
  double nll_f = 0, nll_q = 0, kl_sum = 0, kl_max = 0;
  int agree = 0, n = 0;
  for (int p = prefill; p < ctx; ++p) {
    const std::span<const TokenId> t(&toks[static_cast<size_t>(p)], 1);
    run(f16, t, p, lf);
    run(q8, t, p, lq);
    const auto a = log_softmax(lf.data(), vocab), b = log_softmax(lq.data(), vocab);
    const auto next = static_cast<size_t>(toks[static_cast<size_t>(p) + 1]);
    nll_f -= a[next];
    nll_q -= b[next];
    double kl = 0;
    for (int64_t i = 0; i < vocab; ++i) kl += std::exp(a[static_cast<size_t>(i)]) * (a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
    kl_sum += kl;
    kl_max = std::max(kl_max, kl);
    agree += argmax(lf) == argmax(lq) ? 1 : 0;
    ++n;
  }
  // Greedy continuation from the full context.
  std::vector<TokenId> gf, gq;
  for (int path = 0; path < 2; ++path) {
    Path& p = path == 0 ? f16 : q8;
    std::vector<float>& l = path == 0 ? lf : lq;
    std::vector<TokenId>& g = path == 0 ? gf : gq;
    TokenId t = static_cast<TokenId>(argmax(l));
    for (int i = 0; i < kGen; ++i) {
      g.push_back(t);
      run(p, std::span<const TokenId>(&t, 1), ctx + i, l);
      t = static_cast<TokenId>(argmax(l));
    }
  }
  size_t diverge = 0;
  while (diverge < gf.size() && gf[diverge] == gq[diverge]) ++diverge;
  const KvGeometry gf16 = kv_geometry_for(c, DType::kF16, 16, ctx), gq8 = kv_geometry_for(c, DType::kQ8_0, 16, ctx);
  std::printf("model %s, context %d, last %d positions decoded\n", argv[1], ctx, n);
  std::printf("KV bytes per %d tokens: f16 %.1f MiB, q8_0 %.1f MiB\n", ctx, gf16.total_bytes() / 1048576.0,
              gq8.total_bytes() / 1048576.0);
  std::printf("perplexity f16 %.4f | q8_0 %.4f | change %+.3f%%\n", std::exp(nll_f / n), std::exp(nll_q / n),
              100.0 * (std::exp(nll_q / n) / std::exp(nll_f / n) - 1.0));
  std::printf("KL(f16 || q8_0) mean %.6f max %.6f nats | top-1 agreement %.2f%%\n", kl_sum / n, kl_max,
              100.0 * agree / n);
  std::printf("greedy continuation identical for %zu of %d tokens\n", diverge, kGen);
  return 0;
}

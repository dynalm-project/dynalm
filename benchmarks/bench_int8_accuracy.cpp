// Performance program P3: accuracy of int8-activation decode (DD-053).
//
//   bench_int8_accuracy <model.gguf> [threads]
//
// Feeds a fixed English text one token at a time (the decode shape, M = 1)
// through two KV histories: the fp32-activation path and the int8 path. At
// every position it compares the next-token distributions:
//   perplexity of the true next tokens (fp32 vs int8),
//   mean / max KL(fp32 || int8), top-1 agreement, max |logit difference|.
// Then both paths generate 64 tokens greedily from the same prompt and the
// first position where they diverge is reported. The int8 path is enabled by
// default only if these stay within the thresholds in DD-053.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"
#include "runtime/engine.h"

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
    "travelers on the road.";

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
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_int8_accuracy <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto lm = load_model(argv[1]);
  if (!lm.ok()) {
    std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
    return 1;
  }
  const ModelConfig& c = (*lm)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  auto tf = Transformer::create(c, (*lm)->weights, be, 64);
  if (!tf.ok()) {
    std::fprintf(stderr, "%s\n", tf.status().to_string().c_str());
    return 1;
  }
  const std::vector<TokenId> text = (*lm)->tokenizer->encode(kText, true, false);
  const int64_t vocab = c.vocab_size;
  const int32_t ctx = 2 * (static_cast<int32_t>(text.size()) + 96);
  auto make_kv = [&]() -> std::unique_ptr<KvBlockPool> {
    auto p = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, ctx), be);
    if (!p.ok()) std::abort();
    return std::move(*p);
  };

  // One token at `pos` through one path; logits into `out`.
  auto step = [&](KvBlockPool& kv, KvBlockTable& table, TokenId t, int32_t pos, int mode, std::vector<float>& out) {
    if (!table.reserve(pos + 1).ok()) std::abort();
    const SeqBatch b{std::span<const TokenId>(&t, 1), pos, table.block_table(), true};
    ExecutionPlan plan = (*tf)->planner().plan({&b, 1});
    plan.kernels.int8_decode_max_rows = mode == 1 || mode == 3 ? 4 : 0;
    plan.kernels.int8_ffn_down = mode != 3;
    plan.kernels.expand_min_rows = mode == 2 ? 1 : 1 << 30;
    out.resize(static_cast<size_t>(vocab));
    if (!(*tf)->forward_batch({&b, 1}, kv, out, &plan).ok()) std::abort();
  };

  auto compare = [&](int alt, const char* label) {
    std::unique_ptr<KvBlockPool> kv_fp = make_kv(), kv_i8 = make_kv();
  // --- teacher-forced comparison over the text ---
  KvBlockTable tab_fp(*kv_fp), tab_i8(*kv_i8);
  std::vector<float> lf, li;
  double nll_fp = 0, nll_i8 = 0, kl_sum = 0, kl_max = 0, dmax = 0;
  int agree = 0, n = 0;
  for (size_t p = 0; p + 1 < text.size(); ++p) {
    step(*kv_fp, tab_fp, text[p], static_cast<int32_t>(p), 0, lf);
    step(*kv_i8, tab_i8, text[p], static_cast<int32_t>(p), alt, li);
    const auto a = log_softmax(lf.data(), vocab), b = log_softmax(li.data(), vocab);
    const auto next = static_cast<size_t>(text[p + 1]);
    nll_fp -= a[next];
    nll_i8 -= b[next];
    double kl = 0;
    for (int64_t i = 0; i < vocab; ++i) {
      kl += std::exp(a[static_cast<size_t>(i)]) * (a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
      dmax = std::max(dmax, std::abs(static_cast<double>(lf[static_cast<size_t>(i)]) - li[static_cast<size_t>(i)]));
    }
    kl_sum += kl;
    kl_max = std::max(kl_max, kl);
    agree += argmax(lf) == argmax(li) ? 1 : 0;
    ++n;
  }

  // --- free-running greedy continuation from the first 32 tokens ---
  tab_fp.release();
  tab_i8.release();
  const size_t prompt = std::min<size_t>(32, text.size());
  std::vector<TokenId> out_fp, out_i8;
  for (int path = 0; path < 2; ++path) {
    KvBlockPool& kv = path == 0 ? *kv_fp : *kv_i8;
    KvBlockTable table(kv);
    std::vector<float> logits;
    TokenId t = text[0];
    std::vector<TokenId>& out = path == 0 ? out_fp : out_i8;
    for (int32_t pos = 0; pos < static_cast<int32_t>(prompt) + 64; ++pos) {
      step(kv, table, t, pos, path == 1 ? alt : 0, logits);
      const auto next = static_cast<TokenId>(argmax(logits));
      if (pos + 1 < static_cast<int32_t>(prompt)) {
        t = text[static_cast<size_t>(pos + 1)];
      } else {
        out.push_back(next);
        t = next;
      }
    }
  }
  size_t diverge = 0;
  while (diverge < out_fp.size() && out_fp[diverge] == out_i8[diverge]) ++diverge;

  std::printf("\n[%s vs fp32 fused]\n", label);
  std::printf("teacher-forced positions: %d\n", n);
  std::printf("perplexity fp32 %.4f | alt %.4f | change %+.3f%%\n", std::exp(nll_fp / n), std::exp(nll_i8 / n),
              100.0 * (std::exp(nll_i8 / n) / std::exp(nll_fp / n) - 1.0));
  std::printf("KL(fp32 || alt) mean %.6f max %.6f nats\n", kl_sum / n, kl_max);
  std::printf("top-1 agreement %.2f%% | max |logit diff| %.4f\n", 100.0 * agree / n, dmax);
  std::printf("greedy continuation: identical for %zu of %zu tokens\n", diverge, out_fp.size());
  };
  std::printf("model: %s (%s), %d threads\n", argv[1], (*lm)->quantization.c_str(), threads);
  compare(2, "fp32 expand path (summation-order noise floor)");
  compare(1, "int8 activations");
  compare(3, "int8 activations except ffn_down");
  return 0;
}

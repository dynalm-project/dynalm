// DD-066 accuracy: grouped GQA attention vs the per-head path.
//
//   bench_attn_accuracy <model.gguf> [contexts=64,1024,3072] [threads] [mode=ab|floor]
//
// mode=floor compares two pre-DD-066 runs that differ only in float ordering
// (per-pair vs split-K attention): the noise floor any exact reordering of
// attention arithmetic produces, against which the ab differences are judged.
//
// For each context length L: a text of L tokens is prefilled in 256-token
// chunks (prefill attention) through two KV histories, one per attention
// path; then the next 32 true tokens are fed one at a time (decode attention).
// At the last prompt position and the 32 decode positions the logits are
// compared: max |difference|, RMSE, mean / max KL, top-1 agreement, top-5
// overlap, and the perplexity of the true next tokens. Finally both paths
// generate 48 tokens greedily from position L and the first divergence is
// reported. The pass criterion is the project's accuracy contract (DD-053):
// perplexity change <= 1%, mean KL <= 0.0025 nats.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_backend.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "runtime/sequence.h"

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

std::vector<int64_t> top_k(const std::vector<float>& v, int k) {
  std::vector<int64_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) { return v[a] > v[b]; });
  idx.resize(static_cast<size_t>(k));
  return idx;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_attn_accuracy <model.gguf> [contexts=64,1024,3072] [threads]\n");
    return 1;
  }
  std::vector<int32_t> contexts;
  {
    std::stringstream ss(argc > 2 ? argv[2] : "64,1024,3072");
    for (std::string t; std::getline(ss, t, ',');) contexts.push_back(std::atoi(t.c_str()));
  }
  const int threads = argc > 3 && std::atoi(argv[3]) > 0 ? std::atoi(argv[3]) : cpu_info().physical_cores;
  const bool floor_mode = argc > 4 && std::string(argv[4]) == "floor";
  auto lm = load_model(argv[1]);
  if (!lm.ok()) {
    std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
    return 1;
  }
  const ModelConfig& c = (*lm)->config;
  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  constexpr int32_t kChunk = 256, kDecode = 32, kGreedy = 48;
  auto tf = Transformer::create(c, (*lm)->weights, be, kChunk);
  if (!tf.ok()) {
    std::fprintf(stderr, "%s\n", tf.status().to_string().c_str());
    return 1;
  }
  const int32_t max_ctx = *std::max_element(contexts.begin(), contexts.end());
  // A long document: the text repeated with numbered section markers.
  std::vector<TokenId> text;
  for (int sec = 1; static_cast<int32_t>(text.size()) < max_ctx + kDecode + 1; ++sec) {
    const std::string part = "Section " + std::to_string(sec) + ". " + kText + "\n";
    const std::vector<TokenId> t = (*lm)->tokenizer->encode(part, sec == 1, false);
    text.insert(text.end(), t.begin(), t.end());
  }
  const int64_t vocab = c.vocab_size;
  const int32_t gqa = c.num_heads / std::max(1, c.num_kv_heads);
  if (floor_mode) std::printf("mode: noise floor (per-pair vs split-K, both pre-DD-066)\n");
  std::printf("model: %s (%s) | heads %d, kv heads %d (group %d), head dim %d | %d threads\n", argv[1],
              (*lm)->quantization.c_str(), c.num_heads, c.num_kv_heads, gqa, c.head_dim, threads);
  std::printf("%7s %9s %10s %12s %12s %8s %8s %10s %10s %12s\n", "context", "positions", "max|dl|", "rmse", "KL mean",
              "top1%", "top5%", "ppl old", "ppl new", "greedy same");

  auto kv_pool = [&](int32_t tokens) {
    auto p = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, tokens + 64), be);
    if (!p.ok()) std::abort();
    return std::move(*p);
  };
  bool pass = true;
  for (int32_t L : contexts) {
    auto kv_old = kv_pool(L + kGreedy + kDecode), kv_new = kv_pool(L + kGreedy + kDecode);
    KvBlockTable t_old(*kv_old), t_new(*kv_new);
    // Runs tokens [from, from+n) of `toks` at positions [pos, pos+n) through one path; logits of the last token.
    auto run = [&](KvBlockPool& kv, KvBlockTable& table, const TokenId* toks, int32_t pos, int32_t n, bool grouped,
                   std::vector<float>& out) {
      if (!table.reserve(pos + n).ok()) std::abort();
      out.resize(static_cast<size_t>(vocab));
      for (int32_t p0 = 0; p0 < n; p0 += kChunk) {
        const int32_t len = std::min(kChunk, n - p0);
        const bool last = p0 + len == n;
        const SeqBatch b{std::span<const TokenId>(toks + p0, static_cast<size_t>(len)), pos + p0, table.block_table(), last};
        ExecutionPlan plan = (*tf)->planner().plan({&b, 1});
        if (floor_mode) {  // "grouped" selects the second variant: split-K instead of per-pair
          plan.kernels.grouped_attention = false;
          plan.kernels.attention_full = plan.kernels.attention_window =
              grouped ? AttentionStrategy::kSplitK : AttentionStrategy::kPerPair;
        } else {
          plan.kernels.grouped_attention = grouped;
        }
        if (!(*tf)->forward_batch({&b, 1}, kv, last ? std::span<float>(out) : std::span<float>(), &plan).ok()) {
          std::abort();
        }
      }
    };
    std::vector<float> lo, ln;
    double dmax = 0, sq = 0, kl_sum = 0, nll_o = 0, nll_n = 0;
    int64_t cnt = 0;
    int agree = 0, top5 = 0, positions = 0;
    auto compare = [&](size_t next) {
      const auto a = log_softmax(lo.data(), vocab), b = log_softmax(ln.data(), vocab);
      double kl = 0;
      for (int64_t i = 0; i < vocab; ++i) {
        const double d = static_cast<double>(lo[static_cast<size_t>(i)]) - ln[static_cast<size_t>(i)];
        dmax = std::max(dmax, std::abs(d));
        sq += d * d;
        kl += std::exp(a[static_cast<size_t>(i)]) * (a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
      }
      cnt += vocab;
      kl_sum += kl;
      nll_o -= a[next];
      nll_n -= b[next];
      const auto ko = top_k(lo, 5), kn = top_k(ln, 5);
      agree += ko[0] == kn[0] ? 1 : 0;
      for (int64_t x : ko) top5 += std::count(kn.begin(), kn.end(), x) > 0 ? 1 : 0;
      ++positions;
    };
    // Prefill L tokens (chunked), compare at the last prompt position.
    run(*kv_old, t_old, text.data(), 0, L, false, lo);
    run(*kv_new, t_new, text.data(), 0, L, true, ln);
    compare(static_cast<size_t>(text[static_cast<size_t>(L)]));
    // Teacher-forced decode of the next kDecode true tokens.
    for (int32_t d = 0; d < kDecode; ++d) {
      const int32_t pos = L + d;
      run(*kv_old, t_old, &text[static_cast<size_t>(pos)], pos, 1, false, lo);
      run(*kv_new, t_new, &text[static_cast<size_t>(pos)], pos, 1, true, ln);
      compare(static_cast<size_t>(text[static_cast<size_t>(pos + 1)]));
    }
    // Greedy continuation from position L (KV truncated back to the prompt).
    t_old.truncate(L);
    t_new.truncate(L);
    std::vector<TokenId> g_old, g_new;
    for (int path = 0; path < 2; ++path) {
      KvBlockPool& kv = path == 0 ? *kv_old : *kv_new;
      KvBlockTable& table = path == 0 ? t_old : t_new;
      std::vector<TokenId>& out = path == 0 ? g_old : g_new;
      std::vector<float> logits;
      TokenId t = text[static_cast<size_t>(L)];
      for (int32_t i = 0; i < kGreedy; ++i) {
        run(kv, table, &t, L + i, 1, path == 1, logits);
        t = static_cast<TokenId>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        out.push_back(t);
      }
    }
    size_t same = 0;
    while (same < g_old.size() && g_old[same] == g_new[same]) ++same;
    const double ppl_o = std::exp(nll_o / positions), ppl_n = std::exp(nll_n / positions);
    const double kl_mean = kl_sum / positions;
    std::printf("%7d %9d %10.2e %12.2e %12.2e %7.1f%% %7.1f%% %10.4f %10.4f %7zu / %d\n", L, positions, dmax,
                std::sqrt(sq / static_cast<double>(cnt)), kl_mean, 100.0 * agree / positions,
                100.0 * top5 / (5.0 * positions), ppl_o, ppl_n, same, kGreedy);
    if (std::abs(ppl_n / ppl_o - 1.0) > 0.01 || kl_mean > 0.0025) pass = false;
  }
  std::printf("accuracy contract (DD-053: perplexity <= 1%%, mean KL <= 0.0025): %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 2;
}

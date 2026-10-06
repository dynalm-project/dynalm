// Phase 26: sampling (DD-043).

#include "sampling/sampler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "runtime/engine.h"
#include "common/core.h"

namespace dynalm {
namespace {

std::vector<double> softmax(const std::vector<float>& l, float t) {
  const float mx = *std::max_element(l.begin(), l.end());
  std::vector<double> p(l.size());
  double s = 0;
  for (size_t i = 0; i < l.size(); ++i) s += p[i] = std::exp((l[i] - mx) / t);
  for (double& v : p) v /= s;
  return p;
}

// Empirical distribution of `n` draws.
std::vector<double> draws(Sampler& s, const std::vector<float>& logits, int n) {
  std::vector<double> f(logits.size());
  std::vector<float> row;
  for (int i = 0; i < n; ++i) {
    row = logits;
    f[static_cast<size_t>(s.sample(row, {}))] += 1.0 / n;
  }
  return f;
}

SamplingParams with_seed(SamplingParams p, uint64_t seed = 7) {
  p.seed = seed;
  p.has_seed = true;
  return p;
}

TEST(Rng, PortableSequence) {
  // splitmix64-seeded xoshiro256**, values computed independently in Python.
  Rng r(42);
  EXPECT_EQ(r.next(), 0x15780b2e0c2ec716ull);
  EXPECT_EQ(r.next(), 0x6104d9866d113a7eull);
  EXPECT_EQ(r.next(), 0xae17533239e499a1ull);
  Rng u(1);
  for (int i = 0; i < 1000; ++i) {
    const double x = u.uniform();
    ASSERT_GE(x, 0.0);
    ASSERT_LT(x, 1.0);
  }
}

TEST(Sampler, GreedyPaths) {
  std::vector<float> l = {0.1f, 2.0f, 2.0f, -1.0f};
  EXPECT_EQ(sample_greedy(l), 1);  // ties -> lowest id
  Sampler zero(with_seed({}));
  EXPECT_EQ(zero.sample(l, {}), 1);
  SamplingParams k1;
  k1.temperature = 1.5f;
  k1.top_k = 1;
  Sampler topk1(with_seed(k1));
  for (int i = 0; i < 20; ++i) {
    std::vector<float> row = l;
    EXPECT_EQ(topk1.sample(row, {}), 1);
  }
}

TEST(Sampler, MatchesSoftmaxWithTemperature) {
  const std::vector<float> l = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f};
  for (float t : {1.0f, 0.5f, 1.7f}) {
    SamplingParams p;
    p.temperature = t;
    Sampler s(with_seed(p, 11));
    const auto f = draws(s, l, 200000);
    const auto want = softmax(l, t);
    for (size_t i = 0; i < l.size(); ++i) EXPECT_NEAR(f[i], want[i], 0.006) << "T=" << t << " token " << i;
  }
}

TEST(Sampler, TopKTopPMinPSupport) {
  // Probabilities 0.5, 0.3, 0.15, 0.05 at T = 1.
  const std::vector<float> l = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  auto check = [&](SamplingParams p, const std::vector<double>& want) {
    p.temperature = 1.0f;
    Sampler s(with_seed(p, 3));
    const auto f = draws(s, l, 100000);
    for (size_t i = 0; i < l.size(); ++i) EXPECT_NEAR(f[i], want[i], 0.006) << i;
  };
  SamplingParams k2;
  k2.top_k = 2;
  check(k2, {0.625, 0.375, 0, 0});  // renormalized top-2
  SamplingParams p75;
  p75.top_p = 0.75f;
  check(p75, {0.625, 0.375, 0, 0});  // 0.5 < 0.75, 0.5 + 0.3 >= 0.75
  SamplingParams p95;
  p95.top_p = 0.95f;
  check(p95, {0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0});
  SamplingParams mp;
  mp.min_p = 0.4f;  // keep p >= 0.2
  check(mp, {0.625, 0.375, 0, 0});
}

TEST(Sampler, PenaltiesExact) {
  SamplingParams p;
  p.repetition_penalty = 2.0f;
  p.frequency_penalty = 0.5f;
  p.presence_penalty = 0.25f;
  Sampler s(with_seed(p));
  std::vector<float> l = {2.0f, -1.0f, 0.5f, 3.0f};
  const std::vector<TokenId> ctx = {0, 0, 1};
  s.apply_penalties(l, ctx);
  EXPECT_FLOAT_EQ(l[0], 2.0f / 2 - 2 * 0.5f - 0.25f);   // seen twice
  EXPECT_FLOAT_EQ(l[1], -1.0f * 2 - 1 * 0.5f - 0.25f);  // negative logit: multiplied
  EXPECT_FLOAT_EQ(l[2], 0.5f);                          // unseen: unchanged
  EXPECT_FLOAT_EQ(l[3], 3.0f);
  // Window: only the last token counts.
  p.penalty_last_n = 1;
  Sampler w(with_seed(p));
  std::vector<float> l2 = {2.0f, -1.0f, 0.5f, 3.0f};
  w.apply_penalties(l2, ctx);
  EXPECT_FLOAT_EQ(l2[0], 2.0f);
  EXPECT_FLOAT_EQ(l2[1], -2.75f);
  // A penalized greedy sampler avoids the repeated argmax.
  SamplingParams g;
  g.repetition_penalty = 10.0f;
  Sampler gs(with_seed(g));
  std::vector<float> l3 = {5.0f, 4.0f};
  const std::vector<TokenId> ctx0 = {0};
  EXPECT_EQ(gs.sample(l3, ctx0), 1);
}

TEST(Sampler, LargeVocabNucleusIsExact) {
  // 152k random logits: every sampled token must lie in the brute-force nucleus.
  constexpr size_t kVocab = 151936;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  std::vector<float> l(kVocab);
  for (float& v : l) v = nd(rng);
  SamplingParams p;
  p.temperature = 0.8f;
  p.top_p = 0.9f;
  const auto probs = softmax(l, p.temperature);
  std::vector<size_t> order(kVocab);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return l[a] > l[b]; });
  std::set<size_t> nucleus;
  double mass = 0;
  for (size_t i : order) {
    if (mass >= 0.9) break;
    mass += probs[i];
    nucleus.insert(i);
  }
  Sampler s(with_seed(p));
  for (int i = 0; i < 300; ++i) {
    std::vector<float> row = l;
    EXPECT_TRUE(nucleus.count(static_cast<size_t>(s.sample(row, {}))));
  }
}

TEST(Sampler, SeedDeterminism) {
  const std::vector<float> l = {0.0f, 0.3f, 0.6f, 0.9f, 1.2f};
  SamplingParams p;
  p.temperature = 1.0f;
  auto run = [&](uint64_t seed) {
    Sampler s(with_seed(p, seed));
    std::vector<TokenId> out;
    for (int i = 0; i < 64; ++i) {
      std::vector<float> row = l;
      out.push_back(s.sample(row, {}));
    }
    return out;
  };
  EXPECT_EQ(run(123), run(123));
  EXPECT_NE(run(123), run(124));
}

TEST(Sampler, ValidatesParams) {
  SamplingParams p;
  EXPECT_TRUE(p.validate().ok());
  for (auto mutate : {+[](SamplingParams& q) { q.top_p = 0; }, +[](SamplingParams& q) { q.top_k = -1; },
                      +[](SamplingParams& q) { q.temperature = -1; }, +[](SamplingParams& q) { q.min_p = 2; },
                      +[](SamplingParams& q) { q.repetition_penalty = 0; }}) {
    SamplingParams q;
    mutate(q);
    EXPECT_FALSE(q.validate().ok());
  }
}

// Through the engine: seeded requests reproduce, greedy is unchanged.
std::string generate(Engine& e, const SamplingParams& sp) {
  GenerateParams gp;
  gp.max_tokens = 24;
  gp.stop_at_eog = false;
  gp.sampling = sp;
  auto s = e.generate_text("sampling test", false, gp);
  EXPECT_TRUE(s.ok());
  std::string text;
  StreamEvent ev;
  while ((*s)->next(ev) && !ev.done) text += ev.text;
  return text + ev.text;
}

TEST(SamplingEngine, SeededRequestsReproduce) {
  EngineOptions o;
  o.model_path = std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf";
  o.threads = 2;
  o.kv_tokens = 1024;
  auto e = Engine::create(o);
  ASSERT_TRUE(e.ok());
  SamplingParams sp;
  sp.temperature = 1.5f;
  sp.top_p = 0.95f;
  const std::string a = generate(**e, with_seed(sp, 1)), b = generate(**e, with_seed(sp, 1));
  const std::string c = generate(**e, with_seed(sp, 2));
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_EQ(generate(**e, {}), generate(**e, {}));  // greedy

  // Invalid params fail the request cleanly.
  SamplingParams bad;
  bad.top_p = -1.0f;
  GenerateParams gp;
  gp.sampling = bad;
  auto s = (*e)->generate_text("x", false, gp);
  ASSERT_TRUE(s.ok());
  StreamEvent ev;
  while ((*s)->next(ev) && !ev.done) {
  }
  EXPECT_EQ(ev.finish, StreamFinish::kError);
}

}  // namespace
}  // namespace dynalm

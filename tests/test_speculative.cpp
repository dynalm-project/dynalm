// Phase 27: speculative decoding (DD-044).

#include "runtime/speculative.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

// A model with its backend, KV pool and transformer.
struct Loaded {
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<ThreadPool> pool;
  std::unique_ptr<CpuBackend> backend;
  std::unique_ptr<KvBlockPool> kv;
  std::unique_ptr<Transformer> tf;
};

Loaded load(const std::string& path, int32_t max_batch = 16) {
  Loaded l;
  auto m = load_model(path);
  EXPECT_TRUE(m.ok()) << m.status().to_string();
  l.model = std::move(*m);
  l.pool = std::make_unique<ThreadPool>(2);
  l.backend = std::make_unique<CpuBackend>(*l.pool, select_best_isa(cpu_info().features));
  const ModelConfig& c = l.model->config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 16, DType::kF32};
  auto kv = KvBlockPool::create(g, *l.backend);
  auto tf = Transformer::create(c, l.model->weights, *l.backend, max_batch);
  EXPECT_TRUE(kv.ok() && tf.ok());
  l.kv = std::move(*kv);
  l.tf = std::move(*tf);
  return l;
}

std::vector<TokenId> plain_greedy(Loaded& l, const std::vector<TokenId>& prompt, int32_t n) {
  Generator g(*l.tf, *l.kv, *l.model->tokenizer);
  GenerateOptions o;
  o.max_new_tokens = n;
  o.stop_at_eog = false;
  std::vector<TokenId> out;
  EXPECT_TRUE(g.generate(prompt, o, [&](TokenId t) { out.push_back(t); return true; }).ok());
  return out;
}

std::vector<TokenId> speculative(Loaded& l, Drafter& d, const std::vector<TokenId>& prompt, int32_t n, int32_t k,
                                 SpeculativeStats* st, SamplingParams sp = {}, bool adaptive = false) {
  SpeculativeGenerator g(*l.tf, *l.kv, *l.model->tokenizer, d);
  GenerateOptions o;
  o.max_new_tokens = n;
  o.stop_at_eog = false;
  SpeculativeOptions so;
  so.draft_tokens = k;
  so.sampling = sp;
  so.adaptive = adaptive;
  std::vector<TokenId> out;
  EXPECT_TRUE(g.generate(prompt, o, so, [&](TokenId t) { out.push_back(t); return true; }, st).ok());
  return out;
}

// A prompt with repetition, so prompt lookup finds continuations.
const std::vector<TokenId> kPrompt = {5, 17, 99, 3, 200, 42, 7, 5, 17, 99, 3, 200, 42, 7, 5, 17, 99};

TEST(SpecController, WarmsUpEveryArmThenPicksTheFastest) {
  SpecController c(5);  // arms 5, 3, 0
  std::vector<int32_t> seen;
  // The very first drafting round is a catch-up round and is not measured.
  for (int i = 0; i < 3 * SpecController::kWarmupRounds + 1; ++i) {
    const int32_t k = c.next_k();
    seen.push_back(k);
    // Plain: 1 token in 10 ms. k=3: 2 tokens in 15 ms. k=5: 2 tokens in 20 ms.
    c.record(k, k == 0 ? 1 : 2, k == 0 ? 10.0 : k == 3 ? 15.0 : 20.0);
  }
  EXPECT_EQ(std::count(seen.begin(), seen.end(), 5), SpecController::kWarmupRounds + 1);
  for (int32_t k : {3, 0}) EXPECT_EQ(std::count(seen.begin(), seen.end(), k), SpecController::kWarmupRounds);
  EXPECT_EQ(c.best_k(), 3);
  EXPECT_NEAR(c.rate(3), 2.0 / 15.0, 1e-12);
}

TEST(SpecController, TurnsSpeculationOffWhenItLosesAndBackOnWhenItWins) {
  SpecController c(4);  // arms 4, 3, 0
  bool drafts_pay = false;
  auto round = [&] {
    const int32_t k = c.next_k();
    // Plain decoding: 1 token / 10 ms. Speculation: 1.2 tokens / 14 ms while drafts
    // are poor, 4 tokens / 16 ms once they are good (repetitive text).
    if (k == 0) c.record(k, 1, 10.0);
    else c.record(k, drafts_pay ? 4 : 1, drafts_pay ? 16.0 : 14.0);
    return k;
  };
  int plain = 0;
  for (int i = 0; i < 400; ++i) plain += round() == 0;
  EXPECT_EQ(c.best_k(), 0);
  EXPECT_GT(plain, 370);  // warm-up and backed-off probes cost a few rounds, the rest run plain

  // The next probe (at most kReprobeMax rounds away) finds the change.
  drafts_pay = true;
  int spec = 0;
  for (int i = 0; i < 400; ++i) spec += round() != 0;
  EXPECT_NE(c.best_k(), 0);
  EXPECT_GT(spec, 400 - SpecController::kReprobeMax - 20);
}

TEST(Speculative, AdaptiveGreedyIsExact) {
  Loaded target = load(data("tiny_llama.gguf"));
  const std::vector<TokenId> want = plain_greedy(target, kPrompt, 60);
  Loaded other = load(data("tiny_qwen2.gguf"));
  auto other_d = ModelDrafter::create(*other.tf, *other.kv, *other.model->tokenizer, *target.model->tokenizer);
  ASSERT_TRUE(other_d.ok());
  NgramDrafter ngram;
  for (Drafter* d : {static_cast<Drafter*>(&ngram), static_cast<Drafter*>(other_d->get())}) {
    for (int32_t k : {1, 4, 6}) {
      SpeculativeStats st;
      EXPECT_EQ(speculative(target, *d, kPrompt, 60, k, &st, {}, /*adaptive=*/true), want) << d->name() << k;
      EXPECT_EQ(st.generated, 60);
    }
  }
  EXPECT_EQ(target.kv->free_blocks(), target.kv->num_blocks());
}

TEST(Ngram, ProposesContinuationOfLatestMatch) {
  NgramDrafter d(3, 2);
  std::vector<TokenId> out;
  const std::vector<TokenId> ctx = {1, 2, 3, 9, 9, 1, 2, 3, 4, 5, 1, 2, 3};
  d.propose(ctx, 3, out);
  EXPECT_EQ(out, (std::vector<TokenId>{4, 5, 1}));  // latest earlier "1 2 3" is followed by 4 5 1
  d.propose(std::vector<TokenId>{7, 8}, 3, out);
  EXPECT_TRUE(out.empty());
}

TEST(Speculative, GreedyIsExactWithEveryDrafter) {
  Loaded target = load(data("tiny_llama.gguf"));
  const std::vector<TokenId> want = plain_greedy(target, kPrompt, 40);

  NgramDrafter ngram;
  SpeculativeStats st_ngram;
  EXPECT_EQ(speculative(target, ngram, kPrompt, 40, 4, &st_ngram), want);

  // The model drafting for itself: every draft is accepted.
  Loaded self = load(data("tiny_llama.gguf"));
  auto self_d = ModelDrafter::create(*self.tf, *self.kv, *self.model->tokenizer, *target.model->tokenizer);
  ASSERT_TRUE(self_d.ok()) << self_d.status().to_string();
  SpeculativeStats st_self;
  EXPECT_EQ(speculative(target, **self_d, kPrompt, 40, 4, &st_self), want);
  EXPECT_DOUBLE_EQ(st_self.acceptance(), 1.0);
  EXPECT_GT(st_self.tokens_per_pass(), 4.0);  // k accepted + 1 bonus per pass (until the end)

  // A different model with the same vocabulary: partial acceptance, same output.
  Loaded other = load(data("tiny_qwen2.gguf"));
  auto other_d = ModelDrafter::create(*other.tf, *other.kv, *other.model->tokenizer, *target.model->tokenizer);
  ASSERT_TRUE(other_d.ok());
  SpeculativeStats st_other;
  for (int32_t k : {1, 3, 6}) EXPECT_EQ(speculative(target, **other_d, kPrompt, 40, k, &st_other), want) << k;
  EXPECT_LT(st_other.acceptance(), 1.0);

  // All KV returned after generations with rollbacks.
  EXPECT_EQ(target.kv->free_blocks(), target.kv->num_blocks());
}

TEST(Speculative, VerificationPreservesTheSamplingDistribution) {
  // Token distribution after verifying a fixed draft must equal sample()'s.
  const std::vector<float> logits = {1.0f, 0.2f, 2.0f, -0.5f, 1.4f};
  SamplingParams sp;
  sp.temperature = 1.0f;
  sp.top_k = 4;  // token 3 is outside the kept set
  sp.seed = 99;
  sp.has_seed = true;
  // Reference distribution: top-4 softmax.
  std::vector<double> want(logits.size());
  double z = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (i != 3) z += want[i] = std::exp(logits[i]);
  }
  for (double& w : want) w /= z;
  for (TokenId draft : {2, 1, 3}) {  // high-probability, low-probability, impossible draft
    Sampler s(sp);
    std::vector<double> freq(logits.size());
    constexpr int kN = 200000;
    int accepted_n = 0;
    for (int i = 0; i < kN; ++i) {
      std::vector<float> row = logits;
      bool accepted = false;
      const TokenId t = s.sample_speculative(row, {}, draft, accepted);
      freq[static_cast<size_t>(t)] += 1.0 / kN;
      accepted_n += accepted;
      EXPECT_EQ(accepted, t == draft);
    }
    for (size_t i = 0; i < logits.size(); ++i) EXPECT_NEAR(freq[i], want[i], 0.005) << "draft " << draft << " token " << i;
    EXPECT_NEAR(static_cast<double>(accepted_n) / kN, want[static_cast<size_t>(draft)], 0.005);
  }
}

TEST(Speculative, SampledGenerationIsSeededAndValid) {
  Loaded target = load(data("tiny_llama.gguf"));
  NgramDrafter ngram;
  SamplingParams sp;
  sp.temperature = 0.9f;
  sp.top_p = 0.95f;
  sp.seed = 5;
  sp.has_seed = true;
  SpeculativeStats st;
  const auto a = speculative(target, ngram, kPrompt, 30, 4, &st, sp);
  const auto b = speculative(target, ngram, kPrompt, 30, 4, &st, sp);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a.size(), 30u);
  for (TokenId t : a) {
    EXPECT_GE(t, 0);
    EXPECT_LT(t, target.model->config.vocab_size);
  }
}

TEST(Speculative, RejectsIncompatibleDraftVocabulary) {
  Loaded target = load(data("tiny_llama.gguf"));
  Loaded draft = load(data("tiny_llama.gguf"));
  // Same size, different text for one token: build a tokenizer variant.
  TokenizerData td;
  td.model = TokenizerData::Model::kBpe;
  for (TokenId i = 0; i < target.model->tokenizer->vocab_size(); ++i) {
    td.tokens.push_back(target.model->tokenizer->token_text(i));
  }
  td.tokens[42] = "different";
  auto other_tok = Tokenizer::create(td);
  ASSERT_TRUE(other_tok.ok());
  EXPECT_FALSE(ModelDrafter::create(*draft.tf, *draft.kv, **other_tok, *target.model->tokenizer).ok());
}

}  // namespace
}  // namespace engine

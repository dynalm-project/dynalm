// Phase 11: several sequences in one forward pass must produce exactly the
// same results as running each sequence alone (rows are independent).

#include <gtest/gtest.h>

#include <numeric>
#include <random>

#include "backends/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "model/transformer.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

class Batching : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    auto m = load_model(data("tiny_" + GetParam() + ".gguf"));
    ASSERT_TRUE(m.ok()) << m.status().to_string();
    model = std::move(*m);
    const ModelConfig& c = model->config;
    auto p = KvBlockPool::create(KvGeometry{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 4, 128,
                                            DType::kF32},
                                 be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
    auto t = Transformer::create(c, model->weights, be, 64);
    ASSERT_TRUE(t.ok());
    tf = std::move(*t);
  }

  std::vector<TokenId> random_tokens(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> v(n);
    for (auto& t : v) t = static_cast<TokenId>(rng() % 256);
    return v;
  }

  ThreadPool tp{3};
  CpuBackend be{tp, CpuIsa::kGeneric};
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<Transformer> tf;
};

TEST_P(Batching, MixedBatchEqualsSequentialBitExact) {
  const int32_t vocab = static_cast<int32_t>(model->config.vocab_size);
  // Four sequences with different histories, then one mixed step: two decode
  // rows, one prefill chunk of 5 and one prefill chunk of 3 without logits.
  struct Case {
    std::vector<TokenId> history, step;
    bool want_logits;
  };
  std::vector<Case> cases = {{random_tokens(9, 1), random_tokens(1, 11), true},
                             {random_tokens(3, 2), random_tokens(1, 12), true},
                             {random_tokens(6, 3), random_tokens(5, 13), true},
                             {random_tokens(1, 4), random_tokens(3, 14), false}};

  auto run_history = [&](KvBlockTable& t, const Case& cs) {
    ASSERT_TRUE(t.reserve(static_cast<int64_t>(cs.history.size() + cs.step.size())).ok());
    std::vector<float> scratch(static_cast<size_t>(vocab));
    std::vector<int32_t> pos(cs.history.size());
    std::iota(pos.begin(), pos.end(), 0);
    ASSERT_TRUE(tf->forward(cs.history, pos, *pool, t.block_table(), scratch).ok());
  };

  // Reference: each sequence alone.
  std::vector<std::vector<float>> want;
  for (const Case& cs : cases) {
    KvBlockTable t(*pool);
    run_history(t, cs);
    std::vector<float> l(static_cast<size_t>(vocab));
    std::vector<int32_t> pos(cs.step.size());
    std::iota(pos.begin(), pos.end(), static_cast<int32_t>(cs.history.size()));
    ASSERT_TRUE(tf->forward(cs.step, pos, *pool, t.block_table(), l).ok());
    if (cs.want_logits) want.push_back(l);
  }

  // Batched: same histories (run separately), then one batched step.
  std::vector<KvBlockTable> tables;
  for (const Case& cs : cases) {
    tables.emplace_back(*pool);
    run_history(tables.back(), cs);
  }
  std::vector<SeqBatch> batch;
  for (size_t i = 0; i < cases.size(); ++i) {
    batch.push_back({cases[i].step, static_cast<int32_t>(cases[i].history.size()), tables[i].block_table(),
                     cases[i].want_logits});
  }
  std::vector<float> got(want.size() * static_cast<size_t>(vocab));
  ASSERT_TRUE(tf->forward_batch(batch, *pool, got).ok());
  for (size_t s = 0; s < want.size(); ++s) {
    for (int32_t i = 0; i < vocab; ++i) {
      ASSERT_EQ(got[s * static_cast<size_t>(vocab) + static_cast<size_t>(i)], want[s][static_cast<size_t>(i)])
          << GetParam() << " seq " << s << " logit " << i;
    }
  }
}

TEST_P(Batching, ValidatesBatch) {
  KvBlockTable t(*pool);
  ASSERT_TRUE(t.reserve(8).ok());
  const TokenId toks[] = {1, 2};
  std::vector<float> l(static_cast<size_t>(model->config.vocab_size));
  // Logits buffer sized for the wrong number of rows.
  const SeqBatch two[] = {{toks, 0, t.block_table(), true}, {toks, 2, t.block_table(), true}};
  EXPECT_EQ(tf->forward_batch(two, *pool, l).code(), StatusCode::kInvalidArgument);
  // Position without a block.
  const SeqBatch far[] = {{toks, 100, t.block_table(), true}};
  EXPECT_EQ(tf->forward_batch(far, *pool, l).code(), StatusCode::kInvalidArgument);
  // Too many tokens for max_batch_tokens (64).
  std::vector<TokenId> big(65, 1);
  KvBlockTable u(*pool);
  ASSERT_TRUE(u.reserve(65).ok());
  const SeqBatch huge[] = {{big, 0, u.block_table(), true}};
  EXPECT_EQ(tf->forward_batch(huge, *pool, l).code(), StatusCode::kInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(Arch, Batching, ::testing::Values("llama", "gemma3", "phi3"),
                         [](const ::testing::TestParamInfo<std::string>& p) { return p.param; });

}  // namespace
}  // namespace engine

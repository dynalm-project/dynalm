#include "runtime/sequence.h"

#include <gtest/gtest.h>

#include "backends/cpu/cpu_backend.h"

namespace engine {
namespace {

TokenizerData tiny_vocab() {
  TokenizerData d;
  d.model = TokenizerData::Model::kBpe;
  for (int i = 0; i < 10; ++i) d.tokens.push_back("t" + std::to_string(i));
  d.types.assign(10, TokenType::kNormal);
  d.types[9] = TokenType::kControl;
  d.eos = 9;
  return d;
}

class SequenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto cache = KvBlockPool::create(KvGeometry{1, 1, 4, 4, 4, 8, DType::kF32}, backend);
    ASSERT_TRUE(cache.ok());
    cache_ = std::move(*cache);
    auto tok = Tokenizer::create(tiny_vocab());
    ASSERT_TRUE(tok.ok()) << tok.status().to_string();
    tok_ = std::move(*tok);
  }
  ThreadPool pool{1};
  CpuBackend backend{pool, CpuIsa::kGeneric};
  std::unique_ptr<KvBlockPool> cache_;
  std::unique_ptr<Tokenizer> tok_;
};

TEST_F(SequenceTest, PrefillDecodeFinishByLength) {
  const TokenId prompt[] = {1, 2, 3, 4, 5, 6};
  SequenceState s(7, prompt, StopParams{2, true}, *cache_);
  EXPECT_EQ(s.status(), SequenceStatus::kWaiting);
  EXPECT_EQ(s.pending(), 6);

  ASSERT_TRUE(s.reserve_kv(4).ok());
  s.mark_computed(4);  // chunked prefill
  EXPECT_EQ(s.status(), SequenceStatus::kPrefill);
  ASSERT_TRUE(s.reserve_kv(2).ok());
  s.mark_computed(2);
  EXPECT_EQ(s.status(), SequenceStatus::kDecode);
  EXPECT_EQ(s.block_table().size(), 2u);  // 6 tokens / block size 4
  EXPECT_EQ(cache_->free_blocks(), 6);

  s.append_token(1, *tok_);
  EXPECT_EQ(s.pending(), 1);
  ASSERT_TRUE(s.reserve_kv(1).ok());
  s.mark_computed(1);
  s.append_token(2, *tok_);  // second generated token hits max_new_tokens
  EXPECT_EQ(s.status(), SequenceStatus::kFinished);
  EXPECT_EQ(s.finish_reason(), FinishReason::kLength);
  EXPECT_EQ(s.generated(), 2);
  EXPECT_EQ(cache_->free_blocks(), 8);  // released on finish
}

TEST_F(SequenceTest, FinishOnEogReleasesKv) {
  const TokenId prompt[] = {1};
  SequenceState s(1, prompt, StopParams{100, true}, *cache_);
  ASSERT_TRUE(s.reserve_kv(1).ok());
  s.mark_computed(1);
  s.append_token(9, *tok_);  // EOS
  EXPECT_EQ(s.finish_reason(), FinishReason::kEog);
  EXPECT_EQ(cache_->free_blocks(), 8);
}

TEST_F(SequenceTest, EogIgnoredWhenDisabled) {
  const TokenId prompt[] = {1};
  SequenceState s(1, prompt, StopParams{100, false}, *cache_);
  ASSERT_TRUE(s.reserve_kv(1).ok());
  s.mark_computed(1);
  s.append_token(9, *tok_);
  EXPECT_EQ(s.status(), SequenceStatus::kDecode);
}

TEST_F(SequenceTest, CancelAndFailReleaseKv) {
  const TokenId prompt[] = {1, 2, 3, 4, 5};
  {
    SequenceState s(1, prompt, StopParams{}, *cache_);
    ASSERT_TRUE(s.reserve_kv(5).ok());
    EXPECT_EQ(cache_->free_blocks(), 6);
    s.cancel();
    EXPECT_EQ(s.status(), SequenceStatus::kCancelled);
    EXPECT_EQ(cache_->free_blocks(), 8);
    s.cancel();  // idempotent
  }
  {
    SequenceState s(2, prompt, StopParams{}, *cache_);
    ASSERT_TRUE(s.reserve_kv(5).ok());
    s.fail(Internal("boom"));
    EXPECT_EQ(s.status(), SequenceStatus::kError);
    EXPECT_EQ(s.error().code(), StatusCode::kInternal);
    EXPECT_EQ(cache_->free_blocks(), 8);
  }
}

TEST_F(SequenceTest, KvExhaustionIsReported) {
  std::vector<TokenId> prompt(40, 1);  // 40 tokens > 8 blocks * 4
  SequenceState s(1, prompt, StopParams{}, *cache_);
  EXPECT_EQ(s.reserve_kv(40).code(), StatusCode::kResourceExhausted);
}

TEST(KvSizing, GeometryAndBudget) {
  ModelConfig c;
  c.num_layers = 30;
  c.num_kv_heads = 3;
  c.head_dim = c.head_dim_v = 64;
  const KvGeometry g = kv_geometry_for(c, DType::kF16, 16, 1000);
  EXPECT_EQ(g.num_blocks, 63);  // ceil(1000 / 16)
  EXPECT_EQ(g.total_bytes(), c.kv_bytes_per_token(DType::kF16) * 63 * 16);
  EXPECT_EQ(kv_tokens_for_budget(c, DType::kF16, 23040 * 100), 100);  // 23040 B/token
}

}  // namespace
}  // namespace engine

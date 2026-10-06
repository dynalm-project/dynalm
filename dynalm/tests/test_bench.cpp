// Phase 21: benchmark framework (load generator) correctness.

#include "bench/loadgen.h"

#include <gtest/gtest.h>
#include "common/core.h"

namespace dynalm::bench {
namespace {

TEST(Bench, PercentilesNearestRank) {
  std::vector<double> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  const Percentiles p = percentiles(v);
  EXPECT_EQ(p.p50, 50);
  EXPECT_EQ(p.p90, 90);
  EXPECT_EQ(p.p95, 95);
  EXPECT_EQ(p.p99, 99);
  EXPECT_DOUBLE_EQ(p.mean, 50.5);
  EXPECT_EQ(percentiles({}).p99, 0);
  EXPECT_EQ(percentiles({7}).p50, 7);
}

class BenchEngine : public ::testing::Test {
 protected:
  void SetUp() override {
    EngineOptions o;
    o.model_path = std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf";
    o.threads = 2;
    o.kv_tokens = 4096;
    o.max_batch_tokens = 64;
    auto e = Engine::create(o);
    ASSERT_TRUE(e.ok());
    eng = std::move(*e);
  }
  std::unique_ptr<Engine> eng;
};

TEST_F(BenchEngine, PromptsHaveExactTokenLengthAndUniquePrefixes) {
  const Tokenizer& tok = *eng->model().tokenizer;
  for (int32_t n : {16, 50, 100}) {
    const std::string a = make_prompt(tok, n, 0), b = make_prompt(tok, n, 1);
    const auto ids = tok.encode(a, /*add_special=*/false, false);
    EXPECT_EQ(static_cast<int32_t>(ids.size()), n);
    EXPECT_NE(a.substr(0, 10), b.substr(0, 10));  // no shared prefix across requests
  }
}

TEST_F(BenchEngine, RunPointMeasuresEveryRequest) {
  auto target = make_engine_target(*eng);
  PointConfig cfg;
  cfg.concurrency = 3;
  cfg.prompt_tokens = 20;
  cfg.output_tokens = 12;
  cfg.requests = 6;
  const PointResult r = run_point(*target, *eng->model().tokenizer, cfg);
  EXPECT_EQ(r.completed, 6);
  EXPECT_EQ(r.errors, 0);
  EXPECT_DOUBLE_EQ(r.mean_completion_tokens, 12);  // ignore EOS: exact output length
  EXPECT_GT(r.output_tok_s, 0);
  EXPECT_GT(r.ttft_ms.p50, 0);
  EXPECT_LE(r.ttft_ms.p50, r.ttft_ms.p99);
  EXPECT_GT(r.e2e_ms.p50, 0);
  EXPECT_GT(r.rss_mb, 0);
  const std::string j = to_json(r, "tiny", "test");
  EXPECT_NE(j.find("\"ttft_ms\":{"), std::string::npos);
  EXPECT_NE(j.find("\"concurrency\":3"), std::string::npos);
}

TEST_F(BenchEngine, PointsDoNotReuseEachOthersPrefixes) {
  auto target = make_engine_target(*eng);
  PointConfig cfg;
  cfg.concurrency = 2;
  cfg.prompt_tokens = 64;  // 4 full KV blocks each
  cfg.output_tokens = 2;
  cfg.requests = 4;
  (void)run_point(*target, *eng->model().tokenizer, cfg);
  const uint64_t hits = eng->stats().prefix.hit_tokens;
  (void)run_point(*target, *eng->model().tokenizer, cfg);
  // Only the short shared "Request " lead-in may match; never a full prompt block.
  EXPECT_LT(eng->stats().prefix.hit_tokens - hits, 16u * 5);
}

}  // namespace
}  // namespace dynalm::bench

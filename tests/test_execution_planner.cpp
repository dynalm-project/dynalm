// Performance program P2: the execution planner (DD-051). The planner owns
// every per-step execution decision; with it the engine must produce exactly
// the results it produced before, and each strategy it can choose must be
// numerically equivalent.

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "backends/cpu/cpu_backend.h"
#include "execution/batch_planner.h"
#include "loader/model_loader.h"
#include "model/transformer.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

ModelConfig config(int32_t heads, int32_t layers, int32_t window = 0) {
  ModelConfig c;
  c.num_heads = heads;
  c.num_layers = layers;
  c.sliding_window = window;
  if (window > 0) {
    for (int l = 0; l < layers; ++l) c.sliding_layers.push_back(l % 2 == 0);  // alternate window / full
  }
  return c;
}

HardwareProfile hw(int32_t threads) {
  HardwareProfile h;
  h.threads = threads;
  return h;
}

TEST(BatchPlannerTest, PhaseAndShape) {
  const BatchPlanner planner(config(8, 2), hw(8));
  const std::vector<TokenId> one = {5}, five = {1, 2, 3, 4, 5};
  const std::vector<int32_t> table = {0};

  const SeqBatch d1{one, 100, table, true}, d2{one, 7, table, true};
  ExecutionPlan p = planner.plan(std::vector<SeqBatch>{d1, d2});
  EXPECT_EQ(p.phase, StepPhase::kDecode);
  EXPECT_EQ(p.rows, 2);
  EXPECT_EQ(p.decode_rows, 2);
  EXPECT_EQ(p.logit_rows, 2);
  EXPECT_EQ(p.max_context, 101);
  EXPECT_EQ(p.attention_work, 101 + 8);

  const SeqBatch pf{five, 10, table, false};
  p = planner.plan(std::vector<SeqBatch>{pf});
  EXPECT_EQ(p.phase, StepPhase::kPrefill);
  EXPECT_EQ(p.prefill_rows, 5);
  EXPECT_EQ(p.logit_rows, 0);
  EXPECT_EQ(p.attention_work, 11 + 12 + 13 + 14 + 15);

  p = planner.plan(std::vector<SeqBatch>{d1, pf});
  EXPECT_EQ(p.phase, StepPhase::kMixed);
  EXPECT_EQ(p.sequences, 2);
}

TEST(BatchPlannerTest, AttentionSplitsOnlyWhenPairsCannotFillTheThreads) {
  const BatchPlanner planner(config(4, 2), hw(10));
  const std::vector<TokenId> one = {5};
  const std::vector<int32_t> table = {0};
  // One decode row x 4 heads = 4 pairs < 20, context 2000 > 512: split.
  ExecutionPlan p = planner.plan(std::vector<SeqBatch>{SeqBatch{one, 1999, table, true}});
  EXPECT_EQ(p.kernels.attention_full, AttentionStrategy::kSplitK);
  // Short context: the merge would cost more than it saves.
  p = planner.plan(std::vector<SeqBatch>{SeqBatch{one, 100, table, true}});
  EXPECT_EQ(p.kernels.attention_full, AttentionStrategy::kPerPair);
  // Eight rows = 32 pairs >= 20: per-pair tasks already fill the pool.
  std::vector<SeqBatch> eight(8, SeqBatch{one, 1999, table, true});
  p = planner.plan(eight);
  EXPECT_EQ(p.kernels.attention_full, AttentionStrategy::kPerPair);
}

TEST(BatchPlannerTest, SlidingWindowLayersGetTheirOwnDecision) {
  const BatchPlanner planner(config(4, 4, /*window=*/256), hw(10));
  const std::vector<TokenId> one = {5};
  const std::vector<int32_t> table = {0};
  const ExecutionPlan p = planner.plan(std::vector<SeqBatch>{SeqBatch{one, 4000, table, true}});
  EXPECT_EQ(p.kernels.attention_full, AttentionStrategy::kSplitK);      // 4001 tokens of context
  EXPECT_EQ(p.kernels.attention_window, AttentionStrategy::kPerPair);   // capped at 256
}

TEST(BatchPlannerTest, KeepsTheBaseKernelSettings) {
  KernelPlan base;
  base.expand_min_rows = 5;
  base.gemm_k_block = 512;
  const BatchPlanner planner(config(4, 2), hw(4), base);
  const std::vector<TokenId> one = {5};
  const std::vector<int32_t> table = {0};
  const ExecutionPlan p = planner.plan(std::vector<SeqBatch>{SeqBatch{one, 3, table, true}});
  EXPECT_EQ(p.kernels.expand_min_rows, 5);
  EXPECT_EQ(p.kernels.gemm_k_block, 512);
}

// --- with a real model ---------------------------------------------------------------

class PlannedForward : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    auto m = load_model(data("tiny_" + GetParam() + ".gguf"));
    ASSERT_TRUE(m.ok()) << m.status().to_string();
    model = std::move(*m);
    const ModelConfig& c = model->config;
    auto p = KvBlockPool::create(KvGeometry{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 128,
                                            DType::kF32},
                                 be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
    auto t = Transformer::create(c, model->weights, be, 64);
    ASSERT_TRUE(t.ok());
    tf = std::move(*t);
  }

  // Prefills `n` random tokens in chunks, then decodes one token, returning
  // its logits computed with `plan_override` applied to the planner's plan.
  std::vector<float> run(int32_t n, void (*plan_override)(ExecutionPlan&)) {
    std::mt19937 rng(7);
    std::vector<TokenId> toks(static_cast<size_t>(n) + 1);
    for (auto& t : toks) t = static_cast<TokenId>(rng() % 200);
    KvBlockTable table(*pool);
    EXPECT_TRUE(table.reserve(n + 1).ok());
    const auto vocab = static_cast<size_t>(model->config.vocab_size);
    std::vector<float> logits(vocab);
    for (int32_t s = 0; s < n; s += 64) {
      const int32_t len = std::min(64, n - s);
      const SeqBatch b{std::span<const TokenId>(toks).subspan(static_cast<size_t>(s), static_cast<size_t>(len)), s,
                       table.block_table(), false};
      EXPECT_TRUE(tf->forward_batch({&b, 1}, *pool, {}).ok());
    }
    const SeqBatch d{std::span<const TokenId>(toks).subspan(static_cast<size_t>(n), 1), n, table.block_table(), true};
    ExecutionPlan plan = tf->planner().plan({&d, 1});
    plan_override(plan);
    EXPECT_TRUE(tf->forward_batch({&d, 1}, *pool, logits, &plan).ok());
    return logits;
  }

  ThreadPool tp{4};
  CpuBackend be{tp, CpuIsa::kGeneric};
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<Transformer> tf;
};

TEST_P(PlannedForward, PlannerReproducesThePrePlannerRulesExactly) {
  // 1500 tokens of context with one decode row: the planner chooses split-K,
  // exactly what the backend's per-call rule (kAuto) chooses.
  const auto planned = run(1500, [](ExecutionPlan&) {});
  const auto automatic = run(1500, [](ExecutionPlan& p) {
    p.kernels.attention_full = p.kernels.attention_window = AttentionStrategy::kAuto;
  });
  ASSERT_EQ(planned.size(), automatic.size());
  for (size_t i = 0; i < planned.size(); ++i) ASSERT_EQ(planned[i], automatic[i]) << i;  // bit-identical
}

TEST_P(PlannedForward, ForcedStrategiesAgree) {
  const auto split = run(1500, [](ExecutionPlan& p) {
    p.kernels.attention_full = p.kernels.attention_window = AttentionStrategy::kSplitK;
  });
  const auto per_pair = run(1500, [](ExecutionPlan& p) {
    p.kernels.attention_full = p.kernels.attention_window = AttentionStrategy::kPerPair;
  });
  double max_diff = 0;
  for (size_t i = 0; i < split.size(); ++i) max_diff = std::max(max_diff, std::fabs(double{split[i]} - per_pair[i]));
  EXPECT_LT(max_diff, 1e-3);  // log-sum-exp merge vs single pass: float rounding only
}

TEST_P(PlannedForward, ExpandThresholdIsHonored) {
  // Forcing the expand (GEMM) path for a single row must match the fused path.
  const auto fused = run(40, [](ExecutionPlan& p) { p.kernels.expand_min_rows = 2; });
  const auto expanded = run(40, [](ExecutionPlan& p) { p.kernels.expand_min_rows = 1; });
  double max_diff = 0;
  for (size_t i = 0; i < fused.size(); ++i) max_diff = std::max(max_diff, std::fabs(double{fused[i]} - expanded[i]));
  EXPECT_LT(max_diff, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(Arch, PlannedForward, ::testing::Values("llama", "gemma2", "qwen2"),
                         [](const auto& pi) { return pi.param; });

}  // namespace
}  // namespace engine

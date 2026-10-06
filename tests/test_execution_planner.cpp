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
#include "dtype/fp16.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

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

// P5 (DD-054): GQA-grouped attention for one decode row (the case where
// (row, KV head) units are fewer than threads, so heads are split into
// sub-groups) against a naive per-head softmax, for every strategy.
TEST(GqaAttention, OneRowMatchesNaiveInEveryLayout) {
  for (DType kvt : {DType::kF32, DType::kF16}) {
    ThreadPool tp(8);
    CpuBackend be(tp, select_best_isa(cpu_info().features));
    KvGeometry g;
    g.num_layers = 1;
    g.num_kv_heads = 2;
    g.head_dim = 16;
    g.head_dim_v = 16;
    g.block_size = 16;
    g.num_blocks = 80;  // 1280 tokens
    g.dtype = kvt;
    auto cache = KvBlockPool::create(g, be);
    ASSERT_TRUE(cache.ok());
    std::vector<int32_t> table(80);
    for (int32_t i = 0; i < 80; ++i) table[static_cast<size_t>(i)] = (i * 13) % 80;
    const KvLayerView kv = (*cache)->layer_view(0, table);
    const int64_t T = 1100, H = 14;  // 7 query heads per KV head
    std::mt19937 rng(4);
    std::normal_distribution<float> nd(0, 1);
    auto fill = [&](int64_t r, int64_t c) {
      auto t = Tensor::empty(DType::kF32, {r, c});
      for (int64_t i = 0; i < r * c; ++i) t->data_as<float>()[i] = nd(rng);
      return std::move(*t);
    };
    Tensor k = fill(T, 32), v = fill(T, 32);
    std::vector<int32_t> pos(T), row_seq(T, 0);
    for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    const KvLayerView views[] = {kv};
    be.kv_store(k, v, pos, row_seq, views);
    // KV as stored (fp16-rounded when kvt is F16), for the reference.
    auto stored = [&](const Tensor& src, int64_t t, int64_t c) {
      const float x = src.data_as<const float>()[t * 32 + c];
      return kvt == DType::kF16 ? fp16_to_fp32(fp32_to_fp16(x)) : x;
    };
    Tensor q = fill(1, H * 16);
    const int32_t qpos[] = {static_cast<int32_t>(T - 1)};
    const int32_t qseq[] = {0};
    // Naive reference.
    std::vector<double> ref(static_cast<size_t>(H * 16));
    for (int64_t h = 0; h < H; ++h) {
      const int64_t kvh = h / 7;
      std::vector<double> s(static_cast<size_t>(T));
      double mx = -1e300;
      for (int64_t t = 0; t < T; ++t) {
        double d = 0;
        for (int64_t c = 0; c < 16; ++c) d += double{q.data_as<float>()[h * 16 + c]} * stored(k, t, kvh * 16 + c);
        s[static_cast<size_t>(t)] = d * 0.25;
        mx = std::max(mx, s[static_cast<size_t>(t)]);
      }
      double z = 0;
      for (int64_t t = 0; t < T; ++t) z += std::exp(s[static_cast<size_t>(t)] - mx);
      for (int64_t c = 0; c < 16; ++c) {
        double a = 0;
        for (int64_t t = 0; t < T; ++t) a += std::exp(s[static_cast<size_t>(t)] - mx) / z * stored(v, t, kvh * 16 + c);
        ref[static_cast<size_t>(h * 16 + c)] = a;
      }
    }
    std::vector<std::vector<float>> results;
    for (AttentionStrategy st : {AttentionStrategy::kPerPair, AttentionStrategy::kSplitK, AttentionStrategy::kAuto}) {
      KernelPlan kp = KernelPlan::defaults();
      kp.attention_full = st;
      be.set_kernel_plan(kp);
      auto out = Tensor::zeros(DType::kF32, {1, H * 16});
      AttentionParams ap;
      ap.q = q;
      ap.out = *out;
      ap.positions = qpos;
      ap.row_seq = qseq;
      ap.kv = views;
      ap.num_heads = static_cast<int32_t>(H);
      ap.scale = 0.25f;
      be.attention(ap);
      for (int64_t i = 0; i < H * 16; ++i) {
        ASSERT_NEAR(out->data_as<float>()[i], ref[static_cast<size_t>(i)], 1e-4)
            << dtype_name(kvt) << " strategy " << attention_strategy_name(st) << " i=" << i;
      }
      results.emplace_back(out->data_as<float>(), out->data_as<float>() + H * 16);
    }
    // kAuto picks split-K here (1 row, 1100 tokens); per-pair (sub-grouped)
    // and split-K agree to float rounding, as above.
    EXPECT_EQ(results[1], results[2]);
  }
}

// DD-066: grouped GQA kernels (all heads of a KV head per K/V load) against
// the per-head path and a naive double-precision softmax: f32/f16 KV, head
// dims that are and are not multiples of 8, group sizes 1..9 (9 = two passes
// of at most 8 heads), three decode rows from two sequences with scrambled
// block tables, on the best ISA and on the generic kernels.
TEST(GqaAttention, GroupedMatchesPerHeadAndNaive) {
  for (CpuIsa isa : {select_best_isa(cpu_info().features), CpuIsa::kGeneric}) {
    for (DType kvt : {DType::kF32, DType::kF16}) {
      for (int32_t hd : {64, 128, 72}) {
        for (int32_t group : {1, 2, 3, 4, 6, 7, 8, 9}) {
          ThreadPool tp(4);
          CpuBackend be(tp, isa);
          constexpr int32_t kNkv = 2, kBlocks = 80, kBs = 16;
          KvGeometry g;
          g.num_layers = 1;
          g.num_kv_heads = kNkv;
          g.head_dim = hd;
          g.head_dim_v = hd;
          g.block_size = kBs;
          g.num_blocks = kBlocks;
          g.dtype = kvt;
          auto cache = KvBlockPool::create(g, be);
          ASSERT_TRUE(cache.ok());
          // Two sequences, 40 blocks each, interleaved and scrambled.
          std::vector<int32_t> t0(40), t1(40);
          for (int32_t i = 0; i < 40; ++i) {
            t0[static_cast<size_t>(i)] = (i * 2 * 13) % kBlocks;
            t1[static_cast<size_t>(i)] = (i * 2 * 13 + 1) % kBlocks;
          }
          const KvLayerView views[] = {(*cache)->layer_view(0, t0), (*cache)->layer_view(0, t1)};
          const int64_t T = 600, H = static_cast<int64_t>(group) * kNkv, W = kNkv * hd;
          std::mt19937 rng(static_cast<unsigned>(hd * 31 + group));
          std::normal_distribution<float> nd(0, 1);
          auto fill = [&](int64_t r, int64_t c) {
            auto t = Tensor::empty(DType::kF32, {r, c});
            for (int64_t i = 0; i < r * c; ++i) t->data_as<float>()[i] = nd(rng);
            return std::move(*t);
          };
          // T positions of K/V for each sequence.
          Tensor k0 = fill(T, W), v0 = fill(T, W), k1 = fill(T, W), v1 = fill(T, W);
          std::vector<int32_t> pos(static_cast<size_t>(T)), seq0(static_cast<size_t>(T), 0), seq1(static_cast<size_t>(T), 1);
          for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
          be.kv_store(k0, v0, pos, seq0, views);
          be.kv_store(k1, v1, pos, seq1, views);
          auto stored = [&](const Tensor& src, int64_t t, int64_t c) {
            const float x = src.data_as<const float>()[t * W + c];
            return kvt == DType::kF16 ? fp16_to_fp32(fp32_to_fp16(x)) : x;
          };
          // Rows: seq 0 at 299, seq 1 at 599, seq 0 at 517 (causal ranges differ).
          const int32_t qpos[] = {299, 599, 517};
          const int32_t qseq[] = {0, 1, 0};
          Tensor q = fill(3, H * hd);
          const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
          std::vector<double> ref(static_cast<size_t>(3 * H * hd));
          for (int64_t r = 0; r < 3; ++r) {
            const Tensor& kk = qseq[r] == 0 ? k0 : k1;
            const Tensor& vv = qseq[r] == 0 ? v0 : v1;
            const int64_t n = qpos[r] + 1;
            for (int64_t h = 0; h < H; ++h) {
              const int64_t kvh = h / group;
              std::vector<double> s(static_cast<size_t>(n));
              double mx = -1e300;
              for (int64_t t = 0; t < n; ++t) {
                double d = 0;
                for (int64_t c = 0; c < hd; ++c) {
                  d += double{q.data_as<float>()[r * H * hd + h * hd + c]} * stored(kk, t, kvh * hd + c);
                }
                s[static_cast<size_t>(t)] = d * scale;
                mx = std::max(mx, s[static_cast<size_t>(t)]);
              }
              double z = 0;
              for (int64_t t = 0; t < n; ++t) z += std::exp(s[static_cast<size_t>(t)] - mx);
              for (int64_t c = 0; c < hd; ++c) {
                double a = 0;
                for (int64_t t = 0; t < n; ++t) a += std::exp(s[static_cast<size_t>(t)] - mx) / z * stored(vv, t, kvh * hd + c);
                ref[static_cast<size_t>(r * H * hd + h * hd + c)] = a;
              }
            }
          }
          std::vector<float> outs[2];
          for (int grouped = 0; grouped < 2; ++grouped) {
            KernelPlan kp = KernelPlan::defaults();
            kp.grouped_attention = grouped == 1;
            kp.attention_full = AttentionStrategy::kPerPair;
            be.set_kernel_plan(kp);
            auto out = Tensor::zeros(DType::kF32, {3, H * hd});
            AttentionParams ap;
            ap.q = q;
            ap.out = *out;
            ap.positions = qpos;
            ap.row_seq = qseq;
            ap.kv = views;
            ap.num_heads = static_cast<int32_t>(H);
            ap.scale = scale;
            be.attention(ap);
            outs[grouped].assign(out->data_as<float>(), out->data_as<float>() + 3 * H * hd);
            for (int64_t i = 0; i < 3 * H * hd; ++i) {
              ASSERT_NEAR(outs[grouped][static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4)
                  << isa_name(isa) << " " << dtype_name(kvt) << " hd=" << hd << " group=" << group
                  << " grouped=" << grouped << " i=" << i;
            }
          }
          double max_diff = 0;
          for (size_t i = 0; i < outs[0].size(); ++i) max_diff = std::max(max_diff, std::fabs(double{outs[0][i]} - outs[1][i]));
          EXPECT_LT(max_diff, 1e-5) << isa_name(isa) << " " << dtype_name(kvt) << " hd=" << hd << " group=" << group;
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Arch, PlannedForward, ::testing::Values("llama", "gemma2", "qwen2"),
                         [](const auto& pi) { return pi.param; });

}  // namespace
}  // namespace engine

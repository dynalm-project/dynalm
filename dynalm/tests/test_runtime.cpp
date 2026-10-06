#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/tensor/fp16.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/architecture.h"
#include "model/transformer.h"
#include "runtime/generator.h"
#include "dynacore/execution/thread_pool.h"
#include "sampling/sampler.h"
#include "test_models.h"
#include "common/core.h"

namespace dynalm {
namespace {

// ---------------------------------------------------------------------------
// ThreadPool

TEST(ThreadPool, CoversEveryIndexExactlyOnce) {
  ThreadPool pool(4);
  for (size_t n : {1u, 7u, 64u, 1000u, 12345u}) {
    std::vector<std::atomic<int>> hits(n);
    pool.parallel_for(n, 3, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) hits[i].fetch_add(1);
    });
    for (size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i].load(), 1) << "n=" << n << " i=" << i;
  }
}

TEST(ThreadPool, ManySmallLaunches) {
  ThreadPool pool(4);
  std::atomic<int64_t> sum{0};
  for (int it = 0; it < 5000; ++it) {
    pool.parallel_for(8, 1, [&](size_t b, size_t e) { sum.fetch_add(static_cast<int64_t>(e - b)); });
  }
  EXPECT_EQ(sum.load(), 5000 * 8);
}

TEST(ThreadPool, SingleThreadRunsInline) {
  ThreadPool pool(1);
  int calls = 0;
  pool.parallel_for(100, 10, [&](size_t b, size_t e) {
    ++calls;
    EXPECT_EQ(b, 0u);
    EXPECT_EQ(e, 100u);
  });
  EXPECT_EQ(calls, 1);
}

// ---------------------------------------------------------------------------
// CPU backend ops vs naive references

class CpuOps : public ::testing::Test {
 protected:
  ThreadPool pool{3};
  CpuDevice be{pool, CpuIsa::kGeneric};
  std::mt19937 rng{7};

  Tensor rand(DType t, TensorShape s) {
    auto f = Tensor::empty(DType::kF32, s);
    std::normal_distribution<float> nd(0, 1);
    for (int64_t i = 0; i < f->numel(); ++i) f->data_as<float>()[i] = nd(rng);
    if (t == DType::kF32) return *f;
    auto h = Tensor::empty(DType::kF16, s);
    for (int64_t i = 0; i < f->numel(); ++i) {
      h->data_as<uint16_t>()[i] = fp32_to_fp16(f->data_as<float>()[i]);
    }
    return *h;
  }
  static float at(const Tensor& t, int64_t i) {
    return t.dtype() == DType::kF16 ? fp16_to_fp32(t.data_as<uint16_t>()[i]) : t.data_as<float>()[i];
  }
};

TEST_F(CpuOps, MatmulMatchesNaive) {
  for (DType wt : {DType::kF32, DType::kF16}) {
    const int64_t m = 5, k = 37, n = 29;
    Tensor x = rand(DType::kF32, {m, k}), w = rand(wt, {n, k}), b = rand(DType::kF32, {n});
    auto y = Tensor::zeros(DType::kF32, {m, n});
    const TensorView bv = b;
    be.matmul(x, w, &bv, *y);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double ref = at(b, j);
        for (int64_t q = 0; q < k; ++q) ref += static_cast<double>(at(x, i * k + q)) * at(w, j * k + q);
        ASSERT_NEAR(y->data_as<float>()[i * n + j], ref, 1e-4) << dtype_name(wt);
      }
    }
  }
}

TEST_F(CpuOps, RmsNormAndInPlace) {
  Tensor x = rand(DType::kF32, {3, 16}), w = rand(DType::kF32, {16});
  auto y = Tensor::zeros(DType::kF32, {3, 16});
  be.rms_norm(x, w, 1e-5f, *y);
  for (int r = 0; r < 3; ++r) {
    double ss = 0;
    for (int i = 0; i < 16; ++i) ss += at(x, r * 16 + i) * at(x, r * 16 + i);
    const double inv = 1.0 / std::sqrt(ss / 16 + 1e-5);
    for (int i = 0; i < 16; ++i) ASSERT_NEAR(y->data_as<float>()[r * 16 + i], at(x, r * 16 + i) * inv * at(w, i), 1e-5);
  }
  be.rms_norm(x, w, 1e-5f, x);  // aliasing allowed
  for (int i = 0; i < 48; ++i) ASSERT_NEAR(x.data_as<float>()[i], y->data_as<float>()[i], 1e-6);
}

TEST_F(CpuOps, RopeStyles) {
  RopeConfig rc;
  rc.dim = 8;
  rc.freq_base = 10000;
  const int32_t pos[] = {0, 5};
  for (RopeStyle style : {RopeStyle::kInterleaved, RopeStyle::kHalfSplit}) {
    rc.style = style;
    Tensor x = rand(DType::kF32, {2, 16});  // 2 rows, 2 heads x 8
    auto orig = Tensor::empty(DType::kF32, {2, 16});
    std::memcpy(orig->data(), x.data(), 2 * 16 * 4);
    be.rope(x, 2, 8, pos, rc, nullptr);
    for (int r = 0; r < 2; ++r) {
      for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
          const double th = pos[r] * std::pow(10000.0, -2.0 * i / 8);
          const int a = style == RopeStyle::kInterleaved ? 2 * i : i;
          const int b = style == RopeStyle::kInterleaved ? 2 * i + 1 : i + 4;
          const float* o = orig->data_as<float>() + r * 16 + h * 8;
          const float* n = x.data_as<float>() + r * 16 + h * 8;
          ASSERT_NEAR(n[a], o[a] * std::cos(th) - o[b] * std::sin(th), 1e-5);
          ASSERT_NEAR(n[b], o[a] * std::sin(th) + o[b] * std::cos(th), 1e-5);
        }
      }
    }
  }
}

// Attention through a scrambled block table must equal a naive computation
// over contiguous K/V, for GQA, f32/f16 KV and sliding windows.
TEST_F(CpuOps, AttentionPagedMatchesNaive) {
  for (DType kvt : {DType::kF32, DType::kF16}) {
    for (int32_t window : {0, 3}) {
      KvGeometry g;
      g.num_layers = 1;
      g.num_kv_heads = 2;
      g.head_dim = 8;
      g.head_dim_v = 8;
      g.block_size = 4;
      g.num_blocks = 8;
      g.dtype = kvt;
      auto cache = KvBlockPool::create(g, be);
      ASSERT_TRUE(cache.ok());
      const int32_t table[] = {5, 2, 7};  // positions 0..11 scattered
      const KvLayerView kv = (*cache)->layer_view(0, table);

      const int64_t T = 10, H = 4;
      Tensor k = rand(DType::kF32, {T, 16}), v = rand(DType::kF32, {T, 16});
      std::vector<int32_t> pos(T);
      std::iota(pos.begin(), pos.end(), 0);
      const std::vector<int32_t> row_seq(T, 0);
      const KvLayerView views[] = {kv};
      be.kv_store(k, v, pos, row_seq, views);

      Tensor q = rand(DType::kF32, {T, H * 8});
      auto out = Tensor::zeros(DType::kF32, {T, H * 8});
      AttentionParams ap;
      ap.q = q;
      ap.out = *out;
      ap.positions = pos;
      ap.row_seq = row_seq;
      ap.kv = views;
      ap.num_heads = static_cast<int32_t>(H);
      ap.scale = 0.35f;
      ap.sliding_window = window;
      be.attention(ap);

      auto kvv = [&](const Tensor& t, int64_t tok, int64_t h, int i) {
        const float f = t.data_as<float>()[tok * 16 + h * 8 + i];
        return kvt == DType::kF16 ? fp16_to_fp32(fp32_to_fp16(f)) : f;
      };
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
          const int64_t kh = h / 2;
          const int64_t lo = window > 0 ? std::max<int64_t>(0, t - window + 1) : 0;
          std::vector<double> s;
          double mx = -1e30;
          for (int64_t u = lo; u <= t; ++u) {
            double d = 0;
            for (int i = 0; i < 8; ++i) d += q.data_as<float>()[t * 32 + h * 8 + i] * kvv(k, u, kh, i);
            s.push_back(d * 0.35);
            mx = std::max(mx, s.back());
          }
          double sum = 0;
          for (double& e : s) sum += (e = std::exp(e - mx));
          for (int i = 0; i < 8; ++i) {
            double ref = 0;
            for (int64_t u = lo; u <= t; ++u) ref += s[static_cast<size_t>(u - lo)] / sum * kvv(v, u, kh, i);
            ASSERT_NEAR(out->data_as<float>()[t * 32 + h * 8 + i], ref, 1e-4)
                << dtype_name(kvt) << " window=" << window;
          }
        }
      }
    }
  }
}

// Decode-shaped attention (few rows, long context) takes the split-K path:
// chunk partials merged with log-sum-exp must equal a single softmax.
TEST_F(CpuOps, AttentionSplitKMatchesNaive) {
  for (DType kvt : {DType::kF32, DType::kF16}) {
    for (int32_t window : {0, 600}) {
      KvGeometry g;
      g.num_layers = 1;
      g.num_kv_heads = 2;
      g.head_dim = 8;
      g.head_dim_v = 8;
      g.block_size = 16;
      g.num_blocks = 64;  // 1024 tokens
      g.dtype = kvt;
      auto cache = KvBlockPool::create(g, be);
      ASSERT_TRUE(cache.ok());
      std::vector<int32_t> table(64);
      for (int32_t i = 0; i < 64; ++i) table[static_cast<size_t>(i)] = (i * 37) % 64;  // scrambled
      const KvLayerView kv = (*cache)->layer_view(0, table);

      const int64_t T = 900, H = 4;
      Tensor k = rand(DType::kF32, {T, 16}), v = rand(DType::kF32, {T, 16});
      std::vector<int32_t> pos(T), seq0(T, 0);
      std::iota(pos.begin(), pos.end(), 0);
      const KvLayerView views[] = {kv};
      be.kv_store(k, v, pos, seq0, views);

      // Two query rows: one at position 899 (long), one at 40 (short: most
      // of its chunks are empty).
      const int32_t qpos[] = {899, 40};
      const int32_t qseq[] = {0, 0};
      Tensor q = rand(DType::kF32, {2, H * 8});
      auto out = Tensor::zeros(DType::kF32, {2, H * 8});
      AttentionParams ap;
      ap.q = q;
      ap.out = *out;
      ap.positions = qpos;
      ap.row_seq = qseq;
      ap.kv = views;
      ap.num_heads = static_cast<int32_t>(H);
      ap.scale = 0.35f;
      ap.sliding_window = window;
      be.attention(ap);

      auto kvv = [&](const Tensor& t, int64_t tok, int64_t h, int i) {
        const float f = t.data_as<float>()[tok * 16 + h * 8 + i];
        return kvt == DType::kF16 ? fp16_to_fp32(fp32_to_fp16(f)) : f;
      };
      for (int r = 0; r < 2; ++r) {
        const int64_t p = qpos[r];
        const int64_t lo = window > 0 ? std::max<int64_t>(0, p - window + 1) : 0;
        for (int64_t h = 0; h < H; ++h) {
          std::vector<double> s;
          double mx = -1e30;
          for (int64_t u = lo; u <= p; ++u) {
            double d = 0;
            for (int i = 0; i < 8; ++i) d += q.data_as<float>()[r * 32 + h * 8 + i] * kvv(k, u, h / 2, i);
            s.push_back(d * 0.35);
            mx = std::max(mx, s.back());
          }
          double sum = 0;
          for (double& e : s) sum += (e = std::exp(e - mx));
          for (int i = 0; i < 8; ++i) {
            double ref = 0;
            for (int64_t u = lo; u <= p; ++u) ref += s[static_cast<size_t>(u - lo)] / sum * kvv(v, u, h / 2, i);
            ASSERT_NEAR(out->data_as<float>()[r * 32 + h * 8 + i], ref, 2e-4)
                << dtype_name(kvt) << " window=" << window << " row " << r;
          }
        }
      }
    }
  }
}

TEST(Sampler, GreedyPicksFirstMax) {
  const float l[] = {0.1f, 3.0f, -1.0f, 3.0f};
  EXPECT_EQ(sample_greedy(l), 1);
}

TEST(KvBlockPool, BlockAccountingAndExhaustion) {
  ThreadPool pool(1);
  CpuDevice be(pool, CpuIsa::kGeneric);
  KvGeometry g{1, 1, 4, 4, 4, 3, DType::kF32};
  auto cache = KvBlockPool::create(g, be);
  ASSERT_TRUE(cache.ok());
  {
    KvBlockTable s(**cache);
    ASSERT_TRUE(s.reserve(9).ok());  // 3 blocks
    EXPECT_EQ((*cache)->free_blocks(), 0);
    KvBlockTable t(**cache);
    EXPECT_EQ(t.reserve(1).code(), StatusCode::kResourceExhausted);
  }
  EXPECT_EQ((*cache)->free_blocks(), 3);  // released on destruction
}

// ---------------------------------------------------------------------------
// Architecture validation

TEST(Architecture, Registry) {
  ASSERT_NE(find_architecture("llama"), nullptr);
  EXPECT_EQ(find_architecture("made-up"), nullptr);
}

// ---------------------------------------------------------------------------
// Golden: runtime vs the NumPy reference (tools/ref_llama.py)

struct RefData {
  std::vector<TokenId> tokens, greedy;
  std::vector<std::pair<TokenId, float>> top;
  double sum = 0, sumsq = 0;
};

RefData load_ref(const std::string& path) {
  RefData r;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "tokens:") {
      for (TokenId t; ss >> t;) r.tokens.push_back(t);
    } else if (key == "greedy:") {
      for (TokenId t; ss >> t;) r.greedy.push_back(t);
    } else if (key == "top:") {
      for (std::string p; ss >> p;) {
        const auto c = p.find(':');
        r.top.emplace_back(std::stoi(p.substr(0, c)), std::stof(p.substr(c + 1)));
      }
    } else if (key == "stats:") {
      ss >> r.sum >> r.sumsq;
    }
  }
  return r;
}

struct RealModel {
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<ThreadPool> pool;
  std::unique_ptr<CpuDevice> backend;
  std::unique_ptr<KvBlockPool> cache;
  std::unique_ptr<Transformer> transformer;
};

std::optional<RealModel> open_real(DType kv, int32_t batch, std::string path = dynalm::testing::smollm_model()) {
  if (!dynalm::testing::exists(path)) return std::nullopt;
  RealModel r;
  auto m = load_model(path);
  EXPECT_TRUE(m.ok()) << m.status().to_string();
  if (!m.ok()) return std::nullopt;
  r.model = std::move(*m);
  r.pool = std::make_unique<ThreadPool>(4);
  r.backend = std::make_unique<CpuDevice>(*r.pool, CpuIsa::kGeneric);
  const ModelConfig& c = r.model->config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 16, kv};
  auto cache = KvBlockPool::create(g, *r.backend);
  EXPECT_TRUE(cache.ok());
  r.cache = std::move(*cache);
  auto t = Transformer::create(c, r.model->weights, *r.backend, batch);
  EXPECT_TRUE(t.ok()) << t.status().to_string();
  if (!t.ok()) return std::nullopt;
  r.transformer = std::move(*t);
  return r;
}

TEST(RuntimeGolden, SmolLm2LogitsMatchReference) {
  const RefData ref = load_ref(std::string(ENGINE_TEST_DATA_DIR) + "/ref_smollm2.txt");
  ASSERT_FALSE(ref.tokens.empty());
  auto rm = open_real(DType::kF32, 64);
  if (!rm) GTEST_SKIP() << "test model not present";
  KvBlockTable seq(*rm->cache);
  ASSERT_TRUE(seq.reserve(static_cast<int64_t>(ref.tokens.size())).ok());
  std::vector<int32_t> pos(ref.tokens.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> logits(static_cast<size_t>(rm->model->config.vocab_size));
  ASSERT_TRUE(rm->transformer->forward(ref.tokens, pos, *rm->cache, seq.block_table(), logits).ok());

  float max_err = 0;
  for (const auto& [id, want] : ref.top) max_err = std::max(max_err, std::abs(logits[static_cast<size_t>(id)] - want));
  EXPECT_LT(max_err, 2e-3f) << "max |logit - ref| over top-32";
  EXPECT_EQ(sample_greedy(logits), ref.top.front().first);
  double sum = 0, sumsq = 0;
  for (float l : logits) {
    sum += l;
    sumsq += static_cast<double>(l) * l;
  }
  EXPECT_NEAR(sum / ref.sum, 1.0, 1e-4);
  EXPECT_NEAR(sumsq / ref.sumsq, 1.0, 1e-4);
}

TEST(RuntimeGolden, SmolLm2GreedyMatchesReference) {
  const RefData ref = load_ref(std::string(ENGINE_TEST_DATA_DIR) + "/ref_smollm2.txt");
  for (DType kv : {DType::kF32, DType::kF16}) {
    auto rm = open_real(kv, 16);  // 16 < prompt length: exercises chunked prefill
    if (!rm) GTEST_SKIP() << "test model not present";
    Generator gen(*rm->transformer, *rm->cache, *rm->model->tokenizer);
    GenerateOptions opts;
    opts.max_new_tokens = static_cast<int32_t>(ref.greedy.size());
    opts.stop_at_eog = false;
    std::vector<TokenId> out;
    ASSERT_TRUE(gen.generate(ref.tokens, opts, [&](TokenId t) { out.push_back(t); return true; }).ok());
    EXPECT_EQ(out, ref.greedy) << "kv=" << dtype_name(kv);
    EXPECT_EQ(rm->cache->free_blocks(), rm->cache->geometry().num_blocks);  // blocks released
  }
}

TEST(RuntimeGolden, ChunkedPrefillEqualsSinglePass) {
  const RefData ref = load_ref(std::string(ENGINE_TEST_DATA_DIR) + "/ref_smollm2.txt");
  std::vector<float> a, b;
  for (int32_t batch : {64, 5}) {
    auto rm = open_real(DType::kF32, batch);
    if (!rm) GTEST_SKIP() << "test model not present";
    KvBlockTable seq(*rm->cache);
    ASSERT_TRUE(seq.reserve(static_cast<int64_t>(ref.tokens.size())).ok());
    std::vector<float> logits(static_cast<size_t>(rm->model->config.vocab_size));
    for (size_t off = 0; off < ref.tokens.size(); off += static_cast<size_t>(batch)) {
      const size_t n = std::min(ref.tokens.size() - off, static_cast<size_t>(batch));
      std::vector<int32_t> pos(n);
      std::iota(pos.begin(), pos.end(), static_cast<int32_t>(off));
      ASSERT_TRUE(rm->transformer->forward(std::span(ref.tokens).subspan(off, n), pos, *rm->cache,
                                           seq.block_table(), logits).ok());
    }
    (batch == 64 ? a : b) = logits;
  }
  ASSERT_EQ(a.size(), b.size());
  float max_diff = 0;
  for (size_t i = 0; i < a.size(); ++i) max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  EXPECT_LT(max_diff, 1e-3f);
}

// Q8_0 execution vs the NumPy reference on the SAME dequantized Q8_0 weights
// (gguf-py dequantization): must agree as tightly as the f16 path. Plus a
// quality check against the f16 reference: quantization may change late
// near-ties, but not the top-1 token or the logits beyond Q8 noise.
TEST(RuntimeGolden, SmolLm2Q8_0MatchesReference) {
  const RefData ref = load_ref(std::string(ENGINE_TEST_DATA_DIR) + "/ref_smollm2_q8_0.txt");
  const RefData f16 = load_ref(std::string(ENGINE_TEST_DATA_DIR) + "/ref_smollm2.txt");
  auto rm = open_real(DType::kF32, 64, dynalm::testing::smollm_q8_model());
  if (!rm) GTEST_SKIP() << "Q8_0 test model not present";
  ASSERT_EQ(rm->model->weights.get(TensorRole::kAttnQ, 0)->dtype(), DType::kQ8_0);
  KvBlockTable seq(*rm->cache);
  ASSERT_TRUE(seq.reserve(static_cast<int64_t>(ref.tokens.size())).ok());
  std::vector<int32_t> pos(ref.tokens.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> logits(static_cast<size_t>(rm->model->config.vocab_size));
  ASSERT_TRUE(rm->transformer->forward(ref.tokens, pos, *rm->cache, seq.block_table(), logits).ok());
  float err_q8 = 0, err_f16 = 0;
  for (const auto& [id, want] : ref.top) err_q8 = std::max(err_q8, std::abs(logits[static_cast<size_t>(id)] - want));
  for (const auto& [id, want] : f16.top) err_f16 = std::max(err_f16, std::abs(logits[static_cast<size_t>(id)] - want));
  EXPECT_LT(err_q8, 2e-3f) << "vs Q8_0 reference";
  EXPECT_LT(err_f16, 0.5f) << "vs f16 reference (quantization noise)";
  EXPECT_EQ(sample_greedy(logits), f16.top.front().first);
  std::printf("[ q8_0 ] max top-32 logit error: vs q8 ref %.5f, vs f16 ref %.4f\n", err_q8, err_f16);
  seq.release();

  Generator gen(*rm->transformer, *rm->cache, *rm->model->tokenizer);
  GenerateOptions opts;
  opts.max_new_tokens = static_cast<int32_t>(ref.greedy.size());
  opts.stop_at_eog = false;
  std::vector<TokenId> out;
  ASSERT_TRUE(gen.generate(ref.tokens, opts, [&](TokenId t) { out.push_back(t); return true; }).ok());
  EXPECT_EQ(out, ref.greedy);
}

TEST(RuntimeGolden, RejectsInvalidInput) {
  auto rm = open_real(DType::kF32, 8);
  if (!rm) GTEST_SKIP() << "test model not present";
  KvBlockTable seq(*rm->cache);
  ASSERT_TRUE(seq.reserve(4).ok());
  std::vector<float> logits(static_cast<size_t>(rm->model->config.vocab_size));
  const TokenId bad[] = {-1};
  const int32_t p0[] = {0};
  EXPECT_FALSE(rm->transformer->forward(bad, p0, *rm->cache, seq.block_table(), logits).ok());
  const TokenId ok[] = {1};
  const int32_t far[] = {1000};  // no block for this position
  EXPECT_FALSE(rm->transformer->forward(ok, far, *rm->cache, seq.block_table(), logits).ok());
}

TEST(CpuDevice, MatmulManyMatchesIndividualMatmuls) {
  // MoE batches: several independent (x, w, y) jobs with 1-3 rows run in one
  // parallel region (fused dot path); larger jobs fall back to matmul().
  ThreadPool pool(4);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  std::mt19937 rng(25);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  constexpr int64_t kK = 64;
  for (const std::vector<int64_t>& rows : {std::vector<int64_t>{1, 2, 3, 1}, std::vector<int64_t>{1, 5, 2}}) {
    std::vector<Tensor> xs, ws, ys, refs;
    std::vector<Device::MatmulJob> jobs;
    for (size_t j = 0; j < rows.size(); ++j) {
      const int64_t n = 17 + 13 * static_cast<int64_t>(j);  // not a multiple of the 16-row chunk
      Tensor x = *Tensor::empty(DType::kF32, {rows[j], kK});
      Tensor w = *Tensor::empty(DType::kF16, {n, kK});
      for (int64_t i = 0; i < x.numel(); ++i) x.data_as<float>()[i] = u(rng);
      for (int64_t i = 0; i < w.numel(); ++i) w.data_as<uint16_t>()[i] = fp32_to_fp16(u(rng));
      ys.push_back(*Tensor::zeros(DType::kF32, {rows[j], n}));
      refs.push_back(*Tensor::zeros(DType::kF32, {rows[j], n}));
      be.matmul(x, w, nullptr, refs.back());
      jobs.push_back({x.view(), w.view(), ys.back().view()});
      xs.push_back(std::move(x));
      ws.push_back(std::move(w));
    }
    be.matmul_many(jobs);
    for (size_t j = 0; j < jobs.size(); ++j) {
      for (int64_t i = 0; i < ys[j].numel(); ++i) {
        const float want = refs[j].data_as<float>()[i];
        EXPECT_NEAR(ys[j].data_as<float>()[i], want, 1e-4f + 1e-5f * std::abs(want)) << "job " << j << " elem " << i;
      }
    }
  }
}

}  // namespace
}  // namespace dynalm

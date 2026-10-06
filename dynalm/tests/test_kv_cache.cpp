#include "kv_cache/kv_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "common/core.h"

namespace dynalm {
namespace {

class KvPaged : public ::testing::Test {
 protected:
  void SetUp() override {
    // 2 layers, 1 kv head, head_dim 4, block 4 tokens, 16 blocks, f32.
    auto p = KvBlockPool::create(KvGeometry{2, 1, 4, 4, 4, 16, DType::kF32}, be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
  }

  // Writes value `v` into every K/V element of `pos` (all layers).
  void write(const KvBlockTable& t, int64_t pos, float v) {
    for (int l = 0; l < 2; ++l) {
      const KvLayerView kv = pool->layer_view(l, t.block_table());
      for (int i = 0; i < 4; ++i) {
        static_cast<float*>(kv.k)[kv.k_offset(pos, 0) + i] = v;
        static_cast<float*>(kv.v)[kv.v_offset(pos, 0) + i] = -v;
      }
    }
  }
  float read_k(const KvBlockTable& t, int64_t pos, int layer = 1) {
    const KvLayerView kv = pool->layer_view(layer, t.block_table());
    return static_cast<float*>(kv.k)[kv.k_offset(pos, 0)];
  }
  float read_v(const KvBlockTable& t, int64_t pos) {
    const KvLayerView kv = pool->layer_view(0, t.block_table());
    return static_cast<float*>(kv.v)[kv.v_offset(pos, 0)];
  }

  ThreadPool tp{1};
  CpuDevice be{tp, CpuIsa::kGeneric};
  std::unique_ptr<KvBlockPool> pool;
};

TEST_F(KvPaged, RefCountLifecycle) {
  auto b = pool->allocate();
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(pool->ref_count(*b), 1);
  pool->retain(*b);
  EXPECT_EQ(pool->ref_count(*b), 2);
  EXPECT_EQ(pool->free_blocks(), 15);
  pool->release(*b);
  EXPECT_EQ(pool->free_blocks(), 15);  // still referenced: never freed
  pool->release(*b);
  EXPECT_EQ(pool->free_blocks(), 16);
}

TEST_F(KvPaged, ReserveTruncateRelease) {
  KvBlockTable t(*pool);
  ASSERT_TRUE(t.reserve(10).ok());  // 3 blocks of 4
  EXPECT_EQ(t.num_blocks(), 3);
  EXPECT_EQ(pool->used_blocks(), 3);
  t.truncate(5);  // positions 0..4 need 2 blocks
  EXPECT_EQ(t.num_blocks(), 2);
  EXPECT_EQ(pool->used_blocks(), 2);
  t.truncate(0);
  EXPECT_EQ(pool->used_blocks(), 0);
}

TEST_F(KvPaged, CloneSharesUntilRelease) {
  KvBlockTable a(*pool);
  ASSERT_TRUE(a.reserve(8).ok());
  for (int p = 0; p < 8; ++p) write(a, p, static_cast<float>(p));
  {
    KvBlockTable b = a.clone();
    EXPECT_EQ(b.block_table()[0], a.block_table()[0]);
    EXPECT_EQ(pool->ref_count(a.block_table()[0]), 2);
    EXPECT_EQ(pool->used_blocks(), 2);  // no copies
    EXPECT_EQ(read_k(b, 5), 5.0f);
  }
  EXPECT_EQ(pool->ref_count(a.block_table()[0]), 1);
  EXPECT_EQ(read_k(a, 7), 7.0f);  // original intact after clone released
}

TEST_F(KvPaged, CopyOnWriteIsolatesForks) {
  KvBlockTable a(*pool);
  ASSERT_TRUE(a.reserve(6).ok());  // block 0 full, block 1 partial (positions 4,5)
  for (int p = 0; p < 6; ++p) write(a, p, static_cast<float>(p + 1));
  KvBlockTable b = a.clone();

  // Fork b continues writing at position 6 (inside shared block 1).
  ASSERT_TRUE(b.make_writable(6, 7).ok());
  EXPECT_NE(b.block_table()[1], a.block_table()[1]);  // block 1 copied
  EXPECT_EQ(b.block_table()[0], a.block_table()[0]);  // block 0 still shared
  write(b, 6, 100.0f);
  write(b, 4, 42.0f);

  // b sees the copied history plus its own writes; a is unaffected.
  EXPECT_EQ(read_k(b, 5), 6.0f);
  EXPECT_EQ(read_v(b, 5), -6.0f);
  EXPECT_EQ(read_k(b, 4), 42.0f);
  EXPECT_EQ(read_k(a, 4), 5.0f);
  EXPECT_EQ(pool->ref_count(a.block_table()[1]), 1);
  EXPECT_EQ(pool->ref_count(a.block_table()[0]), 2);

  // Exclusive blocks are not copied again.
  const int32_t before = b.block_table()[1];
  ASSERT_TRUE(b.make_writable(4, 8).ok());
  EXPECT_EQ(b.block_table()[1], before);
}

TEST_F(KvPaged, ExhaustionAndRecovery) {
  KvBlockTable a(*pool);
  ASSERT_TRUE(a.reserve(64).ok());  // all 16 blocks
  KvBlockTable b(*pool);
  EXPECT_EQ(b.reserve(1).code(), StatusCode::kResourceExhausted);
  KvBlockTable c = a.clone();
  EXPECT_EQ(c.make_writable(0, 1).code(), StatusCode::kResourceExhausted);  // COW needs a free block
  a.release();
  c.release();
  EXPECT_TRUE(b.reserve(1).ok());
}

TEST_F(KvPaged, MoveTransfersOwnership) {
  KvBlockTable a(*pool);
  ASSERT_TRUE(a.reserve(4).ok());
  KvBlockTable b(std::move(a));
  EXPECT_EQ(a.num_blocks(), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.num_blocks(), 1);
  EXPECT_EQ(pool->used_blocks(), 1);
  KvBlockTable c(*pool);
  c = std::move(b);
  EXPECT_EQ(pool->used_blocks(), 1);
}

// Many threads allocating, cloning and releasing concurrently: refcounts and
// the free list must stay consistent (run under TSAN in the Linux gate).
TEST_F(KvPaged, ConcurrentStress) {
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      std::mt19937 rng(static_cast<unsigned>(t));
      for (int it = 0; it < 2000; ++it) {
        KvBlockTable a(*pool);
        if (!a.reserve(1 + static_cast<int64_t>(rng() % 8)).ok()) continue;
        KvBlockTable b = a.clone();
        if (rng() % 2) a.release();
        (void)b.make_writable(0, 1);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(pool->free_blocks(), 16);
  for (int32_t b = 0; b < 16; ++b) EXPECT_EQ(pool->ref_count(b), 0) << b;
}

// q8_0 KV (DD-074): sizing, validation, and logits close to f16 KV.
TEST(KvQ8, GeometryBytesAndValidation) {
  ThreadPool pool(1);
  CpuDevice cpu(pool, CpuIsa::kGeneric);
  KvGeometry g{2, 2, 64, 64, 16, 8, DType::kQ8_0};
  EXPECT_EQ(g.k_block_bytes(), 2 * 16 * 64 / 32 * 34);  // 34-byte blocks of 32 values
  EXPECT_EQ(g.bytes_per_layer(), 2 * g.k_block_bytes() * 8);
  auto ok = KvBlockPool::create(g, cpu);
  EXPECT_TRUE(ok.ok()) << ok.status().to_string();
  KvGeometry bad{2, 2, 48, 48, 16, 8, DType::kQ8_0};
  auto rejected = KvBlockPool::create(bad, cpu);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), StatusCode::kUnsupported);
}

TEST(KvQ8, LogitsTrackF16Cache) {
  for (const char* arch : {"qwen3", "gemma3"}) {  // head_dim 32: q8_0-compatible tiny models
    auto m = load_model(std::string(ENGINE_TEST_DATA_DIR) + "/tiny_" + arch + ".gguf");
    ASSERT_TRUE(m.ok());
    const ModelConfig& c = (*m)->config;
    ThreadPool pool(2);
    CpuDevice cpu(pool, select_best_isa(cpu_info().features));
    auto tf = Transformer::create(c, (*m)->weights, cpu, 64);
    ASSERT_TRUE(tf.ok());
    auto run = [&](DType dt) {
      auto kv = KvBlockPool::create(KvGeometry{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 8, dt}, cpu);
      EXPECT_TRUE(kv.ok());
      KvBlockTable t(**kv);
      EXPECT_TRUE(t.reserve(40).ok());
      std::vector<TokenId> toks(40);
      for (size_t i = 0; i < toks.size(); ++i) toks[i] = static_cast<TokenId>(3 + (i * 7) % 200);
      std::vector<int32_t> pos(toks.size());
      std::iota(pos.begin(), pos.end(), 0);
      std::vector<float> logits(static_cast<size_t>(c.vocab_size));
      EXPECT_TRUE((*tf)->forward(toks, pos, **kv, t.block_table(), logits).ok());
      return logits;
    };
    const std::vector<float> a = run(DType::kF16), b = run(DType::kQ8_0);
    float amax = 0, dmax = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      amax = std::max(amax, std::fabs(a[i]));
      dmax = std::max(dmax, std::fabs(a[i] - b[i]));
    }
    EXPECT_LT(dmax, 0.02f * amax + 1e-3f) << arch;  // ~2^-8 relative KV error, not a different model
    EXPECT_EQ(std::max_element(a.begin(), a.end()) - a.begin(), std::max_element(b.begin(), b.end()) - b.begin())
        << arch;
  }
}

}  // namespace
}  // namespace dynalm

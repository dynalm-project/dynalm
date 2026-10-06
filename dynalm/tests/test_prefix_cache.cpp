#include "prefix_cache/prefix_cache.h"

#include <gtest/gtest.h>

#include <map>
#include <random>

#include "dynacore/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "runtime/generator.h"
#include "scheduler/scheduler.h"

namespace engine {
namespace {

// --- PrefixCache unit tests (block size 4) ---

std::unique_ptr<PrefixCache> make_cache(PrefixCacheKind k, KvBlockPool& pool, int32_t max_blocks = 0) {
  return k == PrefixCacheKind::kRadix ? make_radix_prefix_cache(pool, max_blocks)
                                      : make_hash_prefix_cache(pool, max_blocks);
}
std::string kind_name(const ::testing::TestParamInfo<PrefixCacheKind>& p) {
  return p.param == PrefixCacheKind::kRadix ? "Radix" : "Hash";
}

class PrefixCacheUnit : public ::testing::TestWithParam<PrefixCacheKind> {
 protected:
  void SetUp() override {
    auto p = KvBlockPool::create(KvGeometry{1, 1, 4, 4, 4, 32, DType::kF32}, be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
  }
  // A sequence "computing" tokens: returns its table (blocks allocated).
  KvBlockTable seq(size_t n) {
    KvBlockTable t(*pool);
    EXPECT_TRUE(t.reserve(static_cast<int64_t>(n)).ok());
    return t;
  }
  ThreadPool tp{1};
  CpuBackend be{tp, CpuIsa::kGeneric};
  std::unique_ptr<KvBlockPool> pool;
};

TEST_P(PrefixCacheUnit, InsertLookupLongestPrefix) {
  auto cache_ptr = make_cache(GetParam(), *pool);
  PrefixCache& cache = *cache_ptr;
  const std::vector<TokenId> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // 2 full blocks + 2
  KvBlockTable ta = seq(a.size());
  cache.insert(a, ta.block_table(), 0, 2);
  EXPECT_EQ(cache.stats().cached_blocks, 2);
  EXPECT_EQ(pool->ref_count(ta.block_table()[0]), 2);  // sequence + cache

  const bool radix = GetParam() == PrefixCacheKind::kRadix;
  // Same first block, second block diverges after 2 tokens.
  const std::vector<TokenId> b = {1, 2, 3, 4, 5, 6, 0, 8, 9};
  auto m = cache.lookup(b, 100);
  EXPECT_EQ(m.tokens, radix ? 6 : 4);  // radix also reuses the 2 matching tokens of block 1
  ASSERT_EQ(m.blocks.size(), radix ? 2u : 1u);
  EXPECT_EQ(m.blocks[0], ta.block_table()[0]);
  EXPECT_EQ(pool->ref_count(m.blocks[0]), 3);  // returned reference belongs to the caller
  if (radix) {
    EXPECT_NE(m.blocks[1], ta.block_table()[1]);  // a private copy, not the shared block
    EXPECT_EQ(pool->ref_count(m.blocks[1]), 1);
  }
  for (int32_t blk : m.blocks) pool->release(blk);

  // Capped by max_tokens: hash at block granularity, radix at token granularity.
  auto m2 = cache.lookup(a, 7);
  EXPECT_EQ(m2.tokens, radix ? 7 : 4);
  for (int32_t blk : m2.blocks) pool->release(blk);
  auto m3 = cache.lookup(a, 9);
  EXPECT_EQ(m3.tokens, 8);
  for (int32_t blk : m3.blocks) pool->release(blk);

  // No match when the first block differs, even if later blocks agree.
  const std::vector<TokenId> c = {0, 2, 3, 4, 5, 6, 7, 8};
  auto m4 = cache.lookup(c, 100);
  EXPECT_EQ(m4.tokens, 0);
  for (int32_t blk : m4.blocks) pool->release(blk);
  EXPECT_EQ(pool->used_blocks(), ta.num_blocks());  // cache shares ta's blocks; no leaked copies
}

TEST_P(PrefixCacheUnit, BlocksNeverFreedWhileReferenced) {
  auto cache_ptr = make_cache(GetParam(), *pool);
  PrefixCache& cache = *cache_ptr;
  const std::vector<TokenId> a = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    KvBlockTable ta = seq(a.size());
    cache.insert(a, ta.block_table(), 0, 2);
    EXPECT_EQ(cache.evict(10), 0);  // still used by the sequence
  }
  EXPECT_EQ(pool->used_blocks(), 2);  // sequence gone; cache keeps them
  auto m = cache.lookup(a, 100);      // another sequence adopts them
  EXPECT_EQ(cache.evict(10), 0);      // in use again: not evictable
  for (int32_t b : m.blocks) pool->release(b);
  EXPECT_EQ(cache.evict(10), 2);  // now only the cache holds them
  EXPECT_EQ(pool->used_blocks(), 0);
}

TEST_P(PrefixCacheUnit, EvictsLeavesFirstInLruOrder) {
  auto cache_ptr = make_cache(GetParam(), *pool);
  PrefixCache& cache = *cache_ptr;
  const std::vector<TokenId> a = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};  // chain of 3
  const std::vector<TokenId> b = {9, 9, 9, 9};                          // separate chain
  {
    KvBlockTable ta = seq(a.size()), tb = seq(b.size());
    cache.insert(a, ta.block_table(), 0, 3);
    cache.insert(b, tb.block_table(), 0, 1);
  }
  EXPECT_EQ(cache.evict(1), 1);  // oldest leaf: the tail of chain a
  auto m = cache.lookup(a, 100);
  EXPECT_EQ(m.tokens, 8);  // first two blocks of a still reachable
  for (int32_t blk : m.blocks) pool->release(blk);
  EXPECT_EQ(cache.evict(10), 3);
  EXPECT_EQ(cache.stats().cached_blocks, 0);
}

TEST_P(PrefixCacheUnit, CapacityBoundsCache) {
  auto cache_ptr = make_cache(GetParam(), *pool, /*max_blocks=*/2);
  PrefixCache& cache = *cache_ptr;
  std::mt19937 rng(1);
  for (int i = 0; i < 5; ++i) {
    std::vector<TokenId> t(4);
    for (auto& x : t) x = static_cast<TokenId>(rng() % 1000);
    KvBlockTable tt = seq(4);
    cache.insert(t, tt.block_table(), 0, 1);
  }
  EXPECT_LE(cache.stats().cached_blocks, 2);
  EXPECT_EQ(pool->used_blocks(), cache.stats().cached_blocks);
}

INSTANTIATE_TEST_SUITE_P(Kinds, PrefixCacheUnit, ::testing::Values(PrefixCacheKind::kRadix, PrefixCacheKind::kHash),
                         kind_name);

// Radix-only: a partially matching block is copied, never shared, and the
// copy's first r rows equal the cached block's.
TEST(RadixPrefixCache, PartialBlockIsPrivateCopy) {
  ThreadPool tp(1);
  CpuBackend be(tp, CpuIsa::kGeneric);
  auto pool = KvBlockPool::create(KvGeometry{1, 1, 4, 4, 4, 8, DType::kF32}, be);
  ASSERT_TRUE(pool.ok());
  auto cache = make_radix_prefix_cache(**pool);
  KvBlockTable t(**pool);
  ASSERT_TRUE(t.reserve(4).ok());
  const KvLayerView kv = (*pool)->layer_view(0, t.block_table());
  for (int p = 0; p < 4; ++p) static_cast<float*>(kv.k)[kv.k_offset(p, 0)] = 10.0f + static_cast<float>(p);
  const std::vector<TokenId> a = {7, 8, 9, 10};
  cache->insert(a, t.block_table(), 0, 1);
  const std::vector<TokenId> q = {7, 8, 99, 100, 101};
  auto m = cache->lookup(q, 4);
  ASSERT_EQ(m.tokens, 2);
  ASSERT_EQ(m.blocks.size(), 1u);
  const int32_t copy[] = {m.blocks[0]};
  const KvLayerView ck = (*pool)->layer_view(0, copy);
  EXPECT_EQ(static_cast<float*>(ck.k)[ck.k_offset(0, 0)], 10.0f);
  EXPECT_EQ(static_cast<float*>(ck.k)[ck.k_offset(1, 0)], 11.0f);
  EXPECT_EQ(cache->stats().partial_hit_tokens, 2u);
  (*pool)->release(m.blocks[0]);
}

// --- Scheduler with prefix reuse (tiny model, block size 4) ---

class PrefixScheduling : public ::testing::TestWithParam<PrefixCacheKind> {
 protected:
  void SetUp() override {
    auto m = load_model(std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf");
    ASSERT_TRUE(m.ok());
    model = std::move(*m);
    make_pool(128);
    auto t = Transformer::create(model->config, model->weights, be, 64);
    ASSERT_TRUE(t.ok());
    tf = std::move(*t);
  }
  void make_pool(int32_t blocks) {
    const ModelConfig& c = model->config;
    auto p = KvBlockPool::create(KvGeometry{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 4, blocks,
                                            DType::kF32},
                                 be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
  }
  std::vector<TokenId> solo(const std::vector<TokenId>& p, int32_t n) {
    auto sp = KvBlockPool::create(pool->geometry(), be);
    Generator g(*tf, **sp, *model->tokenizer);
    GenerateOptions o;
    o.max_new_tokens = n;
    o.stop_at_eog = false;
    std::vector<TokenId> out;
    EXPECT_TRUE(g.generate(p, o, [&](TokenId t) { out.push_back(t); return true; }).ok());
    return out;
  }
  ThreadPool tp{3};
  CpuBackend be{tp, CpuIsa::kGeneric};
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<Transformer> tf;
};

TEST_P(PrefixScheduling, SharedSystemPromptIsReusedAndExact) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{.prefix_cache_kind = GetParam()});
  std::map<uint64_t, std::vector<TokenId>> out;
  const std::vector<TokenId> system = {11, 22, 33, 44, 55, 66, 77, 88, 99, 12, 23, 34, 45, 56, 67, 78, 89, 90};
  std::vector<std::pair<uint64_t, std::vector<TokenId>>> want;
  for (int i = 0; i < 4; ++i) {
    std::vector<TokenId> p = system;
    p.push_back(static_cast<TokenId>(100 + i));
    p.push_back(static_cast<TokenId>(150 + i));
    Request r;
    r.prompt = p;
    r.stop = StopParams{6, false};
    r.on_event = [&out](const RequestEvent& ev) {
      if (!ev.finished) out[ev.request_id].push_back(ev.token);
    };
    const uint64_t id = sched.submit(std::move(r));
    want.emplace_back(id, solo(p, 6));
    sched.run_until_idle();  // sequential arrivals: later ones hit the cache
  }
  for (const auto& [id, tokens] : want) EXPECT_EQ(out[id], tokens) << "request " << id;
  const PrefixCacheStats& st = sched.prefix_cache()->stats();
  // Hash: the 4 full system blocks (16 tokens). Radix: all 18 shared tokens
  // (2 more from the partially matching 5th block).
  const uint64_t reused = GetParam() == PrefixCacheKind::kRadix ? 18u : 16u;
  EXPECT_EQ(st.hit_tokens, 3u * reused);
  EXPECT_GT(st.hit_rate(), 0.5);
  // Rows computed: 20 + 3 * (20 - reused) prompt rows + decode rows, instead of 4 * 20.
  EXPECT_EQ(sched.stats().tokens_computed, 20u + 3u * (20u - reused) + 4u * 5u);
}

TEST_P(PrefixScheduling, FullyCachedPromptStillComputesLastToken) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{.prefix_cache_kind = GetParam()});
  std::map<uint64_t, std::vector<TokenId>> out;
  const std::vector<TokenId> p = {5, 6, 7, 8, 9, 10, 11, 12};  // exactly 2 blocks
  for (int round = 0; round < 2; ++round) {
    Request r;
    r.prompt = p;
    r.stop = StopParams{4, false};
    r.on_event = [&out, round](const RequestEvent& ev) {
      if (!ev.finished) out[static_cast<uint64_t>(round)].push_back(ev.token);
    };
    sched.submit(std::move(r));
    sched.run_until_idle();
  }
  EXPECT_EQ(out[0], solo(p, 4));
  EXPECT_EQ(out[1], out[0]);
  // Hash reuses 1 whole block; radix reuses 7 of 8 tokens (only the last is recomputed).
  EXPECT_EQ(sched.prefix_cache()->stats().hit_tokens, GetParam() == PrefixCacheKind::kRadix ? 7u : 4u);
}

TEST_P(PrefixScheduling, CacheEvictedBeforePreemptingUnderPressure) {
  make_pool(16);  // 64 tokens of KV
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{.prefix_cache_kind = GetParam()});
  std::map<uint64_t, std::vector<TokenId>> out;
  auto run = [&](unsigned seed, size_t len) {
    std::mt19937 rng(seed);
    std::vector<TokenId> p(len);
    for (auto& t : p) t = static_cast<TokenId>(rng() % 256);
    Request r;
    r.prompt = p;
    r.stop = StopParams{8, false};
    r.on_event = [&out](const RequestEvent& ev) {
      if (!ev.finished) out[ev.request_id].push_back(ev.token);
    };
    const uint64_t id = sched.submit(std::move(r));
    sched.run_until_idle();
    EXPECT_EQ(out[id], solo(p, 8));
  };
  for (unsigned s = 0; s < 6; ++s) run(s, 40);  // each fills most of the pool; cache must yield
  EXPECT_GT(sched.prefix_cache()->stats().evicted_blocks, 0u);
  EXPECT_EQ(sched.stats().preemptions, 0u);
  EXPECT_EQ(sched.stats().failed, 0u);
}

INSTANTIATE_TEST_SUITE_P(Kinds, PrefixScheduling, ::testing::Values(PrefixCacheKind::kRadix, PrefixCacheKind::kHash),
                         kind_name);

}  // namespace
}  // namespace engine

#pragma once

// Hash-based prefix cache: reuse KV blocks across requests that share a
// token prefix (system prompts, few-shot examples, multi-turn history).
//
// Granularity is one full KV block. Block i of a token sequence is keyed by a
// chained hash h_i = H(h_{i-1}, tokens of block i), so a key identifies the
// whole prefix up to and including that block. Each entry also stores the
// block's tokens and its parent key; lookups verify both, so a 64-bit hash
// collision can never return the wrong KV.
//
// The cache holds one pool reference per cached block; a block is freed only
// when the cache evicts it AND no sequence uses it. Only full blocks are
// cached, and full blocks are never written again, so sharing is safe.
//
// Eviction: LRU among entries that (a) have no cached children (leaves first,
// so chains stay reachable) and (b) are referenced by nobody but the cache.
//
// Not thread-safe: owned and used by the scheduler thread only.

#include <cstdint>
#include <list>
#include <span>
#include <unordered_map>
#include <vector>

#include "kv_cache/kv_cache.h"
#include "tokenizer/tokenizer.h"

namespace engine {

struct PrefixCacheStats {
  uint64_t lookups = 0;
  uint64_t lookup_tokens = 0;  // tokens eligible for reuse across lookups
  uint64_t hit_tokens = 0;     // tokens actually reused
  uint64_t inserted_blocks = 0;
  uint64_t evicted_blocks = 0;
  int64_t cached_blocks = 0;
  double hit_rate() const { return lookup_tokens ? static_cast<double>(hit_tokens) / lookup_tokens : 0.0; }
};

class PrefixCache {
 public:
  // `max_blocks` caps how many blocks the cache may hold (0 = no cap beyond
  // the pool; eviction then happens only on demand).
  PrefixCache(KvBlockPool& pool, int32_t max_blocks = 0);
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  struct Match {
    std::vector<int32_t> blocks;  // retained on behalf of the caller
    int32_t tokens = 0;
  };
  // Longest cached prefix of `tokens`, in full blocks, covering at most
  // `max_tokens` tokens. The returned blocks carry one reference each that
  // the caller now owns (e.g. via KvBlockTable::append_shared).
  Match lookup(std::span<const TokenId> tokens, int32_t max_tokens);

  // Offers the full blocks [first_block, num_full_blocks) of a sequence whose
  // block i holds tokens [i*bs, (i+1)*bs). New ones are retained by the cache.
  void insert(std::span<const TokenId> tokens, std::span<const int32_t> blocks, int32_t first_block,
              int32_t num_full_blocks);

  // Frees up to `n` blocks that only the cache references. Returns the count.
  int32_t evict(int32_t n);

  const PrefixCacheStats& stats() const { return stats_; }

 private:
  struct Entry {
    uint64_t parent = 0;  // key of the previous block (0 = none)
    std::vector<TokenId> tokens;
    int32_t block = -1;
    int32_t children = 0;
    std::list<uint64_t>::iterator lru;  // position in lru_ (front = most recent)
  };

  static uint64_t chain_hash(uint64_t parent, std::span<const TokenId> tokens);
  void touch(Entry& e, uint64_t key);
  bool evict_one();

  KvBlockPool& pool_;
  int32_t block_size_;
  int32_t max_blocks_;
  std::unordered_map<uint64_t, Entry> map_;
  std::list<uint64_t> lru_;
  PrefixCacheStats stats_;
};

}  // namespace engine

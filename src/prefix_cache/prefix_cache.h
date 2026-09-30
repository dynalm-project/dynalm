#pragma once

// Prefix caches: reuse KV blocks across requests that share a token prefix
// (system prompts, few-shot examples, multi-turn history).
//
// Hash cache (Phase 15):
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
#include <memory>
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
  uint64_t partial_hit_tokens = 0;  // of which reused via a partially matching block (radix)
  uint64_t inserted_blocks = 0;
  uint64_t evicted_blocks = 0;
  int64_t cached_blocks = 0;
  double hit_rate() const { return lookup_tokens ? static_cast<double>(hit_tokens) / lookup_tokens : 0.0; }
};

// Common interface of the hash (Phase 15) and radix (Phase 16) caches.
class PrefixCache {
 public:
  struct Match {
    // Blocks to adopt, each carrying one reference now owned by the caller.
    // All but possibly the last are shared, immutable cached blocks; a radix
    // cache may end the match with a private copy of a partially matching
    // block, so `tokens` need not be a multiple of the block size.
    std::vector<int32_t> blocks;
    int32_t tokens = 0;
  };

  virtual ~PrefixCache() = default;

  // Longest reusable prefix of `tokens`, covering at most `max_tokens`.
  virtual Match lookup(std::span<const TokenId> tokens, int32_t max_tokens) = 0;
  // Offers the full blocks [first_block, num_full_blocks) of a sequence whose
  // block i holds tokens [i*bs, (i+1)*bs). New ones are retained by the cache.
  virtual void insert(std::span<const TokenId> tokens, std::span<const int32_t> blocks, int32_t first_block,
                      int32_t num_full_blocks) = 0;
  // Frees up to `n` blocks that only the cache references. Returns the count.
  virtual int32_t evict(int32_t n) = 0;
  virtual const PrefixCacheStats& stats() const = 0;
};

// Chained-hash cache (full blocks only). `max_blocks` 0 = bounded by the pool.
std::unique_ptr<PrefixCache> make_hash_prefix_cache(KvBlockPool& pool, int32_t max_blocks = 0);
// Radix tree of blocks with token-granular partial-block reuse (DD-030).
std::unique_ptr<PrefixCache> make_radix_prefix_cache(KvBlockPool& pool, int32_t max_blocks = 0);

}  // namespace engine

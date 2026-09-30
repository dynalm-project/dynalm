#include "prefix_cache/prefix_cache.h"

#include <algorithm>

namespace engine {
namespace {

// SplitMix64 finalizer: strong 64-bit mixing for chaining token ids.
inline uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

}  // namespace

PrefixCache::PrefixCache(KvBlockPool& pool, int32_t max_blocks)
    : pool_(pool), block_size_(pool.geometry().block_size), max_blocks_(max_blocks) {}

PrefixCache::~PrefixCache() {
  for (auto& [key, e] : map_) pool_.release(e.block);
}

uint64_t PrefixCache::chain_hash(uint64_t parent, std::span<const TokenId> tokens) {
  uint64_t h = mix(parent ^ 0x5bd1e995ull);
  for (TokenId t : tokens) h = mix(h ^ static_cast<uint32_t>(t));
  return h == 0 ? 1 : h;  // 0 is reserved for "no parent"
}

void PrefixCache::touch(Entry& e, uint64_t key) {
  lru_.erase(e.lru);
  lru_.push_front(key);
  e.lru = lru_.begin();
}

PrefixCache::Match PrefixCache::lookup(std::span<const TokenId> tokens, int32_t max_tokens) {
  Match m;
  ++stats_.lookups;
  const int32_t eligible = std::min(static_cast<int32_t>(tokens.size()), max_tokens) / block_size_ * block_size_;
  stats_.lookup_tokens += static_cast<uint64_t>(std::max(eligible, 0));
  uint64_t parent = 0;
  for (int32_t off = 0; off + block_size_ <= eligible; off += block_size_) {
    const auto chunk = tokens.subspan(static_cast<size_t>(off), static_cast<size_t>(block_size_));
    const uint64_t key = chain_hash(parent, chunk);
    auto it = map_.find(key);
    // Verify: same parent chain and identical tokens (collision-proof).
    if (it == map_.end() || it->second.parent != parent ||
        !std::equal(chunk.begin(), chunk.end(), it->second.tokens.begin())) {
      break;
    }
    pool_.retain(it->second.block);
    m.blocks.push_back(it->second.block);
    touch(it->second, key);
    parent = key;
  }
  m.tokens = static_cast<int32_t>(m.blocks.size()) * block_size_;
  stats_.hit_tokens += static_cast<uint64_t>(m.tokens);
  return m;
}

void PrefixCache::insert(std::span<const TokenId> tokens, std::span<const int32_t> blocks, int32_t first_block,
                         int32_t num_full_blocks) {
  // Walk the chain from the start to recover parent keys (cheap: hashing only).
  uint64_t parent = 0;
  for (int32_t i = 0; i < num_full_blocks; ++i) {
    const auto chunk = tokens.subspan(static_cast<size_t>(i) * static_cast<size_t>(block_size_),
                                      static_cast<size_t>(block_size_));
    const uint64_t key = chain_hash(parent, chunk);
    if (i >= first_block) {
      auto it = map_.find(key);
      if (it == map_.end()) {
        if (max_blocks_ > 0 && static_cast<int32_t>(map_.size()) >= max_blocks_ && !evict_one()) return;
        Entry e;
        e.parent = parent;
        e.tokens.assign(chunk.begin(), chunk.end());
        e.block = blocks[static_cast<size_t>(i)];
        pool_.retain(e.block);
        lru_.push_front(key);
        e.lru = lru_.begin();
        map_.emplace(key, std::move(e));
        if (parent != 0) {
          if (auto p = map_.find(parent); p != map_.end()) ++p->second.children;
        }
        ++stats_.inserted_blocks;
      } else if (it->second.parent != parent ||
                 !std::equal(chunk.begin(), chunk.end(), it->second.tokens.begin())) {
        return;  // hash collision with a different prefix: keep the existing entry, stop here
      } else {
        touch(it->second, key);
      }
    }
    parent = key;
  }
  stats_.cached_blocks = static_cast<int64_t>(map_.size());
}

bool PrefixCache::evict_one() {
  // Oldest first; skip blocks still used by sequences and non-leaf entries.
  for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
    auto e = map_.find(*it);
    if (e->second.children > 0 || pool_.ref_count(e->second.block) != 1) continue;
    if (e->second.parent != 0) {
      if (auto p = map_.find(e->second.parent); p != map_.end()) --p->second.children;
    }
    pool_.release(e->second.block);
    lru_.erase(std::next(it).base());
    map_.erase(e);
    ++stats_.evicted_blocks;
    stats_.cached_blocks = static_cast<int64_t>(map_.size());
    return true;
  }
  return false;
}

int32_t PrefixCache::evict(int32_t n) {
  int32_t freed = 0;
  while (freed < n && evict_one()) ++freed;
  return freed;
}

}  // namespace engine

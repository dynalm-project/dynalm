// Radix prefix cache (Phase 16).
//
// A tree whose edges are KV blocks: the path root → n spells a token prefix
// in block-size steps, and node n owns one pool reference to the block that
// holds its tokens' K/V. Children are indexed by a hash of their block's
// tokens and always verified token by token.
//
// Beyond the hash cache it matches below block granularity: when the query
// diverges inside a cached block, the child sharing the longest token prefix
// r (1 <= r < block_size) is copied into a fresh private block and r extra
// tokens are reused. Those rows are exactly what the sequence would have
// computed (same tokens at the same positions after the same prefix), and the
// copy is exclusive, so the sequence can write the rest of the block.
//
// Eviction: LRU over leaves referenced only by the cache; evicting a leaf can
// turn its parent into an evictable leaf.

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>

#include "prefix_cache/prefix_cache.h"

namespace engine {
namespace {

uint64_t block_key(std::span<const TokenId> tokens) {
  uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a over token ids
  for (TokenId t : tokens) {
    h ^= static_cast<uint32_t>(t);
    h *= 0x100000001b3ull;
  }
  return h;
}

class RadixPrefixCache final : public PrefixCache {
 public:
  RadixPrefixCache(KvBlockPool& pool, int32_t max_blocks)
      : pool_(pool), bs_(pool.geometry().block_size), max_blocks_(max_blocks) {}

  ~RadixPrefixCache() override { release_subtree(root_); }

  Match lookup(std::span<const TokenId> tokens, int32_t max_tokens) override {
    Match m;
    ++stats_.lookups;
    const auto limit = static_cast<size_t>(std::max(0, std::min(static_cast<int32_t>(tokens.size()), max_tokens)));
    stats_.lookup_tokens += limit;
    Node* node = &root_;
    size_t off = 0;
    // Whole blocks.
    while (off + static_cast<size_t>(bs_) <= limit) {
      Node* child = find_child(*node, tokens.subspan(off, static_cast<size_t>(bs_)));
      if (!child) break;
      pool_.retain(child->block);
      m.blocks.push_back(child->block);
      touch(*child);
      node = child;
      off += static_cast<size_t>(bs_);
    }
    // Partial block: the child sharing the longest token prefix.
    const size_t remaining = std::min(limit - off, static_cast<size_t>(bs_ - 1));
    if (remaining > 0) {
      Node* best = nullptr;
      size_t best_r = 0;
      for (auto& [key, child] : node->children) {
        size_t r = 0;
        while (r < remaining && child->tokens[r] == tokens[off + r]) ++r;
        if (r > best_r) {
          best_r = r;
          best = child.get();
        }
      }
      if (best) {
        auto copy = pool_.allocate();
        if (copy.ok()) {  // no free block: skip the partial reuse, it's an optimization
          pool_.copy_block(*copy, best->block);
          m.blocks.push_back(*copy);
          off += best_r;
          stats_.partial_hit_tokens += best_r;
          touch(*best);
        }
      }
    }
    m.tokens = static_cast<int32_t>(off);
    stats_.hit_tokens += off;
    return m;
  }

  void insert(std::span<const TokenId> tokens, std::span<const int32_t> blocks, int32_t first_block,
              int32_t num_full_blocks) override {
    Node* node = &root_;
    for (int32_t i = 0; i < num_full_blocks; ++i) {
      const auto chunk = tokens.subspan(static_cast<size_t>(i) * static_cast<size_t>(bs_), static_cast<size_t>(bs_));
      // Blocks before first_block were cached earlier; if they were evicted
      // since, they are re-inserted here too (the sequence still holds them).
      (void)first_block;
      Node* child = find_child(*node, chunk);
      if (!child) {
        if (max_blocks_ > 0 && cached_ >= max_blocks_ && !evict_one()) return;
        auto n = std::make_unique<Node>();
        n->tokens.assign(chunk.begin(), chunk.end());
        n->block = blocks[static_cast<size_t>(i)];
        n->parent = node;
        pool_.retain(n->block);
        lru_.push_front(n.get());
        n->lru = lru_.begin();
        child = n.get();
        node->children.emplace(block_key(chunk), std::move(n));
        ++cached_;
        ++stats_.inserted_blocks;
      } else {
        touch(*child);
      }
      node = child;
    }
    stats_.cached_blocks = cached_;
  }

  int32_t evict(int32_t n) override {
    int32_t freed = 0;
    while (freed < n && evict_one()) ++freed;
    return freed;
  }

  const PrefixCacheStats& stats() const override { return stats_; }

 private:
  struct Node {
    std::vector<TokenId> tokens;
    int32_t block = -1;
    Node* parent = nullptr;
    // Keyed by block_key(tokens); collisions are resolved by a vector bucket.
    std::unordered_multimap<uint64_t, std::unique_ptr<Node>> children;
    std::list<Node*>::iterator lru;
  };

  Node* find_child(Node& node, std::span<const TokenId> chunk) {
    auto [lo, hi] = node.children.equal_range(block_key(chunk));
    for (auto it = lo; it != hi; ++it) {
      if (std::equal(chunk.begin(), chunk.end(), it->second->tokens.begin())) return it->second.get();
    }
    return nullptr;
  }

  void touch(Node& n) {
    lru_.erase(n.lru);
    lru_.push_front(&n);
    n.lru = lru_.begin();
  }

  bool evict_one() {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
      Node* n = *it;
      if (!n->children.empty() || pool_.ref_count(n->block) != 1) continue;
      pool_.release(n->block);
      lru_.erase(std::next(it).base());
      Node* parent = n->parent;
      auto [lo, hi] = parent->children.equal_range(block_key(n->tokens));
      for (auto c = lo; c != hi; ++c) {
        if (c->second.get() == n) {
          parent->children.erase(c);  // destroys n
          break;
        }
      }
      --cached_;
      ++stats_.evicted_blocks;
      stats_.cached_blocks = cached_;
      return true;
    }
    return false;
  }

  void release_subtree(Node& n) {
    for (auto& [key, child] : n.children) {
      release_subtree(*child);
      pool_.release(child->block);
    }
  }

  KvBlockPool& pool_;
  int32_t bs_;
  int32_t max_blocks_;
  Node root_;
  std::list<Node*> lru_;
  int64_t cached_ = 0;
  PrefixCacheStats stats_;
};

}  // namespace

std::unique_ptr<PrefixCache> make_radix_prefix_cache(KvBlockPool& pool, int32_t max_blocks) {
  return std::make_unique<RadixPrefixCache>(pool, max_blocks);
}

}  // namespace engine

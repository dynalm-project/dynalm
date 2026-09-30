#pragma once

// KV-cache storage: one K pool and one V pool per layer, each an array of
// fixed-size blocks (layout in kv_layout.h), allocated once up front.
//
// Block allocation here is a simple LIFO free list for single-threaded use.
// Phase 10 replaces it with a reference-counted, concurrent block pool; the
// physical layout and KvLayerView stay the same.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "backends/backend.h"
#include "common/status.h"
#include "kv_cache/kv_layout.h"

namespace engine {

class KvCache {
 public:
  static Result<std::unique_ptr<KvCache>> create(const KvGeometry& geom, Backend& backend);

  const KvGeometry& geometry() const { return geom_; }
  int32_t free_blocks() const { return static_cast<int32_t>(free_.size()); }

  // kResourceExhausted when no block is free.
  Result<int32_t> allocate_block();
  void free_block(int32_t block);

  KvLayerView layer_view(int32_t layer, std::span<const int32_t> block_table) const {
    return KvLayerView{k_[static_cast<size_t>(layer)]->data(), v_[static_cast<size_t>(layer)]->data(), &geom_,
                       block_table};
  }

 private:
  KvGeometry geom_;
  std::vector<std::shared_ptr<Storage>> k_, v_;
  std::vector<int32_t> free_;
};

// A sequence's view of the cache: its block table and token count.
class KvSequence {
 public:
  explicit KvSequence(KvCache& cache) : cache_(&cache) {}
  ~KvSequence() { release(); }
  KvSequence(const KvSequence&) = delete;
  KvSequence& operator=(const KvSequence&) = delete;

  // Ensures blocks exist for positions [0, n_tokens).
  Status reserve(int64_t n_tokens);
  void release();

  std::span<const int32_t> block_table() const { return blocks_; }
  int64_t capacity() const { return static_cast<int64_t>(blocks_.size()) * cache_->geometry().block_size; }

 private:
  KvCache* cache_;
  std::vector<int32_t> blocks_;
};

}  // namespace engine

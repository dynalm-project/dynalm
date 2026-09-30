#include "kv_cache/kv_cache.h"

#include <string>

namespace engine {

Result<std::unique_ptr<KvCache>> KvCache::create(const KvGeometry& geom, Backend& backend) {
  if (geom.num_layers <= 0 || geom.num_kv_heads <= 0 || geom.head_dim <= 0 || geom.head_dim_v <= 0 ||
      geom.block_size <= 0 || geom.num_blocks <= 0) {
    return InvalidArgument("KvCache: invalid geometry");
  }
  if (geom.dtype != DType::kF32 && geom.dtype != DType::kF16) {
    return Unsupported("KvCache: dtype " + std::string(dtype_name(geom.dtype)) + " not supported");
  }
  auto cache = std::unique_ptr<KvCache>(new KvCache());
  cache->geom_ = geom;
  const auto elem = static_cast<size_t>(dtype_block_bytes(geom.dtype));
  const size_t k_bytes = static_cast<size_t>(geom.k_block_elems() * geom.num_blocks) * elem;
  const size_t v_bytes = static_cast<size_t>(geom.v_block_elems() * geom.num_blocks) * elem;
  for (int32_t l = 0; l < geom.num_layers; ++l) {
    ENGINE_ASSIGN_OR_RETURN(auto k, backend.allocate(k_bytes));
    ENGINE_ASSIGN_OR_RETURN(auto v, backend.allocate(v_bytes));
    cache->k_.push_back(std::move(k));
    cache->v_.push_back(std::move(v));
  }
  // LIFO free list, lowest block ids handed out first.
  cache->free_.reserve(static_cast<size_t>(geom.num_blocks));
  for (int32_t b = geom.num_blocks - 1; b >= 0; --b) cache->free_.push_back(b);
  return cache;
}

Result<int32_t> KvCache::allocate_block() {
  if (free_.empty()) return ResourceExhausted("KV cache is full");
  const int32_t b = free_.back();
  free_.pop_back();
  return b;
}

void KvCache::free_block(int32_t block) { free_.push_back(block); }

Status KvSequence::reserve(int64_t n_tokens) {
  while (capacity() < n_tokens) {
    ENGINE_ASSIGN_OR_RETURN(int32_t b, cache_->allocate_block());
    blocks_.push_back(b);
  }
  return Status::Ok();
}

void KvSequence::release() {
  for (int32_t b : blocks_) cache_->free_block(b);
  blocks_.clear();
}

}  // namespace engine

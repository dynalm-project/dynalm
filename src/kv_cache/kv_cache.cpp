#include "kv_cache/kv_cache.h"

#include <cassert>
#include <cstring>
#include <string>

namespace engine {

// ---------------------------------------------------------------------------
// KvBlockPool

Result<std::unique_ptr<KvBlockPool>> KvBlockPool::create(const KvGeometry& geom, Backend& backend) {
  if (geom.num_layers <= 0 || geom.num_kv_heads <= 0 || geom.head_dim <= 0 || geom.head_dim_v <= 0 ||
      geom.block_size <= 0 || geom.num_blocks <= 0) {
    return InvalidArgument("KvBlockPool: invalid geometry");
  }
  if (geom.dtype != DType::kF32 && geom.dtype != DType::kF16) {
    return Unsupported("KvBlockPool: dtype " + std::string(dtype_name(geom.dtype)) + " not supported");
  }
  auto pool = std::unique_ptr<KvBlockPool>(new KvBlockPool());
  pool->geom_ = geom;
  pool->backend_ = &backend;
  const auto elem = static_cast<size_t>(dtype_block_bytes(geom.dtype));
  const size_t k_bytes = static_cast<size_t>(geom.k_block_elems() * geom.num_blocks) * elem;
  const size_t v_bytes = static_cast<size_t>(geom.v_block_elems() * geom.num_blocks) * elem;
  for (int32_t l = 0; l < geom.num_layers; ++l) {
    ENGINE_ASSIGN_OR_RETURN(auto k, backend.allocate(k_bytes));
    ENGINE_ASSIGN_OR_RETURN(auto v, backend.allocate(v_bytes));
    pool->k_.push_back(std::move(k));
    pool->v_.push_back(std::move(v));
  }
  pool->refs_ = std::make_unique<std::atomic<int32_t>[]>(static_cast<size_t>(geom.num_blocks));
  pool->free_.reserve(static_cast<size_t>(geom.num_blocks));
  for (int32_t b = geom.num_blocks - 1; b >= 0; --b) pool->free_.push_back(b);  // lowest ids first
  return pool;
}

int32_t KvBlockPool::free_blocks() const {
  std::lock_guard<std::mutex> lock(free_mu_);
  return static_cast<int32_t>(free_.size());
}

Result<int32_t> KvBlockPool::allocate() {
  int32_t b;
  {
    std::lock_guard<std::mutex> lock(free_mu_);
    if (free_.empty()) return ResourceExhausted("KV cache is full");
    b = free_.back();
    free_.pop_back();
  }
  refs_[static_cast<size_t>(b)].store(1, std::memory_order_release);
  return b;
}

void KvBlockPool::retain(int32_t block) {
  [[maybe_unused]] const int32_t prev = refs_[static_cast<size_t>(block)].fetch_add(1, std::memory_order_relaxed);
  assert(prev > 0 && "retain of a free KV block");
}

void KvBlockPool::release(int32_t block) {
  // acq_rel: writes made by any holder happen-before the block is reused.
  const int32_t prev = refs_[static_cast<size_t>(block)].fetch_sub(1, std::memory_order_acq_rel);
  assert(prev > 0 && "double release of a KV block");
  if (prev == 1) {
    std::lock_guard<std::mutex> lock(free_mu_);
    free_.push_back(block);
  }
}

void KvBlockPool::copy_block(int32_t dst, int32_t src) {
  const auto elem = static_cast<size_t>(dtype_block_bytes(geom_.dtype));
  const size_t kb = static_cast<size_t>(geom_.k_block_elems()) * elem;
  const size_t vb = static_cast<size_t>(geom_.v_block_elems()) * elem;
  for (int32_t l = 0; l < geom_.num_layers; ++l) {
    auto* k = static_cast<std::byte*>(k_[static_cast<size_t>(l)]->data());
    auto* v = static_cast<std::byte*>(v_[static_cast<size_t>(l)]->data());
    backend_->copy(k + static_cast<size_t>(dst) * kb, k + static_cast<size_t>(src) * kb, kb);
    backend_->copy(v + static_cast<size_t>(dst) * vb, v + static_cast<size_t>(src) * vb, vb);
  }
}

// ---------------------------------------------------------------------------
// KvBlockTable

KvBlockTable& KvBlockTable::operator=(KvBlockTable&& o) noexcept {
  if (this != &o) {
    release();
    pool_ = o.pool_;
    blocks_ = std::move(o.blocks_);
    o.blocks_.clear();
  }
  return *this;
}

Status KvBlockTable::reserve(int64_t n_tokens) {
  while (capacity() < n_tokens) {
    ENGINE_ASSIGN_OR_RETURN(int32_t b, pool_->allocate());
    blocks_.push_back(b);
  }
  return Status::Ok();
}

void KvBlockTable::truncate(int64_t n_tokens) {
  const int32_t bs = pool_->geometry().block_size;
  const auto keep = static_cast<size_t>((n_tokens + bs - 1) / bs);
  while (blocks_.size() > keep) {
    pool_->release(blocks_.back());
    blocks_.pop_back();
  }
}

KvBlockTable KvBlockTable::clone() const {
  KvBlockTable t(*pool_);
  t.blocks_ = blocks_;
  for (int32_t b : blocks_) pool_->retain(b);
  return t;
}

Status KvBlockTable::make_writable(int64_t begin, int64_t end) {
  if (end <= begin) return Status::Ok();
  const int32_t bs = pool_->geometry().block_size;
  const auto first = static_cast<size_t>(begin / bs);
  const auto last = static_cast<size_t>((end - 1) / bs);
  if (last >= blocks_.size()) return InvalidArgument("make_writable beyond reserved KV");
  for (size_t i = first; i <= last; ++i) {
    const int32_t b = blocks_[i];
    if (pool_->ref_count(b) == 1) continue;  // already exclusive
    ENGINE_ASSIGN_OR_RETURN(int32_t copy, pool_->allocate());
    pool_->copy_block(copy, b);
    pool_->release(b);
    blocks_[i] = copy;
  }
  return Status::Ok();
}

void KvBlockTable::release() {
  for (int32_t b : blocks_) pool_->release(b);
  blocks_.clear();
}

}  // namespace engine

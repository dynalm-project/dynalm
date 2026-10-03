#pragma once

// Paged KV cache.
//
//   KvBlockPool  — owns the physical K/V storage (one pool per layer, all
//                  allocated up front) and per-block reference counts.
//   KvBlockTable — one sequence's logical→physical block map.
//
// A block is returned to the free list only when its last reference is
// released, so sequences (and, later, the prefix cache) can share blocks
// safely. Sharing is read-only: before writing into a shared block, a table
// calls make_writable(), which copies the block (copy-on-write).
//
// Thread safety: all pool operations are safe from multiple threads.
// Refcount changes are atomic; the free list takes a short mutex (see DD-024
// for the measured contention). A KvBlockTable is used by one thread at a time.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "backends/backend.h"
#include "common/status.h"
#include "kv_cache/kv_layout.h"

namespace engine {

class KvBlockPool {
 public:
  static Result<std::unique_ptr<KvBlockPool>> create(const KvGeometry& geom, Backend& backend);

  const KvGeometry& geometry() const { return geom_; }
  int32_t num_blocks() const { return geom_.num_blocks; }
  int32_t free_blocks() const;
  int32_t used_blocks() const { return num_blocks() - free_blocks(); }

  // New block with refcount 1; kResourceExhausted when none is free.
  Result<int32_t> allocate();
  // Adds a reference to an allocated block.
  void retain(int32_t block);
  // Drops a reference; the block becomes free at zero.
  void release(int32_t block);
  int32_t ref_count(int32_t block) const {
    return refs_[static_cast<size_t>(block)].load(std::memory_order_acquire);
  }

  // Copies K and V of every layer from src to dst (through the backend:
  // KV may live in device memory, DD-045).
  void copy_block(int32_t dst, int32_t src);

  KvLayerView layer_view(int32_t layer, std::span<const int32_t> block_table) const {
    return KvLayerView{k_[static_cast<size_t>(layer)]->data(), v_[static_cast<size_t>(layer)]->data(), &geom_,
                       block_table};
  }

 private:
  KvBlockPool() = default;

  KvGeometry geom_;
  Backend* backend_ = nullptr;
  std::vector<std::shared_ptr<Storage>> k_, v_;
  std::unique_ptr<std::atomic<int32_t>[]> refs_;
  mutable std::mutex free_mu_;
  std::vector<int32_t> free_;  // LIFO: recently freed blocks are cache-warm
};

class KvBlockTable {
 public:
  explicit KvBlockTable(KvBlockPool& pool) : pool_(&pool) {}
  ~KvBlockTable() { release(); }
  KvBlockTable(KvBlockTable&& o) noexcept : pool_(o.pool_), blocks_(std::move(o.blocks_)) { o.blocks_.clear(); }
  KvBlockTable& operator=(KvBlockTable&& o) noexcept;
  KvBlockTable(const KvBlockTable&) = delete;
  KvBlockTable& operator=(const KvBlockTable&) = delete;

  // Maps new blocks until positions [0, n_tokens) are covered.
  Status reserve(int64_t n_tokens);
  // Unmaps blocks no longer needed for positions [0, n_tokens) (KV rollback).
  void truncate(int64_t n_tokens);
  // Maps an already-retained shared block at the end (ownership of that
  // reference moves to this table). Used by the prefix cache.
  void append_shared(int32_t block) { blocks_.push_back(block); }
  // New table sharing every block of this one (fork).
  KvBlockTable clone() const;
  // Ensures the blocks covering positions [begin, end) are exclusively owned,
  // copying shared ones. Must be called before writing K/V there.
  Status make_writable(int64_t begin, int64_t end);
  // Drops all references.
  void release();

  std::span<const int32_t> block_table() const { return blocks_; }
  int32_t num_blocks() const { return static_cast<int32_t>(blocks_.size()); }
  int64_t capacity() const { return static_cast<int64_t>(blocks_.size()) * pool_->geometry().block_size; }
  KvBlockPool& pool() const { return *pool_; }

 private:
  KvBlockPool* pool_;
  std::vector<int32_t> blocks_;
};

}  // namespace engine

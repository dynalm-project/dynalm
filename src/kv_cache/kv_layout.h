#pragma once

// Physical KV-cache layout shared by the cache manager and attention kernels.
//
// Storage is block-based from the start: each layer owns a pool of blocks,
// and a sequence maps logical token positions to blocks through a block
// table. Within a block, K is laid out [kv_head][slot][head_dim] so one head's
// keys are contiguous across tokens (the access pattern of decode attention).
//
//   K(layer, block, head, slot) = k_base[layer]
//       + ((block * n_kv_heads + head) * block_size + slot) * head_dim
//
// V uses the same formula with head_dim_v. The contiguous single-sequence
// case is simply a block table [0, 1, 2, ...].

#include <cstdint>
#include <span>

#include "dtype/dtype.h"

namespace engine {

struct KvGeometry {
  int32_t num_layers = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t head_dim_v = 0;
  int32_t block_size = 16;
  int32_t num_blocks = 0;
  DType dtype = DType::kF32;

  int64_t k_block_elems() const { return static_cast<int64_t>(num_kv_heads) * block_size * head_dim; }
  int64_t v_block_elems() const { return static_cast<int64_t>(num_kv_heads) * block_size * head_dim_v; }
  int64_t bytes_per_layer() const {
    return (k_block_elems() + v_block_elems()) * num_blocks * dtype_block_bytes(dtype);
  }
  int64_t total_bytes() const { return bytes_per_layer() * num_layers; }
};

// Everything a kernel needs to address one layer of one sequence's KV.
struct KvLayerView {
  void* k = nullptr;  // layer base
  void* v = nullptr;
  const KvGeometry* geom = nullptr;
  std::span<const int32_t> block_table;  // logical block -> physical block

  int64_t k_offset(int64_t pos, int32_t head) const {
    const int32_t bs = geom->block_size;
    const int64_t block = block_table[static_cast<size_t>(pos / bs)];
    return ((block * geom->num_kv_heads + head) * bs + pos % bs) * geom->head_dim;
  }
  int64_t v_offset(int64_t pos, int32_t head) const {
    const int32_t bs = geom->block_size;
    const int64_t block = block_table[static_cast<size_t>(pos / bs)];
    return ((block * geom->num_kv_heads + head) * bs + pos % bs) * geom->head_dim_v;
  }
};

}  // namespace engine

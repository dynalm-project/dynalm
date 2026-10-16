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
//
// Slot-major layout (GPU devices, docs/gpu-backend.md): within a block, each
// slot holds all KV heads of one token, [slot][kv_head][head_dim], so every
// physical slot ("cell") is one row of n_kv_heads * head_dim elements and a
// layer's whole pool is one strided [head_dim, cells, kv_heads] view:
//
//   K(layer, block, head, slot) = k_base[layer]
//       + ((block * block_size + slot) * n_kv_heads + head) * head_dim
//
// Attention can then read the pool in place, with the block table expressed
// as a mask over cells. Blocks, block tables, sharing and copy-on-write are
// identical in both layouts; only the order inside a block differs. CPU
// kernels use the head-major layout.
//
// Element types: f32, f16, or q8_0 (DD-074). Offsets are in elements; for a
// block format they are always multiples of its 32-element block (head_dim
// must be a multiple of 32), and byte_offset() converts.

#include <cstddef>
#include <cstdint>
#include <span>

#include "dynacore/tensor/dtype.h"

namespace dynacore {

enum class KvLayout : uint8_t {
  kHeadMajor,  // [block][kv_head][slot][dim] (CPU kernels)
  kSlotMajor,  // [block][slot][kv_head][dim] (GPU, in-place paged attention)
};

struct KvGeometry {
  int32_t num_layers = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t head_dim_v = 0;
  int32_t block_size = 16;
  int32_t num_blocks = 0;
  DType dtype = DType::kF32;
  KvLayout layout = KvLayout::kHeadMajor;

  int64_t k_block_elems() const { return static_cast<int64_t>(num_kv_heads) * block_size * head_dim; }
  int64_t v_block_elems() const { return static_cast<int64_t>(num_kv_heads) * block_size * head_dim_v; }
  // Bytes of `elems` consecutive elements (block formats: whole blocks).
  int64_t bytes_of(int64_t elems) const { return elems / dtype_block_elems(dtype) * dtype_block_bytes(dtype); }
  int64_t k_block_bytes() const { return bytes_of(k_block_elems()); }
  int64_t v_block_bytes() const { return bytes_of(v_block_elems()); }
  int64_t bytes_per_layer() const { return (k_block_bytes() + v_block_bytes()) * num_blocks; }
  int64_t total_bytes() const { return bytes_per_layer() * num_layers; }
};

// Everything a kernel needs to address one layer of one sequence's KV.
struct KvLayerView {
  void* k = nullptr;  // layer base
  void* v = nullptr;
  const KvGeometry* geom = nullptr;
  std::span<const int32_t> block_table;  // logical block -> physical block

  int64_t k_offset(int64_t pos, int32_t head) const { return row(pos, head) * geom->head_dim; }
  int64_t v_offset(int64_t pos, int32_t head) const { return row(pos, head) * geom->head_dim_v; }
  // Index of the head_dim-element row holding (pos, head) in the layer pool.
  int64_t row(int64_t pos, int32_t head) const {
    const int32_t bs = geom->block_size;
    const int64_t block = block_table[static_cast<size_t>(pos / bs)];
    if (geom->layout == KvLayout::kSlotMajor) return (block * bs + pos % bs) * geom->num_kv_heads + head;
    return (block * geom->num_kv_heads + head) * bs + pos % bs;
  }
  // Address of element offset `off` (from k_offset / v_offset) in a base.
  void* k_at(int64_t off) const { return static_cast<std::byte*>(k) + geom->bytes_of(off); }
  void* v_at(int64_t off) const { return static_cast<std::byte*>(v) + geom->bytes_of(off); }
};

}  // namespace dynacore

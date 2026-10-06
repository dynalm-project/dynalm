#pragma once

// Element / storage types.
//
// A DType describes how a tensor's bytes are laid out, including block-
// quantized formats where a fixed number of elements share scales inside one
// packed block. The enum is engine-owned: file formats (GGUF, SafeTensors)
// map their own type IDs onto it in their loaders.
//
// Weight, activation, accumulator and KV precisions are chosen independently
// per model/op (see PrecisionConfig in model_ir); a DType is only the storage
// type of one tensor.

#include <cstdint>
#include <string_view>

namespace engine {

enum class DType : uint8_t {
  kF32 = 0,
  kF16,
  kBF16,
  kI8,
  kI16,
  kI32,
  // Block-quantized (GGML-compatible block layouts).
  kQ4_0,
  kQ4_1,
  kQ5_0,
  kQ5_1,
  kQ8_0,
  kQ8_1,
  kQ2_K,
  kQ3_K,
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_K,
  kCount,
};

struct DTypeInfo {
  std::string_view name;
  int32_t block_elems;  // elements per block (1 for scalar types)
  int32_t block_bytes;  // bytes per block
  bool quantized;
  bool floating;        // scalar floating-point type
};

const DTypeInfo& dtype_info(DType t);

inline std::string_view dtype_name(DType t) { return dtype_info(t).name; }
inline bool dtype_is_quantized(DType t) { return dtype_info(t).quantized; }
inline int32_t dtype_block_elems(DType t) { return dtype_info(t).block_elems; }
inline int32_t dtype_block_bytes(DType t) { return dtype_info(t).block_bytes; }

// Bytes occupied by `n` consecutive elements. For quantized types `n` must be
// a multiple of the block size; returns -1 otherwise.
int64_t dtype_row_bytes(DType t, int64_t n);

// Parses a name as printed by dtype_name ("f32", "q4_K", ...).
bool parse_dtype(std::string_view s, DType& out);

}  // namespace engine

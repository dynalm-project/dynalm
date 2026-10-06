#pragma once

// Row conversion to fp32: the reference path for every weight dtype.
//
// Kernels use these for embedding lookups, as the generic fallback, and as the
// oracle that optimized quantized dot products are tested against.

#include <cstdint>

#include "dynacore/tensor/dtype.h"

namespace engine {

// Converts `n` elements starting at `src` to fp32. `n` must be a multiple of
// the dtype's block size. Returns false for dtypes without a converter.
bool dequantize_row(DType type, const void* src, float* dst, int64_t n);

bool dequant_supported(DType type);

}  // namespace engine

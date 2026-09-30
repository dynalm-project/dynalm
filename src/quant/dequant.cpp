#include "quant/dequant.h"

#include <cstring>

#include "dtype/fp16.h"

namespace engine {

bool dequant_supported(DType type) {
  switch (type) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
      return true;
    default:
      return false;
  }
}

bool dequantize_row(DType type, const void* src, float* dst, int64_t n) {
  switch (type) {
    case DType::kF32:
      std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
      return true;
    case DType::kF16: {
      const auto* h = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(h[i]);
      return true;
    }
    case DType::kBF16: {
      const auto* h = static_cast<const uint16_t*>(src);
      for (int64_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(h[i]);
      return true;
    }
    default:
      return false;
  }
}

}  // namespace engine

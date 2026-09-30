#include "quant/dequant.h"

#include <cstring>

#include "dtype/fp16.h"
#include "quant/quant_formats.h"

namespace engine {

bool dequant_supported(DType type) {
  switch (type) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kQ4_0:
    case DType::kQ4_1:
    case DType::kQ5_0:
    case DType::kQ5_1:
    case DType::kQ8_0:
    case DType::kQ8_1:
    case DType::kQ2_K:
    case DType::kQ3_K:
    case DType::kQ4_K:
    case DType::kQ5_K:
    case DType::kQ6_K:
    case DType::kQ8_K:
      return true;
    default:
      return false;
  }
}

bool dequantize_row(DType type, const void* src, float* dst, int64_t n) {
  using namespace quant;
  const int64_t nb = n / dtype_block_elems(type);
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
    case DType::kQ4_0: dequantize_q4_0(static_cast<const BlockQ4_0*>(src), dst, nb); return true;
    case DType::kQ4_1: dequantize_q4_1(static_cast<const BlockQ4_1*>(src), dst, nb); return true;
    case DType::kQ5_0: dequantize_q5_0(static_cast<const BlockQ5_0*>(src), dst, nb); return true;
    case DType::kQ5_1: dequantize_q5_1(static_cast<const BlockQ5_1*>(src), dst, nb); return true;
    case DType::kQ8_0: dequantize_q8_0(static_cast<const BlockQ8_0*>(src), dst, nb); return true;
    case DType::kQ8_1: dequantize_q8_1(static_cast<const BlockQ8_1*>(src), dst, nb); return true;
    case DType::kQ2_K: dequantize_q2_K(static_cast<const BlockQ2_K*>(src), dst, nb); return true;
    case DType::kQ3_K: dequantize_q3_K(static_cast<const BlockQ3_K*>(src), dst, nb); return true;
    case DType::kQ4_K: dequantize_q4_K(static_cast<const BlockQ4_K*>(src), dst, nb); return true;
    case DType::kQ5_K: dequantize_q5_K(static_cast<const BlockQ5_K*>(src), dst, nb); return true;
    case DType::kQ6_K: dequantize_q6_K(static_cast<const BlockQ6_K*>(src), dst, nb); return true;
    case DType::kQ8_K: dequantize_q8_K(static_cast<const BlockQ8_K*>(src), dst, nb); return true;
    default:
      return false;
  }
}

}  // namespace engine

#pragma once

// GGML-compatible block quantization formats: block layouts and scalar
// reference dequantization.
//
// These are the correctness oracles for the whole quantization subsystem:
// optimized kernels (quantized dot products, SIMD, repacked layouts) are tested
// against them, and they are themselves tested against gguf-py's independent
// NumPy implementations (tests/data/quant, tools/make_quant_fixtures.py).

#include <cstdint>

namespace engine::quant {

inline constexpr int kQK = 32;     // legacy block size
inline constexpr int kQK_K = 256;  // k-quant super-block size

#pragma pack(push, 1)
struct BlockQ4_0 { uint16_t d; uint8_t qs[kQK / 2]; };
struct BlockQ4_1 { uint16_t d, m; uint8_t qs[kQK / 2]; };
struct BlockQ5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[kQK / 2]; };
struct BlockQ5_1 { uint16_t d, m; uint8_t qh[4]; uint8_t qs[kQK / 2]; };
struct BlockQ8_0 { uint16_t d; int8_t qs[kQK]; };
struct BlockQ8_1 { uint16_t d, s; int8_t qs[kQK]; };
struct BlockQ2_K { uint8_t scales[kQK_K / 16]; uint8_t qs[kQK_K / 4]; uint16_t d, dmin; };
struct BlockQ3_K { uint8_t hmask[kQK_K / 8]; uint8_t qs[kQK_K / 4]; uint8_t scales[12]; uint16_t d; };
struct BlockQ4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[kQK_K / 2]; };
struct BlockQ5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[kQK_K / 8]; uint8_t qs[kQK_K / 2]; };
struct BlockQ6_K { uint8_t ql[kQK_K / 2]; uint8_t qh[kQK_K / 4]; int8_t scales[kQK_K / 16]; uint16_t d; };
struct BlockQ8_K { float d; int8_t qs[kQK_K]; int16_t bsums[kQK_K / 16]; };
#pragma pack(pop)

static_assert(sizeof(BlockQ4_0) == 18 && sizeof(BlockQ4_1) == 20 && sizeof(BlockQ5_0) == 22 &&
              sizeof(BlockQ5_1) == 24 && sizeof(BlockQ8_0) == 34 && sizeof(BlockQ8_1) == 36);
static_assert(sizeof(BlockQ2_K) == 84 && sizeof(BlockQ3_K) == 110 && sizeof(BlockQ4_K) == 144 &&
              sizeof(BlockQ5_K) == 176 && sizeof(BlockQ6_K) == 210 && sizeof(BlockQ8_K) == 292);

// Dequantize `nb` consecutive blocks into `y` (nb * block_elems floats).
void dequantize_q4_0(const BlockQ4_0* x, float* y, int64_t nb);
void dequantize_q4_1(const BlockQ4_1* x, float* y, int64_t nb);
void dequantize_q5_0(const BlockQ5_0* x, float* y, int64_t nb);
void dequantize_q5_1(const BlockQ5_1* x, float* y, int64_t nb);
void dequantize_q8_0(const BlockQ8_0* x, float* y, int64_t nb);
void dequantize_q8_1(const BlockQ8_1* x, float* y, int64_t nb);
void dequantize_q2_K(const BlockQ2_K* x, float* y, int64_t nb);
void dequantize_q3_K(const BlockQ3_K* x, float* y, int64_t nb);
void dequantize_q4_K(const BlockQ4_K* x, float* y, int64_t nb);
void dequantize_q5_K(const BlockQ5_K* x, float* y, int64_t nb);
void dequantize_q6_K(const BlockQ6_K* x, float* y, int64_t nb);
void dequantize_q8_K(const BlockQ8_K* x, float* y, int64_t nb);

// 6-bit scale/min unpacking shared by Q4_K and Q5_K (8 sub-blocks).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

}  // namespace engine::quant

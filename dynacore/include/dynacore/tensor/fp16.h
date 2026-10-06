#pragma once

// Scalar IEEE fp16 / bfloat16 <-> fp32 conversion.
//
// Branch-light bit manipulation (after Maratyszcza's FP16 library): exact,
// round-to-nearest-even, handles subnormals, infinities and NaN. SIMD backends
// use hardware conversion (F16C etc.); these are the portable reference and
// the generic-backend implementation.
//
// Requires IEEE-conforming float arithmetic: do not build with /fp:fast or
// -ffast-math.

#include <bit>
#include <cstdint>

namespace dynacore {

inline float fp16_to_fp32(uint16_t h) {
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;

  // Normal numbers: shift the exponent/mantissa into fp32 position, then fix
  // the exponent bias with a multiply (also maps fp16 inf/NaN to fp32 inf/NaN).
  constexpr uint32_t exp_offset = 0xE0u << 23;
  constexpr float exp_scale = 0x1.0p-112f;
  const float normalized = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;

  // Subnormals: place the mantissa in a float with exponent 2^-1 and subtract
  // the implicit 0.5, which leaves m * 2^-24 exactly.
  constexpr uint32_t magic_mask = 126u << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized = std::bit_cast<float>((two_w >> 17) | magic_mask) - magic_bias;

  constexpr uint32_t denormalized_cutoff = 1u << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? std::bit_cast<uint32_t>(denormalized)
                                          : std::bit_cast<uint32_t>(normalized));
  return std::bit_cast<float>(result);
}

inline uint16_t fp32_to_fp16(float f) {
  // Scale into the fp16 range so the FPU performs the mantissa rounding
  // (round-to-nearest-even) and overflow to infinity for us.
  constexpr float scale_to_inf = 0x1.0p+112f;
  constexpr float scale_to_zero = 0x1.0p-110f;
  const uint32_t w = std::bit_cast<uint32_t>(f);
  const float abs_f = std::bit_cast<float>(w & 0x7FFFFFFFu);
  float base = (abs_f * scale_to_inf) * scale_to_zero;

  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & 0x80000000u;
  uint32_t bias = shl1_w & 0xFF000000u;
  if (bias < 0x71000000u) bias = 0x71000000u;

  base = std::bit_cast<float>((bias >> 1) + 0x07800000u) + base;
  const uint32_t bits = std::bit_cast<uint32_t>(base);
  const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
  const uint32_t mantissa_bits = bits & 0x00000FFFu;
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

inline float bf16_to_fp32(uint16_t h) { return std::bit_cast<float>(static_cast<uint32_t>(h) << 16); }

inline uint16_t fp32_to_bf16(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<uint16_t>((u >> 16) | 0x40u);  // NaN: keep sign, force quiet
  }
  // Round to nearest even on the dropped 16 bits.
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace dynacore

#pragma once

// exp(x) for x <= 0, without libm (DD-043, reused by attention in DD-056).
//
// Cody-Waite range reduction (x = n ln2 + r) and a degree-6 polynomial for
// e^r; relative error ~1e-7 against std::exp. Plain multiplies and adds only:
// the same result on every compiler and ISA, and loops over arrays of it
// vectorize without fast-math. Inputs below -87 flush to e^-87 (~1.6e-38).

#include <algorithm>
#include <bit>
#include <cstdint>

namespace dynacore {

inline float exp_nonpos(float x) {
  x = std::max(x, -87.0f);
  // Round to nearest without a libm call: adding 1.5 * 2^23 pushes the
  // fraction out of the mantissa (exact for |v| < 2^22).
  const float n = (x * 1.44269504088896341f + 12582912.0f) - 12582912.0f;
  const float r = x - n * 0.693145751953125f - n * 1.428606765330187e-06f;
  float p = 1.0f / 720;
  p = p * r + 1.0f / 120;
  p = p * r + 1.0f / 24;
  p = p * r + 1.0f / 6;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  const auto bits = static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23;
  return p * std::bit_cast<float>(bits);
}

}  // namespace dynacore

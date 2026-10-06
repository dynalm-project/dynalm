#include "dynacore/tensor/dtype.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "dynacore/tensor/fp16.h"

namespace dynacore {
namespace {

TEST(DType, BlockGeometryMatchesGgml) {
  struct Case {
    DType t;
    int elems, bytes;
  };
  const Case cases[] = {
      {DType::kF32, 1, 4},     {DType::kF16, 1, 2},     {DType::kBF16, 1, 2},
      {DType::kQ4_0, 32, 18},  {DType::kQ4_1, 32, 20},  {DType::kQ5_0, 32, 22},
      {DType::kQ5_1, 32, 24},  {DType::kQ8_0, 32, 34},  {DType::kQ8_1, 32, 36},
      {DType::kQ2_K, 256, 84}, {DType::kQ3_K, 256, 110}, {DType::kQ4_K, 256, 144},
      {DType::kQ5_K, 256, 176}, {DType::kQ6_K, 256, 210}, {DType::kQ8_K, 256, 292},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(std::string(dtype_name(c.t)));
    EXPECT_EQ(dtype_block_elems(c.t), c.elems);
    EXPECT_EQ(dtype_block_bytes(c.t), c.bytes);
    EXPECT_EQ(dtype_is_quantized(c.t), c.elems > 1);
  }
}

TEST(DType, RowBytes) {
  EXPECT_EQ(dtype_row_bytes(DType::kF32, 4096), 16384);
  EXPECT_EQ(dtype_row_bytes(DType::kQ4_K, 4096), 4096 / 256 * 144);
  EXPECT_EQ(dtype_row_bytes(DType::kQ8_0, 64), 68);
  EXPECT_EQ(dtype_row_bytes(DType::kQ8_0, 33), -1);  // not block-aligned
  EXPECT_EQ(dtype_row_bytes(DType::kF16, 0), 0);
}

TEST(DType, NameRoundTrip) {
  for (int i = 0; i < static_cast<int>(DType::kCount); ++i) {
    const auto t = static_cast<DType>(i);
    DType parsed = DType::kF32;
    ASSERT_TRUE(parse_dtype(dtype_name(t), parsed)) << dtype_name(t);
    EXPECT_EQ(parsed, t);
  }
  DType out;
  EXPECT_FALSE(parse_dtype("q4_k_m", out));  // a file-level mix, not a block type
}

TEST(Fp16, ExhaustiveRoundTrip) {
  for (uint32_t h = 0; h <= 0xFFFF; ++h) {
    const float f = fp16_to_fp32(static_cast<uint16_t>(h));
    const bool is_nan = (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0;
    if (is_nan) {
      ASSERT_TRUE(std::isnan(f)) << std::hex << h;
      ASSERT_TRUE(std::isnan(fp16_to_fp32(fp32_to_fp16(f))));
    } else {
      ASSERT_EQ(fp32_to_fp16(f), h) << std::hex << h;
    }
  }
}

TEST(Fp16, KnownValues) {
  EXPECT_EQ(fp32_to_fp16(1.0f), 0x3C00);
  EXPECT_EQ(fp32_to_fp16(-2.0f), 0xC000);
  EXPECT_EQ(fp32_to_fp16(65504.0f), 0x7BFF);                     // max finite
  EXPECT_EQ(fp32_to_fp16(65520.0f), 0x7C00);                     // rounds to +inf
  EXPECT_EQ(fp32_to_fp16(std::ldexp(1.0f, -24)), 0x0001);        // min subnormal
  EXPECT_EQ(fp32_to_fp16(std::ldexp(1.0f, -25)), 0x0000);        // tie -> even (0)
  EXPECT_EQ(fp32_to_fp16(std::ldexp(3.0f, -25)), 0x0002);        // tie 1.5 ulp -> even (2)
  EXPECT_EQ(fp32_to_fp16(std::numeric_limits<float>::infinity()), 0x7C00);
  EXPECT_EQ(fp32_to_fp16(-0.0f), 0x8000);
  EXPECT_EQ(fp16_to_fp32(0x3555), 0.333251953125f);
}

// Independent slow reference: nearest fp16 by exhaustive search over finite
// halves of the right sign, ties to even mantissa.
uint16_t fp16_reference(float f) {
  if (std::isnan(f)) return 0x7E00;
  const uint16_t sign = std::signbit(f) ? 0x8000 : 0;
  const double a = std::fabs(static_cast<double>(f));
  if (a >= 65520.0) return sign | 0x7C00;
  uint16_t best = 0;
  double best_err = INFINITY;
  for (uint16_t h = 0; h <= 0x7BFF; ++h) {
    const double v = fp16_to_fp32(h);
    const double err = std::fabs(v - a);
    if (err < best_err || (err == best_err && (h & 1) == 0)) {
      best = h;
      best_err = err;
    }
  }
  return sign | best;
}

TEST(Fp16, RandomMatchesReference) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> mag(-30.0f, 17.0f);
  for (int i = 0; i < 300; ++i) {
    const float f = std::ldexp(1.0f + (rng() % 1000000) / 1e6f, static_cast<int>(mag(rng))) *
                    ((rng() & 1) ? -1.0f : 1.0f);
    ASSERT_EQ(fp32_to_fp16(f), fp16_reference(f)) << f;
  }
}

TEST(Bf16, Conversions) {
  EXPECT_EQ(fp32_to_bf16(1.0f), 0x3F80);
  EXPECT_EQ(bf16_to_fp32(0x3F80), 1.0f);
  // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1.0078125: ties to even (1.0).
  EXPECT_EQ(fp32_to_bf16(1.0f + std::ldexp(1.0f, -8)), 0x3F80);
  // 1 + 3*2^-8 is halfway between 0x3F81 and 0x3F82: ties to even (0x3F82).
  EXPECT_EQ(fp32_to_bf16(1.0f + 3 * std::ldexp(1.0f, -8)), 0x3F82);
  EXPECT_TRUE(std::isnan(bf16_to_fp32(fp32_to_bf16(std::nanf("")))));
  EXPECT_EQ(fp32_to_bf16(std::numeric_limits<float>::infinity()), 0x7F80);
  for (uint32_t h = 0; h <= 0xFFFF; ++h) {
    const float f = bf16_to_fp32(static_cast<uint16_t>(h));
    if (!std::isnan(f)) {
      ASSERT_EQ(fp32_to_bf16(f), h);
    }
  }
}

}  // namespace
}  // namespace dynacore

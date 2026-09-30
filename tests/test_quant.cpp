// Dequantization vs gguf-py's independent NumPy implementations
// (fixtures from tools/make_quant_fixtures.py).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iterator>

#include "dtype/fp16.h"
#include "quant/dequant.h"
#include "quant/quant_formats.h"

namespace engine {
namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

class Dequant : public ::testing::TestWithParam<DType> {};

TEST_P(Dequant, MatchesGgufPy) {
  const DType t = GetParam();
  const std::string base = std::string(ENGINE_TEST_DATA_DIR) + "/quant/" + std::string(dtype_name(t));
  const auto raw = read_file(base + ".bin");
  const auto exp = read_file(base + ".f32");
  ASSERT_FALSE(raw.empty()) << base;
  ASSERT_EQ(raw.size() % static_cast<size_t>(dtype_block_bytes(t)), 0u);
  const int64_t n = static_cast<int64_t>(raw.size()) / dtype_block_bytes(t) * dtype_block_elems(t);
  ASSERT_EQ(exp.size(), static_cast<size_t>(n) * 4);

  std::vector<float> got(static_cast<size_t>(n));
  ASSERT_TRUE(dequant_supported(t));
  ASSERT_TRUE(dequantize_row(t, raw.data(), got.data(), n));
  const auto* want = reinterpret_cast<const float*>(exp.data());
  for (int64_t i = 0; i < n; ++i) {
    // Same operations in the same order: results must be bit-identical up to
    // float rounding of the final multiply-add.
    ASSERT_NEAR(got[static_cast<size_t>(i)], want[i], 1e-6f + 1e-6f * std::abs(want[i]))
        << dtype_name(t) << " element " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Ggml, Dequant,
                         ::testing::Values(DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ5_1, DType::kQ8_0,
                                           DType::kQ2_K, DType::kQ3_K, DType::kQ4_K, DType::kQ5_K, DType::kQ6_K),
                         [](const ::testing::TestParamInfo<DType>& p) { return std::string(dtype_name(p.param)); });

TEST(Dequant, Q8KByHand) {
  quant::BlockQ8_K b{};
  b.d = 0.5f;
  for (int i = 0; i < quant::kQK_K; ++i) b.qs[i] = static_cast<int8_t>(i - 128);
  std::vector<float> y(quant::kQK_K);
  ASSERT_TRUE(dequantize_row(DType::kQ8_K, &b, y.data(), quant::kQK_K));
  EXPECT_FLOAT_EQ(y[0], -64.0f);
  EXPECT_FLOAT_EQ(y[255], 63.5f);
}

TEST(Dequant, Q8_0ByHand) {
  quant::BlockQ8_0 b{};
  b.d = fp32_to_fp16(0.25f);
  for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<int8_t>(i - 16);
  std::vector<float> y(32);
  ASSERT_TRUE(dequantize_row(DType::kQ8_0, &b, y.data(), 32));
  EXPECT_FLOAT_EQ(y[0], -4.0f);
  EXPECT_FLOAT_EQ(y[31], 3.75f);
}

}  // namespace
}  // namespace engine

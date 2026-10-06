#include "loader/gguf/gguf.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "dynacore/tensor/fp16.h"
#include "gguf_builder.h"
#include "test_models.h"

namespace engine::gguf {
namespace {

using engine::testing::GgufBuilder;

std::vector<uint8_t> f32_bytes(const std::vector<float>& v) {
  std::vector<uint8_t> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

GgufBuilder sample_builder() {
  GgufBuilder b;
  b.kv_str("general.architecture", "llama")
      .kv_u32("llama.block_count", 2)
      .kv_f32("llama.rope.freq_base", 10000.0f)
      .kv_i32("llama.signed_neg", -3)
      .kv_bool("tokenizer.ggml.add_bos_token", true)
      .kv_str_array("tokenizer.ggml.tokens", {"<s>", "</s>", "hello", "world"})
      .kv_f32_array("tokenizer.ggml.scores", {0.f, -1.f, -2.5f, -3.f})
      .kv_i32_array("tokenizer.ggml.token_type", {3, 3, 1, 1})
      // 2x3 f32 matrix: ne = {3, 2}
      .tensor("w", {3, 2}, 0, f32_bytes({1, 2, 3, 4, 5, 6}))
      // 64-element q8_0 row (2 blocks = 68 bytes), zeros
      .tensor("q", {64}, 8, std::vector<uint8_t>(68, 0))
      // iq4_xs: known geometry, not executable
      .tensor("iq", {256}, 23, std::vector<uint8_t>(136, 0));
  return b;
}

// Removes the temp files this test wrote (all mappings are closed by then).
class GgufTest : public ::testing::Test {
 protected:
  void TearDown() override {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(std::filesystem::temp_directory_path(), ec)) {
      if (e.path().filename().string().starts_with("engine_test_")) std::filesystem::remove(e.path(), ec);
    }
  }
};

Result<std::unique_ptr<GgufFile>> open_bytes(const std::vector<uint8_t>& bytes, const std::string& name) {
  return GgufFile::open(GgufBuilder::write_temp(bytes, name));
}

TEST_F(GgufTest, ParsesHeaderAndMetadata) {
  auto g = open_bytes(sample_builder().build(), "meta");
  ASSERT_TRUE(g.ok()) << g.status().to_string();
  const GgufFile& f = **g;
  EXPECT_EQ(f.version(), 3u);
  EXPECT_EQ(f.alignment(), 32u);
  EXPECT_EQ(f.metadata().size(), 8u);

  EXPECT_EQ(*f.get_string("general.architecture"), "llama");
  EXPECT_EQ(*f.get_uint("llama.block_count"), 2u);
  EXPECT_EQ(*f.get_int("llama.block_count"), 2);
  EXPECT_FLOAT_EQ(static_cast<float>(*f.get_float("llama.rope.freq_base")), 10000.0f);
  EXPECT_EQ(*f.get_int("llama.signed_neg"), -3);
  EXPECT_FALSE(f.get_uint("llama.signed_neg").ok());
  EXPECT_TRUE(*f.get_bool("tokenizer.ggml.add_bos_token"));

  EXPECT_EQ(f.get_uint("missing").status().code(), StatusCode::kNotFound);
  EXPECT_EQ(f.get_uint("general.architecture").status().code(), StatusCode::kInvalidArgument);
}

TEST_F(GgufTest, DecodesArrays) {
  auto g = open_bytes(sample_builder().build(), "arrays");
  ASSERT_TRUE(g.ok());
  auto toks = GgufFile::array_strings(*(*g)->get_array("tokenizer.ggml.tokens"));
  ASSERT_TRUE(toks.ok());
  ASSERT_EQ(toks->size(), 4u);
  EXPECT_EQ((*toks)[2], "hello");

  auto scores = GgufFile::array_floats(*(*g)->get_array("tokenizer.ggml.scores"));
  ASSERT_TRUE(scores.ok());
  EXPECT_FLOAT_EQ((*scores)[2], -2.5f);

  auto types = GgufFile::array_ints(*(*g)->get_array("tokenizer.ggml.token_type"));
  ASSERT_TRUE(types.ok());
  EXPECT_EQ((*types)[0], 3);

  EXPECT_FALSE(GgufFile::array_floats(*(*g)->get_array("tokenizer.ggml.tokens")).ok());
}

TEST_F(GgufTest, TensorDirectoryAndZeroCopyLoad) {
  auto g = open_bytes(sample_builder().build(), "tensors");
  ASSERT_TRUE(g.ok());
  const GgufFile& f = **g;
  ASSERT_EQ(f.tensors().size(), 3u);

  const TensorInfo* w = f.find_tensor("w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->shape, (TensorShape{2, 3}));  // reversed from ne={3,2}
  EXPECT_EQ(w->dtype, DType::kF32);
  EXPECT_EQ(w->nbytes, 24u);
  EXPECT_EQ(w->offset % f.alignment(), 0u);

  auto t = f.load_tensor(*w);
  ASSERT_TRUE(t.ok()) << t.status().to_string();
  EXPECT_EQ(t->data_as<float>()[4], 5.0f);  // row 1, col 1
  // Zero-copy: data lies inside the mapping.
  EXPECT_GE(static_cast<const std::byte*>(t->data()), f.file().data());
  EXPECT_LT(static_cast<const std::byte*>(t->data()), f.file().data() + f.file().size());

  const TensorInfo* q = f.find_tensor("q");
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->dtype, DType::kQ8_0);
  EXPECT_EQ(q->nbytes, 68u);

  const TensorInfo* iq = f.find_tensor("iq");
  ASSERT_NE(iq, nullptr);
  EXPECT_FALSE(iq->dtype.has_value());
  EXPECT_EQ(ggml_type_name(iq->ggml_type), "iq4_xs");
  EXPECT_EQ(f.load_tensor(*iq).status().code(), StatusCode::kUnsupported);

  EXPECT_EQ(f.total_tensor_bytes(), 24u + 68u + 136u);
}

TEST_F(GgufTest, TensorOutlivesFileObject) {
  auto g = open_bytes(sample_builder().build(), "lifetime");
  ASSERT_TRUE(g.ok());
  auto t = (*g)->load_tensor(*(*g)->find_tensor("w"));
  ASSERT_TRUE(t.ok());
  g = Status(StatusCode::kCancelled, "drop");  // destroy the GgufFile
  EXPECT_EQ(t->data_as<float>()[5], 6.0f);     // mapping kept alive by the tensor
}

TEST_F(GgufTest, CustomAlignment) {
  GgufBuilder b;
  b.kv_u32("general.alignment", 64).tensor("a", {4}, 0, f32_bytes({1, 2, 3, 4}))
      .tensor("b", {4}, 0, f32_bytes({5, 6, 7, 8}));
  auto g = open_bytes(b.build(64), "align64");
  ASSERT_TRUE(g.ok()) << g.status().to_string();
  EXPECT_EQ((*g)->alignment(), 64u);
  auto t = (*g)->load_tensor(*(*g)->find_tensor("b"));
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->data_as<float>()[0], 5.0f);
}

TEST_F(GgufTest, RejectsBadMagicAndVersion) {
  auto bytes = sample_builder().build();
  bytes[0] = 'X';
  EXPECT_EQ(open_bytes(bytes, "badmagic").status().code(), StatusCode::kCorrupt);

  GgufBuilder v1(1);
  EXPECT_EQ(open_bytes(v1.build(), "v1").status().code(), StatusCode::kUnsupported);
}

TEST_F(GgufTest, MissingFileIsNotFound) {
  EXPECT_EQ(GgufFile::open("definitely/not/here.gguf").status().code(), StatusCode::kNotFound);
}

// Every truncation of a valid file must fail cleanly (no crash, no UB).
TEST_F(GgufTest, EveryTruncationFailsCleanly) {
  const auto bytes = sample_builder().build();
  for (size_t len = 1; len < bytes.size(); ++len) {
    std::vector<uint8_t> cut(bytes.begin(), bytes.begin() + len);
    auto g = open_bytes(cut, "trunc");
    ASSERT_FALSE(g.ok()) << "accepted truncation at " << len;
  }
}

// Random byte corruption must never crash; it may or may not parse.
TEST_F(GgufTest, RandomCorruptionNeverCrashes) {
  const auto bytes = sample_builder().build();
  std::srand(42);
  for (int iter = 0; iter < 300; ++iter) {
    auto mutated = bytes;
    for (int k = 0; k < 4; ++k) mutated[std::rand() % mutated.size()] = static_cast<uint8_t>(std::rand());
    auto g = open_bytes(mutated, "fuzz");
    if (g.ok()) {
      for (const auto& t : (*g)->tensors()) (void)(*g)->load_tensor(t);
    }
  }
}

TEST_F(GgufTest, RejectsDuplicateKeysAndTensors) {
  GgufBuilder dk;
  dk.kv_u32("a", 1).kv_u32("a", 2);
  EXPECT_EQ(open_bytes(dk.build(), "dupkey").status().code(), StatusCode::kCorrupt);

  GgufBuilder dt;
  dt.tensor("t", {4}, 0, f32_bytes({1, 2, 3, 4})).tensor("t", {4}, 0, f32_bytes({1, 2, 3, 4}));
  EXPECT_EQ(open_bytes(dt.build(), "duptensor").status().code(), StatusCode::kCorrupt);
}

TEST_F(GgufTest, RejectsTensorPastEof) {
  GgufBuilder b;
  b.tensor("t", {1024}, 0, f32_bytes({1, 2, 3, 4}));  // claims 4 KiB, provides 16 bytes
  EXPECT_EQ(open_bytes(b.build(), "pasteof").status().code(), StatusCode::kCorrupt);
}

// Optional: parse a real model when ENGINE_TEST_MODEL points at a GGUF file.
TEST_F(GgufTest, RealModelIfAvailable) {
  const std::string path = engine::testing::smollm_model();
  if (!engine::testing::exists(path)) GTEST_SKIP() << "test model not present";
  auto g = GgufFile::open(path);
  ASSERT_TRUE(g.ok()) << g.status().to_string();
  EXPECT_TRUE((*g)->get_string("general.architecture").ok());
  EXPECT_GT((*g)->tensors().size(), 0u);
  for (const auto& t : (*g)->tensors()) {
    if (t.dtype) {
      auto tensor = (*g)->load_tensor(t);
      ASSERT_TRUE(tensor.ok()) << t.name;
    }
  }
}

}  // namespace
}  // namespace engine::gguf

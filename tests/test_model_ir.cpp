#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "gguf_builder.h"
#include "test_models.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"

namespace engine {
namespace {

using engine::testing::GgufBuilder;

ModelConfig valid_config() {
  ModelConfig c;
  c.architecture = "llama";
  c.vocab_size = 100;
  c.hidden_size = 64;
  c.intermediate_size = 128;
  c.num_layers = 2;
  c.num_heads = 4;
  c.num_kv_heads = 2;
  c.head_dim = 16;
  c.head_dim_v = 16;
  c.context_length = 256;
  c.rope.dim = 16;
  return c;
}

TEST(ModelConfig, ValidateAcceptsConsistentConfig) {
  EXPECT_TRUE(valid_config().validate().ok());
  EXPECT_EQ(valid_config().gqa_group(), 2);
}

TEST(ModelConfig, ValidateRejectsInconsistencies) {
  auto c = valid_config();
  c.num_kv_heads = 3;  // 4 % 3 != 0
  EXPECT_FALSE(c.validate().ok());
  c = valid_config();
  c.rope.dim = 32;  // > head_dim
  EXPECT_FALSE(c.validate().ok());
  c = valid_config();
  c.rope.dim = 15;  // odd
  EXPECT_FALSE(c.validate().ok());
  c = valid_config();
  c.num_layers = 0;
  EXPECT_FALSE(c.validate().ok());
  c = valid_config();
  c.sliding_layers = {true};  // wrong size
  EXPECT_FALSE(c.validate().ok());
}

TEST(ModelConfig, KvBytesPerToken) {
  auto c = valid_config();
  // 2 kv heads * (16 + 16) dims * 2 layers * 2 bytes (f16)
  EXPECT_EQ(c.kv_bytes_per_token(DType::kF16), 2 * 32 * 2 * 2);
  EXPECT_EQ(c.kv_bytes_per_token(DType::kF32), 2 * 32 * 2 * 4);
}

TEST(ModelConfig, SlidingWindowPerLayer) {
  auto c = valid_config();
  EXPECT_FALSE(c.layer_uses_sliding_window(0));
  c.sliding_window = 128;
  EXPECT_TRUE(c.layer_uses_sliding_window(1));  // no pattern = all layers
  c.sliding_layers = {true, false};
  EXPECT_TRUE(c.layer_uses_sliding_window(0));
  EXPECT_FALSE(c.layer_uses_sliding_window(1));
}

TEST(TensorRegistry, AddFindAndErrors) {
  TensorRegistry reg(2);
  auto t = Tensor::zeros(DType::kF32, {4});
  ASSERT_TRUE(t.ok());
  EXPECT_TRUE(reg.add(TensorRole::kTokenEmbedding, -1, *t).ok());
  EXPECT_TRUE(reg.add(TensorRole::kAttnQ, 1, *t).ok());
  EXPECT_EQ(reg.size(), 2u);
  EXPECT_TRUE(reg.has(TensorRole::kAttnQ, 1));
  EXPECT_FALSE(reg.has(TensorRole::kAttnQ, 0));
  EXPECT_EQ(reg.get(TensorRole::kAttnK, 0).status().code(), StatusCode::kNotFound);

  EXPECT_EQ(reg.add(TensorRole::kAttnQ, 1, *t).code(), StatusCode::kAlreadyExists);
  EXPECT_FALSE(reg.add(TensorRole::kAttnQ, 2, *t).ok());          // layer out of range
  EXPECT_FALSE(reg.add(TensorRole::kAttnQ, -1, *t).ok());         // per-layer role needs a layer
  EXPECT_FALSE(reg.add(TensorRole::kTokenEmbedding, 0, *t).ok()); // global role with layer
  EXPECT_EQ(reg.total_bytes(), 32);
}

TEST(GgufNames, ParseTensorName) {
  TensorRole role;
  int layer;
  ASSERT_TRUE(gguf::parse_tensor_name("token_embd.weight", role, layer));
  EXPECT_EQ(role, TensorRole::kTokenEmbedding);
  EXPECT_EQ(layer, -1);
  ASSERT_TRUE(gguf::parse_tensor_name("blk.12.attn_q.weight", role, layer));
  EXPECT_EQ(role, TensorRole::kAttnQ);
  EXPECT_EQ(layer, 12);
  ASSERT_TRUE(gguf::parse_tensor_name("blk.0.ffn_down.weight", role, layer));
  EXPECT_EQ(role, TensorRole::kFfnDown);
  ASSERT_TRUE(gguf::parse_tensor_name("blk.3.attn_k.bias", role, layer));
  EXPECT_EQ(role, TensorRole::kAttnKBias);

  EXPECT_FALSE(gguf::parse_tensor_name("blk.x.attn_q.weight", role, layer));
  EXPECT_FALSE(gguf::parse_tensor_name("blk.1.unknown.weight", role, layer));
  EXPECT_FALSE(gguf::parse_tensor_name("blk..attn_q.weight", role, layer));
  EXPECT_FALSE(gguf::parse_tensor_name("blk.-1.attn_q.weight", role, layer));
  EXPECT_FALSE(gguf::parse_tensor_name("blk.1", role, layer));
}

TEST(GgufNames, FileTypeNames) {
  EXPECT_EQ(gguf::file_type_name(15), "Q4_K_M");
  EXPECT_EQ(gguf::file_type_name(7), "Q8_0");
  EXPECT_EQ(gguf::file_type_name(999), "unknown");
}

std::vector<uint8_t> zeros_f32(size_t n) { return std::vector<uint8_t>(n * 4, 0); }

GgufBuilder tiny_llama() {
  GgufBuilder b;
  b.kv_str("general.architecture", "llama")
      .kv_str("general.name", "tiny")
      .kv_u32("llama.embedding_length", 8)
      .kv_u32("llama.block_count", 1)
      .kv_u32("llama.attention.head_count", 2)
      .kv_u32("llama.feed_forward_length", 16)
      .kv_u32("llama.context_length", 64)
      .kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-6f)
      .kv_f32("llama.rope.freq_base", 500000.0f)
      .kv_str_array("tokenizer.ggml.tokens", {"a", "b", "c", "d", "e"})
      .tensor("token_embd.weight", {8, 5}, 0, zeros_f32(40))
      .tensor("output_norm.weight", {8}, 0, zeros_f32(8))
      .tensor("blk.0.attn_q.weight", {8, 8}, 0, zeros_f32(64))
      .tensor("blk.0.mystery.weight", {8}, 0, zeros_f32(8));
  return b;
}

class ModelIrGguf : public ::testing::Test {
 protected:
  void TearDown() override {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(std::filesystem::temp_directory_path(), ec)) {
      if (e.path().filename().string().starts_with("engine_test_")) std::filesystem::remove(e.path(), ec);
    }
  }
};

TEST_F(ModelIrGguf, ReadsConfigWithDefaults) {
  auto g = gguf::GgufFile::open(GgufBuilder::write_temp(tiny_llama().build(), "cfg"));
  ASSERT_TRUE(g.ok()) << g.status().to_string();
  auto c = gguf::read_model_config(**g);
  ASSERT_TRUE(c.ok()) << c.status().to_string();
  EXPECT_EQ(c->architecture, "llama");
  EXPECT_EQ(c->name, "tiny");
  EXPECT_EQ(c->hidden_size, 8);
  EXPECT_EQ(c->num_layers, 1);
  EXPECT_EQ(c->num_heads, 2);
  EXPECT_EQ(c->num_kv_heads, 2);  // defaulted to head_count
  EXPECT_EQ(c->head_dim, 4);      // hidden / heads
  EXPECT_EQ(c->head_dim_v, 4);
  EXPECT_EQ(c->rope.dim, 4);
  EXPECT_FLOAT_EQ(c->rope.freq_base, 500000.0f);
  EXPECT_FLOAT_EQ(c->norm_eps, 1e-6f);
  EXPECT_EQ(c->norm, NormType::kRmsNorm);
  EXPECT_EQ(c->vocab_size, 5);    // from the tokenizer vocab
  EXPECT_TRUE(c->tied_embeddings);
  EXPECT_TRUE(c->validate().ok()) << c->validate().to_string();
}

TEST_F(ModelIrGguf, MissingRequiredKeyIsNotFound) {
  GgufBuilder b;
  b.kv_str("general.architecture", "llama").kv_u32("llama.block_count", 1);
  auto g = gguf::GgufFile::open(GgufBuilder::write_temp(b.build(), "missing"));
  ASSERT_TRUE(g.ok());
  EXPECT_EQ(gguf::read_model_config(**g).status().code(), StatusCode::kNotFound);
}

TEST_F(ModelIrGguf, MapsTensorsByRole) {
  auto g = gguf::GgufFile::open(GgufBuilder::write_temp(tiny_llama().build(), "map"));
  ASSERT_TRUE(g.ok());
  auto m = gguf::map_tensors(**g, 1);
  ASSERT_TRUE(m.ok()) << m.status().to_string();
  EXPECT_EQ(m->registry.size(), 3u);
  auto emb = m->registry.get(TensorRole::kTokenEmbedding);
  ASSERT_TRUE(emb.ok());
  EXPECT_EQ(emb->shape(), (TensorShape{5, 8}));  // [vocab, hidden]
  EXPECT_TRUE(m->registry.has(TensorRole::kAttnQ, 0));
  ASSERT_EQ(m->unmapped.size(), 1u);
  EXPECT_EQ(m->unmapped[0], "blk.0.mystery.weight");

  // A tensor for a layer beyond block_count is corrupt.
  EXPECT_EQ(gguf::map_tensors(**g, 0).status().code(), StatusCode::kCorrupt);
}

TEST(MemoryEstimate, AddsWeightsKvAndScratch) {
  auto c = valid_config();
  const auto m = estimate_memory(c, 1000, 256, DType::kF16, 16);
  EXPECT_EQ(m.weight_bytes, 1000);
  EXPECT_EQ(m.kv_bytes, c.kv_bytes_per_token(DType::kF16) * 256);
  EXPECT_GT(m.activation_bytes, 0);
  EXPECT_EQ(m.total(), m.weight_bytes + m.kv_bytes + m.activation_bytes);
}

TEST(ModelIrReal, SmolLm2ConfigIfAvailable) {
  const std::string path = engine::testing::smollm_model();
  if (!engine::testing::exists(path)) GTEST_SKIP() << "test model not present";
  auto g = gguf::GgufFile::open(path);
  ASSERT_TRUE(g.ok());
  auto c = gguf::read_model_config(**g);
  ASSERT_TRUE(c.ok()) << c.status().to_string();
  EXPECT_TRUE(c->validate().ok()) << c->validate().to_string();
  auto m = gguf::map_tensors(**g, c->num_layers);
  ASSERT_TRUE(m.ok()) << m.status().to_string();
  EXPECT_TRUE(m->unmapped.empty()) << m->unmapped.front();
  EXPECT_TRUE(m->registry.has(TensorRole::kTokenEmbedding));
}

}  // namespace
}  // namespace engine

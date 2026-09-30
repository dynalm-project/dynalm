// Every supported architecture, end to end, against the NumPy reference:
// tiny random-weight GGUFs + fixtures from tools/make_tiny_models.py.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

#include "backends/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/generator.h"
#include "sampling/sampler.h"
#include "test_models.h"

namespace engine {
namespace {

struct Ref {
  std::vector<TokenId> tokens, greedy;
  std::vector<std::pair<TokenId, float>> top;
};

Ref load_ref(const std::string& path) {
  Ref r;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "tokens:") {
      for (TokenId t; ss >> t;) r.tokens.push_back(t);
    } else if (key == "greedy:") {
      for (TokenId t; ss >> t;) r.greedy.push_back(t);
    } else if (key == "top:") {
      for (std::string p; ss >> p;) {
        const auto c = p.find(':');
        r.top.emplace_back(std::stoi(p.substr(0, c)), std::stof(p.substr(c + 1)));
      }
    }
  }
  return r;
}

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

class TinyArch : public ::testing::TestWithParam<std::string> {};

TEST_P(TinyArch, MatchesReference) {
  const std::string arch = GetParam();
  auto m = load_model(data("tiny_" + arch + ".gguf"));
  ASSERT_TRUE(m.ok()) << m.status().to_string();
  const ModelConfig& c = (*m)->config;
  const Ref ref = load_ref(data("ref_tiny_" + arch + ".txt"));
  ASSERT_FALSE(ref.tokens.empty());

  ThreadPool pool(3);
  CpuBackend be(pool, CpuIsa::kGeneric);
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 4, 16, DType::kF32};
  auto cache = KvCache::create(g, be);
  ASSERT_TRUE(cache.ok());

  // Logits after the prompt (single pass).
  auto t = Transformer::create(c, (*m)->weights, be, 32);
  ASSERT_TRUE(t.ok()) << t.status().to_string();
  {
    KvSequence seq(**cache);
    ASSERT_TRUE(seq.reserve(static_cast<int64_t>(ref.tokens.size())).ok());
    std::vector<int32_t> pos(ref.tokens.size());
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits(static_cast<size_t>(c.vocab_size));
    ASSERT_TRUE((*t)->forward(ref.tokens, pos, **cache, seq.block_table(), logits).ok());
    for (const auto& [id, want] : ref.top) {
      ASSERT_NEAR(logits[static_cast<size_t>(id)], want, 2e-3f + 1e-4f * std::abs(want))
          << arch << " token " << id;
    }
  }

  // Greedy continuation with chunked prefill (batch 5 < prompt 12).
  auto t5 = Transformer::create(c, (*m)->weights, be, 5);
  ASSERT_TRUE(t5.ok());
  Generator gen(**t5, **cache, *(*m)->tokenizer);
  GenerateOptions opts;
  opts.max_new_tokens = static_cast<int32_t>(ref.greedy.size());
  opts.stop_at_eog = false;
  std::vector<TokenId> out;
  ASSERT_TRUE(gen.generate(ref.tokens, opts, [&](TokenId id) { out.push_back(id); return true; }).ok());
  EXPECT_EQ(out, ref.greedy) << arch;
}

INSTANTIATE_TEST_SUITE_P(All, TinyArch,
                         ::testing::Values("llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"),
                         [](const ::testing::TestParamInfo<std::string>& p) { return p.param; });

// Real f16 checkpoints vs the reference (skipped when the files are absent;
// fetch with tools/fetch_models.sh f16).
struct RealCase {
  std::string model, fixture;
};

class RealArch : public ::testing::TestWithParam<int> {};

TEST_P(RealArch, MatchesReference) {
  const RealCase cases[] = {{engine::testing::qwen_f16_model(), "ref_qwen25.txt"},
                            {engine::testing::gemma_model(), "ref_gemma3.txt"},
                            {engine::testing::qwen_q4_model(), "ref_qwen25_q4_k_m.txt"}};
  const RealCase rc = cases[GetParam()];
  if (!engine::testing::exists(rc.model)) GTEST_SKIP() << rc.model << " not present";
  if (!std::filesystem::exists(data(rc.fixture))) GTEST_SKIP() << rc.fixture << " not generated";
  auto m = load_model(rc.model);
  ASSERT_TRUE(m.ok()) << m.status().to_string();
  const ModelConfig& c = (*m)->config;
  const Ref ref = load_ref(data(rc.fixture));

  ThreadPool pool(4);
  CpuBackend be(pool, CpuIsa::kGeneric);
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 16, DType::kF32};
  auto cache = KvCache::create(g, be);
  ASSERT_TRUE(cache.ok());
  auto t = Transformer::create(c, (*m)->weights, be, 64);
  ASSERT_TRUE(t.ok()) << t.status().to_string();
  KvSequence seq(**cache);
  ASSERT_TRUE(seq.reserve(static_cast<int64_t>(ref.tokens.size())).ok());
  std::vector<int32_t> pos(ref.tokens.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  ASSERT_TRUE((*t)->forward(ref.tokens, pos, **cache, seq.block_table(), logits).ok());
  for (const auto& [id, want] : ref.top) {
    ASSERT_NEAR(logits[static_cast<size_t>(id)], want, 5e-3f + 2e-4f * std::abs(want)) << rc.fixture << " " << id;
  }
  seq.release();

  Generator gen(**t, **cache, *(*m)->tokenizer);
  GenerateOptions opts;
  opts.max_new_tokens = static_cast<int32_t>(ref.greedy.size());
  opts.stop_at_eog = false;
  std::vector<TokenId> out;
  ASSERT_TRUE(gen.generate(ref.tokens, opts, [&](TokenId id) { out.push_back(id); return true; }).ok());
  EXPECT_EQ(out, ref.greedy);
}

std::string real_case_name(const ::testing::TestParamInfo<int>& p) {
  static const char* kNames[] = {"Qwen25", "Gemma3", "Qwen25Q4KM"};
  return kNames[p.param];
}

INSTANTIATE_TEST_SUITE_P(Real, RealArch, ::testing::Values(0, 1, 2), real_case_name);

TEST(Architectures, ConfigureSetsFamilySemantics) {
  auto qwen2 = load_model(data("tiny_qwen2.gguf"));
  ASSERT_TRUE(qwen2.ok());
  EXPECT_EQ((*qwen2)->architecture->name(), "Qwen");
  EXPECT_TRUE((*qwen2)->config.attn_qkv_bias);
  EXPECT_EQ((*qwen2)->config.rope.style, RopeStyle::kHalfSplit);
  EXPECT_TRUE((*qwen2)->config.tied_embeddings);

  auto g3 = load_model(data("tiny_gemma3.gguf"));
  ASSERT_TRUE(g3.ok());
  const ModelConfig& c = (*g3)->config;
  EXPECT_TRUE(c.attn_qk_norm && c.post_attn_norm && c.post_ffn_norm && c.has_rope_local);
  EXPECT_EQ(c.activation, Activation::kGeluTanh);
  EXPECT_TRUE(c.layer_uses_sliding_window(0));
  EXPECT_FALSE(c.layer_uses_sliding_window(5));  // every 6th layer is global
  EXPECT_FLOAT_EQ(c.layer_rope(0).freq_base, 10000.0f);
  EXPECT_FLOAT_EQ(c.layer_rope(5).freq_base, 1000000.0f);

  auto phi = load_model(data("tiny_phi3.gguf"));
  ASSERT_TRUE(phi.ok());
  EXPECT_TRUE((*phi)->weights.has(TensorRole::kFfnGateUp, 0));  // remapped from ffn_up
  EXPECT_FALSE((*phi)->weights.has(TensorRole::kFfnUp, 0));
  EXPECT_TRUE((*phi)->weights.has(TensorRole::kAttnQkv, 0));
}

TEST(Architectures, UnsupportedArchitectureIsClearError) {
  auto m = load_model(data("golden_smollm2.txt"));  // not a GGUF
  EXPECT_EQ(m.status().code(), StatusCode::kUnsupported);
}

}  // namespace
}  // namespace engine

// Phase 23: SafeTensors + Hugging Face model directories.
//
// - Parser: valid files, and hostile headers rejected.
// - config.json / tensor-name / RoPE-factor translation.
// - Equivalence: every tiny GGUF fixture and its HF export
//   (tools/make_tiny_hf.py: un-permuted Llama Q/K, Gemma norms as w - 1)
//   load to identical logits and tokenization.
// - Real checkpoints, when present: SmolLM2 / Qwen2.5 safetensors vs GGUF.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "dynacore/cpu/cpu_backend.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "loader/hf/hf_model.h"
#include "loader/hf/hf_tokenizer.h"
#include "loader/model_loader.h"
#include "loader/safetensors/safetensors.h"
#include "model/transformer.h"
#include "runtime/engine.h"
#include "runtime/generator.h"
#include "test_models.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

// Writes a safetensors file with the given header JSON and data bytes.
std::string write_st(const std::string& name, const std::string& header, const std::string& body) {
  const std::string path = ::testing::TempDir() + name;
  std::ofstream f(path, std::ios::binary);
  uint64_t n = header.size();
  char len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<char>((n >> (8 * i)) & 0xff);
  f.write(len, 8);
  f << header << body;
  return path;
}

TEST(SafeTensors, ParsesValidFileZeroCopy) {
  std::string body(24, '\0');
  const float vals[4] = {1.0f, -2.0f, 3.5f, 0.25f};
  std::memcpy(body.data(), vals, 16);
  const uint16_t h[4] = {0x3c00, 0x4000, 0xc000, 0x0000};  // f16 1, 2, -2, 0
  std::memcpy(body.data() + 16, h, 8);
  const std::string path = write_st(
      "ok.safetensors",
      R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[4],"data_offsets":[16,24]},)"
      R"("__metadata__":{"format":"pt"}})",
      body);
  auto st = safetensors::SafeTensorsFile::open(path);
  ASSERT_TRUE(st.ok()) << st.status().to_string();
  ASSERT_EQ((*st)->tensors().size(), 2u);
  EXPECT_EQ((*st)->metadata().at("format"), "pt");
  const safetensors::TensorInfo* a = (*st)->find("a");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->shape, (TensorShape{2, 2}));
  auto t = (*st)->load_tensor(*a);
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->data_as<float>()[2], 3.5f);
  EXPECT_EQ((*st)->total_tensor_bytes(), 24u);
}

TEST(SafeTensors, RejectsHostileHeaders) {
  const std::string body(16, '\0');
  const char* bad[] = {
      R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,32]}})",                // past end of file
      R"({"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,16]}})",                // size != dtype x shape
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})",                   // end < begin
      R"({"a":{"dtype":"Q4","shape":[2],"data_offsets":[0,8]}})",                    // unknown dtype
      R"({"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})",                  // negative dim
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",  // overlap
      R"({"a":{"dtype":"F32","shape":[2]}})",                                        // missing offsets
      R"([1,2,3])",                                                                  // not an object
      R"({"a":)",                                                                    // truncated JSON
  };
  for (const char* h : bad) {
    EXPECT_FALSE(safetensors::SafeTensorsFile::open(write_st("bad.safetensors", h, body)).ok()) << h;
  }
  // Header length larger than the file.
  const std::string path = ::testing::TempDir() + "short.safetensors";
  {
    std::ofstream f(path, std::ios::binary);
    const char len[8] = {static_cast<char>(0xff), 0x7f, 0, 0, 0, 0, 0, 0};
    f.write(len, 8);
    f << "{}";
  }
  EXPECT_FALSE(safetensors::SafeTensorsFile::open(path).ok());
}

TEST(SafeTensors, NonFloatTensorsLoadOnlyWhenUsed) {
  const std::string path = write_st("i64.safetensors", R"({"steps":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}})",
                                    std::string(8, '\0'));
  auto st = safetensors::SafeTensorsFile::open(path);
  ASSERT_TRUE(st.ok());
  EXPECT_FALSE((*st)->tensors()[0].dtype.has_value());
  EXPECT_EQ((*st)->load_tensor((*st)->tensors()[0]).status().code(), StatusCode::kUnsupported);
}

json::Value J(const char* s) {
  auto v = json::parse(s);
  EXPECT_TRUE(v.ok()) << s;
  return v.ok() ? *v : json::Value();
}

TEST(HfConfig, TranslatesFamiliesAndOptions) {
  auto llama = hf::read_config(J(R"({"model_type":"mistral","vocab_size":32000,"hidden_size":4096,
      "intermediate_size":14336,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,
      "max_position_embeddings":32768,"rms_norm_eps":1e-5,"rope_theta":1000000.0,"sliding_window":4096})"));
  ASSERT_TRUE(llama.ok()) << llama.status().to_string();
  EXPECT_EQ(llama->architecture, "llama");
  EXPECT_EQ(llama->head_dim, 128);
  EXPECT_EQ(llama->num_kv_heads, 8);
  EXPECT_EQ(llama->sliding_window, 0);  // ignored for Mistral, as on the GGUF path
  EXPECT_FALSE(llama->qk_rows_interleaved);
  EXPECT_FLOAT_EQ(llama->rope.freq_base, 1e6f);

  auto qwen = hf::read_config(J(R"({"model_type":"qwen2","vocab_size":151936,"hidden_size":896,
      "intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,
      "sliding_window":32768,"use_sliding_window":false})"));
  ASSERT_TRUE(qwen.ok());
  EXPECT_EQ(qwen->sliding_window, 0);

  auto g3 = hf::read_config(J(R"({"model_type":"gemma3_text","vocab_size":262144,"hidden_size":640,
      "intermediate_size":2048,"num_hidden_layers":18,"num_attention_heads":4,"num_key_value_heads":1,
      "head_dim":256,"sliding_window":512,"rope_scaling":{"rope_type":"linear","factor":8.0}})"));
  ASSERT_TRUE(g3.ok());
  EXPECT_EQ(g3->architecture, "gemma3");
  EXPECT_EQ(g3->sliding_window, 512);
  EXPECT_EQ(g3->rope.scaling, RopeScaling::kLinear);
  EXPECT_FLOAT_EQ(g3->rope.scaling_factor, 8.0f);

  EXPECT_EQ(hf::read_config(J(R"({"model_type":"gemma3"})")).status().code(), StatusCode::kUnsupported);
  EXPECT_EQ(hf::read_config(J(R"({"model_type":"bert"})")).status().code(), StatusCode::kUnsupported);
  EXPECT_FALSE(hf::read_config(J(R"({"model_type":"llama","vocab_size":"many"})")).ok());
}

TEST(HfConfig, TensorNames) {
  TensorRole r;
  int l;
  ASSERT_TRUE(hf::parse_tensor_name("model.layers.7.self_attn.k_proj.weight", "llama", r, l));
  EXPECT_EQ(r, TensorRole::kAttnK);
  EXPECT_EQ(l, 7);
  ASSERT_TRUE(hf::parse_tensor_name("model.layers.0.post_attention_layernorm.weight", "llama", r, l));
  EXPECT_EQ(r, TensorRole::kFfnNorm);
  ASSERT_TRUE(hf::parse_tensor_name("model.layers.0.post_attention_layernorm.weight", "gemma3", r, l));
  EXPECT_EQ(r, TensorRole::kPostAttnNorm);
  ASSERT_TRUE(hf::parse_tensor_name("lm_head.weight", "qwen2", r, l));
  EXPECT_EQ(l, -1);
  EXPECT_FALSE(hf::parse_tensor_name("model.layers.0.self_attn.rotary_emb.inv_freq", "llama", r, l));
  EXPECT_FALSE(hf::parse_tensor_name("model.layers.x.mlp.up_proj.weight", "llama", r, l));
}

TEST(HfConfig, Llama3RopeFactors) {
  // Llama 3.1: theta 500k, factor 8, low 1, high 4, original context 8192.
  const json::Value cfg = J(R"({"rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,
      "high_freq_factor":4.0,"original_max_position_embeddings":8192}})");
  auto f = hf::llama3_rope_factors(cfg, 128, 500000.0f);
  ASSERT_TRUE(f.ok());
  ASSERT_EQ(f->size(), 64u);
  EXPECT_FLOAT_EQ(f->front(), 1.0f);  // high frequencies untouched
  EXPECT_FLOAT_EQ(f->back(), 8.0f);   // low frequencies scaled by the full factor
  for (size_t i = 1; i < f->size(); ++i) EXPECT_GE((*f)[i], (*f)[i - 1]);  // smooth, monotonic
  EXPECT_TRUE(hf::llama3_rope_factors(J("{}"), 128, 500000.0f)->empty());
}

// --- equivalence -------------------------------------------------------------

std::vector<float> prompt_logits(const LoadedModel& m, const std::vector<TokenId>& tokens) {
  ThreadPool pool(2);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  const ModelConfig& c = m.config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 64, DType::kF32};
  auto cache = KvBlockPool::create(g, be);
  auto t = Transformer::create(c, m.weights, be, 64);
  EXPECT_TRUE(cache.ok() && t.ok());
  if (!cache.ok() || !t.ok()) return {};
  KvBlockTable seq(**cache);
  EXPECT_TRUE(seq.reserve(static_cast<int64_t>(tokens.size())).ok());
  std::vector<int32_t> pos(tokens.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  EXPECT_TRUE((*t)->forward(tokens, pos, **cache, seq.block_table(), logits).ok());
  return logits;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float d = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
  return a.size() == b.size() ? d : INFINITY;
}

class TinyHf : public ::testing::TestWithParam<std::string> {};

TEST_P(TinyHf, MatchesGguf) {
  const std::string arch = GetParam();
  auto g = load_model(data("tiny_" + arch + ".gguf"));
  auto h = load_model(data("hf_tiny_" + arch));
  ASSERT_TRUE(g.ok()) << g.status().to_string();
  ASSERT_TRUE(h.ok()) << h.status().to_string();
  EXPECT_EQ((*h)->format, "safetensors");
  EXPECT_EQ((*h)->quantization, "F16");
  EXPECT_EQ((*h)->architecture, (*g)->architecture);
  EXPECT_EQ((*h)->config.rope.style, (*g)->config.rope.style == RopeStyle::kInterleaved ? RopeStyle::kHalfSplit
                                                                                       : (*g)->config.rope.style);
  EXPECT_EQ((*h)->chat_template.has_value(), (*g)->chat_template.has_value());

  const std::string text = "Hello, tiny model! 123";
  EXPECT_EQ((*h)->tokenizer->encode(text, true, false), (*g)->tokenizer->encode(text, true, false));
  EXPECT_EQ((*h)->tokenizer->bos(), (*g)->tokenizer->bos());
  EXPECT_EQ((*h)->tokenizer->eos(), (*g)->tokenizer->eos());

  const std::vector<TokenId> prompt = {5, 17, 99, 3, 200, 42, 7, 128, 64, 11, 250, 33};
  const std::vector<float> lg = prompt_logits(**g, prompt), lh = prompt_logits(**h, prompt);
  // Same weights, different but equivalent layouts: only float rounding may differ.
  const float d = max_abs_diff(lg, lh);
  std::printf("[ tiny %-6s ] max |logit diff| GGUF vs HF = %g\n", arch.c_str(), d);
  EXPECT_LT(d, 1e-4f) << arch;
}

INSTANTIATE_TEST_SUITE_P(All, TinyHf,
                         ::testing::Values("llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "mixtral",
                                           "qwen2moe", "qwen3moe", "granitemoe"),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });

TEST(TinyHfSharded, IndexJsonShardsLoadLikeOneFile) {
  // Re-split hf_tiny_qwen2 into two shards plus model.safetensors.index.json.
  namespace fs = std::filesystem;
  const fs::path src = data("hf_tiny_qwen2");
  const fs::path dir = fs::path(::testing::TempDir()) / "hf_tiny_qwen2_sharded";
  fs::remove_all(dir);
  fs::create_directories(dir);
  for (const char* f : {"config.json", "tokenizer.json", "tokenizer_config.json"}) {
    fs::copy_file(src / f, dir / f);
  }
  auto st = safetensors::SafeTensorsFile::open((src / "model.safetensors").string());
  ASSERT_TRUE(st.ok());
  std::string header[2] = {"{", "{"}, body[2], weight_map;
  size_t i = 0;
  for (const auto& t : (*st)->tensors()) {
    const int s = static_cast<int>(i++ % 2);
    std::string dims;
    for (int64_t d : t.shape.dims()) dims += (dims.empty() ? "" : ",") + std::to_string(d);
    if (header[s].size() > 1) header[s] += ",";
    header[s] += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype_name + "\",\"shape\":[" + dims +
                 "],\"data_offsets\":[" + std::to_string(body[s].size()) + "," +
                 std::to_string(body[s].size() + t.nbytes) + "]}";
    body[s].append(reinterpret_cast<const char*>((*st)->file().data() + t.offset), t.nbytes);
    weight_map += std::string(weight_map.empty() ? "" : ",") + "\"" + t.name + "\":\"model-0000" +
                  std::to_string(s + 1) + "-of-00002.safetensors\"";
  }
  for (int s = 0; s < 2; ++s) {
    header[s] += "}";
    header[s].append((8 - header[s].size() % 8) % 8, ' ');
    const std::string p = write_st("shard.tmp", header[s], body[s]);
    fs::rename(p, dir / ("model-0000" + std::to_string(s + 1) + "-of-00002.safetensors"));
  }
  std::ofstream(dir / "model.safetensors.index.json") << "{\"metadata\":{},\"weight_map\":{" << weight_map << "}}";

  auto one = load_model(src.string());
  auto two = load_model(dir.string());
  ASSERT_TRUE(two.ok()) << two.status().to_string();
  const std::vector<TokenId> prompt = {5, 17, 99, 3, 200};
  EXPECT_EQ(max_abs_diff(prompt_logits(**one, prompt), prompt_logits(**two, prompt)), 0.0f);

  // A shard path escaping the directory is refused.
  std::ofstream(dir / "model.safetensors.index.json") << R"({"weight_map":{"x":"../evil.safetensors"}})";
  EXPECT_FALSE(load_model(dir.string()).ok());
}

// tokenizer.json alone (no weights) vs the GGUF tokenizer of the same model:
// covers the SentencePiece-style path (Gemma 3: byte fallback, merge-rank
// scores) and both byte-level BPE families.
TEST(HfTokenizer, MatchesGgufTokenizers) {
  const std::string root = std::string(ENGINE_SOURCE_DIR) + "/models/";
  const std::pair<std::string, std::string> pairs[] = {
      {root + "hf/gemma3/tokenizer.json", testing::gemma_model()},
      {root + "hf/qwen25/tokenizer.json", testing::qwen_model()},
      {root + "hf/smollm2/tokenizer.json", testing::smollm_model()},
  };
  int checked = 0;
  for (const auto& [tj_path, gguf] : pairs) {
    if (!testing::exists(tj_path) || !testing::exists(gguf)) continue;
    auto tj = hf::read_json_file(tj_path);
    ASSERT_TRUE(tj.ok()) << tj.status().to_string();
    hf::TokenizerFiles files;
    files.tokenizer = &*tj;
    auto data = hf::read_tokenizer(files);
    ASSERT_TRUE(data.ok()) << tj_path << ": " << data.status().to_string();
    auto tok = Tokenizer::create(std::move(*data));
    ASSERT_TRUE(tok.ok());
    auto g = load_model(gguf);
    ASSERT_TRUE(g.ok());
    for (const char* s : {"Hello world, this is a test.", "Numbers 1234567 and émojis 🙂 and 中文 text",
                          "  leading spaces\n\n\ttabs and trailing  ", "code: for (int i = 0; i < n; ++i) {}",
                          "Ünïcödé façade naïve résumé", "<start_of_turn>user\nhi<end_of_turn>"}) {
      // add_special=false: BOS policy lives in tokenizer_config.json, not here.
      EXPECT_EQ((*tok)->encode(s, false, true), (*g)->tokenizer->encode(s, false, true)) << tj_path << ": " << s;
      EXPECT_EQ((*tok)->decode((*tok)->encode(s, false, false)), s) << tj_path;
    }
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no tokenizer.json / GGUF pairs present";
}

struct RealPair {
  const char* name;
  std::string hf_dir;
  std::string gguf;
  // True if the GGUF was converted from exactly this checkpoint. Qwen's
  // official GGUFs ("v0.1") come from a different snapshot than the HF repo:
  // even F32 norm weights differ, so only tokenizer and behaviour are compared.
  bool weight_twin;
};

class RealHf : public ::testing::TestWithParam<int> {};

TEST_P(RealHf, MatchesGgufTwin) {
  const std::string root = std::string(ENGINE_SOURCE_DIR) + "/models/";
  const RealPair pairs[] = {
      {"smollm2", root + "st/smollm2-135m-instruct", testing::smollm_model(), true},
      {"qwen25", root + "st/qwen2.5-0.5b-instruct", testing::qwen_f16_model(), false},
  };
  const RealPair& p = pairs[GetParam()];
  if (!testing::exists(p.hf_dir + "/model.safetensors") || !testing::exists(p.gguf)) {
    GTEST_SKIP() << "missing " << p.hf_dir << " or " << p.gguf;
  }
  auto h = load_model(p.hf_dir);
  ASSERT_TRUE(h.ok()) << h.status().to_string();
  auto g = load_model(p.gguf);
  ASSERT_TRUE(g.ok());
  EXPECT_EQ((*h)->quantization, "BF16");
  ASSERT_TRUE((*h)->chat_template.has_value());

  // Tokenization (plain text, specials parsed, unicode) is identical.
  for (const char* s : {"Hello world, this is a test.", "<|im_start|>user\nHi!<|im_end|>\n",
                        "Numbers 1234567 and émojis 🙂 and 中文", "  leading spaces\n\n\ttabs"}) {
    EXPECT_EQ((*h)->tokenizer->encode(s, true, true), (*g)->tokenizer->encode(s, true, true)) << s;
  }
  EXPECT_EQ((*h)->tokenizer->eos(), (*g)->tokenizer->eos());

  if (!p.weight_twin) {
    // Not converted from this checkpoint: check the model itself answers.
    EngineOptions o;
    o.model_path = p.hf_dir;
    o.threads = 4;
    o.kv_tokens = 512;
    auto e = Engine::create(o);
    ASSERT_TRUE(e.ok()) << e.status().to_string();
    const ChatMessage msgs[] = {{"user", "What is 7 times 6? Answer with the number only."}};
    GenerateParams gp;
    gp.max_tokens = 8;
    auto s = (*e)->generate_chat(msgs, gp);
    ASSERT_TRUE(s.ok());
    std::string text;
    StreamEvent ev;
    while ((*s)->next(ev) && !ev.done) text += ev.text;
    text += ev.text;
    EXPECT_NE(text.find("42"), std::string::npos) << text;
    return;
  }

  // BF16 checkpoint vs its F16 conversion: logits agree to rounding.
  const std::vector<TokenId> prompt = (*g)->tokenizer->encode("The capital of France is", true, false);
  const std::vector<float> lg = prompt_logits(**g, prompt), lh = prompt_logits(**h, prompt);
  ASSERT_EQ(lg.size(), lh.size());
  const float d = max_abs_diff(lg, lh);
  std::printf("[ %s ] max |logit diff| GGUF F16 vs safetensors BF16 = %g\n", p.name, d);
  EXPECT_LT(d, 0.05f);
  EXPECT_EQ(std::max_element(lg.begin(), lg.end()) - lg.begin(), std::max_element(lh.begin(), lh.end()) - lh.begin());
}

INSTANTIATE_TEST_SUITE_P(Real, RealHf, ::testing::Values(0, 1),
                         [](const ::testing::TestParamInfo<int>& i) { return i.param == 0 ? "smollm2" : "qwen25"; });

}  // namespace
}  // namespace engine

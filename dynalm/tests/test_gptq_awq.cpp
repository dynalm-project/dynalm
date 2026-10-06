// Phase 24: GPTQ / AWQ packed weights.
//
// Fixtures (tools/make_tiny_quant_hf.py) quantize the tiny Llama's linears in
// five ways and pack them as AutoGPTQ / AutoAWQ do; each has a _ref twin with
// W = scale * (q - zero) computed independently in NumPy. The engine must
// repack to the expected executable layout, reproduce the reference weights
// (bit-exact where the layout allows) and the reference logits.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numeric>

#include "runtime/engine.h"

#include "api/json.h"
#include "dynacore/cpu/cpu_device.h"
#include "loader/hf/hf_model.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/quantization/dequant.h"
#include "loader/gptq_awq.h"
#include "common/core.h"

namespace dynalm {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

json::Value J(const char* s) {
  auto v = json::parse(s);
  EXPECT_TRUE(v.ok()) << s;
  return v.ok() ? *v : json::Value();
}

TEST(QuantConfig, ParsesGptqAndAwq) {
  auto none = hf::read_quantization(J(R"({"model_type":"llama"})"));
  ASSERT_TRUE(none.ok());
  EXPECT_FALSE(none->has_value());

  auto g = hf::read_quantization(J(R"({"quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,
      "desc_act":true,"sym":false}})"));
  ASSERT_TRUE(g.ok() && g->has_value()) << g.status().to_string();
  EXPECT_EQ((*g)->method, quant::PackedMethod::kGptq);
  EXPECT_TRUE((*g)->desc_act);
  EXPECT_FALSE((*g)->sym);
  EXPECT_TRUE((*g)->zero_minus_one);  // checkpoint_format defaults to v1 "gptq"
  EXPECT_EQ((*g)->describe(), "GPTQ int4 g128 asym act-order");

  auto a = hf::read_quantization(J(R"({"quantization_config":{"quant_method":"awq","bits":4,"group_size":128,
      "zero_point":true,"version":"gemm"}})"));
  ASSERT_TRUE(a.ok() && a->has_value());
  EXPECT_EQ((*a)->method, quant::PackedMethod::kAwq);
  EXPECT_FALSE((*a)->zero_minus_one);

  // Clear errors for what is not supported.
  for (const char* bad : {R"({"quantization_config":{"quant_method":"bitsandbytes"}})",
                          R"({"quantization_config":{"quant_method":"fp8"}})",
                          R"({"quantization_config":{"quant_method":"gptq","bits":3}})",
                          R"({"quantization_config":{"quant_method":"awq","version":"gemv"}})",
                          R"({"quantization_config":{"quant_method":"gptq","checkpoint_format":"marlin"}})"}) {
    EXPECT_EQ(hf::read_quantization(J(bad)).status().code(), StatusCode::kUnsupported) << bad;
  }
}

TEST(PackedLayout, GptqAndAwqUnpackByHand) {
  // 8 outputs x 8 inputs, one group, codes q[o][i] = (o + i) % 16, zero = 3.
  constexpr int kIn = 8, kOut = 8;
  int32_t gptq_w[kOut] = {};  // [in/8 = 1, out]
  int32_t awq_w[kIn] = {};    // [in, out/8 = 1]
  constexpr int kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  for (int o = 0; o < kOut; ++o) {
    for (int i = 0; i < kIn; ++i) gptq_w[o] |= static_cast<int32_t>(((o + i) % 16) << (4 * i));
  }
  for (int i = 0; i < kIn; ++i) {
    for (int k = 0; k < 8; ++k) awq_w[i] |= static_cast<int32_t>(((kOrder[k] + i) % 16) << (4 * k));
  }
  int32_t zeros_v1 = 0, zeros_awq = 0;
  for (int o = 0; o < kOut; ++o) zeros_v1 |= 2 << (4 * o);   // stored zero - 1
  for (int k = 0; k < 8; ++k) zeros_awq |= 3 << (4 * k);
  uint16_t scales[kOut];
  for (uint16_t& s : scales) s = 0x3c00;  // 1.0

  quant::PackedScheme gs;
  gs.group_size = -1;
  quant::PackedScheme as = gs;
  as.method = quant::PackedMethod::kAwq;
  as.zero_minus_one = false;
  const quant::PackedLinear gw{gptq_w, scales, &zeros_v1, nullptr, kIn, kOut};
  const quant::PackedLinear aw{awq_w, scales, &zeros_awq, nullptr, kIn, kOut};
  for (const auto& [s, w] : {std::pair{gs, gw}, std::pair{as, aw}}) {
    float out[kOut * kIn];
    ASSERT_TRUE(quant::dequantize(s, w, out).ok());
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) EXPECT_EQ(out[o * kIn + i], static_cast<float>((o + i) % 16 - 3)) << o << "," << i;
    }
  }
  // Out-of-range g_idx is corrupt data, not a crash.
  const int32_t bad_gidx[kIn] = {0, 0, 0, 7, 0, 0, 0, 0};
  const quant::PackedLinear bw{gptq_w, scales, &zeros_v1, bad_gidx, kIn, kOut};
  float out[kOut * kIn];
  EXPECT_EQ(quant::dequantize(gs, bw, out).code(), StatusCode::kCorrupt);
}

struct Variant {
  std::string name;
  std::string target;      // expected repack layout for every linear
  float weight_tol;        // vs the NumPy reference weights
  float logit_tol;
};

std::vector<float> prompt_logits(const LoadedModel& m, const std::vector<TokenId>& tokens) {
  ThreadPool pool(2);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  const ModelConfig& c = m.config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, 16, DType::kF32};
  auto cache = KvBlockPool::create(g, be);
  auto t = Transformer::create(c, m.weights, be, 32);
  EXPECT_TRUE(cache.ok() && t.ok());
  KvBlockTable seq(**cache);
  EXPECT_TRUE(seq.reserve(static_cast<int64_t>(tokens.size())).ok());
  std::vector<int32_t> pos(tokens.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  EXPECT_TRUE((*t)->forward(tokens, pos, **cache, seq.block_table(), logits).ok());
  return logits;
}

class PackedModel : public ::testing::TestWithParam<Variant> {};

TEST_P(PackedModel, RepacksToReference) {
  const Variant& v = GetParam();
  auto q = load_model(data("hf_tiny_llama_" + v.name));
  auto r = load_model(data("hf_tiny_llama_" + v.name + "_ref"));
  ASSERT_TRUE(q.ok()) << q.status().to_string();
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  // 2 layers x 7 linears, all in the expected layout.
  EXPECT_NE((*q)->quantization.find("-> " + v.target + " x14"), std::string::npos) << (*q)->quantization;

  float worst = 0;
  for (int l = 0; l < (*q)->config.num_layers; ++l) {
    for (TensorRole role : {TensorRole::kAttnQ, TensorRole::kAttnK, TensorRole::kAttnV, TensorRole::kAttnOutput,
                            TensorRole::kFfnGate, TensorRole::kFfnUp, TensorRole::kFfnDown}) {
      const Tensor* tq = (*q)->weights.find(role, l);
      const Tensor* tr = (*r)->weights.find(role, l);
      ASSERT_TRUE(tq && tr);
      ASSERT_EQ(tq->shape(), tr->shape());
      ASSERT_EQ(tr->dtype(), DType::kF32);
      const int64_t rows = tq->shape()[0], cols = tq->shape()[1];
      std::vector<float> row(static_cast<size_t>(cols));
      for (int64_t i = 0; i < rows; ++i) {
        ASSERT_TRUE(dequantize_row(tq->dtype(), static_cast<const char*>(tq->data()) + i * dtype_row_bytes(tq->dtype(), cols),
                                   row.data(), cols));
        for (int64_t j = 0; j < cols; ++j) {
          worst = std::max(worst, std::abs(row[static_cast<size_t>(j)] - tr->data_as<float>()[i * cols + j]));
        }
      }
    }
  }
  std::printf("[ %-24s ] -> %-4s max |w - ref| = %g\n", v.name.c_str(), v.target.c_str(), worst);
  EXPECT_LE(worst, v.weight_tol);

  const std::vector<TokenId> prompt = {5, 17, 99, 3, 200, 42, 7, 128, 64, 11, 250, 33};
  const auto lq = prompt_logits(**q, prompt), lr = prompt_logits(**r, prompt);
  float d = 0;
  for (size_t i = 0; i < lq.size(); ++i) d = std::max(d, std::abs(lq[i] - lr[i]));
  std::printf("[ %-24s ] max |logit - ref| = %g\n", v.name.c_str(), d);
  EXPECT_LT(d, v.logit_tol);
}

INSTANTIATE_TEST_SUITE_P(
    All, PackedModel,
    ::testing::Values(Variant{"gptq_int4_sym_g32", "Q4_0", 0.0f, 1e-3f},      // bit-exact layout
                      Variant{"gptq_int8_sym_g32", "Q8_0", 0.0f, 1e-3f},      // bit-exact layout
                      Variant{"gptq_int4_asym_g64", "Q4_1", 2e-3f, 2e-2f},    // fp16-rounded min
                      Variant{"awq_int4_g32", "Q4_1", 2e-3f, 2e-2f},          // fp16-rounded min
                      Variant{"gptq_int4_actorder_g32", "F16", 2e-3f, 2e-2f}),  // act-order: F16
    [](const ::testing::TestParamInfo<Variant>& i) { return i.param.name; });

// Official Qwen2.5-0.5B-Instruct GPTQ-Int4 (sym, g128) and AWQ (zero point,
// g128) checkpoints, when downloaded to models/st/.
class RealPacked : public ::testing::TestWithParam<std::pair<const char*, const char*>> {};

TEST_P(RealPacked, LoadsAndAnswers) {
  const auto& [dir, target] = GetParam();
  const std::string path = std::string(ENGINE_SOURCE_DIR) + "/models/st/" + dir;
  if (!std::filesystem::exists(path + "/model.safetensors")) GTEST_SKIP() << "missing " << path;
  EngineOptions o;
  o.model_path = path;
  o.threads = 4;
  o.kv_tokens = 512;
  auto e = Engine::create(o);
  ASSERT_TRUE(e.ok()) << e.status().to_string();
  EXPECT_NE((*e)->model().quantization.find(std::string("-> ") + target + " x168"), std::string::npos)
      << (*e)->model().quantization;
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
}

INSTANTIATE_TEST_SUITE_P(Qwen, RealPacked,
                         ::testing::Values(std::pair{"qwen2.5-0.5b-instruct-gptq-int4", "Q4_0"},
                                           std::pair{"qwen2.5-0.5b-instruct-awq", "Q4_1"}),
                         [](const auto& i) { return std::string(i.param.second); });

}  // namespace
}  // namespace dynalm

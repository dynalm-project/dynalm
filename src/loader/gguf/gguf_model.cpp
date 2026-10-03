#include "loader/gguf/gguf_model.h"

#include <charconv>
#include <string>

namespace engine::gguf {
namespace {

// Reads `<arch>.<suffix>` as an integer; rejects per-layer arrays, which the
// engine does not support yet (e.g. variable head counts per layer).
Result<int64_t> arch_int(const GgufFile& f, const std::string& arch, std::string_view suffix) {
  const std::string key = arch + "." + std::string(suffix);
  if (const Value* v = f.find(key); v && v->type == ValueType::kArray) {
    return Unsupported("per-layer '" + key + "' is not supported");
  }
  return f.get_int(key);
}

// Optional integer: default if missing, error if present but malformed.
Result<int64_t> arch_int_or(const GgufFile& f, const std::string& arch, std::string_view suffix,
                            int64_t fallback) {
  auto r = arch_int(f, arch, suffix);
  if (!r.ok() && r.status().code() == StatusCode::kNotFound) return fallback;
  return r;
}

Result<double> arch_float_or(const GgufFile& f, const std::string& arch, std::string_view suffix,
                             double fallback) {
  auto r = f.get_float(arch + "." + std::string(suffix));
  if (!r.ok() && r.status().code() == StatusCode::kNotFound) return fallback;
  return r;
}

}  // namespace

Result<ModelConfig> read_model_config(const GgufFile& f) {
  ModelConfig c;
  ENGINE_ASSIGN_OR_RETURN(std::string_view arch_sv, f.get_string("general.architecture"));
  const std::string arch(arch_sv);
  c.architecture = arch;
  // llama.cpp's converter permutes Llama-derived Q/K rows for interleaved RoPE.
  c.qk_rows_interleaved = arch == "llama" || arch == "granite" || arch == "granitemoe";
  if (auto name = f.get_string("general.name"); name.ok()) c.name = std::string(*name);

  ENGINE_ASSIGN_OR_RETURN(int64_t hidden, arch_int(f, arch, "embedding_length"));
  ENGINE_ASSIGN_OR_RETURN(int64_t layers, arch_int(f, arch, "block_count"));
  ENGINE_ASSIGN_OR_RETURN(int64_t heads, arch_int(f, arch, "attention.head_count"));
  ENGINE_ASSIGN_OR_RETURN(int64_t kv_heads, arch_int_or(f, arch, "attention.head_count_kv", heads));
  ENGINE_ASSIGN_OR_RETURN(int64_t ffn, arch_int_or(f, arch, "feed_forward_length", 0));
  ENGINE_ASSIGN_OR_RETURN(int64_t ctx, arch_int_or(f, arch, "context_length", 2048));
  if (heads <= 0) return Corrupt("attention.head_count must be > 0");
  ENGINE_ASSIGN_OR_RETURN(int64_t head_dim, arch_int_or(f, arch, "attention.key_length", hidden / heads));
  ENGINE_ASSIGN_OR_RETURN(int64_t head_dim_v, arch_int_or(f, arch, "attention.value_length", head_dim));

  c.hidden_size = hidden;
  c.num_layers = static_cast<int32_t>(layers);
  c.num_heads = static_cast<int32_t>(heads);
  c.num_kv_heads = static_cast<int32_t>(kv_heads);
  c.intermediate_size = ffn;
  c.context_length = ctx;
  c.head_dim = static_cast<int32_t>(head_dim);
  c.head_dim_v = static_cast<int32_t>(head_dim_v);

  // Norm epsilon: RMSNorm and LayerNorm models use different keys.
  if (auto e = f.get_float(arch + ".attention.layer_norm_rms_epsilon"); e.ok()) {
    c.norm = NormType::kRmsNorm;
    c.norm_eps = static_cast<float>(*e);
  } else if (auto e2 = f.get_float(arch + ".attention.layer_norm_epsilon"); e2.ok()) {
    c.norm = NormType::kLayerNorm;
    c.norm_eps = static_cast<float>(*e2);
  }

  // Vocab size: explicit key, else the tokenizer vocab, else the embedding rows.
  if (auto v = arch_int(f, arch, "vocab_size"); v.ok()) {
    c.vocab_size = *v;
  } else if (auto toks = f.get_array("tokenizer.ggml.tokens"); toks.ok()) {
    c.vocab_size = static_cast<int64_t>(toks->count);
  } else if (const TensorInfo* emb = f.find_tensor("token_embd.weight")) {
    c.vocab_size = emb->shape[0];
  }

  // RoPE (style is chosen by the adapter).
  ENGINE_ASSIGN_OR_RETURN(int64_t rope_dim, arch_int_or(f, arch, "rope.dimension_count", head_dim));
  ENGINE_ASSIGN_OR_RETURN(double base, arch_float_or(f, arch, "rope.freq_base", 10000.0));
  c.rope.dim = static_cast<int32_t>(rope_dim);
  c.rope.freq_base = static_cast<float>(base);
  if (auto st = f.get_string(arch + ".rope.scaling.type"); st.ok()) {
    if (*st == "linear") c.rope.scaling = RopeScaling::kLinear;
    else if (*st == "yarn") c.rope.scaling = RopeScaling::kYarn;
    else if (*st != "none") return Unsupported("rope scaling type '" + std::string(*st) + "'");
  }
  ENGINE_ASSIGN_OR_RETURN(double sf, arch_float_or(f, arch, "rope.scaling.factor", 1.0));
  c.rope.scaling_factor = static_cast<float>(sf);
  // Older files express linear scaling only as rope.scale_linear.
  if (auto lin = f.get_float(arch + ".rope.scale_linear"); lin.ok() && *lin != 1.0) {
    c.rope.scaling = RopeScaling::kLinear;
    c.rope.scaling_factor = static_cast<float>(*lin);
  }
  ENGINE_ASSIGN_OR_RETURN(c.rope.original_context,
                          arch_int_or(f, arch, "rope.scaling.original_context_length", 0));

  ENGINE_ASSIGN_OR_RETURN(int64_t window, arch_int_or(f, arch, "attention.sliding_window", 0));
  c.sliding_window = static_cast<int32_t>(window);
  ENGINE_ASSIGN_OR_RETURN(double attn_cap, arch_float_or(f, arch, "attn_logit_softcapping", 0.0));
  ENGINE_ASSIGN_OR_RETURN(double final_cap, arch_float_or(f, arch, "final_logit_softcapping", 0.0));
  c.attn_logit_softcap = static_cast<float>(attn_cap);
  c.final_logit_softcap = static_cast<float>(final_cap);

  ENGINE_ASSIGN_OR_RETURN(int64_t n_exp, arch_int_or(f, arch, "expert_count", 0));
  ENGINE_ASSIGN_OR_RETURN(int64_t n_used, arch_int_or(f, arch, "expert_used_count", 0));
  ENGINE_ASSIGN_OR_RETURN(int64_t n_shared, arch_int_or(f, arch, "expert_shared_count", 0));
  ENGINE_ASSIGN_OR_RETURN(int64_t exp_ffn, arch_int_or(f, arch, "expert_feed_forward_length", 0));
  ENGINE_ASSIGN_OR_RETURN(int64_t shared_ffn, arch_int_or(f, arch, "expert_shared_feed_forward_length", 0));
  c.moe.num_experts = static_cast<int32_t>(n_exp);
  c.moe.experts_per_token = static_cast<int32_t>(n_used);
  c.moe.num_shared_experts = static_cast<int32_t>(n_shared);
  // Mixtral / Granite-MoE store the expert width as feed_forward_length.
  c.moe.expert_intermediate_size = exp_ffn > 0 ? exp_ffn : (n_exp > 0 ? ffn : 0);
  c.moe.shared_intermediate_size = shared_ffn;
  if (auto norm = f.get_bool(arch + ".expert_weights_norm"); norm.ok()) c.moe.normalize_topk = *norm;

  // Granite multipliers (absent elsewhere: identity).
  ENGINE_ASSIGN_OR_RETURN(double emb_scale, arch_float_or(f, arch, "embedding_scale", 1.0));
  ENGINE_ASSIGN_OR_RETURN(double res_scale, arch_float_or(f, arch, "residual_scale", 1.0));
  ENGINE_ASSIGN_OR_RETURN(double att_scale, arch_float_or(f, arch, "attention.scale", 0.0));
  ENGINE_ASSIGN_OR_RETURN(double logit_scale, arch_float_or(f, arch, "logit_scale", 1.0));
  c.embedding_scale = static_cast<float>(emb_scale);
  c.residual_scale = static_cast<float>(res_scale);
  c.attn_scale = static_cast<float>(att_scale);
  c.logit_scale = static_cast<float>(logit_scale);

  // Tied embeddings: no separate output projection.
  c.tied_embeddings = f.find_tensor("output.weight") == nullptr;
  return c;
}

// ---------------------------------------------------------------------------

namespace {

struct NameRole {
  std::string_view suffix;
  TensorRole role;
};

constexpr NameRole kGlobal[] = {
    {"token_embd.weight", TensorRole::kTokenEmbedding},
    {"output_norm.weight", TensorRole::kOutputNorm},
    {"output_norm.bias", TensorRole::kOutputNormBias},
    {"output.weight", TensorRole::kOutput},
    {"rope_freqs.weight", TensorRole::kRopeFreqs},
};

constexpr NameRole kLayer[] = {
    {"attn_norm.weight", TensorRole::kAttnNorm},
    {"attn_norm.bias", TensorRole::kAttnNormBias},
    {"attn_q.weight", TensorRole::kAttnQ},
    {"attn_k.weight", TensorRole::kAttnK},
    {"attn_v.weight", TensorRole::kAttnV},
    {"attn_qkv.weight", TensorRole::kAttnQkv},
    {"attn_q.bias", TensorRole::kAttnQBias},
    {"attn_k.bias", TensorRole::kAttnKBias},
    {"attn_v.bias", TensorRole::kAttnVBias},
    {"attn_qkv.bias", TensorRole::kAttnQkvBias},
    {"attn_output.weight", TensorRole::kAttnOutput},
    {"attn_output.bias", TensorRole::kAttnOutputBias},
    {"attn_q_norm.weight", TensorRole::kAttnQNorm},
    {"attn_k_norm.weight", TensorRole::kAttnKNorm},
    {"post_attention_norm.weight", TensorRole::kPostAttnNorm},
    {"ffn_norm.weight", TensorRole::kFfnNorm},
    {"ffn_norm.bias", TensorRole::kFfnNormBias},
    {"ffn_gate.weight", TensorRole::kFfnGate},
    {"ffn_up.weight", TensorRole::kFfnUp},
    {"ffn_down.weight", TensorRole::kFfnDown},
    {"ffn_up.bias", TensorRole::kFfnUpBias},
    {"ffn_down.bias", TensorRole::kFfnDownBias},
    {"post_ffw_norm.weight", TensorRole::kPostFfnNorm},
    {"ffn_gate_inp.weight", TensorRole::kFfnRouter},
    {"ffn_gate_exps.weight", TensorRole::kFfnGateExperts},
    {"ffn_up_exps.weight", TensorRole::kFfnUpExperts},
    {"ffn_down_exps.weight", TensorRole::kFfnDownExperts},
    {"ffn_gate_inp_shexp.weight", TensorRole::kFfnSharedRouter},
    {"ffn_gate_shexp.weight", TensorRole::kFfnGateShared},
    {"ffn_up_shexp.weight", TensorRole::kFfnUpShared},
    {"ffn_down_shexp.weight", TensorRole::kFfnDownShared},
};

}  // namespace

std::string_view file_type_name(uint32_t ft) {
  switch (ft) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 7: return "Q8_0";
    case 8: return "Q5_0";
    case 9: return "Q5_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K_S";
    case 12: return "Q3_K_M";
    case 13: return "Q3_K_L";
    case 14: return "Q4_K_S";
    case 15: return "Q4_K_M";
    case 16: return "Q5_K_S";
    case 17: return "Q5_K_M";
    case 18: return "Q6_K";
    case 19: return "IQ2_XXS";
    case 20: return "IQ2_XS";
    case 21: return "Q2_K_S";
    case 22: return "IQ3_XS";
    case 23: return "IQ3_XXS";
    case 24: return "IQ1_S";
    case 25: return "IQ4_NL";
    case 26: return "IQ3_S";
    case 27: return "IQ3_M";
    case 28: return "IQ2_S";
    case 29: return "IQ2_M";
    case 30: return "IQ4_XS";
    case 31: return "IQ1_M";
    case 32: return "BF16";
    case 36: return "TQ1_0";
    case 37: return "TQ2_0";
    default: return "unknown";
  }
}

bool parse_tensor_name(std::string_view name, TensorRole& role, int& layer) {
  for (const auto& e : kGlobal) {
    if (name == e.suffix) {
      role = e.role;
      layer = -1;
      return true;
    }
  }
  constexpr std::string_view kBlk = "blk.";
  if (!name.starts_with(kBlk)) return false;
  name.remove_prefix(kBlk.size());
  const auto dot = name.find('.');
  if (dot == std::string_view::npos || dot == 0) return false;
  int l = 0;
  const auto [ptr, ec] = std::from_chars(name.data(), name.data() + dot, l);
  if (ec != std::errc() || ptr != name.data() + dot || l < 0) return false;
  const std::string_view rest = name.substr(dot + 1);
  for (const auto& e : kLayer) {
    if (rest == e.suffix) {
      role = e.role;
      layer = l;
      return true;
    }
  }
  return false;
}

Result<TensorMapping> map_tensors(const GgufFile& f, int num_layers) {
  TensorMapping m{TensorRegistry(num_layers), {}};
  for (const TensorInfo& info : f.tensors()) {
    TensorRole role;
    int layer;
    if (!parse_tensor_name(info.name, role, layer)) {
      m.unmapped.emplace_back(info.name);
      continue;
    }
    if (layer >= num_layers) {
      return Corrupt("tensor '" + std::string(info.name) + "' refers to layer " + std::to_string(layer) +
                     " but the model has " + std::to_string(num_layers));
    }
    ENGINE_ASSIGN_OR_RETURN(Tensor t, f.load_tensor(info));
    ENGINE_RETURN_IF_ERROR(m.registry.add(role, layer, std::move(t)));
  }
  return m;
}

}  // namespace engine::gguf

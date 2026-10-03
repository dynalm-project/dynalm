#include "loader/hf/hf_model.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>

#include "quant/dequant.h"

namespace engine::hf {
namespace fs = std::filesystem;
namespace {

const json::Value* field(const json::Value& o, std::string_view key) {
  const json::Value* v = o.find(key);
  return v && !v->is_null() ? v : nullptr;
}

Result<int64_t> get_int(const json::Value& o, std::string_view key) {
  const json::Value* v = field(o, key);
  if (!v) return NotFound("config.json: missing '" + std::string(key) + "'");
  if (!v->is_number() || std::floor(v->as_number()) != v->as_number()) {
    return Corrupt("config.json: '" + std::string(key) + "' is not an integer");
  }
  return static_cast<int64_t>(v->as_number());
}

Result<int64_t> get_int_or(const json::Value& o, std::string_view key, int64_t fallback) {
  auto r = get_int(o, key);
  if (!r.ok() && r.status().code() == StatusCode::kNotFound) return fallback;
  return r;
}

double get_float_or(const json::Value& o, std::string_view key, double fallback) {
  const json::Value* v = field(o, key);
  return v && v->is_number() ? v->as_number() : fallback;
}

struct NameRole {
  std::string_view name;
  TensorRole role;
};

constexpr NameRole kGlobal[] = {
    {"model.embed_tokens.weight", TensorRole::kTokenEmbedding},
    {"model.norm.weight", TensorRole::kOutputNorm},
    {"lm_head.weight", TensorRole::kOutput},
};

constexpr NameRole kLayer[] = {
    {"input_layernorm.weight", TensorRole::kAttnNorm},
    {"self_attn.q_proj.weight", TensorRole::kAttnQ},
    {"self_attn.k_proj.weight", TensorRole::kAttnK},
    {"self_attn.v_proj.weight", TensorRole::kAttnV},
    {"self_attn.qkv_proj.weight", TensorRole::kAttnQkv},
    {"self_attn.q_proj.bias", TensorRole::kAttnQBias},
    {"self_attn.k_proj.bias", TensorRole::kAttnKBias},
    {"self_attn.v_proj.bias", TensorRole::kAttnVBias},
    {"self_attn.o_proj.weight", TensorRole::kAttnOutput},
    {"self_attn.o_proj.bias", TensorRole::kAttnOutputBias},
    {"self_attn.q_norm.weight", TensorRole::kAttnQNorm},
    {"self_attn.k_norm.weight", TensorRole::kAttnKNorm},
    {"pre_feedforward_layernorm.weight", TensorRole::kFfnNorm},
    {"post_feedforward_layernorm.weight", TensorRole::kPostFfnNorm},
    {"mlp.gate_proj.weight", TensorRole::kFfnGate},
    {"mlp.up_proj.weight", TensorRole::kFfnUp},
    {"mlp.gate_up_proj.weight", TensorRole::kFfnUp},  // Phi-3: fused gate|up, like its GGUF
    {"mlp.down_proj.weight", TensorRole::kFfnDown},
    // MoE routers and shared experts (per-expert tensors: parse_expert_name).
    {"block_sparse_moe.gate.weight", TensorRole::kFfnRouter},          // Mixtral
    {"block_sparse_moe.router.layer.weight", TensorRole::kFfnRouter},  // Granite-MoE
    {"block_sparse_moe.input_linear.weight", TensorRole::kFfnGateUp},  // Granite-MoE [E, 2F, D], split at load
    {"block_sparse_moe.output_linear.weight", TensorRole::kFfnDownExperts},
    {"mlp.gate.weight", TensorRole::kFfnRouter},                       // Qwen-MoE
    {"mlp.shared_expert.gate_proj.weight", TensorRole::kFfnGateShared},
    {"mlp.shared_expert.up_proj.weight", TensorRole::kFfnUpShared},
    {"mlp.shared_expert.down_proj.weight", TensorRole::kFfnDownShared},
    {"mlp.shared_expert_gate.weight", TensorRole::kFfnSharedRouter},   // [1, D], reshaped at load
};

// Replaces a (small) norm tensor by an F32 copy holding 1 + w.
Status fold_one_plus(TensorRegistry& w, TensorRole role, int layer) {
  std::optional<Tensor> t = w.take(role, layer);
  if (!t) return Status::Ok();
  ENGINE_ASSIGN_OR_RETURN(Tensor f, Tensor::empty(DType::kF32, t->shape()));
  if (!dequantize_row(t->dtype(), t->data(), f.data_as<float>(), t->numel())) {
    return Unsupported("cannot convert norm weight of type " + std::string(dtype_name(t->dtype())));
  }
  for (int64_t i = 0; i < f.numel(); ++i) f.data_as<float>()[i] += 1.0f;
  return w.add(role, layer, std::move(f));
}

}  // namespace

bool is_hf_model(const std::string& path) {
  std::error_code ec;
  if (fs::is_directory(path, ec)) return fs::exists(fs::path(path) / "config.json", ec);
  return fs::path(path).extension() == ".safetensors";
}

Result<ModelFiles> locate(const std::string& path) {
  std::error_code ec;
  ModelFiles f;
  if (fs::is_directory(path, ec)) {
    f.dir = path;
  } else if (fs::path(path).extension() == ".safetensors" && fs::exists(path, ec)) {
    f.dir = fs::path(path).parent_path().string();
    if (f.dir.empty()) f.dir = ".";
  } else {
    return NotFound("not a Hugging Face model directory or .safetensors file: " + path);
  }
  const fs::path dir(f.dir);
  if (!fs::exists(dir / "config.json", ec)) return NotFound("config.json not found in " + f.dir);
  if (fs::exists(dir / "model.safetensors.index.json", ec)) {
    // Sharded: every distinct file named by the weight map.
    ENGINE_ASSIGN_OR_RETURN(json::Value idx, read_json_file((dir / "model.safetensors.index.json").string()));
    const json::Value* map = idx.find("weight_map");
    if (!map || !map->is_object()) return Corrupt("model.safetensors.index.json: missing weight_map");
    std::vector<std::string> names;
    for (const auto& [tensor, file] : map->as_object()) {
      if (!file.is_string() || file.as_string().find("..") != std::string::npos ||
          fs::path(file.as_string()).is_absolute()) {
        return Corrupt("model.safetensors.index.json: bad shard name for '" + tensor + "'");
      }
      names.push_back(file.as_string());
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    for (const std::string& n : names) f.weights.push_back((dir / n).string());
  } else if (fs::path(path).extension() == ".safetensors") {
    f.weights.push_back(path);
  } else if (fs::exists(dir / "model.safetensors", ec)) {
    f.weights.push_back((dir / "model.safetensors").string());
  } else {
    return NotFound("no model.safetensors or model.safetensors.index.json in " + f.dir);
  }
  return f;
}

Result<json::Value> read_json_file(const std::string& path, size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return NotFound("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  json::ParseLimits limits;
  limits.max_bytes = max_bytes;
  limits.max_depth = 64;
  auto v = json::parse(text, limits);
  if (!v.ok()) return Corrupt(path + ": " + v.status().message());
  return v;
}

Result<ModelConfig> read_config(const json::Value& cfg) {
  if (!cfg.is_object()) return Corrupt("config.json is not an object");
  const json::Value* mt = field(cfg, "model_type");
  if (!mt || !mt->is_string()) return Corrupt("config.json: missing model_type");
  const std::string& type = mt->as_string();
  ModelConfig c;
  if (type == "llama" || type == "mistral" || type == "mixtral") c.architecture = "llama";
  else if (type == "qwen2" || type == "qwen3" || type == "gemma" || type == "gemma2" || type == "phi3" ||
           type == "granite" || type == "granitemoe") c.architecture = type;
  else if (type == "qwen2_moe") c.architecture = "qwen2moe";
  else if (type == "qwen3_moe") c.architecture = "qwen3moe";
  else if (type == "gemma3_text") c.architecture = "gemma3";
  else if (type == "gemma3") return Unsupported("multimodal Gemma 3 checkpoints are not supported; use a text-only (gemma3_text) model");
  else return Unsupported("Hugging Face model_type '" + type + "' is not supported yet");
  if (const json::Value* n = field(cfg, "_name_or_path"); n && n->is_string()) c.name = n->as_string();

  ENGINE_ASSIGN_OR_RETURN(c.vocab_size, get_int(cfg, "vocab_size"));
  ENGINE_ASSIGN_OR_RETURN(c.hidden_size, get_int(cfg, "hidden_size"));
  ENGINE_ASSIGN_OR_RETURN(c.intermediate_size, get_int(cfg, "intermediate_size"));
  ENGINE_ASSIGN_OR_RETURN(int64_t layers, get_int(cfg, "num_hidden_layers"));
  ENGINE_ASSIGN_OR_RETURN(int64_t heads, get_int(cfg, "num_attention_heads"));
  if (heads <= 0 || layers <= 0) return Corrupt("config.json: layer and head counts must be > 0");
  ENGINE_ASSIGN_OR_RETURN(int64_t kv_heads, get_int_or(cfg, "num_key_value_heads", heads));
  ENGINE_ASSIGN_OR_RETURN(int64_t head_dim, get_int_or(cfg, "head_dim", c.hidden_size / heads));
  ENGINE_ASSIGN_OR_RETURN(c.context_length, get_int_or(cfg, "max_position_embeddings", 2048));
  c.num_layers = static_cast<int32_t>(layers);
  c.num_heads = static_cast<int32_t>(heads);
  c.num_kv_heads = static_cast<int32_t>(kv_heads);
  c.head_dim = static_cast<int32_t>(head_dim);
  c.head_dim_v = c.head_dim;
  c.norm = NormType::kRmsNorm;
  c.norm_eps = static_cast<float>(get_float_or(cfg, "rms_norm_eps", 1e-6));
  c.qk_rows_interleaved = false;
  // Final say comes from the adapter (lm_head present or not); this is the
  // declared intent, shown by `dynalm inspect`. HF's default is tied.
  const json::Value* tie = field(cfg, "tie_word_embeddings");
  c.tied_embeddings = !(tie && tie->is_bool() && !tie->as_bool());

  c.rope.freq_base = static_cast<float>(get_float_or(cfg, "rope_theta", 10000.0));
  c.rope.dim = static_cast<int32_t>(std::lround(head_dim * get_float_or(cfg, "partial_rotary_factor", 1.0)));
  if (const json::Value* rs = field(cfg, "rope_scaling"); rs && rs->is_object()) {
    const json::Value* t = field(*rs, "rope_type");
    if (!t) t = field(*rs, "type");
    const std::string kind = t && t->is_string() ? t->as_string() : "";
    if (kind == "linear") {
      c.rope.scaling = RopeScaling::kLinear;
      c.rope.scaling_factor = static_cast<float>(get_float_or(*rs, "factor", 1.0));
    } else if (kind == "yarn") {
      c.rope.scaling = RopeScaling::kYarn;
      c.rope.scaling_factor = static_cast<float>(get_float_or(*rs, "factor", 1.0));
      c.rope.original_context = static_cast<int64_t>(get_float_or(*rs, "original_max_position_embeddings", 0));
    } else if (kind == "llama3") {
      // Handled as rope_freqs factors (apply_conventions), as in GGUF files.
    } else if (!kind.empty() && kind != "default") {
      return Unsupported("rope_scaling type '" + kind + "' is not supported yet");
    }
  }

  // Sliding windows: Gemma 2/3 and Phi-3 use them; Qwen2 configs carry a
  // window but disable it; llama.cpp ignores Mistral v0.1's window, and so
  // does the GGUF path.
  const json::Value* use_sw = field(cfg, "use_sliding_window");
  const bool sw_enabled = c.architecture != "llama" && !(use_sw && use_sw->is_bool() && !use_sw->as_bool());
  if (sw_enabled) {
    ENGINE_ASSIGN_OR_RETURN(int64_t window, get_int_or(cfg, "sliding_window", 0));
    c.sliding_window = static_cast<int32_t>(window);
  }
  c.attn_logit_softcap = static_cast<float>(get_float_or(cfg, "attn_logit_softcapping", 0.0));
  c.final_logit_softcap = static_cast<float>(get_float_or(cfg, "final_logit_softcapping", 0.0));

  // MoE: Mixtral / Granite-MoE ("num_local_experts", expert width =
  // intermediate_size) and Qwen-MoE ("num_experts", "moe_intermediate_size").
  ENGINE_ASSIGN_OR_RETURN(int64_t n_exp, get_int_or(cfg, "num_local_experts", 0));
  if (n_exp == 0 && (type == "qwen2_moe" || type == "qwen3_moe")) {
    ENGINE_ASSIGN_OR_RETURN(n_exp, get_int_or(cfg, "num_experts", 0));
  }
  if (n_exp > 0) {
    ENGINE_ASSIGN_OR_RETURN(int64_t used, get_int(cfg, "num_experts_per_tok"));
    ENGINE_ASSIGN_OR_RETURN(int64_t moe_ff, get_int_or(cfg, "moe_intermediate_size", c.intermediate_size));
    ENGINE_ASSIGN_OR_RETURN(int64_t shared_ff, get_int_or(cfg, "shared_expert_intermediate_size", 0));
    ENGINE_ASSIGN_OR_RETURN(int64_t sparse_step, get_int_or(cfg, "decoder_sparse_step", 1));
    const json::Value* dense_layers = field(cfg, "mlp_only_layers");
    if (sparse_step != 1 || (dense_layers && dense_layers->is_array() && !dense_layers->as_array().empty())) {
      return Unsupported("MoE models mixing dense and sparse layers are not supported yet");
    }
    c.moe.num_experts = static_cast<int32_t>(n_exp);
    c.moe.experts_per_token = static_cast<int32_t>(used);
    c.moe.expert_intermediate_size = moe_ff;
    c.moe.shared_intermediate_size = shared_ff;
    c.moe.num_shared_experts = shared_ff > 0 ? 1 : 0;
  }
  // Granite multipliers.
  c.embedding_scale = static_cast<float>(get_float_or(cfg, "embedding_multiplier", 1.0));
  c.residual_scale = static_cast<float>(get_float_or(cfg, "residual_multiplier", 1.0));
  c.attn_scale = static_cast<float>(get_float_or(cfg, "attention_multiplier", 0.0));
  c.logit_scale = static_cast<float>(get_float_or(cfg, "logits_scaling", 1.0));
  return c;
}

bool parse_expert_name(std::string_view name, TensorRole& role, int& layer, int& expert) {
  constexpr std::string_view kPrefix = "model.layers.";
  if (!name.starts_with(kPrefix)) return false;
  name.remove_prefix(kPrefix.size());
  auto number = [](std::string_view& s, int& out) {
    const size_t dot = s.find('.');
    if (dot == std::string_view::npos || dot == 0) return false;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + dot, out);
    if (ec != std::errc() || p != s.data() + dot || out < 0) return false;
    s.remove_prefix(dot + 1);
    return true;
  };
  if (!number(name, layer)) return false;
  for (std::string_view base : {"block_sparse_moe.experts.", "mlp.experts."}) {
    if (!name.starts_with(base)) continue;
    name.remove_prefix(base.size());
    if (!number(name, expert)) return false;
    static constexpr std::pair<std::string_view, TensorRole> kParts[] = {
        {"w1.weight", TensorRole::kFfnGateExperts},        {"w3.weight", TensorRole::kFfnUpExperts},
        {"w2.weight", TensorRole::kFfnDownExperts},        {"gate_proj.weight", TensorRole::kFfnGateExperts},
        {"up_proj.weight", TensorRole::kFfnUpExperts},     {"down_proj.weight", TensorRole::kFfnDownExperts}};
    for (const auto& [suffix, r] : kParts) {
      if (name == suffix) {
        role = r;
        return true;
      }
    }
    return false;
  }
  return false;
}

bool parse_tensor_name(std::string_view name, std::string_view arch, TensorRole& role, int& layer) {
  for (const auto& e : kGlobal) {
    if (name == e.name) {
      role = e.role;
      layer = -1;
      return true;
    }
  }
  constexpr std::string_view kPrefix = "model.layers.";
  if (!name.starts_with(kPrefix)) return false;
  name.remove_prefix(kPrefix.size());
  const size_t dot = name.find('.');
  if (dot == std::string_view::npos || dot == 0) return false;
  int l = 0;
  const auto [ptr, ec] = std::from_chars(name.data(), name.data() + dot, l);
  if (ec != std::errc() || ptr != name.data() + dot || l < 0) return false;
  const std::string_view rest = name.substr(dot + 1);
  layer = l;
  if (rest == "post_attention_layernorm.weight") {
    // Gemma 2/3 "sandwich" norms: this one follows attention; the FFN input
    // norm is pre_feedforward_layernorm. Elsewhere it is the FFN input norm.
    role = (arch == "gemma2" || arch == "gemma3") ? TensorRole::kPostAttnNorm : TensorRole::kFfnNorm;
    return true;
  }
  for (const auto& e : kLayer) {
    if (rest == e.name) {
      role = e.role;
      return true;
    }
  }
  return false;
}

Result<std::optional<quant::PackedScheme>> read_quantization(const json::Value& cfg) {
  const json::Value* qc = field(cfg, "quantization_config");
  if (!qc) return std::optional<quant::PackedScheme>{};
  if (!qc->is_object()) return Corrupt("config.json: quantization_config is not an object");
  const json::Value* m = field(*qc, "quant_method");
  const std::string method = m && m->is_string() ? m->as_string() : "";
  auto flag = [&](std::string_view key, bool fallback) {
    const json::Value* v = field(*qc, key);
    return v && v->is_bool() ? v->as_bool() : fallback;
  };
  quant::PackedScheme s;
  if (method == "gptq") {
    s.method = quant::PackedMethod::kGptq;
    ENGINE_ASSIGN_OR_RETURN(int64_t bits, get_int_or(*qc, "bits", 4));
    ENGINE_ASSIGN_OR_RETURN(int64_t group, get_int_or(*qc, "group_size", 128));
    s.bits = static_cast<int>(bits);
    s.group_size = static_cast<int>(group);
    s.sym = flag("sym", true);
    s.desc_act = flag("desc_act", false);
    const json::Value* fmt = field(*qc, "checkpoint_format");
    const std::string f = fmt && fmt->is_string() ? fmt->as_string() : "gptq";
    if (f != "gptq" && f != "gptq_v2") return Unsupported("GPTQ checkpoint_format '" + f + "'");
    s.zero_minus_one = f == "gptq";
  } else if (method == "awq") {
    s.method = quant::PackedMethod::kAwq;
    ENGINE_ASSIGN_OR_RETURN(int64_t bits, get_int_or(*qc, "bits", 4));
    ENGINE_ASSIGN_OR_RETURN(int64_t group, get_int_or(*qc, "group_size", 128));
    s.bits = static_cast<int>(bits);
    s.group_size = static_cast<int>(group);
    s.sym = !flag("zero_point", true);
    s.zero_minus_one = false;
    const json::Value* ver = field(*qc, "version");
    const std::string v = ver && ver->is_string() ? ver->as_string() : "gemm";
    if (v != "gemm" && v != "GEMM") {
      return Unsupported("AWQ version '" + v + "' (only the GEMM packing is supported)");
    }
  } else {
    return Unsupported("quantization method '" + method +
                       "' is not supported (GPTQ and AWQ are; GGUF quantizations are always supported)");
  }
  ENGINE_RETURN_IF_ERROR(quant::packed_shapes(s, 32, 32).status());  // validates bits / group size
  return std::optional<quant::PackedScheme>(s);
}

bool split_packed_name(std::string_view name, std::string& weight_name, std::string& component) {
  for (std::string_view c : {"qweight", "qzeros", "scales", "g_idx"}) {
    if (name.size() > c.size() + 1 && name.ends_with(c) && name[name.size() - c.size() - 1] == '.') {
      weight_name = std::string(name.substr(0, name.size() - c.size())) + "weight";
      component = std::string(c);
      return true;
    }
  }
  return false;
}

Result<std::vector<float>> llama3_rope_factors(const json::Value& cfg, int32_t rope_dim, float base) {
  const json::Value* rs = field(cfg, "rope_scaling");
  if (!rs || !rs->is_object()) return std::vector<float>{};
  const json::Value* t = field(*rs, "rope_type");
  if (!t) t = field(*rs, "type");
  if (!t || !t->is_string() || t->as_string() != "llama3") return std::vector<float>{};
  const double factor = get_float_or(*rs, "factor", 8.0);
  const double low = get_float_or(*rs, "low_freq_factor", 1.0);
  const double high = get_float_or(*rs, "high_freq_factor", 4.0);
  const double old_ctx = get_float_or(*rs, "original_max_position_embeddings", 8192.0);
  if (factor <= 0 || high <= low || old_ctx <= 0) return Corrupt("config.json: invalid llama3 rope_scaling");
  const double low_wavelen = old_ctx / low, high_wavelen = old_ctx / high;
  std::vector<float> out;
  for (int32_t i = 0; i < rope_dim; i += 2) {
    const double freq = 1.0 / std::pow(static_cast<double>(base), static_cast<double>(i) / rope_dim);
    const double wavelen = 2.0 * std::numbers::pi / freq;
    double f;
    if (wavelen < high_wavelen) {
      f = 1.0;
    } else if (wavelen > low_wavelen) {
      f = factor;
    } else {
      const double smooth = (old_ctx / wavelen - low) / (high - low);
      f = 1.0 / ((1.0 - smooth) / factor + smooth);
    }
    out.push_back(static_cast<float>(f));
  }
  return out;
}

Status apply_conventions(const json::Value& cfg, const ModelConfig& c, TensorRegistry& w) {
  if (c.architecture == "gemma" || c.architecture == "gemma2" || c.architecture == "gemma3") {
    ENGINE_RETURN_IF_ERROR(fold_one_plus(w, TensorRole::kOutputNorm, -1));
    for (int l = 0; l < c.num_layers; ++l) {
      for (TensorRole r : {TensorRole::kAttnNorm, TensorRole::kFfnNorm, TensorRole::kPostAttnNorm,
                           TensorRole::kPostFfnNorm, TensorRole::kAttnQNorm, TensorRole::kAttnKNorm}) {
        ENGINE_RETURN_IF_ERROR(fold_one_plus(w, r, l));
      }
    }
  }
  for (int l = 0; l < c.num_layers && c.moe.num_experts > 0; ++l) {
    // Granite-MoE input_linear [E, 2F, D] = gate|up: zero-copy halves.
    if (const Tensor* gu = w.find(TensorRole::kFfnGateUp, l); gu && gu->shape().rank() == 3) {
      const int64_t f = gu->shape()[1] / 2;
      ENGINE_ASSIGN_OR_RETURN(Tensor gate, gu->slice(1, 0, f));
      ENGINE_ASSIGN_OR_RETURN(Tensor up, gu->slice(1, f, f));
      w.take(TensorRole::kFfnGateUp, l);
      ENGINE_RETURN_IF_ERROR(w.add(TensorRole::kFfnGateExperts, l, std::move(gate)));
      ENGINE_RETURN_IF_ERROR(w.add(TensorRole::kFfnUpExperts, l, std::move(up)));
    }
    // Qwen-MoE shared_expert_gate is a [1, D] linear; the IR stores it as [D].
    if (const Tensor* g = w.find(TensorRole::kFfnSharedRouter, l); g && g->shape().rank() == 2) {
      ENGINE_ASSIGN_OR_RETURN(Tensor v, g->reshape({g->numel()}));
      w.take(TensorRole::kFfnSharedRouter, l);
      ENGINE_RETURN_IF_ERROR(w.add(TensorRole::kFfnSharedRouter, l, std::move(v)));
    }
  }
  if (c.architecture == "llama") {
    ENGINE_ASSIGN_OR_RETURN(std::vector<float> factors, llama3_rope_factors(cfg, c.rope.dim, c.rope.freq_base));
    if (!factors.empty()) {
      ENGINE_ASSIGN_OR_RETURN(Tensor t, Tensor::empty(DType::kF32, {static_cast<int64_t>(factors.size())}));
      std::copy(factors.begin(), factors.end(), t.data_as<float>());
      ENGINE_RETURN_IF_ERROR(w.add(TensorRole::kRopeFreqs, -1, std::move(t)));
    }
  }
  return Status::Ok();
}

}  // namespace engine::hf

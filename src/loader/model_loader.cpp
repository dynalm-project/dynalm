#include "loader/model_loader.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "loader/hf/hf_model.h"
#include "loader/hf/hf_tokenizer.h"
#include "loader/safetensors/safetensors.h"
#include "logging/log.h"
#include "quant/dequant.h"

namespace engine {
namespace {

Result<std::string> detect_format(const std::string& path) {
  if (hf::is_hf_model(path)) return std::string("safetensors");
  std::ifstream in(path, std::ios::binary);
  if (!in) return NotFound("cannot open model file: " + path);
  char magic[4] = {};
  in.read(magic, 4);
  if (in.gcount() == 4 && std::memcmp(magic, "GGUF", 4) == 0) return std::string("gguf");
  return Unsupported("unrecognized model format (expected a GGUF file, a Hugging Face model directory or a "
                     ".safetensors file): " + path);
}

std::string join(const std::vector<std::string_view>& v) {
  std::string s;
  for (auto x : v) s += (s.empty() ? "" : ", ") + std::string(x);
  return s;
}

Status find_adapter(LoadedModel& m) {
  m.architecture = find_architecture(m.config.architecture);
  if (!m.architecture) {
    return Unsupported("architecture '" + m.config.architecture + "' is not supported (supported: " +
                       join(supported_architecture_ids()) + ")");
  }
  return Status::Ok();
}

// Format-independent tail: adapter semantics, weight validation, tokenizer
// and chat template.
Status finish(LoadedModel& m, TokenizerData tok_data, const std::string& chat_template) {
  ENGINE_RETURN_IF_ERROR(m.architecture->configure(m.config, m.weights));
  ENGINE_RETURN_IF_ERROR(m.architecture->prepare_weights(m.config, m.weights));
  ENGINE_RETURN_IF_ERROR(m.architecture->validate(m.config, m.weights));
  if (!m.unmapped_tensors.empty()) {
    LOG_WARN("{} tensor(s) in the model are not used by the engine, e.g. '{}'", m.unmapped_tensors.size(),
             m.unmapped_tensors.front());
  }
  ENGINE_ASSIGN_OR_RETURN(m.tokenizer, Tokenizer::create(std::move(tok_data)));
  // Some models pad the embedding matrix; the tokenizer may be smaller, never larger.
  if (m.tokenizer->vocab_size() > m.config.vocab_size) {
    return Corrupt("tokenizer vocab (" + std::to_string(m.tokenizer->vocab_size()) + ") larger than model vocab (" +
                   std::to_string(m.config.vocab_size) + ")");
  }
  if (auto tmpl = ChatTemplate::from_jinja(chat_template); tmpl.ok()) m.chat_template = std::move(*tmpl);
  return Status::Ok();
}

Result<std::unique_ptr<LoadedModel>> load_gguf(const std::string& path) {
  ENGINE_ASSIGN_OR_RETURN(auto file, gguf::GgufFile::open(path));
  auto m = std::make_unique<LoadedModel>();
  m->path = path;
  m->format = "gguf";
  if (auto ft = file->get_uint("general.file_type"); ft.ok()) {
    m->quantization = std::string(gguf::file_type_name(static_cast<uint32_t>(*ft)));
  }
  ENGINE_ASSIGN_OR_RETURN(m->config, gguf::read_model_config(*file));
  ENGINE_RETURN_IF_ERROR(find_adapter(*m));
  ENGINE_ASSIGN_OR_RETURN(auto mapping, gguf::map_tensors(*file, m->config.num_layers));
  m->weights = std::move(mapping.registry);
  m->unmapped_tensors = std::move(mapping.unmapped);
  m->weight_bytes = static_cast<int64_t>(file->total_tensor_bytes());
  for (const auto& info : file->tensors()) {
    if (info.dtype && !dequant_supported(*info.dtype)) {
      return Unsupported("tensor '" + std::string(info.name) + "' has type " + std::string(dtype_name(*info.dtype)) +
                         ", which has no CPU kernel yet");
    }
  }
  ENGINE_ASSIGN_OR_RETURN(TokenizerData tok_data, gguf::read_tokenizer_data(*file));
  ENGINE_RETURN_IF_ERROR(finish(*m, std::move(tok_data), gguf::read_chat_template(*file)));
  return m;
}

// The tensors making up one GPTQ/AWQ linear layer.
struct PackedParts {
  TensorRole role = TensorRole::kCount;
  int layer = -1;
  std::map<std::string, const safetensors::TensorInfo*> parts;  // qweight, scales, qzeros, g_idx
};

// Validates and repacks every packed layer into the registry (DD-041).
// Returns a summary such as "Q4_0 x168".
Result<std::string> repack_packed_layers(const quant::PackedScheme& s, const std::map<std::string, PackedParts>& layers,
                                         const std::vector<std::unique_ptr<safetensors::SafeTensorsFile>>& shards,
                                         LoadedModel& m) {
  auto data_of = [&](const safetensors::TensorInfo* t) -> const void* {
    for (const auto& sh : shards) {
      if (!sh->tensors().empty() && t >= sh->tensors().data() && t < sh->tensors().data() + sh->tensors().size()) {
        return sh->file().data() + t->offset;
      }
    }
    return nullptr;
  };
  std::map<std::string, int> counts;
  const int per = 32 / s.bits;
  for (const auto& [name, p] : layers) {
    auto part = [&](const char* c) -> const safetensors::TensorInfo* {
      auto it = p.parts.find(c);
      return it == p.parts.end() ? nullptr : it->second;
    };
    const auto *qw = part("qweight"), *sc = part("scales"), *qz = part("qzeros"), *gi = part("g_idx");
    if (!qw || !sc || !qz) return Corrupt("packed layer '" + name + "' needs qweight, scales and qzeros");
    if (p.layer >= m.config.num_layers) return Corrupt("packed layer '" + name + "' has an out-of-range layer");
    if (qw->dtype_name != "I32" || qz->dtype_name != "I32" || sc->dtype_name != "F16" || (gi && gi->dtype_name != "I32") ||
        qw->shape.rank() != 2 || sc->shape.rank() != 2 || qz->shape.rank() != 2 || (gi && gi->shape.rank() != 1)) {
      return Corrupt("packed layer '" + name + "': unexpected component dtypes or ranks");
    }
    quant::PackedLinear w;
    if (s.method == quant::PackedMethod::kGptq) {
      w.in = qw->shape[0] * per;
      w.out = qw->shape[1];
    } else {
      w.in = qw->shape[0];
      w.out = qw->shape[1] * per;
    }
    ENGINE_ASSIGN_OR_RETURN(quant::PackedShapes shp, quant::packed_shapes(s, w.in, w.out));
    if (qw->shape[0] != shp.qweight_rows || qw->shape[1] != shp.qweight_cols || sc->shape[0] != shp.groups ||
        sc->shape[1] != w.out || qz->shape[0] != shp.groups || qz->shape[1] != shp.qzeros_cols ||
        (gi && gi->shape[0] != w.in)) {
      return Corrupt("packed layer '" + name + "': component shapes do not match " + s.describe());
    }
    w.qweight = static_cast<const int32_t*>(data_of(qw));
    w.scales = static_cast<const uint16_t*>(data_of(sc));
    w.qzeros = static_cast<const int32_t*>(data_of(qz));
    w.g_idx = gi ? static_cast<const int32_t*>(data_of(gi)) : nullptr;
    if (!w.qweight || !w.scales || !w.qzeros || (gi && !w.g_idx)) return Internal("packed component not mapped");
    quant::RepackTarget target;
    ENGINE_ASSIGN_OR_RETURN(Tensor t, quant::repack(s, w, &target));
    m.weight_bytes += static_cast<int64_t>(dtype_row_bytes(t.dtype(), w.in)) * w.out;
    ++counts[std::string(quant::repack_target_name(target))];
    ENGINE_RETURN_IF_ERROR(m.weights.add(p.role, p.layer, std::move(t)));
  }
  std::string summary;
  for (const auto& [target, n] : counts) summary += (summary.empty() ? "" : ", ") + target + " x" + std::to_string(n);
  if (counts.count("F16")) {
    LOG_WARN("{} packed layer(s) use act-order or a group size that is not a multiple of 32; they run as F16",
             counts["F16"]);
  }
  return summary.empty() ? std::string("no packed layers") : summary;
}

Result<std::unique_ptr<LoadedModel>> load_safetensors(const std::string& path) {
  namespace fs = std::filesystem;
  ENGINE_ASSIGN_OR_RETURN(hf::ModelFiles files, hf::locate(path));
  const fs::path dir(files.dir);
  ENGINE_ASSIGN_OR_RETURN(json::Value cfg, hf::read_json_file((dir / "config.json").string(), 4 << 20));

  auto m = std::make_unique<LoadedModel>();
  m->path = path;
  m->format = "safetensors";
  ENGINE_ASSIGN_OR_RETURN(m->config, hf::read_config(cfg));
  ENGINE_RETURN_IF_ERROR(find_adapter(*m));

  ENGINE_ASSIGN_OR_RETURN(std::optional<quant::PackedScheme> scheme, hf::read_quantization(cfg));

  // Map every shard's tensors; the tensors keep their mapped file alive.
  // GPTQ/AWQ components are gathered per linear layer and repacked below.
  m->weights = TensorRegistry(m->config.num_layers);
  std::map<std::string, int64_t> bytes_by_dtype;
  std::map<std::string, PackedParts> packed;  // by linear weight name
  std::map<std::pair<TensorRole, int>, std::vector<Tensor>> expert_parts;  // per-expert 2-D tensors
  const int32_t num_experts = m->config.moe.num_experts;
  if (scheme && num_experts > 0) return Unsupported("GPTQ/AWQ mixture-of-experts checkpoints are not supported yet");
  std::vector<std::unique_ptr<safetensors::SafeTensorsFile>> shards;
  for (const std::string& shard : files.weights) {
    ENGINE_ASSIGN_OR_RETURN(auto st, safetensors::SafeTensorsFile::open(shard));
    for (const safetensors::TensorInfo& info : st->tensors()) {
      TensorRole role;
      int layer, expert;
      if (num_experts > 0 && hf::parse_expert_name(info.name, role, layer, expert)) {
        if (layer >= m->config.num_layers || expert >= num_experts) {
          return Corrupt("expert tensor '" + info.name + "' is out of range");
        }
        auto& slot = expert_parts[{role, layer}];
        slot.resize(static_cast<size_t>(num_experts));
        ENGINE_ASSIGN_OR_RETURN(slot[static_cast<size_t>(expert)], st->load_tensor(info));
        m->weight_bytes += static_cast<int64_t>(info.nbytes);
        bytes_by_dtype[info.dtype_name] += static_cast<int64_t>(info.nbytes);
        continue;
      }
      std::string weight_name, component;
      if (scheme && hf::split_packed_name(info.name, weight_name, component) &&
          hf::parse_tensor_name(weight_name, m->config.architecture, role, layer)) {
        PackedParts& p = packed[weight_name];
        p.role = role;
        p.layer = layer;
        p.parts[component] = &info;
        continue;
      }
      if (!hf::parse_tensor_name(info.name, m->config.architecture, role, layer)) {
        m->unmapped_tensors.push_back(info.name);
        continue;
      }
      if (layer >= m->config.num_layers) {
        return Corrupt("tensor '" + info.name + "' refers to layer " + std::to_string(layer) + " but the model has " +
                       std::to_string(m->config.num_layers));
      }
      ENGINE_ASSIGN_OR_RETURN(Tensor t, st->load_tensor(info));
      ENGINE_RETURN_IF_ERROR(m->weights.add(role, layer, std::move(t)));
      m->weight_bytes += static_cast<int64_t>(info.nbytes);
      bytes_by_dtype[info.dtype_name] += static_cast<int64_t>(info.nbytes);
    }
    shards.push_back(std::move(st));
  }
  // Stack per-expert matrices into the IR's [experts, rows, cols] tensors
  // (one copy at load; GGUF and Granite-MoE checkpoints are already 3-D).
  for (auto& [key, parts] : expert_parts) {
    const Tensor& first = parts.front();
    for (size_t e = 0; e < parts.size(); ++e) {
      if (!parts[e].defined() || parts[e].dtype() != first.dtype() || !(parts[e].shape() == first.shape()) ||
          first.shape().rank() != 2) {
        return Corrupt("expert " + std::to_string(e) + " of " + std::string(tensor_role_name(key.first)) + " (layer " +
                       std::to_string(key.second) + ") is missing or inconsistent");
      }
    }
    const int64_t rows = first.shape()[0], cols = first.shape()[1];
    const auto bytes = static_cast<size_t>(dtype_row_bytes(first.dtype(), cols) * rows);
    ENGINE_ASSIGN_OR_RETURN(Tensor stacked, Tensor::empty(first.dtype(), {num_experts, rows, cols}));
    for (size_t e = 0; e < parts.size(); ++e) {
      std::memcpy(static_cast<std::byte*>(stacked.data()) + e * bytes, parts[e].data(), bytes);
    }
    ENGINE_RETURN_IF_ERROR(m->weights.add(key.first, key.second, std::move(stacked)));
  }
  // Reported "quantization": the dominant weight dtype (F32 / F16 / BF16).
  for (const auto& [name, bytes] : bytes_by_dtype) {
    if (m->quantization.empty() || bytes > bytes_by_dtype[m->quantization]) m->quantization = name;
  }
  if (scheme) {
    ENGINE_ASSIGN_OR_RETURN(std::string summary, repack_packed_layers(*scheme, packed, shards, *m));
    m->quantization = scheme->describe() + " -> " + summary;
  }
  ENGINE_RETURN_IF_ERROR(hf::apply_conventions(cfg, m->config, m->weights));

  ENGINE_ASSIGN_OR_RETURN(json::Value tok_json, hf::read_json_file((dir / "tokenizer.json").string()));
  auto optional_json = [&](const char* name) -> std::optional<json::Value> {
    auto v = hf::read_json_file((dir / name).string(), 16 << 20);
    if (v.ok()) return std::move(*v);
    return std::nullopt;
  };
  const std::optional<json::Value> tok_cfg = optional_json("tokenizer_config.json");
  const std::optional<json::Value> gen_cfg = optional_json("generation_config.json");
  hf::TokenizerFiles tf;
  tf.tokenizer = &tok_json;
  tf.tokenizer_config = tok_cfg ? &*tok_cfg : nullptr;
  tf.generation_config = gen_cfg ? &*gen_cfg : nullptr;
  tf.config = &cfg;
  ENGINE_ASSIGN_OR_RETURN(TokenizerData tok_data, hf::read_tokenizer(tf));
  ENGINE_RETURN_IF_ERROR(finish(*m, std::move(tok_data), hf::read_chat_template(tf.tokenizer_config)));
  return m;
}

}  // namespace

Result<std::unique_ptr<LoadedModel>> load_model(const std::string& path) {
  ENGINE_ASSIGN_OR_RETURN(std::string format, detect_format(path));
  if (format == "safetensors") return load_safetensors(path);
  return load_gguf(path);
}

}  // namespace engine

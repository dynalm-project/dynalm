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

  // Map every shard's tensors; the tensors keep their mapped file alive.
  m->weights = TensorRegistry(m->config.num_layers);
  std::map<std::string, int64_t> bytes_by_dtype;
  for (const std::string& shard : files.weights) {
    ENGINE_ASSIGN_OR_RETURN(auto st, safetensors::SafeTensorsFile::open(shard));
    for (const safetensors::TensorInfo& info : st->tensors()) {
      TensorRole role;
      int layer;
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
  }
  // Reported "quantization": the dominant weight dtype (F32 / F16 / BF16).
  for (const auto& [name, bytes] : bytes_by_dtype) {
    if (m->quantization.empty() || bytes > bytes_by_dtype[m->quantization]) m->quantization = name;
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

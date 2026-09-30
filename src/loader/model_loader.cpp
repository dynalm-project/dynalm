#include "loader/model_loader.h"

#include <cstring>
#include <fstream>

#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "logging/log.h"
#include "quant/dequant.h"

namespace engine {
namespace {

Result<std::string> detect_format(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return NotFound("cannot open model file: " + path);
  char magic[4] = {};
  in.read(magic, 4);
  if (in.gcount() == 4 && std::memcmp(magic, "GGUF", 4) == 0) return std::string("gguf");
  return Unsupported("unrecognized model format (only GGUF is supported): " + path);
}

std::string join(const std::vector<std::string_view>& v) {
  std::string s;
  for (auto x : v) s += (s.empty() ? "" : ", ") + std::string(x);
  return s;
}

}  // namespace

Result<std::unique_ptr<LoadedModel>> load_model(const std::string& path) {
  ENGINE_ASSIGN_OR_RETURN(std::string format, detect_format(path));
  ENGINE_ASSIGN_OR_RETURN(auto file, gguf::GgufFile::open(path));

  auto m = std::make_unique<LoadedModel>();
  m->path = path;
  m->format = format;
  if (auto ft = file->get_uint("general.file_type"); ft.ok()) {
    m->quantization = std::string(gguf::file_type_name(static_cast<uint32_t>(*ft)));
  }

  ENGINE_ASSIGN_OR_RETURN(m->config, gguf::read_model_config(*file));
  m->architecture = find_architecture(m->config.architecture);
  if (!m->architecture) {
    return Unsupported("architecture '" + m->config.architecture + "' is not supported (supported: " +
                       join(supported_architecture_ids()) + ")");
  }

  ENGINE_ASSIGN_OR_RETURN(auto mapping, gguf::map_tensors(*file, m->config.num_layers));
  m->weights = std::move(mapping.registry);
  m->unmapped_tensors = std::move(mapping.unmapped);
  m->weight_bytes = static_cast<int64_t>(file->total_tensor_bytes());

  ENGINE_RETURN_IF_ERROR(m->architecture->configure(m->config, m->weights));
  ENGINE_RETURN_IF_ERROR(m->architecture->validate(m->config, m->weights));

  for (const auto& info : file->tensors()) {
    if (info.dtype && !dequant_supported(*info.dtype)) {
      return Unsupported("tensor '" + std::string(info.name) + "' has type " +
                         std::string(dtype_name(*info.dtype)) + ", which has no CPU kernel yet");
    }
  }
  if (!m->unmapped_tensors.empty()) {
    LOG_WARN("{} tensor(s) in the file are not used by the engine, e.g. '{}'", m->unmapped_tensors.size(),
             m->unmapped_tensors.front());
  }

  ENGINE_ASSIGN_OR_RETURN(TokenizerData tok_data, gguf::read_tokenizer_data(*file));
  ENGINE_ASSIGN_OR_RETURN(m->tokenizer, Tokenizer::create(std::move(tok_data)));
  if (m->tokenizer->vocab_size() != m->config.vocab_size) {
    // Some models pad the embedding matrix; the tokenizer may be smaller.
    if (m->tokenizer->vocab_size() > m->config.vocab_size) {
      return Corrupt("tokenizer vocab (" + std::to_string(m->tokenizer->vocab_size()) +
                     ") larger than model vocab (" + std::to_string(m->config.vocab_size) + ")");
    }
  }

  if (auto tmpl = ChatTemplate::from_jinja(gguf::read_chat_template(*file)); tmpl.ok()) {
    m->chat_template = std::move(*tmpl);
  }
  return m;
}

}  // namespace engine

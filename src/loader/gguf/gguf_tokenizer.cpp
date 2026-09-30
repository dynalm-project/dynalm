#include "loader/gguf/gguf_tokenizer.h"

namespace engine::gguf {
namespace {

TokenId optional_id(const GgufFile& f, std::string_view key) {
  auto v = f.get_int(key);
  return v.ok() ? static_cast<TokenId>(*v) : kNoToken;
}

bool optional_bool(const GgufFile& f, std::string_view key, bool fallback) {
  auto v = f.get_bool(key);
  return v.ok() ? *v : fallback;
}

}  // namespace

Result<TokenizerData> read_tokenizer_data(const GgufFile& f) {
  TokenizerData d;
  ENGINE_ASSIGN_OR_RETURN(std::string_view model, f.get_string("tokenizer.ggml.model"));
  if (model == "gpt2") {
    d.model = TokenizerData::Model::kBpe;
  } else if (model == "llama") {
    d.model = TokenizerData::Model::kSpm;
  } else {
    return Unsupported("tokenizer model '" + std::string(model) + "' is not supported yet");
  }
  if (auto pre = f.get_string("tokenizer.ggml.pre"); pre.ok()) d.pre = std::string(*pre);

  ENGINE_ASSIGN_OR_RETURN(Array tok_arr, f.get_array("tokenizer.ggml.tokens"));
  ENGINE_ASSIGN_OR_RETURN(auto toks, GgufFile::array_strings(tok_arr));
  d.tokens.assign(toks.begin(), toks.end());

  if (auto a = f.get_array("tokenizer.ggml.scores"); a.ok()) {
    ENGINE_ASSIGN_OR_RETURN(d.scores, GgufFile::array_floats(*a));
  }
  if (auto a = f.get_array("tokenizer.ggml.token_type"); a.ok()) {
    ENGINE_ASSIGN_OR_RETURN(auto types, GgufFile::array_ints(*a));
    d.types.reserve(types.size());
    for (int64_t t : types) {
      d.types.push_back((t >= 0 && t <= 6) ? static_cast<TokenType>(t) : TokenType::kNormal);
    }
  }
  if (auto a = f.get_array("tokenizer.ggml.merges"); a.ok()) {
    ENGINE_ASSIGN_OR_RETURN(auto merges, GgufFile::array_strings(*a));
    d.merges.assign(merges.begin(), merges.end());
  }

  d.bos = optional_id(f, "tokenizer.ggml.bos_token_id");
  d.eos = optional_id(f, "tokenizer.ggml.eos_token_id");
  d.unk = optional_id(f, "tokenizer.ggml.unknown_token_id");
  d.pad = optional_id(f, "tokenizer.ggml.padding_token_id");
  for (std::string_view key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
    if (TokenId id = optional_id(f, key); id != kNoToken) d.eog.push_back(id);
  }

  // Defaults follow the model families: SPM vocabularies add BOS, byte-level
  // BPE ones generally don't unless the metadata says so.
  const bool spm = d.model == TokenizerData::Model::kSpm;
  d.add_bos = optional_bool(f, "tokenizer.ggml.add_bos_token", spm);
  d.add_eos = optional_bool(f, "tokenizer.ggml.add_eos_token", false);
  d.add_space_prefix = optional_bool(f, "tokenizer.ggml.add_space_prefix", spm);
  return d;
}

std::string read_chat_template(const GgufFile& f) {
  auto t = f.get_string("tokenizer.chat_template");
  return t.ok() ? std::string(*t) : std::string();
}

}  // namespace engine::gguf

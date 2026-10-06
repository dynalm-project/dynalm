#pragma once

// Hugging Face tokenizer files -> TokenizerData (the same structure the GGUF
// loader fills), so one tokenizer implementation serves both formats.
//
// Supported tokenizer.json models (DD-040):
// - byte-level BPE (GPT-2 style ByteLevel pre-tokenizer/decoder) whose
//   pre-tokenizer matches a known split (GPT-2, Llama 3, Qwen2, SmolLM/StarCoder);
// - SentencePiece-style BPE with byte fallback ("▁" for spaces: Gemma, Llama 2,
//   Mistral), mapped to the SPM tokenizer with merge rank as piece score.
// Anything else (Unigram, WordPiece, unknown splits) is rejected clearly.

#include <string>

#include "api/json.h"
#include "dynacore/base/status.h"
#include "tokenizer/tokenizer.h"

namespace engine::hf {

struct TokenizerFiles {
  const json::Value* tokenizer = nullptr;          // tokenizer.json (required)
  const json::Value* tokenizer_config = nullptr;   // tokenizer_config.json (optional)
  const json::Value* generation_config = nullptr;  // generation_config.json (optional)
  const json::Value* config = nullptr;             // config.json (optional, token ids)
};

Result<TokenizerData> read_tokenizer(const TokenizerFiles& files);

// The chat template from tokenizer_config.json ("" if none).
std::string read_chat_template(const json::Value* tokenizer_config);

}  // namespace engine::hf

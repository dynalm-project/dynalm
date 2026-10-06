#pragma once

// GGUF `tokenizer.ggml.*` / `tokenizer.chat_template` → format-neutral data.

#include <string>

#include "dynacore/base/status.h"
#include "loader/gguf/gguf.h"
#include "tokenizer/tokenizer.h"

namespace engine::gguf {

Result<TokenizerData> read_tokenizer_data(const GgufFile& file);

// Empty string when the file carries no template.
std::string read_chat_template(const GgufFile& file);

}  // namespace engine::gguf

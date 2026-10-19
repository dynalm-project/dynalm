#pragma once

// GGUF `tokenizer.ggml.*` / `tokenizer.chat_template` → format-neutral data.

#include <string>
#include <string_view>

#include "dynacore/base/status.h"
#include "loader/gguf/gguf.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm::gguf {

Result<TokenizerData> read_tokenizer_data(const GgufFile& file);

// Whether the engine has a tokenizer for `tokenizer.ggml.model` = `model` and
// `tokenizer.ggml.pre` = `pre` (empty: the key is absent, the GPT-2 split).
// The one rule both read_tokenizer_data / Tokenizer::create (at load) and
// `dynalm pull --check` (from the first 256 KiB) apply.
Status check_tokenizer_kind(std::string_view model, std::string_view pre);

// Empty string when the file carries no template.
std::string read_chat_template(const GgufFile& file);

}  // namespace dynalm::gguf

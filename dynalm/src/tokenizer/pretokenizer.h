#pragma once

// Byte-level BPE pre-tokenizers: split text into words before BPE merges.
//
// Each variant is a hand-written matcher equivalent to the regex the model's
// HF tokenizer uses (std::regex has no Unicode classes and is slow). The
// variants and their source regexes:
//
//   kGpt2:      's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//   kLlama3:    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//               ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   kQwen2:     as kLlama3 with \p{N} (single digits)
//   kStarCoder: split every \p{N} into its own word, then kGpt2 on the rest
//               (SmolLM, StarCoder, Refact, Command-R)

#include <string_view>
#include <vector>

#include "dynacore/base/status.h"

namespace engine {

enum class PreTokenizer { kGpt2, kLlama3, kQwen2, kStarCoder };

// Maps a GGUF `tokenizer.ggml.pre` id (e.g. "llama-bpe", "qwen2", "smollm").
// Unknown ids are an error: silently using the wrong split changes token IDs.
Result<PreTokenizer> pretokenizer_from_name(std::string_view name);

// Appends the words of `text` to `out` as views into `text`. Every byte of
// `text` lands in exactly one word, in order.
void pretokenize(PreTokenizer kind, std::string_view text, std::vector<std::string_view>& out);

}  // namespace engine

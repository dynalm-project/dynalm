#pragma once

// Tokenizer abstraction.
//
// Built from a format-neutral TokenizerData (the GGUF loader fills it from
// `tokenizer.ggml.*`; a future HF loader from tokenizer.json). Two models:
//   kBpe — byte-level BPE with merge ranks (GPT-2, Llama-3, Qwen2, SmolLM)
//   kSpm — SentencePiece-style BPE by piece score, with byte fallback
//          (Llama-2, Mistral, Gemma, Phi-3)
//
// Token IDs are never hard-coded: BOS/EOS/end-of-generation sets come from
// model metadata.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dynacore/base/status.h"

namespace engine {

using TokenId = int32_t;
inline constexpr TokenId kNoToken = -1;

enum class TokenType : uint8_t {
  kUndefined = 0,
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,      // <s>, <|im_start|> ... matched only when parsing specials
  kUserDefined = 4,  // added tokens: always matched as a whole
  kUnused = 5,
  kByte = 6,         // <0xAB> byte-fallback pieces (SPM)
};

struct TokenizerData {
  enum class Model { kBpe, kSpm };
  Model model = Model::kBpe;
  std::string pre = "default";  // BPE pre-tokenizer id

  std::vector<std::string> tokens;
  std::vector<float> scores;      // SPM merge priority (higher first)
  std::vector<TokenType> types;   // empty = all normal
  std::vector<std::string> merges;  // BPE, "left right", highest priority first

  TokenId bos = kNoToken;
  TokenId eos = kNoToken;
  TokenId unk = kNoToken;
  TokenId pad = kNoToken;
  std::vector<TokenId> eog;  // additional end-of-generation tokens (eot, eom, ...)

  bool add_bos = false;
  bool add_eos = false;
  bool add_space_prefix = true;  // SPM only
};

class Tokenizer {
 public:
  static Result<std::unique_ptr<Tokenizer>> create(TokenizerData data);
  virtual ~Tokenizer() = default;

  // Appends tokens for `text`. `add_special` adds BOS/EOS per the model's
  // flags. `parse_special` recognizes control-token text such as
  // "<|im_start|>" (use for trusted templates, never for raw user input).
  Status encode(std::string_view text, bool add_special, bool parse_special,
                std::vector<TokenId>& out) const;
  std::vector<TokenId> encode(std::string_view text, bool add_special = true,
                              bool parse_special = false) const;

  // Appends the bytes of one token. Control tokens render only when
  // `render_special`. Output may end mid-UTF-8 sequence (see Utf8Buffer).
  void decode_token(TokenId id, std::string& out, bool render_special = false) const;
  std::string decode(std::span<const TokenId> ids, bool render_special = false) const;

  int32_t vocab_size() const { return static_cast<int32_t>(data_.tokens.size()); }
  const std::string& token_text(TokenId id) const { return data_.tokens[static_cast<size_t>(id)]; }
  TokenType token_type(TokenId id) const;
  TokenId bos() const { return data_.bos; }
  TokenId eos() const { return data_.eos; }
  bool is_eog(TokenId id) const { return eog_.contains(id); }
  const std::unordered_set<TokenId>& eog_tokens() const { return eog_; }
  TokenizerData::Model model() const { return data_.model; }

  // Looks up an exact token string; kNoToken if absent.
  TokenId find_token(std::string_view text) const;

 protected:
  explicit Tokenizer(TokenizerData data);
  Status init_common();

  // Tokenizes a fragment containing no special tokens.
  virtual void encode_fragment(std::string_view text, bool at_start, std::vector<TokenId>& out) const = 0;
  // Bytes of a normal/byte token.
  virtual void decode_normal(TokenId id, std::string& out) const = 0;

  TokenizerData data_;
  std::vector<std::string_view> vocab_keys_;  // views into data_.tokens
  struct Hash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
  };
  std::unordered_map<std::string_view, TokenId, Hash, std::equal_to<>> vocab_;
  // Control + user-defined tokens bucketed by first byte, longest text first.
  std::vector<TokenId> specials_by_byte_[256];
  std::unordered_set<TokenId> eog_;
};

// Accumulates decoded bytes and releases only complete UTF-8 sequences, so a
// streamed character split across tokens is never emitted half-way.
class Utf8Buffer {
 public:
  // Appends `bytes`; moves every complete prefix into `out`.
  void push(std::string_view bytes, std::string& out);
  // Remaining bytes (incomplete sequence) at end of stream.
  void flush(std::string& out);

 private:
  std::string pending_;
};

}  // namespace engine

#pragma once

// Token ids → streamable text for one request.
//
// Guarantees:
//   - output is always valid, complete UTF-8 (a character split across tokens
//     is emitted once complete);
//   - stop strings are never emitted, even partially: text is held back only
//     while it could still be the start of a stop string (the minimal
//     suffix), so normal text flows immediately;
//   - control tokens (EOS, <|im_end|>, ...) are not rendered.

#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

class TextStreamer {
 public:
  TextStreamer(const Tokenizer& tokenizer, std::vector<std::string> stop_strings);

  struct Delta {
    std::string text;      // safe to emit now
    bool stopped = false;  // a stop string completed; `text` ends before it
  };

  // Adds one generated token.
  Delta push(TokenId id);
  // End of generation (no stop matched): everything still held back.
  std::string finish();

  bool stopped() const { return stopped_; }

 private:
  // Length of the longest suffix of pending_ that is a proper prefix of a stop string.
  size_t held_suffix() const;

  const Tokenizer& tokenizer_;
  std::vector<std::string> stops_;
  Utf8Buffer utf8_;
  std::string piece_, pending_;
  bool stopped_ = false;
};

}  // namespace dynalm

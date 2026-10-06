#pragma once

// Minimal Unicode support for tokenization: UTF-8 decoding and the three
// general categories the pre-tokenizers need.

#include <cstdint>
#include <string>
#include <string_view>
#include "common/core.h"

namespace dynalm::unicode {

struct CodepointRange {
  uint32_t first, last;
};

// Decodes one code point at `s[i]`, advancing `i`. Invalid or truncated
// sequences decode as the single byte value (0x80..0xFF) so every input byte
// is consumed exactly once and nothing is dropped.
uint32_t decode_utf8(std::string_view s, size_t& i);

void append_utf8(uint32_t cp, std::string& out);

bool is_letter(uint32_t cp);
bool is_number(uint32_t cp);
bool is_whitespace(uint32_t cp);

// Length of a complete UTF-8 sequence starting with lead byte `c`, or 0 if
// `c` is not a valid lead byte.
int utf8_sequence_length(unsigned char c);

}  // namespace dynalm::unicode

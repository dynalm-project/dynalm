#include "tokenizer/unicode.h"

#include <algorithm>
#include <iterator>

namespace dynalm::unicode {
namespace {

#include "tokenizer/unicode_tables.inc"
#include "common/core.h"

template <size_t N>
bool in_ranges(const CodepointRange (&ranges)[N], uint32_t cp) {
  // Binary search for the last range with first <= cp.
  auto it = std::upper_bound(std::begin(ranges), std::end(ranges), cp,
                             [](uint32_t v, const CodepointRange& r) { return v < r.first; });
  return it != std::begin(ranges) && cp <= std::prev(it)->last;
}

}  // namespace

int utf8_sequence_length(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return c >= 0xC2 ? 2 : 0;  // C0/C1 are overlong
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return c <= 0xF4 ? 4 : 0;
  return 0;
}

uint32_t decode_utf8(std::string_view s, size_t& i) {
  const auto c0 = static_cast<unsigned char>(s[i]);
  const int len = utf8_sequence_length(c0);
  if (len == 1 || len == 0 || i + len > s.size()) {
    ++i;
    return c0;
  }
  uint32_t cp = c0 & (0xFF >> (len + 1));
  for (int k = 1; k < len; ++k) {
    const auto c = static_cast<unsigned char>(s[i + k]);
    if ((c & 0xC0) != 0x80) {
      ++i;
      return c0;
    }
    cp = (cp << 6) | (c & 0x3F);
  }
  // Reject overlong encodings and surrogates.
  static constexpr uint32_t kMin[] = {0, 0, 0x80, 0x800, 0x10000};
  if (cp < kMin[len] || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
    ++i;
    return c0;
  }
  i += len;
  return cp;
}

void append_utf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

bool is_letter(uint32_t cp) {
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return in_ranges(kLetterRanges, cp);
}

bool is_number(uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(kNumberRanges, cp);
}

bool is_whitespace(uint32_t cp) {
  if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
  return in_ranges(kWhitespaceRanges, cp);
}

}  // namespace dynalm::unicode

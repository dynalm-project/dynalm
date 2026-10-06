#include "tokenizer/pretokenizer.h"

#include <string>

#include "tokenizer/unicode.h"
#include "common/core.h"

namespace dynalm {
namespace {

using unicode::is_letter;
using unicode::is_number;
using unicode::is_whitespace;

// Code points with their starting byte offsets (offsets[n] = text size).
// Matchers take `n`, the logical end of the text being split, which is
// smaller than size() when splitting a sub-segment.
struct Cps {
  std::vector<uint32_t> cp;
  std::vector<size_t> off;

  void decode(std::string_view s) {
    cp.clear();
    off.clear();
    for (size_t i = 0; i < s.size();) {
      off.push_back(i);
      cp.push_back(unicode::decode_utf8(s, i));
    }
    off.push_back(s.size());
  }
  size_t size() const { return cp.size(); }
};

bool is_other(uint32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); }
bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Length (in code points) of a contraction at i: 's 't 'm 'd 're 've 'll.
size_t match_contraction(const Cps& s, size_t i, size_t n, bool ignore_case) {
  if (s.cp[i] != '\'' || i + 1 >= n) return 0;
  auto at = [&](size_t k) { return ignore_case ? lower(s.cp[k]) : s.cp[k]; };
  const uint32_t a = at(i + 1);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  if (i + 2 < n) {
    const uint32_t b = at(i + 2);
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  }
  return 0;
}

template <typename Pred>
size_t run_end(const Cps& s, size_t i, size_t n, Pred pred) {
  while (i < n && pred(s.cp[i])) ++i;
  return i;
}

// \s+(?!\S) then \s+ : a whitespace run leaves its last char for the next
// word when followed by non-space (so " word" keeps its leading space).
size_t match_trailing_ws(const Cps& s, size_t i, size_t n) {
  const size_t e = run_end(s, i, n, is_whitespace);
  if (e == n || e - i == 1) return e - i;
  return e - i - 1;
}

size_t match_gpt2(const Cps& s, size_t i, size_t n) {
  if (size_t m = match_contraction(s, i, n, false)) return m;
  // ` ?\p{L}+ | ` ?\p{N}+ | ` ?[^\s\p{L}\p{N}]+` — classes are disjoint.
  const size_t k = (s.cp[i] == ' ' && i + 1 < n) ? i + 1 : i;
  const uint32_t c = s.cp[k];
  if (is_letter(c)) return run_end(s, k, n, is_letter) - i;
  if (is_number(c)) return run_end(s, k, n, is_number) - i;
  if (is_other(c)) return run_end(s, k, n, is_other) - i;
  return match_trailing_ws(s, i, n);
}

size_t match_llama3(const Cps& s, size_t i, size_t n, size_t max_digits) {
  const uint32_t c = s.cp[i];
  if (size_t m = match_contraction(s, i, n, true)) return m;
  // [^\r\n\p{L}\p{N}]?\p{L}+
  if (is_letter(c)) return run_end(s, i, n, is_letter) - i;
  if (!is_crlf(c) && !is_number(c) && i + 1 < n && is_letter(s.cp[i + 1])) {
    return run_end(s, i + 1, n, is_letter) - i;
  }
  // \p{N}{1,max}
  if (is_number(c)) {
    size_t e = i;
    while (e < n && e - i < max_digits && is_number(s.cp[e])) ++e;
    return e - i;
  }
  // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
  const size_t k = (c == ' ' && i + 1 < n) ? i + 1 : i;
  if (is_other(s.cp[k])) {
    const size_t e = run_end(s, k, n, is_other);
    return run_end(s, e, n, is_crlf) - i;
  }
  // \s*[\r\n]+ : whitespace up to and including the last newline of the run.
  const size_t e = run_end(s, i, n, is_whitespace);
  for (size_t m = e; m > i; --m) {
    if (is_crlf(s.cp[m - 1])) return m - i;
  }
  return match_trailing_ws(s, i, n);
}

template <typename Match>
void split(const Cps& s, std::string_view text, size_t begin, size_t end, Match match,
           std::vector<std::string_view>& out) {
  for (size_t i = begin; i < end;) {
    size_t len = match(i, end);
    if (len == 0) len = 1;  // unreachable for well-formed matchers; never stall
    out.push_back(text.substr(s.off[i], s.off[i + len] - s.off[i]));
    i += len;
  }
}

}  // namespace

Result<PreTokenizer> pretokenizer_from_name(std::string_view name) {
  if (name == "default" || name == "gpt2" || name == "gpt-2" || name == "mpt" || name == "olmo") {
    return PreTokenizer::kGpt2;
  }
  if (name == "llama3" || name == "llama-bpe" || name == "llama-v3" || name == "dbrx" ||
      name == "smaug-bpe" || name == "falcon3") {
    return PreTokenizer::kLlama3;
  }
  if (name == "qwen2" || name == "deepseek-r1-qwen" || name == "megrez") {
    return PreTokenizer::kQwen2;
  }
  if (name == "smollm" || name == "starcoder" || name == "refact" || name == "command-r" ||
      name == "codeshell" || name == "exaone" || name == "minerva-7b") {
    return PreTokenizer::kStarCoder;
  }
  return Unsupported("pre-tokenizer '" + std::string(name) + "' is not supported yet");
}

void pretokenize(PreTokenizer kind, std::string_view text, std::vector<std::string_view>& out) {
  // Reused per thread: tokenization runs per request, not per token, but
  // prompts can be long and this avoids regrowing two vectors each call.
  thread_local Cps s;
  s.decode(text);
  const size_t n = s.size();

  switch (kind) {
    case PreTokenizer::kGpt2:
      split(s, text, 0, n, [&](size_t i, size_t e) { return match_gpt2(s, i, e); }, out);
      break;
    case PreTokenizer::kLlama3:
      split(s, text, 0, n, [&](size_t i, size_t e) { return match_llama3(s, i, e, 3); }, out);
      break;
    case PreTokenizer::kQwen2:
      split(s, text, 0, n, [&](size_t i, size_t e) { return match_llama3(s, i, e, 1); }, out);
      break;
    case PreTokenizer::kStarCoder: {
      // Stage 1: every number code point is its own word. Stage 2: GPT-2
      // split of the text between numbers (matching never crosses a digit).
      size_t seg = 0;
      for (size_t i = 0; i <= n; ++i) {
        if (i == n || is_number(s.cp[i])) {
          split(s, text, seg, i, [&](size_t k, size_t e) { return match_gpt2(s, k, e); }, out);
          if (i < n) out.push_back(text.substr(s.off[i], s.off[i + 1] - s.off[i]));
          seg = i + 1;
        }
      }
      break;
    }
  }
}

}  // namespace dynalm

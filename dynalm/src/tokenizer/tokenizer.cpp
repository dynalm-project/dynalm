#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>

#include "tokenizer/pretokenizer.h"
#include "tokenizer/unicode.h"

namespace engine {

// ===========================================================================
// Common

Tokenizer::Tokenizer(TokenizerData data) : data_(std::move(data)) {}

TokenType Tokenizer::token_type(TokenId id) const {
  if (data_.types.empty()) return TokenType::kNormal;
  return data_.types[static_cast<size_t>(id)];
}

TokenId Tokenizer::find_token(std::string_view text) const {
  auto it = vocab_.find(text);
  return it == vocab_.end() ? kNoToken : it->second;
}

Status Tokenizer::init_common() {
  const size_t n = data_.tokens.size();
  if (n == 0) return InvalidArgument("tokenizer: empty vocabulary");
  if (!data_.types.empty() && data_.types.size() != n) return Corrupt("tokenizer: token_type size mismatch");
  if (!data_.scores.empty() && data_.scores.size() != n) return Corrupt("tokenizer: scores size mismatch");
  auto check_id = [&](TokenId id, const char* what) -> Status {
    if (id != kNoToken && (id < 0 || static_cast<size_t>(id) >= n)) {
      return Corrupt(std::string("tokenizer: ") + what + " id out of range");
    }
    return Status::Ok();
  };
  ENGINE_RETURN_IF_ERROR(check_id(data_.bos, "bos"));
  ENGINE_RETURN_IF_ERROR(check_id(data_.eos, "eos"));
  ENGINE_RETURN_IF_ERROR(check_id(data_.unk, "unk"));
  ENGINE_RETURN_IF_ERROR(check_id(data_.pad, "pad"));
  for (TokenId id : data_.eog) ENGINE_RETURN_IF_ERROR(check_id(id, "eog"));

  vocab_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    // First occurrence wins on duplicate strings (some vocabs have them).
    vocab_.emplace(data_.tokens[i], static_cast<TokenId>(i));
  }

  for (size_t i = 0; i < n; ++i) {
    const TokenType t = token_type(static_cast<TokenId>(i));
    if ((t == TokenType::kControl || t == TokenType::kUserDefined) && !data_.tokens[i].empty()) {
      specials_by_byte_[static_cast<unsigned char>(data_.tokens[i][0])].push_back(static_cast<TokenId>(i));
    }
  }
  for (auto& bucket : specials_by_byte_) {
    std::stable_sort(bucket.begin(), bucket.end(), [&](TokenId a, TokenId b) {
      return data_.tokens[static_cast<size_t>(a)].size() > data_.tokens[static_cast<size_t>(b)].size();
    });
  }

  // End-of-generation: EOS, explicit extras, and well-known chat terminators
  // present as control tokens (metadata often omits eot ids).
  if (data_.eos != kNoToken) eog_.insert(data_.eos);
  for (TokenId id : data_.eog) eog_.insert(id);
  static constexpr std::string_view kTerminators[] = {
      "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>",
      "<|end_of_text|>", "<｜end▁of▁sentence｜>", "<EOT>", "<|return|>"};
  for (std::string_view t : kTerminators) {
    const TokenId id = find_token(t);
    if (id != kNoToken && token_type(id) == TokenType::kControl) eog_.insert(id);
  }
  return Status::Ok();
}

Status Tokenizer::encode(std::string_view text, bool add_special, bool parse_special,
                         std::vector<TokenId>& out) const {
  if (add_special && data_.add_bos && data_.bos != kNoToken) out.push_back(data_.bos);

  // Split on special-token text (longest match wins), BPE the gaps.
  size_t frag_start = 0;
  bool at_start = true;
  auto flush = [&](size_t end) {
    if (end > frag_start) {
      encode_fragment(text.substr(frag_start, end - frag_start), at_start, out);
      at_start = false;
    }
  };
  for (size_t i = 0; i < text.size();) {
    TokenId matched = kNoToken;
    for (TokenId id : specials_by_byte_[static_cast<unsigned char>(text[i])]) {
      if (!parse_special && token_type(id) == TokenType::kControl) continue;
      const std::string& s = data_.tokens[static_cast<size_t>(id)];
      if (text.compare(i, s.size(), s) == 0) {
        matched = id;
        break;
      }
    }
    if (matched == kNoToken) {
      ++i;
      continue;
    }
    flush(i);
    out.push_back(matched);
    at_start = false;
    i += data_.tokens[static_cast<size_t>(matched)].size();
    frag_start = i;
  }
  flush(text.size());

  if (add_special && data_.add_eos && data_.eos != kNoToken) out.push_back(data_.eos);
  return Status::Ok();
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool add_special, bool parse_special) const {
  std::vector<TokenId> out;
  (void)encode(text, add_special, parse_special, out);
  return out;
}

void Tokenizer::decode_token(TokenId id, std::string& out, bool render_special) const {
  if (id < 0 || id >= vocab_size()) return;
  switch (token_type(id)) {
    case TokenType::kControl:
    case TokenType::kUnknown:
    case TokenType::kUnused:
      if (render_special) out += data_.tokens[static_cast<size_t>(id)];
      return;
    case TokenType::kUserDefined:
      out += data_.tokens[static_cast<size_t>(id)];
      return;
    default:
      decode_normal(id, out);
  }
}

std::string Tokenizer::decode(std::span<const TokenId> ids, bool render_special) const {
  std::string out;
  for (TokenId id : ids) decode_token(id, out, render_special);
  // SPM with a space prefix: the leading "▁" of the first piece is synthetic.
  if (data_.model == TokenizerData::Model::kSpm && data_.add_space_prefix && !out.empty() &&
      out[0] == ' ') {
    out.erase(0, 1);
  }
  return out;
}

// ===========================================================================
// Byte-level BPE

namespace {

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to
// U+0100 onwards, so every byte is a visible character in the vocab.
struct ByteMap {
  uint32_t byte_to_cp[256];
  std::unordered_map<uint32_t, uint8_t> cp_to_byte;

  ByteMap() {
    uint32_t next = 256;
    for (uint32_t b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
      byte_to_cp[b] = printable ? b : next++;
      cp_to_byte[byte_to_cp[b]] = static_cast<uint8_t>(b);
    }
  }
};

const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

class BpeTokenizer final : public Tokenizer {
 public:
  explicit BpeTokenizer(TokenizerData data) : Tokenizer(std::move(data)) {}

  Status init() {
    ENGINE_RETURN_IF_ERROR(init_common());
    ENGINE_ASSIGN_OR_RETURN(pre_, pretokenizer_from_name(data_.pre));
    const ByteMap& bm = byte_map();
    for (int b = 0; b < 256; ++b) {
      std::string s;
      unicode::append_utf8(bm.byte_to_cp[b], s);
      byte_token_[b] = find_token(s);
    }
    merges_.reserve(data_.merges.size());
    for (size_t rank = 0; rank < data_.merges.size(); ++rank) {
      const std::string& m = data_.merges[rank];
      const size_t sp = m.find(' ', 1);  // a leading space would be a mapped byte, never raw
      if (sp == std::string::npos) return Corrupt("tokenizer: malformed merge '" + m + "'");
      const std::string_view left(m.data(), sp);
      const std::string_view right(m.data() + sp + 1, m.size() - sp - 1);
      const TokenId l = find_token(left), r = find_token(right);
      const TokenId merged = find_token(std::string(left) + std::string(right));
      if (l == kNoToken || r == kNoToken || merged == kNoToken) continue;
      merges_.emplace(key(l, r), Merge{static_cast<int32_t>(rank), merged});
    }
    return Status::Ok();
  }

 protected:
  void encode_fragment(std::string_view text, bool /*at_start*/, std::vector<TokenId>& out) const override {
    thread_local std::vector<std::string_view> words;
    words.clear();
    pretokenize(pre_, text, words);
    for (std::string_view w : words) encode_word(w, out);
  }

  void decode_normal(TokenId id, std::string& out) const override {
    const std::string& t = data_.tokens[static_cast<size_t>(id)];
    const ByteMap& bm = byte_map();
    for (size_t i = 0; i < t.size();) {
      const uint32_t cp = unicode::decode_utf8(t, i);
      auto it = bm.cp_to_byte.find(cp);
      if (it != bm.cp_to_byte.end()) {
        out += static_cast<char>(it->second);
      } else {
        unicode::append_utf8(cp, out);  // not byte-mapped: emit as-is
      }
    }
  }

 private:
  struct Merge {
    int32_t rank;
    TokenId merged;
  };
  static uint64_t key(TokenId l, TokenId r) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(l)) << 32) | static_cast<uint32_t>(r);
  }

  // Standard BPE: repeatedly merge the adjacent pair with the lowest rank.
  // Doubly linked symbols + min-heap with lazy invalidation: O(n log n).
  void encode_word(std::string_view word, std::vector<TokenId>& out) const {
    struct Sym {
      TokenId id;
      int prev, next;
    };
    thread_local std::vector<Sym> syms;
    syms.clear();
    for (unsigned char b : word) {
      const TokenId id = byte_token_[b];
      syms.push_back({id == kNoToken ? data_.unk : id, static_cast<int>(syms.size()) - 1,
                      static_cast<int>(syms.size()) + 1});
    }
    if (syms.empty()) return;
    syms.back().next = -1;

    struct Cand {
      int32_t rank;
      int left;
      TokenId l, r;  // ids at push time, to detect stale entries
      bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : left > o.left; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<>> heap;
    auto push = [&](int left) {
      if (left < 0) return;
      const int right = syms[static_cast<size_t>(left)].next;
      if (right < 0) return;
      const TokenId l = syms[static_cast<size_t>(left)].id, r = syms[static_cast<size_t>(right)].id;
      if (l == kNoToken || r == kNoToken) return;
      auto it = merges_.find(key(l, r));
      if (it != merges_.end()) heap.push({it->second.rank, left, l, r});
    };
    for (int i = 0; i + 1 < static_cast<int>(syms.size()); ++i) push(i);

    while (!heap.empty()) {
      const Cand c = heap.top();
      heap.pop();
      Sym& left = syms[static_cast<size_t>(c.left)];
      if (left.id != c.l || left.next < 0) continue;  // stale
      Sym& right = syms[static_cast<size_t>(left.next)];
      if (right.id != c.r) continue;  // stale
      left.id = merges_.at(key(c.l, c.r)).merged;
      right.id = kNoToken;  // tombstone
      left.next = right.next;
      if (left.next >= 0) syms[static_cast<size_t>(left.next)].prev = c.left;
      push(left.prev);
      push(c.left);
    }
    for (int i = 0; i >= 0; i = syms[static_cast<size_t>(i)].next) {
      if (syms[static_cast<size_t>(i)].id != kNoToken) out.push_back(syms[static_cast<size_t>(i)].id);
    }
  }

  PreTokenizer pre_ = PreTokenizer::kGpt2;
  TokenId byte_token_[256];
  std::unordered_map<uint64_t, Merge> merges_;
};

// ===========================================================================
// SentencePiece BPE

class SpmTokenizer final : public Tokenizer {
 public:
  explicit SpmTokenizer(TokenizerData data) : Tokenizer(std::move(data)) {}

  Status init() {
    ENGINE_RETURN_IF_ERROR(init_common());
    if (data_.scores.empty()) return Corrupt("tokenizer: SPM model without scores");
    for (int b = 0; b < 256; ++b) {
      char name[8];
      std::snprintf(name, sizeof(name), "<0x%02X>", b);
      byte_token_[b] = find_token(name);
    }
    return Status::Ok();
  }

 protected:
  void encode_fragment(std::string_view text, bool at_start, std::vector<TokenId>& out) const override {
    // Normalize: optional leading space, then ' ' -> U+2581.
    thread_local std::string norm;
    norm.clear();
    if (data_.add_space_prefix && at_start) norm += "\xE2\x96\x81";
    for (char c : text) {
      if (c == ' ') {
        norm += "\xE2\x96\x81";
      } else {
        norm += c;
      }
    }

    struct Sym {
      size_t begin, len;
      int prev, next;
    };
    thread_local std::vector<Sym> syms;
    syms.clear();
    for (size_t i = 0; i < norm.size();) {
      const size_t start = i;
      unicode::decode_utf8(norm, i);
      syms.push_back({start, i - start, static_cast<int>(syms.size()) - 1,
                      static_cast<int>(syms.size()) + 1});
    }
    if (syms.empty()) return;
    syms.back().next = -1;

    struct Cand {
      float score;
      int left;
      size_t len;  // merged byte length at push time (stale check)
      bool operator<(const Cand& o) const { return score != o.score ? score < o.score : left > o.left; }
    };
    std::priority_queue<Cand> heap;  // highest score first, then leftmost
    auto push = [&](int left) {
      if (left < 0) return;
      const Sym& l = syms[static_cast<size_t>(left)];
      if (l.next < 0) return;
      const Sym& r = syms[static_cast<size_t>(l.next)];
      const std::string_view piece(norm.data() + l.begin, l.len + r.len);
      const TokenId id = find_token(piece);
      if (id != kNoToken) heap.push({data_.scores[static_cast<size_t>(id)], left, piece.size()});
    };
    for (int i = 0; i + 1 < static_cast<int>(syms.size()); ++i) push(i);

    while (!heap.empty()) {
      const Cand c = heap.top();
      heap.pop();
      Sym& l = syms[static_cast<size_t>(c.left)];
      if (l.len == 0 || l.next < 0) continue;
      Sym& r = syms[static_cast<size_t>(l.next)];
      if (l.len + r.len != c.len) continue;  // stale
      l.len += r.len;
      r.len = 0;
      l.next = r.next;
      if (l.next >= 0) syms[static_cast<size_t>(l.next)].prev = c.left;
      push(l.prev);
      push(c.left);
    }

    for (int i = 0; i >= 0; i = syms[static_cast<size_t>(i)].next) {
      const Sym& s = syms[static_cast<size_t>(i)];
      const std::string_view piece(norm.data() + s.begin, s.len);
      const TokenId id = find_token(piece);
      if (id != kNoToken) {
        out.push_back(id);
        continue;
      }
      // Byte fallback, else UNK.
      for (unsigned char b : piece) {
        const TokenId bt = byte_token_[b];
        out.push_back(bt != kNoToken ? bt : data_.unk);
      }
    }
  }

  void decode_normal(TokenId id, std::string& out) const override {
    const std::string& t = data_.tokens[static_cast<size_t>(id)];
    if (token_type(id) == TokenType::kByte && t.size() == 6 && t.starts_with("<0x")) {
      out += static_cast<char>(std::stoi(t.substr(3, 2), nullptr, 16));
      return;
    }
    for (size_t i = 0; i < t.size(); ++i) {
      if (t.compare(i, 3, "\xE2\x96\x81") == 0) {
        out += ' ';
        i += 2;
      } else {
        out += t[i];
      }
    }
  }

 private:
  TokenId byte_token_[256];
};

}  // namespace

Result<std::unique_ptr<Tokenizer>> Tokenizer::create(TokenizerData data) {
  if (data.model == TokenizerData::Model::kBpe) {
    auto t = std::make_unique<BpeTokenizer>(std::move(data));
    ENGINE_RETURN_IF_ERROR(t->init());
    return std::unique_ptr<Tokenizer>(std::move(t));
  }
  auto t = std::make_unique<SpmTokenizer>(std::move(data));
  ENGINE_RETURN_IF_ERROR(t->init());
  return std::unique_ptr<Tokenizer>(std::move(t));
}

// ===========================================================================

void Utf8Buffer::push(std::string_view bytes, std::string& out) {
  pending_.append(bytes);
  // Find the end of the last complete sequence.
  size_t complete = 0;
  for (size_t i = 0; i < pending_.size();) {
    const int len = unicode::utf8_sequence_length(static_cast<unsigned char>(pending_[i]));
    if (len == 0) {  // invalid lead byte: pass through
      complete = ++i;
      continue;
    }
    if (i + static_cast<size_t>(len) > pending_.size()) break;
    i += static_cast<size_t>(len);
    complete = i;
  }
  out.append(pending_, 0, complete);
  pending_.erase(0, complete);
}

void Utf8Buffer::flush(std::string& out) {
  out += pending_;
  pending_.clear();
}

}  // namespace engine

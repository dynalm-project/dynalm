#pragma once

// Structured output: constrains generation to JSON, optionally matching a
// JSON Schema (OpenAI `response_format`: json_object, json_schema).
//
// A JsonGrammar is compiled once per request from the schema. A
// JsonConstraint tracks one sequence's position in it with a small
// character-level pushdown matcher, and decides which vocabulary tokens may
// come next: a token is allowed when all of its bytes keep the output a valid
// prefix of a matching document; end-of-generation tokens only once the
// document is complete.
//
// Sampling (scheduler): the token chosen without the constraint is checked
// first; only when it is not allowed is the whole vocabulary masked (every
// disallowed logit set to -inf) and the token chosen again. Masking walks the
// vocabulary in byte order and reuses the matcher state of the shared prefix,
// so tokens with a common rejected prefix are skipped together.
//
// Supported schema subset (anything else is refused with a clear error, never
// silently ignored): type (object, array, string, number, integer, boolean,
// null, or a list of these), properties, required, additionalProperties
// (false, or absent with properties), items, enum, const, anyOf/oneOf whose
// alternatives start with different characters (e.g. string | null).
// Annotations (title, description, $schema, default, examples) are ignored.
// Object keys may come in any order; each at most once. The document starts
// at its first byte (no leading whitespace), and at most 32 whitespace bytes
// may follow each other, so a model cannot spend its tokens on whitespace.

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "dynacore/base/status.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

namespace json {
class Value;
}

class JsonGrammar {
 public:
  enum class Kind : uint8_t { kAny, kObject, kArray, kString, kNumber, kInteger, kLiteral, kUnion };
  struct Node {
    Kind kind = Kind::kAny;
    std::vector<std::string> keys;     // kObject: property names, JSON-escaped (no quotes)
    std::vector<uint32_t> values;      // kObject: node per property; kUnion: alternatives
    uint64_t required = 0;             // kObject: bit per property
    bool free_keys = false;            // kObject without properties: any keys, any values
    uint32_t items = 0;                // kArray: node of every element
    std::vector<std::string> literals;  // kLiteral: serialized JSON values (enum, const, true/false, null)
    std::array<uint64_t, 4> first{};   // bytes a value of this node can start with
  };

  // `schema` nullptr: any JSON object (response_format json_object).
  static Result<std::shared_ptr<const JsonGrammar>> compile(const json::Value* schema);

  const Node& node(uint32_t i) const { return nodes_[i]; }
  uint32_t root() const { return root_; }
  uint32_t any() const { return 0; }
  uint32_t free_object() const { return 1; }
  uint32_t any_array() const { return 2; }
  uint32_t number() const { return 3; }
  uint32_t boolean() const { return 4; }
  uint32_t null() const { return 5; }

 private:
  Result<uint32_t> add(const json::Value& schema, int depth);
  uint32_t push(Node n);
  std::vector<Node> nodes_;
  uint32_t root_ = 0;
};

// Byte strings of a tokenizer's vocabulary in sorted order, with the common
// prefix length of each with its predecessor. Built once per tokenizer.
class VocabBytes {
 public:
  static std::shared_ptr<const VocabBytes> build(const Tokenizer& tok);
  // From the bytes of each token id (tests); empty = a control token.
  static std::shared_ptr<const VocabBytes> from_bytes(std::vector<std::string> bytes);
  std::span<const TokenId> order() const { return order_; }
  const std::string& bytes(TokenId id) const { return bytes_[static_cast<size_t>(id)]; }
  uint16_t lcp(size_t k) const { return lcp_[k]; }  // with order_[k - 1]
  size_t max_len() const { return max_len_; }
  int32_t size() const { return static_cast<int32_t>(bytes_.size()); }

 private:
  std::vector<std::string> bytes_;  // per token id; empty for control tokens
  std::vector<TokenId> order_;      // ids with non-empty bytes, sorted by bytes
  std::vector<uint16_t> lcp_;
  size_t max_len_ = 0;
};

class JsonConstraint {
 public:
  static constexpr int kMaxDepth = 32;
  static constexpr int kMaxWhitespace = 32;  // consecutive whitespace between JSON tokens; none before the document

  struct Frame {
    uint8_t kind = 0;
    uint8_t phase = 0;
    uint16_t idx = 0;
    uint32_t node = 0;
    uint64_t mask = 0;
  };
  struct State {
    std::array<Frame, kMaxDepth> f;
    uint8_t depth = 0;
    uint8_t ws = 0;
    bool done = false;
  };

  JsonConstraint(std::shared_ptr<const JsonGrammar> grammar, std::shared_ptr<const VocabBytes> vocab,
                 std::unordered_set<TokenId> eog);

  // Whether `id` may come next.
  bool allows(TokenId id) const;
  // Sets the logit of every token that may not come next to -inf. False when
  // no token may come next (then `logits` is unspecified).
  bool mask(std::span<float> logits) const;
  // Advances past `id`, which must be allowed.
  void accept(TokenId id);
  // A complete document has been generated.
  bool complete() const { return can_finish(state_); }

  // The matcher on raw bytes (tests and the token checks above).
  bool feed(State& s, uint8_t c) const;
  bool can_finish(const State& s) const;
  const State& state() const { return state_; }

 private:
  static void copy(State& dst, const State& src);
  bool feed_bytes(State& s, const std::string& b) const;
  bool push_value(State& s, uint32_t node) const;
  bool start_value(State& s, uint8_t c) const;
  void value_done(State& s) const;

  std::shared_ptr<const JsonGrammar> g_;
  std::shared_ptr<const VocabBytes> vocab_;
  std::unordered_set<TokenId> eog_;
  State state_;
  mutable std::vector<State> scratch_;
};

}  // namespace dynalm

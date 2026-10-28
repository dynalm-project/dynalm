#include "sampling/json_constraint.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <string_view>

#include "api/json.h"

namespace dynalm {
namespace {

using Kind = JsonGrammar::Kind;
using Node = JsonGrammar::Node;

// Matcher frame kinds. A kValue frame is replaced in place by the concrete
// frame once the value's first byte is seen.
enum FrameKind : uint8_t { kValue, kObject, kKey, kArray, kString, kNumber, kLiteral };

constexpr int kMaxSchemaDepth = 32;
constexpr size_t kMaxKeys = 64;      // properties per object; enum values (bit masks)
constexpr size_t kMaxKeyBytes = 250;  // a key or literal must fit the uint16 position
constexpr int kMaxNumberChars = 40;

bool is_ws(uint8_t c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
bool has(const std::array<uint64_t, 4>& s, uint8_t c) { return (s[c >> 6] >> (c & 63)) & 1; }
void add_byte(std::array<uint64_t, 4>& s, uint8_t c) { s[c >> 6] |= uint64_t{1} << (c & 63); }
void add_digits(std::array<uint64_t, 4>& s) {
  add_byte(s, '-');
  for (uint8_t c = '0'; c <= '9'; ++c) add_byte(s, c);
}
uint64_t all_bits(size_t n) { return n >= 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1; }
bool number_accepting(uint8_t phase) { return phase == 2 || phase == 3 || phase == 5 || phase == 8; }

// The JSON text of a key or string literal without its quotes.
std::string escaped(const std::string& s) {
  std::string d = json::dump(json::Value(s));
  return d.substr(1, d.size() - 2);
}

}  // namespace

// ---------------------------------------------------------------------------
// JsonGrammar

uint32_t JsonGrammar::push(Node n) {
  switch (n.kind) {
    case Kind::kAny:
      for (uint8_t c : std::string_view("{[\"tfn")) add_byte(n.first, c);
      add_digits(n.first);
      break;
    case Kind::kObject: add_byte(n.first, '{'); break;
    case Kind::kArray: add_byte(n.first, '['); break;
    case Kind::kString: add_byte(n.first, '"'); break;
    case Kind::kNumber:
    case Kind::kInteger: add_digits(n.first); break;
    case Kind::kLiteral:
      for (const std::string& l : n.literals) add_byte(n.first, static_cast<uint8_t>(l[0]));
      break;
    case Kind::kUnion:
      for (uint32_t a : n.values) {
        for (size_t w = 0; w < 4; ++w) n.first[w] |= nodes_[a].first[w];
      }
      break;
  }
  nodes_.push_back(std::move(n));
  return static_cast<uint32_t>(nodes_.size() - 1);
}

Result<std::shared_ptr<const JsonGrammar>> JsonGrammar::compile(const json::Value* schema) {
  auto g = std::make_shared<JsonGrammar>();
  // Built-in nodes, at the indices the accessors name.
  g->push(Node{.kind = Kind::kAny});
  g->push(Node{.kind = Kind::kObject, .free_keys = true});
  g->push(Node{.kind = Kind::kArray, .items = 0});
  g->push(Node{.kind = Kind::kNumber});
  g->push(Node{.kind = Kind::kLiteral, .literals = {"true", "false"}});
  g->push(Node{.kind = Kind::kLiteral, .literals = {"null"}});
  g->push(Node{.kind = Kind::kString});
  if (schema == nullptr) {
    g->root_ = g->free_object();
  } else {
    auto r = g->add(*schema, 0);
    if (!r.ok()) return r.status();
    g->root_ = *r;
  }
  return std::shared_ptr<const JsonGrammar>(std::move(g));
}

Result<uint32_t> JsonGrammar::add(const json::Value& s, int depth) {
  if (depth > kMaxSchemaDepth) return InvalidArgument("JSON schema: nested deeper than 32 levels");
  if (s.is_bool()) {
    if (s.as_bool()) return any();
    return InvalidArgument("JSON schema: a 'false' schema matches nothing");
  }
  if (!s.is_object()) return InvalidArgument("JSON schema: a schema must be an object");
  static constexpr std::string_view kKnown[] = {
      "type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf", "oneOf",
      // Annotations: no effect on what is generated.
      "title", "description", "$schema", "$id", "$comment", "default", "examples"};
  for (const auto& [k, v] : s.as_object()) {
    (void)v;
    if (std::find(std::begin(kKnown), std::end(kKnown), k) == std::end(kKnown)) {
      return InvalidArgument("JSON schema: keyword '" + k + "' is not supported");
    }
  }
  auto literal_node = [&](std::vector<std::string> lits) -> Result<uint32_t> {
    if (lits.empty()) return InvalidArgument("JSON schema: 'enum' must not be empty");
    if (lits.size() > kMaxKeys) return InvalidArgument("JSON schema: 'enum' has more than 64 values");
    for (const std::string& l : lits) {
      if (l.size() > kMaxKeyBytes) return InvalidArgument("JSON schema: an 'enum' value is longer than 250 bytes");
    }
    return push(Node{.kind = Kind::kLiteral, .literals = std::move(lits)});
  };
  if (const json::Value* c = s.find("const")) return literal_node({json::dump(*c)});
  if (const json::Value* e = s.find("enum")) {
    if (!e->is_array()) return InvalidArgument("JSON schema: 'enum' must be an array");
    std::vector<std::string> lits;
    for (const json::Value& v : e->as_array()) lits.push_back(json::dump(v));
    std::sort(lits.begin(), lits.end());
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    return literal_node(std::move(lits));
  }
  auto union_node = [&](std::vector<uint32_t> alts) -> Result<uint32_t> {
    if (alts.size() == 1) return alts[0];
    // Alternatives must be told apart by their first byte (string | null,
    // object | array, ...): the matcher never backtracks.
    std::array<uint64_t, 4> seen{};
    for (uint32_t a : alts) {
      for (size_t w = 0; w < 4; ++w) {
        if ((seen[w] & nodes_[a].first[w]) != 0) {
          return InvalidArgument(
              "JSON schema: anyOf/oneOf and type lists are supported only when the alternatives start "
              "differently (e.g. string or null, object or array)");
        }
        seen[w] |= nodes_[a].first[w];
      }
    }
    return push(Node{.kind = Kind::kUnion, .values = std::move(alts)});
  };
  for (std::string_view key : {"anyOf", "oneOf"}) {
    if (const json::Value* a = s.find(key)) {
      if (!a->is_array() || a->as_array().empty()) {
        return InvalidArgument("JSON schema: '" + std::string(key) + "' must be a non-empty array");
      }
      std::vector<uint32_t> alts;
      for (const json::Value& sub : a->as_array()) {
        auto r = add(sub, depth + 1);
        if (!r.ok()) return r.status();
        alts.push_back(*r);
      }
      return union_node(std::move(alts));
    }
  }

  auto typed = [&](std::string_view type) -> Result<uint32_t> {
    if (type == "string") return push(Node{.kind = Kind::kString});
    if (type == "number") return push(Node{.kind = Kind::kNumber});
    if (type == "integer") return push(Node{.kind = Kind::kInteger});
    if (type == "boolean") return boolean();
    if (type == "null") return null();
    if (type == "array") {
      uint32_t items = any();
      if (const json::Value* it = s.find("items")) {
        auto r = add(*it, depth + 1);
        if (!r.ok()) return r.status();
        items = *r;
      }
      return push(Node{.kind = Kind::kArray, .items = items});
    }
    if (type == "object") {
      const json::Value* props = s.find("properties");
      const json::Value* extra = s.find("additionalProperties");
      if (props != nullptr && !props->is_object()) return InvalidArgument("JSON schema: 'properties' must be an object");
      if (props == nullptr || props->as_object().empty()) {
        if (extra != nullptr && !extra->is_bool()) {
          return InvalidArgument("JSON schema: 'additionalProperties' with a schema is not supported");
        }
        if (s.find("required") != nullptr) return InvalidArgument("JSON schema: 'required' needs 'properties'");
        if (extra != nullptr && !extra->as_bool()) return push(Node{.kind = Kind::kObject});  // only {}
        return free_object();
      }
      if (extra != nullptr && !(extra->is_bool() && !extra->as_bool())) {
        return InvalidArgument(
            "JSON schema: with 'properties', only 'additionalProperties': false is supported (other keys are never "
            "generated)");
      }
      if (props->as_object().size() > kMaxKeys) return InvalidArgument("JSON schema: more than 64 properties");
      Node n{.kind = Kind::kObject};
      for (const auto& [name, sub] : props->as_object()) {
        std::string k = escaped(name);
        if (k.empty() || k.size() > kMaxKeyBytes) {
          return InvalidArgument("JSON schema: property names must be 1 to 250 bytes");
        }
        auto r = add(sub, depth + 1);
        if (!r.ok()) return r.status();
        n.keys.push_back(std::move(k));
        n.values.push_back(*r);
      }
      if (const json::Value* req = s.find("required")) {
        if (!req->is_array()) return InvalidArgument("JSON schema: 'required' must be an array");
        for (const json::Value& r : req->as_array()) {
          if (!r.is_string()) return InvalidArgument("JSON schema: 'required' must list property names");
          const auto it = std::find(n.keys.begin(), n.keys.end(), escaped(r.as_string()));
          if (it == n.keys.end()) {
            return InvalidArgument("JSON schema: required property '" + r.as_string() + "' is not in 'properties'");
          }
          n.required |= uint64_t{1} << (it - n.keys.begin());
        }
      }
      return push(std::move(n));
    }
    return InvalidArgument("JSON schema: unknown type '" + std::string(type) + "'");
  };

  const json::Value* type = s.find("type");
  if (type == nullptr) {
    if (s.find("properties") != nullptr) return typed("object");
    if (s.find("items") != nullptr) return typed("array");
    return any();
  }
  if (type->is_string()) return typed(type->as_string());
  if (!type->is_array() || type->as_array().empty()) {
    return InvalidArgument("JSON schema: 'type' must be a string or a non-empty array");
  }
  std::vector<uint32_t> alts;
  for (const json::Value& t : type->as_array()) {
    if (!t.is_string()) return InvalidArgument("JSON schema: 'type' lists must contain strings");
    auto r = typed(t.as_string());
    if (!r.ok()) return r.status();
    alts.push_back(*r);
  }
  return union_node(std::move(alts));
}

// ---------------------------------------------------------------------------
// VocabBytes

std::shared_ptr<const VocabBytes> VocabBytes::build(const Tokenizer& tok) {
  std::vector<std::string> bytes(static_cast<size_t>(tok.vocab_size()));
  for (TokenId id = 0; id < tok.vocab_size(); ++id) tok.decode_token(id, bytes[static_cast<size_t>(id)], false);
  return from_bytes(std::move(bytes));
}

std::shared_ptr<const VocabBytes> VocabBytes::from_bytes(std::vector<std::string> bytes) {
  auto v = std::make_shared<VocabBytes>();
  v->bytes_ = std::move(bytes);
  for (size_t id = 0; id < v->bytes_.size(); ++id) {
    if (!v->bytes_[id].empty()) v->order_.push_back(static_cast<TokenId>(id));
  }
  std::sort(v->order_.begin(), v->order_.end(),
            [&](TokenId a, TokenId b) { return v->bytes(a) < v->bytes(b); });
  v->lcp_.assign(v->order_.size(), 0);
  for (size_t k = 0; k < v->order_.size(); ++k) {
    const std::string& cur = v->bytes(v->order_[k]);
    v->max_len_ = std::max(v->max_len_, cur.size());
    if (k == 0) continue;
    const std::string& prev = v->bytes(v->order_[k - 1]);
    size_t l = 0;
    while (l < cur.size() && l < prev.size() && cur[l] == prev[l] && l < 0xffff) ++l;
    v->lcp_[k] = static_cast<uint16_t>(l);
  }
  return v;
}

// ---------------------------------------------------------------------------
// JsonConstraint

JsonConstraint::JsonConstraint(std::shared_ptr<const JsonGrammar> grammar, std::shared_ptr<const VocabBytes> vocab,
                               std::unordered_set<TokenId> eog)
    : g_(std::move(grammar)), vocab_(std::move(vocab)), eog_(std::move(eog)) {
  push_value(state_, g_->root());
  scratch_.resize(vocab_->max_len() + 1);
}

void JsonConstraint::copy(State& dst, const State& src) {
  dst.depth = src.depth;
  dst.ws = src.ws;
  dst.done = src.done;
  std::copy_n(src.f.begin(), src.depth, dst.f.begin());
}

bool JsonConstraint::push_value(State& s, uint32_t node) const {
  if (s.depth >= kMaxDepth) return false;
  s.f[s.depth++] = Frame{kValue, 0, 0, node, 0};
  return true;
}

// Replaces the top kValue frame by the frame for the value starting with `c`.
bool JsonConstraint::start_value(State& s, uint8_t c) const {
  uint32_t id = s.f[s.depth - 1].node;
  for (;;) {
    const Node& n = g_->node(id);
    if (n.kind == Kind::kUnion) {
      const auto it = std::find_if(n.values.begin(), n.values.end(),
                                   [&](uint32_t a) { return has(g_->node(a).first, c); });
      if (it == n.values.end()) return false;
      id = *it;
      continue;
    }
    if (n.kind == Kind::kAny) {
      if (c == '{') id = g_->free_object();
      else if (c == '[') id = g_->any_array();
      else if (c == '"') id = 6;  // the built-in string node
      else if (c == '-' || (c >= '0' && c <= '9')) id = g_->number();
      else if (c == 't' || c == 'f') id = g_->boolean();
      else if (c == 'n') id = g_->null();
      else return false;
      continue;
    }
    break;
  }
  const Node& n = g_->node(id);
  if (!has(n.first, c)) return false;
  Frame f{0, 0, 0, id, 0};
  switch (n.kind) {
    case Kind::kObject: f.kind = kObject; break;
    case Kind::kArray: f.kind = kArray; break;
    case Kind::kString: f.kind = kString; break;
    case Kind::kNumber:
    case Kind::kInteger: f.kind = kNumber; break;
    case Kind::kLiteral:
      f.kind = kLiteral;
      f.mask = all_bits(n.literals.size());
      break;
    default: return false;
  }
  s.f[s.depth - 1] = f;
  return true;
}

// The top frame's value is complete and has been popped.
void JsonConstraint::value_done(State& s) const {
  if (s.depth == 0) {
    s.done = true;
    return;
  }
  Frame& p = s.f[s.depth - 1];
  if (p.kind == kObject) {
    p.phase = p.phase == 3 ? 5 : 2;  // a member value, or a free-form key
  } else if (p.kind == kArray) {
    p.phase = 5;
    ++p.mask;
  }
}

bool JsonConstraint::feed(State& s, uint8_t c) const {
  for (;;) {
    if (s.depth == 0) return false;  // the document is complete
    Frame& t = s.f[s.depth - 1];
    const Node& n = g_->node(t.node);
    switch (t.kind) {
      case kValue:
        // No whitespace before the document: a model stuck on whitespace
        // would otherwise spend its tokens there.
        if (is_ws(c)) return s.depth > 1 && ++s.ws <= kMaxWhitespace;
        s.ws = 0;
        if (!start_value(s, c)) return false;
        continue;  // the concrete frame takes `c`

      case kObject:
        if (t.phase == 0) {
          if (c != '{') return false;
          t.phase = 1;
          return true;
        }
        if (is_ws(c)) return ++s.ws <= kMaxWhitespace;
        s.ws = 0;
        switch (t.phase) {
          case 1:
          case 4:
            if (c == '}' && t.phase == 1) {
              if ((n.required & ~t.mask) != 0) return false;
              --s.depth;
              value_done(s);
              return true;
            }
            if (c != '"' || s.depth >= kMaxDepth) return false;
            if (n.free_keys) {
              s.f[s.depth++] = Frame{kString, 1, 0, 6, 0};
              return true;
            }
            if ((all_bits(n.keys.size()) & ~t.mask) == 0) return false;
            s.f[s.depth++] = Frame{kKey, 0, 0, t.node, all_bits(n.keys.size()) & ~t.mask};
            return true;
          case 2:
            if (c != ':') return false;
            t.phase = 3;
            return push_value(s, n.free_keys ? g_->any() : n.values[t.idx]);
          case 5:
            if (c == ',') {
              if (!n.free_keys && (all_bits(n.keys.size()) & ~t.mask) == 0) return false;  // no key left
              t.phase = 4;
              return true;
            }
            if (c != '}' || (n.required & ~t.mask) != 0) return false;
            --s.depth;
            value_done(s);
            return true;
          default: return false;
        }

      case kKey: {
        uint64_t m = 0;
        for (uint64_t b = t.mask; b != 0; b &= b - 1) {
          const size_t i = static_cast<size_t>(std::countr_zero(b));
          if (t.idx < n.keys[i].size() && static_cast<uint8_t>(n.keys[i][t.idx]) == c) m |= uint64_t{1} << i;
        }
        if (m != 0) {
          t.mask = m;
          ++t.idx;
          return true;
        }
        if (c != '"') return false;
        // Escaped keys never end in an unpaired backslash, so a quote that
        // continues a key and a quote that ends one cannot both match.
        for (uint64_t b = t.mask; b != 0; b &= b - 1) {
          const size_t i = static_cast<size_t>(std::countr_zero(b));
          if (n.keys[i].size() == t.idx) {
            --s.depth;
            Frame& o = s.f[s.depth - 1];
            o.idx = static_cast<uint16_t>(i);
            o.mask |= uint64_t{1} << i;
            o.phase = 2;
            return true;
          }
        }
        return false;
      }

      case kArray:
        if (t.phase == 0) {
          if (c != '[') return false;
          t.phase = 1;
          return true;
        }
        if (is_ws(c)) return ++s.ws <= kMaxWhitespace;
        s.ws = 0;
        switch (t.phase) {
          case 1:
            if (c == ']') {
              --s.depth;
              value_done(s);
              return true;
            }
            [[fallthrough]];
          case 4:
            t.phase = 3;
            if (!push_value(s, n.items)) return false;
            continue;
          case 5:
            if (c == ',') {
              t.phase = 4;
              return true;
            }
            if (c != ']') return false;
            --s.depth;
            value_done(s);
            return true;
          default: return false;
        }

      case kString:
        switch (t.phase) {
          case 0:
            if (c != '"') return false;
            t.phase = 1;
            return true;
          case 1:
            if (c == '"') {
              --s.depth;
              value_done(s);
              return true;
            }
            if (c == '\\') t.phase = 2;
            return c >= 0x20;
          case 2:
            if (c == 'u') {
              t.phase = 3;
              return true;
            }
            if (c == 0 || std::strchr("\"\\/bfnrt", c) == nullptr) return false;
            t.phase = 1;
            return true;
          default:  // 3..6: the four hex digits of \uXXXX
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
            t.phase = t.phase == 6 ? 1 : static_cast<uint8_t>(t.phase + 1);
            return true;
        }

      case kNumber: {
        // RFC 8259 number: -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?
        const bool integer = n.kind == Kind::kInteger;
        const bool digit = c >= '0' && c <= '9';
        const bool dot = !integer && c == '.';
        const bool exp = !integer && (c == 'e' || c == 'E');
        int next = -1;
        switch (t.phase) {
          case 0: next = c == '-' ? 1 : c == '0' ? 2 : digit ? 3 : -1; break;
          case 1: next = c == '0' ? 2 : digit ? 3 : -1; break;
          case 2: next = dot ? 4 : exp ? 6 : -1; break;
          case 3: next = digit ? 3 : dot ? 4 : exp ? 6 : -1; break;
          case 4: next = digit ? 5 : -1; break;
          case 5: next = digit ? 5 : exp ? 6 : -1; break;
          case 6: next = c == '+' || c == '-' ? 7 : digit ? 8 : -1; break;
          case 7:
          case 8: next = digit ? 8 : -1; break;
          default: break;
        }
        if (next >= 0) {
          if (++t.idx > kMaxNumberChars) return false;
          t.phase = static_cast<uint8_t>(next);
          return true;
        }
        if (!number_accepting(t.phase)) return false;
        --s.depth;  // the number ended; `c` belongs to the parent
        value_done(s);
        continue;
      }

      case kLiteral: {
        uint64_t m = 0;
        bool ended = false;  // a literal is complete before `c`
        for (uint64_t b = t.mask; b != 0; b &= b - 1) {
          const size_t i = static_cast<size_t>(std::countr_zero(b));
          const std::string& l = n.literals[i];
          if (t.idx < l.size() && static_cast<uint8_t>(l[t.idx]) == c) m |= uint64_t{1} << i;
          if (l.size() == t.idx) ended = true;
        }
        if (m != 0) {
          t.mask = m;
          ++t.idx;
          bool full = false, longer = false;
          for (uint64_t b = m; b != 0; b &= b - 1) {
            const size_t len = n.literals[static_cast<size_t>(std::countr_zero(b))].size();
            full |= len == t.idx;
            longer |= len > t.idx;
          }
          if (full && !longer) {
            --s.depth;
            value_done(s);
          }
          return true;
        }
        if (!ended) return false;
        --s.depth;  // e.g. enum [1, 12] after "1": `c` belongs to the parent
        value_done(s);
        continue;
      }

      default: return false;
    }
  }
}

bool JsonConstraint::can_finish(const State& s) const {
  if (s.done) return true;
  if (s.depth != 1) return false;
  const Frame& t = s.f[0];
  if (t.kind == kNumber) return number_accepting(t.phase);
  if (t.kind == kLiteral) {
    const Node& n = g_->node(t.node);
    for (uint64_t b = t.mask; b != 0; b &= b - 1) {
      if (n.literals[static_cast<size_t>(std::countr_zero(b))].size() == t.idx) return true;
    }
  }
  return false;
}

bool JsonConstraint::feed_bytes(State& s, const std::string& b) const {
  for (char ch : b) {
    if (!feed(s, static_cast<uint8_t>(ch))) return false;
  }
  return true;
}

bool JsonConstraint::allows(TokenId id) const {
  if (id < 0 || id >= vocab_->size()) return false;
  if (eog_.contains(id)) return can_finish(state_);
  const std::string& b = vocab_->bytes(id);
  if (b.empty()) return false;
  State& s = scratch_[0];
  copy(s, state_);
  return feed_bytes(s, b);
}

bool JsonConstraint::mask(std::span<float> logits) const {
  constexpr float kOff = -std::numeric_limits<float>::infinity();
  const int32_t n = std::min<int32_t>(vocab_->size(), static_cast<int32_t>(logits.size()));
  for (int32_t id = n; id < static_cast<int32_t>(logits.size()); ++id) logits[static_cast<size_t>(id)] = kOff;
  bool any = false;
  // scratch_[d]: the state after the first d bytes of the current token.
  // Sorted order: a token shares its first lcp bytes with the previous one,
  // so those states are reused; when the previous token failed within the
  // shared prefix, this one fails too.
  copy(scratch_[0], state_);
  size_t valid = 0;                                        // scratch_[0..valid] are current
  size_t dead = std::numeric_limits<size_t>::max();        // a prefix of this length is rejected
  const auto order = vocab_->order();
  for (size_t k = 0; k < order.size(); ++k) {
    const TokenId id = order[k];
    const std::string& b = vocab_->bytes(id);
    const size_t l = k == 0 ? 0 : vocab_->lcp(k);
    bool ok = true;
    if (dead <= l) {
      ok = false;
    } else {
      dead = std::numeric_limits<size_t>::max();
      for (size_t d = std::min(valid, l); d < b.size(); ++d) {
        copy(scratch_[d + 1], scratch_[d]);
        if (!feed(scratch_[d + 1], static_cast<uint8_t>(b[d]))) {
          dead = d + 1;
          valid = d;
          ok = false;
          break;
        }
        valid = d + 1;
      }
    }
    if (id >= n) continue;
    if (ok && !eog_.contains(id)) {
      any = true;
    } else if (!eog_.contains(id)) {
      logits[static_cast<size_t>(id)] = kOff;
    }
  }
  // Tokens without bytes (control tokens) never fit, except end-of-generation
  // once the document is complete.
  const bool finish = can_finish(state_);
  for (int32_t id = 0; id < n; ++id) {
    if (eog_.contains(id)) {
      if (finish) any = true;
      else logits[static_cast<size_t>(id)] = kOff;
    } else if (vocab_->bytes(id).empty()) {
      logits[static_cast<size_t>(id)] = kOff;
    }
  }
  return any;
}

void JsonConstraint::accept(TokenId id) {
  if (eog_.contains(id)) return;
  (void)feed_bytes(state_, vocab_->bytes(id));
}

}  // namespace dynalm

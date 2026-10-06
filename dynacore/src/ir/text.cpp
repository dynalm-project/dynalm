#include "dynacore/ir/text.h"

#include <cctype>
#include <charconv>
#include <cstdio>
#include <unordered_map>

namespace dynacore::ir {

bool op_writes_in_place(OpKind k) {
  switch (k) {
    case OpKind::kRope:
    case OpKind::kScale:
    case OpKind::kSoftcap:
    case OpKind::kFill:
    case OpKind::kKvWrite:
    case OpKind::kScatterAdd:
      return true;
    default:
      return false;
  }
}

namespace {

std::string attr_to_string(const Attr& a) {
  if (const auto* i = std::get_if<int64_t>(&a)) return std::to_string(*i);
  if (const auto* d = std::get_if<double>(&a)) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", *d);
    std::string s = buf;
    // Keep floats recognizably floating so they parse back as doubles.
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
  }
  const std::string& s = std::get<std::string>(a);
  bool ident = !s.empty() && (std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_');
  for (char c : s) ident = ident && (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.');
  return ident ? s : "\"" + s + "\"";
}

}  // namespace

std::string print_graph(const Graph& g, const PrintOptions& opts) {
  std::string out = "graph @" + g.name() + " {\n";
  for (const Op& op : g.ops()) {
    const Value& r = g.value(op.result);
    std::string line = "  %" + std::to_string(r.id) + " = " + std::string(op_name(op.kind));
    const bool leaf = op.kind == OpKind::kInput || op.kind == OpKind::kWeight || op.kind == OpKind::kKvCache ||
                      op.kind == OpKind::kConstant;
    if (leaf) line += " \"" + r.name + "\"";
    if (op.kind == OpKind::kConstant) {
      if (const ConstData* c = g.const_data(r.id)) {
        line += " [";
        for (size_t i = 0; i < c->i32.size(); ++i) line += (i ? "," : "") + std::to_string(c->i32[i]);
        for (size_t i = 0; i < c->f32.size(); ++i) line += (i ? "," : "") + attr_to_string(Attr(double{c->f32[i]}));
        line += "]";
      }
    }
    for (size_t i = 0; i < op.inputs.size(); ++i) line += (i ? ", %" : " %") + std::to_string(op.inputs[i]);
    Attrs attrs = op.attrs;
    // Aliasing that the op kind does not imply.
    if (!leaf && !op_writes_in_place(op.kind)) {
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        if (g.value(op.inputs[i]).buffer == r.buffer) attrs["alias"] = static_cast<int64_t>(i);
      }
    }
    if (!attrs.empty()) {
      line += " {";
      bool first = true;
      for (const auto& [k, v] : attrs) {
        line += (first ? "" : ", ") + k + "=" + attr_to_string(v);
        first = false;
      }
      line += "}";
    }
    line += " : " + type_to_string(r.type);
    std::string note;
    if (opts.annotations && !op.kernel.empty()) note += " kernel=" + op.kernel;
    if (opts.annotations && op.fusion_group >= 0) note += " fusion=" + std::to_string(op.fusion_group);
    if (opts.names && !leaf && !r.name.empty()) note += " " + r.name;
    if (!note.empty()) line += "  //" + note;
    out += line + "\n";
  }
  out += "}\n";
  return out;
}

// --- parser ---------------------------------------------------------------------

namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  Result<Graph> parse() {
    skip();
    if (!eat_word("graph")) return error("expected 'graph'");
    skip();
    std::string name = "graph";
    if (peek() == '@') {
      ++pos_;
      name = ident();
    }
    Graph g(name);
    skip();
    if (!eat('{')) return error("expected '{'");
    for (;;) {
      skip();
      if (eat('}')) break;
      if (pos_ >= s_.size()) return error("unexpected end of input (missing '}')");
      ENGINE_RETURN_IF_ERROR(statement(g));
    }
    skip();
    if (pos_ != s_.size()) return error("trailing text after the graph");
    return g;
  }

 private:
  Status statement(Graph& g) {
    if (!eat('%')) return error("expected %value");
    const std::string result = ident();
    if (result.empty()) return error("expected a value name after %");
    if (names_.count(result)) return error("value %" + result + " is defined twice");
    skip();
    if (!eat('=')) return error("expected '='");
    skip();
    const std::string opname = ident();
    OpKind kind;
    if (!parse_op_name(opname, kind)) return error("unknown op '" + opname + "'");
    skip();
    const bool leaf = kind == OpKind::kInput || kind == OpKind::kWeight || kind == OpKind::kKvCache ||
                      kind == OpKind::kConstant;
    std::string debug_name;
    if (leaf) {
      ENGINE_ASSIGN_OR_RETURN(debug_name, quoted());
      skip();
    }
    ConstData data;
    std::vector<Attr> const_values;
    if (kind == OpKind::kConstant) {
      if (!eat('[')) return error("expected [values] for constant");
      skip();
      while (!eat(']')) {
        if (pos_ >= s_.size()) return error("unterminated constant list");
        ENGINE_ASSIGN_OR_RETURN(Attr v, attr_value());
        if (std::holds_alternative<std::string>(v)) return error("constant values must be numbers");
        const_values.push_back(std::move(v));
        skip();
        eat(',');
        skip();
      }
      skip();
    }
    std::vector<ValueId> inputs;
    while (peek() == '%') {
      ++pos_;
      const std::string in = ident();
      auto it = names_.find(in);
      if (it == names_.end()) return error("use of undefined value %" + in);
      inputs.push_back(it->second);
      skip();
      if (!eat(',')) break;
      skip();
    }
    skip();
    Attrs attrs;
    if (eat('{')) {
      for (;;) {
        skip();
        if (eat('}')) break;
        const std::string key = ident();
        if (key.empty()) return error("expected attribute name");
        skip();
        if (!eat('=')) return error("expected '=' after attribute " + key);
        skip();
        ENGINE_ASSIGN_OR_RETURN(Attr v, attr_value());
        attrs[key] = std::move(v);
        skip();
        eat(',');
      }
      skip();
    }
    if (!eat(':')) return error("expected ': type'");
    skip();
    ENGINE_ASSIGN_OR_RETURN(Type type, parse_type());

    ValueId id;
    if (leaf) {
      if (!inputs.empty() || !attrs.empty()) return error(opname + " takes no operands or attributes");
      if (kind == OpKind::kConstant) {
        const bool floats = type.kind == ValueKind::kTensor && type.dtype == DType::kF32;
        if ((type.kind != ValueKind::kIndex && !floats) ||
            type.numel() != static_cast<int64_t>(const_values.size())) {
          return error("constant type must be index<i32>[" + std::to_string(const_values.size()) + "] or tensor<f32>[" +
                       std::to_string(const_values.size()) + "]");
        }
        for (const Attr& v : const_values) {
          const double d = std::holds_alternative<int64_t>(v) ? static_cast<double>(std::get<int64_t>(v))
                                                                : std::get<double>(v);
          if (floats) data.f32.push_back(static_cast<float>(d));
          else data.i32.push_back(static_cast<int32_t>(d));
        }
        id = g.add_const(type, std::move(data), debug_name);
      } else {
        const ValueKind want = kind == OpKind::kWeight    ? ValueKind::kWeight
                               : kind == OpKind::kKvCache ? ValueKind::kKv
                                                          : type.kind;
        if (type.kind != want) return error(opname + " must have a " + std::string(value_kind_name(want)) + " type");
        id = g.add_leaf(kind, type, debug_name);
      }
    } else {
      int64_t alias = -1;
      if (auto it = attrs.find("alias"); it != attrs.end()) {
        alias = std::holds_alternative<int64_t>(it->second) ? std::get<int64_t>(it->second) : -1;
        attrs.erase(it);
        if (alias < 0 || alias >= static_cast<int64_t>(inputs.size())) return error("alias out of range");
      }
      std::vector<Type> types;
      for (ValueId v : inputs) types.push_back(g.value(v).type);
      Result<Type> inferred = infer_type(kind, types, attrs);
      if (!inferred.ok()) return error(inferred.status().message());
      if (!result_type_matches(*inferred, type)) {
        return error("declared type " + type_to_string(type) + " but " + opname + " produces " +
                     type_to_string(*inferred));
      }
      int32_t buffer = -1;
      if (op_writes_in_place(kind) && !inputs.empty()) buffer = g.value(inputs[0]).buffer;
      if (alias >= 0) buffer = g.value(inputs[static_cast<size_t>(alias)]).buffer;
      id = g.add_op(kind, inputs, type, std::move(attrs), buffer);
    }
    names_[result] = id;
    return Status::Ok();
  }

  Result<Type> parse_type() {
    const std::string kind = ident();
    Type t;
    if (kind == "tensor") t.kind = ValueKind::kTensor;
    else if (kind == "weight") t.kind = ValueKind::kWeight;
    else if (kind == "kv") t.kind = ValueKind::kKv;
    else if (kind == "index") t.kind = ValueKind::kIndex;
    else return error("unknown type kind '" + kind + "' (tensor, weight, kv, index)");
    if (!eat('<')) return error("expected '<' in type");
    const std::string dt = ident();
    if (!parse_dtype(dt, t.dtype)) return error("unknown dtype '" + dt + "'");
    if (!eat('>')) return error("expected '>' in type");
    if (!eat('[')) return error("expected '[' in type");
    while (!eat(']')) {
      skip();
      if (std::isdigit(static_cast<unsigned char>(peek()))) {
        ENGINE_ASSIGN_OR_RETURN(int64_t n, integer());
        t.shape.push_back(Dim::of(n));
      } else {
        const std::string sym = ident();
        if (sym.empty()) return error("expected a dimension");
        t.shape.push_back(Dim::symbol(sym));
      }
      skip();
      eat(',');
    }
    if (t.kind == ValueKind::kWeight && dtype_is_quantized(t.dtype)) {
      t.layout = Layout{LayoutKind::kBlocked, dtype_block_elems(t.dtype)};
    }
    const size_t save = pos_;
    skip_spaces();
    if (std::isalpha(static_cast<unsigned char>(peek()))) {
      const size_t word_start = pos_;
      const std::string lay = ident();
      bool found = false;
      for (LayoutKind k : {LayoutKind::kRowMajor, LayoutKind::kColMajor, LayoutKind::kStrided, LayoutKind::kBlocked,
                           LayoutKind::kPacked, LayoutKind::kPaged}) {
        if (layout_kind_name(k) == lay) {
          t.layout = Layout{k, 0};
          found = true;
        }
      }
      if (!found) {
        pos_ = word_start;
        return error("unknown layout '" + lay + "'");
      }
      if (eat('(')) {
        ENGINE_ASSIGN_OR_RETURN(int64_t p, integer());
        t.layout.param = p;
        if (!eat(')')) return error("expected ')' after layout parameter");
      }
    } else {
      pos_ = save;
    }
    skip_spaces();
    if (peek() == '@') {
      ++pos_;
      if (ident() != "gpu") return error("expected @gpu");
      t.device = DeviceType::kCuda;
    }
    return t;
  }

  Result<Attr> attr_value() {
    if (peek() == '"') {
      ENGINE_ASSIGN_OR_RETURN(std::string s, quoted());
      return Attr(std::move(s));
    }
    const char c = peek();
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.') {
      const size_t start = pos_;
      while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.' ||
                                  s_[pos_] == '-' || s_[pos_] == '+')) {
        ++pos_;
      }
      const std::string_view num = s_.substr(start, pos_ - start);
      if (num.find_first_of(".eEn") == std::string_view::npos) {
        int64_t v = 0;
        auto [p, ec] = std::from_chars(num.data(), num.data() + num.size(), v);
        if (ec == std::errc() && p == num.data() + num.size()) return Attr(v);
      }
      try {
        size_t used = 0;
        const double d = std::stod(std::string(num), &used);
        if (used == num.size()) return Attr(d);
      } catch (...) {
      }
      return error("bad number '" + std::string(num) + "'");
    }
    const std::string word = ident();
    if (word.empty()) return error("expected an attribute value");
    return Attr(word);
  }

  Result<int64_t> integer() {
    const size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    int64_t v = 0;
    auto [p, ec] = std::from_chars(s_.data() + start, s_.data() + pos_, v);
    if (ec != std::errc() || p == s_.data() + start) return error("expected an integer");
    return v;
  }

  Result<std::string> quoted() {
    if (!eat('"')) return error("expected a quoted string");
    std::string out;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      if (s_[pos_] == '\n') return error("unterminated string");
      out += s_[pos_++];
    }
    if (!eat('"')) return error("unterminated string");
    return out;
  }

  std::string ident() {
    const size_t start = pos_;
    while (pos_ < s_.size() &&
           (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_' || s_[pos_] == '.')) {
      ++pos_;
    }
    return std::string(s_.substr(start, pos_ - start));
  }

  bool eat_word(std::string_view w) {
    if (s_.substr(pos_, w.size()) != w) return false;
    pos_ += w.size();
    return true;
  }
  bool eat(char c) {
    if (peek() != c) return false;
    ++pos_;
    return true;
  }
  char peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  void skip_spaces() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t')) ++pos_;
  }
  // Whitespace, newlines and // comments.
  void skip() {
    for (;;) {
      while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
      if (s_.substr(pos_, 2) == "//") {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
        continue;
      }
      return;
    }
  }

  Status error(const std::string& msg) const {
    int line = 1, col = 1;
    for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
      if (s_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    return InvalidArgument("ir:" + std::to_string(line) + ":" + std::to_string(col) + ": " + msg);
  }

  std::string_view s_;
  size_t pos_ = 0;
  std::unordered_map<std::string, ValueId> names_;
};

}  // namespace

Result<Graph> parse_graph(std::string_view text) {
  // Bound the input: the parser is also fed untrusted files by dynacorec.
  if (text.size() > (64u << 20)) return InvalidArgument("ir: input larger than 64 MiB");
  return Parser(text).parse();
}

}  // namespace dynacore::ir

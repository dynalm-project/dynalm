#include "api/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "tokenizer/unicode.h"

namespace engine::json {

const Value* Value::find(std::string_view key) const {
  if (!is_object()) return nullptr;
  const Object& o = as_object();
  auto it = o.find(key);
  return it == o.end() ? nullptr : &it->second;
}

namespace {

class Parser {
 public:
  Parser(std::string_view s, const ParseLimits& lim) : s_(s), lim_(lim) {}

  Result<Value> run() {
    if (s_.size() > lim_.max_bytes) return InvalidArgument("JSON input exceeds " + std::to_string(lim_.max_bytes) + " bytes");
    skip_ws();
    Value v;
    ENGINE_RETURN_IF_ERROR(value(v, 0));
    skip_ws();
    if (i_ != s_.size()) return error("trailing characters");
    return v;
  }

 private:
  Status error(const std::string& what) const {
    return InvalidArgument("invalid JSON at byte " + std::to_string(i_) + ": " + what);
  }
  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }
  bool consume(std::string_view lit) {
    if (s_.compare(i_, lit.size(), lit) != 0) return false;
    i_ += lit.size();
    return true;
  }

  Status value(Value& out, int depth) {
    if (depth > lim_.max_depth) return error("nesting deeper than " + std::to_string(lim_.max_depth));
    if (i_ >= s_.size()) return error("unexpected end of input");
    switch (s_[i_]) {
      case '{': return object(out, depth);
      case '[': return array(out, depth);
      case '"': {
        std::string str;
        ENGINE_RETURN_IF_ERROR(string(str));
        out = Value(std::move(str));
        return Status::Ok();
      }
      case 't':
        if (!consume("true")) return error("invalid literal");
        out = Value(true);
        return Status::Ok();
      case 'f':
        if (!consume("false")) return error("invalid literal");
        out = Value(false);
        return Status::Ok();
      case 'n':
        if (!consume("null")) return error("invalid literal");
        out = Value();
        return Status::Ok();
      default:
        return number(out);
    }
  }

  Status object(Value& out, int depth) {
    ++i_;  // '{'
    Object o;
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      out = Value(std::move(o));
      return Status::Ok();
    }
    for (;;) {
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != '"') return error("expected object key");
      std::string key;
      ENGINE_RETURN_IF_ERROR(string(key));
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') return error("expected ':'");
      ++i_;
      skip_ws();
      Value v;
      ENGINE_RETURN_IF_ERROR(value(v, depth + 1));
      o.insert_or_assign(std::move(key), std::move(v));  // last duplicate wins
      skip_ws();
      if (i_ >= s_.size()) return error("unterminated object");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        out = Value(std::move(o));
        return Status::Ok();
      }
      return error("expected ',' or '}'");
    }
  }

  Status array(Value& out, int depth) {
    ++i_;  // '['
    Array a;
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      out = Value(std::move(a));
      return Status::Ok();
    }
    for (;;) {
      skip_ws();
      Value v;
      ENGINE_RETURN_IF_ERROR(value(v, depth + 1));
      a.push_back(std::move(v));
      skip_ws();
      if (i_ >= s_.size()) return error("unterminated array");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        out = Value(std::move(a));
        return Status::Ok();
      }
      return error("expected ',' or ']'");
    }
  }

  Status hex4(uint32_t& cp) {
    if (i_ + 4 > s_.size()) return error("truncated \\u escape");
    cp = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_++];
      cp <<= 4;
      if (c >= '0' && c <= '9') cp |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') cp |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') cp |= static_cast<uint32_t>(c - 'A' + 10);
      else return error("invalid hex digit in \\u escape");
    }
    return Status::Ok();
  }

  Status string(std::string& out) {
    ++i_;  // opening quote
    for (;;) {
      if (i_ >= s_.size()) return error("unterminated string");
      const auto c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        ++i_;
        return Status::Ok();
      }
      if (c < 0x20) return error("control character in string");
      if (c != '\\') {
        out += static_cast<char>(c);
        ++i_;
        continue;
      }
      if (++i_ >= s_.size()) return error("truncated escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp;
          ENGINE_RETURN_IF_ERROR(hex4(cp));
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate: needs a low one
            if (!consume("\\u")) return error("unpaired surrogate");
            uint32_t lo;
            ENGINE_RETURN_IF_ERROR(hex4(lo));
            if (lo < 0xDC00 || lo > 0xDFFF) return error("invalid low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return error("unpaired surrogate");
          }
          unicode::append_utf8(cp, out);
          break;
        }
        default:
          return error("invalid escape");
      }
    }
  }

  Status number(Value& out) {
    const size_t start = i_;
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    if (i_ >= s_.size()) return error("invalid number");
    if (s_[i_] == '0') {
      ++i_;
    } else if (s_[i_] >= '1' && s_[i_] <= '9') {
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    } else {
      return error("unexpected character");
    }
    if (i_ < s_.size() && s_[i_] == '.') {
      ++i_;
      if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return error("invalid fraction");
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return error("invalid exponent");
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    // strtod needs a terminated buffer; numbers are short.
    const std::string num(s_.substr(start, i_ - start));
    out = Value(std::strtod(num.c_str(), nullptr));
    return Status::Ok();
  }

  std::string_view s_;
  ParseLimits lim_;
  size_t i_ = 0;
};

}  // namespace

Result<Value> parse(std::string_view text, const ParseLimits& limits) { return Parser(text, limits).run(); }

void append_quoted(std::string_view s, std::string& out) {
  out += '"';
  for (char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  out += '"';
}

void dump_to(const Value& v, std::string& out) {
  if (v.is_null()) {
    out += "null";
  } else if (v.is_bool()) {
    out += v.as_bool() ? "true" : "false";
  } else if (v.is_number()) {
    const double d = v.as_number();
    if (!std::isfinite(d)) {
      out += "null";
    } else if (d == std::floor(d) && std::abs(d) < 9e15) {
      out += std::to_string(static_cast<int64_t>(d));
    } else {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.17g", d);
      out += buf;
    }
  } else if (v.is_string()) {
    append_quoted(v.as_string(), out);
  } else if (v.is_array()) {
    out += '[';
    bool first = true;
    for (const Value& e : v.as_array()) {
      if (!first) out += ',';
      first = false;
      dump_to(e, out);
    }
    out += ']';
  } else {
    out += '{';
    bool first = true;
    for (const auto& [k, e] : v.as_object()) {
      if (!first) out += ',';
      first = false;
      append_quoted(k, out);
      out += ':';
      dump_to(e, out);
    }
    out += '}';
  }
}

std::string dump(const Value& v) {
  std::string s;
  dump_to(v, s);
  return s;
}

}  // namespace engine::json

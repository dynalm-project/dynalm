#pragma once

// Minimal JSON for the HTTP API: a value type, a strict RFC 8259 parser with
// explicit limits, and a compact serializer.
//
// The parser treats input as hostile: nesting depth and input size are
// bounded, every malformed input yields kInvalidArgument with a byte offset,
// and it never throws or reads out of bounds. Numbers are stored as double;
// strings are UTF-8 (\u escapes, including surrogate pairs, are decoded).

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "dynacore/base/status.h"
#include "common/core.h"

namespace dynalm::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value, std::less<>>;

class Value {
 public:
  Value() = default;                                    // null
  Value(std::nullptr_t) {}                              // NOLINT
  Value(bool b) : v_(b) {}                              // NOLINT
  Value(int v) : v_(static_cast<double>(v)) {}          // NOLINT
  Value(int64_t v) : v_(static_cast<double>(v)) {}      // NOLINT
  Value(double v) : v_(v) {}                            // NOLINT
  Value(const char* s) : v_(std::string(s)) {}          // NOLINT
  Value(std::string s) : v_(std::move(s)) {}            // NOLINT
  Value(Array a) : v_(std::make_shared<Array>(std::move(a))) {}    // NOLINT
  Value(Object o) : v_(std::make_shared<Object>(std::move(o))) {}  // NOLINT

  bool is_null() const { return std::holds_alternative<std::monostate>(v_); }
  bool is_bool() const { return std::holds_alternative<bool>(v_); }
  bool is_number() const { return std::holds_alternative<double>(v_); }
  bool is_string() const { return std::holds_alternative<std::string>(v_); }
  bool is_array() const { return std::holds_alternative<std::shared_ptr<Array>>(v_); }
  bool is_object() const { return std::holds_alternative<std::shared_ptr<Object>>(v_); }

  bool as_bool() const { return std::get<bool>(v_); }
  double as_number() const { return std::get<double>(v_); }
  const std::string& as_string() const { return std::get<std::string>(v_); }
  const Array& as_array() const { return *std::get<std::shared_ptr<Array>>(v_); }
  const Object& as_object() const { return *std::get<std::shared_ptr<Object>>(v_); }
  Array& as_array() { return *std::get<std::shared_ptr<Array>>(v_); }
  Object& as_object() { return *std::get<std::shared_ptr<Object>>(v_); }

  // Object member lookup; nullptr if not an object or missing.
  const Value* find(std::string_view key) const;

 private:
  std::variant<std::monostate, bool, double, std::string, std::shared_ptr<Array>, std::shared_ptr<Object>> v_;
};

struct ParseLimits {
  size_t max_bytes = 8 << 20;  // 8 MiB
  int max_depth = 64;
};

Result<Value> parse(std::string_view text, const ParseLimits& limits = {});

// Compact serialization (no whitespace). Integral numbers print without a
// fraction; non-finite numbers serialize as null.
std::string dump(const Value& v);
void dump_to(const Value& v, std::string& out);

// Appends `s` as a quoted, escaped JSON string.
void append_quoted(std::string_view s, std::string& out);

}  // namespace dynalm::json

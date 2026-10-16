#pragma once

// Environment-variable helpers for tuning switches.
//
// An empty value counts as unset: `VAR=` (or a variable another tool left
// behind with an empty value) must not turn a switch into 0. That once
// collapsed the scheduler's decode protection to a 1-row prefill budget
// (DYNALM_PROTECT_MIN= / DYNALM_DECODE_TARGET=). Numeric parsers accept only a
// complete number, so "16x" or "abc" are rejected rather than read as 16 or 0.

#include <cerrno>
#include <cstdlib>

namespace dynacore {

// The variable's value, or nullptr when it is unset or empty.
inline const char* env_value(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

// Whole-string integer / floating-point parse; false (out untouched) when s is
// null, empty, not a number, has trailing characters, or is out of range.
inline bool parse_env_long(const char* s, long& out) {
  if (s == nullptr || s[0] == '\0') return false;
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE) return false;
  out = v;
  return true;
}
inline bool parse_env_double(const char* s, double& out) {
  if (s == nullptr || s[0] == '\0') return false;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(s, &end);
  if (end == s || *end != '\0' || errno == ERANGE) return false;
  out = v;
  return true;
}

}  // namespace dynacore

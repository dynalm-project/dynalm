#pragma once

// Byte sizes on the command line: "8GB", "8G", "512MB", "512M", "4096" (bytes).
// Units are binary (G = 2^30). False on anything else or zero.

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

namespace dynalm::cli {

inline bool parse_byte_size(std::string_view s, int64_t& out) {
  size_t i = 0;
  double v = 0;
  bool digits = false, dot = false;
  double scale = 1;
  for (; i < s.size(); ++i) {
    const char ch = s[i];
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      digits = true;
      if (dot) {
        scale /= 10;
        v += (ch - '0') * scale;
      } else {
        v = v * 10 + (ch - '0');
      }
    } else if (ch == '.' && !dot) {
      dot = true;
    } else {
      break;
    }
  }
  if (!digits) return false;
  std::string unit(s.substr(i));
  for (char& ch : unit) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  if (unit.size() == 3 && unit.substr(1) == "IB") unit = unit.substr(0, 1);
  if (unit.size() == 2 && unit[1] == 'B') unit = unit.substr(0, 1);
  double mul = 1;
  if (unit == "K") mul = 1024.0;
  else if (unit == "M") mul = 1024.0 * 1024;
  else if (unit == "G") mul = 1024.0 * 1024 * 1024;
  else if (unit == "T") mul = 1024.0 * 1024 * 1024 * 1024;
  else if (!unit.empty() && unit != "B") return false;
  out = static_cast<int64_t>(v * mul);
  return out > 0;
}

}  // namespace dynalm::cli

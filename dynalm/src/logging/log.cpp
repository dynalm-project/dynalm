#include "logging/log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include "common/core.h"

namespace dynalm::log {

std::string_view level_name(Level l) {
  switch (l) {
    case Level::kTrace: return "TRACE";
    case Level::kDebug: return "DEBUG";
    case Level::kInfo: return "INFO";
    case Level::kWarn: return "WARN";
    case Level::kError: return "ERROR";
    case Level::kOff: return "OFF";
  }
  return "?";
}

bool parse_level(std::string_view s, Level& out) {
  static constexpr std::pair<std::string_view, Level> kLevels[] = {
      {"trace", Level::kTrace}, {"debug", Level::kDebug}, {"info", Level::kInfo},
      {"warn", Level::kWarn},   {"error", Level::kError}, {"off", Level::kOff}};
  for (const auto& [name, lvl] : kLevels) {
    if (s == name) {
      out = lvl;
      return true;
    }
  }
  return false;
}

namespace detail {

void write(Level level, std::string_view msg) {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
  const std::time_t t = system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::string line = std::format("{:02}:{:02}:{:02}.{:03} [{}] {}\n", tm.tm_hour, tm.tm_min,
                                 tm.tm_sec, ms, level_name(level), msg);
  // One fwrite per line; the mutex also keeps ordering sane across threads.
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  std::fwrite(line.data(), 1, line.size(), stderr);
}

}  // namespace detail
}  // namespace dynalm::log

#pragma once

// Leveled logging.
//
// Messages below the active level cost one relaxed atomic load and a branch:
// arguments are not formatted. Output is line-atomic (one write per line) so
// concurrent threads never interleave within a line. Do not log per token.

#include <atomic>
#include <format>
#include <string_view>

namespace engine::log {

enum class Level : int { kTrace = 0, kDebug, kInfo, kWarn, kError, kOff };

namespace detail {
inline std::atomic<int> g_level{static_cast<int>(Level::kInfo)};
void write(Level level, std::string_view msg);
}  // namespace detail

inline void set_level(Level level) {
  detail::g_level.store(static_cast<int>(level), std::memory_order_relaxed);
}
inline Level level() { return static_cast<Level>(detail::g_level.load(std::memory_order_relaxed)); }
inline bool enabled(Level l) {
  return static_cast<int>(l) >= detail::g_level.load(std::memory_order_relaxed);
}

// Parses "trace|debug|info|warn|error|off" (case-sensitive). Returns false on
// unknown input and leaves `out` unchanged.
bool parse_level(std::string_view s, Level& out);
std::string_view level_name(Level l);

template <typename... Args>
void logf(Level l, std::format_string<Args...> fmt, Args&&... args) {
  if (!enabled(l)) return;
  detail::write(l, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace engine::log

#define ENGINE_LOG(level, ...)                                                   \
  do {                                                                           \
    if (::engine::log::enabled(::engine::log::Level::k##level))                  \
      ::engine::log::logf(::engine::log::Level::k##level, __VA_ARGS__);          \
  } while (0)

#define LOG_TRACE(...) ENGINE_LOG(Trace, __VA_ARGS__)
#define LOG_DEBUG(...) ENGINE_LOG(Debug, __VA_ARGS__)
#define LOG_INFO(...) ENGINE_LOG(Info, __VA_ARGS__)
#define LOG_WARN(...) ENGINE_LOG(Warn, __VA_ARGS__)
#define LOG_ERROR(...) ENGINE_LOG(Error, __VA_ARGS__)

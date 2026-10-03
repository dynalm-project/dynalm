#pragma once

// Configuration sources for CLI commands (spec §42): a config file, environment
// variables and the command line, merged into one argument list that the
// command's own parser consumes. Later sources override earlier ones:
//
//   config file  <  DYNALM_* environment  <  command line
//
// Config file: one `key = value` (or bare `key` for flags) per line, `#`
// comments; keys are the long option names without "--" (e.g. `threads = 8`).
// Environment: DYNALM_<KEY> with '-' as '_' (e.g. DYNALM_HTTP_THREADS=32).
// The file is named by `--config PATH` on the command line or DYNALM_CONFIG.

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/status.h"

namespace engine {

struct OptionSpec {
  std::string_view name;  // long name without "--"
  bool takes_value = true;
};

// Returns the merged argument list (without --config). Unknown keys in the
// config file are errors; environment variables are looked up only for the
// listed options. `env` is injectable for tests (nullptr = process env).
using EnvLookup = const char* (*)(const char* name);
Result<std::vector<std::string>> merge_config(std::span<const std::string_view> cli_args,
                                              std::span<const OptionSpec> options, EnvLookup env = nullptr);

// Parses a config file's contents into "--key [value]" arguments.
Result<std::vector<std::string>> parse_config_text(std::string_view text, std::span<const OptionSpec> options,
                                                   std::string_view origin);

// "auto" or a non-negative integer; auto maps to 0 (the engine's AUTO mode).
bool parse_int_or_auto(std::string_view s, int& out);

}  // namespace engine

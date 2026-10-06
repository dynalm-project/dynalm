#pragma once

// Configuration sources for CLI commands: a config file, environment
// variables and the command line, merged into one argument list that the
// command's own parser consumes. Later sources override earlier ones:
//
//   config file  <  DYNALM_* environment  <  command line
//
// Config file, either format:
//   YAML subset (DD-070): two levels of `section:` / `  key: value`, `#`
//     comments, optional quotes. Keys are the documented ones in yaml_keys()
//     (e.g. runtime.threads, server.port); a key that the running command does
//     not use (server.port for `dynalm run`) is skipped, an unknown key is an
//     error with its line number. No YAML library: lists, anchors, multi-line
//     values and deeper nesting are rejected.
//   Flat: one `key = value` (or bare `key` for flags) per line; keys are the
//     command's long option names without "--" (e.g. `threads = 8`).
// Environment: DYNALM_<KEY> with '-' as '_' (e.g. DYNALM_HTTP_THREADS=32).
// The file is `--config PATH`, else $DYNALM_CONFIG, else ~/.dynalm/config.yaml
// when it exists.

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/core.h"
#include "dynacore/base/status.h"

namespace dynalm {

struct OptionSpec {
  std::string_view name;  // long name without "--"
  bool takes_value = true;
};

// One documented YAML key and the command-line option it sets.
struct YamlKey {
  std::string_view key;            // "runtime.threads"
  std::string_view option;         // "threads"
  std::string_view default_value;  // shown by `dynalm config show`
  std::string_view help;
  bool inverted_flag = false;      // server.admin: false sets --disable-admin
};
std::span<const YamlKey> yaml_keys();

// ~/.dynalm (%USERPROFILE%\.dynalm on Windows).
std::string dynalm_home();
// $DYNALM_CONFIG, else dynalm_home()/config.yaml (whether or not it exists).
std::string default_config_path();

// Returns the merged argument list (without --config). Unknown keys in the
// config file are errors; environment variables are looked up only for the
// listed options. `env` is injectable for tests (nullptr = process env).
using EnvLookup = const char* (*)(const char* name);
Result<std::vector<std::string>> merge_config(std::span<const std::string_view> cli_args,
                                              std::span<const OptionSpec> options, EnvLookup env = nullptr);

// Parses a config file's contents (either format) into "--key [value]"
// arguments for a command accepting `options`.
Result<std::vector<std::string>> parse_config_text(std::string_view text, std::span<const OptionSpec> options,
                                                   std::string_view origin);

// The YAML keys a config file sets, as (yaml key, value) pairs in file order.
// Only for YAML files; flat files return their keys unchanged.
Result<std::vector<std::pair<std::string, std::string>>> parse_config_entries(std::string_view text,
                                                                              std::string_view origin);

// "auto" or a non-negative integer; auto maps to 0 (the engine's AUTO mode).
bool parse_int_or_auto(std::string_view s, int& out);

}  // namespace dynalm

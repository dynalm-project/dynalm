#include "config/config.h"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace dynalm {
namespace {

// The documented configuration (docs/configuration.md). Intentionally small:
// what a user tunes, not every engine knob.
constexpr YamlKey kYamlKeys[] = {
    {"model.path", "model", "", "model file, directory or name (qwen3:4b)"},
    {"model.id", "model-id", "file name", "id reported by /v1/models"},
    {"runtime.device", "backend", "auto", "auto | cpu (GPU devices are not built yet)"},
    {"runtime.threads", "threads", "auto", "compute threads; auto = physical cores"},
    {"runtime.context_length", "ctx", "auto", "KV cache capacity in tokens; auto = from free RAM"},
    {"runtime.batch_tokens", "batch", "256", "max tokens per forward pass"},
    {"runtime.int8_decode", "int8-decode", "4", "int8 activations for matmuls of <= N rows; 0 = off"},
    {"scheduler.max_concurrent_requests", "max-active", "64", "concurrent requests before 503"},
    {"scheduler.policy", "policy", "balanced", "balanced | latency | throughput"},
    {"kv_cache.dtype", "kv", "f16", "f16 | f32"},
    {"sampling.temperature", "temperature", "1.0 (serve), 0.8 (chat)", "default when a request omits it"},
    {"sampling.max_tokens", "max-tokens", "1024 (serve), 2048 (chat)", "default max_tokens per request"},
    {"server.host", "host", "127.0.0.1", "bind address (0.0.0.0 to expose)"},
    {"server.port", "port", "8000", "HTTP port"},
    {"server.http_threads", "http-threads", "auto", "HTTP workers; auto = max_concurrent_requests + 8"},
    {"server.request_timeout", "request-timeout", "600", "per-request timeout in seconds; 0 = none"},
    {"server.shutdown_timeout", "shutdown-timeout", "30", "drain time on stop, seconds"},
    {"server.admin", "disable-admin", "true", "POST /admin/shutdown (used by dynalm stop)", true},
};

std::string_view trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
  return s;
}

const OptionSpec* find_option(std::span<const OptionSpec> options, std::string_view name) {
  for (const OptionSpec& o : options) {
    if (o.name == name) return &o;
  }
  return nullptr;
}

const YamlKey* find_yaml_key(std::string_view key) {
  for (const YamlKey& k : kYamlKeys) {
    if (k.key == key) return &k;
  }
  return nullptr;
}

std::string env_name(std::string_view option) {
  std::string n = "DYNALM_";
  for (char c : option) n += c == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return n;
}

bool truthy(std::string_view v) { return v == "1" || v == "true" || v == "yes" || v == "on"; }
bool falsy(std::string_view v) { return v == "0" || v == "false" || v == "no" || v == "off"; }

const char* process_env(const char* name) { return std::getenv(name); }

// Strips a trailing comment that is not inside quotes.
std::string_view strip_comment(std::string_view line) {
  char quote = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quote) {
      if (c == quote) quote = 0;
    } else if (c == '"' || c == '\'') {
      quote = c;
    } else if (c == '#') {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string_view unquote(std::string_view v) {
  if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) return v.substr(1, v.size() - 2);
  return v;
}

// A file is YAML when a non-comment line has a ':' and no '='.
bool looks_like_yaml(std::string_view text) {
  while (!text.empty()) {
    const size_t nl = text.find('\n');
    std::string_view line = trim(strip_comment(text.substr(0, nl)));
    text = nl == std::string_view::npos ? std::string_view() : text.substr(nl + 1);
    if (line.empty()) continue;
    return line.find('=') == std::string_view::npos && line.find(':') != std::string_view::npos;
  }
  return false;
}

struct Entry {
  std::string key, value;
  int line = 0;
};

Result<std::vector<Entry>> parse_yaml(std::string_view text, std::string_view origin) {
  std::vector<Entry> out;
  std::string section;
  int line_no = 0;
  while (!text.empty()) {
    const size_t nl = text.find('\n');
    std::string_view raw = text.substr(0, nl);
    text = nl == std::string_view::npos ? std::string_view() : text.substr(nl + 1);
    ++line_no;
    if (!raw.empty() && raw.back() == '\r') raw.remove_suffix(1);
    const std::string where = std::string(origin) + ":" + std::to_string(line_no);
    std::string_view line = strip_comment(raw);
    if (trim(line).empty()) continue;
    size_t indent = 0;
    while (indent < line.size() && (line[indent] == ' ' || line[indent] == '\t')) {
      if (line[indent] == '\t') return InvalidArgument(where + ": tabs are not allowed for indentation");
      ++indent;
    }
    line = trim(line);
    if (line.starts_with("- ") || line == "-") return InvalidArgument(where + ": lists are not supported");
    const size_t colon = line.find(':');
    if (colon == std::string_view::npos) return InvalidArgument(where + ": expected 'key: value'");
    const std::string_view key = trim(line.substr(0, colon));
    const std::string_view value = unquote(trim(line.substr(colon + 1)));
    if (key.empty()) return InvalidArgument(where + ": empty key");
    if (indent == 0) {
      if (value.empty()) {  // section header
        section = std::string(key);
        continue;
      }
      section.clear();
      out.push_back({std::string(key), std::string(value), line_no});  // top-level `threads: 8`
      continue;
    }
    if (section.empty()) return InvalidArgument(where + ": indented key without a section");
    if (value.empty()) return InvalidArgument(where + ": '" + std::string(key) + "' needs a value (no deeper nesting)");
    out.push_back({section + "." + std::string(key), std::string(value), line_no});
  }
  return out;
}

}  // namespace

std::span<const YamlKey> yaml_keys() { return kYamlKeys; }

std::string dynalm_home() {
#if defined(_WIN32)
  const char* home = std::getenv("USERPROFILE");
#else
  const char* home = std::getenv("HOME");
#endif
  const std::filesystem::path base =
      (home != nullptr && *home != '\0') ? std::filesystem::path(home) : std::filesystem::current_path();
  return (base / ".dynalm").string();
}

std::string default_config_path() {
  if (const char* p = std::getenv("DYNALM_CONFIG"); p != nullptr && *p != '\0') return p;
  return (std::filesystem::path(dynalm_home()) / "config.yaml").string();
}

bool parse_int_or_auto(std::string_view s, int& out) {
  if (s == "auto") {
    out = 0;
    return true;
  }
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size() && out >= 0;
}

Result<std::vector<std::pair<std::string, std::string>>> parse_config_entries(std::string_view text,
                                                                              std::string_view origin) {
  std::vector<std::pair<std::string, std::string>> out;
  if (!looks_like_yaml(text)) return out;
  ENGINE_ASSIGN_OR_RETURN(std::vector<Entry> entries, parse_yaml(text, origin));
  for (Entry& e : entries) out.emplace_back(std::move(e.key), std::move(e.value));
  return out;
}

Result<std::vector<std::string>> parse_config_text(std::string_view text, std::span<const OptionSpec> options,
                                                   std::string_view origin) {
  std::vector<std::string> args;
  auto add = [&](const OptionSpec& opt, std::string_view key, std::string_view value, bool has_value,
                 const std::string& where) -> Status {
    if (opt.takes_value) {
      if (!has_value || value.empty()) return InvalidArgument(where + ": '" + std::string(key) + "' needs a value");
      args.push_back("--" + std::string(opt.name));
      args.emplace_back(value);
    } else if (!has_value || truthy(value)) {
      args.push_back("--" + std::string(opt.name));
    } else if (!falsy(value)) {
      return InvalidArgument(where + ": '" + std::string(key) + "' is a flag (true/false)");
    }
    return Status::Ok();
  };

  if (looks_like_yaml(text)) {
    ENGINE_ASSIGN_OR_RETURN(std::vector<Entry> entries, parse_yaml(text, origin));
    for (const Entry& e : entries) {
      const std::string where = std::string(origin) + ":" + std::to_string(e.line);
      const YamlKey* yk = find_yaml_key(e.key);
      std::string_view option_name = yk != nullptr ? yk->option : std::string_view(e.key);
      // Top-level keys may name an option directly (`threads: 8`).
      const OptionSpec* opt = find_option(options, option_name);
      if (yk == nullptr && opt == nullptr) {
        return InvalidArgument(where + ": unknown key '" + e.key + "' (see docs/configuration.md)");
      }
      if (opt == nullptr) continue;  // a documented key this command does not use
      std::string_view value = e.value;
      if (option_name == "backend" && value == "auto") continue;  // auto = the default device
      if (yk != nullptr && yk->inverted_flag) {
        if (!truthy(value) && !falsy(value)) return InvalidArgument(where + ": '" + e.key + "' is true/false");
        if (falsy(value)) args.push_back("--" + std::string(opt->name));
        continue;
      }
      ENGINE_RETURN_IF_ERROR(add(*opt, e.key, value, true, where));
    }
    return args;
  }

  int line_no = 0;
  while (!text.empty()) {
    const size_t nl = text.find('\n');
    std::string_view line = text.substr(0, nl);
    text = nl == std::string_view::npos ? std::string_view() : text.substr(nl + 1);
    ++line_no;
    line = trim(strip_comment(line));
    if (line.empty()) continue;
    const size_t eq = line.find('=');
    const std::string_view key = trim(line.substr(0, eq));
    const std::string_view value = eq == std::string_view::npos ? std::string_view() : trim(line.substr(eq + 1));
    const std::string where = std::string(origin) + ":" + std::to_string(line_no);
    const OptionSpec* opt = find_option(options, key);
    if (!opt) return InvalidArgument(where + ": unknown key '" + std::string(key) + "'");
    ENGINE_RETURN_IF_ERROR(add(*opt, key, value, eq != std::string_view::npos, where));
  }
  return args;
}

Result<std::vector<std::string>> merge_config(std::span<const std::string_view> cli_args,
                                              std::span<const OptionSpec> options, EnvLookup env) {
  const bool real_env = env == nullptr;
  if (!env) env = process_env;
  std::vector<std::string> cli;
  std::string config_path;
  bool explicit_path = false;
  if (const char* p = env("DYNALM_CONFIG"); p != nullptr && *p != '\0') {
    config_path = p;
    explicit_path = true;
  }
  for (size_t i = 0; i < cli_args.size(); ++i) {
    if (cli_args[i] == "--config") {
      if (i + 1 >= cli_args.size()) return InvalidArgument("--config needs a file path");
      config_path = std::string(cli_args[++i]);
      explicit_path = true;
    } else {
      cli.emplace_back(cli_args[i]);
    }
  }
  // Zero configuration: ~/.dynalm/config.yaml applies when it exists. Tests
  // that inject an environment never see the user's file.
  if (!explicit_path && real_env) {
    std::error_code ec;
    const std::string def = default_config_path();
    if (std::filesystem::is_regular_file(def, ec)) config_path = def;
  }

  std::vector<std::string> merged;
  if (!config_path.empty()) {
    std::ifstream in(config_path, std::ios::binary);
    if (!in) return IoError("cannot open config file " + config_path);
    std::stringstream ss;
    ss << in.rdbuf();
    ENGINE_ASSIGN_OR_RETURN(std::vector<std::string> file_args, parse_config_text(ss.str(), options, config_path));
    merged = std::move(file_args);
  }
  for (const OptionSpec& o : options) {
    const char* v = env(env_name(o.name).c_str());
    if (!v) continue;
    if (o.takes_value) {
      merged.push_back("--" + std::string(o.name));
      merged.emplace_back(v);
    } else if (truthy(v)) {
      merged.push_back("--" + std::string(o.name));
    }
  }
  merged.insert(merged.end(), cli.begin(), cli.end());
  return merged;
}

}  // namespace dynalm

#include "config/config.h"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace engine {
namespace {

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

std::string env_name(std::string_view option) {
  std::string n = "ENGINE_";
  for (char c : option) n += c == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return n;
}

bool truthy(std::string_view v) { return v == "1" || v == "true" || v == "yes" || v == "on"; }

const char* process_env(const char* name) { return std::getenv(name); }

}  // namespace

bool parse_int_or_auto(std::string_view s, int& out) {
  if (s == "auto") {
    out = 0;
    return true;
  }
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size() && out >= 0;
}

Result<std::vector<std::string>> parse_config_text(std::string_view text, std::span<const OptionSpec> options,
                                                   std::string_view origin) {
  std::vector<std::string> args;
  int line_no = 0;
  while (!text.empty()) {
    const size_t nl = text.find('\n');
    std::string_view line = text.substr(0, nl);
    text = nl == std::string_view::npos ? std::string_view() : text.substr(nl + 1);
    ++line_no;
    if (const size_t hash = line.find('#'); hash != std::string_view::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    const size_t eq = line.find('=');
    const std::string_view key = trim(line.substr(0, eq));
    const std::string_view value = eq == std::string_view::npos ? std::string_view() : trim(line.substr(eq + 1));
    const std::string where = std::string(origin) + ":" + std::to_string(line_no);
    const OptionSpec* opt = find_option(options, key);
    if (!opt) return InvalidArgument(where + ": unknown key '" + std::string(key) + "'");
    if (opt->takes_value) {
      if (eq == std::string_view::npos || value.empty()) {
        return InvalidArgument(where + ": '" + std::string(key) + "' needs a value");
      }
      args.push_back("--" + std::string(key));
      args.emplace_back(value);
    } else if (eq == std::string_view::npos || truthy(value)) {
      args.push_back("--" + std::string(key));
    } else if (!(value == "0" || value == "false" || value == "no" || value == "off")) {
      return InvalidArgument(where + ": '" + std::string(key) + "' is a flag (true/false)");
    }
  }
  return args;
}

Result<std::vector<std::string>> merge_config(std::span<const std::string_view> cli_args,
                                              std::span<const OptionSpec> options, EnvLookup env) {
  if (!env) env = process_env;
  std::vector<std::string> cli;
  std::string config_path;
  if (const char* p = env("ENGINE_CONFIG")) config_path = p;
  for (size_t i = 0; i < cli_args.size(); ++i) {
    if (cli_args[i] == "--config") {
      if (i + 1 >= cli_args.size()) return InvalidArgument("--config needs a file path");
      config_path = std::string(cli_args[++i]);
    } else {
      cli.emplace_back(cli_args[i]);
    }
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

}  // namespace engine

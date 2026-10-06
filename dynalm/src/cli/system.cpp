// System commands:
//   dynalm doctor [--json]          system report for bug reports (diagnostics/doctor.h)
//   dynalm config [show|path|init]  effective configuration and the config file

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "cli/commands.h"
#include "config/config.h"
#include "diagnostics/doctor.h"
#include "common/core.h"

namespace dynalm::cli {
namespace {

constexpr const char* kConfigTemplate =
    "# DynaLM configuration (docs/configuration.md).\n"
    "# Every key is optional: delete what you do not need. Precedence:\n"
    "#   this file < DYNALM_* environment variables < command-line options\n"
    "\n"
    "model:\n"
    "  # path: qwen3:4b           # model for `dynalm serve` when none is given\n"
    "\n"
    "runtime:\n"
    "  threads: auto              # auto = physical cores\n"
    "  context_length: auto       # KV cache tokens; auto = from free RAM\n"
    "  # device: auto             # cpu (GPU devices are not built yet)\n"
    "\n"
    "scheduler:\n"
    "  max_concurrent_requests: 64\n"
    "  policy: balanced           # balanced | latency | throughput\n"
    "\n"
    "kv_cache:\n"
    "  dtype: f16                 # f16 | f32\n"
    "\n"
    "server:\n"
    "  host: 127.0.0.1            # 0.0.0.0 to accept other machines\n"
    "  port: 8000\n";

int config_show(const std::string& path) {
  std::map<std::string, std::string> file_values;
  std::error_code ec;
  const bool exists = std::filesystem::is_regular_file(path, ec);
  if (exists) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    auto entries = parse_config_entries(ss.str(), path);
    if (!entries.ok()) {
      std::fprintf(stderr, "config: %s\n", entries.status().message().c_str());
      return 1;
    }
    for (auto& [k, v] : *entries) file_values[k] = v;
  }
  std::printf("config file: %s%s\n\n", path.c_str(), exists ? "" : " (not present; defaults apply)");
  std::printf("%-36s %-24s %s\n", "KEY", "VALUE", "SOURCE");
  for (const YamlKey& k : yaml_keys()) {
    std::string env = "DYNALM_";
    for (char c : k.option) env += c == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::string value(k.default_value), source = "default";
    if (auto it = file_values.find(std::string(k.key)); it != file_values.end()) {
      value = it->second;
      source = "file";
    }
    if (const char* v = std::getenv(env.c_str()); v != nullptr && !k.inverted_flag) {
      value = v;
      source = env;
    }
    if (value.empty()) value = "-";
    std::printf("%-36s %-24s %s\n", std::string(k.key).c_str(), value.c_str(), source.c_str());
  }
  std::printf("\nCommand-line options override all of these.\n");
  return 0;
}

int config_init(const std::string& path, bool force) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::exists(path, ec) && !force) {
    std::fprintf(stderr, "config: %s exists (use --force to overwrite)\n", path.c_str());
    return 1;
  }
  fs::create_directories(fs::path(path).parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  out << kConfigTemplate;
  if (!out) {
    std::fprintf(stderr, "config: cannot write %s\n", path.c_str());
    return 1;
  }
  std::printf("wrote %s\n", path.c_str());
  return 0;
}

}  // namespace

int cmd_doctor(std::span<const std::string_view> args) {
  bool json = false;
  for (std::string_view a : args) {
    if (a == "--json") {
      json = true;
    } else {
      std::fprintf(stderr, "usage: dynalm doctor [--json]\n");
      return 1;
    }
  }
  const DoctorReport r = collect_doctor_report();
  const std::string out = json ? format_doctor_json(r) : format_doctor_text(r);
  std::fwrite(out.data(), 1, out.size(), stdout);
  return r.ready ? 0 : 1;
}

int cmd_config(std::span<const std::string_view> args) {
  std::string sub = args.empty() ? "show" : std::string(args[0]);
  std::string path = default_config_path();
  bool force = false;
  for (size_t i = args.empty() ? 0 : 1; i < args.size(); ++i) {
    if (args[i] == "--force") {
      force = true;
    } else if (args[i] == "--config" && i + 1 < args.size()) {
      path = args[++i];
    } else {
      sub = "help";
    }
  }
  if (sub == "show") return config_show(path);
  if (sub == "path") {
    std::printf("%s\n", path.c_str());
    return 0;
  }
  if (sub == "init") return config_init(path, force);
  std::fprintf(stderr,
               "usage: dynalm config [show|path|init] [--config FILE] [--force]\n"
               "  show   effective settings and where each comes from (default)\n"
               "  path   the config file used: $DYNALM_CONFIG or ~/.dynalm/config.yaml\n"
               "  init   write a commented starter config.yaml\n");
  return 1;
}

}  // namespace dynalm::cli

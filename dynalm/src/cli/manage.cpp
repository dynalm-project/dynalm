// Model management commands:
//   dynalm list [dir]                 GGUF files and Hugging Face model directories under dir
//                                     (default $DYNALM_MODELS_DIR or ./models)
//   dynalm pull <link>                download a GGUF model (cli/pull.cpp)
//   dynalm rm <model>... [-y]         delete downloaded models (GGUF files, partial downloads,
//                                     Hugging Face model directories)
//   dynalm stop [--host H] [--port P] ask a running `dynalm serve` to drain and exit
//
// A server process serves exactly one model (DD-039), so `dynalm unload` is
// the same operation as `dynalm stop`.

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "cli/commands.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "loader/hf/hf_model.h"
#include "loader/model_source.h"
#include "model/architecture.h"
#if ENGINE_HAS_SERVER
#include "server/server.h"
#include "common/core.h"
#endif

namespace dynalm::cli {

int cmd_list(std::span<const std::string_view> args) {
  namespace fs = std::filesystem;
  std::string dir = "models";
  if (const char* env = std::getenv("DYNALM_MODELS_DIR")) dir = env;
  if (!args.empty()) dir = args[0];
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    std::fprintf(stderr, "list: '%s' is not a directory\n", dir.c_str());
    return 1;
  }
  std::vector<fs::path> files, parts, hf_dirs;
  for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (it.depth() > 2) it.disable_recursion_pending();
    if (it->is_regular_file(ec) && it->path().extension() == ".gguf") files.push_back(it->path());
    if (it->is_regular_file(ec) && it->path().filename().string().ends_with(".gguf.part")) parts.push_back(it->path());
    if (it->is_directory(ec) && hf::is_hf_model(it->path().string()) &&
        hf::locate(it->path().string()).ok()) {
      hf_dirs.push_back(it->path());
    }
  }
  std::sort(hf_dirs.begin(), hf_dirs.end());
  std::sort(files.begin(), files.end());
  std::printf("%-48s %-10s %-10s %9s %8s  %s\n", "MODEL", "ARCH", "QUANT", "SIZE MiB", "CONTEXT", "STATUS");
  for (const fs::path& p : files) {
    const std::string rel = fs::relative(p, dir, ec).generic_string();
    auto g = gguf::GgufFile::open(p.string());
    if (!g.ok()) {
      std::printf("%-48s %-10s %-10s %9s %8s  unreadable: %s\n", rel.c_str(), "-", "-", "-", "-",
                  g.status().message().c_str());
      continue;
    }
    const gguf::GgufFile& f = **g;
    auto arch = f.get_string("general.architecture");
    const std::string arch_s = arch.ok() ? std::string(*arch) : "?";
    std::string quant = "?";
    if (auto ft = f.get_uint("general.file_type"); ft.ok()) quant = std::string(gguf::file_type_name(static_cast<uint32_t>(*ft)));
    std::string ctx = "?";
    if (auto c = f.get_uint(arch_s + ".context_length"); c.ok()) ctx = std::to_string(*c);
    std::printf("%-48s %-10s %-10s %9.1f %8s  %s\n", rel.c_str(), arch_s.c_str(), quant.c_str(),
                static_cast<double>(f.file().size()) / (1024.0 * 1024.0), ctx.c_str(),
                gguf_support_status(f).c_str());
  }
  // Hugging Face directories (config.json + safetensors).
  for (const fs::path& p : hf_dirs) {
    const std::string rel = fs::relative(p, dir, ec).generic_string() + "/";
    double mib = 0;
    if (auto f = hf::locate(p.string()); f.ok()) {
      for (const std::string& w : f->weights) mib += static_cast<double>(fs::file_size(w, ec)) / (1024.0 * 1024.0);
    }
    auto cfg_json = hf::read_json_file((p / "config.json").string(), 4 << 20);
    auto cfg = cfg_json.ok() ? hf::read_config(*cfg_json) : Result<ModelConfig>(cfg_json.status());
    const json::Value* dt = cfg_json.ok() ? cfg_json->find("torch_dtype") : nullptr;
    const std::string dtype = dt && dt->is_string() ? dt->as_string() : "?";
    if (!cfg.ok()) {
      std::printf("%-48s %-10s %-10s %9.1f %8s  %s\n", rel.c_str(), "-", dtype.c_str(), mib, "-",
                  cfg.status().message().c_str());
      continue;
    }
    std::printf("%-48s %-10s %-10s %9.1f %8lld  %s\n", rel.c_str(), cfg->architecture.c_str(), dtype.c_str(), mib,
                static_cast<long long>(cfg->context_length),
                find_architecture(cfg->architecture) ? "ok (safetensors)" : "unsupported architecture");
  }
  for (const fs::path& p : parts) {
    std::printf("%-48s %-10s %-10s %9.1f %8s  %s\n", fs::relative(p, dir, ec).generic_string().c_str(), "-", "-",
                static_cast<double>(fs::file_size(p, ec)) / (1024.0 * 1024.0), "-",
                "partial download (rerun dynalm pull to resume, or dynalm rm)");
  }
  if (files.empty() && parts.empty() && hf_dirs.empty()) std::printf("(no models under %s)\n", dir.c_str());
  return 0;
}

// Size of a file or, for a directory, of every file under it.
static uintmax_t disk_size(const std::filesystem::path& p) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(p, ec)) return fs::file_size(p, ec);
  uintmax_t total = 0;
  for (auto it = fs::recursive_directory_iterator(p, ec); !ec && it != fs::recursive_directory_iterator();
       it.increment(ec)) {
    if (it->is_regular_file(ec)) total += it->file_size(ec);
  }
  return total;
}

int cmd_rm(std::span<const std::string_view> args) {
  namespace fs = std::filesystem;
  std::string dir = "models";
  if (const char* env = std::getenv("DYNALM_MODELS_DIR"); env && *env) dir = env;
  bool yes = false, bad_arg = false;
  std::vector<std::string> names;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "-y" || args[i] == "--yes") yes = true;
    else if (args[i] == "--dir" && i + 1 < args.size()) dir = args[++i];
    else if (!args[i].starts_with("-")) names.emplace_back(args[i]);
    else bad_arg = true;
  }
  if (names.empty() || bad_arg) {
    std::fprintf(stderr,
                 "usage: dynalm rm <model>... [-y] [--dir DIR]\n"
                 "  <model>  a name as shown by `dynalm list` (the .gguf extension is optional), or a path\n"
                 "  -y       do not ask for confirmation\n"
                 "Deletes only GGUF files, partial downloads (.gguf.part) and Hugging Face model directories.\n");
    return 1;
  }
  int failed = 0;
  for (const std::string& name : names) {
    // A path as given, else a name under the models directory.
    std::error_code ec;
    fs::path target;
    for (const fs::path& c : {fs::path(name), fs::path(dir) / name, fs::path(dir) / (name + ".gguf"),
                              fs::path(dir) / (name + ".part"), fs::path(dir) / (name + ".gguf.part")}) {
      if (fs::exists(c, ec)) {
        target = c;
        break;
      }
    }
    if (target.empty()) {
      std::fprintf(stderr, "rm: no model '%s' (see dynalm list)\n", name.c_str());
      ++failed;
      continue;
    }
    // Only things that are models: never an arbitrary file or folder.
    const std::string fname = target.filename().string();
    const bool is_dir = fs::is_directory(target, ec);
    const bool ok_file =
        fs::is_regular_file(target, ec) && (fname.ends_with(".gguf") || fname.ends_with(".gguf.part"));
    if (!ok_file && !(is_dir && hf::is_hf_model(target.string()))) {
      std::fprintf(stderr, "rm: refusing to delete '%s': not a GGUF file or Hugging Face model directory\n",
                   target.string().c_str());
      ++failed;
      continue;
    }
    const double mib = static_cast<double>(disk_size(target)) / (1024.0 * 1024.0);
    if (!yes) {
      std::printf("delete %s%s (%.1f MiB)? [y/N] ", target.string().c_str(),
                  is_dir ? " and everything in it" : "", mib);
      std::fflush(stdout);
      char line[16] = {};
      if (!std::fgets(line, sizeof line, stdin) || (line[0] != 'y' && line[0] != 'Y')) {
        std::printf("kept %s\n", target.string().c_str());
        continue;
      }
    }
    if (is_dir) fs::remove_all(target, ec);
    else fs::remove(target, ec);
    if (ec) {
      std::fprintf(stderr, "rm: cannot delete %s: %s (is a dynalm server using it? stop it first)\n",
                   target.string().c_str(), ec.message().c_str());
      ++failed;
      continue;
    }
    std::printf("deleted %s (%.1f MiB freed)\n", target.string().c_str(), mib);
  }
  return failed ? 1 : 0;
}

int cmd_stop(std::span<const std::string_view> args) {
#if ENGINE_HAS_SERVER
  std::string host = "127.0.0.1";
  int port = 8000;
  for (size_t i = 0; i < args.size(); ++i) {
    bool ok = i + 1 < args.size();
    if (ok && args[i] == "--host") host = args[++i];
    else if (ok && args[i] == "--port") {
      const std::string_view v = args[++i];
      auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), port);
      ok = ec == std::errc() && p == v.data() + v.size();
    } else ok = false;
    if (!ok) {
      std::fprintf(stderr, "usage: dynalm stop [--host 127.0.0.1] [--port 8000]\n");
      return 1;
    }
  }
  if (Status st = request_server_shutdown(host, port); !st.ok()) {
    std::fprintf(stderr, "stop: %s\n", st.to_string().c_str());
    return 1;
  }
  std::printf("server at %s:%d is draining and will exit\n", host.c_str(), port);
  return 0;
#else
  (void)args;
  std::fprintf(stderr, "dynalm: built without the server (ENABLE_SERVER=OFF)\n");
  return 2;
#endif
}

}  // namespace dynalm::cli

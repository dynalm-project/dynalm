// Model management commands:
//   dynalm models [--dir DIR]         GGUF files and Hugging Face model directories in the model
//                                     store ($DYNALM_MODELS_DIR or ~/.dynalm/models) and ./models
//   dynalm models --available         names accepted by pull/run (registry/model_registry.h)
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
#include "config/config.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "loader/hf/hf_model.h"
#include "loader/model_source.h"
#include "model/architecture.h"
#include "registry/model_registry.h"
#include "common/core.h"
#if ENGINE_HAS_SERVER
#include "server/server.h"
#endif

namespace dynalm::cli {

namespace {

std::string human_size(double bytes) {
  char buf[32];
  if (bytes >= 1e9) {
    std::snprintf(buf, sizeof buf, "%.1f GB", bytes / 1e9);
  } else {
    std::snprintf(buf, sizeof buf, "%.0f MB", bytes / 1e6);
  }
  return buf;
}

// "~/.dynalm/models" instead of the full home path, for narrow terminals.
std::string short_dir(const std::string& dir) {
  const std::string home = std::filesystem::path(dynalm_home()).parent_path().string();
  if (!home.empty() && dir.rfind(home, 0) == 0) return "~" + dir.substr(home.size());
  return dir;
}

struct ModelRow {
  std::string name, format, size, location, quant, status;
};

void print_rows(const std::vector<ModelRow>& rows) {
  std::printf("%-40s %-12s %8s  %-20s %-12s %s\n", "NAME", "FORMAT", "SIZE", "LOCATION", "QUANTIZATION", "STATUS");
  for (const ModelRow& r : rows) {
    std::printf("%-40s %-12s %8s  %-20s %-12s %s\n", r.name.c_str(), r.format.c_str(), r.size.c_str(),
                r.location.c_str(), r.quant.c_str(), r.status.c_str());
  }
}

void scan_dir(const std::string& dir, std::vector<ModelRow>& rows) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::vector<fs::path> files, parts, hf_dirs;
  for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (it.depth() > 2) it.disable_recursion_pending();
    if (it->is_regular_file(ec) && it->path().extension() == ".gguf") files.push_back(it->path());
    if (it->is_regular_file(ec) && it->path().filename().string().ends_with(".gguf.part")) parts.push_back(it->path());
    if (it->is_directory(ec) && hf::is_hf_model(it->path().string()) && hf::locate(it->path().string()).ok()) {
      hf_dirs.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  std::sort(hf_dirs.begin(), hf_dirs.end());
  const std::string loc = short_dir(dir);
  for (const fs::path& p : files) {
    ModelRow r;
    const RegistryEntry* e = registry_entry_for_file(p.filename().string());
    r.name = e != nullptr ? std::string(e->name) : fs::relative(p, dir, ec).generic_string();
    r.format = "GGUF";
    r.location = loc;
    auto g = gguf::GgufFile::open(p.string());
    if (!g.ok()) {
      r.size = human_size(static_cast<double>(fs::file_size(p, ec)));
      r.quant = "-";
      r.status = "unreadable: " + g.status().message();
      rows.push_back(std::move(r));
      continue;
    }
    const gguf::GgufFile& f = **g;
    r.size = human_size(static_cast<double>(f.file().size()));
    r.quant = "?";
    if (auto ft = f.get_uint("general.file_type"); ft.ok()) {
      r.quant = std::string(gguf::file_type_name(static_cast<uint32_t>(*ft)));
    }
    const std::string st = gguf_support_status(f);
    r.status = st == "ok" ? "ready" : st;
    rows.push_back(std::move(r));
  }
  for (const fs::path& p : hf_dirs) {
    ModelRow r;
    r.name = fs::relative(p, dir, ec).generic_string() + "/";
    r.format = "SafeTensors";
    r.location = loc;
    double bytes = 0;
    if (auto f = hf::locate(p.string()); f.ok()) {
      for (const std::string& w : f->weights) bytes += static_cast<double>(fs::file_size(w, ec));
    }
    r.size = human_size(bytes);
    auto cfg_json = hf::read_json_file((p / "config.json").string(), 4 << 20);
    auto cfg = cfg_json.ok() ? hf::read_config(*cfg_json) : Result<ModelConfig>(cfg_json.status());
    const json::Value* dt = cfg_json.ok() ? cfg_json->find("torch_dtype") : nullptr;
    r.quant = dt && dt->is_string() ? dt->as_string() : "?";
    if (!cfg.ok()) {
      r.status = cfg.status().message();
    } else {
      r.status = find_architecture(cfg->architecture) ? "ready" : "unsupported architecture " + cfg->architecture;
    }
    rows.push_back(std::move(r));
  }
  for (const fs::path& p : parts) {
    ModelRow r;
    std::string file = p.filename().string();
    file.resize(file.size() - 5);  // drop ".part"
    const RegistryEntry* e = registry_entry_for_file(file);
    r.name = e != nullptr ? std::string(e->name) : fs::relative(p, dir, ec).generic_string();
    r.format = "GGUF";
    r.size = human_size(static_cast<double>(fs::file_size(p, ec)));
    r.location = loc;
    r.quant = e != nullptr ? std::string(e->quant) : "-";
    r.status = "partial (dynalm pull to resume)";
    rows.push_back(std::move(r));
  }
}

int list_available() {
  std::printf("%-20s %-12s %-8s %8s  %-6s %s\n", "NAME", "ARCH", "QUANT", "SIZE", "LOCAL", "DESCRIPTION");
  for (const RegistryEntry& e : registry_entries()) {
    auto r = resolve_model(e.name);
    std::printf("%-20s %-12s %-8s %8s  %-6s %s\n", std::string(e.name).c_str(), std::string(e.arch).c_str(),
                std::string(e.quant).c_str(), human_size(static_cast<double>(e.size_bytes)).c_str(),
                r.ok() && r->downloaded ? "yes" : "-", std::string(e.note).c_str());
  }
  std::printf("\nShort names work too: qwen3, llama, llama:3b, gemma, gemma:270m, phi, ...\n");
  return 0;
}

}  // namespace

int cmd_models(std::span<const std::string_view> args) {
  if (!args.empty() && (args[0] == "rm" || args[0] == "delete")) return cmd_rm(args.subspan(1));
  if (!args.empty() && args[0] == "pull") return cmd_pull(args.subspan(1));
  if (!args.empty() && (args[0] == "ls" || args[0] == "list")) args = args.subspan(1);
  std::vector<std::string> dirs;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--available" || args[i] == "-a") return list_available();
    if (args[i] == "--dir" && i + 1 < args.size()) {
      dirs.emplace_back(args[++i]);
    } else if (!args[i].starts_with("-") && dirs.empty()) {
      dirs.emplace_back(args[i]);  // `dynalm list DIR`, the pre-R1 form
    } else {
      std::fprintf(stderr,
                   "usage: dynalm models [ls] [--dir DIR] [--available]\n"
                   "       dynalm models rm <name>... [-y]\n"
                   "       dynalm models pull <name>\n"
                   "Lists local models in %s and ./models (or DIR).\n"
                   "--available lists the model names `dynalm pull` and `dynalm run` accept.\n",
                   models_dir().c_str());
      return 1;
    }
  }
  if (dirs.empty()) dirs = model_search_dirs();
  std::error_code ec;
  std::vector<ModelRow> rows;
  for (const std::string& d : dirs) {
    if (!std::filesystem::is_directory(d, ec)) {
      std::fprintf(stderr, "models: '%s' is not a directory\n", d.c_str());
      return 1;
    }
    scan_dir(d, rows);
  }
  if (rows.empty()) {
    std::printf("No local models in %s.\nDownload one:  dynalm pull qwen3:4b   (list: dynalm models --available)\n",
                short_dir(models_dir()).c_str());
    return 0;
  }
  print_rows(rows);
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
  std::vector<std::string> dirs = model_search_dirs();
  bool yes = false, bad_arg = false;
  std::vector<std::string> names;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "-y" || args[i] == "--yes") yes = true;
    else if (args[i] == "--dir" && i + 1 < args.size()) dirs = {std::string(args[++i])};
    else if (!args[i].starts_with("-")) names.emplace_back(args[i]);
    else bad_arg = true;
  }
  if (names.empty() || bad_arg) {
    std::fprintf(stderr,
                 "usage: dynalm rm <model>... [-y] [--dir DIR]\n"
                 "  <model>  a name as shown by `dynalm models` (the .gguf extension is optional), or a path\n"
                 "  -y       do not ask for confirmation\n"
                 "Deletes only GGUF files, partial downloads (.gguf.part) and Hugging Face model directories.\n");
    return 1;
  }
  int failed = 0;
  for (const std::string& name : names) {
    // A path as given, else a name under the models directory.
    std::error_code ec;
    fs::path target;
    std::vector<fs::path> candidates = {fs::path(name)};
    const RegistryEntry* entry = is_model_name(name) ? find_registry_entry(name) : nullptr;
    for (const std::string& dir : dirs) {
      if (entry != nullptr) {
        candidates.push_back(fs::path(dir) / std::string(entry->file()));
        candidates.push_back(fs::path(dir) / (std::string(entry->file()) + ".part"));
      }
      for (const std::string& c : {name, name + ".gguf", name + ".part", name + ".gguf.part"}) {
        candidates.push_back(fs::path(dir) / c);
      }
    }
    for (const fs::path& c : candidates) {
      if (fs::exists(c, ec)) {
        target = c;
        break;
      }
    }
    if (target.empty()) {
      std::fprintf(stderr, "rm: no model '%s' (see dynalm models)\n", name.c_str());
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

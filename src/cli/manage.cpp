// Model management commands (spec §41):
//   engine list [dir]                 GGUF models under dir (default $ENGINE_MODELS_DIR or ./models)
//   engine stop [--host H] [--port P] ask a running `engine serve` to drain and exit
//
// A server process serves exactly one model (DD-039), so `engine unload` is
// the same operation as `engine stop`.

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
#include "model/architecture.h"
#if ENGINE_HAS_SERVER
#include "server/server.h"
#endif

namespace engine::cli {

int cmd_list(std::span<const std::string_view> args) {
  namespace fs = std::filesystem;
  std::string dir = "models";
  if (const char* env = std::getenv("ENGINE_MODELS_DIR")) dir = env;
  if (!args.empty()) dir = args[0];
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    std::fprintf(stderr, "list: '%s' is not a directory\n", dir.c_str());
    return 1;
  }
  std::vector<fs::path> files;
  for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (it.depth() > 2) it.disable_recursion_pending();
    if (it->is_regular_file(ec) && it->path().extension() == ".gguf") files.push_back(it->path());
  }
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
    const bool supported = arch.ok() && find_architecture(*arch) != nullptr;
    std::printf("%-48s %-10s %-10s %9.1f %8s  %s\n", rel.c_str(), arch_s.c_str(), quant.c_str(),
                static_cast<double>(f.file().size()) / (1024.0 * 1024.0), ctx.c_str(),
                supported ? "ok" : "unsupported architecture");
  }
  if (files.empty()) std::printf("(no .gguf files under %s)\n", dir.c_str());
  return 0;
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
      std::fprintf(stderr, "usage: engine stop [--host 127.0.0.1] [--port 8000]\n");
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
  std::fprintf(stderr, "engine: built without the server (ENABLE_SERVER=OFF)\n");
  return 2;
#endif
}

}  // namespace engine::cli

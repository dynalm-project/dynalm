// dynalm pull <link> [-o DIR] [--force]
//
// Downloads one GGUF model file into the models directory. The link is a
// Hugging Face file link (page or download form), the short form
// <owner>/<repo>/<file>.gguf, or any direct http(s) URL. Before the full
// download, the first 256 KiB are fetched and the architecture is checked, so
// an unsupported model is refused in seconds instead of after many GB.
//
// Transfers use the system `curl` (shipped with Windows 10+, macOS and every
// Linux distribution), which brings TLS, proxies, redirects, retries and
// resume without adding a TLS library to DynaLM (DD-047). HF_TOKEN, if set, is
// sent to huggingface.co for gated models (via a private header file, not the
// command line).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "cli/commands.h"
#include "loader/gguf/gguf.h"
#include "loader/model_source.h"
#include "model/architecture.h"

#if defined(_WIN32)
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace engine::cli {
namespace {

namespace fs = std::filesystem;

constexpr size_t kProbeBytes = 256 * 1024;

// Runs argv[0] from PATH and waits; returns its exit code, -1 if it could not start.
int run_process(const std::vector<std::string>& argv) {
#if defined(_WIN32)
  // _spawnvp joins arguments with spaces: quote those that contain one. Links
  // are validated to contain no spaces or quotes; paths cannot contain quotes.
  std::vector<std::string> quoted;
  for (const std::string& a : argv) {
    quoted.push_back(a.find_first_of(" \t") == std::string::npos && !a.empty() ? a : "\"" + a + "\"");
  }
  std::vector<const char*> p;
  for (const std::string& a : quoted) p.push_back(a.c_str());
  p.push_back(nullptr);
  const intptr_t rc = _spawnvp(_P_WAIT, p[0], p.data());
  return rc < 0 ? -1 : static_cast<int>(rc);
#else
  std::vector<char*> p;
  for (const std::string& a : argv) p.push_back(const_cast<char*>(a.c_str()));
  p.push_back(nullptr);
  pid_t pid = 0;
  if (posix_spawnp(&pid, p[0], nullptr, nullptr, p.data(), environ) != 0) return -1;
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string temp_path(const char* tag) {
  std::random_device rd;
  return (fs::temp_directory_path() / ("dynalm-" + std::string(tag) + "-" + std::to_string(rd()))).string();
}

// Deletes a temporary file on scope exit.
struct TempFile {
  std::string path;
  ~TempFile() {
    if (path.empty()) return;
    std::error_code ec;
    fs::remove(path, ec);
  }
};

const char* hf_token() {
  if (const char* t = std::getenv("HF_TOKEN"); t && *t) return t;
  if (const char* t = std::getenv("HUGGING_FACE_HUB_TOKEN"); t && *t) return t;
  return nullptr;
}

// curl's exit codes, in words.
std::string curl_error(int rc) {
  switch (rc) {
    case -1: return "could not run curl (install it, or put it on PATH)";
    case 6: return "could not resolve the host (offline?)";
    case 7: return "could not connect to the server";
    case 22: return "the server returned an HTTP error: the file does not exist, or the model is gated "
                    "(accept its license on Hugging Face and set HF_TOKEN)";
    case 23: return "could not write the file (disk full or no permission?)";
    case 28: return "the connection timed out";
    case 35: case 60: return "TLS/certificate error";
    default: return "curl failed with exit code " + std::to_string(rc);
  }
}

void print_usage() {
  std::fprintf(stderr,
               "usage: dynalm pull <link> [-o DIR] [--force]\n"
               "\n"
               "  <link>   a .gguf file on Hugging Face or any http(s) URL, e.g.\n"
               "             https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf\n"
               "             Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf\n"
               "  -o DIR   where to save (default $DYNALM_MODELS_DIR or ./models)\n"
               "  --force  download even if the model looks unsupported, or is already present\n"
               "\n"
               "Interrupted downloads resume when the same command is run again.\n"
               "Gated models: accept the license on Hugging Face, then set HF_TOKEN.\n");
}

}  // namespace

int cmd_pull(std::span<const std::string_view> args) {
  std::string link, dir = "models";
  if (const char* env = std::getenv("DYNALM_MODELS_DIR"); env && *env) dir = env;
  bool force = false;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--force") force = true;
    else if ((args[i] == "-o" || args[i] == "--output") && i + 1 < args.size()) dir = args[++i];
    else if (args[i] == "-h" || args[i] == "--help") { print_usage(); return 0; }
    else if (link.empty() && !args[i].starts_with("-")) link = args[i];
    else { print_usage(); return 1; }
  }
  if (link.empty()) { print_usage(); return 1; }

  auto url = resolve_model_url(link);
  if (!url.ok()) {
    std::fprintf(stderr, "pull: %s\n", url.status().message().c_str());
    return 1;
  }
  const std::string name = url_file_name(*url);
  if (name.size() <= 5 || name.substr(name.size() - 5) != ".gguf") {
    std::fprintf(stderr, "pull: '%s' is not a .gguf file (pull downloads single GGUF files)\n", name.c_str());
    return 1;
  }

  std::error_code ec;
  fs::create_directories(dir, ec);
  if (!fs::is_directory(dir, ec)) {
    std::fprintf(stderr, "pull: cannot create directory '%s'\n", dir.c_str());
    return 1;
  }
  const fs::path dest = fs::path(dir) / name;
  const bool present = fs::exists(dest, ec);

  std::vector<std::string> common = {"curl", "-fL", "--retry", "5", "--retry-delay", "3"};
  TempFile header;
  if (const char* tok = hf_token(); tok && url->starts_with("https://huggingface.co/")) {
    // A file only this user can read keeps the token out of the process list.
    header.path = temp_path("auth");
    { std::ofstream(header.path, std::ios::binary).flush(); }
    fs::permissions(header.path, fs::perms::owner_read | fs::perms::owner_write, ec);
    std::ofstream(header.path, std::ios::binary) << "Authorization: Bearer " << tok << "\n";
    common.insert(common.end(), {"-H", "@" + header.path});
  }

  if (!present || force) {
    // 1. Check the header before downloading gigabytes. --max-filesize stops a
    //    server that ignores the range request from sending the whole file.
    std::printf("checking %s\n", url->c_str());
    std::fflush(stdout);
    TempFile probe{temp_path("probe")};
    std::vector<std::string> cmd = common;
    cmd.insert(cmd.end(), {"-sS", "--max-filesize", std::to_string(4 * kProbeBytes), "-r",
                           "0-" + std::to_string(kProbeBytes - 1), "-o", probe.path, *url});
    const int rc = run_process(cmd);
    if (rc == 0) {
      std::vector<char> head(kProbeBytes);
      std::ifstream in(probe.path, std::ios::binary);
      in.read(head.data(), static_cast<std::streamsize>(head.size()));
      head.resize(static_cast<size_t>(in.gcount()));
      auto arch = peek_gguf_architecture(std::as_bytes(std::span(head)));
      if (arch.ok()) {
        if (find_architecture(*arch) == nullptr) {
          std::fprintf(stderr, "pull: architecture '%s' is not supported by DynaLM, so this model cannot run\n",
                       arch->c_str());
          if (!force) {
            std::fprintf(stderr, "      (supported models: docs/model-support.md; download anyway with --force)\n");
            return 1;
          }
        } else {
          std::printf("architecture %s: supported\n", arch->c_str());
        }
      } else if (arch.status().code() == StatusCode::kNotFound) {
        std::printf("note: could not read the architecture from the file header; downloading anyway\n");
      } else {
        std::fprintf(stderr, "pull: %s\n", arch.status().message().c_str());
        return 1;
      }
    } else if (rc == 63) {
      std::printf("note: the server does not support partial downloads; skipping the pre-check\n");
    } else {
      std::fprintf(stderr, "pull: %s\n", curl_error(rc).c_str());
      return 1;
    }

    // 2. Download to <name>.part (resumable), then rename into place.
    const std::string part = dest.string() + ".part";
    std::printf("downloading to %s\n", dest.string().c_str());
    std::fflush(stdout);
    cmd = common;
    cmd.insert(cmd.end(), {"-#", "-C", "-", "-o", part, *url});
    if (const int dl = run_process(cmd); dl != 0) {
      std::fprintf(stderr, "pull: %s\n      run the same command again to resume\n", curl_error(dl).c_str());
      return 1;
    }
    fs::rename(part, dest, ec);
    if (ec) {
      std::fprintf(stderr, "pull: cannot rename %s: %s\n", part.c_str(), ec.message().c_str());
      return 1;
    }
  } else {
    std::printf("%s is already downloaded (use --force to download again)\n", dest.string().c_str());
  }

  // 3. Full check of what arrived: metadata and every tensor type.
  auto f = gguf::GgufFile::open(dest.string());
  if (!f.ok()) {
    std::fprintf(stderr, "pull: the downloaded file is not a valid GGUF: %s\n", f.status().to_string().c_str());
    return 1;
  }
  const std::string verdict = gguf_support_status(**f);
  std::printf("%s %s (%.1f MiB): %s\n", present && !force ? "found" : "saved", dest.string().c_str(),
              static_cast<double>((*f)->file().size()) / (1024.0 * 1024.0), verdict.c_str());
  if (verdict == "ok") {
    std::printf("try it:  dynalm run \"%s\" -p \"Hello\"\n", dest.generic_string().c_str());
  }
  return 0;
}

}  // namespace engine::cli

// `dynalm` command-line entry point.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "cli/commands.h"
#include "common/version.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/system_info.h"
#include "dynacore/version.h"
#include "common/core.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#endif

namespace {

#if defined(_WIN32)
// Model output is UTF-8. A Windows console in its default code page shows the
// bytes of every non-ASCII character as garbage, so
// switch the console to UTF-8 while dynalm runs and restore it on exit.
UINT g_prev_output_cp = 0;
void restore_console_cp() { SetConsoleOutputCP(g_prev_output_cp); }

void use_utf8_console() {
  const UINT cp = GetConsoleOutputCP();
  if (cp == 0 || cp == CP_UTF8 || !SetConsoleOutputCP(CP_UTF8)) return;  // no console, or already UTF-8
  g_prev_output_cp = cp;
  std::atexit(restore_console_cp);
}
// Arguments, paths and std::filesystem are UTF-8 too: cli/dynalm.manifest sets
// the process code page to UTF-8 (Windows 10 1903+), so non-English prompts
// reaches the tokenizer intact.
#endif



void print_usage() {
  std::printf(
      "usage: dynalm [-q | -v | --log-level LEVEL] <command> [args]\n"
      "\n"
      "Get started:\n"
      "  dynalm doctor                 check this machine\n"
      "  dynalm pull qwen3:4b          download a model (dynalm models --available lists names)\n"
      "  dynalm run qwen3:4b           chat in the terminal\n"
      "  dynalm serve qwen3:4b         OpenAI-compatible API on http://127.0.0.1:8000/v1\n"
      "\n"
      "commands:\n"
      "  run <model> [-p TEXT]   chat, or generate once with -p (dynalm run for options)\n"
      "  serve <model>           OpenAI-compatible HTTP server (dynalm serve for options)\n"
      "  pull <model|link>       download a model by name, Hugging Face link or URL\n"
      "  models [rm|--available] local models; remove one; names you can pull\n"
      "  inspect <model>         architecture, parameters, quantization, memory estimate\n"
      "  benchmark <model>       throughput and latency (in-process or --url server)\n"
      "  doctor [--json]         system report: CPU, ISA, RAM, GPU, device, status\n"
      "  config [show|path|init] effective configuration and ~/.dynalm/config.yaml\n"
      "  stop                    ask a local server to drain and exit (--host, --port)\n"
      "  version                 DynaLM and DynaCore versions, build configuration\n"
      "\n"
      "<model> is a name (qwen3:4b, llama:3b, gemma:270m), a .gguf file or a Hugging Face directory.\n"
      "Log levels: quiet, normal, verbose, debug, trace (-q = quiet, -v = verbose; or DYNALM_LOG_LEVEL).\n");
}

int cmd_version() {
  std::printf("DynaLM   %s\n", ENGINE_VERSION_STRING);
  std::printf("DynaCore %s\n", dynacore::kVersionString);
  std::printf("build:    %s, %s %s, %s\n", ENGINE_BUILD_TYPE, ENGINE_COMPILER_ID, ENGINE_COMPILER_VERSION,
              dynacore::os_info().arch.c_str());
  std::printf("kernels:  generic%s%s%s%s (selected at run time: %s)\n", ENGINE_HAS_AVX2 ? " avx2" : "",
              ENGINE_HAS_AVX512 ? " avx512" : "", ENGINE_HAS_AMX ? " amx" : "", ENGINE_HAS_NEON ? " neon" : "",
              std::string(dynalm::isa_name(dynalm::select_isa(dynalm::cpu_info().features))).c_str());
  std::printf("devices:  cpu; cuda, hip, metal, vulkan: designed, not built (DD-045)\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  use_utf8_console();
#endif
  std::vector<std::string_view> args(argv + 1, argv + argc);

  // Global options precede the command. DYNALM_LOG_LEVEL sets the default.
  auto set_level = [](std::string_view name) {
    dynalm::log::Level lvl;
    if (!dynalm::log::parse_level(name, lvl)) {
      std::fprintf(stderr, "dynalm: invalid log level '%.*s' (quiet, normal, verbose, debug, trace)\n",
                   static_cast<int>(name.size()), name.data());
      return false;
    }
    dynalm::log::set_level(lvl);
    return true;
  };
  if (const char* env = std::getenv("DYNALM_LOG_LEVEL"); env != nullptr && *env != '\0' && !set_level(env)) return 1;
  size_t i = 0;
  while (i < args.size() && args[i].starts_with("-")) {
    if (args[i] == "--log-level" && i + 1 < args.size()) {
      if (!set_level(args[i + 1])) return 1;
      i += 2;
    } else if (args[i] == "-q" || args[i] == "--quiet") {
      set_level("quiet");
      ++i;
    } else if (args[i] == "-v" || args[i] == "--verbose") {
      set_level("verbose");
      ++i;
    } else if (args[i] == "--help" || args[i] == "-h") {
      print_usage();
      return 0;
    } else if (args[i] == "--version") {
      return cmd_version();
    } else {
      std::fprintf(stderr, "dynalm: unknown option '%.*s'\n", static_cast<int>(args[i].size()),
                   args[i].data());
      return 1;
    }
  }

  if (i >= args.size()) {
    print_usage();
    return 1;
  }

  const std::string_view cmd = args[i];
  const auto rest = std::span(args).subspan(i + 1);
  if (cmd == "version") return cmd_version();
  if (cmd == "doctor" || cmd == "info") return dynalm::cli::cmd_doctor(rest);  // info: pre-R1 name
  if (cmd == "config") return dynalm::cli::cmd_config(rest);
  if (cmd == "inspect") return dynalm::cli::cmd_inspect(rest);
  if (cmd == "run") return dynalm::cli::cmd_run(rest);
  if (cmd == "chat") return dynalm::cli::cmd_run(rest);  // run without -p
#if ENGINE_HAS_SERVER
  if (cmd == "serve") return dynalm::cli::cmd_serve(rest);
#else
  if (cmd == "serve") {
    std::fprintf(stderr, "dynalm: built without the server (ENABLE_SERVER=OFF)\n");
    return 2;
  }
#endif
  if (cmd == "benchmark") return dynalm::cli::cmd_benchmark(rest);
  if (cmd == "pull") return dynalm::cli::cmd_pull(rest);
  if (cmd == "models" || cmd == "list" || cmd == "ls") return dynalm::cli::cmd_models(rest);  // list: pre-R1 name
  if (cmd == "rm" || cmd == "delete") return dynalm::cli::cmd_rm(rest);
  // One model per server process (DD-039): unloading it means stopping it.
  if (cmd == "stop" || cmd == "unload") return dynalm::cli::cmd_stop(rest);
  if (cmd == "help") {
    print_usage();
    return 0;
  }

  std::fprintf(stderr, "dynalm: unknown command '%.*s'\n", static_cast<int>(cmd.size()), cmd.data());
  print_usage();
  return 1;
}

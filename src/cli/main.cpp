// `dynalm` command-line entry point.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "cli/commands.h"
#include "common/version.h"
#include "logging/log.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

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

using engine::CpuIsa;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

void print_usage() {
  std::printf(
      "usage: dynalm [--log-level trace|debug|info|warn|error|off] <command> [args]\n"
      "\n"
      "commands:\n"
      "  version              print version and build configuration\n"
      "  info                 print detected hardware and selected CPU backend\n"
      "  inspect <model>      show model metadata, config and memory estimate\n"
      "  run <model> -p TEXT  generate from a prompt (dynalm run for options)\n"
      "  chat <model>         interactive chat, like `ollama run` (also: run <model> without -p)\n"
      "  serve <model>        OpenAI-compatible HTTP server (dynalm serve for options)\n"
      "  benchmark <model>    load test (in-process or --url server), P50-P99 latency\n"
      "  pull <link>          download a GGUF model (Hugging Face link or URL) into ./models\n"
      "  rm <model>...        delete downloaded models (asks first; -y to skip)\n"
      "  list [dir]           GGUF models under dir (default ./models or $DYNALM_MODELS_DIR)\n"
      "  stop | unload        ask a local server to drain and exit (--host, --port)\n");
}

int cmd_version() {
  std::printf("DynaLM %s\n", ENGINE_VERSION_STRING);
  std::printf("build:    %s, %s %s\n", ENGINE_BUILD_TYPE, ENGINE_COMPILER_ID, ENGINE_COMPILER_VERSION);
  std::printf("kernels:  generic%s%s%s%s\n", ENGINE_HAS_AVX2 ? " avx2" : "", ENGINE_HAS_AVX512 ? " avx512" : "",
              ENGINE_HAS_AMX ? " amx" : "", ENGINE_HAS_NEON ? " neon" : "");
  std::printf("backends: cpu (built); cuda, hip, metal, vulkan: designed, not built (DD-045)\n");
  return 0;
}

std::string feature_list(const engine::CpuFeatures& f) {
  std::string s;
  auto add = [&](bool on, const char* name) {
    if (!on) return;
    if (!s.empty()) s += ' ';
    s += name;
  };
  add(f.sse42, "sse4.2");
  add(f.avx, "avx");
  add(f.avx2, "avx2");
  add(f.fma, "fma");
  add(f.f16c, "f16c");
  add(f.avx_vnni, "avx-vnni");
  add(f.avx512f, "avx512f");
  add(f.avx512bw, "avx512bw");
  add(f.avx512vl, "avx512vl");
  add(f.avx512dq, "avx512dq");
  add(f.avx512_vnni, "avx512-vnni");
  add(f.avx512_bf16, "avx512-bf16");
  add(f.amx_tile, "amx-tile");
  add(f.amx_int8, "amx-int8");
  add(f.amx_bf16, "amx-bf16");
  add(f.neon, "neon");
  add(f.arm_dotprod, "dotprod");
  return s.empty() ? "(none)" : s;
}

int cmd_info() {
  const engine::CpuInfo& cpu = engine::cpu_info();
  const engine::MemoryInfo mem = engine::memory_info();

  std::printf("CPU:            %s (%s)\n", cpu.brand.c_str(), cpu.vendor.c_str());
  std::printf("Cores:          %d physical, %d logical", cpu.physical_cores, cpu.logical_cores);
  if (cpu.performance_cores > 0) {
    std::printf(" (hybrid: %d P-cores, %d E-cores)", cpu.performance_cores, cpu.efficiency_cores);
  }
  std::printf("\n");
  std::printf("Caches:         L1d %lld KiB/core, L2 %lld KiB, L3 %lld KiB\n",
              static_cast<long long>(cpu.l1d_bytes / 1024), static_cast<long long>(cpu.l2_bytes / 1024),
              static_cast<long long>(cpu.l3_bytes / 1024));
  std::printf("Features:       %s\n", feature_list(cpu.features).c_str());
  std::printf("RAM:            %.1f GiB total, %.1f GiB available\n", mem.total_bytes / kGiB,
              mem.available_bytes / kGiB);

  std::string usable;
  for (CpuIsa isa : engine::usable_isas(cpu.features)) {
    if (!usable.empty()) usable += ", ";
    usable += engine::isa_name(isa);
  }
  std::printf("Usable kernels: %s\n", usable.c_str());
  std::printf("Backend:        CPU/%s\n",
              std::string(engine::isa_name(engine::select_best_isa(cpu.features))).c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  use_utf8_console();
#endif
  std::vector<std::string_view> args(argv + 1, argv + argc);

  // Global options precede the command.
  size_t i = 0;
  while (i < args.size() && args[i].starts_with("--")) {
    if (args[i] == "--log-level" && i + 1 < args.size()) {
      engine::log::Level lvl;
      if (!engine::log::parse_level(args[i + 1], lvl)) {
        std::fprintf(stderr, "dynalm: invalid log level '%.*s'\n",
                     static_cast<int>(args[i + 1].size()), args[i + 1].data());
        return 1;
      }
      engine::log::set_level(lvl);
      i += 2;
    } else if (args[i] == "--help") {
      print_usage();
      return 0;
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
  if (cmd == "version") return cmd_version();
  if (cmd == "info") return cmd_info();
  if (cmd == "inspect") return engine::cli::cmd_inspect(std::span(args).subspan(i + 1));
  if (cmd == "run") return engine::cli::cmd_run(std::span(args).subspan(i + 1));
  if (cmd == "chat") return engine::cli::cmd_run(std::span(args).subspan(i + 1));  // run without -p
#if ENGINE_HAS_SERVER
  if (cmd == "serve") return engine::cli::cmd_serve(std::span(args).subspan(i + 1));
#else
  if (cmd == "serve") {
    std::fprintf(stderr, "dynalm: built without the server (ENABLE_SERVER=OFF)\n");
    return 2;
  }
#endif
  if (cmd == "benchmark") return engine::cli::cmd_benchmark(std::span(args).subspan(i + 1));
  if (cmd == "pull") return engine::cli::cmd_pull(std::span(args).subspan(i + 1));
  if (cmd == "rm" || cmd == "delete") return engine::cli::cmd_rm(std::span(args).subspan(i + 1));
  if (cmd == "list") return engine::cli::cmd_list(std::span(args).subspan(i + 1));
  // One model per server process (DD-039): unloading it means stopping it.
  if (cmd == "stop" || cmd == "unload") return engine::cli::cmd_stop(std::span(args).subspan(i + 1));
  if (cmd == "help") {
    print_usage();
    return 0;
  }

  std::fprintf(stderr, "dynalm: unknown command '%.*s'\n", static_cast<int>(cmd.size()), cmd.data());
  print_usage();
  return 1;
}

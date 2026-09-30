// `engine` command-line entry point.

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "cli/commands.h"
#include "common/version.h"
#include "logging/log.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

namespace {

using engine::CpuIsa;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

void print_usage() {
  std::printf(
      "usage: engine [--log-level trace|debug|info|warn|error|off] <command> [args]\n"
      "\n"
      "commands:\n"
      "  version              print version and build configuration\n"
      "  info                 print detected hardware and selected CPU backend\n"
      "  inspect <model>      show model metadata            (phase 2/3)\n"
      "  run <model>          generate from a prompt         (phase 6)\n"
      "  serve <model>        start OpenAI-compatible server (phase 20)\n"
      "  benchmark <model>    run the benchmark suite        (phase 21)\n"
      "  list | stop | unload model management               (phase 22)\n");
}

int cmd_version() {
  std::printf("engine %s\n", ENGINE_VERSION_STRING);
  std::printf("build:    %s, %s %s\n", ENGINE_BUILD_TYPE, ENGINE_COMPILER_ID, ENGINE_COMPILER_VERSION);
  std::printf("kernels:  generic%s%s%s\n", ENGINE_HAS_AVX2 ? " avx2" : "",
              ENGINE_HAS_AVX512 ? " avx512" : "", ENGINE_HAS_AMX ? " amx" : "");
  std::printf("backends: cpu\n");
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

int not_implemented(std::string_view cmd, int phase) {
  std::fprintf(stderr, "engine: '%.*s' is not implemented yet (planned for phase %d)\n",
               static_cast<int>(cmd.size()), cmd.data(), phase);
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> args(argv + 1, argv + argc);

  // Global options precede the command.
  size_t i = 0;
  while (i < args.size() && args[i].starts_with("--")) {
    if (args[i] == "--log-level" && i + 1 < args.size()) {
      engine::log::Level lvl;
      if (!engine::log::parse_level(args[i + 1], lvl)) {
        std::fprintf(stderr, "engine: invalid log level '%.*s'\n",
                     static_cast<int>(args[i + 1].size()), args[i + 1].data());
        return 1;
      }
      engine::log::set_level(lvl);
      i += 2;
    } else if (args[i] == "--help") {
      print_usage();
      return 0;
    } else {
      std::fprintf(stderr, "engine: unknown option '%.*s'\n", static_cast<int>(args[i].size()),
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
  if (cmd == "run") return not_implemented(cmd, 6);
  if (cmd == "serve") return not_implemented(cmd, 20);
  if (cmd == "benchmark") return not_implemented(cmd, 21);
  if (cmd == "list" || cmd == "stop" || cmd == "unload") return not_implemented(cmd, 22);
  if (cmd == "help") {
    print_usage();
    return 0;
  }

  std::fprintf(stderr, "engine: unknown command '%.*s'\n", static_cast<int>(cmd.size()), cmd.data());
  print_usage();
  return 1;
}

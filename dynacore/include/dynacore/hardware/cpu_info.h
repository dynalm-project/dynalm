#pragma once

// Runtime hardware detection. Queried once at startup; feeds the kernel
// dispatcher (ISA choice) and AUTO configuration (threads, memory budgets).

#include <cstdint>
#include <string>
#include <vector>

namespace dynacore {

struct CpuFeatures {
  // x86. Each flag means "the CPU supports it AND the OS saves the state".
  bool sse42 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx_vnni = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512dq = false;
  bool avx512_vnni = false;
  bool avx512_bf16 = false;
  bool amx_tile = false;
  bool amx_int8 = false;
  bool amx_bf16 = false;
  // arm64
  bool neon = false;
  bool arm_dotprod = false;
};

struct CpuInfo {
  std::string vendor;
  std::string brand;
  CpuFeatures features;

  int logical_cores = 0;
  int physical_cores = 0;
  // Hybrid CPUs (Intel Alder Lake+). 0 performance cores means "not hybrid /
  // unknown": treat all physical cores as equal.
  int performance_cores = 0;
  int efficiency_cores = 0;

  // Per-core L1d/L2, shared L3, in bytes (0 = unknown). Used for GEMM tiling.
  int64_t l1d_bytes = 0;
  int64_t l2_bytes = 0;
  int64_t l3_bytes = 0;

  // First logical CPU of each physical core, performance cores first (empty
  // when unknown, e.g. macOS). Used to place one compute thread per core.
  std::vector<int> core_first_cpu;
};

struct MemoryInfo {
  int64_t total_bytes = 0;
  int64_t available_bytes = 0;
};

// Detected once and cached; thread-safe.
const CpuInfo& cpu_info();

// Queried fresh on each call (available memory changes).
MemoryInfo memory_info();

// Detection entry point without caching; exposed for tests.
CpuInfo detect_cpu_info();

}  // namespace dynacore

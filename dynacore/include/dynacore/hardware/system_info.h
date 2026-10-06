#pragma once

// Operating system and accelerator detection for diagnostics and device
// selection. Detection never requires a vendor SDK at build time: GPUs are
// probed through the driver library at run time (CUDA driver API via
// nvcuda.dll / libcuda.so.1), so a CPU-only build still reports the GPU it
// could use if the CUDA device were compiled in.

#include <cstdint>
#include <string>
#include <vector>

namespace dynacore {

struct OsInfo {
  std::string name;     // "Windows", "Linux", "macOS"
  std::string version;  // "10.0.26200", "Ubuntu 24.04.1 LTS", "14.5"
  std::string arch;     // "x86_64", "arm64"
};

// Detected once and cached; thread-safe.
const OsInfo& os_info();

struct GpuInfo {
  std::string vendor;  // "NVIDIA"
  std::string name;
  int64_t memory_bytes = 0;
  std::string driver;  // driver API version, e.g. "12.4"
};

// GPUs visible through installed drivers (empty when none or no driver).
// Probed once and cached; thread-safe.
const std::vector<GpuInfo>& gpu_info();

}  // namespace dynacore

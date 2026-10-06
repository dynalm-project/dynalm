#include "dynacore/hardware/system_info.h"

#include <cstdio>
#include <cstring>

#include "dynacore/base/platform.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define DYNACORE_CU_API __stdcall
#else
#include <dlfcn.h>
#include <sys/utsname.h>
#define DYNACORE_CU_API
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <fstream>

namespace dynacore {
namespace {

std::string machine_arch() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#else
  return "unknown";
#endif
}

OsInfo detect_os() {
  OsInfo os;
  os.arch = machine_arch();
#if defined(_WIN32)
  os.name = "Windows";
  // GetVersionEx lies without a manifest entry; RtlGetVersion does not.
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
    auto fn = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")));
    OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof(v);
    if (fn != nullptr && fn(&v) == 0) {
      os.version = std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) + "." +
                   std::to_string(v.dwBuildNumber);
    }
  }
#elif defined(__APPLE__)
  os.name = "macOS";
  char buf[64] = {};
  size_t len = sizeof(buf);
  if (sysctlbyname("kern.osproductversion", buf, &len, nullptr, 0) == 0) os.version = buf;
#else
  os.name = "Linux";
  std::ifstream f("/etc/os-release");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("PRETTY_NAME=", 0) == 0) {
      os.version = line.substr(12);
      if (os.version.size() >= 2 && os.version.front() == '"') os.version = os.version.substr(1, os.version.size() - 2);
    }
  }
  utsname u{};
  if (uname(&u) == 0) os.version += (os.version.empty() ? "" : ", kernel ") + std::string(u.release);
#endif
  return os;
}

// CUDA driver API, resolved at run time (no SDK, no link dependency).
using CuInit = int(DYNACORE_CU_API*)(unsigned);
using CuDriverGetVersion = int(DYNACORE_CU_API*)(int*);
using CuDeviceGetCount = int(DYNACORE_CU_API*)(int*);
using CuDeviceGet = int(DYNACORE_CU_API*)(int*, int);
using CuDeviceGetName = int(DYNACORE_CU_API*)(char*, int, int);
using CuDeviceTotalMem = int(DYNACORE_CU_API*)(size_t*, int);

void* open_cuda_driver() {
#if defined(_WIN32)
  return reinterpret_cast<void*>(LoadLibraryA("nvcuda.dll"));
#elif defined(__APPLE__)
  return nullptr;
#else
  return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

std::vector<GpuInfo> detect_gpus() {
  std::vector<GpuInfo> gpus;
  void* lib = open_cuda_driver();
  if (lib == nullptr) return gpus;
  // The library stays loaded for the life of the process (closing a CUDA
  // driver mid-process is unsafe).
  auto init = reinterpret_cast<CuInit>(symbol(lib, "cuInit"));
  auto drv = reinterpret_cast<CuDriverGetVersion>(symbol(lib, "cuDriverGetVersion"));
  auto count = reinterpret_cast<CuDeviceGetCount>(symbol(lib, "cuDeviceGetCount"));
  auto get = reinterpret_cast<CuDeviceGet>(symbol(lib, "cuDeviceGet"));
  auto name = reinterpret_cast<CuDeviceGetName>(symbol(lib, "cuDeviceGetName"));
  auto mem = reinterpret_cast<CuDeviceTotalMem>(symbol(lib, "cuDeviceTotalMem_v2"));
  if (!init || !count || !get || !name || init(0) != 0) return gpus;
  int version = 0;
  if (drv) drv(&version);
  int n = 0;
  if (count(&n) != 0) return gpus;
  for (int i = 0; i < n; ++i) {
    int dev = 0;
    if (get(&dev, i) != 0) continue;
    GpuInfo g;
    g.vendor = "NVIDIA";
    char buf[256] = {};
    if (name(buf, sizeof(buf) - 1, dev) == 0) g.name = buf;
    size_t bytes = 0;
    if (mem && mem(&bytes, dev) == 0) g.memory_bytes = static_cast<int64_t>(bytes);
    if (version > 0) g.driver = std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
    gpus.push_back(std::move(g));
  }
  return gpus;
}

}  // namespace

const OsInfo& os_info() {
  static const OsInfo info = detect_os();
  return info;
}

const std::vector<GpuInfo>& gpu_info() {
  static const std::vector<GpuInfo> gpus = detect_gpus();
  return gpus;
}

}  // namespace dynacore

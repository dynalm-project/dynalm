#include "dynacore/hardware/process_stats.h"

#include "dynacore/base/platform.h"

#if ENGINE_OS_WINDOWS
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#if ENGINE_OS_MACOS
#include <mach/mach.h>
#endif

#include <fstream>
#include <string>
#endif

namespace engine {

#if ENGINE_OS_WINDOWS

int64_t process_rss_bytes() {
  PROCESS_MEMORY_COUNTERS pmc{};
  return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) ? static_cast<int64_t>(pmc.WorkingSetSize) : 0;
}

int64_t process_peak_rss_bytes() {
  PROCESS_MEMORY_COUNTERS pmc{};
  return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) ? static_cast<int64_t>(pmc.PeakWorkingSetSize)
                                                                       : 0;
}

double process_cpu_seconds() {
  FILETIME create, exit, kernel, user;
  if (!GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) return 0;
  auto to_s = [](const FILETIME& ft) {
    return static_cast<double>((static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) * 1e-7;
  };
  return to_s(kernel) + to_s(user);
}

#else

#if !ENGINE_OS_MACOS
namespace {
int64_t status_kb(const char* key) {
  std::ifstream in("/proc/self/status");
  std::string k;
  int64_t v = 0;
  std::string unit;
  while (in >> k) {
    if (k == key) {
      in >> v;
      return v * 1024;
    }
    std::getline(in, unit);
  }
  return 0;
}
}  // namespace
#endif

#if ENGINE_OS_MACOS
int64_t process_rss_bytes() {
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<int64_t>(info.resident_size);
}
int64_t process_peak_rss_bytes() {
  rusage ru{};
  return getrusage(RUSAGE_SELF, &ru) == 0 ? static_cast<int64_t>(ru.ru_maxrss) : 0;  // bytes on macOS
}
#else
int64_t process_rss_bytes() { return status_kb("VmRSS:"); }
int64_t process_peak_rss_bytes() { return status_kb("VmHWM:"); }
#endif

double process_cpu_seconds() {
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
  return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) +
         static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
}

#endif

}  // namespace engine

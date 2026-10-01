#include "platform/process_stats.h"

#include "common/platform.h"

#if ENGINE_OS_WINDOWS
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>

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

int64_t process_rss_bytes() { return status_kb("VmRSS:"); }
int64_t process_peak_rss_bytes() { return status_kb("VmHWM:"); }

double process_cpu_seconds() {
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
  return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) +
         static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
}

#endif

}  // namespace engine

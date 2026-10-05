#include "platform/perf_counters.h"

#include <vector>

#include "common/platform.h"

#if ENGINE_OS_WINDOWS
#include <windows.h>
#include <powerbase.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
#if ENGINE_OS_LINUX
#include <dirent.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#endif

namespace engine {

PerfSample PerfSample::delta(const PerfSample& a, const PerfSample& b) {
  auto d = [](int64_t x, int64_t y) { return x < 0 || y < 0 ? int64_t{-1} : y - x; };
  PerfSample r;
  r.cycles = d(a.cycles, b.cycles);
  r.instructions = d(a.instructions, b.instructions);
  r.cache_references = d(a.cache_references, b.cache_references);
  r.cache_misses = d(a.cache_misses, b.cache_misses);
  r.branch_misses = d(a.branch_misses, b.branch_misses);
  r.l1d_misses = d(a.l1d_misses, b.l1d_misses);
  r.cpu_migrations = d(a.cpu_migrations, b.cpu_migrations);
  r.context_switches = d(a.context_switches, b.context_switches);
  r.page_faults = d(a.page_faults, b.page_faults);
  r.major_faults = d(a.major_faults, b.major_faults);
  return r;
}

#if ENGINE_OS_LINUX

namespace {

// The counters in PerfSample that come from perf_event_open, in order.
enum Event { kCycles, kInstructions, kCacheRefs, kCacheMisses, kBranchMisses, kL1dMisses, kMigrations, kNumEvents };

struct EventSpec {
  uint32_t type;
  uint64_t config;
};

constexpr EventSpec kSpecs[kNumEvents] = {
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
    {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                             (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
};

int open_event(const EventSpec& s, pid_t tid) {
  perf_event_attr a;
  std::memset(&a, 0, sizeof a);
  a.size = sizeof a;
  a.type = s.type;
  a.config = s.config;
  a.exclude_kernel = 1;  // allowed at perf_event_paranoid <= 2
  a.exclude_hv = 1;
  a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
  return static_cast<int>(syscall(SYS_perf_event_open, &a, tid, -1, -1, 0));
}

std::vector<pid_t> process_threads() {
  std::vector<pid_t> tids;
  if (DIR* d = opendir("/proc/self/task")) {
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] != '.') tids.push_back(static_cast<pid_t>(std::atol(e->d_name)));
    }
    closedir(d);
  }
  return tids;
}

}  // namespace

struct PerfCounters::Impl {
  std::vector<int> fds[kNumEvents];  // one per thread, per event
  ~Impl() {
    for (auto& v : fds) {
      for (int fd : v) close(fd);
    }
  }
  // Sum over threads, scaled for multiplexing; -1 if no thread could open it.
  int64_t total(Event e) const {
    if (fds[e].empty()) return -1;
    double sum = 0;
    for (int fd : fds[e]) {
      uint64_t v[3] = {0, 0, 0};  // value, time enabled, time running
      if (::read(fd, v, sizeof v) != static_cast<ssize_t>(sizeof v)) continue;
      sum += v[2] > 0 && v[2] < v[1] ? static_cast<double>(v[0]) * static_cast<double>(v[1]) / static_cast<double>(v[2])
                                     : static_cast<double>(v[0]);
    }
    return static_cast<int64_t>(sum);
  }
};

std::unique_ptr<PerfCounters> PerfCounters::open() {
  std::unique_ptr<PerfCounters> pc(new PerfCounters());
  pc->impl_ = std::make_unique<Impl>();
  int first_errno = 0;
  for (pid_t tid : process_threads()) {
    for (int e = 0; e < kNumEvents; ++e) {
      const int fd = open_event(kSpecs[e], tid);
      if (fd >= 0) {
        pc->impl_->fds[e].push_back(fd);
      } else if (first_errno == 0 && kSpecs[e].type != PERF_TYPE_SOFTWARE) {
        first_errno = errno;
      }
    }
  }
  pc->hw_available_ = !pc->impl_->fds[kCycles].empty() && !pc->impl_->fds[kInstructions].empty();
  if (!pc->hw_available_) {
    if (first_errno == ENOENT || first_errno == EOPNOTSUPP || first_errno == ENODEV) {
      pc->hw_reason_ = "no hardware PMU exposed to this OS (e.g. a VM such as WSL2)";
    } else if (first_errno == EACCES || first_errno == EPERM) {
      pc->hw_reason_ = "perf_event_paranoid or a seccomp policy forbids perf_event_open";
    } else {
      pc->hw_reason_ = std::string("perf_event_open failed: ") + std::strerror(first_errno);
    }
  }
  return pc;
}

PerfCounters::~PerfCounters() = default;

PerfSample PerfCounters::read() const {
  PerfSample s;
  s.cycles = impl_->total(kCycles);
  s.instructions = impl_->total(kInstructions);
  s.cache_references = impl_->total(kCacheRefs);
  s.cache_misses = impl_->total(kCacheMisses);
  s.branch_misses = impl_->total(kBranchMisses);
  s.l1d_misses = impl_->total(kL1dMisses);
  s.cpu_migrations = impl_->total(kMigrations);
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    s.context_switches = ru.ru_nvcsw + ru.ru_nivcsw;
    s.page_faults = ru.ru_minflt + ru.ru_majflt;
    s.major_faults = ru.ru_majflt;
  }
  return s;
}

double cpu_current_mhz() {
  double sum = 0;
  int n = 0;
  for (int cpu = 0; cpu < 4096; ++cpu) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq");
    double khz = 0;
    if (!(f >> khz)) break;  // no cpufreq (VM), or past the last CPU
    sum += khz / 1000.0;
    ++n;
  }
  if (n > 0) return sum / n;
  // VMs often lack cpufreq; /proc/cpuinfo reports the clock the hypervisor shows.
  std::ifstream info("/proc/cpuinfo");
  std::string line;
  while (std::getline(info, line)) {
    if (line.rfind("cpu MHz", 0) == 0) {
      const size_t colon = line.find(':');
      if (colon != std::string::npos) {
        sum += std::atof(line.c_str() + colon + 1);
        ++n;
      }
    }
  }
  return n > 0 ? sum / n : -1;
}

#else  // !Linux

struct PerfCounters::Impl {};

std::unique_ptr<PerfCounters> PerfCounters::open() {
  std::unique_ptr<PerfCounters> pc(new PerfCounters());
  pc->impl_ = std::make_unique<Impl>();
  pc->hw_reason_ = "hardware counters are read only on Linux (perf_event_open)";
  return pc;
}

PerfCounters::~PerfCounters() = default;

PerfSample PerfCounters::read() const {
  PerfSample s;
#if ENGINE_OS_WINDOWS
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) s.page_faults = pmc.PageFaultCount;
#else
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    s.context_switches = ru.ru_nvcsw + ru.ru_nivcsw;
    s.page_faults = ru.ru_minflt + ru.ru_majflt;
    s.major_faults = ru.ru_majflt;
  }
#endif
  return s;
}

double cpu_current_mhz() {
#if ENGINE_OS_WINDOWS
  // CallNtPowerInformation(ProcessorInformation) fills one record per logical CPU.
  struct ProcessorPowerInformation {
    ULONG Number, MaxMhz, CurrentMhz, MhzLimit, MaxIdleState, CurrentIdleState;
  };
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  std::vector<ProcessorPowerInformation> info(si.dwNumberOfProcessors);
  if (CallNtPowerInformation(ProcessorInformation, nullptr, 0, info.data(),
                             static_cast<ULONG>(info.size() * sizeof(ProcessorPowerInformation))) != 0) {
    return -1;
  }
  double sum = 0;
  for (const auto& p : info) sum += p.CurrentMhz;
  return info.empty() ? -1 : sum / static_cast<double>(info.size());
#else
  return -1;
#endif
}

#endif

}  // namespace engine

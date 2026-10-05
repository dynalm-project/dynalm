#include "platform/cpu_info.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "common/platform.h"

#if ENGINE_ARCH_X86_64
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if ENGINE_OS_WINDOWS
#include <windows.h>
#include <vector>
#elif ENGINE_OS_LINUX
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>
#elif ENGINE_OS_MACOS
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace engine {
namespace {

#if ENGINE_ARCH_X86_64

struct CpuidRegs {
  uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidRegs cpuid(uint32_t leaf, uint32_t subleaf = 0) {
  CpuidRegs r;
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<uint32_t>(regs[0]);
  r.ebx = static_cast<uint32_t>(regs[1]);
  r.ecx = static_cast<uint32_t>(regs[2]);
  r.edx = static_cast<uint32_t>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

constexpr bool bit(uint32_t v, int b) { return (v >> b) & 1u; }

void detect_x86(CpuInfo& info) {
  const CpuidRegs l0 = cpuid(0);
  const uint32_t max_leaf = l0.eax;
  char vendor[13] = {};
  std::memcpy(vendor + 0, &l0.ebx, 4);
  std::memcpy(vendor + 4, &l0.edx, 4);
  std::memcpy(vendor + 8, &l0.ecx, 4);
  info.vendor = vendor;

  if (cpuid(0x80000000).eax >= 0x80000004) {
    char brand[49] = {};
    for (uint32_t i = 0; i < 3; ++i) {
      const CpuidRegs r = cpuid(0x80000002 + i);
      std::memcpy(brand + 16 * i + 0, &r.eax, 4);
      std::memcpy(brand + 16 * i + 4, &r.ebx, 4);
      std::memcpy(brand + 16 * i + 8, &r.ecx, 4);
      std::memcpy(brand + 16 * i + 12, &r.edx, 4);
    }
    std::string b = brand;
    const auto first = b.find_first_not_of(' ');
    const auto last = b.find_last_not_of(' ');
    info.brand = first == std::string::npos ? "" : b.substr(first, last - first + 1);
  }

  if (max_leaf < 1) return;
  const CpuidRegs l1 = cpuid(1);
  CpuFeatures& f = info.features;
  f.sse42 = bit(l1.ecx, 20);

  // The OS must enable the register state (XCR0) for AVX / AVX-512 / AMX,
  // otherwise executing those instructions faults even if cpuid reports them.
  const bool osxsave = bit(l1.ecx, 27);
  const uint64_t xcr0 = osxsave ? xgetbv0() : 0;
  const bool os_avx = (xcr0 & 0x6) == 0x6;                   // XMM | YMM
  const bool os_avx512 = os_avx && (xcr0 & 0xE0) == 0xE0;    // opmask | ZMM_hi256 | hi16_ZMM
  const bool os_amx = (xcr0 & 0x60000) == 0x60000;           // XTILECFG | XTILEDATA
  // Linux additionally requires arch_prctl(ARCH_REQ_XCOMP_PERM) before AMX
  // use; that request belongs in the AMX backend, not in detection.

  f.avx = os_avx && bit(l1.ecx, 28);
  f.fma = os_avx && bit(l1.ecx, 12);
  f.f16c = os_avx && bit(l1.ecx, 29);

  if (max_leaf >= 7) {
    const CpuidRegs l7 = cpuid(7, 0);
    f.avx2 = os_avx && bit(l7.ebx, 5);
    f.avx512f = os_avx512 && bit(l7.ebx, 16);
    f.avx512dq = os_avx512 && bit(l7.ebx, 17);
    f.avx512bw = os_avx512 && bit(l7.ebx, 30);
    f.avx512vl = os_avx512 && bit(l7.ebx, 31);
    f.avx512_vnni = os_avx512 && bit(l7.ecx, 11);
    f.amx_bf16 = os_amx && bit(l7.edx, 22);
    f.amx_tile = os_amx && bit(l7.edx, 24);
    f.amx_int8 = os_amx && bit(l7.edx, 25);
    if (l7.eax >= 1) {
      const CpuidRegs l71 = cpuid(7, 1);
      f.avx_vnni = os_avx && bit(l71.eax, 4);
      f.avx512_bf16 = os_avx512 && bit(l71.eax, 5);
    }
  }
}

#endif  // ENGINE_ARCH_X86_64

#if ENGINE_OS_WINDOWS

void detect_topology_windows(CpuInfo& info) {
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
  if (len == 0) return;
  std::vector<unsigned char> buf(len);
  auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationAll, base, &len)) return;

  int physical = 0, logical = 0;
  BYTE max_eff = 0, min_eff = 0xFF;
  std::vector<std::pair<BYTE, int>> cores;  // (efficiency class, logical count)
  std::vector<std::pair<BYTE, int>> first;  // (efficiency class, first logical CPU in group 0)
  for (DWORD off = 0; off < len;) {
    auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
    if (e->Relationship == RelationProcessorCore) {
      ++physical;
      int threads = 0;
      for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
        threads += static_cast<int>(__popcnt64(e->Processor.GroupMask[g].Mask));
      }
      logical += threads;
      const BYTE eff = e->Processor.EfficiencyClass;
      max_eff = std::max(max_eff, eff);
      min_eff = std::min(min_eff, eff);
      cores.emplace_back(eff, threads);
      if (e->Processor.GroupCount >= 1 && e->Processor.GroupMask[0].Group == 0 && e->Processor.GroupMask[0].Mask) {
        unsigned long bit = 0;
        _BitScanForward64(&bit, e->Processor.GroupMask[0].Mask);
        first.emplace_back(eff, static_cast<int>(bit));
      }
    } else if (e->Relationship == RelationCache) {
      const CACHE_RELATIONSHIP& c = e->Cache;
      if (c.Level == 1 && c.Type == CacheData && info.l1d_bytes == 0) info.l1d_bytes = c.CacheSize;
      if (c.Level == 2 && info.l2_bytes == 0) info.l2_bytes = c.CacheSize;
      if (c.Level == 3) info.l3_bytes = std::max<int64_t>(info.l3_bytes, c.CacheSize);
    }
    off += e->Size;
  }
  info.physical_cores = physical;
  info.logical_cores = logical;
  std::stable_sort(first.begin(), first.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (const auto& f : first) info.core_first_cpu.push_back(f.second);
  // EfficiencyClass: higher = more performant. Only meaningful when classes differ.
  if (max_eff != min_eff) {
    for (const auto& [eff, threads] : cores) {
      (eff == max_eff ? info.performance_cores : info.efficiency_cores) += 1;
    }
  }
}

#elif ENGINE_OS_LINUX

std::string read_first_line(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  return line;
}

// Parses a sysfs cpu list like "0-3,8,10-11" into a count.
int count_cpu_list(const std::string& list) {
  int count = 0;
  std::stringstream ss(list);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) continue;
    const auto dash = part.find('-');
    if (dash == std::string::npos) {
      ++count;
    } else {
      count += std::stoi(part.substr(dash + 1)) - std::stoi(part.substr(0, dash)) + 1;
    }
  }
  return count;
}

int64_t parse_cache_size(const std::string& s) {
  if (s.empty()) return 0;
  int64_t v = std::stoll(s);
  if (s.back() == 'K') v *= 1024;
  if (s.back() == 'M') v *= 1024 * 1024;
  return v;
}

void detect_topology_linux(CpuInfo& info) {
  info.logical_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  std::set<std::pair<int, int>> cores;  // (package, core)
  std::map<std::pair<int, int>, int> first;  // core -> first logical CPU
  for (int cpu = 0; cpu < info.logical_cores; ++cpu) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const std::string pkg = read_first_line(base + "physical_package_id");
    const std::string core = read_first_line(base + "core_id");
    if (pkg.empty() || core.empty()) continue;
    cores.emplace(std::stoi(pkg), std::stoi(core));
    first.emplace(std::make_pair(std::stoi(pkg), std::stoi(core)), cpu);
  }
  info.physical_cores = cores.empty() ? info.logical_cores : static_cast<int>(cores.size());

  // Intel hybrid: separate PMUs list their logical CPUs. P-cores have 2
  // threads with HT; E-cores have 1. Report core counts, not thread counts.
  const std::string pcpus = read_first_line("/sys/devices/cpu_core/cpus");
  const std::string ecpus = read_first_line("/sys/devices/cpu_atom/cpus");
  if (!pcpus.empty() && !ecpus.empty()) {
    info.efficiency_cores = count_cpu_list(ecpus);
    info.performance_cores = info.physical_cores - info.efficiency_cores;
  }
  // P-cores (hyper-threaded: a sibling list with two CPUs) first.
  std::vector<std::pair<int, int>> order;  // (is_e_core, cpu)
  for (const auto& [core, cpu] : first) {
    const std::string sib =
        read_first_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    order.emplace_back(info.performance_cores > 0 && count_cpu_list(sib) < 2 ? 1 : 0, cpu);
  }
  std::stable_sort(order.begin(), order.end());
  for (const auto& o : order) info.core_first_cpu.push_back(o.second);

  for (int idx = 0; idx < 8; ++idx) {
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/";
    const std::string level = read_first_line(base + "level");
    if (level.empty()) break;
    const std::string type = read_first_line(base + "type");
    const int64_t size = parse_cache_size(read_first_line(base + "size"));
    if (level == "1" && type == "Data") info.l1d_bytes = size;
    if (level == "2") info.l2_bytes = size;
    if (level == "3") info.l3_bytes = size;
  }
}

#endif

#if ENGINE_OS_MACOS

int64_t sysctl_int(const char* name) {
  int64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
  if (len == sizeof(int32_t)) return static_cast<int64_t>(*reinterpret_cast<int32_t*>(&v));
  return v;
}

// hw.perflevel0 = performance cores, hw.perflevel1 = efficiency cores
// (Apple Silicon; absent on Intel Macs).
void detect_topology_macos(CpuInfo& info) {
  info.physical_cores = static_cast<int>(sysctl_int("hw.physicalcpu"));
  info.logical_cores = static_cast<int>(sysctl_int("hw.logicalcpu"));
  info.performance_cores = static_cast<int>(sysctl_int("hw.perflevel0.physicalcpu"));
  info.efficiency_cores = static_cast<int>(sysctl_int("hw.perflevel1.physicalcpu"));
  info.l1d_bytes = sysctl_int("hw.l1dcachesize");
  info.l2_bytes = sysctl_int("hw.l2cachesize");
  info.l3_bytes = sysctl_int("hw.l3cachesize");
  char brand[256] = {};
  size_t len = sizeof(brand);
  if (info.brand.empty() && sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) {
    info.brand = brand;
  }
  if (info.vendor == "ARM") info.vendor = "Apple";
}

#endif

}  // namespace

CpuInfo detect_cpu_info() {
  CpuInfo info;
#if ENGINE_ARCH_X86_64
  detect_x86(info);
#elif ENGINE_ARCH_ARM64
  info.vendor = "ARM";
  info.features.neon = true;  // mandatory on AArch64
#if defined(__ARM_FEATURE_DOTPROD)
  info.features.arm_dotprod = true;
#endif
#endif

#if ENGINE_OS_WINDOWS
  detect_topology_windows(info);
#elif ENGINE_OS_LINUX
  detect_topology_linux(info);
#elif ENGINE_OS_MACOS
  detect_topology_macos(info);
#endif

  if (info.logical_cores <= 0) {
    info.logical_cores = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  if (info.physical_cores <= 0) info.physical_cores = info.logical_cores;
  return info;
}

const CpuInfo& cpu_info() {
  static const CpuInfo info = detect_cpu_info();
  return info;
}

MemoryInfo memory_info() {
  MemoryInfo m;
#if ENGINE_OS_WINDOWS
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (GlobalMemoryStatusEx(&st)) {
    m.total_bytes = static_cast<int64_t>(st.ullTotalPhys);
    m.available_bytes = static_cast<int64_t>(st.ullAvailPhys);
  }
#elif ENGINE_OS_LINUX
  std::ifstream in("/proc/meminfo");
  std::string key;
  int64_t value_kb = 0;
  std::string unit;
  while (in >> key >> value_kb >> unit) {
    if (key == "MemTotal:") m.total_bytes = value_kb * 1024;
    if (key == "MemAvailable:") m.available_bytes = value_kb * 1024;
  }
#elif ENGINE_OS_MACOS
  m.total_bytes = sysctl_int("hw.memsize");
  // Available ~ free + inactive (reclaimable) pages, like Activity Monitor.
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) ==
      KERN_SUCCESS) {
    const auto page = static_cast<int64_t>(vm_kernel_page_size);
    m.available_bytes = (static_cast<int64_t>(vm.free_count) + static_cast<int64_t>(vm.inactive_count)) * page;
  }
#endif
  return m;
}

}  // namespace engine

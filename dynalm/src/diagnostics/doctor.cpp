#include "diagnostics/doctor.h"

#include <cstdio>
#include <filesystem>

#include "api/json.h"
#include "common/version.h"
#include "config/config.h"
#include "dynacore/device/device_registry.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/system_info.h"
#include "dynacore/version.h"
#include "registry/model_registry.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dynalm {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

#if defined(_WIN32)
// Smart App Control in enforce mode blocks freshly built or downloaded
// unsigned executables ("An Application Control policy has blocked this file").
bool smart_app_control_enforced() {
  DWORD value = 0, size = sizeof(value);
  const LSTATUS rc = RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\CI\\Policy",
                                  L"VerifiedAndReputablePolicyState", RRF_RT_REG_DWORD, nullptr, &value, &size);
  return rc == ERROR_SUCCESS && value == 1;
}
#endif

int32_t count_local_models() {
  namespace fs = std::filesystem;
  int32_t n = 0;
  std::error_code ec;
  for (const std::string& dir : model_search_dirs()) {
    for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      if (it.depth() > 2) it.disable_recursion_pending();
      if (it->is_regular_file(ec) && it->path().extension() == ".gguf") ++n;
    }
  }
  return n;
}

std::string gib(int64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.1f GiB", static_cast<double>(bytes) / kGiB);
  return buf;
}

std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string s;
  for (const std::string& x : v) s += (s.empty() ? "" : sep) + x;
  return s;
}

}  // namespace

DoctorReport collect_doctor_report() {
  DoctorReport r;
  r.dynalm_version = ENGINE_VERSION_STRING;
  r.dynacore_version = dynacore::kVersionString;
  r.build_type = ENGINE_BUILD_TYPE;
  r.compiler = std::string(ENGINE_COMPILER_ID) + " " + ENGINE_COMPILER_VERSION;
  for (CpuIsa isa : {CpuIsa::kGeneric, CpuIsa::kAvx2, CpuIsa::kAvx512, CpuIsa::kAmx, CpuIsa::kNeon}) {
    if (isa_compiled(isa)) r.compiled_isas.emplace_back(isa_name(isa));
  }
  for (DeviceKind k : compiled_devices()) r.compiled_devices.emplace_back(device_kind_name(k));

  const OsInfo& os = os_info();
  r.os = os.name;
  r.os_version = os.version;
  r.arch = os.arch;

  const CpuInfo& cpu = cpu_info();
  r.cpu = cpu.brand;
  r.cpu_vendor = cpu.vendor;
  r.physical_cores = cpu.physical_cores;
  r.logical_cores = cpu.logical_cores;
  r.performance_cores = cpu.performance_cores;
  r.efficiency_cores = cpu.efficiency_cores;
  r.l2_bytes = cpu.l2_bytes;
  r.l3_bytes = cpu.l3_bytes;
  const CpuFeatures& f = cpu.features;
  if (os.arch == "arm64") {
    r.isa = {{"NEON", f.neon}, {"DOTPROD", f.arm_dotprod}};
  } else {
    r.isa = {{"AVX2", f.avx2},           {"FMA", f.fma},       {"F16C", f.f16c},
             {"AVX-VNNI", f.avx_vnni},   {"AVX512", f.avx512f}, {"AVX512-VNNI", f.avx512_vnni},
             {"AMX", f.amx_tile && f.amx_int8}};
  }

  const MemoryInfo mem = memory_info();
  r.ram_total = mem.total_bytes;
  r.ram_available = mem.available_bytes;

  for (const GpuInfo& g : gpu_info()) r.gpus.push_back({g.vendor + " " + g.name, g.memory_bytes, g.driver});

  const CpuIsa best = select_best_isa(f);
  r.device = "CPU";
  r.kernel = std::string(isa_name(best));
  r.threads = cpu.physical_cores > 0 ? cpu.physical_cores : 1;
  r.model_store = models_dir();
  r.local_models = count_local_models();
  if (std::string path = default_config_path(); std::filesystem::exists(path)) r.config_file = path;

  // Warnings: only things a user can act on.
  if (r.ram_available > 0 && r.ram_available < static_cast<int64_t>(2 * kGiB)) {
    r.warnings.push_back("only " + gib(r.ram_available) +
                         " RAM available: models above ~1.5 GB will page from disk and run slowly. "
                         "Close other programs or use a smaller model (dynalm models --available).");
  }
  if (r.ram_available > 0 && r.ram_available < static_cast<int64_t>(kGiB / 2)) r.ready = false;
  if (best == CpuIsa::kGeneric && (f.avx2 || f.neon)) {
    r.warnings.push_back("this CPU has SIMD support the binary was not built with; rebuild with ENABLE_AVX2/NEON "
                         "for several times the speed.");
  }
  if (!r.gpus.empty()) {
    r.warnings.push_back("an NVIDIA GPU is present, but this build has no CUDA device (DynaCore CUDA backend is a "
                         "later milestone); DynaLM runs on the CPU.");
  }
  if (r.local_models == 0) r.warnings.push_back("no local models yet: dynalm pull qwen3:4b");
#if defined(_WIN32)
  if (smart_app_control_enforced()) {
    r.warnings.push_back("Windows Smart App Control is on: it can block unsigned, freshly built dynalm binaries "
                         "(\"An Application Control policy has blocked this file\"). Use a signed release.");
  }
#endif
  return r;
}

std::string format_doctor_text(const DoctorReport& r) {
  std::string s;
  char buf[512];
  auto line = [&](const char* label, const std::string& value) {
    std::snprintf(buf, sizeof buf, "  %-10s %s\n", label, value.c_str());
    s += buf;
  };
  s += "DynaLM System Report\n";
  s += "--------------------\n";
  line("DynaLM", r.dynalm_version + " (" + r.build_type + ", " + r.compiler + ")");
  line("DynaCore", r.dynacore_version + " (kernels: " + join(r.compiled_isas, " ") + "; devices: " +
                       join(r.compiled_devices, " ") + ")");
  line("OS", r.os + " " + r.os_version + " (" + r.arch + ")");
  std::string cores = std::to_string(r.physical_cores) + " cores, " + std::to_string(r.logical_cores) + " threads";
  if (r.performance_cores > 0) {
    cores += " (" + std::to_string(r.performance_cores) + " P-cores + " + std::to_string(r.efficiency_cores) +
             " E-cores)";
  }
  line("CPU", r.cpu);
  line("", cores);
  std::snprintf(buf, sizeof buf, "L2 %lld KiB, L3 %lld KiB", static_cast<long long>(r.l2_bytes / 1024),
                static_cast<long long>(r.l3_bytes / 1024));
  line("Caches", buf);
  line("RAM", gib(r.ram_total) + " total, " + gib(r.ram_available) + " available");
  std::string isa;
  for (const auto& [name, on] : r.isa) isa += name + (on ? " yes  " : " no  ");
  line("ISA", isa);
  if (r.gpus.empty()) {
    line("GPU", "not detected");
  } else {
    for (const DoctorReport::Gpu& g : r.gpus) {
      line("GPU", g.name + ", " + gib(g.memory_bytes) + (g.driver.empty() ? "" : ", CUDA driver " + g.driver));
    }
  }
  line("CUDA", "not built into this binary");
  line("Device", r.device + " / " + r.kernel + ", " + std::to_string(r.threads) + " threads");
  line("Models", std::to_string(r.local_models) + " local, store " + r.model_store);
  line("Config", r.config_file.empty() ? "none (defaults; dynalm config init)" : r.config_file);
  for (const std::string& w : r.warnings) s += "  warning:   " + w + "\n";
  line("Status", r.ready ? (r.warnings.empty() ? "Ready" : "Ready, with warnings") : "Not ready");
  return s;
}

std::string format_doctor_json(const DoctorReport& r) {
  json::Object isa;
  for (const auto& [name, on] : r.isa) isa[name] = on;
  json::Array gpus;
  for (const DoctorReport::Gpu& g : r.gpus) {
    gpus.emplace_back(json::Object{{"name", g.name}, {"memory_bytes", g.memory_bytes}, {"driver", g.driver}});
  }
  auto strings = [](const std::vector<std::string>& v) {
    json::Array a;
    for (const std::string& x : v) a.emplace_back(x);
    return a;
  };
  json::Object o{
      {"dynalm_version", r.dynalm_version},
      {"dynacore_version", r.dynacore_version},
      {"build_type", r.build_type},
      {"compiler", r.compiler},
      {"compiled_isas", strings(r.compiled_isas)},
      {"compiled_devices", strings(r.compiled_devices)},
      {"os", json::Object{{"name", r.os}, {"version", r.os_version}, {"arch", r.arch}}},
      {"cpu", json::Object{{"name", r.cpu},
                           {"vendor", r.cpu_vendor},
                           {"physical_cores", r.physical_cores},
                           {"logical_cores", r.logical_cores},
                           {"performance_cores", r.performance_cores},
                           {"efficiency_cores", r.efficiency_cores},
                           {"l2_bytes", r.l2_bytes},
                           {"l3_bytes", r.l3_bytes},
                           {"isa", isa}}},
      {"ram", json::Object{{"total_bytes", r.ram_total}, {"available_bytes", r.ram_available}}},
      {"gpus", gpus},
      {"device", r.device},
      {"kernel", r.kernel},
      {"threads", r.threads},
      {"model_store", r.model_store},
      {"local_models", r.local_models},
      {"config_file", r.config_file},
      {"warnings", strings(r.warnings)},
      {"ready", r.ready},
  };
  return json::dump(json::Value(std::move(o))) + "\n";
}

}  // namespace dynalm

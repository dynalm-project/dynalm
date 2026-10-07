#include "diagnostics/doctor.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

#include "api/json.h"
#include "common/version.h"
#include "config/config.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/device/device_registry.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/system_info.h"
#include "dynacore/quantization/quant_formats.h"
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

TensorView view2(void* p, DType d, int64_t r, int64_t c) {
  return TensorView(p, *TensorLayout::contiguous(d, TensorShape{r, c}));
}

// A few kernels on the selected ISA against the generic kernels: a Q8_0
// matmul (int8 and fp32 paths) and an RMSNorm. Catches a broken or
// mis-detected SIMD tier on a fresh install in milliseconds.
bool kernel_self_test(CpuIsa isa, std::string& detail) {
  ThreadPool pool(2);
  CpuDevice fast(pool, isa), ref(pool, CpuIsa::kGeneric);
  // Same arithmetic on both sides (fp32 activations), so only summation order
  // differs; int8 activations (DD-053) are checked separately for finiteness.
  KernelPlan fp32 = KernelPlan::defaults();
  fp32.int8_decode_max_rows = 0;
  fast.set_kernel_plan(fp32);
  ref.set_kernel_plan(fp32);
  constexpr int64_t k = 256, n = 64;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::vector<float> w(static_cast<size_t>(n * k));
  for (float& v : w) v = nd(rng);
  std::vector<quant::BlockQ8_0> wq(static_cast<size_t>(n * k / quant::kQK));
  quant::quantize_q8_0(w.data(), wq.data(), static_cast<int64_t>(wq.size()));
  const TensorView wv = view2(wq.data(), DType::kQ8_0, n, k);
  double worst = 0;
  for (int64_t m : {1, 8}) {  // int8 decode path, then the fp32 GEMM path
    std::vector<float> x(static_cast<size_t>(m * k)), a(static_cast<size_t>(m * n)), b(a.size());
    for (float& v : x) v = nd(rng);
    fast.matmul(view2(x.data(), DType::kF32, m, k), wv, nullptr, view2(a.data(), DType::kF32, m, n));
    ref.matmul(view2(x.data(), DType::kF32, m, k), wv, nullptr, view2(b.data(), DType::kF32, m, n));
    for (size_t i = 0; i < a.size(); ++i) {
      if (!std::isfinite(a[i])) {
        detail = "non-finite matmul output";
        return false;
      }
      worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / (1.0 + std::fabs(b[i])));
    }
  }
  std::vector<float> x(k), g(k, 1.0f), a(k), b(k);
  for (float& v : x) v = nd(rng);
  fast.rms_norm(view2(x.data(), DType::kF32, 1, k), TensorView(g.data(), *TensorLayout::contiguous(DType::kF32, TensorShape{k})),
                1e-6f, view2(a.data(), DType::kF32, 1, k));
  ref.rms_norm(view2(x.data(), DType::kF32, 1, k), TensorView(g.data(), *TensorLayout::contiguous(DType::kF32, TensorShape{k})),
               1e-6f, view2(b.data(), DType::kF32, 1, k));
  for (int64_t i = 0; i < k; ++i) worst = std::max(worst, std::fabs(static_cast<double>(a[static_cast<size_t>(i)]) - b[static_cast<size_t>(i)]));
  bool finite = true;
  {
    std::vector<float> x1(k), y1(n);
    for (float& v : x1) v = nd(rng);
    fast.set_kernel_plan(KernelPlan::defaults());
    fast.matmul(view2(x1.data(), DType::kF32, 1, k), wv, nullptr, view2(y1.data(), DType::kF32, 1, n));
    for (float v : y1) finite = finite && std::isfinite(v);
  }
  const bool ok = worst < 1e-3 && finite;
  char buf[96];
  std::snprintf(buf, sizeof buf, "kernel self-test %s vs generic: %s (max rel. diff %.1e)",
                std::string(isa_name(isa)).c_str(), ok ? "passed" : "FAILED", worst);
  detail = buf;
  return ok;
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
  r.cpu = cpu.brand.empty() ? cpu.vendor : cpu.brand;
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
    r.isa = {{"AVX2", f.avx2},         {"FMA", f.fma},          {"F16C", f.f16c},
             {"AVX-VNNI", f.avx_vnni}, {"AVX-512", f.avx512f},  {"AMX", f.amx_tile && f.amx_int8}};
  }

  const MemoryInfo mem = memory_info();
  r.ram_total = mem.total_bytes;
  r.ram_available = mem.available_bytes;
  for (const GpuInfo& g : gpu_info()) r.gpus.push_back({g.vendor + " " + g.name, g.memory_bytes, g.driver});

  const CpuIsa isa = select_isa(f, &r.isa_note);
  r.device = "CPU";
  r.kernel = std::string(isa_name(isa));
  r.threads = cpu.physical_cores > 0 ? cpu.physical_cores : 1;

  r.executable = executable_path();
  if (!r.executable.empty()) {
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(r.executable).parent_path();
    r.dynacorec_found = std::filesystem::exists(dir / "dynacorec", ec) || std::filesystem::exists(dir / "dynacorec.exe", ec);
  }
  r.model_store = models_dir();
  r.local_models = count_local_models();
  if (std::string path = default_config_path(); std::filesystem::exists(path)) r.config_file = path;

  r.dynacore_ok = kernel_self_test(isa, r.dynacore_detail);
  if (!r.dynacore_ok) r.ready = false;

  // Warnings: only things a user can act on.
  if (r.ram_available > 0 && r.ram_available < static_cast<int64_t>(2 * kGiB)) {
    r.warnings.push_back("only " + gib(r.ram_available) +
                         " RAM available: models above ~1.5 GB will page from disk and run slowly. "
                         "Close other programs or pick a smaller model (dynalm models --available).");
  }
  if (r.ram_available > 0 && r.ram_available < static_cast<int64_t>(kGiB / 2)) r.ready = false;
  if (isa == CpuIsa::kGeneric && (f.avx2 || f.neon) && r.isa_note.empty()) {
    r.warnings.push_back("this CPU has SIMD support this build lacks; a release build is several times faster.");
  }
  if (!r.isa_note.empty() && r.isa_note != "forced by DYNACORE_ISA") r.warnings.push_back(r.isa_note);
  if (!r.gpus.empty()) {
    r.warnings.push_back("a GPU is present; this release runs on the CPU only.");
  }
  if (r.local_models == 0) r.warnings.push_back("no local models yet: dynalm pull qwen3:4b");
#if defined(_WIN32)
  if (smart_app_control_enforced()) {
    r.warnings.push_back("Windows Smart App Control is on: it can block unsigned dynalm builds "
                         "(\"An Application Control policy has blocked this file\").");
  }
#endif
  return r;
}

std::string format_doctor_text(const DoctorReport& r) {
  std::string s;
  char buf[640];
  auto line = [&](const char* label, const std::string& value) {
    std::snprintf(buf, sizeof buf, "%-14s %s\n", label, value.c_str());
    s += buf;
  };
  s += "DynaLM Doctor\n\n";
  line("Version:", "DynaLM " + r.dynalm_version + ", DynaCore " + r.dynacore_version + " (" + r.build_type + ", " +
                       r.compiler + ")");
  line("Platform:", r.os + " " + r.os_version);
  line("Architecture:", r.arch);
  std::string cores = std::to_string(r.physical_cores) + " cores, " + std::to_string(r.logical_cores) + " threads";
  if (r.performance_cores > 0) {
    cores += ", " + std::to_string(r.performance_cores) + " P + " + std::to_string(r.efficiency_cores) + " E";
  }
  line("CPU:", r.cpu + " (" + cores + ")");
  std::string have, missing;
  for (const auto& [name, on] : r.isa) (on ? have : missing) += (on ? have : missing).empty() ? name : " " + name;
  line("CPU features:", (have.empty() ? "none" : have) + (missing.empty() ? "" : "  (not available: " + missing + ")"));
  line("Backend:", r.device + " / " + r.kernel + (r.isa_note.empty() ? "" : "  [" + r.isa_note + "]"));
  line("Threads:", std::to_string(r.threads));
  line("Memory:", gib(r.ram_total) + " total, " + gib(r.ram_available) + " available");
  if (r.gpus.empty()) {
    line("GPU:", "none detected (not used by this build)");
  } else {
    for (const DoctorReport::Gpu& g : r.gpus) line("GPU:", g.name + " (not used by this build)");
  }
  line("Installed:", r.executable.empty() ? "unknown" : r.executable);
  line("Compiler:", r.dynacorec_found ? "dynacorec found" : "dynacorec not found next to dynalm");
  line("Models:", std::to_string(r.local_models) + " local, store " + r.model_store);
  line("Config:", r.config_file.empty() ? "none (defaults)" : r.config_file);
  s += "\n";
  line("DynaCore:", std::string(r.dynacore_ok ? "OK" : "FAILED") + " (" + r.dynacore_detail + ")");
  line("Runtime:", r.dynacore_ok ? "OK" : "not usable until the kernel self-test passes");
  if (!r.warnings.empty()) {
    s += "\nWarnings:\n";
    for (const std::string& w : r.warnings) s += "  - " + w + "\n";
  }
  s += "\n";
  line("Status:", r.ready ? "Ready" : "Not ready");
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
      {"isa_note", r.isa_note},
      {"threads", r.threads},
      {"executable", r.executable},
      {"dynacorec_found", r.dynacorec_found},
      {"model_store", r.model_store},
      {"local_models", r.local_models},
      {"config_file", r.config_file},
      {"dynacore_ok", r.dynacore_ok},
      {"dynacore_detail", r.dynacore_detail},
      {"warnings", strings(r.warnings)},
      {"ready", r.ready},
  };
  return json::dump(json::Value(std::move(o))) + "\n";
}

}  // namespace dynalm

#include "runtime/autotune.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include "api/json.h"
#include "common/version.h"
#include "config/config.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/system_info.h"
#include "runtime/engine.h"

namespace dynalm {

std::string_view tune_mode_name(TuneMode m) {
  switch (m) {
    case TuneMode::kAuto: return "auto";
    case TuneMode::kPerformance: return "performance";
    case TuneMode::kExact: return "exact";
    case TuneMode::kSafe: return "safe";
  }
  return "?";
}

Result<TuneMode> parse_tune_mode(std::string_view s) {
  for (TuneMode m : {TuneMode::kAuto, TuneMode::kPerformance, TuneMode::kExact, TuneMode::kSafe}) {
    if (tune_mode_name(m) == s) return m;
  }
  return InvalidArgument("unknown mode '" + std::string(s) + "' (auto, performance, exact, safe)");
}

Status apply_tune_mode(TuneMode m, EngineOptions& o) {
  switch (m) {
    case TuneMode::kAuto:
    case TuneMode::kPerformance:
      return Status::Ok();  // the device plan already picks the fastest validated path
    case TuneMode::kExact:
      // DD-053 everywhere: GPUs (checked per family) and DynaCore's CPU
      // kernels (int8 per-32 where validated), never ggml's 8-bit CPU path.
      if (!o.backend_auto && o.backend == DeviceKind::kGgmlCpu) {
        return InvalidArgument("--mode exact: --backend ggml-cpu is below DD-053 accuracy; use --backend cpu");
      }
      o.cpu_ggml = false;
      return Status::Ok();
    case TuneMode::kSafe:
      // The fp32 reference kernels on the CPU: no GPU, no int8, no ggml. An
      // explicitly requested GPU (--gpu, a named GPU) is an error, never a
      // silent switch to the CPU.
      if (o.backend_require_gpu) return InvalidArgument("--mode safe runs on the CPU only; it cannot be combined with --gpu");
      if (!o.backend_auto && o.backend != DeviceKind::kCpu) {
        return InvalidArgument("--mode safe runs on the CPU's reference kernels; it cannot use --backend " +
                               std::string(device_kind_name(o.backend)));
      }
      o.backend_auto = o.backend_require_gpu = o.cpu_ggml = false;
      o.backend = DeviceKind::kCpu;
      o.int8_decode_rows = 0;
      return Status::Ok();
  }
  return Status::Ok();
}

namespace {

std::string features_string(const dynacore::CpuFeatures& f) {
  std::string s;
  auto add = [&](bool on, const char* name) {
    if (on) s += (s.empty() ? "" : " ") + std::string(name);
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
  add(f.arm_i8mm, "i8mm");
  add(f.arm_fp16, "fp16");
  add(f.arm_bf16, "bf16");
  add(f.arm_sve, "sve");
  add(f.arm_sme, "sme");
  return s.empty() ? "none" : s;
}

// The binary itself: any rebuild (a kernel, a flag, a ggml update) changes its
// size or modification time, and with it the key.
std::string build_stamp() {
  std::error_code ec;
  const std::filesystem::path exe = dynacore::executable_path();
  const auto size = std::filesystem::file_size(exe, ec);
  if (ec) return "unknown";
  const auto mtime = std::filesystem::last_write_time(exe, ec);
  if (ec) return "unknown";
  return std::to_string(static_cast<unsigned long long>(size)) + " bytes @" +
         std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

std::string iso_now() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

}  // namespace

std::string tune_machine_key() {
  const dynacore::CpuInfo& c = dynacore::cpu_info();
  std::ostringstream k;
  k << c.brand << " | " << features_string(c.features) << " | " << c.physical_cores << "c/" << c.performance_cores
    << "p | " << dynacore::os_info().name << " " << dynacore::os_info().arch << " | DynaLM " << ENGINE_VERSION_STRING
    << " " << ENGINE_COMPILER_ID << " " << ENGINE_COMPILER_VERSION << " | ggml " << dynacore::ggml_revision()
    << " | " << ENGINE_BUILD_TYPE << " binary " << build_stamp();
  return k.str();
}

std::string tune_model_key(const ModelConfig& c, std::string_view quantization) {
  return c.architecture + "/" + std::to_string(c.hidden_size) + "x" + std::to_string(c.num_layers) + "/" +
         std::string(quantization);
}

const TuneEntry* TuneCache::find(std::string_view device, std::string_view model) const {
  for (const TuneEntry& e : entries) {
    if (e.device == device && e.model == model) return &e;
  }
  return nullptr;
}

void TuneCache::put(TuneEntry e) {
  for (TuneEntry& x : entries) {
    if (x.device == e.device && x.model == e.model) {
      x = std::move(e);
      return;
    }
  }
  entries.push_back(std::move(e));
}

std::string tune_cache_path() {
  if (const char* p = std::getenv("DYNALM_AUTOTUNE_CACHE"); p != nullptr && *p != '\0') return p;
  return (std::filesystem::path(dynalm_home()) / "autotune.json").string();
}

TuneCache load_tune_cache(const std::string& path, std::string* why) {
  TuneCache c;
  auto reason = [&](std::string w) {
    if (why != nullptr) *why = std::move(w);
    return TuneCache{};
  };
  std::ifstream f(path, std::ios::binary);
  if (!f) return reason("no tuning cache yet (dynalm autotune --run MODEL makes one)");
  std::stringstream ss;
  ss << f.rdbuf();
  auto v = json::parse(ss.str());
  if (!v.ok() || !v->is_object()) return reason("unreadable tuning cache " + path + " (ignored)");
  const json::Value* ver = v->find("version");
  if (ver == nullptr || !ver->is_number() || static_cast<int>(ver->as_number()) != TuneCache::kVersion) {
    return reason("tuning cache of another format version (ignored; run dynalm autotune --run)");
  }
  const json::Value* mk = v->find("machine");
  if (mk == nullptr || !mk->is_string() || mk->as_string() != tune_machine_key()) {
    return reason("tuning cache from another machine or build (ignored; run dynalm autotune --run)");
  }
  c.machine = mk->as_string();
  if (const json::Value* es = v->find("entries"); es != nullptr && es->is_array()) {
    for (const json::Value& e : es->as_array()) {
      if (!e.is_object()) continue;
      TuneEntry t;
      auto str = [&](const char* k, std::string& out) {
        if (const json::Value* x = e.find(k); x != nullptr && x->is_string()) out = x->as_string();
      };
      auto num = [&](const char* k) {
        const json::Value* x = e.find(k);
        return x != nullptr && x->is_number() ? x->as_number() : 0.0;
      };
      str("device", t.device);
      str("model", t.model);
      str("conditions", t.conditions);
      str("measured_at", t.measured_at);
      t.threads = static_cast<int32_t>(num("threads"));
      t.tok_s = num("tok_s");
      t.baseline_tok_s = num("baseline_tok_s");
      if (!t.device.empty() && !t.model.empty() && t.threads > 0) c.entries.push_back(std::move(t));
    }
  }
  if (why != nullptr) *why = std::to_string(c.entries.size()) + " tuned setting(s) for this machine and build";
  return c;
}

Status save_tune_cache(const TuneCache& c, const std::string& path) {
  json::Array es;
  for (const TuneEntry& e : c.entries) {
    json::Object o;
    o["device"] = e.device;
    o["model"] = e.model;
    o["threads"] = static_cast<double>(e.threads);
    o["tok_s"] = e.tok_s;
    o["baseline_tok_s"] = e.baseline_tok_s;
    o["conditions"] = e.conditions;
    o["measured_at"] = e.measured_at.empty() ? iso_now() : e.measured_at;
    es.push_back(json::Value(std::move(o)));
  }
  json::Object root;
  root["version"] = static_cast<double>(TuneCache::kVersion);
  root["machine"] = c.machine.empty() ? tune_machine_key() : c.machine;
  root["entries"] = json::Value(std::move(es));
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return Internal("cannot write " + tmp);
    f << json::dump(json::Value(std::move(root))) << "\n";
    if (!f) return Internal("cannot write " + tmp);
  }
  std::filesystem::rename(tmp, path, ec);  // atomic replace: a reader never sees half a file
  if (ec) return Internal("cannot replace " + path + ": " + ec.message());
  return Status::Ok();
}

int32_t tuned_threads(DeviceKind device, const ModelConfig& c, std::string_view quantization) {
  if (const char* v = std::getenv("DYNALM_AUTOTUNE"); v != nullptr && std::string_view(v) == "0") return 0;
  static std::mutex mu;
  static std::string loaded_path;
  static TuneCache cache;
  const std::string path = tune_cache_path();
  std::lock_guard<std::mutex> lock(mu);
  if (path != loaded_path) {  // read once per process (and again if the path changes)
    cache = load_tune_cache(path);
    loaded_path = path;
  }
  const TuneEntry* e = cache.find(device_kind_name(device), tune_model_key(c, quantization));
  return e != nullptr ? e->threads : 0;
}

TuneConditions judge_tune_conditions(const std::vector<std::string>& notes, const std::vector<std::string>& problems,
                                     bool first_check) {
  TuneConditions c;
  if (!problems.empty()) {
    for (const std::string& s : problems) c.text += (c.text.empty() ? "" : ", ") + s;
    return c;
  }
  if (notes.empty()) {
    c.quiet = !first_check;  // after the run, nothing new to report is fine
    c.text = first_check ? "conditions unknown (power and load cannot be read on this system)" : "no change";
    return c;
  }
  c.quiet = true;
  for (const std::string& s : notes) c.text += (c.text.empty() ? "" : ", ") + s;
  return c;
}

std::string autotune_report() {
  const dynacore::CpuInfo& c = dynacore::cpu_info();
  const dynacore::MemoryInfo mem = dynacore::memory_info();
  std::ostringstream r;
  auto gb = [](int64_t b) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(b) / 1e9);
    return std::string(buf);
  };
  r << "Hardware (detected at run time)\n";
  r << "  CPU:       " << c.brand << ", " << c.physical_cores << " cores";
  if (c.performance_cores > 0) r << " (" << c.performance_cores << " performance + " << c.efficiency_cores << " efficiency)";
  r << "\n  features:  " << features_string(c.features) << "\n";
  r << "  memory:    " << gb(mem.total_bytes) << " total, " << gb(mem.available_bytes) << " available\n";
  std::string gpus;
  for (DeviceKind k : compiled_devices()) {
    if (k != DeviceKind::kCpu) gpus += (gpus.empty() ? "" : ", ") + std::string(device_kind_name(k));
  }
  r << "  GPU backends built in: " << (gpus.empty() ? "none (CPU-only build)" : gpus) << "\n\n";

  const dynacore::CpuIsa best = dynacore::select_isa(c.features);
  const dynacore::CpuKernels k = dynacore::make_cpu_kernels(best);
  const bool ggml_cpu = dynacore::ggml_gpu_compiled(dynacore::DeviceType::kCpu);
  r << "CPU kernels\n";
  r << "  DynaCore (native-cpu): " << dynacore::isa_name(best) << " tier; int8 activations (DD-053): "
    << (k.int8_simd ? "SIMD" : "not accelerated (fp32 kernels)");
  if (best == dynacore::CpuIsa::kNeon) {
    r << (c.features.arm_dotprod ? " [SDOT]" : " [this CPU has no dot-product instructions]");
  }
  r << "; int16 (FFN down, opt-in): " << (k.quantize_act16 != nullptr ? "available" : "not available") << "\n";
  r << "  ggml (cpu, fast path): "
    << (ggml_cpu ? "built in; repacked int8 GEMM/GEMV, performance cores only by default"
                 : "not built in (ENABLE_GGML_CPU=OFF)")
    << "\n\n";

  r << "Execution modes (--mode)\n";
  const std::string gpu_first = gpus.empty() ? "" : "a built-in GPU that runs the model (" + gpus + "), else ";
  const std::string fast_cpu = ggml_cpu ? "ggml's CPU kernels (llama.cpp-level accuracy)" : "DynaCore's CPU kernels";
  r << "  auto, performance: " << gpu_first << fast_cpu << "\n";
  r << "  exact:             " << gpu_first << "DynaCore's CPU kernels (" << (k.int8_simd ? "int8 per-32, " : "")
    << "DD-053)\n";
  r << "  safe:              DynaCore's fp32 CPU reference kernels only\n\n";

  std::string why;
  const TuneCache cache = load_tune_cache(tune_cache_path(), &why);
  r << "Tuning cache: " << tune_cache_path() << "\n  " << why << "\n";
  for (const TuneEntry& e : cache.entries) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  %-9s %-36s threads %2d  %.1f tok/s (default %.1f)  %s, %s\n", e.device.c_str(),
                  e.model.c_str(), e.threads, e.tok_s, e.baseline_tok_s, e.conditions.c_str(), e.measured_at.c_str());
    r << buf;
  }
  if (const char* v = std::getenv("DYNALM_AUTOTUNE"); v != nullptr && std::string_view(v) == "0") {
    r << "  DYNALM_AUTOTUNE=0: tuned settings are not used\n";
  }
  return r.str();
}

}  // namespace dynalm

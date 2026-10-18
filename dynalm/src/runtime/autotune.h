#pragma once

// AutoTune: hardware-aware choices on top of the device plan
// (runtime/device_select.h), observable and cached.
//
//   modes   auto / performance: the fastest validated path (GPU first, then
//           ggml's CPU kernels at llama.cpp-level accuracy, then DynaCore's);
//           exact: DD-053 everywhere (GPUs checked per family, DynaCore's CPU
//           kernels, never the 8-bit-activation ggml CPU kernels);
//           safe: DynaCore's fp32 CPU kernels only (no GPU, no int8): the
//           reference path, for debugging and odd hardware.
//   report  what the machine has (CPU features detected at run time, cores,
//           memory, built-in GPU backends) and what each mode would run.
//   cache   tuned settings (today: CPU threads per device and model), measured
//           by `dynalm autotune --run` and stored in a versioned JSON file
//           keyed by the CPU, its features, the OS and the DynaLM build; any
//           change of that key invalidates every entry. The Engine reads it
//           only when the caller left the setting on auto (--threads not given)
//           and never when DYNALM_AUTOTUNE=0.

#include <string>
#include <string_view>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/device/device_registry.h"
#include "model_ir/model_config.h"
#include "common/core.h"

namespace dynalm {

struct EngineOptions;

enum class TuneMode { kAuto, kPerformance, kExact, kSafe };
std::string_view tune_mode_name(TuneMode m);
Result<TuneMode> parse_tune_mode(std::string_view s);  // auto | performance | exact | safe
// Applies a mode on top of the device choice already in `o` (apply_backend_choice).
// exact / safe also refuse a device choice that contradicts them (--backend ggml-cpu).
Status apply_tune_mode(TuneMode m, EngineOptions& o);

// What identifies a measurement's validity: hardware, OS and build.
std::string tune_machine_key();
// What identifies a model for tuning: architecture, width, depth, quantization.
std::string tune_model_key(const ModelConfig& c, std::string_view quantization);

struct TuneEntry {
  std::string device;     // device_kind_name: "ggml-cpu", "cpu", ...
  std::string model;      // tune_model_key
  int32_t threads = 0;
  double tok_s = 0;       // decode tokens/s at that setting
  double baseline_tok_s = 0;  // at the default setting
  std::string conditions;     // "AC, load 1.2" when measured
  std::string measured_at;    // ISO-8601 UTC
};

struct TuneCache {
  static constexpr int kVersion = 1;
  std::string machine;  // tune_machine_key() when written
  std::vector<TuneEntry> entries;
  const TuneEntry* find(std::string_view device, std::string_view model) const;
  void put(TuneEntry e);  // replaces the entry of the same device and model
};

// $DYNALM_AUTOTUNE_CACHE, else dynalm_home()/autotune.json.
std::string tune_cache_path();
// A missing, unreadable, other-version or other-machine file loads as empty
// (with the reason in `why` when given): never an error, never stale settings.
TuneCache load_tune_cache(const std::string& path, std::string* why = nullptr);
Status save_tune_cache(const TuneCache& c, const std::string& path);

// Threads tuned for this device and model on this machine, 0 when none (or
// DYNALM_AUTOTUNE=0). Reads the cache file once per process (again when
// tune_cache_path() changes).
int32_t tuned_threads(dynacore::DeviceKind device, const ModelConfig& c, std::string_view quantization);

// Whether a tuning run's machine state makes its measurement meaningful.
// `notes`: what was checked and is fine ("AC", "load 0.4"); `problems`: what
// is not ("on battery", "busy (load 3.1)"). Not quiet when there is a problem,
// or when `first_check` (the check before the run) could read nothing at all
// (no power or load source on this OS): an unverified measurement is not
// stored without --force.
struct TuneConditions {
  bool quiet = false;
  std::string text;
};
TuneConditions judge_tune_conditions(const std::vector<std::string>& notes, const std::vector<std::string>& problems,
                                     bool first_check);

// Human-readable report: hardware, built-in backends, kernels per CPU path,
// what each mode selects, and the cache state.
std::string autotune_report();

}  // namespace dynalm

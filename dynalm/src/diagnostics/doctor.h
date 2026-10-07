#pragma once

// `dynalm doctor`: one report with everything an installation problem report
// needs (versions, platform, CPU and ISA, the backend DynaLM will use,
// memory, install location, model store), a short DynaCore kernel self-test,
// and actionable warnings. Text for people, JSON (--json) for issue forms.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/core.h"

namespace dynalm {

struct DoctorReport {
  // Build.
  std::string dynalm_version, dynacore_version, build_type, compiler;
  std::vector<std::string> compiled_isas;     // "generic", "avx2", ...
  std::vector<std::string> compiled_devices;  // "cpu"
  // System.
  std::string os, os_version, arch;
  std::string cpu, cpu_vendor;
  int32_t physical_cores = 0, logical_cores = 0, performance_cores = 0, efficiency_cores = 0;
  int64_t l2_bytes = 0, l3_bytes = 0;
  std::vector<std::pair<std::string, bool>> isa;  // feature name -> present
  int64_t ram_total = 0, ram_available = 0;
  struct Gpu {
    std::string name;
    int64_t memory_bytes = 0;
    std::string driver;
  };
  std::vector<Gpu> gpus;
  // What DynaLM will use.
  std::string device;    // "CPU"
  std::string kernel;    // selected ISA tier
  std::string isa_note;  // DYNACORE_ISA override / fallback, empty if none
  int32_t threads = 0;   // default compute threads
  // Installation.
  std::string executable;
  bool dynacorec_found = false;
  std::string model_store;
  int32_t local_models = 0;
  std::string config_file;  // empty when none
  // Checks.
  bool dynacore_ok = false;
  std::string dynacore_detail;
  // Problems worth acting on, one line each.
  std::vector<std::string> warnings;
  bool ready = true;
};

DoctorReport collect_doctor_report();
std::string format_doctor_text(const DoctorReport& r);
std::string format_doctor_json(const DoctorReport& r);

}  // namespace dynalm

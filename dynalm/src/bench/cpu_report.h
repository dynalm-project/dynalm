#pragma once

// dynalm cpu-report (docs/perf/cpu-report-design.md): what to measure on this
// CPU, and the report. The pieces here are pure functions (unit-tested); the
// command itself (cli/cpu_report.cpp) runs the servers and the load.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bench/loadgen.h"
#include "dynacore/hardware/cpu_info.h"
#include "common/core.h"

namespace dynalm::cpu_report {

// One way of running DynaLM (or llama.cpp) on the CPU.
struct Path {
  std::string id;     // stable key in the JSON: "default", "ggml-cpu", "dynacore", "qmm-neon_dot", ...
  std::string label;  // for the Markdown
  bool llama = false;                 // llama.cpp's llama-server instead of dynalm serve
  std::vector<std::string> serve_args;  // after `dynalm serve <model>` (or llama-server's)
  std::vector<std::pair<std::string, std::string>> env;  // set for the server process
  // Accuracy check device for run_perplexity: "native" (DynaCore) or "ggml-cpu"; empty = not checked.
  std::string score_device;
};

// The paths for a CPU with these features. `native_default`: DynaCore's
// kernels are already DynaLM's default CPU path here (SME CPUs today).
std::vector<Path> plan_paths(const dynacore::CpuFeatures& f, bool native_default, int threads);

struct ModelSpec {
  std::string id;    // "1.5b", "4b", "8b"
  std::string ref;   // for dynalm pull
  int64_t bytes = 0;  // download size
  int64_t min_ram_bytes = 0;  // skipped below this
};
const std::vector<ModelSpec>& all_models();
// The models of a run: `ids` ("1.5b,4b"; empty = all, or 1.5b only with quick),
// minus those that need more RAM than `ram_bytes` (reported as skipped).
struct ModelChoice {
  std::vector<ModelSpec> run;
  std::vector<std::string> skipped;  // "8b: needs 12 GB of RAM, this machine has 8.0 GB"
  std::string error;                 // an unknown id
};
ModelChoice choose_models(const std::string& ids, bool quick, int64_t ram_bytes);

// The pinned llama.cpp release (its CPU build for this OS and architecture).
struct LlamaAsset {
  std::string build;  // "b11695"
  std::string url;
  std::string sha256;
  bool zip = false;
};
// os: "macos" | "linux" | "windows"; arch: "arm64" | "x86_64". Empty url: none for this platform.
LlamaAsset llama_asset(const std::string& os, const std::string& arch);

// Fixed workload (the same as the published M4 results).
struct Workload {
  int32_t prompt_tokens = 512;
  int32_t output_tokens = 128;
  std::vector<int32_t> users = {1, 4};
  int rounds = 2;
};

// One measured point: one path, one model, one user count, one round.
struct Point {
  std::string path, model;
  int32_t users = 1;
  int round = 0;
  double tok_s = 0, ttft_p50_ms = 0, itl_p50_ms = 0, itl_p99_ms = 0;
  int32_t completed = 0, errors = 0;
};

// Agreement of one path with llama.cpp on the same tokens (llama-perplexity's
// --kl-divergence-base file; docs/perf/cpu-model-sweep.md method). This is
// agreement with llama.cpp, not DD-053 (which compares against DynaLM's own
// fp32 reference).
struct Accuracy {
  std::string path, model;
  bool ran = false;
  std::string error;
  double top1_pct = 0, kl_mean = 0, ppl = 0, llama_ppl = 0;
  int64_t positions = 0;
  bool pass = false;
};
// The verdict: top-1 agreement at least the 94 % that llama.cpp reaches
// against itself with only its batch size changed (cpu-model-sweep.md), and
// a perplexity at most 1 % above llama.cpp's (the DD-053 perplexity limit).
inline constexpr double kMinTop1Pct = 94.0;
inline constexpr double kMaxPplIncreasePct = 1.0;
bool accuracy_pass(double top1_pct, double ppl, double llama_ppl);

struct Report {
  std::string dynalm_version;
  std::string machine_json;  // dynalm doctor --json
  std::string cpu_name, os;
  std::string features;      // "avx2 fma avx_vnni" / "neon dotprod i8mm sme"
  int64_t ram_bytes = 0;
  int threads = 0;
  bool controlled = true;
  std::vector<std::string> notes;  // why uncontrolled, skipped models, llama.cpp source, ...
  std::string llama_source;        // "b11695 (pinned download, sha256 ...)" or "--llama PATH (version ...)"
  Workload workload;
  std::vector<Path> paths;
  std::vector<std::string> models;
  std::vector<Point> points;
  std::vector<Accuracy> accuracy;
};

// Medians over rounds for (path, model, users); nullptr-safe lookups in the renderers.
std::string to_json(const Report& r);
std::string to_markdown(const Report& r);
// The CPU features cpu_info detects, as a space-separated list.
std::string feature_list(const dynacore::CpuFeatures& f);

}  // namespace dynalm::cpu_report

// dynalm autotune: what this machine can run and how (runtime/autotune.h).
//
//   dynalm autotune                     report: hardware, kernels, modes, cache
//   dynalm autotune --run MODEL         measure CPU thread counts for MODEL and
//                                       store the fastest (a bounded search)
//   dynalm autotune --clear             delete the tuning cache
//
// A measurement is only stored when the machine was quiet (on AC power, low
// load, not thermally limited); otherwise it is shown, marked unreliable, and
// kept only with --force. Only DynaLM's own kernels run: nothing is downloaded
// or compiled at tuning time.

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <stdlib.h>  // getloadavg
#endif

#include "cli/commands.h"
#include "dynacore/base/timer.h"
#include "dynacore/hardware/cpu_info.h"
#include "runtime/autotune.h"
#include "runtime/engine.h"

namespace dynalm::cli {
namespace {

void usage() {
  std::fprintf(stderr,
               "usage: dynalm autotune [--run MODEL [options] | --clear]\n"
               "  (no option)        show the hardware, kernels, modes and tuned settings\n"
               "  --run MODEL        measure decode speed at a few CPU thread counts and store the\n"
               "                     fastest for this model, device, machine and build\n"
               "  --backend B        device to tune with --run: cpu (default) | native-cpu | ggml-cpu\n"
               "  --budget S         stop trying thread counts after about S seconds (default 120)\n"
               "  --tokens N         tokens generated per measurement (default 64)\n"
               "  --force            store the result even when the machine was not quiet\n"
               "  --clear            delete the tuning cache (DYNALM_AUTOTUNE=0 ignores it instead)\n"
               "The cache is %s\n",
               tune_cache_path().c_str());
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

std::string run_cmd(const char* cmd) {
#if defined(_WIN32)
  (void)cmd;
  return {};
#else
  std::string out;
  if (FILE* p = popen(cmd, "r")) {
    char buf[256];
    while (std::fgets(buf, sizeof(buf), p) != nullptr) out += buf;
    pclose(p);
  }
  return out;
#endif
}

// `with_load`: false after a run, whose own threads raise the load average.
TuneConditions check_conditions(bool with_load) {
  std::vector<std::string> notes, problems;
#if defined(__APPLE__)
  const std::string batt = run_cmd("pmset -g batt 2>/dev/null");
  if (batt.find("Battery Power") != std::string::npos) problems.push_back("on battery");
  else if (batt.find("AC Power") != std::string::npos) notes.push_back("AC");
  const std::string therm = run_cmd("pmset -g therm 2>/dev/null");
  if (const size_t p = therm.find("CPU_Speed_Limit"); p != std::string::npos) {
    const int limit = std::atoi(therm.c_str() + therm.find('=', p) + 1);
    if (limit > 0 && limit < 100) problems.push_back("thermally limited (CPU speed limit " + std::to_string(limit) + "%)");
  }
#elif defined(__linux__)
  std::error_code ec;
  bool have_ac = false, ac_online = false;
  for (const auto& d : std::filesystem::directory_iterator("/sys/class/power_supply", ec)) {
    std::FILE* f = std::fopen((d.path() / "type").c_str(), "r");
    if (f == nullptr) continue;
    char type[32] = {};
    const bool read = std::fgets(type, sizeof(type), f) != nullptr;
    std::fclose(f);
    if (!read || std::string_view(type).substr(0, 5) != "Mains") continue;
    have_ac = true;
    if (std::FILE* o = std::fopen((d.path() / "online").c_str(), "r")) {
      ac_online = ac_online || std::fgetc(o) == '1';
      std::fclose(o);
    }
  }
  if (have_ac && !ac_online) problems.push_back("on battery");
  else if (have_ac) notes.push_back("AC");
#endif
#if !defined(_WIN32)
  double load[1] = {0};
  if (with_load && getloadavg(load, 1) == 1) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "load %.1f", load[0]);
    // More than a core's worth of other work skews a multi-threaded run.
    if (load[0] > 1.5) problems.push_back(std::string("busy (") + buf + ")");
    else notes.push_back(buf);
  }
#endif
  return judge_tune_conditions(notes, problems, with_load);
}

// Decode tokens/s of one greedy generation (first token excluded: it carries
// the prompt), or < 0 on error.
double measure_decode(Engine& e, int tokens) {
  GenerateParams p;
  p.max_tokens = tokens;
  p.stop_at_eog = false;
  auto s = e.generate_text("The history of the city begins with", /*parse_special=*/false, p);
  if (!s.ok()) return -1;
  const Stopwatch t;
  StreamEvent ev;
  double first_ms = -1;
  while ((*s)->next(ev)) {
    if (first_ms < 0 && !ev.text.empty()) first_ms = t.elapsed_ms();
    if (ev.done) break;
  }
  if (ev.finish == StreamFinish::kError || first_ms < 0 || ev.completion_tokens < 2) return -1;
  const double ms = t.elapsed_ms() - first_ms;
  return ms > 0 ? (ev.completion_tokens - 1) * 1e3 / ms : -1;
}

int cmd_autotune_run(std::string path, const std::string& backend, int budget_s, int tokens, bool force) {
  if (auto resolved = ensure_model(path, true); resolved.ok()) {
    path = *resolved;
  } else {
    std::fprintf(stderr, "autotune: %s\n", resolved.status().message().c_str());
    return 1;
  }
  EngineOptions base;
  if (Status st = apply_backend_choice(backend, base); !st.ok()) {
    std::fprintf(stderr, "autotune: %s\n", st.message().c_str());
    return 1;
  }
  base.model_path = path;
  base.kv_tokens = 1024;
  // Thread counts to try: the defaults of both CPU paths, and a few around the
  // performance-core count, small to large. Bounded by the budget.
  const CpuInfo& cpu = cpu_info();
  const int phys = std::max(1, cpu.physical_cores);
  const int perf = cpu.performance_cores > 0 ? cpu.performance_cores : phys;
  std::vector<int> cand = {perf, phys, std::max(1, perf - 2), std::max(1, perf / 2), std::min(phys, perf + 2)};
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

  const TuneConditions before = check_conditions(true);
  std::printf("Tuning CPU threads for %s\n  machine: %s\n", path.c_str(), before.text.c_str());
  if (!before.quiet) std::printf("  warning: the machine is not quiet; results will be marked unreliable\n");

  struct Point { int threads; double tok_s; };
  std::vector<Point> points;
  DeviceKind kind = DeviceKind::kCpu;
  std::string model_key;
  int default_threads = 0;
  const Stopwatch budget;
  for (int th : cand) {
    if (!points.empty() && budget.elapsed_ms() > budget_s * 1e3) {
      std::printf("  budget of %d s reached; %zu of %zu thread counts tried\n", budget_s, points.size(), cand.size());
      break;
    }
    EngineOptions o = base;
    o.threads = th;
    auto eng = Engine::create(o);
    if (!eng.ok()) {
      std::fprintf(stderr, "autotune: %s\n", eng.status().to_string().c_str());
      return 1;
    }
    Engine& e = **eng;
    kind = e.device_kind();
    if (kind != DeviceKind::kCpu && kind != DeviceKind::kGgmlCpu) {
      std::fprintf(stderr, "autotune: only CPU devices take a thread count (this one runs on %s)\n",
                   e.backend_name().c_str());
      return 1;
    }
    model_key = tune_model_key(e.model().config, e.model().quantization);
    default_threads = kind == DeviceKind::kGgmlCpu ? perf : phys;
    (void)measure_decode(e, std::min(tokens, 16));  // warm-up: page in weights, settle clocks
    std::vector<double> runs;
    for (int r = 0; r < 3; ++r) {
      const double v = measure_decode(e, tokens);
      if (v > 0) runs.push_back(v);
    }
    if (runs.empty()) {
      std::fprintf(stderr, "autotune: generation failed at %d threads\n", th);
      return 1;
    }
    std::sort(runs.begin(), runs.end());
    const double med = runs[runs.size() / 2];
    const double spread = (runs.back() - runs.front()) / med;
    std::printf("  %-9s %2d threads: %6.1f tok/s (3 runs, spread %.0f%%)%s\n",
                std::string(device_kind_name(kind)).c_str(), th, med, spread * 100,
                spread > 0.10 ? "  noisy" : "");
    points.push_back({th, med});
  }

  const Point best = *std::max_element(points.begin(), points.end(),
                                       [](const Point& a, const Point& b) { return a.tok_s < b.tok_s; });
  const auto def = std::find_if(points.begin(), points.end(), [&](const Point& p) { return p.threads == default_threads; });
  const double def_tok_s = def != points.end() ? def->tok_s : 0;
  // Keep the default unless another count is clearly faster: a 3% gain is
  // within run-to-run noise on a laptop.
  const Point chosen = def != points.end() && best.tok_s < def_tok_s * 1.03 ? *def : best;
  std::printf("Result: %d threads, %.1f tok/s (default %d threads: %.1f tok/s)\n", chosen.threads, chosen.tok_s,
              default_threads, def_tok_s);

  const TuneConditions after = check_conditions(false);
  const bool quiet = before.quiet && after.quiet;
  if (!quiet && !force) {
    std::printf("Not stored: the machine was not quiet (%s). Retry on AC power with nothing else running, or "
                "pass --force.\n",
                (before.quiet ? after : before).text.c_str());
    return 2;
  }
  std::string why;
  TuneCache cache = load_tune_cache(tune_cache_path(), &why);
  cache.machine = tune_machine_key();
  TuneEntry entry;
  entry.device = device_kind_name(kind);
  entry.model = model_key;
  entry.threads = chosen.threads;
  entry.tok_s = chosen.tok_s;
  entry.baseline_tok_s = def_tok_s;
  entry.conditions = quiet ? before.text : "UNRELIABLE (" + (before.quiet ? after : before).text + ")";
  cache.put(std::move(entry));
  if (Status st = save_tune_cache(cache, tune_cache_path()); !st.ok()) {
    std::fprintf(stderr, "autotune: %s\n", st.message().c_str());
    return 1;
  }
  std::printf("Stored in %s (used when --threads is not given; DYNALM_AUTOTUNE=0 ignores it)\n",
              tune_cache_path().c_str());
  return 0;
}

}  // namespace

int cmd_autotune(std::span<const std::string_view> args) {
  std::string model, backend = "cpu";
  int budget_s = 120, tokens = 64;
  bool run = false, clear = false, force = false;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "--run") ok = !(model = std::string(value())).empty(), run = true;
    else if (a == "--backend") ok = !(backend = std::string(value())).empty();
    else if (a == "--budget") ok = parse_int(value(), budget_s) && budget_s > 0;
    else if (a == "--tokens") ok = parse_int(value(), tokens) && tokens >= 8;
    else if (a == "--force") force = true;
    else if (a == "--clear") clear = true;
    else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else ok = false;
    if (!ok) {
      std::fprintf(stderr, "autotune: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (run && clear) {
    usage();
    return 1;
  }
  if (clear) {
    const std::string p = tune_cache_path();
    std::error_code ec;
    const bool removed = std::filesystem::remove(p, ec);
    if (ec) {
      std::fprintf(stderr, "autotune: cannot remove %s: %s\n", p.c_str(), ec.message().c_str());
      return 1;
    }
    std::printf("%s %s\n", removed ? "Removed" : "No tuning cache at", p.c_str());
    return 0;
  }
  if (run) return cmd_autotune_run(model, backend, budget_s, tokens, force);
  std::printf("%s", autotune_report().c_str());
  return 0;
}

}  // namespace dynalm::cli

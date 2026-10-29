// dynalm cpu-report [--quick] [--models 1.5b,4b,8b] [--llama DIR] [--out DIR]
//                   [--rounds N] [--threads N] [--anonymous] [--dry-run]
//
// Measures DynaLM's CPU paths against llama.cpp on this machine and writes a
// report to attach to issue #31 (docs/perf/cpu-report-design.md): what is
// measured comes from bench/cpu_report.h (plan_paths), this file runs it.
//
// Internal (run by cpu-report itself, one process per path so each path's
// environment switches take effect):
//   dynalm cpu-report --score <device> <model.gguf> <text> <llama kld file> <threads>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "api/json.h"
#include "bench/cpu_report.h"
#include "bench/loadgen.h"
#include "bench/perplexity.h"
#include "cli/commands.h"
#include "common/process.h"
#include "common/sha256.h"
#include "common/version.h"
#include "diagnostics/doctor.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/system_info.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "registry/model_registry.h"
#include "runtime/engine.h"
#include "common/core.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>  // gethostname
#endif

#include "httplib.h"

namespace dynalm::cli {
namespace {

namespace fs = std::filesystem;
namespace cr = cpu_report;

constexpr const char* kWikiUrl = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip";
constexpr const char* kWikiZipSha = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11";
constexpr const char* kWikiTestSha = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08";
constexpr int kScoreCtx = 512, kScoreChunks = 10;
constexpr int kCtxPerUser = 4096;  // context per user, as the published M4 measurements

void usage() {
  std::fprintf(stderr,
               "usage: dynalm cpu-report [options]\n"
               "Measures DynaLM's CPU paths against llama.cpp on this machine (CPU only) and writes\n"
               "cpu-report-<host>-<time>.json and .md to attach to issue #31.\n"
               "  --quick            Qwen2.5-1.5B only, 1 round (about 15 minutes)\n"
               "  --models LIST      1.5b,4b,8b (default all; 8b needs 12 GB of RAM)\n"
               "  --llama DIR        a llama.cpp build to use (llama-server, llama-perplexity) instead of\n"
               "                     downloading the pinned release\n"
               "  --out DIR          where to write the report (default: current directory)\n"
               "  --rounds N         rounds per point (default 2; --quick: 1)\n"
               "  --threads N        threads for every engine (default: DynaLM's default)\n"
               "  --anonymous        no host name in the report or its file name\n"
               "  --dry-run          print what would run, then exit\n"
               "Downloads: the models (1.1 / 2.5 / 5.0 GB), llama.cpp b11695 (~15 MB) and wikitext-2\n"
               "(5 MB), each checked against its SHA-256. Prompts and outputs are not stored.\n");
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Runs a program to completion, its output in `log` (empty: discarded).
int run_wait(const std::vector<std::string>& argv, const ChildProcess::Env& env, const std::string& log,
             int64_t timeout_ms = 3600 * 1000) {
  auto c = ChildProcess::start(argv, env, log);
  if (!c.ok()) return -1;
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while ((*c)->running()) {
    if (std::chrono::steady_clock::now() > until) return (*c)->stop(2000), -2;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return (*c)->stop();
}

// A program on PATH (curl, tar, unzip, python3); empty if absent.
std::string which(const std::string& name) {
  const char* path = std::getenv("PATH");
  if (path == nullptr) return {};
#if defined(_WIN32)
  const char sep = ';';
  const std::vector<std::string> exts = {".exe", ""};
#else
  const char sep = ':';
  const std::vector<std::string> exts = {""};
#endif
  std::stringstream ss(path);
  for (std::string dir; std::getline(ss, dir, sep);) {
    for (const std::string& e : exts) {
      std::error_code ec;
      const fs::path p = fs::path(dir) / (name + e);
      if (fs::is_regular_file(p, ec)) return p.string();
    }
  }
  return {};
}

// Downloads `url` to `dest` and checks its SHA-256 (kept when it already matches).
Status fetch(const std::string& url, const std::string& sha, const fs::path& dest, const std::string& log) {
  std::error_code ec;
  if (fs::exists(dest, ec)) {
    if (auto h = sha256_file(dest.string()); h.ok() && *h == sha) return Status::Ok();
  }
  const std::string curl = which("curl");
  if (curl.empty()) return Unsupported("curl is needed to download " + url);
  std::printf("  downloading %s\n", url.c_str());
  std::fflush(stdout);
  if (run_wait({curl, "-fL", "--retry", "3", "-sS", "-o", dest.string(), url}, {}, log) != 0) {
    return IoError("download failed: " + url + " (log: " + log + ")");
  }
  auto h = sha256_file(dest.string());
  if (!h.ok()) return h.status();
  if (*h != sha) {
    fs::remove(dest, ec);
    return IoError("sha256 mismatch for " + url + ": got " + *h + ", expected " + sha);
  }
  return Status::Ok();
}

// Unpacks a .tar.gz or .zip into `dir` (tar; for zip on Linux, unzip or python3).
Status unpack(const fs::path& archive, const fs::path& dir, bool zip, const std::string& log) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string tar = which("tar");
#if !defined(__linux__)
  if (!tar.empty() && run_wait({tar, "-xf", archive.string(), "-C", dir.string()}, {}, log) == 0) return Status::Ok();
#else
  if (!zip && !tar.empty() && run_wait({tar, "-xzf", archive.string(), "-C", dir.string()}, {}, log) == 0) return Status::Ok();
  if (zip) {
    if (const std::string u = which("unzip"); !u.empty() && run_wait({u, "-oq", archive.string(), "-d", dir.string()}, {}, log) == 0) {
      return Status::Ok();
    }
    if (const std::string py = which("python3");
        !py.empty() && run_wait({py, "-m", "zipfile", "-e", archive.string(), dir.string()}, {}, log) == 0) {
      return Status::Ok();
    }
  }
#endif
  return IoError("cannot unpack " + archive.string() + " (log: " + log + ")");
}

fs::path find_file(const fs::path& dir, const std::string& name) {
  std::error_code ec;
  for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (it->is_regular_file(ec) && it->path().filename() == name) return it->path();
  }
  return {};
}

std::string exe(const std::string& name) {
#if defined(_WIN32)
  return name + ".exe";
#else
  return name;
#endif
}

// Power and load (as tools/bench/baseline.sh): an empty string when fine, else why not.
std::string power_problem() {
#if defined(_WIN32)
  SYSTEM_POWER_STATUS s{};
  if (GetSystemPowerStatus(&s) && s.ACLineStatus == 0) return "running on battery";
  return {};
#elif defined(__APPLE__)
  const std::string pmset = "/usr/bin/pmset";
  const fs::path tmp = fs::temp_directory_path() / "dynalm-cpu-report-pmset.txt";
  std::error_code ec;
  fs::remove(tmp, ec);
  if (run_wait({pmset, "-g", "batt"}, {}, tmp.string(), 10000) != 0) return {};
  const std::string out = read_file(tmp.string());
  fs::remove(tmp, ec);
  if (out.find("'Battery Power'") != std::string::npos) return "running on battery";
  return {};
#else
  std::error_code ec;
  bool mains_seen = false, mains_online = false;
  for (const auto& e : fs::directory_iterator("/sys/class/power_supply", ec)) {
    if (read_file((e.path() / "type").string()).rfind("Mains", 0) != 0) continue;
    mains_seen = true;
    mains_online = mains_online || read_file((e.path() / "online").string()).rfind("1", 0) == 0;
  }
  return mains_seen && !mains_online ? "running on battery" : std::string();
#endif
}

double load_average() {
#if defined(_WIN32)
  return -1;  // not available
#else
  double l[1] = {-1};
  return getloadavg(l, 1) == 1 ? l[0] : -1;
#endif
}

// Waits up to 60 s for the 1-minute load to fall below the limit; false if it stays above.
bool wait_for_quiet(double limit) {
  for (int i = 0; i < 12; ++i) {
    const double l = load_average();
    if (l < 0 || l < limit) return true;
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
  return false;
}

int free_port(int start) {
  for (int p = start; p < start + 200; ++p) {
    httplib::Client c("127.0.0.1", p);
    c.set_connection_timeout(0, 200000);
    if (!c.Get("/health")) return p;  // nothing listening
  }
  return start;
}

// Waits until the server answers GET /health with 200; false if it exits or times out.
bool wait_ready(ChildProcess& proc, int port, int64_t timeout_ms) {
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  httplib::Client c("127.0.0.1", port);
  c.set_connection_timeout(1, 0);
  c.set_read_timeout(5, 0);
  while (std::chrono::steady_clock::now() < until) {
    if (!proc.running()) return false;
    if (auto r = c.Get("/health"); r && r->status == 200) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return false;
}

std::string timestamp() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char b[32];
  std::strftime(b, sizeof(b), "%Y%m%d-%H%M", &tm);
  return b;
}

std::string host_name() {
#if defined(_WIN32)
  char b[256];
  DWORD n = sizeof(b);
  return GetComputerNameA(b, &n) ? std::string(b, n) : "host";
#else
  char b[256] = {};
  return gethostname(b, sizeof(b) - 1) == 0 ? std::string(b) : "host";
#endif
}

std::string safe_name(std::string s) {
  for (char& ch : s) {
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') ch = '-';
  }
  return s.empty() ? "host" : s;
}

// The internal scorer: one path's agreement with llama.cpp, in this process.
int score_main(std::span<const std::string_view> a) {
  if (a.size() != 5) return 2;
  PerplexityOptions o;
  o.device = std::string(a[0]);
  o.model_path = std::string(a[1]);
  o.text_path = std::string(a[2]);
  o.kld_ref = std::string(a[3]);
  o.threads = std::atoi(std::string(a[4]).c_str());
  o.ctx = kScoreCtx;
  o.max_chunks = kScoreChunks;
  auto r = run_perplexity(o);
  if (!r.ok()) {
    std::printf("SCORE_ERROR %s\n", r.status().message().c_str());
    return 1;
  }
  std::printf("SCORE {\"ppl\": %.6f, \"top1\": %.4f, \"kl_mean\": %.8f, \"positions\": %lld}\n", r->ppl,
              r->parity.top1_pct(), r->parity.kl_mean, static_cast<long long>(r->parity.positions));
  return 0;
}

double parse_llama_ppl(const std::string& log) {
  const size_t i = log.rfind("Final estimate: PPL = ");
  if (i == std::string::npos) return 0;
  return std::atof(log.c_str() + i + 22);
}

}  // namespace

int cmd_cpu_report(std::span<const std::string_view> args) {
  if (!args.empty() && args[0] == "--score") return score_main(args.subspan(1));
  bool quick = false, anonymous = false, dry_run = false;
  std::string models_arg, llama_dir, out_dir = ".";
  int rounds = -1, threads = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string { return i + 1 < args.size() ? std::string(args[++i]) : std::string(); };
    if (a == "--quick") quick = true;
    else if (a == "--anonymous") anonymous = true;
    else if (a == "--dry-run") dry_run = true;
    else if (a == "--models") models_arg = value();
    else if (a == "--llama") llama_dir = value();
    else if (a == "--out") out_dir = value();
    else if (a == "--rounds") rounds = std::atoi(value().c_str());
    else if (a == "--threads") threads = std::atoi(value().c_str());
    else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      usage();
      return 1;
    }
  }
  const dynacore::CpuInfo& ci = cpu_info();
  if (threads <= 0) threads = ci.performance_cores > 0 ? ci.performance_cores : ci.physical_cores;
  if (rounds <= 0) rounds = quick ? 1 : 2;

  cr::Report rep;
  rep.dynalm_version = ENGINE_VERSION_STRING;
  rep.cpu_name = ci.brand;
  rep.os = os_info().name + " " + os_info().version + " (" + os_info().arch + ")";
  rep.features = cr::feature_list(ci.features);
  rep.ram_bytes = memory_info().total_bytes;
  rep.threads = threads;
  rep.workload.rounds = rounds;
  rep.paths = cr::plan_paths(ci.features, native_cpu_default(), threads);
  const cr::ModelChoice mc = cr::choose_models(models_arg, quick, rep.ram_bytes);
  if (!mc.error.empty()) {
    std::fprintf(stderr, "cpu-report: %s\n", mc.error.c_str());
    return 1;
  }
  for (const std::string& s : mc.skipped) rep.notes.push_back("skipped " + s);
  for (const cr::ModelSpec& m : mc.run) rep.models.push_back(m.id);
  if (mc.run.empty()) {
    std::fprintf(stderr, "cpu-report: no model to run\n");
    return 1;
  }

  std::printf("cpu-report: %s, %s\n  features: %s\n  threads %d, %d round(s), models:", rep.cpu_name.c_str(),
              rep.os.c_str(), rep.features.c_str(), threads, rounds);
  for (const std::string& m : rep.models) std::printf(" %s", m.c_str());
  std::printf("\n  paths:\n");
  for (const cr::Path& p : rep.paths) {
    std::printf("    %-14s %s", p.id.c_str(), p.label.c_str());
    for (const auto& [k, v] : p.env) std::printf("  [%s=%s]", k.c_str(), v.c_str());
    std::printf("\n");
  }
  if (dry_run) return 0;

  // Work directory: downloads and logs.
  const fs::path work = fs::path(models_dir()).parent_path() / "cpu-report";
  std::error_code ec;
  fs::create_directories(work / "logs", ec);
  const std::string log_dl = (work / "logs" / "download.log").string();

  // llama.cpp.
  fs::path llama_server, llama_ppl;
  if (!llama_dir.empty()) {
    llama_server = find_file(llama_dir, exe("llama-server"));
    llama_ppl = find_file(llama_dir, exe("llama-perplexity"));
    rep.llama_source = "--llama " + llama_dir;
  } else {
    std::string os = os_info().name == "macOS" ? "macos" : os_info().name == "Windows" ? "windows" : "linux";
    const cr::LlamaAsset asset = cr::llama_asset(os, os_info().arch);
    if (asset.url.empty()) {
      std::fprintf(stderr, "cpu-report: no pinned llama.cpp build for %s %s; use --llama DIR\n", os.c_str(),
                   os_info().arch.c_str());
      return 1;
    }
    const fs::path dir = work / ("llama-" + asset.build);
    const fs::path archive = work / fs::path(asset.url).filename();
    if (Status st = fetch(asset.url, asset.sha256, archive, log_dl); !st.ok()) {
      std::fprintf(stderr, "cpu-report: %s\n", st.message().c_str());
      return 1;
    }
    if (find_file(dir, exe("llama-server")).empty()) {
      if (Status st = unpack(archive, dir, asset.zip, log_dl); !st.ok()) {
        std::fprintf(stderr, "cpu-report: %s\n", st.message().c_str());
        return 1;
      }
    }
    llama_server = find_file(dir, exe("llama-server"));
    llama_ppl = find_file(dir, exe("llama-perplexity"));
    rep.llama_source = "llama.cpp " + asset.build + " (pinned release, sha256 " + asset.sha256.substr(0, 16) + "…)";
  }
  if (llama_server.empty()) {
    std::fprintf(stderr, "cpu-report: llama-server not found%s\n", llama_dir.empty() ? "" : (" in " + llama_dir).c_str());
    return 1;
  }
#if !defined(_WIN32)
  fs::permissions(llama_server, fs::perms::owner_exec, fs::perm_options::add, ec);
  if (!llama_ppl.empty()) fs::permissions(llama_ppl, fs::perms::owner_exec, fs::perm_options::add, ec);
#endif
  const std::string llama_lib = llama_server.parent_path().string();
  const ChildProcess::Env llama_env = {{"LD_LIBRARY_PATH", llama_lib}, {"DYLD_LIBRARY_PATH", llama_lib}};
  {
    const fs::path vlog = work / "logs" / "llama-version.log";
    fs::remove(vlog, ec);
    run_wait({llama_server.string(), "--version"}, llama_env, vlog.string(), 30000);
    const std::string v = read_file(vlog.string());
    const size_t b = v.find("version:");
    if (b != std::string::npos) rep.notes.push_back("llama.cpp " + v.substr(b, v.find('\n', b) - b));
  }

  // Preconditions.
  if (const std::string p = power_problem(); !p.empty()) {
    rep.controlled = false;
    rep.notes.push_back("uncontrolled: " + p);
    std::printf("  warning: %s; the report is marked uncontrolled\n", p.c_str());
  }
  const double load_limit = std::max(1.0, 0.25 * ci.physical_cores);
  if (load_average() < 0) rep.notes.push_back("machine load not checked (not available on this OS)");

  // Models.
  std::vector<std::string> model_paths;
  for (const cr::ModelSpec& m : mc.run) {
    const std::string_view pa[] = {m.ref};
    if (cmd_pull(pa) != 0) {
      std::fprintf(stderr, "cpu-report: could not download %s\n", m.ref.c_str());
      return 1;
    }
    model_paths.push_back((fs::path(models_dir()) / fs::path(m.ref).filename()).string());
  }

  // Agreement with llama.cpp (1.5B only).
  const auto it15 = std::find(rep.models.begin(), rep.models.end(), "1.5b");
  if (it15 != rep.models.end() && !llama_ppl.empty()) {
    const std::string model = model_paths[static_cast<size_t>(it15 - rep.models.begin())];
    std::printf("agreement with llama.cpp (1.5b, %d x %d tokens of wikitext-2)\n", kScoreChunks, kScoreCtx);
    const fs::path wzip = work / "wikitext-2-raw-v1.zip";
    fs::path wiki = find_file(work / "wikitext", "wiki.test.raw");
    Status st = Status::Ok();
    if (wiki.empty()) {
      st = fetch(kWikiUrl, kWikiZipSha, wzip, log_dl);
      if (st.ok()) st = unpack(wzip, work / "wikitext", true, log_dl);
      wiki = find_file(work / "wikitext", "wiki.test.raw");
    }
    if (st.ok() && (wiki.empty() || [&] {
                      auto h = sha256_file(wiki.string());
                      return !h.ok() || *h != kWikiTestSha;
                    }())) {
      st = IoError("wiki.test.raw missing or not the expected file");
    }
    const fs::path kld = work / "llama-1.5b.kld";
    const std::string plog = (work / "logs" / "llama-perplexity.log").string();
    double llama_pplx = 0;
    if (st.ok()) {
      fs::remove(plog, ec);
      const int rc = run_wait({llama_ppl.string(), "-m", model, "-f", wiki.string(), "-c", std::to_string(kScoreCtx),
                               "--chunks", std::to_string(kScoreChunks), "-ngl", "0", "-t", std::to_string(threads),
                               "--kl-divergence-base", kld.string()},
                              llama_env, plog);
      llama_pplx = parse_llama_ppl(read_file(plog));
      if (rc != 0 || llama_pplx <= 0) st = IoError("llama-perplexity failed (log: " + plog + ")");
    }
    for (const cr::Path& p : rep.paths) {
      if (p.llama || p.score_device.empty()) continue;
      cr::Accuracy acc;
      acc.path = p.id;
      acc.model = "1.5b";
      acc.llama_ppl = llama_pplx;
      if (!st.ok()) {
        acc.error = st.message();
        rep.accuracy.push_back(acc);
        continue;
      }
      const fs::path slog = work / "logs" / ("score-" + p.id + ".log");
      fs::remove(slog, ec);
      run_wait({executable_path(), "cpu-report", "--score", p.score_device, model, wiki.string(), kld.string(),
                std::to_string(threads)},
               p.env, slog.string());
      const std::string out = read_file(slog.string());
      const size_t s = out.find("SCORE {");
      if (s == std::string::npos) {
        const size_t e = out.find("SCORE_ERROR ");
        acc.error = e != std::string::npos ? out.substr(e + 12, out.find('\n', e) - e - 12) : "scorer failed (log: " + slog.string() + ")";
      } else if (auto j = json::parse(out.substr(s + 6, out.find('\n', s) - s - 6)); j.ok()) {
        acc.ran = true;
        acc.ppl = j->find("ppl")->as_number();
        acc.top1_pct = j->find("top1")->as_number();
        acc.kl_mean = j->find("kl_mean")->as_number();
        acc.positions = static_cast<int64_t>(j->find("positions")->as_number());
        acc.pass = cr::accuracy_pass(acc.top1_pct, acc.ppl, acc.llama_ppl);
      }
      std::printf("  %-14s %s\n", p.id.c_str(),
                  acc.ran ? (std::to_string(acc.top1_pct).substr(0, 5) + " % top-1, " + (acc.pass ? "pass" : "FAIL")).c_str()
                          : ("not run: " + acc.error).c_str());
      rep.accuracy.push_back(acc);
    }
    fs::remove(kld, ec);  // ~0.8 GB for a 151k-token vocabulary; only needed for the scores above
  }

  // Speed.
  int32_t prompt_base = 0;
  const int32_t max_users = *std::max_element(rep.workload.users.begin(), rep.workload.users.end());
  for (size_t mi = 0; mi < mc.run.size(); ++mi) {
    const std::string& model = model_paths[mi];
    // The tokenizer only (exact-length prompts), not the weights: a server has them.
    auto gf = gguf::GgufFile::open(model);
    auto td = gf.ok() ? gguf::read_tokenizer_data(**gf) : Result<TokenizerData>(gf.status());
    auto tok = td.ok() ? Tokenizer::create(std::move(*td)) : Result<std::unique_ptr<Tokenizer>>(td.status());
    if (!tok.ok()) {
      std::fprintf(stderr, "cpu-report: %s: %s\n", model.c_str(), tok.status().message().c_str());
      return 1;
    }
    for (int round = 0; round < rounds; ++round) {
      std::vector<const cr::Path*> order;
      for (const cr::Path& p : rep.paths) order.push_back(&p);
      if (round % 2 == 1) std::reverse(order.begin(), order.end());  // engine order alternates
      for (const cr::Path* p : order) {
        if (!wait_for_quiet(load_limit) && rep.controlled) {
          rep.controlled = false;
          rep.notes.push_back("uncontrolled: machine load stayed above " + std::to_string(load_limit).substr(0, 4));
        }
        const int port = free_port(18900);
        std::vector<std::string> argv;
        ChildProcess::Env env = p->env;
        if (p->llama) {
          argv = {llama_server.string(), "-m", model, "--port", std::to_string(port), "-c",
                  std::to_string(kCtxPerUser * max_users), "-np", std::to_string(max_users)};
          env.insert(env.end(), llama_env.begin(), llama_env.end());
        } else {
          argv = {executable_path(), "serve", model, "--port", std::to_string(port), "--ctx",
                  std::to_string(kCtxPerUser * max_users)};
        }
        argv.insert(argv.end(), p->serve_args.begin(), p->serve_args.end());
        const fs::path slog = work / "logs" / ("server-" + mc.run[mi].id + "-" + p->id + "-r" + std::to_string(round) + ".log");
        fs::remove(slog, ec);
        std::printf("%s  round %d  %-14s ", mc.run[mi].id.c_str(), round + 1, p->id.c_str());
        std::fflush(stdout);
        auto proc = ChildProcess::start(argv, env, slog.string());
        if (!proc.ok() || !wait_ready(**proc, port, 600000)) {
          std::printf("server did not start (log: %s)\n", slog.string().c_str());
          rep.notes.push_back(mc.run[mi].id + " " + p->id + " round " + std::to_string(round + 1) +
                              ": server did not start");
          continue;
        }
        auto target = bench::make_http_target("http://127.0.0.1:" + std::to_string(port), mc.run[mi].id);
        if (!target.ok()) {
          std::printf("%s\n", target.status().message().c_str());
          continue;
        }
        for (int32_t u : rep.workload.users) {
          bench::PointConfig cfg;
          cfg.concurrency = u;
          cfg.prompt_tokens = rep.workload.prompt_tokens;
          cfg.output_tokens = rep.workload.output_tokens;
          cfg.requests = std::max(4, 2 * u);
          cfg.warmup_requests = 1;
          cfg.prompt_base = prompt_base;
          prompt_base += 1000;  // unique prompts: no prompt or prefix cache hits across points
          const bench::PointResult r = bench::run_point(**target, **tok, cfg);
          cr::Point pt{p->id, mc.run[mi].id, u, round, r.output_tok_s, r.ttft_ms.p50, r.itl_ms.p50, r.itl_ms.p99,
                       r.completed, r.errors};
          rep.points.push_back(pt);
          std::printf(" %du %.1f tok/s", u, r.output_tok_s);
          std::fflush(stdout);
        }
        std::printf("\n");
        (*proc)->stop();
      }
    }
  }

  // The report.
  rep.machine_json = format_doctor_json(collect_doctor_report());
  const std::string base =
      "cpu-report-" + (anonymous ? std::string("anonymous") : safe_name(host_name())) + "-" + timestamp();
  fs::create_directories(out_dir, ec);
  const fs::path jpath = fs::path(out_dir) / (base + ".json"), mpath = fs::path(out_dir) / (base + ".md");
  std::string machine = rep.machine_json;
  if (anonymous) {
    const std::string h = host_name();
    for (size_t at; !h.empty() && (at = machine.find(h)) != std::string::npos;) machine.replace(at, h.size(), "anonymous");
    rep.machine_json = machine;
  }
  std::ofstream(jpath, std::ios::binary) << cr::to_json(rep);
  const std::string md = cr::to_markdown(rep);
  std::ofstream(mpath, std::ios::binary) << md;
  std::printf("\n%s\nwrote %s\n      %s\nAttach both to https://github.com/dynalm-project/dynalm/issues/31\n", md.c_str(),
              jpath.string().c_str(), mpath.string().c_str());
  return 0;
}

}  // namespace dynalm::cli

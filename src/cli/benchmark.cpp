// `engine benchmark <model.gguf> [options]`: load test the engine in-process
// or any OpenAI-compatible server (--url), sweeping concurrency x prompt
// length x output length, with P50/P90/P95/P99 latency.

#include <charconv>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "bench/loadgen.h"
#include "cli/commands.h"
#include "common/version.h"
#include "loader/model_loader.h"
#include "logging/log.h"
#include "platform/cpu_info.h"
#include "runtime/engine.h"

namespace engine::cli {
namespace {

void usage() {
  std::fprintf(stderr,
               "usage: engine benchmark <model.gguf> [options]\n"
               "  --concurrency LIST    e.g. 1,4,16 (default 1,4)\n"
               "  --prompt LIST         prompt lengths in tokens (default 128,512)\n"
               "  --output LIST         output lengths in tokens (default 128)\n"
               "  --requests N          requests per point (default 2 x concurrency, min 4)\n"
               "  --url http://H:P      benchmark an OpenAI-compatible server instead of in-process\n"
               "  --model-name NAME     model field sent to --url (default: file stem)\n"
               "  -t, --threads N       compute threads (in-process)\n"
               "  -c, --ctx N           KV capacity in tokens (in-process, default 32768)\n"
               "  --out FILE            append JSON lines (one per point)\n");
}

bool parse_list(std::string_view s, std::vector<int32_t>& out) {
  out.clear();
  std::stringstream ss{std::string(s)};
  std::string item;
  while (std::getline(ss, item, ',')) {
    int v = 0;
    auto [p, ec] = std::from_chars(item.data(), item.data() + item.size(), v);
    if (ec != std::errc() || p != item.data() + item.size() || v <= 0) return false;
    out.push_back(v);
  }
  return !out.empty();
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

}  // namespace

int cmd_benchmark(std::span<const std::string_view> args) {
  std::string path, url, model_name, out_file;
  std::vector<int32_t> conc = {1, 4}, prompts = {128, 512}, outputs = {128};
  int requests = 0, threads = 0, ctx = 32768;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "--concurrency") ok = parse_list(value(), conc);
    else if (a == "--prompt") ok = parse_list(value(), prompts);
    else if (a == "--output") ok = parse_list(value(), outputs);
    else if (a == "--requests") ok = parse_int(value(), requests);
    else if (a == "--url") url = value();
    else if (a == "--model-name") model_name = value();
    else if (a == "-t" || a == "--threads") ok = parse_int(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int(value(), ctx);
    else if (a == "--out") out_file = value();
    else if (path.empty() && !a.starts_with("-")) path = a;
    else ok = false;
    if (!ok) {
      std::fprintf(stderr, "benchmark: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (path.empty()) {
    usage();
    return 1;
  }

  // The model file supplies the tokenizer used to size prompts exactly (also
  // for --url targets serving the same model).
  std::unique_ptr<Engine> eng;
  std::unique_ptr<LoadedModel> tok_model;
  std::unique_ptr<bench::Target> target;
  const Tokenizer* tokenizer = nullptr;
  if (url.empty()) {
    EngineOptions o;
    o.model_path = path;
    o.threads = threads;
    o.kv_tokens = ctx;
    auto e = Engine::create(o);
    if (!e.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", e.status().to_string().c_str());
      return 1;
    }
    eng = std::move(*e);
    tokenizer = eng->model().tokenizer.get();
    target = bench::make_engine_target(*eng);
  } else {
#if ENGINE_HAS_SERVER
    auto m = load_model(path);
    if (!m.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", m.status().to_string().c_str());
      return 1;
    }
    tok_model = std::move(*m);
    tokenizer = tok_model->tokenizer.get();
    if (model_name.empty()) model_name = std::filesystem::path(path).stem().string();
    auto t = bench::make_http_target(url, model_name);
    if (!t.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", t.status().to_string().c_str());
      return 1;
    }
    target = std::move(*t);
#else
    std::fprintf(stderr, "benchmark: --url needs a build with ENABLE_SERVER=ON\n");
    return 1;
#endif
  }

  const CpuInfo& cpu = cpu_info();
  const std::string hardware = cpu.brand + " (" + std::to_string(cpu.physical_cores) + "C/" +
                               std::to_string(cpu.logical_cores) + "T), " + "engine " + ENGINE_VERSION_STRING;
  const std::string model = std::filesystem::path(path).filename().string();
  std::printf("target: %s\nmodel: %s\nhardware: %s\n\n", target->name().c_str(), model.c_str(), hardware.c_str());
  std::printf("%4s %6s %5s | %8s %8s | %8s %8s %8s | %7s %7s %7s | %7s %7s | %6s\n", "conc", "prompt", "out",
              "out t/s", "in t/s", "TTFT p50", "p90", "p99", "ITL p50", "p90", "p99", "TPOT50", "E2E p99", "err");
  std::ofstream jsonl;
  if (!out_file.empty()) jsonl.open(out_file, std::ios::app);

  for (int32_t p : prompts) {
    for (int32_t o : outputs) {
      for (int32_t c : conc) {
        bench::PointConfig cfg{c, p, o, requests};
        bench::PointResult r = bench::run_point(*target, *tokenizer, cfg);
        std::printf("%4d %6d %5d | %8.1f %8.1f | %8.0f %8.0f %8.0f | %7.1f %7.1f %7.1f | %7.1f %7.0f | %6d\n", c, p, o,
                    r.output_tok_s, r.input_tok_s, r.ttft_ms.p50, r.ttft_ms.p90, r.ttft_ms.p99, r.itl_ms.p50,
                    r.itl_ms.p90, r.itl_ms.p99, r.tpot_ms.p50, r.e2e_ms.p99, r.errors);
        std::fflush(stdout);
        if (!r.first_error.empty()) std::fprintf(stderr, "  first error: %s\n", r.first_error.c_str());
        if (jsonl) jsonl << bench::to_json(r, model, hardware) << "\n";
      }
    }
  }
  return 0;
}

}  // namespace engine::cli

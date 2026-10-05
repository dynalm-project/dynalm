// `dynalm benchmark <model> [options]`: load test the engine in-process
// or any OpenAI-compatible server (--url), sweeping concurrency x prompt
// length x output length, with P50/P90/P95/P99 latency.

#include <algorithm>
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
#include "platform/perf_counters.h"
#include "runtime/engine.h"
#include "runtime/thread_pool.h"

namespace engine::cli {
namespace {

void usage() {
  std::fprintf(stderr,
               "usage: dynalm benchmark <model> [options]\n"
               "  --concurrency LIST    e.g. 1,4,16 (default 1,4)\n"
               "  --prompt LIST         prompt lengths in tokens (default 128,512)\n"
               "  --prompt-mix LIST     mixed lengths in one point: request i uses LIST[i %% n] tokens\n"
               "                        (e.g. 64,512,2048; replaces --prompt)\n"
               "  --output LIST         output lengths in tokens (default 128)\n"
               "  --requests N          requests per point (default 2 x concurrency, min 4)\n"
               "  --url http://H:P      benchmark an OpenAI-compatible server instead of in-process\n"
               "  --model-name NAME     model field sent to --url (default: file stem)\n"
               "  -t, --threads N       compute threads (in-process)\n"
               "  -c, --ctx N           KV capacity in tokens (in-process, default 32768)\n"
               "  --int8-decode N       int8 activations for matmuls of <= N rows (in-process; 0 = off)\n"
               "  --prefill-budget N    prompt tokens per scheduler step (in-process, default 64)\n"
               "  --decode-budget N     decode rows per scheduler step (in-process, default 64)\n"
               "  --chunk N             max prompt tokens per sequence per step (0 = budget; default 32)\n"
               "  --batch N             max tokens per forward pass (default: budgets + 1, min 256)\n"
               "  --policy P            balanced | latency | throughput (in-process, DD-061)\n"
               "  --out FILE            append JSON lines (one per point)\n"
               "  --no-diag             skip the bandwidth probe and per-point diagnostics\n");
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
  std::vector<int32_t> conc = {1, 4}, prompts = {128, 512}, outputs = {128}, prompt_mix;
  bool diag = true;
  int requests = 0, threads = 0, ctx = 32768, int8_rows = -1;
  int prefill_budget = -1, decode_budget = -1, chunk = -1, batch = -1;
  SchedulerPolicy policy = SchedulerPolicy::kBalanced;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "--concurrency") ok = parse_list(value(), conc);
    else if (a == "--prompt") ok = parse_list(value(), prompts);
    else if (a == "--prompt-mix") ok = parse_list(value(), prompt_mix);
    else if (a == "--no-diag") diag = false;
    else if (a == "--output") ok = parse_list(value(), outputs);
    else if (a == "--requests") ok = parse_int(value(), requests);
    else if (a == "--url") url = value();
    else if (a == "--model-name") model_name = value();
    else if (a == "-t" || a == "--threads") ok = parse_int(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int(value(), ctx);
    else if (a == "--int8-decode") ok = parse_int(value(), int8_rows) && int8_rows >= 0;
    else if (a == "--prefill-budget") ok = parse_int(value(), prefill_budget) && prefill_budget > 0;
    else if (a == "--decode-budget") ok = parse_int(value(), decode_budget) && decode_budget > 0;
    else if (a == "--chunk") ok = parse_int(value(), chunk) && chunk >= 0;
    else if (a == "--batch") ok = parse_int(value(), batch) && batch > 0;
    else if (a == "--policy") ok = parse_scheduler_policy(value(), policy);
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
  bench::RunContext run_ctx;
  std::unique_ptr<PerfCounters> perf;
  const Tokenizer* tokenizer = nullptr;
  if (url.empty()) {
    EngineOptions o;
    o.model_path = path;
    o.threads = threads;
    o.kv_tokens = ctx;
    o.int8_decode_rows = int8_rows;
    o.scheduler.policy = policy;
    if (prefill_budget > 0) o.scheduler.prefill_token_budget = prefill_budget;
    if (decode_budget > 0) o.scheduler.decode_token_budget = decode_budget;
    if (chunk >= 0) o.scheduler.max_prefill_chunk = chunk;
    o.max_batch_tokens = batch > 0 ? batch
                                   : std::max(o.max_batch_tokens, o.scheduler.prefill_token_budget +
                                                                      o.scheduler.decode_token_budget + 1);
    auto e = Engine::create(o);
    if (!e.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", e.status().to_string().c_str());
      return 1;
    }
    eng = std::move(*e);
    tokenizer = eng->model().tokenizer.get();
    target = bench::make_engine_target(*eng);
    if (diag) {
      // A separate pool of the same size measures the DRAM read ceiling once.
      {
        ThreadPool probe(eng->threads());
        run_ctx.peak_bw_gbs = bench::measure_read_bandwidth_gbs(probe);
      }
      perf = PerfCounters::open();  // after the engine has started its workers
      run_ctx.perf = perf.get();
    }
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
                               std::to_string(cpu.logical_cores) + "T), " + "DynaLM " + ENGINE_VERSION_STRING;
  const std::string model = std::filesystem::path(path).filename().string();
  std::printf("target: %s\nmodel: %s\nhardware: %s\n", target->name().c_str(), model.c_str(), hardware.c_str());
  if (run_ctx.peak_bw_gbs > 0) std::printf("measured DRAM read ceiling: %.1f GB/s\n", run_ctx.peak_bw_gbs);
  if (perf && !perf->hardware_available()) {
    std::printf("hardware counters: unavailable (%s)\n", perf->hardware_reason().c_str());
  }
  std::printf("\n");
  std::printf("%4s %6s %5s | %8s %8s | %8s %8s %8s | %7s %7s %7s | %7s %7s | %6s\n", "conc", "prompt", "out",
              "out t/s", "in t/s", "TTFT p50", "p90", "p99", "ITL p50", "p90", "p99", "TPOT50", "E2E p99", "err");
  std::ofstream jsonl;
  if (!out_file.empty()) jsonl.open(out_file, std::ios::app);

  if (!prompt_mix.empty()) prompts = {0};  // one pass; the mix sets each request's length
  for (int32_t p : prompts) {
    for (int32_t o : outputs) {
      for (int32_t c : conc) {
        bench::PointConfig cfg{c, p, o, requests, prompt_mix};
        bench::PointResult r = bench::run_point(*target, *tokenizer, cfg, run_ctx);
        std::printf("%4d %6d %5d | %8.1f %8.1f | %8.0f %8.0f %8.0f | %7.1f %7.1f %7.1f | %7.1f %7.0f | %6d\n", c,
                    r.cfg.prompt_tokens, o,
                    r.output_tok_s, r.input_tok_s, r.ttft_ms.p50, r.ttft_ms.p90, r.ttft_ms.p99, r.itl_ms.p50,
                    r.itl_ms.p90, r.itl_ms.p99, r.tpot_ms.p50, r.e2e_ms.p99, r.errors);
        std::fflush(stdout);
        if (!r.first_error.empty()) std::fprintf(stderr, "  first error: %s\n", r.first_error.c_str());
        if (const bench::Diagnostics& d = r.diag; d.valid) {
          const double tail = d.forward_ms > 0 && d.pool_tail_wait_ms >= 0 ? 100 * d.pool_tail_wait_ms / d.forward_ms : -1;
          std::printf("     -> %s | decode step %.1f ms x %.1f rows | %.0f regions/step, tail wait %.0f%% | "
                      "est. %.1f of %.1f GB/s | %.0f MHz\n",
                      std::string(bench::bottleneck_name(d.bottleneck.primary)).c_str(), d.decode_step_ms,
                      d.mean_decode_rows, d.steps ? d.pool_regions / static_cast<double>(d.steps) : 0.0, tail,
                      d.est_decode_bw_gbs, d.peak_bw_gbs, d.cpu_mhz);
        }
        if (jsonl) jsonl << bench::to_json(r, model, hardware) << "\n";
      }
    }
  }
  return 0;
}

}  // namespace engine::cli

// `dynalm benchmark <model> [options]`: load test the engine in-process
// or any OpenAI-compatible server (--url), sweeping concurrency x prompt
// length x output length, with P50/P90/P95/P99 latency.

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "bench/loadgen.h"
#include "cli/commands.h"
#include "common/version.h"
#include "loader/model_loader.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/perf_counters.h"
#include "runtime/autotune.h"
#include "runtime/engine.h"
#include "dynacore/execution/thread_pool.h"
#include "common/core.h"

namespace dynalm::cli {
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
               "  --arrival-rate R      open loop: Poisson arrivals at R requests/s, at most --concurrency\n"
               "                        in flight; latencies count from the scheduled arrival (default: closed loop)\n"
               "  --seed N              arrival schedule seed for --arrival-rate (default 1)\n"
               "  --prompt-base N       first prompt index (default 0): give each benchmark process against\n"
               "                        the same server its own range, so no prompt repeats (prompt caches)\n"
               "  --arrival-trace FILE  open loop with explicit arrivals: \"offset_ms,prompt_tokens\" per line\n"
               "                        (replaces --prompt/--prompt-mix/--requests and --arrival-rate)\n"
               "  --request-log FILE    append one JSON line per request (scheduled, dispatch, TTFT, E2E, ...)\n"
               "  --warmup-requests N   full-size uncounted requests before each point (one at a time,\n"
               "                        longest prompt first; default 0)\n"
               "  --url http://H:P      benchmark an OpenAI-compatible server instead of in-process\n"
               "  --model-name NAME     model field sent to --url (default: file stem)\n"
               "  -t, --threads N       compute threads (in-process)\n"
               "  -c, --ctx N           KV capacity in tokens (in-process, default 32768)\n"
               "  --int8-decode N       int8 activations for matmuls of <= N rows (in-process; 0 = off)\n"
               "  --prefill-budget N    prompt tokens per scheduler step (in-process, default 64; Metal 256)\n"
               "  --decode-budget N     decode rows per scheduler step (in-process, default 64)\n"
               "  --chunk N             max prompt tokens per sequence per step (0 = budget; default 32; Metal 128)\n"
               "  --solo-prefill-chunk N  EXPERIMENTAL: chunk and budget N in steps without decode rows\n"
               "  --spec ngram|DRAFT    speculative decoding for every stream (in-process): prompt lookup,\n"
               "                        or a draft model file sharing the vocabulary\n"
               "  --spec-k K            with --spec: most drafts per stream and step (default 4)\n"
               "                        (in-process; 0 = off; default off on the CPU, 256 on Metal)\n"
               "  --solo-prefill-split  with --solo-prefill-chunk: split it across prefilling sequences\n"
               "  --prefill-order O     EXPERIMENTAL: fifo (default) | srpt (shortest remaining prompt first)\n"
               "  --prefill-aging-ms N  with srpt: prompts prefilling longer than N ms go first (default 20000)\n"
               "  --step-trace FILE     append the scheduler's per-step and per-request trace (JSON lines)\n"
               "  --batch N             max tokens per forward pass (default: budgets + 1, min 256 and the solo chunk)\n"
               "  --policy P            balanced | latency | throughput (in-process, DD-061)\n"
               "  --execution MODE      auto (default: compiled on GPUs and qmm CPUs) | reference | compiled (in-process; DD-072)\n"
               "  --gpu / --cpu         in-process device (default auto: GPU first, else CPU)\n"
               "  --backend B           auto | gpu | cpu | ggml-cpu | metal | vulkan | cuda (in-process)\n"
               "  --mode M              auto | performance | exact | safe (in-process; see dynalm autotune)\n"
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
  std::string path, url, model_name, out_file, trace_file, request_log_file;
  std::vector<int32_t> conc = {1, 4}, prompts = {128, 512}, outputs = {128}, prompt_mix;
  bool diag = true;
  int requests = 0, threads = 0, ctx = 32768, int8_rows = -1;
  int prefill_budget = -1, decode_budget = -1, chunk = -1, batch = -1, solo_chunk = -1;
  bool spec_on = false;  // --spec ngram|DRAFT.gguf
  std::string draft_path;
  int32_t spec_k = 4;
  bool solo_split = false;
  PrefillOrder prefill_order = PrefillOrder::kFifo;
  int prefill_aging_ms = -1;
  std::string step_trace_file;
  double arrival_rate = 0;
  int seed = 1, warmup_requests = 0, prompt_base = 0;
  SchedulerPolicy policy = SchedulerPolicy::kBalanced;
  ExecutionMode execution = ExecutionMode::kAuto;
  EngineOptions device;  // --backend / --cpu / --gpu (default auto)
  (void)apply_backend_choice("auto", device);
  TuneMode mode = TuneMode::kAuto;
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
    else if (a == "--arrival-rate") {
      const std::string v(value());
      char* end = nullptr;
      arrival_rate = std::strtod(v.c_str(), &end);
      ok = !v.empty() && end == v.c_str() + v.size() && arrival_rate > 0;
    }
    else if (a == "--seed") ok = parse_int(value(), seed) && seed >= 0;
    else if (a == "--prompt-base") ok = parse_int(value(), prompt_base) && prompt_base >= 0 && prompt_base <= 1000000000;
    else if (a == "--warmup-requests") ok = parse_int(value(), warmup_requests) && warmup_requests >= 0;
    else if (a == "--arrival-trace") ok = !(trace_file = std::string(value())).empty();
    else if (a == "--request-log") ok = !(request_log_file = std::string(value())).empty();
    else if (a == "--url") url = value();
    else if (a == "--model-name") model_name = value();
    else if (a == "-t" || a == "--threads") ok = parse_int(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int(value(), ctx);
    else if (a == "--int8-decode") ok = parse_int(value(), int8_rows) && int8_rows >= 0;
    else if (a == "--prefill-budget") ok = parse_int(value(), prefill_budget) && prefill_budget > 0;
    else if (a == "--decode-budget") ok = parse_int(value(), decode_budget) && decode_budget > 0;
    else if (a == "--chunk") ok = parse_int(value(), chunk) && chunk >= 0;
    else if (a == "--solo-prefill-chunk") ok = parse_int(value(), solo_chunk) && solo_chunk >= 0;
    else if (a == "--spec") {
      const std::string v(value());
      spec_on = !v.empty() && v != "off";
      if (spec_on && v != "ngram") draft_path = v;  // a draft model file
    }
    else if (a == "--spec-k") ok = parse_int(value(), spec_k) && spec_k >= 1 && spec_k <= 16;
    else if (a == "--solo-prefill-split") solo_split = true;
    else if (a == "--prefill-order") ok = parse_prefill_order(value(), prefill_order);
    else if (a == "--prefill-aging-ms") ok = parse_int(value(), prefill_aging_ms) && prefill_aging_ms >= 0;
    else if (a == "--step-trace") ok = !(step_trace_file = std::string(value())).empty();
    else if (a == "--batch") ok = parse_int(value(), batch) && batch > 0;
    else if (a == "--policy") ok = parse_scheduler_policy(value(), policy);
    else if (a == "--execution") {
      auto x = parse_execution_mode(value());
      ok = x.ok();
      if (ok) execution = *x;
    } else if (a == "--backend") {
      if (Status st = apply_backend_choice(value(), device); !st.ok()) {
        std::fprintf(stderr, "benchmark: %s\n", st.message().c_str());
        ok = false;
      }
    } else if (a == "--cpu" || a == "--gpu") {
      (void)apply_backend_choice(a.substr(2), device);
    } else if (a == "--mode") {
      auto m = parse_tune_mode(value());
      ok = m.ok();
      if (ok) mode = *m;
      else std::fprintf(stderr, "%s: %s\n", "benchmark", m.status().message().c_str());
    }
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
  // A trace run must never fall back to closed-loop or Poisson arrivals.
  std::vector<bench::PointConfig::TraceEntry> arrival_trace;
  if (!trace_file.empty()) {
    if (arrival_rate > 0) {
      std::fprintf(stderr, "benchmark: --arrival-trace and --arrival-rate are exclusive\n");
      return 1;
    }
    auto t = bench::load_arrival_trace(trace_file);
    if (!t.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", t.status().message().c_str());
      return 1;
    }
    arrival_trace = std::move(t).value();
  }
  std::ofstream step_trace_log;
  if (!step_trace_file.empty()) {
    if (!url.empty()) {
      std::fprintf(stderr, "benchmark: --step-trace needs the in-process engine (no --url)\n");
      return 1;
    }
    step_trace_log.open(step_trace_file, std::ios::app);
    if (!step_trace_log) {
      std::fprintf(stderr, "benchmark: cannot open --step-trace %s\n", step_trace_file.c_str());
      return 1;
    }
  }
  std::ofstream request_log;
  if (!request_log_file.empty()) {
    request_log.open(request_log_file, std::ios::app);
    if (!request_log) {
      std::fprintf(stderr, "benchmark: cannot open --request-log %s\n", request_log_file.c_str());
      return 1;
    }
  }
  if (auto resolved = ensure_model(path, true); resolved.ok()) {
    path = *resolved;
  } else {
    std::fprintf(stderr, "benchmark: %s\n", resolved.status().message().c_str());
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
    o.execution = execution;
    o.backend = device.backend;
    o.backend_auto = device.backend_auto;
    o.backend_require_gpu = device.backend_require_gpu;
    o.cpu_ggml = device.cpu_ggml;
    o.model_path = path;
    o.threads = threads;
    o.kv_tokens = ctx;
    o.int8_decode_rows = int8_rows;
    if (Status st = apply_tune_mode(mode, o); !st.ok()) {
      std::fprintf(stderr, "benchmark: %s\n", st.message().c_str());
      return 1;
    }
    o.scheduler.policy = policy;
    // Explicit step budgets replace the device's scheduling profile.
    if (prefill_budget > 0 || decode_budget > 0 || chunk >= 0) o.device_scheduler_defaults = false;
    if (prefill_budget > 0) o.scheduler.prefill_token_budget = prefill_budget;
    if (decode_budget > 0) o.scheduler.decode_token_budget = decode_budget;
    if (chunk >= 0) o.scheduler.max_prefill_chunk = chunk;
    if (solo_chunk >= 0) o.scheduler.solo_prefill_chunk = solo_chunk;
    o.scheduler.spec_k = spec_on ? spec_k : 0;
    o.draft_model_path = draft_path;
    if (solo_split) o.scheduler.solo_prefill_split = true;
    o.scheduler.prefill_order = prefill_order;
    if (prefill_aging_ms >= 0) o.scheduler.prefill_aging_ms = prefill_aging_ms;
    if (!step_trace_file.empty()) o.scheduler.trace = true;
    o.max_batch_tokens = batch > 0 ? batch : 0;  // 0 = the device default (Engine)
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

  if (!prompt_mix.empty() || !arrival_trace.empty()) prompts = {0};  // one pass; the mix / trace sets each length
  for (int32_t p : prompts) {
    for (int32_t o : outputs) {
      for (int32_t c : conc) {
        bench::PointConfig cfg{c, p, o, requests, prompt_mix};
        cfg.arrival_rate = arrival_rate;
        cfg.seed = static_cast<uint32_t>(seed);
        cfg.prompt_base = prompt_base;
        cfg.warmup_requests = warmup_requests;
        cfg.arrival_trace = arrival_trace;
        cfg.arrival_trace_path = trace_file;
        bench::PointResult r = bench::run_point(*target, *tokenizer, cfg, run_ctx);
        if (!arrival_trace.empty() && r.arrival_mode != "trace") {
          std::fprintf(stderr, "benchmark: internal error: trace run measured as '%s'\n", r.arrival_mode.c_str());
          return 1;
        }
        std::printf("%4d %6d %5d | %8.1f %8.1f | %8.0f %8.0f %8.0f | %7.1f %7.1f %7.1f | %7.1f %7.0f | %6d\n", c,
                    r.cfg.prompt_tokens, o,
                    r.output_tok_s, r.input_tok_s, r.ttft_ms.p50, r.ttft_ms.p90, r.ttft_ms.p99, r.itl_ms.p50,
                    r.itl_ms.p90, r.itl_ms.p99, r.tpot_ms.p50, r.e2e_ms.p99, r.errors);
        std::fflush(stdout);
        if (!r.first_error.empty()) std::fprintf(stderr, "  first error: %s\n", r.first_error.c_str());
        for (const bench::PromptGroup& g : r.by_prompt) {
          std::printf("     prompt %5d: %3d done | TTFT p50 %8.0f p95 %8.0f max %8.0f | ITL p99 %7.1f max %7.1f\n",
                      g.prompt_tokens, g.completed, g.ttft_ms.p50, g.ttft_ms.p95, g.ttft_ms.max, g.itl_ms.p99,
                      g.itl_ms.max);
        }
        if (r.arrival_mode != "closed") {
          std::printf("     open loop (%s, digest %s): scheduled %.4f req/s, achieved %.4f req/s; start delay p50 %.1f "
                      "p99 %.1f max %.1f ms; %d cap-bound (no free slot at arrival); ITL max %.1f ms\n",
                      r.arrival_mode.c_str(), r.arrival_digest.c_str(), r.trace_arrival_rate, r.achieved_arrival_rate,
                      r.start_delay_ms.p50, r.start_delay_ms.p99, r.start_delay_ms_max, r.cap_bound_requests, r.itl_ms.max);
        }
        if (request_log) {
          for (const bench::RequestResult& q : r.requests) request_log << bench::request_to_json(r, q) << "\n";
          request_log.flush();
        }
        if (step_trace_log) {
          for (const std::string& line : bench::trace_to_json_lines(r)) step_trace_log << line << "\n";
          step_trace_log.flush();
        }
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

}  // namespace dynalm::cli

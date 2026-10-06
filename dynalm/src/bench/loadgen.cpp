#include "bench/loadgen.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

#include "api/json.h"
#include "dynacore/base/timer.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/process_stats.h"
#include "dynacore/tensor/dtype.h"

namespace engine::bench {

Percentiles percentiles(std::vector<double> v) {
  Percentiles p;
  if (v.empty()) return p;
  std::sort(v.begin(), v.end());
  auto at = [&](double q) {
    const auto rank = static_cast<size_t>(std::ceil(q / 100.0 * static_cast<double>(v.size())));
    return v[std::clamp<size_t>(rank, 1, v.size()) - 1];
  };
  p.p50 = at(50);
  p.p90 = at(90);
  p.p95 = at(95);
  p.p99 = at(99);
  double sum = 0;
  for (double x : v) sum += x;
  p.mean = sum / static_cast<double>(v.size());
  return p;
}

// --- in-process target ----------------------------------------------------------

namespace {

class EngineTarget final : public Target {
 public:
  explicit EngineTarget(Engine& e) : engine_(e) {}
  std::string name() const override { return "dynalm (in-process)"; }
  bool in_process() const override { return true; }
  Engine* engine() override { return &engine_; }

  RequestResult run(const std::string& prompt, int32_t max_tokens) override {
    RequestResult r;
    GenerateParams p;
    p.max_tokens = max_tokens;
    p.stop_at_eog = false;  // fixed output length
    const int64_t t0 = now_ns();
    auto s = engine_.generate_text(prompt, /*parse_special=*/false, p);
    if (!s.ok()) {
      r.error = s.status().to_string();
      return r;
    }
    StreamEvent ev;
    int64_t last = 0;
    bool first = true;
    while ((*s)->next(ev, std::chrono::milliseconds(600000))) {
      if (!ev.text.empty()) {
        const int64_t now = now_ns();
        if (first) {
          r.ttft_ms = static_cast<double>(now - t0) * 1e-6;
          first = false;
        } else {
          r.gaps_ms.push_back(static_cast<double>(now - last) * 1e-6);
        }
        last = now;
      }
      if (ev.done) break;
    }
    r.e2e_ms = static_cast<double>(now_ns() - t0) * 1e-6;
    if (first) r.ttft_ms = r.e2e_ms;
    r.prompt_tokens = ev.prompt_tokens;
    r.completion_tokens = ev.completion_tokens;
    r.ok = ev.done && (ev.finish == StreamFinish::kLength || ev.finish == StreamFinish::kStop);
    if (!r.ok) r.error = ev.done ? ev.error.to_string() : "timeout";
    return r;
  }

 private:
  Engine& engine_;
};

}  // namespace

std::unique_ptr<Target> make_engine_target(Engine& engine) { return std::make_unique<EngineTarget>(engine); }

// --- workload -----------------------------------------------------------------------

std::string make_prompt(const Tokenizer& tok, int32_t tokens, int32_t request_index) {
  // Varied natural-language filler; unique leading text per request so no
  // request benefits from another's prefix cache.
  static constexpr const char* kWords[] = {
      "river", "copper", "lantern", "orbit", "meadow", "signal", "harbor", "violet", "engine", "summit",
      "canvas", "thunder", "pocket", "glacier", "ribbon", "falcon", "marble", "compass", "ember", "willow"};
  std::string text = "Request " + std::to_string(request_index) + ": write a long, detailed story about";
  uint32_t state = 2166136261u ^ static_cast<uint32_t>(request_index);
  while (static_cast<int32_t>(tok.encode(text, true, false).size()) < tokens + 8) {
    for (int i = 0; i < 16; ++i) {
      state = state * 1664525u + 1013904223u;
      text += ' ';
      text += kWords[(state >> 16) % std::size(kWords)];
    }
  }
  // Trim to exactly `tokens` (decode the first `tokens` ids back to text).
  std::vector<TokenId> ids = tok.encode(text, true, false);
  const size_t bos = (!ids.empty() && ids.front() == tok.bos()) ? 1 : 0;
  ids.resize(std::min(ids.size(), static_cast<size_t>(tokens) + bos));
  return tok.decode(std::span<const TokenId>(ids).subspan(bos));
}

namespace {

// Waits until the engine has retired every request of the point, so its
// published stats include the last step.
EngineStats settled_stats(const Engine& e) {
  EngineStats s = e.stats();
  for (int i = 0; i < 400 && (s.scheduler.running != 0 || s.scheduler.waiting != 0); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    s = e.stats();
  }
  return s;
}

Diagnostics diagnose(const Engine& engine, const EngineStats& a, const EngineStats& b, const PerfSample& perf,
                     double cpu_mhz, double wall_s, const PointConfig& cfg, const RunContext& ctx) {
  Diagnostics d;
  d.valid = true;
  const SchedulerStats& x = a.scheduler;
  const SchedulerStats& y = b.scheduler;
  d.steps = y.steps - x.steps;
  d.steps_decode_only = y.steps_decode_only - x.steps_decode_only;
  d.steps_prefill_only = y.steps_prefill_only - x.steps_prefill_only;
  d.steps_mixed = y.steps_mixed - x.steps_mixed;
  const uint64_t decode_steps = d.steps_decode_only + d.steps_mixed;
  d.mean_decode_rows = decode_steps ? static_cast<double>(y.decode_rows_total - x.decode_rows_total) / decode_steps : 0;
  d.plan_ms = y.plan_ms - x.plan_ms;
  d.prefix_lookup_ms = y.prefix_lookup_ms - x.prefix_lookup_ms;
  d.kv_reserve_ms = y.kv_reserve_ms - x.kv_reserve_ms;
  d.forward_ms = y.forward_ms - x.forward_ms;
  d.forward_decode_only_ms = y.forward_decode_only_ms - x.forward_decode_only_ms;
  d.forward_prefill_only_ms = y.forward_prefill_only_ms - x.forward_prefill_only_ms;
  d.forward_mixed_ms = y.forward_mixed_ms - x.forward_mixed_ms;
  d.decode_step_ms = d.steps_decode_only ? d.forward_decode_only_ms / static_cast<double>(d.steps_decode_only) : 0;
  d.sample_ms = y.sample_ms - x.sample_ms;
  d.emit_ms = y.emit_ms - x.emit_ms;
  d.prefix_insert_ms = y.prefix_insert_ms - x.prefix_insert_ms;
  d.tokenize_ms = b.tokenize_ms - a.tokenize_ms;
  const uint64_t admitted = y.admitted - x.admitted;
  d.queue_wait_mean_ms = admitted ? (y.queue_ms_total - x.queue_ms_total) / static_cast<double>(admitted) : 0;
  if (b.profiling && a.profiling) {
    for (size_t op = 0; op < b.forward.ns.size(); ++op) {
      const double ms = static_cast<double>(b.forward.ns[op] - a.forward.ns[op]) * 1e-6;
      if (ms > 0) d.op_ms.emplace_back(std::string(forward_op_name(static_cast<ForwardOp>(op))), ms);
    }
    d.pool_regions = static_cast<double>(b.pool.regions - a.pool.regions);
    d.pool_region_ms = static_cast<double>(b.pool.region_ns - a.pool.region_ns) * 1e-6;
    d.pool_tail_wait_ms = static_cast<double>(b.pool.tail_wait_ns - a.pool.tail_wait_ns) * 1e-6;
    d.pool_sleeps = static_cast<double>(b.pool.sleeps - a.pool.sleeps);
  }
  d.perf = perf;
  d.cpu_mhz = cpu_mhz;

  // Modelled decode traffic: weights for the step's rows, plus each row's KV
  // at the point's mean context (prompt + half the output).
  const LoadedModel& m = engine.model();
  const KvGeometry& g = engine.kv_geometry();
  const double ctx_len = cfg.prompt_tokens + 0.5 * cfg.output_tokens;
  const double rows = std::max(1.0, d.mean_decode_rows);
  const double step_bytes =
      decode_weight_bytes_per_step(m, rows) +
      rows * kv_bytes_per_token_read(g.num_layers, g.num_kv_heads, g.head_dim, g.head_dim_v,
                                     static_cast<int32_t>(dtype_block_bytes(g.dtype)), ctx_len);
  if (d.forward_decode_only_ms > 0) {
    d.est_decode_bw_gbs = step_bytes * static_cast<double>(d.steps_decode_only) / (d.forward_decode_only_ms * 1e-3) / 1e9;
  }
  d.peak_bw_gbs = ctx.peak_bw_gbs;

  ClassifierInputs in;
  in.wall_s = wall_s;
  in.forward_s = d.forward_ms * 1e-3;
  in.decode_forward_s = d.forward_decode_only_ms * 1e-3;
  in.prefill_forward_s = (d.forward_prefill_only_ms + d.forward_mixed_ms) * 1e-3;
  in.host_s = (d.plan_ms + d.sample_ms + d.emit_ms + d.prefix_insert_ms) * 1e-3;
  in.est_decode_bw_gbs = d.est_decode_bw_gbs;
  in.peak_bw_gbs = d.peak_bw_gbs;
  if (d.pool_regions >= 0) {
    in.pool_region_s = d.pool_region_ms * 1e-3;
    in.pool_tail_wait_s = d.pool_tail_wait_ms * 1e-3;
    in.pool_regions = d.pool_regions;
    in.regions_per_step = d.steps ? d.pool_regions / static_cast<double>(d.steps) : -1;
  }
  if (perf.major_faults >= 0) in.major_fault_mb_s = static_cast<double>(perf.major_faults) * 4096.0 / wall_s / 1e6;
  if (perf.cycles > 0 && perf.instructions >= 0) {
    in.ipc = static_cast<double>(perf.instructions) / static_cast<double>(perf.cycles);
  }
  if (perf.cache_references > 0 && perf.cache_misses >= 0) {
    in.llc_miss_ratio = static_cast<double>(perf.cache_misses) / static_cast<double>(perf.cache_references);
  }
  d.bottleneck = classify(in);
  return d;
}

}  // namespace

PointResult run_point(Target& target, const Tokenizer& tokenizer, const PointConfig& cfg_in, const RunContext& ctx) {
  PointConfig cfg = cfg_in;
  if (cfg.requests <= 0) cfg.requests = std::max(4, 2 * cfg.concurrency);
  auto prompt_len = [&](int32_t i) {
    return cfg.prompt_mix.empty() ? cfg.prompt_tokens : cfg.prompt_mix[static_cast<size_t>(i) % cfg.prompt_mix.size()];
  };
  if (!cfg.prompt_mix.empty()) {
    int64_t sum = 0;
    for (int32_t i = 0; i < cfg.requests; ++i) sum += prompt_len(i);
    cfg.prompt_tokens = static_cast<int32_t>(sum / cfg.requests);
  }
  PointResult out;
  out.cfg = cfg;
  out.target = target.name();

  // Prompt indices never repeat within a process: a later point must not hit
  // the prefix cache (ours or the server's) warmed by an earlier point.
  static std::atomic<int32_t> next_prompt{0};
  const int32_t base = next_prompt.fetch_add(cfg.requests + 1);
  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(cfg.requests));
  for (int32_t i = 0; i < cfg.requests; ++i) prompts.push_back(make_prompt(tokenizer, prompt_len(i), base + i));

  Engine* engine = target.engine();
  // Per-op + thread-pool accounting (a few clock reads per op), with diagnostics only.
  if (engine) engine->set_profiling(ctx.perf != nullptr);

  // Uncounted warm-up: pages in mmapped weights / server caches so the first
  // measured request doesn't carry cold-start cost.
  (void)target.run(make_prompt(tokenizer, std::min<int32_t>(cfg.prompt_tokens, 32), base + cfg.requests), 4);

  const EngineStats before = engine ? settled_stats(*engine) : EngineStats{};
  const PerfSample perf0 = ctx.perf ? ctx.perf->read() : PerfSample{};

  std::mutex mu;
  std::vector<RequestResult> results;
  std::atomic<int32_t> next{0};
  std::atomic<bool> done{false};
  // CPU clock sampled through the run (frequency drops under sustained load).
  std::vector<double> mhz;
  // Only with diagnostics: on Windows the clock query may interrupt cores.
  std::thread clock_sampler([&] {
    while (ctx.perf && !done.load()) {
      if (const double f = cpu_current_mhz(); f > 0) mhz.push_back(f);
      for (int i = 0; i < 25 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  const double cpu0 = process_cpu_seconds();
  const int64_t t0 = now_ns();
  int64_t peak_rss = 0;
  std::vector<std::thread> clients;
  for (int32_t c = 0; c < cfg.concurrency; ++c) {
    clients.emplace_back([&] {
      for (int32_t i; (i = next.fetch_add(1)) < cfg.requests;) {
        RequestResult r = target.run(prompts[static_cast<size_t>(i)], cfg.output_tokens);
        std::lock_guard<std::mutex> lock(mu);
        if (target.in_process()) peak_rss = std::max(peak_rss, process_rss_bytes());
        results.push_back(std::move(r));
      }
    });
  }
  for (auto& t : clients) t.join();
  out.wall_s = static_cast<double>(now_ns() - t0) * 1e-9;
  const double cpu1 = process_cpu_seconds();
  done.store(true);
  clock_sampler.join();

  std::vector<double> ttft, itl, tpot, e2e;
  int64_t out_tokens = 0, in_tokens = 0;
  for (const RequestResult& r : results) {
    if (!r.ok) {
      ++out.errors;
      if (out.first_error.empty()) out.first_error = r.error;
      continue;
    }
    ++out.completed;
    ttft.push_back(r.ttft_ms);
    e2e.push_back(r.e2e_ms);
    itl.insert(itl.end(), r.gaps_ms.begin(), r.gaps_ms.end());
    if (r.completion_tokens > 1) tpot.push_back((r.e2e_ms - r.ttft_ms) / (r.completion_tokens - 1));
    out_tokens += r.completion_tokens;
    in_tokens += r.prompt_tokens > 0 ? r.prompt_tokens : cfg.prompt_tokens;
  }
  out.ttft_ms = percentiles(ttft);
  out.itl_ms = percentiles(itl);
  out.tpot_ms = percentiles(tpot);
  out.e2e_ms = percentiles(e2e);
  out.output_tok_s = static_cast<double>(out_tokens) / out.wall_s;
  out.input_tok_s = static_cast<double>(in_tokens) / out.wall_s;
  out.mean_completion_tokens = out.completed ? static_cast<double>(out_tokens) / out.completed : 0;
  if (target.in_process()) {
    out.rss_mb = static_cast<double>(peak_rss) / (1024.0 * 1024.0);
    out.cpu_util = (cpu1 - cpu0) / out.wall_s / cpu_info().logical_cores;
  }
  if (engine) {
    const EngineStats after = settled_stats(*engine);
    const PerfSample perf = ctx.perf ? PerfSample::delta(perf0, ctx.perf->read()) : PerfSample{};
    double mean_mhz = -1;
    if (!mhz.empty()) {
      mean_mhz = 0;
      for (double f : mhz) mean_mhz += f;
      mean_mhz /= static_cast<double>(mhz.size());
    }
    out.diag = diagnose(*engine, before, after, perf, mean_mhz, out.wall_s, cfg, ctx);
  }
  return out;
}

std::string to_json(const PointResult& r, const std::string& model, const std::string& hardware) {
  auto pct = [](const Percentiles& p) {
    json::Object o;
    o["p50"] = p.p50;
    o["p90"] = p.p90;
    o["p95"] = p.p95;
    o["p99"] = p.p99;
    o["mean"] = p.mean;
    return json::Value(std::move(o));
  };
  json::Object o;
  o["target"] = r.target;
  o["model"] = model;
  o["hardware"] = hardware;
  o["concurrency"] = r.cfg.concurrency;
  o["prompt_tokens"] = r.cfg.prompt_tokens;
  o["output_tokens"] = r.cfg.output_tokens;
  o["requests"] = r.cfg.requests;
  o["completed"] = r.completed;
  o["errors"] = r.errors;
  o["wall_s"] = r.wall_s;
  o["output_tok_s"] = r.output_tok_s;
  o["input_tok_s"] = r.input_tok_s;
  o["mean_completion_tokens"] = r.mean_completion_tokens;
  o["ttft_ms"] = pct(r.ttft_ms);
  o["itl_ms"] = pct(r.itl_ms);
  o["tpot_ms"] = pct(r.tpot_ms);
  o["e2e_ms"] = pct(r.e2e_ms);
  o["rss_mb"] = r.rss_mb;
  o["cpu_util"] = r.cpu_util;
  if (!r.first_error.empty()) o["first_error"] = r.first_error;
  if (!r.cfg.prompt_mix.empty()) {
    json::Array mix;
    for (int32_t v : r.cfg.prompt_mix) mix.push_back(v);
    o["prompt_mix"] = std::move(mix);
  }
  if (const Diagnostics& d = r.diag; d.valid) {
    auto num_or_null = [](double v) { return v < 0 ? json::Value() : json::Value(v); };
    auto cnt_or_null = [](int64_t v) { return v < 0 ? json::Value() : json::Value(static_cast<double>(v)); };
    json::Object g;
    g["steps"] = static_cast<double>(d.steps);
    g["steps_decode_only"] = static_cast<double>(d.steps_decode_only);
    g["steps_prefill_only"] = static_cast<double>(d.steps_prefill_only);
    g["steps_mixed"] = static_cast<double>(d.steps_mixed);
    g["mean_decode_rows"] = d.mean_decode_rows;
    g["decode_step_ms"] = d.decode_step_ms;
    json::Object ms;
    ms["plan"] = d.plan_ms;
    ms["prefix_lookup"] = d.prefix_lookup_ms;
    ms["kv_reserve"] = d.kv_reserve_ms;
    ms["forward"] = d.forward_ms;
    ms["forward_decode_only"] = d.forward_decode_only_ms;
    ms["forward_prefill_only"] = d.forward_prefill_only_ms;
    ms["forward_mixed"] = d.forward_mixed_ms;
    ms["sample"] = d.sample_ms;
    ms["emit"] = d.emit_ms;
    ms["prefix_insert"] = d.prefix_insert_ms;
    ms["tokenize"] = d.tokenize_ms;
    g["time_ms"] = std::move(ms);
    g["queue_wait_mean_ms"] = d.queue_wait_mean_ms;
    json::Object ops;
    for (const auto& [name, v] : d.op_ms) ops[name] = v;
    g["op_ms"] = std::move(ops);
    json::Object pool;
    pool["regions"] = num_or_null(d.pool_regions);
    pool["region_ms"] = num_or_null(d.pool_region_ms);
    pool["tail_wait_ms"] = num_or_null(d.pool_tail_wait_ms);
    pool["sleeps"] = num_or_null(d.pool_sleeps);
    g["pool"] = std::move(pool);
    json::Object pc;
    pc["cycles"] = cnt_or_null(d.perf.cycles);
    pc["instructions"] = cnt_or_null(d.perf.instructions);
    pc["llc_references"] = cnt_or_null(d.perf.cache_references);
    pc["llc_misses"] = cnt_or_null(d.perf.cache_misses);
    pc["l1d_misses"] = cnt_or_null(d.perf.l1d_misses);
    pc["branch_misses"] = cnt_or_null(d.perf.branch_misses);
    pc["cpu_migrations"] = cnt_or_null(d.perf.cpu_migrations);
    pc["context_switches"] = cnt_or_null(d.perf.context_switches);
    pc["page_faults"] = cnt_or_null(d.perf.page_faults);
    pc["major_faults"] = cnt_or_null(d.perf.major_faults);
    g["counters"] = std::move(pc);
    g["cpu_mhz"] = num_or_null(d.cpu_mhz);
    g["est_decode_bw_gbs"] = num_or_null(d.est_decode_bw_gbs);
    g["peak_bw_gbs"] = num_or_null(d.peak_bw_gbs);
    json::Object bn;
    bn["primary"] = std::string(bottleneck_name(d.bottleneck.primary));
    json::Array all, ev;
    for (Bottleneck b : d.bottleneck.all) all.push_back(std::string(bottleneck_name(b)));
    for (const std::string& e : d.bottleneck.evidence) ev.push_back(e);
    bn["all"] = std::move(all);
    bn["evidence"] = std::move(ev);
    g["bottleneck"] = std::move(bn);
    o["diag"] = std::move(g);
  }
  return json::dump(o);
}

}  // namespace engine::bench

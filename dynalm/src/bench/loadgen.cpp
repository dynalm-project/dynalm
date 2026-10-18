#include "bench/loadgen.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <thread>

#include "api/json.h"
#include "dynacore/base/timer.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/process_stats.h"
#include "dynacore/tensor/dtype.h"
#include "common/core.h"

namespace dynalm::bench {

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
  p.max = v.back();
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
  d.spec_drafted = y.spec_drafted - x.spec_drafted;
  d.spec_accepted = y.spec_accepted - x.spec_accepted;
  d.steps_solo_prefill = y.steps_solo_prefill - x.steps_solo_prefill;
  d.steps_solo_prefill_skipped = y.steps_solo_prefill_skipped - x.steps_solo_prefill_skipped;
  d.steps_prefill_srpt = y.steps_prefill_srpt - x.steps_prefill_srpt;
  d.forward_solo_prefill_ms = y.forward_solo_prefill_ms - x.forward_solo_prefill_ms;
  d.solo_prefill_step_ms_max = y.solo_prefill_step_ms_max;
  d.solo_prefill_chunk = y.solo_prefill_chunk;
  d.solo_prefill_split = y.solo_prefill_split;
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
    if (d.pool_region_ms > 0 && a.pool.busy_ns.size() == b.pool.busy_ns.size()) {
      for (size_t t = 0; t < b.pool.busy_ns.size(); ++t) {
        d.pool_thread_util.push_back(static_cast<double>(b.pool.busy_ns[t] - a.pool.busy_ns[t]) * 1e-6 / d.pool_region_ms);
      }
    }
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
  const bool traced = !cfg.arrival_trace.empty();
  if (traced) cfg.requests = static_cast<int32_t>(cfg.arrival_trace.size());  // the trace defines the requests
  if (cfg.requests <= 0) cfg.requests = std::max(4, 2 * cfg.concurrency);
  auto prompt_len = [&](int32_t i) {
    if (traced) return cfg.arrival_trace[static_cast<size_t>(i)].prompt_tokens;
    return cfg.prompt_mix.empty() ? cfg.prompt_tokens : cfg.prompt_mix[static_cast<size_t>(i) % cfg.prompt_mix.size()];
  };
  if (!cfg.prompt_mix.empty() || traced) {
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
  const int32_t base = next_prompt.fetch_add(cfg.requests + 1 + std::max(0, cfg.warmup_requests));
  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(cfg.requests));
  for (int32_t i = 0; i < cfg.requests; ++i) prompts.push_back(make_prompt(tokenizer, prompt_len(i), base + i));

  Engine* engine = target.engine();
  // Per-op + thread-pool accounting (a few clock reads per op), with diagnostics only.
  if (engine) engine->set_profiling(ctx.perf != nullptr);

  // Uncounted warm-up: pages in mmapped weights / server caches so the first
  // measured request doesn't carry cold-start cost.
  (void)target.run(make_prompt(tokenizer, std::min<int32_t>(cfg.prompt_tokens, 32), base + cfg.requests), 4);
  // Optional full-size warm-up (uncounted, one at a time): brings clocks and
  // caches to the point's steady state. Request k uses the point's k-th
  // longest distinct prompt length (cycling) and the full output length.
  if (cfg.warmup_requests > 0) {
    std::vector<int32_t> lens;
    for (int32_t i = 0; i < cfg.requests; ++i) lens.push_back(prompt_len(i));
    std::sort(lens.begin(), lens.end(), std::greater<>());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    for (int32_t k = 0; k < cfg.warmup_requests; ++k) {
      const int32_t len = lens[static_cast<size_t>(k) % lens.size()];
      (void)target.run(make_prompt(tokenizer, len, base + cfg.requests + 1 + k), cfg.output_tokens);
    }
  }

  const EngineStats before = engine ? settled_stats(*engine) : EngineStats{};
  const std::vector<dynacore::MatmulKernelStats> mm_before = dynacore::CpuDevice::matmul_stats();
  if (engine) {  // drop the warm-up's trace (empty unless the scheduler trace is on)
    (void)engine->take_step_trace();
    (void)engine->take_request_trace();
  }
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
  // Open loop: arrival offsets (ns from t0), fixed before the clock starts:
  // from the trace, or Poisson (exponential gaps from the seed).
  std::vector<int64_t> arrival_ns;
  if (traced) {
    for (const PointConfig::TraceEntry& e : cfg.arrival_trace) arrival_ns.push_back(e.offset_ms * 1000000);
    out.arrival_mode = "trace";
  } else if (cfg.arrival_rate > 0) {
    std::mt19937_64 rng(cfg.seed);
    std::exponential_distribution<double> gap_s(cfg.arrival_rate);
    double t = 0;
    for (int32_t i = 0; i < cfg.requests; ++i) {
      t += gap_s(rng);
      arrival_ns.push_back(static_cast<int64_t>(t * 1e9));
    }
    out.arrival_mode = "poisson";
  }
  {  // digest of the schedule: matched arms must have the same one
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xff;
        h *= 1099511628211ull;
      }
    };
    for (int32_t i = 0; i < cfg.requests; ++i) {
      mix(arrival_ns.empty() ? 0 : static_cast<uint64_t>(arrival_ns[static_cast<size_t>(i)]));
      mix(static_cast<uint64_t>(prompt_len(i)));
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    out.arrival_digest = buf;
  }
  if (arrival_ns.size() > 1 && arrival_ns.back() > arrival_ns.front()) {
    out.trace_arrival_rate = static_cast<double>(arrival_ns.size() - 1) / (static_cast<double>(arrival_ns.back() - arrival_ns.front()) * 1e-9);
  }
  const double cpu0 = process_cpu_seconds();
  const int64_t t0 = now_ns();
  int64_t peak_rss = 0;
  std::vector<std::thread> clients;
  for (int32_t c = 0; c < cfg.concurrency; ++c) {
    clients.emplace_back([&] {
      int32_t served = 0;  // requests this client slot has already sent
      for (int32_t i; (i = next.fetch_add(1)) < cfg.requests; ++served) {
        double delay_ms = 0;
        bool slot_late = false;
        if (!arrival_ns.empty()) {
          // Wait for this request's arrival; if every slot was busy past it,
          // the lateness is part of its latency (no coordinated omission).
          const int64_t due = t0 + arrival_ns[static_cast<size_t>(i)];
          const int64_t picked = now_ns();
          // Cap-bound: this slot was still serving an earlier request at the
          // arrival (picked late). A slot's first pick can only be late by
          // thread start-up, which is not queueing.
          slot_late = served > 0 && picked > due + 1000000;
          // Sleep to ~20 ms before the arrival, then yield until it: a sleep can
          // overshoot by one OS timer period (~15.6 ms on Windows), which would
          // otherwise add up to tens of ms of lateness to the request.
          if (const int64_t wait = due - picked - 20000000; wait > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(wait));
          while (now_ns() < due) std::this_thread::yield();
          delay_ms = std::max<double>(0, static_cast<double>(now_ns() - due) * 1e-6);
        }
        const int64_t dispatch = now_ns();
        RequestResult r = target.run(prompts[static_cast<size_t>(i)], cfg.output_tokens);
        r.index = i;
        r.scheduled_ms = arrival_ns.empty() ? -1.0 : static_cast<double>(arrival_ns[static_cast<size_t>(i)]) * 1e-6;
        r.dispatch_ms = static_cast<double>(dispatch - t0) * 1e-6;
        r.requested_prompt_tokens = prompt_len(i);
        r.start_delay_ms = delay_ms;
        r.cap_bound = slot_late;
        r.ttft_ms += delay_ms;
        r.e2e_ms += delay_ms;
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
  if (!cfg.prompt_mix.empty() || traced) {
    struct Samples {
      std::vector<double> ttft, itl, e2e;
    };
    std::map<int32_t, Samples> groups;
    for (const RequestResult& r : results) {
      if (!r.ok) continue;
      Samples& g = groups[r.requested_prompt_tokens];
      g.ttft.push_back(r.ttft_ms);
      g.e2e.push_back(r.e2e_ms);
      g.itl.insert(g.itl.end(), r.gaps_ms.begin(), r.gaps_ms.end());
    }
    for (auto& [len, g] : groups) {
      out.by_prompt.push_back({len, static_cast<int32_t>(g.ttft.size()), percentiles(g.ttft), percentiles(g.itl),
                               percentiles(g.e2e)});
    }
  }
  if (!arrival_ns.empty() && !results.empty()) {
    std::vector<double> delays;
    double first = 1e300, last = -1e300;
    for (const RequestResult& r : results) {
      out.start_delay_ms_mean += r.start_delay_ms;
      out.start_delay_ms_max = std::max(out.start_delay_ms_max, r.start_delay_ms);
      delays.push_back(r.start_delay_ms);
      if (r.cap_bound) ++out.cap_bound_requests;
      first = std::min(first, r.dispatch_ms);
      last = std::max(last, r.dispatch_ms);
    }
    out.start_delay_ms_mean /= static_cast<double>(results.size());
    out.start_delay_ms = percentiles(delays);
    if (results.size() > 1 && last > first) out.achieved_arrival_rate = static_cast<double>(results.size() - 1) / ((last - first) * 1e-3);
  }
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
    const std::vector<dynacore::MatmulKernelStats> mm_after = dynacore::CpuDevice::matmul_stats();
    const bool timed = dynacore::CpuDevice::matmul_timing();
    for (size_t i = 0; i < mm_after.size() && i < mm_before.size(); ++i) {
      const uint64_t calls = mm_after[i].calls - mm_before[i].calls;
      if (calls == 0) continue;
      Diagnostics::MatmulKernelUse u;
      u.name = std::string(dynacore::matmul_kernel_name(mm_after[i].id));
      u.calls = static_cast<double>(calls);
      u.rows = static_cast<double>(mm_after[i].rows - mm_before[i].rows);
      u.macs = static_cast<double>(mm_after[i].macs - mm_before[i].macs);
      u.ms = timed ? static_cast<double>(mm_after[i].ns - mm_before[i].ns) * 1e-6 : -1.0;
      out.diag.matmul_kernels.push_back(std::move(u));
    }
  }
  out.t0_ns = t0;
  if (engine) {
    out.step_trace = engine->take_step_trace();
    out.request_trace = engine->take_request_trace();
  }
  std::sort(results.begin(), results.end(), [](const RequestResult& a, const RequestResult& b) { return a.index < b.index; });
  for (RequestResult& r : results) r.gaps_ms.clear();  // per-request records keep timings, not every gap
  out.requests = std::move(results);
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
    o["max"] = p.max;
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
  o["warmup_requests"] = r.cfg.warmup_requests;
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
    json::Array groups;
    for (const PromptGroup& g : r.by_prompt) {
      json::Object go;
      go["prompt_tokens"] = g.prompt_tokens;
      go["completed"] = g.completed;
      go["ttft_ms"] = pct(g.ttft_ms);
      go["itl_ms"] = pct(g.itl_ms);
      go["e2e_ms"] = pct(g.e2e_ms);
      groups.push_back(json::Value(std::move(go)));
    }
    o["by_prompt"] = std::move(groups);
  }
  o["arrival"] = r.arrival_mode;
  o["arrival_digest"] = r.arrival_digest;
  if (r.arrival_mode != "closed") {
    if (r.arrival_mode == "poisson") {
      o["arrival_rate"] = r.cfg.arrival_rate;
      o["seed"] = static_cast<double>(r.cfg.seed);
    } else {
      o["arrival_trace"] = r.cfg.arrival_trace_path;
    }
    o["trace_arrival_rate"] = r.trace_arrival_rate;
    o["achieved_arrival_rate"] = r.achieved_arrival_rate;
    o["start_delay_ms_mean"] = r.start_delay_ms_mean;
    o["start_delay_ms_max"] = r.start_delay_ms_max;
    o["start_delay_ms"] = pct(r.start_delay_ms);
    o["cap_bound_requests"] = r.cap_bound_requests;
  }
  if (const Diagnostics& d = r.diag; d.valid) {
    auto num_or_null = [](double v) { return v < 0 ? json::Value() : json::Value(v); };
    auto cnt_or_null = [](int64_t v) { return v < 0 ? json::Value() : json::Value(static_cast<double>(v)); };
    json::Object g;
    g["steps"] = static_cast<double>(d.steps);
    g["steps_decode_only"] = static_cast<double>(d.steps_decode_only);
    g["steps_prefill_only"] = static_cast<double>(d.steps_prefill_only);
    g["steps_mixed"] = static_cast<double>(d.steps_mixed);
    g["spec_drafted"] = static_cast<double>(d.spec_drafted);
    g["spec_accepted"] = static_cast<double>(d.spec_accepted);
    g["mean_decode_rows"] = d.mean_decode_rows;
    g["decode_step_ms"] = d.decode_step_ms;
    g["steps_solo_prefill"] = static_cast<double>(d.steps_solo_prefill);
    g["steps_solo_prefill_skipped"] = static_cast<double>(d.steps_solo_prefill_skipped);
    g["steps_prefill_srpt"] = static_cast<double>(d.steps_prefill_srpt);
    g["forward_solo_prefill_ms"] = d.forward_solo_prefill_ms;
    g["solo_prefill_step_ms_mean"] = d.steps_solo_prefill ? d.forward_solo_prefill_ms / static_cast<double>(d.steps_solo_prefill) : 0.0;
    g["solo_prefill_step_ms_max"] = d.solo_prefill_step_ms_max;
    g["solo_prefill_chunk"] = d.solo_prefill_chunk;
    g["solo_prefill_split"] = d.solo_prefill_split;
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
    json::Object mmk;
    for (const Diagnostics::MatmulKernelUse& u : d.matmul_kernels) {
      json::Object ko;
      ko["calls"] = u.calls;
      ko["rows"] = u.rows;
      ko["macs"] = u.macs;
      ko["ms"] = num_or_null(u.ms);
      mmk[u.name] = std::move(ko);
    }
    g["matmul_kernels"] = std::move(mmk);
    json::Object pool;
    pool["regions"] = num_or_null(d.pool_regions);
    pool["region_ms"] = num_or_null(d.pool_region_ms);
    pool["tail_wait_ms"] = num_or_null(d.pool_tail_wait_ms);
    pool["sleeps"] = num_or_null(d.pool_sleeps);
    if (!d.pool_thread_util.empty()) {
      json::Array util;
      for (double u : d.pool_thread_util) util.push_back(u);
      pool["thread_util"] = std::move(util);
    }
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

std::string request_to_json(const PointResult& point, const RequestResult& r) {
  json::Object o;
  o["concurrency"] = point.cfg.concurrency;
  o["point_prompt_tokens"] = point.cfg.prompt_tokens;
  o["arrival"] = point.arrival_mode;
  o["arrival_digest"] = point.arrival_digest;
  o["index"] = r.index;
  o["prompt_tokens"] = r.requested_prompt_tokens;
  o["prompt_tokens_reported"] = r.prompt_tokens;
  o["scheduled_ms"] = r.scheduled_ms;
  o["dispatch_ms"] = r.dispatch_ms;
  o["start_delay_ms"] = r.start_delay_ms;
  o["cap_bound"] = r.cap_bound;
  o["ttft_ms"] = r.ttft_ms;
  o["e2e_ms"] = r.e2e_ms;
  o["completion_tokens"] = r.completion_tokens;
  o["ok"] = r.ok;
  if (!r.error.empty()) o["error"] = r.error;
  return json::dump(json::Value(std::move(o)));
}

std::vector<std::string> trace_to_json_lines(const PointResult& point) {
  std::vector<std::string> lines;
  const auto ms = [&](int64_t ns) { return ns == 0 ? -1.0 : static_cast<double>(ns - point.t0_ns) * 1e-6; };
  for (const StepTrace& s : point.step_trace) {
    json::Object o;
    o["type"] = "step";
    o["step"] = static_cast<double>(s.step);
    o["plan_ms"] = ms(s.plan_start_ns);
    o["forward_start_ms"] = ms(s.forward_start_ns);
    o["forward_end_ms"] = ms(s.forward_end_ns);
    o["forward_ms"] = static_cast<double>(s.forward_end_ns - s.forward_start_ns) * 1e-6;
    o["prefill_rows"] = s.prefill_rows;
    o["decode_rows"] = s.decode_rows;
    o["prefill_budget"] = s.prefill_budget;
    o["prefill_chunk"] = s.prefill_chunk;
    o["prefilling"] = s.prefilling;
    o["waiting"] = s.waiting;
    o["solo"] = s.solo;
    o["solo_skipped"] = s.solo_skipped;
    o["srpt"] = s.srpt;
    json::Array rows;
    for (const StepTraceRow& r : s.rows) {
      json::Object ro;
      ro["id"] = static_cast<double>(r.request_id);
      ro["rows"] = r.rows;
      ro["pos"] = r.start_pos;
      ro["decode"] = r.decode;
      rows.push_back(json::Value(std::move(ro)));
    }
    o["rows"] = std::move(rows);
    json::Array unserved;
    for (uint64_t id : s.prefill_unserved) unserved.push_back(static_cast<double>(id));
    o["prefill_unserved"] = std::move(unserved);
    lines.push_back(json::dump(json::Value(std::move(o))));
  }
  for (const RequestTrace& r : point.request_trace) {
    json::Object o;
    o["type"] = "request";
    o["id"] = static_cast<double>(r.id);
    o["prompt_tokens"] = r.prompt_tokens;
    o["generated"] = r.generated;
    o["submit_ms"] = ms(r.submit_ns);
    o["admit_ms"] = ms(r.admit_ns);
    o["first_scheduled_ms"] = ms(r.first_scheduled_ns);
    o["first_token_ms"] = ms(r.first_token_ns);
    o["finish_ms"] = ms(r.finish_ns);
    o["prefill_steps"] = r.prefill_steps;
    o["prefill_unserved_steps"] = r.prefill_unserved_steps;
    o["prefill_forward_ms"] = r.prefill_forward_ms;
    o["preemptions"] = r.preemptions;
    o["status"] = std::string(sequence_status_name(r.status));
    lines.push_back(json::dump(json::Value(std::move(o))));
  }
  return lines;
}

Result<std::vector<PointConfig::TraceEntry>> load_arrival_trace(const std::string& path) {
  std::ifstream in(path);
  if (!in) return NotFound("cannot open arrival trace " + path);
  std::vector<PointConfig::TraceEntry> out;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    long long off = -1, len = -1;
    char extra = 0;
    if (std::sscanf(line.c_str(), "%lld,%lld%c", &off, &len, &extra) != 2 || off < 0 || len <= 0) {
      return InvalidArgument("arrival trace " + path + " line " + std::to_string(line_no) + ": expected offset_ms,prompt_tokens");
    }
    if (!out.empty() && off < out.back().offset_ms) {
      return InvalidArgument("arrival trace " + path + " line " + std::to_string(line_no) + ": offsets must not decrease");
    }
    out.push_back({static_cast<int64_t>(off), static_cast<int32_t>(len)});
  }
  if (out.empty()) return InvalidArgument("arrival trace " + path + " has no entries");
  return out;
}

}  // namespace dynalm::bench

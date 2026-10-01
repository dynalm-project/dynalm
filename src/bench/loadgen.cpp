#include "bench/loadgen.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

#include "api/json.h"
#include "common/timer.h"
#include "platform/cpu_info.h"
#include "platform/process_stats.h"

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
  std::string name() const override { return "engine (in-process)"; }
  bool in_process() const override { return true; }

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

PointResult run_point(Target& target, const Tokenizer& tokenizer, const PointConfig& cfg_in) {
  PointConfig cfg = cfg_in;
  if (cfg.requests <= 0) cfg.requests = std::max(4, 2 * cfg.concurrency);
  PointResult out;
  out.cfg = cfg;
  out.target = target.name();

  // Prompt indices never repeat within a process: a later point must not hit
  // the prefix cache (ours or the server's) warmed by an earlier point.
  static std::atomic<int32_t> next_prompt{0};
  const int32_t base = next_prompt.fetch_add(cfg.requests + 1);
  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(cfg.requests));
  for (int32_t i = 0; i < cfg.requests; ++i) prompts.push_back(make_prompt(tokenizer, cfg.prompt_tokens, base + i));

  // Uncounted warm-up: pages in mmapped weights / server caches so the first
  // measured request doesn't carry cold-start cost.
  (void)target.run(make_prompt(tokenizer, std::min<int32_t>(cfg.prompt_tokens, 32), base + cfg.requests), 4);

  std::mutex mu;
  std::vector<RequestResult> results;
  std::atomic<int32_t> next{0};
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
    out.cpu_util = (process_cpu_seconds() - cpu0) / out.wall_s / cpu_info().logical_cores;
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
  return json::dump(o);
}

}  // namespace engine::bench

// Load-generator test: closed loop, Poisson and trace (deterministic) arrivals.
//
//   test_loadgen_openloop <model.gguf>
//
// Uses only the model's tokenizer (to size prompts); requests go to a fake
// target that sleeps a fixed service time, so arrival scheduling, dispatch,
// start delay, the arrival digest, per-request records and the trace reader are
// checked without running inference:
//   - closed loop: mode "closed", one record per request, no scheduled time;
//   - Poisson: the same seed gives the same digest, another seed a different one;
//     arrivals do not wait for completions (achieved rate ~ scheduled rate);
//     latencies count from the scheduled arrival;
//   - client cap: with one slot and slow service, requests start late, are
//     counted as cap-bound, and the delay is inside their TTFT;
//   - trace: exact offsets and prompt lengths, the trace replaces --requests,
//     the same digest on every run, one group per prompt length;
//   - load_arrival_trace rejects bad lines, decreasing offsets and empty files.
// Exit status 0 = all checks passed.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "bench/loadgen.h"
#include "loader/model_loader.h"
#include "common/core.h"

namespace {

using namespace dynalm;

int g_checks = 0, g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (0)

class FakeTarget final : public bench::Target {
 public:
  explicit FakeTarget(int service_ms) : service_ms_(service_ms) {}
  std::string name() const override { return "fake"; }
  bench::RequestResult run(const std::string&, int32_t max_tokens) override {
    ++calls_;
    std::this_thread::sleep_for(std::chrono::milliseconds(service_ms_));
    bench::RequestResult r;
    r.ok = true;
    r.ttft_ms = service_ms_ / 2.0;
    r.e2e_ms = service_ms_;
    r.completion_tokens = max_tokens;
    for (int i = 1; i < max_tokens; ++i) r.gaps_ms.push_back(service_ms_ / 2.0 / std::max(1, max_tokens - 1));
    return r;
  }
  std::atomic<int> calls_{0};

 private:
  int service_ms_;
};

std::string write_file(const std::string& name, const std::string& text) {
  const auto p = std::filesystem::temp_directory_path() / name;
  std::ofstream(p, std::ios::binary) << text;
  return p.string();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: test_loadgen_openloop <model.gguf>\n");
    return 1;
  }
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const Tokenizer& tok = *(*m)->tokenizer;

  // 1. Closed loop.
  {
    FakeTarget t(10);
    bench::PointConfig cfg{2, 16, 4, 6, {}};
    const bench::PointResult r = bench::run_point(t, tok, cfg);
    CHECK(r.arrival_mode == "closed");
    CHECK(r.completed == 6 && r.errors == 0);
    CHECK(r.requests.size() == 6);
    bool sched_none = true;
    for (const auto& q : r.requests) sched_none = sched_none && q.scheduled_ms < 0;
    CHECK(sched_none);
    CHECK(bench::to_json(r, "m", "h").find("\"arrival\":\"closed\"") != std::string::npos);
    std::printf("  closed loop: mode closed, 6 records, no scheduled arrival\n");
  }

  // 2. Poisson: digests, independence from completions, latency from the scheduled arrival.
  std::string digest_s7;
  {
    FakeTarget t(20);
    bench::PointConfig cfg{8, 16, 4, 24, {}};
    cfg.arrival_rate = 6.0;
    cfg.seed = 7;
    const bench::PointResult a = bench::run_point(t, tok, cfg);
    const bench::PointResult b = bench::run_point(t, tok, cfg);
    cfg.seed = 8;
    const bench::PointResult c = bench::run_point(t, tok, cfg);
    digest_s7 = a.arrival_digest;
    CHECK(a.arrival_mode == "poisson");
    CHECK(a.arrival_digest == b.arrival_digest && a.arrival_digest != c.arrival_digest);
    CHECK(a.completed == 24 && a.requests.size() == 24);
    bool same_sched = a.requests.size() == b.requests.size();
    for (size_t i = 0; same_sched && i < a.requests.size(); ++i) same_sched = a.requests[i].scheduled_ms == b.requests[i].scheduled_ms;
    CHECK(same_sched);
    const double rel = std::fabs(a.achieved_arrival_rate - a.trace_arrival_rate) / a.trace_arrival_rate;
    CHECK(rel < 0.03);
    // Free slots: no request is cap-bound (exact), and lateness is only OS
    // scheduling jitter (measured 0.2-8.9 ms p99 on the i7-1255U; ~30 ms before
    // the precise wake).
    CHECK(a.cap_bound_requests == 0);
    CHECK(a.start_delay_ms.p50 < 2.0 && a.start_delay_ms.p99 < 25.0);
    bool ttft_from_schedule = true;
    for (const auto& q : a.requests) ttft_from_schedule = ttft_from_schedule && std::fabs(q.ttft_ms - (10.0 + q.start_delay_ms)) < 1e-6;
    CHECK(ttft_from_schedule);
    std::printf("  poisson: same seed same digest (%s), other seed differs; achieved %.3f vs scheduled %.3f req/s; "
                "start delay p99 %.1f ms\n",
                a.arrival_digest.c_str(), a.achieved_arrival_rate, a.trace_arrival_rate, a.start_delay_ms.p99);
  }

  // 3. Client cap binds: one slot, slow service, fast arrivals.
  {
    FakeTarget t(300);
    bench::PointConfig cfg{1, 16, 4, 6, {}};
    cfg.arrival_rate = 20.0;
    cfg.seed = 3;
    const bench::PointResult r = bench::run_point(t, tok, cfg);
    CHECK(r.cap_bound_requests >= 4);
    bool delay_in_ttft = true;
    for (const auto& q : r.requests) delay_in_ttft = delay_in_ttft && q.ttft_ms >= q.start_delay_ms;
    CHECK(delay_in_ttft);
    CHECK(r.achieved_arrival_rate < 0.5 * r.trace_arrival_rate);
    std::printf("  cap bound: %d of 6 found no free slot; achieved %.2f vs scheduled %.2f req/s (flagged, not hidden)\n",
                r.cap_bound_requests, r.achieved_arrival_rate, r.trace_arrival_rate);
  }

  // 4. Trace: exact offsets and lengths; the trace replaces requests.
  {
    const std::string path = write_file("dynalm_test_trace.csv", "# offset_ms,prompt_tokens\n0,16\n100,32\n\n400,16\n1000,64\n");
    auto tr = bench::load_arrival_trace(path);
    CHECK(tr.ok() && tr->size() == 4);
    FakeTarget t(20);
    bench::PointConfig cfg{4, 128, 4, 99, {}};
    cfg.arrival_trace = *tr;
    cfg.arrival_trace_path = path;
    const bench::PointResult a = bench::run_point(t, tok, cfg);
    const bench::PointResult b = bench::run_point(t, tok, cfg);
    CHECK(a.arrival_mode == "trace");
    CHECK(a.cfg.requests == 4 && a.completed == 4 && a.requests.size() == 4);
    const double want[] = {0, 100, 400, 1000};
    const int lens[] = {16, 32, 16, 64};
    bool exact = true, dispatch_ok = true;
    for (size_t i = 0; i < a.requests.size(); ++i) {
      exact = exact && a.requests[i].index == static_cast<int>(i) && a.requests[i].scheduled_ms == want[i] &&
              a.requests[i].requested_prompt_tokens == lens[i];
      dispatch_ok = dispatch_ok && a.requests[i].dispatch_ms >= want[i] && a.requests[i].dispatch_ms <= want[i] + 25.0 &&
                    !a.requests[i].cap_bound;
    }
    CHECK(exact);
    CHECK(dispatch_ok);
    CHECK(a.arrival_digest == b.arrival_digest && a.arrival_digest != digest_s7);
    CHECK(a.by_prompt.size() == 3);
    CHECK(std::fabs(a.trace_arrival_rate - 3.0) < 1e-9);  // (4 - 1) / 1.0 s
    const std::string pj = bench::to_json(a, "m", "h"), rj = bench::request_to_json(a, a.requests[1]);
    CHECK(pj.find("\"arrival\":\"trace\"") != std::string::npos && pj.find("\"arrival_digest\"") != std::string::npos);
    CHECK(rj.find("\"scheduled_ms\":100") != std::string::npos && rj.find("\"prompt_tokens\":32") != std::string::npos);
    std::printf("  trace: 4 requests at 0/100/400/1000 ms (requests=99 ignored), lengths 16/32/16/64, digest %s on both runs\n",
                a.arrival_digest.c_str());
  }

  // 5. Trace reader errors.
  {
    CHECK(!bench::load_arrival_trace(write_file("dynalm_test_bad1.csv", "0,16\n50\n")).ok());
    CHECK(!bench::load_arrival_trace(write_file("dynalm_test_bad2.csv", "100,16\n50,16\n")).ok());
    CHECK(!bench::load_arrival_trace(write_file("dynalm_test_bad3.csv", "# nothing\n")).ok());
    CHECK(!bench::load_arrival_trace(write_file("dynalm_test_bad4.csv", "0,0\n")).ok());
    CHECK(!bench::load_arrival_trace("does-not-exist.csv").ok());
    std::printf("  trace reader: rejects a short line, decreasing offsets, empty file, zero length, missing file\n");
  }

  if (g_failures) {
    std::printf("FAILED: %d of %d checks\n", g_failures, g_checks);
    return 1;
  }
  std::printf("all load-generator open-loop tests passed (%d checks)\n", g_checks);
  return 0;
}

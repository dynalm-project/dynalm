// Scheduler test: experimental prefill order (SchedulerConfig::prefill_order) and the scheduler trace.
//
//   test_prefill_priority <model.gguf> [threads]
//
// Drives the real Scheduler on a real model and checks:
//   - kFifo (default) row-for-row against hand-derived traces; the trace (SchedulerConfig::trace)
//     does not change any row;
//   - kShortestRemaining: short prompts that arrive behind a long one get the prefill budget first;
//     aging 0 reproduces FIFO exactly; with aging a long prompt still progresses under a stream of
//     short arrivals;
//   - correctness under reordering: every prompt is computed exactly once, in order (contiguous KV
//     positions from the trace), decode positions advance by one, every request ends with exactly
//     max_new tokens, cancellation of a waiting long prompt leaves the others intact, and the generated
//     tokens are compared with FIFO and with each request run alone;
//   - request lifecycle timestamps are ordered (submit <= admit <= first scheduled <= first token <= finish).
// The prefix cache is off; prompts are fixed per request (same tokens in every run). Exit 0 = all passed.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "runtime/sequence.h"
#include "scheduler/scheduler.h"
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

struct Spec {
  int seed = 0;          // prompt content (same seed = same tokens in every run)
  int prompt = 0;
  int max_new = 0;
  int submit_after = 0;  // submitted before step submit_after + 1
  int cancel_after = 0;  // cancelled after this step (0 = never)
};

struct Outcome {
  uint64_t id = 0;
  std::vector<TokenId> tokens;
  int first_token_step = -1;
  bool finished = false;
  SequenceStatus status = SequenceStatus::kWaiting;
  FinishReason reason = FinishReason::kNone;
};

struct Run {
  std::vector<int> prefill, decode;  // per step, from SchedulerStats
  std::vector<Outcome> out;
  SchedulerStats stats;
  std::vector<StepTrace> steps;
  std::vector<RequestTrace> requests;
};

class Fixture {
 public:
  Fixture(const LoadedModel& m, ThreadPool& pool) : m_(m), be_(pool, select_best_isa(cpu_info().features)) {
    auto kv = KvBlockPool::create(kv_geometry_for(m.config, DType::kF16, 16, 8192), be_);
    auto tf = Transformer::create(m.config, m.weights, be_, 256);
    if (!kv.ok() || !tf.ok()) {
      std::fprintf(stderr, "setup failed\n");
      std::exit(1);
    }
    kv_ = std::move(kv).value();
    tf_ = std::move(tf).value();
  }

  // The configuration a scheduler built from `cfg` reports as in effect.
  SchedulerConfig effective(const SchedulerConfig& cfg) {
    Scheduler sched(*tf_, *kv_, *m_.tokenizer, cfg);
    return sched.config();
  }

  Run run(SchedulerConfig cfg, const std::vector<Spec>& specs) {
    cfg.enable_prefix_cache = false;
    Scheduler sched(*tf_, *kv_, *m_.tokenizer, cfg);
    Run r;
    r.out.resize(specs.size());
    int step_no = 0;
    auto submit_due = [&](int after) {
      for (size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].submit_after != after) continue;
        Request q;
        for (int t = 0; t < specs[i].prompt; ++t) q.prompt.push_back(static_cast<TokenId>(1000 + (specs[i].seed * 7919 + t * 31) % 20000));
        q.stop = StopParams{specs[i].max_new, false};
        Outcome* o = &r.out[i];
        q.on_event = [o, &step_no](const RequestEvent& ev) {
          if (ev.finished) {
            o->finished = true;
            o->status = ev.status;
            o->reason = ev.reason;
            return;
          }
          if (o->tokens.empty()) o->first_token_step = step_no;
          o->tokens.push_back(ev.token);
        };
        r.out[i].id = sched.submit(std::move(q));
      }
    };
    submit_due(0);
    for (int guard = 0; guard < 4000; ++guard) {
      const uint64_t before = sched.stats().steps;
      step_no = static_cast<int>(before) + 1;
      sched.step();
      const SchedulerStats& st = sched.stats();
      if (st.steps != before) {
        r.prefill.push_back(st.last_prefill_rows);
        r.decode.push_back(st.last_decode_rows);
        for (size_t i = 0; i < specs.size(); ++i) {
          if (specs[i].cancel_after == static_cast<int>(st.steps)) sched.cancel(r.out[i].id);
        }
        submit_due(static_cast<int>(st.steps));
      }
      bool pending = false;
      for (const Spec& s : specs) pending = pending || s.submit_after > static_cast<int>(st.steps);
      if (sched.idle() && !pending) break;
    }
    r.stats = sched.stats();
    r.steps = sched.take_step_trace();
    r.requests = sched.take_request_trace();
    return r;
  }

 private:
  const LoadedModel& m_;
  CpuDevice be_;
  std::unique_ptr<KvBlockPool> kv_;
  std::unique_ptr<Transformer> tf_;
};

std::string fmt(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}

void expect_seq(const char* what, const std::vector<int>& got, const std::vector<int>& want) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what, fmt(got).c_str(), fmt(want).c_str());
  }
}

std::vector<int> run_of(int value, int n) { return std::vector<int>(static_cast<size_t>(n), value); }
std::vector<int> cat(std::initializer_list<std::vector<int>> parts) {
  std::vector<int> v;
  for (const auto& p : parts) v.insert(v.end(), p.begin(), p.end());
  return v;
}

// Every finished request: exactly max_new tokens (kLength). Cancelled ones: status kCancelled.
void expect_outcomes(const char* what, const Run& r, const std::vector<Spec>& specs) {
  for (size_t i = 0; i < specs.size(); ++i) {
    const Outcome& o = r.out[i];
    bool ok;
    if (specs[i].cancel_after > 0) {
      ok = o.finished && o.status == SequenceStatus::kCancelled;
    } else {
      ok = o.finished && static_cast<int>(o.tokens.size()) == specs[i].max_new && o.reason == FinishReason::kLength &&
           o.status == SequenceStatus::kFinished;
    }
    ++g_checks;
    if (!ok) {
      ++g_failures;
      std::fprintf(stderr, "FAIL %s request %zu: finished %d tokens %zu status %d\n", what, i, o.finished ? 1 : 0,
                   o.tokens.size(), static_cast<int>(o.status));
    }
  }
}

// KV positions from the trace: each prompt computed exactly once, in order; decode positions advance by one.
void expect_positions(const char* what, const Run& r, const std::vector<Spec>& specs) {
  std::map<uint64_t, size_t> slot;
  for (size_t i = 0; i < specs.size(); ++i) slot[r.out[i].id] = i;
  std::map<uint64_t, int> next;  // next expected position
  bool ok = true;
  for (const StepTrace& s : r.steps) {
    for (const StepTraceRow& row : s.rows) {
      int& pos = next[row.request_id];
      if (row.start_pos != pos || row.rows <= 0 || (row.decode && row.rows != 1)) ok = false;
      pos += row.rows;
    }
  }
  for (size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].cancel_after > 0) continue;
    // prompt + (max_new - 1) positions are computed (the last token is never fed back)
    if (next[r.out[i].id] != specs[i].prompt + specs[i].max_new - 1) ok = false;
  }
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: KV positions not contiguous / not covering each prompt exactly once\n", what);
  }
}

// Lifecycle timestamps ordered; trace counters consistent with the step trace.
void expect_lifecycle(const char* what, const Run& r) {
  bool ok = !r.requests.empty();
  for (const RequestTrace& q : r.requests) {
    if (q.status == SequenceStatus::kCancelled) continue;
    ok = ok && q.submit_ns > 0 && q.submit_ns <= q.admit_ns && q.admit_ns <= q.first_scheduled_ns &&
         q.first_scheduled_ns <= q.first_token_ns && q.first_token_ns <= q.finish_ns && q.prefill_forward_ms > 0;
    int chunks = 0, unserved = 0;
    for (const StepTrace& s : r.steps) {
      for (const StepTraceRow& row : s.rows) chunks += (row.request_id == q.id && !row.decode) ? 1 : 0;
      for (uint64_t id : s.prefill_unserved) unserved += id == q.id ? 1 : 0;
    }
    ok = ok && chunks == q.prefill_steps && unserved == q.prefill_unserved_steps;
  }
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: request lifecycle trace inconsistent\n", what);
  }
}

SchedulerConfig cfg(PrefillOrder order, bool trace = true, int aging_ms = 20000) {
  SchedulerConfig c;  // defaults: budget 64, chunk 32, decode budget 64, DD-079 on
  c.prefill_order = order;
  c.trace = trace;
  c.prefill_aging_ms = aging_ms;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: test_prefill_priority <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  ThreadPool pool(threads);
  Fixture fx(**m, pool);
  std::printf("model: %s  threads %d\n", argv[1], threads);
  CHECK(SchedulerConfig{}.prefill_order == PrefillOrder::kFifo && !SchedulerConfig{}.trace);

  // Scheduler::config() reports the budgets in effect (serve logs these):
  // policy overrides applied, the requested values otherwise.
  {
    SchedulerConfig lat;
    lat.policy = SchedulerPolicy::kLatencyFirst;
    const SchedulerConfig el = fx.effective(lat);
    CHECK(el.prefill_token_budget == 32 && el.protected_prefill_min == 16);
    SchedulerConfig thr;
    thr.policy = SchedulerPolicy::kThroughputFirst;
    const SchedulerConfig et = fx.effective(thr);
    CHECK(et.prefill_token_budget == 128 && !et.decode_protect);
    const SchedulerConfig eb = fx.effective(SchedulerConfig{});
    CHECK(eb.prefill_token_budget == SchedulerConfig{}.prefill_token_budget &&
          eb.decode_token_budget == SchedulerConfig{}.decode_token_budget);
  }

  // A 600-token prompt alone, then two 128-token prompts arrive after step 1.
  const std::vector<Spec> hol = {{1, 600, 2}, {2, 128, 2, 1}, {3, 128, 2, 1}};

  // 1. FIFO (reference): long 32 + first short 32 per step; the second short waits for the first.
  const Run fifo = fx.run(cfg(PrefillOrder::kFifo), hol);
  expect_seq("fifo prefill rows", fifo.prefill, cat({{32}, run_of(64, 4), {64}, run_of(64, 3), {32}, run_of(32, 8), {24, 0}}));
  expect_seq("fifo decode rows", fifo.decode, cat({run_of(0, 5), {1}, run_of(0, 3), {1}, run_of(0, 9), {1}}));
  CHECK(fifo.out[1].first_token_step == 5 && fifo.out[2].first_token_step == 9 && fifo.out[0].first_token_step == 19);
  CHECK(fifo.stats.steps_prefill_srpt == 0);
  expect_outcomes("fifo", fifo, hol);
  expect_positions("fifo", fifo, hol);
  expect_lifecycle("fifo", fifo);
  std::printf("  fifo: shorts first tokens at steps 5 and 9, long at 19\n");

  // 2. The trace is statistics only: the same rows without it.
  {
    const Run plain = fx.run(cfg(PrefillOrder::kFifo, false), hol);
    CHECK(plain.prefill == fifo.prefill && plain.decode == fifo.decode);
    CHECK(plain.steps.empty() && plain.requests.empty());
    bool same_tokens = true;
    for (size_t i = 0; i < hol.size(); ++i) same_tokens = same_tokens && plain.out[i].tokens == fifo.out[i].tokens;
    CHECK(same_tokens);
    std::printf("  trace off: identical rows and tokens (trace is behaviour-neutral)\n");
  }

  // 3. Shortest remaining first: both shorts get the budget (32 + 32) from step 2 and finish at step 5;
  //    the long prompt waits 4 steps, then computes alone (first token at step 23 instead of 19).
  const Run srpt = fx.run(cfg(PrefillOrder::kShortestRemaining), hol);
  expect_seq("srpt prefill rows", srpt.prefill, cat({{32}, run_of(64, 4), {32}, run_of(32, 16), {24, 0}}));
  expect_seq("srpt decode rows", srpt.decode, cat({run_of(0, 5), {2}, run_of(0, 17), {1}}));
  CHECK(srpt.out[1].first_token_step == 5 && srpt.out[2].first_token_step == 5 && srpt.out[0].first_token_step == 23);
  CHECK(srpt.stats.steps_prefill_srpt == 4);
  expect_outcomes("srpt", srpt, hol);
  expect_positions("srpt", srpt, hol);
  expect_lifecycle("srpt", srpt);
  {
    // The trace shows the long prompt unserved in steps 2-5 (interrupted), then resumed at position 32.
    int unserved = 0, resumed_at = -1;
    for (const StepTrace& s : srpt.steps) {
      for (uint64_t id : s.prefill_unserved) unserved += id == srpt.out[0].id ? 1 : 0;
      for (const StepTraceRow& row : s.rows) {
        if (row.request_id == srpt.out[0].id && s.step == 6) resumed_at = row.start_pos;
      }
    }
    CHECK(unserved == 4 && resumed_at == 32);
  }
  std::printf("  srpt: shorts first tokens at steps 5 and 5, long interrupted 4 steps, resumed at position 32, first token 23\n");

  // 4. Aging 0: every prompt counts as waiting too long -> FIFO order, identical rows.
  {
    const Run aged = fx.run(cfg(PrefillOrder::kShortestRemaining, true, 0), hol);
    CHECK(aged.prefill == fifo.prefill && aged.decode == fifo.decode);
    CHECK(aged.out[2].first_token_step == 9);
    std::printf("  srpt with aging 0: identical to fifo\n");
  }

  // 5. Generated tokens under reordering: compared with FIFO and with each request run alone.
  {
    int same_fifo = 0, same_alone_srpt = 0, same_alone_fifo = 0;
    for (size_t i = 0; i < hol.size(); ++i) {
      Spec alone = hol[i];
      alone.submit_after = 0;
      const Run a = fx.run(cfg(PrefillOrder::kFifo, false), {alone});
      same_fifo += srpt.out[i].tokens == fifo.out[i].tokens ? 1 : 0;
      same_alone_srpt += srpt.out[i].tokens == a.out[0].tokens ? 1 : 0;
      same_alone_fifo += fifo.out[i].tokens == a.out[0].tokens ? 1 : 0;
    }
    std::printf("  tokens: srpt == fifo for %d/3 requests; == alone: srpt %d/3, fifo %d/3\n", same_fifo, same_alone_srpt,
                same_alone_fifo);
    CHECK(same_alone_srpt >= same_alone_fifo);  // reordering must not be worse than FIFO's own batching effects
  }

  // 6. Cancelling the long prompt while it waits (after step 3): the shorts finish normally.
  {
    std::vector<Spec> c = hol;
    c[0].cancel_after = 3;
    const Run r = fx.run(cfg(PrefillOrder::kShortestRemaining), c);
    expect_outcomes("srpt cancel", r, c);
    expect_positions("srpt cancel", r, c);
    CHECK(r.out[1].first_token_step == 5 && r.out[2].first_token_step == 5);
    std::printf("  srpt: long prompt cancelled while waiting; shorts complete (first tokens step 5)\n");
  }

  // 7. Starvation guard: a stream of short prompts (one per step for 24 steps) behind a long one.
  //    Without aging the long prompt waits until the stream ends; with aging 1500 ms it is served earlier.
  {
    std::vector<Spec> stream = {{10, 600, 2}};
    for (int k = 1; k <= 24; ++k) stream.push_back({10 + k, 128, 2, k});
    const Run starve = fx.run(cfg(PrefillOrder::kShortestRemaining, true, 1000000), stream);
    const Run aged = fx.run(cfg(PrefillOrder::kShortestRemaining, true, 1500), stream);
    expect_outcomes("srpt stream (no aging)", starve, stream);
    expect_outcomes("srpt stream (aging 1500 ms)", aged, stream);
    expect_positions("srpt stream (aging 1500 ms)", aged, stream);
    CHECK(aged.out[0].first_token_step < starve.out[0].first_token_step);
    std::printf("  srpt stream of 24 shorts: long first token at step %d without aging, %d with aging 1500 ms\n",
                starve.out[0].first_token_step, aged.out[0].first_token_step);
  }

  if (g_failures) {
    std::printf("FAILED: %d of %d checks\n", g_failures, g_checks);
    return 1;
  }
  std::printf("all prefill-priority scheduler tests passed (%d checks)\n", g_checks);
  return 0;
}

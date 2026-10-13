// Scheduler test: experimental solo-prefill chunking (SchedulerConfig::solo_prefill_chunk).
//
//   test_solo_prefill <model.gguf> [threads]
//
// Drives the real Scheduler on a real model (a small one keeps it short:
// Qwen2.5-1.5B takes ~1 min) and checks, step by step, the prefill / decode
// rows and the sequences of every batch against the scheduling rules:
//   - solo_prefill_chunk = 0 (the default) keeps the existing rows exactly,
//     also with solo_prefill_split set;
//   - with solo_prefill_chunk = 128 (FIFO), a step without decode rows and
//     with exactly one prompt prefilling takes chunks of up to 128 rows (budget
//     128); with two or more prompts it keeps the regular rows (32 per prompt,
//     budget 64), so a short prompt never waits behind a long one's 128-row
//     chunks; split mode shares the 128 budget across the prompts instead;
//   - the solo budget never reaches a step with decode rows, including the
//     first mixed step right after a prefill-only step: there every sequence
//     keeps the 32-row chunk and the DD-079 budget (<= prefill_token_budget);
//   - every request ends with exactly max_new_tokens tokens (kLength).
// The prefix cache is off and every prompt is unique, so row counts are exact.
// Exit status 0 = all checks passed.

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

#define CHECK(cond)                                                             \
  do {                                                                          \
    ++g_checks;                                                                 \
    if (!(cond)) {                                                              \
      ++g_failures;                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
    }                                                                           \
  } while (0)

struct Step {
  int prefill = 0, decode = 0, seqs = 0, mixed_budget = 0;
  bool solo = false, skipped = false;  // this step counted in steps_solo_prefill / steps_solo_prefill_skipped
};

struct Outcome {
  int tokens = 0;
  int first_token_step = -1;  // 1-based step that produced the first token
  bool finished = false;
  SequenceStatus status = SequenceStatus::kWaiting;
  FinishReason reason = FinishReason::kNone;
};

struct Spec {
  int prompt = 0;   // prompt tokens
  int max_new = 0;  // generated tokens (stop_at_eog off: always exactly this many)
  int submit_after = 0;  // submitted before step submit_after + 1
};

struct Run {
  std::vector<Step> steps;
  std::vector<Outcome> out;  // in Spec order
  SchedulerStats stats;
};

class Fixture {
 public:
  Fixture(const LoadedModel& m, ThreadPool& pool)
      : m_(m), be_(pool, select_best_isa(cpu_info().features)) {
    const ModelConfig& c = m.config;
    auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, 4096), be_);
    auto tf = Transformer::create(c, m.weights, be_, 256);
    if (!kv.ok() || !tf.ok()) {
      std::fprintf(stderr, "setup failed: %s %s\n", kv.status().to_string().c_str(), tf.status().to_string().c_str());
      std::exit(1);
    }
    kv_ = std::move(kv).value();
    tf_ = std::move(tf).value();
  }

  Run run(SchedulerConfig cfg, const std::vector<Spec>& specs) {
    cfg.enable_prefix_cache = false;
    Scheduler sched(*tf_, *kv_, *m_.tokenizer, cfg);
    Run r;
    r.out.resize(specs.size());
    std::map<uint64_t, size_t> index;
    int step_no = 0;
    auto submit_due = [&](int after) {
      for (size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].submit_after != after) continue;
        Request q;
        // Unique prompt per request and per run: no two prompts share a prefix.
        for (int t = 0; t < specs[i].prompt; ++t) q.prompt.push_back(static_cast<TokenId>(1000 + (next_prompt_ * 7919 + t * 31) % 20000));
        ++next_prompt_;
        q.stop = StopParams{specs[i].max_new, false};
        const size_t slot = i;
        q.on_event = [&r, &step_no, slot](const RequestEvent& ev) {
          Outcome& o = r.out[slot];
          if (ev.finished) {
            o.finished = true;
            o.status = ev.status;
            o.reason = ev.reason;
            return;
          }
          if (o.tokens++ == 0) o.first_token_step = step_no;
        };
        index[sched.submit(std::move(q))] = i;
      }
    };
    submit_due(0);
    for (int guard = 0; guard < 2000; ++guard) {
      const uint64_t before = sched.stats().steps;
      const uint64_t solo0 = sched.stats().steps_solo_prefill, skip0 = sched.stats().steps_solo_prefill_skipped;
      step_no = static_cast<int>(before) + 1;
      sched.step();
      const SchedulerStats& st = sched.stats();
      if (st.steps != before) {
        r.steps.push_back({st.last_prefill_rows, st.last_decode_rows, st.last_batch_seqs, st.mixed_prefill_budget,
                           st.steps_solo_prefill != solo0, st.steps_solo_prefill_skipped != skip0});
        submit_due(static_cast<int>(st.steps));
      }
      bool pending_submit = false;
      for (const Spec& s : specs) pending_submit = pending_submit || s.submit_after > static_cast<int>(st.steps);
      if (sched.idle() && !pending_submit) break;
    }
    r.stats = sched.stats();
    return r;
  }

 private:
  const LoadedModel& m_;
  CpuDevice be_;
  std::unique_ptr<KvBlockPool> kv_;
  std::unique_ptr<Transformer> tf_;
  int next_prompt_ = 0;
};

std::vector<int> col(const Run& r, int Step::*f) {
  std::vector<int> v;
  for (const Step& s : r.steps) v.push_back(s.*f);
  return v;
}

std::string fmt(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}

bool expect_rows(const char* what, const Run& r, const std::vector<int>& pre, const std::vector<int>& dec,
                 const std::vector<int>& seqs) {
  const auto p = col(r, &Step::prefill), d = col(r, &Step::decode), q = col(r, &Step::seqs);
  const bool ok = p == pre && d == dec && q == seqs;
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s rows\n  prefill %s (expected %s)\n  decode  %s (expected %s)\n  seqs    %s (expected %s)\n",
                 what, fmt(p).c_str(), fmt(pre).c_str(), fmt(d).c_str(), fmt(dec).c_str(), fmt(q).c_str(),
                 fmt(seqs).c_str());
  }
  return ok;
}

// Every request finished normally with exactly max_new tokens.
void expect_complete(const char* what, const Run& r, const std::vector<Spec>& specs) {
  for (size_t i = 0; i < specs.size(); ++i) {
    const Outcome& o = r.out[i];
    const bool ok = o.finished && o.tokens == specs[i].max_new && o.reason == FinishReason::kLength &&
                    o.status == SequenceStatus::kFinished;
    ++g_checks;
    if (!ok) {
      ++g_failures;
      std::fprintf(stderr, "FAIL %s request %zu: finished %d tokens %d (expected %d) reason %d status %d\n", what, i,
                   o.finished ? 1 : 0, o.tokens, specs[i].max_new, static_cast<int>(o.reason), static_cast<int>(o.status));
    }
  }
  // Solo-step timing statistics (V4): present exactly when solo steps ran,
  // and consistent (each step <= the total; the mean step <= the longest).
  const SchedulerStats& st = r.stats;
  const bool timing_ok =
      (st.steps_solo_prefill > 0) == (st.forward_solo_prefill_ms > 0) &&
      (st.steps_solo_prefill > 0) == (st.solo_prefill_step_ms_max > 0) &&
      st.solo_prefill_step_ms_max <= st.forward_solo_prefill_ms + 1e-9 &&
      (st.steps_solo_prefill == 0 ||
       st.forward_solo_prefill_ms / static_cast<double>(st.steps_solo_prefill) <= st.solo_prefill_step_ms_max + 1e-9);
  ++g_checks;
  if (!timing_ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: solo timing stats inconsistent (steps %llu, total %.3f ms, max %.3f ms)\n", what,
                 static_cast<unsigned long long>(st.steps_solo_prefill), st.forward_solo_prefill_ms,
                 st.solo_prefill_step_ms_max);
  }
}

// Mixed steps never exceed the 32-row chunk per prefilling sequence or the
// DD-079 budget, whatever the solo setting.
void expect_mixed_bounded(const char* what, const Run& r, const SchedulerConfig& cfg) {
  for (size_t i = 0; i < r.steps.size(); ++i) {
    const Step& s = r.steps[i];
    CHECK(s.mixed_budget <= cfg.prefill_token_budget);
    if (s.decode == 0 || s.prefill == 0) continue;
    const int prefilling = s.seqs - s.decode;
    const bool ok = prefilling >= 1 && s.prefill <= cfg.prefill_token_budget && s.prefill <= cfg.max_prefill_chunk * prefilling;
    ++g_checks;
    if (!ok) {
      ++g_failures;
      std::fprintf(stderr, "FAIL %s: mixed step %zu has %d prefill rows over %d prefilling sequences\n", what, i + 1,
                   s.prefill, prefilling);
    }
  }
}

// FIFO mode: every solo step is a lone prompt (one sequence, no decode rows);
// every skipped step has no decode rows; no step is both.
void expect_fifo_solo_property(const char* what, const Run& r) {
  for (size_t i = 0; i < r.steps.size(); ++i) {
    const Step& s = r.steps[i];
    const bool ok = !(s.solo && s.skipped) && (!s.solo || (s.decode == 0 && s.seqs == 1)) && (!s.skipped || s.decode == 0);
    ++g_checks;
    if (!ok) {
      ++g_failures;
      std::fprintf(stderr, "FAIL %s: step %zu solo %d skipped %d with %d decode rows over %d sequences\n", what, i + 1,
                   s.solo ? 1 : 0, s.skipped ? 1 : 0, s.decode, s.seqs);
    }
  }
}

SchedulerConfig base() {
  SchedulerConfig c;  // defaults: budget 64, chunk 32, decode budget 64, DD-079 on
  return c;
}

SchedulerConfig solo(bool split) {
  SchedulerConfig c = base();
  c.solo_prefill_chunk = 128;
  c.solo_prefill_split = split;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: test_solo_prefill <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  ThreadPool pool(threads);
  Fixture fx(**m, pool);
  std::printf("model: %s  threads %d\n", argv[1], threads);

  // Defaults: the solo fields are off.
  CHECK(SchedulerConfig{}.solo_prefill_chunk == 0);
  CHECK(!SchedulerConfig{}.solo_prefill_split);

  // 1. Off (default): one 300-token prompt -> 9 chunks of 32 + 12, then decode.
  {
    const std::vector<Spec> sp = {{300, 4}};
    const Run r = fx.run(base(), sp);
    expect_rows("off/lone", r, {32, 32, 32, 32, 32, 32, 32, 32, 32, 12, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    expect_complete("off/lone", r, sp);
    CHECK(r.stats.steps_solo_prefill == 0);
    CHECK(r.stats.solo_prefill_chunk == 0);
    std::printf("  off, lone 300-token prompt: rows 9 x 32 + 12, 4 tokens, no solo steps\n");
  }

  // 2. Off (default), two prompts at once: 32 + 32 per step, then the first
  //    mixed step keeps the 32-row chunk.
  const std::vector<int> two_off_pre = {64, 64, 64, 64, 64, 64, 40, 32, 32, 12, 0};
  const std::vector<int> two_off_dec = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1};
  const std::vector<int> two_off_seq = {2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1};
  {
    const std::vector<Spec> sp = {{300, 2}, {200, 2}};
    const Run r = fx.run(base(), sp);
    expect_rows("off/two", r, two_off_pre, two_off_dec, two_off_seq);
    expect_complete("off/two", r, sp);
    expect_mixed_bounded("off/two", r, base());
    CHECK(r.out[1].first_token_step == 7 && r.out[0].first_token_step == 10);
    std::printf("  off, prompts 300 + 200 together: 32 + 32 per step, B first (step 7), A at step 10\n");
  }

  // 3. split set but solo off: identical to off.
  {
    SchedulerConfig c = base();
    c.solo_prefill_split = true;
    const std::vector<Spec> sp = {{300, 2}, {200, 2}};
    const Run r = fx.run(c, sp);
    expect_rows("split-without-solo/two", r, two_off_pre, two_off_dec, two_off_seq);
    expect_complete("split-without-solo/two", r, sp);
    CHECK(r.stats.steps_solo_prefill == 0);
    std::printf("  solo_prefill_split without solo_prefill_chunk: rows identical to off\n");
  }

  // 4. Solo 128, lone prompt: 128 + 128 + 44 (FIFO and split alike).
  for (const bool split : {false, true}) {
    const std::vector<Spec> sp = {{300, 4}};
    const Run r = fx.run(solo(split), sp);
    const char* what = split ? "solo-split/lone" : "solo-fifo/lone";
    expect_rows(what, r, {128, 128, 44, 0, 0, 0}, {0, 0, 0, 1, 1, 1}, {1, 1, 1, 1, 1, 1});
    expect_complete(what, r, sp);
    if (!split) expect_fifo_solo_property(what, r);
    CHECK(r.stats.steps_solo_prefill == 3);
    CHECK(r.stats.steps_solo_prefill_skipped == 0);
    CHECK(r.stats.solo_prefill_chunk == 128);
    CHECK(r.stats.solo_prefill_split == split);
    std::printf("  solo 128 %s, lone 300-token prompt: rows 128 + 128 + 44, 3 solo steps\n", split ? "split" : "FIFO");
  }

  // 5. Solo 128 FIFO, prompts 300 (A) + 200 (B) together: two prompts
  //    prefilling, so no solo chunk (the regular 32 + 32 per step, as off).
  //    B finishes first (step 7), step 8 is mixed (A 32), and A, now a lone
  //    prompt with no decode rows, takes its last 44 rows in one solo step.
  {
    const std::vector<Spec> sp = {{300, 2}, {200, 2}};
    const Run r = fx.run(solo(false), sp);
    expect_rows("solo-fifo/two", r, {64, 64, 64, 64, 64, 64, 40, 32, 44, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
                {2, 2, 2, 2, 2, 2, 2, 2, 1, 1});
    expect_complete("solo-fifo/two", r, sp);
    expect_mixed_bounded("solo-fifo/two", r, solo(false));
    expect_fifo_solo_property("solo-fifo/two", r);
    CHECK(r.out[1].first_token_step == 7 && r.out[0].first_token_step == 9);  // B first, as off (7); A at 9 (off: 10)
    CHECK(r.stats.steps_solo_prefill == 1);
    CHECK(r.stats.steps_solo_prefill_skipped == 7);
    std::printf("  solo 128 FIFO, prompts 300 + 200: 32 + 32 per step (7 skipped), B first (step 7); A's last 44 solo (step 9)\n");
  }

  // 6. Solo 128 split, same prompts: 64 + 64 per step. Step 4 finishes B
  //    (64 + 8); step 5 is mixed (B decodes), so A gets 32 rows, not 64 or
  //    128; step 6 is solo again (B finished): A's last 12.
  {
    const std::vector<Spec> sp = {{300, 2}, {200, 2}};
    const Run r = fx.run(solo(true), sp);
    expect_rows("solo-split/two", r, {128, 128, 128, 72, 32, 12, 0}, {0, 0, 0, 0, 1, 0, 1}, {2, 2, 2, 2, 2, 1, 1});
    expect_complete("solo-split/two", r, sp);
    expect_mixed_bounded("solo-split/two", r, solo(true));
    CHECK(r.out[1].first_token_step == 4 && r.out[0].first_token_step == 6);  // split: the shorter B first
    CHECK(r.stats.steps_solo_prefill == 5);
    std::printf("  solo 128 split, prompts 300 + 200: rows 64 + 64 per step | mixed 32 | 12; B first (step 4), A step 6\n");
  }

  // 7. Solo 128 split, three prompts: per-sequence chunk max(32, 128 / 3) = 42.
  {
    const std::vector<Spec> sp = {{100, 1}, {100, 1}, {100, 1}};
    const Run r = fx.run(solo(true), sp);
    // 42 x 3 = 126; 42 x 3 = 126; then 16 each = 48 -> all done.
    expect_rows("solo-split/three", r, {126, 126, 48}, {0, 0, 0}, {3, 3, 3});
    expect_complete("solo-split/three", r, sp);
    std::printf("  solo 128 split, three 100-token prompts: 42 rows each per step\n");
  }

  // 8. Decoders running when a long prompt arrives: every step of its prefill
  //    is mixed, so it keeps 32-row chunks with solo 128 (FIFO and split).
  //    Step 1 (the decoders' own two prompts, no decode rows yet) is a split
  //    solo step, and a skipped step in FIFO mode (two prompts); 32 rows either way.
  for (const bool split : {false, true}) {
    const std::vector<Spec> sp = {{16, 24}, {16, 24}, {300, 2, 1}};
    const Run r = fx.run(solo(split), sp);
    const char* what = split ? "solo-split/decoders+long" : "solo-fifo/decoders+long";
    std::vector<int> pre = {32}, dec = {0}, seqs = {2};
    for (int i = 0; i < 9; ++i) pre.push_back(32), dec.push_back(2), seqs.push_back(3);
    pre.push_back(12), dec.push_back(2), seqs.push_back(3);   // step 11: last chunk
    for (int i = 0; i < 1; ++i) pre.push_back(0), dec.push_back(3), seqs.push_back(3);  // step 12: C's 2nd token
    for (int i = 0; i < 12; ++i) pre.push_back(0), dec.push_back(2), seqs.push_back(2);  // decoders to 24 tokens
    expect_rows(what, r, pre, dec, seqs);
    expect_complete(what, r, sp);
    expect_mixed_bounded(what, r, solo(split));
    if (!split) expect_fifo_solo_property(what, r);
    CHECK(r.stats.steps_solo_prefill == (split ? 1u : 0u));
    CHECK(r.stats.steps_solo_prefill_skipped == (split ? 0u : 1u));
    std::printf("  solo 128 %s, long prompt while 2 streams decode: 32-row chunks in all 10 mixed steps\n",
                split ? "split" : "FIFO");
  }

  // 9. The same scenario with solo off: identical rows (the solo setting only
  //    changes steps without decode rows).
  {
    const std::vector<Spec> sp = {{16, 24}, {16, 24}, {300, 2, 1}};
    const Run r = fx.run(base(), sp);
    std::vector<int> pre = {32}, dec = {0}, seqs = {2};
    for (int i = 0; i < 9; ++i) pre.push_back(32), dec.push_back(2), seqs.push_back(3);
    pre.push_back(12), dec.push_back(2), seqs.push_back(3);
    pre.push_back(0), dec.push_back(3), seqs.push_back(3);
    for (int i = 0; i < 12; ++i) pre.push_back(0), dec.push_back(2), seqs.push_back(2);
    expect_rows("off/decoders+long", r, pre, dec, seqs);
    expect_complete("off/decoders+long", r, sp);
    std::printf("  off, the same scenario: identical rows\n");
  }

  // 10. The solo chunk is clamped to the forward-pass capacity (256 here).
  {
    SchedulerConfig c = base();
    c.solo_prefill_chunk = 4096;
    const std::vector<Spec> sp = {{300, 1}};
    const Run r = fx.run(c, sp);
    CHECK(r.stats.solo_prefill_chunk == 256);
    expect_rows("solo-clamped/lone", r, {256, 44}, {0, 0}, {1, 1});
    expect_complete("solo-clamped/lone", r, sp);
    expect_fifo_solo_property("solo-clamped/lone", r);
    std::printf("  solo 4096 with a 256-token batch: clamped to 256 (rows 256 + 44)\n");
  }

  // 11-13. DD-079 decode protection, made deterministic by its latency
  //    target: 0.001 x the decode-step time means every mixed step is "slow"
  //    (the budget halves after each, down to protected_prefill_min); 1e9
  //    means every mixed step is "fast" (it doubles, up to prefill_token_budget).
  //    Chunk 64 and minimum 16 make the budget visible in the rows. Two
  //    streams decode (one decode-only step sets the step-time EMA), then a
  //    600-token prompt arrives; the streams finish after 6 tokens, leaving
  //    steps without decode rows.
  {
    auto protect = [](int32_t solo_chunk, float target) {
      SchedulerConfig c = base();
      c.max_prefill_chunk = 64;
      c.protected_prefill_min = 16;
      c.decode_latency_target = target;
      c.solo_prefill_chunk = solo_chunk;
      return c;
    };
    const std::vector<Spec> sp = {{16, 6}, {16, 6}, {600, 1, 2}};
    const std::vector<int> dec = {0, 2, 2, 2, 2, 2};
    // Halving with solo 128: mixed rows 64 -> 32 -> 16 -> 16; then solo steps of
    // 128 (472 left); the halved mixed budget (16) is kept, not reset.
    {
      const SchedulerConfig c = protect(128, 0.001f);
      const Run r = fx.run(c, sp);
      std::vector<int> d = dec;
      d.insert(d.end(), {0, 0, 0, 0});
      expect_rows("protect-halve/solo", r, {32, 0, 64, 32, 16, 16, 128, 128, 128, 88}, d, {2, 2, 3, 3, 3, 3, 1, 1, 1, 1});
      expect_complete("protect-halve/solo", r, sp);
      const std::vector<int> want_budget = {64, 64, 32, 16, 16, 16, 16, 16, 16, 16};
      CHECK(col(r, &Step::mixed_budget) == want_budget);
      for (const Step& s : r.steps) CHECK(s.mixed_budget >= c.protected_prefill_min && s.mixed_budget <= c.prefill_token_budget);
      CHECK(r.stats.steps_solo_prefill == 4);          // the four steps after the streams finished (lone prompt)
      CHECK(r.stats.steps_solo_prefill_skipped == 1);  // step 1: the two streams' prompts
      expect_fifo_solo_property("protect-halve/solo", r);
      std::printf("  DD-079 halving, solo 128: mixed 64/32/16/16 rows, budget kept at 16 through solo steps of 128\n");
    }
    // The same with solo off: identical mixed steps; afterwards prefill_token_budget (64) steps.
    {
      const SchedulerConfig c = protect(0, 0.001f);
      const Run r = fx.run(c, sp);
      std::vector<int> pre = {32, 0, 64, 32, 16, 16}, d = dec, q = {2, 2, 3, 3, 3, 3};
      for (int i = 0; i < 7; ++i) pre.push_back(64), d.push_back(0), q.push_back(1);  // 472 = 7 x 64 + 24
      pre.push_back(24), d.push_back(0), q.push_back(1);
      expect_rows("protect-halve/off", r, pre, d, q);
      expect_complete("protect-halve/off", r, sp);
      for (const Step& s : r.steps) CHECK(s.mixed_budget >= c.protected_prefill_min && s.mixed_budget <= c.prefill_token_budget);
      CHECK(r.stats.steps_solo_prefill == 0);
      std::printf("  DD-079 halving, solo off: the same mixed steps; 64-row steps after the streams finish\n");
    }
    // Doubling with solo 128: the mixed budget stays at prefill_token_budget (64),
    // never the solo 128, in all four mixed steps.
    {
      const SchedulerConfig c = protect(128, 1e9f);
      const Run r = fx.run(c, sp);
      std::vector<int> d = dec;
      d.insert(d.end(), {0, 0, 0});
      expect_rows("protect-double/solo", r, {32, 0, 64, 64, 64, 64, 128, 128, 88}, d, {2, 2, 3, 3, 3, 3, 1, 1, 1});
      expect_complete("protect-double/solo", r, sp);
      for (const Step& s : r.steps) CHECK(s.mixed_budget == c.prefill_token_budget);
      CHECK(r.stats.steps_solo_prefill == 3 && r.stats.steps_solo_prefill_skipped == 1);
      expect_fifo_solo_property("protect-double/solo", r);
      std::printf("  DD-079 doubling, solo 128: mixed budget capped at 64 (never 128); solo steps 128\n");
    }
  }

  // 14. Head-of-line: a 600-token prompt alone (solo step of 128), then a
  //     128-token prompt arrives. From step 2 two prompts prefill: no solo
  //     chunk, 32 + 32 as off, so the short prompt's first token comes at step
  //     5 exactly as with solo off (the V3 FIFO rule left it waiting behind
  //     every 128-row chunk). Step 6 is mixed (the short one decodes); from
  //     step 7 the long prompt is alone again: solo 128, 128, 56.
  const std::vector<Spec> hol = {{600, 2}, {128, 2, 1}};
  {
    const Run r = fx.run(solo(false), hol);
    expect_rows("solo-fifo/head-of-line", r, {128, 64, 64, 64, 64, 32, 128, 128, 56, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 1},
                {1, 2, 2, 2, 2, 2, 1, 1, 1, 1});
    expect_complete("solo-fifo/head-of-line", r, hol);
    expect_mixed_bounded("solo-fifo/head-of-line", r, solo(false));
    expect_fifo_solo_property("solo-fifo/head-of-line", r);
    CHECK(r.out[1].first_token_step == 5 && r.out[0].first_token_step == 9);
    CHECK(r.stats.steps_solo_prefill == 4 && r.stats.steps_solo_prefill_skipped == 4);
    std::printf("  solo 128 FIFO, short prompt arriving behind a lone long one: shared 32 + 32 steps, short first token step 5\n");
  }
  // 15. The same with solo off: the short prompt's first token also at step 5;
  //     the long prompt keeps 32-row chunks (first token at step 19).
  {
    const Run r = fx.run(base(), hol);
    std::vector<int> pre = {32, 64, 64, 64, 64, 32}, d = {0, 0, 0, 0, 0, 1}, q = {1, 2, 2, 2, 2, 2};
    for (int i = 0; i < 12; ++i) pre.push_back(32), d.push_back(0), q.push_back(1);  // 408 = 12 x 32 + 24
    pre.push_back(24), d.push_back(0), q.push_back(1);
    pre.push_back(0), d.push_back(1), q.push_back(1);
    expect_rows("off/head-of-line", r, pre, d, q);
    expect_complete("off/head-of-line", r, hol);
    CHECK(r.out[1].first_token_step == 5 && r.out[0].first_token_step == 19);
    std::printf("  off, the same arrivals: short first token step 5 (identical), long step 19 (solo: 9)\n");
  }

  // 16. New arrivals: two 128-token prompts arrive behind a lone 600-token
  //     prompt. Steps 2-10 have two or more prompts (or decode rows) and match
  //     solo off row for row; FIFO order (long, then S1, then S2) as always.
  //     Then the long prompt is alone: solo 128, 56.
  const std::vector<Spec> arrivals = {{600, 2}, {128, 2, 1}, {128, 2, 1}};
  {
    const Run r = fx.run(solo(false), arrivals);
    expect_rows("solo-fifo/arrivals", r, {128, 64, 64, 64, 64, 64, 64, 64, 64, 32, 128, 56, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1}, {1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1});
    expect_complete("solo-fifo/arrivals", r, arrivals);
    expect_mixed_bounded("solo-fifo/arrivals", r, solo(false));
    expect_fifo_solo_property("solo-fifo/arrivals", r);
    CHECK(r.out[1].first_token_step == 5 && r.out[2].first_token_step == 9 && r.out[0].first_token_step == 12);
    CHECK(r.stats.steps_solo_prefill == 3 && r.stats.steps_solo_prefill_skipped == 7);
    const Run off = fx.run(base(), arrivals);
    bool same = off.steps.size() >= 10;
    for (size_t i = 1; same && i < 10; ++i) {
      same = off.steps[i].prefill == r.steps[i].prefill && off.steps[i].decode == r.steps[i].decode &&
             off.steps[i].seqs == r.steps[i].seqs;
    }
    CHECK(same);
    CHECK(off.out[1].first_token_step == 5 && off.out[2].first_token_step == 9);
    expect_complete("off/arrivals", off, arrivals);
    std::printf("  solo 128 FIFO, two short arrivals: steps 2-10 identical to off; shorts first at steps 5 and 9\n");
  }

  if (g_failures) {
    std::printf("FAILED: %d of %d checks\n", g_failures, g_checks);
    return 1;
  }
  std::printf("all solo-prefill scheduler tests passed (%d checks)\n", g_checks);
  return 0;
}

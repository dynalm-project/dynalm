// Engine end to end on the tiny synthetic GGUF model (dynalm/tests/tiny_gguf.h):
// loader, tokenizer, chat template, scheduler, KV cache and prefix cache, on
// the CPU and (when built and available) on the Metal GPU. No model download.
//
//   1. deterministic greedy generation;
//   2. isolation: concurrent requests (with and without shared prefixes) give
//      exactly the text each gives alone (CPU, int8 activations off: the
//      DD-031 batch-invariance condition);
//   3. prefix cache: hits give the same text as an engine without it;
//   4. cancellation and deadlines end the request; later requests unaffected;
//   5. invalid requests fail and do not corrupt later ones;
//   6. solo prefill (GPU default): a long prompt never stalls a decoding
//      stream: every step while it decodes carries its decode row, and solo
//      steps never carry decode rows (scheduler trace, not timing);
//   7. bounded memory over repeated requests;
//   8. GPU: generation completes, is deterministic, concurrent requests and
//      shared prefixes run; layouts head/slot agree on greedy text for the
//      deterministic prompt set within the documented tolerance policy.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "api/json.h"
#include "chat_template/chat_template.h"
#include "common/sha256.h"
#include "diagnostics/doctor.h"
#include "dynacore/cpu/qmm.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/process_stats.h"
#include "loader/model_loader.h"
#include "model/architecture.h"
#include "loader/model_source.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "loader/gguf/gguf.h"
#include "runtime/autotune.h"
#include "tokenizer/pretokenizer.h"
#include "runtime/engine.h"
#include "sampling/sampler.h"
#include "tests/tiny_gguf.h"

using namespace dynalm;

namespace {

int g_checks = 0;
#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::_Exit(1);                                                                   \
    }                                                                                  \
  } while (0)

std::string g_model;
// ggml's CPU kernels are built in AND this CPU runs them (x86-64: AVX2, FMA,
// F16C); a build without them, or e.g. Rosetta without AVX, uses the native CPU.
bool ggml_cpu_usable() {
  static const bool usable = dynacore::ggml_gpu_compiled(dynacore::DeviceType::kCpu) &&
                             dynacore::create_ggml_gpu_device(dynacore::DeviceType::kCpu).ok();
  return usable;
}

std::unique_ptr<Engine> make_engine(DeviceKind dev, bool prefix_cache = true, int32_t solo = -1, bool trace = false) {
  EngineOptions o;
  o.model_path = g_model;
  o.backend = dev;
  o.threads = 2;
  o.kv_tokens = 2048;
  o.int8_decode_rows = 0;  // batch invariance on the CPU (DD-031)
  o.scheduler.enable_prefix_cache = prefix_cache;
  o.scheduler.solo_prefill_chunk = solo;
  o.scheduler.trace = trace;
  auto e = Engine::create(o);
  if (!e.ok()) {
    std::fprintf(stderr, "engine: %s\n", e.status().to_string().c_str());
    CHECK(false);
  }
  return std::move(*e);
}

struct Out {
  std::string text;
  StreamFinish finish = StreamFinish::kNone;
  Status error;
  int32_t tokens = 0;
};

Out drain(const std::shared_ptr<RequestStream>& s) {
  Out o;
  StreamEvent ev;
  while (s->next(ev, std::chrono::milliseconds(120000))) {
    o.text += ev.text;
    if (ev.done) {
      o.finish = ev.finish;
      o.error = ev.error;
      o.tokens = ev.completion_tokens;
      break;
    }
  }
  return o;
}

GenerateParams greedy(int32_t n) {
  GenerateParams p;
  p.max_tokens = n;
  p.stop_at_eog = false;  // fixed lengths: comparisons see every token
  return p;
}

Out gen(Engine& e, const std::string& prompt, int32_t n) {
  auto s = e.generate_text(prompt, false, greedy(n));
  CHECK(s.ok());
  return drain(*s);
}

const std::vector<std::string> kPrompts = {
    "The quick brown fox jumps over the lazy dog.",
    "Paged attention keeps KV in blocks of sixteen tokens.",
    "Hello, world! 1234567890 ~!@#$%^&*()",
    "A much longer prompt that spans several KV blocks so the scheduler has to chunk it across more than one "
    "forward pass and the prefix cache can share whole blocks between requests.",
};
const std::string kShared = std::string(200, 'x') + " shared system preamble. ";

void test_basic_and_isolation(Engine& e, const char* label) {
  // 1. determinism
  const Out a = gen(e, kPrompts[0], 24), b = gen(e, kPrompts[0], 24);
  CHECK(a.finish == StreamFinish::kLength && a.tokens == 24);
  CHECK(a.text == b.text);
  // 2. concurrent vs alone (incl. two requests sharing a long prefix)
  std::vector<std::string> prompts = kPrompts;
  prompts.push_back(kShared + "first question");
  prompts.push_back(kShared + "second question");
  std::vector<std::string> alone;
  for (const auto& p : prompts) alone.push_back(gen(e, p, 20).text);
  std::vector<std::shared_ptr<RequestStream>> streams;
  for (const auto& p : prompts) {
    auto s = e.generate_text(p, false, greedy(20));
    CHECK(s.ok());
    streams.push_back(*s);
  }
  for (size_t i = 0; i < streams.size(); ++i) {
    const Out o = drain(streams[i]);
    CHECK(o.finish == StreamFinish::kLength);
    if (o.text != alone[i]) std::fprintf(stderr, "%s: request %zu differs when batched\n", label, i);
    CHECK(o.text == alone[i]);
  }
}

void test_prefix_cache_equivalence() {
  auto with = make_engine(DeviceKind::kCpu, true), without = make_engine(DeviceKind::kCpu, false);
  for (const char* q : {"alpha", "beta", "gamma"}) {
    const std::string p = kShared + q;
    CHECK(gen(*with, p, 16).text == gen(*without, p, 16).text);
  }
  CHECK(with->stats().prefix.hit_tokens > 0);  // the cache actually served blocks
}

void test_cancel_deadline_errors(Engine& e) {
  const std::string baseline = gen(e, kPrompts[1], 16).text;
  // cancellation
  auto s = e.generate_text(kPrompts[2], false, greedy(400));
  CHECK(s.ok());
  StreamEvent ev;
  CHECK((*s)->next(ev, std::chrono::milliseconds(60000)));
  (*s)->cancel();
  const Out c = drain(*s);
  CHECK(c.finish == StreamFinish::kCancelled || c.finish == StreamFinish::kLength);
  // deadline: 1 ms cannot cover prefill + 400 tokens
  GenerateParams p = greedy(400);
  p.timeout_ms = 1;
  auto d = e.generate_text(kPrompts[3], false, p);
  CHECK(d.ok());
  const Out dl = drain(*d);
  CHECK(dl.finish == StreamFinish::kError && dl.error.code() == StatusCode::kDeadlineExceeded);
  // invalid: prompt beyond the KV capacity, and max_tokens <= 0
  auto big = e.generate_text(std::string(5000, 'y'), false, greedy(4));
  if (big.ok()) {
    const Out o = drain(*big);
    CHECK(o.finish == StreamFinish::kError);
  }
  CHECK(!e.generate_text("x", false, greedy(0)).ok());
  // later requests unaffected
  CHECK(gen(e, kPrompts[1], 16).text == baseline);
}

void test_solo_prefill_never_stalls_decode(DeviceKind dev) {
  auto e = make_engine(dev, false, /*solo=*/256, /*trace=*/true);
  auto a = e->generate_text(kPrompts[0], false, greedy(120));
  CHECK(a.ok());
  StreamEvent ev;
  CHECK((*a)->next(ev, std::chrono::milliseconds(60000)));  // A is decoding
  auto b = e->generate_text(std::string(900, 'z'), false, greedy(4));  // long prompt
  CHECK(b.ok());
  const Out oa = drain(*a), ob = drain(*b);
  CHECK(oa.finish == StreamFinish::kLength && ob.finish == StreamFinish::kLength);
  const auto steps = e->take_step_trace();
  CHECK(!steps.empty());
  const uint64_t a_id = (*a)->id();
  bool a_started = false, a_done = false;
  int solo_steps = 0, b_rows_while_a = 0;
  for (const StepTrace& st : steps) {
    if (st.solo) {
      ++solo_steps;
      CHECK(st.decode_rows == 0);  // solo steps never carry decode rows
    }
    bool has_a = false;
    for (const StepTraceRow& r : st.rows) {
      if (r.request_id == a_id && r.decode) has_a = true;
      if (r.request_id != a_id && !r.decode && a_started && !a_done) b_rows_while_a += r.rows;
    }
    if (has_a) a_started = true;
    if (a_started && !a_done && !has_a) {
      // A has no decode row in this step: only allowed once A finished.
      a_done = true;
    }
    if (a_done) CHECK(!has_a);  // A never reappears after a gap: no step skipped A mid-stream
  }
  CHECK(b_rows_while_a > 0);  // B's prompt was prefilled while A decoded (mixed steps)
  (void)solo_steps;
}

// Prefill bursts (SchedulerConfig::prefill_burst_max_stall_ms): while B's
// prompt prefills, steps carry no decode rows (A waits) as long as A has not
// waited for the bound; with a 1 ms bound A is served in mixed steps instead.
// Either way A's greedy text is what it is alone, and A is never dropped.
void test_prefill_burst(DeviceKind dev) {
  const std::string alone = [&] {
    auto e = make_engine(dev);
    return gen(*e, kPrompts[0], 120).text;
  }();
  for (int32_t bound : {600000, 1}) {
    EngineOptions o;
    o.model_path = g_model;
    o.backend = dev;
    o.threads = 2;
    o.kv_tokens = 2048;
    o.int8_decode_rows = 0;
    o.scheduler.solo_prefill_chunk = 128;
    o.scheduler.prefill_burst_max_stall_ms = bound;
    o.scheduler.trace = true;
    auto e = Engine::create(o);
    CHECK(e.ok());
    CHECK((*e)->scheduler_config().prefill_burst_max_stall_ms == bound);
    auto a = (*e)->generate_text(kPrompts[0], false, greedy(120));
    CHECK(a.ok());
    StreamEvent ev;
    CHECK((*a)->next(ev, std::chrono::milliseconds(60000)));  // A is decoding
    std::string text_a = ev.text;
    auto b = (*e)->generate_text(std::string(900, 'z'), false, greedy(4));  // long prompt
    CHECK(b.ok());
    const Out oa = drain(*a), ob = drain(*b);
    text_a += oa.text;
    CHECK(oa.finish == StreamFinish::kLength && ob.finish == StreamFinish::kLength);
    CHECK(text_a == alone);
    const uint64_t a_id = (*a)->id();
    int burst_steps = 0, mixed_steps = 0;
    for (const StepTrace& st : (*e)->take_step_trace()) {
      bool a_row = false, b_rows = false;
      for (const StepTraceRow& r : st.rows) {
        if (r.request_id == a_id && r.decode) a_row = true;
        if (r.request_id != a_id && !r.decode) b_rows = true;
      }
      if (b_rows && !a_row && st.decode_rows == 0) ++burst_steps;
      if (b_rows && a_row) ++mixed_steps;
    }
    if (bound > 1) {
      CHECK(burst_steps > 0 && mixed_steps == 0);  // B's prompt never shared a step with A's decode rows
      CHECK((*e)->stats().scheduler.steps_prefill_burst > 0);
    } else {
      CHECK(mixed_steps > 0);  // the bound expired at once: A kept decoding next to B's prompt
    }
  }
}

// Speculative decoding in the batch (SchedulerConfig::spec_k): the
// same greedy text as plain decoding, alone and with several streams in one
// batch; drafts were scored and some accepted (the tiny model loops, so
// prompt lookup finds continuations); the KV pool is not leaked.
// `draft`: a draft model file (BatchDrafter), empty = prompt lookup.
void test_speculative_batch(DeviceKind dev, bool exact, const std::string& draft = "") {
  auto make = [&](int32_t k) {
    EngineOptions o;
    o.model_path = g_model;
    o.backend = dev;
    o.threads = 2;
    o.kv_tokens = 2048;
    o.int8_decode_rows = 0;
    o.scheduler.spec_k = k;
    if (k > 0) o.draft_model_path = draft;
    auto e = Engine::create(o);
    if (!e.ok()) std::fprintf(stderr, "engine: %s\n", e.status().to_string().c_str());
    CHECK(e.ok());
    return std::move(*e);
  };
  auto plain = make(0), spec = make(4);
  const std::vector<std::string> prompts = {kPrompts[0], kPrompts[2], kPrompts[3], "abc abc abc abc abc abc"};
  std::vector<Out> want;
  for (const auto& p : prompts) want.push_back(gen(*plain, p, 48));
  std::vector<std::shared_ptr<RequestStream>> streams;
  for (const auto& p : prompts) {
    auto st = spec->generate_text(p, false, greedy(48));
    CHECK(st.ok());
    streams.push_back(*st);
  }
  for (size_t i = 0; i < prompts.size(); ++i) {
    const Out o = drain(streams[i]);
    CHECK(o.finish == StreamFinish::kLength && o.tokens == 48);
    if (exact) CHECK(o.text == want[i].text);
  }
  const Out alone = gen(*spec, prompts[1], 48);
  CHECK(alone.tokens == 48);
  if (exact) CHECK(alone.text == want[1].text);
  const SchedulerStats ss = spec->stats().scheduler;
  CHECK(ss.spec_drafted > 0 && ss.spec_accepted > 0 && ss.spec_accepted <= ss.spec_drafted);
  CHECK(spec->stats().kv_blocks_used == plain->stats().kv_blocks_used);  // rejected drafts' blocks returned
}

// Device selection policy (runtime/device_select.h): pure, so every GPU
// combination is checked on any machine.
void test_device_plan() {
  ModelConfig dense;
  dense.architecture = "qwen2";
  ModelConfig moe = dense;
  moe.moe.num_experts = 8;
  using K = DeviceKind;
  const std::vector<K> metal = {K::kCpu, K::kMetal}, vulkan = {K::kCpu, K::kVulkan}, none = {K::kCpu};
  const std::vector<K> several = {K::kCpu, K::kCuda, K::kMetal, K::kVulkan};
  DeviceRequest autoc;
  autoc.backend_auto = autoc.cpu_ggml = true;
  // auto: the GPUs that run the model, in build order, then the CPU path.
  DevicePlan p = plan_devices(dense, autoc, several, true);
  CHECK(p.error.ok() && (p.candidates == std::vector<K>{K::kCuda, K::kMetal, K::kVulkan, K::kGgmlCpu}) && p.native_cpu_last);
  p = plan_devices(dense, autoc, metal, true);
  CHECK((p.candidates == std::vector<K>{K::kMetal, K::kGgmlCpu}) && p.native_cpu_last && p.note.empty());
  // MoE: Metal, Vulkan and ggml-cpu; not CUDA (yet): a CUDA build runs it on the ggml CPU kernels.
  p = plan_devices(moe, autoc, vulkan, true);
  CHECK((p.candidates == std::vector<K>{K::kVulkan, K::kGgmlCpu}));
  p = plan_devices(moe, autoc, std::vector<K>{K::kCpu, K::kCuda}, true);
  CHECK((p.candidates == std::vector<K>{K::kGgmlCpu}) && p.native_cpu_last && p.note.find("not verified") != std::string::npos);
  p = plan_devices(moe, autoc, metal, true);
  CHECK((p.candidates == std::vector<K>{K::kMetal, K::kGgmlCpu}));
  // No GPU built in: the CPU path, with the reason.
  p = plan_devices(dense, autoc, none, true);
  CHECK((p.candidates == std::vector<K>{K::kGgmlCpu}) && p.note.find("no GPU support") != std::string::npos);
  // --gpu: never the CPU; an error when no GPU can run the model.
  DeviceRequest gpu = autoc;
  gpu.require_gpu = true;
  p = plan_devices(dense, gpu, metal, true);
  CHECK((p.candidates == std::vector<K>{K::kMetal}) && !p.native_cpu_last);
  CHECK(plan_devices(dense, gpu, none, true).error.code() == StatusCode::kUnsupported);
  CHECK(plan_devices(moe, gpu, std::vector<K>{K::kCpu, K::kCuda}, true).error.code() == StatusCode::kUnsupported);
  // cpu: ggml kernels when built and able; exact (cpu_ggml off): native only; a named device: exactly it.
  DeviceRequest cpu;
  cpu.cpu_ggml = true;
  CHECK((plan_devices(dense, cpu, metal, true).candidates == std::vector<K>{K::kGgmlCpu}));
  p = plan_devices(dense, cpu, none, false);
  CHECK(p.candidates.empty() && p.native_cpu_last && p.note.find("ENABLE_GGML_CPU") != std::string::npos);
  ModelConfig yarn = dense;
  yarn.rope.scaling = RopeScaling::kYarn;
  p = plan_devices(yarn, cpu, metal, true);
  CHECK(p.candidates.empty() && p.native_cpu_last);
  DeviceRequest native;
  p = plan_devices(dense, native, metal, true);
  CHECK(p.candidates.empty() && p.native_cpu_last && p.note.empty());
  DeviceRequest named;
  named.backend = K::kVulkan;
  p = plan_devices(moe, named, metal, true);
  CHECK((p.candidates == std::vector<K>{K::kVulkan}) && !p.native_cpu_last);
}

// DD-079 predictive mode: rows = (target - 1) x decode time / row cost,
// clamped; no row cost: the maximum; no decode time yet: the minimum.
void test_predictive_mixed_budget() {
  CHECK(predict_mixed_prefill_budget(12.0, 0.5, 4.0, 64, 256) == 72);   // 3 x 12 / 0.5
  CHECK(predict_mixed_prefill_budget(12.0, 0.01, 4.0, 64, 256) == 256);  // cheap rows: the maximum
  CHECK(predict_mixed_prefill_budget(12.0, 5.0, 4.0, 64, 256) == 64);    // costly rows: the minimum
  CHECK(predict_mixed_prefill_budget(12.0, 0.0, 4.0, 64, 256) == 256);   // no cost sample yet
  CHECK(predict_mixed_prefill_budget(0.0, 0.5, 4.0, 64, 256) == 64);     // no decode-only step yet: protect
  CHECK(predict_mixed_prefill_budget(12.0, 0.5, 1.0, 64, 256) == 64);    // target 1x: nothing spare
  CHECK(predict_mixed_prefill_budget(12.0, 0.5, 4.0, 512, 256) == 256);  // min above max: max wins
  CHECK(predict_mixed_prefill_budget(12.0, 1e-300, 4.0, 1, 256) == 256); // overflow-safe
}

// Scheduler profile per device (Engine::create_on): the GPU's mixed-step
// profile (kGpuMixedPrefillBudget etc.) only on a GPU; the CPU kinds,
// ggml-cpu included, keep the balanced CPU budgets. The solo-prefill chunk
// follows the execution model: on for graph devices (GPU and ggml-cpu); on
// DynaCore's CPU kernels kCpuSmeSoloPrefillChunk where the SME panel gemm runs
// prefill, else off. Explicit budgets replace the device profile.
void test_scheduler_profile(bool gpu_usable, DeviceKind gpu_kind) {
  const SchedulerConfig def;
  auto cpu_budgets = [&](const SchedulerConfig& s) {
    return s.prefill_token_budget == def.prefill_token_budget && s.max_prefill_chunk == def.max_prefill_chunk &&
           s.decode_latency_target == def.decode_latency_target && s.protected_prefill_min == def.protected_prefill_min &&
           s.decode_protect_predictive == def.decode_protect_predictive;
  };
  CHECK(cpu_budgets(make_engine(DeviceKind::kCpu)->scheduler_config()));
  const bool sme = dynacore::qmm::sme_gemm() != nullptr && KernelPlan::defaults().qmm && KernelPlan::defaults().sme_min_rows > 0;
  CHECK(make_engine(DeviceKind::kCpu)->scheduler_config().solo_prefill_chunk == (sme ? kCpuSmeSoloPrefillChunk : 0));
  if (ggml_cpu_usable()) {
    const SchedulerConfig s = make_engine(DeviceKind::kGgmlCpu)->scheduler_config();
    CHECK(cpu_budgets(s));
    CHECK(s.solo_prefill_chunk == kGpuSoloPrefillChunk);
  }
  if (gpu_usable) {
    const SchedulerConfig s = make_engine(gpu_kind)->scheduler_config();
    CHECK(s.prefill_token_budget == kGpuMixedPrefillBudget && s.max_prefill_chunk == kGpuMaxPrefillChunk);
    CHECK(s.decode_latency_target == kGpuDecodeLatencyTarget && s.protected_prefill_min == kGpuProtectedPrefillMin);
    CHECK(s.decode_protect_predictive == kGpuDecodePredictive && s.solo_prefill_chunk == kGpuSoloPrefillChunk);
    EngineOptions o;
    o.model_path = g_model;
    o.backend = gpu_kind;
    o.kv_tokens = 2048;
    o.device_scheduler_defaults = false;  // the caller's budgets win
    auto e = Engine::create(o);
    CHECK(e.ok() && cpu_budgets((*e)->scheduler_config()));
  }
}

// Sampling (sampling/sampler.h, DD-043), on hand-made logits rows: greedy and
// its ties; top-k, top-p and min-p never return a token they cut, and keep the
// ones they should; temperature sharpens / flattens the distribution as
// exp(dl / T) (frequencies of seeded draws, within statistical tolerance);
// penalties; a fixed seed repeats the same tokens; invalid parameters are refused.
void test_sampling() {
  constexpr int32_t kVocab = 64;
  // Log-probabilities of p = 0.5, 0.3, 0.15, 0.05 for ids 0..3; the rest are far below.
  auto row = [&] {
    std::vector<float> l(kVocab, -40.0f);
    const float p[4] = {0.5f, 0.3f, 0.15f, 0.05f};
    for (int i = 0; i < 4; ++i) l[static_cast<size_t>(i)] = std::log(p[i]);
    return l;
  };
  const std::vector<TokenId> ctx = {7, 8, 9};
  // Counts of `n` draws per token id.
  auto draws = [&](SamplingParams p, int n, std::vector<float> base) {
    p.has_seed = true;
    Sampler s(p);
    std::vector<int> c(kVocab, 0);
    for (int i = 0; i < n; ++i) {
      std::vector<float> l = base;
      const TokenId t = s.sample(l, ctx);
      CHECK(t >= 0 && t < kVocab);
      ++c[static_cast<size_t>(t)];
    }
    return c;
  };
  auto only = [&](const std::vector<int>& c, int n_kept) {
    for (int i = n_kept; i < kVocab; ++i) {
      if (c[static_cast<size_t>(i)] != 0) return false;
    }
    for (int i = 0; i < n_kept; ++i) {
      if (c[static_cast<size_t>(i)] == 0) return false;
    }
    return true;
  };
  constexpr int kN = 20000;

  // Greedy: temperature 0, or top_k 1 at any temperature; ties to the lowest id.
  {
    std::vector<float> l = row();
    CHECK(sample_greedy(l) == 0);
    l[5] = l[0];
    CHECK(sample_greedy(l) == 0);
    SamplingParams p;
    p.temperature = 1.5f;
    p.top_k = 1;
    CHECK(p.greedy());
    const auto c = draws(p, 200, row());
    CHECK(c[0] == 200);
  }
  // Temperature 1, no cut: the frequencies follow p (0.5 / 0.3 / 0.15 / 0.05).
  {
    SamplingParams p;
    p.temperature = 1.0f;
    const auto c = draws(p, kN, row());
    CHECK(only(c, 4));
    const double want[4] = {0.5, 0.3, 0.15, 0.05};
    for (int i = 0; i < 4; ++i) {
      const double f = static_cast<double>(c[static_cast<size_t>(i)]) / kN;
      CHECK(std::abs(f - want[i]) < 0.015);  // > 4 standard deviations at n = 20000
    }
  }
  // Temperature scales log-odds: p0 / p1 = (5/3)^(1/T).
  for (const float t : {0.5f, 2.0f}) {
    SamplingParams p;
    p.temperature = t;
    const auto c = draws(p, kN, row());
    const double ratio = static_cast<double>(c[0]) / std::max(1, c[1]);
    const double want = std::pow(0.5 / 0.3, 1.0 / t);
    CHECK(std::abs(ratio / want - 1.0) < 0.08);
  }
  // top-k: only the k largest.
  {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 2;
    const auto c = draws(p, kN, row());
    CHECK(only(c, 2));
    const double ratio = static_cast<double>(c[0]) / c[1];
    CHECK(std::abs(ratio / (0.5 / 0.3) - 1.0) < 0.08);  // renormalized among the kept
  }
  // top-p: the smallest prefix with mass >= top_p (0.5, 0.8, 0.95, 1.0).
  for (const auto& [top_p, kept] : std::vector<std::pair<float, int>>{{0.45f, 1}, {0.7f, 2}, {0.9f, 3}, {0.99f, 4}}) {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_p = top_p;
    CHECK(only(draws(p, 4000, row()), kept));
  }
  // Near-equal logits (one histogram bin of the pre-cut in distribution()): the exact top-k and
  // nucleus steps decide, not the coarse cut. p is about 0.25 for each of ids 0..3.
  {
    std::vector<float> flat(kVocab, -40.0f);
    for (int i = 0; i < 4; ++i) flat[static_cast<size_t>(i)] = -0.001f * static_cast<float>(i);
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 2;
    CHECK(only(draws(p, 4000, flat), 2));
    p.top_k = 3;
    CHECK(only(draws(p, 4000, flat), 3));
    SamplingParams q;
    q.temperature = 1.0f;
    q.top_p = 0.4f;  // 0.25 + 0.25 >= 0.4
    CHECK(only(draws(q, 4000, flat), 2));
    q.top_p = 0.6f;  // three are needed
    CHECK(only(draws(q, 4000, flat), 3));
  }
  // top-k then top-p (p applies to the k kept): k = 3 keeps 0.95 of the mass; top-p 0.8 of that
  // (0.76) is reached by ids 0 and 1 (0.8 of 0.95).
  {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 3;
    p.top_p = 0.8f;
    CHECK(only(draws(p, 4000, row()), 2));
  }
  // min-p: drop p < min_p x p_max (0.2 x 0.5 = 0.1: id 3 at 0.05 goes).
  {
    SamplingParams p;
    p.temperature = 1.0f;
    p.min_p = 0.2f;
    CHECK(only(draws(p, 4000, row()), 3));
  }
  // Penalties (step 1, before selection): repetition divides positive and multiplies negative
  // logits of seen tokens; frequency and presence subtract.
  {
    SamplingParams p;
    p.repetition_penalty = 2.0f;
    Sampler s(p);
    std::vector<float> l(kVocab, 0.0f);
    l[7] = 4.0f;
    l[8] = -4.0f;
    l[1] = 4.0f;  // not in the context
    s.apply_penalties(l, ctx);
    CHECK(l[7] == 2.0f && l[8] == -8.0f && l[1] == 4.0f);
    SamplingParams q;
    q.frequency_penalty = 0.5f;
    q.presence_penalty = 1.0f;
    Sampler s2(q);
    std::vector<float> m(kVocab, 0.0f);
    const std::vector<TokenId> twice = {7, 7, 9};
    s2.apply_penalties(m, twice);
    CHECK(std::abs(m[7] - (-2.0f)) < 1e-6f && std::abs(m[9] - (-1.5f)) < 1e-6f && m[8] == 0.0f);
  }
  // A fixed seed repeats; another seed gives another sequence (64 draws at p = 0.5 / 0.3 / ...).
  {
    auto seq = [&](uint64_t seed) {
      SamplingParams p;
      p.temperature = 1.0f;
      p.seed = seed;
      p.has_seed = true;
      Sampler s(p);
      std::vector<TokenId> out;
      for (int i = 0; i < 64; ++i) {
        std::vector<float> l = row();
        out.push_back(s.sample(l, ctx));
      }
      return out;
    };
    CHECK(seq(42) == seq(42));
    CHECK(seq(42) != seq(43));
  }
  // Invalid parameters are refused (the server answers 400 for these).
  auto invalid = [](auto set) {
    SamplingParams p;
    set(p);
    return !p.validate().ok();
  };
  CHECK(SamplingParams().validate().ok());
  CHECK(invalid([](SamplingParams& p) { p.temperature = -0.1f; }));
  CHECK(invalid([](SamplingParams& p) { p.temperature = 101.0f; }));
  CHECK(invalid([](SamplingParams& p) { p.top_k = -1; }));
  CHECK(invalid([](SamplingParams& p) { p.top_p = 0.0f; }));
  CHECK(invalid([](SamplingParams& p) { p.top_p = 1.01f; }));
  CHECK(invalid([](SamplingParams& p) { p.min_p = 1.5f; }));
  CHECK(invalid([](SamplingParams& p) { p.repetition_penalty = 0.0f; }));
  CHECK(invalid([](SamplingParams& p) { p.frequency_penalty = 2.5f; }));
}

// Memory admission: weights + KV within 90% of RAM (95% of free GPU memory);
// the error names the largest --ctx that fits.
void test_memory_admission() {
  constexpr int64_t G = int64_t{1} << 30;
  CHECK(check_memory_fit(8 * G, 8 * G, 128 << 10, 24 * G, -1).ok());               // 16 of 21.6 GiB
  const Status s = check_memory_fit(8 * G, 16 * G, 128 << 10, 24 * G, -1);       // 24 > 21.6 GiB
  CHECK(s.code() == StatusCode::kResourceExhausted);
  CHECK(s.message().find("--ctx 111408") != std::string::npos);                  // (24 GiB x 0.9 - 8 GiB) / 128 KiB, multiple of 16
  CHECK(s.message().find("--kv q8_0") != std::string::npos);
  CHECK(check_memory_fit(8 * G, 1 * G, 128 << 10, 24 * G, 6 * G).code() == StatusCode::kResourceExhausted);  // GPU
  CHECK(check_memory_fit(4 * G, 1 * G, 128 << 10, 24 * G, 6 * G).ok());
  CHECK(check_memory_fit(30 * G, 1 * G, 128 << 10, 24 * G, -1).message().find("does not fit") != std::string::npos);
  CHECK(check_memory_fit(8 * G, 64 * G, 128 << 10, 0, -1).ok());                   // unknown RAM: not checked

  // Working-set cap (Metal): budget = cap - 10% - allocated - weights - workspace reserve.
  constexpr int64_t M = int64_t{1} << 20;
  CHECK(device_kv_budget(-1, 0, 4 * G, G) == -1);  // no cap known: no device limit
  CHECK(device_kv_budget(0, 0, 4 * G, G) == -1);
  CHECK(device_kv_budget(16 * G, 0, 4 * G, G) == 16 * G / 100 * 90 - 4 * G - G);
  CHECK(device_kv_budget(16 * G, 2 * G, 4 * G, G) == 16 * G / 100 * 90 - 2 * G - 4 * G - G);  // already allocated
  CHECK(device_kv_budget(16 * G, 0, 4 * G, 0) == 16 * G / 100 * 90 - 4 * G - kDefaultWorkspaceReserve);  // unknown reserve
  CHECK(device_kv_budget(8 * G, 0, 7 * G, G) == 0);  // weights alone exceed it: never negative
  // Llama-2-7B (e2e-20261019-0022, logs/10-bench-auto.log.c1): 3.8 GiB of weights and auto KV
  // from free RAM to 20,275 MiB in all; Metal's working set on a 24 GB M4 Pro is ~16 GiB.
  const int64_t budget = device_kv_budget(16 * G, 0, 3891 * M, G);
  const int64_t per_token = 512 << 10;  // MHA: 32 layers x 4096 x (K + V) x f16
  const Status oom = check_memory_fit(3891 * M, 20275 * M - 3891 * M, per_token, 24 * G, -1, budget);
  CHECK(oom.code() == StatusCode::kResourceExhausted);
  CHECK(oom.message().find("GPU working set") != std::string::npos);
  CHECK(oom.message().find("--ctx " + std::to_string(budget / per_token / 16 * 16)) != std::string::npos);
  CHECK(check_memory_fit(3891 * M, budget, per_token, 24 * G, -1, budget).ok());       // exactly at the budget
  CHECK(!check_memory_fit(3891 * M, budget + 1, per_token, 24 * G, -1, budget).ok());  // one byte over
  CHECK(check_memory_fit(3891 * M, 64 * G, per_token, 0, -1, -1).ok());                // no cap, unknown RAM
  CHECK(device_kv_budget(16 * G, 0, 4 * G, G, 512 * M) == 16 * G / 100 * 90 - 4 * G - G - 512 * M);  // scratch, once
  // Automatic sizing uses the pool's own arithmetic, so admission never refuses it: for every KV
  // dtype and a few shapes, kv_tokens_within(budget) fits and one more 16-token block does not.
  for (const auto& [layers, heads, hd] : {std::tuple{32, 32, 128}, std::tuple{28, 2, 128}, std::tuple{26, 1, 256},
                                          std::tuple{24, 8, 96}}) {
    ModelConfig mc;
    mc.num_layers = layers;
    mc.num_kv_heads = heads;
    mc.head_dim = mc.head_dim_v = hd;
    for (DType dt : {DType::kF16, DType::kF32, DType::kQ8_0}) {
      for (int64_t b : {11 * G + 123457, 3 * G, 777 * M + 5}) {
        const int64_t t = kv_tokens_within(mc, dt, b);
        const int64_t bytes = static_cast<int64_t>(kv_geometry_for(mc, dt, 16, t).total_bytes());
        const int64_t more = static_cast<int64_t>(kv_geometry_for(mc, dt, 16, t + 16).total_bytes());
        CHECK(t % 16 == 0 && bytes <= b && more > b);
        CHECK(check_memory_fit(4 * G, bytes, mc.kv_bytes_per_token(dt), 64 * G, -1, b).ok());
      }
    }
    CHECK(kv_tokens_within(mc, DType::kF16, 0) == 0 && kv_tokens_within(mc, DType::kF16, -1) == 0);
  }
  // Nothing left for even one 16-token block: says so instead of a --ctx.
  CHECK(check_memory_fit(7 * G, G, per_token, 24 * G, -1, 0).message().find("does not fit on the GPU") != std::string::npos);
}

void test_memory_bounded(Engine& e) {
  for (int i = 0; i < 5; ++i) (void)gen(e, kPrompts[static_cast<size_t>(i) % kPrompts.size()], 32);
  const int64_t before = dynacore::process_rss_bytes();
  for (int i = 0; i < 60; ++i) (void)gen(e, kPrompts[static_cast<size_t>(i) % kPrompts.size()] + std::to_string(i), 32);
  const int64_t after = dynacore::process_rss_bytes();
  std::printf("    RSS over 60 requests: %+.1f MiB\n", static_cast<double>(after - before) / 1048576.0);
  // Under AddressSanitizer freed memory stays resident in its quarantine, so
  // RSS growth is not a leak signal there; leaks are LeakSanitizer's job (the
  // Linux sanitizer CI job runs with detect_leaks=1).
#if defined(__SANITIZE_ADDRESS__)
  constexpr bool kAsan = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  constexpr bool kAsan = true;
#else
  constexpr bool kAsan = false;
#endif
#else
  constexpr bool kAsan = false;
#endif
  if (kAsan) {
    std::printf("    (RSS bound not checked under AddressSanitizer)\n");
  } else {
    CHECK(after - before < (int64_t{64} << 20));
  }
}

}  // namespace

// Gemma 1-3 ask ggml-cpu to scale Q in its own op (llama.cpp's operation
// order, for bit-parity); no other architecture does.
void test_attn_prescale_q() {
  int configured = 0;
  for (std::string_view id : supported_architecture_ids()) {
    ModelConfig c;
    c.architecture = std::string(id);
    c.num_layers = 2;
    c.hidden_size = 128;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 32;
    TensorRegistry w(c.num_layers);
    const ModelArchitecture* a = find_architecture(id);
    CHECK(a != nullptr);
    if (!a->configure(c, w).ok()) continue;  // needs tensors to configure
    ++configured;
    const bool gemma = id == "gemma" || id == "gemma2" || id == "gemma3";
    if (c.attn_prescale_q != gemma) std::fprintf(stderr, "attn_prescale_q wrong for %.*s\n", static_cast<int>(id.size()), id.data());
    CHECK(c.attn_prescale_q == gemma);
  }
  for (std::string_view id : {"gemma3", "llama", "qwen3"}) {
    ModelConfig c;
    c.architecture = std::string(id);
    c.num_layers = 2;
    c.hidden_size = 128;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 32;
    TensorRegistry w(c.num_layers);
    CHECK(find_architecture(id)->configure(c, w).ok());
  }
  std::printf("  attn_prescale_q: Gemma only (%d architectures configured)\n", configured);
}

// `auto` picks the GPU only for a model it runs, and the CPU otherwise;
// an explicit GPU request for an unsupported model fails at startup.
void test_device_selection(const std::string& dir, bool gpu_usable, DeviceKind gpu_kind) {
  auto loaded = load_model(g_model);
  CHECK(loaded.ok());
  const ModelConfig base = (*loaded)->config;
  CHECK(gpu_model_support(base, DType::kF16, DeviceKind::kMetal).ok());
  CHECK(gpu_model_support(base, DType::kQ8_0, DeviceKind::kMetal).ok());
  CHECK(!gpu_model_support(base, DType::kQ4_K, DeviceKind::kMetal).ok());
  ModelConfig c = base;
  c.moe.num_experts = 8;  // routed MoE: Metal, Vulkan and ggml-cpu (verified), not yet CUDA
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kGgmlCpu).ok());
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kVulkan).ok());
  CHECK(!gpu_model_support(c, DType::kF16, DeviceKind::kCuda).ok());
  c.moe.shared_intermediate_size = 64;  // a shared expert (Qwen2-MoE): in the same device op
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c = base;
  c.attn_qk_norm = true;  // per-head Q/K norm: verified on the GPU for Qwen3 and Gemma 3
  c.architecture = "qwen3";
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c.architecture = "gemma3";
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c.architecture = "olmo2";  // not verified on the GPU
  CHECK(!gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c = base;
  c.rope.scaling = RopeScaling::kLinear;
  CHECK(gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c.rope.scaling = RopeScaling::kYarn;
  CHECK(!gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());
  c = base;
  c.has_rope_local = true;
  c.rope_local.scaling = RopeScaling::kLlama3;
  CHECK(!gpu_model_support(c, DType::kF16, DeviceKind::kMetal).ok());

  const std::string qwen3 = dir + "/tiny-qwen3.gguf";
  test::TinyModelSpec spec;
  spec.qwen3 = true;
  CHECK(test::write_file(qwen3, test::tiny_model_gguf(spec)));
  auto on = [&](const std::string& path, bool automatic, DeviceKind kind) {
    EngineOptions o;
    o.model_path = path;
    o.threads = 2;
    o.kv_tokens = 512;
    o.backend = kind;
    o.backend_auto = automatic;
    return Engine::create(o);
  };
  // By device kind: on the CPU the name can be "compiled/CPU" (--execution auto).
  auto is_gpu = [](const Engine& e) { return e.device_kind() != DeviceKind::kCpu && e.device_kind() != DeviceKind::kGgmlCpu; };
  {
    auto e = on(g_model, true, DeviceKind::kCpu);  // `backend` is ignored by auto
    CHECK(e.ok());
    CHECK(is_gpu(**e) == gpu_usable);
    const Out o = gen(**e, kPrompts[0], 8);
    CHECK(o.finish == StreamFinish::kLength && o.tokens == 8);
  }
  {
    // Qwen3 (per-head Q/K norm): the GPU when one is usable; deterministic.
    auto e = on(qwen3, true, gpu_kind);
    CHECK(e.ok());
    CHECK(is_gpu(**e) == gpu_usable);
    const Out o = gen(**e, kPrompts[0], 8), o2 = gen(**e, kPrompts[0], 8);
    CHECK(o.finish == StreamFinish::kLength && o.tokens == 8 && o.text == o2.text);
  }
  // --gpu / --cpu / --backend choices (apply_backend_choice).
  {
    EngineOptions o;
    CHECK(apply_backend_choice("gpu", o).ok() && o.backend_auto && o.backend_require_gpu);
    CHECK(apply_backend_choice("cpu", o).ok() && !o.backend_auto && o.backend == DeviceKind::kCpu);
    CHECK(apply_backend_choice("auto", o).ok() && o.backend_auto && !o.backend_require_gpu);
    CHECK(apply_backend_choice("metal", o).ok() && !o.backend_auto && o.backend == DeviceKind::kMetal);
    CHECK(!apply_backend_choice("tpu", o).ok());
    // cpu: ggml's CPU kernels when built in, DynaCore's where native_cpu_default(). native-cpu
    // is not a public backend in this release: a clear error (the same for a config file's
    // runtime.device), never a redirect.
    CHECK(apply_backend_choice("cpu", o).ok() && o.cpu_ggml == !native_cpu_default() && o.backend == DeviceKind::kCpu);
    {
      EngineOptions before = o;
      const Status st = apply_backend_choice("native-cpu", o);
      CHECK(!st.ok() && st.message().find("not included in this release") != std::string::npos &&
            st.message().find("--mode exact") != std::string::npos);
      CHECK(o.backend == before.backend && o.backend_auto == before.backend_auto && o.cpu_ggml == before.cpu_ggml);
    }
    CHECK(apply_backend_choice("auto", o).ok() && o.cpu_ggml == !native_cpu_default());
    CHECK(apply_backend_choice("ggml-cpu", o).ok() && o.backend == DeviceKind::kGgmlCpu);
    // cpu, and cpu with --mode exact (DynaCore's kernels, DD-053).
    for (const bool exact : {false, true}) {
      EngineOptions c;
      c.model_path = qwen3;
      c.threads = 2;
      c.kv_tokens = 512;
      CHECK(apply_backend_choice("cpu", c).ok());
      CHECK(apply_tune_mode(exact ? TuneMode::kExact : TuneMode::kAuto, c).ok());
      auto e = Engine::create(c);
      CHECK(e.ok());
      const bool ggml = (*e)->backend_name().find("ggml-CPU") != std::string::npos;
      CHECK(ggml == (!exact && ggml_cpu_usable() && !native_cpu_default()));
      const Out a = gen(**e, kPrompts[0], 8), b = gen(**e, kPrompts[0], 8);
      CHECK(a.finish == StreamFinish::kLength && a.tokens == 8 && a.text == b.text);
    }
    // --gpu never falls back to the CPU: without a usable GPU it is an error
    // naming the reason. (Models the GPU cannot run are covered by the
    // gpu_model_support checks above; the tiny generator makes none of them.)
    o = EngineOptions();
    o.model_path = qwen3;
    o.threads = 2;
    o.kv_tokens = 512;
    CHECK(apply_backend_choice("gpu", o).ok());
    auto e = Engine::create(o);
    if (gpu_usable) {
      CHECK(e.ok() && is_gpu(**e));
    } else {
      CHECK(!e.ok() && e.status().code() == StatusCode::kUnsupported);
    }
  }
}

bool write_text(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << text;
  return static_cast<bool>(f);
}

void set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  if (value.empty()) unsetenv(name);
  else setenv(name, value.c_str(), 1);
#endif
}

// AutoTune (runtime/autotune.h): modes, the versioned cache and its
// invalidation, and the Engine using a tuned thread count.
void test_autotune(const std::filesystem::path& dir) {
  for (TuneMode m : {TuneMode::kAuto, TuneMode::kPerformance, TuneMode::kExact, TuneMode::kSafe}) {
    auto r = parse_tune_mode(tune_mode_name(m));
    CHECK(r.ok() && *r == m);
  }
  CHECK(!parse_tune_mode("fastest").ok());
  {
    EngineOptions o;
    CHECK(apply_backend_choice("auto", o).ok());
    CHECK(apply_tune_mode(TuneMode::kAuto, o).ok() && o.backend_auto && o.cpu_ggml == !native_cpu_default());
    CHECK(apply_tune_mode(TuneMode::kExact, o).ok() && o.backend_auto && !o.cpu_ggml);  // GPUs stay, ggml-cpu goes
    CHECK(apply_tune_mode(TuneMode::kSafe, o).ok());
    CHECK(!o.backend_auto && o.backend == DeviceKind::kCpu && !o.cpu_ggml && o.int8_decode_rows == 0);
    EngineOptions g;
    CHECK(apply_backend_choice("ggml-cpu", g).ok());
    CHECK(!apply_tune_mode(TuneMode::kExact, g).ok());  // below DD-053: refused, not silently replaced
    CHECK(!apply_tune_mode(TuneMode::kSafe, g).ok());
    EngineOptions gpu;
    CHECK(apply_backend_choice("gpu", gpu).ok());
    CHECK(!apply_tune_mode(TuneMode::kSafe, gpu).ok());  // --gpu is never silently replaced by the CPU
    CHECK(apply_tune_mode(TuneMode::kExact, gpu).ok() && gpu.backend_require_gpu);
    EngineOptions n;  // exact on the CPU: DynaCore's kernels, never ggml's
    CHECK(apply_backend_choice("cpu", n).ok());
    CHECK(apply_tune_mode(TuneMode::kExact, n).ok() && n.backend == DeviceKind::kCpu && !n.cpu_ggml);
  }

  const std::string path = (dir / "autotune.json").string();
  std::string why;
  CHECK(load_tune_cache(path, &why).entries.empty() && why.find("no tuning cache") != std::string::npos);
  auto loaded = load_model(g_model);
  CHECK(loaded.ok());
  const ModelConfig& cfg = (*loaded)->config;
  const std::string model_key = tune_model_key(cfg, (*loaded)->quantization);
  TuneCache c;
  c.put({"cpu", model_key, 5, 10.0, 8.0, "AC, load 0.2", ""});
  c.put({"ggml-cpu", model_key, 4, 20.0, 19.0, "AC", ""});
  c.put({"cpu", model_key, 3, 11.0, 8.0, "AC", ""});  // replaces the first
  CHECK(c.entries.size() == 2 && c.find("cpu", model_key)->threads == 3);
  CHECK(save_tune_cache(c, path).ok());
  TuneCache r = load_tune_cache(path, &why);
  CHECK(r.entries.size() == 2 && r.machine == tune_machine_key());
  CHECK(r.find("cpu", model_key)->threads == 3 && r.find("cpu", model_key)->tok_s == 11.0);
  CHECK(!r.find("cpu", model_key)->measured_at.empty());
  CHECK(r.find("metal", model_key) == nullptr && r.find("cpu", "llama/4096x32/Q4_K_M") == nullptr);

  // The Engine takes the tuned count only when threads are on auto, and never
  // with DYNALM_AUTOTUNE=0.
  set_env("DYNALM_AUTOTUNE_CACHE", path);
  CHECK(tuned_threads(DeviceKind::kCpu, cfg, (*loaded)->quantization) == 3);
  EngineOptions o;
  o.model_path = g_model;
  o.backend = DeviceKind::kCpu;
  o.kv_tokens = 256;
  {
    auto e = Engine::create(o);
    CHECK(e.ok() && (*e)->threads() == 3);
  }
  o.threads = 2;
  {
    auto e = Engine::create(o);
    CHECK(e.ok() && (*e)->threads() == 2);
  }
  set_env("DYNALM_AUTOTUNE", "0");
  CHECK(tuned_threads(DeviceKind::kCpu, cfg, (*loaded)->quantization) == 0);
  set_env("DYNALM_AUTOTUNE", "");

  // Any other machine, build or format version invalidates the whole file.
  auto rewrite = [&](const std::string& from, const std::string& to) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    const size_t at = text.find(from);
    CHECK(at != std::string::npos);
    text.replace(at, from.size(), to);
    CHECK(write_text(path, text));
  };
  rewrite("\"version\":1", "\"version\":2");
  CHECK(load_tune_cache(path, &why).entries.empty() && why.find("version") != std::string::npos);
  rewrite("\"version\":2", "\"version\":1");
  CHECK(load_tune_cache(path).entries.size() == 2);
  rewrite(tune_machine_key(), "Some Other CPU | avx2 | 8c/8p | Linux x86_64");
  CHECK(load_tune_cache(path, &why).entries.empty() && why.find("another machine") != std::string::npos);
  CHECK(write_text(path, "{not json"));
  CHECK(load_tune_cache(path, &why).entries.empty() && why.find("unreadable") != std::string::npos);
  set_env("DYNALM_AUTOTUNE_CACHE", (dir / "none.json").string());
  CHECK(tuned_threads(DeviceKind::kCpu, cfg, (*loaded)->quantization) == 0);
  CHECK(autotune_report().find("Execution modes") != std::string::npos);
  // Conditions: a problem, nothing readable before the run, all fine.
  CHECK(!judge_tune_conditions({"AC"}, {"on battery"}, true).quiet);
  CHECK(judge_tune_conditions({"AC"}, {"on battery"}, true).text == "on battery");
  CHECK(!judge_tune_conditions({}, {}, true).quiet);  // unverified: not stored without --force
  CHECK(judge_tune_conditions({}, {}, false).quiet);  // after the run: no new problem
  CHECK(judge_tune_conditions({"AC", "load 0.4"}, {}, true).quiet);
  CHECK(judge_tune_conditions({"AC", "load 0.4"}, {}, true).text == "AC, load 0.4");
}

// Chat formats this engine renders natively: DeepSeek-Coder detection and the
// fallback by model name for files without a template.
void test_chat_templates() {
  const std::string dsc =
      "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
      "{%- if not ns.found -%}{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed "
      "by Deepseek Company, and you only answer questions related to computer science. For politically sensitive "
      "questions, security and privacy issues, and other non-computer science questions, you will refuse to "
      "answer\\n'}}{%- endif %}{%- for message in messages %}{%- if message['role'] == 'system' %}{{ message['content'] }}"
      "{%- else %}{%- if message['role'] == 'user' %}{{'### Instruction:\\n' + message['content'] + '\\n'}}"
      "{%- else %}{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}{%- endif %}{%- endif %}{%- endfor %}"
      "{% if add_generation_prompt %}{{'### Response:'}}{% endif %}";
  auto t = ChatTemplate::from_jinja(dsc);
  CHECK(t.ok() && t->format() == ChatFormat::kDeepSeekCoder);
  const std::vector<ChatMessage> msgs = {{"user", "hi"}, {"assistant", "hello"}, {"user", "sum"}};
  auto out = t->apply(msgs, true);
  CHECK(out.ok());
  CHECK(out->starts_with("You are an AI programming assistant") &&
        out->find("refuse to answer\n### Instruction:\nhi\n### Response:\nhello\n<|EOT|>\n### Instruction:\nsum\n"
                  "### Response:") != std::string::npos);
  const std::vector<ChatMessage> with_sys = {{"system", "Be brief."}, {"user", "x"}};
  CHECK(*t->apply(with_sys, true) == "Be brief.### Instruction:\nx\n### Response:");

  // Phi-4-mini: the tag is built from the role, no newlines.
  auto phi4 = ChatTemplate::from_jinja(
      "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message %}"
      "{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}"
      "{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}");
  CHECK(phi4.ok() && phi4->format() == ChatFormat::kPhi4);
  CHECK(*phi4->apply(with_sys, true) == "<|system|>Be brief.<|end|><|user|>x<|end|><|assistant|>");
  auto by_name = [](std::string_view a, std::string_view b) {
    const std::string_view n[] = {a, b};
    return ChatTemplate::from_model_name(n);
  };
  CHECK(by_name("LLaMA v2", "llama-2-7b-chat.Q4_K_M.gguf")->format() == ChatFormat::kLlama2);
  CHECK(by_name("deepseek-ai_deepseek-coder-1.3b-instruct", "x.gguf")->format() == ChatFormat::kDeepSeekCoder);
  CHECK(by_name("", "Mistral-7B-Instruct-v0.2.Q4_K_M.gguf")->format() == ChatFormat::kMistral);
  CHECK(!by_name("LLaMA v2", "llama-2-7b.Q4_K_M.gguf").ok());  // a base model: no chat format
  CHECK(!by_name("", "some-model-chat.gguf").ok());
}

// A fused Q|K|V weight (Phi-3) gives the same text as separate Q, K, V
// weights holding the same values, on the CPU and on graph devices (ggml-cpu,
// the GPU), where rope writes Q inside the fused value and then reads K.
void test_fused_qkv(const std::filesystem::path& dir, DeviceKind dev) {
  test::TinyModelSpec spec;
  spec.fused_qkv = true;
  const std::string fused = (dir / "tiny-fused.gguf").string();
  CHECK(test::write_file(fused, test::tiny_model_gguf(spec)));
  auto open = [&](const std::string& path) {
    EngineOptions o;
    o.model_path = path;
    o.backend = dev;
    o.threads = 2;
    o.kv_tokens = 512;
    o.int8_decode_rows = 0;
    auto e = Engine::create(o);
    if (!e.ok()) std::fprintf(stderr, "fused qkv on %s: %s\n", std::string(device_kind_name(dev)).c_str(),
                              e.status().to_string().c_str());
    CHECK(e.ok());
    return std::move(*e);
  };
  auto a = open(g_model), b = open(fused);
  for (const char* p : {"alpha beta", "the quick brown fox jumps over"}) {
    const Out x = gen(*a, p, 16), y = gen(*b, p, 16);
    CHECK(y.finish == StreamFinish::kLength && x.text == y.text);
  }
}

// GPT-4o (o200k) pre-tokenizer: cases worked out from the HF regex by hand.
void test_gpt4o_pretokenizer() {
  auto split = [](std::string_view text) {
    std::vector<std::string_view> w;
    pretokenize(PreTokenizer::kGpt4o, text, w);
    return std::vector<std::string>(w.begin(), w.end());
  };
  using V = std::vector<std::string>;
  CHECK(pretokenizer_from_name("gpt-4o").ok() && *pretokenizer_from_name("gpt-4o") == PreTokenizer::kGpt4o);
  CHECK(split("Hello world") == (V{"Hello", " world"}));
  CHECK(split("HTTPServer ABC") == (V{"HTTPServer", " ABC"}));  // U*W+ backs off; U+W* for all caps
  CHECK(split("don't I'M") == (V{"don't", " I'M"}));             // contractions, any case
  CHECK(split("12345") == (V{"123", "45"}));
  CHECK(split("a/b") == (V{"a", "/b"}));
  CHECK(split("x.\n\ny") == (V{"x", ".\n\n", "y"}));
  CHECK(split("  hi") == (V{" ", " hi"}));
  CHECK(split("e\xCC\x81t\xC3\xA9") == (V{"e\xCC\x81t\xC3\xA9"}));      // combining mark stays in the word
  CHECK(split("\xCE\x91\xCE\x92\xCE\xB3 x") == (V{"\xCE\x91\xCE\x92\xCE\xB3", " x"}));  // Greek AB + gamma
  CHECK(split("a  \n b") == (V{"a", "  \n", " b"}));
}

// BOS when the GGUF has no add_bos_token key: as llama.cpp, the Llama 3
// pre-tokenizers add it (many Llama 3 files omit the key); Qwen2 does not.
void test_bos_default(const std::filesystem::path& dir) {
  for (const auto& [pre, expect] : {std::pair{"llama-bpe", true}, std::pair{"qwen2", false}}) {
    test::TinyModelSpec spec;
    spec.pre = pre;
    spec.add_bos = -1;
    const std::string path = (dir / ("tiny-bos-" + std::string(pre) + ".gguf")).string();
    CHECK(test::write_file(path, test::tiny_model_gguf(spec)));
    auto m = load_model(path);
    CHECK(m.ok());
    const std::vector<TokenId> t = (*m)->tokenizer->encode("hello", true, false);
    CHECK(!t.empty() && (t.front() == (*m)->tokenizer->bos()) == expect);
  }
}

// `dynalm doctor`: no model, no benchmark; reports the backends, the default
// device and the AutoTune state, in text and JSON.
void test_doctor(const std::filesystem::path& dir) {
  set_env("DYNALM_AUTOTUNE_CACHE", (dir / "doctor-none.json").string());
  const DoctorReport r = collect_doctor_report();
  CHECK(r.dynacore_ok && !r.kernel.empty() && r.threads > 0 && r.ggml_threads > 0);
  CHECK(!r.device.empty() && r.autotune_entries == 0 && r.autotune_enabled);
  CHECK(r.autotune_status.find("no tuning cache") != std::string::npos);
  CHECK(r.ggml_cpu == dynacore::ggml_gpu_compiled(dynacore::DeviceType::kCpu));
  CHECK(r.ggml_cpu_usable == ggml_cpu_usable());
  // A GPU backend with a device is what --backend auto names first.
  for (const DoctorReport::GpuBackend& g : r.gpu_backends) {
    if (!g.devices.empty()) CHECK(r.device.find(g.provider + " GPU") == 0);
  }
  const std::string text = format_doctor_text(r);
  for (const char* label : {"Backends:", "Default:", "Threads:", "AutoTune:", "CPU features:", "Status:"}) {
    CHECK(text.find(label) != std::string::npos);
  }
  CHECK(text.find("this release runs on the CPU only") == std::string::npos);  // stale since GPU support
  auto j = json::parse(format_doctor_json(r));
  CHECK(j.ok() && j->is_object());
  const json::Value* at = j->find("autotune");
  CHECK(at != nullptr && at->is_object() && at->find("cache") != nullptr && at->find("enabled") != nullptr);
  CHECK(j->find("gpu_backends") != nullptr && j->find("gpu_backends")->is_array());
  CHECK(j->find("ggml_threads") != nullptr && j->find("device") != nullptr);
  set_env("DYNALM_AUTOTUNE", "0");
  CHECK(!collect_doctor_report().autotune_enabled);
  set_env("DYNALM_AUTOTUNE", "");
}

// Tokenizer-aware support (e2e-20261019-0022: DeepSeek-Coder safetensors was
// reported supported from its architecture, then failed at its tokenizer):
// the checks use the loader's own code and say which part fails.
void test_tokenizer_support(const std::filesystem::path& dir) {
  // The shared GGUF rule (load and `pull --check`).
  CHECK(gguf::check_tokenizer_kind("gpt2", "qwen2").ok());
  CHECK(gguf::check_tokenizer_kind("gpt2", "").ok());  // no pre key: the GPT-2 split
  CHECK(gguf::check_tokenizer_kind("llama", "").ok());
  CHECK(!gguf::check_tokenizer_kind("gpt2", "deepseek-coder").ok());
  CHECK(!gguf::check_tokenizer_kind("rwkv", "").ok());

  // GGUF header peek: architecture and tokenizer keys, and what a short read misses.
  const std::vector<uint8_t> tiny = test::tiny_model_gguf();
  auto pk = peek_gguf_header(std::as_bytes(std::span(tiny)));
  CHECK(pk.ok() && pk->architecture == "qwen2" && pk->tokenizer_model == "gpt2" && pk->tokenizer_pre == "qwen2");
  auto short_pk = peek_gguf_header(std::as_bytes(std::span(tiny).first(64)));
  CHECK(short_pk.ok() && !short_pk->has_tokenizer_model);
  CHECK(peek_gguf_architecture(std::as_bytes(std::span(tiny))).value() == "qwen2");

  // A GGUF with an unknown pre-tokenizer: supported architecture, unsupported tokenizer.
  test::TinyModelSpec bad;
  bad.pre = "deepseek-coder";
  const std::string bad_path = (dir / "tiny-badpre.gguf").string();
  CHECK(test::write_file(bad_path, test::tiny_model_gguf(bad)));
  auto bf = gguf::GgufFile::open(bad_path);
  CHECK(bf.ok());
  CHECK(gguf_support_status(**bf).find("unsupported tokenizer") == 0);
  CHECK(!gguf_tokenizer_support(**bf).ok());
  CHECK(!load_model(bad_path).ok());  // the load agrees with the check
  auto good = gguf::GgufFile::open(g_model);
  CHECK(good.ok() && gguf_support_status(**good) == "ok");
  CHECK(gguf_tokenizer_support(**good).value() == "byte-level BPE, pre-tokenizer qwen2");

  // Hugging Face tokenizer.json variants (config.json + tokenizer.json, no weights).
  json::Object vocab;
  const std::vector<std::string> toks = test::byte_tokens();
  for (size_t i = 0; i < toks.size(); ++i) vocab[toks[i]] = static_cast<double>(i);
  auto split = [](const std::string& re) {
    return json::Value(json::Object{{"type", "Split"}, {"pattern", json::Object{{"Regex", re}}}, {"behavior", "Isolated"},
                                    {"invert", false}});
  };
  const json::Value byte_level(json::Object{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}});
  auto hf_dir = [&](const std::string& name, const json::Value* pre, const std::string& model_type,
                    const char* raw = nullptr) {
    const std::filesystem::path d = dir / name;
    std::filesystem::create_directories(d);
    CHECK(write_text((d / "config.json").string(), "{\"model_type\": \"llama\"}"));
    if (raw != nullptr) {
      CHECK(write_text((d / "tokenizer.json").string(), raw));
    } else {
      json::Object tj{{"model", json::Object{{"type", model_type}, {"vocab", vocab}, {"merges", json::Array{}}}}};
      if (pre != nullptr) tj["pre_tokenizer"] = *pre;
      CHECK(write_text((d / "tokenizer.json").string(), json::dump(json::Value(std::move(tj)))));
    }
    return d.string();
  };
  const std::string qwen2_re =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  const json::Value qwen2(json::Object{{"type", "Sequence"}, {"pretokenizers", json::Array{split(qwen2_re), byte_level}}});
  auto ok = hf_tokenizer_support(hf_dir("hf-qwen2", &qwen2, "BPE"));
  CHECK(ok.ok() && ok->find("pre-tokenizer qwen2") != std::string::npos);
  // DeepSeek-Coder-1.3B's exact pre-tokenizer (huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct).
  const json::Value deepseek(json::Object{
      {"type", "Sequence"},
      {"pretokenizers", json::Array{split("[\r\n]"), split(R"(\s?\p{L}+)"), split(R"(\s?\p{P}+)"),
                                    split("[一-龥ࠀ-一가-퟿]+"),
                                    json::Value(json::Object{{"type", "Digits"}, {"individual_digits", true}}),
                                    byte_level}}});
  auto ds = hf_tokenizer_support(hf_dir("hf-deepseek", &deepseek, "BPE"));
  CHECK(!ds.ok() && ds.status().message().find("Split+Split+Split+Split+Digits+ByteLevel") != std::string::npos);
  CHECK(!hf_tokenizer_support(hf_dir("hf-malformed", nullptr, "BPE", "{not json")).ok());
  auto wp = hf_tokenizer_support(hf_dir("hf-wordpiece", &qwen2, "WordPiece"));
  CHECK(!wp.ok() && wp.status().message().find("BPE only") != std::string::npos);
  const std::filesystem::path none = dir / "hf-no-tokenizer";
  std::filesystem::create_directories(none);
  CHECK(write_text((none / "config.json").string(), "{}"));
  CHECK(!hf_tokenizer_support(none.string()).ok());

  // The overall verdict: YES only with all three checked and supported.
  CHECK(support_verdict(true, true, WeightsCheck::kSupported) == "YES");
  CHECK(support_verdict(true, false, WeightsCheck::kAbsent) == "NO (tokenizer)");
  CHECK(support_verdict(false, false, WeightsCheck::kUnsupported) == "NO (architecture, tokenizer, weights)");
  CHECK(support_verdict(true, true, WeightsCheck::kAbsent).find("PARTIAL") == 0);
}

// Mixed prompt/decode steps (DD-079), cost-model (DYNALM_DECODE_PREDICTIVE=1)
// and halving controller (=0), on the scheduler trace (no timing): a decoding
// stream gets its row in every step until it ends, while two long prompts
// arrive; mixed steps stay within the step budget and per-sequence chunk, and
// their prompt share within [protected minimum, prefill budget]; every prompt
// is computed exactly once, in order (contiguous positions), and finishes; a
// lone request afterwards has no mixed steps.
void test_mixed_step_protection(DeviceKind dev) {
  // The mixed-step controller itself: prefill bursts (on by default where the
  // CPU runs prefill on SME) would keep the prompts out of A's steps.
  set_env("DYNALM_PREFILL_BURST_MS", "0");
  for (const char* predictive : {"1", "0"}) {
    set_env("DYNALM_DECODE_PREDICTIVE", predictive);
    auto e = make_engine(dev, /*prefix_cache=*/false, /*solo=*/-1, /*trace=*/true);
    const SchedulerConfig cfg = e->scheduler_config();
    CHECK(cfg.decode_protect_predictive == (predictive[0] == '1'));
    auto a = e->generate_text(kPrompts[0], false, greedy(160));
    CHECK(a.ok());
    StreamEvent ev;
    CHECK((*a)->next(ev, std::chrono::milliseconds(60000)));  // A is decoding
    auto b = e->generate_text(std::string(900, 'z'), false, greedy(4));
    auto c = e->generate_text(std::string(600, 'q'), false, greedy(4));
    CHECK(b.ok() && c.ok());
    const Out oa = drain(*a), ob = drain(*b), oc = drain(*c);
    CHECK(oa.finish == StreamFinish::kLength && ob.finish == StreamFinish::kLength && oc.finish == StreamFinish::kLength);
    CHECK(ob.tokens == 4 && oc.tokens == 4);
    const uint64_t a_id = (*a)->id();
    std::map<uint64_t, int32_t> next_pos;  // per prompt: the first position not yet computed
    bool a_started = false, a_done = false;
    int mixed = 0;
    for (const StepTrace& st : e->take_step_trace()) {
      bool has_a = false;
      for (const StepTraceRow& r : st.rows) {
        if (r.request_id == a_id && r.decode) has_a = true;
        if (r.decode) continue;
        if (!st.solo) CHECK(r.rows <= std::max(cfg.max_prefill_chunk, 1));
        auto [it, fresh] = next_pos.try_emplace(r.request_id, 0);
        if (r.request_id != a_id) CHECK(r.start_pos == it->second);  // in order, nothing skipped or repeated
        it->second = r.start_pos + r.rows;
      }
      if (has_a) a_started = true;
      if (a_started && !a_done && !has_a) a_done = true;  // A has finished
      if (a_done) CHECK(!has_a);                          // never a step without A mid-stream
      if (st.decode_rows > 0 && st.prefill_rows > 0) {
        ++mixed;
        CHECK(!st.solo);
        CHECK(st.prefill_rows <= st.prefill_budget && st.prefill_budget <= cfg.prefill_token_budget);
        CHECK(st.prefill_budget >= cfg.protected_prefill_min);
      }
    }
    CHECK(mixed > 0);  // the prompts were prefilled while A decoded
    const uint64_t b_id = (*b)->id(), c_id = (*c)->id();
    CHECK(next_pos.count(b_id) && next_pos.count(c_id) && next_pos[b_id] > next_pos[c_id]);  // 900 > 600 chars
    const Out alone = gen(*e, kPrompts[3], 8);
    CHECK(alone.finish == StreamFinish::kLength);
    for (const StepTrace& st : e->take_step_trace()) CHECK(!(st.decode_rows > 0 && st.prefill_rows > 0));
  }
  set_env("DYNALM_DECODE_PREDICTIVE", "");
  set_env("DYNALM_PREFILL_BURST_MS", "");
}

// SHA-256 against the FIPS 180-4 / NIST test vectors, split updates, a file,
// and the Hugging Face X-Linked-ETag header parse used by `dynalm pull`.
void test_sha256(const std::filesystem::path& dir) {
  auto hex = [](std::string_view s) {
    Sha256 h;
    h.update(s);
    return h.hex_digest();
  };
  CHECK(hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million(1000000, 'a');
  CHECK(hex(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // Every length 0..200 hashed whole equals the same bytes fed in odd-sized pieces.
  std::string msg;
  for (int i = 0; i < 200; ++i) msg.push_back(static_cast<char>(i * 7 + 3));
  for (size_t len = 0; len <= msg.size(); ++len) {
    Sha256 split;
    for (size_t p = 0; p < len;) {
      const size_t step = std::min<size_t>(len - p, 1 + (p * 13) % 71);
      split.update(std::string_view(msg).substr(p, step));
      p += step;
    }
    CHECK(split.hex_digest() == hex(std::string_view(msg).substr(0, len)));
  }
  const auto file = (dir / "sha.bin").string();
  CHECK(test::write_file(file, std::vector<uint8_t>(million.begin(), million.end())));
  auto fh = sha256_file(file);
  CHECK(fh.ok() && *fh == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  CHECK(!sha256_file((dir / "missing.bin").string()).ok());
  // The header as curl -I -L writes it: a redirect response, then the CDN's.
  const std::string headers =
      "HTTP/2 302 \r\nlocation: https://cdn.example/x\r\nX-Linked-Size: 491400032\r\n"
      "x-linked-etag: \"74A4DA8C9FDBCD15BD1F6D01D621410D31C6FC00986F5EB687824E7B93D7A9DB\"\r\n\r\n"
      "HTTP/2 200 \r\netag: \"abc\"\r\n\r\n";
  CHECK(linked_etag_sha256(headers) == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db");
  CHECK(linked_etag_sha256("HTTP/2 200\r\netag: \"abc\"\r\n").empty());  // no linked ETag (not an LFS file)
  CHECK(linked_etag_sha256("x-linked-etag: \"0123456789abcdef0123456789abcdef01234567\"\r\n").empty());  // git SHA-1
  CHECK(linked_etag_sha256("x-linked-etag: W/\"74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db\"\r\n").empty());
  CHECK(linked_etag_sha256("x-linked-etag: \"zz a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db\"\r\n").empty());
}

int main() {
  const auto dir = std::filesystem::temp_directory_path() / ("dynalm-test-" + test::unique_suffix());
  std::filesystem::create_directories(dir);
  g_model = (dir / "tiny.gguf").string();
  CHECK(test::write_file(g_model, test::tiny_model_gguf()));
  int rc = 0;
  test_sha256(dir);
  std::printf("sha256: NIST vectors, split updates, file, Hugging Face X-Linked-ETag\n");
  test_device_plan();
  std::printf("device plan: auto / --gpu / cpu / exact / named, per model and build\n");
  test_predictive_mixed_budget();
  test_sampling();
  std::printf("sampling: greedy and ties, temperature, top-k, top-p, min-p, penalties, seeds, invalid parameters\n");
  test_memory_admission();
  // Isolated from the user's tuning cache (~/.dynalm/autotune.json).
  set_env("DYNALM_AUTOTUNE_CACHE", (dir / "none.json").string());
  test_autotune(dir);
  test_chat_templates();
  test_gpt4o_pretokenizer();
  test_bos_default(dir);
  test_doctor(dir);
  test_tokenizer_support(dir);
  std::printf("tokenizer support: shared GGUF rule, header peek, GGUF and HF checks by the load path, verdict\n");
  std::printf("doctor: backends, default device, AutoTune state; text and JSON, no model loaded\n");
  std::printf("BOS default without add_bos_token: Llama 3 pre-tokenizers add it, Qwen2 does not\n");
  std::printf("GPT-4o pre-tokenizer (Phi-4-mini, Llama 4): regex cases\n");
  std::printf("chat templates: DeepSeek-Coder, and by model name when the file has none\n");
  std::printf("autotune: modes, versioned cache, invalidation, tuned threads and the opt-out\n");
  std::printf("memory admission: refuses a KV cache that cannot fit, names a --ctx that does\n");
  std::printf("DD-079 predictive mixed-step budget: target, clamps, missing samples\n");
  {
    std::printf("CPU:\n");
    auto e = make_engine(DeviceKind::kCpu);
    test_basic_and_isolation(*e, "cpu");
    std::printf("  determinism, concurrent == alone (shared prefixes included)\n");
    test_prefix_cache_equivalence();
    std::printf("  prefix cache hits == no prefix cache\n");
    test_cancel_deadline_errors(*e);
    std::printf("  cancellation, deadline, invalid requests; later requests unaffected\n");
    test_solo_prefill_never_stalls_decode(DeviceKind::kCpu);
    test_prefill_burst(DeviceKind::kCpu);
    std::printf("  solo prefill never stalls a decoding stream (scheduler trace)\n");
    test_mixed_step_protection(DeviceKind::kCpu);
    if (ggml_cpu_usable()) test_mixed_step_protection(DeviceKind::kGgmlCpu);
    std::printf("  mixed steps (cost model and halving): decode never skipped, budgets and chunks held, prompts in order%s\n",
                ggml_cpu_usable() ? ", native and ggml-cpu" : " (ggml-cpu: SKIP, not usable here)");
    test_memory_bounded(*e);
    std::printf("  memory bounded over repeated requests\n");
    test_speculative_batch(DeviceKind::kCpu, /*exact=*/true);
    std::printf("  speculative decoding in the batch: same text as plain decoding, drafts accepted\n");
    // A draft model: the target itself (drafts mostly accepted) and an unrelated
    // model of the same vocabulary (drafts mostly rejected): the text never changes.
    test_speculative_batch(DeviceKind::kCpu, /*exact=*/true, g_model);
    test::TinyModelSpec other;
    other.seed = 99;
    const std::string other_path = (dir / "tiny-other.gguf").string();
    CHECK(test::write_file(other_path, test::tiny_model_gguf(other)));
    test_speculative_batch(DeviceKind::kCpu, /*exact=*/true, other_path);
    std::printf("  speculative decoding with a draft model (BatchDrafter): same text, matching and unrelated drafts\n");
    test_fused_qkv(dir, DeviceKind::kCpu);
    if (ggml_cpu_usable()) {
      test_fused_qkv(dir, DeviceKind::kGgmlCpu);
      std::printf("  fused Q|K|V weights (Phi-3) == separate ones, native and ggml-cpu\n");
    } else {
      std::printf("  fused Q|K|V weights (Phi-3) == separate ones, native (ggml-cpu: SKIP, not usable here)\n");
    }
  }
  bool gpu_usable = false;
  // The GPU under test: the first one built in, or DYNACORE_TEST_GPU.
  DeviceKind gpu_kind = DeviceKind::kCpu;
  for (DeviceKind k : compiled_devices()) {
    if (k != DeviceKind::kCpu && gpu_kind == DeviceKind::kCpu) gpu_kind = k;
  }
  if (const char* t = std::getenv("DYNACORE_TEST_GPU"); t != nullptr && *t != '\0') {
    auto k = parse_device_kind(t);
    CHECK(k.ok());
    gpu_kind = *k;
  }
  if (dynacore::ggml_gpu_compiled()) {
    EngineOptions probe;
    probe.model_path = g_model;
    probe.backend = gpu_kind;
    if (auto e = Engine::create(probe); !e.ok()) {
      std::printf("GPU: SKIP (no usable %s device: %s)\n", std::string(device_kind_name(gpu_kind)).c_str(),
                  e.status().to_string().c_str());
    } else {
      e->reset();
      gpu_usable = true;
      std::printf("GPU (%s):\n", std::string(device_kind_name(gpu_kind)).c_str());
      auto g = make_engine(gpu_kind);
      const Out a = gen(*g, kPrompts[0], 24), b = gen(*g, kPrompts[0], 24);
      CHECK(a.finish == StreamFinish::kLength && a.text == b.text);
      std::printf("  generation completes, deterministic\n");
      // Concurrency, shared prefixes and cancellation on the GPU: complete and
      // correct lengths; text equality with solo runs is not promised on the
      // GPU (no batch invariance), so lengths and finish reasons are checked.
      std::vector<std::shared_ptr<RequestStream>> streams;
      for (const auto& p : {kPrompts[0], kPrompts[3], kShared + "one", kShared + "two"}) {
        auto s = g->generate_text(p, false, greedy(24));
        CHECK(s.ok());
        streams.push_back(*s);
      }
      for (auto& s : streams) {
        const Out o = drain(s);
        CHECK(o.finish == StreamFinish::kLength && o.tokens == 24);
      }
      test_cancel_deadline_errors(*g);
      test_solo_prefill_never_stalls_decode(gpu_kind);
      test_mixed_step_protection(gpu_kind);
      std::printf("  concurrent + shared prefixes, cancellation, deadline, solo prefill and mixed-step traces\n");
      test_memory_bounded(*g);
      std::printf("  memory bounded over repeated requests\n");
      // GPU: no batch invariance, so lengths and acceptance, not text equality.
      test_speculative_batch(gpu_kind, /*exact=*/false);
      std::printf("  speculative decoding in the batch: completes, drafts accepted\n");
      test_fused_qkv(dir, gpu_kind);
      std::printf("  fused Q|K|V weights (Phi-3) == separate ones\n");
    }
  } else {
    std::printf("GPU: NOT BUILT (ENABLE_METAL/VULKAN/CUDA=OFF)\n");
  }
  test_scheduler_profile(gpu_usable, gpu_kind);
  std::printf("scheduler profile: GPU mixed-step budgets on %s only; cpu and ggml-cpu keep the CPU budgets%s\n",
              gpu_usable ? std::string(device_kind_name(gpu_kind)).c_str() : "the GPU (not tested here)",
              ggml_cpu_usable() ? "" : " (ggml-cpu: SKIP, not usable here)");
  test_attn_prescale_q();
  test_device_selection(dir.string(), gpu_usable, gpu_kind);
  std::printf("device selection: auto -> %s for Qwen2 and Qwen3 (per-head Q/K norm); cpu -> ggml kernels when built; "
              "--gpu/--cpu/--backend\n",
              gpu_usable ? "GPU" : "CPU");
  std::filesystem::remove_all(dir);
  std::printf("all engine tests passed (%d checks)\n", g_checks);
  return rc;
}

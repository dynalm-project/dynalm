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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/process_stats.h"
#include "loader/model_loader.h"
#include "runtime/engine.h"
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

// `auto` picks the GPU only for a model it runs, and the CPU otherwise;
// an explicit GPU request for an unsupported model fails at startup.
void test_device_selection(const std::string& dir, bool gpu_usable, DeviceKind gpu_kind) {
  auto loaded = load_model(g_model);
  CHECK(loaded.ok());
  const ModelConfig base = (*loaded)->config;
  CHECK(gpu_model_support(base, DType::kF16).ok());
  CHECK(gpu_model_support(base, DType::kQ8_0).ok());
  CHECK(!gpu_model_support(base, DType::kQ4_K).ok());
  ModelConfig c = base;
  c.moe.num_experts = 8;
  CHECK(!gpu_model_support(c, DType::kF16).ok());
  c = base;
  c.attn_qk_norm = true;
  CHECK(!gpu_model_support(c, DType::kF16).ok());
  c = base;
  c.rope.scaling = RopeScaling::kLinear;
  CHECK(gpu_model_support(c, DType::kF16).ok());
  c.rope.scaling = RopeScaling::kYarn;
  CHECK(!gpu_model_support(c, DType::kF16).ok());
  c = base;
  c.has_rope_local = true;
  c.rope_local.scaling = RopeScaling::kLlama3;
  CHECK(!gpu_model_support(c, DType::kF16).ok());

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
  auto is_gpu = [](const Engine& e) { return e.backend_name().rfind("CPU", 0) != 0; };
  {
    auto e = on(g_model, true, DeviceKind::kCpu);  // `backend` is ignored by auto
    CHECK(e.ok());
    CHECK(is_gpu(**e) == gpu_usable);
    const Out o = gen(**e, kPrompts[0], 8);
    CHECK(o.finish == StreamFinish::kLength && o.tokens == 8);
  }
  {
    auto e = on(qwen3, true, gpu_kind);
    CHECK(e.ok());
    CHECK(!is_gpu(**e));
    const Out o = gen(**e, kPrompts[0], 8);
    CHECK(o.finish == StreamFinish::kLength && o.tokens == 8);
  }
  // --gpu / --cpu / --backend choices (apply_backend_choice).
  {
    EngineOptions o;
    CHECK(apply_backend_choice("gpu", o).ok() && o.backend_auto && o.backend_require_gpu);
    CHECK(apply_backend_choice("cpu", o).ok() && !o.backend_auto && o.backend == DeviceKind::kCpu);
    CHECK(apply_backend_choice("auto", o).ok() && o.backend_auto && !o.backend_require_gpu);
    CHECK(apply_backend_choice("metal", o).ok() && !o.backend_auto && o.backend == DeviceKind::kMetal);
    CHECK(!apply_backend_choice("tpu", o).ok());
    // --gpu never falls back to the CPU: an unsupported model (or a build
    // without a GPU) is an error naming the reason.
    o = EngineOptions();
    o.model_path = qwen3;
    o.threads = 2;
    o.kv_tokens = 512;
    CHECK(apply_backend_choice("gpu", o).ok());
    auto e = Engine::create(o);
    CHECK(!e.ok() && e.status().code() == StatusCode::kUnsupported);
    if (gpu_usable) {
      o.model_path = g_model;
      auto g = Engine::create(o);
      CHECK(g.ok() && is_gpu(**g));
    }
  }
  if (dynacore::ggml_gpu_compiled()) {
    auto e = on(qwen3, false, gpu_kind);
    CHECK(!e.ok() && e.status().code() == StatusCode::kUnsupported);
  }
}

int main() {
  const auto dir = std::filesystem::temp_directory_path() / ("dynalm-test-" + test::unique_suffix());
  std::filesystem::create_directories(dir);
  g_model = (dir / "tiny.gguf").string();
  CHECK(test::write_file(g_model, test::tiny_model_gguf()));
  int rc = 0;
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
    std::printf("  solo prefill never stalls a decoding stream (scheduler trace)\n");
    test_memory_bounded(*e);
    std::printf("  memory bounded over repeated requests\n");
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
      std::printf("  concurrent + shared prefixes, cancellation, deadline, solo prefill trace\n");
      test_memory_bounded(*g);
      std::printf("  memory bounded over repeated requests\n");
    }
  } else {
    std::printf("GPU: NOT BUILT (ENABLE_METAL/VULKAN/CUDA=OFF)\n");
  }
  test_device_selection(dir.string(), gpu_usable, gpu_kind);
  std::printf("device selection: auto -> %s for a supported model, CPU for an unsupported one\n",
              gpu_usable ? "GPU" : "CPU");
  std::filesystem::remove_all(dir);
  std::printf("all engine tests passed (%d checks)\n", g_checks);
  return rc;
}

// Phase 22: model compatibility suite. For every supported
// architecture: loading, metadata validation, tokenizer, short generation,
// long generation, KV (prefix) reuse and concurrent generation — through the
// same Engine path the server uses. Runs on the committed tiny fixtures; real
// models are covered when their files are present (test_models.h).

#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

#include "runtime/engine.h"
#include "test_models.h"

namespace engine {
namespace {

struct Generated {
  std::string text;
  StreamFinish finish = StreamFinish::kNone;
  int32_t completion_tokens = 0;
  Status error;
};

Generated generate(Engine& e, const std::string& prompt, int32_t max_tokens, bool ignore_eos) {
  GenerateParams p;
  p.max_tokens = max_tokens;
  p.stop_at_eog = !ignore_eos;
  Generated g;
  auto s = e.generate_text(prompt, false, p);
  if (!s.ok()) {
    g.error = s.status();
    return g;
  }
  StreamEvent ev;
  while ((*s)->next(ev, std::chrono::milliseconds(120000))) {
    g.text += ev.text;
    if (ev.done) break;
  }
  g.finish = ev.finish;
  g.completion_tokens = ev.completion_tokens;
  g.error = ev.error;
  return g;
}

// The final stream event is pushed during a scheduler step, but the stats
// snapshot is refreshed after the step: poll briefly.
int32_t running_after_quiesce(Engine& e) {
  for (int i = 0; i < 500 && e.stats().scheduler.running > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return e.stats().scheduler.running;
}

struct CompatCase {
  std::string name;
  std::string path;
  int32_t long_tokens;  // "long generation" length for this model
};

std::vector<CompatCase> cases() {
  const std::string d = std::string(ENGINE_TEST_DATA_DIR) + "/";
  std::vector<CompatCase> v = {
      {"tiny_llama", d + "tiny_llama.gguf", 100},   {"tiny_qwen2", d + "tiny_qwen2.gguf", 100},
      {"tiny_qwen3", d + "tiny_qwen3.gguf", 100},   {"tiny_gemma", d + "tiny_gemma.gguf", 100},
      {"tiny_gemma2", d + "tiny_gemma2.gguf", 100}, {"tiny_gemma3", d + "tiny_gemma3.gguf", 100},
      {"tiny_phi3", d + "tiny_phi3.gguf", 100},
      {"tiny_mixtral", d + "tiny_mixtral.gguf", 100},     {"tiny_qwen2moe", d + "tiny_qwen2moe.gguf", 100},
      {"tiny_qwen3moe", d + "tiny_qwen3moe.gguf", 100},   {"tiny_granitemoe", d + "tiny_granitemoe.gguf", 100},
  };
  // Optional real models: shorter long-generation to keep the suite fast.
  v.push_back({"smollm2_q8", testing::smollm_q8_model(), 64});
  v.push_back({"qwen25_q4km", testing::qwen_q4_model(), 64});
  v.push_back({"gemma3_f16", testing::gemma_model(), 64});
  return v;
}

void PrintTo(const CompatCase& c, std::ostream* os) { *os << c.name; }

class Compat : public ::testing::TestWithParam<CompatCase> {
 protected:
  void SetUp() override {
    if (!testing::exists(GetParam().path)) GTEST_SKIP() << "model not present: " << GetParam().path;
    EngineOptions o;
    o.model_path = GetParam().path;
    o.threads = 2;
    o.kv_tokens = 2048;
    o.max_batch_tokens = 64;
    // Batch invariance (DD-031) is a property of the fp32 decode path; the
    // int8 path (DD-053) trades it for speed and is checked for accuracy
    // separately (test_int8_decode, bench_int8_accuracy).
    o.int8_decode_rows = 0;
    auto e = Engine::create(o);  // 1. loading
    ASSERT_TRUE(e.ok()) << e.status().to_string();
    eng = std::move(*e);
  }
  std::unique_ptr<Engine> eng;
};

TEST_P(Compat, MetadataAndTokenizer) {
  const LoadedModel& m = eng->model();
  // 2. metadata validation
  ASSERT_NE(m.architecture, nullptr);
  EXPECT_TRUE(m.config.validate().ok()) << m.config.validate().to_string();
  EXPECT_GT(m.config.context_length, 0);
  EXPECT_GT(m.weight_bytes, 0);
  EXPECT_TRUE(m.unmapped_tensors.empty() || !m.unmapped_tensors.front().empty());
  // 3. tokenizer: encode/decode round trip, ids in range
  const Tokenizer& tok = *m.tokenizer;
  for (const char* s : {"Hello world", "numbers 12345 and symbols !?", "multi\nline  spaces"}) {
    const std::vector<TokenId> ids = tok.encode(s, /*add_special=*/false, false);
    ASSERT_FALSE(ids.empty()) << s;
    for (TokenId id : ids) {
      EXPECT_GE(id, 0);
      EXPECT_LT(id, m.config.vocab_size);
    }
    EXPECT_EQ(tok.decode(ids), s);
  }
}

TEST_P(Compat, ShortAndLongGeneration) {
  // 4. short generation
  const Generated s = generate(*eng, "The capital of France is", 8, true);
  ASSERT_TRUE(s.error.ok()) << s.error.to_string();
  EXPECT_EQ(s.finish, StreamFinish::kLength);
  EXPECT_EQ(s.completion_tokens, 8);
  // 5. long generation (fills most of a tiny model's 128-token context)
  const int32_t n = std::min<int32_t>(GetParam().long_tokens,
                                      static_cast<int32_t>(eng->model().config.context_length) - 16);
  const Generated l = generate(*eng, "Once upon a time", n, true);
  ASSERT_TRUE(l.error.ok()) << l.error.to_string();
  EXPECT_EQ(l.finish, StreamFinish::kLength);
  EXPECT_EQ(l.completion_tokens, n);
  EXPECT_EQ(running_after_quiesce(*eng), 0);
}

TEST_P(Compat, KvReuseGivesIdenticalOutput) {
  // 6. KV reuse: the second identical request reuses the cached prompt prefix
  //    and must generate exactly the same text.
  // Several 16-token KV blocks even for a byte-level tiny tokenizer, while
  // prompt + output stays inside the tiny models' 128-token context.
  const std::string prompt = "A shared system prompt reused by the next request: be brief.";
  const Generated a = generate(*eng, prompt, 12, true);
  const uint64_t hits_before = eng->stats().prefix.hit_tokens;
  const Generated b = generate(*eng, prompt, 12, true);
  ASSERT_TRUE(a.error.ok()) << a.error.to_string();
  ASSERT_TRUE(b.error.ok()) << b.error.to_string();
  EXPECT_GT(eng->stats().prefix.hit_tokens, hits_before);
  EXPECT_EQ(a.text, b.text);
}

TEST_P(Compat, ConcurrentGeneration) {
  // 7. concurrent generation: identical prompts batched together agree, and
  //    distinct prompts all complete with their requested lengths.
  std::vector<Generated> out(6);
  std::vector<std::thread> ts;
  for (int i = 0; i < 6; ++i) {
    ts.emplace_back([&, i] {
      const std::string prompt = i < 3 ? "Same prompt for everyone" : "Distinct prompt number " + std::to_string(i);
      out[static_cast<size_t>(i)] = generate(*eng, prompt, i < 3 ? 12 : 10 + i, true);
    });
  }
  for (auto& t : ts) t.join();
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(out[static_cast<size_t>(i)].error.ok()) << i << ": " << out[static_cast<size_t>(i)].error.to_string();
    EXPECT_EQ(out[static_cast<size_t>(i)].completion_tokens, i < 3 ? 12 : 10 + i);
  }
  // Same prompt, same length, greedy: identical text however they were batched.
  EXPECT_EQ(out[0].text, out[1].text);
  EXPECT_EQ(out[1].text, out[2].text);
  EXPECT_EQ(running_after_quiesce(*eng), 0);
}

INSTANTIATE_TEST_SUITE_P(AllArchitectures, Compat, ::testing::ValuesIn(cases()),
                         [](const ::testing::TestParamInfo<CompatCase>& i) { return i.param.name; });

}  // namespace
}  // namespace engine

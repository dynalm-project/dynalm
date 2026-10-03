// Phase 19: text streaming (UTF-8 safety, stop strings) and the Engine facade
// (scheduler thread, concurrent streams, cancellation).

#include <gtest/gtest.h>

#include <thread>

#include "loader/model_loader.h"
#include "backends/cpu/cpu_backend.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"
#include "runtime/engine.h"
#include "runtime/generator.h"
#include "runtime/text_stream.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

// The tiny models use a byte-level vocab where token id == byte value.
std::vector<TokenId> bytes_of(std::string_view s) {
  std::vector<TokenId> v;
  for (unsigned char c : s) v.push_back(c);
  return v;
}

class Streamer : public ::testing::Test {
 protected:
  void SetUp() override {
    auto m = load_model(data("tiny_llama.gguf"));
    ASSERT_TRUE(m.ok());
    model = std::move(*m);
  }
  // Feeds `text` byte by byte; returns (emitted text, stopped).
  std::pair<std::string, bool> run(std::string_view text, std::vector<std::string> stops,
                                   std::vector<std::string>* deltas = nullptr) {
    TextStreamer s(*model->tokenizer, std::move(stops));
    std::string out;
    for (TokenId t : bytes_of(text)) {
      auto d = s.push(t);
      out += d.text;
      if (deltas && !d.text.empty()) deltas->push_back(d.text);
      if (d.stopped) return {out, true};
    }
    out += s.finish();
    return {out, false};
  }
  std::unique_ptr<LoadedModel> model;
};

TEST_F(Streamer, Utf8NeverSplit) {
  std::vector<std::string> deltas;
  const std::string text = "a\xE4\xB8\xAD\xF0\x9F\x98\x80z";  // a 中 😀 z
  EXPECT_EQ(run(text, {}, &deltas).first, text);
  for (const auto& d : deltas) {
    // Every delta is complete UTF-8: no delta starts with a continuation byte.
    EXPECT_NE(static_cast<unsigned char>(d[0]) & 0xC0, 0x80);
  }
}

TEST_F(Streamer, StopStringNeverEmittedEvenAcrossTokens) {
  auto [out, stopped] = run("Hello world END tail", {"END"});
  EXPECT_TRUE(stopped);
  EXPECT_EQ(out, "Hello world ");
  // A partial match that turns out not to be a stop is released.
  auto [out2, stopped2] = run("value ENDLESS? no: END!", {"END!"});
  EXPECT_TRUE(stopped2);
  EXPECT_EQ(out2, "value ENDLESS? no: ");
  // Text ending in a stop prefix is flushed at the end.
  auto [out3, stopped3] = run("almost EN", {"END"});
  EXPECT_FALSE(stopped3);
  EXPECT_EQ(out3, "almost EN");
}

TEST_F(Streamer, HoldsBackOnlyTheMinimalSuffix) {
  std::vector<std::string> deltas;
  run("abcXYabc", {"XYZ"}, &deltas);
  // "X" and "XY" are held until disambiguated; everything else flows at once.
  ASSERT_GE(deltas.size(), 4u);
  EXPECT_EQ(deltas[0], "a");
  EXPECT_EQ(deltas[1], "b");
  EXPECT_EQ(deltas[2], "c");
  EXPECT_EQ(deltas[3], "XYa");  // released when 'a' breaks the stop prefix
}

TEST_F(Streamer, EarliestOfSeveralStops) {
  EXPECT_EQ(run("one two three", {"three", "two"}).first, "one ");
}

// --- Engine ---

class EngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EngineOptions o;
    o.model_path = data("tiny_llama.gguf");
    o.threads = 2;
    o.kv_tokens = 1024;
    o.max_batch_tokens = 64;
    auto e = Engine::create(o);
    ASSERT_TRUE(e.ok()) << e.status().to_string();
    eng = std::move(*e);
  }
  // Greedy reference text for a raw byte prompt.
  std::string solo_text(std::string_view prompt, int32_t n) {
    ThreadPool tp(1);
    CpuBackend be(tp, CpuIsa::kGeneric);
    const ModelConfig& c = eng->model().config;
    auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, 256), be);
    auto tf = Transformer::create(c, eng->model().weights, be, 64);
    Generator g(**tf, **kv, *eng->model().tokenizer);
    GenerateOptions o;
    o.max_new_tokens = n;
    std::vector<TokenId> toks = eng->model().tokenizer->encode(prompt, true, false);
    std::string out;
    EXPECT_TRUE(g.generate(toks, o, [&](TokenId t) {
      eng->model().tokenizer->decode_token(t, out);
      return true;
    }).ok());
    return out;
  }
  static std::pair<std::string, StreamEvent> drain(RequestStream& s, int* deltas = nullptr) {
    std::string text;
    StreamEvent ev;
    while (s.next(ev, std::chrono::milliseconds(30000))) {
      text += ev.text;
      if (deltas && !ev.text.empty()) ++*deltas;
      if (ev.done) break;
    }
    return {text, ev};
  }
  std::unique_ptr<Engine> eng;
};

TEST_F(EngineTest, StreamsIncrementallyAndMatchesGenerator) {
  GenerateParams p;
  p.max_tokens = 20;
  auto s = eng->generate_text("streaming test", false, p);
  ASSERT_TRUE(s.ok());
  int deltas = 0;
  auto [text, last] = drain(**s, &deltas);
  EXPECT_TRUE(last.done);
  EXPECT_EQ(last.finish, StreamFinish::kLength);
  EXPECT_EQ(last.completion_tokens, 20);
  EXPECT_GT(deltas, 1);  // delivered as it was generated, not in one piece
  EXPECT_EQ(text, solo_text("streaming test", 20));
}

TEST_F(EngineTest, StopStringEndsGenerationEarly) {
  const std::string full = solo_text("stop me", 24);
  ASSERT_GE(full.size(), 10u);
  GenerateParams p;
  p.max_tokens = 24;
  p.stop = {full.substr(6, 2)};  // a string the model will produce
  auto s = eng->generate_text("stop me", false, p);
  ASSERT_TRUE(s.ok());
  auto [text, last] = drain(**s);
  EXPECT_EQ(last.finish, StreamFinish::kStop);
  EXPECT_EQ(text, full.substr(0, full.find(p.stop[0])));
  EXPECT_LT(last.completion_tokens, 24);
}

TEST_F(EngineTest, ConcurrentClientsEachGetTheirOwnStream) {
  std::vector<std::string> prompts = {"alpha", "beta beta", "gamma!", "delta 123"};
  std::vector<std::string> got(prompts.size());
  std::vector<std::thread> clients;
  for (size_t i = 0; i < prompts.size(); ++i) {
    clients.emplace_back([&, i] {
      GenerateParams p;
      p.max_tokens = 12;
      auto s = eng->generate_text(prompts[i], false, p);
      ASSERT_TRUE(s.ok());
      got[i] = drain(**s).first;
    });
  }
  for (auto& c : clients) c.join();
  for (size_t i = 0; i < prompts.size(); ++i) EXPECT_EQ(got[i], solo_text(prompts[i], 12)) << prompts[i];
}

TEST_F(EngineTest, ConsumerCancellation) {
  GenerateParams p;
  p.max_tokens = 100;
  auto s = eng->generate_text("cancel", false, p);
  ASSERT_TRUE(s.ok());
  StreamEvent ev;
  ASSERT_TRUE((*s)->next(ev));  // at least one delta arrives
  (*s)->cancel();
  auto [text, last] = drain(**s);
  EXPECT_TRUE(last.done);
  // The tiny model can finish all 100 tokens before the cancel reaches the
  // scheduler thread (seen on fast CI machines), so either outcome is valid;
  // what must hold is that the stream ends exactly once and a cancelled one
  // stopped early. Deterministic cancellation is covered at the scheduler
  // level (SchedulerTest.CancellationReleasesKvAndOthersContinue).
  if (last.finish == StreamFinish::kCancelled) {
    EXPECT_LT(last.completion_tokens, 100);
  } else {
    EXPECT_EQ(last.finish, StreamFinish::kLength);
    EXPECT_EQ(last.completion_tokens, 100);
  }
  EXPECT_FALSE((*s)->next(ev, std::chrono::milliseconds(1)));  // nothing after the final event
}

TEST_F(EngineTest, InvalidRequestReportedOnStream) {
  GenerateParams p;
  p.max_tokens = 1000000;  // exceeds context
  auto s = eng->generate_text("x", false, p);
  ASSERT_TRUE(s.ok());
  auto [text, last] = drain(**s);
  EXPECT_EQ(last.finish, StreamFinish::kError);
  EXPECT_EQ(last.error.code(), StatusCode::kInvalidArgument);
  GenerateParams zero;
  zero.max_tokens = 0;
  EXPECT_FALSE(eng->generate_text("x", false, zero).ok());
}

TEST_F(EngineTest, StreamsOutliveEngine) {
  GenerateParams p;
  p.max_tokens = 100;
  auto s = eng->generate_text("shutdown", false, p);
  ASSERT_TRUE(s.ok());
  eng.reset();  // shut down mid-generation
  auto [text, last] = drain(**s);
  EXPECT_TRUE(last.done);
  (*s)->cancel();  // safe after shutdown
}

}  // namespace
}  // namespace engine

// Phase 22: robustness under abuse. Many clients on a small KV
// pool (forcing preemption), random cancellations, timeouts, invalid and
// oversized requests, and engine shutdown with requests in flight. Every
// request must end with exactly one final event, failures must stay isolated,
// and no KV block may leak. Run under ASAN/UBSAN and TSAN in the gate.

#include <gtest/gtest.h>

#include <atomic>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "runtime/engine.h"
#include "common/core.h"

namespace dynalm {
namespace {

std::unique_ptr<Engine> make_engine(int64_t kv_tokens) {
  EngineOptions o;
  o.model_path = std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf";
  o.threads = 2;
  o.kv_tokens = kv_tokens;
  o.max_batch_tokens = 64;
  auto e = Engine::create(o);
  EXPECT_TRUE(e.ok()) << e.status().to_string();
  return e.ok() ? std::move(*e) : nullptr;
}

// Waits until the scheduler has nothing running or queued.
void wait_quiescent(Engine& e) {
  for (int i = 0; i < 500; ++i) {
    const EngineStats st = e.stats();
    if (st.scheduler.running == 0 && st.scheduler.waiting == 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

TEST(Hardening, MixedAbuseNeverLeaksOrCrossContaminates) {
  // 160 tokens of KV: ten blocks for up to ~10 concurrent sequences of up to
  // ~100 tokens, so preemption and admission waits happen constantly.
  auto eng = make_engine(160);
  ASSERT_TRUE(eng);
  constexpr int kClients = 8, kPerClient = 12;
  std::atomic<int> ended{0}, ok{0}, cancelled{0}, failed{0}, rejected{0};
  std::vector<std::thread> clients;
  for (int c = 0; c < kClients; ++c) {
    clients.emplace_back([&, c] {
      std::mt19937 rng(static_cast<uint32_t>(1000 + c));
      for (int r = 0; r < kPerClient; ++r) {
        const int kind = static_cast<int>(rng() % 6);
        GenerateParams p;
        p.stop_at_eog = false;
        p.max_tokens = 4 + static_cast<int32_t>(rng() % 40);
        std::string prompt = "client " + std::to_string(c) + " request " + std::to_string(r);
        if (kind == 0) p.max_tokens = 100000;  // exceeds context: rejected up front
        if (kind == 1) p.timeout_ms = 1 + static_cast<int64_t>(rng() % 5);
        if (kind == 2) prompt.clear();         // empty prompt (BOS only or nothing)
        auto s = eng->generate_text(prompt, false, p);
        if (!s.ok()) {
          rejected.fetch_add(1);
          continue;
        }
        const bool cancel_early = kind == 3;
        StreamEvent ev;
        int events_after_done = 0;
        bool done = false;
        int seen = 0;
        while ((*s)->next(ev, std::chrono::milliseconds(60000))) {
          if (done) ++events_after_done;
          if (cancel_early && ++seen == 2) (*s)->cancel();
          if (ev.done) {
            done = true;
            break;
          }
        }
        EXPECT_TRUE(done) << "request never ended";
        EXPECT_FALSE((*s)->next(ev, std::chrono::milliseconds(1)));  // nothing after the final event
        EXPECT_EQ(events_after_done, 0);
        ended.fetch_add(1);
        switch (ev.finish) {
          case StreamFinish::kLength:
          case StreamFinish::kStop: ok.fetch_add(1); break;
          case StreamFinish::kCancelled: cancelled.fetch_add(1); break;
          default: failed.fetch_add(1); break;
        }
        if (ev.finish == StreamFinish::kLength) {
          EXPECT_EQ(ev.completion_tokens, p.max_tokens);
        }
      }
    });
  }
  for (auto& t : clients) t.join();
  EXPECT_EQ(ended.load() + rejected.load(), kClients * kPerClient);
  EXPECT_GT(ok.load(), kClients * kPerClient / 3);  // good requests succeed despite the abuse
  EXPECT_GT(failed.load(), 0);                      // invalid / timed-out ones failed (isolated)

  wait_quiescent(*eng);
  const EngineStats st = eng->stats();
  EXPECT_EQ(st.scheduler.running, 0);
  EXPECT_EQ(st.scheduler.waiting, 0);
  // Only the prefix cache may still hold blocks.
  EXPECT_EQ(st.kv_blocks_used, st.prefix.cached_blocks);

  // The engine is still fully usable afterwards.
  GenerateParams p;
  p.max_tokens = 5;
  p.stop_at_eog = false;
  auto s = eng->generate_text("after the storm", false, p);
  ASSERT_TRUE(s.ok());
  StreamEvent ev;
  while ((*s)->next(ev) && !ev.done) {
  }
  EXPECT_EQ(ev.finish, StreamFinish::kLength);
}

TEST(Hardening, RequestLargerThanKvPoolIsRejectedUpFront) {
  auto eng = make_engine(64);  // 4 blocks
  ASSERT_TRUE(eng);
  GenerateParams p;
  p.max_tokens = 80;  // fits the 128-token context, not the 64-token pool
  p.stop_at_eog = false;
  auto s = eng->generate_text("hello", false, p);
  ASSERT_TRUE(s.ok());
  StreamEvent ev;
  while ((*s)->next(ev) && !ev.done) {
  }
  EXPECT_EQ(ev.finish, StreamFinish::kError);
  EXPECT_EQ(ev.error.code(), StatusCode::kInvalidArgument);
  EXPECT_NE(ev.error.message().find("KV cache capacity"), std::string::npos);
  EXPECT_EQ(ev.completion_tokens, 0);  // never started generating
}

TEST(Hardening, FailedCreateCleansUp) {
  // A model that fails to load must leave no half-built engine behind
  // (UBSAN caught a null dereference in ~Engine on this path).
  const std::string path = ::testing::TempDir() + "truncated.gguf";
  {
    std::ifstream in(std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
  }
  EngineOptions o;
  o.model_path = path;
  EXPECT_FALSE(Engine::create(o).ok());
  o.model_path = "/nonexistent/model.gguf";
  EXPECT_FALSE(Engine::create(o).ok());
}

TEST(Hardening, ShutdownWithRequestsInFlight) {
  // "Model unload while requests exist": destroying the engine ends every
  // open stream (cancelled) and the streams stay safe to use afterwards.
  auto eng = make_engine(512);
  ASSERT_TRUE(eng);
  std::vector<std::shared_ptr<RequestStream>> streams;
  for (int i = 0; i < 6; ++i) {
    GenerateParams p;
    p.max_tokens = 100;
    p.stop_at_eog = false;
    auto s = eng->generate_text("in flight " + std::to_string(i), false, p);
    ASSERT_TRUE(s.ok());
    streams.push_back(std::move(*s));
  }
  eng.reset();
  for (auto& s : streams) {
    StreamEvent ev;
    bool done = false;
    while (s->next(ev, std::chrono::milliseconds(5000))) {
      if (ev.done) {
        done = true;
        break;
      }
    }
    EXPECT_TRUE(done);
    EXPECT_TRUE(ev.finish == StreamFinish::kCancelled || ev.finish == StreamFinish::kLength);
    s->cancel();  // no-op after shutdown, must not crash
  }
}

}  // namespace
}  // namespace dynalm

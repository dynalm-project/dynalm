// Continuous batching: every request's output must equal running it alone,
// regardless of arrival order, interleaving, cancellation or preemption.

#include "scheduler/scheduler.h"

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <thread>

#include "backends/cpu/cpu_backend.h"
#include "loader/model_loader.h"
#include "runtime/generator.h"

namespace engine {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto m = load_model(data("tiny_llama.gguf"));
    ASSERT_TRUE(m.ok()) << m.status().to_string();
    model = std::move(*m);
    make_pool(128);
    auto t = Transformer::create(model->config, model->weights, be, 64);
    ASSERT_TRUE(t.ok());
    tf = std::move(*t);
  }
  void make_pool(int32_t blocks) {
    const ModelConfig& c = model->config;
    auto p = KvBlockPool::create(KvGeometry{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 4, blocks,
                                            DType::kF32},
                                 be);
    ASSERT_TRUE(p.ok());
    pool = std::move(*p);
  }

  std::vector<TokenId> prompt(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> v(n);
    for (auto& t : v) t = static_cast<TokenId>(rng() % 256);
    return v;
  }

  // Reference: greedy generation of one request alone.
  std::vector<TokenId> solo(const std::vector<TokenId>& p, int32_t max_new) {
    auto solo_pool = KvBlockPool::create(pool->geometry(), be);
    Generator gen(*tf, **solo_pool, *model->tokenizer);
    GenerateOptions o;
    o.max_new_tokens = max_new;
    o.stop_at_eog = false;
    std::vector<TokenId> out;
    EXPECT_TRUE(gen.generate(p, o, [&](TokenId t) { out.push_back(t); return true; }).ok());
    return out;
  }

  struct Collected {
    std::map<uint64_t, std::vector<TokenId>> tokens;
    std::map<uint64_t, RequestEvent> final_event;
  };
  Request make_request(const std::vector<TokenId>& p, int32_t max_new, Collected& out) {
    Request r;
    r.prompt = p;
    r.stop = StopParams{max_new, false};
    r.on_event = [&out](const RequestEvent& ev) {
      if (ev.finished) {
        out.final_event[ev.request_id] = ev;
      } else {
        out.tokens[ev.request_id].push_back(ev.token);
      }
    };
    return r;
  }

  ThreadPool tp{3};
  CpuBackend be{tp, CpuIsa::kGeneric};
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<Transformer> tf;
};

TEST_F(SchedulerTest, StaggeredArrivalsMatchSoloRuns) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  std::vector<std::pair<uint64_t, std::vector<TokenId>>> want;
  // Requests arrive while others are mid-generation (join/leave freely).
  const std::vector<std::pair<size_t, int32_t>> shapes = {{5, 12}, {17, 4}, {3, 9}, {30, 6}, {8, 15}, {1, 3}};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto p = prompt(shapes[i].first, static_cast<unsigned>(i + 1));
    const uint64_t id = sched.submit(make_request(p, shapes[i].second, out));
    want.emplace_back(id, solo(p, shapes[i].second));
    for (int s = 0; s < 3; ++s) sched.step();
  }
  sched.run_until_idle();
  for (const auto& [id, tokens] : want) {
    EXPECT_EQ(out.tokens[id], tokens) << "request " << id;
    EXPECT_EQ(out.final_event[id].status, SequenceStatus::kFinished);
    EXPECT_EQ(out.final_event[id].reason, FinishReason::kLength);
  }
  EXPECT_EQ(sched.stats().completed, shapes.size());
  EXPECT_EQ(pool->free_blocks(), pool->num_blocks());  // no leaked KV
}

TEST_F(SchedulerTest, BatchesManySequencesPerStep) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  for (int i = 0; i < 8; ++i) sched.submit(make_request(prompt(4, 100 + i), 20, out));
  sched.step();  // all 8 prompts (32 tokens) prefilled in one batch
  EXPECT_EQ(sched.stats().last_batch_seqs, 8);
  EXPECT_EQ(sched.stats().last_batch_rows, 32);
  sched.step();  // then 8 decode rows together
  EXPECT_EQ(sched.stats().last_batch_seqs, 8);
  EXPECT_EQ(sched.stats().last_batch_rows, 8);
  sched.run_until_idle();
  EXPECT_EQ(sched.stats().completed, 8u);
}

TEST_F(SchedulerTest, TokenBudgetSplitsLongPrompts) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{16, 16});
  Collected out;
  const auto p = prompt(50, 7);
  const uint64_t id = sched.submit(make_request(p, 5, out));
  sched.step();
  EXPECT_EQ(sched.stats().last_batch_rows, 16);  // prompt chunked by the budget
  sched.run_until_idle();
  EXPECT_EQ(out.tokens[id], solo(p, 5));
}

TEST_F(SchedulerTest, CancellationReleasesKvAndOthersContinue) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  const auto pa = prompt(10, 1), pb = prompt(10, 2);
  const uint64_t a = sched.submit(make_request(pa, 40, out));
  const uint64_t b = sched.submit(make_request(pb, 10, out));
  for (int s = 0; s < 4; ++s) sched.step();
  sched.cancel(a);
  sched.run_until_idle();
  EXPECT_EQ(out.final_event[a].status, SequenceStatus::kCancelled);
  EXPECT_LT(out.tokens[a].size(), 40u);
  EXPECT_EQ(out.tokens[b], solo(pb, 10));
  EXPECT_EQ(pool->free_blocks(), pool->num_blocks());
}

TEST_F(SchedulerTest, InvalidRequestsFailAlone) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  const uint64_t empty = sched.submit(make_request({}, 5, out));
  const uint64_t bad_tok = sched.submit(make_request({1, 99999, 3}, 5, out));
  const uint64_t too_long = sched.submit(make_request(prompt(4, 1), 100000, out));
  const auto p = prompt(6, 3);
  const uint64_t good = sched.submit(make_request(p, 6, out));
  sched.run_until_idle();
  for (uint64_t id : {empty, bad_tok, too_long}) {
    EXPECT_EQ(out.final_event[id].status, SequenceStatus::kError) << id;
    EXPECT_EQ(out.final_event[id].error.code(), StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(out.tokens[good], solo(p, 6));
  EXPECT_EQ(sched.stats().failed, 3u);
}

TEST_F(SchedulerTest, PreemptionUnderKvPressureKeepsResultsExact) {
  make_pool(12);  // 48 tokens of KV for all requests together
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  std::vector<std::pair<uint64_t, std::vector<TokenId>>> want;
  for (int i = 0; i < 4; ++i) {
    const auto p = prompt(8, 50 + static_cast<unsigned>(i));
    const uint64_t id = sched.submit(make_request(p, 14, out));  // 4 x 22 tokens > 48
    want.emplace_back(id, solo(p, 14));
  }
  sched.run_until_idle();
  EXPECT_GT(sched.stats().preemptions, 0u);
  for (const auto& [id, tokens] : want) EXPECT_EQ(out.tokens[id], tokens) << "request " << id;
  EXPECT_EQ(pool->free_blocks(), pool->num_blocks());
}

TEST_F(SchedulerTest, RequestLargerThanCacheFailsCleanly) {
  make_pool(4);  // 16 tokens total
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 16});
  Collected out;
  const uint64_t id = sched.submit(make_request(prompt(30, 1), 5, out));
  sched.run_until_idle();
  EXPECT_EQ(out.final_event[id].status, SequenceStatus::kError);
  EXPECT_EQ(out.final_event[id].error.code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(pool->free_blocks(), pool->num_blocks());
}

TEST_F(SchedulerTest, ConcurrentSubmitters) {
  Scheduler sched(*tf, *pool, *model->tokenizer, SchedulerConfig{64, 32});
  Collected out;
  std::mutex mu;  // callbacks run on this (scheduler) thread; map guarded for the submitters' reads
  std::atomic<int> submitted{0};
  std::vector<std::thread> producers;
  for (int t = 0; t < 4; ++t) {
    producers.emplace_back([&, t] {
      for (int i = 0; i < 5; ++i) {
        Request r;
        r.prompt = prompt(3 + static_cast<size_t>(i), static_cast<unsigned>(t * 10 + i));
        r.stop = StopParams{4, false};
        r.on_event = [&](const RequestEvent& ev) {
          std::lock_guard<std::mutex> lock(mu);
          if (ev.finished) out.final_event[ev.request_id] = ev;
        };
        sched.submit(std::move(r));
        submitted.fetch_add(1);
      }
    });
  }
  while (submitted.load() < 20 || !sched.idle()) sched.step();
  for (auto& p : producers) p.join();
  sched.run_until_idle();
  std::lock_guard<std::mutex> lock(mu);
  EXPECT_EQ(out.final_event.size(), 20u);
  EXPECT_EQ(sched.stats().completed, 20u);
}

}  // namespace
}  // namespace engine

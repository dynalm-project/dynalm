// Performance program P1 (measurement, DD-050): engine/scheduler timing
// accounting, thread-pool statistics, counters, the bandwidth probe, the
// decode traffic model and the bottleneck classifier.

#include <gtest/gtest.h>

#include <atomic>

#include "bench/analysis.h"
#include "bench/loadgen.h"
#include "dynacore/hardware/perf_counters.h"
#include "dynacore/execution/thread_pool.h"

namespace engine::bench {
namespace {

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

// --- classifier -------------------------------------------------------------------

ClassifierInputs decode_heavy() {
  ClassifierInputs in;
  in.wall_s = 10;
  in.forward_s = 9;
  in.decode_forward_s = 8;
  in.prefill_forward_s = 1;
  in.host_s = 0.2;
  return in;
}

TEST(Classifier, MemoryBoundWhenDecodeRunsNearTheBandwidthCeiling) {
  ClassifierInputs in = decode_heavy();
  in.est_decode_bw_gbs = 14;
  in.peak_bw_gbs = 18;
  const Classification c = classify(in);
  EXPECT_EQ(c.primary, Bottleneck::kMemoryBound);
  ASSERT_FALSE(c.evidence.empty());
  EXPECT_NE(c.evidence[0].find("78%"), std::string::npos) << c.evidence[0];
}

TEST(Classifier, SynchronizationVersusImbalanceDependsOnRegionLength) {
  ClassifierInputs in = decode_heavy();
  in.pool_tail_wait_s = 3;  // a third of forward time idle at region ends
  in.pool_region_s = 8;
  in.pool_regions = 400000;  // 20 us regions
  in.regions_per_step = 100;
  EXPECT_EQ(classify(in).primary, Bottleneck::kSynchronizationBound);
  in.pool_regions = 20000;  // 400 us regions: the wait is uneven work, not hand-off cost
  EXPECT_EQ(classify(in).primary, Bottleneck::kLoadImbalanced);
}

TEST(Classifier, DispatchBoundOnManyTinyRegionsOrHostOverhead) {
  ClassifierInputs in = decode_heavy();
  in.pool_tail_wait_s = 0.5;
  in.pool_region_s = 4;
  in.pool_regions = 400000;  // 10 us each
  in.regions_per_step = 300;
  EXPECT_EQ(classify(in).primary, Bottleneck::kDispatchBound);
  ClassifierInputs host = decode_heavy();
  host.host_s = 4;  // 4 s of scheduler work next to 9 s of forward
  EXPECT_EQ(classify(host).primary, Bottleneck::kDispatchBound);
}

TEST(Classifier, IoOutranksEverythingAndComputeNeedsPrefill) {
  ClassifierInputs in = decode_heavy();
  in.est_decode_bw_gbs = 14;
  in.peak_bw_gbs = 18;
  in.major_fault_mb_s = 200;
  const Classification c = classify(in);
  EXPECT_EQ(c.primary, Bottleneck::kIoBound);
  EXPECT_EQ(c.all.size(), 2u);  // memory-bound is still reported as secondary

  ClassifierInputs pf;
  pf.wall_s = 10;
  pf.forward_s = 9;
  pf.prefill_forward_s = 7;
  pf.decode_forward_s = 2;
  pf.est_decode_bw_gbs = 3;
  pf.peak_bw_gbs = 18;
  pf.host_s = 0.1;
  EXPECT_EQ(classify(pf).primary, Bottleneck::kComputeBound);
}

TEST(Classifier, CacheBoundNeedsCountersAndMixedWhenNothingDominates) {
  ClassifierInputs in = decode_heavy();
  in.est_decode_bw_gbs = 8;  // 44% of the ceiling: neither memory- nor clearly compute-bound
  in.peak_bw_gbs = 18;
  EXPECT_EQ(classify(in).primary, Bottleneck::kMixed);
  in.llc_miss_ratio = 0.45;
  EXPECT_EQ(classify(in).primary, Bottleneck::kCacheBound);
}

TEST(Classifier, DecodeWithIdleMemoryIsComputeBound) {
  ClassifierInputs in = decode_heavy();
  in.est_decode_bw_gbs = 3.6;  // 21% of the ceiling, no synchronization signal
  in.peak_bw_gbs = 17.4;
  const Classification c = classify(in);
  EXPECT_EQ(c.primary, Bottleneck::kComputeBound);
  EXPECT_NE(c.evidence[0].find("dequantization"), std::string::npos);
}

// --- traffic model -------------------------------------------------------------------

TEST(TrafficModel, DenseReadsEveryWeightOnceMoeOnlyTouchedExperts) {
  auto dense = load_model(data("tiny_llama.gguf"));
  ASSERT_TRUE(dense.ok());
  const double w1 = decode_weight_bytes_per_step(**dense, 1);
  const double w8 = decode_weight_bytes_per_step(**dense, 8);
  EXPECT_GT(w1, 0);
  EXPECT_LE(w1, static_cast<double>((*dense)->weights.total_bytes()));
  EXPECT_NEAR(w8, w1, 0.01 * w1);  // more rows barely change a dense model's weight traffic

  auto moe = load_model(data("tiny_mixtral.gguf"));
  ASSERT_TRUE(moe.ok());
  const double m1 = decode_weight_bytes_per_step(**moe, 1);
  const double m16 = decode_weight_bytes_per_step(**moe, 16);
  EXPECT_LT(m1, m16);  // more rows touch more experts
  EXPECT_LE(m16, static_cast<double>((*moe)->weights.total_bytes()) + 1);
}

TEST(TrafficModel, KvBytesPerToken) {
  // 24 layers x 2 KV heads x (64 + 64) x 2 bytes x 1000 tokens.
  EXPECT_DOUBLE_EQ(kv_bytes_per_token_read(24, 2, 64, 64, 2, 1000), 24.0 * 2 * 128 * 2 * 1000);
}

// --- counters, pool, bandwidth ---------------------------------------------------------

TEST(PerfCountersTest, OpenReadAndDeltaNeverFail) {
  auto pc = PerfCounters::open();
  ASSERT_NE(pc, nullptr);
  const PerfSample a = pc->read();
  volatile double x = 0;
  for (int i = 0; i < 2000000; ++i) x = x + i;
  const PerfSample b = pc->read();
  const PerfSample d = PerfSample::delta(a, b);
  if (pc->hardware_available()) {
    EXPECT_GT(d.instructions, 0);
  } else {
    EXPECT_EQ(d.instructions, -1);
    EXPECT_FALSE(pc->hardware_reason().empty());
  }
  EXPECT_GE(d.page_faults, -1);
  PerfSample na;  // all unavailable
  EXPECT_EQ(PerfSample::delta(na, b).cycles, -1);
}

TEST(ThreadPoolStatsTest, CountsRegionsOnlyWhileEnabled) {
  ThreadPool pool(4);
  std::atomic<int> sum{0};
  pool.parallel_for(100, 1, [&](size_t b, size_t e) { sum += static_cast<int>(e - b); });
  EXPECT_EQ(pool.stats().regions, 0u);
  pool.set_stats_enabled(true);
  for (int i = 0; i < 10; ++i) pool.parallel_for(100, 1, [&](size_t b, size_t e) { sum += static_cast<int>(e - b); });
  pool.parallel_for(1, 4, [&](size_t b, size_t e) { sum += static_cast<int>(e - b); });  // inline
  const ThreadPoolStats s = pool.stats();
  EXPECT_EQ(s.regions, 10u);
  EXPECT_EQ(s.inline_regions, 1u);
  EXPECT_GT(s.region_ns, 0);
  EXPECT_GE(s.region_ns, s.tail_wait_ns);
  EXPECT_EQ(sum.load(), 1101);
}

TEST(Bandwidth, ProbeReturnsAPlausibleRate) {
  ThreadPool pool(2);
  const double gbs = measure_read_bandwidth_gbs(pool, size_t{16} << 20);
  EXPECT_GT(gbs, 0.1);
  EXPECT_LT(gbs, 10000);
}

// --- end to end ---------------------------------------------------------------------------

TEST(Diagnostics, RunPointAccountsForEveryStep) {
  EngineOptions o;
  o.model_path = data("tiny_llama.gguf");
  o.threads = 2;
  o.kv_tokens = 4096;
  o.max_batch_tokens = 64;
  auto e = Engine::create(o);
  ASSERT_TRUE(e.ok());
  auto target = make_engine_target(**e);
  auto perf = PerfCounters::open();
  RunContext ctx;
  ctx.perf = perf.get();
  ctx.peak_bw_gbs = 10;
  PointConfig cfg;
  cfg.concurrency = 3;
  cfg.output_tokens = 10;
  cfg.requests = 6;
  cfg.prompt_mix = {16, 40};
  const PointResult r = run_point(*target, *(*e)->model().tokenizer, cfg, ctx);
  ASSERT_EQ(r.completed, 6);
  EXPECT_EQ(r.cfg.prompt_tokens, 28);  // mean of the mix
  const Diagnostics& d = r.diag;
  ASSERT_TRUE(d.valid);
  EXPECT_EQ(d.steps, d.steps_decode_only + d.steps_prefill_only + d.steps_mixed);
  EXPECT_GT(d.steps_decode_only, 0u);
  EXPECT_GT(d.forward_ms, 0);
  EXPECT_NEAR(d.forward_ms, d.forward_decode_only_ms + d.forward_prefill_only_ms + d.forward_mixed_ms, 1e-6);
  EXPECT_GE(d.mean_decode_rows, 1.0);
  EXPECT_GT(d.tokenize_ms, 0);
  EXPECT_FALSE(d.op_ms.empty());  // profiling was on
  EXPECT_GT(d.pool_regions, 0);
  EXPECT_GT(d.est_decode_bw_gbs, 0);
  EXPECT_FALSE(d.bottleneck.evidence.empty());
  const std::string js = to_json(r, "tiny", "test");
  EXPECT_NE(js.find("\"bottleneck\""), std::string::npos);
  EXPECT_NE(js.find("\"prompt_mix\""), std::string::npos);
}

}  // namespace
}  // namespace engine::bench

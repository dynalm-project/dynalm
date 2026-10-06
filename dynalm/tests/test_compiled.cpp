// Compiled execution (DD-071, DD-072): every architecture through the
// DynaCore recording device. Deferred mode with all fusions must reproduce
// the reference CPU device bit for bit (same kernels, fused differently);
// trace mode must produce IR that verifies.

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/ir/text.h"
#include "dynacore/ir/verifier.h"
#include "loader/model_loader.h"
#include "runtime/engine.h"
#include "runtime/generator.h"
#include "scheduler/scheduler.h"
#include "common/core.h"

namespace dynalm {
namespace {

namespace ir = dynacore::ir;

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

struct Stack {
  std::unique_ptr<KvBlockPool> kv;
  std::unique_ptr<Transformer> tf;
};

Stack build(const LoadedModel& m, Device& be, int32_t blocks = 32) {
  const ModelConfig& c = m.config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, blocks, DType::kF16};
  auto kv = KvBlockPool::create(g, be);
  auto tf = Transformer::create(c, m.weights, be, 32);
  EXPECT_TRUE(kv.ok() && tf.ok()) << (tf.ok() ? "" : tf.status().to_string());
  return {std::move(*kv), std::move(*tf)};
}

std::unique_ptr<ir::RecordingDevice> compiled(Device& inner, int threads) {
  auto rec = std::make_unique<ir::RecordingDevice>(inner, ir::RecordingDevice::Mode::kDeferred);
  ir::CompileOptions co;
  co.verify = true;  // tests verify every compiled segment
  co.cost = ir::CostModel::from(KernelPlan::defaults(), threads);
  rec->set_planner(ir::make_planner(co));
  return rec;
}

class CompiledArch : public ::testing::TestWithParam<std::string> {};

TEST_P(CompiledArch, MatchesReferenceBitForBit) {
  auto m = load_model(data("tiny_" + GetParam() + ".gguf"));
  ASSERT_TRUE(m.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  auto rec = compiled(cpu, 2);
  Stack a = build(**m, cpu), b = build(**m, *rec);

  const std::vector<TokenId> prompt = {5, 17, 99, 3, 200, 42, 7, 128, 64, 11, 250, 33, 5, 17, 99};
  GenerateOptions o;
  o.max_new_tokens = 24;
  o.stop_at_eog = false;
  std::vector<TokenId> out_ref, out_comp;
  Generator ga(*a.tf, *a.kv, *(*m)->tokenizer), gb(*b.tf, *b.kv, *(*m)->tokenizer);
  ASSERT_TRUE(ga.generate(prompt, o, [&](TokenId t) { out_ref.push_back(t); return true; }).ok());
  ASSERT_TRUE(gb.generate(prompt, o, [&](TokenId t) { out_comp.push_back(t); return true; }).ok());
  EXPECT_EQ(out_comp, out_ref);
  const ir::RecordingStats& st = rec->stats();
  EXPECT_EQ(st.fallbacks, 0);
  EXPECT_GT(st.cache_hits, 0) << "decode steps repeat one structure; the plan cache must hit";

  KvBlockTable ta(*a.kv), tb(*b.kv);
  ASSERT_TRUE(ta.reserve(static_cast<int64_t>(prompt.size())).ok() &&
              tb.reserve(static_cast<int64_t>(prompt.size())).ok());
  std::vector<int32_t> pos(prompt.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> la(static_cast<size_t>((*m)->config.vocab_size)), lb(la.size());
  ASSERT_TRUE(a.tf->forward(prompt, pos, *a.kv, ta.block_table(), la).ok());
  ASSERT_TRUE(b.tf->forward(prompt, pos, *b.kv, tb.block_table(), lb).ok());
  EXPECT_EQ(la, lb);
}

TEST_P(CompiledArch, TraceProducesVerifiedIr) {
  auto m = load_model(data("tiny_" + GetParam() + ".gguf"));
  ASSERT_TRUE(m.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  ir::RecordingDevice trace(cpu, ir::RecordingDevice::Mode::kTrace);
  Stack s = build(**m, trace);
  KvBlockTable t(*s.kv);
  ASSERT_TRUE(t.reserve(8).ok());
  const std::vector<TokenId> toks = {5, 17, 99, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  std::vector<float> logits(static_cast<size_t>((*m)->config.vocab_size));
  trace.begin_graph("prefill");
  ASSERT_TRUE(s.tf->forward(toks, pos, *s.kv, t.block_table(), logits).ok());
  const ir::Graph& g = trace.graph();
  EXPECT_TRUE(ir::verify(g).ok()) << ir::verify(g).message();
  // The text form round-trips (parse(print(g)) prints the same).
  const std::string text = ir::print_graph(g, {.annotations = false, .names = false});
  auto parsed = ir::parse_graph(text);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(ir::print_graph(*parsed, {.annotations = false, .names = false}), text);
}

INSTANTIATE_TEST_SUITE_P(All, CompiledArch,
                         ::testing::Values("llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "mixtral",
                                           "qwen2moe", "qwen3moe", "granitemoe"),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });

TEST(Compiled, SchedulerWithPrefixCacheMatchesReference) {
  auto m = load_model(data("tiny_llama.gguf"));
  ASSERT_TRUE(m.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  auto rec = compiled(cpu, 2);
  auto run = [&](Device& be) {
    Stack s = build(**m, be, 64);
    Scheduler sched(*s.tf, *s.kv, *(*m)->tokenizer, SchedulerConfig{});
    std::map<uint64_t, std::vector<TokenId>> out;
    std::vector<TokenId> shared(40);
    std::iota(shared.begin(), shared.end(), 10);
    for (int i = 0; i < 6; ++i) {
      Request r;
      r.prompt = shared;
      r.prompt.push_back(static_cast<TokenId>(100 + i));
      if (i % 2) r.prompt.resize(37);  // partial-block prefix match -> copy-on-write
      r.stop = StopParams{12, false};
      r.on_event = [&out](const RequestEvent& ev) {
        if (!ev.finished) out[ev.request_id].push_back(ev.token);
      };
      sched.submit(std::move(r));
      sched.step();
    }
    sched.run_until_idle();
    std::vector<std::vector<TokenId>> v;
    for (auto& [id, toks] : out) v.push_back(toks);
    return v;
  };
  EXPECT_EQ(run(*rec), run(cpu));
  EXPECT_EQ(rec->stats().fallbacks, 0);
}

TEST(Compiled, EngineExecutionModeOption) {
  EXPECT_EQ(*parse_execution_mode("compiled"), ExecutionMode::kCompiled);
  EXPECT_EQ(*parse_execution_mode("reference"), ExecutionMode::kReference);
  EXPECT_FALSE(parse_execution_mode("jit").ok());
  EngineOptions o;
  o.model_path = data("tiny_qwen2.gguf");
  o.threads = 2;
  o.kv_tokens = 256;
  o.execution = ExecutionMode::kCompiled;
  auto e = Engine::create(o);
  ASSERT_TRUE(e.ok()) << e.status().to_string();
  EXPECT_NE((*e)->backend_name().find("compiled/"), std::string::npos);
  GenerateParams params;
  params.max_tokens = 8;
  params.stop_at_eog = false;
  auto stream = (*e)->generate_text("hello world", false, params);
  ASSERT_TRUE(stream.ok());
  int tokens = 0;
  StreamEvent ev;
  while ((*stream)->next(ev)) {
    if (!ev.text.empty()) ++tokens;
    if (ev.done) {
      EXPECT_EQ(ev.finish, StreamFinish::kLength);
      break;
    }
  }
  EXPECT_EQ(ev.completion_tokens, 8);
  (void)tokens;
}

}  // namespace
}  // namespace dynalm

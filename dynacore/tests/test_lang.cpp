// DynaCore language (DD-073): parsing, semantic errors, lowering to IR, and
// execution of the lowered graph through the compiler pipeline.

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/executor.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/text.h"
#include "dynacore/ir/verifier.h"
#include "dynacore/lang/lang.h"

namespace dynacore::lang {
namespace {

std::string example() {
  std::ifstream in(std::string(DYNACORE_SOURCE_DIR) + "/examples/dynacore/decoder_layer.dyna");
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string error_of(std::string_view src) {
  auto p = compile(src, "t.dyna");
  return p.ok() ? "" : p.status().message();
}

TEST(Lang, CompilesTheExampleLayer) {
  auto p = compile(example(), "decoder_layer.dyna");
  ASSERT_TRUE(p.ok()) << p.status().message();
  EXPECT_TRUE(ir::verify(p->graph).ok()) << ir::verify(p->graph).message();
  EXPECT_EQ(p->outputs.size(), 1u);
  EXPECT_EQ(p->symbols.at("M"), 1);
  int gqa = 0, qmm = 0, bias = 0;
  for (const ir::Op& op : p->graph.ops()) {
    gqa += op.kind == ir::OpKind::kGqa;
    qmm += op.kind == ir::OpKind::kQuantizedMatMul || op.kind == ir::OpKind::kMatMul;
    bias += (op.kind == ir::OpKind::kMatMul || op.kind == ir::OpKind::kQuantizedMatMul) && op.inputs.size() == 3;
  }
  EXPECT_EQ(gqa, 1);   // 12 heads over 2 KV heads
  EXPECT_EQ(qmm, 7);   // q k v o gate up down
  EXPECT_EQ(bias, 3);  // (x @ W) + b folds into the matmul for q, k, v
}

TEST(Lang, ScheduleSetsFusionAndOptimizerFuses) {
  auto p = compile(example(), "decoder_layer.dyna");
  ASSERT_TRUE(p.ok());
  ir::FusionReport rep;
  ir::Graph g = p->graph;
  ir::select_kernels(g, ir::CostModel{});
  ir::plan_execution(g, p->fusion, ir::CostModel{}, &rep);
  EXPECT_EQ(rep.groups, 1);  // Q/K/V
  EXPECT_EQ(rep.gated, 1);   // gate/up/swiglu
  auto off = compile("graph g(M=1) { input x : tensor<f32>[M, 256]\n weight w : q8_0[64, 256]\n y = x @ w\n"
                     "output y\n schedule { fuse none; group none } }");
  ASSERT_TRUE(off.ok()) << off.status().message();
  EXPECT_FALSE(off->fusion.gated_mlp);
  EXPECT_FALSE(off->fusion.group_shared_input);
}

TEST(Lang, ErrorsNameTheLocation) {
  EXPECT_NE(error_of("graph g { y = x }").find("t.dyna:1:"), std::string::npos);
  EXPECT_NE(error_of("graph g { y = x }").find("undefined name 'x'"), std::string::npos);
  EXPECT_NE(error_of("graph g {\n input x : tensor<f32>[4, 8]\n y = frob(x)\n output y }").find("t.dyna:3:"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, 8]\n weight w : q4_K[4, 8]\n y = x @ w\n output y }")
                .find("whole q4_K blocks"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, D] }").find("unknown dimension 'D'"), std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, 8]\n x = x\n output x }").find("cannot assign"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, 8]\n y = x @ x\n output y }").find("must be a weight"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, 8] }").find("no 'output'"), std::string::npos);
  EXPECT_NE(error_of("graph g {\n config heads = 30\n input q : tensor<f32>[1, 30 * 64]\n input p : index[1]\n"
                     " kv c : kv<f16>[4, 4, 16, 64]\n a = attention(q, c, p, p, heads, 4, 64)\n output a }")
                .find("GQA requires query_heads % kv_heads == 0"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4, 8]\n output x\n schedule { prefetch weights } }")
                .find("unknown schedule directive"),
            std::string::npos);
  EXPECT_NE(error_of("graph g { input x : tensor<f32>[4,8] $ }").find("unexpected character"), std::string::npos);
  EXPECT_FALSE(compile("").ok());
}

TEST(Lang, MalformedSourceNeverCrashes) {
  const std::string src = example();
  for (size_t n = 0; n < src.size(); n += 11) (void)compile(src.substr(0, n));
  std::mt19937 rng(3);
  for (int i = 0; i < 300; ++i) {
    std::string s = src;
    s[rng() % s.size()] = static_cast<char>(32 + rng() % 95);
    (void)compile(s);
  }
}

TEST(Lang, LoweredGraphRunsAndFusionIsExact) {
  auto p = compile(example(), "decoder_layer.dyna");
  ASSERT_TRUE(p.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  ir::Graph g = p->graph;
  ir::CompileOptions co;
  co.verify = true;
  co.fusion = p->fusion;
  auto plan = ir::compile_segment(g, co);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  auto ex = ir::GraphExecutor::create(g, cpu, ir::ExecutorOptions{});
  ASSERT_TRUE(ex.ok()) << ex.status().message();
  ASSERT_TRUE((*ex)->run(nullptr).ok());
  const TensorView y = (*ex)->view(p->outputs[0]);
  std::vector<float> a(y.data_as<float>(), y.data_as<float>() + y.numel());
  ASSERT_TRUE((*ex)->run(&*plan).ok());
  std::vector<float> b(y.data_as<float>(), y.data_as<float>() + y.numel());
  EXPECT_EQ(a, b);
  for (float v : a) ASSERT_TRUE(std::isfinite(v));
}

TEST(Lang, UnsupportedOpsAreReportedBeforeRunning) {
  auto p = compile("graph g { input x : tensor<f32>[2, 8]\n y = softmax(x)\n output y }");
  ASSERT_TRUE(p.ok()) << p.status().message();
  ThreadPool pool(1);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  auto ex = ir::GraphExecutor::create(p->graph, cpu, {});
  ASSERT_FALSE(ex.ok());
  EXPECT_EQ(ex.status().code(), StatusCode::kUnsupported);
}

}  // namespace
}  // namespace dynacore::lang

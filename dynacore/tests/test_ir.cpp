// DynaCore IR (DD-071, DD-072): types, builder, verifier, text round trip,
// passes, and the fused CPU kernels the compiler emits.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/ir.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/ir/text.h"
#include "dynacore/ir/verifier.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

namespace dynacore::ir {
namespace {

Type t(DType d, std::initializer_list<int64_t> dims) {
  Shape s;
  for (int64_t x : dims) s.push_back(Dim::of(x));
  return tensor_type(d, s);
}
Type w(DType d, std::initializer_list<int64_t> dims) {
  Shape s;
  for (int64_t x : dims) s.push_back(Dim::of(x));
  return weight_type(d, s);
}

// One decoder layer's attention + gated MLP, built by hand.
Graph decoder_layer(int64_t heads = 32, int64_t kv_heads = 4) {
  Graph g("layer");
  Builder b(g);
  const int64_t hd = 128, hidden = heads * hd, ff = 512;
  const ValueId x = b.input("x", tensor_type(DType::kF32, {Dim::symbol("M"), Dim::of(hidden)}));
  const ValueId pos = b.input("positions", index_type({Dim::symbol("M")}));
  const ValueId seq = b.input("row_seq", index_type({Dim::symbol("M")}));
  const ValueId norm_w = b.weight("attn_norm", w(DType::kF32, {hidden}));
  const ValueId wq = b.weight("wq", w(DType::kQ4_K, {hidden, hidden}));
  const ValueId wk = b.weight("wk", w(DType::kQ4_K, {kv_heads * hd, hidden}));
  const ValueId wv = b.weight("wv", w(DType::kQ6_K, {kv_heads * hd, hidden}));
  const ValueId kv = b.kv_cache("kv0", kv_type(DType::kF16, 64, kv_heads, 16, hd));
  const ValueId xn = b.rms_norm(x, norm_w, 1e-6);
  const ValueId q = b.rope(b.matmul(xn, wq), pos, heads, hd);
  const ValueId k = b.rope(b.matmul(xn, wk), pos, kv_heads, hd);
  const ValueId v = b.matmul(xn, wv);
  const ValueId kv1 = b.kv_write(kv, k, v, pos, seq);
  const ValueId a = b.attention(q, kv1, pos, seq, heads, kv_heads, hd, 1.0 / std::sqrt(128.0));
  const ValueId h = b.add(x, a);
  const ValueId wg = b.weight("w_gate", w(DType::kQ4_K, {ff, hidden}));
  const ValueId wu = b.weight("w_up", w(DType::kQ4_K, {ff, hidden}));
  const ValueId hn = b.rms_norm(h, norm_w, 1e-6);
  const ValueId gate = b.matmul(hn, wg);
  const ValueId up = b.matmul(hn, wu);
  b.act_mul(gate, up, "silu");
  EXPECT_TRUE(b.status().ok()) << b.status().message();
  return g;
}

TEST(IrTypes, PrintAndBytes) {
  EXPECT_EQ(type_to_string(w(DType::kQ4_K, {4096, 4096})), "weight<q4_K>[4096,4096] blocked(256)");
  EXPECT_EQ(type_to_string(kv_type(DType::kF16, 8, 4, 16, 128)), "kv<f16>[8,4,16,128] paged(16)");
  EXPECT_EQ(type_to_string(tensor_type(DType::kF32, {Dim::symbol("M"), Dim::of(64)})), "tensor<f32>[M,64]");
  EXPECT_EQ(w(DType::kQ4_K, {256, 512}).bytes(), 256 * 2 * 144);  // 2 blocks of 144 bytes per row
  EXPECT_EQ(tensor_type(DType::kF32, {Dim::symbol("M")}).bytes(), -1);
}

TEST(IrBuilder, InfersTypesThroughALayer) {
  Graph g = decoder_layer();
  EXPECT_TRUE(verify(g).ok()) << verify(g).message();
  const Op& attn = g.ops()[static_cast<size_t>(g.value(static_cast<ValueId>(g.values().size()) - 1).producer)];
  EXPECT_EQ(attn.kind, OpKind::kActMul);
  // The attention op is gqa (32 query heads over 4 KV heads).
  bool found = false;
  for (const Op& op : g.ops()) found = found || op.kind == OpKind::kGqa;
  EXPECT_TRUE(found);
}

TEST(IrBuilder, RejectsBadGqaGrouping) {
  Graph g("bad");
  Builder b(g);
  const ValueId q = b.input("q", tensor_type(DType::kF32, {Dim::of(1), Dim::of(30 * 64)}));
  const ValueId kv = b.kv_cache("kv", kv_type(DType::kF16, 4, 4, 16, 64));
  const ValueId pos = b.input("p", index_type({Dim::of(1)}));
  b.attention(q, kv, pos, pos, 30, 4, 64, 0.125);
  ASSERT_FALSE(b.status().ok());
  EXPECT_NE(b.status().message().find("GQA requires query_heads % kv_heads == 0 (query_heads = 30, kv_heads = 4)"),
            std::string::npos)
      << b.status().message();
}

TEST(IrBuilder, RejectsShapeAndQuantizationErrors) {
  Graph g("bad");
  Builder b(g);
  const ValueId x = b.input("x", t(DType::kF32, {2, 100}));
  b.matmul(x, b.weight("w", w(DType::kQ4_K, {8, 100})));  // K=100 is not whole 256-blocks
  EXPECT_NE(b.status().message().find("whole q4_K blocks"), std::string::npos) << b.status().message();
  Graph g2("bad2");
  Builder b2(g2);
  b2.matmul(b2.input("x", t(DType::kF32, {2, 64})), b2.weight("w", w(DType::kF16, {8, 32})));
  EXPECT_NE(b2.status().message().find("inner dimensions differ"), std::string::npos);
}

TEST(IrVerifier, CatchesUseBeforeDefinitionAndTypeLies) {
  Graph g = decoder_layer();
  // Swap two compute ops: an operand is now used before it is defined.
  std::vector<Op> ops(g.ops().begin(), g.ops().end());
  size_t a = 0;
  while (ops[a].kind != OpKind::kRmsNorm) ++a;
  std::swap(ops[a], ops[a + 1]);
  Graph broken = g;
  broken.set_ops(ops);
  EXPECT_FALSE(verify(broken).ok());
  EXPECT_NE(verify(broken).message().find("used before it is defined"), std::string::npos) << verify(broken).message();
  // A result type that its operands cannot produce.
  Graph lie = g;
  lie.mutable_value(static_cast<ValueId>(lie.values().size()) - 1).type = t(DType::kF16, {1, 2});
  EXPECT_FALSE(verify(lie).ok());
}

TEST(IrVerifier, MemoryBudget) {
  Graph g = decoder_layer();
  VerifyOptions o;
  o.memory_budget = 1 << 20;  // 1 MiB: far below the weights
  o.symbol_size = 4;
  EXPECT_FALSE(verify(g, o).ok());
  o.memory_budget = int64_t{1} << 32;
  EXPECT_TRUE(verify(g, o).ok());
}

TEST(IrText, RoundTripAndErrors) {
  Graph g = decoder_layer();
  const std::string text = print_graph(g);
  auto p = parse_graph(text);
  ASSERT_TRUE(p.ok()) << p.status().message();
  EXPECT_EQ(print_graph(*p), text);
  EXPECT_TRUE(verify(*p).ok());

  const char* hand =
      "graph @tiny {\n"
      "  %x = input \"x\" : tensor<f32>[M,256]\n"
      "  %w = weight \"w\" : weight<q8_0>[64,256]\n"
      "  %y = qmatmul %x, %w : tensor<f32>[M,64]   // comment\n"
      "  %z = scale %y {s=0.5} : tensor<f32>[M,64]\n"
      "}\n";
  auto h = parse_graph(hand);
  ASSERT_TRUE(h.ok()) << h.status().message();
  EXPECT_EQ(h->value(3).buffer, h->value(2).buffer);  // scale writes in place

  auto bad_type = parse_graph("graph { %x = input \"x\" : tensor<f32>[4,8]\n %y = softmax %x : tensor<f32>[4,9] }");
  ASSERT_FALSE(bad_type.ok());
  EXPECT_NE(bad_type.status().message().find("declared type"), std::string::npos);
  auto undefined = parse_graph("graph {\n  %y = softmax %nope : tensor<f32>[1]\n}");
  ASSERT_FALSE(undefined.ok());
  EXPECT_NE(undefined.status().message().find("ir:2:"), std::string::npos) << undefined.status().message();
  EXPECT_FALSE(parse_graph("graph { %x = frobnicate : tensor<f32>[1] }").ok());
  EXPECT_FALSE(parse_graph("graph { %x = input \"x\" : tensor<f99>[1] }").ok());
  EXPECT_FALSE(parse_graph("").ok());
  EXPECT_FALSE(parse_graph("graph {").ok());
}

TEST(IrText, MalformedInputNeverCrashes) {
  // Cheap fuzz: every prefix and single-byte corruption of a valid program
  // parses or fails cleanly.
  const std::string text = print_graph(decoder_layer());
  for (size_t n = 0; n < text.size(); n += 7) (void)parse_graph(text.substr(0, n));
  std::mt19937 rng(42);
  for (int i = 0; i < 300; ++i) {
    std::string s = text;
    s[rng() % s.size()] = static_cast<char>(rng() % 128);
    auto r = parse_graph(s);
    if (r.ok()) (void)verify(*r);
  }
}

TEST(IrPasses, FusionFindsGroupsAndGatedMlp) {
  Graph g = decoder_layer();
  CostModel cm;
  FusionReport rep;
  const ExecPlan plan = plan_execution(g, FusionOptions{}, cm, &rep);
  // In this hand-built layer RoPE follows each projection, so Q/K/V are not
  // back to back and form no group; the gated MLP is fused.
  EXPECT_EQ(rep.gated, 1);
  bool gated = false;
  for (const ExecStep& s : plan.steps) gated = gated || s.kind == ExecStep::Kind::kGatedMatmul;
  EXPECT_TRUE(gated);
  // Every compute op appears exactly once.
  std::vector<int> seen(g.ops().size(), 0);
  for (const ExecStep& s : plan.steps)
    for (int32_t op : s.ops) ++seen[static_cast<size_t>(op)];
  for (size_t i = 0; i < g.ops().size(); ++i) {
    const OpKind k = g.ops()[i].kind;
    const bool leaf = k == OpKind::kInput || k == OpKind::kWeight || k == OpKind::kKvCache || k == OpKind::kConstant;
    EXPECT_EQ(seen[i], leaf ? 0 : 1) << i;
  }
}

TEST(IrPasses, GroupsBackToBackSharedInputMatmuls) {
  Graph g("qkv");
  Builder b(g);
  const ValueId x = b.input("x", t(DType::kF32, {1, 256}));
  const ValueId q = b.matmul(x, b.weight("wq", w(DType::kQ8_0, {256, 256})));
  const ValueId k = b.matmul(x, b.weight("wk", w(DType::kQ8_0, {64, 256})));
  const ValueId v = b.matmul(x, b.weight("wv", w(DType::kQ8_0, {64, 256})));
  b.matmul(q, b.weight("wo", w(DType::kQ8_0, {256, 256})));  // depends on q: not grouped
  (void)k;
  (void)v;
  FusionReport rep;
  const ExecPlan plan = plan_execution(g, FusionOptions{}, CostModel{}, &rep);
  EXPECT_EQ(rep.groups, 1);
  EXPECT_EQ(rep.grouped_ops, 3);
  ASSERT_GE(plan.steps.size(), 2u);
  EXPECT_EQ(plan.steps[0].kind, ExecStep::Kind::kMatmulGroup);
}

TEST(IrPasses, KernelSelectionFollowsRowsAndPlan) {
  Graph g("k");
  Builder b(g);
  const ValueId x1 = b.input("x1", t(DType::kF32, {1, 256}));
  const ValueId x64 = b.input("x64", t(DType::kF32, {64, 256}));
  const ValueId wq = b.weight("w", w(DType::kQ4_K, {128, 256}));
  b.matmul(x1, wq);
  b.matmul(x64, wq);
  select_kernels(g, CostModel{});
  std::vector<std::string> kernels;
  for (const Op& op : g.ops()) if (!op.kernel.empty()) kernels.push_back(op.kernel);
  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(kernels[0], "cpu.int8_dot.q4_K");
  EXPECT_EQ(kernels[1], "cpu.panel_gemm.q4_K");
}

TEST(IrPasses, MemoryAnalysisSeesReuse) {
  Graph g = decoder_layer();
  const MemoryReport r = analyze_memory(g);
  EXPECT_GT(r.values, 5);
  EXPECT_GT(r.peak_live_bytes, 0);
  EXPECT_LE(r.peak_live_bytes, r.activation_bytes);
}

// --- fused CPU kernels the compiler emits ------------------------------------

// Reference Q8_0 quantization: per 32-value block, d = max|x| / 127.
std::vector<std::byte> quantize_q8(const std::vector<float>& w, int64_t rows, int64_t k) {
  std::vector<quant::BlockQ8_0> blocks(static_cast<size_t>(rows * k / 32));
  for (size_t b = 0; b < blocks.size(); ++b) {
    const float* x = w.data() + b * 32;
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0f;
    blocks[b].d = fp32_to_fp16(d);
    for (int i = 0; i < 32; ++i) blocks[b].qs[i] = static_cast<int8_t>(std::lround(d > 0 ? x[i] / d : 0.0f));
  }
  std::vector<std::byte> out(blocks.size() * sizeof(quant::BlockQ8_0));
  std::memcpy(out.data(), blocks.data(), out.size());
  return out;
}

TensorView view2(void* p, DType d, int64_t r, int64_t c) {
  return TensorView(p, *TensorLayout::contiguous(d, TensorShape{r, c}));
}

TEST(FusedKernels, MatmulGatedEqualsUnfusedSequence) {
  ThreadPool pool(3);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  const int64_t m = 2, k = 256, n = 96;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 0.5f);
  std::vector<float> x(static_cast<size_t>(m * k)), wg(static_cast<size_t>(n * k)), wu(wg.size());
  for (float& v : x) v = nd(rng);
  for (float& v : wg) v = nd(rng);
  for (float& v : wu) v = nd(rng);
  auto qg = quantize_q8(wg, n, k), qu = quantize_q8(wu, n, k);
  std::vector<float> fused(static_cast<size_t>(m * n)), ref(fused.size()), gs(fused.size()), us(fused.size());
  const TensorView xv = view2(x.data(), DType::kF32, m, k), gv = view2(qg.data(), DType::kQ8_0, n, k),
                   uv = view2(qu.data(), DType::kQ8_0, n, k);
  cpu.matmul_gated(Activation::kSilu, xv, gv, uv, view2(fused.data(), DType::kF32, m, n),
                   view2(gs.data(), DType::kF32, m, n), view2(us.data(), DType::kF32, m, n));
  cpu.Device::matmul_gated(Activation::kSilu, xv, gv, uv, view2(ref.data(), DType::kF32, m, n),
                           view2(gs.data(), DType::kF32, m, n), view2(us.data(), DType::kF32, m, n));
  EXPECT_EQ(fused, ref);  // same dot kernels, same activation: bit-exact
}

TEST(FusedKernels, MatmulManyWithBiasAndSharedInput) {
  ThreadPool pool(3);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  const int64_t m = 1, k = 256;
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0, 0.5f);
  std::vector<float> x(static_cast<size_t>(k)), w1(static_cast<size_t>(64 * k)), w2(static_cast<size_t>(32 * k)),
      b1(64), b2(32);
  for (auto* v : {&x, &w1, &w2, &b1, &b2})
    for (float& f : *v) f = nd(rng);
  auto q1 = quantize_q8(w1, 64, k), q2 = quantize_q8(w2, 32, k);
  std::vector<float> y1(64), y2(32), r1(64), r2(32);
  const TensorView xv = view2(x.data(), DType::kF32, m, k);
  const TensorView bv1(b1.data(), *TensorLayout::contiguous(DType::kF32, TensorShape{64}));
  const TensorView bv2(b2.data(), *TensorLayout::contiguous(DType::kF32, TensorShape{32}));
  const Device::MatmulJob jobs[] = {
      {xv, view2(q1.data(), DType::kQ8_0, 64, k), view2(y1.data(), DType::kF32, m, 64), &bv1},
      {xv, view2(q2.data(), DType::kQ8_0, 32, k), view2(y2.data(), DType::kF32, m, 32), &bv2}};
  cpu.matmul_many(jobs);
  cpu.matmul(xv, view2(q1.data(), DType::kQ8_0, 64, k), &bv1, view2(r1.data(), DType::kF32, m, 64));
  cpu.matmul(xv, view2(q2.data(), DType::kQ8_0, 32, k), &bv2, view2(r2.data(), DType::kF32, m, 32));
  EXPECT_EQ(y1, r1);
  EXPECT_EQ(y2, r2);
}

}  // namespace
}  // namespace dynacore::ir

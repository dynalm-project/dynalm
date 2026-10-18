// Graph IR -> DynaKernel lowering: equivalence with GraphExecutor, explicit
// rejections, bindings. Self-contained: synthetic data, no model files.
//
// Tolerance: GraphExecutor and ReferenceBackend both call CpuDevice::matmul
// with KernelPlan::defaults() (act = auto) on the same ISA tier, and each
// output element is computed by one thread in a fixed order, so the two
// results must be bit-identical; any difference is a lowering or binding bug.
// Separately, both are checked against a double-precision reference.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/lower.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/executor.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/verifier.h"
#include "dynacore/lang/lang.h"
#include "dynacore/quantization/quant_formats.h"

using namespace dynacore;
using namespace dynacore::quant;
namespace gir = dynacore::ir;
using dk::ActFormat;
using dk::Bindings;
using dk::LowerIssueKind;
using dk::LowerOptions;
using dk::LowerResult;
using dk::ReferenceBackend;

namespace {

int g_checks = 0;

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

bool contains(const std::string& s, const std::string& n) { return s.find(n) != std::string::npos; }

bool has_issue(const LowerResult& r, LowerIssueKind k, const std::string& needle) {
  for (const dk::LowerIssue& i : r.issues) {
    if (i.kind == k && contains(i.message, needle)) return true;
  }
  return false;
}

std::vector<float> random_floats(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 7;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>((s >> 8) & 0xffff) / 32768.0f - 1.0f;
  }
  return v;
}

TensorView dense_view(void* data, DType t, std::initializer_list<int64_t> dims) {
  auto l = TensorLayout::contiguous(t, TensorShape(dims));
  CHECK(l.ok());
  return TensorView(data, *l);
}

CpuIsa this_isa() { return select_isa(cpu_info().features); }

gir::Type f32_tensor(gir::Dim m, int64_t k) { return gir::tensor_type(DType::kF32, {m, gir::Dim::of(k)}); }

// Caller data for one graph leaf: f32 rows, or GGUF Q8_0 blocks.
struct Leaf {
  std::vector<float> f;
  std::vector<BlockQ8_0> q;
  std::vector<float> deq;  // what the stored weight decodes to (reference)
  TensorView view;
};

Leaf f32_leaf(int64_t rows, int64_t cols, uint32_t seed) {
  Leaf l;
  l.f = random_floats(static_cast<size_t>(rows * cols), seed);
  l.deq = l.f;
  l.view = cols > 0 && rows > 0 ? dense_view(l.f.data(), DType::kF32, {rows, cols}) : TensorView();
  return l;
}

Leaf q8_leaf(int64_t rows, int64_t cols, uint32_t seed) {
  Leaf l;
  l.f = random_floats(static_cast<size_t>(rows * cols), seed);
  l.q.resize(static_cast<size_t>(rows * cols / 32));
  quantize_q8_0(l.f.data(), l.q.data(), static_cast<int64_t>(l.q.size()));
  l.deq.resize(l.f.size());
  dequantize_q8_0(l.q.data(), l.deq.data(), static_cast<int64_t>(l.q.size()));
  l.view = dense_view(l.q.data(), DType::kQ8_0, {rows, cols});
  return l;
}

Leaf bias_leaf(int64_t n, uint32_t seed) {
  Leaf l;
  l.f = random_floats(static_cast<size_t>(n), seed);
  l.deq = l.f;
  l.view = dense_view(l.f.data(), DType::kF32, {n});
  return l;
}

std::vector<double> ref_matmul(const std::vector<double>& x, const std::vector<float>& w, const std::vector<float>* b,
                               int64_t m, int64_t n, int64_t k) {
  std::vector<double> y(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = b ? (*b)[static_cast<size_t>(j)] : 0.0;
      for (int64_t t = 0; t < k; ++t) acc += x[static_cast<size_t>(i * k + t)] * w[static_cast<size_t>(j * k + t)];
      y[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return y;
}

// Runs `g` through GraphExecutor and through lowering + ReferenceBackend over
// the same bound leaves; returns both results of `out` (each m x n).
struct Both {
  std::vector<float> executor, kernel;
};

Both run_both(const gir::Graph& g, const std::map<gir::ValueId, TensorView>& leaves, gir::ValueId out, int64_t m,
              int64_t n, const std::map<std::string, int64_t>& symbols, ActFormat act = ActFormat::kAuto) {
  ThreadPool pool(4);
  CpuDevice dev(pool, this_isa());
  gir::ExecutorOptions eo;
  eo.symbols = symbols;
  auto ex = gir::GraphExecutor::create(g, dev, eo);
  CHECK(ex.ok());
  for (const auto& [v, view] : leaves) CHECK((*ex)->bind(v, view).ok());
  CHECK((*ex)->run().ok());
  const TensorView ev = (*ex)->view(out);
  Both r;
  r.executor.assign(ev.data_as<const float>(), ev.data_as<const float>() + m * n);

  LowerOptions lo;
  lo.symbols = symbols;
  lo.act = act;
  const LowerResult lr = dk::lower(g, lo);
  if (!lr.ok()) std::fprintf(stderr, "%s\n", lr.status().to_string().c_str());
  CHECK(lr.ok());
  ReferenceBackend be(pool, this_isa());
  Bindings bind;
  std::vector<std::vector<float>> results;  // kernel-side result buffers
  results.reserve(lr.lowered.values.size());
  for (const auto& [gv, kv] : lr.lowered.values) {
    const dk::Value& v = lr.lowered.program.value(kv);
    if (v.role == dk::ValueRole::kResult) {
      results.emplace_back(static_cast<size_t>(v.layout.span_bytes() / 4), std::nanf(""));
      bind.bind(kv, TensorView(results.back().data(), v.layout), v.layout.span_bytes());
    } else {
      bind.bind(kv, leaves.at(gv));
    }
  }
  const Status s = be.run(lr.lowered.program, bind);
  if (!s.ok()) std::fprintf(stderr, "%s\n", s.to_string().c_str());
  CHECK(s.ok());
  const TensorView* kvw = bind.find(lr.lowered.kernel_value(out));
  CHECK(kvw != nullptr);
  r.kernel.assign(kvw->data_as<const float>(), kvw->data_as<const float>() + m * n);
  return r;
}

double rel_err(const std::vector<float>& y, const std::vector<double>& ref) {
  double mr = 0, me = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    mr = std::max(mr, std::abs(ref[i]));
    me = std::max(me, std::abs(static_cast<double>(y[i]) - ref[i]));
  }
  return me / (mr + 1e-6);
}

// --- equivalence ----------------------------------------------------------------------------

void test_single_matmul_shapes() {
  // M symbolic (bound per case), N and K static; f32 weights, with bias.
  const int64_t cases[][3] = {{1, 7, 5}, {3, 16, 32}, {7, 33, 96}, {33, 70, 300}, {64, 8, 1100}};
  for (const auto& c : cases) {
    const int64_t m = c[0], n = c[1], k = c[2];
    gir::Graph g("one");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::symbol("M"), k));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(n), gir::Dim::of(k)}));
    const gir::ValueId bias = b.weight("b", gir::weight_type(DType::kF32, {gir::Dim::of(n)}));
    const gir::ValueId y = b.matmul(x, w, bias);
    CHECK(b.status().ok());
    Leaf lx = f32_leaf(m, k, 1), lw = f32_leaf(n, k, 2), lb = bias_leaf(n, 3);
    const Both r = run_both(g, {{x, lx.view}, {w, lw.view}, {bias, lb.view}}, y, m, n, {{"M", m}});
    CHECK(r.executor == r.kernel);  // bit-identical (see header)
    std::vector<double> xd(lx.f.begin(), lx.f.end());
    CHECK(rel_err(r.kernel, ref_matmul(xd, lw.deq, &lb.f, m, n, k)) < 1e-5);
  }
  std::printf("  single matmul, %zu shapes: bit-identical to GraphExecutor\n", std::size(cases));
}

void test_chain_quantized_and_shared_input() {
  // h = x . w1^T + b1 (f32); y = h . w2^T (Q8_0, canonicalized to quantized_matmul);
  // z = x . w3^T (shares x with the first op).
  // M: above the int8 row limit (4; 64 on ARM64), so both layers run fp32.
  const int64_t m = std::max<int64_t>(5, KernelPlan::defaults().int8_decode_max_rows + 1), k = 64, hdim = 96, n = 40,
                n3 = 24;
  gir::Graph g("chain");
  gir::Builder b(g);
  const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::symbol("M"), k));
  const gir::ValueId w1 = b.weight("w1", gir::weight_type(DType::kF32, {gir::Dim::of(hdim), gir::Dim::of(k)}));
  const gir::ValueId b1 = b.weight("b1", gir::weight_type(DType::kF32, {gir::Dim::of(hdim)}));
  const gir::ValueId w2 = b.weight("w2", gir::weight_type(DType::kQ8_0, {gir::Dim::of(n), gir::Dim::of(hdim)}));
  const gir::ValueId w3 = b.weight("w3", gir::weight_type(DType::kF32, {gir::Dim::of(n3), gir::Dim::of(k)}));
  const gir::ValueId h = b.matmul(x, w1, b1);
  const gir::ValueId y = b.matmul(h, w2);
  const gir::ValueId z = b.matmul(x, w3);
  CHECK(b.status().ok());
  CHECK(gir::canonicalize(g) == 1);  // w2 is quantized: matmul -> quantized_matmul
  CHECK(g.op(g.value(y).producer).kind == gir::OpKind::kQuantizedMatMul);

  Leaf lx = f32_leaf(m, k, 4), l1 = f32_leaf(hdim, k, 5), lb = bias_leaf(hdim, 6), l2 = q8_leaf(n, hdim, 7),
       l3 = f32_leaf(n3, k, 8);
  const std::map<gir::ValueId, TensorView> leaves = {{x, lx.view}, {w1, l1.view}, {b1, lb.view}, {w2, l2.view},
                                                     {w3, l3.view}};
  // Lowered structure: 3 kernel ops in graph order, x mapped once, h a result feeding op 1.
  LowerOptions lo;
  lo.symbols = {{"M", m}};
  const LowerResult lr = dk::lower(g, lo);
  CHECK(lr.ok());
  const dk::Program& p = lr.lowered.program;
  CHECK(p.ops().size() == 3);
  CHECK(p.values().size() == 8);  // x w1 b1 h w2 y w3 z
  const dk::ValueId kx = lr.lowered.kernel_value(x), kh = lr.lowered.kernel_value(h);
  CHECK(p.ops()[0].inputs[0] == kx && p.ops()[2].inputs[0] == kx);
  CHECK(p.ops()[0].output == kh && p.ops()[1].inputs[0] == kh);
  CHECK(p.value(kh).role == dk::ValueRole::kResult && p.value(kx).role == dk::ValueRole::kActivation);
  CHECK(p.value(lr.lowered.kernel_value(w2)).layout.dtype == DType::kQ8_0);
  CHECK(lr.lowered.ops.at(g.value(y).producer) == 1);
  CHECK(std::get<dk::MatMulAttrs>(p.ops()[1].attrs).k == hdim);

  // Values: both executions over the same leaves.
  const Both ry = run_both(g, leaves, y, m, n, lo.symbols);
  const Both rz = run_both(g, leaves, z, m, n3, lo.symbols);
  CHECK(ry.executor == ry.kernel && rz.executor == rz.kernel);
  // Reference: h in f32 (as stored between ops), then Q8_0-decoded w2. M rows:
  // above the int8 row limit, so both layers run fp32.
  std::vector<double> xd(lx.f.begin(), lx.f.end());
  const std::vector<double> hd = ref_matmul(xd, l1.deq, &lb.f, m, hdim, k);
  std::vector<double> hf(hd.size());
  for (size_t i = 0; i < hd.size(); ++i) hf[i] = static_cast<float>(hd[i]);
  CHECK(rel_err(ry.kernel, ref_matmul(hf, l2.deq, nullptr, m, n, hdim)) < 1e-5);
  CHECK(rel_err(rz.kernel, ref_matmul(xd, l3.deq, nullptr, m, n3, k)) < 1e-5);
  std::printf("  chain + Q8_0 + shared input: structure ok, bit-identical to GraphExecutor\n");
}

void test_small_rows_int8_path() {
  // M = 2 with a Q8_0 weight: on AVX2 both layers take the int8 activation
  // path (auto). Still bit-identical, because both run the same kernel.
  gir::Graph g("decode");
  gir::Builder b(g);
  const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::of(2), 128));
  const gir::ValueId w = b.weight("w", gir::weight_type(DType::kQ8_0, {gir::Dim::of(48), gir::Dim::of(128)}));
  const gir::ValueId y = b.matmul(x, w);
  CHECK(b.status().ok());
  Leaf lx = f32_leaf(2, 128, 9), lw = q8_leaf(48, 128, 10);
  const Both r = run_both(g, {{x, lx.view}, {w, lw.view}}, y, 2, 48, {});
  CHECK(r.executor == r.kernel);
  std::vector<double> xd(lx.f.begin(), lx.f.end());
  CHECK(rel_err(r.kernel, ref_matmul(xd, lw.deq, nullptr, 2, 48, 128)) < 2e-2);  // int8 activations, DD-053
  // The comparison is sensitive: an f32 activation contract changes the bits
  // exactly when the int8 path exists on this tier.
  const Both f = run_both(g, {{x, lx.view}, {w, lw.view}}, y, 2, 48, {}, ActFormat::kF32);
  ThreadPool probe_pool(1);  // must outlive the backend
  ReferenceBackend probe(probe_pool, this_isa());
  CHECK((f.executor == f.kernel) == !probe.target().int8_dot);
  CHECK(rel_err(f.kernel, ref_matmul(xd, lw.deq, nullptr, 2, 48, 128)) < 1e-5);
  std::printf("  M=2 Q8_0 auto: bit-identical to GraphExecutor (int8_dot=%d; f32 contract %s)\n",
              probe.target().int8_dot ? 1 : 0, f.executor == f.kernel ? "same bits" : "different bits");
}

// --- rejections ---------------------------------------------------------------------------

void test_rejections() {
  {  // Unsupported op: nothing is lowered (no partial program), status kUnsupported.
    gir::Graph g("norm");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::of(2), 8));
    const gir::ValueId nw = b.weight("nw", gir::weight_type(DType::kF32, {gir::Dim::of(8)}));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}));
    b.matmul(b.rms_norm(x, nw, 1e-5), w);
    CHECK(b.status().ok());
    const LowerResult r = dk::lower(g);
    CHECK(!r.ok() && r.status().code() == StatusCode::kUnsupported);
    CHECK(has_issue(r, LowerIssueKind::kUnsupportedOp, "graph op 3 (rmsnorm): no DynaKernel lowering for 'rmsnorm'"));
    CHECK(has_issue(r, LowerIssueKind::kUnsupportedType, "graph op 4 (matmul): x %3 comes from 'rmsnorm'"));
    CHECK(r.lowered.program.ops().empty() && r.lowered.values.empty());
    CHECK(r.issues[0].graph_op >= 0);
  }
  {  // f16 activations: representable in the graph, not in the kernel IR.
    gir::Graph g("half");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", gir::tensor_type(DType::kF16, {gir::Dim::of(2), gir::Dim::of(8)}));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}));
    b.matmul(x, w);
    CHECK(b.status().ok());
    const LowerResult r = dk::lower(g);
    CHECK(r.status().code() == StatusCode::kUnsupported);
    CHECK(has_issue(r, LowerIssueKind::kUnsupportedType, "x %0 'x' is f16; DynaKernel matmul takes f32 activations"));
  }
  {  // Non-f32 bias.
    gir::Graph g("hbias");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::of(2), 8));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}));
    const gir::ValueId bias = b.weight("b", gir::weight_type(DType::kF16, {gir::Dim::of(4)}));
    b.matmul(x, w, bias);
    CHECK(b.status().ok());
    CHECK(has_issue(dk::lower(g), LowerIssueKind::kUnsupportedType, "the kernel adds f32 bias"));
  }
  {  // Right-hand side computed by an earlier op: not a kernel weight.
    gir::Graph g("xw");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::of(8), 8));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(8), gir::Dim::of(8)}));
    const gir::ValueId h = b.matmul(x, w);
    b.matmul(x, h);
    CHECK(b.status().ok());
    CHECK(has_issue(dk::lower(g), LowerIssueKind::kUnsupportedType, "must be a weight leaf"));
  }
  {  // Unbound symbol: an error, never a default of 1.
    gir::Graph g("sym");
    gir::Builder b(g);
    const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::symbol("M"), 8));
    const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}));
    b.matmul(x, w);
    const LowerResult r = dk::lower(g);
    CHECK(r.status().code() == StatusCode::kInvalidArgument);
    CHECK(has_issue(r, LowerIssueKind::kUnboundSymbol, "dimension 'M' of %0 'x' has no value"));
    LowerOptions bad;
    bad.symbols = {{"M", 0}};
    CHECK(has_issue(dk::lower(g, bad), LowerIssueKind::kUnboundSymbol, "symbol 'M' is 0"));
  }
  {  // Invalid graph (result type that infer_type disagrees with): ir::verify's message is kept.
    gir::Graph g("bad");
    const gir::ValueId x = g.add_leaf(gir::OpKind::kInput, f32_tensor(gir::Dim::of(2), 8), "x");
    const gir::ValueId w =
        g.add_leaf(gir::OpKind::kWeight, gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}), "w");
    g.add_op(gir::OpKind::kMatMul, {x, w}, f32_tensor(gir::Dim::of(2), 5));  // should be [2, 4]
    const LowerResult r = dk::lower(g);
    CHECK(r.status().code() == StatusCode::kInvalidArgument);
    CHECK(!r.issues.empty() && r.issues[0].kind == LowerIssueKind::kInvalidGraph);
    CHECK(r.lowered.program.ops().empty());
  }
  {  // A layout DynaKernel cannot represent (column-major input).
    gir::Graph g("colmajor");
    gir::Type t = f32_tensor(gir::Dim::of(2), 8);
    t.layout = gir::Layout{gir::LayoutKind::kColMajor, 0};
    const gir::ValueId x = g.add_leaf(gir::OpKind::kInput, t, "x");
    const gir::ValueId w =
        g.add_leaf(gir::OpKind::kWeight, gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}), "w");
    g.add_op(gir::OpKind::kMatMul, {x, w}, f32_tensor(gir::Dim::of(2), 4));
    const LowerResult r = dk::lower(g);
    if (r.issues.size() && r.issues[0].kind == LowerIssueKind::kInvalidGraph) {
      std::printf("    (graph verifier rejects a col-major matmul operand first: %s)\n", r.issues[0].message.c_str());
    } else {
      CHECK(has_issue(r, LowerIssueKind::kUnsupportedType, "has layout col_major(0)"));
    }
    CHECK(!r.ok() && r.lowered.program.ops().empty());
  }
  std::printf("  rejections: ok\n");
}

void test_layouts_and_unused_leaves() {
  // A strided result (written into rows 16 floats apart) maps to exact byte
  // strides; unused leaves (an index array, a KV cache) are not lowered and
  // need no binding. GraphExecutor allocates every value densely, so strided
  // execution is compared with the double reference, not with the executor.
  gir::Graph g("strided");
  const gir::ValueId pos = g.add_leaf(gir::OpKind::kInput, gir::index_type({gir::Dim::of(3)}), "pos");
  const gir::ValueId kv = g.add_leaf(gir::OpKind::kKvCache, gir::kv_type(DType::kF16, 4, 2, 16, 8), "cache");
  const gir::ValueId x = g.add_leaf(gir::OpKind::kInput, f32_tensor(gir::Dim::of(3), 32), "x");
  const gir::ValueId w =
      g.add_leaf(gir::OpKind::kWeight, gir::weight_type(DType::kQ8_0, {gir::Dim::of(10), gir::Dim::of(32)}), "w");
  gir::Type yt = f32_tensor(gir::Dim::of(3), 10);
  yt.layout = gir::Layout{gir::LayoutKind::kStrided, 16};
  const gir::ValueId y = g.add_op(gir::OpKind::kQuantizedMatMul, {x, w}, yt);
  CHECK(gir::verify(g).ok());
  const LowerResult r = dk::lower(g);
  if (!r.ok()) std::fprintf(stderr, "%s\n", r.status().to_string().c_str());
  CHECK(r.ok());
  CHECK(r.lowered.kernel_value(pos) == dk::kNoValue && r.lowered.kernel_value(kv) == dk::kNoValue);
  const dk::Value& ky = r.lowered.program.value(r.lowered.kernel_value(y));
  CHECK(ky.layout.strides[0] == 16 * 4 && ky.layout.strides[1] == 4);
  CHECK(r.lowered.program.value(r.lowered.kernel_value(w)).layout.is_contiguous());

  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  Leaf lx = f32_leaf(3, 32, 11), lw = q8_leaf(10, 32, 12);
  std::vector<float> out(3 * 16, 77.0f);
  Bindings bind;
  bind.bind(r.lowered.kernel_value(x), lx.view);
  bind.bind(r.lowered.kernel_value(w), lw.view);
  bind.bind(r.lowered.kernel_value(y), TensorView(out.data(), ky.layout), static_cast<int64_t>(out.size() * 4));
  CHECK(be.run(r.lowered.program, bind).ok());
  std::vector<double> xd(lx.f.begin(), lx.f.end());
  const std::vector<double> ref = ref_matmul(xd, lw.deq, nullptr, 3, 10, 32);
  double mr = 0, me = 0;
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 10; ++j) {
      mr = std::max(mr, std::abs(ref[static_cast<size_t>(i * 10 + j)]));
      me = std::max(me, std::abs(out[static_cast<size_t>(i * 16 + j)] - ref[static_cast<size_t>(i * 10 + j)]));
    }
    for (int64_t j = 10; j < 16; ++j) CHECK(out[static_cast<size_t>(i * 16 + j)] == 77.0f);
  }
  CHECK(me / (mr + 1e-6) < 2e-2);  // M = 3 may take the int8 path on AVX2
  std::printf("  strided result + unused leaves: ok\n");
}

void test_bindings_after_lowering() {
  // Errors at execution leave every result buffer untouched.
  gir::Graph g("bind");
  gir::Builder b(g);
  const gir::ValueId x = b.input("x", f32_tensor(gir::Dim::of(2), 8));
  const gir::ValueId w = b.weight("w", gir::weight_type(DType::kF32, {gir::Dim::of(4), gir::Dim::of(8)}));
  const gir::ValueId y = b.matmul(x, w);
  const LowerResult r = dk::lower(g);
  CHECK(r.ok());
  ThreadPool pool(1);
  ReferenceBackend be(pool, this_isa());
  Leaf lx = f32_leaf(2, 8, 13), lw = f32_leaf(4, 8, 14);
  std::vector<float> out(8, -3.0f);
  const dk::ValueId kx = r.lowered.kernel_value(x), kw = r.lowered.kernel_value(w), ky = r.lowered.kernel_value(y);
  auto untouched = [&] {
    for (float v : out) {
      if (v != -3.0f) return false;
    }
    return true;
  };
  {
    Bindings bind;  // y missing
    bind.bind(kx, lx.view);
    bind.bind(kw, lw.view);
    CHECK(contains(be.run(r.lowered.program, bind).message(), "is not bound") && untouched());
  }
  {
    Bindings bind;  // weight bound with the wrong shape (graph semantics: [N, K] = [4, 8])
    bind.bind(kx, lx.view);
    bind.bind(kw, dense_view(lw.f.data(), DType::kF32, {8, 4}));
    bind.bind(ky, dense_view(out.data(), DType::kF32, {2, 4}));
    CHECK(contains(be.run(r.lowered.program, bind).message(), "does not match the declared") && untouched());
  }
  {
    Bindings bind;  // result buffer too small
    bind.bind(kx, lx.view);
    bind.bind(kw, lw.view);
    bind.bind(ky, dense_view(out.data(), DType::kF32, {2, 4}), 7 * 4);
    CHECK(contains(be.run(r.lowered.program, bind).message(), "but the buffer has 28") && untouched());
  }
  {
    Bindings bind;
    bind.bind(kx, lx.view);
    bind.bind(kw, lw.view);
    bind.bind(ky, dense_view(out.data(), DType::kF32, {2, 4}), 8 * 4);
    CHECK(be.run(r.lowered.program, bind).ok() && !untouched());
  }
  {  // f32 activation contract passes through lowering to every op.
    LowerOptions lo;
    lo.act = ActFormat::kF32;
    const LowerResult r32 = dk::lower(g, lo);
    CHECK(r32.ok() && std::get<dk::MatMulAttrs>(r32.lowered.program.ops()[0].attrs).act == ActFormat::kF32);
  }
  std::printf("  bindings after lowering: ok\n");
}

// A graph compiled from DynaCore language source (dynacore/lang), with Q4_K,
// Q6_K and Q8_0 weights. GraphExecutor fills every leaf (random activations,
// a valid constant byte pattern for quantized weights); the kernel program
// is bound to those same buffers, so only the lowering differs.
void test_language_graph() {
  constexpr const char* kSrc = R"(
graph proj(M = 3) {
  config hidden = 256, ff = 512
  input  x  : tensor<f32>[M, hidden]
  weight wq : q4_K[hidden, hidden]
  weight bq : f32[hidden]
  weight wu : q8_0[ff, hidden]
  weight wd : q6_K[hidden, ff]
  q = x @ wq + bq
  y = (q @ wu) @ wd
  output y
}
graph mlp(M = 2) {
  config hidden = 256, ff = 512
  input  x      : tensor<f32>[M, hidden]
  weight w_gate : q4_K[ff, hidden]
  weight w_up   : q4_K[ff, hidden]
  weight w_down : q6_K[hidden, ff]
  y = swiglu(x @ w_gate, x @ w_up) @ w_down
  output y
}
)";
  for (int64_t m : {1, 2, 3, 9}) {
    auto prog = lang::compile(kSrc, "<test>", "proj");
    if (!prog.ok()) std::fprintf(stderr, "%s\n", prog.status().to_string().c_str());
    CHECK(prog.ok());
    const gir::Graph& g = prog->graph;
    const gir::ValueId y = prog->outputs.at(0);
    const std::map<std::string, int64_t> symbols = {{"M", m}};
    ThreadPool pool(4);
    CpuDevice dev(pool, this_isa());
    gir::ExecutorOptions eo;
    eo.symbols = symbols;
    auto ex = gir::GraphExecutor::create(g, dev, eo);
    CHECK(ex.ok());
    CHECK((*ex)->run().ok());

    LowerOptions lo;
    lo.symbols = symbols;
    const LowerResult lr = dk::lower(g, lo);
    if (!lr.ok()) std::fprintf(stderr, "%s\n", lr.status().to_string().c_str());
    CHECK(lr.ok());
    CHECK(lr.lowered.program.ops().size() == 3);
    bool q4 = false, q6 = false;
    for (const dk::Value& v : lr.lowered.program.values()) {
      q4 = q4 || v.layout.dtype == DType::kQ4_K;
      q6 = q6 || v.layout.dtype == DType::kQ6_K;
    }
    CHECK(q4 && q6);
    ReferenceBackend be(pool, this_isa());
    Bindings bind;
    std::vector<std::vector<float>> results;
    results.reserve(lr.lowered.values.size());
    for (const auto& [gv, kv] : lr.lowered.values) {
      const dk::Value& v = lr.lowered.program.value(kv);
      if (v.role == dk::ValueRole::kResult) {
        results.emplace_back(static_cast<size_t>(v.layout.span_bytes() / 4), std::nanf(""));
        bind.bind(kv, TensorView(results.back().data(), v.layout), v.layout.span_bytes());
      } else {
        bind.bind(kv, (*ex)->view(gv));  // the executor's own leaf storage
      }
    }
    CHECK(be.run(lr.lowered.program, bind).ok());
    const TensorView ev = (*ex)->view(y);
    const TensorView* kv = bind.find(lr.lowered.kernel_value(y));
    CHECK(kv != nullptr && ev.numel() == kv->numel());
    CHECK(std::memcmp(ev.data(), kv->data(), static_cast<size_t>(ev.numel()) * 4) == 0);
    bool finite = true;
    for (int64_t i = 0; i < kv->numel(); ++i) finite = finite && std::isfinite(kv->data_as<const float>()[i]);
    CHECK(finite);
  }
  {  // The gated MLP form needs act_mul: rejected as a whole, not partially lowered.
    auto prog = lang::compile(kSrc, "<test>", "mlp");
    CHECK(prog.ok());
    LowerOptions lo;
    lo.symbols = {{"M", 2}};
    const LowerResult r = dk::lower(prog->graph, lo);
    CHECK(r.status().code() == StatusCode::kUnsupported);
    CHECK(has_issue(r, LowerIssueKind::kUnsupportedOp, "no DynaKernel lowering for 'act_mul'"));
    CHECK(r.lowered.program.ops().empty());
  }
  std::printf("  language graph (Q4_K, Q6_K, Q8_0): bit-identical to GraphExecutor; swiglu rejected\n");
}

}  // namespace

int main() {
  std::printf("test_dynakernel_lowering (isa %s)\n", isa_name(this_isa()).data());
  test_single_matmul_shapes();
  test_chain_quantized_and_shared_input();
  test_small_rows_int8_path();
  test_rejections();
  test_layouts_and_unused_leaves();
  test_bindings_after_lowering();
  test_language_graph();
  std::printf("all lowering tests passed (%d checks)\n", g_checks);
  return 0;
}

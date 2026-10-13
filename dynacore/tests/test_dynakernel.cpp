// DynaKernel IR v0: construction, verification, reference execution.
// Self-contained: no model files, no server, synthetic data only.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/program.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/quantization/quant_formats.h"

using namespace dynacore;
using namespace dynacore::dk;
using namespace dynacore::quant;  // GGUF block structs and quantize_q8_0 / dequantize_q8_0

namespace {

int g_checks = 0;

#define CHECK(cond)                                                               \
  do {                                                                            \
    ++g_checks;                                                                   \
    if (!(cond)) {                                                                \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

bool contains(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

// Some diagnostic contains `needle`.
bool has_diag(const Program& p, const std::string& needle) {
  for (const Diagnostic& d : verify_all(p)) {
    if (contains(d.message, needle)) return true;
  }
  return false;
}

std::vector<float> random_floats(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>((s >> 8) & 0xffff) / 32768.0f - 1.0f;  // [-1, 1)
  }
  return v;
}

TensorView view_of(void* data, DType dtype, int64_t rows, int64_t cols) {
  auto l = TensorLayout::contiguous(dtype, TensorShape{rows, cols});
  CHECK(l.ok());
  return TensorView(data, *l);
}

TensorView vec_view(float* data, int64_t n) {
  auto l = TensorLayout::contiguous(DType::kF32, TensorShape{n});
  CHECK(l.ok());
  return TensorView(data, *l);
}

// y = x . w^T (+ b) in double precision, w given as dense fp32 rows.
std::vector<double> reference(const std::vector<float>& x, const std::vector<float>& w, const float* b, int64_t m,
                              int64_t n, int64_t k) {
  std::vector<double> y(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = b ? b[j] : 0.0;
      for (int64_t t = 0; t < k; ++t) acc += static_cast<double>(x[i * k + t]) * w[j * k + t];
      y[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return y;
}

// max |a - b| / (max |b| + 1e-6) over a [m, n] result read through row stride `ld`.
double rel_error(const float* y, int64_t ld, const std::vector<double>& ref, int64_t m, int64_t n) {
  double max_ref = 0, max_err = 0;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const double r = ref[static_cast<size_t>(i * n + j)];
      max_ref = std::max(max_ref, std::abs(r));
      max_err = std::max(max_err, std::abs(static_cast<double>(y[i * ld + j]) - r));
    }
  }
  return max_err / (max_ref + 1e-6);
}

CpuIsa this_isa() { return select_isa(cpu_info().features); }

// --- construction --------------------------------------------------------------------

void test_construction() {
  ProgramBuilder b;
  const ValueId x = b.activation("x", DType::kF32, 2, 4);
  const ValueId w1 = b.weight("w1", DType::kF32, 3, 4);
  const ValueId w2 = b.weight("w2", DType::kF32, 5, 3);
  const ValueId bias = b.bias("b2", 5);
  const ValueId h = b.matmul(x, w1, kNoValue, {.name = "up"});
  const ValueId y = b.matmul(h, w2, bias, {.act = ActFormat::kF32, .name = "down"});
  auto p = b.finish();
  CHECK(p.ok());
  CHECK(p->ops().size() == 2);
  CHECK(p->values().size() == 6);
  const Op& up = p->ops()[0];
  const Op& down = p->ops()[1];
  CHECK(up.inputs.size() == 2 && up.inputs[0] == x && up.inputs[1] == w1 && up.output == h);
  CHECK(down.inputs.size() == 3 && down.inputs[0] == h && down.inputs[2] == bias && down.output == y);
  const auto& a = std::get<MatMulAttrs>(down.attrs);
  CHECK(a.m == 2 && a.n == 5 && a.k == 3 && a.act == ActFormat::kF32);
  CHECK(p->value(h).role == ValueRole::kResult && p->value(h).layout.shape == (TensorShape{2, 3}));
  // Deterministic text form.
  const std::string t1 = to_text(*p), t2 = to_text(*p);
  CHECK(t1 == t2);
  CHECK(contains(t1, "matmul %4 <- %0 %1 {m=2 n=3 k=4 accum=f32 act=auto} \"up\""));
  CHECK(contains(t1, "matmul %5 <- %4 %2 %3 {m=2 n=5 k=3 accum=f32 act=f32} \"down\""));
  CHECK(contains(t1, "%0 = activation f32[2, 4] strides(16,4) \"x\""));
  std::printf("  construction: ok\n");
}

// --- verification --------------------------------------------------------------------

// A valid two-value program skeleton the cases below break in one place.
struct Fixture {
  Program p;
  ValueId x, w, y;
  explicit Fixture(int64_t m = 2, int64_t n = 3, int64_t k = 4, DType wt = DType::kF32) {
    ProgramBuilder b;
    x = b.activation("x", DType::kF32, m, k);
    w = b.weight("w", wt, n, k);
    y = b.matmul(x, w);
    auto r = b.finish();
    CHECK(r.ok());
    p = std::move(*r);
  }
};

Op matmul_op(ValueId x, ValueId w, ValueId y, int64_t m, int64_t n, int64_t k) {
  Op op;
  op.kind = OpKind::kMatMul;
  op.inputs = {x, w};
  op.output = y;
  op.attrs = MatMulAttrs{m, n, k};
  return op;
}

void test_verifier() {
  CHECK(verify(Fixture().p).ok());

  {  // K mismatch, through the builder: finish() refuses.
    ProgramBuilder b;
    const ValueId x = b.activation("x", DType::kF32, 2, 64);
    const ValueId w = b.weight("w", DType::kF32, 8, 128);
    b.matmul(x, w, kNoValue, {.name = "q"});
    auto r = b.finish();
    CHECK(!r.ok());
    CHECK(r.status().code() == StatusCode::kInvalidArgument);
    CHECK(contains(r.status().message(), "op 0 (matmul 'q'): x has 64 columns but w has 128 (K mismatch)"));
  }
  {  // Undefined operand.
    Fixture f;
    Program p = f.p;
    const ValueId y2 = p.add_value(p.value(f.y));
    p.add_op(matmul_op(99, f.w, y2, 2, 3, 4));
    CHECK(has_diag(p, "op 1 (matmul): operand 0 references undefined value %99"));
    CHECK(!verify(p).ok());
  }
  {  // Undefined output.
    Fixture f;
    Program p = f.p;
    p.add_op(matmul_op(f.x, f.w, 42, 2, 3, 4));
    CHECK(has_diag(p, "output references undefined value %42"));
  }
  {  // Output written twice.
    Fixture f;
    Program p = f.p;
    p.add_op(matmul_op(f.x, f.w, f.y, 2, 3, 4));
    CHECK(has_diag(p, "is already written by op 0"));
  }
  {  // Result read before it is written (op order is execution order).
    ProgramBuilder b;
    const ValueId x = b.activation("x", DType::kF32, 2, 3);
    const ValueId w = b.weight("w", DType::kF32, 3, 3);
    const ValueId h = b.value("h", ValueRole::kResult, *TensorLayout::contiguous(DType::kF32, TensorShape{2, 3}));
    const ValueId y = b.value("y", ValueRole::kResult, *TensorLayout::contiguous(DType::kF32, TensorShape{2, 3}));
    b.matmul_into(h, w, kNoValue, y);  // reads h ...
    b.matmul_into(x, w, kNoValue, h);  // ... which is only written here
    auto r = b.finish();
    CHECK(!r.ok());
    CHECK(contains(r.status().message(), "is read before any op writes it"));
  }
  {  // Output declared with the wrong role / shape / dtype.
    Fixture f;
    Program p = f.p;
    Value bad = p.value(f.y);
    bad.role = ValueRole::kWeight;
    const ValueId wy = p.add_value(bad);
    p.add_op(matmul_op(f.x, f.w, wy, 2, 3, 4));
    CHECK(has_diag(p, "must be a result value"));

    Program q = f.p;
    Value wrong = q.value(f.y);
    wrong.layout = *TensorLayout::contiguous(DType::kF32, TensorShape{2, 5});
    const ValueId y5 = q.add_value(wrong);
    q.add_op(matmul_op(f.x, f.w, y5, 2, 3, 4));
    CHECK(has_diag(q, "output is [2, 5], expected [2, 3]"));

    Program h = f.p;
    Value half = h.value(f.y);
    half.layout = *TensorLayout::contiguous(DType::kF16, TensorShape{2, 3});
    const ValueId yh = h.add_value(half);
    h.add_op(matmul_op(f.x, f.w, yh, 2, 3, 4));
    CHECK(has_diag(h, "output must be f32, is f16"));
  }
  {  // Result never written.
    Fixture f;
    Program p = f.p;
    p.add_value(p.value(f.y));
    CHECK(has_diag(p, "result is never written"));
  }
  {  // Attributes that disagree with operands are not repaired.
    Fixture f;
    Program p;
    for (const Value& v : f.p.values()) p.add_value(v);
    p.add_op(matmul_op(f.x, f.w, f.y, 2, 3, 8));
    CHECK(has_diag(p, "attributes M,N,K = 2,3,8 disagree with operands 2,3,4"));
  }
  {  // Unsupported op kind: kUnsupported, not kInvalidArgument.
    Fixture f;
    Program p = f.p;
    Op att;
    att.kind = OpKind::kAttention;
    att.name = "attn";
    att.inputs = {f.x};
    att.output = p.add_value(p.value(f.y));
    p.add_op(att);
    const Status s = verify(p);
    CHECK(s.code() == StatusCode::kUnsupported);
    CHECK(contains(s.message(), "op 1 (attention 'attn'): op kind is declared but not defined in DynaKernel v0"));
  }
  {  // Quantized K not a whole number of blocks.
    ProgramBuilder b;
    TensorLayout l;
    l.dtype = DType::kQ8_0;
    l.shape = TensorShape{4, 40};  // Q8_0 blocks hold 32 values
    const ValueId x = b.activation("x", DType::kF32, 1, 40);
    const ValueId w = b.value("w", ValueRole::kWeight, l);
    b.matmul(x, w);
    auto r = b.finish();
    CHECK(!r.ok());
    CHECK(contains(r.status().message(), "not a multiple of the q8_0 block size 32"));
  }
  {  // Layouts: non-contiguous innermost, overlapping rows, strided weight, bad alignment.
    TensorLayout l = *TensorLayout::contiguous(DType::kF32, TensorShape{2, 4});
    l.strides[1] = 8;
    Fixture f;
    Program p = f.p;
    p.add_value(Value{kNoValue, "gap", ValueRole::kActivation, l, 0});
    CHECK(has_diag(p, "innermost stride 8 (must be 4"));

    TensorLayout o = *TensorLayout::contiguous(DType::kF32, TensorShape{2, 4});
    o.strides[0] = 8;
    Program q = f.p;
    q.add_value(Value{kNoValue, "ov", ValueRole::kActivation, o, 0});
    CHECK(has_diag(q, "overlaps the previous one (needs >= 16)"));

    TensorLayout ws = *TensorLayout::contiguous(DType::kF32, TensorShape{3, 4});
    ws.strides[0] = 32;  // rows padded: a valid view, but weights must be dense
    ProgramBuilder b;
    const ValueId x = b.activation("x", DType::kF32, 2, 4);
    const ValueId w = b.value("w", ValueRole::kWeight, ws);
    b.matmul(x, w);
    auto r = b.finish();
    CHECK(!r.ok() && contains(r.status().message(), "w must be dense row-major"));

    Program a = f.p;
    a.add_value(Value{kNoValue, "al", ValueRole::kActivation, *TensorLayout::contiguous(DType::kF32, TensorShape{1, 4}), 24});
    CHECK(has_diag(a, "alignment 24 is not a power of two"));
  }
  {  // Wrong operand roles and dtypes; bias shape; a result used as its own operand.
    ProgramBuilder b;
    const ValueId x = b.activation("x", DType::kF16, 2, 4);
    const ValueId w = b.activation("w", DType::kF32, 3, 4);  // not a weight
    const ValueId bias = b.bias("b", 4);                      // should be 3
    b.matmul(x, w, bias);
    auto r = b.finish();
    CHECK(!r.ok());
    // The same program built raw, to inspect every diagnostic.
    Program raw;
    const ValueId rx = raw.add_value(Value{kNoValue, "x", ValueRole::kActivation,
                                           *TensorLayout::contiguous(DType::kF16, TensorShape{2, 4}), 0});
    const ValueId rw = raw.add_value(Value{kNoValue, "w", ValueRole::kActivation,
                                           *TensorLayout::contiguous(DType::kF32, TensorShape{3, 4}), 0});
    const ValueId rb = raw.add_value(Value{kNoValue, "b", ValueRole::kBias,
                                           *TensorLayout::contiguous(DType::kF32, TensorShape{4}), 0});
    const ValueId ry = raw.add_value(Value{kNoValue, "y", ValueRole::kResult,
                                           *TensorLayout::contiguous(DType::kF32, TensorShape{2, 3}), 0});
    Op op = matmul_op(rx, rw, ry, 2, 3, 4);
    op.inputs.push_back(rb);
    raw.add_op(op);
    CHECK(has_diag(raw, "x must be f32, is f16"));
    CHECK(has_diag(raw, "w %1 'w' must be a weight"));
    CHECK(has_diag(raw, "bias must be a dense f32 [3] vector"));

    Fixture f;
    Program ip = f.p;
    const ValueId y2 = ip.add_value(ip.value(f.y));
    ip.add_op(matmul_op(y2, f.w, y2, 2, 3, 3));
    CHECK(has_diag(ip, "is read before any op writes it"));
  }
  std::printf("  verifier: ok\n");
}

// --- execution --------------------------------------------------------------------------

void test_known_fp32() {
  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  // [[1,2,3],[4,5,6]] . [[1,0,1],[0,1,0]]^T + [10, 20] = [[14, 22], [20, 25]]
  std::vector<float> x = {1, 2, 3, 4, 5, 6}, w = {1, 0, 1, 0, 1, 0}, bias = {10, 20}, y(4, -1.0f);
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, 2, 3);
  const ValueId vw = b.weight("w", DType::kF32, 2, 3);
  const ValueId vb = b.bias("b", 2);
  const ValueId vy = b.matmul(vx, vw, vb);
  auto p = b.finish();
  CHECK(p.ok());
  CHECK(be.check(*p).ok());
  Bindings bind;
  bind.bind(vx, view_of(x.data(), DType::kF32, 2, 3));
  bind.bind(vw, view_of(w.data(), DType::kF32, 2, 3));
  bind.bind(vb, vec_view(bias.data(), 2));
  bind.bind(vy, view_of(y.data(), DType::kF32, 2, 2));
  const Status s = be.run(*p, bind);
  CHECK(s.ok());
  CHECK(y[0] == 14 && y[1] == 22 && y[2] == 20 && y[3] == 25);
  std::printf("  known fp32 product: ok (%s, %d threads)\n", be.target().isa.c_str(), be.target().threads);
}

void run_shape(ReferenceBackend& be, int64_t m, int64_t n, int64_t k, ActFormat act, DType wt, double tol) {
  const std::vector<float> x = random_floats(static_cast<size_t>(m * k), static_cast<uint32_t>(m * 131 + k));
  const std::vector<float> wf = random_floats(static_cast<size_t>(n * k), static_cast<uint32_t>(n * 17 + k));
  const std::vector<float> bias = random_floats(static_cast<size_t>(n), 7);
  // Weight storage in its real format; the reference uses exactly the values
  // that storage decodes to (GGUF dequantization), not the pre-quantization floats.
  std::vector<float> wref = wf;
  std::vector<BlockQ8_0> q;
  void* wdata = const_cast<float*>(wf.data());
  if (wt == DType::kQ8_0) {
    q.resize(static_cast<size_t>(n * k / 32));
    quantize_q8_0(wf.data(), q.data(), static_cast<int64_t>(q.size()));
    dequantize_q8_0(q.data(), wref.data(), static_cast<int64_t>(q.size()));
    wdata = q.data();
  }
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, m, k);
  const ValueId vw = b.weight("w", wt, n, k);
  const ValueId vb = b.bias("b", n);
  const ValueId vy = b.matmul(vx, vw, vb, {.act = act});
  auto p = b.finish();
  CHECK(p.ok());
  std::vector<float> y(static_cast<size_t>(m * n), std::nanf(""));
  Bindings bind;
  bind.bind(vx, view_of(const_cast<float*>(x.data()), DType::kF32, m, k));
  bind.bind(vw, view_of(wdata, wt, n, k));
  bind.bind(vb, vec_view(const_cast<float*>(bias.data()), n));
  bind.bind(vy, view_of(y.data(), DType::kF32, m, n));
  CHECK(be.run(*p, bind).ok());
  const double err = rel_error(y.data(), n, reference(x, wref, bias.data(), m, n, k), m, n);
  if (!(err <= tol)) {
    std::fprintf(stderr, "  M=%lld N=%lld K=%lld %s act=%s: rel error %.3g > %.3g\n", static_cast<long long>(m),
                 static_cast<long long>(n), static_cast<long long>(k), dtype_name(wt).data(),
                 act_format_name(act).data(), err, tol);
  }
  CHECK(err <= tol);
}

void test_shapes() {
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  // fp32 weights: fused dot (M = 1) and fp32 panel GEMM (M >= 2), square and not.
  const int64_t shapes[][3] = {{1, 1, 1}, {1, 7, 5}, {3, 5, 7}, {4, 64, 32}, {5, 33, 96}, {17, 40, 64}, {33, 70, 300}};
  for (const auto& s : shapes) {
    run_shape(be, s[0], s[1], s[2], ActFormat::kAuto, DType::kF32, 1e-5);
    run_shape(be, s[0], s[1], s[2], ActFormat::kF32, DType::kF32, 1e-5);
  }
  std::printf("  fp32 shapes: ok (%zu shapes x 2 formats)\n", std::size(shapes));
}

void test_q8_0() {
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  // GGUF Q8_0 weights (quantize_q8_0 + dequantize_q8_0, the loader's own block
  // format). fp32 activations: the error vs the dequantized weights is
  // accumulation rounding only.
  for (int64_t m : {1, 2, 4, 9, 32}) run_shape(be, m, 48, 128, ActFormat::kF32, DType::kQ8_0, 1e-5);
  // auto: rows <= 4 may take the int8 activation path on AVX2 (per-32 int8
  // scales, DD-053), which rounds activations to 8 bits: a documented ~1% bound
  // relative to the largest output here, not a bug.
  for (int64_t m : {1, 2, 4, 9, 32}) run_shape(be, m, 48, 128, ActFormat::kAuto, DType::kQ8_0, 2e-2);
  std::printf("  q8_0 weights: ok (f32 within 1e-5, auto within 2e-2)\n");

  // The contract is observable: with integer-dot kernels, auto at 2 rows takes
  // the int8 path (different bits from f32); without them, both are fp32.
  const int64_t m = 2, n = 32, k = 64;
  const std::vector<float> x = random_floats(m * k, 5), wf = random_floats(n * k, 6);
  std::vector<BlockQ8_0> q(n * k / 32);
  quantize_q8_0(wf.data(), q.data(), static_cast<int64_t>(q.size()));
  auto run_with = [&](ActFormat act) {
    ProgramBuilder b;
    const ValueId vx = b.activation("x", DType::kF32, m, k);
    const ValueId vw = b.weight("w", DType::kQ8_0, n, k);
    const ValueId vy = b.matmul(vx, vw, kNoValue, {.act = act});
    auto p = b.finish();
    CHECK(p.ok());
    std::vector<float> y(m * n, 0.0f);
    Bindings bi;
    bi.bind(vx, view_of(const_cast<float*>(x.data()), DType::kF32, m, k));
    bi.bind(vw, view_of(q.data(), DType::kQ8_0, n, k));
    bi.bind(vy, view_of(y.data(), DType::kF32, m, n));
    CHECK(be.run(*p, bi).ok());
    return y;
  };
  const bool same = run_with(ActFormat::kAuto) == run_with(ActFormat::kF32);
  CHECK(same == !be.target().int8_dot);
  std::printf("  activation contract observable: ok (int8_dot=%d, auto %s f32)\n", be.target().int8_dot ? 1 : 0,
              same ? "==" : "!=");
}

void test_chain_and_strided_output() {
  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  const int64_t m = 3, k = 8, h = 6, n = 4;
  const std::vector<float> x = random_floats(m * k, 1), w1 = random_floats(h * k, 2), w2 = random_floats(n * h, 3);
  // Second result written into columns [2, 6) of a wider [3, 10] buffer.
  std::vector<float> hid(m * h, 0), wide(m * 10, 123.0f);
  TensorLayout yl = *TensorLayout::contiguous(DType::kF32, TensorShape{m, n});
  yl.strides[0] = 10 * sizeof(float);
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, m, k);
  const ValueId v1 = b.weight("w1", DType::kF32, h, k);
  const ValueId v2 = b.weight("w2", DType::kF32, n, h);
  const ValueId vh = b.matmul(vx, v1, kNoValue, {.name = "h"});
  const ValueId vy = b.value("y", ValueRole::kResult, yl);
  b.matmul_into(vh, v2, kNoValue, vy, {.name = "y"});
  auto p = b.finish();
  CHECK(p.ok());
  Bindings bind;
  bind.bind(vx, view_of(const_cast<float*>(x.data()), DType::kF32, m, k));
  bind.bind(v1, view_of(const_cast<float*>(w1.data()), DType::kF32, h, k));
  bind.bind(v2, view_of(const_cast<float*>(w2.data()), DType::kF32, n, h));
  bind.bind(vh, view_of(hid.data(), DType::kF32, m, h));
  bind.bind(vy, TensorView(wide.data() + 2, yl));
  CHECK(be.run(*p, bind).ok());
  const std::vector<double> rh = reference(x, w1, nullptr, m, h, k);
  std::vector<float> hf(rh.size());  // the kernel stores h as f32 before op 2 reads it
  for (size_t i = 0; i < rh.size(); ++i) hf[i] = static_cast<float>(rh[i]);
  const std::vector<double> ry = reference(hf, w2, nullptr, m, n, h);
  CHECK(rel_error(wide.data() + 2, 10, ry, m, n) < 1e-5);
  for (int64_t i = 0; i < m; ++i) {  // columns outside the view untouched
    CHECK(wide[i * 10 + 0] == 123.0f && wide[i * 10 + 1] == 123.0f && wide[i * 10 + 9] == 123.0f);
  }
  std::printf("  two-op chain, strided output: ok\n");
}

void test_backend_rejects() {
  ThreadPool pool(1);
  ReferenceBackend be(pool, this_isa());
  std::vector<float> x(8, 1.0f), w(12, 1.0f), y(6, -7.0f), big(64, 0.0f);
  auto make = [](ActFormat act) {
    ProgramBuilder b;
    const ValueId vx = b.activation("x", DType::kF32, 2, 4, 64);
    const ValueId vw = b.weight("w", DType::kF32, 3, 4);
    b.matmul(vx, vw, kNoValue, {.act = act});
    auto p = b.finish();
    CHECK(p.ok());
    return std::move(*p);
  };
  const Program p = make(ActFormat::kAuto);
  auto good = [&] {
    Bindings bind;
    alignas(64) static float xa[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    bind.bind(0, view_of(xa, DType::kF32, 2, 4));
    bind.bind(1, view_of(w.data(), DType::kF32, 3, 4));
    bind.bind(2, view_of(y.data(), DType::kF32, 2, 3));
    return bind;
  };
  // Every rejection leaves the output untouched: nothing runs.
  auto untouched = [&] { return y[0] == -7.0f && y[5] == -7.0f; };
  {
    Bindings bind = good();
    Bindings missing;
    missing.bind(0, *bind.find(0));
    missing.bind(1, *bind.find(1));
    const Status s = be.run(p, missing);
    CHECK(s.code() == StatusCode::kInvalidArgument);
    CHECK(contains(s.message(), "value %2 is not bound") && untouched());
  }
  {
    Bindings bind = good();
    bind.bind(1, view_of(w.data(), DType::kF32, 4, 3));  // wrong shape
    const Status s = be.run(p, bind);
    CHECK(contains(s.message(), "does not match the declared dtype, shape and strides") && untouched());
  }
  {
    Bindings bind = good();
    bind.bind(0, view_of(reinterpret_cast<float*>(reinterpret_cast<char*>(big.data()) + 4), DType::kF32, 2, 4));
    const Status s = be.run(p, bind);
    CHECK(contains(s.message(), "64-byte aligned") && untouched());
  }
  {
    Bindings bind = good();
    bind.bind(2, view_of(w.data(), DType::kF32, 2, 3));  // result over the weight
    const Status s = be.run(p, bind);
    CHECK(contains(s.message(), "overlaps the buffer of %1"));
  }
  {
    const Program q = make(ActFormat::kInt8);
    const Status s = be.check(q);
    CHECK(s.code() == StatusCode::kUnsupported && contains(s.message(), "activation format int8 cannot be requested"));
    CHECK(be.run(q, good()).code() == StatusCode::kUnsupported && untouched());
  }
  {  // Unverified program: run() verifies first.
    Program bad = p;
    bad.add_op(matmul_op(0, 1, 77, 2, 3, 4));
    CHECK(be.run(bad, good()).code() == StatusCode::kInvalidArgument && untouched());
  }
  CHECK(be.run(p, good()).ok());
  CHECK(y[0] == 4.0f && y[5] == 4.0f);
  std::printf("  backend rejections: ok\n");
}

// Review fixes: element-multiple strides, natural alignment, capacity,
// check() on unverified input, all diagnostics reported at once.
void test_bounds_and_alignment() {
  {  // A row stride that is not whole floats would be divided by sizeof(float).
    TensorLayout l = *TensorLayout::contiguous(DType::kF32, TensorShape{2, 4});
    l.strides[0] = 18;
    Program p;
    p.add_value(Value{kNoValue, "odd", ValueRole::kActivation, l, 0});
    CHECK(has_diag(p, "stride 18 of dimension 0 is not a multiple of the 4-byte element"));
  }
  {  // Several independent problems: every one reported, first one returned.
    Program raw;
    const ValueId x = raw.add_value(Value{kNoValue, "x", ValueRole::kActivation,
                                          *TensorLayout::contiguous(DType::kF32, TensorShape{2, 4}), 0});
    const ValueId w = raw.add_value(Value{kNoValue, "w", ValueRole::kWeight,
                                          *TensorLayout::contiguous(DType::kF32, TensorShape{3, 5}), 0});
    const ValueId y = raw.add_value(Value{kNoValue, "y", ValueRole::kResult,
                                          *TensorLayout::contiguous(DType::kF32, TensorShape{2, 2}), 0});
    raw.add_op(matmul_op(x, w, y, 2, 3, 4));
    CHECK(verify_all(raw).size() == 2);  // K mismatch, output shape
    CHECK(contains(verify(raw).message(), "(+1 more)"));
  }
  ThreadPool pool(1);
  ReferenceBackend be(pool, this_isa());
  {  // check() verifies first: no out-of-range access on malformed input.
    Program bad;
    Op op;
    op.kind = OpKind::kMatMul;
    op.inputs = {5};
    op.output = 9;
    op.attrs = MatMulAttrs{};
    bad.add_op(op);
    CHECK(be.check(bad).code() == StatusCode::kInvalidArgument);
    Program empty;
    CHECK(be.check(empty).ok());  // nothing to run is valid
  }
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, 2, 4);
  const ValueId vw = b.weight("w", DType::kF32, 3, 4);
  const ValueId vy = b.matmul(vx, vw);
  auto p = b.finish();
  CHECK(p.ok());
  alignas(16) float xs[8 + 4] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(16) float ws[12] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  alignas(16) float ys[6] = {-1, -1, -1, -1, -1, -1};
  auto bind_all = [&](Bindings& bi) {
    bi.bind(vx, view_of(xs, DType::kF32, 2, 4));
    bi.bind(vw, view_of(ws, DType::kF32, 3, 4));
    bi.bind(vy, view_of(ys, DType::kF32, 2, 3));
  };
  {  // An f32 pointer off by one byte: below natural alignment, rejected.
    Bindings bi;
    bind_all(bi);
    bi.bind(vx, view_of(reinterpret_cast<char*>(xs) + 1, DType::kF32, 2, 4));
    const Status s = be.run(*p, bi);
    CHECK(contains(s.message(), "data is not 4-byte aligned") && ys[0] == -1);
  }
  {  // Declared capacity smaller than the layout's span.
    Bindings bi;
    bind_all(bi);
    bi.bind(vy, view_of(ys, DType::kF32, 2, 3), 5 * sizeof(float));
    const Status s = be.run(*p, bi);
    CHECK(contains(s.message(), "layout spans 24 bytes but the buffer has 20") && ys[0] == -1);
  }
  {  // Tensor bindings take capacity from their storage; a slice at the end is too short.
    auto t = Tensor::zeros(DType::kF32, TensorShape{2, 3});
    CHECK(t.ok());
    Bindings bi;
    bind_all(bi);
    bi.bind(vy, *t);
    CHECK(be.run(*p, bi).ok());
    CHECK(t->data_as<float>()[0] == 4 && t->data_as<float>()[2] == 12 && t->data_as<float>()[5] == 12);
    CHECK(bi.entry(vy)->capacity >= 24);
    auto big = Tensor::zeros(DType::kF32, TensorShape{3, 3});
    CHECK(big.ok());
    auto tail = big->slice(0, 1, 2);  // rows 1..2: 24 bytes left, exactly enough
    CHECK(tail.ok());
    Bindings ok2;
    bind_all(ok2);
    ok2.bind(vy, *tail);
    CHECK(ok2.entry(vy)->capacity >= 24 && ok2.entry(vy)->capacity < bi.entry(vy)->capacity + 36);
    CHECK(be.run(*p, ok2).ok());
  }
  std::printf("  bounds, alignment, capacity: ok\n");
}

// Strided activations and results through each CPU path: fused dot (M = 1),
// int8 rows (Q8_0, M <= 4 on AVX2), fp32 panel (M >= 2 on f32 weights).
void test_strided_paths() {
  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  struct Case {
    int64_t m;
    DType wt;
    ActFormat act;
    double tol;
  };
  const Case cases[] = {{1, DType::kF32, ActFormat::kAuto, 1e-5}, {1, DType::kQ8_0, ActFormat::kF32, 1e-5},
                        {2, DType::kQ8_0, ActFormat::kAuto, 2e-2}, {3, DType::kF32, ActFormat::kAuto, 1e-5},
                        {6, DType::kQ8_0, ActFormat::kAuto, 1e-5}};
  const int64_t n = 16, k = 64, ldx = 80, ldy = 24;  // padded rows
  for (const Case& c : cases) {
    std::vector<float> xs(static_cast<size_t>(c.m * ldx), 9.0f), ys(static_cast<size_t>(c.m * ldy), 55.0f);
    const std::vector<float> xd = random_floats(static_cast<size_t>(c.m * k), 11);
    for (int64_t i = 0; i < c.m; ++i) std::memcpy(&xs[i * ldx], &xd[i * k], k * sizeof(float));
    std::vector<float> wf = random_floats(n * k, 12), wref = wf;
    std::vector<BlockQ8_0> q(n * k / 32);
    void* wdata = wf.data();
    if (c.wt == DType::kQ8_0) {
      quantize_q8_0(wf.data(), q.data(), static_cast<int64_t>(q.size()));
      dequantize_q8_0(q.data(), wref.data(), static_cast<int64_t>(q.size()));
      wdata = q.data();
    }
    TensorLayout xl = *TensorLayout::contiguous(DType::kF32, TensorShape{c.m, k});
    xl.strides[0] = ldx * sizeof(float);
    TensorLayout yl = *TensorLayout::contiguous(DType::kF32, TensorShape{c.m, n});
    yl.strides[0] = ldy * sizeof(float);
    ProgramBuilder b;
    const ValueId vx = b.value("x", ValueRole::kActivation, xl);
    const ValueId vw = b.weight("w", c.wt, n, k);
    const ValueId vy = b.value("y", ValueRole::kResult, yl);
    b.matmul_into(vx, vw, kNoValue, vy, {.act = c.act});
    auto p = b.finish();
    CHECK(p.ok());
    Bindings bi;
    bi.bind(vx, TensorView(xs.data(), xl), static_cast<int64_t>(xs.size() * sizeof(float)));
    bi.bind(vw, view_of(wdata, c.wt, n, k));
    bi.bind(vy, TensorView(ys.data(), yl), static_cast<int64_t>(ys.size() * sizeof(float)));
    CHECK(be.run(*p, bi).ok());
    CHECK(rel_error(ys.data(), ldy, reference(xd, wref, nullptr, c.m, n, k), c.m, n) <= c.tol);
    for (int64_t i = 0; i < c.m; ++i) {
      for (int64_t j = n; j < ldy; ++j) CHECK(ys[i * ldy + j] == 55.0f);  // padding untouched
    }
  }
  std::printf("  strided x/y through fused, int8 and panel paths: ok\n");
}

void test_defaults_untouched() {
  // The production kernel plan is not changed by this module, and a backend's
  // per-op plans stay on its own device.
  const KernelPlan& d = KernelPlan::defaults();
  if (std::getenv("DYNACORE_INT8_DECODE_ROWS") == nullptr) CHECK(d.int8_decode_max_rows == 4);
  CHECK(ReferenceBackend::plan_for(MatMulAttrs{}) == d);
  MatMulAttrs f32;
  f32.act = ActFormat::kF32;
  const KernelPlan p = ReferenceBackend::plan_for(f32);
  CHECK(p.int8_decode_max_rows == 0 && p.int16_decode_max_rows == 0);
  ThreadPool pool(1);
  CpuDevice model_device(pool, this_isa());
  ReferenceBackend be(pool, this_isa());
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, 1, 4);
  const ValueId vw = b.weight("w", DType::kF32, 2, 4);
  const ValueId vy = b.matmul(vx, vw, kNoValue, {.act = ActFormat::kF32});
  auto prog = b.finish();
  CHECK(prog.ok());
  std::vector<float> x(4, 1.0f), w(8, 0.5f), y(2, 0.0f);
  Bindings bind;
  bind.bind(vx, view_of(x.data(), DType::kF32, 1, 4));
  bind.bind(vw, view_of(w.data(), DType::kF32, 2, 4));
  bind.bind(vy, view_of(y.data(), DType::kF32, 1, 2));
  CHECK(be.run(*prog, bind).ok());
  CHECK(y[0] == 2.0f && y[1] == 2.0f);
  CHECK(model_device.kernel_plan() == d);
  std::printf("  production defaults unchanged: ok\n");
}

}  // namespace

int main() {
  std::printf("test_dynakernel (isa %s)\n", isa_name(this_isa()).data());
  test_construction();
  test_verifier();
  test_known_fp32();
  test_shapes();
  test_q8_0();
  test_chain_and_strided_output();
  test_backend_rejects();
  test_bounds_and_alignment();
  test_strided_paths();
  test_defaults_untouched();
  std::printf("all DynaKernel tests passed (%d checks)\n", g_checks);
  return 0;
}

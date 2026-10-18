// Generated C++ backend (codegen stage 0): provenance and determinism of the
// generated unit, proof that the generated code runs, accuracy against an
// independent double-precision Q4_K decode, agreement with the reference
// backend's paths within their bounds, and rejection of everything outside
// the generated family. No model files.
//
// Bounds (u = 2^-24, T = |d*sc*q| + |dmin*m|, S = sum |x| T):
//   generated kernel and fp32 reference paths: |e| <= (K + 5) u (S + |bias|)
//     (fp32 weight value: 2u T; k-order fp32 accumulation: K u S; bias add: u);
//   int8 reference paths add sum (h/2) |w|, h the activation rounding step
//     (per-32: amax32/127; sub-scaled DD-077: u_s * m_b).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/generated_backend.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;
using namespace dynacore::dk;
using namespace dynacore::quant;

extern "C" const dynacore::dk::GeneratedKernel* dk_generated_q4k_kernels(size_t* count);
extern "C" uint64_t dk_generated_q4k_fingerprint();
extern "C" uint32_t dk_generated_q4k_version();
#if DK_GENERATED_HAS_AVX2
extern "C" const dynacore::dk::GeneratedKernel* dk_generated_q4k_avx2_kernels(size_t* count);
extern "C" uint64_t dk_generated_q4k_avx2_fingerprint();
extern "C" uint32_t dk_generated_q4k_avx2_version();
#endif

namespace {

int g_checks = 0;
constexpr double kU = 1.0 / 16777216.0;

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

CpuIsa this_isa() { return select_isa(cpu_info().features); }

// One generated unit: its table and identity. The suite runs once per variant.
struct Variant {
  CodegenIsa isa = CodegenIsa::kScalar;
  std::span<const GeneratedKernel> kernels;
  uint64_t fingerprint = 0;
  uint32_t version = 0;
};

Variant scalar_variant() {
  size_t n = 0;
  const GeneratedKernel* k = dk_generated_q4k_kernels(&n);
  return {CodegenIsa::kScalar, {k, n}, dk_generated_q4k_fingerprint(), dk_generated_q4k_version()};
}

#if DK_GENERATED_HAS_AVX2
Variant avx2_variant() {
  size_t n = 0;
  const GeneratedKernel* k = dk_generated_q4k_avx2_kernels(&n);
  return {CodegenIsa::kAvx2, {k, n}, dk_generated_q4k_avx2_fingerprint(), dk_generated_q4k_avx2_version()};
}
#endif

Variant g_v;  // the variant under test
std::span<const GeneratedKernel> table() { return g_v.kernels; }
const char* vname() { return codegen_isa_name(g_v.isa).data(); }

// --- independent Q4_K decode and test data (same method as test_dynakernel_quant) ---

struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed * 2654435761u + 0x9e3779b9u) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  double uniform(double lo, double hi) { return lo + (hi - lo) * (next() / 4294967296.0); }
};

struct Weight {
  int64_t n = 0, k = 0;
  std::vector<BlockQ4_K> blocks;
  std::vector<double> w, mag;  // exact value and magnitude T per element
};

// Q4_K sub-block j's 6-bit scale and min, from the 12 packed bytes (written
// from the GGUF format, independently of the loader).
void scale_min(const uint8_t* s, int j, int& sc, int& m) {
  if (j < 4) {
    sc = s[j] & 63;
    m = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

Weight make_weight(int64_t n, int64_t k, uint32_t seed) {
  Weight w;
  w.n = n;
  w.k = k;
  const int64_t nb = n * k / 256;
  w.blocks.resize(static_cast<size_t>(nb));
  Rng r(seed);
  for (int64_t i = 0; i < nb; ++i) {
    BlockQ4_K& b = w.blocks[static_cast<size_t>(i)];
    auto* bytes = reinterpret_cast<uint8_t*>(&b);
    for (size_t t = 0; t < sizeof(BlockQ4_K); ++t) bytes[t] = static_cast<uint8_t>(r.next());
    b.d = fp32_to_fp16(static_cast<float>(((r.next() & 1) ? 1 : -1) * r.uniform(1e-3, 2e-2)));
    b.dmin = fp32_to_fp16(static_cast<float>(((r.next() & 1) ? 1 : -1) * r.uniform(1e-3, 2e-2)));
    if (i % 7 == 3) b.d = fp32_to_fp16(0.0f);           // zero scale
    if (i % 11 == 5) std::memset(b.scales, 0xFF, 12);  // saturated 6-bit scales and mins
  }
  w.w.resize(static_cast<size_t>(n * k));
  w.mag.resize(w.w.size());
  for (int64_t i = 0; i < nb; ++i) {
    const BlockQ4_K& b = w.blocks[static_cast<size_t>(i)];
    const double d = fp16_to_fp32(b.d), dmin = fp16_to_fp32(b.dmin);
    for (int g = 0; g < 4; ++g) {
      for (int half = 0; half < 2; ++half) {
        int sc = 0, m = 0;
        scale_min(b.scales, 2 * g + half, sc, m);
        for (int l = 0; l < 32; ++l) {
          const int q = half == 0 ? (b.qs[32 * g + l] & 0x0F) : (b.qs[32 * g + l] >> 4);
          const size_t at = static_cast<size_t>(i * 256 + 64 * g + 32 * half + l);
          w.w[at] = d * sc * q - dmin * m;
          w.mag[at] = std::abs(d * sc * q) + std::abs(dmin * m);
        }
      }
    }
  }
  return w;
}

// Rows: smooth, an outlier channel, tiny values, a zero 32-block, large mixed
// signs, all zero.
std::vector<float> make_x(int64_t m, int64_t k, int64_t ld, uint32_t seed) {
  Rng r(seed);
  std::vector<float> x(static_cast<size_t>(m * ld), 999.0f);  // padding is never read
  for (int64_t i = 0; i < m; ++i) {
    float* row = x.data() + i * ld;
    const int kind = static_cast<int>(i % 6);
    for (int64_t t = 0; t < k; ++t) {
      double v = r.uniform(-1, 1) + r.uniform(-1, 1) + r.uniform(-1, 1);
      if (kind == 2) v *= 1e-3;
      if (kind == 4) v *= 20;
      if (kind == 5 && m > 1) v = 0;
      row[t] = static_cast<float>(v);
    }
    if (kind == 1) row[(i * 37 + 11) % k] = (i & 1) ? -40.0f : 40.0f;
    if (kind == 3) std::fill(row + 32, row + 64, 0.0f);
  }
  return x;
}

enum class Path { kFp32, kInt8Per32, kInt8Subscale };

std::vector<double> act_steps(const float* x, int64_t k, Path p) {
  std::vector<double> h(static_cast<size_t>(k), 0.0);
  auto amax = [&](int64_t at, int64_t len) {
    float a = 0;
    for (int64_t t = at; t < at + len; ++t) a = std::max(a, std::fabs(x[t]));
    return a;
  };
  if (p == Path::kInt8Per32) {
    for (int64_t b = 0; b < k; b += 32) {
      const double step = amax(b, 32) / 127.0f;
      for (int64_t t = b; t < b + 32; ++t) h[static_cast<size_t>(t)] = step;
    }
  } else if (p == Path::kInt8Subscale) {
    for (int64_t s = 0; s < k; s += 256) {
      const float u = amax(s, 256) / (127.0f * 128.0f);
      for (int64_t b = s; b < s + 256; b += 32) {
        int mult = 1;
        if (u > 0) mult = std::clamp(static_cast<int>(std::ceil(amax(b, 32) / (127.0f * u))), 1, 128);
        for (int64_t t = b; t < b + 32; ++t) h[static_cast<size_t>(t)] = static_cast<double>(u) * mult;
      }
    }
  }
  return h;
}

struct Stats {
  double max_abs = 0, max_rel = 0, max_ratio = 0;
  int cases = 0;
  void print(const char* name) const {
    std::printf("  %-26s %5d %11.3g %11.3g %9.3f\n", name, cases, max_abs, max_rel, max_ratio);
  }
};

// y (row stride ldy) against the exact reference within the path's bound.
void check_bound(const Weight& w, const std::vector<float>& x, int64_t ldx, const float* bias, const float* y,
                 int64_t ldy, int64_t m, Path p, Stats& st) {
  double max_ref = 0, case_abs = 0;
  for (int64_t i = 0; i < m; ++i) {
    const float* xr = x.data() + i * ldx;
    const std::vector<double> h = act_steps(xr, w.k, p);
    for (int64_t j = 0; j < w.n; ++j) {
      double ref = bias ? bias[j] : 0.0, s = bias ? std::abs(bias[j]) : 0.0, qerr = 0;
      for (int64_t t = 0; t < w.k; ++t) {
        const size_t wi = static_cast<size_t>(j * w.k + t);
        const double hv = h[static_cast<size_t>(t)];
        ref += static_cast<double>(xr[t]) * w.w[wi];
        s += (std::abs(xr[t]) + hv / 2) * w.mag[wi];
        qerr += hv / 2 * std::abs(w.w[wi]);
      }
      const double bound = (static_cast<double>(w.k) + 5) * kU * s + qerr + 1e-30;
      const double e = std::abs(static_cast<double>(y[i * ldy + j]) - ref);
      if (!(e <= bound)) {
        std::fprintf(stderr, "  M=%lld N=%lld K=%lld y[%lld,%lld]: |e| %.3g > bound %.3g\n", static_cast<long long>(m),
                     static_cast<long long>(w.n), static_cast<long long>(w.k), static_cast<long long>(i),
                     static_cast<long long>(j), e, bound);
      }
      CHECK(e <= bound);
      max_ref = std::max(max_ref, std::abs(ref));
      case_abs = std::max(case_abs, e);
      st.max_ratio = std::max(st.max_ratio, e / bound);
    }
  }
  st.max_abs = std::max(st.max_abs, case_abs);
  if (max_ref > 0) st.max_rel = std::max(st.max_rel, case_abs / max_ref);
  ++st.cases;
}

TensorView view2(const void* data, DType t, int64_t r, int64_t c, int64_t row_stride_elems = 0) {
  TensorLayout l = *TensorLayout::contiguous(t, TensorShape{r, c});
  if (row_stride_elems > 0) l.strides[0] = row_stride_elems * static_cast<int64_t>(sizeof(float));
  return TensorView(const_cast<void*>(data), l);
}

struct Matmul {
  Program p;
  ValueId x, w, b, y;
};

Matmul build(int64_t m, int64_t n, int64_t k, bool bias, ActFormat act = ActFormat::kAuto, int64_t ldx = 0,
             int64_t ldy = 0) {
  ProgramBuilder pb;
  Matmul r;
  TensorLayout xl = *TensorLayout::contiguous(DType::kF32, TensorShape{m, k});
  if (ldx > 0) xl.strides[0] = ldx * 4;
  TensorLayout yl = *TensorLayout::contiguous(DType::kF32, TensorShape{m, n});
  if (ldy > 0) yl.strides[0] = ldy * 4;
  r.x = pb.value("x", ValueRole::kActivation, xl);
  r.w = pb.weight("w", DType::kQ4_K, n, k);
  r.b = bias ? pb.bias("b", n) : kNoValue;
  r.y = pb.value("y", ValueRole::kResult, yl);
  pb.matmul_into(r.x, r.w, r.b, r.y, {.act = act});
  auto p = pb.finish();
  CHECK(p.ok());
  r.p = std::move(*p);
  return r;
}

Status run_on(Backend& be, const Matmul& mm, const Weight& w, const std::vector<float>& x, int64_t ldx,
              const std::vector<float>& bias, std::vector<float>& y, int64_t ldy, int64_t m) {
  Bindings bi;
  bi.bind(mm.x, view2(x.data(), DType::kF32, m, w.k, ldx), static_cast<int64_t>(x.size() * 4));
  bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, w.n, w.k),
          static_cast<int64_t>(w.blocks.size() * sizeof(BlockQ4_K)));
  if (mm.b != kNoValue) {
    bi.bind(mm.b, TensorView(const_cast<float*>(bias.data()), *TensorLayout::contiguous(DType::kF32, TensorShape{w.n})));
  }
  bi.bind(mm.y, view2(y.data(), DType::kF32, m, w.n, ldy), static_cast<int64_t>(y.size() * 4));
  return be.run(mm.p, bi);
}

// --- tests ---------------------------------------------------------------------------------

void test_provenance() {
  const std::vector<Q4KMatmulSpec> specs = q4k_test_specs();
  auto a = generate_q4k_source(specs, g_v.isa), b = generate_q4k_source(specs, g_v.isa);
  CHECK(a.ok() && b.ok() && a->text == b->text && a->fingerprint == b->fingerprint);  // deterministic
  // The linked unit is the one this generator produces for these specs.
  CHECK(g_v.fingerprint == a->fingerprint);
  CHECK(g_v.version == kCodegenVersion);
  const auto t = table();
  CHECK(t.size() == specs.size());
  for (size_t i = 0; i < t.size(); ++i) {
    CHECK(t[i].spec == specs[i] && t[i].fn != nullptr && t[i].calls != nullptr && t[i].isa == g_v.isa);
    CHECK(kernel_symbol(specs[i], g_v.isa) == t[i].symbol);
  }
  // The generator refuses what it cannot emit.
  const Q4KMatmulSpec bad[] = {{0, 8, 256, false}, {9, 8, 256, false}, {1, 8, 300, false}, {1, 0, 256, false},
                               {1, 8, 0, false}};
  for (const Q4KMatmulSpec& s : bad) {
    const Q4KMatmulSpec one[] = {s};
    CHECK(generate_q4k_source(one).status().code() == StatusCode::kUnsupported);
  }
  const Q4KMatmulSpec dup[] = {{2, 8, 256, false}, {2, 8, 256, false}};
  CHECK(generate_q4k_source(dup).status().code() == StatusCode::kInvalidArgument);
  CHECK(generate_q4k_source(std::span<const Q4KMatmulSpec>()).status().code() == StatusCode::kInvalidArgument);
  std::printf("  [%s] provenance: %zu kernels, fingerprint %016llx matches a fresh generation; deterministic\n",
              vname(), t.size(), static_cast<unsigned long long>(g_v.fingerprint));
}

// The generated function runs (its own counter moves, once per column chunk)
// and nothing else computes the result: a table whose entry points at a stub
// makes the backend produce the stub's output.
void stub_kernel(const void*, const float*, int64_t, const float*, float* y, int64_t ys, int64_t n0, int64_t n1) {
  for (int64_t j = n0; j < n1; ++j) y[j] = 12345.0f, y[ys + j] = 12345.0f;
}
uint64_t stub_calls() { return 0; }

void test_execution_proof() {
  ThreadPool pool(4);
  GeneratedBackend be(pool, table());
  const Weight w = make_weight(24, 1280, 5);
  const std::vector<float> x = make_x(2, 1280, 1280, 6);
  std::vector<float> y(2 * 24, std::nanf("")), none;
  const Matmul mm = build(2, 24, 1280, false);
  const GeneratedKernel* g = be.find(mm.p.ops()[0]);
  CHECK(g != nullptr && std::string(g->symbol) == kernel_symbol({2, 24, 1280, false}, g_v.isa) && g->isa == g_v.isa);
  uint64_t others0 = 0;
  for (const GeneratedKernel& k : table()) others0 += &k == g ? 0 : k.calls();
  const uint64_t c0 = g->calls();
  CHECK(run_on(be, mm, w, x, 1280, none, y, 24, 2).ok());
  CHECK(g->calls() - c0 == 3);  // ceil(24 / 8) column chunks
  uint64_t others1 = 0;
  for (const GeneratedKernel& k : table()) others1 += &k == g ? 0 : k.calls();
  CHECK(others1 == others0);
  Stats st;
  check_bound(w, x, 1280, nullptr, y.data(), 24, 2, Path::kFp32, st);
  // Same program, same table layout, one entry swapped for a stub.
  std::vector<GeneratedKernel> swapped(table().begin(), table().end());
  for (GeneratedKernel& k : swapped) {
    if (k.spec == g->spec) {
      k.fn = &stub_kernel;
      k.calls = &stub_calls;
    }
  }
  GeneratedBackend stubbed(pool, swapped);
  std::vector<float> ys(2 * 24, 0.0f);
  CHECK(run_on(stubbed, mm, w, x, 1280, none, ys, 24, 2).ok());
  CHECK(std::all_of(ys.begin(), ys.end(), [](float v) { return v == 12345.0f; }));
  CHECK(be.target().backend == "generated" && be.target().isa == codegen_isa_name(g_v.isa));
  std::printf("  [%s] execution proof: %s ran 3 chunks, no other kernel ran; a stubbed entry is what executes\n",
              vname(), g->symbol);
}

void test_accuracy() {
  ThreadPool pool(4);
  GeneratedBackend gen(pool, table());
  ReferenceBackend ref(pool, this_isa());
  ReferenceBackend packed_ref(pool, this_isa());
  const bool int8_dot = ref.target().int8_dot, int8_sx = ref.target().int8_subscaled;
  Stats s_gen, s_f32, s_per32, s_sx, s_packed;
  int same_as_f32 = 0, total = 0, packed_cases = 0;
  uint32_t seed = 100;
  std::vector<std::unique_ptr<Weight>> keep;  // packed sources stay alive until released
  for (const GeneratedKernel& g : table()) {
    const Q4KMatmulSpec& s = g.spec;
    keep.push_back(std::make_unique<Weight>(make_weight(s.n, s.k, seed++)));
    const Weight& w = *keep.back();
    const std::vector<float> x = make_x(s.m, s.k, s.k, seed++);
    std::vector<float> bias(static_cast<size_t>(s.n));
    Rng r(seed++);
    for (float& b : bias) b = static_cast<float>(r.uniform(-2, 2));
    const float* bp = s.bias ? bias.data() : nullptr;
    std::vector<float> yg(static_cast<size_t>(s.m * s.n));
    const Matmul mm = build(s.m, s.n, s.k, s.bias);
    CHECK(run_on(gen, mm, w, x, s.k, bias, yg, s.n, s.m).ok());
    check_bound(w, x, s.k, bp, yg.data(), s.n, s.m, Path::kFp32, s_gen);
    // Reference, f32 contract (fused dot or fp32 panel).
    std::vector<float> yf(yg.size());
    const Matmul mf = build(s.m, s.n, s.k, s.bias, ActFormat::kF32);
    CHECK(run_on(ref, mf, w, x, s.k, bias, yf, s.n, s.m).ok());
    check_bound(w, x, s.k, bp, yf.data(), s.n, s.m, Path::kFp32, s_f32);
    same_as_f32 += yg == yf;
    ++total;
    // Reference, auto contract (int8 on AVX2), unpacked.
    std::vector<float> ya(yg.size());
    CHECK(run_on(ref, mm, w, x, s.k, bias, ya, s.n, s.m).ok());
    // The auto contract: per-32 for one row; sub-scaled for 2..8 rows on a
    // tier with that family (AVX2); per-32 up to int8_decode_max_rows on NEON with SDOT.
    const int64_t per32_rows = KernelPlan::defaults().int8_decode_max_rows;
    const Path pa = !int8_dot ? Path::kFp32
                    : s.m == 1 ? Path::kInt8Per32
                    : int8_sx  ? (s.m <= 8 ? Path::kInt8Subscale : Path::kFp32)
                               : (s.m <= per32_rows ? Path::kInt8Per32 : Path::kFp32);
    check_bound(w, x, s.k, bp, ya.data(), s.n, s.m, pa, pa == Path::kInt8Per32 ? s_per32 : pa == Path::kInt8Subscale ? s_sx : s_f32);
    // Reference, auto contract, prepacked interleaved Q4_K (AVX2, N % 8 == 0, M 2..8).
    if (int8_sx && s.n % 8 == 0 && s.m >= 2) {
      auto pk = packed_ref.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, s.n, s.k));
      if (!pk.ok()) std::fprintf(stderr, "%s\n", pk.status().to_string().c_str());
      CHECK(pk.ok());
      std::vector<float> yp(yg.size());
      CHECK(run_on(packed_ref, mm, w, x, s.k, bias, yp, s.n, s.m).ok());
      check_bound(w, x, s.k, bp, yp.data(), s.n, s.m, Path::kInt8Subscale, s_packed);
      ++packed_cases;
    }
  }
  for (const auto& w : keep) packed_ref.release_packed(w->blocks.data());
  std::printf("  %-26s %5s %11s %11s %9s\n", "implementation", "cases", "max |err|", "max rel", "err/bound");
  s_gen.print(g_v.isa == CodegenIsa::kAvx2 ? "generated avx2 (fp32)" : "generated scalar (fp32)");
  s_f32.print("reference f32 contract");
  if (s_per32.cases) s_per32.print("reference int8 per-32");
  if (s_sx.cases) s_sx.print("reference int8 sub-scaled");
  if (s_packed.cases) s_packed.print("reference packed q4_K");
  std::printf("  generated == reference f32 bit for bit in %d/%d cases (summation order differs otherwise)\n",
              same_as_f32, total);
}

void test_strides_and_threads() {
  // Strided x and y (padded rows), bias, and thread-count independence.
  const Weight w = make_weight(33, 256, 200);
  const int64_t m = 5, ldx = 300, ldy = 40;
  const std::vector<float> x = make_x(m, 256, ldx, 201);
  std::vector<float> bias(33);
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.25f * static_cast<float>(i) - 3.0f;
  const Matmul mm = build(m, 33, 256, true, ActFormat::kF32, ldx, ldy);
  std::vector<float> y1(static_cast<size_t>(m * ldy), -7.0f), y4 = y1;
  ThreadPool p1(1), p4(4);
  GeneratedBackend b1(p1, table()), b4(p4, table());
  CHECK(run_on(b1, mm, w, x, ldx, bias, y1, ldy, m).ok());
  CHECK(run_on(b4, mm, w, x, ldx, bias, y4, ldy, m).ok());
  CHECK(y1 == y4);  // one thread per output element, fixed k order
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 33; j < ldy; ++j) CHECK(y1[static_cast<size_t>(i * ldy + j)] == -7.0f);  // padding untouched
  }
  Stats st;
  check_bound(w, x, ldx, bias.data(), y1.data(), ldy, m, Path::kFp32, st);
  std::printf("  strided x (ld %lld) / y (ld %lld) with bias: within bound (%.3f); 1 and 4 threads bit-identical\n",
              static_cast<long long>(ldx), static_cast<long long>(ldy), st.max_ratio);
}

void test_rejections() {
  ThreadPool pool(2);
  GeneratedBackend be(pool, table());
  auto rejected = [&](const Program& p, StatusCode code, const char* needle) {
    const Status s = be.check(p);
    if (s.code() != code || s.message().find(needle) == std::string::npos) {
      std::fprintf(stderr, "  unexpected: %s\n", s.to_string().c_str());
    }
    return s.code() == code && s.message().find(needle) != std::string::npos;
  };
  CHECK(rejected(build(9, 8, 256, false).p, StatusCode::kUnsupported, "M must be 1..8"));
  CHECK(rejected(build(2, 8, 512, false).p, StatusCode::kUnsupported, "in this build's table"));
  CHECK(rejected(build(2, 8, 256, true).p, StatusCode::kUnsupported, "dk_q4k_m2_n8_k256_b1"));  // no bias variant
  CHECK(rejected(build(2, 8, 256, false, ActFormat::kInt8).p, StatusCode::kUnsupported, "is not generated"));
  {  // Q6_K weight.
    ProgramBuilder pb;
    const ValueId x = pb.activation("x", DType::kF32, 2, 256);
    const ValueId w = pb.weight("w", DType::kQ6_K, 8, 256);
    pb.matmul(x, w);
    auto p = pb.finish();
    CHECK(p.ok());
    CHECK(rejected(*p, StatusCode::kUnsupported, "take q4_K weights, not q6_K"));
  }
  {  // Unverified program: verify() runs first.
    Program bad = build(2, 8, 256, false).p;
    Op op = bad.ops()[0];
    op.output = 99;
    bad.add_op(op);
    CHECK(be.check(bad).code() == StatusCode::kInvalidArgument);
  }
  // Bindings: misaligned x, short capacity, overlapping output; nothing written.
  const Weight w = make_weight(8, 256, 300);
  const Matmul mm = build(2, 8, 256, false);
  std::vector<float> x = make_x(2, 256, 256, 301), y(16, -1.0f), big(1024, 0.0f);
  auto untouched = [&] { return std::all_of(y.begin(), y.end(), [](float v) { return v == -1.0f; }); };
  {
    Bindings bi;
    bi.bind(mm.x, view2(reinterpret_cast<char*>(big.data()) + 2, DType::kF32, 2, 256));
    bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, 8, 256));
    bi.bind(mm.y, view2(y.data(), DType::kF32, 2, 8));
    CHECK(be.run(mm.p, bi).message().find("aligned") != std::string::npos && untouched());
  }
  {
    Bindings bi;
    bi.bind(mm.x, view2(x.data(), DType::kF32, 2, 256));
    bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, 8, 256), 8 * 144 - 1);
    bi.bind(mm.y, view2(y.data(), DType::kF32, 2, 8));
    CHECK(be.run(mm.p, bi).message().find("but the buffer has") != std::string::npos && untouched());
  }
  {
    Bindings bi;
    bi.bind(mm.x, view2(x.data(), DType::kF32, 2, 256));
    bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, 8, 256));
    bi.bind(mm.y, view2(x.data() + 4, DType::kF32, 2, 8));  // y inside x
    CHECK(be.run(mm.p, bi).message().find("overlaps") != std::string::npos);
    CHECK(x[4] == make_x(2, 256, 256, 301)[4]);
  }
  std::printf("  rejections: M=9, missing shape, missing bias variant, int8 contract, q6_K, unverified, "
              "alignment, capacity, overlap\n");
}

Variant avx2_variant_or_empty() {
#if DK_GENERATED_HAS_AVX2
  return avx2_variant();
#else
  return Variant{CodegenIsa::kAvx2, {}, 0, 0};
#endif
}

// Whether this process may run the AVX2 table: built with it, and the AVX2
// tier selected (hardware, and DYNACORE_ISA not forcing a lower tier).
bool avx2_runs_here() { return DK_GENERATED_HAS_AVX2 && this_isa() == CpuIsa::kAvx2; }

// Which generated backend can run on this tier, asserted rather than
// assumed; a refused backend writes nothing.
void test_tier_rules() {
  ThreadPool pool(2);
  const Variant sv = scalar_variant();
  GeneratedBackend scalar(pool, sv.kernels);
  CHECK(scalar.runnable().ok());  // portable C++: every tier
  const Variant av = avx2_variant_or_empty();
  if (!DK_GENERATED_HAS_AVX2) {
    std::printf("  tier rules: this build has no AVX2 unit (ENABLE_AVX2 off); scalar runs\n");
    return;
  }
  GeneratedBackend avx2(pool, av.kernels);
  const Matmul mm = build(2, 24, 1280, false);
  const Weight w = make_weight(24, 1280, 7);
  const std::vector<float> x = make_x(2, 1280, 1280, 8);
  std::vector<float> y(48, -3.0f), none;
  if (this_isa() == CpuIsa::kAvx2) {
    CHECK(avx2.runnable().ok() && avx2.check(mm.p).ok());
  } else {
    // generic tier (no AVX2, or DYNACORE_ISA=generic): refused, never run.
    const uint64_t c0 = av.kernels.empty() ? 0 : av.kernels[0].calls();
    CHECK(avx2.runnable().code() == StatusCode::kUnsupported);
    CHECK(avx2.check(mm.p).code() == StatusCode::kUnsupported);
    const Status s = run_on(avx2, mm, w, x, 1280, none, y, 24, 2);
    CHECK(s.code() == StatusCode::kUnsupported && s.message().find("need the avx2 tier") != std::string::npos);
    CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == -3.0f; }));
    uint64_t total = 0;
    for (const GeneratedKernel& k : av.kernels) total += k.calls();
    CHECK(total == 0 && c0 == 0);  // no AVX2 kernel ever ran in this process
  }
  // A table mixing variants is refused (one ISA per backend).
  std::vector<GeneratedKernel> mixed(sv.kernels.begin(), sv.kernels.end());
  mixed.push_back(av.kernels[0]);
  GeneratedBackend bad(pool, mixed);
  CHECK(bad.runnable().code() == StatusCode::kInvalidArgument);
  std::printf("  tier rules (%s tier): scalar runnable; avx2 %s; mixed table refused\n", isa_name(this_isa()).data(),
              this_isa() == CpuIsa::kAvx2 ? "runnable" : "refused with kUnsupported, nothing written, 0 calls");
}

// The AVX2 kernels (not the scalar ones) execute for the AVX2 table, and the
// two variants differ only in summation order: both within the bound.
void test_scalar_vs_avx2() {
  ThreadPool pool(4);
  const Variant sv = scalar_variant(), av = avx2_variant_or_empty();
  GeneratedBackend scalar(pool, sv.kernels), avx2(pool, av.kernels);
  int identical = 0, cases = 0;
  double max_diff_ratio = 0;
  uint32_t seed = 500;
  for (const GeneratedKernel& g : av.kernels) {
    const Q4KMatmulSpec& s = g.spec;
    const Weight w = make_weight(s.n, s.k, seed++);
    const std::vector<float> x = make_x(s.m, s.k, s.k, seed++);
    std::vector<float> bias(static_cast<size_t>(s.n), 0.5f);
    std::vector<float> ys(static_cast<size_t>(s.m * s.n)), ya(ys.size());
    const Matmul mm = build(s.m, s.n, s.k, s.bias);
    uint64_t scalar_calls0 = 0, avx2_calls0 = 0;
    for (const GeneratedKernel& k : sv.kernels) scalar_calls0 += k.calls();
    for (const GeneratedKernel& k : av.kernels) avx2_calls0 += k.calls();
    CHECK(run_on(avx2, mm, w, x, s.k, bias, ya, s.n, s.m).ok());
    uint64_t scalar_calls1 = 0, avx2_calls1 = 0;
    for (const GeneratedKernel& k : sv.kernels) scalar_calls1 += k.calls();
    for (const GeneratedKernel& k : av.kernels) avx2_calls1 += k.calls();
    CHECK(scalar_calls1 == scalar_calls0 && avx2_calls1 > avx2_calls0);  // only AVX2 code ran
    CHECK(run_on(scalar, mm, w, x, s.k, bias, ys, s.n, s.m).ok());
    identical += ys == ya;
    ++cases;
    Stats st;
    check_bound(w, x, s.k, s.bias ? bias.data() : nullptr, ya.data(), s.n, s.m, Path::kFp32, st);
    max_diff_ratio = std::max(max_diff_ratio, st.max_ratio);
  }
  std::printf("  scalar vs avx2: %d/%d bit-identical (avx2 max err/bound %.3f); every avx2 run moved only avx2 counters\n",
              identical, cases, max_diff_ratio);
}

}  // namespace

int main() {
  std::printf("test_dynakernel_codegen (reference isa %s)\n", isa_name(this_isa()).data());
  // Packing must not depend on free RAM (DD-078 budget) for these small weights.
#if defined(_WIN32)
  _putenv_s("DYNACORE_Q4_REPACK", "always");
#else
  setenv("DYNACORE_Q4_REPACK", "always", 1);
#endif
  auto suite = [] {
    test_provenance();
    test_execution_proof();
    test_accuracy();
    test_strides_and_threads();
    test_rejections();
  };
  g_v = scalar_variant();
  suite();
  test_tier_rules();
  if (avx2_runs_here()) {
    g_v = avx2_variant_or_empty();
    suite();
    test_scalar_vs_avx2();
  }
  std::printf("all codegen tests passed (%d checks)\n", g_checks);
  return 0;
}

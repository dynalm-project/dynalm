// Generated INT8 Q4_K kernels (prototype): provenance, ISA guard, proof of
// execution, accuracy against the independent double-precision Q4_K decode
// within the sub-scaled int8 bound, comparison with the reference int8 paths
// (same activation codes), packing rules and rejections. No model files.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/generated_backend.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "q4k_reference.h"

using namespace dynacore;
using namespace dynacore::dk;
using q4kref::Path;
using q4kref::Stats;
using q4kref::Weight;

#define CHECK Q4K_CHECK

#if DK_GENERATED_HAS_AVX2
extern "C" const dynacore::dk::GeneratedInt8Kernel* dk_generated_q4k_int8_avx2_kernels(size_t* count);
extern "C" uint64_t dk_generated_q4k_int8_avx2_fingerprint();
extern "C" uint32_t dk_generated_q4k_int8_avx2_version();
#endif

namespace {

CpuIsa this_isa() { return select_isa(cpu_info().features); }

std::span<const GeneratedInt8Kernel> table() {
#if DK_GENERATED_HAS_AVX2
  size_t n = 0;
  const GeneratedInt8Kernel* k = dk_generated_q4k_int8_avx2_kernels(&n);
  return {k, n};
#else
  return {};
#endif
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

Bindings bind(const Matmul& mm, const Weight& w, const std::vector<float>& x, int64_t ldx,
              const std::vector<float>& bias, std::vector<float>& y, int64_t ldy, int64_t m) {
  Bindings bi;
  bi.bind(mm.x, view2(x.data(), DType::kF32, m, w.k, ldx), static_cast<int64_t>(x.size() * 4));
  bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, w.n, w.k));
  if (mm.b != kNoValue) {
    bi.bind(mm.b, TensorView(const_cast<float*>(bias.data()), *TensorLayout::contiguous(DType::kF32, TensorShape{w.n})));
  }
  bi.bind(mm.y, view2(y.data(), DType::kF32, m, w.n, ldy), static_cast<int64_t>(y.size() * 4));
  return bi;
}

TensorView wview(const Weight& w) { return view2(w.blocks.data(), DType::kQ4_K, w.n, w.k); }

void test_provenance() {
  const std::vector<Q4KMatmulSpec> specs = q4k_int8_test_specs();
  auto a = generate_q4k_int8_source(specs), b = generate_q4k_int8_source(specs);
  CHECK(a.ok() && b.ok() && a->text == b->text);
#if DK_GENERATED_HAS_AVX2
  CHECK(dk_generated_q4k_int8_avx2_fingerprint() == a->fingerprint);
  CHECK(dk_generated_q4k_int8_avx2_version() == kCodegenVersion);
#endif
  const auto t = table();
  CHECK(t.size() == (DK_GENERATED_HAS_AVX2 ? specs.size() : 0));
  for (size_t i = 0; i < t.size(); ++i) CHECK(t[i].spec == specs[i] && int8_kernel_symbol(specs[i]) == t[i].symbol);
  for (const Q4KMatmulSpec& s : {Q4KMatmulSpec{2, 12, 256, false}, Q4KMatmulSpec{9, 8, 256, false},
                                 Q4KMatmulSpec{2, 8, 300, false}, Q4KMatmulSpec{0, 8, 256, false}}) {
    const Q4KMatmulSpec one[] = {s};
    CHECK(generate_q4k_int8_source(one).status().code() == StatusCode::kUnsupported);
  }
  const Q4KMatmulSpec dup[] = {{2, 8, 256, false}, {2, 8, 256, false}};
  CHECK(generate_q4k_int8_source(dup).status().code() == StatusCode::kInvalidArgument);
  std::printf("  provenance: %zu int8 kernels; generator deterministic; rejects N %% 8, M 9, K 300, M 0, duplicates\n",
              t.size());
}

// On a non-AVX2 tier (or DYNACORE_ISA=generic) nothing generated runs.
void test_tier_rules() {
  ThreadPool pool(2);
  GeneratedInt8Backend be(pool, table());
  const Matmul mm = build(2, 24, 1280, false);
  const Weight w = q4kref::make_weight(24, 1280, 1);
  const std::vector<float> x = q4kref::make_x(2, 1280, 1280, 2);
  std::vector<float> y(48, -4.0f), none;
  if (this_isa() == CpuIsa::kAvx2 && DK_GENERATED_HAS_AVX2) {
    CHECK(be.runnable().ok() && be.check(mm.p).ok());
    std::printf("  tier rules (avx2): runnable\n");
    return;
  }
  CHECK(be.runnable().code() == StatusCode::kUnsupported);
  const Status s = be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2));
  CHECK(s.code() == StatusCode::kUnsupported);
  CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == -4.0f; }));
  uint64_t calls = 0;
  for (const GeneratedInt8Kernel& k : table()) calls += k.calls();
  CHECK(calls == 0);
  std::printf("  tier rules (%s): refused with kUnsupported, nothing written, 0 generated calls\n",
              isa_name(this_isa()).data());
}

void stub_int8(const BlockQ4_Kx8*, const ActSuperQ8*, const float*, float* y, int64_t ys, int64_t g0, int64_t g1) {
  for (int64_t g = g0; g < g1; ++g) {
    for (int i = 0; i < 8; ++i) y[8 * g + i] = 777.0f, y[ys + 8 * g + i] = 777.0f;
  }
}
uint64_t stub_calls() { return 0; }

void test_execution_proof() {
  ThreadPool pool(4);
  GeneratedInt8Backend be(pool, table());
  const Weight w = q4kref::make_weight(24, 1280, 3);
  const std::vector<float> x = q4kref::make_x(2, 1280, 1280, 4);
  std::vector<float> y(48, std::nanf("")), none;
  const Matmul mm = build(2, 24, 1280, false);
  CHECK(be.prepack_weight(wview(w)).ok() && be.packed_count() == 1);
  const GeneratedInt8Kernel* g = be.find(mm.p.ops()[0]);
  CHECK(g != nullptr && std::string(g->symbol) == "dk_q4k_sx8_avx2_m2_n24_k1280_b0");
  uint64_t others0 = 0;
  for (const GeneratedInt8Kernel& k : table()) others0 += &k == g ? 0 : k.calls();
  const uint64_t c0 = g->calls();
  CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).ok());
  CHECK(g->calls() - c0 == 3);  // 3 groups of 8 rows, grain 1
  uint64_t others1 = 0;
  for (const GeneratedInt8Kernel& k : table()) others1 += &k == g ? 0 : k.calls();
  CHECK(others1 == others0);
  Stats st;
  q4kref::check_bound(w, x, 1280, nullptr, y.data(), 24, 2, Path::kInt8Subscale, st);
  std::vector<GeneratedInt8Kernel> swapped(table().begin(), table().end());
  for (GeneratedInt8Kernel& k : swapped) {
    if (k.spec == g->spec) k.fn = &stub_int8, k.calls = &stub_calls;
  }
  GeneratedInt8Backend stubbed(pool, swapped);
  CHECK(stubbed.prepack_weight(wview(w)).ok());
  std::vector<float> ys(48, 0.0f);
  CHECK(stubbed.run(mm.p, bind(mm, w, x, 1280, none, ys, 24, 2)).ok());
  CHECK(std::all_of(ys.begin(), ys.end(), [](float v) { return v == 777.0f; }));
  CHECK(be.target().backend == "generated-int8" && be.target().int8_dot);
  std::printf("  execution proof: %s ran 3 group chunks, no other kernel ran; a stubbed entry is what executes\n",
              g->symbol);
}

void test_accuracy_vs_reference() {
  ThreadPool pool(4);
  GeneratedInt8Backend gen(pool, table());
  ReferenceBackend ref(pool, this_isa()), ref_pk(pool, this_isa());
  Stats s_gen, s_ref_sx, s_ref_pk, s_ref_per32;
  int same_pk = 0, same_sx = 0, n_pk = 0, n_sx = 0;
  uint32_t seed = 100;
  std::vector<std::unique_ptr<Weight>> keep;  // packed sources stay alive until released
  for (const GeneratedInt8Kernel& g : table()) {
    const Q4KMatmulSpec& s = g.spec;
    keep.push_back(std::make_unique<Weight>(q4kref::make_weight(s.n, s.k, seed++)));
    const Weight& w = *keep.back();
    const std::vector<float> x = q4kref::make_x(s.m, s.k, s.k, seed++);
    std::vector<float> bias(static_cast<size_t>(s.n));
    q4kref::Rng r(seed++);
    for (float& v : bias) v = static_cast<float>(r.uniform(-2, 2));
    const float* bp = s.bias ? bias.data() : nullptr;
    const Matmul mm = build(s.m, s.n, s.k, s.bias);
    CHECK(gen.prepack_weight(wview(w)).ok());
    std::vector<float> yg(static_cast<size_t>(s.m * s.n));
    CHECK(gen.run(mm.p, bind(mm, w, x, s.k, bias, yg, s.n, s.m)).ok());
    q4kref::check_bound(w, x, s.k, bp, yg.data(), s.n, s.m, Path::kInt8Subscale, s_gen);
    // Reference auto contract: M = 1 per-32 (other activation format), M >= 2 sub-scaled rows.
    std::vector<float> yr(yg.size());
    CHECK(ref.run(mm.p, bind(mm, w, x, s.k, bias, yr, s.n, s.m)).ok());
    if (s.m == 1) {
      q4kref::check_bound(w, x, s.k, bp, yr.data(), s.n, s.m, Path::kInt8Per32, s_ref_per32);
    } else {
      q4kref::check_bound(w, x, s.k, bp, yr.data(), s.n, s.m, Path::kInt8Subscale, s_ref_sx);
      same_sx += yg == yr;
      ++n_sx;
      // Reference packed sub-scaled (same codes, same layout, different kernel).
      CHECK(ref_pk.prepack_weight(wview(w)).ok());
      std::vector<float> yp(yg.size());
      CHECK(ref_pk.run(mm.p, bind(mm, w, x, s.k, bias, yp, s.n, s.m)).ok());
      q4kref::check_bound(w, x, s.k, bp, yp.data(), s.n, s.m, Path::kInt8Subscale, s_ref_pk);
      same_pk += yg == yp;
      ++n_pk;
    }
  }
  for (const auto& w : keep) {
    CHECK(gen.release_packed(w->blocks.data()));
    ref_pk.release_packed(w->blocks.data());
  }
  CHECK(gen.packed_count() == 0);
  std::printf("  %-28s %5s %11s %11s %9s\n", "implementation", "cases", "max |err|", "max rel", "err/bound");
  s_gen.print("generated int8 sx x8 (M 1-8)");
  s_ref_sx.print("reference int8 sx rows");
  s_ref_pk.print("reference int8 sx packed");
  s_ref_per32.print("reference int8 per-32 (M 1)");
  std::printf("  generated == reference packed bit for bit in %d/%d cases; == reference sx rows in %d/%d\n", same_pk,
              n_pk, same_sx, n_sx);
}

void test_strides_threads_bias() {
  const Weight w = q4kref::make_weight(24, 1280, 200);
  const int64_t m = 5, ldx = 1300, ldy = 40;
  const std::vector<float> x = q4kref::make_x(m, 1280, ldx, 201);
  std::vector<float> bias(24);
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.5f * static_cast<float>(i) - 4.0f;
  const Matmul mm = build(m, 24, 1280, true, ActFormat::kInt8, ldx, ldy);  // explicit int8 contract
  std::vector<float> y1(static_cast<size_t>(m * ldy), -7.0f), y4 = y1;
  ThreadPool p1(1), p4(4);
  GeneratedInt8Backend b1(p1, table()), b4(p4, table());
  CHECK(b1.prepack_weight(wview(w)).ok() && b4.prepack_weight(wview(w)).ok());
  CHECK(b1.run(mm.p, bind(mm, w, x, ldx, bias, y1, ldy, m)).ok());
  CHECK(b4.run(mm.p, bind(mm, w, x, ldx, bias, y4, ldy, m)).ok());
  CHECK(y1 == y4);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 24; j < ldy; ++j) CHECK(y1[static_cast<size_t>(i * ldy + j)] == -7.0f);
  }
  Stats st;
  q4kref::check_bound(w, x, ldx, bias.data(), y1.data(), ldy, m, Path::kInt8Subscale, st);
  std::printf("  strided x/y, bias, int8 contract: within bound (%.3f); 1 and 4 threads bit-identical\n", st.max_ratio);
}

void test_rejections_and_packing() {
  ThreadPool pool(2);
  GeneratedInt8Backend be(pool, table());
  const Weight w = q4kref::make_weight(24, 1280, 300);
  const std::vector<float> x = q4kref::make_x(2, 1280, 1280, 301);
  std::vector<float> y(48, -1.0f), none;
  auto untouched = [&] { return std::all_of(y.begin(), y.end(), [](float v) { return v == -1.0f; }); };
  // Contracts and shapes the generated family does not cover.
  CHECK(be.check(build(2, 24, 1280, false, ActFormat::kF32).p).message().find("does not allow int8") != std::string::npos);
  CHECK(be.check(build(9, 24, 1280, false).p).code() == StatusCode::kUnsupported);
  CHECK(be.check(build(2, 33, 256, false).p).message().find("multiple of 8") != std::string::npos);
  CHECK(be.check(build(2, 24, 512, false).p).message().find("in this build's table") != std::string::npos);
  {
    ProgramBuilder pb;
    const ValueId vx = pb.activation("x", DType::kF32, 2, 256);
    const ValueId vw = pb.weight("w", DType::kQ6_K, 8, 256);
    pb.matmul(vx, vw);
    CHECK(be.check(*pb.finish()).message().find("take q4_K weights") != std::string::npos);
  }
  const Matmul mm = build(2, 24, 1280, false);
  // Not prepacked: refused, nothing written.
  Status s = be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2));
  CHECK(s.code() == StatusCode::kInvalidArgument && s.message().find("not prepacked") != std::string::npos && untouched());
  // Prepack validation.
  const Weight odd = q4kref::make_weight(12, 256, 302);
  CHECK(be.prepack_weight(view2(odd.blocks.data(), DType::kQ4_K, 12, 256)).status().code() == StatusCode::kUnsupported);
  CHECK(be.prepack_weight(view2(reinterpret_cast<const char*>(w.blocks.data()) + 1, DType::kQ4_K, 8, 256)).status().code() ==
        StatusCode::kInvalidArgument);
  CHECK(be.prepack_weight(wview(w), 24 * 5 * 144 - 1).status().code() == StatusCode::kInvalidArgument);
  CHECK(be.prepack_weight(TensorView()).status().code() == StatusCode::kInvalidArgument);
  CHECK(be.prepack_weight(wview(w), 24 * 5 * 144).ok());
  CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, 8, 1280 * 3)).status().code() == StatusCode::kInvalidArgument);
  CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).ok() && !untouched());
  // Stale source: refused; restoring the bytes makes it valid again.
  Weight& mw = const_cast<Weight&>(w);
  const uint8_t saved = mw.blocks[3].qs[7];
  mw.blocks[3].qs[7] ^= 0x5A;
  std::fill(y.begin(), y.end(), -1.0f);
  s = be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2));
  CHECK(s.message().find("changed since prepack_weight") != std::string::npos && untouched());
  mw.blocks[3].qs[7] = saved;
  CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).ok());
  // Released: the copy is gone; the weight must be packed again.
  CHECK(be.release_packed(w.blocks.data()) && be.packed_count() == 0 && !be.release_packed(w.blocks.data()));
  std::fill(y.begin(), y.end(), -1.0f);
  CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).message().find("not prepacked") != std::string::npos &&
        untouched());
  CHECK(be.prepack_weight(wview(w)).ok());  // owned copies: a released address can be packed again
  std::printf("  rejections and packing: f32 contract, M 9, N 33, missing shape, q6_K, unpacked, stale, released; "
              "prepack validation\n");
}

}  // namespace

int main() {
  std::printf("test_dynakernel_int8gen (isa %s)\n", isa_name(this_isa()).data());
  test_provenance();
  test_tier_rules();
  if (this_isa() == CpuIsa::kAvx2 && DK_GENERATED_HAS_AVX2) {
#if defined(_WIN32)
    _putenv_s("DYNACORE_Q4_REPACK", "always");  // reference packing independent of free RAM
#else
    setenv("DYNACORE_Q4_REPACK", "always", 1);
#endif
    test_execution_proof();
    test_accuracy_vs_reference();
    test_strides_threads_bias();
    test_rejections_and_packing();
  }
  std::printf("all generated int8 tests passed (%d checks)\n", q4kref::g_checks);
  return 0;
}

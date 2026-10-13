// DynaKernel -> LLVM native backend (prototype): IR emission, LLVM
// verification, native compilation, ISA of the generated code, exactness
// against the generated C++ INT8 kernel, accuracy against the independent
// double-precision Q4_K decode, contract rejections, failure handling, tier
// fallback and resource lifetime. No model files.
//
// Environment:
//   DYNACORE_LLVM_C       path of LLVM-C.dll / libLLVM.so (else the default name)
//   DK_TEST_EXPECT_LLVM   1: fail if LLVM is unavailable; 0: fail if it is
//                         available (proves the fallback run really had no LLVM)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/compact_layout.h"
#include "dynacore/dynakernel/generated_backend.h"
#include "dynacore/dynakernel/llvm_backend.h"
#include "dynacore/dynakernel/llvm_jit.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "q4k_reference.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace dynacore;
using namespace dynacore::dk;
using q4kref::Path;
using q4kref::Stats;
using q4kref::Weight;

#define CHECK Q4K_CHECK

#if DK_GENERATED_HAS_AVX2
extern "C" const dynacore::dk::GeneratedInt8Kernel* dk_generated_q4k_int8_avx2_kernels(size_t* count);
#endif

namespace {

CpuIsa this_isa() { return select_isa(cpu_info().features); }

std::span<const GeneratedInt8Kernel> gen_table() {
#if DK_GENERATED_HAS_AVX2
  size_t n = 0;
  const GeneratedInt8Kernel* k = dk_generated_q4k_int8_avx2_kernels(&n);
  return {k, n};
#else
  return {};
#endif
}

constexpr LlvmLowering kLowerings[] = {LlvmLowering::kPortable, LlvmLowering::kAvx2, LlvmLowering::kAvxVnni};

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }
size_t count_of(const std::string& s, const char* what) {
  size_t n = 0;
  for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
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

bool llvm_ok() { return llvm_available().ok(); }

// --- 1. IR emission (no LLVM needed) ---------------------------------------------------------------

void test_ir_emission() {
  const Q4KMatmulSpec s{3, 16, 512, true};
  for (LlvmLowering l : kLowerings) {
    auto a = emit_q4k_int8_llvm_ir(s, l), b = emit_q4k_int8_llvm_ir(s, l);
    CHECK(a.ok() && b.ok() && *a == *b);
    const std::string sym = llvm_kernel_symbol(s, l);
    CHECK(contains(*a, ("define void @" + sym + "(").c_str()));
    CHECK(contains(*a, ("define i64 @" + sym + "_calls()").c_str()));
    CHECK(count_of(*a, "call <8 x float> @llvm.fma.v8f32") == 3 * 10);  // per row: 2 halves x (4 minimum terms + 1)
    CHECK(contains(*a, "pmadd.ub.sw") == (l == LlvmLowering::kAvx2));
    CHECK(contains(*a, "vpdpbusd") == (l == LlvmLowering::kAvxVnni));
    CHECK(!contains(*a, "llvm.x86") == (l == LlvmLowering::kPortable));
    CHECK(!contains(*a, "fast") && !contains(*a, "contract"));  // no relaxed floating point
  }
  CHECK(llvm_kernel_symbol({8, 1536, 1536, false}, LlvmLowering::kAvx2) == "dk_llvm_q4k_sx8_avx2_m8_n1536_k1536_b0");
  for (const Q4KMatmulSpec& bad : {Q4KMatmulSpec{9, 8, 256, false}, Q4KMatmulSpec{0, 8, 256, false},
                                   Q4KMatmulSpec{2, 12, 256, false}, Q4KMatmulSpec{2, 8, 300, false},
                                   Q4KMatmulSpec{2, 8, 0, false}}) {
    CHECK(emit_q4k_int8_llvm_ir(bad, LlvmLowering::kAvx2).status().code() == StatusCode::kUnsupported);
  }
  CHECK(emit_q4k_int8_llvm_ir(s, static_cast<LlvmLowering>(7)).status().code() == StatusCode::kInvalidArgument);
  CHECK(llvm_feature_enabled("+sse2,+avx2,-avx512f", "avx2") && !llvm_feature_enabled("+sse2,+avx2,-avx512f", "avx512f"));
  CHECK(!llvm_feature_enabled("+avx2,-avx2", "avx2") && !llvm_feature_enabled("+avx2", "avx") &&
        llvm_feature_enabled("+avxvnni", "avxvnni"));
  std::printf("  IR emission: deterministic for 3 lowerings; intrinsics per lowering; strict fp; rejects M 9, M 0, "
              "N 12, K 300, K 0, unknown lowering\n");
}

// --- 2. availability and fallback ------------------------------------------------------------------

void test_availability(const char* expect) {
  const LlvmRuntimeInfo& i = llvm_runtime();
  if (i.available) {
    std::printf("  LLVM %u.%u.%u from %s; host %s, cpu %s; avx2 %d fma %d avxvnni %d avx512f %d\n", i.major, i.minor,
                i.patch, i.library.c_str(), i.host_triple.c_str(), i.host_cpu.c_str(),
                llvm_feature_enabled(i.host_features, "avx2"), llvm_feature_enabled(i.host_features, "fma"),
                llvm_feature_enabled(i.host_features, "avxvnni"), llvm_feature_enabled(i.host_features, "avx512f"));
  } else {
    std::printf("  LLVM unavailable: %s\n", i.error.c_str());
  }
  if (expect != nullptr) CHECK(i.available == (expect[0] == '1'));
  CHECK(llvm_available().ok() == i.available);
  if (!i.available) {
    CHECK(llvm_available().code() == StatusCode::kUnsupported);
    CHECK(LlvmJit::create().status().code() == StatusCode::kUnsupported);
  }
}

// Whatever is missing (LLVM, the AVX2 tier, VNNI), the backend refuses before
// writing anything, and the reference backend computes the result.
void test_fallback() {
  ThreadPool pool(2);
  const Weight w = q4kref::make_weight(24, 1280, 11);
  const std::vector<float> x = q4kref::make_x(2, 1280, 1280, 12);
  const Matmul mm = build(2, 24, 1280, false);
  int refused = 0;
  for (LlvmLowering l : kLowerings) {
    LlvmInt8Backend be(pool, l);
    const Status r = llvm_lowering_runnable(l);
    if (r.ok()) continue;
    std::vector<float> y(48, -4.0f), none;
    CHECK(be.check(mm.p).code() == StatusCode::kUnsupported);
    CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).code() == StatusCode::kUnsupported);
    CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == -4.0f; }));
    CHECK(be.compiled().empty());
    ReferenceBackend ref(pool, this_isa());
    CHECK(ref.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).ok());
    Stats st;
    q4kref::check_bound(w, x, 1280, nullptr, y.data(), 24, 2, this_isa() == CpuIsa::kAvx2 ? Path::kInt8Subscale : Path::kFp32,
                        st);
    std::printf("  fallback (%s): refused (%s); reference backend result within bound\n",
                llvm_lowering_name(l).data(), r.message().c_str());
    ++refused;
  }
  if (refused == 0) std::printf("  fallback: every lowering runnable here (refusal paths exercised by the no-LLVM run)\n");
}

// --- 3. compilation, verification and failure handling ---------------------------------------------

const char* kBadSyntax = "define void @f( {\n";
// Parses, but %b is used where it is not defined (dominance): only the verifier rejects it.
const char* kBadVerify =
    "define void @g(i1 %c) {\nentry:\n  br i1 %c, label %a, label %b\na:\n  %v = add i32 1, 2\n  br label %b\n"
    "b:\n  %u = add i32 %v, 1\n  ret void\n}\n";

void test_compile_failures() {
  auto j = LlvmJit::create(llvm_tier_target());  // runs on every tier: portable lowering below
  CHECK(j.ok());
  LlvmJit& jit = **j;
  Status s = jit.compile(kBadSyntax, "f").status();
  CHECK(s.code() == StatusCode::kInternal && contains(s.message(), "rejected the generated IR"));
  s = jit.compile(kBadVerify, "g").status();
  CHECK(s.code() == StatusCode::kInternal && contains(s.message(), "verification"));
  // Valid IR without the `<symbol>_calls` entry: code generation succeeds, the
  // lookup fails, and the module is removed again.
  const Q4KMatmulSpec spec{1, 8, 256, false};
  const std::string sym = llvm_kernel_symbol(spec, LlvmLowering::kPortable);
  std::string ir = *emit_q4k_int8_llvm_ir(spec, LlvmLowering::kPortable);
  std::string no_calls = ir;
  const size_t at = no_calls.find("define i64 @" + sym + "_calls()");
  no_calls.replace(at, std::string("define i64 @" + sym + "_calls()").size(), "define i64 @other_calls()");
  s = jit.compile(no_calls, sym).status();
  CHECK(s.code() == StatusCode::kInternal && contains(s.message(), "native code generation"));
  // After the failures the same symbol compiles cleanly (nothing was left behind)...
  LlvmCompileStats st;
  auto ok = jit.compile(ir, sym, &st);
  CHECK(ok.ok() && ok->fn != nullptr && ok->calls != nullptr);
  CHECK(st.parse_ms > 0 && st.optimize_ms > 0 && st.codegen_ms > 0 && st.total_ms >= st.codegen_ms && st.ir_bytes == ir.size());
  // ...and only once.
  CHECK(jit.compile(ir, sym).status().code() == StatusCode::kInvalidArgument);
  CHECK(jit.compile_kernel(spec, LlvmLowering::kPortable).status().code() == StatusCode::kInvalidArgument);
  // Intrinsic lowerings for a target without the instruction set: refused
  // before LLVM sees them (LLVM itself would terminate the process).
  auto base = LlvmJit::create({"x86-64", ""});
  CHECK(base.ok());
  for (LlvmLowering l : {LlvmLowering::kAvx2, LlvmLowering::kAvxVnni}) {
    CHECK(llvm_lowering_compatible((*base)->target(), l).code() == StatusCode::kUnsupported);
    CHECK((*base)->compile_kernel(spec, l).status().code() == StatusCode::kUnsupported);
    CHECK((*base)->kernel_assembly(spec, l).status().code() == StatusCode::kUnsupported);
  }
  CHECK(llvm_lowering_compatible({"x", "+avx2,+avxvnni"}, LlvmLowering::kAvxVnni).ok() &&
        llvm_lowering_compatible({"x", "+avx2,-avxvnni"}, LlvmLowering::kAvxVnni).code() == StatusCode::kUnsupported &&
        llvm_lowering_compatible({"x", ""}, LlvmLowering::kPortable).ok());
  CHECK((*base)->compile_kernel(spec, LlvmLowering::kPortable).ok());
  std::printf("  compile failures: syntax and verifier errors reported, failed lookup rolled back, duplicate refused, "
              "avx2/avxvnni IR refused for an x86-64 baseline target; 1x8x256 compiled in %.1f ms (parse %.1f, "
              "verify %.1f, O3 %.1f, codegen %.1f)\n",
              st.total_ms, st.parse_ms, st.verify_ms, st.optimize_ms, st.codegen_ms);
}

// --- 4. instruction sets of the generated code -----------------------------------------------------

void test_assembly_isa() {
  const Q4KMatmulSpec s{4, 16, 512, false};
  for (LlvmLowering l : {LlvmLowering::kAvx2, LlvmLowering::kAvxVnni, LlvmLowering::kPortable}) {
    if (!llvm_lowering_runnable(l).ok()) continue;
    auto j = LlvmJit::create(llvm_lowering_target(l));
    CHECK(j.ok());
    auto as = (*j)->kernel_assembly(s, l);
    CHECK(as.ok());
    const size_t mbw = count_of(*as, "vpmaddubsw"), dp = count_of(*as, "vpdpbusd"), mwd = count_of(*as, "vpmaddwd"),
                 mlw = count_of(*as, "vpmullw"), fma = count_of(*as, "vfmadd") + count_of(*as, "vfnmadd");
    CHECK(contains(*as, "ymm"));
    CHECK(fma > 0);
    // Per super-block: 4 rows x 8 sub-blocks x 8 chunks (more if LLVM unrolls the super-block loop).
    constexpr size_t kPerSb = 4 * 8 * 8;
    if (l == LlvmLowering::kAvx2) CHECK(mbw > 0 && mbw % kPerSb == 0 && dp == 0);
    if (l == LlvmLowering::kAvxVnni) CHECK(dp > 0 && dp % kPerSb == 0 && mbw == 0);
    std::printf("  asm %-8s (%s): vpmaddubsw %zu, vpdpbusd %zu, vpmaddwd %zu, vpmullw %zu, fma %zu\n",
                llvm_lowering_name(l).data(), (*j)->target().cpu.c_str(), mbw, dp, mwd, mlw, fma);
  }
  // Portable IR compiled for the x86-64 baseline: SSE2 only, fused multiply-adds by libm call.
  auto base = LlvmJit::create({"x86-64", ""});
  CHECK(base.ok());
  auto as = (*base)->kernel_assembly(s, LlvmLowering::kPortable);
  CHECK(as.ok());
  CHECK(!contains(*as, "ymm") && !contains(*as, "zmm") && !contains(*as, "vfmadd") && !contains(*as, "vpmadd"));
  CHECK(contains(*as, "fmaf") && contains(*as, "xmm"));
  std::printf("  asm portable (x86-64 baseline): xmm only, no ymm/zmm or VEX multiply-adds, fma via fmaf calls\n");
}

// --- 5. exactness and accuracy ---------------------------------------------------------------------

// Every test specialization, every runnable lowering: bit-identical to the
// generated C++ INT8 kernel (same codes, same fp32 sequence), and within the
// sub-scaled int8 bound of the double-precision decode.
void test_exactness() {
  ThreadPool pool(4);
  GeneratedInt8Backend gen(pool, gen_table());
  for (LlvmLowering l : kLowerings) {
    if (!llvm_lowering_runnable(l).ok()) continue;
    LlvmInt8Backend be(pool, l);
    Stats st;
    int same = 0, cases = 0;
    double compile_ms = 0;
    uint32_t seed = 500;
    std::vector<std::unique_ptr<Weight>> keep;
    for (const GeneratedInt8Kernel& g : gen_table()) {
      const Q4KMatmulSpec& s = g.spec;
      keep.push_back(std::make_unique<Weight>(q4kref::make_weight(s.n, s.k, seed++)));
      const Weight& w = *keep.back();
      const std::vector<float> x = q4kref::make_x(s.m, s.k, s.k, seed++);
      std::vector<float> bias(static_cast<size_t>(s.n));
      q4kref::Rng r(seed++);
      for (float& v : bias) v = static_cast<float>(r.uniform(-2, 2));
      const Matmul mm = build(s.m, s.n, s.k, s.bias);
      CHECK(gen.prepack_weight(wview(w)).ok() && be.prepack_weight(wview(w)).ok());
      std::vector<float> yg(static_cast<size_t>(s.m * s.n)), yl(yg.size(), std::nanf(""));
      CHECK(gen.run(mm.p, bind(mm, w, x, s.k, bias, yg, s.n, s.m)).ok());
      CHECK(be.run(mm.p, bind(mm, w, x, s.k, bias, yl, s.n, s.m)).ok());
      q4kref::check_bound(w, x, s.k, s.bias ? bias.data() : nullptr, yl.data(), s.n, s.m, Path::kInt8Subscale, st);
      CHECK(std::memcmp(yg.data(), yl.data(), yg.size() * sizeof(float)) == 0);
      same += 1;
      ++cases;
      compile_ms += be.compiled().back().stats.total_ms;
    }
    for (const auto& w : keep) CHECK(gen.release_packed(w->blocks.data()) && be.release_packed(w->blocks.data()));
    CHECK(be.compiled().size() == gen_table().size() && be.packed_count() == 0);
    char name[64];
    std::snprintf(name, sizeof(name), "llvm %s (M 1-8)", llvm_lowering_name(l).data());
    st.print(name);
    std::printf("    bit-identical to the generated C++ int8 kernel in %d/%d cases; %zu kernels compiled in %.0f ms "
                "(%.1f ms each)\n",
                same, cases, be.compiled().size(), compile_ms, compile_ms / std::max(1, cases));
  }
}

// The portable lowering compiled for the x86-64 baseline (SSE2, fmaf calls)
// called directly: same bits as the AVX2 C++ kernel.
void test_baseline_execution() {
  const Q4KMatmulSpec s{3, 24, 1280, true};
  auto j = LlvmJit::create({"x86-64", ""});
  CHECK(j.ok());
  auto c = (*j)->compile_kernel(s, LlvmLowering::kPortable);
  CHECK(c.ok());
  const Weight w = q4kref::make_weight(s.n, s.k, 41);
  const std::vector<float> x = q4kref::make_x(s.m, s.k, s.k, 42);
  std::vector<float> bias(static_cast<size_t>(s.n), 0.25f);
  const int64_t sb = s.k / 256, groups = s.n / 8;
  std::vector<BlockQ4_Kx8> packed(static_cast<size_t>(groups * sb));
  for (int64_t g = 0; g < groups; ++g) {
    repack_q4_K_x8(w.blocks.data(), dtype_row_bytes(DType::kQ4_K, s.k), 8 * g, s.k, packed.data() + g * sb);
  }
  const CpuKernels k = make_cpu_kernels(CpuIsa::kAvx2);
  std::vector<ActBlockQ8> act(static_cast<size_t>(s.m * (s.k / 32)));
  for (int64_t r = 0; r < s.m; ++r) k.quantize_act_sx(x.data() + r * s.k, act.data() + r * (s.k / 32), s.k);
  const auto* xa = reinterpret_cast<const ActSuperQ8*>(act.data());
  std::vector<float> yb(static_cast<size_t>(s.m * s.n)), yg(yb.size());
  auto fn = reinterpret_cast<GeneratedQ4KInt8Fn>(c->fn);
  auto calls = reinterpret_cast<uint64_t (*)()>(c->calls);
  fn(packed.data(), xa, bias.data(), yb.data(), s.n, 0, groups);
  fn(packed.data(), xa, bias.data(), yb.data(), s.n, 1, 1);  // empty range: writes nothing
  CHECK(calls() == 2);
  const GeneratedInt8Kernel* ref = nullptr;
  for (const GeneratedInt8Kernel& g : gen_table()) ref = g.spec == s ? &g : ref;
  CHECK(ref != nullptr);
  ref->fn(packed.data(), xa, bias.data(), yg.data(), s.n, 0, groups);
  CHECK(std::memcmp(yb.data(), yg.data(), yb.size() * sizeof(float)) == 0);
  Stats st;
  q4kref::check_bound(w, x, s.k, bias.data(), yb.data(), s.n, s.m, Path::kInt8Subscale, st);
  std::printf("  baseline x86-64 (SSE2) portable kernel: bit-identical to the AVX2 C++ kernel; native call counter %llu\n",
              static_cast<unsigned long long>(calls()));
}

// --- 6. backend contract ---------------------------------------------------------------------------

void test_contract() {
  ThreadPool pool(2);
  LlvmInt8Backend be(pool, LlvmLowering::kAvx2);
  const Weight w = q4kref::make_weight(24, 1280, 300);
  const std::vector<float> x = q4kref::make_x(2, 1280, 1280, 301);
  std::vector<float> y(48, -1.0f), none;
  auto untouched = [&] { return std::all_of(y.begin(), y.end(), [](float v) { return v == -1.0f; }); };
  CHECK(contains(be.check(build(2, 24, 1280, false, ActFormat::kF32).p).message(), "does not allow int8"));
  CHECK(be.check(build(9, 24, 1280, false).p).code() == StatusCode::kUnsupported);
  CHECK(contains(be.check(build(2, 33, 256, false).p).message(), "multiple of 8"));
  {
    ProgramBuilder pb;
    pb.matmul(pb.activation("x", DType::kF32, 2, 256), pb.weight("w", DType::kQ6_K, 8, 256));
    CHECK(contains(be.check(*pb.finish()).message(), "take q4_K weights"));
  }
  CHECK(be.compiled().empty());  // check() compiles nothing
  const Matmul mm = build(2, 24, 1280, false);
  CHECK(be.prepare(mm.p).ok() && be.compiled().size() == 1 && be.prepare(mm.p).ok() && be.compiled().size() == 1);
  // Not prepacked: refused, nothing written.
  Status s = be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2));
  CHECK(s.code() == StatusCode::kInvalidArgument && contains(s.message(), "not prepacked") && untouched());
  CHECK(be.prepack_weight(wview(w)).ok());
  const GeneratedInt8Kernel* g = be.find(mm.p.ops()[0]);
  CHECK(g != nullptr && std::string(g->symbol) == "dk_llvm_q4k_sx8_avx2_m2_n24_k1280_b0");
  const uint64_t c0 = g->calls();
  CHECK(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).ok() && !untouched());
  CHECK(g->calls() - c0 == 3);  // 3 groups of 8 rows, grain 1: counted by the native code
  // Stale source: refused.
  Weight& mw = const_cast<Weight&>(w);
  mw.blocks[2].qs[5] ^= 0x33;
  std::fill(y.begin(), y.end(), -1.0f);
  CHECK(contains(be.run(mm.p, bind(mm, w, x, 1280, none, y, 24, 2)).message(), "changed since prepack_weight") && untouched());
  mw.blocks[2].qs[5] ^= 0x33;
  CHECK(be.target().backend == "llvm-int8" && be.target().isa == "avx2" && be.target().int8_dot);
  // Compiled shape, run() fast path (no prepare): the executor still enforces
  // the contract and the program; nothing is written.
  std::fill(y.begin(), y.end(), -1.0f);
  const Matmul mf = build(2, 24, 1280, false, ActFormat::kF32);
  CHECK(be.find(mf.p.ops()[0]) != nullptr);
  CHECK(be.run(mf.p, bind(mf, w, x, 1280, none, y, 24, 2)).code() == StatusCode::kUnsupported && untouched());
  {
    Program bad = mm.p;
    bad.add_value(Value{kNoValue, "dangling", ValueRole::kResult, *TensorLayout::contiguous(DType::kF32, TensorShape{2, 24})});
    CHECK(be.run(bad, bind(mm, w, x, 1280, none, y, 24, 2)).code() != StatusCode::kOk && untouched());
  }
  // Strided x / y, bias, explicit int8 contract; 1 and 4 threads identical.
  const int64_t m = 5, ldx = 1300, ldy = 40;
  const std::vector<float> xs = q4kref::make_x(m, 1280, ldx, 302);
  std::vector<float> bias(24);
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.5f * static_cast<float>(i) - 4.0f;
  const Matmul ms = build(m, 24, 1280, true, ActFormat::kInt8, ldx, ldy);
  std::vector<float> y1(static_cast<size_t>(m * ldy), -7.0f), y4 = y1;
  ThreadPool p1(1), p4(4);
  LlvmInt8Backend b1(p1), b4(p4);
  CHECK(b1.prepack_weight(wview(w)).ok() && b4.prepack_weight(wview(w)).ok());
  CHECK(b1.run(ms.p, bind(ms, w, xs, ldx, bias, y1, ldy, m)).ok() && b4.run(ms.p, bind(ms, w, xs, ldx, bias, y4, ldy, m)).ok());
  CHECK(y1 == y4);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 24; j < ldy; ++j) CHECK(y1[static_cast<size_t>(i * ldy + j)] == -7.0f);
  }
  Stats st;
  q4kref::check_bound(w, xs, ldx, bias.data(), y1.data(), ldy, m, Path::kInt8Subscale, st);
  std::printf("  contract: f32, M 9, N 33, q6_K refused; prepare idempotent; unpacked and stale refused; native call "
              "counter 3; strided/bias/int8 within bound (%.3f), 1 == 4 threads\n",
              st.max_ratio);
}

// --- 7. resource lifetime --------------------------------------------------------------------------

size_t private_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    return pmc.PrivateUsage;
  }
#endif
  return 0;
}

void test_lifetime() {
  ThreadPool pool(2);
  const Weight w = q4kref::make_weight(16, 768, 77);
  const std::vector<float> x = q4kref::make_x(4, 768, 768, 78);
  const Matmul a = build(4, 16, 768, false), b = build(1, 16, 768, false);
  std::vector<float> ref(64), none;
  const int live0 = LlvmJit::live_instances();
  auto cycle = [&](std::vector<float>& y) {
    LlvmInt8Backend be(pool);
    CHECK(be.prepack_weight(wview(w)).ok());
    CHECK(be.run(a.p, bind(a, w, x, 768, none, y, 16, 4)).ok());
    std::vector<float> y1(16);
    CHECK(be.run(b.p, bind(b, w, x, 768, none, y1, 16, 1)).ok());
    CHECK(be.compiled().size() == 2 && LlvmJit::live_instances() == live0 + 1);
  };
  cycle(ref);  // warm-up: LLVM's one-time allocations
  CHECK(LlvmJit::live_instances() == live0);
  const size_t before = private_bytes();
  constexpr int kCycles = 20;
  for (int i = 0; i < kCycles; ++i) {
    std::vector<float> y(64);
    cycle(y);
    CHECK(y == ref);
  }
  const size_t after = private_bytes();
  CHECK(LlvmJit::live_instances() == live0);
  const double growth_mb = (static_cast<double>(after) - static_cast<double>(before)) / (1024.0 * 1024.0);
  if (before != 0) CHECK(growth_mb < 16.0);
  std::printf("  lifetime: %d create/compile/run/destroy cycles, live JITs back to %d, private bytes %+.2f MB\n", kCycles,
              live0, growth_mb);
}

// --- 8. experimental parallel activation quantization -------------------------------------------

// Serial and parallel quantization give identical outputs (same quantizer on
// the same super-blocks) for odd super-block counts, strided rows, bias,
// several pool sizes and thresholds, both executors; repeated runs agree.
void test_parallel_quant() {
  struct Shape {
    int64_t m, n, k, ldx;
    bool bias;
  };
  // 9 super-blocks x 3 rows (chunks span rows), 35 x 8, strided x, 1 super-block x 2.
  const Shape shapes[] = {{3, 64, 2304, 2304, false}, {8, 24, 8960, 8960, true}, {5, 24, 1280, 1300, true}, {2, 8, 256, 256, false}};
  int cases = 0, parallel_regions = 0;
  for (const Shape& sh : shapes) {
    const Weight w = q4kref::make_weight(sh.n, sh.k, static_cast<uint32_t>(700 + sh.k));
    const std::vector<float> x = q4kref::make_x(sh.m, sh.k, sh.ldx, static_cast<uint32_t>(800 + sh.k));
    std::vector<float> bias(static_cast<size_t>(sh.n));
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.25f * static_cast<float>(i) - 1.0f;
    const Matmul mm = build(sh.m, sh.n, sh.k, sh.bias, ActFormat::kAuto, sh.ldx, 0);
    std::vector<float> y_serial(static_cast<size_t>(sh.m * sh.n));
    for (int threads : {1, 4, 10}) {
      ThreadPool pool(threads);
      LlvmInt8Backend be(pool);
      CHECK(be.prepack_weight(wview(w)).ok());
      RunOptions serial;
      CHECK(be.run(mm.p, bind(mm, w, x, sh.ldx, bias, y_serial, sh.n, sh.m), serial).ok());
      Stats st;
      q4kref::check_bound(w, x, sh.ldx, sh.bias ? bias.data() : nullptr, y_serial.data(), sh.n, sh.m, Path::kInt8Subscale, st);
      GeneratedInt8Backend gen(pool, gen_table());
      const bool has_gen = gen.find(mm.p.ops()[0]) != nullptr;
      if (has_gen) CHECK(gen.prepack_weight(wview(w)).ok());
      for (int64_t threshold : {int64_t{0}, int64_t{96}, int64_t{1} << 40}) {
        RunOptions par;
        par.experimental_parallel_quant = true;
        par.parallel_quant_min_superblocks = threshold;
        const uint64_t misuse0 = pool.misuse_count();
        for (int rep = 0; rep < 5; ++rep) {
          std::vector<float> y(y_serial.size(), std::nanf(""));
          CHECK(be.run(mm.p, bind(mm, w, x, sh.ldx, bias, y, sh.n, sh.m), par).ok());
          CHECK(std::memcmp(y.data(), y_serial.data(), y.size() * sizeof(float)) == 0);
          if (has_gen) {
            std::vector<float> yg(y_serial.size(), std::nanf(""));
            CHECK(gen.run(mm.p, bind(mm, w, x, sh.ldx, bias, yg, sh.n, sh.m), par).ok());
            CHECK(std::memcmp(yg.data(), y_serial.data(), yg.size() * sizeof(float)) == 0);
          }
          ++cases;
        }
        CHECK(pool.misuse_count() == misuse0);
        parallel_regions += threads > 1 && sh.m * (sh.k / 256) >= threshold;
      }
    }
  }
  std::printf("  parallel quantization: %d runs bit-identical to serial (4 shapes incl. strided x and row-spanning "
              "chunks, 1/4/10 threads, thresholds 0/96/off, llvm and generated executors); %d configs took the "
              "parallel path; no pool misuse\n",
              cases, parallel_regions);
}

// Concurrent callers. Separate pools and backends on separate threads: all
// exact. Two callers sharing one pool: the pool contract is reported
// (kInternal) whenever a region found the pool busy, and outputs stay exact.
void test_concurrent_callers() {
  const Weight w = q4kref::make_weight(64, 2304, 901);
  const std::vector<float> x = q4kref::make_x(8, 2304, 2304, 902);
  const Matmul mm = build(8, 64, 2304, false);
  std::vector<float> y_ref(8 * 64), none;
  {
    ThreadPool pool(1);
    LlvmInt8Backend be(pool);
    CHECK(be.prepack_weight(wview(w)).ok() && be.run(mm.p, bind(mm, w, x, 2304, none, y_ref, 64, 8)).ok());
  }
  RunOptions par;
  par.experimental_parallel_quant = true;
  par.parallel_quant_min_superblocks = 0;
  constexpr int kThreads = 4, kIters = 40;
  {
    std::vector<int> bad(kThreads, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
      ts.emplace_back([&, t] {
        ThreadPool pool(3);
        LlvmInt8Backend be(pool);
        if (!be.prepack_weight(wview(w)).ok()) ++bad[static_cast<size_t>(t)];
        for (int i = 0; i < kIters; ++i) {
          std::vector<float> y(y_ref.size());
          const Status s = be.run(mm.p, bind(mm, w, x, 2304, none, y, 64, 8), par);
          if (!s.ok() || y != y_ref) ++bad[static_cast<size_t>(t)];
        }
      });
    }
    for (auto& th : ts) th.join();
    for (int b : bad) CHECK(b == 0);
  }
  int reported = 0, exact = 0;
  {
    ThreadPool shared(4);
    LlvmInt8Backend a(shared), b(shared);
    CHECK(a.prepack_weight(wview(w)).ok() && b.prepack_weight(wview(w)).ok());
    CHECK(a.prepare(mm.p).ok() && b.prepare(mm.p).ok());  // compile first: only execution overlaps
    std::vector<int> rep(2, 0), ok(2, 0), wrong(2, 0);
    auto caller = [&](LlvmInt8Backend& be, int id) {
      for (int i = 0; i < kIters; ++i) {
        std::vector<float> y(y_ref.size());
        const Status s = be.run(mm.p, bind(mm, w, x, 2304, none, y, 64, 8), par);
        if (s.ok()) ++ok[static_cast<size_t>(id)];
        else if (s.code() == StatusCode::kInternal && contains(s.message(), "thread pool contract")) ++rep[static_cast<size_t>(id)];
        else ++wrong[static_cast<size_t>(id)];
        if (y != y_ref) ++wrong[static_cast<size_t>(id)];
      }
    };
    std::thread t1(caller, std::ref(a), 0), t2(caller, std::ref(b), 1);
    t1.join();
    t2.join();
    CHECK(wrong[0] == 0 && wrong[1] == 0);
    reported = rep[0] + rep[1];
    exact = ok[0] + ok[1] + reported;
  }
  std::printf("  concurrent callers: %d threads x %d runs on separate pools exact; shared pool: %d/%d runs exact, %d "
              "reported the pool contract violation\n",
              kThreads, kIters, exact, 2 * kIters, reported);
}

// --- 9. experimental prefetch option (Phase 3) ------------------------------------------------------

void test_prefetch_option() {
  const Q4KMatmulSpec s{4, 24, 1280, true};
  for (LlvmLowering l : kLowerings) {
    CHECK(*emit_q4k_int8_llvm_ir(s, l, LlvmKernelOptions{}) == *emit_q4k_int8_llvm_ir(s, l));  // defaults: same IR
    CHECK(llvm_kernel_symbol(s, l, LlvmKernelOptions{}) == llvm_kernel_symbol(s, l));
  }
  const LlvmKernelOptions pf{2432};
  const std::string ir = *emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2, pf);
  CHECK(count_of(ir, "call void @llvm.prefetch.p0") == 19);  // one per cache line of a 1216-byte block
  CHECK(llvm_kernel_symbol(s, LlvmLowering::kAvx2, pf) == "dk_llvm_q4k_sx8_avx2_m4_n24_k1280_b1_pf2432");
  for (int bad : {-64, 100, (1 << 20) + 64}) {
    CHECK(emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2, LlvmKernelOptions{bad}).status().code() == StatusCode::kInvalidArgument);
  }
  if (!llvm_ok() || !llvm_lowering_runnable(LlvmLowering::kAvx2).ok()) {
    std::printf("  prefetch option: IR checks only (LLVM or avx2 unavailable)\n");
    return;
  }
  auto j = LlvmJit::create(llvm_lowering_target(LlvmLowering::kAvx2));
  CHECK(j.ok());
  auto as = (*j)->kernel_assembly(s, LlvmLowering::kAvx2, pf);
  CHECK(as.ok() && count_of(*as, "prefetcht0") > 0);
  // Bit-identical to the control on several shapes, both called directly.
  const CpuKernels k = make_cpu_kernels(CpuIsa::kAvx2);
  int cases = 0;
  for (const Q4KMatmulSpec& sp : {Q4KMatmulSpec{4, 24, 1280, true}, Q4KMatmulSpec{8, 64, 2304, false}, Q4KMatmulSpec{5, 16, 768, false}}) {
    auto c0 = (*j)->compile_kernel(sp, LlvmLowering::kAvx2);
    CHECK(c0.ok());
    void* f0 = c0->fn;
    for (int dist : {64, 1216, 2432, 9728}) {
      auto c1 = (*j)->compile_kernel(sp, LlvmLowering::kAvx2, nullptr, LlvmKernelOptions{dist});
      CHECK(c1.ok());
      const Weight w = q4kref::make_weight(sp.n, sp.k, static_cast<uint32_t>(1000 + dist));
      const std::vector<float> x = q4kref::make_x(sp.m, sp.k, sp.k, static_cast<uint32_t>(2000 + dist));
      std::vector<float> bias(static_cast<size_t>(sp.n), 0.5f);
      const int64_t sb = sp.k / 256, groups = sp.n / 8;
      std::vector<BlockQ4_Kx8> packed(static_cast<size_t>(groups * sb));
      for (int64_t g = 0; g < groups; ++g) {
        repack_q4_K_x8(w.blocks.data(), dtype_row_bytes(DType::kQ4_K, sp.k), 8 * g, sp.k, packed.data() + g * sb);
      }
      std::vector<ActBlockQ8> act(static_cast<size_t>(sp.m * (sp.k / 32)));
      for (int64_t r = 0; r < sp.m; ++r) k.quantize_act_sx(x.data() + r * sp.k, act.data() + r * (sp.k / 32), sp.k);
      const auto* xa = reinterpret_cast<const ActSuperQ8*>(act.data());
      std::vector<float> y0(static_cast<size_t>(sp.m * sp.n)), y1(y0.size(), std::nanf(""));
      const float* bp = sp.bias ? bias.data() : nullptr;
      reinterpret_cast<GeneratedQ4KInt8Fn>(f0)(packed.data(), xa, bp, y0.data(), sp.n, 0, groups);
      reinterpret_cast<GeneratedQ4KInt8Fn>(c1->fn)(packed.data(), xa, bp, y1.data(), sp.n, 0, groups);
      CHECK(std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) == 0);
      Stats st;
      q4kref::check_bound(w, x, sp.k, bp, y1.data(), sp.n, sp.m, Path::kInt8Subscale, st);
      ++cases;
    }
  }
  std::printf("  prefetch option: default IR unchanged, 19 prefetches per super-block, invalid distances refused, "
              "prefetcht0 in the assembly; %d shape x distance cases bit-identical to the control\n",
              cases);
}

// --- 10. experimental compact packed layouts (Phase 4) -----------------------------------------------

constexpr PackedLayout kCompact[] = {PackedLayout::kX8Fp16, PackedLayout::kX8Scales6, PackedLayout::kX8Fp16Scales6};

// Weights with the edge cases the layouts must carry exactly: make_weight's
// d = 0 blocks (every 7th) and saturated 6-bit scales/mins (every 11th),
// plus fp16 subnormal, negative-zero and extreme scales in the first blocks.
Weight edge_weight(int64_t n, int64_t k, uint32_t seed) {
  Weight w = q4kref::make_weight(n, k, seed);
  const uint16_t specials[][2] = {{0x0001, 0x8000}, {0x8001, 0x03FF}, {0x7BFF, 0xFBFF}, {0x0400, 0x0000}};
  for (size_t i = 0; i < 4 && i < w.blocks.size(); ++i) {
    w.blocks[i].d = specials[i][0];
    w.blocks[i].dmin = specials[i][1];
    // Recompute the exact reference values of the changed block (as make_weight does).
    const double d = fp16_to_fp32(w.blocks[i].d), dmin = fp16_to_fp32(w.blocks[i].dmin);
    const int64_t row = static_cast<int64_t>(i) / (k / 256), col0 = (static_cast<int64_t>(i) % (k / 256)) * 256;
    for (int g = 0; g < 4; ++g) {
      for (int half = 0; half < 2; ++half) {
        int sc = 0, m = 0;
        q4kref::scale_min(w.blocks[i].scales, 2 * g + half, sc, m);
        for (int l = 0; l < 32; ++l) {
          const int q = half == 0 ? (w.blocks[i].qs[32 * g + l] & 0x0F) : (w.blocks[i].qs[32 * g + l] >> 4);
          const size_t at = static_cast<size_t>(row * k + col0 + 64 * g + 32 * half + l);
          w.w[at] = d * sc * q - dmin * m;
          w.mag[at] = std::abs(d * sc * q) + std::abs(dmin * m);
        }
      }
    }
  }
  return w;
}

// Every compact layout decodes to exactly BlockQ4_Kx8's values, element by element.
void test_layout_packing() {
  CHECK(packed_layout_info(PackedLayout::kX8).block_bytes == 1216 && packed_layout_info(PackedLayout::kX8Fp16).block_bytes == 1184 &&
        packed_layout_info(PackedLayout::kX8Scales6).block_bytes == 1184 &&
        packed_layout_info(PackedLayout::kX8Fp16Scales6).block_bytes == 1152);
  CHECK(!valid_packed_layout(static_cast<PackedLayout>(9)));
  int64_t elements = 0;
  for (const auto& sh : {std::pair<int64_t, int64_t>{8, 256}, {24, 1280}, {64, 2304}}) {
    const Weight w = edge_weight(sh.first, sh.second, static_cast<uint32_t>(sh.second));
    const int64_t sb = w.k / 256, rb = dtype_row_bytes(DType::kQ4_K, w.k);
    std::vector<BlockQ4_Kx8> x8(static_cast<size_t>(sb));
    for (int64_t g = 0; g < w.n / 8; ++g) {
      repack_q4_K_x8(w.blocks.data(), rb, 8 * g, w.k, x8.data());
      for (PackedLayout l : kCompact) {
        std::vector<uint8_t> buf(static_cast<size_t>(sb * packed_layout_info(l).block_bytes) + 1, 0xA5);
        repack_q4_K_x8_layout(l, w.blocks.data(), rb, 8 * g, w.k, buf.data());
        CHECK(buf.back() == 0xA5);  // writes exactly k / 256 blocks
        for (int r = 0; r < 8; ++r) {
          for (int64_t i = 0; i < w.k; ++i) {
            const float a = packed_q4_K_x8_value(x8.data(), r, i), b = packed_layout_value(l, buf.data(), r, i);
            CHECK(std::memcmp(&a, &b, 4) == 0);
            ++elements;
          }
        }
      }
    }
  }
  std::printf("  layout packing: x8 1216, h/s 1184, hs 1152 bytes per block; %lld element decodes bit-identical to "
              "BlockQ4_Kx8 (incl. d = 0, saturated scales, fp16 subnormal / -0 / max)\n",
              static_cast<long long>(elements));
}

void test_layout_ir() {
  const Q4KMatmulSpec s{2, 16, 512, false};
  CHECK(*emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2, LlvmKernelOptions{0, PackedLayout::kX8}) ==
        *emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2));
  for (PackedLayout l : kCompact) {
    const LlvmKernelOptions o{0, l};
    const std::string ir = *emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2, o);
    CHECK(contains(ir, "fpext <8 x half>") == packed_layout_info(l).fp16);
    CHECK(llvm_kernel_symbol(s, LlvmLowering::kAvx2, o) ==
          "dk_llvm_q4k_sx8_avx2_m2_n16_k512_b0_l" + std::string(packed_layout_name(l)));
  }
  CHECK(emit_q4k_int8_llvm_ir(s, LlvmLowering::kAvx2, LlvmKernelOptions{0, static_cast<PackedLayout>(9)}).status().code() ==
        StatusCode::kInvalidArgument);
  std::printf("  layout IR: default layout IR unchanged, fp16 widening only in h/hs, suffixed symbols, unknown layout refused\n");
}

// LLVM kernels on every compact layout are bit-identical to the x8 kernels,
// for all 40 INT8 test specs and every runnable lowering, through the backend.
void test_layout_exactness() {
  ThreadPool pool(4);
  int cases = 0;
  for (LlvmLowering low : kLowerings) {
    if (!llvm_lowering_runnable(low).ok()) continue;
    LlvmInt8Backend base(pool, low);
    std::vector<std::unique_ptr<LlvmInt8Backend>> lay;
    for (PackedLayout l : kCompact) lay.push_back(std::make_unique<LlvmInt8Backend>(pool, low, LlvmKernelOptions{0, l}));
    uint32_t seed = 3000;
    std::vector<std::unique_ptr<Weight>> keep;
    for (const GeneratedInt8Kernel& g : gen_table()) {
      const Q4KMatmulSpec& sp = g.spec;
      keep.push_back(std::make_unique<Weight>(edge_weight(sp.n, sp.k, seed++)));
      const Weight& w = *keep.back();
      const std::vector<float> x = q4kref::make_x(sp.m, sp.k, sp.k, seed++);
      std::vector<float> bias(static_cast<size_t>(sp.n), -0.75f);
      const Matmul mm = build(sp.m, sp.n, sp.k, sp.bias);
      std::vector<float> y0(static_cast<size_t>(sp.m * sp.n));
      CHECK(base.prepack_weight(wview(w)).ok() && base.run(mm.p, bind(mm, w, x, sp.k, bias, y0, sp.n, sp.m)).ok());
      Stats st;
      q4kref::check_bound(w, x, sp.k, sp.bias ? bias.data() : nullptr, y0.data(), sp.n, sp.m, Path::kInt8Subscale, st);
      for (auto& be : lay) {
        std::vector<float> y(y0.size(), std::nanf(""));
        auto pw = be->prepack_weight(wview(w));
        CHECK(pw.ok() && pw->packed_bytes == (sp.n / 8) * (sp.k / 256) * packed_layout_info(be->options().layout).block_bytes);
        CHECK(be->run(mm.p, bind(mm, w, x, sp.k, bias, y, sp.n, sp.m)).ok());
        CHECK(std::memcmp(y.data(), y0.data(), y.size() * sizeof(float)) == 0);
        ++cases;
      }
    }
  }
  std::printf("  layout exactness: %d (layout x lowering x spec) runs bit-identical to the x8 layout, within bound\n", cases);
}

void test_layout_rules() {
  // fp16 layouts need F16C: refused for an x86-64 baseline target; the s layout compiles there.
  auto base = LlvmJit::create({"x86-64", ""});
  CHECK(base.ok());
  const Q4KMatmulSpec s{3, 24, 1280, true};
  for (PackedLayout l : kCompact) {
    const Status c = llvm_kernel_compatible((*base)->target(), LlvmLowering::kPortable, LlvmKernelOptions{0, l});
    CHECK(c.ok() == !packed_layout_info(l).fp16);
    CHECK((*base)->compile_kernel(s, LlvmLowering::kPortable, nullptr, LlvmKernelOptions{0, l}).ok() == c.ok());
  }
  // Executor: the layout is fixed once weights are packed; stale detection and release still work.
  ThreadPool pool(2);
  LlvmInt8Backend be(pool, LlvmLowering::kAvx2, LlvmKernelOptions{0, PackedLayout::kX8Fp16Scales6});
  GeneratedInt8Backend ex(pool, gen_table());
  const Weight w = q4kref::make_weight(24, 1280, 4100);
  CHECK(ex.set_packed_layout(PackedLayout::kX8Scales6).ok() && ex.prepack_weight(wview(w)).ok());
  CHECK(ex.set_packed_layout(PackedLayout::kX8).code() == StatusCode::kInvalidArgument);
  CHECK(ex.set_packed_layout(PackedLayout::kX8Scales6).ok() && ex.release_packed(w.blocks.data()) && ex.set_packed_layout(PackedLayout::kX8).ok());
  const std::vector<float> x = q4kref::make_x(3, 1280, 1280, 4101);
  std::vector<float> bias(24, 1.0f), y(72, -1.0f);
  const Matmul mm = build(3, 24, 1280, true);
  CHECK(be.prepack_weight(wview(w)).ok() && be.run(mm.p, bind(mm, w, x, 1280, bias, y, 24, 3)).ok());
  Weight& mw = const_cast<Weight&>(w);
  mw.blocks[1].scales[3] ^= 0x41;
  std::fill(y.begin(), y.end(), -1.0f);
  CHECK(contains(be.run(mm.p, bind(mm, w, x, 1280, bias, y, 24, 3)).message(), "changed since prepack_weight"));
  CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == -1.0f; }));
  mw.blocks[1].scales[3] ^= 0x41;
  CHECK(be.release_packed(w.blocks.data()) && be.packed_count() == 0);
  std::printf("  layout rules: fp16 layouts refused without F16C (s compiles on x86-64 baseline), layout fixed while "
              "packed, stale weights refused, release works\n");
}

}  // namespace

int main() {
  std::printf("test_dynakernel_llvm (isa %s)\n", isa_name(this_isa()).data());
#if defined(_WIN32)
  _putenv_s("DYNACORE_Q4_REPACK", "always");  // reference packing independent of free RAM
#else
  setenv("DYNACORE_Q4_REPACK", "always", 1);
#endif
  test_ir_emission();
  test_availability(std::getenv("DK_TEST_EXPECT_LLVM"));
  test_fallback();
  if (llvm_ok()) {
    test_compile_failures();
    test_assembly_isa();
    if (this_isa() == CpuIsa::kAvx2 && DK_GENERATED_HAS_AVX2) {
      test_exactness();
      test_baseline_execution();
      test_contract();
      test_lifetime();
      test_parallel_quant();
      test_concurrent_callers();
      test_prefetch_option();
      test_layout_packing();
      test_layout_ir();
      test_layout_exactness();
      test_layout_rules();
    }
  }
  std::printf("all llvm backend tests passed (%d checks)\n", q4kref::g_checks);
  return 0;
}

// Experimental 8-row packed Q4_K INT8 kernel (KernelPlan::packed_int8_8rows):
// off by default; when on, 5..8 rows (and 8-row tiles beyond) share one pass
// over the interleaved weights. Rows are independent in the kernel, so the
// 8-row result must be bit-identical to the default 4-row tiles; both are also
// checked against the independent double-precision decode within the
// sub-scaled int8 bound. Evidence of execution: CpuDevice's dispatch counter.
// No model files.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "q4k_reference.h"

using namespace dynacore;
using namespace dynacore::dk;
using q4kref::Path;
using q4kref::Stats;
using q4kref::Weight;

#define CHECK Q4K_CHECK

namespace {

CpuIsa this_isa() { return select_isa(cpu_info().features); }

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
  _putenv_s(k, v);
#else
  if (v[0] == '\0') unsetenv(k); else setenv(k, v, 1);
#endif
}

TensorView view2(const void* data, DType t, int64_t r, int64_t c, int64_t row_stride_elems = 0) {
  TensorLayout l = *TensorLayout::contiguous(t, TensorShape{r, c});
  if (row_stride_elems > 0) l.strides[0] = row_stride_elems * static_cast<int64_t>(sizeof(float));
  return TensorView(const_cast<void*>(data), l);
}

struct Mm {
  Program p;
  ValueId x, w, b, y;
};

Mm build(int64_t m, int64_t n, int64_t k, bool bias, int64_t ldx = 0, int64_t ldy = 0) {
  ProgramBuilder pb;
  Mm r;
  TensorLayout xl = *TensorLayout::contiguous(DType::kF32, TensorShape{m, k});
  if (ldx > 0) xl.strides[0] = ldx * 4;
  TensorLayout yl = *TensorLayout::contiguous(DType::kF32, TensorShape{m, n});
  if (ldy > 0) yl.strides[0] = ldy * 4;
  r.x = pb.value("x", ValueRole::kActivation, xl);
  r.w = pb.weight("w", DType::kQ4_K, n, k);
  r.b = bias ? pb.bias("b", n) : kNoValue;
  r.y = pb.value("y", ValueRole::kResult, yl);
  pb.matmul_into(r.x, r.w, r.b, r.y);  // auto contract: int8 allowed
  auto p = pb.finish();
  CHECK(p.ok());
  r.p = std::move(*p);
  return r;
}

// Runs one matmul; returns y (row stride ldy) and the 8-row dispatch delta.
std::vector<float> run(ReferenceBackend& be, const Mm& mm, const Weight& w, const std::vector<float>& x, int64_t ldx,
                       const std::vector<float>& bias, int64_t m, int64_t ldy, bool rows8, uint64_t* dispatches) {
  std::vector<float> y(static_cast<size_t>(m * ldy), -9.0f);
  Bindings bi;
  bi.bind(mm.x, view2(x.data(), DType::kF32, m, w.k, ldx), static_cast<int64_t>(x.size() * 4));
  bi.bind(mm.w, view2(w.blocks.data(), DType::kQ4_K, w.n, w.k));
  if (mm.b != kNoValue) bi.bind(mm.b, TensorView(const_cast<float*>(bias.data()), *TensorLayout::contiguous(DType::kF32, TensorShape{w.n})));
  bi.bind(mm.y, view2(y.data(), DType::kF32, m, w.n, ldy), static_cast<int64_t>(y.size() * 4));
  RunOptions ro;
  ro.experimental_packed_8rows = rows8;
  const uint64_t c0 = CpuDevice::experimental_rows8_dispatches();
  const Status s = be.run(mm.p, bi, ro);
  if (!s.ok()) std::fprintf(stderr, "%s\n", s.to_string().c_str());
  CHECK(s.ok());
  if (dispatches) *dispatches = CpuDevice::experimental_rows8_dispatches() - c0;
  return y;
}

std::vector<float> make_bias(int64_t n, uint32_t seed) {
  q4kref::Rng r(seed);
  std::vector<float> b(static_cast<size_t>(n));
  for (float& v : b) v = static_cast<float>(r.uniform(-2, 2));
  return b;
}

void test_defaults() {
  CHECK(!KernelPlan{}.packed_int8_8rows);
  const char* env = std::getenv("DYNACORE_PACKED_8ROWS");
  if (env == nullptr) CHECK(!KernelPlan::defaults().packed_int8_8rows);
  CHECK(!ReferenceBackend::plan_for(MatMulAttrs{}).packed_int8_8rows || env != nullptr);
  CHECK(!RunOptions{}.experimental_packed_8rows);
  std::printf("  defaults: packed_int8_8rows off (KernelPlan, defaults(), plan_for, RunOptions)\n");
}

// Same inputs with the flag off and on: identical bits, both within bound;
// the 8-row path runs only when on and M > 4.
void test_on_off(ThreadPool& pool) {
  ReferenceBackend be(pool, this_isa());  // fresh: released addresses are never reused across sub-tests
  struct Shape {
    int64_t n, k;
    bool bias;
  };
  const Shape shapes[] = {{8, 256, false}, {24, 1280, true}, {64, 2304, false}, {1536, 256, false}};
  Stats st_off, st_on;
  int cases = 0;
  uint32_t seed = 10;
  std::vector<std::unique_ptr<Weight>> keep;
  for (const Shape& sh : shapes) {
    keep.push_back(std::make_unique<Weight>(q4kref::make_weight(sh.n, sh.k, seed++)));
    const Weight& w = *keep.back();
    CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, w.n, w.k)).ok());
    const std::vector<float> bias = make_bias(sh.n, seed++);
    for (int64_t m = 1; m <= 8; ++m) {
      const std::vector<float> x = q4kref::make_x(m, sh.k, sh.k, seed++);
      const Mm mm = build(m, sh.n, sh.k, sh.bias);
      uint64_t d_off = 0, d_on = 0;
      const std::vector<float> y_off = run(be, mm, w, x, sh.k, bias, m, sh.n, false, &d_off);
      const std::vector<float> y_on = run(be, mm, w, x, sh.k, bias, m, sh.n, true, &d_on);
      // Default: never the 8-row path, unless DYNACORE_PACKED_8ROWS=1 turned it on.
      CHECK((d_off > 0) == (KernelPlan::defaults().packed_int8_8rows && m > 4));
      CHECK((d_on > 0) == (m > 4));        // on: only above the 4-row tile
      CHECK(y_on == y_off);                // rows are independent: same bits
      const Path p = m == 1 ? Path::kInt8Per32 : Path::kInt8Subscale;  // M = 1 stays per-32 (not packed)
      q4kref::check_bound(w, x, sh.k, sh.bias ? bias.data() : nullptr, y_off.data(), sh.n, m, p, st_off);
      q4kref::check_bound(w, x, sh.k, sh.bias ? bias.data() : nullptr, y_on.data(), sh.n, m, p, st_on);
      ++cases;
    }
  }
  for (const auto& w : keep) be.release_packed(w->blocks.data());
  std::printf("  on/off over %d cases (M 1-8, 4 shapes, bias): bit-identical; 8-row dispatch only when on and M > 4\n",
              cases);
  std::printf("  %-28s %5s %11s %11s %9s\n", "path", "cases", "max |err|", "max rel", "err/bound");
  st_off.print("default (4-row tiles)");
  st_on.print("experimental (8-row tiles)");
}

// N not a multiple of 8: no interleaved copy, so the row kernels run with
// the flag on (never the 8-row path).
void test_odd_n(ThreadPool& pool) {
  ReferenceBackend be(pool, this_isa());  // fresh: released addresses are never reused across sub-tests
  for (int64_t n : {12, 33}) {
    const Weight w = q4kref::make_weight(n, 768, 50 + static_cast<uint32_t>(n));
    CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, n, 768)).status().code() == StatusCode::kUnsupported);
    const std::vector<float> x = q4kref::make_x(8, 768, 768, 60), none;
    uint64_t d = 0;
    const std::vector<float> y = run(be, build(8, n, 768, false), w, x, 768, none, 8, n, true, &d);
    CHECK(d == 0);
    Stats st;
    q4kref::check_bound(w, x, 768, nullptr, y.data(), n, 8, Path::kInt8Subscale, st);
  }
  std::printf("  odd N (12, 33): not packable; flag on falls back to the row kernel, within bound\n");
}

// Strided x and y, bias, M = 7 (one partial 8-row tile): padding untouched.
void test_strides(ThreadPool& pool) {
  ReferenceBackend be(pool, this_isa());  // fresh: released addresses are never reused across sub-tests
  const Weight w = q4kref::make_weight(24, 1280, 70);
  CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, 24, 1280)).ok());
  const int64_t m = 7, ldx = 1300, ldy = 40;
  const std::vector<float> x = q4kref::make_x(m, 1280, ldx, 71);
  const std::vector<float> bias = make_bias(24, 72);
  const Mm mm = build(m, 24, 1280, true, ldx, ldy);
  uint64_t d = 0;
  const std::vector<float> y_on = run(be, mm, w, x, ldx, bias, m, ldy, true, &d);
  const std::vector<float> y_off = run(be, mm, w, x, ldx, bias, m, ldy, false, nullptr);
  CHECK(d > 0 && y_on == y_off);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 24; j < ldy; ++j) CHECK(y_on[static_cast<size_t>(i * ldy + j)] == -9.0f);
  }
  Stats st;
  q4kref::check_bound(w, x, ldx, bias.data(), y_on.data(), ldy, m, Path::kInt8Subscale, st);
  be.release_packed(w.blocks.data());
  std::printf("  strided x/y + bias, M = 7: identical, padding untouched, within bound (%.3f)\n", st.max_ratio);
}

// More than 8 rows on the interleaved copy (the opt-in prefill path,
// DYNACORE_Q4_REPACK_MAX_ROWS set at process start): 8-row tiles plus a tail.
void test_tiles_beyond_8(ThreadPool& pool) {
  ReferenceBackend be(pool, this_isa());  // fresh: released addresses are never reused across sub-tests
  if (KernelPlan::defaults().q4_repack_max_rows < 16) {
    std::printf("  (tiles beyond 8 rows not covered: DYNACORE_Q4_REPACK_MAX_ROWS < 16)\n");
    return;
  }
  const Weight w = q4kref::make_weight(1024, 256, 80);  // N >= q4_repack_min_out
  CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, 1024, 256)).ok());
  const std::vector<float> none;
  for (int64_t m : {12, 16}) {
    const std::vector<float> x = q4kref::make_x(m, 256, 256, 81);
    const Mm mm = build(m, 1024, 256, false);
    uint64_t d = 0;
    const std::vector<float> y_on = run(be, mm, w, x, 256, none, m, 1024, true, &d);
    const std::vector<float> y_off = run(be, mm, w, x, 256, none, m, 1024, false, nullptr);
    CHECK(d > 0 && y_on == y_off);
    Stats st;
    q4kref::check_bound(w, x, 256, nullptr, y_on.data(), 1024, m, Path::kInt8Subscale, st);
  }
  be.release_packed(w.blocks.data());
  std::printf("  M = 12 (8 + 4 tile) and 16 (8 + 8) on the packed prefill path: identical, within bound\n");
}

// Grouped execution (matmul_many shares int8_rows).
void test_grouped(ThreadPool& pool) {
  ReferenceBackend be(pool, this_isa());  // fresh: released addresses are never reused across sub-tests
  const Weight w1 = q4kref::make_weight(24, 1280, 90), w2 = q4kref::make_weight(64, 1280, 91);
  CHECK(be.prepack_weight(view2(w1.blocks.data(), DType::kQ4_K, 24, 1280)).ok());
  CHECK(be.prepack_weight(view2(w2.blocks.data(), DType::kQ4_K, 64, 1280)).ok());
  const std::vector<float> x = q4kref::make_x(8, 1280, 1280, 92);
  ProgramBuilder pb;
  const ValueId vx = pb.activation("x", DType::kF32, 8, 1280);
  const ValueId a = pb.weight("a", DType::kQ4_K, 24, 1280), b = pb.weight("b", DType::kQ4_K, 64, 1280);
  const ValueId ya = pb.matmul(vx, a), yb = pb.matmul(vx, b);
  auto p = pb.finish();
  CHECK(p.ok());
  auto go = [&](bool rows8, uint64_t& d) {
    std::vector<float> oa(8 * 24), ob(8 * 64);
    Bindings bi;
    bi.bind(vx, view2(x.data(), DType::kF32, 8, 1280));
    bi.bind(a, view2(w1.blocks.data(), DType::kQ4_K, 24, 1280));
    bi.bind(b, view2(w2.blocks.data(), DType::kQ4_K, 64, 1280));
    bi.bind(ya, view2(oa.data(), DType::kF32, 8, 24));
    bi.bind(yb, view2(ob.data(), DType::kF32, 8, 64));
    RunOptions ro;
    ro.group_independent = true;
    ro.experimental_packed_8rows = rows8;
    const uint64_t c0 = CpuDevice::experimental_rows8_dispatches();
    CHECK(be.run(*p, bi, ro).ok());
    d = CpuDevice::experimental_rows8_dispatches() - c0;
    oa.insert(oa.end(), ob.begin(), ob.end());
    return oa;
  };
  uint64_t d_on = 0, d_off = 0;
  const std::vector<float> on = go(true, d_on), off = go(false, d_off);
  CHECK(d_on > 0 && (d_off > 0) == KernelPlan::defaults().packed_int8_8rows && on == off);
  be.release_packed(w1.blocks.data());
  be.release_packed(w2.blocks.data());
  std::printf("  grouped matmul_many (2 jobs, M = 8): identical; 8-row path used only when on\n");
}

// With DYNACORE_VNNI=1 the x8 kernel is the VNNI one, which has no 8-row
// variant: the flag must not route anything to the AVX2 8-row kernel.
void test_vnni_stand_down() {
  if (!cpu_info().features.avx_vnni) {
    std::printf("  (VNNI stand-down not covered: this CPU has no AVX-VNNI)\n");
    return;
  }
  set_env("DYNACORE_VNNI", "1");
  ThreadPool pool(2);
  ReferenceBackend vnni(pool, this_isa());  // kernel table built with the VNNI x8 kernel
  set_env("DYNACORE_VNNI", "");
  const Weight w = q4kref::make_weight(24, 1280, 95);
  CHECK(vnni.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, 24, 1280)).ok());
  const std::vector<float> x = q4kref::make_x(8, 1280, 1280, 96), none;
  uint64_t d = 0;
  const std::vector<float> y = run(vnni, build(8, 24, 1280, false), w, x, 1280, none, 8, 24, true, &d);
  CHECK(d == 0);
  Stats st;
  q4kref::check_bound(w, x, 1280, nullptr, y.data(), 24, 8, Path::kInt8Subscale, st);
  vnni.release_packed(w.blocks.data());
  std::printf("  DYNACORE_VNNI=1: 8-row path stands down (0 dispatches), VNNI result within bound\n");
}

}  // namespace

int main() {
  // Both before KernelPlan::defaults() is first read: reference packing
  // independent of free RAM, and the opt-in packed prefill path so tiles
  // beyond 8 rows can be exercised.
  set_env("DYNACORE_Q4_REPACK", "always");
  set_env("DYNACORE_Q4_REPACK_MAX_ROWS", "16");
  std::printf("test_packed_8rows (isa %s)\n", isa_name(this_isa()).data());
  test_defaults();
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  if (!be.target().int8_subscaled) {
    // Generic and NEON tiers: no interleaved kernels; the flag changes nothing.
    const Weight w = q4kref::make_weight(24, 1280, 5);
    CHECK(be.prepack_weight(view2(w.blocks.data(), DType::kQ4_K, 24, 1280)).status().code() == StatusCode::kUnsupported);
    const std::vector<float> x = q4kref::make_x(8, 1280, 1280, 6), none;
    uint64_t d = 0;
    const Mm mm = build(8, 24, 1280, false);
    const std::vector<float> on = run(be, mm, w, x, 1280, none, 8, 24, true, &d);
    const std::vector<float> off = run(be, mm, w, x, 1280, none, 8, 24, false, nullptr);
    CHECK(d == 0 && on == off);
    Stats st;
    q4kref::check_bound(w, x, 1280, nullptr, on.data(), 24, 8, Path::kFp32, st);
    std::printf("  generic tier: flag on changes nothing (0 dispatches, fp32 path, within bound)\n");
  } else {
    test_on_off(pool);
    test_odd_n(pool);
    test_strides(pool);
    test_tiles_beyond_8(pool);
    test_grouped(pool);
    test_vnni_stand_down();
  }
  std::printf("all packed 8-row tests passed (%d checks)\n", q4kref::g_checks);
  return 0;
}

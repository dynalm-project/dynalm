// Experimental int16 prefill path (KernelPlan::int16_prefill_max_rows): off by
// default; when on, matmuls of 9..N rows on an interleaved Q4_K copy (or a
// Q6_K one with DYNACORE_Q6_PACKED=1) with >= q4_repack_min_out output rows
// use int16 activations in 8-row tiles instead of fp32 expand.
//
// Checks: bit-identical to the existing 4-row int16 packed path (same per-row
// arithmetic, DD-081/DD-082); the AVX-VNNI tiles bit-identical to the AVX2
// ones; within an int16 activation bound of the exact double-precision Q4_K
// decode; strided x/y, bias, 1 == 4 threads; gating (<= 8 rows, small N,
// unpacked weights keep their old path); dispatch counter as proof of
// execution. No model files.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/quantization/repack.h"
#include "dynacore/tensor/fp16.h"
#include "q4k_reference.h"

using namespace dynacore;
using q4kref::Weight;

#define CHECK Q4K_CHECK

namespace {

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
  _putenv_s(k, v);
#else
  if (v[0] == '\0') unsetenv(k); else setenv(k, v, 1);
#endif
}

TensorView fview(const float* p, int64_t r, int64_t c, int64_t ld = 0) {
  TensorLayout l = *TensorLayout::contiguous(DType::kF32, TensorShape{r, c});
  if (ld > 0) l.strides[0] = ld * static_cast<int64_t>(sizeof(float));
  return TensorView(const_cast<float*>(p), l);
}
TensorView qview(const void* p, DType t, int64_t n, int64_t k) {
  return TensorView(const_cast<void*>(p), *TensorLayout::contiguous(t, TensorShape{n, k}));
}

KernelPlan plan_new() {  // the experimental path; int8 off so no other activation path interferes
  KernelPlan p = KernelPlan::defaults();
  p.int8_decode_max_rows = 0;
  p.int16_decode_max_rows = 0;
  p.int16_prefill_max_rows = 128;  // up to the 128-row solo-prefill chunk (scheduler solo_prefill_chunk)
  p.int16_prefill_min_weights = 0;  // the test shapes are small; the size gate is tested separately
  return p;
}
KernelPlan plan_old16() {  // the existing int16 packed path in 4-row tiles (DD-081)
  KernelPlan p = KernelPlan::defaults();
  p.int8_decode_max_rows = 0;
  p.int16_decode_max_rows = 128;
  p.int16_prefill_max_rows = 0;
  return p;
}
KernelPlan plan_fp32() {
  KernelPlan p = KernelPlan::defaults();
  p.int8_decode_max_rows = 0;
  p.int16_decode_max_rows = 0;
  p.int16_prefill_max_rows = 0;
  return p;
}

// y (row stride ldy) against the exact decode within the int16 activation
// bound: (K + 5) u S + sum (h_t / 2) |w|, h_t = amax of x's 32-block / 32767.
void check_int16(const Weight& w, const std::vector<float>& x, int64_t ldx, const float* bias, const float* y, int64_t ldy,
                 int64_t m, double* worst) {
  for (int64_t i = 0; i < m; ++i) {
    const float* xr = x.data() + i * ldx;
    std::vector<double> h(static_cast<size_t>(w.k));
    for (int64_t b = 0; b < w.k; b += 32) {
      float a = 0;
      for (int64_t t = b; t < b + 32; ++t) a = std::max(a, std::fabs(xr[t]));
      for (int64_t t = b; t < b + 32; ++t) h[static_cast<size_t>(t)] = a / 32767.0;
    }
    for (int64_t j = 0; j < w.n; ++j) {
      double ref = bias ? bias[j] : 0.0, s = bias ? std::abs(bias[j]) : 0.0, q = 0;
      for (int64_t t = 0; t < w.k; ++t) {
        const size_t wi = static_cast<size_t>(j * w.k + t);
        ref += static_cast<double>(xr[t]) * w.w[wi];
        s += (std::abs(xr[t]) + h[static_cast<size_t>(t)] / 2) * w.mag[wi];
        q += h[static_cast<size_t>(t)] / 2 * std::abs(w.w[wi]);
      }
      const double bound = (static_cast<double>(w.k) + 5) * q4kref::kU * s + q + 1e-30;
      const double e = std::abs(static_cast<double>(y[i * ldy + j]) - ref);
      if (!(e <= bound)) std::fprintf(stderr, "  y[%lld,%lld] |e| %.3g > bound %.3g\n", (long long)i, (long long)j, e, bound);
      CHECK(e <= bound);
      *worst = std::max(*worst, e / bound);
    }
  }
}

void test_defaults() {
  if (std::getenv("DYNACORE_INT16_PREFILL_ROWS") == nullptr) CHECK(KernelPlan::defaults().int16_prefill_max_rows == 0);
  const CpuKernels k = make_cpu_kernels(CpuIsa::kAvx2);
  CHECK(k.dot_q16_x8_q4_K_rows8 != nullptr && k.dot_q16_x8_q6_K_rows8 != nullptr);
  std::printf("  defaults: int16_prefill_max_rows 0 (off); 8-row int16 tiles registered for Q4_K and Q6_K\n");
}

// Q4_K: new path == existing 4-row int16 path, bit for bit, and within the int16 bound.
void test_q4k_identity(ThreadPool& pool) {
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t n = 1024, k = 1280;
  const Weight w = q4kref::make_weight(n, k, 71);
  dev.prepack_weight(qview(w.blocks.data(), DType::kQ4_K, n, k), true);
  std::vector<float> bias(static_cast<size_t>(n));
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.01f * static_cast<float>(i % 97) - 0.4f;
  const TensorView bias1 = TensorView(bias.data(), *TensorLayout::contiguous(DType::kF32, TensorShape{n}));
  double worst = 0;
  int cases = 0;
  for (int64_t m : {9, 12, 16, 17, 33, 36, 40, 48, 64, 96, 128}) {
    const std::vector<float> x = q4kref::make_x(m, k, k, static_cast<uint32_t>(100 + m));
    std::vector<float> y_new(static_cast<size_t>(m * n), std::nanf("")), y_old(y_new.size(), std::nanf(""));
    const uint64_t c0 = CpuDevice::experimental_int16_prefill_calls();
    dev.set_kernel_plan(plan_new());
    dev.matmul(fview(x.data(), m, k), qview(w.blocks.data(), DType::kQ4_K, n, k), &bias1, fview(y_new.data(), m, n));
    CHECK(CpuDevice::experimental_int16_prefill_calls() == c0 + 1);
    dev.set_kernel_plan(plan_old16());
    dev.matmul(fview(x.data(), m, k), qview(w.blocks.data(), DType::kQ4_K, n, k), &bias1, fview(y_old.data(), m, n));
    CHECK(CpuDevice::experimental_int16_prefill_calls() == c0 + 1);
    CHECK(std::memcmp(y_new.data(), y_old.data(), y_new.size() * sizeof(float)) == 0);
    check_int16(w, x, k, bias.data(), y_new.data(), n, m, &worst);
    ++cases;
  }
  std::printf("  Q4_K: %d row counts (9..128) bit-identical to the 4-row int16 path; within the int16 bound "
              "(worst %.3f of bound); counter +1 per matmul\n",
              cases, worst);
}

// Strided x and y, 1 and 4 threads identical.
void test_strides_threads() {
  const int64_t n = 1024, k = 768, m = 37, ldx = 800, ldy = 1040;
  const Weight w = q4kref::make_weight(n, k, 72);
  const std::vector<float> x = q4kref::make_x(m, k, ldx, 73);
  std::vector<float> y1(static_cast<size_t>(m * ldy), -3.0f), y4 = y1;
  for (int t : {1, 4}) {
    ThreadPool pool(t);
    CpuDevice dev(pool, CpuIsa::kAvx2);
    dev.prepack_weight(qview(w.blocks.data(), DType::kQ4_K, n, k), true);
    dev.set_kernel_plan(plan_new());
    std::vector<float>& y = t == 1 ? y1 : y4;
    dev.matmul(fview(x.data(), m, k, ldx), qview(w.blocks.data(), DType::kQ4_K, n, k), nullptr, fview(y.data(), m, n, ldy));
  }
  CHECK(y1 == y4);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = n; j < ldy; ++j) CHECK(y1[static_cast<size_t>(i * ldy + j)] == -3.0f);
  }
  double worst = 0;
  check_int16(w, x, ldx, nullptr, y1.data(), ldy, m, &worst);
  std::printf("  strided x (ld %lld) and y (ld %lld), M %lld: 1 == 4 threads, padding untouched, within bound (%.3f)\n",
              (long long)ldx, (long long)ldy, (long long)m, worst);
}

// Gating: <= 8 rows, N < q4_repack_min_out and unpacked weights keep the path
// they had (fp32 expand here), with no dispatch.
void test_gating(ThreadPool& pool) {
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t k = 512;
  const Weight big = q4kref::make_weight(1024, k, 74), small = q4kref::make_weight(1016, k, 75), loose = q4kref::make_weight(1024, k, 76);
  dev.prepack_weight(qview(big.blocks.data(), DType::kQ4_K, 1024, k), true);
  dev.prepack_weight(qview(small.blocks.data(), DType::kQ4_K, 1016, k), true);  // packed, but below min_out
  auto run = [&](const Weight& w, int64_t m, const KernelPlan& kp) {
    const std::vector<float> x = q4kref::make_x(m, k, k, static_cast<uint32_t>(200 + m));
    std::vector<float> y(static_cast<size_t>(m * w.n));
    dev.set_kernel_plan(kp);
    dev.matmul(fview(x.data(), m, k), qview(w.blocks.data(), DType::kQ4_K, w.n, k), nullptr, fview(y.data(), m, w.n));
    return y;
  };
  const uint64_t c0 = CpuDevice::experimental_int16_prefill_calls();
  for (int64_t m : {1, 4, 8}) CHECK(run(big, m, plan_new()) == run(big, m, plan_fp32()));
  CHECK(run(small, 36, plan_new()) == run(small, 36, plan_fp32()));
  CHECK(run(loose, 36, plan_new()) == run(loose, 36, plan_fp32()));
  KernelPlan capped = plan_new();
  capped.int16_prefill_max_rows = 32;  // M 36 is above the cap
  CHECK(run(big, 36, capped) == run(big, 36, plan_fp32()));
  capped.int16_prefill_max_rows = 64;  // DYNACORE_INT16_PREFILL_ROWS=64: 96/128-row solo-prefill chunks stay on fp32
  CHECK(run(big, 96, capped) == run(big, 96, plan_fp32()));
  CHECK(run(big, 128, capped) == run(big, 128, plan_fp32()));
  KernelPlan sized = plan_new();
  sized.int16_prefill_min_weights = KernelPlan::defaults().int16_prefill_min_weights;  // 1024 x 512 is below it
  CHECK(sized.int16_prefill_min_weights == (int64_t{4} << 20));
  CHECK(run(big, 36, sized) == run(big, 36, plan_fp32()));
  CHECK(CpuDevice::experimental_int16_prefill_calls() == c0);
  run(big, 9, plan_new());
  CHECK(CpuDevice::experimental_int16_prefill_calls() == c0 + 1);
  std::printf("  gating: M 1/4/8, N 1016 < min_out, unpacked weight, M above the cap (36 > 32, 96/128 > 64), weight below the 4M-element "
              "size gate: unchanged (== fp32), no dispatch; M 9 dispatches\n");
}

// Random Q6_K blocks (valid fp16 scales).
std::vector<quant::BlockQ6_K> make_q6k(int64_t n, int64_t k, uint32_t seed) {
  std::vector<quant::BlockQ6_K> b(static_cast<size_t>(n * k / 256));
  q4kref::Rng r(seed);
  for (auto& blk : b) {
    auto* bytes = reinterpret_cast<uint8_t*>(&blk);
    for (size_t i = 0; i < sizeof(blk); ++i) bytes[i] = static_cast<uint8_t>(r.next());
    blk.d = fp32_to_fp16(static_cast<float>(r.uniform(1e-4, 2e-3)));
  }
  return b;
}

// Q6_K (DYNACORE_Q6_PACKED=1 set in main): new path == 4-row int16 Q6_K path,
// and close to fp32 expand.
void test_q6k(ThreadPool& pool) {
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t n = 1024, k = 1024;
  const auto w = make_q6k(n, k, 81);
  dev.prepack_weight(qview(w.data(), DType::kQ6_K, n, k), true);
  double worst_rel = 0;
  for (int64_t m : {9, 33, 40, 96, 128}) {
    const std::vector<float> x = q4kref::make_x(m, k, k, static_cast<uint32_t>(300 + m));
    std::vector<float> y_new(static_cast<size_t>(m * n)), y_old(y_new.size()), y_f(y_new.size());
    const uint64_t c0 = CpuDevice::experimental_int16_prefill_calls();
    for (int path = 0; path < 3; ++path) {
      dev.set_kernel_plan(path == 0 ? plan_new() : path == 1 ? plan_old16() : plan_fp32());
      std::vector<float>& y = path == 0 ? y_new : path == 1 ? y_old : y_f;
      dev.matmul(fview(x.data(), m, k), qview(w.data(), DType::kQ6_K, n, k), nullptr, fview(y.data(), m, n));
    }
    CHECK(CpuDevice::experimental_int16_prefill_calls() == c0 + 1);
    CHECK(std::memcmp(y_new.data(), y_old.data(), y_new.size() * sizeof(float)) == 0);
    double ymax = 0, dmax = 0;
    for (size_t i = 0; i < y_f.size(); ++i) {
      ymax = std::max(ymax, std::abs(static_cast<double>(y_f[i])));
      dmax = std::max(dmax, std::abs(static_cast<double>(y_new[i]) - y_f[i]));
    }
    CHECK(dmax <= 1e-3 * ymax);
    worst_rel = std::max(worst_rel, dmax / ymax);
  }
  std::printf("  Q6_K: M 9/33/40/96/128 bit-identical to the 4-row int16 Q6_K path; max |int16 - fp32| %.2g of max |y|\n", worst_rel);
}

// AVX-VNNI tiles (vpdpwssd) == AVX2 tiles, bit for bit, at every tile height.
void test_vnni_identity() {
  if (!cpu_info().features.avx_vnni) {
    std::printf("  vnni: CPU has no AVX-VNNI; AVX2 tiles only\n");
    return;
  }
  set_env("DYNACORE_INT16_PREFILL_VNNI", "0");
  const CpuKernels avx2 = make_cpu_kernels(CpuIsa::kAvx2);
  set_env("DYNACORE_INT16_PREFILL_VNNI", "");
  const CpuKernels vnni = make_cpu_kernels(CpuIsa::kAvx2);
  CHECK(avx2.dot_q16_x8_q4_K_rows8 != vnni.dot_q16_x8_q4_K_rows8 && avx2.dot_q16_x8_q6_K_rows8 != vnni.dot_q16_x8_q6_K_rows8);
  const int64_t k = 1536, nb = k / 32;
  const Weight w = q4kref::make_weight(8, k, 91);
  const auto w6 = make_q6k(8, k, 92);
  std::vector<BlockQ4_Kx8> p4(static_cast<size_t>(k / 256));
  std::vector<BlockQ6_Kx8> p6(static_cast<size_t>(k / 256));
  repack_q4_K_x8(w.blocks.data(), dtype_row_bytes(DType::kQ4_K, k), 0, k, p4.data());
  repack_q6_K_x8(w6.data(), dtype_row_bytes(DType::kQ6_K, k), 0, k, p6.data());
  const std::vector<float> x = q4kref::make_x(8, k, k, 93);
  std::vector<ActBlockQ16> act(static_cast<size_t>(8 * nb));
  for (int r = 0; r < 8; ++r) avx2.quantize_act16(x.data() + r * k, act.data() + r * nb, k);
  const ActBlockQ16* xr[8];
  for (int r = 0; r < 8; ++r) xr[r] = act.data() + r * nb;
  for (int m = 1; m <= 8; ++m) {
    float a[64], b[64];
    avx2.dot_q16_x8_q4_K_rows8(p4.data(), xr, m, k, a);
    vnni.dot_q16_x8_q4_K_rows8(p4.data(), xr, m, k, b);
    CHECK(std::memcmp(a, b, sizeof(float) * 8 * static_cast<size_t>(m)) == 0);
    avx2.dot_q16_x8_q6_K_rows8(p6.data(), xr, m, k, a);
    vnni.dot_q16_x8_q6_K_rows8(p6.data(), xr, m, k, b);
    CHECK(std::memcmp(a, b, sizeof(float) * 8 * static_cast<size_t>(m)) == 0);
  }
  std::printf("  vnni: vpdpwssd tiles == AVX2 tiles bit for bit, Q4_K and Q6_K, 1..8 rows\n");
}

}  // namespace

int main() {
  const CpuIsa isa = select_isa(cpu_info().features);
  std::printf("test_int16_prefill (isa %s)\n", isa_name(isa).data());
  if (isa != CpuIsa::kAvx2) {
    std::printf("no avx2 tier: the int16 kernels are AVX2-only; nothing to test\n");
    return 0;
  }
  set_env("DYNACORE_Q4_REPACK", "always");  // packing independent of free RAM
  set_env("DYNACORE_Q6_PACKED", "1");       // read once, before the first Q6_K prepack
  ThreadPool pool(4);
  test_defaults();
  test_q4k_identity(pool);
  test_strides_threads();
  test_gating(pool);
  test_q6k(pool);
  test_vnni_identity();
  std::printf("all int16 prefill tests passed (%d checks)\n", q4kref::g_checks);
  return 0;
}

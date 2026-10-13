// Single-thread FFN prefill kernel throughput on one pinned logical CPU
// (DynaLM Next, Phase 1 performance inventory).
//
// For a Q4_K weight [N, K] and M activation rows, times one thread computing
// y = x . w^T with
//   expand : dequantize 4-row panels (K-slices of 1024) + gemm_panel, as
//            CpuDevice::matmul's fp32 prefill path;
//   i16avx2: quantize_act16 + AVX2 int16 x8 tiles (8 rows per call);
//   i16vnni: quantize_act16 + AVX-VNNI int16 x8 tiles;
// on the interleaved copy for the int16 paths. Reports ms per call and GMAC/s.
// No model files.
//
//   bench_ffn_kernels [cpu=0] [N=1024] [K=1536] [M list=8,16,32,48,64] [reps=0 (auto)]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/thread_qos.h"
#include "dynacore/quantization/repack.h"
#include "dynacore/tensor/dtype.h"
#include "../tests/q4k_reference.h"

using namespace dynacore;

namespace dynacore {
CpuKernels::DotQ16SX8Fn avxvnni_q16s_variant(int r);  // kernels_avxvnni.cpp study hooks
int64_t avxvnni_probe_dpbusd(int64_t iters);
int64_t avxvnni_probe_fma(int64_t iters);
}

namespace {

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
  _putenv_s(k, v);
#else
  setenv(k, v, 1);
#endif
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  std::string t(s);
  size_t p = 0;
  while (p < t.size()) {
    const size_t q = t.find(',', p);
    v.push_back(std::atoi(t.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const int cpu = argc > 1 ? std::atoi(argv[1]) : 0;
  if (argc > 2 && std::string(argv[2]) == "probe") {
    // Instruction throughput: ymm vpdpbusd vs vfmadd231ps, 12 independent chains.
    request_full_speed_process();
    request_full_speed_thread();
    if (!pin_current_thread(cpu)) std::printf("warning: could not pin to cpu %d\n", cpu);
    const int64_t iters = 50'000'000;
    int64_t sink = avxvnni_probe_dpbusd(iters / 10) + avxvnni_probe_fma(iters / 10);  // warm
    double best_d = 1e30, best_f = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      double t0 = now_ms();
      sink += avxvnni_probe_dpbusd(iters);
      best_d = std::min(best_d, now_ms() - t0);
      t0 = now_ms();
      sink += avxvnni_probe_fma(iters);
      best_f = std::min(best_f, now_ms() - t0);
    }
    const double n = 12.0 * static_cast<double>(iters);
    std::printf("cpu %d probe: vpdpbusd ymm %.2f G/s | vfmadd231ps ymm %.2f G/s | ratio dpbusd/fma %.2f (sink %lld)\n", cpu,
                n / best_d * 1e-6, n / best_f * 1e-6, best_f / best_d, static_cast<long long>(sink & 1));
    return 0;
  }
  const int64_t n = argc > 2 ? std::atoll(argv[2]) : 1024;
  const int64_t k = argc > 3 ? std::atoll(argv[3]) : 1536;
  const std::vector<int> ms = parse_list(argc > 4 ? argv[4] : "8,16,32,48,64");
  const int reps_arg = argc > 5 ? std::atoi(argv[5]) : 0;
  request_full_speed_process();
  request_full_speed_thread();
  if (!pin_current_thread(cpu)) std::printf("warning: could not pin to cpu %d\n", cpu);

  set_env("DYNACORE_INT16_PREFILL_VNNI", "0");
  const CpuKernels ka = make_cpu_kernels(CpuIsa::kAvx2);
  set_env("DYNACORE_INT16_PREFILL_VNNI", "1");
  const CpuKernels kv = make_cpu_kernels(CpuIsa::kAvx2);
  const bool has_vnni = cpu_info().features.avx_vnni && kv.dot_q16_x8_q4_K_rows8 != ka.dot_q16_x8_q4_K_rows8;

  const q4kref::Weight w = q4kref::make_weight(n, k, 7);
  const int64_t row_bytes = dtype_row_bytes(DType::kQ4_K, k), nb = k / 256, groups = n / 8;
  std::vector<BlockQ4_Kx8> packed(static_cast<size_t>(groups * nb));
  for (int64_t g = 0; g < groups; ++g) repack_q4_K_x8(w.blocks.data(), row_bytes, g * 8, k, packed.data() + g * nb);
  const int mmax = *std::max_element(ms.begin(), ms.end());
  const std::vector<float> x = q4kref::make_x(mmax, k, k, 11);
  std::vector<float> y(static_cast<size_t>(mmax * n));
  std::vector<ActBlockQ16> act(static_cast<size_t>(mmax * (k / 32)));
  std::vector<float> panel(static_cast<size_t>(4 * 1024));
  const auto* wbase = reinterpret_cast<const std::byte*>(w.blocks.data());
  const DequantFn dequant = ka.dequant_for(DType::kQ4_K);

  auto run_expand = [&](int m) {
    const int64_t kc = std::min<int64_t>(1024, k);
    for (int64_t k0 = 0; k0 < k; k0 += kc) {
      const int64_t len = std::min(kc, k - k0);
      for (int64_t j0 = 0; j0 < n; j0 += 4) {
        for (int r = 0; r < 4; ++r) dequant(wbase + (j0 + r) * row_bytes + dtype_row_bytes(DType::kQ4_K, k0), panel.data() + r * len, len);
        ka.gemm_panel(panel.data(), 4, x.data() + k0, k, m, len, y.data() + j0, n, k0 > 0);
      }
    }
  };
  auto run_i16 = [&](const CpuKernels& kk, int m) {
    const int64_t nbk = k / 32;
    for (int r = 0; r < m; ++r) kk.quantize_act16(x.data() + r * k, act.data() + r * nbk, k);
    float out[8 * 8];
    const ActBlockQ16* xr[8];
    for (int64_t g = 0; g < groups; ++g) {
      for (int r0 = 0; r0 < m; r0 += 8) {
        const int mm = std::min(8, m - r0);
        for (int t = 0; t < mm; ++t) xr[t] = act.data() + (r0 + t) * nbk;
        kk.dot_q16_x8_q4_K_rows8(packed.data() + g * nb, xr, mm, k, out);
        for (int t = 0; t < mm; ++t) std::copy(out + 8 * t, out + 8 * t + 8, y.data() + (r0 + t) * n + 8 * g);
      }
    }
  };
  std::vector<ActBlockQ16S> acts(act.size());
  auto run_i16s = [&](CpuKernels::DotQ16SX8Fn fn, int m) {
    const int64_t nbk = k / 32;
    for (int r = 0; r < m; ++r) ka.quantize_act16(x.data() + r * k, act.data() + r * nbk, k);
    ka.split_act16(act.data(), acts.data(), m * nbk);
    float out[8 * 8];
    const ActBlockQ16S* xr[8];
    for (int64_t g = 0; g < groups; ++g) {
      for (int r0 = 0; r0 < m; r0 += 8) {
        const int mm = std::min(8, m - r0);
        for (int t = 0; t < mm; ++t) xr[t] = acts.data() + (r0 + t) * nbk;
        fn(packed.data() + g * nb, xr, mm, k, out);
        for (int t = 0; t < mm; ++t) std::copy(out + 8 * t, out + 8 * t + 8, y.data() + (r0 + t) * n + 8 * g);
      }
    }
  };
  auto time = [&](auto&& fn) {
    fn();  // warm
    int reps = reps_arg;
    if (reps <= 0) {
      const double t0 = now_ms();
      fn();
      const double one = std::max(now_ms() - t0, 1e-3);
      reps = std::max(3, static_cast<int>(300.0 / one));
    }
    std::vector<double> t(static_cast<size_t>(reps));
    for (int i = 0; i < reps; ++i) {
      const double t0 = now_ms();
      fn();
      t[static_cast<size_t>(i)] = now_ms() - t0;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };
  std::printf("cpu %d | N %lld K %lld | vnni %s\n", cpu, static_cast<long long>(n), static_cast<long long>(k), has_vnni ? "yes" : "no");
  std::printf("  M |  expand ms  GMAC/s | i16avx2 ms  GMAC/s | i16vnni ms  GMAC/s |"
              " split3 ms GMAC/s | split2 ms GMAC/s | split4 ms GMAC/s | bit-identical\n");
  for (int m : ms) {
    const double macs = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double te = time([&] { run_expand(m); });
    const double ta = time([&] { run_i16(ka, m); });
    const double tv = has_vnni ? time([&] { run_i16(kv, m); }) : 0.0;
    double ts[3] = {0, 0, 0};
    bool ident = false;
    if (has_vnni) {
      const int rs[3] = {3, 2, 4};
      for (int v = 0; v < 3; ++v) ts[v] = time([&] { run_i16s(avxvnni_q16s_variant(rs[v]), m); });
      run_i16(ka, m);
      const std::vector<float> ref(y.begin(), y.begin() + static_cast<int64_t>(m) * n);
      ident = true;
      for (int v = 0; v < 3; ++v) {
        run_i16s(avxvnni_q16s_variant(rs[v]), m);
        ident = ident && std::equal(ref.begin(), ref.end(), y.begin());
      }
    }
    auto g = [&](double t) { return t > 0 ? macs / t * 1e-6 : 0.0; };
    std::printf("%3d | %9.3f %7.2f | %9.3f %7.2f | %9.3f %7.2f | %8.3f %6.2f | %8.3f %6.2f | %8.3f %6.2f | %s\n", m, te, g(te), ta,
                g(ta), tv, g(tv), ts[0], g(ts[0]), ts[1], g(ts[1]), ts[2], g(ts[2]), ident ? "yes" : "NO");
  }
  return 0;
}

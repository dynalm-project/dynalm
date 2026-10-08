// Phase 17 benchmark: fused dequantize-dot throughput per weight type,
// generic vs AVX2 tier (one 4096-element row, the decode matmul inner loop).

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/quantization/repack.h"
#include "bench_harness.h"
#include "dynacore/tensor/fp16.h"
#include "dynacore/hardware/cpu_info.h"

int main() {
  using namespace dynacore;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  CpuKernels generic;
  register_generic_kernels(generic);
  CpuKernels avx2 = generic;
  const bool have_avx2 = isa_supported(CpuIsa::kAvx2, cpu_info().features) && register_avx2_kernels(avx2);

  constexpr int64_t kN = 4096;
  std::mt19937 rng(1);
  std::vector<float> x(kN);
  for (auto& v : x) v = static_cast<float>(rng() % 1000) / 1000.0f - 0.5f;
  std::printf("%-6s %14s %14s %9s %12s\n", "type", "generic ns", "avx2 ns", "speedup", "avx2 GB/s");
  for (DType t : {DType::kF32, DType::kF16, DType::kBF16, DType::kQ8_0, DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ4_K,
                  DType::kQ5_K, DType::kQ6_K}) {
    const int64_t bytes = dtype_row_bytes(t, kN);
    std::vector<uint8_t> w(static_cast<size_t>(bytes));
    // Quantized: small random bytes keep fp16 scale fields finite. Float
    // types need real values: tiny byte patterns would be denormals, which
    // are pathologically slow and would misrepresent the kernel.
    for (auto& b : w) b = static_cast<uint8_t>(rng() & 0x3F);
    for (int64_t i = 0; i < kN && !dtype_is_quantized(t); ++i) {
      const float v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      if (t == DType::kF32) reinterpret_cast<float*>(w.data())[i] = v;
      if (t == DType::kF16) reinterpret_cast<uint16_t*>(w.data())[i] = fp32_to_fp16(v);
      if (t == DType::kBF16) reinterpret_cast<uint16_t*>(w.data())[i] = fp32_to_bf16(v);
    }
    float sink = 0;
    const bench::Options opt{.warmup_samples = 5, .samples = 200, .batch = 20};
    const auto g = bench::run([&] { sink += generic.vec_dot_for(t)(w.data(), x.data(), kN); }, opt);
    double a_p50 = 0;
    if (have_avx2) {
      const auto a = bench::run([&] { sink += avx2.vec_dot_for(t)(w.data(), x.data(), kN); }, opt);
      a_p50 = a.p50;
    }
    bench::do_not_optimize(sink);
    std::printf("%-6s %14.0f %14.0f %8.1fx %12.2f\n", std::string(dtype_name(t)).c_str(), g.p50, a_p50,
                a_p50 > 0 ? g.p50 / a_p50 : 0.0, a_p50 > 0 ? bytes / a_p50 : 0.0);
  }

  // Row dequantization (the batched-matmul expand path), one 4096-element row.
  std::printf("\ndequant row (%lld elements):\n%-6s %14s %14s %9s\n", static_cast<long long>(kN), "type",
              "generic ns", "avx2 ns", "speedup");
  for (DType t : {DType::kF16, DType::kBF16, DType::kQ8_0, DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ4_K, DType::kQ6_K}) {
    std::vector<uint8_t> w(static_cast<size_t>(dtype_row_bytes(t, kN)));
    for (auto& b : w) b = static_cast<uint8_t>(rng() & 0x3F);
    std::vector<float> out(kN);
    const bench::Options opt{.warmup_samples = 5, .samples = 200, .batch = 20};
    const auto g = bench::run([&] { generic.dequant_for(t)(w.data(), out.data(), kN); }, opt);
    const auto a = bench::run([&] { avx2.dequant_for(t)(w.data(), out.data(), kN); }, opt);
    bench::do_not_optimize(out[0]);
    std::printf("%-6s %14.0f %14.0f %8.1fx\n", std::string(dtype_name(t)).c_str(), g.p50, a.p50, g.p50 / a.p50);
  }

  // int8 decode kernels (DD-053, DD-075), single thread, weights hot in cache:
  // the compute cost per 256-value super-block for m = 1..4 activation rows,
  // per-32 activations ("q8") vs super-block activations ("sb").
  if (have_avx2) {
    std::printf("\nint8/int16 dot rows (single thread, hot cache, ns per super-block):\n%-6s %4s %10s %10s %10s %10s %10s %10s %10s\n",
                "type", "m", "fp32", "q8", "sb", "sx", "q16", "sx-x8", "x8-vnni");
    constexpr int64_t kK = 1536, kRows = 64;
    for (DType t : {DType::kQ4_K, DType::kQ6_K}) {
      const int64_t row_bytes = dtype_row_bytes(t, kK);
      std::vector<uint8_t> w(static_cast<size_t>(row_bytes * kRows));
      for (auto& b : w) b = static_cast<uint8_t>(rng() & 0x3F);
      std::vector<float> xs(static_cast<size_t>(4 * kK));
      for (auto& v : xs) v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      std::vector<ActBlockQ8> a8(static_cast<size_t>(4 * kK / 32)), asb(a8.size());
      std::vector<ActBlockQ16> a16(a8.size());
      std::vector<ActBlockQ8> asx(a8.size());
      for (int r = 0; r < 4; ++r) {
        avx2.quantize_act(xs.data() + r * kK, a8.data() + r * kK / 32, kK);
        avx2.quantize_act_sb(xs.data() + r * kK, asb.data() + r * kK / 32, kK);
        avx2.quantize_act16(xs.data() + r * kK, a16.data() + r * kK / 32, kK);
        avx2.quantize_act_sx(xs.data() + r * kK, asx.data() + r * kK / 32, kK);
      }
      const ActBlockQ8* psx[4] = {asx.data(), asx.data() + kK / 32, asx.data() + 2 * kK / 32, asx.data() + 3 * kK / 32};
      // Interleaved copy (DD-078), Q4_K only.
      std::vector<BlockQ4_Kx8> packed;
      if (t == DType::kQ4_K) {
        packed.resize(static_cast<size_t>(kRows / 8 * kK / 256));
        for (int64_t g = 0; g < kRows / 8; ++g) repack_q4_K_x8(w.data(), row_bytes, 8 * g, kK, packed.data() + g * (kK / 256));
      }
      const ActBlockQ16* p16[4] = {a16.data(), a16.data() + kK / 32, a16.data() + 2 * kK / 32,
                                   a16.data() + 3 * kK / 32};
      const ActBlockQ8* p8[4] = {a8.data(), a8.data() + kK / 32, a8.data() + 2 * kK / 32, a8.data() + 3 * kK / 32};
      const ActBlockQ8* psb[4] = {asb.data(), asb.data() + kK / 32, asb.data() + 2 * kK / 32, asb.data() + 3 * kK / 32};
      for (int m = 1; m <= 4; ++m) {
        float out[4];
        const bench::Options opt{.warmup_samples = 5, .samples = 100, .batch = 5};
        auto sweep = [&](DotQ8RowsFn f, const ActBlockQ8* const* p) {
          for (int64_t j = 0; j < kRows; ++j) f(w.data() + j * row_bytes, p, m, kK, out);
          bench::do_not_optimize(out[0]);
        };
        const double sbs = static_cast<double>(kRows * kK / 256);
        const auto f = bench::run([&] {
          for (int64_t j = 0; j < kRows; ++j) {
            for (int r = 0; r < m; ++r) out[r] = avx2.vec_dot_for(t)(w.data() + j * row_bytes, xs.data() + r * kK, kK);
          }
          bench::do_not_optimize(out[0]);
        }, opt);
        const auto q = bench::run([&] { sweep(avx2.dot_q8_rows_for(t), p8); }, opt);
        const auto s = bench::run([&] { sweep(avx2.dot_q8_sb_rows_for(t), psb); }, opt);
        const auto x2 = bench::run([&] { sweep(avx2.dot_q8_sx_rows_for(t), psx); }, opt);
        const auto h = bench::run([&] {
          for (int64_t j = 0; j < kRows; ++j) avx2.dot_q16_rows_for(t)(w.data() + j * row_bytes, p16, m, kK, out);
          bench::do_not_optimize(out[0]);
        }, opt);
        double x8 = 0, x8v = 0;
        if (!packed.empty() && cpu_info().features.avx_vnni) {
          CpuKernels vnni = avx2;
          if (register_avxvnni_kernels(vnni)) {
            float out8[32];
            const auto rv = bench::run([&] {
              for (int64_t g = 0; g < kRows / 8; ++g) vnni.dot_q8_sx_x8_q4_K(packed.data() + g * (kK / 256), psx, m, kK, out8);
              bench::do_not_optimize(out8[0]);
            }, opt);
            x8v = rv.p50 / sbs;
          }
        }
        if (!packed.empty()) {
          float out8[32];
          const auto r8 = bench::run([&] {
            for (int64_t g = 0; g < kRows / 8; ++g) avx2.dot_q8_sx_x8_q4_K(packed.data() + g * (kK / 256), psx, m, kK, out8);
            bench::do_not_optimize(out8[0]);
          }, opt);
          x8 = r8.p50 / sbs;
        }
        std::printf("%-6s %4d %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n", std::string(dtype_name(t)).c_str(), m,
                    f.p50 / sbs, q.p50 / sbs, s.p50 / sbs, x2.p50 / sbs, h.p50 / sbs, x8, x8v);
      }
    }
  }

  // GEMM panel microkernel, single thread: 4 weight rows x m activation rows.
  std::printf("\ngemm_panel (single thread, nr=4):\n");
  for (int64_t m : {2, 64, 256}) {
    for (int64_t k : {896, 4864}) {
      std::vector<float> w(static_cast<size_t>(4 * k)), xs(static_cast<size_t>(m * k)), y(static_cast<size_t>(m * 4));
      for (auto& v : w) v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      for (auto& v : xs) v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      const auto& kk = have_avx2 ? avx2 : generic;
      const auto s = bench::run([&] { kk.gemm_panel(w.data(), 4, xs.data(), k, m, k, y.data(), 4, false); },
                                {.warmup_samples = 3, .samples = 50, .batch = 5});
      const double flops = 2.0 * 4 * static_cast<double>(m) * static_cast<double>(k);
      std::printf("  m=%-4lld k=%-5lld p50 %10.0f ns  %6.1f GFLOPS\n", static_cast<long long>(m),
                  static_cast<long long>(k), s.p50, flops / s.p50);
    }
  }
  return 0;
}

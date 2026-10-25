// qmm kernels (dynacore/cpu/qmm.h) against CpuDevice::matmul's current path
// on transformer shapes, multithreaded. Synthetic weights (no model files).
//
//   bench_qmm [threads] [--quick]
//
// Rows: m = 1 / 4 / 8 (decode) and 64 / 512 (prefill). "device" is
// CpuDevice::matmul with the engine's ARM plan (int8 up to 64 rows, int16 FFN
// down); "qmm" quantizes the activations and runs gemv (m <= 8) or gemm.
// Reports p50 ms and effective weight bandwidth (GB/s) or GOPS.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bench_harness.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/cpu/qmm.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;

namespace {

std::mt19937 g_rng(7);

Tensor random_weights(DType t, int64_t n, int64_t k) {
  auto w = Tensor::empty(t, {n, k});
  auto* p = static_cast<uint8_t*>(w->data());
  const int64_t bytes = dtype_row_bytes(t, k) * n;
  for (int64_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(g_rng());
  // Sane fp16 scales: overwrite each block's fp16 fields.
  const int64_t be = dtype_block_elems(t), bb = dtype_block_bytes(t);
  std::vector<int> fields;
  if (t == DType::kQ8_0 || t == DType::kQ4_0) fields = {0};
  if (t == DType::kQ4_K || t == DType::kQ5_K) fields = {0, 2};
  if (t == DType::kQ6_K) fields = {208};
  const uint16_t h = fp32_to_fp16(0.01f);
  for (int64_t b = 0; b < n * k / be; ++b) {
    for (int f : fields) std::memcpy(p + b * bb + f, &h, 2);
  }
  return std::move(*w);
}

struct Shape {
  const char* name;
  DType t;
  int64_t n, k;
  bool int16;  // FFN down: int16 activations
};

}  // namespace

int main(int argc, char** argv) {
  int threads = cpu_info().physical_cores;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else {
      threads = std::atoi(argv[i]);
    }
  }
  const qmm::Kernels& kern = qmm::best_kernels();
  std::printf("cpu: %s, %d threads, qmm tier %s\n\n", cpu_info().brand.c_str(), threads, kern.name);
  ThreadPool pool(threads);
  CpuDevice dev(pool, select_best_isa(cpu_info().features));

  const Shape shapes[] = {
      {"1.5B gate/up", DType::kQ4_K, 8960, 1536, false}, {"1.5B down q4", DType::kQ4_K, 1536, 8960, true},
      {"1.5B down q6", DType::kQ6_K, 1536, 8960, true},  {"1.5B attn_o", DType::kQ4_K, 1536, 1536, false},
      {"1.5B lm_head", DType::kQ6_K, 151936, 1536, false}, {"8B gate q8", DType::kQ8_0, 14336, 4096, false},
      {"q5_K gate", DType::kQ5_K, 8960, 1536, false},    {"q4_0 gate", DType::kQ4_0, 8960, 1536, false},
  };
  if (std::getenv("BENCH_QMM_STEP") != nullptr) {
    // A decode step's matvecs (Qwen2.5-1.5B shapes, all Q4_K, 28 layers,
    // cold: 770 MB): one parallel region each, packed weights, as CpuDevice
    // runs them. Reports the step time per thread count.
    struct M { int64_t n, k; };
    const M layer[] = {{1536, 1536}, {256, 1536}, {256, 1536}, {1536, 1536}, {8960, 1536}, {8960, 1536}, {1536, 8960}};
    std::vector<std::vector<std::byte>> packs;
    std::vector<M> shapes;
    for (int l = 0; l < 28; ++l) {
      for (const M& m : layer) {
        const Tensor w = random_weights(DType::kQ4_K, m.n, m.k);
        packs.emplace_back(qmm::packed_bytes(DType::kQ4_K, m.n, m.k));
        qmm::pack_groups(DType::kQ4_K, w.data(), dtype_row_bytes(DType::kQ4_K, m.k), m.k, 0, m.n / 4, packs.back().data());
        shapes.push_back(m);
      }
    }
    size_t total = 0;
    for (const auto& pk : packs) total += pk.size();
    auto x = Tensor::empty(DType::kF32, {1, 8960});
    auto y = Tensor::empty(DType::kF32, {1, 8960});
    std::normal_distribution<float> nd(0, 1);
    for (int64_t i = 0; i < 8960; ++i) x->data_as<float>()[i] = nd(g_rng);
    qmm::ActBuffer a1536, a8960;
    a1536.prepare(qmm::ActMode::kInt8, 1, 1536, false);
    a1536.quantize_rows(x->data_as<float>(), 1536, 0, 1, kern);
    a8960.prepare(qmm::ActMode::kInt8, 1, 8960, false);
    a8960.quantize_rows(x->data_as<float>(), 8960, 0, 1, kern);
    for (int t : {4, 8, 10, 12}) {
      ThreadPool tp(t);
      auto step = [&] {
        for (size_t i = 0; i < packs.size(); ++i) {
          const M& m = shapes[i];
          const qmm::Acts& a = m.k == 1536 ? a1536.acts() : a8960.acts();
          const int64_t groups = m.n / 4, gu = 4;  // 16 rows per item
          const size_t items = static_cast<size_t>((groups + gu - 1) / gu);
          tp.parallel_for(items, std::max<size_t>(1, items / (static_cast<size_t>(t) * 32)), [&](size_t b, size_t e) {
            kern.gemv_packed(DType::kQ4_K, packs[i].data(), m.k, static_cast<int64_t>(b) * gu,
                             std::min<int64_t>(groups, static_cast<int64_t>(e) * gu), a, 0, 1, y->data_as<float>(), m.n, nullptr);
          });
        }
      };
      const auto st = bench::run(step, {.warmup_samples = 1, .samples = 8, .batch = 1});
      std::printf("decode-step matvecs (%zu regions, %.0f MB), %2d threads: %.2f ms p50 (%.2f min), %.1f GB/s\n", packs.size(),
                  static_cast<double>(total) / 1e6, t, st.p50 * 1e-6, st.min * 1e-6, static_cast<double>(total) / st.p50);
    }
    return 0;
  }
  if (std::getenv("BENCH_QMM_ROTATE") != nullptr) {
    // Decode-like: 40 distinct cold 8960 x 1536 Q4_K matrices (308 MB) used in
    // turn, one gemv each (as the layers of a decode step).
    constexpr int kMats = 40;
    std::vector<Tensor> ws;
    for (int i = 0; i < kMats; ++i) ws.push_back(random_weights(DType::kQ4_K, 8960, 1536));
    const int64_t rb = dtype_row_bytes(DType::kQ4_K, 1536);
    auto x = Tensor::empty(DType::kF32, {1, 1536});
    auto y = Tensor::empty(DType::kF32, {1, 8960});
    std::normal_distribution<float> nd(0, 1);
    for (int64_t i = 0; i < 1536; ++i) x->data_as<float>()[i] = nd(g_rng);
    for (int t : {4, 8, 12}) {
      for (int unit : {16, 64, 256}) {
        ThreadPool tp(t);
        qmm::ActBuffer ab;
        ab.prepare(qmm::ActMode::kInt8, 1, 1536, false);
        ab.quantize_rows(x->data_as<float>(), 1536, 0, 1, kern);
        auto pass = [&] {
          for (int i = 0; i < kMats; ++i) {
            tp.parallel_for(static_cast<size_t>((8960 + unit - 1) / unit), 1, [&](size_t b, size_t e) {
              kern.gemv(DType::kQ4_K, ws[static_cast<size_t>(i)].data(), rb, static_cast<int64_t>(b) * unit,
                        std::min<int64_t>(8960, static_cast<int64_t>(e) * unit), ab.acts(), 0, 1, y->data_as<float>(), 8960,
                        nullptr);
            });
          }
        };
        const auto st = bench::run(pass, {.warmup_samples = 1, .samples = 8, .batch = 1});
        std::printf("rotating cold 8960x1536 q4_K, %2d threads, unit %3d: %.1f us/op, %.1f GB/s\n", t, unit,
                    st.p50 * 1e-3 / kMats, static_cast<double>(rb * 8960) * kMats / st.p50);
        // The same on packed copies (4-row groups).
        static std::vector<std::vector<std::byte>> packs;
        if (packs.empty()) {
          for (int i = 0; i < kMats; ++i) {
            packs.emplace_back(qmm::packed_bytes(DType::kQ4_K, 8960, 1536));
            qmm::pack_groups(DType::kQ4_K, ws[static_cast<size_t>(i)].data(), rb, 1536, 0, 8960 / 4, packs.back().data());
          }
        }
        auto ppass = [&] {
          for (int i = 0; i < kMats; ++i) {
            const int64_t gu = unit / 4;
            tp.parallel_for(static_cast<size_t>((2240 + gu - 1) / gu), 1, [&](size_t b, size_t e) {
              kern.gemv_packed(DType::kQ4_K, packs[static_cast<size_t>(i)].data(), 1536, static_cast<int64_t>(b) * gu,
                               std::min<int64_t>(2240, static_cast<int64_t>(e) * gu), ab.acts(), 0, 1, y->data_as<float>(), 8960,
                               nullptr);
            });
          }
        };
        const auto sp = bench::run(ppass, {.warmup_samples = 1, .samples = 8, .batch = 1});
        std::printf("   packed                                         : %.1f us/op, %.1f GB/s\n", sp.p50 * 1e-3 / kMats,
                    static_cast<double>(qmm::packed_bytes(DType::kQ4_K, 8960, 1536)) * kMats / sp.p50);
      }
    }
    return 0;
  }
  if (std::getenv("BENCH_QMM_SME") != nullptr) {
    // SME fp32 panel gemm vs the qmm int8 gemm (both through CpuDevice::matmul on a
    // prepacked weight, as in the engine), and the weight-panel dequantization alone.
    if (qmm::sme_gemm() == nullptr) {
      std::printf("no SME tier on this CPU / build\n");
      return 0;
    }
    std::printf("%-14s %5s %5s | %9s %7s | %9s %7s | %9s\n", "shape", "type", "m", "qmm ms", "GOPS", "sme ms", "GOPS",
                "panels ms");
    for (const Shape& s : shapes) {
      if (s.n > 100000) continue;
      const Tensor w = random_weights(s.t, s.n, s.k);
      dev.prepack_weight(w, s.int16);
      for (int64_t m : {32, 64, 128, 512}) {
        auto x = Tensor::empty(DType::kF32, {m, s.k});
        auto y = Tensor::empty(DType::kF32, {m, s.n});
        std::normal_distribution<float> nd(0, 1);
        for (int64_t i = 0; i < m * s.k; ++i) x->data_as<float>()[i] = nd(g_rng);
        KernelPlan plan = KernelPlan::defaults();
        plan.int8_decode_max_rows = s.int16 ? 0 : 1 << 30;
        plan.int16_decode_max_rows = s.int16 ? 1 << 30 : 0;
        plan.sme_min_rows = 0;
        dev.set_kernel_plan(plan);
        const auto sq = bench::run([&] { dev.matmul(*x, w, nullptr, *y); }, {.warmup_samples = 2, .samples = 7, .batch = 1});
        plan.sme_min_rows = 1;
        dev.set_kernel_plan(plan);
        const auto ss = bench::run([&] { dev.matmul(*x, w, nullptr, *y); }, {.warmup_samples = 2, .samples = 7, .batch = 1});
        const int64_t rb = dtype_row_bytes(s.t, s.k), panels = (s.n + qmm::kSmePanel - 1) / qmm::kSmePanel;
        const auto sp = bench::run([&] {
          pool.parallel_for(static_cast<size_t>(panels), 1, [&](size_t b, size_t e) {
            thread_local std::vector<float> wp;
            wp.resize(static_cast<size_t>(s.k * qmm::kSmePanel));
            for (size_t i = b; i < e; ++i) {
              qmm::weight_panel_f32(s.t, w.data(), rb, nullptr, s.n, s.k, static_cast<int64_t>(i) * qmm::kSmePanel, wp.data());
            }
          });
        }, {.warmup_samples = 1, .samples = 5, .batch = 1});
        const double ops = 2.0 * static_cast<double>(m * s.n * s.k);
        std::printf("%-14s %5s %5lld | %9.3f %7.0f | %9.3f %7.0f | %9.3f\n", s.name, std::string(dtype_name(s.t)).c_str(),
                    static_cast<long long>(m), sq.p50 * 1e-6, ops / sq.p50, ss.p50 * 1e-6, ops / ss.p50, sp.p50 * 1e-6);
      }
    }
    return 0;
  }
  if (std::getenv("BENCH_QMM_BIG") != nullptr) {
    // Cold weights (151 MB, beyond every cache): decode bandwidth vs threads.
    const Tensor w = random_weights(DType::kQ4_K, 65536, 4096);
    const int64_t rb = dtype_row_bytes(DType::kQ4_K, 4096);
    auto x = Tensor::empty(DType::kF32, {1, 4096});
    auto y = Tensor::empty(DType::kF32, {1, 65536});
    std::normal_distribution<float> nd(0, 1);
    for (int64_t i = 0; i < 4096; ++i) x->data_as<float>()[i] = nd(g_rng);
    for (int t : {1, 2, 4, 8, 10, 12}) {
      ThreadPool tp(t);
      qmm::ActBuffer ab;
      auto run = [&] {
        ab.prepare(qmm::ActMode::kInt8, 1, 4096, false);
        ab.quantize_rows(x->data_as<float>(), 4096, 0, 1, kern);
        const int64_t unit = 16;
        tp.parallel_for(65536 / unit, 1, [&](size_t b, size_t e) {
          kern.gemv(DType::kQ4_K, w.data(), rb, static_cast<int64_t>(b) * unit, static_cast<int64_t>(e) * unit, ab.acts(), 0, 1,
                    y->data_as<float>(), 65536, nullptr);
        });
      };
      const auto st = bench::run(run, {.warmup_samples = 2, .samples = 10, .batch = 1});
      std::printf("cold q4_K gemv 65536x4096, %2d threads: %.3f ms, %.1f GB/s\n", t, st.p50 * 1e-6,
                  static_cast<double>(rb * 65536) / st.p50);
    }
    return 0;
  }
  std::printf("%-14s %5s %5s | %10s %8s | %10s %8s | %6s\n", "shape", "type", "m", "device ms", "GB/s|GOP", "qmm ms",
              "GB/s|GOP", "speed");
  qmm::ActBuffer acts;
  std::vector<qmm::GemmScratch> scratch(static_cast<size_t>(threads) * 2);
  for (const Shape& s : shapes) {
    if (quick && s.n > 100000) continue;
    const Tensor w = random_weights(s.t, s.n, s.k);
    const int64_t rb = dtype_row_bytes(s.t, s.k);
    for (int64_t m : {1, 2, 4, 8, 64, 512}) {
      if (quick && m == 512 && s.n * s.k > 20000000) continue;
      if (s.n > 100000 && m > 8) continue;  // vocabulary projection: decode only
      auto x = Tensor::empty(DType::kF32, {m, s.k});
      auto y = Tensor::empty(DType::kF32, {m, s.n});
      std::normal_distribution<float> nd(0, 1);
      for (int64_t i = 0; i < m * s.k; ++i) x->data_as<float>()[i] = nd(g_rng);

      KernelPlan plan = KernelPlan::defaults();
      plan.int8_decode_max_rows = s.int16 ? 0 : 64;
      plan.int16_decode_max_rows = s.int16 ? 64 : 0;
      plan.qmm = false;  // the pre-qmm path
      dev.set_kernel_plan(plan);
      const int samples = m >= 64 ? 5 : 30;
      const auto sd = bench::run([&] { dev.matmul(*x, w, nullptr, *y); }, {.warmup_samples = 2, .samples = samples, .batch = 1});

      const qmm::ActMode mode = s.int16 ? qmm::ActMode::kInt16 : qmm::ActMode::kInt8;
      static const int64_t gemm_min = std::getenv("BENCH_QMM_GEMM_MIN") ? std::atoll(std::getenv("BENCH_QMM_GEMM_MIN")) : qmm::kGemvMaxRows + 1;
      const bool use_gemm = m >= gemm_min && kern.gemm_matrix;
      auto run_qmm = [&] {
        acts.prepare(mode, m, s.k, use_gemm);
        pool.parallel_for(static_cast<size_t>(m), 8, [&](size_t b, size_t e) {
          acts.quantize_rows(x->data_as<float>(), s.k, static_cast<int64_t>(b), static_cast<int64_t>(e), kern);
        });
        const qmm::Acts& a = acts.acts();
        float* yp = y->data_as<float>();
        if (use_gemm) {
          const int64_t unit = 16;
          const auto chunks = static_cast<size_t>((s.n + unit - 1) / unit);
          pool.parallel_for(chunks, 1, [&](size_t b, size_t e) {
            thread_local qmm::GemmScratch sc;
            kern.gemm(s.t, w.data(), rb, static_cast<int64_t>(b) * unit, std::min<int64_t>(s.n, static_cast<int64_t>(e) * unit),
                      a, yp, s.n, nullptr, sc);
          });
        } else {
          const int64_t unit = 16;
          const auto chunks = static_cast<size_t>((s.n + unit - 1) / unit);
          pool.parallel_for(chunks, 1, [&](size_t b, size_t e) {
            const int64_t n0 = static_cast<int64_t>(b) * unit, n1 = std::min<int64_t>(s.n, static_cast<int64_t>(e) * unit);
            for (int64_t r0 = 0; r0 < m; r0 += qmm::kGemvMaxRows) {
              kern.gemv(s.t, w.data(), rb, n0, n1, a, r0, static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, m - r0)),
                        yp, s.n, nullptr);
            }
          });
        }
      };
      const auto sq = bench::run(run_qmm, {.warmup_samples = 2, .samples = samples, .batch = 1});
      const double wbytes = static_cast<double>(rb * s.n);
      const double ops = 2.0 * static_cast<double>(m) * s.n * s.k;
      auto rate = [&](double ns) { return m <= 8 ? wbytes / ns : ops / ns; };
      std::printf("%-14s %5s %5lld | %10.3f %8.1f | %10.3f %8.1f | %5.2fx\n", s.name, std::string(dtype_name(s.t)).c_str(),
                  static_cast<long long>(m), sd.p50 * 1e-6, rate(sd.p50), sq.p50 * 1e-6, rate(sq.p50), sd.p50 / sq.p50);
    }
  }
  return 0;
}

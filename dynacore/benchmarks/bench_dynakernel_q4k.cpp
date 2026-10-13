// bench_dynakernel_q4k: controlled, model-free microbenchmark of Q4_K matmul
// implementations reachable through DynaKernel (docs/dynakernel.md).
//
//   bench_dynakernel_q4k [--threads 1,10] [--samples 15] [--target-ms 20] [--csv FILE] [--int8-only] [--dram]
// --dram: separate DRAM-sized configuration (int8 paths, M 2, 4, 8); see run_dram().
//
// Implementations (same synthetic weights, inputs, shapes and output layout):
//   gen_scalar   generated portable C++            fp32 activations
//   gen_avx2     generated AVX2 + FMA              fp32 activations   (AVX2 tier only)
//   ref_f32      ReferenceBackend, f32 contract    fp32 (fused dot M=1, fp32 panel M>=2)
//   ref_int8     ReferenceBackend, auto contract   int8 activations (per-32 M=1, sub-scaled M 2..8)
//   ref_int8_pk  ReferenceBackend, auto, prepacked int8 sub-scaled, interleaved Q4_K (M 2..8)
//   gen_int8     GeneratedInt8Backend: generated AVX2 int8 sub-scaled kernel on
//                its own interleaved copy, activations by quantize_act_sx (M 1..8)
// --int8-only runs ref_int8, ref_int8_pk and gen_int8 only.
// fp32 and int8 rows compute different numerics (int8 rounds activations);
// they are listed side by side, never as the same workload.
//
// Method: every output is verified against a double-precision reference
// (independent Q4_K decode) within its path's error bound before timing;
// allocation, generation, compilation and prepacking happen before timing.
// Each sample times a batch of run() calls sized to --target-ms; samples of
// all implementations are interleaved (round-robin) so drift affects them
// alike; 3 warm-up rounds are discarded. Reported: median, p10, p90 and the
// p10-p90 spread of per-call time, GFLOP/s (2 M N K) and weight GB/s. After
// timing, each implementation's output is recomputed and must be bit-identical
// to its verified output. run() includes the DynaKernel per-call checks
// (verify, kernel lookup, binding validation) for every implementation alike;
// prepacked-source fingerprinting is turned off for timing (check_packed).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bench_harness.h"
#include "dynacore/base/timer.h"
#include "dynacore/cpu/cpu_device.h"
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

extern "C" const dynacore::dk::GeneratedKernel* dk_generated_q4k_bench_kernels(size_t* count);
#if DK_GENERATED_HAS_AVX2
extern "C" const dynacore::dk::GeneratedKernel* dk_generated_q4k_bench_avx2_kernels(size_t* count);
extern "C" const dynacore::dk::GeneratedInt8Kernel* dk_generated_q4k_bench_int8_avx2_kernels(size_t* count);
#endif

namespace {

constexpr double kU = 1.0 / 16777216.0;

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "bench_dynakernel_q4k: %s\n", m.c_str());
  std::exit(1);
}

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

// --- synthetic data and independent reference -----------------------------------------------

void scale_min(const uint8_t* s, int j, int& sc, int& m) {
  if (j < 4) {
    sc = s[j] & 63;
    m = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

struct Weight {
  int64_t n = 0, k = 0;
  std::vector<BlockQ4_K> blocks;
};

Weight make_weight(int64_t n, int64_t k, uint32_t seed) {
  Weight w;
  w.n = n;
  w.k = k;
  const int64_t nb = n * k / 256;
  w.blocks.resize(static_cast<size_t>(nb));
  Rng r(seed);
  for (BlockQ4_K& b : w.blocks) {
    auto* bytes = reinterpret_cast<uint8_t*>(&b);
    for (size_t t = 0; t < sizeof(BlockQ4_K); ++t) bytes[t] = static_cast<uint8_t>(r.next());
    b.d = fp32_to_fp16(static_cast<float>(r.uniform(1e-3, 1e-2)));
    b.dmin = fp32_to_fp16(static_cast<float>(r.uniform(1e-3, 1e-2)));
  }
  return w;
}

std::vector<float> make_x(int64_t m, int64_t k, uint32_t seed) {
  Rng r(seed);
  std::vector<float> x(static_cast<size_t>(m * k));
  for (float& v : x) v = static_cast<float>(r.uniform(-1, 1) + r.uniform(-1, 1));
  return x;
}

// Exact reference y and per-element bounds for a path. Weight values are
// decoded per column on the fly (double), so memory stays O(K).
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

// Max |e| / bound over all elements (must be <= 1).
double verify(const Weight& w, const std::vector<float>& x, int64_t m, const std::vector<float>& y, Path p) {
  std::vector<double> wv(static_cast<size_t>(w.k)), tv(wv.size());
  std::vector<std::vector<double>> h(static_cast<size_t>(m));
  for (int64_t i = 0; i < m; ++i) h[static_cast<size_t>(i)] = act_steps(x.data() + i * w.k, w.k, p);
  const int64_t nb = w.k / 256;
  double worst = 0;
  for (int64_t j = 0; j < w.n; ++j) {
    for (int64_t b = 0; b < nb; ++b) {
      const BlockQ4_K& blk = w.blocks[static_cast<size_t>(j * nb + b)];
      const double d = fp16_to_fp32(blk.d), dmin = fp16_to_fp32(blk.dmin);
      for (int g = 0; g < 4; ++g) {
        for (int half = 0; half < 2; ++half) {
          int sc = 0, mm = 0;
          scale_min(blk.scales, 2 * g + half, sc, mm);
          for (int l = 0; l < 32; ++l) {
            const int q = half == 0 ? (blk.qs[32 * g + l] & 0x0F) : (blk.qs[32 * g + l] >> 4);
            const size_t at = static_cast<size_t>(b * 256 + 64 * g + 32 * half + l);
            wv[at] = d * sc * q - dmin * mm;
            tv[at] = std::abs(d * sc * q) + std::abs(dmin * mm);
          }
        }
      }
    }
    for (int64_t i = 0; i < m; ++i) {
      const float* xr = x.data() + i * w.k;
      const std::vector<double>& hi = h[static_cast<size_t>(i)];
      double ref = 0, s = 0, qerr = 0;
      for (int64_t t = 0; t < w.k; ++t) {
        ref += static_cast<double>(xr[t]) * wv[static_cast<size_t>(t)];
        s += (std::abs(xr[t]) + hi[static_cast<size_t>(t)] / 2) * tv[static_cast<size_t>(t)];
        qerr += hi[static_cast<size_t>(t)] / 2 * std::abs(wv[static_cast<size_t>(t)]);
      }
      const double bound = (static_cast<double>(w.k) + 5) * kU * s + qerr + 1e-30;
      worst = std::max(worst, std::abs(static_cast<double>(y[static_cast<size_t>(i * w.n + j)]) - ref) / bound);
    }
  }
  return worst;
}

// --- implementations -------------------------------------------------------------------------

struct Impl {
  std::string name;
  std::string numerics;  // what it computes
  std::string executes;  // what actually runs
  Path path = Path::kFp32;
  Backend* backend = nullptr;
  ReferenceBackend* ref = nullptr;  // set for reference paths (RunOptions)
  ActFormat act = ActFormat::kAuto;
  GeneratedInt8Backend* gi8 = nullptr;  // set for the generated int8 path (RunOptions)
  bool rows8 = false;                   // reference: experimental 8-row packed tiles (--dram only)
};

struct Case {
  Program p;
  Bindings b;
  std::vector<float> y;
};

// Bindings hold raw pointers into the Case (its output buffer), so a Case
// must not move after it is built: it lives on the heap.
std::unique_ptr<Case> make_case(const Weight& w, const std::vector<float>& x, int64_t m, ActFormat act) {
  ProgramBuilder pb;
  const ValueId vx = pb.activation("x", DType::kF32, m, w.k);
  const ValueId vw = pb.weight("w", DType::kQ4_K, w.n, w.k);
  const ValueId vy = pb.matmul(vx, vw, kNoValue, {.act = act});
  auto p = pb.finish();
  if (!p.ok()) die(p.status().to_string());
  auto cp = std::make_unique<Case>();
  Case& c = *cp;
  c.p = std::move(*p);
  c.y.assign(static_cast<size_t>(m * w.n), std::nanf(""));
  auto lay = [](DType t, int64_t r, int64_t cols) { return *TensorLayout::contiguous(t, TensorShape{r, cols}); };
  c.b.bind(vx, TensorView(const_cast<float*>(x.data()), lay(DType::kF32, m, w.k)));
  c.b.bind(vw, TensorView(const_cast<BlockQ4_K*>(w.blocks.data()), lay(DType::kQ4_K, w.n, w.k)));
  c.b.bind(vy, TensorView(c.y.data(), lay(DType::kF32, m, w.n)));
  return cp;
}

Status run_impl(const Impl& im, Case& c) {
  RunOptions ro;
  ro.check_packed = false;  // fingerprinting a multi-MB weight per call would dominate the timing
  ro.experimental_packed_8rows = im.rows8;
  if (im.ref != nullptr) return im.ref->run(c.p, c.b, ro);
  if (im.gi8 != nullptr) return im.gi8->run(c.p, c.b, ro);
  return im.backend->run(c.p, c.b);
}

struct Result {
  double p10 = 0, p50 = 0, p90 = 0;
};

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  std::string cur;
  for (const char* p = s;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!cur.empty()) v.push_back(std::atoi(cur.c_str()));
      cur.clear();
      if (*p == '\0') break;
    } else {
      cur += *p;
    }
  }
  return v;
}

// --- DRAM-sized mode (--dram) ---------------------------------------------------------------
//
// The default cases re-run one weight, which stays in the last-level cache
// (every bench shape is <= ~8 MB). Here each implementation gets R distinct
// weight copies whose packed working set is >= max(4 x LLC, 64 MB), and every
// timed call uses the next copy round-robin: a copy is touched again only
// after > 3 x LLC of other weights has streamed through, so (for an LRU-like
// cache) its lines are gone. Evidence, printed per row: the same kernel on
// copy 0 only ("hot", cache-resident control, interleaved in the same
// rounds), the achieved weight GB/s, and its ratio to a streaming read
// bandwidth probe over 256 MB measured with the same pool.
//
// Compared: ref_int8_pk (CpuDevice interleaved sub-scaled kernel on the
// device's packed copies) and gen_int8 (generated kernel on the backend's own
// packed copies): identical weights, activations, shapes, threads, outputs
// verified for every copy before timing and re-checked after.

// Best-of-5 streaming read of `buf` with every pool thread (GB/s).
double read_bandwidth_gbs(ThreadPool& pool, const std::vector<uint64_t>& buf) {
  const size_t chunks = static_cast<size_t>(pool.size()) * 16, per = buf.size() / chunks;
  std::vector<uint64_t> sums(chunks);
  double best = 0;
  for (int rep = 0; rep < 6; ++rep) {
    const int64_t t0 = now_ns();
    pool.parallel_for(chunks, 1, [&](size_t b, size_t e) {
      for (size_t c = b; c < e; ++c) {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const uint64_t* p = buf.data() + c * per;
        for (size_t i = 0; i + 4 <= per; i += 4) s0 += p[i], s1 += p[i + 1], s2 += p[i + 2], s3 += p[i + 3];
        sums[c] = s0 + s1 + s2 + s3;
      }
    });
    const double s = static_cast<double>(now_ns() - t0) * 1e-9;
    uint64_t all = 0;
    for (uint64_t v : sums) all += v;
    bench::do_not_optimize(all);
    if (rep > 0) best = std::max(best, static_cast<double>(chunks * per * sizeof(uint64_t)) / s / 1e9);
  }
  return best;
}

int run_dram(const std::vector<int>& threads_list, int samples, double target_ms, const std::string& csv,
             const GeneratedInt8Kernel* t_int8, size_t n_int8, CpuIsa tier) {
  const CpuInfo& cpu = cpu_info();
  if (cpu.l3_bytes <= 0) {
    std::printf("DRAM mode: the last-level cache size is unknown on this system, so a working set larger than it\n"
                "cannot be sized; stopping without results.\n");
    return 2;
  }
  if (t_int8 == nullptr || n_int8 == 0) {
    std::printf("DRAM mode: this build has no generated int8 unit (ENABLE_AVX2 off); nothing to compare.\n");
    return 2;
  }
  const int64_t target_ws = std::max<int64_t>(4 * cpu.l3_bytes, int64_t{64} << 20);
  std::printf("DRAM mode: LLC %.1f MB (L2 %.2f MB per core); per-implementation packed working set >= %.0f MB, "
              "round-robin over distinct copies\n\n",
              cpu.l3_bytes / 1048576.0, cpu.l2_bytes / 1048576.0, target_ws / 1048576.0);
  std::FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (out) {
    // The default columns, then the DRAM-mode columns (text fields quoted).
    std::fprintf(out, "threads,m,n,k,impl,numerics,executes,p10_us,p50_us,p90_us,spread_pct,gflops,weight_gbs,"
                      "err_over_bound,mode,copies,working_set_mb,probe_gbs\n");
  }
  std::vector<uint64_t> probe(static_cast<size_t>((int64_t{256} << 20) / 8));
  for (size_t i = 0; i < probe.size(); ++i) probe[i] = i * 2654435761u;
  for (int threads : threads_list) {
    ThreadPool pool(threads);
    pool.pin_caller();
    if (!ReferenceBackend(pool, tier).target().int8_dot || !GeneratedInt8Backend(pool, {t_int8, n_int8}).runnable().ok()) {
      std::printf("DRAM mode needs the AVX2 tier (int8 kernels); this process selected %s.\n", isa_name(tier).data());
      if (out) std::fclose(out);
      return 2;
    }
    const double bw = read_bandwidth_gbs(pool, probe);
    std::printf("== threads %d: streaming read probe %.1f GB/s (256 MB, best of 5) ==\n", threads, bw);
    std::printf("%2s %5s %5s  %-12s %-5s %6s %7s %9s %9s %9s %6s %8s %7s %6s %7s\n", "M", "N", "K", "impl", "mode",
                "copies", "ws MB", "p10 us", "p50 us", "p90 us", "sprd%", "GFLOP/s", "GB/s", "%bw", "e/bound");
    uint32_t seed = 7000;
    const std::vector<Q4KMatmulSpec> specs = q4k_bench_specs();
    for (size_t si = 0; si < specs.size();) {
      const int64_t n = specs[si].n, k = specs[si].k;
      const int64_t packed_bytes = (n / 8) * (k / 256) * static_cast<int64_t>(sizeof(BlockQ4_Kx8));
      const int copies = static_cast<int>(std::max<int64_t>(2, (target_ws + packed_bytes - 1) / packed_bytes));
      // Fresh backends per shape: the device's packed copies are freed with
      // its backend, so recycled addresses never meet a retired registration.
      ReferenceBackend ref_pk(pool, tier), ref_pk8(pool, tier);  // separate packed copies per implementation
      GeneratedInt8Backend gen(pool, {t_int8, n_int8});
      std::vector<std::unique_ptr<Weight>> ws;
      for (int c = 0; c < copies; ++c) {
        ws.push_back(std::make_unique<Weight>(make_weight(n, k, seed++)));
        const TensorView v(ws.back()->blocks.data(), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{n, k}));
        const auto pr = ref_pk.prepack_weight(v);
        if (!pr.ok()) die("DRAM mode: reference prepack failed: " + pr.status().to_string());
        const auto pr8 = ref_pk8.prepack_weight(v);
        if (!pr8.ok()) die("DRAM mode: reference (8-row) prepack failed: " + pr8.status().to_string());
        const auto pg = gen.prepack_weight(v);
        if (!pg.ok()) die("DRAM mode: generated prepack failed: " + pg.status().to_string());
      }
      for (; si < specs.size() && specs[si].n == n && specs[si].k == k; ++si) {
        const int64_t m = specs[si].m;
        if (m < 2) continue;  // the packed reference path serves M 2..8
        const std::vector<float> x = make_x(m, k, seed++);
        struct Variant {
          std::string name, mode;
          Impl impl;
          int copies;
          std::vector<std::unique_ptr<Case>> cases;
          std::vector<double> us;
          double err = 0;
          uint64_t rows8 = 0;  // 8-row packed dispatches while this variant was timed
        };
        std::vector<Variant> vs;
        for (int which = 0; which < 3; ++which) {
          for (int dram = 1; dram >= 0; --dram) {
            Variant v;
            v.name = which == 0 ? "ref_int8_pk" : which == 1 ? "ref_int8_pk8" : "gen_int8";
            v.mode = dram ? "dram" : "hot";
            v.copies = dram ? copies : 1;
            if (which == 0) {
              v.impl = {v.name, "int8 act, sub-scaled", "CpuDevice int8 interleaved Q4_K x8", Path::kInt8Subscale,
                        nullptr, &ref_pk, ActFormat::kAuto};
            } else if (which == 1) {
              v.impl = {v.name, "int8 act, sub-scaled", "CpuDevice int8 interleaved Q4_K x8, 8-row tiles (experimental)",
                        Path::kInt8Subscale, nullptr, &ref_pk8, ActFormat::kAuto};
              v.impl.rows8 = true;
            } else {
              v.impl = {v.name, "int8 act, sub-scaled", int8_kernel_symbol(specs[si]), Path::kInt8Subscale, nullptr,
                        nullptr, ActFormat::kAuto};
              v.impl.gi8 = &gen;
            }
            for (int c = 0; c < v.copies; ++c) v.cases.push_back(make_case(*ws[static_cast<size_t>(c)], x, m, ActFormat::kAuto));
            vs.push_back(std::move(v));
          }
        }
        // Verify every copy's output (dram variants cover all copies).
        for (Variant& v : vs) {
          for (int c = 0; c < v.copies; ++c) {
            Case& cs = *v.cases[static_cast<size_t>(c)];
            const Status st = run_impl(v.impl, cs);
            if (!st.ok()) die(v.name + ": " + st.to_string());
            if (v.mode == "dram") {
              const double e = verify(*ws[static_cast<size_t>(c)], x, m, cs.y, Path::kInt8Subscale);
              if (!(e <= 1.0)) die(v.name + ": copy " + std::to_string(c) + " outside its error bound");
              v.err = std::max(v.err, e);
            }
          }
        }
        for (Variant& v : vs) {  // hot copy 0 must equal the verified dram copy 0
          if (v.mode == "hot") {
            const auto& d = std::find_if(vs.begin(), vs.end(), [&](const Variant& o) { return o.name == v.name && o.mode == "dram"; });
            if (v.cases[0]->y != d->cases[0]->y) die(v.name + ": hot and dram outputs differ for copy 0");
            v.err = d->err;
          }
        }
        // Calls per sample: whole rotations, sized to target_ms from a calibration pass.
        std::vector<int> passes(vs.size());
        for (size_t i = 0; i < vs.size(); ++i) {
          const int64_t t0 = now_ns();
          for (int c = 0; c < vs[i].copies; ++c) (void)run_impl(vs[i].impl, *vs[i].cases[static_cast<size_t>(c)]);
          const double one = std::max(1.0, static_cast<double>(now_ns() - t0));
          passes[i] = std::max(1, static_cast<int>(target_ms * 1e6 / one));
        }
        for (int round = 0; round < samples + 3; ++round) {
          for (size_t i = 0; i < vs.size(); ++i) {
            const uint64_t d0 = CpuDevice::experimental_rows8_dispatches();
            const int64_t t0 = now_ns();
            for (int p = 0; p < passes[i]; ++p) {
              for (int c = 0; c < vs[i].copies; ++c) (void)run_impl(vs[i].impl, *vs[i].cases[static_cast<size_t>(c)]);
            }
            const double per = static_cast<double>(now_ns() - t0) / (passes[i] * vs[i].copies) / 1e3;
            vs[i].rows8 += CpuDevice::experimental_rows8_dispatches() - d0;
            if (round >= 3) vs[i].us.push_back(per);
          }
        }
        double p50_of[3][2] = {};
        for (Variant& v : vs) {
          // Determinism: first and last copy reproduce their verified outputs.
          for (int c : {0, v.copies - 1}) {
            Case& cs = *v.cases[static_cast<size_t>(c)];
            const std::vector<float> before = cs.y;
            std::fill(cs.y.begin(), cs.y.end(), std::nanf(""));
            if (!run_impl(v.impl, cs).ok() || cs.y != before) die(v.name + ": output changed after timing");
          }
          std::vector<double> sorted = v.us;
          std::sort(sorted.begin(), sorted.end());
          const double p10 = bench::percentile(sorted, 10), p50 = bench::percentile(sorted, 50),
                       p90 = bench::percentile(sorted, 90);
          const double gflops = 2.0 * m * static_cast<double>(n) * k / (p50 * 1e3);
          const double gbs = static_cast<double>(packed_bytes) / (p50 * 1e3);
          const double ws_mb = static_cast<double>(packed_bytes) * v.copies / 1048576.0;
          p50_of[v.name == "ref_int8_pk" ? 0 : v.name == "ref_int8_pk8" ? 1 : 2][v.mode == "dram"] = p50;
          std::printf("%2lld %5lld %5lld  %-12s %-5s %6d %7.1f %9.1f %9.1f %9.1f %6.1f %8.2f %7.2f %5.0f%% %7.3f\n",
                      static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), v.name.c_str(),
                      v.mode.c_str(), v.copies, ws_mb, p10, p50, p90, 100.0 * (p90 - p10) / p50, gflops, gbs,
                      100.0 * gbs / bw, v.err);
          if (out) {
            std::fprintf(out, "%d,%lld,%lld,%lld,\"%s\",\"%s\",\"%s\",%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.4f,\"%s\",%d,%.2f,%.2f\n",
                         threads, static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
                         v.name.c_str(), v.impl.numerics.c_str(), v.impl.executes.c_str(), p10, p50, p90,
                         100.0 * (p90 - p10) / p50, gflops, gbs, v.err, v.mode.c_str(), v.copies, ws_mb, bw);
          }
        }
        std::printf("   8-row dispatches while timed: ref_int8_pk %llu, ref_int8_pk8 %llu, gen_int8 %llu\n",
                    static_cast<unsigned long long>(vs[0].rows8 + vs[1].rows8),
                    static_cast<unsigned long long>(vs[2].rows8 + vs[3].rows8),
                    static_cast<unsigned long long>(vs[4].rows8 + vs[5].rows8));
        std::printf("   speedup vs ref_int8_pk (dram, hot): ref_int8_pk8 %.2f, %.2f; gen_int8 %.2f, %.2f; "
                    "dram / hot time: ref %.2f, ref8 %.2f, gen %.2f\n\n",
                    p50_of[0][1] / p50_of[1][1], p50_of[0][0] / p50_of[1][0], p50_of[0][1] / p50_of[2][1],
                    p50_of[0][0] / p50_of[2][0], p50_of[0][1] / p50_of[0][0], p50_of[1][1] / p50_of[1][0],
                    p50_of[2][1] / p50_of[2][0]);
      }
      for (const auto& w : ws) {
        ref_pk.release_packed(w->blocks.data());
        ref_pk8.release_packed(w->blocks.data());
        gen.release_packed(w->blocks.data());
      }
    }
  }
  if (out) std::fclose(out);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // progress is visible during long runs
  const CpuInfo& cpu = cpu_info();
  std::vector<int> threads_list = {1, std::max(1, cpu.physical_cores)};
  int samples = 15;
  double target_ms = 20;
  std::string csv;
  bool int8_only = false, dram = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads_list = parse_list(argv[++i]);
    else if (a == "--samples" && i + 1 < argc) samples = std::max(5, std::atoi(argv[++i]));
    else if (a == "--target-ms" && i + 1 < argc) target_ms = std::max(1.0, std::atof(argv[++i]));
    else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--int8-only") int8_only = true;
    else if (a == "--dram") dram = true;
    else die("usage: bench_dynakernel_q4k [--threads 1,10] [--samples 15] [--target-ms 20] [--csv FILE] [--int8-only] [--dram]");
  }
#if defined(_WIN32)
  _putenv_s("DYNACORE_Q4_REPACK", "always");  // packing independent of free RAM
#else
  setenv("DYNACORE_Q4_REPACK", "always", 1);
#endif
  const CpuIsa tier = select_isa(cpu.features);
  size_t n_scalar = 0, n_avx2 = 0;
  const GeneratedKernel* t_scalar = dk_generated_q4k_bench_kernels(&n_scalar);
  const GeneratedKernel* t_avx2 = nullptr;
  const GeneratedInt8Kernel* t_int8 = nullptr;
  size_t n_int8 = 0;
#if DK_GENERATED_HAS_AVX2
  t_avx2 = dk_generated_q4k_bench_avx2_kernels(&n_avx2);
  t_int8 = dk_generated_q4k_bench_int8_avx2_kernels(&n_int8);
#endif
  std::printf("bench_dynakernel_q4k\n");
  std::printf("cpu: %s (%d cores, %d threads)  tier: %s  compiler: ", cpu.brand.c_str(), cpu.physical_cores,
              cpu.logical_cores, isa_name(tier).data());
#if defined(_MSC_VER)
  std::printf("MSVC %d", _MSC_FULL_VER);
#elif defined(__clang__)
  std::printf("clang %d.%d", __clang_major__, __clang_minor__);
#else
  std::printf("gcc %d.%d", __GNUC__, __GNUC_MINOR__);
#endif
#if defined(NDEBUG)
  std::printf(" (optimized, NDEBUG)");
#else
  std::printf(" (DEBUG BUILD: timings not meaningful)");
#endif
  std::printf("\nsamples %d (+3 warm-up rounds), ~%.0f ms per sample, interleaved; generated units: scalar %zu, avx2 %zu\n\n",
              samples, target_ms, n_scalar, n_avx2);
  if (dram) {
    std::printf("thread pinning: %s\n", ThreadPool::pinning_enabled() ? "on (DYNACORE_PIN_THREADS=1)" : "off");
    return run_dram(threads_list, samples, target_ms, csv, t_int8, n_int8, tier);
  }
  std::FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (out) {
    std::fprintf(out, "threads,m,n,k,impl,numerics,executes,p10_us,p50_us,p90_us,spread_pct,gflops,weight_gbs,err_over_bound\n");
  }

  std::printf("thread pinning: %s\n\n", ThreadPool::pinning_enabled() ? "on (DYNACORE_PIN_THREADS=1)" : "off");
  for (int threads : threads_list) {
    ThreadPool pool(threads);
    pool.pin_caller();  // with DYNACORE_PIN_THREADS=1: caller on core 0, worker i on core i + 1 (no-op otherwise)
    GeneratedBackend gen_scalar(pool, {t_scalar, n_scalar});
    GeneratedBackend gen_avx2(pool, {t_avx2, n_avx2});
    const bool avx2_ok = t_avx2 != nullptr && gen_avx2.runnable().ok();
    GeneratedInt8Backend gen_int8(pool, {t_int8, n_int8});
    const bool int8gen_ok = t_int8 != nullptr && gen_int8.runnable().ok();
    ReferenceBackend ref(pool, tier), ref_pk(pool, tier);
    const bool int8 = ref.target().int8_dot;
    std::printf("== threads %d ==\n", threads);
    std::printf("%5s %5s %5s  %-11s %-22s %-34s %9s %9s %9s %6s %8s %7s %7s\n", "M", "N", "K", "impl", "numerics",
                "executes", "p10 us", "p50 us", "p90 us", "sprd%", "GFLOP/s", "GB/s", "e/bound");
    uint32_t seed = 1;
    int64_t last_n = -1, last_k = -1;
    // Weights stay alive (at their addresses) until their packed copies are released.
    std::vector<std::unique_ptr<Weight>> weights;
    Weight* w = nullptr;
    for (size_t si = 0; si < n_scalar; ++si) {
      const Q4KMatmulSpec& s = t_scalar[si].spec;
      if (s.n != last_n || s.k != last_k) {
        weights.push_back(std::make_unique<Weight>(make_weight(s.n, s.k, seed++)));
        w = weights.back().get();
        last_n = s.n;
        last_k = s.k;
        if (int8 && s.n % 8 == 0) {
          auto pk = ref_pk.prepack_weight(
              TensorView(w->blocks.data(), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{s.n, s.k})));
          if (!pk.ok()) die("prepack: " + pk.status().to_string());
        }
        if (int8gen_ok) {
          auto pk = gen_int8.prepack_weight(
              TensorView(w->blocks.data(), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{s.n, s.k})));
          if (!pk.ok()) die("generated int8 prepack: " + pk.status().to_string());
        }
      }
      const std::vector<float> x = make_x(s.m, s.k, seed++);
      std::vector<Impl> impls;
      if (!int8_only) {
        impls.push_back({"gen_scalar", "fp32", kernel_symbol(s, CodegenIsa::kScalar), Path::kFp32, &gen_scalar, nullptr});
      }
      if (avx2_ok && !int8_only) {
        impls.push_back({"gen_avx2", "fp32", kernel_symbol(s, CodegenIsa::kAvx2), Path::kFp32, &gen_avx2, nullptr});
      }
      if (!int8_only) {
        impls.push_back({"ref_f32", "fp32",
                         s.m == 1 ? "CpuDevice fused dot (vec_dot_q4_K)" : "CpuDevice fp32 panel GEMM", Path::kFp32,
                         nullptr, &ref, ActFormat::kF32});
      }
      if (int8) {
        const bool per32 = s.m == 1;
        impls.push_back({"ref_int8", per32 ? "int8 act, per-32" : "int8 act, sub-scaled",
                         per32 ? "CpuDevice int8 rows (per-32)" : "CpuDevice int8 rows (sub-scaled)",
                         per32 ? Path::kInt8Per32 : Path::kInt8Subscale, nullptr, &ref, ActFormat::kAuto});
        if (s.m >= 2 && s.n % 8 == 0) {
          impls.push_back({"ref_int8_pk", "int8 act, sub-scaled", "CpuDevice int8 interleaved Q4_K x8",
                           Path::kInt8Subscale, nullptr, &ref_pk, ActFormat::kAuto});
        }
      }
      if (int8gen_ok) {
        Impl gi{"gen_int8", "int8 act, sub-scaled", int8_kernel_symbol(s), Path::kInt8Subscale, nullptr, nullptr,
                ActFormat::kAuto};
        gi.gi8 = &gen_int8;
        impls.push_back(gi);
      }
      // Set up, verify (before any timing), and size each batch.
      std::vector<std::unique_ptr<Case>> cases;
      std::vector<double> err(impls.size());
      std::vector<int> batch(impls.size());
      for (size_t i = 0; i < impls.size(); ++i) {
        cases.push_back(make_case(*w, x, s.m, impls[i].act));
        const Status st = run_impl(impls[i], *cases[i]);
        if (!st.ok()) die(impls[i].name + ": " + st.to_string());
        err[i] = verify(*w, x, s.m, cases[i]->y, impls[i].path);
        if (!(err[i] <= 1.0)) die(impls[i].name + ": output outside its error bound (" + std::to_string(err[i]) + ")");
        const int64_t t0 = now_ns();
        for (int r = 0; r < 3; ++r) (void)run_impl(impls[i], *cases[i]);
        const double one = std::max(1.0, static_cast<double>(now_ns() - t0) / 3.0);
        batch[i] = std::max(1, static_cast<int>(target_ms * 1e6 / one));
      }
      const std::vector<std::vector<float>> verified = [&] {
        std::vector<std::vector<float>> v;
        for (const auto& c : cases) v.push_back(c->y);
        return v;
      }();
      // Interleaved timing: 3 warm-up rounds, then `samples` rounds.
      std::vector<std::vector<double>> us(impls.size());
      for (int round = 0; round < samples + 3; ++round) {
        for (size_t i = 0; i < impls.size(); ++i) {
          const int64_t t0 = now_ns();
          for (int r = 0; r < batch[i]; ++r) (void)run_impl(impls[i], *cases[i]);
          const double per = static_cast<double>(now_ns() - t0) / batch[i] / 1e3;
          if (round >= 3) us[i].push_back(per);
        }
      }
      for (size_t i = 0; i < impls.size(); ++i) {
        // Determinism: a fresh run must reproduce the verified output exactly.
        std::fill(cases[i]->y.begin(), cases[i]->y.end(), std::nanf(""));
        if (!run_impl(impls[i], *cases[i]).ok() || cases[i]->y != verified[i]) {
          die(impls[i].name + ": output changed after timing");
        }
        const bench::Stats st = bench::summarize(us[i]);
        std::vector<double> sorted = us[i];
        std::sort(sorted.begin(), sorted.end());
        const double p10 = bench::percentile(sorted, 10);
        const double flops = 2.0 * s.m * static_cast<double>(s.n) * s.k / (st.p50 * 1e3);
        const double gbs = static_cast<double>(w->blocks.size() * sizeof(BlockQ4_K)) / (st.p50 * 1e3);
        const double spread = 100.0 * (st.p90 - p10) / st.p50;
        std::printf("%5d %5lld %5lld  %-11s %-22s %-34s %9.1f %9.1f %9.1f %6.1f %8.2f %7.2f %7.3f\n", s.m,
                    static_cast<long long>(s.n), static_cast<long long>(s.k), impls[i].name.c_str(),
                    impls[i].numerics.c_str(), impls[i].executes.c_str(), p10, st.p50, st.p90, spread, flops, gbs, err[i]);
        if (out) {
          // Text fields quoted: the numerics labels contain commas.
          std::fprintf(out, "%d,%d,%lld,%lld,\"%s\",\"%s\",\"%s\",%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.4f\n", threads, s.m,
                       static_cast<long long>(s.n), static_cast<long long>(s.k), impls[i].name.c_str(),
                       impls[i].numerics.c_str(), impls[i].executes.c_str(), p10, st.p50, st.p90, spread, flops, gbs,
                       err[i]);
        }
      }
      std::printf("\n");
    }
    for (const auto& wt : weights) {
      ref_pk.release_packed(wt->blocks.data());
      if (int8gen_ok) gen_int8.release_packed(wt->blocks.data());
    }
  }
  if (out) std::fclose(out);
  return 0;
}

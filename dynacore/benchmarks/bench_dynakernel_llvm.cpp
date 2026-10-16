// LLVM-generated INT8 Q4_K kernels against the existing paths (docs/dynakernel.md,
// "LLVM native backend"). A kernel microbenchmark: it is not evidence about
// end-to-end inference.
//
//   bench_dynakernel_llvm [--threads 1,10] [--m 1,2,4,5,8] [--samples 15] [--target-ms 20] [--csv FILE]
//   bench_dynakernel_llvm --breakdown [--dram] [same options]   (per-stage timing; see run_breakdown)
//   bench_dynakernel_llvm --bandwidth [--chunks 32,8] [--prefetch 0,2432] [--vnni] [same options]   (kernel-only GB/s; run_bandwidth)
//   bench_dynakernel_llvm --layouts x8,h,s,hs [--calls] [same options]   (compact packed layouts; run_layouts)
//
// Needs the LLVM C API library (DYNACORE_LLVM_C) and the AVX2 tier. Per shape
// (Qwen2.5-1.5B projections, N x K = 1536 x 1536, 8960 x 1536, 1536 x 8960),
// row count M and thread count, the same weights and activations go through:
//   ref          ReferenceBackend, auto contract, GGUF weights (DynaCore's
//                default dispatch: M 1 int8 per-32, M 2-8 int8 sub-scaled
//                rows, unpacked)
//   ref_pk       ReferenceBackend with the interleaved copy (DD-078): DynaCore's
//                packed int8 sub-scaled kernel; M 5-8 with the experimental
//                8-row tiles (RunOptions::experimental_packed_8rows)
//   ref_pk_vnni  as ref_pk with DynaCore's opt-in AVX-VNNI kernel (DD-080,
//                DYNACORE_VNNI=1), M 2-4 only (it has no 8-row variant)
//   gen_int8     generated C++ INT8 kernel (GeneratedInt8Backend), M 1, 2, 4, 8
//   llvm_*       LlvmInt8Backend, lowering portable / avx2 / avxvnni
// gen_int8 and llvm_* quantize activations with DynaCore's quantize_act_sx
// and run on the same packed copy layout and work split.
//
// Every output is checked against the double-precision Q4_K decode within its
// path's bound before timing, and must be unchanged after timing; llvm_*
// outputs must equal gen_int8's bit for bit. Compilation is timed separately
// (once per kernel, before any timed run). Samples interleave the
// implementations; p10/p50/p90 are per call. Weights are at most 7.7 MB per
// shape, so they stay largely cache-resident: this measures compute, not
// DRAM bandwidth.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/compact_layout.h"
#include "dynacore/dynakernel/generated_backend.h"
#include "dynacore/dynakernel/llvm_backend.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/quantization/repack.h"
#include "q4k_reference.h"

#if DK_GENERATED_HAS_AVX2
extern "C" const dynacore::dk::GeneratedInt8Kernel* dk_generated_q4k_bench_int8_avx2_kernels(size_t* count);
#endif

using namespace dynacore;
using namespace dynacore::dk;
using q4kref::Path;
using q4kref::Weight;

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "bench_dynakernel_llvm: %s\n", m.c_str());
  std::exit(1);
}

void set_env(const char* k, const char* v) {
#if defined(_WIN32)
  _putenv_s(k, v);
#else
  if (*v) setenv(k, v, 1);
  else unsetenv(k);
#endif
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    v.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return v;
}

struct Case {
  Program p;
  Bindings b;
  std::vector<float> y;
};

std::unique_ptr<Case> make_case(const Weight& w, const std::vector<float>& x, int64_t m) {
  ProgramBuilder pb;
  const ValueId vx = pb.activation("x", DType::kF32, m, w.k);
  const ValueId vw = pb.weight("w", DType::kQ4_K, w.n, w.k);
  const ValueId vy = pb.matmul(vx, vw);
  auto p = pb.finish();
  if (!p.ok()) die(p.status().to_string());
  auto c = std::make_unique<Case>();
  c->p = std::move(*p);
  c->y.assign(static_cast<size_t>(m * w.n), std::nanf(""));
  auto lay = [](DType t, int64_t r, int64_t cols) { return *TensorLayout::contiguous(t, TensorShape{r, cols}); };
  c->b.bind(vx, TensorView(const_cast<float*>(x.data()), lay(DType::kF32, m, w.k)));
  c->b.bind(vw, TensorView(const_cast<quant::BlockQ4_K*>(w.blocks.data()), lay(DType::kQ4_K, w.n, w.k)));
  c->b.bind(vy, TensorView(c->y.data(), lay(DType::kF32, m, w.n)));
  return c;
}

struct Impl {
  std::string name, numerics;
  Path path;
  ReferenceBackend* ref = nullptr;
  bool rows8 = false;
  GeneratedInt8Backend* gen = nullptr;
  LlvmInt8Backend* llvm = nullptr;
  std::unique_ptr<Case> c;
  std::vector<double> us;  // per-call samples
  std::vector<float> verified;
};

Status run_impl(Impl& im) {
  RunOptions ro;
  ro.check_packed = false;  // re-fingerprinting MBs of weights per call would dominate
  ro.experimental_packed_8rows = im.rows8;
  if (im.ref != nullptr) return im.ref->run(im.c->p, im.c->b, ro);
  if (im.gen != nullptr) return im.gen->run(im.c->p, im.c->b, ro);
  return im.llvm->run(im.c->p, im.c->b, ro);
}

double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(std::min<double>(static_cast<double>(v.size() - 1), std::floor(q * (v.size() - 1) + 0.5)))];
}

// --- --breakdown: where the time of one LLVM matmul goes ---------------------------------------
//
// Times, per call and in isolation (interleaved samples, same pool):
//   ref_pk    ReferenceBackend, prepacked (the DynaCore packed path)
//   llvm      LlvmInt8Backend::run (prepare + executor)
//   llvm_pq   the same with RunOptions::experimental_parallel_quant (default threshold)
//   llvm_pqa  the same with parallel quantization at any size (threshold 0)
//   exec      GeneratedInt8Backend::run on the same compiled kernel (executor only)
//   quant     quantize_act_sx over the M rows, serial (what the executor does)
//   kernel    the compiled kernel in one parallel region over 8-row groups,
//             executor grain, pre-quantized activations
//   region    the same parallel region with an empty body (fork/join floor)
//   checks    GeneratedInt8Backend::check + validate_bindings
//   prepare   LlvmInt8Backend::prepare on an already compiled program
// --dram rotates over weight copies (identical bytes, distinct memory, each
// implementation its own packed copies) whose packed working set is at least
// max(4 x LLC, 64 MB), so every call streams its weights from DRAM.
struct Component {
  std::string name;
  std::function<void()> fn;
  std::vector<double> us;
};

void time_components(std::vector<Component>& comps, int samples, double target_ms) {
  std::vector<int> batch(comps.size(), 1);
  for (size_t i = 0; i < comps.size(); ++i) {
    for (int r = 0; r < 3; ++r) comps[i].fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; ++r) comps[i].fn();
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
    batch[i] = std::max(1, static_cast<int>(target_ms / std::max(one, 1e-4)));
  }
  for (int s = 0; s < samples; ++s) {
    for (size_t i = 0; i < comps.size(); ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < batch[i]; ++r) comps[i].fn();
      comps[i].us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / batch[i]);
    }
  }
}

int run_breakdown(const std::vector<int>& threads_list, const std::vector<int>& ms, int samples, double target_ms,
                  const std::string& csv, bool dram) {
  const CpuInfo& cpu = cpu_info();
  const int64_t target_ws = std::max<int64_t>(4 * cpu.l3_bytes, int64_t{64} << 20);
  std::printf("breakdown (%s): LLC %.1f MB%s; samples %d x ~%.0f ms; pin %s\n", dram ? "dram" : "hot",
              cpu.l3_bytes / 1048576.0, dram ? (", working set >= " + std::to_string(target_ws >> 20) + " MB").c_str() : "",
              samples, target_ms, std::getenv("DYNACORE_PIN_THREADS") ? std::getenv("DYNACORE_PIN_THREADS") : "unset");
  FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (!csv.empty() && out == nullptr) die("cannot write " + csv);
  if (out) std::fprintf(out, "mode,copies,threads,m,n,k,component,p10_us,p50_us,p90_us\n");
  const CpuKernels kern = make_cpu_kernels(CpuIsa::kAvx2);
  const int64_t shapes[][2] = {{1536, 1536}, {8960, 1536}, {1536, 8960}};
  uint32_t seed = 900;
  for (const auto& sh : shapes) {
    const int64_t n = sh[0], k = sh[1], sb = k / 256, groups = n / 8;
    const int64_t packed_bytes = groups * sb * static_cast<int64_t>(sizeof(BlockQ4_Kx8));
    const int copies = dram ? static_cast<int>(std::max<int64_t>(2, (target_ws + packed_bytes - 1) / packed_bytes)) : 1;
    const Weight w0 = q4kref::make_weight(n, k, seed++);
    std::vector<std::vector<quant::BlockQ4_K>> wc(static_cast<size_t>(copies), w0.blocks);  // copy 0 is verified
    std::vector<std::vector<BlockQ4_Kx8>> packed(static_cast<size_t>(copies));
    for (int c = 0; c < copies; ++c) {
      packed[static_cast<size_t>(c)].resize(static_cast<size_t>(groups * sb));
      for (int64_t g = 0; g < groups; ++g) {
        repack_q4_K_x8(wc[static_cast<size_t>(c)].data(), dtype_row_bytes(DType::kQ4_K, k), 8 * g, k,
                       packed[static_cast<size_t>(c)].data() + g * sb);
      }
    }
    auto wview_c = [&](int c) {
      return TensorView(wc[static_cast<size_t>(c)].data(), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{n, k}));
    };
    for (int t : threads_list) {
      ThreadPool pool(t);
      pool.pin_caller();  // DYNACORE_PIN_THREADS=1: the calling thread (all work at 1 thread) on the first core
      ReferenceBackend ref_pk(pool, CpuIsa::kAvx2);
      LlvmInt8Backend llvm(pool, LlvmLowering::kAvx2);
      for (int c = 0; c < copies; ++c) {
        if (!ref_pk.prepack_weight(wview_c(c)).ok() || !llvm.prepack_weight(wview_c(c)).ok()) die("prepack failed");
      }
      std::printf("\n== N %lld K %lld, %d thread(s), %d weight cop%s (%.1f MB packed each)\n", static_cast<long long>(n),
                  static_cast<long long>(k), t, copies, copies == 1 ? "y" : "ies", packed_bytes / 1048576.0);
      std::printf("%2s  %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s | %6s %6s %6s %6s %6s\n", "M", "ref_pk", "llvm", "llvm_pq",
                  "llvm_pqa", "exec", "quant", "kernel", "region", "checks", "prepare", "q/llvm", "ovh/ll", "k/rpk", "pq/ll",
                  "pqa/ll");
      for (int m : ms) {
        const std::vector<float> x = q4kref::make_x(m, k, k, seed++);
        ProgramBuilder pb;
        const ValueId vx = pb.activation("x", DType::kF32, m, k);
        const ValueId vw = pb.weight("w", DType::kQ4_K, n, k);
        const ValueId vy = pb.matmul(vx, vw);
        const Program p = *pb.finish();
        std::vector<float> y_ref(static_cast<size_t>(m * n)), y_llvm(y_ref.size()), y_exec(y_ref.size()), y_k(y_ref.size()),
            y_pq(y_ref.size()), y_pqa(y_ref.size());
        auto bind_c = [&](int c, std::vector<float>& y) {
          Bindings b;
          auto lay = [](DType dt, int64_t r, int64_t cols) { return *TensorLayout::contiguous(dt, TensorShape{r, cols}); };
          b.bind(vx, TensorView(const_cast<float*>(x.data()), lay(DType::kF32, m, k)));
          b.bind(vw, wview_c(c));
          b.bind(vy, TensorView(y.data(), lay(DType::kF32, m, n)));
          return b;
        };
        std::vector<Bindings> b_ref, b_llvm, b_exec, b_pq, b_pqa;
        for (int c = 0; c < copies; ++c) {
          b_ref.push_back(bind_c(c, y_ref));
          b_llvm.push_back(bind_c(c, y_llvm));
          b_exec.push_back(bind_c(c, y_exec));
          b_pq.push_back(bind_c(c, y_pq));
          b_pqa.push_back(bind_c(c, y_pqa));
        }
        if (const Status s = llvm.prepare(p); !s.ok()) die(s.to_string());
        const GeneratedInt8Kernel kentry = *llvm.find(p.ops()[0]);
        GeneratedInt8Backend exec(pool, {&kentry, 1});
        for (int c = 0; c < copies; ++c) {
          if (!exec.prepack_weight(wview_c(c)).ok()) die("prepack (exec) failed");
        }
        RunOptions ro;
        ro.check_packed = false;
        ro.experimental_packed_8rows = m > 4;
        RunOptions ro_pq = ro, ro_pqa = ro;
        ro_pq.experimental_parallel_quant = ro_pqa.experimental_parallel_quant = true;
        ro_pqa.parallel_quant_min_superblocks = 0;
        const int64_t nb = k / kActQ8Block;
        std::vector<ActBlockQ8> act(static_cast<size_t>(m * nb));
        const size_t grain = std::max<size_t>(1, static_cast<size_t>(groups) / (static_cast<size_t>(pool.size()) * 32));
        int c_ref = 0, c_llvm = 0, c_exec = 0, c_k = 0, c_pq = 0, c_pqa = 0;
        auto next = [copies](int& c) {
          const int v = c;
          c = (c + 1) % copies;
          return v;
        };
        std::vector<Component> comps;
        comps.push_back({"ref_pk", [&] { (void)ref_pk.run(p, b_ref[static_cast<size_t>(next(c_ref))], ro); }, {}});
        comps.push_back({"llvm", [&] { (void)llvm.run(p, b_llvm[static_cast<size_t>(next(c_llvm))], ro); }, {}});
        comps.push_back({"llvm_pq", [&] { (void)llvm.run(p, b_pq[static_cast<size_t>(next(c_pq))], ro_pq); }, {}});
        comps.push_back({"llvm_pqa", [&] { (void)llvm.run(p, b_pqa[static_cast<size_t>(next(c_pqa))], ro_pqa); }, {}});
        comps.push_back({"exec", [&] { (void)exec.run(p, b_exec[static_cast<size_t>(next(c_exec))], ro); }, {}});
        comps.push_back({"quant", [&] {
                           for (int64_t r = 0; r < m; ++r) kern.quantize_act_sx(x.data() + r * k, act.data() + r * nb, k);
                         }, {}});
        comps.push_back({"kernel", [&] {
                           const BlockQ4_Kx8* wp = packed[static_cast<size_t>(next(c_k))].data();
                           const auto* xa = reinterpret_cast<const ActSuperQ8*>(act.data());
                           pool.parallel_for(static_cast<size_t>(groups), grain, [&](size_t g0, size_t g1) {
                             kentry.fn(wp, xa, nullptr, y_k.data(), n, static_cast<int64_t>(g0), static_cast<int64_t>(g1));
                           });
                         }, {}});
        comps.push_back({"region", [&] { pool.parallel_for(static_cast<size_t>(groups), grain, [](size_t, size_t) {}); }, {}});
        comps.push_back({"checks", [&] {
                           (void)exec.check(p);
                           (void)validate_bindings(p, b_exec[0]);
                         }, {}});
        comps.push_back({"prepare", [&] { (void)llvm.prepare(p); }, {}});
        // Correctness before timing: llvm == llvm_pq == llvm_pqa == exec == kernel-only, bit for bit;
        // llvm and ref within bound.
        for (int c = 0; c < copies; ++c) {
          for (int i = 0; i < 7; ++i) comps[static_cast<size_t>(i)].fn();
          if (y_llvm != y_exec || y_llvm != y_k || y_llvm != y_pq || y_llvm != y_pqa) {
            die("breakdown: llvm, parallel-quant, exec and kernel-only outputs differ");
          }
        }
        q4kref::Stats st;
        q4kref::check_bound(w0, x, k, nullptr, y_llvm.data(), n, m, Path::kInt8Subscale, st);
        q4kref::check_bound(w0, x, k, nullptr, y_ref.data(), n, m, m == 1 ? Path::kInt8Per32 : Path::kInt8Subscale, st);
        const std::vector<float> verified = y_llvm;
        if (pool.misuse_count() != 0) die("thread pool misuse");
        time_components(comps, samples, target_ms);
        comps[1].fn();
        comps[2].fn();
        comps[3].fn();
        if (y_llvm != verified || y_pq != verified || y_pqa != verified) die("breakdown: llvm output changed after timing");
        if (pool.misuse_count() != 0) die("thread pool misuse");
        double p50[10];
        for (size_t i = 0; i < comps.size(); ++i) {
          p50[i] = pct(comps[i].us, 0.5);
          if (out) {
            std::fprintf(out, "%s,%d,%d,%d,%lld,%lld,%s,%.2f,%.2f,%.2f\n", dram ? "dram" : "hot", copies, t, m,
                         static_cast<long long>(n), static_cast<long long>(k), comps[i].name.c_str(), pct(comps[i].us, 0.1),
                         p50[i], pct(comps[i].us, 0.9));
          }
        }
        // q/llvm: serial quantization share of the LLVM call; ovh/ll: llvm - quant - kernel share; k/rpk: kernel-only
        // vs ref_pk; pq/ll, pqa/ll: parallel-quant time over serial-quant time (< 1 = faster).
        std::printf("%2d  %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f | %5.0f%% %5.0f%% %5.2fx %5.3f %5.3f\n", m,
                    p50[0], p50[1], p50[2], p50[3], p50[4], p50[5], p50[6], p50[7], p50[8], p50[9], 100 * p50[5] / p50[1],
                    100 * (p50[1] - p50[5] - p50[6]) / p50[1], p50[0] / p50[6], p50[2] / p50[1], p50[3] / p50[1]);
      }
    }
  }
  if (out) std::fclose(out);
  return 0;
}

std::atomic<uint64_t> g_sink{0};  // keeps probe and traversal reads alive

// --- --bandwidth: kernel-only DRAM bandwidth (Phase 3) -----------------------------------------
//
// All kernels run on ONE shared set of packed BlockQ4_Kx8 copies (identical
// bytes, distinct memory) and the same pre-quantized activations, in one
// parallel region over 8-row groups with the executor's grain, on the same
// pool. Nothing else is timed: no quantization, checks, allocation or packing.
//   dc        DynaCore's packed kernel (dot_q8_sx_x8_q4_K, 4-row tiles; M 5-8
//             with the experimental 8-row kernel), called as CpuDevice does
//   gen       generated C++ INT8 kernel (bench table: M 1, 2, 4, 8)
//   llvm      LLVM avx2 lowering (compiled once per spec before timing)
//   traverse  the same region and chunking over the same bytes, summing them:
//             no compute, the achievable rate for this access pattern
// "dram" rotates over C copies whose total is >= max(4 x LLC, 64 MB); "hot"
// re-runs copy 0 (cache-resident control), interleaved in the same rounds.
// GB/s = packed weight bytes per call / time: an ESTIMATE of application
// traffic (activations, ~0.3 KB per 256 values per row, stay in cache and are
// not counted; outputs, M x N x 4 bytes, are not counted). No hardware
// counters on this platform, so actual DRAM traffic is not measured.
// probe: streaming read of 256 MB with the same pool (median of 10 passes).
double read_probe_gbs(ThreadPool& pool, const std::vector<uint64_t>& buf, double* best) {
  const size_t chunks = static_cast<size_t>(pool.size()) * 16, per = buf.size() / chunks;
  std::vector<uint64_t> sums(chunks);
  std::vector<double> gbs;
  for (int rep = 0; rep < 11; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    pool.parallel_for(chunks, 1, [&](size_t b, size_t e) {
      for (size_t c = b; c < e; ++c) {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        const uint64_t* p = buf.data() + c * per;
        for (size_t i = 0; i + 4 <= per; i += 4) s0 += p[i], s1 += p[i + 1], s2 += p[i + 2], s3 += p[i + 3];
        sums[c] = s0 + s1 + s2 + s3;
      }
    });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rep > 0) gbs.push_back(static_cast<double>(chunks * per * sizeof(uint64_t)) / s / 1e9);
  }
  uint64_t all = 0;
  for (uint64_t v : sums) all += v;
  g_sink.fetch_add(all, std::memory_order_relaxed);
  *best = *std::max_element(gbs.begin(), gbs.end());
  return pct(gbs, 0.5);
}

int run_bandwidth(const std::vector<int>& threads_list, const std::vector<int>& ms, int samples, double target_ms,
                  const std::string& csv, const std::vector<int>& chunks_per_thread, const std::vector<int>& prefetch,
                  bool vnni) {
  const CpuInfo& cpu = cpu_info();
  const int64_t target_ws = std::max<int64_t>(4 * cpu.l3_bytes, int64_t{64} << 20);
  std::printf("bandwidth: LLC %.1f MB, dram working set >= %lld MB; samples %d x ~%.0f ms; pin %s\n", cpu.l3_bytes / 1048576.0,
              static_cast<long long>(target_ws >> 20), samples, target_ms,
              std::getenv("DYNACORE_PIN_THREADS") ? std::getenv("DYNACORE_PIN_THREADS") : "unset");
  FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (!csv.empty() && out == nullptr) die("cannot write " + csv);
  if (out) std::fprintf(out, "threads,m,n,k,impl,mode,copies,p10_us,p50_us,p90_us,gbs,probe_gbs\n");
  const CpuKernels kern = make_cpu_kernels(CpuIsa::kAvx2);
  size_t ngen = 0;
#if DK_GENERATED_HAS_AVX2
  const GeneratedInt8Kernel* gen_k = dk_generated_q4k_bench_int8_avx2_kernels(&ngen);
#else
  const GeneratedInt8Kernel* gen_k = nullptr;
#endif
  auto jit = LlvmJit::create(llvm_lowering_target(LlvmLowering::kAvx2));
  if (!jit.ok()) die(jit.status().to_string());
  std::map<std::string, GeneratedQ4KInt8Fn> compiled;  // across thread counts
  const bool with_vnni = vnni && llvm_lowering_runnable(LlvmLowering::kAvxVnni).ok();
  if (vnni && !with_vnni) std::printf("--vnni: the avxvnni lowering is not runnable here; skipped\n");
  auto jit_vnni = LlvmJit::create(llvm_lowering_target(LlvmLowering::kAvxVnni));
  if (with_vnni && !jit_vnni.ok()) die(jit_vnni.status().to_string());
  std::vector<uint64_t> probe_buf(static_cast<size_t>((int64_t{256} << 20) / 8));
  for (size_t i = 0; i < probe_buf.size(); ++i) probe_buf[i] = i * 2654435761u;
  const int64_t shapes[][2] = {{1536, 1536}, {8960, 1536}, {1536, 8960}};
  uint32_t seed = 1300;
  for (int t : threads_list) {
    ThreadPool pool(t);
    pool.pin_caller();
    double probe_best = 0;
    const double probe = read_probe_gbs(pool, probe_buf, &probe_best);
    std::printf("\n== %d thread(s): streaming read probe %.1f GB/s median, %.1f best\n", t, probe, probe_best);
    for (const auto& sh : shapes) {
      const int64_t n = sh[0], k = sh[1], sb = k / 256, groups = n / 8;
      const int64_t packed_bytes = groups * sb * static_cast<int64_t>(sizeof(BlockQ4_Kx8));
      const int copies = static_cast<int>(std::max<int64_t>(2, (target_ws + packed_bytes - 1) / packed_bytes));
      const Weight w0 = q4kref::make_weight(n, k, seed++);
      std::vector<std::vector<BlockQ4_Kx8>> packed(static_cast<size_t>(copies));
      packed[0].resize(static_cast<size_t>(groups * sb));
      for (int64_t g = 0; g < groups; ++g) {
        repack_q4_K_x8(w0.blocks.data(), dtype_row_bytes(DType::kQ4_K, k), 8 * g, k, packed[0].data() + g * sb);
      }
      for (int c = 1; c < copies; ++c) packed[static_cast<size_t>(c)] = packed[0];
      std::printf("-- N %lld K %lld: %.2f MB packed per call, %d copies (%.0f MB)\n", static_cast<long long>(n),
                  static_cast<long long>(k), packed_bytes / 1048576.0, copies, copies * packed_bytes / 1048576.0);
      std::printf("%2s %-12s %5s %9s %9s %9s %8s %6s\n", "M", "impl", "mode", "p10 us", "p50 us", "p90 us", "GB/s", "%probe");
      for (int m : ms) {
        const std::vector<float> x = q4kref::make_x(m, k, k, seed++);
        const int64_t nb = k / kActQ8Block;
        std::vector<ActBlockQ8> act(static_cast<size_t>(m * nb));
        for (int64_t r = 0; r < m; ++r) kern.quantize_act_sx(x.data() + r * k, act.data() + r * nb, k);
        const auto* xa = reinterpret_cast<const ActSuperQ8*>(act.data());
        const ActBlockQ8* xr[kPackedRowsMax];
        for (int r = 0; r < m; ++r) xr[r] = act.data() + r * nb;
        const Q4KMatmulSpec spec{m, n, k, false};
        const GeneratedInt8Kernel* gk = nullptr;
        for (size_t i = 0; i < ngen; ++i) gk = gen_k[i].spec == spec ? &gen_k[i] : gk;
        struct Impl2 {
          std::string name;
          std::function<void(const BlockQ4_Kx8*, float*, size_t, size_t)> fn;  // groups [g0, g1)
          int cpt;                                                            // chunks per thread
          std::vector<float> y;
        };
        std::vector<Impl2> impls;
        auto dc = [&, m, k, sb, n](const BlockQ4_Kx8* wp, float* y, size_t g0, size_t g1) {
          float o[8 * kPackedRowsMax];
          for (size_t g = g0; g < g1; ++g) {
            if (m > kDotRowsMax && kern.dot_q8_sx_x8_q4_K_rows8 != nullptr) {
              kern.dot_q8_sx_x8_q4_K_rows8(wp + g * sb, xr, m, k, o);
              for (int r = 0; r < m; ++r) std::memcpy(y + r * n + 8 * g, o + 8 * r, 32);
            } else {
              for (int r0 = 0; r0 < m; r0 += kDotRowsMax) {
                const int mm = std::min(kDotRowsMax, m - r0);
                kern.dot_q8_sx_x8_q4_K(wp + g * sb, xr + r0, mm, k, o);
                for (int r = 0; r < mm; ++r) std::memcpy(y + (r0 + r) * n + 8 * g, o + 8 * r, 32);
              }
            }
          }
        };
        impls.push_back({"dc", dc, 32, {}});
        if (gk != nullptr) {
          impls.push_back({"gen", [&, gk, n](const BlockQ4_Kx8* wp, float* y, size_t g0, size_t g1) {
                             gk->fn(wp, xa, nullptr, y, n, static_cast<int64_t>(g0), static_cast<int64_t>(g1));
                           }, 32, {}});
        }
        // LLVM kernels: the control (no prefetch) at every chunking, then prefetch variants at default chunking.
        std::vector<std::pair<std::string, GeneratedQ4KInt8Fn>> lk;
        for (int pf : prefetch) {
          LlvmKernelOptions ko;
          ko.prefetch_bytes = pf;
          const std::string sym = llvm_kernel_symbol(spec, LlvmLowering::kAvx2, ko);
          if (!compiled.count(sym)) {
            auto ir = emit_q4k_int8_llvm_ir(spec, LlvmLowering::kAvx2, ko);
            if (!ir.ok()) die(ir.status().to_string());
            LlvmCompileStats st;
            auto c = (*jit)->compile(*ir, sym, &st);
            if (!c.ok()) die(c.status().to_string());
            compiled[sym] = reinterpret_cast<GeneratedQ4KInt8Fn>(c->fn);
            std::printf("   compiled %s: %.1f ms (setup, not timed)\n", sym.c_str(), st.total_ms);
          }
          lk.push_back({pf == 0 ? "llvm" : "llvm_pf" + std::to_string(pf), compiled[sym]});
        }
        if (with_vnni) {  // the AVX-VNNI lowering, default options, on its own JIT (its target keeps +avxvnni)
          const std::string sym = llvm_kernel_symbol(spec, LlvmLowering::kAvxVnni);
          if (!compiled.count(sym)) {
            LlvmCompileStats st;
            auto c = (*jit_vnni)->compile_kernel(spec, LlvmLowering::kAvxVnni, &st);
            if (!c.ok()) die(c.status().to_string());
            compiled[sym] = reinterpret_cast<GeneratedQ4KInt8Fn>(c->fn);
            std::printf("   compiled %s: %.1f ms (setup, not timed)\n", sym.c_str(), st.total_ms);
          }
          lk.push_back({"llvm_vnni", compiled[sym]});
        }
        for (const auto& [name, fn] : lk) {
          for (int cpt : chunks_per_thread) {
            if (name != "llvm" && cpt != chunks_per_thread.front()) continue;
            const std::string nm = cpt == 32 ? name : name + "_c" + std::to_string(cpt);
            GeneratedQ4KInt8Fn f = fn;
            impls.push_back({nm, [f, xa, n](const BlockQ4_Kx8* wp, float* y, size_t g0, size_t g1) {
                               f(wp, xa, nullptr, y, n, static_cast<int64_t>(g0), static_cast<int64_t>(g1));
                             }, cpt, {}});
          }
        }
        impls.push_back({"traverse", [sb](const BlockQ4_Kx8* wp, float*, size_t g0, size_t g1) {
                           const auto* p = reinterpret_cast<const uint64_t*>(wp + g0 * sb);
                           const size_t words = (g1 - g0) * static_cast<size_t>(sb) * sizeof(BlockQ4_Kx8) / 8;
                           uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                           for (size_t i = 0; i + 4 <= words; i += 4) s0 += p[i], s1 += p[i + 1], s2 += p[i + 2], s3 += p[i + 3];
                           g_sink.fetch_add(s0 + s1 + s2 + s3, std::memory_order_relaxed);
                         }, 32, {}});
        for (Impl2& im : impls) im.y.assign(static_cast<size_t>(m * n), std::nanf(""));
        // Components: each impl in dram (rotating copies) and hot (copy 0) mode.
        std::vector<Component> comps;
        std::vector<int> cidx(impls.size(), 0);
        for (size_t i = 0; i < impls.size(); ++i) {
          const size_t grain = std::max<size_t>(1, static_cast<size_t>(groups) / (static_cast<size_t>(t) * impls[i].cpt));
          for (int dram = 1; dram >= 0; --dram) {
            comps.push_back({impls[i].name + (dram ? "|dram" : "|hot"), [&, i, grain, dram, copies] {
                               const int c = dram ? cidx[i] : 0;
                               if (dram) cidx[i] = (cidx[i] + 1) % copies;
                               const BlockQ4_Kx8* wp = packed[static_cast<size_t>(c)].data();
                               float* y = impls[i].y.data();
                               pool.parallel_for(static_cast<size_t>(groups), grain,
                                                 [&](size_t g0, size_t g1) { impls[i].fn(wp, y, g0, g1); });
                             }, {}});
          }
        }
        // Correctness before timing: every copy's output (copies are identical bytes) equals copy 0's;
        // llvm variants == gen bit for bit; dc and llvm within the double-precision bound.
        auto verify_all = [&](const char* when) {
          const std::vector<float>* ref_llvm = nullptr;
          for (size_t i = 0; i < impls.size(); ++i) {
            if (impls[i].name == "traverse") continue;
            std::vector<float> first;
            for (int c = 0; c < copies; ++c) {
              comps[2 * i].fn();  // dram: next copy
              if (c == 0) first = impls[i].y;
              else if (impls[i].y != first) die(impls[i].name + ": copies give different outputs " + when);
            }
            comps[2 * i + 1].fn();  // hot
            if (impls[i].y != first) die(impls[i].name + ": hot and dram outputs differ " + when);
            if (impls[i].name.rfind("llvm", 0) == 0) {  // every LLVM variant (incl. vnni) == the control, bit for bit
              if (ref_llvm == nullptr) ref_llvm = &impls[i].y;
              else if (impls[i].y != *ref_llvm) die(impls[i].name + ": differs from the llvm control " + when);
            }
            if (impls[i].name == "gen" || impls[i].name == "dc" || &impls[i].y == ref_llvm) {
              q4kref::Stats st;
              q4kref::check_bound(w0, x, k, nullptr, impls[i].y.data(), n, m, Path::kInt8Subscale, st);
            }
          }
          for (size_t i = 0; i < impls.size(); ++i) {
            if (impls[i].name == "gen" && ref_llvm != nullptr && impls[i].y != *ref_llvm) die("llvm != gen " + std::string(when));
          }
        };
        verify_all("before timing");
        time_components(comps, samples, target_ms);
        verify_all("after timing");
        if (pool.misuse_count() != 0) die("thread pool misuse");
        for (size_t i = 0; i < comps.size(); ++i) {
          const double p10 = pct(comps[i].us, 0.1), p50 = pct(comps[i].us, 0.5), p90 = pct(comps[i].us, 0.9);
          const double gbs = packed_bytes / (p50 * 1e3);
          const std::string& cn = comps[i].name;
          const std::string name = cn.substr(0, cn.find('|')), mode = cn.substr(cn.find('|') + 1);
          std::printf("%2d %-12s %5s %9.1f %9.1f %9.1f %8.2f %5.0f%%\n", m, name.c_str(), mode.c_str(), p10, p50, p90, gbs,
                      100 * gbs / probe);
          if (out) {
            std::fprintf(out, "%d,%d,%lld,%lld,\"%s\",%s,%d,%.2f,%.2f,%.2f,%.3f,%.3f\n", t, m, static_cast<long long>(n),
                         static_cast<long long>(k), name.c_str(), mode.c_str(), mode == "dram" ? copies : 1, p10, p50, p90,
                         gbs, probe);
          }
        }
      }
    }
  }
  if (out) std::fclose(out);
  return 0;
}

// --- --layouts: compact packed layouts (Phase 4) -----------------------------------------------
//
// Per shape, row count and thread count, for every requested layout L (x8 =
// BlockQ4_Kx8, the control; h, s, hs: compact_layout.h):
//   k_L     LLVM avx2 kernel compiled for L, kernel only (pre-quantized
//           activations, one region, executor grain), on L's own packed copies
//   trav_L  the same region over the same bytes, no compute
//   call_L  (--calls) LlvmInt8Backend::run with layout L: quantization, checks,
//           region and kernel, on rotating GGUF source copies prepacked by it
//   dc      DynaCore's packed kernel on the x8 copies (context)
// "dram" rotates over copies whose total is >= max(4 x LLC, 64 MB) for each
// layout separately; "hot" re-runs copy 0. GB/s = layout bytes per call / time
// (estimate; see --bandwidth). Packing time: one full-matrix pack, median of 5.
// All outputs equal k_x8's bit for bit (same values in every layout); k_x8 and
// dc are within the double-precision bound. Checked before and after timing.
int run_layouts(const std::vector<int>& threads_list, const std::vector<int>& ms, int samples, double target_ms,
                const std::string& csv, const std::vector<PackedLayout>& layouts, bool calls) {
  const CpuInfo& cpu = cpu_info();
  const int64_t target_ws = std::max<int64_t>(4 * cpu.l3_bytes, int64_t{64} << 20);
  std::printf("layouts: LLC %.1f MB, dram working set >= %lld MB per layout; samples %d x ~%.0f ms; pin %s\n",
              cpu.l3_bytes / 1048576.0, static_cast<long long>(target_ws >> 20), samples, target_ms,
              std::getenv("DYNACORE_PIN_THREADS") ? std::getenv("DYNACORE_PIN_THREADS") : "unset");
  FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (!csv.empty() && out == nullptr) die("cannot write " + csv);
  if (out) std::fprintf(out, "threads,m,n,k,impl,layout,mode,copies,bytes_per_call,p10_us,p50_us,p90_us,gbs,probe_gbs\n");
  const CpuKernels kern = make_cpu_kernels(CpuIsa::kAvx2);
  auto jit = LlvmJit::create(llvm_lowering_target(LlvmLowering::kAvx2));
  if (!jit.ok()) die(jit.status().to_string());
  std::map<std::string, GeneratedQ4KInt8Fn> compiled;
  std::vector<uint64_t> probe_buf(static_cast<size_t>((int64_t{256} << 20) / 8));
  for (size_t i = 0; i < probe_buf.size(); ++i) probe_buf[i] = i * 2654435761u;
  const int64_t shapes[][2] = {{1536, 1536}, {8960, 1536}, {1536, 8960}};
  uint32_t seed = 1700;
  for (const auto& sh : shapes) {
    const int64_t n = sh[0], k = sh[1], sb = k / 256, groups = n / 8, rb = dtype_row_bytes(DType::kQ4_K, k);
    const Weight w0 = q4kref::make_weight(n, k, seed++);
    // Packed copies per layout, and the packing time of one full matrix.
    struct LayoutData {
      PackedLayout l;
      int64_t bytes;  // per call (whole matrix)
      std::vector<std::vector<uint8_t>> copies;
      double pack_ms;
    };
    std::vector<LayoutData> ld;
    std::printf("\n-- N %lld K %lld: GGUF %.2f MB\n", static_cast<long long>(n), static_cast<long long>(k), n * rb / 1048576.0);
    for (PackedLayout l : layouts) {
      LayoutData d{l, groups * sb * packed_layout_info(l).block_bytes, {}, 0};
      const int copies = static_cast<int>(std::max<int64_t>(2, (target_ws + d.bytes - 1) / d.bytes));
      std::vector<double> t_ms;
      std::vector<uint8_t> buf(static_cast<size_t>(d.bytes));
      for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t g = 0; g < groups; ++g) {
          repack_q4_K_x8_layout(l, w0.blocks.data(), rb, 8 * g, k, buf.data() + g * sb * packed_layout_info(l).block_bytes);
        }
        t_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      d.pack_ms = pct(t_ms, 0.5);
      d.copies.assign(static_cast<size_t>(copies), buf);
      std::printf("   layout %-2s: %7.3f MB per call (%+.1f%% vs x8), %2d copies (%.0f MB), pack %.2f ms single-thread (%.0f MB/s of output)\n",
                  std::string(packed_layout_name(l)).c_str(), d.bytes / 1048576.0,
                  100.0 * (static_cast<double>(d.bytes) / (groups * sb * 1216.0) - 1), copies, copies * d.bytes / 1048576.0,
                  d.pack_ms, d.bytes / 1048.576 / d.pack_ms);
      if (out) {
        std::fprintf(out, "0,0,%lld,%lld,\"pack\",%s,setup,1,%lld,%.4f,%.4f,%.4f,0,0\n", static_cast<long long>(n),
                     static_cast<long long>(k), std::string(packed_layout_name(l)).c_str(), static_cast<long long>(d.bytes),
                     pct(t_ms, 0.1) * 1e3, d.pack_ms * 1e3, pct(t_ms, 0.9) * 1e3);
      }
      ld.push_back(std::move(d));
    }
    const int gguf_copies = static_cast<int>(std::max<int64_t>(2, (target_ws + n * rb - 1) / (n * rb)));
    std::vector<std::vector<quant::BlockQ4_K>> src;
    if (calls) src.assign(static_cast<size_t>(gguf_copies), w0.blocks);
    for (int t : threads_list) {
      ThreadPool pool(t);
      pool.pin_caller();
      double probe_best = 0;
      const double probe = read_probe_gbs(pool, probe_buf, &probe_best);
      std::printf("== %d thread(s), N %lld K %lld: read probe %.1f GB/s median\n", t, static_cast<long long>(n),
                  static_cast<long long>(k), probe);
      std::vector<std::unique_ptr<LlvmInt8Backend>> backends;
      if (calls) {
        for (const LayoutData& d : ld) {
          backends.push_back(std::make_unique<LlvmInt8Backend>(pool, LlvmLowering::kAvx2, LlvmKernelOptions{0, d.l}));
          for (auto& s : src) {
            const TensorView v(s.data(), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{n, k}));
            if (!backends.back()->prepack_weight(v).ok()) die("prepack failed");
          }
        }
      }
      std::printf("%2s %-10s %5s %9s %9s %9s %8s %6s %7s\n", "M", "impl", "mode", "p10 us", "p50 us", "p90 us", "GB/s", "%probe",
                  "vs x8");
      for (int m : ms) {
        const std::vector<float> x = q4kref::make_x(m, k, k, seed++);
        const int64_t nb = k / kActQ8Block;
        std::vector<ActBlockQ8> act(static_cast<size_t>(m * nb));
        for (int64_t r = 0; r < m; ++r) kern.quantize_act_sx(x.data() + r * k, act.data() + r * nb, k);
        const auto* xa = reinterpret_cast<const ActSuperQ8*>(act.data());
        const ActBlockQ8* xr[kPackedRowsMax];
        for (int r = 0; r < m; ++r) xr[r] = act.data() + r * nb;
        const Q4KMatmulSpec spec{m, n, k, false};
        ProgramBuilder pb;
        const ValueId vx = pb.activation("x", DType::kF32, m, k);
        const ValueId vw = pb.weight("w", DType::kQ4_K, n, k);
        const ValueId vy = pb.matmul(vx, vw);
        const Program prog = *pb.finish();
        struct Imp {
          std::string name, layout;
          int64_t bytes;
          int copies;
          std::function<void(int copy, float* y)> fn;
          std::vector<float> y;
          bool verify;
        };
        std::vector<Imp> imps;
        const size_t grain = std::max<size_t>(1, static_cast<size_t>(groups) / (static_cast<size_t>(t) * 32));
        for (const LayoutData& d : ld) {
          const LlvmKernelOptions ko{0, d.l};
          const std::string sym = llvm_kernel_symbol(spec, LlvmLowering::kAvx2, ko);
          if (!compiled.count(sym)) {
            LlvmCompileStats st;
            auto c = (*jit)->compile_kernel(spec, LlvmLowering::kAvx2, &st, ko);
            if (!c.ok()) die(c.status().to_string());
            compiled[sym] = reinterpret_cast<GeneratedQ4KInt8Fn>(c->fn);
            std::printf("   compiled %s: %.1f ms (setup, not timed)\n", sym.c_str(), st.total_ms);
          }
          const GeneratedQ4KInt8Fn f = compiled[sym];
          const LayoutData* dp = &d;
          imps.push_back({"k_" + std::string(packed_layout_name(d.l)), std::string(packed_layout_name(d.l)), d.bytes,
                          static_cast<int>(d.copies.size()),
                          [&pool, f, xa, n, groups, grain, dp](int c, float* y) {
                            const auto* wp = reinterpret_cast<const BlockQ4_Kx8*>(dp->copies[static_cast<size_t>(c)].data());
                            pool.parallel_for(static_cast<size_t>(groups), grain, [&](size_t g0, size_t g1) {
                              f(wp, xa, nullptr, y, n, static_cast<int64_t>(g0), static_cast<int64_t>(g1));
                            });
                          },
                          {}, true});
        }
        for (const LayoutData& d : ld) {
          const LayoutData* dp = &d;
          imps.push_back({"trav_" + std::string(packed_layout_name(d.l)), std::string(packed_layout_name(d.l)), d.bytes,
                          static_cast<int>(d.copies.size()),
                          [&pool, dp, groups, grain](int c, float*) {
                            const uint8_t* base = dp->copies[static_cast<size_t>(c)].data();
                            const size_t per_group = static_cast<size_t>(dp->bytes / groups);
                            pool.parallel_for(static_cast<size_t>(groups), grain, [&](size_t g0, size_t g1) {
                              const auto* p = reinterpret_cast<const uint64_t*>(base + g0 * per_group);
                              const size_t words = (g1 - g0) * per_group / 8;
                              uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                              for (size_t i = 0; i + 4 <= words; i += 4) s0 += p[i], s1 += p[i + 1], s2 += p[i + 2], s3 += p[i + 3];
                              g_sink.fetch_add(s0 + s1 + s2 + s3, std::memory_order_relaxed);
                            });
                          },
                          {}, false});
        }
        {
          const LayoutData* x8 = nullptr;
          for (const LayoutData& d : ld) x8 = d.l == PackedLayout::kX8 ? &d : x8;
          if (x8 != nullptr) {
            imps.push_back({"dc", "x8", x8->bytes, static_cast<int>(x8->copies.size()),
                            [&, x8](int c, float* y) {
                              const auto* wp = reinterpret_cast<const BlockQ4_Kx8*>(x8->copies[static_cast<size_t>(c)].data());
                              pool.parallel_for(static_cast<size_t>(groups), grain, [&](size_t g0, size_t g1) {
                                float o[8 * kPackedRowsMax];
                                for (size_t g = g0; g < g1; ++g) {
                                  if (m > kDotRowsMax && kern.dot_q8_sx_x8_q4_K_rows8 != nullptr) {
                                    kern.dot_q8_sx_x8_q4_K_rows8(wp + g * sb, xr, m, k, o);
                                    for (int r = 0; r < m; ++r) std::memcpy(y + r * n + 8 * g, o + 8 * r, 32);
                                  } else {
                                    for (int r0 = 0; r0 < m; r0 += kDotRowsMax) {
                                      const int mm = std::min(kDotRowsMax, m - r0);
                                      kern.dot_q8_sx_x8_q4_K(wp + g * sb, xr + r0, mm, k, o);
                                      for (int r = 0; r < mm; ++r) std::memcpy(y + (r0 + r) * n + 8 * g, o + 8 * r, 32);
                                    }
                                  }
                                }
                              });
                            },
                            {}, false});
          }
        }
        RunOptions ro;
        ro.check_packed = false;
        for (size_t bi = 0; bi < backends.size(); ++bi) {
          LlvmInt8Backend* be = backends[bi].get();
          if (!be->prepare(prog).ok()) die("prepare failed");
          imps.push_back({"call_" + std::string(packed_layout_name(ld[bi].l)), std::string(packed_layout_name(ld[bi].l)),
                          ld[bi].bytes, gguf_copies,
                          [&, be, vx, vw, vy](int c, float* y) {
                            Bindings b;
                            auto lay = [](DType dt, int64_t r, int64_t cols) { return *TensorLayout::contiguous(dt, TensorShape{r, cols}); };
                            b.bind(vx, TensorView(const_cast<float*>(x.data()), lay(DType::kF32, m, k)));
                            b.bind(vw, TensorView(src[static_cast<size_t>(c)].data(), lay(DType::kQ4_K, n, k)));
                            b.bind(vy, TensorView(y, lay(DType::kF32, m, n)));
                            if (const Status s = be->run(prog, b, ro); !s.ok()) die(s.to_string());
                          },
                          {}, true});
        }
        for (Imp& im : imps) im.y.assign(static_cast<size_t>(m * n), std::nanf(""));
        std::vector<Component> comps;
        std::vector<int> cidx(imps.size(), 0);
        for (size_t i = 0; i < imps.size(); ++i) {
          for (int dram = 1; dram >= 0; --dram) {
            comps.push_back({imps[i].name + (dram ? "|dram" : "|hot"), [&, i, dram] {
                               const int c = dram ? cidx[i] : 0;
                               if (dram) cidx[i] = (cidx[i] + 1) % imps[i].copies;
                               imps[i].fn(c, imps[i].y.data());
                             }, {}});
          }
        }
        auto verify = [&](const char* when) {
          const std::vector<float>* ref = nullptr;
          for (size_t i = 0; i < imps.size(); ++i) {
            if (imps[i].name.rfind("trav_", 0) == 0) continue;
            std::vector<float> first;
            for (int c = 0; c < imps[i].copies; ++c) {
              comps[2 * i].fn();
              if (c == 0) first = imps[i].y;
              else if (imps[i].y != first) die(imps[i].name + ": copies differ " + when);
            }
            comps[2 * i + 1].fn();
            if (imps[i].y != first) die(imps[i].name + ": hot and dram differ " + when);
            if (imps[i].name == "k_x8") {
              ref = &imps[i].y;
              q4kref::Stats st;
              q4kref::check_bound(w0, x, k, nullptr, imps[i].y.data(), n, m, Path::kInt8Subscale, st);
            } else if (imps[i].name == "dc") {
              q4kref::Stats st;
              q4kref::check_bound(w0, x, k, nullptr, imps[i].y.data(), n, m, Path::kInt8Subscale, st);
            }
          }
          for (size_t i = 0; i < imps.size(); ++i) {
            if (imps[i].verify && ref != nullptr && imps[i].y != *ref) die(imps[i].name + ": differs from k_x8 " + when);
          }
        };
        verify("before timing");
        time_components(comps, samples, target_ms);
        verify("after timing");
        if (pool.misuse_count() != 0) die("thread pool misuse");
        std::map<std::string, double> x8_p50;  // "<kind>|<mode>" -> p50 of the x8 variant
        for (size_t i = 0; i < comps.size(); ++i) {
          const std::string& cn = comps[i].name;
          const std::string name = cn.substr(0, cn.find('|')), mode = cn.substr(cn.find('|') + 1);
          if (name.size() > 3 && name.substr(name.size() - 3) == "_x8") x8_p50[name.substr(0, name.size() - 2) + mode] = pct(comps[i].us, 0.5);
        }
        for (size_t i = 0; i < comps.size(); ++i) {
          const Imp& im = imps[i / 2];
          const double p10 = pct(comps[i].us, 0.1), p50 = pct(comps[i].us, 0.5), p90 = pct(comps[i].us, 0.9);
          const double gbs = im.bytes / (p50 * 1e3);
          const std::string& cn = comps[i].name;
          const std::string mode = cn.substr(cn.find('|') + 1);
          const std::string kind = im.name.substr(0, im.name.find('_') + 1);
          const auto it = x8_p50.find(kind + mode);
          const double rel = it != x8_p50.end() && im.name != "dc" ? p50 / it->second : 0;
          std::printf("%2d %-10s %5s %9.1f %9.1f %9.1f %8.2f %5.0f%% %7.3f\n", m, im.name.c_str(), mode.c_str(), p10, p50, p90,
                      gbs, 100 * gbs / probe, rel);
          if (out) {
            std::fprintf(out, "%d,%d,%lld,%lld,\"%s\",%s,%s,%d,%lld,%.2f,%.2f,%.2f,%.3f,%.3f\n", t, m, static_cast<long long>(n),
                         static_cast<long long>(k), im.name.c_str(), im.layout.c_str(), mode.c_str(),
                         mode == "dram" ? im.copies : 1, static_cast<long long>(im.bytes), p10, p50, p90, gbs, probe);
          }
        }
      }
    }
  }
  if (out) std::fclose(out);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> threads_list = {1, 10}, ms = {1, 2, 4, 5, 8};
  int samples = 15;
  double target_ms = 20;
  std::string csv;
  bool breakdown = false, dram = false, bandwidth = false, want_vnni = false, layouts_mode = false, calls = false;
  std::vector<PackedLayout> layouts = {PackedLayout::kX8, PackedLayout::kX8Fp16, PackedLayout::kX8Scales6};
  std::vector<int> cpt = {32}, pf = {0};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads_list = parse_list(argv[++i]);
    else if (a == "--m" && i + 1 < argc) ms = parse_list(argv[++i]);
    else if (a == "--samples" && i + 1 < argc) samples = std::max(3, std::atoi(argv[++i]));
    else if (a == "--target-ms" && i + 1 < argc) target_ms = std::atof(argv[++i]);
    else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--breakdown") breakdown = true;
    else if (a == "--dram") dram = true;
    else if (a == "--bandwidth") bandwidth = true;
    else if (a == "--vnni") want_vnni = true;
    else if (a == "--calls") calls = true;
    else if (a == "--layouts" && i + 1 < argc) {
      layouts_mode = true;
      layouts.clear();
      for (std::string rest = argv[++i]; !rest.empty();) {
        const std::string name = rest.substr(0, rest.find(','));
        rest = rest.find(',') == std::string::npos ? "" : rest.substr(rest.find(',') + 1);
        bool found = false;
        for (PackedLayout l : {PackedLayout::kX8, PackedLayout::kX8Fp16, PackedLayout::kX8Scales6, PackedLayout::kX8Fp16Scales6}) {
          if (packed_layout_name(l) == name) layouts.push_back(l), found = true;
        }
        if (!found) die("unknown layout " + name + " (x8, h, s, hs)");
      }
    }
    else if (a == "--chunks" && i + 1 < argc) cpt = parse_list(argv[++i]);
    else if (a == "--prefetch" && i + 1 < argc) pf = parse_list(argv[++i]);
    else die("usage: bench_dynakernel_llvm [--threads 1,10] [--m 1,2,4,5,8] [--samples 15] [--target-ms 20] [--csv FILE] "
             "[--breakdown [--dram]] [--bandwidth [--chunks 32,8] [--prefetch 0,2432] [--vnni]] [--layouts x8,h,s,hs [--calls]]");
  }
  const CpuIsa tier = select_isa(cpu_info().features);
  if (tier != CpuIsa::kAvx2 || !DK_GENERATED_HAS_AVX2) die("needs the avx2 tier");
  if (const Status s = llvm_available(); !s.ok()) die(s.message());
  set_env("DYNACORE_Q4_REPACK", "always");  // reference packing independent of free RAM
  if (breakdown) return run_breakdown(threads_list, ms, samples, target_ms, csv, dram);
  if (bandwidth) return run_bandwidth(threads_list, ms, samples, target_ms, csv, cpt, pf, want_vnni);
  if (layouts_mode) return run_layouts(threads_list, ms, samples, target_ms, csv, layouts, calls);
  if (dram) die("--dram needs --breakdown");
  const LlvmRuntimeInfo& li = llvm_runtime();
  std::vector<LlvmLowering> lowerings;
  for (LlvmLowering l : {LlvmLowering::kPortable, LlvmLowering::kAvx2, LlvmLowering::kAvxVnni}) {
    if (llvm_lowering_runnable(l).ok()) lowerings.push_back(l);
  }
  const bool vnni = cpu_info().features.avx_vnni;
  size_t ngen = 0;
#if DK_GENERATED_HAS_AVX2
  const GeneratedInt8Kernel* gen_k = dk_generated_q4k_bench_int8_avx2_kernels(&ngen);
#else
  const GeneratedInt8Kernel* gen_k = nullptr;
#endif
  std::printf("bench_dynakernel_llvm: LLVM %u.%u.%u, cpu %s, %s; samples %d x ~%.0f ms; pin %s\n", li.major, li.minor, li.patch,
              li.host_cpu.c_str(), cpu_info().brand.c_str(), samples, target_ms,
              std::getenv("DYNACORE_PIN_THREADS") ? std::getenv("DYNACORE_PIN_THREADS") : "unset");
  FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
  if (!csv.empty() && out == nullptr) die("cannot write " + csv);
  if (out) {
    std::fprintf(out, "threads,m,n,k,impl,numerics,p10_us,p50_us,p90_us,spread_pct,gflops,weight_gbs,err_over_bound,"
                      "same_as_gen,compile_ms\n");
  }
  const int64_t shapes[][2] = {{1536, 1536}, {8960, 1536}, {1536, 8960}};
  uint32_t seed = 1;
  struct Compile {
    std::string lowering;
    Q4KMatmulSpec spec;
    LlvmCompileStats st;
  };
  std::vector<Compile> compiles;
  for (const auto& sh : shapes) {
    const int64_t n = sh[0], k = sh[1];
    const Weight w = q4kref::make_weight(n, k, seed++);
    const int64_t wbytes = dtype_row_bytes(DType::kQ4_K, k) * n;
    for (int t : threads_list) {
      ThreadPool pool(t);
      pool.pin_caller();  // DYNACORE_PIN_THREADS=1: the calling thread (all work at 1 thread) on the first core
      ReferenceBackend ref(pool, tier), ref_pk(pool, tier);
      std::unique_ptr<ReferenceBackend> ref_vnni;
      if (vnni) {
        set_env("DYNACORE_VNNI", "1");  // read when the device builds its kernel table
        ref_vnni = std::make_unique<ReferenceBackend>(pool, tier);
        set_env("DYNACORE_VNNI", "");
      }
      GeneratedInt8Backend gen(pool, {gen_k, ngen});
      std::vector<std::unique_ptr<LlvmInt8Backend>> llvm;
      const TensorView wv(const_cast<quant::BlockQ4_K*>(w.blocks.data()), *TensorLayout::contiguous(DType::kQ4_K, TensorShape{n, k}));
      if (!ref_pk.prepack_weight(wv).ok() || !gen.prepack_weight(wv).ok()) die("prepack failed");
      if (ref_vnni && !ref_vnni->prepack_weight(wv).ok()) die("prepack (vnni) failed");
      for (LlvmLowering l : lowerings) {
        llvm.push_back(std::make_unique<LlvmInt8Backend>(pool, l));
        if (!llvm.back()->prepack_weight(wv).ok()) die("prepack (llvm) failed");
      }
      std::printf("\n== N %lld K %lld, %d thread(s)\n", static_cast<long long>(n), static_cast<long long>(k), t);
      std::printf("%2s  %-12s %-28s %9s %9s %9s %6s %7s %7s %6s %5s %8s\n", "M", "impl", "numerics", "p10 us", "p50 us",
                  "p90 us", "sprd%", "GFLOP/s", "GB/s", "e/bnd", "=gen", "vs pk");
      for (int m : ms) {
        const std::vector<float> x = q4kref::make_x(m, k, k, seed++);
        std::vector<Impl> impls;
        const Path ref_path = m == 1 ? Path::kInt8Per32 : m <= 8 ? Path::kInt8Subscale : Path::kFp32;
        const char* ref_num = m == 1 ? "int8 per-32" : m <= 8 ? "int8 sub-scaled rows" : "fp32 act";
        impls.push_back({"ref", ref_num, ref_path, &ref});
        impls.push_back({"ref_pk", m == 1 ? "int8 per-32" : m <= 4 ? "int8 sx packed x8" : "int8 sx packed 8-row (exp)",
                         m == 1 ? Path::kInt8Per32 : Path::kInt8Subscale, &ref_pk, m > 4});
        if (ref_vnni && m >= 2 && m <= 4) impls.push_back({"ref_pk_vnni", "int8 sx packed x8 vnni", Path::kInt8Subscale, ref_vnni.get()});
        const Q4KMatmulSpec spec{m, n, k, false};
        bool has_gen = false;
        for (size_t i = 0; i < ngen; ++i) has_gen = has_gen || gen_k[i].spec == spec;
        if (has_gen) {
          Impl g{"gen_int8", "int8 sx packed (C++ gen)", Path::kInt8Subscale};
          g.gen = &gen;
          impls.push_back(std::move(g));
        }
        for (size_t li2 = 0; li2 < lowerings.size(); ++li2) {
          Impl im{"llvm_" + std::string(llvm_lowering_name(lowerings[li2])), "int8 sx packed (LLVM)", Path::kInt8Subscale};
          im.llvm = llvm[li2].get();
          impls.push_back(std::move(im));
        }
        for (Impl& im : impls) im.c = make_case(w, x, m);
        // Compile (timed separately), then verify every output before timing.
        std::vector<double> compile_ms(impls.size(), 0);
        for (size_t i = 0; i < impls.size(); ++i) {
          if (impls[i].llvm == nullptr) continue;
          if (const Status s = impls[i].llvm->prepare(impls[i].c->p); !s.ok()) die(impls[i].name + ": " + s.to_string());
          const auto& ck = impls[i].llvm->compiled().back();
          compile_ms[i] = ck.stats.total_ms;
          if (t == threads_list.front()) compiles.push_back({std::string(llvm_lowering_name(impls[i].llvm->lowering())), spec, ck.stats});
        }
        const std::vector<float>* gen_y = nullptr;
        std::vector<double> ratio(impls.size(), 0);
        for (size_t i = 0; i < impls.size(); ++i) {
          Impl& im = impls[i];
          if (const Status s = run_impl(im); !s.ok()) die(im.name + ": " + s.to_string());
          q4kref::Stats st;
          q4kref::check_bound(w, x, k, nullptr, im.c->y.data(), n, m, im.path, st);
          ratio[i] = st.max_ratio;
          im.verified = im.c->y;
          if (im.gen != nullptr) gen_y = &im.verified;
        }
        std::vector<int> same(impls.size(), -1);
        for (size_t i = 0; i < impls.size(); ++i) {
          if (gen_y != nullptr && impls[i].llvm != nullptr) {
            same[i] = impls[i].verified == *gen_y;
            if (!same[i]) die(impls[i].name + ": output differs from gen_int8");
          }
        }
        // Warm-up and calibration: calls per sample so a sample takes ~target_ms.
        std::vector<int> batch(impls.size(), 1);
        for (size_t i = 0; i < impls.size(); ++i) {
          for (int r = 0; r < 3; ++r) (void)run_impl(impls[i]);
          const auto t0 = std::chrono::steady_clock::now();
          (void)run_impl(impls[i]);
          const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          batch[i] = std::max(1, static_cast<int>(target_ms / std::max(one, 1e-3)));
        }
        for (int s = 0; s < samples; ++s) {
          for (size_t i = 0; i < impls.size(); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < batch[i]; ++r) (void)run_impl(impls[i]);
            impls[i].us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / batch[i]);
          }
        }
        double pk50 = 0;
        for (Impl& im : impls) {
          if (!run_impl(im).ok() || im.c->y != im.verified) die(im.name + ": output changed after timing");
          if (im.name == "ref_pk") pk50 = pct(im.us, 0.5);
        }
        for (size_t i = 0; i < impls.size(); ++i) {
          const Impl& im = impls[i];
          const double p10 = pct(im.us, 0.1), p50 = pct(im.us, 0.5), p90 = pct(im.us, 0.9);
          const double gflops = 2.0 * m * n * k / (p50 * 1e3), gbs = wbytes / (p50 * 1e3);
          std::printf("%2d  %-12s %-28s %9.1f %9.1f %9.1f %6.1f %7.1f %7.2f %6.3f %5s %7.2fx\n", m, im.name.c_str(),
                      im.numerics.c_str(), p10, p50, p90, 100 * (p90 - p10) / p50, gflops, gbs, ratio[i],
                      same[i] < 0 ? "-" : same[i] ? "yes" : "no", pk50 / p50);
          if (out) {
            std::fprintf(out, "%d,%d,%lld,%lld,\"%s\",\"%s\",%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.4f,%d,%.2f\n", t, m,
                         static_cast<long long>(n), static_cast<long long>(k), im.name.c_str(), im.numerics.c_str(), p10,
                         p50, p90, 100 * (p90 - p10) / p50, gflops, gbs, ratio[i], same[i], compile_ms[i]);
          }
        }
      }
      for (auto& b : llvm) b->release_packed(w.blocks.data());
    }
  }
  std::printf("\nLLVM compilation (once per kernel, before timing; ms):\n%-9s %2s %5s %5s %8s %8s %8s %8s %8s %8s\n", "lowering",
              "M", "N", "K", "IR KB", "parse", "verify", "O3", "codegen", "total");
  for (const Compile& c : compiles) {
    std::printf("%-9s %2d %5lld %5lld %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n", c.lowering.c_str(), c.spec.m,
                static_cast<long long>(c.spec.n), static_cast<long long>(c.spec.k), c.st.ir_bytes / 1024.0, c.st.parse_ms,
                c.st.verify_ms, c.st.optimize_ms, c.st.codegen_ms, c.st.total_ms);
  }
  if (out) std::fclose(out);
  return 0;
}

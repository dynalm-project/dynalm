// CPU matmul kernel registry (DynaLM Next, matmul_registry.h). No model files.
//
// Checks:
//  1. select_matmul == a verbatim copy of the CpuDevice::matmul if-chain it
//     replaced, over a grid of shapes, weight types, plans and tier
//     capabilities (default behaviour unchanged by construction);
//  2. every kernel the device can run, through CpuDevice::matmul: the kernel
//     plan_matmul names is the one the counters record, and its output is
//     within its accuracy class of an fp64 reference (fp32 / int16 / int8);
//     results identical for 1 and 4 threads;
//  3. split-byte int16 tiles bit-identical to the int16 tiles (M 9..128,
//     strided x and y, bias, threads);
//  4. diagnostics: counters, timing, shape table, explain text, reasons;
//  5. validation and kernel-list parsing; disabled kernels fall through;
//  6. external kernels: registration errors, opt-in, eligibility, fallback;
//  7. two devices on two threads at once give the serial results.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "dynacore/base/env.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/kernel/matmul_registry.h"
#include "dynacore/quantization/dequant.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"
#include "q4k_reference.h"

using namespace dynacore;
using Id = MatmulKernelId;

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

// ---------------------------------------------------------------------------
// 1. The if-chain of CpuDevice::matmul / int8_kernel before the registry,
//    copied with kernel-table pointers replaced by caps booleans. Returns the
//    path that would have run.
constexpr Id kNone = Id::kCount;

Id legacy_int8(const MatmulShape& s, const KernelPlan& p, const MatmulCaps& c) {
  const int64_t m = s.m, k = s.k, n = s.n;
  const DType wt = s.weight;
  if (p.int8_decode_max_rows <= 0 || !c.int8_accelerated || k % kActQ8Block != 0) return kNone;
  const bool sb_rows = k % 256 == 0 && m <= std::max(p.int8_decode_max_rows, p.int8_superblock_max_rows);
  if (!sb_rows && wt == DType::kQ4_K && k % 256 == 0 && m <= p.q4_repack_max_rows && n >= p.q4_repack_min_out &&
      p.q4_repack_min_rows > 0 && p.int8_subscale_min_rows > 0 && m >= p.int8_subscale_min_rows && c.quantize_act_sx) {
    if (c.packed_q4) return c.dot_q8_sx_rows ? Id::kInt8PrefillX8 : kNone;  // returned even with a null dot
  }
  if (sb_rows && p.int8_subscale_min_rows > 0 && m >= p.int8_subscale_min_rows && c.quantize_act_sx) {
    if (c.dot_q8_sx_rows) {
      const bool packed = wt == DType::kQ4_K && p.q4_repack_min_rows > 0 && m >= p.q4_repack_min_rows && c.packed_q4;
      if (!packed) return Id::kInt8Subscaled;
      return (p.packed_int8_8rows && m > kDotRowsMax && c.dot_q8_sx_x8_rows8) ? Id::kInt8SubscaledX8Rows8 : Id::kInt8SubscaledX8;
    }
  }
  if (sb_rows && p.int8_superblock_min_rows > 0 && m >= p.int8_superblock_min_rows && c.quantize_act_sb) {
    if (c.dot_q8_sb_rows) return Id::kInt8Superblock;
  }
  if (m > p.int8_decode_max_rows) return kNone;
  return c.dot_q8_rows ? Id::kInt8Per32 : kNone;
}

Id legacy_select(const MatmulShape& s, const KernelPlan& p, const MatmulCaps& c) {
  const int64_t m = s.m, k = s.k, n = s.n;
  const DType wt = s.weight;
  if (const Id i8 = legacy_int8(s, p, c); i8 != kNone) return i8;
  if (p.int16_prefill_max_rows > 0 && m > kPackedRowsMax && m <= p.int16_prefill_max_rows && n >= p.q4_repack_min_out &&
      n * k >= p.int16_prefill_min_weights && c.int8_accelerated && k % 256 == 0 && c.quantize_act16 && c.q16_packed_enabled) {
    bool pk = false;
    if (wt == DType::kQ4_K && c.dot_q16_x8_q4_rows8) {
      pk = c.packed_q4;
    } else if (wt == DType::kQ6_K && c.dot_q16_x8_q6_rows8 && c.q6_packed_enabled) {
      pk = c.packed_q6;
    }
    if (pk) return Id::kInt16PrefillX8;
  }
  const bool try16 = p.int16_decode_max_rows > 0 && p.q4_repack_min_rows > 0 && m >= p.q4_repack_min_rows && c.q16_packed_enabled;
  bool packed16 = false;
  if (try16 && wt == DType::kQ4_K && c.dot_q16_x8_q4) {
    packed16 = c.packed_q4;
  } else if (try16 && wt == DType::kQ6_K && c.dot_q16_x8_q6 && c.q6_packed_enabled) {
    packed16 = c.packed_q6;
  }
  if (packed16 && m <= std::max(p.int16_decode_max_rows, p.int8_superblock_max_rows) && c.int8_accelerated && k % 256 == 0 &&
      c.quantize_act16) {
    return Id::kInt16X8;
  }
  if (m <= p.int16_decode_max_rows && c.int8_accelerated && k % kActQ8Block == 0 && c.quantize_act16) {
    if (c.dot_q16_rows) return Id::kInt16Rows;
  }
  const bool expand = wt != DType::kF32 && m >= p.expand_min_rows;
  const bool f32_panel = wt == DType::kF32 && m >= p.expand_min_rows && s.x_dense;
  if ((expand || f32_panel) && s.x_dense) return Id::kF32ExpandGemm;
  return Id::kF32RowDot;
}

MatmulCaps caps_avx2(DType wt, bool packed, bool vnni_split) {
  MatmulCaps c;
  const bool kq = wt == DType::kQ4_K || wt == DType::kQ5_K || wt == DType::kQ6_K || wt == DType::kQ2_K || wt == DType::kQ3_K;
  const bool int8_type = kq || wt == DType::kQ4_0 || wt == DType::kQ8_0 || wt == DType::kQ5_0;
  c.int8_accelerated = true;
  c.packed_q4 = packed && wt == DType::kQ4_K;
  c.packed_q6 = packed && wt == DType::kQ6_K;
  c.quantize_act = c.quantize_act_sb = c.quantize_act_sx = c.quantize_act16 = true;
  c.dot_q8_rows = int8_type;
  c.dot_q8_sb_rows = kq;
  c.dot_q8_sx_rows = kq;
  c.dot_q8_sx_x8_rows8 = true;
  c.dot_q16_x8_q4 = c.dot_q16_x8_q6 = c.dot_q16_x8_q4_rows8 = c.dot_q16_x8_q6_rows8 = true;
  c.dot_q16s_x8_q4_rows8 = vnni_split;
  c.split_act16 = true;
  c.dot_q16_rows = kq || wt == DType::kQ8_0;
  return c;
}

void test_legacy_equivalence() {
  const int64_t ms[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 31, 32, 33, 48, 64, 65, 128, 129};
  const int64_t ns[] = {256, 1016, 1024, 1536, 8960};
  const int64_t ks[] = {64, 96, 256, 1536, 8960};
  const DType wts[] = {DType::kF32, DType::kF16, DType::kQ4_0, DType::kQ8_0, DType::kQ4_K, DType::kQ5_K, DType::kQ6_K};
  std::vector<KernelPlan> plans;
  plans.push_back(KernelPlan::defaults());
  auto add = [&](auto&& f) {
    KernelPlan p = KernelPlan::defaults();
    f(p);
    plans.push_back(p);
  };
  add([](KernelPlan& p) { p.int8_decode_max_rows = 0; });
  add([](KernelPlan& p) { p.int16_prefill_max_rows = 64; });
  add([](KernelPlan& p) { p.int16_prefill_max_rows = 128; p.int16_prefill_min_weights = 0; });
  add([](KernelPlan& p) { p.q4_repack_max_rows = 64; });
  add([](KernelPlan& p) { p.int8_superblock_min_rows = 4; });
  add([](KernelPlan& p) { p.int8_subscale_min_rows = 0; });
  add([](KernelPlan& p) { p.packed_int8_8rows = true; });
  add([](KernelPlan& p) { p.int16_decode_max_rows = 8; });
  add([](KernelPlan& p) { p.int16_decode_max_rows = 4; p.int8_decode_max_rows = 0; });
  add([](KernelPlan& p) { p.expand_min_rows = 16; });
  add([](KernelPlan& p) { p.q4_repack_min_rows = 0; p.int16_prefill_max_rows = 64; });
  add([](KernelPlan& p) { p.int8_decode_max_rows = 8; p.q4_repack_max_rows = 32; p.packed_int8_8rows = true; });
  int64_t compared = 0;
  for (DType wt : wts) {
    std::vector<MatmulCaps> cv;
    cv.push_back(caps_avx2(wt, true, true));
    cv.push_back(caps_avx2(wt, false, true));
    cv.push_back(caps_avx2(wt, true, false));
    MatmulCaps generic;  // generic tier: no integer kernels
    generic.quantize_act = generic.quantize_act16 = true;
    generic.dot_q8_rows = true;
    cv.push_back(generic);
    MatmulCaps q16off = caps_avx2(wt, true, true);
    q16off.q16_packed_enabled = false;
    cv.push_back(q16off);
    MatmulCaps q6on = caps_avx2(wt, true, true);
    q6on.q6_packed_enabled = true;
    cv.push_back(q6on);
    MatmulCaps no_sx = caps_avx2(wt, true, true);
    no_sx.dot_q8_sx_rows = false;
    cv.push_back(no_sx);
    for (const KernelPlan& p : plans) {
      for (const MatmulCaps& c : cv) {
        for (int64_t m : ms) {
          for (int64_t n : ns) {
            for (int64_t k : ks) {
              if (dtype_block_elems(wt) > 1 && k % dtype_block_elems(wt) != 0) continue;
              for (bool dense : {true, false}) {
                const MatmulShape s{m, n, k, wt, dense};
                const Id want = legacy_select(s, p, c);
                const MatmulSelection got = select_matmul(s, p, c);
                // The split kernel is off in every plan here, so the legacy
                // chain must agree exactly.
                if (got.id != want) {
                  std::fprintf(stderr, "mismatch: %s m %lld n %lld k %lld dense %d: legacy %s registry %s\n",
                               std::string(dtype_name(wt)).c_str(), static_cast<long long>(m), static_cast<long long>(n),
                               static_cast<long long>(k), dense ? 1 : 0, std::string(matmul_kernel_name(want)).c_str(),
                               select_matmul(s, p, c, true).explain().c_str());
                }
                CHECK(got.id == want);
                // explain mode never changes the choice
                CHECK(select_matmul(s, p, c, true).id == want);
                ++compared;
              }
            }
          }
        }
      }
    }
  }
  std::printf("  legacy equivalence: %lld selections identical to the old if-chain\n", static_cast<long long>(compared));

  // With the split switch on, it replaces exactly the Q4_K int16 prefill tiles.
  KernelPlan sp = KernelPlan::defaults();
  sp.int16_prefill_max_rows = 64;
  sp.int16_prefill_split = true;
  for (int64_t m : ms) {
    for (DType wt : {DType::kQ4_K, DType::kQ6_K}) {
      MatmulCaps c = caps_avx2(wt, true, true);
      c.q6_packed_enabled = true;
      const MatmulShape s{m, 8960, 1536, wt, true};
      KernelPlan off = sp;
      off.int16_prefill_split = false;
      const Id base = select_matmul(s, off, c).id;
      const Id got = select_matmul(s, sp, c).id;
      CHECK(got == ((base == Id::kInt16PrefillX8 && wt == DType::kQ4_K) ? Id::kInt16SplitPrefillX8 : base));
      MatmulCaps nosplit = c;
      nosplit.dot_q16s_x8_q4_rows8 = false;  // no AVX-VNNI: the int16 tiles stay
      CHECK(select_matmul(s, sp, nosplit).id == base);
    }
  }
}

// ---------------------------------------------------------------------------
// 2-3. Execution through CpuDevice.

struct QWeight {
  DType type;
  int64_t n, k;
  std::vector<uint8_t> bytes;
  std::vector<float> f;  // dequantized [n, k]
};

QWeight make_q(DType t, int64_t n, int64_t k, uint32_t seed) {
  QWeight w{t, n, k, {}, {}};
  const int64_t rb = dtype_row_bytes(t, k);
  w.bytes.resize(static_cast<size_t>(n * rb));
  q4kref::Rng r(seed);
  if (t == DType::kQ4_K) {
    const q4kref::Weight q = q4kref::make_weight(n, k, seed);
    std::memcpy(w.bytes.data(), q.blocks.data(), w.bytes.size());
  } else if (t == DType::kF32) {
    auto* p = reinterpret_cast<float*>(w.bytes.data());
    for (int64_t i = 0; i < n * k; ++i) p[i] = static_cast<float>(r.uniform(-0.05, 0.05));
  } else if (t == DType::kF16) {
    auto* p = reinterpret_cast<uint16_t*>(w.bytes.data());
    for (int64_t i = 0; i < n * k; ++i) p[i] = fp32_to_fp16(static_cast<float>(r.uniform(-0.05, 0.05)));
  } else {
    for (uint8_t& b : w.bytes) b = static_cast<uint8_t>(r.next());
    // Sane block scales: overwrite the fp16 scale fields with small values.
    const int64_t nb = n * k / dtype_block_elems(t);
    const int32_t bb = dtype_info(t).block_bytes;
    for (int64_t i = 0; i < nb; ++i) {
      uint8_t* blk = w.bytes.data() + i * bb;
      const uint16_t d = fp32_to_fp16(static_cast<float>(r.uniform(1e-3, 1e-2)));
      if (t == DType::kQ6_K) {
        std::memcpy(blk + bb - 2, &d, 2);  // d is the last field
      } else {
        std::memcpy(blk, &d, 2);  // Q8_0 / Q4_0: d first
      }
    }
  }
  w.f.resize(static_cast<size_t>(n * k));
  for (int64_t i = 0; i < n; ++i) CHECK(dequantize_row(t, w.bytes.data() + i * rb, w.f.data() + i * k, k));
  return w;
}

// max |y - ref| / max |ref| over the m x n output (fp64 reference).
double rel_err(const QWeight& w, const std::vector<float>& x, int64_t ldx, const float* bias, const float* y, int64_t ldy, int64_t m) {
  double emax = 0, rmax = 1e-30;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < w.n; ++j) {
      double s = bias ? bias[j] : 0.0;
      for (int64_t t = 0; t < w.k; ++t) s += static_cast<double>(x[i * ldx + t]) * w.f[j * w.k + t];
      emax = std::max(emax, std::abs(s - y[i * ldy + j]));
      rmax = std::max(rmax, std::abs(s));
    }
  }
  return emax / rmax;
}

double tolerance(MatmulActFormat a) {
  switch (a) {
    case MatmulActFormat::kF32: return 2e-5;
    case MatmulActFormat::kInt16:
    case MatmulActFormat::kInt16Split: return 5e-4;
    default: return 4e-2;  // int8 activations
  }
}

uint64_t calls_of(Id id) { return CpuDevice::matmul_stats()[static_cast<size_t>(id)].calls; }

void test_execution() {
  ThreadPool pool1(1), pool4(4);
  const int64_t k = 512;
  struct Case {
    DType t;
    int64_t n;
  };
  const Case cases[] = {{DType::kQ4_K, 1024}, {DType::kQ6_K, 1024}, {DType::kQ8_0, 256}, {DType::kF16, 256}, {DType::kF32, 256}};
  std::vector<KernelPlan> plans;
  plans.push_back(KernelPlan::defaults());
  {
    KernelPlan p = KernelPlan::defaults();
    p.int16_prefill_max_rows = 128;
    p.int16_prefill_min_weights = 0;
    plans.push_back(p);
    p.int16_prefill_split = true;
    plans.push_back(p);
  }
  {
    KernelPlan p = KernelPlan::defaults();
    p.int16_decode_max_rows = 8;  // the FFN-down plan: int8 off, int16 on (transformer.cpp)
    p.int8_decode_max_rows = 0;
    plans.push_back(p);
  }
  {
    KernelPlan p = KernelPlan::defaults();
    p.int8_superblock_min_rows = 2;
    p.int8_subscale_min_rows = 0;
    plans.push_back(p);
  }
  {
    KernelPlan p = KernelPlan::defaults();
    p.packed_int8_8rows = true;
    plans.push_back(p);
  }
  {
    KernelPlan p = KernelPlan::defaults();
    p.q4_repack_max_rows = 64;  // DD-078 int8 prefill (fails DD-053; exercised for correctness only)
    plans.push_back(p);
  }
  {
    KernelPlan p = KernelPlan::defaults();
    p.matmul_disabled = 0;
    (void)parse_matmul_kernel_list("all-accelerated", p.matmul_disabled);
    plans.push_back(p);
  }
  const int64_t ms[] = {1, 2, 4, 5, 8, 9, 16, 33, 64, 128};
  bool seen[kMatmulKernelCount] = {};
  int runs = 0;
  for (const Case& cs : cases) {
    const QWeight w = make_q(cs.t, cs.n, k, 17 + static_cast<uint32_t>(cs.t));
    std::vector<float> bias(static_cast<size_t>(cs.n));
    for (int64_t j = 0; j < cs.n; ++j) bias[static_cast<size_t>(j)] = 0.01f * static_cast<float>(j % 7) - 0.03f;
    CpuDevice d1(pool1, CpuIsa::kAvx2), d4(pool4, CpuIsa::kAvx2);
    d1.prepack_weight(qview(w.bytes.data(), cs.t, cs.n, k), true);
    d4.prepack_weight(qview(w.bytes.data(), cs.t, cs.n, k), true);
    for (const KernelPlan& p : plans) {
      d1.set_kernel_plan(p);
      d4.set_kernel_plan(p);
      for (int64_t m : ms) {
        const std::vector<float> x = q4kref::make_x(m, k, k, 5 + static_cast<uint32_t>(m));
        std::vector<float> y1(static_cast<size_t>(m * cs.n)), y4(y1.size());
        const TensorView wv = qview(w.bytes.data(), cs.t, cs.n, k);
        const MatmulSelection sel = d1.plan_matmul(fview(x.data(), m, k), wv);
        CHECK(d4.plan_matmul(fview(x.data(), m, k), wv).id == sel.id);
        const uint64_t c0 = calls_of(sel.id);
        d1.matmul(fview(x.data(), m, k), wv, nullptr, fview(y1.data(), m, cs.n));
        CHECK(calls_of(sel.id) == c0 + 1);  // the named kernel is the one that ran
        d4.matmul(fview(x.data(), m, k), wv, nullptr, fview(y4.data(), m, cs.n));
        CHECK(y1 == y4);  // thread count never changes results
        const double e = rel_err(w, x, k, nullptr, y1.data(), cs.n, m);
        if (e > tolerance(sel.act)) {
          std::fprintf(stderr, "error %.3g > %.3g for %s m %lld (%s)\n", e, tolerance(sel.act),
                       std::string(dtype_name(cs.t)).c_str(), static_cast<long long>(m), sel.explain().c_str());
        }
        CHECK(e <= tolerance(sel.act));
        if (p.matmul_disabled != 0) CHECK(sel.id == Id::kF32ExpandGemm || sel.id == Id::kF32RowDot);
        seen[static_cast<size_t>(sel.id)] = true;
        ++runs;
      }
    }
    // bias and strided rows on the selected default path
    {
      d4.set_kernel_plan(KernelPlan::defaults());
      const int64_t m = 9, ldx = k + 13, ldy = cs.n + 5;
      const std::vector<float> x = q4kref::make_x(m, k, ldx, 99);
      std::vector<float> y(static_cast<size_t>(m * ldy), 7.0f);
      std::vector<float> bv = bias;
      TensorView b = fview(bv.data(), 1, cs.n);
      d4.matmul(fview(x.data(), m, k, ldx), qview(w.bytes.data(), cs.t, cs.n, k), &b, fview(y.data(), m, cs.n, ldy));
      CHECK(rel_err(w, x, ldx, bias.data(), y.data(), ldy, m) <= tolerance(MatmulActFormat::kF32));
      for (int64_t i = 0; i < m; ++i) CHECK(y[static_cast<size_t>(i * ldy + cs.n)] == 7.0f);  // padding untouched
    }
  }
  const bool vnni = cpu_info().features.avx_vnni;
  int kinds = 0;
  for (size_t i = 0; i < kMatmulKernelCount; ++i) {
    kinds += seen[i] ? 1 : 0;
    if (!seen[i]) std::printf("  (not exercised: %s)\n", std::string(matmul_kernel_name(static_cast<Id>(i))).c_str());
  }
  CHECK(seen[static_cast<size_t>(Id::kInt8Per32)] && seen[static_cast<size_t>(Id::kInt8SubscaledX8)] &&
        seen[static_cast<size_t>(Id::kInt8Subscaled)] && seen[static_cast<size_t>(Id::kInt8Superblock)] &&
        seen[static_cast<size_t>(Id::kInt8SubscaledX8Rows8)] && seen[static_cast<size_t>(Id::kInt16PrefillX8)] &&
        seen[static_cast<size_t>(Id::kInt16X8)] && seen[static_cast<size_t>(Id::kF32ExpandGemm)] &&
        seen[static_cast<size_t>(Id::kF32RowDot)]);
  if (vnni) CHECK(seen[static_cast<size_t>(Id::kInt16SplitPrefillX8)]);
  std::printf("  execution: %d matmuls over %d kernels, each within its accuracy class of fp64, 1 thread == 4 threads\n", runs, kinds);
}

void test_split_identity() {
  if (!cpu_info().features.avx_vnni) {
    std::printf("  split-byte identity: skipped (no AVX-VNNI)\n");
    return;
  }
  const int64_t n = 1024, k = 1536;
  const QWeight w = make_q(DType::kQ4_K, n, k, 3);
  std::vector<float> bias(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j) bias[static_cast<size_t>(j)] = static_cast<float>(j % 5) * 0.25f - 0.5f;
  int compared = 0;
  for (int threads : {1, 3, 10}) {
    ThreadPool pool(threads);
    CpuDevice dev(pool, CpuIsa::kAvx2);
    dev.prepack_weight(qview(w.bytes.data(), DType::kQ4_K, n, k), true);
    KernelPlan p16 = KernelPlan::defaults();
    p16.int16_prefill_max_rows = 128;
    p16.int16_prefill_min_weights = 0;  // 1024 x 1536 is below the 4M serving gate
    KernelPlan ps = p16;
    ps.int16_prefill_split = true;
    for (int64_t m : {9, 10, 11, 12, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 63, 64, 65, 96, 127, 128}) {
      for (bool strided : {false, true}) {
        const int64_t ldx = strided ? k + 32 : k, ldy = strided ? n + 3 : n;
        const std::vector<float> x = q4kref::make_x(m, k, ldx, 1000 + static_cast<uint32_t>(m));
        std::vector<float> ya(static_cast<size_t>(m * ldy), 0.0f), yb(ya.size(), 0.0f);
        TensorView b = fview(bias.data(), 1, n);
        const TensorView wv = qview(w.bytes.data(), DType::kQ4_K, n, k);
        dev.set_kernel_plan(p16);
        CHECK(dev.plan_matmul(fview(x.data(), m, k, ldx), wv).id == Id::kInt16PrefillX8);
        dev.matmul(fview(x.data(), m, k, ldx), wv, &b, fview(ya.data(), m, n, ldy));
        dev.set_kernel_plan(ps);
        CHECK(dev.plan_matmul(fview(x.data(), m, k, ldx), wv).id == Id::kInt16SplitPrefillX8);
        const uint64_t c0 = calls_of(Id::kInt16SplitPrefillX8);
        dev.matmul(fview(x.data(), m, k, ldx), wv, &b, fview(yb.data(), m, n, ldy));
        CHECK(calls_of(Id::kInt16SplitPrefillX8) == c0 + 1);
        CHECK(std::memcmp(ya.data(), yb.data(), ya.size() * sizeof(float)) == 0);  // bit for bit
        ++compared;
      }
    }
  }
  std::printf("  split-byte identity: %d matmuls bit-identical to the int16 tiles (M 9..128, strided, bias, 1/3/10 threads)\n",
              compared);
}

// Wide decode (KernelPlan::decode_wide_max_rows; the batch planner raises
// int8_superblock_max_rows in decode-only steps): a 9..16-row matmul runs the
// int8 / int16 decode kernels, and every row equals that row computed in a
// batch of <= 4 rows under the default plan, bit for bit.
void test_wide_decode() {
  ThreadPool pool(4);
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t k = 1536;
  struct Case {
    DType t;
    int64_t n;
    bool down;  // the FFN-down plan: int8 off, int16 on
    Id want;
  };
  const Case cases[] = {{DType::kQ4_K, 1024, false, Id::kInt8SubscaledX8},
                        {DType::kQ6_K, 1024, false, Id::kInt8Subscaled},
                        {DType::kQ4_K, 1536, true, Id::kInt16X8}};
  int compared = 0;
  for (const Case& cs : cases) {
    const QWeight w = make_q(cs.t, cs.n, k, 40 + static_cast<uint32_t>(cs.t));
    const TensorView wv = qview(w.bytes.data(), cs.t, cs.n, k);
    dev.prepack_weight(wv, cs.down);
    KernelPlan base = KernelPlan::defaults();
    if (cs.down) {
      base.int16_decode_max_rows = base.int8_decode_max_rows;  // transformer.cpp down plan
      base.int8_decode_max_rows = 0;
    }
    // Wide decode with the 8-row tiles (wide_rows8, the default) and without.
    for (bool rows8 : {true, false}) {
      KernelPlan wide = base;
      wide.int8_superblock_max_rows = 16;
      wide.wide_rows8 = rows8;
      // Q4_K above 8 rows: the 8-row tiles of the same kernel family when allowed and available.
      const bool tiles8 = rows8 && cs.want == Id::kInt8SubscaledX8 && dev.kernels().dot_q8_sx_x8_q4_K_rows8 != nullptr;
      const Id want_wide = tiles8 ? Id::kInt8SubscaledX8Rows8 : cs.want;
      for (int64_t m : {9, 12, 16}) {
        const std::vector<float> x = q4kref::make_x(m, k, k, 300 + static_cast<uint32_t>(m));
        std::vector<float> yw(static_cast<size_t>(m * cs.n)), yc(yw.size());
        dev.set_kernel_plan(base);
        CHECK(dev.plan_matmul(fview(x.data(), m, k), wv).id != cs.want);  // today: fp32 above 8 rows
        dev.set_kernel_plan(wide);
        CHECK(dev.plan_matmul(fview(x.data(), m, k), wv).id == want_wide);
        const uint64_t c0 = calls_of(want_wide);
        dev.matmul(fview(x.data(), m, k), wv, nullptr, fview(yw.data(), m, cs.n));
        CHECK(calls_of(want_wide) == c0 + 1);  // the wide kernel is the one that ran
        dev.set_kernel_plan(base);
        for (int64_t r0 = 0; r0 < m; r0 += 4) {
          const int64_t mm = std::min<int64_t>(4, m - r0);
          if (mm >= 2) CHECK(dev.plan_matmul(fview(x.data() + r0 * k, mm, k), wv).id == cs.want);
          dev.matmul(fview(x.data() + r0 * k, mm, k), wv, nullptr, fview(yc.data() + r0 * cs.n, mm, cs.n));
        }
        // rows of a final 1-row chunk use the per-32 kernel; compare the rest
        const int64_t full = m % 4 == 1 ? m - 1 : m;
        CHECK(std::memcmp(yw.data(), yc.data(), static_cast<size_t>(full * cs.n) * sizeof(float)) == 0);
        CHECK(rel_err(w, x, k, nullptr, yw.data(), cs.n, m) <= tolerance(cs.down ? MatmulActFormat::kInt16 : MatmulActFormat::kInt8Subscaled));
        ++compared;
      }
    }
    // Rows 5..8 keep the 4-row tiles with wide decode alone (wide_rows8 only acts above 8 rows).
    if (cs.want == Id::kInt8SubscaledX8) {
      KernelPlan wide = base;
      wide.int8_superblock_max_rows = 16;
      dev.set_kernel_plan(wide);
      const std::vector<float> x8 = q4kref::make_x(8, k, k, 77);
      CHECK(dev.plan_matmul(fview(x8.data(), 8, k), wv).id == Id::kInt8SubscaledX8);
    }
  }
  std::printf("  wide decode: %d matmuls of 9..16 rows (8-row and 4-row tiles) bit-identical per row to <= 4-row batches "
              "(int8 x8, int8 rows, int16 x8)\n", compared);
}

// A tier without SIMD integer kernels (generic: CPUs without AVX2, or
// DYNACORE_ISA=generic) with every accelerated path asked for: selection must
// fall back to the fp32 kernels and results stay correct.
void test_generic_fallback() {
  ThreadPool pool(2);
  CpuDevice dev(pool, CpuIsa::kGeneric);
  KernelPlan p = KernelPlan::defaults();
  p.int16_prefill_max_rows = 128;
  p.int16_prefill_min_weights = 0;
  p.int16_prefill_split = true;
  p.int16_decode_max_rows = 8;
  p.int8_superblock_max_rows = 16;  // what wide decode sets
  p.q4_repack_max_rows = 64;
  dev.set_kernel_plan(p);
  int runs = 0;
  for (DType t : {DType::kQ4_K, DType::kQ6_K, DType::kQ8_0}) {
    const int64_t n = 256, k = 512;
    const QWeight w = make_q(t, n, k, 60 + static_cast<uint32_t>(t));
    const TensorView wv = qview(w.bytes.data(), t, n, k);
    dev.prepack_weight(wv, true);  // no-op on this tier
    for (int64_t m : {1, 4, 9, 16, 33}) {
      const std::vector<float> x = q4kref::make_x(m, k, k, 500 + static_cast<uint32_t>(m));
      std::vector<float> y(static_cast<size_t>(m * n));
      const MatmulSelection sel = dev.plan_matmul(fview(x.data(), m, k), wv);
      CHECK(sel.id == Id::kF32ExpandGemm || sel.id == Id::kF32RowDot);
      dev.matmul(fview(x.data(), m, k), wv, nullptr, fview(y.data(), m, n));
      CHECK(rel_err(w, x, k, nullptr, y.data(), n, m) <= tolerance(MatmulActFormat::kF32));
      ++runs;
    }
  }
  std::printf("  generic tier: %d matmuls with every accelerated path requested fall back to fp32, correct\n", runs);
}

// Environment helpers (dynacore/base/env.h): empty = unset; whole-string numbers only.
void test_env_helpers() {
  long l = 7;
  double d = 7;
  CHECK(!parse_env_long(nullptr, l) && !parse_env_long("", l) && !parse_env_long("abc", l) && !parse_env_long("16x", l) && l == 7);
  CHECK(parse_env_long("16", l) && l == 16 && parse_env_long("-3", l) && l == -3 && parse_env_long("0", l) && l == 0);
  CHECK(!parse_env_long("99999999999999999999999", l));
  CHECK(!parse_env_double("", d) && !parse_env_double("3.0x", d) && !parse_env_double("x", d) && d == 7);
  CHECK(parse_env_double("3", d) && d == 3.0 && parse_env_double("0.5", d) && d == 0.5);
  set_env("DYNACORE_TEST_ENV_HELPER", "16");
  CHECK(env_value("DYNACORE_TEST_ENV_HELPER") != nullptr && std::string(env_value("DYNACORE_TEST_ENV_HELPER")) == "16");
  set_env("DYNACORE_TEST_ENV_HELPER", "");  // unsets on Windows; empty value on POSIX: both read as unset
  CHECK(env_value("DYNACORE_TEST_ENV_HELPER") == nullptr);
  CHECK(env_value("DYNACORE_TEST_ENV_NEVER_SET") == nullptr);
#if !defined(_WIN32)
  setenv("DYNACORE_TEST_ENV_HELPER", "", 1);  // an explicitly empty value
  CHECK(std::getenv("DYNACORE_TEST_ENV_HELPER") != nullptr && env_value("DYNACORE_TEST_ENV_HELPER") == nullptr);
#endif
  std::printf("  environment helpers: empty = unset, whole-string numbers only\n");
}

// ---------------------------------------------------------------------------
// 4. Diagnostics.
void test_diagnostics() {
  ThreadPool pool(2);
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t n = 1024, k = 512;
  const QWeight w = make_q(DType::kQ4_K, n, k, 9);
  dev.prepack_weight(qview(w.bytes.data(), DType::kQ4_K, n, k), true);
  const std::vector<float> x = q4kref::make_x(32, k, k, 4);
  std::vector<float> y(static_cast<size_t>(32 * n));
  CpuDevice::reset_matmul_stats();
  for (const MatmulKernelStats& s : CpuDevice::matmul_stats()) CHECK(s.calls == 0 && s.rows == 0 && s.macs == 0 && s.ns == 0);
  dev.set_kernel_plan(KernelPlan::defaults());
  const TensorView wv = qview(w.bytes.data(), DType::kQ4_K, n, k);
  // defaults at 32 rows: fp32 expand, with the int8 and int16 kernels passed over
  const MatmulSelection sel = dev.plan_matmul(fview(x.data(), 32, k), wv, true);
  CHECK(sel.id == Id::kF32ExpandGemm);
  const std::string ex = sel.explain();
  CHECK(ex.find("f32_expand_gemm") == 0);
  CHECK(ex.find("int16_prefill_x8=not_enabled") != std::string::npos);
  CHECK(ex.find("int8_per32=rows") != std::string::npos);
  CHECK(ex.find("external=not_enabled") != std::string::npos);
  CpuDevice::set_matmul_timing(false);
  dev.matmul(fview(x.data(), 32, k), wv, nullptr, fview(y.data(), 32, n));
  dev.matmul(fview(x.data(), 1, k), wv, nullptr, fview(y.data(), 1, n));
  auto st = CpuDevice::matmul_stats();
  const MatmulKernelStats& e = st[static_cast<size_t>(Id::kF32ExpandGemm)];
  CHECK(e.calls == 1 && e.rows == 32 && e.macs == static_cast<uint64_t>(32 * n * k) && e.ns == 0);
  CHECK(st[static_cast<size_t>(Id::kInt8Per32)].calls == 1);
  CHECK(CpuDevice::matmul_shape_stats().empty());  // nothing recorded while timing is off
  CpuDevice::set_matmul_timing(true);
  CHECK(CpuDevice::matmul_timing());
  dev.matmul(fview(x.data(), 32, k), wv, nullptr, fview(y.data(), 32, n));
  st = CpuDevice::matmul_stats();
  CHECK(st[static_cast<size_t>(Id::kF32ExpandGemm)].calls == 2 && st[static_cast<size_t>(Id::kF32ExpandGemm)].ns > 0);
  const auto shapes = CpuDevice::matmul_shape_stats();
  CHECK(shapes.size() == 1 && shapes[0].id == Id::kF32ExpandGemm && shapes[0].n == n && shapes[0].k == k && shapes[0].m == 32 &&
        shapes[0].calls == 1 && shapes[0].weight == DType::kQ4_K);
  CpuDevice::set_matmul_timing(false);
  // descriptors: one per id, unique names, defaults consistent with accuracy
  int defaults_on = 0;
  for (const MatmulKernelDesc& d : matmul_kernels()) {
    CHECK(matmul_kernel_name(d.id) == d.name);
    MatmulKernelId back;
    CHECK(parse_matmul_kernel(d.name, back) && back == d.id);
    if (d.accuracy == MatmulAccuracy::kFailsDd053 || d.accuracy == MatmulAccuracy::kExperimental) CHECK(!d.default_on);
    defaults_on += d.default_on ? 1 : 0;
  }
  CHECK(matmul_kernels().size() == kMatmulKernelCount);
  CHECK(!matmul_kernel_desc(Id::kInt16SplitPrefillX8).default_on && !matmul_kernel_desc(Id::kInt16PrefillX8).default_on);
  std::printf("  diagnostics: counters, timing, shape table, explain reasons, %d default-on kernels of %zu\n", defaults_on,
              kMatmulKernelCount);
}

// ---------------------------------------------------------------------------
// 5. Validation, kernel lists, disabled kernels.
void test_validation() {
  CHECK(validate_matmul_shape({4, 8, 256, DType::kQ4_K, true}).ok());
  CHECK(!validate_matmul_shape({0, 8, 256, DType::kQ4_K, true}).ok());
  CHECK(!validate_matmul_shape({4, 0, 256, DType::kQ4_K, true}).ok());
  CHECK(!validate_matmul_shape({4, 8, 100, DType::kQ4_K, true}).ok());  // not a multiple of 256
  CHECK(!validate_matmul_shape({4, 8, 48, DType::kQ8_0, true}).ok());   // not a multiple of 32
  CHECK(validate_matmul_shape({4, 8, 7, DType::kF32, true}).ok());
  CHECK(validate_matmul_shape({4, 8, 7, DType::kI8, true}).code() == StatusCode::kUnsupported);
  uint32_t mask = 0;
  CHECK(parse_matmul_kernel_list("int8_per32, int16_x8", mask).ok());
  CHECK(mask == (matmul_kernel_bit(Id::kInt8Per32) | matmul_kernel_bit(Id::kInt16X8)));
  CHECK(!parse_matmul_kernel_list("int8_per32,nope", mask).ok());
  CHECK(mask == (matmul_kernel_bit(Id::kInt8Per32) | matmul_kernel_bit(Id::kInt16X8)));  // unchanged on error
  CHECK(!parse_matmul_kernel_list("f32_expand_gemm", mask).ok());                         // the reference stays
  CHECK(parse_matmul_kernel_list("all-accelerated", mask).ok());
  CHECK((mask & matmul_kernel_bit(Id::kF32ExpandGemm)) == 0 && (mask & matmul_kernel_bit(Id::kF32RowDot)) == 0);
  CHECK((mask & matmul_kernel_bit(Id::kInt8Per32)) != 0 && (mask & matmul_kernel_bit(Id::kInt16SplitPrefillX8)) != 0);
  CHECK(parse_matmul_kernel_list("", mask).ok() && mask == 0);
  // disabled kernels fall through to the next eligible one, with a reason
  const MatmulCaps c = caps_avx2(DType::kQ4_K, true, true);
  KernelPlan p = KernelPlan::defaults();
  CHECK(select_matmul({4, 1024, 1536, DType::kQ4_K, true}, p, c).id == Id::kInt8SubscaledX8);
  p.matmul_disabled = matmul_kernel_bit(Id::kInt8SubscaledX8);
  const MatmulSelection s = select_matmul({4, 1024, 1536, DType::kQ4_K, true}, p, c, true);
  CHECK(s.id == Id::kInt8Subscaled);
  CHECK(s.explain().find("int8_subscaled_x8=disabled") != std::string::npos);
  p.matmul_disabled |= matmul_kernel_bit(Id::kInt8Subscaled);
  CHECK(select_matmul({4, 1024, 1536, DType::kQ4_K, true}, p, c).id == Id::kInt8Per32);
  p.matmul_disabled |= matmul_kernel_bit(Id::kInt8Per32);
  CHECK(select_matmul({4, 1024, 1536, DType::kQ4_K, true}, p, c).id == Id::kF32ExpandGemm);
  CHECK(select_matmul({1, 1024, 1536, DType::kQ4_K, true}, p, c).id == Id::kF32RowDot);
  std::printf("  validation, kernel lists, disabled-kernel fall-through\n");
}

// ---------------------------------------------------------------------------
// 6. External kernels.
void test_external() {
  ThreadPool pool(2);
  CpuDevice dev(pool, CpuIsa::kAvx2);
  const int64_t n = 256, k = 512;
  const QWeight w = make_q(DType::kQ8_0, n, k, 21);
  std::atomic<int> runs{0};
  ExternalMatmulKernel ext;
  ext.name = "test_naive";
  ext.eligible = [](const MatmulShape& s, const void*) { return s.m == 7 && s.weight == DType::kQ8_0; };
  ext.run = [&](const TensorView& x, const TensorView& wv, const TensorView* bias, const TensorView& y) {
    (void)wv;
    (void)bias;
    ++runs;
    const int64_t m = x.dim(0);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double s = 0;
        for (int64_t t = 0; t < k; ++t) s += static_cast<double>(x.data_as<const float>()[i * k + t]) * w.f[j * k + t];
        y.data_as<float>()[i * n + j] = static_cast<float>(s);
      }
    }
  };
  CHECK(dev.register_external_matmul(ext).ok());
  CHECK(dev.register_external_matmul(ext).code() == StatusCode::kAlreadyExists);
  ExternalMatmulKernel bad;
  CHECK(!dev.register_external_matmul(bad).ok());
  bad.name = "no_functions";
  CHECK(!dev.register_external_matmul(bad).ok());
  CHECK(dev.external_matmul_count() == 1);
  const std::vector<float> x = q4kref::make_x(7, k, k, 8);
  std::vector<float> y(static_cast<size_t>(7 * n));
  const TensorView wv = qview(w.bytes.data(), DType::kQ8_0, n, k);
  KernelPlan p = KernelPlan::defaults();
  dev.set_kernel_plan(p);  // off by default: never consulted
  CHECK(dev.plan_matmul(fview(x.data(), 7, k), wv).id != Id::kExternal);
  dev.matmul(fview(x.data(), 7, k), wv, nullptr, fview(y.data(), 7, n));
  CHECK(runs == 0);
  p.external_matmul = true;
  dev.set_kernel_plan(p);
  const uint64_t c0 = calls_of(Id::kExternal);
  CHECK(dev.plan_matmul(fview(x.data(), 7, k), wv).id == Id::kExternal);
  dev.matmul(fview(x.data(), 7, k), wv, nullptr, fview(y.data(), 7, n));
  CHECK(runs == 1 && calls_of(Id::kExternal) == c0 + 1);
  CHECK(rel_err(w, x, k, nullptr, y.data(), n, 7) <= 1e-6);
  // ineligible shape: the built-in kernels, unchanged
  CHECK(dev.plan_matmul(fview(x.data(), 6, k), wv).id != Id::kExternal);
  dev.matmul(fview(x.data(), 6, k), wv, nullptr, fview(y.data(), 6, n));
  CHECK(runs == 1);
  // disabled by name
  p.matmul_disabled = matmul_kernel_bit(Id::kExternal);
  dev.set_kernel_plan(p);
  CHECK(dev.plan_matmul(fview(x.data(), 7, k), wv).id != Id::kExternal);
  std::printf("  external kernels: registration errors, opt-in, eligibility, fall-back, disable\n");
}

// ---------------------------------------------------------------------------
// 7. Two devices, two threads.
void test_concurrency() {
  const int64_t n = 1024, k = 512;
  const QWeight w = make_q(DType::kQ4_K, n, k, 31);
  const TensorView wv = qview(w.bytes.data(), DType::kQ4_K, n, k);
  const int64_t ms[] = {1, 4, 9, 33, 64};
  KernelPlan p = KernelPlan::defaults();
  p.int16_prefill_max_rows = 64;
  p.int16_prefill_min_weights = 0;
  p.int16_prefill_split = cpu_info().features.avx_vnni;
  std::vector<std::vector<float>> xs, ref;
  {
    ThreadPool pool(2);
    CpuDevice dev(pool, CpuIsa::kAvx2);
    dev.prepack_weight(wv, true);
    dev.set_kernel_plan(p);
    for (int64_t m : ms) {
      xs.push_back(q4kref::make_x(m, k, k, 70 + static_cast<uint32_t>(m)));
      ref.emplace_back(static_cast<size_t>(m * n));
      dev.matmul(fview(xs.back().data(), m, k), wv, nullptr, fview(ref.back().data(), m, n));
    }
  }
  std::atomic<int> bad{0};
  auto worker = [&](int seed) {
    ThreadPool pool(2);
    CpuDevice dev(pool, CpuIsa::kAvx2);
    dev.prepack_weight(wv, true);
    dev.set_kernel_plan(p);
    for (int it = 0; it < 40; ++it) {
      const size_t i = static_cast<size_t>((it + seed) % static_cast<int>(std::size(ms)));
      std::vector<float> y(ref[i].size());
      dev.matmul(fview(xs[i].data(), ms[i], k), wv, nullptr, fview(y.data(), ms[i], n));
      if (y != ref[i]) ++bad;
    }
  };
  const uint64_t before = [] {
    uint64_t s = 0;
    for (const MatmulKernelStats& st : CpuDevice::matmul_stats()) s += st.calls;
    return s;
  }();
  std::thread a(worker, 0), b(worker, 3);
  a.join();
  b.join();
  uint64_t after = 0;
  for (const MatmulKernelStats& st : CpuDevice::matmul_stats()) after += st.calls;
  CHECK(bad == 0);
  CHECK(after - before == 80);  // no lost counter updates
  std::printf("  concurrency: 2 devices x 40 matmuls on 2 threads == serial results, counters exact\n");
}

}  // namespace

int main() {
  set_env("DYNACORE_Q4_REPACK", "always");  // packing independent of free RAM
  set_env("DYNACORE_Q6_PACKED", "1");       // read once: lets the Q6_K int16 tiles run
  if (!cpu_info().features.avx2) {
    std::printf("matmul registry tests: no AVX2; selection tests only\n");
    test_legacy_equivalence();
    test_validation();
    test_env_helpers();
    test_generic_fallback();
    std::printf("all matmul registry tests passed (%d checks)\n", q4kref::g_checks);
    return 0;
  }
  test_legacy_equivalence();
  test_validation();
  test_env_helpers();
  test_diagnostics();
  test_execution();
  test_split_identity();
  test_wide_decode();
  test_generic_fallback();
  test_external();
  test_concurrency();
  std::printf("all matmul registry tests passed (%d checks)\n", q4kref::g_checks);
  return 0;
}

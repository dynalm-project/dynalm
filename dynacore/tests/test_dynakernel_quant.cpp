// DynaKernel: Q4_K / Q6_K numerical accuracy and grouped execution.
//
// Reference: an independent double-precision decoder of the GGUF Q4_K / Q6_K
// block layouts (written from the format, not shared with the loader) gives
// the exact real value w of every stored weight, and y_ref = sum x * w is
// accumulated in double. The loader's dequantizer, the CPU tier's own
// dequantizer and (optionally) gguf-py fixtures are checked against it.
//
// Tolerances are worst-case bounds derived per element, not fitted:
//   u = 2^-24 (fp32 unit roundoff), T_t = |d*sc*q_t| + |dmin*m_t| (the
//   magnitudes the weight value is formed from), S = sum_t |x_t| T_t.
//   fp32 paths (fused dot, fp32 panel GEMM):  |e| <= (K + 4) u S
//     (fp32 rounding of each dequantized weight: 2u T_t; fp32 accumulation
//     of K products: K u S).
//   int8 paths: activations are rounded to a step h_t (half a step error at
//   most), so additionally |e| <= sum_t (h_t / 2) |w_t|, with h_t from the
//   path's quantizer: per-32 h = amax32 / 127; sub-scaled (DD-077)
//   h = u_s * m_b, u_s = amax256 / (127 * 128), m_b = ceil(amax32 / (127 u_s)).
//   The fp32 term then uses |x_t| + h_t / 2.
//
// Optional: DK_QUANT_FIXTURES=<dir> made by tools/make_quant_fixtures.py
// (gguf-py) adds an external oracle for the dequantizers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/lower.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/executor.h"
#include "dynacore/ir/passes.h"
#include "dynacore/quantization/dequant.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;
using namespace dynacore::dk;
using namespace dynacore::quant;
namespace gir = dynacore::ir;

namespace {

int g_checks = 0;
constexpr double kU = 1.0 / 16777216.0;  // 2^-24

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

CpuIsa this_isa() { return select_isa(cpu_info().features); }

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

// --- independent decoder (double) ------------------------------------------------------

// Value and magnitude T of every element, from the GGUF block fields.
struct Decoded {
  std::vector<double> w, mag;
};

// Q4_K: 8 sub-blocks of 32 with 6-bit scale and min, packed in 12 bytes:
// j < 4: sc = s[j] & 63, m = s[j+4] & 63; j >= 4: low 4 bits from s[j+4],
// high 2 bits from the top bits of s[j-4] (sc) and s[j] (m). Values of 64-value
// group g: low nibbles of qs[32g..32g+31] (sub-block 2g), then high nibbles (2g+1).
void q4_K_scale_min(const uint8_t* s, int j, int& sc, int& m) {
  if (j < 4) {
    sc = s[j] & 63;
    m = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

Decoded decode_q4_K(const BlockQ4_K* b, int64_t nb) {
  Decoded out;
  out.w.resize(static_cast<size_t>(nb * 256));
  out.mag.resize(out.w.size());
  for (int64_t i = 0; i < nb; ++i) {
    const double d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    for (int g = 0; g < 4; ++g) {
      for (int half = 0; half < 2; ++half) {
        int sc = 0, m = 0;
        q4_K_scale_min(b[i].scales, 2 * g + half, sc, m);
        for (int l = 0; l < 32; ++l) {
          const uint8_t byte = b[i].qs[32 * g + l];
          const int q = half == 0 ? (byte & 0x0F) : (byte >> 4);
          const size_t at = static_cast<size_t>(i * 256 + 64 * g + 32 * half + l);
          out.w[at] = d * sc * q - dmin * m;
          out.mag[at] = std::abs(d * sc * q) + std::abs(dmin * m);
        }
      }
    }
  }
  return out;
}

// Q6_K: two halves of 128; in half h, value l (< 32) of each quarter c takes
// 4 low bits from ql (nibble c / 2 of byte l + 32 (c % 2)) and 2 high bits
// from qh[l] (bits 2c..2c+1), minus 32, times int8 scale sc[l / 16 + 2c].
Decoded decode_q6_K(const BlockQ6_K* b, int64_t nb) {
  Decoded out;
  out.w.resize(static_cast<size_t>(nb * 256));
  out.mag.resize(out.w.size());
  for (int64_t i = 0; i < nb; ++i) {
    const double d = fp16_to_fp32(b[i].d);
    for (int h = 0; h < 2; ++h) {
      const uint8_t* ql = b[i].ql + 64 * h;
      const uint8_t* qh = b[i].qh + 32 * h;
      const int8_t* sc = b[i].scales + 8 * h;
      for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
          const uint8_t lo_byte = ql[l + 32 * (c % 2)];
          const int lo = c < 2 ? (lo_byte & 0x0F) : (lo_byte >> 4);
          const int hi = (qh[l] >> (2 * c)) & 3;
          const int q = (lo | (hi << 4)) - 32;
          const size_t at = static_cast<size_t>(i * 256 + 128 * h + 32 * c + l);
          out.w[at] = d * sc[l / 16 + 2 * c] * q;
          out.mag[at] = std::abs(out.w[at]);
        }
      }
    }
  }
  return out;
}

// --- quantized test weights ------------------------------------------------------------

// Random block bytes (every bit pattern of scales and codes is valid), fp16
// super-block scales of both signs, and edge blocks: zero scale, saturated
// 6-bit scales/mins (Q4_K), int8 scale -128 (Q6_K).
std::vector<BlockQ4_K> make_q4_K(int64_t nb, uint32_t seed) {
  Rng r(seed);
  std::vector<BlockQ4_K> v(static_cast<size_t>(nb));
  for (int64_t i = 0; i < nb; ++i) {
    auto* bytes = reinterpret_cast<uint8_t*>(&v[static_cast<size_t>(i)]);
    for (size_t k = 0; k < sizeof(BlockQ4_K); ++k) bytes[k] = static_cast<uint8_t>(r.next());
    const double sd = (r.next() & 1) ? 1 : -1, sm = (r.next() & 1) ? 1 : -1;
    v[static_cast<size_t>(i)].d = fp32_to_fp16(static_cast<float>(sd * r.uniform(1e-3, 2e-2)));
    v[static_cast<size_t>(i)].dmin = fp32_to_fp16(static_cast<float>(sm * r.uniform(1e-3, 2e-2)));
    if (i % 7 == 3) v[static_cast<size_t>(i)].d = fp32_to_fp16(0.0f);
    if (i % 11 == 5) std::memset(v[static_cast<size_t>(i)].scales, 0xFF, 12);
  }
  return v;
}

std::vector<BlockQ6_K> make_q6_K(int64_t nb, uint32_t seed) {
  Rng r(seed);
  std::vector<BlockQ6_K> v(static_cast<size_t>(nb));
  for (int64_t i = 0; i < nb; ++i) {
    auto* bytes = reinterpret_cast<uint8_t*>(&v[static_cast<size_t>(i)]);
    for (size_t k = 0; k < sizeof(BlockQ6_K); ++k) bytes[k] = static_cast<uint8_t>(r.next());
    const double sd = (r.next() & 1) ? 1 : -1;
    v[static_cast<size_t>(i)].d = fp32_to_fp16(static_cast<float>(sd * r.uniform(1e-4, 4e-3)));
    if (i % 5 == 2) v[static_cast<size_t>(i)].d = fp32_to_fp16(0.0f);
    if (i % 3 == 1) v[static_cast<size_t>(i)].scales[(i / 3) % 16] = -128;
  }
  return v;
}

struct QWeight {
  DType dtype;
  std::vector<BlockQ4_K> q4;
  std::vector<BlockQ6_K> q6;
  Decoded dec;
  int64_t n = 0, k = 0;
  void* data() { return dtype == DType::kQ4_K ? static_cast<void*>(q4.data()) : static_cast<void*>(q6.data()); }
};

QWeight make_weight(DType t, int64_t n, int64_t k, uint32_t seed) {
  QWeight w;
  w.dtype = t;
  w.n = n;
  w.k = k;
  const int64_t nb = n * k / 256;
  if (t == DType::kQ4_K) {
    w.q4 = make_q4_K(nb, seed);
    w.dec = decode_q4_K(w.q4.data(), nb);
  } else {
    w.q6 = make_q6_K(nb, seed);
    w.dec = decode_q6_K(w.q6.data(), nb);
  }
  return w;
}

// Activation rows with varied structure: smooth, an outlier channel, tiny
// values, a zero 32-block, large mixed signs, and an all-zero row.
std::vector<float> make_x(int64_t m, int64_t k, uint32_t seed) {
  Rng r(seed);
  std::vector<float> x(static_cast<size_t>(m * k));
  for (int64_t i = 0; i < m; ++i) {
    float* row = x.data() + i * k;
    const int kind = static_cast<int>(i % 6);
    for (int64_t t = 0; t < k; ++t) {
      double v = r.uniform(-1, 1) + r.uniform(-1, 1) + r.uniform(-1, 1);
      if (kind == 2) v *= 1e-3;
      if (kind == 4) v *= 20;
      if (kind == 5 && m > 1) v = 0;  // an all-zero row (int8 scale 0)
      row[t] = static_cast<float>(v);
    }
    if (kind == 1) row[(i * 37 + 11) % k] = (i & 1) ? -40.0f : 40.0f;
    if (kind == 3) std::fill(row + 32, row + 64, 0.0f);
  }
  return x;
}

// --- execution paths ---------------------------------------------------------------------

enum class Path { kFused, kPanel, kInt8Per32, kInt8Subscale, kInt8SubscalePacked };
const char* path_name(Path p) {
  switch (p) {
    case Path::kFused: return "fused_dot";
    case Path::kPanel: return "fp32_panel";
    case Path::kInt8Per32: return "int8_per32";
    case Path::kInt8Subscale: return "int8_subscale";
    case Path::kInt8SubscalePacked: return "int8_sx_packed";
  }
  return "?";
}

// The path CpuDevice::matmul takes for a K-quant weight (k % 256 == 0, dense
// x, no interleaved copy) under ReferenceBackend::plan_for(act): the rules of
// CpuDevice::int8_kernel with KernelPlan::defaults().
Path expected_path(bool int8_dot, ActFormat act, int64_t m) {
  if (act == ActFormat::kAuto && int8_dot) {
    if (m == 1) return Path::kInt8Per32;   // sub-scale needs >= 2 rows
    if (m <= 8) return Path::kInt8Subscale;  // super-block family up to 8 rows
  }
  return m == 1 ? Path::kFused : Path::kPanel;  // expand_min_rows = 2
}

// Rounding step of every activation of row `x` (length k) on `p`.
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
  } else if (p == Path::kInt8Subscale || p == Path::kInt8SubscalePacked) {  // same quantize_act_sx
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

struct Stats {
  double max_abs = 0, max_rel = 0, max_ratio = 0;
  int cases = 0;
};

TensorView view2(void* data, DType t, int64_t r, int64_t c) {
  auto l = TensorLayout::contiguous(t, TensorShape{r, c});
  CHECK(l.ok());
  return TensorView(data, *l);
}

std::vector<float> run_matmul(ReferenceBackend& be, QWeight& w, const std::vector<float>& x, int64_t m, ActFormat act) {
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, m, w.k);
  const ValueId vw = b.weight("w", w.dtype, w.n, w.k);
  const ValueId vy = b.matmul(vx, vw, kNoValue, {.act = act});
  auto p = b.finish();
  CHECK(p.ok());
  std::vector<float> y(static_cast<size_t>(m * w.n), std::nanf(""));
  Bindings bi;
  bi.bind(vx, view2(const_cast<float*>(x.data()), DType::kF32, m, w.k));
  bi.bind(vw, view2(w.data(), w.dtype, w.n, w.k));
  bi.bind(vy, view2(y.data(), DType::kF32, m, w.n));
  const Status s = be.run(*p, bi);
  if (!s.ok()) std::fprintf(stderr, "%s\n", s.to_string().c_str());
  CHECK(s.ok());
  return y;
}

// Checks y against the double reference within the path's bound; folds the
// measurements into `st`.
void check_case(const QWeight& w, const std::vector<float>& x, const std::vector<float>& y, int64_t m, Path p, Stats& st) {
  const int64_t n = w.n, k = w.k;
  double max_ref = 0, case_abs = 0;
  for (int64_t i = 0; i < m; ++i) {
    const float* xr = x.data() + i * k;
    const std::vector<double> h = act_steps(xr, k, p);
    for (int64_t j = 0; j < n; ++j) {
      double ref = 0, s = 0, qerr = 0;
      for (int64_t t = 0; t < k; ++t) {
        const size_t wi = static_cast<size_t>(j * k + t);
        const double xv = xr[t], hv = h[static_cast<size_t>(t)];
        ref += xv * w.dec.w[wi];
        s += (std::abs(xv) + hv / 2) * w.dec.mag[wi];
        qerr += hv / 2 * std::abs(w.dec.w[wi]);
      }
      const double bound = (static_cast<double>(k) + 4) * kU * s + qerr + 1e-30;
      const double e = std::abs(static_cast<double>(y[static_cast<size_t>(i * n + j)]) - ref);
      if (!(e <= bound)) {
        std::fprintf(stderr, "  %s %s M=%lld N=%lld K=%lld y[%lld,%lld]: |e| %.3g > bound %.3g (ref %.6g)\n",
                     dtype_name(w.dtype).data(), path_name(p), static_cast<long long>(m), static_cast<long long>(n),
                     static_cast<long long>(k), static_cast<long long>(i), static_cast<long long>(j), e, bound, ref);
      }
      CHECK(e <= bound);
      max_ref = std::max(max_ref, std::abs(ref));
      case_abs = std::max(case_abs, e);
      st.max_ratio = std::max(st.max_ratio, e / bound);
    }
  }
  st.max_abs = std::max(st.max_abs, case_abs);
  if (max_ref > 0) st.max_rel = std::max(st.max_rel, case_abs / max_ref);
  ++st.cases;
}

// --- dequantizer cross-checks -------------------------------------------------------------

// |got - exact| <= 2u T elementwise (two fp32 roundings of d*sc*q - dmin*m).
double check_dequant(const char* who, const float* got, const Decoded& exact, size_t n) {
  double worst = 0;
  for (size_t t = 0; t < n; ++t) {
    const double e = std::abs(static_cast<double>(got[t]) - exact.w[t]);
    const double bound = 2 * kU * exact.mag[t] + 1e-45;
    if (!(e <= bound)) std::fprintf(stderr, "  %s: element %zu off by %.3g (bound %.3g)\n", who, t, e, bound);
    CHECK(e <= bound);
    worst = std::max(worst, exact.mag[t] > 0 ? e / exact.mag[t] : 0.0);
  }
  return worst;
}

void test_dequantizers() {
  ThreadPool pool(1);
  CpuDevice dev(pool, this_isa());
  for (DType t : {DType::kQ4_K, DType::kQ6_K}) {
    QWeight w = make_weight(t, 16, 1024, t == DType::kQ4_K ? 101 : 202);
    const size_t n = w.dec.w.size();
    std::vector<float> loader(n), tier(n);
    CHECK(dequantize_row(t, w.data(), loader.data(), static_cast<int64_t>(n)));
    dev.kernels().dequant_for(t)(w.data(), tier.data(), static_cast<int64_t>(n));
    const double a = check_dequant("loader", loader.data(), w.dec, n);
    const double b = check_dequant("cpu tier", tier.data(), w.dec, n);
    size_t same = 0;
    for (size_t i = 0; i < n; ++i) same += loader[i] == tier[i];
    std::printf("  %s dequant vs exact decode: loader max %.2g u, %s tier max %.2g u; tier == loader on %zu/%zu\n",
                dtype_name(t).data(), a / kU, isa_name(dev.kernels().isa).data(), b / kU, same, n);
  }
  // External oracle: gguf-py's NumPy dequantization (tools/make_quant_fixtures.py).
  if (const char* dir = std::getenv("DK_QUANT_FIXTURES")) {
    for (DType t : {DType::kQ4_K, DType::kQ6_K}) {
      const std::string base = std::string(dir) + "/" + std::string(dtype_name(t));
      std::ifstream fb(base + ".bin", std::ios::binary), ff(base + ".f32", std::ios::binary);
      CHECK(fb.good() && ff.good());
      std::vector<char> raw((std::istreambuf_iterator<char>(fb)), {});
      std::vector<char> fx((std::istreambuf_iterator<char>(ff)), {});
      const size_t nb = raw.size() / static_cast<size_t>(dtype_block_bytes(t)), n = nb * 256;
      CHECK(nb > 0 && fx.size() == n * 4);
      const float* gguf = reinterpret_cast<const float*>(fx.data());
      const Decoded exact = t == DType::kQ4_K ? decode_q4_K(reinterpret_cast<const BlockQ4_K*>(raw.data()), nb)
                                              : decode_q6_K(reinterpret_cast<const BlockQ6_K*>(raw.data()), nb);
      std::vector<float> loader(n);
      CHECK(dequantize_row(t, raw.data(), loader.data(), static_cast<int64_t>(n)));
      const double g = check_dequant("gguf-py", gguf, exact, n);
      check_dequant("loader (fixture blocks)", loader.data(), exact, n);
      size_t same = 0;
      for (size_t i = 0; i < n; ++i) same += loader[i] == gguf[i];
      std::printf("  %s gguf-py fixture (%zu blocks): within %.2g u of exact decode; loader == gguf-py on %zu/%zu\n",
                  dtype_name(t).data(), nb, g / kU, same, n);
    }
  } else {
    std::printf("  (gguf-py fixtures not checked: set DK_QUANT_FIXTURES)\n");
  }
}

// --- accuracy -----------------------------------------------------------------------------

void test_accuracy() {
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  const bool int8_dot = be.target().int8_dot;
  const int64_t ks[] = {256, 768, 1280, 2304};  // 1, 3, 5 (crosses gemm_k_block 1024), 9 super-blocks
  const int64_t ns[] = {1, 5, 33};
  const int64_t ms[] = {1, 2, 3, 4, 5, 8, 9, 17};
  std::printf("  %-5s %-14s %5s %11s %11s %9s\n", "dtype", "path", "cases", "max |err|", "max rel", "err/bound");
  for (DType t : {DType::kQ4_K, DType::kQ6_K}) {
    Stats st[5];
    uint32_t seed = 1;
    for (int64_t k : ks) {
      for (int64_t n : ns) {
        QWeight w = make_weight(t, n, k, seed++);
        for (int64_t m : ms) {
          const std::vector<float> x = make_x(m, k, seed++);
          std::vector<float> y_f32;
          for (ActFormat act : {ActFormat::kF32, ActFormat::kAuto}) {
            const Path p = expected_path(int8_dot, act, m);
            const std::vector<float> y = run_matmul(be, w, x, m, act);
            check_case(w, x, y, m, p, st[static_cast<int>(p)]);
            if (act == ActFormat::kF32) y_f32 = y;
            // Evidence that the int8 path ran: its rounding changes the bits.
            if (p == Path::kInt8Per32 || p == Path::kInt8Subscale) CHECK(y != y_f32);
            if (act == ActFormat::kAuto && p == expected_path(int8_dot, ActFormat::kF32, m)) CHECK(y == y_f32);
          }
        }
      }
    }
    for (int p = 0; p < 5; ++p) {
      if (st[p].cases == 0) continue;
      std::printf("  %-5s %-14s %5d %11.3g %11.3g %9.3f\n", dtype_name(t).data(), path_name(static_cast<Path>(p)),
                  st[p].cases, st[p].max_abs, st[p].max_rel, st[p].max_ratio);
    }
  }
}

// --- grouped execution ---------------------------------------------------------------------

std::vector<float> random_floats(size_t n, uint32_t seed) {
  Rng r(seed);
  std::vector<float> v(n);
  for (float& f : v) f = static_cast<float>(r.uniform(-1, 1));
  return v;
}

void test_grouping_plan() {
  // Q/K/V sharing x, then an op reading Q (dependency), then one with another contract.
  ProgramBuilder b;
  const ValueId x = b.activation("x", DType::kF32, 2, 256);
  const ValueId wq = b.weight("wq", DType::kQ4_K, 8, 256), wk = b.weight("wk", DType::kQ6_K, 4, 256);
  const ValueId wv = b.weight("wv", DType::kF32, 4, 256), wo = b.weight("wo", DType::kF32, 3, 8);
  const ValueId q = b.matmul(x, wq);
  b.matmul(x, wk);
  b.matmul(x, wv);
  b.matmul(q, wo);                                         // reads q: new call
  b.matmul(x, wk, kNoValue, {.act = ActFormat::kF32});     // other contract: new call
  auto p = b.finish();
  CHECK(p.ok());
  using Calls = std::vector<std::vector<int32_t>>;
  CHECK(ReferenceBackend::plan_calls(*p, RunOptions{}) == (Calls{{0}, {1}, {2}, {3}, {4}}));
  CHECK(ReferenceBackend::plan_calls(*p, RunOptions{.group_independent = true}) == (Calls{{0, 1, 2}, {3}, {4}}));
  std::printf("  grouping plan: independent runs grouped, split at dependencies and contract changes\n");
}

// Independent matmuls of mixed formats, grouped vs one call per op, and both
// vs the double reference.
void test_grouped_vs_single() {
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  const bool int8_dot = be.target().int8_dot;
  for (int64_t k : {512, 1280}) {
    for (int64_t m : {1, 3, 12}) {
      QWeight wa = make_weight(DType::kQ4_K, 24, k, 31), wb = make_weight(DType::kQ6_K, 7, k, 32);
      const std::vector<float> wf = random_floats(static_cast<size_t>(9 * k), 33);
      const std::vector<float> x1 = make_x(m, k, 34), x2 = make_x(m, k, 35);
      ProgramBuilder b;
      const ValueId vx1 = b.activation("x1", DType::kF32, m, k), vx2 = b.activation("x2", DType::kF32, m, k);
      const ValueId va = b.weight("a", DType::kQ4_K, 24, k), vb = b.weight("b", DType::kQ6_K, 7, k);
      const ValueId vf = b.weight("f", DType::kF32, 9, k);
      const ValueId ya = b.matmul(vx1, va), yb = b.matmul(vx1, vb), yf = b.matmul(vx2, vf);
      auto p = b.finish();
      CHECK(p.ok());
      auto run = [&](bool grouped) {
        std::vector<std::vector<float>> out = {std::vector<float>(static_cast<size_t>(m * 24)),
                                               std::vector<float>(static_cast<size_t>(m * 7)),
                                               std::vector<float>(static_cast<size_t>(m * 9))};
        Bindings bi;
        bi.bind(vx1, view2(const_cast<float*>(x1.data()), DType::kF32, m, k));
        bi.bind(vx2, view2(const_cast<float*>(x2.data()), DType::kF32, m, k));
        bi.bind(va, view2(wa.data(), DType::kQ4_K, 24, k));
        bi.bind(vb, view2(wb.data(), DType::kQ6_K, 7, k));
        bi.bind(vf, view2(const_cast<float*>(wf.data()), DType::kF32, 9, k));
        bi.bind(ya, view2(out[0].data(), DType::kF32, m, 24));
        bi.bind(yb, view2(out[1].data(), DType::kF32, m, 7));
        bi.bind(yf, view2(out[2].data(), DType::kF32, m, 9));
        CHECK(be.run(*p, bi, RunOptions{.group_independent = grouped}).ok());
        return out;
      };
      const auto single = run(false), grouped = run(true);
      // Each output against its own reference (ordering: nothing swapped).
      const Path pa = expected_path(int8_dot, ActFormat::kAuto, m);
      Stats st;
      for (const auto* out : {&single, &grouped}) {
        check_case(wa, x1, (*out)[0], m, pa, st);
        check_case(wb, x1, (*out)[1], m, pa, st);
      }
      // f32 weights (no int8 kernel): fp32 bound, magnitudes |w|.
      QWeight wfq;
      wfq.dtype = DType::kF32;
      wfq.n = 9;
      wfq.k = k;
      wfq.dec.w.assign(wf.begin(), wf.end());
      wfq.dec.mag.resize(wf.size());
      for (size_t i = 0; i < wf.size(); ++i) wfq.dec.mag[i] = std::abs(wf[i]);
      check_case(wfq, x2, single[2], m, Path::kPanel, st);
      check_case(wfq, x2, grouped[2], m, Path::kPanel, st);
      // Bitwise expectations, from the two implementations' code paths:
      //  - int8 paths and the single-row fused dot: same kernels -> identical;
      //  - quantized panel, K <= gemm_k_block (1024): identical;
      //  - quantized panel, K > 1024: matmul K-blocks, matmul_many does not;
      //  - f32 weights, M >= 2: matmul uses the panel GEMM, matmul_many the row dot.
      const bool quant_same = pa != Path::kPanel || k <= 1024;
      if (quant_same) CHECK(single[0] == grouped[0] && single[1] == grouped[1]);
      if (m == 1) CHECK(single[2] == grouped[2]);
      std::printf("    K=%-4lld M=%-2lld %-13s q4_K/q6_K %s, f32 %s (max err/bound %.3f)\n", static_cast<long long>(k),
                  static_cast<long long>(m), path_name(pa),
                  single[0] == grouped[0] && single[1] == grouped[1] ? "bit-identical" : "differ (within bound)",
                  single[2] == grouped[2] ? "bit-identical" : "differ (within bound)", st.max_ratio);
    }
  }
}

// GraphExecutor's planned run (matmul group -> matmul_many) vs lowering + grouped run.
void test_vs_executor_plan() {
  const int64_t m = 3, k = 512;
  gir::Graph g("qkv");
  gir::Builder b(g);
  const gir::ValueId x = b.input("x", gir::tensor_type(DType::kF32, {gir::Dim::of(m), gir::Dim::of(k)}));
  const gir::ValueId wq = b.weight("wq", gir::weight_type(DType::kQ4_K, {gir::Dim::of(32), gir::Dim::of(k)}));
  const gir::ValueId wk = b.weight("wk", gir::weight_type(DType::kQ6_K, {gir::Dim::of(16), gir::Dim::of(k)}));
  const gir::ValueId wv = b.weight("wv", gir::weight_type(DType::kQ4_K, {gir::Dim::of(16), gir::Dim::of(k)}));
  const gir::ValueId q = b.matmul(x, wq), kk = b.matmul(x, wk), v = b.matmul(x, wv);
  CHECK(b.status().ok());
  gir::canonicalize(g);
  gir::FusionReport rep;
  const gir::ExecPlan plan = gir::plan_execution(g, gir::FusionOptions{}, gir::CostModel{}, &rep);
  CHECK(rep.groups == 1 && rep.grouped_ops == 3);

  QWeight lq = make_weight(DType::kQ4_K, 32, k, 41), lk = make_weight(DType::kQ6_K, 16, k, 42),
          lv = make_weight(DType::kQ4_K, 16, k, 43);
  const std::vector<float> xs = make_x(m, k, 44);
  ThreadPool pool(4);
  CpuDevice dev(pool, this_isa());
  auto ex = gir::GraphExecutor::create(g, dev, {});
  CHECK(ex.ok());
  CHECK((*ex)->bind(x, view2(const_cast<float*>(xs.data()), DType::kF32, m, k)).ok());
  CHECK((*ex)->bind(wq, view2(lq.data(), DType::kQ4_K, 32, k)).ok());
  CHECK((*ex)->bind(wk, view2(lk.data(), DType::kQ6_K, 16, k)).ok());
  CHECK((*ex)->bind(wv, view2(lv.data(), DType::kQ4_K, 16, k)).ok());
  CHECK((*ex)->run(&plan).ok());

  const LowerResult lr = lower(g);
  CHECK(lr.ok());
  CHECK(ReferenceBackend::plan_calls(lr.lowered.program, RunOptions{.group_independent = true}).size() == 1);
  ReferenceBackend be(pool, this_isa());
  std::vector<float> yq(m * 32), yk(m * 16), yv(m * 16);
  Bindings bi;
  bi.bind(lr.lowered.kernel_value(x), view2(const_cast<float*>(xs.data()), DType::kF32, m, k));
  bi.bind(lr.lowered.kernel_value(wq), view2(lq.data(), DType::kQ4_K, 32, k));
  bi.bind(lr.lowered.kernel_value(wk), view2(lk.data(), DType::kQ6_K, 16, k));
  bi.bind(lr.lowered.kernel_value(wv), view2(lv.data(), DType::kQ4_K, 16, k));
  bi.bind(lr.lowered.kernel_value(q), view2(yq.data(), DType::kF32, m, 32));
  bi.bind(lr.lowered.kernel_value(kk), view2(yk.data(), DType::kF32, m, 16));
  bi.bind(lr.lowered.kernel_value(v), view2(yv.data(), DType::kF32, m, 16));
  CHECK(be.run(lr.lowered.program, bi, RunOptions{.group_independent = true}).ok());
  auto same = [&](gir::ValueId gv, const std::vector<float>& y) {
    const TensorView ev = (*ex)->view(gv);
    return std::memcmp(ev.data(), y.data(), y.size() * 4) == 0;
  };
  CHECK(same(q, yq) && same(kk, yk) && same(v, yv));
  std::printf("  planned GraphExecutor (1 matmul group) vs lowered grouped run: bit-identical\n");
}

bool contains_overlap(const Status& s) { return !s.ok() && s.message().find("overlaps the buffer of") != std::string::npos; }

void test_grouped_failures() {
  // Rejections leave every output untouched, including outputs of ops that
  // come before the failing one: all checks run before any kernel.
  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  ProgramBuilder b;
  const ValueId x = b.activation("x", DType::kF32, 2, 256);
  const ValueId w1 = b.weight("w1", DType::kQ4_K, 8, 256), w2 = b.weight("w2", DType::kQ6_K, 8, 256);
  const ValueId y1 = b.matmul(x, w1), y2 = b.matmul(x, w2);
  auto p = b.finish();
  CHECK(p.ok());
  QWeight a = make_weight(DType::kQ4_K, 8, 256, 51), c = make_weight(DType::kQ6_K, 8, 256, 52);
  const std::vector<float> xs = make_x(2, 256, 53);
  std::vector<float> out(64, -9.0f);
  auto untouched = [&] { return std::all_of(out.begin(), out.end(), [](float v) { return v == -9.0f; }); };
  auto base = [&](Bindings& bi) {
    bi.bind(x, view2(const_cast<float*>(xs.data()), DType::kF32, 2, 256));
    bi.bind(w1, view2(a.data(), DType::kQ4_K, 8, 256));
    bi.bind(w2, view2(c.data(), DType::kQ6_K, 8, 256));
  };
  for (bool grouped : {false, true}) {
    const RunOptions ro{.group_independent = grouped};
    {  // Second output overlaps the first.
      Bindings bi;
      base(bi);
      bi.bind(y1, view2(out.data(), DType::kF32, 2, 8));
      bi.bind(y2, view2(out.data() + 8, DType::kF32, 2, 8));
      CHECK(contains_overlap(be.run(*p, bi, ro)) && untouched());
    }
    {  // Second output too small: the first is not written either.
      Bindings bi;
      base(bi);
      bi.bind(y1, view2(out.data(), DType::kF32, 2, 8), 16 * 4);
      bi.bind(y2, view2(out.data() + 32, DType::kF32, 2, 8), 15 * 4);
      const Status s = be.run(*p, bi, ro);
      CHECK(s.code() == StatusCode::kInvalidArgument && untouched());
    }
    {  // A program the backend cannot run (int8 contract requested).
      ProgramBuilder b2;
      const ValueId x2 = b2.activation("x", DType::kF32, 2, 256);
      const ValueId w = b2.weight("w", DType::kQ4_K, 8, 256);
      b2.matmul(x2, w);
      b2.matmul(x2, w, kNoValue, {.act = ActFormat::kInt8});
      auto q = b2.finish();
      CHECK(q.ok());
      Bindings bi;
      bi.bind(0, view2(const_cast<float*>(xs.data()), DType::kF32, 2, 256));
      bi.bind(1, view2(a.data(), DType::kQ4_K, 8, 256));
      bi.bind(2, view2(out.data(), DType::kF32, 2, 8));
      bi.bind(3, view2(out.data() + 32, DType::kF32, 2, 8));
      CHECK(be.run(*q, bi, ro).code() == StatusCode::kUnsupported && untouched());
    }
  }
  std::printf("  grouped/single rejections: no output written (overlap, capacity, unsupported contract)\n");
}

// --- prepacked (interleaved) Q4_K ---------------------------------------------------------

std::vector<float> run_packed(ReferenceBackend& be, QWeight& w, const std::vector<float>& x, int64_t m,
                              RunOptions ro = {}) {
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, m, w.k);
  const ValueId vw = b.weight("w", w.dtype, w.n, w.k);
  const ValueId vy = b.matmul(vx, vw);
  auto p = b.finish();
  CHECK(p.ok());
  std::vector<float> y(static_cast<size_t>(m * w.n), std::nanf(""));
  Bindings bi;
  bi.bind(vx, view2(const_cast<float*>(x.data()), DType::kF32, m, w.k));
  bi.bind(vw, view2(w.data(), w.dtype, w.n, w.k));
  bi.bind(vy, view2(y.data(), DType::kF32, m, w.n));
  const Status s = be.run(*p, bi, ro);
  if (!s.ok()) std::fprintf(stderr, "%s\n", s.to_string().c_str());
  CHECK(s.ok());
  return y;
}

void test_prepack_validation() {
  ThreadPool pool(2);
  ReferenceBackend be(pool, this_isa());
  QWeight q4 = make_weight(DType::kQ4_K, 16, 512, 61), q6 = make_weight(DType::kQ6_K, 16, 512, 62);
  QWeight odd = make_weight(DType::kQ4_K, 12, 512, 63);
  CHECK(be.prepack_weight(view2(q6.data(), DType::kQ6_K, 16, 512)).status().code() == StatusCode::kUnsupported);
  CHECK(be.prepack_weight(view2(odd.data(), DType::kQ4_K, 12, 512)).status().code() == StatusCode::kUnsupported);
  CHECK(be.prepack_weight(TensorView()).status().code() == StatusCode::kInvalidArgument);
  auto misaligned = be.prepack_weight(view2(reinterpret_cast<char*>(q4.data()) + 1, DType::kQ4_K, 8, 512));
  CHECK(misaligned.status().code() == StatusCode::kInvalidArgument);
  auto short_buf = be.prepack_weight(view2(q4.data(), DType::kQ4_K, 16, 512), 16 * 2 * 144 - 1);
  CHECK(short_buf.status().code() == StatusCode::kInvalidArgument &&
        short_buf.status().message().find("but the buffer has") != std::string::npos);
  if (!be.target().int8_dot) {
    const Status s = be.prepack_weight(view2(q4.data(), DType::kQ4_K, 16, 512)).status();
    CHECK(s.code() == StatusCode::kUnsupported && s.message().find("no interleaved q4_K kernel") != std::string::npos);
    std::printf("  prepack on %s: rejected (no interleaved kernel); packed paths not reachable on this tier\n",
                be.target().isa.c_str());
    return;
  }
  auto ok = be.prepack_weight(view2(q4.data(), DType::kQ4_K, 16, 512), 16 * 2 * 144);
  if (!ok.ok()) std::fprintf(stderr, "%s\n", ok.status().to_string().c_str());
  CHECK(ok.ok() && ok->n == 16 && ok->k == 512 && ok->packed_bytes > 0 && be.packed_weights().size() == 1);
  CHECK(be.prepack_weight(view2(q4.data(), DType::kQ4_K, 16, 512)).ok() && be.packed_weights().size() == 1);
  CHECK(be.prepack_weight(view2(q4.data(), DType::kQ4_K, 8, 1024)).status().code() == StatusCode::kInvalidArgument);
  // Binding the packed source under another shape is refused, not misread.
  ProgramBuilder b;
  const ValueId vx = b.activation("x", DType::kF32, 2, 1024);
  const ValueId vw = b.weight("w", DType::kQ4_K, 8, 1024);
  const ValueId vy = b.matmul(vx, vw);
  auto p = b.finish();
  CHECK(p.ok());
  std::vector<float> x(2 * 1024, 1.0f), y(2 * 8, -5.0f);
  Bindings bi;
  bi.bind(vx, view2(x.data(), DType::kF32, 2, 1024));
  bi.bind(vw, view2(q4.data(), DType::kQ4_K, 8, 1024));
  bi.bind(vy, view2(y.data(), DType::kF32, 2, 8));
  const Status s = be.run(*p, bi);
  CHECK(s.code() == StatusCode::kInvalidArgument && s.message().find("prepacked [16, 512]") != std::string::npos);
  CHECK(y[0] == -5.0f);
  std::printf("  prepack validation: dtype, N %% 8, null, alignment, capacity, re-pack, shape conflict\n");
}

// The packed kernel really computes the result: after packing, the source
// bytes are zeroed (checking disabled); rows 2..8 still match the original
// weights, while 1 row (per-32 kernel) and 9 rows (fp32 panel), which read the
// source, see zeros. Then stale-copy detection, accuracy and grouping.
void test_packed_kernel() {
  ThreadPool pool(4);
  ReferenceBackend be(pool, this_isa());
  if (!be.target().int8_dot) return;  // reported by test_prepack_validation
  {
    QWeight w = make_weight(DType::kQ4_K, 16, 512, 71);
    CHECK(be.prepack_weight(view2(w.data(), DType::kQ4_K, 16, 512)).ok());
    const std::vector<BlockQ4_K> original = w.q4;
    std::memset(w.q4.data(), 0, w.q4.size() * sizeof(BlockQ4_K));
    RunOptions unchecked;
    unchecked.check_packed = false;
    Stats st;
    for (int64_t m : {2, 3, 4, 5, 8}) {
      const std::vector<float> x = make_x(m, 512, 72 + static_cast<uint32_t>(m));
      check_case(w, x, run_packed(be, w, x, m, unchecked), m, Path::kInt8SubscalePacked, st);  // w.dec = original
    }
    for (int64_t m : {1, 9}) {
      const std::vector<float> x = make_x(m, 512, 80);
      const std::vector<float> y = run_packed(be, w, x, m, unchecked);
      CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == 0.0f; }));
    }
    // Stale copy: with checking on (the default), the changed source is refused.
    {
      const std::vector<float> x = make_x(2, 512, 81);
      ProgramBuilder b;
      const ValueId vx = b.activation("x", DType::kF32, 2, 512);
      const ValueId vw = b.weight("w", DType::kQ4_K, 16, 512);
      const ValueId vy = b.matmul(vx, vw);
      auto p = b.finish();
      CHECK(p.ok());
      std::vector<float> y(32, -1.0f);
      Bindings bi;
      bi.bind(vx, view2(const_cast<float*>(x.data()), DType::kF32, 2, 512));
      bi.bind(vw, view2(w.data(), DType::kQ4_K, 16, 512));
      bi.bind(vy, view2(y.data(), DType::kF32, 2, 16));
      const Status s = be.run(*p, bi);
      CHECK(s.code() == StatusCode::kInvalidArgument && s.message().find("interleaved copy is stale") != std::string::npos);
      CHECK(y[0] == -1.0f);
      w.q4 = original;  // same buffer contents again (vector assignment keeps the allocation)
      CHECK(w.data() == be.packed_weights()[0].data);
      CHECK(be.run(*p, bi).ok());
    }
    // Released, then the same buffer holds a different weight of the same
    // shape: the device still has the old copy under this address, but an
    // unregistered weight never uses it.
    CHECK(be.release_packed(w.data()) && be.packed_weights().empty() && !be.release_packed(w.data()));
    const Status again = be.prepack_weight(view2(w.data(), DType::kQ4_K, 16, 512)).status();
    CHECK(again.code() == StatusCode::kUnsupported && again.message().find("released earlier") != std::string::npos);
    QWeight other = make_weight(DType::kQ4_K, 16, 512, 79);
    std::memcpy(w.q4.data(), other.q4.data(), w.q4.size() * sizeof(BlockQ4_K));
    w.dec = other.dec;
    {
      ThreadPool pool3(2);
      ReferenceBackend fresh(pool3, this_isa());
      const std::vector<float> x = make_x(3, 512, 82);
      const std::vector<float> y = run_packed(be, w, x, 3), y_fresh = run_packed(fresh, w, x, 3);
      Stats s2;
      check_case(w, x, y, 3, Path::kInt8Subscale, s2);
      CHECK(y == y_fresh);  // the unpacked kernel, as on a backend that never packed
    }
    std::printf("  packed kernel reached: rows 2-8 match the original weights from the copy alone (err/bound %.3f); "
                "rows 1 and 9 read the zeroed source; stale source refused\n", st.max_ratio);
  }
  // Accuracy over shapes and edge blocks, vs the independent reference; and
  // against the unpacked sub-scaled kernel (another summation order).
  // A fresh backend: `be` refuses to pack an address it released above, and
  // the heap may hand that freed buffer to one of the weights below.
  ThreadPool pool_acc(4);
  ReferenceBackend acc(pool_acc, this_isa());
  Stats packed, unpacked;
  int identical = 0, cases = 0;
  ThreadPool pool2(4);
  ReferenceBackend plain(pool2, this_isa());
  uint32_t seed = 90;
  std::vector<std::unique_ptr<QWeight>> keep;  // packed sources stay alive (and at their addresses) until released
  for (int64_t k : {256, 1280, 2304}) {
    for (int64_t n : {8, 24, 64}) {
      keep.push_back(std::make_unique<QWeight>(make_weight(DType::kQ4_K, n, k, seed++)));
      QWeight& w = *keep.back();
      const auto packed_w = acc.prepack_weight(view2(w.data(), DType::kQ4_K, n, k));
      if (!packed_w.ok()) std::fprintf(stderr, "%s\n", packed_w.status().to_string().c_str());
      CHECK(packed_w.ok());
      for (int64_t m : {2, 3, 4, 5, 8}) {
        const std::vector<float> x = make_x(m, k, seed++);
        const std::vector<float> yp = run_packed(acc, w, x, m), yu = run_packed(plain, w, x, m);
        check_case(w, x, yp, m, Path::kInt8SubscalePacked, packed);
        check_case(w, x, yu, m, Path::kInt8Subscale, unpacked);
        identical += yp == yu;
        ++cases;
        // Grouped (matmul_many) with the packed weight: same kernel per job.
        if (m == 3 && n == 24) {
          ProgramBuilder b;
          const ValueId vx = b.activation("x", DType::kF32, m, k);
          const ValueId vw = b.weight("w", DType::kQ4_K, n, k);
          const ValueId y1 = b.matmul(vx, vw), y2 = b.matmul(vx, vw);
          auto p = b.finish();
          CHECK(p.ok());
          std::vector<float> o1(static_cast<size_t>(m * n)), o2(static_cast<size_t>(m * n));
          Bindings bi;
          bi.bind(vx, view2(const_cast<float*>(x.data()), DType::kF32, m, k));
          bi.bind(vw, view2(w.data(), DType::kQ4_K, n, k));
          bi.bind(y1, view2(o1.data(), DType::kF32, m, n));
          bi.bind(y2, view2(o2.data(), DType::kF32, m, n));
          CHECK(acc.run(*p, bi, RunOptions{.group_independent = true}).ok());
          CHECK(o1 == yp && o2 == yp);
        }
      }
    }
  }
  std::printf("  %-5s %-14s %5s %11s %11s %9s\n", "dtype", "path", "cases", "max |err|", "max rel", "err/bound");
  std::printf("  %-5s %-14s %5d %11.3g %11.3g %9.3f\n", "q4_K", "int8_sx_packed", packed.cases, packed.max_abs,
              packed.max_rel, packed.max_ratio);
  std::printf("  %-5s %-14s %5d %11.3g %11.3g %9.3f\n", "q4_K", "int8_subscale", unpacked.cases, unpacked.max_abs,
              unpacked.max_rel, unpacked.max_ratio);
  std::printf("  packed vs unpacked sub-scaled: bit-identical in %d/%d cases; grouped packed == single packed\n",
              identical, cases);
  for (const auto& w : keep) CHECK(acc.release_packed(w->data()));
  // Defaults untouched: a device that was never asked to prepack has no copies.
  CpuDevice other(pool2, this_isa());
  CHECK(other.prepack_stats().tensors == 0 && plain.packed_weights().empty());
}

}  // namespace

int main() {
  std::printf("test_dynakernel_quant (isa %s)\n", isa_name(this_isa()).data());
  // Per-section check counts: tiers legitimately differ in how many bitwise
  // assertions apply (see docs/dynakernel.md, "Check counts per tier").
  std::vector<std::pair<const char*, int>> counts;
  auto section = [&](const char* name, void (*fn)()) {
    const int before = g_checks;
    fn();
    counts.emplace_back(name, g_checks - before);
  };
  section("dequantizers", test_dequantizers);
  section("accuracy", test_accuracy);
  section("grouping_plan", test_grouping_plan);
  std::printf("  grouped (matmul_many) vs one call per op:\n");
  section("grouped_vs_single", test_grouped_vs_single);
  section("vs_executor_plan", test_vs_executor_plan);
  section("grouped_failures", test_grouped_failures);
  // Packing must not depend on free RAM (DD-078 budget) for these small test weights.
#if defined(_WIN32)
  _putenv_s("DYNACORE_Q4_REPACK", "always");
#else
  setenv("DYNACORE_Q4_REPACK", "always", 1);
#endif
  section("prepack_validation", test_prepack_validation);
  section("packed_kernel", test_packed_kernel);
  std::printf("  checks per section:");
  for (const auto& [name, n] : counts) std::printf(" %s=%d", name, n);
  std::printf("\n");
  std::printf("all quantized/grouped tests passed (%d checks)\n", g_checks);
  return 0;
}

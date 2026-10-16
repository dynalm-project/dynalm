#pragma once

// Test support for Q4_K kernels (header-only): an independent double-precision
// decode of the GGUF Q4_K block layout (written from the format, not shared
// with the loader), synthetic weights with edge blocks, structured
// activations, and the per-element error bounds of docs/dynakernel.md.
//
//   u = 2^-24, T = |d*sc*q| + |dmin*m|, S = sum (|x| + h/2) T
//   |e| <= (K + 5) u (S + |bias|) + sum (h/2) |w|
// with h the activation rounding step of the path (0 for fp32; per-32
// amax32/127; sub-scaled u_s * m_b, DD-077).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

namespace q4kref {

inline int g_checks = 0;

#define Q4K_CHECK(cond)                                                                \
  do {                                                                                 \
    ++q4kref::g_checks;                                                                \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

inline constexpr double kU = 1.0 / 16777216.0;

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

// Q4_K sub-block j's 6-bit scale and min from the 12 packed bytes.
inline void scale_min(const uint8_t* s, int j, int& sc, int& m) {
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
  std::vector<dynacore::quant::BlockQ4_K> blocks;
  std::vector<double> w, mag;  // exact value and magnitude T per element (row-major [N, K])
};

// Random block bytes, fp16 scales of both signs; every 7th block has d = 0,
// every 11th saturated 6-bit scales and mins.
inline Weight make_weight(int64_t n, int64_t k, uint32_t seed) {
  using dynacore::quant::BlockQ4_K;
  Weight w;
  w.n = n;
  w.k = k;
  const int64_t nb = n * k / 256;
  w.blocks.resize(static_cast<size_t>(nb));
  Rng r(seed);
  for (int64_t i = 0; i < nb; ++i) {
    BlockQ4_K& b = w.blocks[static_cast<size_t>(i)];
    auto* bytes = reinterpret_cast<uint8_t*>(&b);
    for (size_t t = 0; t < sizeof(BlockQ4_K); ++t) bytes[t] = static_cast<uint8_t>(r.next());
    b.d = dynacore::fp32_to_fp16(static_cast<float>(((r.next() & 1) ? 1 : -1) * r.uniform(1e-3, 2e-2)));
    b.dmin = dynacore::fp32_to_fp16(static_cast<float>(((r.next() & 1) ? 1 : -1) * r.uniform(1e-3, 2e-2)));
    if (i % 7 == 3) b.d = dynacore::fp32_to_fp16(0.0f);
    if (i % 11 == 5) std::memset(b.scales, 0xFF, 12);
  }
  w.w.resize(static_cast<size_t>(n * k));
  w.mag.resize(w.w.size());
  for (int64_t i = 0; i < nb; ++i) {
    const BlockQ4_K& b = w.blocks[static_cast<size_t>(i)];
    const double d = dynacore::fp16_to_fp32(b.d), dmin = dynacore::fp16_to_fp32(b.dmin);
    for (int g = 0; g < 4; ++g) {
      for (int half = 0; half < 2; ++half) {
        int sc = 0, m = 0;
        scale_min(b.scales, 2 * g + half, sc, m);
        for (int l = 0; l < 32; ++l) {
          const int q = half == 0 ? (b.qs[32 * g + l] & 0x0F) : (b.qs[32 * g + l] >> 4);
          const size_t at = static_cast<size_t>(i * 256 + 64 * g + 32 * half + l);
          w.w[at] = d * sc * q - dmin * m;
          w.mag[at] = std::abs(d * sc * q) + std::abs(dmin * m);
        }
      }
    }
  }
  return w;
}

// Rows (row stride ld >= k): smooth, an outlier channel, tiny values, a zero
// 32-block, large mixed signs, all zero; padding holds 999.
inline std::vector<float> make_x(int64_t m, int64_t k, int64_t ld, uint32_t seed) {
  Rng r(seed);
  std::vector<float> x(static_cast<size_t>(m * ld), 999.0f);
  for (int64_t i = 0; i < m; ++i) {
    float* row = x.data() + i * ld;
    const int kind = static_cast<int>(i % 6);
    for (int64_t t = 0; t < k; ++t) {
      double v = r.uniform(-1, 1) + r.uniform(-1, 1) + r.uniform(-1, 1);
      if (kind == 2) v *= 1e-3;
      if (kind == 4) v *= 20;
      if (kind == 5 && m > 1) v = 0;
      row[t] = static_cast<float>(v);
    }
    if (kind == 1) row[(i * 37 + 11) % k] = (i & 1) ? -40.0f : 40.0f;
    if (kind == 3) std::fill(row + 32, row + 64, 0.0f);
  }
  return x;
}

enum class Path { kFp32, kInt8Per32, kInt8Subscale };

inline std::vector<double> act_steps(const float* x, int64_t k, Path p) {
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

struct Stats {
  double max_abs = 0, max_rel = 0, max_ratio = 0;
  int cases = 0;
  void print(const char* name) const {
    std::printf("  %-28s %5d %11.3g %11.3g %9.3f\n", name, cases, max_abs, max_rel, max_ratio);
  }
};

// y (row stride ldy) against the exact reference within the path's bound.
inline void check_bound(const Weight& w, const std::vector<float>& x, int64_t ldx, const float* bias, const float* y,
                        int64_t ldy, int64_t m, Path p, Stats& st) {
  double max_ref = 0, case_abs = 0;
  for (int64_t i = 0; i < m; ++i) {
    const float* xr = x.data() + i * ldx;
    const std::vector<double> h = act_steps(xr, w.k, p);
    for (int64_t j = 0; j < w.n; ++j) {
      double ref = bias ? bias[j] : 0.0, s = bias ? std::abs(bias[j]) : 0.0, qerr = 0;
      for (int64_t t = 0; t < w.k; ++t) {
        const size_t wi = static_cast<size_t>(j * w.k + t);
        const double hv = h[static_cast<size_t>(t)];
        ref += static_cast<double>(xr[t]) * w.w[wi];
        s += (std::abs(xr[t]) + hv / 2) * w.mag[wi];
        qerr += hv / 2 * std::abs(w.w[wi]);
      }
      const double bound = (static_cast<double>(w.k) + 5) * kU * s + qerr + 1e-30;
      const double e = std::abs(static_cast<double>(y[i * ldy + j]) - ref);
      if (!(e <= bound)) {
        std::fprintf(stderr, "  M=%lld N=%lld K=%lld y[%lld,%lld]: |e| %.3g > bound %.3g\n", static_cast<long long>(m),
                     static_cast<long long>(w.n), static_cast<long long>(w.k), static_cast<long long>(i),
                     static_cast<long long>(j), e, bound);
      }
      Q4K_CHECK(e <= bound);
      max_ref = std::max(max_ref, std::abs(ref));
      case_abs = std::max(case_abs, e);
      st.max_ratio = std::max(st.max_ratio, e / bound);
    }
  }
  st.max_abs = std::max(st.max_abs, case_abs);
  if (max_ref > 0) st.max_rel = std::max(st.max_rel, case_abs / max_ref);
  ++st.cases;
}

}  // namespace q4kref

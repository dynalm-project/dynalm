// AVX2 + FMA + F16C kernels. Compiled with per-file ISA flags (DD-003) and
// only called after runtime detection selected the AVX2 tier.
//
// Quantized formats: each 32-element group is unpacked with byte-level SIMD
// into int8, converted 8 lanes at a time to fp32 and FMA'd with the fp32
// activations. Per-block scales are applied to vector accumulators; there is
// one horizontal sum per call.

#include "dynacore/cpu/cpu_kernels.h"

#if ENGINE_HAS_AVX2 && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dynacore/tensor/fp16.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/quantization/repack.h"

namespace dynacore {
namespace {

using namespace quant;

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

inline __m256 load_i8x8(const int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

// Σ q[i] * x[i] over 16 / 32 int8 weights, as an 8-lane vector.
inline __m256 dot16_i8(const int8_t* q, const float* x) {
  __m256 a = _mm256_mul_ps(load_i8x8(q), _mm256_loadu_ps(x));
  return _mm256_fmadd_ps(load_i8x8(q + 8), _mm256_loadu_ps(x + 8), a);
}
inline __m256 dot32_i8(const int8_t* q, const float* x) {
  __m256 a = _mm256_mul_ps(load_i8x8(q), _mm256_loadu_ps(x));
  a = _mm256_fmadd_ps(load_i8x8(q + 8), _mm256_loadu_ps(x + 8), a);
  __m256 b = _mm256_mul_ps(load_i8x8(q + 16), _mm256_loadu_ps(x + 16));
  b = _mm256_fmadd_ps(load_i8x8(q + 24), _mm256_loadu_ps(x + 24), b);
  return _mm256_add_ps(a, b);
}
inline __m256 sum32(const float* x) {
  return _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(x + 8)),
                       _mm256_add_ps(_mm256_loadu_ps(x + 16), _mm256_loadu_ps(x + 24)));
}

// --- float kernels ------------------------------------------------------------

float dot_f32(const float* a, const float* b, int64_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float s = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

void axpy_f32(float a, const float* x, float* y, int64_t n) {
  const __m256 va = _mm256_set1_ps(a);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += a * x[i];
}

inline __m256 load_f16x8(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

float dot_f16_f32(const uint16_t* a, const float* b, int64_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0;
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(load_f16x8(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(load_f16x8(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(load_f16x8(a + i), _mm256_loadu_ps(b + i), s0);
  float s = hsum(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) s += fp16_to_fp32(a[i]) * b[i];
  return s;
}

void axpy_f16(float a, const uint16_t* x, float* y, int64_t n) {
  const __m256 va = _mm256_set1_ps(a);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, load_f16x8(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += a * fp16_to_fp32(x[i]);
}

float vec_dot_f32(const void* w, const float* x, int64_t n) { return dot_f32(static_cast<const float*>(w), x, n); }
float vec_dot_f16(const void* w, const float* x, int64_t n) {
  return dot_f16_f32(static_cast<const uint16_t*>(w), x, n);
}

inline __m256 load_bf16x8(const uint16_t* p) {
  const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  return _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
}

float vec_dot_bf16(const void* w, const float* x, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  __m256 s = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) s = _mm256_fmadd_ps(load_bf16x8(a + i), _mm256_loadu_ps(x + i), s);
  float r = hsum(s);
  for (; i < n; ++i) r += bf16_to_fp32(a[i]) * x[i];
  return r;
}

void dequant_f16(const void* w, float* out, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, load_f16x8(a + i));
  for (; i < n; ++i) out[i] = fp16_to_fp32(a[i]);
}

// BF16 is the top half of an fp32: widen and shift.
void dequant_bf16(const void* w, float* out, int64_t n) {
  const auto* a = static_cast<const uint16_t*>(w);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    _mm256_storeu_ps(out + i, _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)));
  }
  for (; i < n; ++i) out[i] = bf16_to_fp32(a[i]);
}

// 4 weight rows x 2 activation rows per pass over k: 8 accumulators, and
// each loaded vector feeds several FMAs (x: 4, w: 2).
void gemm_panel(const float* w, int nr, const float* x, int64_t x_stride, int64_t m, int64_t k, float* y,
                int64_t y_stride, bool accumulate) {
  auto put = [accumulate](float& dst, float v) { dst = accumulate ? dst + v : v; };
  if (nr < 4 || k % 8 != 0) {
    for (int64_t i = 0; i < m; ++i) {
      for (int r = 0; r < nr; ++r) put(y[i * y_stride + r], dot_f32(w + r * k, x + i * x_stride, k));
    }
    return;
  }
  const float* w0 = w;
  const float* w1 = w + k;
  const float* w2 = w + 2 * k;
  const float* w3 = w + 3 * k;
  int64_t i = 0;
  // 4 weight rows x 3 activation rows: 12 accumulators + 3 x + 1 w = all 16
  // registers; 12 FMAs per 7 loads, versus 8 per 6 for the 4x2 tile below.
  for (; i + 3 <= m; i += 3) {
    const float* xa = x + i * x_stride;
    const float* xb = xa + x_stride;
    const float* xc = xb + x_stride;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    __m256 c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    for (int64_t kk = 0; kk < k; kk += 8) {
      const __m256 va = _mm256_loadu_ps(xa + kk), vb = _mm256_loadu_ps(xb + kk), vc = _mm256_loadu_ps(xc + kk);
      __m256 vw = _mm256_loadu_ps(w0 + kk);
      a0 = _mm256_fmadd_ps(vw, va, a0);
      b0 = _mm256_fmadd_ps(vw, vb, b0);
      c0 = _mm256_fmadd_ps(vw, vc, c0);
      vw = _mm256_loadu_ps(w1 + kk);
      a1 = _mm256_fmadd_ps(vw, va, a1);
      b1 = _mm256_fmadd_ps(vw, vb, b1);
      c1 = _mm256_fmadd_ps(vw, vc, c1);
      vw = _mm256_loadu_ps(w2 + kk);
      a2 = _mm256_fmadd_ps(vw, va, a2);
      b2 = _mm256_fmadd_ps(vw, vb, b2);
      c2 = _mm256_fmadd_ps(vw, vc, c2);
      vw = _mm256_loadu_ps(w3 + kk);
      a3 = _mm256_fmadd_ps(vw, va, a3);
      b3 = _mm256_fmadd_ps(vw, vb, b3);
      c3 = _mm256_fmadd_ps(vw, vc, c3);
    }
    float* ya = y + i * y_stride;
    float* yb = ya + y_stride;
    float* yc = yb + y_stride;
    put(ya[0], hsum(a0));
    put(ya[1], hsum(a1));
    put(ya[2], hsum(a2));
    put(ya[3], hsum(a3));
    put(yb[0], hsum(b0));
    put(yb[1], hsum(b1));
    put(yb[2], hsum(b2));
    put(yb[3], hsum(b3));
    put(yc[0], hsum(c0));
    put(yc[1], hsum(c1));
    put(yc[2], hsum(c2));
    put(yc[3], hsum(c3));
  }
  for (; i + 2 <= m; i += 2) {
    const float* xa = x + i * x_stride;
    const float* xb = xa + x_stride;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (int64_t kk = 0; kk < k; kk += 8) {
      const __m256 va = _mm256_loadu_ps(xa + kk), vb = _mm256_loadu_ps(xb + kk);
      __m256 vw = _mm256_loadu_ps(w0 + kk);
      a0 = _mm256_fmadd_ps(vw, va, a0);
      b0 = _mm256_fmadd_ps(vw, vb, b0);
      vw = _mm256_loadu_ps(w1 + kk);
      a1 = _mm256_fmadd_ps(vw, va, a1);
      b1 = _mm256_fmadd_ps(vw, vb, b1);
      vw = _mm256_loadu_ps(w2 + kk);
      a2 = _mm256_fmadd_ps(vw, va, a2);
      b2 = _mm256_fmadd_ps(vw, vb, b2);
      vw = _mm256_loadu_ps(w3 + kk);
      a3 = _mm256_fmadd_ps(vw, va, a3);
      b3 = _mm256_fmadd_ps(vw, vb, b3);
    }
    float* ya = y + i * y_stride;
    float* yb = ya + y_stride;
    put(ya[0], hsum(a0));
    put(ya[1], hsum(a1));
    put(ya[2], hsum(a2));
    put(ya[3], hsum(a3));
    put(yb[0], hsum(b0));
    put(yb[1], hsum(b1));
    put(yb[2], hsum(b2));
    put(yb[3], hsum(b3));
  }
  for (; i < m; ++i) {
    for (int r = 0; r < 4; ++r) put(y[i * y_stride + r], dot_f32(w + r * k, x + i * x_stride, k));
  }
}

// --- quantized kernels --------------------------------------------------------

float vec_dot_q8_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(b[i].d)), dot32_i8(b[i].qs, x), acc);
  }
  return hsum(acc);
}

// Q4_1: value = d * q + m (q in 0..15), so dot = d * (q . x) + m * sum(x).
float vec_dot_q4_1(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_1*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  alignas(32) int8_t q[32];
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    _mm_store_si128(reinterpret_cast<__m128i*>(q), _mm_and_si128(bytes, low4));
    _mm_store_si128(reinterpret_cast<__m128i*>(q + 16), _mm_and_si128(_mm_srli_epi16(bytes, 4), low4));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(b[i].d)), dot32_i8(q, x), acc);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(b[i].m)), sum32(x), acc);
  }
  return hsum(acc);
}

float vec_dot_q4_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F), eight = _mm_set1_epi8(8);
  alignas(32) int8_t q[32];
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    _mm_store_si128(reinterpret_cast<__m128i*>(q), _mm_sub_epi8(_mm_and_si128(bytes, low4), eight));
    _mm_store_si128(reinterpret_cast<__m128i*>(q + 16),
                    _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), eight));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(b[i].d)), dot32_i8(q, x), acc);
  }
  return hsum(acc);
}

// Expands 32 bits into 32 bytes: byte e = 0xFF if bit e is set.
inline __m256i bytes_from_bits_32(uint32_t bits) {
  const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
  __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)), shuf);
  const __m256i mask = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ull));
  return _mm256_cmpeq_epi8(_mm256_and_si256(v, mask), mask);
}

float vec_dot_q5_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  alignas(32) int8_t q[32];
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    uint32_t qh;
    std::memcpy(&qh, b[i].qh, 4);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    // Elements 0..15 are low nibbles, 16..31 high nibbles; bit e of qh is bit 4 of element e.
    __m256i v = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), _mm_and_si128(bytes, low4));
    v = _mm256_or_si256(v, _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(q), _mm256_sub_epi8(v, _mm256_set1_epi8(16)));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(b[i].d)), dot32_i8(q, x), acc);
  }
  return hsum(acc);
}

float vec_dot_q4_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  alignas(32) int8_t lo[32], hi[32];
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const uint8_t* qs = b[i].qs;
    for (int is = 0; is < 8; is += 2, qs += 32, x += 64) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
      _mm256_store_si256(reinterpret_cast<__m256i*>(lo), _mm256_and_si256(bytes, low4));
      _mm256_store_si256(reinterpret_cast<__m256i*>(hi), _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4));
      uint8_t sc, m;
      get_scale_min_k4(is, b[i].scales, sc, m);
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc), dot32_i8(lo, x), acc);
      acc = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * m), sum32(x), acc);
      get_scale_min_k4(is + 1, b[i].scales, sc, m);
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc), dot32_i8(hi, x + 32), acc);
      acc = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * m), sum32(x + 32), acc);
    }
  }
  return hsum(acc);
}

float vec_dot_q6_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  alignas(32) int8_t q[4][32];
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    const uint8_t* ql = b[i].ql;
    const uint8_t* qh = b[i].qh;
    const int8_t* sc = b[i].scales;
    for (int part = 0; part < 2; ++part, ql += 64, qh += 32, sc += 8, x += 128) {
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      // Bits 0-1 / 2-3 / 4-5 / 6-7 of qh extend the four nibble groups.
      auto hbits = [&](int shift) {
        return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4);
      };
      const __m256i v1 = _mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0));
      const __m256i v2 = _mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2));
      const __m256i v3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4));
      const __m256i v4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[0]), _mm256_sub_epi8(v1, k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[1]), _mm256_sub_epi8(v2, k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[2]), _mm256_sub_epi8(v3, k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[3]), _mm256_sub_epi8(v4, k32));
      // Group g covers elements [32*g, 32*g + 32): scales sc[2g] (first 16), sc[2g+1] (next 16).
      for (int g = 0; g < 4; ++g) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[2 * g]), dot16_i8(q[g], x + 32 * g), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * sc[2 * g + 1]), dot16_i8(q[g] + 16, x + 32 * g + 16), acc);
      }
    }
  }
  return hsum(acc);
}

// --- quantized dequantization (expand path for batched matmuls) ----------------
// Same SIMD unpacking as the dot kernels, but stores scale * q (+ offset).

// out[0..n) = d * q[0..n) + mn, n a multiple of 8.
inline void store_scaled(const int8_t* q, int n, float d, float mn, float* out) {
  const __m256 vd = _mm256_set1_ps(d), vm = _mm256_set1_ps(mn);
  for (int i = 0; i < n; i += 8) _mm256_storeu_ps(out + i, _mm256_fmadd_ps(load_i8x8(q + i), vd, vm));
}

void dequant_q8_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) store_scaled(b[i].qs, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
}

void dequant_q4_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F), eight = _mm_set1_epi8(8);
  alignas(32) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    _mm_store_si128(reinterpret_cast<__m128i*>(q), _mm_sub_epi8(_mm_and_si128(bytes, low4), eight));
    _mm_store_si128(reinterpret_cast<__m128i*>(q + 16),
                    _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), eight));
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
  }
}

void dequant_q4_1(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_1*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  alignas(32) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    _mm_store_si128(reinterpret_cast<__m128i*>(q), _mm_and_si128(bytes, low4));
    _mm_store_si128(reinterpret_cast<__m128i*>(q + 16), _mm_and_si128(_mm_srli_epi16(bytes, 4), low4));
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), fp16_to_fp32(b[i].m), out);
  }
}

void dequant_q5_0(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  alignas(32) int8_t q[32];
  for (int64_t i = 0; i < n / kQK; ++i, out += kQK) {
    uint32_t qh;
    std::memcpy(&qh, b[i].qh, 4);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    __m256i v = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), _mm_and_si128(bytes, low4));
    v = _mm256_or_si256(v, _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(q), _mm256_sub_epi8(v, _mm256_set1_epi8(16)));
    store_scaled(q, kQK, fp16_to_fp32(b[i].d), 0.0f, out);
  }
}

void dequant_q4_K(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  alignas(32) int8_t lo[32], hi[32];
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const uint8_t* qs = b[i].qs;
    for (int is = 0; is < 8; is += 2, qs += 32, out += 64) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
      _mm256_store_si256(reinterpret_cast<__m256i*>(lo), _mm256_and_si256(bytes, low4));
      _mm256_store_si256(reinterpret_cast<__m256i*>(hi), _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4));
      uint8_t sc, m;
      get_scale_min_k4(is, b[i].scales, sc, m);
      store_scaled(lo, 32, d * sc, -dmin * m, out);
      get_scale_min_k4(is + 1, b[i].scales, sc, m);
      store_scaled(hi, 32, d * sc, -dmin * m, out + 32);
    }
  }
}

void dequant_q6_K(const void* w, float* out, int64_t n) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  alignas(32) int8_t q[4][32];
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    const uint8_t* ql = b[i].ql;
    const uint8_t* qh = b[i].qh;
    const int8_t* sc = b[i].scales;
    for (int part = 0; part < 2; ++part, ql += 64, qh += 32, sc += 8, out += 128) {
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      auto hbits = [&](int shift) {
        return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4);
      };
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[0]),
                         _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[1]),
                         _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[2]),
                         _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32));
      _mm256_store_si256(reinterpret_cast<__m256i*>(q[3]),
                         _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32));
      for (int g = 0; g < 4; ++g) {
        store_scaled(q[g], 16, d * sc[2 * g], 0.0f, out + 32 * g);
        store_scaled(q[g] + 16, 16, d * sc[2 * g + 1], 0.0f, out + 32 * g + 16);
      }
    }
  }
}

// --- block-wise attention (P7, DD-056) -------------------------------------------
// Same per-position arithmetic as dot_f16_f32 / axpy_f16 (results identical),
// without a call and a re-dispatch per position; the value accumulator stays in
// registers across the block.
void attn_scores_f16(const uint16_t* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) {
  for (int64_t t = 0; t < n; ++t) scores[t] = dot_f16_f32(k + t * dim, q, dim) * scale;
}

void attn_accum_f16(const uint16_t* v, int64_t n, int32_t dim, const float* w, float* acc) {
  int32_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 a = _mm256_loadu_ps(acc + i);
    for (int64_t t = 0; t < n; ++t) a = _mm256_fmadd_ps(_mm256_set1_ps(w[t]), load_f16x8(v + t * dim + i), a);
    _mm256_storeu_ps(acc + i, a);
  }
  for (; i < dim; ++i) {
    for (int64_t t = 0; t < n; ++t) acc[i] += w[t] * fp16_to_fp32(v[t * dim + i]);
  }
}

void attn_scores_f32(const float* k, int64_t n, int32_t dim, const float* q, float scale, float* scores) {
  for (int64_t t = 0; t < n; ++t) scores[t] = dot_f32(q, k + t * dim, dim) * scale;
}

void attn_accum_f32(const float* v, int64_t n, int32_t dim, const float* w, float* acc) {
  int32_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 a = _mm256_loadu_ps(acc + i);
    for (int64_t t = 0; t < n; ++t) a = _mm256_fmadd_ps(_mm256_set1_ps(w[t]), _mm256_loadu_ps(v + t * dim + i), a);
    _mm256_storeu_ps(acc + i, a);
  }
  for (; i < dim; ++i) {
    for (int64_t t = 0; t < n; ++t) acc[i] += w[t] * v[t * dim + i];
  }
}

// GQA scores (DD-066): for each position, each 8-wide chunk of k is loaded
// once and multiplied into one accumulator per query head (NH <= 8); the NH
// sums are reduced together by one hadd tree instead of NH horizontal sums.
template <int NH>
void attn_scores_heads_tile(const float* k, int64_t n, int32_t dim, const float* q, float scale, float* scores,
                            int64_t s_stride) {
  const __m256 vs = _mm256_set1_ps(scale);
  for (int64_t t = 0; t < n; ++t) {
    const float* kt = k + t * dim;
    __m256 a[8];
    for (int h = 0; h < 8; ++h) a[h] = _mm256_setzero_ps();
    for (int32_t c = 0; c < dim; c += 8) {
      const __m256 vk = _mm256_loadu_ps(kt + c);
      for (int h = 0; h < NH; ++h) a[h] = _mm256_fmadd_ps(vk, _mm256_loadu_ps(q + h * dim + c), a[h]);
    }
    const __m256 t0 = _mm256_hadd_ps(_mm256_hadd_ps(a[0], a[1]), _mm256_hadd_ps(a[2], a[3]));
    const __m256 t1 = _mm256_hadd_ps(_mm256_hadd_ps(a[4], a[5]), _mm256_hadd_ps(a[6], a[7]));
    const __m256 sum = _mm256_add_ps(_mm256_permute2f128_ps(t0, t1, 0x20), _mm256_permute2f128_ps(t0, t1, 0x31));
    alignas(32) float out[8];
    _mm256_store_ps(out, _mm256_mul_ps(sum, vs));
    for (int h = 0; h < NH; ++h) scores[h * s_stride + t] = out[h];
  }
}

void attn_scores_heads_f32(const float* k, int64_t n, int32_t dim, const float* q, int32_t nh, float scale,
                           float* scores, int64_t s_stride) {
  if (dim % 8 != 0) {
    for (int32_t h = 0; h < nh; ++h) attn_scores_f32(k, n, dim, q + static_cast<int64_t>(h) * dim, scale, scores + h * s_stride);
    return;
  }
  for (int32_t h0 = 0; h0 < nh; h0 += 8) {
    const float* qh = q + static_cast<int64_t>(h0) * dim;
    float* sh = scores + h0 * s_stride;
    switch (std::min(8, nh - h0)) {
      case 1: attn_scores_heads_tile<1>(k, n, dim, qh, scale, sh, s_stride); break;
      case 2: attn_scores_heads_tile<2>(k, n, dim, qh, scale, sh, s_stride); break;
      case 3: attn_scores_heads_tile<3>(k, n, dim, qh, scale, sh, s_stride); break;
      case 4: attn_scores_heads_tile<4>(k, n, dim, qh, scale, sh, s_stride); break;
      case 5: attn_scores_heads_tile<5>(k, n, dim, qh, scale, sh, s_stride); break;
      case 6: attn_scores_heads_tile<6>(k, n, dim, qh, scale, sh, s_stride); break;
      case 7: attn_scores_heads_tile<7>(k, n, dim, qh, scale, sh, s_stride); break;
      default: attn_scores_heads_tile<8>(k, n, dim, qh, scale, sh, s_stride); break;
    }
  }
}

// GQA value accumulation (DD-066): per 8-wide chunk of the head dimension,
// NH accumulators stay in registers over the run and each v vector is loaded
// once for all heads.
template <int NH>
void attn_accum_heads_tile(const float* v, int64_t n, int32_t dim, const float* w, int64_t w_stride, float* acc) {
  for (int32_t c = 0; c < dim; c += 8) {
    __m256 a[NH];
    for (int h = 0; h < NH; ++h) a[h] = _mm256_loadu_ps(acc + h * dim + c);
    for (int64_t t = 0; t < n; ++t) {
      const __m256 vv = _mm256_loadu_ps(v + t * dim + c);
      for (int h = 0; h < NH; ++h) a[h] = _mm256_fmadd_ps(_mm256_broadcast_ss(w + h * w_stride + t), vv, a[h]);
    }
    for (int h = 0; h < NH; ++h) _mm256_storeu_ps(acc + h * dim + c, a[h]);
  }
}

void attn_accum_heads_f32(const float* v, int64_t n, int32_t dim, const float* w, int32_t nh, int64_t w_stride,
                          float* acc) {
  if (dim % 8 != 0) {
    for (int32_t h = 0; h < nh; ++h) attn_accum_f32(v, n, dim, w + h * w_stride, acc + static_cast<int64_t>(h) * dim);
    return;
  }
  for (int32_t h0 = 0; h0 < nh; h0 += 8) {
    const float* wh = w + h0 * w_stride;
    float* ah = acc + static_cast<int64_t>(h0) * dim;
    switch (std::min(8, nh - h0)) {
      case 1: attn_accum_heads_tile<1>(v, n, dim, wh, w_stride, ah); break;
      case 2: attn_accum_heads_tile<2>(v, n, dim, wh, w_stride, ah); break;
      case 3: attn_accum_heads_tile<3>(v, n, dim, wh, w_stride, ah); break;
      case 4: attn_accum_heads_tile<4>(v, n, dim, wh, w_stride, ah); break;
      case 5: attn_accum_heads_tile<5>(v, n, dim, wh, w_stride, ah); break;
      case 6: attn_accum_heads_tile<6>(v, n, dim, wh, w_stride, ah); break;
      case 7: attn_accum_heads_tile<7>(v, n, dim, wh, w_stride, ah); break;
      default: attn_accum_heads_tile<8>(v, n, dim, wh, w_stride, ah); break;
    }
  }
}

// --- int8 activation path (DD-053) ----------------------------------------------
// Weights stay packed; each block is unpacked once to int8 codes and multiplied
// against up to four int8 activation rows held in registers. maddubs does 32
// multiply-adds per instruction (vs 8 for an fp32 FMA), so the single-row case
// stops being compute-bound and the multi-row case reuses every unpacked block.

// Largest |x[i]| over n (multiple of 8) values.
inline float abs_max(const float* x, int64_t n) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  __m256 mx = _mm256_setzero_ps();
  for (int64_t i = 0; i < n; i += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + i)));
  __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
  m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
  m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
  return _mm_cvtss_f32(m4);
}

// One block of 32 values with scale d (codes = round(x / d), |codes| <= 127).
inline void quantize_block(const float* x, float d, ActBlockQ8& out) {
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256 vid = _mm256_set1_ps(d > 0 ? 1.0f / d : 0.0f);
  constexpr int kRound = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  const __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x), vid), kRound));
  const __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 8), vid), kRound));
  const __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 16), vid), kRound));
  const __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 24), vid), kRound));
  // Sum of the codes, before packing.
  const __m256i s = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
  __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
  s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
  // 32 x int32 -> 32 x int8 in element order (packs interleave 128-bit lanes).
  const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.q), _mm256_permutevar8x32_epi32(p, order));
  out.d = d;
  out.sum = _mm_cvtsi128_si32(s4);
}

void quantize_act(const float* x, ActBlockQ8* out, int64_t n) {
  for (int64_t b = 0; b < n / kActQ8Block; ++b, x += kActQ8Block) {
    quantize_block(x, abs_max(x, kActQ8Block) / 127.0f, out[b]);
  }
}

// Super-block activations (DD-075): one scale per 256 values, stored in each
// of the 8 blocks.
void quantize_act_sb(const float* x, ActBlockQ8* out, int64_t n) {
  constexpr int64_t kSb = 256, kPer = kSb / kActQ8Block;
  for (int64_t s = 0; s < n / kSb; ++s, x += kSb, out += kPer) {
    const float d = abs_max(x, kSb) / 127.0f;
    for (int64_t b = 0; b < kPer; ++b) quantize_block(x + b * kActQ8Block, d, out[b]);
  }
}

// 8 int32 partial sums of 32 products: unsigned 8-bit codes x signed int8.
inline __m256i mul_sum_u8_s8(__m256i u, __m256i s) {
  return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
}

// Signed weight codes against M rows: |w| x (x with w's sign).
template <int M>
inline void accumulate_signed(__m256i wq, const ActBlockQ8* const* x, int64_t blk, __m256 scale_w, __m256* acc) {
  const __m256i aw = _mm256_sign_epi8(wq, wq);
  for (int r = 0; r < M; ++r) {
    const ActBlockQ8& a = x[r][blk];
    const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q));
    const __m256i p = mul_sum_u8_s8(aw, _mm256_sign_epi8(xq, wq));
    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_mul_ps(scale_w, _mm256_set1_ps(a.d)), acc[r]);
  }
}

template <int M>
void dot_rows_q8_0_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i) {
    const __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs));
    accumulate_signed<M>(wq, x, i, _mm256_set1_ps(fp16_to_fp32(b[i].d)), acc[i & 1]);
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

template <int M>
void dot_rows_q4_0_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  const __m256i eight = _mm256_set1_epi8(8);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    const __m256i wq = _mm256_sub_epi8(
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), _mm_and_si128(bytes, low4)), eight);
    accumulate_signed<M>(wq, x, i, _mm256_set1_ps(fp16_to_fp32(b[i].d)), acc[i & 1]);
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

template <int M>
void dot_rows_q5_0_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i) {
    uint32_t qh;
    std::memcpy(&qh, b[i].qh, 4);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].qs));
    __m256i v = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), _mm_and_si128(bytes, low4));
    v = _mm256_or_si256(v, _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10)));
    accumulate_signed<M>(_mm256_sub_epi8(v, _mm256_set1_epi8(16)), x, i, _mm256_set1_ps(fp16_to_fp32(b[i].d)), acc[i & 1]);
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

// The 8 6-bit scales (sc) and minimums (mn) of a Q4_K / Q5_K super-block
// (get_scale_min_k4 for j = 0..7) with word operations instead of a per-j loop.
inline void k4_scales_mins(const uint8_t* scales, uint8_t sc[8], uint8_t mn[8]) {
  uint32_t u[3];
  std::memcpy(u, scales, 12);
  constexpr uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
  const uint32_t sc_lo = u[0] & km1, mn_lo = u[1] & km1;
  const uint32_t sc_hi = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
  const uint32_t mn_hi = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
  std::memcpy(sc, &sc_lo, 4);
  std::memcpy(sc + 4, &sc_hi, 4);
  std::memcpy(mn, &mn_lo, 4);
  std::memcpy(mn + 4, &mn_hi, 4);
}

// 8 unsigned bytes times f, as 8 floats.
inline __m256 scaled_u8x8(const uint8_t* v, float f) {
  return _mm256_mul_ps(_mm256_set1_ps(f),
                       _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v)))));
}

// Q4_K: unsigned codes 0..15 per 32-value sub-block, value = d*sc*q - dmin*m.
// Even and odd sub-blocks accumulate separately (two independent FMA and
// correction chains per row): with one chain the dependent FMAs, not the
// arithmetic, bounded single-row decode.
template <int M>
void dot_rows_q4_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc0[M], acc1[M];
  float corr0[M], corr1[M];
  for (int r = 0; r < M; ++r) {
    acc0[r] = acc1[r] = _mm256_setzero_ps();
    corr0[r] = corr1[r] = 0;
  }
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    uint8_t s8[8], m8[8];
    k4_scales_mins(b[i].scales, s8, m8);
    alignas(32) float sc[8], mn[8];
    _mm256_store_ps(sc, scaled_u8x8(s8, fp16_to_fp32(b[i].d)));
    _mm256_store_ps(mn, scaled_u8x8(m8, fp16_to_fp32(b[i].dmin)));
    for (int jp = 0; jp < 4; ++jp) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
      const __m256i lo = _mm256_and_si256(bytes, low4);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
      const int64_t blk = i * 8 + 2 * jp;
      for (int r = 0; r < M; ++r) {
        const ActBlockQ8& a0 = x[r][blk];
        const ActBlockQ8& a1 = x[r][blk + 1];
        const __m256i p0 = mul_sum_u8_s8(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0.q)));
        const __m256i p1 = mul_sum_u8_s8(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1.q)));
        acc0[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(sc[2 * jp] * a0.d), acc0[r]);
        acc1[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(sc[2 * jp + 1] * a1.d), acc1[r]);
        corr0[r] += mn[2 * jp] * a0.d * static_cast<float>(a0.sum);
        corr1[r] += mn[2 * jp + 1] * a1.d * static_cast<float>(a1.sum);
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc0[r], acc1[r])) - (corr0[r] + corr1[r]);
}

// Activation-only terms of M rows of nb blocks (ActBlockQ8 or ActBlockQ16),
// for the multi-row kernels: ad[r][b] = d, ads[r][b] = d * sum (per thread,
// valid until the next call on this thread).
template <int M, typename Act>
void act_terms(const Act* const* x, int64_t nb, const float** ad, const float** ads) {
  thread_local std::vector<float> terms;
  terms.resize(static_cast<size_t>(2 * M * nb));
  for (int r = 0; r < M; ++r) {
    float* d = terms.data() + 2 * r * nb;
    float* ds = d + nb;
    for (int64_t b = 0; b < nb; ++b) {
      d[b] = x[r][b].d;
      ds[b] = x[r][b].d * static_cast<float>(x[r][b].sum);
    }
    ad[r] = d;
    ads[r] = ds;
  }
}

// Multi-row Q4_K (DotQ8RowsNFn): the per-32 arithmetic of dot_rows_q4_K with
// the activation-only terms (block scales d and d * sum) gathered once per
// call, so each super-block takes one vector multiply for its 8 scale
// products and one FMA for the minimum correction instead of 8 + 16 scalar
// operations per row.
template <int M>
void dot_rows_n_q4_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* x,
                       int64_t k, float* const* y, const float* bias) {
  const int64_t nsb = k / kQK_K;
  const float* ad[M];
  const float* ads[M];
  act_terms<M>(x, k / kActQ8Block, ad, ads);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ4_K*>(w + row * w_row_bytes);
    __m256 acc0[M], acc1[M], cor[M];
    for (int r = 0; r < M; ++r) acc0[r] = acc1[r] = cor[r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      uint8_t s8[8], m8[8];
      k4_scales_mins(b[i].scales, s8, m8);
      const __m256 scv = scaled_u8x8(s8, fp16_to_fp32(b[i].d)), mnv = scaled_u8x8(m8, fp16_to_fp32(b[i].dmin));
      alignas(32) float f[M][8];
      for (int r = 0; r < M; ++r) {
        _mm256_store_ps(f[r], _mm256_mul_ps(scv, _mm256_loadu_ps(ad[r] + 8 * i)));
        cor[r] = _mm256_fmadd_ps(mnv, _mm256_loadu_ps(ads[r] + 8 * i), cor[r]);
      }
      for (int jp = 0; jp < 4; ++jp) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
        const __m256i lo = _mm256_and_si256(bytes, low4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
        const int64_t blk = i * 8 + 2 * jp;
        for (int r = 0; r < M; ++r) {
          const __m256i p0 = mul_sum_u8_s8(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][blk].q)));
          const __m256i p1 = mul_sum_u8_s8(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][blk + 1].q)));
          acc0[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_broadcast_ss(&f[r][2 * jp]), acc0[r]);
          acc1[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_broadcast_ss(&f[r][2 * jp + 1]), acc1[r]);
        }
      }
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(_mm256_add_ps(acc0[r], acc1[r])) - hsum(cor[r]) + br;
  }
}

void dot_rows_n_q4_K(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* x, int m,
                     int64_t k, float* const* y, const float* bias) {
  const auto* wb = static_cast<const std::byte*>(w);
  switch (m) {
    case 1: dot_rows_n_q4_K_m<1>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 2: dot_rows_n_q4_K_m<2>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 3: dot_rows_n_q4_K_m<3>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 4: dot_rows_n_q4_K_m<4>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 5: dot_rows_n_q4_K_m<5>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 6: dot_rows_n_q4_K_m<6>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    case 7: dot_rows_n_q4_K_m<7>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
    default: dot_rows_n_q4_K_m<8>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;
  }
}

// Multi-row Q8_0 (DotQ8RowsNFn): dot_rows_q8_0's arithmetic (signed codes,
// one fp16 scale per 32 values) with the activation scales gathered once per
// call; even and odd blocks in two chains. EAGLE-3 heads load as Q8_0.
template <int M>
void dot_rows_n_q8_0_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* x,
                       int64_t k, float* const* y, const float* bias) {
  const int64_t nb = k / kQK;
  const float* ad[M];
  const float* ads[M];
  act_terms<M>(x, nb, ad, ads);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ8_0*>(w + row * w_row_bytes);
    __m256 acc[2][M];
    for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
      const __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs));
      const __m256i aw = _mm256_sign_epi8(wq, wq);
      const float d = fp16_to_fp32(b[i].d);
      for (int r = 0; r < M; ++r) {
        const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][i].q));
        const __m256i p = mul_sum_u8_s8(aw, _mm256_sign_epi8(xq, wq));
        acc[i & 1][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * ad[r][i]), acc[i & 1][r]);
      }
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(_mm256_add_ps(acc[0][r], acc[1][r])) + br;
  }
}

// Multi-row Q6_K (DotQ8RowsNFn): dot_rows_q6_K's arithmetic, with each
// 32-value block's two scale products (d * sc * activation d) formed by two
// vector multiplies per super-block instead of per block and row.
template <int M>
void dot_rows_n_q6_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* x,
                       int64_t k, float* const* y, const float* bias) {
  const int64_t nb = k / kActQ8Block, nsb = k / kQK_K;
  const float* ad[M];
  const float* ads[M];
  act_terms<M>(x, nb, ad, ads);
  // Each block's d twice (its two 16-value scales).
  thread_local std::vector<float> dup;
  dup.resize(static_cast<size_t>(2 * M * nb));
  for (int r = 0; r < M; ++r) {
    for (int64_t b = 0; b < nb; ++b) dup[static_cast<size_t>(2 * (r * nb + b))] = dup[static_cast<size_t>(2 * (r * nb + b) + 1)] = ad[r][b];
  }
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ6_K*>(w + row * w_row_bytes);
    __m256 acc[2][M];
    for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      const __m256 dv = _mm256_set1_ps(fp16_to_fp32(b[i].d));
      const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].scales));
      const __m256 s0 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc8)));
      const __m256 s1 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sc8, 8))));
      alignas(32) float f[M][16];
      for (int r = 0; r < M; ++r) {
        const float* dd = dup.data() + 2 * (r * nb + 8 * i);
        _mm256_store_ps(f[r], _mm256_mul_ps(s0, _mm256_loadu_ps(dd)));
        _mm256_store_ps(f[r] + 8, _mm256_mul_ps(s1, _mm256_loadu_ps(dd + 8)));
      }
      for (int part = 0; part < 2; ++part) {
        const uint8_t* ql = b[i].ql + 64 * part;
        const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
        auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
        const __m256i q[4] = {
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
        for (int g = 0; g < 4; ++g) {
          const int blk = 4 * part + g;
          const __m256i aw = _mm256_sign_epi8(q[g], q[g]);
          for (int r = 0; r < M; ++r) {
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][i * 8 + blk].q));
            const __m256i p = mul_sum_u8_s8(aw, _mm256_sign_epi8(xq, q[g]));
            const __m256 sw = _mm256_set_m128(_mm_broadcast_ss(&f[r][2 * blk + 1]), _mm_broadcast_ss(&f[r][2 * blk]));
            acc[g & 1][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sw, acc[g & 1][r]);
          }
        }
      }
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(_mm256_add_ps(acc[0][r], acc[1][r])) + br;
  }
}

// Q6_K: signed codes -32..31; each 32-value activation block spans two
// 16-value weight scales (lanes 0-3 and 4-7 of the int32 partials).
template <int M>
void dot_rows_q6_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    for (int part = 0; part < 2; ++part) {
      const uint8_t* ql = b[i].ql + 64 * part;
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
      auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
      const __m256i q[4] = {
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
      const int8_t* sc = b[i].scales + 8 * part;
      for (int g = 0; g < 4; ++g) {
        const __m256 scale_w = _mm256_set_m128(_mm_set1_ps(d * sc[2 * g + 1]), _mm_set1_ps(d * sc[2 * g]));
        accumulate_signed<M>(q[g], x, i * 8 + 4 * part + g, scale_w, acc[g & 1]);
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

// --- Super-block activations (DD-075) ---------------------------------------
// With one activation scale per 256 values, the products of a whole K-quant
// super-block are summed in int32, weighted by the integer sub-block scales
// (maddubs -> int16, madd with the scale -> int32), and converted to float
// once per super-block and row. The per-32 path converts and FMAs eight times
// per super-block and row, which made 2-4 row decode compute-bound.

// Q4_K: value = d*sc*q - dmin*m with q in 0..15, sc and m in 0..63.
// |sum| <= 256 * 15 * 127 * 63 < 2^31.
template <int M>
void dot_rows_sb_q4_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc[M];
  float corr[M];
  for (int r = 0; r < M; ++r) {
    acc[r] = _mm256_setzero_ps();
    corr[r] = 0;
  }
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; ++j) get_scale_min_k4(j, b[i].scales, sc[j], mn[j]);
    __m256i sumi[M];
    for (int r = 0; r < M; ++r) sumi[r] = _mm256_setzero_si256();
    for (int jp = 0; jp < 4; ++jp) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
      const __m256i lo = _mm256_and_si256(bytes, low4);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
      const __m256i s0 = _mm256_set1_epi16(sc[2 * jp]), s1 = _mm256_set1_epi16(sc[2 * jp + 1]);
      const int64_t blk = i * 8 + 2 * jp;
      for (int r = 0; r < M; ++r) {
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][blk].q));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][blk + 1].q));
        const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), s0);
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), s1);
        sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p0, p1));
      }
    }
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    for (int r = 0; r < M; ++r) {
      const ActBlockQ8* xb = x[r] + i * 8;
      int32_t msum = 0;
      for (int j = 0; j < 8; ++j) msum += static_cast<int32_t>(mn[j]) * xb[j].sum;
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * xb[0].d), acc[r]);
      corr[r] += dmin * xb[0].d * static_cast<float>(msum);
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(acc[r]) - corr[r];
}

// Q6_K: signed codes -32..31 and signed 8-bit scales per 16 values; the sign
// trick keeps maddubs' first operand unsigned (|q| <= 32).
template <int M>
void dot_rows_sb_q6_K_m(const void* w, const ActBlockQ8* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    __m256i sumi[M];
    for (int r = 0; r < M; ++r) sumi[r] = _mm256_setzero_si256();
    for (int part = 0; part < 2; ++part) {
      const uint8_t* ql = b[i].ql + 64 * part;
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
      auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
      const __m256i q[4] = {
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
      const int8_t* sc = b[i].scales + 8 * part;
      for (int g = 0; g < 4; ++g) {
        // int16 lanes 0-7 hold values 0-15 (scale 2g), lanes 8-15 values 16-31.
        const __m256i scale = _mm256_set_m128i(_mm_set1_epi16(sc[2 * g + 1]), _mm_set1_epi16(sc[2 * g]));
        const __m256i aw = _mm256_sign_epi8(q[g], q[g]);
        const int64_t blk = i * 8 + 4 * part + g;
        for (int r = 0; r < M; ++r) {
          const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][blk].q));
          const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xq, q[g])), scale);
          sumi[r] = _mm256_add_epi32(sumi[r], p);
        }
      }
    }
    const float d = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) {
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * x[r][i * 8].d), acc[r]);
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(acc[r]);
}

// --- Sub-scaled activations (DD-077) -----------------------------------------
// Per 256 values one ActSuperQ8: unit u = amax / (127 * 128), per 32-value
// block the smallest multiplier m in 1..128 with amax_block <= 127 * u * m,
// codes quantized with scale u * m. The kernels multiply the 8 multipliers
// into the integer sub-block scales with one vector multiply per row and
// super-block (sc * m fits int16), pick each sub-block's scale with one
// shuffle, and accumulate the super-block in int32 like the super-block
// kernels. The weight minimums use the precomputed ms[] in one FMA.

// Quantizes 32 values with scale d into q; returns the code sum.
inline int32_t quantize_codes(const float* x, float d, int8_t* q) {
  ActBlockQ8 tmp;
  quantize_block(x, d, tmp);
  std::memcpy(q, tmp.q, 32);
  return tmp.sum;
}

void quantize_act_sx(const float* x, ActBlockQ8* out, int64_t n) {
  constexpr int64_t kSb = 256, kPer = kSb / kActQ8Block;
  auto* sup = reinterpret_cast<ActSuperQ8*>(out);
  for (int64_t s = 0; s < n / kSb; ++s, x += kSb) {
    ActSuperQ8& a = sup[s];
    const float u = abs_max(x, kSb) / (127.0f * kActSxMaxMult);
    a.u = u;
    for (int64_t b = 0; b < kPer; ++b) {
      int32_t m = 1;
      if (u > 0) {
        const float need = abs_max(x + b * kActQ8Block, kActQ8Block) / (127.0f * u);
        m = std::clamp(static_cast<int32_t>(std::ceil(need)), 1, kActSxMaxMult);
      }
      const int32_t sum = quantize_codes(x + b * kActQ8Block, u * static_cast<float>(m), a.q + b * kActQ8Block);
      a.m[b] = static_cast<int16_t>(m);
      a.ms[b] = u * static_cast<float>(m) * static_cast<float>(sum);
    }
    a.pad[0] = a.pad[1] = a.pad[2] = 0;
  }
}

// vpshufb masks: int16 element j of each 128-bit lane, in every position.
alignas(32) constexpr uint8_t kBcast16[8][32] = {
#define DYNACORE_B16(j) {2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, \
                          2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1, 2 * j, 2 * j + 1}
    DYNACORE_B16(0), DYNACORE_B16(1), DYNACORE_B16(2), DYNACORE_B16(3),
    DYNACORE_B16(4), DYNACORE_B16(5), DYNACORE_B16(6), DYNACORE_B16(7)
#undef DYNACORE_B16
};
// Low lane: element 2g; high lane: element 2g + 1 (Q6_K: one 32-value block
// spans two 16-value scales).
alignas(32) constexpr uint8_t kPair16[4][32] = {
#define DYNACORE_P16(g) {4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, 4 * g, 4 * g + 1, \
                          4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3, 4 * g + 2, 4 * g + 3}
    DYNACORE_P16(0), DYNACORE_P16(1), DYNACORE_P16(2), DYNACORE_P16(3)
#undef DYNACORE_P16
};
inline __m256i mask(const uint8_t* m) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(m)); }

// Q4_K: |lane| <= 8 sub-blocks * 2 * 3810 * (63 * 128) < 4.9e8.
template <int M>
void dot_rows_sx_q4_K_m(const void* w, const ActBlockQ8* const* xin, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const ActSuperQ8* x[M];
  for (int r = 0; r < M; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; ++j) get_scale_min_k4(j, b[i].scales, sc[j], mn[j]);
    const __m128i sc16 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sc)));
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const __m256 mnf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mn)))),
                                     _mm256_set1_ps(dmin));
    __m256i scm[M], sumi[M];
    for (int r = 0; r < M; ++r) {
      const ActSuperQ8& a = x[r][i];
      scm[r] = _mm256_broadcastsi128_si256(_mm_mullo_epi16(sc16, _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.m))));
      sumi[r] = _mm256_setzero_si256();
      // Weight minimums: - sum_j dmin * mn_j * (u * m_j * sum_j), folded into acc.
      acc[r] = _mm256_fnmadd_ps(mnf, _mm256_loadu_ps(a.ms), acc[r]);
    }
    for (int jp = 0; jp < 4; ++jp) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
      const __m256i lo = _mm256_and_si256(bytes, low4);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
      const __m256i m0 = mask(kBcast16[2 * jp]), m1 = mask(kBcast16[2 * jp + 1]);
      for (int r = 0; r < M; ++r) {
        const int8_t* q = x[r][i].q + 64 * jp;
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
        const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_shuffle_epi8(scm[r], m0));
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), _mm256_shuffle_epi8(scm[r], m1));
        sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p0, p1));
      }
    }
    for (int r = 0; r < M; ++r) {
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * x[r][i].u), acc[r]);
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(acc[r]);
}

// Q6_K: |sc * m| <= 128 * 128; |lane| <= 8 groups * 2 * 8128 * 16384 < 2.14e9.
template <int M>
void dot_rows_sx_q6_K_m(const void* w, const ActBlockQ8* const* xin, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const ActSuperQ8* x[M];
  for (int r = 0; r < M; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    // 16 signed scales, one per 16 values.
    const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].scales)));
    __m256i scm[M], sumi[M];
    for (int r = 0; r < M; ++r) {
      // m_b for scales 2b and 2b + 1.
      const __m128i m8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[r][i].m));
      const __m256i mm = _mm256_set_m128i(_mm_unpackhi_epi16(m8, m8), _mm_unpacklo_epi16(m8, m8));
      scm[r] = _mm256_mullo_epi16(sc16, mm);
      sumi[r] = _mm256_setzero_si256();
    }
    for (int part = 0; part < 2; ++part) {
      const uint8_t* ql = b[i].ql + 64 * part;
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
      auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
      const __m256i q[4] = {
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
      // This part's 8 scales (16 values each) in both lanes.
      __m256i half[M];
      for (int r = 0; r < M; ++r) {
        half[r] = part == 0 ? _mm256_permute2x128_si256(scm[r], scm[r], 0x00)
                            : _mm256_permute2x128_si256(scm[r], scm[r], 0x11);
      }
      for (int g = 0; g < 4; ++g) {
        const __m256i aw = _mm256_sign_epi8(q[g], q[g]);
        const __m256i pm = mask(kPair16[g]);
        for (int r = 0; r < M; ++r) {
          const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][i].q + 128 * part + 32 * g));
          const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xq, q[g])),
                                              _mm256_shuffle_epi8(half[r], pm));
          sumi[r] = _mm256_add_epi32(sumi[r], p);
        }
      }
    }
    const float d = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) {
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * x[r][i].u), acc[r]);
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(acc[r]);
}

// Multi-row sub-scaled kernels (DotQ8RowsNFn, activations from
// quantize_act_sx): the arithmetic of dot_rows_sx_q4_K / dot_rows_sx_q6_K
// with the weight scales decoded by word operations and the minimum
// correction in its own accumulator, for weight rows [n0, n1) per call.
template <int M>
void dot_rows_n_sx_q4_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* xin,
                          int64_t k, float* const* y, const float* bias) {
  const ActSuperQ8* x[M];
  for (int r = 0; r < M; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  const int64_t nsb = k / kQK_K;
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ4_K*>(w + row * w_row_bytes);
    __m256 acc[M], cor[M];
    for (int r = 0; r < M; ++r) acc[r] = cor[r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      uint8_t s8[8], m8[8];
      k4_scales_mins(b[i].scales, s8, m8);
      const __m128i sc16 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s8)));
      const __m256 mnf = scaled_u8x8(m8, fp16_to_fp32(b[i].dmin));
      const float d = fp16_to_fp32(b[i].d);
      __m256i scm[M], sumi[M];
      for (int r = 0; r < M; ++r) {
        const ActSuperQ8& a = x[r][i];
        scm[r] = _mm256_broadcastsi128_si256(_mm_mullo_epi16(sc16, _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.m))));
        sumi[r] = _mm256_setzero_si256();
        cor[r] = _mm256_fmadd_ps(mnf, _mm256_loadu_ps(a.ms), cor[r]);
      }
      for (int jp = 0; jp < 4; ++jp) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
        const __m256i lo = _mm256_and_si256(bytes, low4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
        const __m256i m0 = mask(kBcast16[2 * jp]), m1 = mask(kBcast16[2 * jp + 1]);
        for (int r = 0; r < M; ++r) {
          const int8_t* q = x[r][i].q + 64 * jp;
          const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
          const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
          const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_shuffle_epi8(scm[r], m0));
          const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), _mm256_shuffle_epi8(scm[r], m1));
          sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p0, p1));
        }
      }
      for (int r = 0; r < M; ++r) acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * x[r][i].u), acc[r]);
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(acc[r]) - hsum(cor[r]) + br;
  }
}

template <int M>
void dot_rows_n_sx_q6_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* xin,
                          int64_t k, float* const* y, const float* bias) {
  const ActSuperQ8* x[M];
  for (int r = 0; r < M; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  const int64_t nsb = k / kQK_K;
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ6_K*>(w + row * w_row_bytes);
    __m256 acc[M];
    for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b[i].scales)));
      __m256i scm[M], sumi[M];
      for (int r = 0; r < M; ++r) {
        const __m128i m8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[r][i].m));
        scm[r] = _mm256_mullo_epi16(sc16, _mm256_set_m128i(_mm_unpackhi_epi16(m8, m8), _mm_unpacklo_epi16(m8, m8)));
        sumi[r] = _mm256_setzero_si256();
      }
      for (int part = 0; part < 2; ++part) {
        const uint8_t* ql = b[i].ql + 64 * part;
        const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
        auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
        const __m256i q[4] = {
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
        __m256i half[M];
        for (int r = 0; r < M; ++r) {
          half[r] = part == 0 ? _mm256_permute2x128_si256(scm[r], scm[r], 0x00) : _mm256_permute2x128_si256(scm[r], scm[r], 0x11);
        }
        for (int g = 0; g < 4; ++g) {
          const __m256i aw = _mm256_sign_epi8(q[g], q[g]);
          const __m256i pm = mask(kPair16[g]);
          for (int r = 0; r < M; ++r) {
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r][i].q + 128 * part + 32 * g));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xq, q[g])), _mm256_shuffle_epi8(half[r], pm));
            sumi[r] = _mm256_add_epi32(sumi[r], p);
          }
        }
      }
      const float d = fp16_to_fp32(b[i].d);
      for (int r = 0; r < M; ++r) acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r]), _mm256_set1_ps(d * x[r][i].u), acc[r]);
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(acc[r]) + br;
  }
}

#define DYNACORE_DOT_ROWS_N_SX(name)                                                                                \
  void name(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ8* const* x, int m, int64_t k,\
            float* const* y, const float* bias) {                                                                  \
    const auto* wb = static_cast<const std::byte*>(w);                                                             \
    switch (m) {                                                                                                   \
      case 1: name##_m<1>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 2: name##_m<2>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 3: name##_m<3>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 4: name##_m<4>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 5: name##_m<5>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 6: name##_m<6>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 7: name##_m<7>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      default: name##_m<8>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                         \
    }                                                                                                              \
  }
DYNACORE_DOT_ROWS_N_SX(dot_rows_n_sx_q4_K)
DYNACORE_DOT_ROWS_N_SX(dot_rows_n_sx_q6_K)
#undef DYNACORE_DOT_ROWS_N_SX

// --- Interleaved Q4_K x 8 rows (DD-078) --------------------------------------
// One 32-byte maddubs covers 4 values of 8 weight rows against a broadcast of
// the activation's 4 bytes; 8 of them (32 values) are summed in int16
// (|lane| <= 8 * 2 * 15 * 127 = 30480), then one madd applies the 8 rows'
// sub-block scale times the activation block's multiplier (sc * m <= 8064, in
// int16), so each int32 lane is one weight row. Half a super-block (4
// sub-blocks) accumulates exactly in int32 (|lane| <= 4 * 2 * 30480 * 8064 <
// 1.97e9) before one conversion: a Q4_K dot is a difference of two large,
// nearly cancelling terms (scaled codes and minimums), so rounding each
// sub-block separately measurably hurt accuracy (DD-078 log).
template <int M>
void dot_x8_sx_q4_K_m(const BlockQ4_Kx8* w, const ActSuperQ8* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ4_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d), dminv = _mm256_loadu_ps(p.dmin);
    __m256i sumi[M];
    __m256 sumf[M];
    for (int r = 0; r < M; ++r) {
      sumi[r] = _mm256_setzero_si256();
      sumf[r] = _mm256_setzero_ps();
    }
    for (int jp = 0; jp < 4; ++jp) {
      __m256i wl[8], wh[8];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.qs[jp][c]));
        wl[c] = _mm256_and_si256(q, low4);
        wh[c] = _mm256_and_si256(_mm256_srli_epi16(q, 4), low4);
      }
      const int j0 = 2 * jp, j1 = 2 * jp + 1;
      auto dup_scales = [&](int j) {
        const __m128i s = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j]));
        return _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(s, s));
      };
      auto min_terms = [&](int j) {  // dmin * mn per row, as the unpacked kernel forms it
        return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j])))),
                             dminv);
      };
      const __m256i s0 = dup_scales(j0), s1 = dup_scales(j1);
      const __m256 mn0 = min_terms(j0), mn1 = min_terms(j1);
      for (int r = 0; r < M; ++r) {
        const ActSuperQ8& a = x[r][i];
        const int8_t* x0 = a.q + 32 * j0;
        const int8_t* x1 = a.q + 32 * j1;
        __m256i sa = _mm256_setzero_si256(), sb = _mm256_setzero_si256();
        for (int c = 0; c < 8; ++c) {
          int32_t b0, b1;
          std::memcpy(&b0, x0 + 4 * c, 4);
          std::memcpy(&b1, x1 + 4 * c, 4);
          sa = _mm256_add_epi16(sa, _mm256_maddubs_epi16(wl[c], _mm256_set1_epi32(b0)));
          sb = _mm256_add_epi16(sb, _mm256_maddubs_epi16(wh[c], _mm256_set1_epi32(b1)));
        }
        const __m256i pa = _mm256_madd_epi16(sa, _mm256_mullo_epi16(s0, _mm256_set1_epi16(a.m[j0])));
        const __m256i pb = _mm256_madd_epi16(sb, _mm256_mullo_epi16(s1, _mm256_set1_epi16(a.m[j1])));
        sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(pa, pb));
        acc[r] = _mm256_fnmadd_ps(mn0, _mm256_set1_ps(a.ms[j0]), acc[r]);
        acc[r] = _mm256_fnmadd_ps(mn1, _mm256_set1_ps(a.ms[j1]), acc[r]);
      }
      if (jp == 1 || jp == 3) {  // end of a half super-block: exact integer -> float
        for (int r = 0; r < M; ++r) {
          sumf[r] = _mm256_add_ps(sumf[r], _mm256_cvtepi32_ps(sumi[r]));
          sumi[r] = _mm256_setzero_si256();
        }
      }
    }
    for (int r = 0; r < M; ++r) {
      acc[r] = _mm256_fmadd_ps(sumf[r], _mm256_mul_ps(dv, _mm256_set1_ps(x[r][i].u)), acc[r]);
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

void dot_x8_sx_q4_K(const void* w, const ActBlockQ8* const* xin, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  const ActSuperQ8* x[kDotRowsMax];
  for (int r = 0; r < m; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  switch (m) {
    case 1: dot_x8_sx_q4_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_sx_q4_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_sx_q4_K_m<3>(pw, x, n, out); break;
    default: dot_x8_sx_q4_K_m<4>(pw, x, n, out); break;
  }
}

// Experimental 8-row entry (KernelPlan::packed_int8_8rows): the same template
// for up to kPackedRowsMax rows, so 5..8 activation rows share one pass over
// each packed weight group. Per-row arithmetic is unchanged.
void dot_x8_sx_q4_K_rows8(const void* w, const ActBlockQ8* const* xin, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  const ActSuperQ8* x[kPackedRowsMax];
  for (int r = 0; r < m; ++r) x[r] = reinterpret_cast<const ActSuperQ8*>(xin[r]);
  switch (m) {
    case 1: dot_x8_sx_q4_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_sx_q4_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_sx_q4_K_m<3>(pw, x, n, out); break;
    case 4: dot_x8_sx_q4_K_m<4>(pw, x, n, out); break;
    case 5: dot_x8_sx_q4_K_m<5>(pw, x, n, out); break;
    case 6: dot_x8_sx_q4_K_m<6>(pw, x, n, out); break;
    case 7: dot_x8_sx_q4_K_m<7>(pw, x, n, out); break;
    default: dot_x8_sx_q4_K_m<8>(pw, x, n, out); break;
  }
}

// --- Interleaved Q4_K x 8 rows with int16 activations (DD-081) --------------
// The FFN down projection keeps int16 activations (int8 breaks the accuracy
// contract there, DD-053/DD-076). On the BlockQ4_Kx8 layout each 4-value chunk
// of 8 rows widens to two int16 vectors (rows 0-3, rows 4-7) shared by every
// activation row; one 64-bit broadcast of the activation's 4 values feeds two
// madd. A 32-value sub-block accumulates exactly in int32 (|lane| <= 8 * 2 *
// 15 * 32767 < 7.9e6, exact in float after the pair add), then one hadd +
// permute puts the 8 rows in order for one float multiply-add with the
// sub-block scale and the activation block's scale.
template <int M>
void dot_x8_q16_q4_K_m(const BlockQ4_Kx8* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  const __m256i row_order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ4_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d), dminv = _mm256_loadu_ps(p.dmin);
    for (int jp = 0; jp < 4; ++jp) {
      // Widened codes: [chunk][0: rows 0-3, 1: rows 4-7] for both sub-blocks.
      __m256i wl[8][2], wh[8][2];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.qs[jp][c]));
        const __m256i lo = _mm256_and_si256(q, low4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), low4);
        wl[c][0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo));
        wl[c][1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1));
        wh[c][0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi));
        wh[c][1] = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1));
      }
      const int j0 = 2 * jp, j1 = 2 * jp + 1;
      auto scale_f = [&](int j) {
        return _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[j])))));
      };
      auto min_f = [&](int j) {
        return _mm256_mul_ps(dminv, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.mn[j])))));
      };
      const __m256 ds0 = scale_f(j0), ds1 = scale_f(j1), dm0 = min_f(j0), dm1 = min_f(j1);
      for (int r = 0; r < M; ++r) {
        const ActBlockQ16& a0 = x[r][i * 8 + j0];
        const ActBlockQ16& a1 = x[r][i * 8 + j1];
        __m256i s0a = _mm256_setzero_si256(), s0b = _mm256_setzero_si256();
        __m256i s1a = _mm256_setzero_si256(), s1b = _mm256_setzero_si256();
        for (int c = 0; c < 8; ++c) {
          int64_t b0, b1;
          std::memcpy(&b0, a0.q + 4 * c, 8);
          std::memcpy(&b1, a1.q + 4 * c, 8);
          const __m256i x0 = _mm256_set1_epi64x(b0), x1 = _mm256_set1_epi64x(b1);
          s0a = _mm256_add_epi32(s0a, _mm256_madd_epi16(wl[c][0], x0));
          s0b = _mm256_add_epi32(s0b, _mm256_madd_epi16(wl[c][1], x0));
          s1a = _mm256_add_epi32(s1a, _mm256_madd_epi16(wh[c][0], x1));
          s1b = _mm256_add_epi32(s1b, _mm256_madd_epi16(wh[c][1], x1));
        }
        const __m256i r0 = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(s0a, s0b), row_order);
        const __m256i r1 = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(s1a, s1b), row_order);
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r0), _mm256_mul_ps(ds0, _mm256_set1_ps(a0.d)), acc[r]);
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r1), _mm256_mul_ps(ds1, _mm256_set1_ps(a1.d)), acc[r]);
        acc[r] = _mm256_fnmadd_ps(dm0, _mm256_set1_ps(a0.d * static_cast<float>(a0.sum)), acc[r]);
        acc[r] = _mm256_fnmadd_ps(dm1, _mm256_set1_ps(a1.d * static_cast<float>(a1.sum)), acc[r]);
      }
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

void dot_x8_q16_q4_K(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q4_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q4_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q4_K_m<3>(pw, x, n, out); break;
    default: dot_x8_q16_q4_K_m<4>(pw, x, n, out); break;
  }
}

// --- Interleaved Q6_K x 8 rows with int16 activations (DD-082) --------------
// Per 4-value chunk of a sub-block pair: 32 bytes of low nibbles (sub-block A
// low, B high) and 16 bytes of 2-bit high planes, rebuilt to signed codes
// (q - 32) once and widened to int16 for every activation row. Each 16-value
// half of a sub-block (one Q6_K scale) accumulates exactly in int32 (|lane|
// after the pair add <= 2 * 4 * 2 * 32 * 32767 < 2^24, exact in float) and
// takes one float multiply-add with d * scale * the activation block scale.
template <int M>
void dot_x8_q16_q6_K_m(const BlockQ6_Kx8* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const __m256i low4 = _mm256_set1_epi8(0x0F), three = _mm256_set1_epi8(3), k32 = _mm256_set1_epi8(32);
  const __m128i low4h = _mm_set1_epi8(0x0F);
  const __m256i row_order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
  __m256 acc[M];
  for (int r = 0; r < M; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const BlockQ6_Kx8& p = w[i];
    const __m256 dv = _mm256_loadu_ps(p.d);
    for (int pp = 0; pp < 4; ++pp) {
      // Signed codes widened to int16: [chunk][A rows 0-3, A rows 4-7, B rows 0-3, B rows 4-7].
      __m256i wq[8][4];
      for (int c = 0; c < 8; ++c) {
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.ql[pp][c]));
        const __m128i hb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p.qh[pp][c]));
        const __m256i h = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(hb, 4), low4h), _mm_and_si128(hb, low4h));
        const __m256i a = _mm256_sub_epi8(
            _mm256_or_si256(_mm256_and_si256(q, low4), _mm256_slli_epi16(_mm256_and_si256(h, three), 4)), k32);
        const __m256i b = _mm256_sub_epi8(
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), low4),
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), three), 4)),
            k32);
        wq[c][0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        wq[c][1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        wq[c][2] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
        wq[c][3] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
      }
      // d * scale for the 4 sixteen-value groups of this pair (A low, A high, B low, B high).
      __m256 ds[4];
      for (int g = 0; g < 4; ++g) {
        ds[g] = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p.sc[4 * pp + g])))));
      }
      for (int r = 0; r < M; ++r) {
        for (int s = 0; s < 2; ++s) {  // sub-block A (s = 0) or B (s = 1)
          const ActBlockQ16& a = x[r][i * 8 + 2 * pp + s];
          __m256i lo_a = _mm256_setzero_si256(), lo_b = _mm256_setzero_si256();
          __m256i hi_a = _mm256_setzero_si256(), hi_b = _mm256_setzero_si256();
          for (int c = 0; c < 4; ++c) {
            int64_t v0, v1;
            std::memcpy(&v0, a.q + 4 * c, 8);
            std::memcpy(&v1, a.q + 16 + 4 * c, 8);
            const __m256i x0 = _mm256_set1_epi64x(v0), x1 = _mm256_set1_epi64x(v1);
            lo_a = _mm256_add_epi32(lo_a, _mm256_madd_epi16(wq[c][2 * s], x0));
            lo_b = _mm256_add_epi32(lo_b, _mm256_madd_epi16(wq[c][2 * s + 1], x0));
            hi_a = _mm256_add_epi32(hi_a, _mm256_madd_epi16(wq[c + 4][2 * s], x1));
            hi_b = _mm256_add_epi32(hi_b, _mm256_madd_epi16(wq[c + 4][2 * s + 1], x1));
          }
          const __m256i lo = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(lo_a, lo_b), row_order);
          const __m256i hi = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(hi_a, hi_b), row_order);
          const __m256 xd = _mm256_set1_ps(a.d);
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo), _mm256_mul_ps(ds[2 * s], xd), acc[r]);
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi), _mm256_mul_ps(ds[2 * s + 1], xd), acc[r]);
        }
      }
    }
  }
  for (int r = 0; r < M; ++r) _mm256_storeu_ps(out + 8 * r, acc[r]);
}

void dot_x8_q16_q6_K(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ6_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q6_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q6_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q6_K_m<3>(pw, x, n, out); break;
    default: dot_x8_q16_q6_K_m<4>(pw, x, n, out); break;
  }
}

// 8-row tiles of the two int16 x8 kernels (KernelPlan::int16_prefill_max_rows):
// up to kPackedRowsMax activation rows per pass over a packed group.
void dot_x8_q16_q4_K_rows8(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ4_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q4_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q4_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q4_K_m<3>(pw, x, n, out); break;
    case 4: dot_x8_q16_q4_K_m<4>(pw, x, n, out); break;
    case 5: dot_x8_q16_q4_K_m<5>(pw, x, n, out); break;
    case 6: dot_x8_q16_q4_K_m<6>(pw, x, n, out); break;
    case 7: dot_x8_q16_q4_K_m<7>(pw, x, n, out); break;
    default: dot_x8_q16_q4_K_m<8>(pw, x, n, out); break;
  }
}

void dot_x8_q16_q6_K_rows8(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {
  const auto* pw = static_cast<const BlockQ6_Kx8*>(w);
  switch (m) {
    case 1: dot_x8_q16_q6_K_m<1>(pw, x, n, out); break;
    case 2: dot_x8_q16_q6_K_m<2>(pw, x, n, out); break;
    case 3: dot_x8_q16_q6_K_m<3>(pw, x, n, out); break;
    case 4: dot_x8_q16_q6_K_m<4>(pw, x, n, out); break;
    case 5: dot_x8_q16_q6_K_m<5>(pw, x, n, out); break;
    case 6: dot_x8_q16_q6_K_m<6>(pw, x, n, out); break;
    case 7: dot_x8_q16_q6_K_m<7>(pw, x, n, out); break;
    default: dot_x8_q16_q6_K_m<8>(pw, x, n, out); break;
  }
}

// --- int16 activation path (DD-076) -----------------------------------------
// Weight codes are widened to int16 and pre-multiplied by their integer
// sub-block scale (|q * sc| <= 4096 for every format here), so each 32-value
// block is two madd_epi16, one add, one convert and one FMA per row. int32
// lanes hold at most 2 * 2 * 4096 * 32767 < 2^31.

void quantize_act16(const float* x, ActBlockQ16* out, int64_t n) {
  constexpr int kRound = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  for (int64_t b = 0; b < n / kActQ8Block; ++b, x += kActQ8Block) {
    const float d = abs_max(x, kActQ8Block) / 32767.0f;
    const __m256 vid = _mm256_set1_ps(d > 0 ? 1.0f / d : 0.0f);
    const __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x), vid), kRound));
    const __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 8), vid), kRound));
    const __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 16), vid), kRound));
    const __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x + 24), vid), kRound));
    const __m256i s = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    // packs interleaves 128-bit lanes; 0xD8 restores element order.
    const __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
    const __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out[b].q), p01);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out[b].q + 16), p23);
    out[b].d = d;
    out[b].sum = _mm_cvtsi128_si32(s4);
  }
}

// Sum of products of 32 int16 weights (w0: values 0-15, w1: 16-31) and one
// activation block, as 8 int32 lanes.
inline __m256i madd16_block(__m256i w0, __m256i w1, const ActBlockQ16& a) {
  return _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q))),
                          _mm256_madd_epi16(w1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q + 16))));
}

template <int M>
void dot_rows16_q8_0_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK; ++i) {
    const __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs));
    const __m256i w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wq));
    const __m256i w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wq, 1));
    const float d = fp16_to_fp32(b[i].d);
    for (int r = 0; r < M; ++r) {
      const ActBlockQ16& a = x[r][i];
      acc[i & 1][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(w0, w1, a)), _mm256_set1_ps(d * a.d), acc[i & 1][r]);
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

template <int M>
void dot_rows16_q4_K_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256 acc0[M], acc1[M];
  float corr0[M], corr1[M];
  for (int r = 0; r < M; ++r) {
    acc0[r] = acc1[r] = _mm256_setzero_ps();
    corr0[r] = corr1[r] = 0;
  }
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    uint8_t sc[8], mn[8];
    k4_scales_mins(b[i].scales, sc, mn);
    for (int jp = 0; jp < 4; ++jp) {
      const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
      const __m256i lo = _mm256_and_si256(bytes, low4);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
      const __m256i s0 = _mm256_set1_epi16(sc[2 * jp]), s1 = _mm256_set1_epi16(sc[2 * jp + 1]);
      const __m256i l0 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo)), s0);
      const __m256i l1 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1)), s0);
      const __m256i h0 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi)), s1);
      const __m256i h1 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1)), s1);
      const int64_t blk = i * 8 + 2 * jp;
      for (int r = 0; r < M; ++r) {
        const ActBlockQ16& a0 = x[r][blk];
        const ActBlockQ16& a1 = x[r][blk + 1];
        acc0[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(l0, l1, a0)), _mm256_set1_ps(d * a0.d), acc0[r]);
        acc1[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(h0, h1, a1)), _mm256_set1_ps(d * a1.d), acc1[r]);
        corr0[r] += dmin * (mn[2 * jp] * a0.d * static_cast<float>(a0.sum));
        corr1[r] += dmin * (mn[2 * jp + 1] * a1.d * static_cast<float>(a1.sum));
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc0[r], acc1[r])) - (corr0[r] + corr1[r]);
}

template <int M>
void dot_rows16_q6_K_m(const void* w, const ActBlockQ16* const* x, int64_t n, float* out) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  __m256 acc[2][M];
  for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    for (int part = 0; part < 2; ++part) {
      const uint8_t* ql = b[i].ql + 64 * part;
      const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
      auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
      const __m256i q[4] = {
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
          _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
      const int8_t* sc = b[i].scales + 8 * part;
      for (int g = 0; g < 4; ++g) {
        const __m256i w0 =
            _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q[g])), _mm256_set1_epi16(sc[2 * g]));
        const __m256i w1 =
            _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(q[g], 1)), _mm256_set1_epi16(sc[2 * g + 1]));
        const int64_t blk = i * 8 + 4 * part + g;
        for (int r = 0; r < M; ++r) {
          const ActBlockQ16& a = x[r][blk];
          acc[g & 1][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(w0, w1, a)), _mm256_set1_ps(d * a.d), acc[g & 1][r]);
        }
      }
    }
  }
  for (int r = 0; r < M; ++r) out[r] = hsum(_mm256_add_ps(acc[0][r], acc[1][r]));
}

// Multi-row int16 Q4_K (DotQ16RowsNFn): dot_rows16_q4_K's arithmetic with
// the activation terms prepared once per call (see dot_rows_n_q4_K_m).
template <int M>
void dot_rows16_n_q4_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ16* const* x,
                         int64_t k, float* const* y, const float* bias) {
  const int64_t nsb = k / kQK_K;
  const float* ad[M];
  const float* ads[M];
  act_terms<M>(x, k / kActQ8Block, ad, ads);
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ4_K*>(w + row * w_row_bytes);
    __m256 acc0[M], acc1[M], cor[M];
    for (int r = 0; r < M; ++r) acc0[r] = acc1[r] = cor[r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      uint8_t sc[8], mn[8];
      k4_scales_mins(b[i].scales, sc, mn);
      const __m256 dv = _mm256_set1_ps(fp16_to_fp32(b[i].d));
      const __m256 mnv = scaled_u8x8(mn, fp16_to_fp32(b[i].dmin));
      alignas(32) float f[M][8];
      for (int r = 0; r < M; ++r) {
        _mm256_store_ps(f[r], _mm256_mul_ps(dv, _mm256_loadu_ps(ad[r] + 8 * i)));
        cor[r] = _mm256_fmadd_ps(mnv, _mm256_loadu_ps(ads[r] + 8 * i), cor[r]);
      }
      for (int jp = 0; jp < 4; ++jp) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs + 32 * jp));
        const __m256i lo = _mm256_and_si256(bytes, low4);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), low4);
        const __m256i s0 = _mm256_set1_epi16(sc[2 * jp]), s1 = _mm256_set1_epi16(sc[2 * jp + 1]);
        const __m256i l0 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo)), s0);
        const __m256i l1 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(lo, 1)), s0);
        const __m256i h0 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi)), s1);
        const __m256i h1 = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(hi, 1)), s1);
        const int64_t blk = i * 8 + 2 * jp;
        for (int r = 0; r < M; ++r) {
          acc0[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(l0, l1, x[r][blk])), _mm256_broadcast_ss(&f[r][2 * jp]), acc0[r]);
          acc1[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(h0, h1, x[r][blk + 1])), _mm256_broadcast_ss(&f[r][2 * jp + 1]),
                                    acc1[r]);
        }
      }
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(_mm256_add_ps(acc0[r], acc1[r])) - hsum(cor[r]) + br;
  }
}

// Multi-row int16 Q6_K (DotQ16RowsNFn): dot_rows16_q6_K's arithmetic, d times
// the activation scales formed once per super-block and row.
template <int M>
void dot_rows16_n_q6_K_m(const std::byte* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const ActBlockQ16* const* x,
                         int64_t k, float* const* y, const float* bias) {
  const int64_t nsb = k / kQK_K;
  const float* ad[M];
  const float* ads[M];
  act_terms<M>(x, k / kActQ8Block, ad, ads);
  const __m256i low4 = _mm256_set1_epi8(0x0F), low2 = _mm256_set1_epi8(0x03), k32 = _mm256_set1_epi8(32);
  for (int64_t row = n0; row < n1; ++row) {
    const auto* b = reinterpret_cast<const BlockQ6_K*>(w + row * w_row_bytes);
    __m256 acc[2][M];
    for (int r = 0; r < M; ++r) acc[0][r] = acc[1][r] = _mm256_setzero_ps();
    for (int64_t i = 0; i < nsb; ++i) {
      const __m256 dv = _mm256_set1_ps(fp16_to_fp32(b[i].d));
      alignas(32) float f[M][8];
      for (int r = 0; r < M; ++r) _mm256_store_ps(f[r], _mm256_mul_ps(dv, _mm256_loadu_ps(ad[r] + 8 * i)));
      for (int part = 0; part < 2; ++part) {
        const uint8_t* ql = b[i].ql + 64 * part;
        const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qh + 32 * part));
        auto hbits = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), low2), 4); };
        const __m256i q[4] = {
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, low4), hbits(0)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, low4), hbits(2)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), low4), hbits(4)), k32),
            _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), low4), hbits(6)), k32)};
        const int8_t* sc = b[i].scales + 8 * part;
        for (int g = 0; g < 4; ++g) {
          const __m256i w0 =
              _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(q[g])), _mm256_set1_epi16(sc[2 * g]));
          const __m256i w1 =
              _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(q[g], 1)), _mm256_set1_epi16(sc[2 * g + 1]));
          const int blk = 4 * part + g;
          for (int r = 0; r < M; ++r) {
            acc[g & 1][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(madd16_block(w0, w1, x[r][i * 8 + blk])),
                                            _mm256_broadcast_ss(&f[r][blk]), acc[g & 1][r]);
          }
        }
      }
    }
    const float br = bias ? bias[row] : 0.0f;
    for (int r = 0; r < M; ++r) y[r][row - n0] = hsum(_mm256_add_ps(acc[0][r], acc[1][r])) + br;
  }
}

#define DYNACORE_DOT_ROWS_N(name, Act)                                                                              \
  void name(const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Act* const* x, int m, int64_t k,     \
            float* const* y, const float* bias) {                                                                  \
    const auto* wb = static_cast<const std::byte*>(w);                                                             \
    switch (m) {                                                                                                   \
      case 1: name##_m<1>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 2: name##_m<2>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 3: name##_m<3>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 4: name##_m<4>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 5: name##_m<5>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 6: name##_m<6>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      case 7: name##_m<7>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                          \
      default: name##_m<8>(wb, w_row_bytes, n0, n1, x, k, y, bias); break;                                         \
    }                                                                                                              \
  }
DYNACORE_DOT_ROWS_N(dot_rows_n_q6_K, ActBlockQ8)
DYNACORE_DOT_ROWS_N(dot_rows_n_q8_0, ActBlockQ8)
DYNACORE_DOT_ROWS_N(dot_rows16_n_q4_K, ActBlockQ16)
DYNACORE_DOT_ROWS_N(dot_rows16_n_q6_K, ActBlockQ16)
#undef DYNACORE_DOT_ROWS_N

#define DYNACORE_DOT_ROWS16(name)                                                                     \
  void name(const void* w, const ActBlockQ16* const* x, int m, int64_t n, float* out) {            \
    switch (m) {                                                                                    \
      case 1: name##_m<1>(w, x, n, out); break;                                                     \
      case 2: name##_m<2>(w, x, n, out); break;                                                     \
      case 3: name##_m<3>(w, x, n, out); break;                                                     \
      default: name##_m<4>(w, x, n, out); break;                                                    \
    }                                                                                               \
  }
DYNACORE_DOT_ROWS16(dot_rows16_q8_0)
DYNACORE_DOT_ROWS16(dot_rows16_q4_K)
DYNACORE_DOT_ROWS16(dot_rows16_q6_K)
#undef DYNACORE_DOT_ROWS16

#define DYNACORE_DOT_ROWS(name)                                                                       \
  void name(const void* w, const ActBlockQ8* const* x, int m, int64_t n, float* out) {             \
    switch (m) {                                                                                    \
      case 1: name##_m<1>(w, x, n, out); break;                                                     \
      case 2: name##_m<2>(w, x, n, out); break;                                                     \
      case 3: name##_m<3>(w, x, n, out); break;                                                     \
      default: name##_m<4>(w, x, n, out); break;                                                    \
    }                                                                                               \
  }
DYNACORE_DOT_ROWS(dot_rows_q8_0)
DYNACORE_DOT_ROWS(dot_rows_q4_0)
DYNACORE_DOT_ROWS(dot_rows_q5_0)
DYNACORE_DOT_ROWS(dot_rows_q4_K)
DYNACORE_DOT_ROWS(dot_rows_q6_K)
DYNACORE_DOT_ROWS(dot_rows_sb_q4_K)
DYNACORE_DOT_ROWS(dot_rows_sb_q6_K)
DYNACORE_DOT_ROWS(dot_rows_sx_q4_K)
DYNACORE_DOT_ROWS(dot_rows_sx_q6_K)
#undef DYNACORE_DOT_ROWS

}  // namespace

bool register_avx2_kernels(CpuKernels& k) {
  k.int8_simd = true;
  k.isa = CpuIsa::kAvx2;
  k.dot_f32 = dot_f32;
  k.axpy_f32 = axpy_f32;
  k.dot_f16_f32 = dot_f16_f32;
  k.axpy_f16 = axpy_f16;
  k.gemm_panel = gemm_panel;
  k.attn_scores_f16 = attn_scores_f16;
  k.attn_scores_heads_f32 = attn_scores_heads_f32;
  k.attn_accum_heads_f32 = attn_accum_heads_f32;
  k.attn_accum_f16 = attn_accum_f16;
  k.attn_scores_f32 = attn_scores_f32;
  k.attn_accum_f32 = attn_accum_f32;
  k.vec_dot[static_cast<size_t>(DType::kF32)] = vec_dot_f32;
  k.vec_dot[static_cast<size_t>(DType::kF16)] = vec_dot_f16;
  k.vec_dot[static_cast<size_t>(DType::kBF16)] = vec_dot_bf16;
  k.vec_dot[static_cast<size_t>(DType::kQ8_0)] = vec_dot_q8_0;
  k.vec_dot[static_cast<size_t>(DType::kQ4_0)] = vec_dot_q4_0;
  k.vec_dot[static_cast<size_t>(DType::kQ5_0)] = vec_dot_q5_0;
  k.vec_dot[static_cast<size_t>(DType::kQ4_K)] = vec_dot_q4_K;
  k.vec_dot[static_cast<size_t>(DType::kQ6_K)] = vec_dot_q6_K;
  k.dequant[static_cast<size_t>(DType::kF16)] = dequant_f16;
  k.dequant[static_cast<size_t>(DType::kBF16)] = dequant_bf16;
  k.dequant[static_cast<size_t>(DType::kQ8_0)] = dequant_q8_0;
  k.dequant[static_cast<size_t>(DType::kQ4_0)] = dequant_q4_0;
  k.dequant[static_cast<size_t>(DType::kQ4_1)] = dequant_q4_1;
  k.vec_dot[static_cast<size_t>(DType::kQ4_1)] = vec_dot_q4_1;
  k.dequant[static_cast<size_t>(DType::kQ5_0)] = dequant_q5_0;
  k.dequant[static_cast<size_t>(DType::kQ4_K)] = dequant_q4_K;
  k.dequant[static_cast<size_t>(DType::kQ6_K)] = dequant_q6_K;
  k.quantize_act = quantize_act;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ8_0)] = dot_rows_q8_0;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ4_0)] = dot_rows_q4_0;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ5_0)] = dot_rows_q5_0;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows_q4_K;
  k.dot_q8_rows_n[static_cast<size_t>(DType::kQ4_K)] = dot_rows_n_q4_K;
  k.dot_q8_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows_q6_K;
  k.dot_q8_rows_n[static_cast<size_t>(DType::kQ6_K)] = dot_rows_n_q6_K;
  k.dot_q8_rows_n[static_cast<size_t>(DType::kQ8_0)] = dot_rows_n_q8_0;
  k.quantize_act_sb = quantize_act_sb;
  k.dot_q8_sb_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows_sb_q4_K;
  k.dot_q8_sb_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows_sb_q6_K;
  k.quantize_act_sx = quantize_act_sx;
  k.dot_q8_sx_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows_sx_q4_K;
  k.dot_q8_sx_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows_sx_q6_K;
  k.dot_q8_sx_rows_n[static_cast<size_t>(DType::kQ4_K)] = dot_rows_n_sx_q4_K;
  k.dot_q8_sx_rows_n[static_cast<size_t>(DType::kQ6_K)] = dot_rows_n_sx_q6_K;
  k.dot_q8_sx_x8_q4_K = dot_x8_sx_q4_K;
  k.dot_q8_sx_x8_q4_K_rows8 = dot_x8_sx_q4_K_rows8;  // used only with KernelPlan::packed_int8_8rows
  k.dot_q16_x8_q4_K = dot_x8_q16_q4_K;
  k.dot_q16_x8_q6_K = dot_x8_q16_q6_K;
  k.dot_q16_x8_q4_K_rows8 = dot_x8_q16_q4_K_rows8;
  k.dot_q16_x8_q6_K_rows8 = dot_x8_q16_q6_K_rows8;
  k.quantize_act16 = quantize_act16;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ8_0)] = dot_rows16_q8_0;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ4_K)] = dot_rows16_q4_K;
  k.dot_q16_rows[static_cast<size_t>(DType::kQ6_K)] = dot_rows16_q6_K;
  k.dot_q16_rows_n[static_cast<size_t>(DType::kQ4_K)] = dot_rows16_n_q4_K;
  k.dot_q16_rows_n[static_cast<size_t>(DType::kQ6_K)] = dot_rows16_n_q6_K;
  return true;
}

}  // namespace dynacore

#else  // AVX2 tier not compiled

namespace dynacore {
bool register_avx2_kernels(CpuKernels&) { return false; }
}  // namespace dynacore

#endif

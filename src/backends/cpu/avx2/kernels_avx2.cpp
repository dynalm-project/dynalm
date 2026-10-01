// AVX2 + FMA + F16C kernels. Compiled with per-file ISA flags (DD-003) and
// only called after runtime detection selected the AVX2 tier.
//
// Quantized formats: each 32-element group is unpacked with byte-level SIMD
// into int8, converted 8 lanes at a time to fp32 and FMA'd with the fp32
// activations. Per-block scales are applied to vector accumulators; there is
// one horizontal sum per call.

#include "backends/cpu/cpu_kernels.h"

#if ENGINE_HAS_AVX2 && (defined(__x86_64__) || defined(_M_X64))

#include <immintrin.h>

#include <cstring>

#include "dtype/fp16.h"
#include "quant/quant_formats.h"

namespace engine {
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

}  // namespace

bool register_avx2_kernels(CpuKernels& k) {
  k.isa = CpuIsa::kAvx2;
  k.dot_f32 = dot_f32;
  k.axpy_f32 = axpy_f32;
  k.dot_f16_f32 = dot_f16_f32;
  k.axpy_f16 = axpy_f16;
  k.gemm_panel = gemm_panel;
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
  k.dequant[static_cast<size_t>(DType::kQ5_0)] = dequant_q5_0;
  k.dequant[static_cast<size_t>(DType::kQ4_K)] = dequant_q4_K;
  k.dequant[static_cast<size_t>(DType::kQ6_K)] = dequant_q6_K;
  return true;
}

}  // namespace engine

#else  // AVX2 tier not compiled

namespace engine {
bool register_avx2_kernels(CpuKernels&) { return false; }
}  // namespace engine

#endif

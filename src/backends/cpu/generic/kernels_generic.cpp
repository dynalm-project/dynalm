// Portable reference kernels (the generic ISA tier).
//
// Fused dequantize-and-dot for the common GGUF formats, plus a universal
// fallback that expands a quantized row one 256-element chunk at a time into
// a stack buffer (no full-row fp32 materialization).

#include <algorithm>
#include <cstring>

#include "backends/cpu/cpu_kernels.h"
#include "dtype/fp16.h"
#include "quant/dequant.h"
#include "quant/quant_formats.h"

namespace engine {
namespace {

using namespace quant;

float dot_f32(const float* a, const float* b, int64_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

void axpy_f32(float a, const float* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += a * x[i];
}

float dot_f16_f32(const uint16_t* a, const float* b, int64_t n) {
  float s = 0;
  for (int64_t i = 0; i < n; ++i) s += fp16_to_fp32(a[i]) * b[i];
  return s;
}

void axpy_f16(float a, const uint16_t* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += a * fp16_to_fp32(x[i]);
}

void gemm_panel(const float* w, int nr, const float* x, int64_t x_stride, int64_t m, int64_t k, float* y,
                int64_t y_stride, bool accumulate) {
  for (int64_t i = 0; i < m; ++i) {
    for (int r = 0; r < nr; ++r) {
      const float d = dot_f32(w + r * k, x + i * x_stride, k);
      y[i * y_stride + r] = accumulate ? y[i * y_stride + r] + d : d;
    }
  }
}

// --- fused dequantize-dot, scalar ---

float vec_dot_f32(const void* w, const float* x, int64_t n) { return dot_f32(static_cast<const float*>(w), x, n); }
float vec_dot_f16(const void* w, const float* x, int64_t n) {
  return dot_f16_f32(static_cast<const uint16_t*>(w), x, n);
}

float vec_dot_q8_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ8_0*>(w);
  float sum = 0;
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    float s = 0;
    for (int j = 0; j < kQK; ++j) s += static_cast<float>(b[i].qs[j]) * x[j];
    sum += fp16_to_fp32(b[i].d) * s;
  }
  return sum;
}

float vec_dot_q4_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_0*>(w);
  float sum = 0;
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    float s = 0;
    for (int j = 0; j < kQK / 2; ++j) {
      s += static_cast<float>((b[i].qs[j] & 0x0F) - 8) * x[j];
      s += static_cast<float>((b[i].qs[j] >> 4) - 8) * x[j + kQK / 2];
    }
    sum += fp16_to_fp32(b[i].d) * s;
  }
  return sum;
}

float vec_dot_q5_0(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ5_0*>(w);
  float sum = 0;
  for (int64_t i = 0; i < n / kQK; ++i, x += kQK) {
    uint32_t qh;
    std::memcpy(&qh, b[i].qh, 4);
    float s = 0;
    for (int j = 0; j < kQK / 2; ++j) {
      const int x0 = ((b[i].qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10)) - 16;
      const int x1 = ((b[i].qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10)) - 16;
      s += static_cast<float>(x0) * x[j] + static_cast<float>(x1) * x[j + kQK / 2];
    }
    sum += fp16_to_fp32(b[i].d) * s;
  }
  return sum;
}

float vec_dot_q4_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ4_K*>(w);
  float sum = 0;
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d), dmin = fp16_to_fp32(b[i].dmin);
    const uint8_t* q = b[i].qs;
    for (int j = 0, is = 0; j < kQK_K; j += 64, is += 2, q += 32, x += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is, b[i].scales, sc, m);
      float dot_lo = 0, sum_lo = 0, dot_hi = 0, sum_hi = 0;
      for (int l = 0; l < 32; ++l) {
        dot_lo += static_cast<float>(q[l] & 0xF) * x[l];
        sum_lo += x[l];
        dot_hi += static_cast<float>(q[l] >> 4) * x[l + 32];
        sum_hi += x[l + 32];
      }
      sum += d * sc * dot_lo - dmin * m * sum_lo;
      get_scale_min_k4(is + 1, b[i].scales, sc, m);
      sum += d * sc * dot_hi - dmin * m * sum_hi;
    }
  }
  return sum;
}

float vec_dot_q6_K(const void* w, const float* x, int64_t n) {
  const auto* b = static_cast<const BlockQ6_K*>(w);
  float sum = 0;
  for (int64_t i = 0; i < n / kQK_K; ++i) {
    const float d = fp16_to_fp32(b[i].d);
    const uint8_t* ql = b[i].ql;
    const uint8_t* qh = b[i].qh;
    const int8_t* sc = b[i].scales;
    for (int part = 0; part < 2; ++part, ql += 64, qh += 32, sc += 8, x += 128) {
      float s[8] = {};
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        s[is + 0] += static_cast<float>(q1) * x[l];
        s[is + 2] += static_cast<float>(q2) * x[l + 32];
        s[is + 4] += static_cast<float>(q3) * x[l + 64];
        s[is + 6] += static_cast<float>(q4) * x[l + 96];
      }
      for (int g = 0; g < 8; ++g) sum += d * static_cast<float>(sc[g]) * s[g];
    }
  }
  return sum;
}

// Universal fallback: expand 256 elements at a time on the stack.
template <DType T>
float vec_dot_chunked(const void* w, const float* x, int64_t n) {
  constexpr int64_t kChunk = 256;
  float buf[kChunk];
  const int64_t be = dtype_block_elems(T);
  const int64_t step = be >= kChunk ? be : kChunk / be * be;
  const auto* src = static_cast<const std::byte*>(w);
  float sum = 0;
  for (int64_t off = 0; off < n; off += step) {
    const int64_t len = std::min(step, n - off);
    dequantize_row(T, src + dtype_row_bytes(T, off), buf, len);
    sum += dot_f32(buf, x + off, len);
  }
  return sum;
}

template <DType T>
void dequant_ref(const void* w, float* out, int64_t n) {
  dequantize_row(T, w, out, n);
}

}  // namespace

void register_generic_kernels(CpuKernels& k) {
  k.isa = CpuIsa::kGeneric;
  k.dot_f32 = dot_f32;
  k.axpy_f32 = axpy_f32;
  k.dot_f16_f32 = dot_f16_f32;
  k.axpy_f16 = axpy_f16;
  k.gemm_panel = gemm_panel;

  auto set = [&](DType t, VecDotFn vd, DequantFn dq) {
    k.vec_dot[static_cast<size_t>(t)] = vd;
    k.dequant[static_cast<size_t>(t)] = dq;
  };
  set(DType::kF32, vec_dot_f32, dequant_ref<DType::kF32>);
  set(DType::kF16, vec_dot_f16, dequant_ref<DType::kF16>);
  set(DType::kBF16, vec_dot_chunked<DType::kBF16>, dequant_ref<DType::kBF16>);
  set(DType::kQ8_0, vec_dot_q8_0, dequant_ref<DType::kQ8_0>);
  set(DType::kQ4_0, vec_dot_q4_0, dequant_ref<DType::kQ4_0>);
  set(DType::kQ5_0, vec_dot_q5_0, dequant_ref<DType::kQ5_0>);
  set(DType::kQ4_K, vec_dot_q4_K, dequant_ref<DType::kQ4_K>);
  set(DType::kQ6_K, vec_dot_q6_K, dequant_ref<DType::kQ6_K>);
  set(DType::kQ4_1, vec_dot_chunked<DType::kQ4_1>, dequant_ref<DType::kQ4_1>);
  set(DType::kQ5_1, vec_dot_chunked<DType::kQ5_1>, dequant_ref<DType::kQ5_1>);
  set(DType::kQ8_1, vec_dot_chunked<DType::kQ8_1>, dequant_ref<DType::kQ8_1>);
  set(DType::kQ2_K, vec_dot_chunked<DType::kQ2_K>, dequant_ref<DType::kQ2_K>);
  set(DType::kQ3_K, vec_dot_chunked<DType::kQ3_K>, dequant_ref<DType::kQ3_K>);
  set(DType::kQ5_K, vec_dot_chunked<DType::kQ5_K>, dequant_ref<DType::kQ5_K>);
  set(DType::kQ8_K, vec_dot_chunked<DType::kQ8_K>, dequant_ref<DType::kQ8_K>);
}

CpuKernels make_cpu_kernels(CpuIsa isa) {
  CpuKernels k;
  register_generic_kernels(k);
  if (isa == CpuIsa::kAvx2 || isa == CpuIsa::kAvx512 || isa == CpuIsa::kAmx) {
    register_avx2_kernels(k);  // AVX-512/AMX tiers build on AVX2 until they have their own kernels
  }
  if (isa == CpuIsa::kNeon) register_neon_kernels(k);
  return k;
}

}  // namespace engine

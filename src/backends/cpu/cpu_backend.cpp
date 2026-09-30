#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "dtype/fp16.h"
#include "quant/dequant.h"

namespace engine {
namespace {

// --- generic primitives ------------------------------------------------------

float dot_f32_generic(const float* a, const float* b, int64_t n) {
  // Four independent accumulators: lets the compiler vectorize/pipeline and
  // keeps rounding behaviour fixed regardless of optimization level.
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

// Pointer to row r of a 2-D view (strides are bytes).
template <typename T>
T* row_ptr(const TensorView& t, int64_t r) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(t.data()) + r * t.stride(0));
}

int64_t rows(const TensorView& t) { return t.rank() == 1 ? 1 : t.dim(0); }
int64_t cols(const TensorView& t) { return t.dim(t.rank() - 1); }

// Chunk size that gives each thread several chunks (load balance across
// P/E cores) without making chunks so small that dispatch dominates.
size_t grain_for(size_t n, int threads, size_t min_grain) {
  return std::max(min_grain, n / (static_cast<size_t>(threads) * 8));
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }
inline float gelu_tanh(float x) {
  constexpr float kC = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(kC * (x + 0.044715f * x * x * x)));
}
inline float apply_act(Activation a, float x) {
  switch (a) {
    case Activation::kSilu: return silu(x);
    case Activation::kGelu: return gelu(x);
    case Activation::kGeluTanh: return gelu_tanh(x);
  }
  return x;
}

// KV element load/store for the supported cache dtypes.
inline float kv_load(const void* base, int64_t idx, DType t) {
  if (t == DType::kF16) return fp16_to_fp32(static_cast<const uint16_t*>(base)[idx]);
  return static_cast<const float*>(base)[idx];
}

}  // namespace

CpuKernels select_cpu_kernels(CpuIsa isa) {
  CpuKernels k;
  k.isa = CpuIsa::kGeneric;  // ISA-specific kernels arrive in Phase 17
  k.dot_f32 = dot_f32_generic;
  (void)isa;
  return k;
}

CpuBackend::CpuBackend(ThreadPool& pool, CpuIsa isa)
    : pool_(pool), k_(select_cpu_kernels(isa)), name_("CPU/" + std::string(isa_name(k_.isa))) {}

Result<std::shared_ptr<Storage>> CpuBackend::allocate(size_t bytes) { return Storage::allocate_host(bytes); }

void CpuBackend::copy(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

bool CpuBackend::supports_weight_type(DType type) const { return dequant_supported(type); }

void CpuBackend::embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) {
  const int64_t dim = cols(table);
  const int64_t row_bytes = dtype_row_bytes(table.dtype(), dim);
  const auto* base = static_cast<const std::byte*>(table.data());
  for (size_t i = 0; i < ids.size(); ++i) {
    dequantize_row(table.dtype(), base + ids[i] * row_bytes, row_ptr<float>(out, static_cast<int64_t>(i)), dim);
  }
}

void CpuBackend::matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) {
  const int64_t m = rows(x), k = cols(x), n = rows(w);
  assert(cols(w) == k && cols(y) == n && rows(y) == m);
  const DType wt = w.dtype();
  const int64_t w_row_bytes = dtype_row_bytes(wt, k);
  const auto* wbase = static_cast<const std::byte*>(w.data());
  const float* b = bias ? bias->data_as<const float>() : nullptr;
  const auto dot = k_.dot_f32;

  // Parallel over output features: each weight row is converted once and
  // reused for every activation row.
  pool_.parallel_for(static_cast<size_t>(n), grain_for(static_cast<size_t>(n), pool_.size(), 4),
                     [&](size_t begin, size_t end) {
    thread_local std::vector<float> wrow;
    if (wt != DType::kF32) wrow.resize(static_cast<size_t>(k));
    for (size_t j = begin; j < end; ++j) {
      const std::byte* src = wbase + static_cast<int64_t>(j) * w_row_bytes;
      const float* wr;
      if (wt == DType::kF32) {
        wr = reinterpret_cast<const float*>(src);
      } else {
        dequantize_row(wt, src, wrow.data(), k);
        wr = wrow.data();
      }
      const float bj = b ? b[j] : 0.0f;
      for (int64_t i = 0; i < m; ++i) row_ptr<float>(y, i)[j] = dot(row_ptr<const float>(x, i), wr, k) + bj;
    }
  });
}

void CpuBackend::rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) {
  const int64_t m = rows(x), d = cols(x);
  const float* wv = weight.data_as<const float>();
  pool_.parallel_for(static_cast<size_t>(m), 1, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      const float* xr = row_ptr<const float>(x, static_cast<int64_t>(r));
      float* yr = row_ptr<float>(y, static_cast<int64_t>(r));
      double ss = 0;  // double: long rows of large activations
      for (int64_t i = 0; i < d; ++i) ss += static_cast<double>(xr[i]) * xr[i];
      const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + eps));
      for (int64_t i = 0; i < d; ++i) yr[i] = xr[i] * inv * wv[i];
    }
  });
}

void CpuBackend::layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
                            const TensorView& y) {
  const int64_t m = rows(x), d = cols(x);
  const float* wv = weight.data_as<const float>();
  const float* bv = bias ? bias->data_as<const float>() : nullptr;
  pool_.parallel_for(static_cast<size_t>(m), 1, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      const float* xr = row_ptr<const float>(x, static_cast<int64_t>(r));
      float* yr = row_ptr<float>(y, static_cast<int64_t>(r));
      double mean = 0, var = 0;
      for (int64_t i = 0; i < d; ++i) mean += xr[i];
      mean /= static_cast<double>(d);
      for (int64_t i = 0; i < d; ++i) var += (xr[i] - mean) * (xr[i] - mean);
      const float inv = static_cast<float>(1.0 / std::sqrt(var / static_cast<double>(d) + eps));
      for (int64_t i = 0; i < d; ++i) {
        yr[i] = static_cast<float>(xr[i] - mean) * inv * wv[i] + (bv ? bv[i] : 0.0f);
      }
    }
  });
}

void CpuBackend::rope(const TensorView& x, int32_t num_heads, int32_t head_dim, std::span<const int32_t> positions,
                      const RopeConfig& rope, const float* freq_factors) {
  if (rope.style == RopeStyle::kNone) return;
  const int32_t rot = rope.dim;
  const int32_t half = rot / 2;
  thread_local std::vector<float> inv_freq;
  inv_freq.resize(static_cast<size_t>(half));
  for (int32_t i = 0; i < half; ++i) {
    float f = std::pow(rope.freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(rot));
    if (freq_factors) f /= freq_factors[i];
    inv_freq[static_cast<size_t>(i)] = f;
  }
  const float pos_scale = rope.scaling == RopeScaling::kLinear ? 1.0f / rope.scaling_factor : 1.0f;
  const bool interleaved = rope.style == RopeStyle::kInterleaved;
  const float* freqs = inv_freq.data();

  pool_.parallel_for(positions.size(), 1, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      float* xr = row_ptr<float>(x, static_cast<int64_t>(r));
      const float p = static_cast<float>(positions[r]) * pos_scale;
      for (int32_t i = 0; i < half; ++i) {
        const float theta = p * freqs[i];
        const float c = std::cos(theta), s = std::sin(theta);
        for (int32_t h = 0; h < num_heads; ++h) {
          float* hx = xr + static_cast<int64_t>(h) * head_dim;
          const int32_t a = interleaved ? 2 * i : i;
          const int32_t b = interleaved ? 2 * i + 1 : i + half;
          const float x0 = hx[a], x1 = hx[b];
          hx[a] = x0 * c - x1 * s;
          hx[b] = x0 * s + x1 * c;
        }
      }
    }
  });
}

void CpuBackend::kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                          const KvLayerView& kv) {
  const KvGeometry& g = *kv.geom;
  for (size_t r = 0; r < positions.size(); ++r) {
    const float* kr = row_ptr<const float>(k, static_cast<int64_t>(r));
    const float* vr = row_ptr<const float>(v, static_cast<int64_t>(r));
    for (int32_t h = 0; h < g.num_kv_heads; ++h) {
      const int64_t ko = kv.k_offset(positions[r], h), vo = kv.v_offset(positions[r], h);
      const float* ks = kr + static_cast<int64_t>(h) * g.head_dim;
      const float* vs = vr + static_cast<int64_t>(h) * g.head_dim_v;
      if (g.dtype == DType::kF16) {
        auto* kd = static_cast<uint16_t*>(kv.k) + ko;
        auto* vd = static_cast<uint16_t*>(kv.v) + vo;
        for (int32_t i = 0; i < g.head_dim; ++i) kd[i] = fp32_to_fp16(ks[i]);
        for (int32_t i = 0; i < g.head_dim_v; ++i) vd[i] = fp32_to_fp16(vs[i]);
      } else {
        std::memcpy(static_cast<float*>(kv.k) + ko, ks, sizeof(float) * static_cast<size_t>(g.head_dim));
        std::memcpy(static_cast<float*>(kv.v) + vo, vs, sizeof(float) * static_cast<size_t>(g.head_dim_v));
      }
    }
  }
}

void CpuBackend::attention(const AttentionParams& p) {
  const KvLayerView& kv = p.kv;
  const KvGeometry& g = *kv.geom;
  const int32_t group = p.num_heads / g.num_kv_heads;
  const int32_t hd = g.head_dim, hdv = g.head_dim_v;
  const size_t m = p.positions.size();
  const DType kt = g.dtype;
  const auto dot = k_.dot_f32;

  pool_.parallel_for(m * static_cast<size_t>(p.num_heads), 1, [&](size_t begin, size_t end) {
    thread_local std::vector<float> scores, kbuf;
    kbuf.resize(static_cast<size_t>(std::max(hd, hdv)));
    for (size_t job = begin; job < end; ++job) {
      const size_t r = job / static_cast<size_t>(p.num_heads);
      const int32_t h = static_cast<int32_t>(job % static_cast<size_t>(p.num_heads));
      const int32_t kvh = h / group;
      const float* q = row_ptr<const float>(p.q, static_cast<int64_t>(r)) + static_cast<int64_t>(h) * hd;
      float* out = row_ptr<float>(p.out, static_cast<int64_t>(r)) + static_cast<int64_t>(h) * hdv;
      const int64_t pos = p.positions[r];
      const int64_t lo = p.sliding_window > 0 ? std::max<int64_t>(0, pos - p.sliding_window + 1) : 0;
      const int64_t n = pos - lo + 1;
      scores.resize(static_cast<size_t>(n));

      float mx = -INFINITY;
      for (int64_t t = 0; t < n; ++t) {
        const int64_t off = kv.k_offset(lo + t, kvh);
        const float* kr;
        if (kt == DType::kF32) {
          kr = static_cast<const float*>(kv.k) + off;
        } else {
          for (int32_t i = 0; i < hd; ++i) kbuf[static_cast<size_t>(i)] = kv_load(kv.k, off + i, kt);
          kr = kbuf.data();
        }
        float s = dot(q, kr, hd) * p.scale;
        if (p.softcap > 0) s = p.softcap * std::tanh(s / p.softcap);
        scores[static_cast<size_t>(t)] = s;
        mx = std::max(mx, s);
      }
      double sum = 0;
      for (int64_t t = 0; t < n; ++t) {
        const float e = std::exp(scores[static_cast<size_t>(t)] - mx);
        scores[static_cast<size_t>(t)] = e;
        sum += e;
      }
      const float inv = static_cast<float>(1.0 / sum);
      std::fill(out, out + hdv, 0.0f);
      for (int64_t t = 0; t < n; ++t) {
        const float w = scores[static_cast<size_t>(t)] * inv;
        const int64_t off = kv.v_offset(lo + t, kvh);
        if (kt == DType::kF32) {
          const float* vr = static_cast<const float*>(kv.v) + off;
          for (int32_t i = 0; i < hdv; ++i) out[i] += w * vr[i];
        } else {
          for (int32_t i = 0; i < hdv; ++i) out[i] += w * kv_load(kv.v, off + i, kt);
        }
      }
    }
  });
}

void CpuBackend::act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) {
  const int64_t m = rows(gate), d = cols(gate);
  pool_.parallel_for(static_cast<size_t>(m), 1, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      const float* gr = row_ptr<const float>(gate, static_cast<int64_t>(r));
      const float* ur = row_ptr<const float>(up, static_cast<int64_t>(r));
      float* o = row_ptr<float>(out, static_cast<int64_t>(r));
      for (int64_t i = 0; i < d; ++i) o[i] = apply_act(act, gr[i]) * ur[i];
    }
  });
}

void CpuBackend::activation(Activation act, const TensorView& x, const TensorView& out) {
  const int64_t m = rows(x), d = cols(x);
  for (int64_t r = 0; r < m; ++r) {
    const float* xr = row_ptr<const float>(x, r);
    float* o = row_ptr<float>(out, r);
    for (int64_t i = 0; i < d; ++i) o[i] = apply_act(act, xr[i]);
  }
}

void CpuBackend::add(const TensorView& a, const TensorView& b, const TensorView& y) {
  const int64_t m = rows(a), d = cols(a);
  for (int64_t r = 0; r < m; ++r) {
    const float* ar = row_ptr<const float>(a, r);
    const float* br = row_ptr<const float>(b, r);
    float* yr = row_ptr<float>(y, r);
    for (int64_t i = 0; i < d; ++i) yr[i] = ar[i] + br[i];
  }
}

void CpuBackend::scale(const TensorView& x, float s) {
  const int64_t m = rows(x), d = cols(x);
  for (int64_t r = 0; r < m; ++r) {
    float* xr = row_ptr<float>(x, r);
    for (int64_t i = 0; i < d; ++i) xr[i] *= s;
  }
}

void CpuBackend::softcap(const TensorView& x, float cap) {
  const int64_t m = rows(x), d = cols(x);
  for (int64_t r = 0; r < m; ++r) {
    float* xr = row_ptr<float>(x, r);
    for (int64_t i = 0; i < d; ++i) xr[i] = cap * std::tanh(xr[i] / cap);
  }
}

}  // namespace engine

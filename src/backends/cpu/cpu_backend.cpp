#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dtype/fp16.h"

namespace engine {
namespace {

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


}  // namespace

CpuBackend::CpuBackend(ThreadPool& pool, CpuIsa isa)
    : pool_(pool), k_(make_cpu_kernels(isa)), name_("CPU/" + std::string(isa_name(k_.isa))) {
  // Tuning override for experiments / the AutoTuner (multiple of 256; 0 = off).
  if (const char* kc = std::getenv("ENGINE_GEMM_KC")) {
    const long v = std::strtol(kc, nullptr, 10);
    gemm_kc_ = v > 0 ? v / 256 * 256 : 0;
  }
  if (const char* em = std::getenv("ENGINE_MATMUL_EXPAND_MIN")) {
    expand_min_rows_ = std::max<long>(1, std::strtol(em, nullptr, 10));
  }
}

Result<std::shared_ptr<Storage>> CpuBackend::allocate(size_t bytes) { return Storage::allocate_host(bytes); }

void CpuBackend::copy(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

bool CpuBackend::supports_weight_type(DType type) const { return k_.vec_dot_for(type) != nullptr; }

void CpuBackend::embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) {
  const int64_t dim = cols(table);
  const int64_t row_bytes = dtype_row_bytes(table.dtype(), dim);
  const auto* base = static_cast<const std::byte*>(table.data());
  for (size_t i = 0; i < ids.size(); ++i) {
    k_.dequant_for(table.dtype())(base + ids[i] * row_bytes, row_ptr<float>(out, static_cast<int64_t>(i)), dim);
  }
}

void CpuBackend::matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) {
  const int64_t m = rows(x), k = cols(x), n = rows(w);
  assert(cols(w) == k && cols(y) == n && rows(y) == m);
  const DType wt = w.dtype();
  const int64_t w_row_bytes = dtype_row_bytes(wt, k);
  const auto* wbase = static_cast<const std::byte*>(w.data());
  const float* b = bias ? bias->data_as<const float>() : nullptr;
  const VecDotFn vec_dot = k_.vec_dot_for(wt);
  const DequantFn dequant = k_.dequant_for(wt);
  const auto dot = k_.dot_f32;
  // Few activation rows (decode): fused dequantize-dot straight from the
  // packed weights. Many rows (prefill): expand each weight row once and
  // reuse it for every activation row.
  const bool expand = wt != DType::kF32 && m >= expand_min_rows_;

  // Prefill path: panels of kPanel weight rows are expanded to fp32 once and
  // multiplied against all activation rows by the register-blocked kernel.
  constexpr int64_t kPanel = 4;
  const bool x_dense = x.stride(0) == k * static_cast<int64_t>(sizeof(float));
  const bool f32_panel = wt == DType::kF32 && m >= expand_min_rows_ && x_dense;
  if ((expand || f32_panel) && x_dense) {
    const size_t panels = static_cast<size_t>((n + kPanel - 1) / kPanel);
    const int64_t y_stride = y.stride(0) / static_cast<int64_t>(sizeof(float));
    const auto* xp = x.data_as<const float>();
    auto* yp = y.data_as<float>();
    // K-blocking (gemm_kc_ > 0): slices of gemm_kc_ columns, so a slice of
    // all activation rows stays cache-resident while a thread sweeps its
    // panels; partial sums accumulate into y. Slices start on multiples of
    // 256, so every block format dequantizes slice by slice.
    const int64_t kc = (gemm_kc_ > 0 && gemm_kc_ < k) ? gemm_kc_ : k;
    pool_.parallel_for(panels, grain_for(panels, pool_.size(), 1), [&](size_t begin, size_t end) {
      thread_local std::vector<float> panel;
      panel.resize(static_cast<size_t>(kPanel * kc));
      for (int64_t k0 = 0; k0 < k; k0 += kc) {
        const int64_t len = std::min(kc, k - k0);
        for (size_t pi = begin; pi < end; ++pi) {
          const int64_t j0 = static_cast<int64_t>(pi) * kPanel;
          const int nr = static_cast<int>(std::min<int64_t>(kPanel, n - j0));
          for (int r = 0; r < nr; ++r) {
            dequant(wbase + (j0 + r) * w_row_bytes + dtype_row_bytes(wt, k0), panel.data() + r * len, len);
          }
          k_.gemm_panel(panel.data(), nr, xp + k0, k, m, len, yp + j0, y_stride, /*accumulate=*/k0 > 0);
        }
      }
      if (b) {
        for (size_t pi = begin; pi < end; ++pi) {
          const int64_t j0 = static_cast<int64_t>(pi) * kPanel;
          const int nr = static_cast<int>(std::min<int64_t>(kPanel, n - j0));
          for (int64_t i = 0; i < m; ++i) {
            for (int r = 0; r < nr; ++r) yp[i * y_stride + j0 + r] += b[j0 + r];
          }
        }
      }
    });
    return;
  }

  pool_.parallel_for(static_cast<size_t>(n), grain_for(static_cast<size_t>(n), pool_.size(), 4),
                     [&](size_t begin, size_t end) {
    thread_local std::vector<float> wrow;
    if (expand) wrow.resize(static_cast<size_t>(k));
    for (size_t j = begin; j < end; ++j) {
      const std::byte* src = wbase + static_cast<int64_t>(j) * w_row_bytes;
      const float bj = b ? b[j] : 0.0f;
      if (expand) {
        dequant(src, wrow.data(), k);
        for (int64_t i = 0; i < m; ++i) row_ptr<float>(y, i)[j] = dot(row_ptr<const float>(x, i), wrow.data(), k) + bj;
      } else {
        for (int64_t i = 0; i < m; ++i) row_ptr<float>(y, i)[j] = vec_dot(src, row_ptr<const float>(x, i), k) + bj;
      }
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
                          std::span<const int32_t> row_seq, std::span<const KvLayerView> kv_views) {
  for (size_t r = 0; r < positions.size(); ++r) {
    const KvLayerView& kv = kv_views[static_cast<size_t>(row_seq[r])];
    const KvGeometry& g = *kv.geom;
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

// Unnormalized attention over key positions [t0, t1) of one (row, head):
// acc = sum_t exp(s_t - mx) * v_t, with mx = max_t s_t and sum = sum_t exp(s_t - mx).
// A full softmax is acc / sum; partials over disjoint ranges merge exactly
// with log-sum-exp rescaling (split-K / flash-decoding).
void CpuBackend::attend_range(const AttentionParams& p, size_t r, int32_t h, int64_t t0, int64_t t1, float* acc,
                              float& mx_out, double& sum_out) {
  const KvGeometry& g = *p.kv.front().geom;
  const int32_t kvh = h / (p.num_heads / g.num_kv_heads);
  const int32_t hd = g.head_dim, hdv = g.head_dim_v;
  const KvLayerView& kv = p.kv[static_cast<size_t>(p.row_seq[r])];
  const float* q = row_ptr<const float>(p.q, static_cast<int64_t>(r)) + static_cast<int64_t>(h) * hd;
  const bool f32 = g.dtype == DType::kF32;
  thread_local std::vector<float> scores;
  scores.resize(static_cast<size_t>(t1 - t0));

  float mx = -INFINITY;
  for (int64_t t = t0; t < t1; ++t) {
    const int64_t off = kv.k_offset(t, kvh);
    float s = (f32 ? k_.dot_f32(q, static_cast<const float*>(kv.k) + off, hd)
                   : k_.dot_f16_f32(static_cast<const uint16_t*>(kv.k) + off, q, hd)) *
              p.scale;
    if (p.softcap > 0) s = p.softcap * std::tanh(s / p.softcap);
    scores[static_cast<size_t>(t - t0)] = s;
    mx = std::max(mx, s);
  }
  double sum = 0;
  std::fill(acc, acc + hdv, 0.0f);
  for (int64_t t = t0; t < t1; ++t) {
    const float e = std::exp(scores[static_cast<size_t>(t - t0)] - mx);
    sum += e;
    const int64_t off = kv.v_offset(t, kvh);
    if (f32) {
      k_.axpy_f32(e, static_cast<const float*>(kv.v) + off, acc, hdv);
    } else {
      k_.axpy_f16(e, static_cast<const uint16_t*>(kv.v) + off, acc, hdv);
    }
  }
  mx_out = mx;
  sum_out = sum;
}

void CpuBackend::attention(const AttentionParams& p) {
  const KvGeometry& g = *p.kv.front().geom;  // all sequences share the pool geometry
  const int32_t hdv = g.head_dim_v;
  const size_t m = p.positions.size();
  const size_t pairs = m * static_cast<size_t>(p.num_heads);
  auto range_of = [&](size_t r, int64_t& lo, int64_t& hi) {
    const int64_t pos = p.positions[r];
    lo = p.sliding_window > 0 ? std::max<int64_t>(0, pos - p.sliding_window + 1) : 0;
    hi = pos + 1;
  };
  int64_t longest = 0;
  for (size_t r = 0; r < m; ++r) {
    int64_t lo, hi;
    range_of(r, lo, hi);
    longest = std::max(longest, hi - lo);
  }

  // Split-K when (row, head) pairs alone cannot occupy the pool (decode) and
  // contexts are long enough to amortize the merge.
  constexpr int64_t kChunk = 256;
  const bool split = pairs < static_cast<size_t>(2 * pool_.size()) && longest > 2 * kChunk;
  if (!split) {
    pool_.parallel_for(pairs, 1, [&](size_t begin, size_t end) {
      for (size_t job = begin; job < end; ++job) {
        const size_t r = job / static_cast<size_t>(p.num_heads);
        const auto h = static_cast<int32_t>(job % static_cast<size_t>(p.num_heads));
        int64_t lo, hi;
        range_of(r, lo, hi);
        float* out = row_ptr<float>(p.out, static_cast<int64_t>(r)) + static_cast<int64_t>(h) * hdv;
        float mx;
        double sum;
        attend_range(p, r, h, lo, hi, out, mx, sum);
        const auto inv = static_cast<float>(1.0 / sum);
        for (int32_t i = 0; i < hdv; ++i) out[i] *= inv;
      }
    });
    return;
  }

  // Pass 1: every (pair, chunk) computes a partial into scratch.
  const auto chunks = static_cast<size_t>((longest + kChunk - 1) / kChunk);
  const size_t stride = static_cast<size_t>(hdv) + 2;  // acc[hdv], max, sum
  split_scratch_.resize(pairs * chunks * stride);
  float* scratch = split_scratch_.data();
  pool_.parallel_for(pairs * chunks, 1, [&](size_t begin, size_t end) {
    for (size_t job = begin; job < end; ++job) {
      const size_t pair = job / chunks, c = job % chunks;
      const size_t r = pair / static_cast<size_t>(p.num_heads);
      const auto h = static_cast<int32_t>(pair % static_cast<size_t>(p.num_heads));
      int64_t lo, hi;
      range_of(r, lo, hi);
      const int64_t t0 = lo + static_cast<int64_t>(c) * kChunk, t1 = std::min(hi, t0 + kChunk);
      float* part = scratch + job * stride;
      if (t0 >= t1) {  // this row's context is shorter than the longest
        part[hdv] = -INFINITY;
        part[hdv + 1] = 0.0f;
        continue;
      }
      float mx;
      double sum;
      attend_range(p, r, h, t0, t1, part, mx, sum);
      part[hdv] = mx;
      part[hdv + 1] = static_cast<float>(sum);
    }
  });
  // Pass 2: merge partials per pair with log-sum-exp rescaling.
  pool_.parallel_for(pairs, 1, [&](size_t begin, size_t end) {
    for (size_t pair = begin; pair < end; ++pair) {
      const size_t r = pair / static_cast<size_t>(p.num_heads);
      const auto h = static_cast<int32_t>(pair % static_cast<size_t>(p.num_heads));
      float* out = row_ptr<float>(p.out, static_cast<int64_t>(r)) + static_cast<int64_t>(h) * hdv;
      const float* parts = scratch + pair * chunks * stride;
      float gmax = -INFINITY;
      for (size_t c = 0; c < chunks; ++c) gmax = std::max(gmax, parts[c * stride + static_cast<size_t>(hdv)]);
      double total = 0;
      std::fill(out, out + hdv, 0.0f);
      for (size_t c = 0; c < chunks; ++c) {
        const float* part = parts + c * stride;
        if (part[hdv] == -INFINITY) continue;
        const float w = std::exp(part[hdv] - gmax);
        total += static_cast<double>(w) * part[hdv + 1];
        k_.axpy_f32(w, part, out, hdv);
      }
      const auto inv = static_cast<float>(1.0 / total);
      for (int32_t i = 0; i < hdv; ++i) out[i] *= inv;
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

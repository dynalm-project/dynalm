#include "dynacore/cpu/cpu_device.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string_view>
#include <tuple>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "dynacore/base/fast_exp.h"
#include "dynacore/base/timer.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/memory/mapped_file.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

namespace dynacore {
namespace {

// int8_rows calls that took the experimental 8-row packed path (test and
// benchmark evidence; relaxed, one increment per call).
std::atomic<uint64_t> g_packed_rows8_dispatches{0};
std::atomic<uint64_t> g_int16_prefill_calls{0};  // KernelPlan::int16_prefill_max_rows matmuls (experimental)

// Matmul registry diagnostics (process-wide, relaxed): per kernel id.
std::atomic<uint64_t> g_mm_calls[kMatmulKernelCount];
std::atomic<uint64_t> g_mm_rows[kMatmulKernelCount];
std::atomic<uint64_t> g_mm_macs[kMatmulKernelCount];
std::atomic<uint64_t> g_mm_ns[kMatmulKernelCount];
std::atomic<bool> g_mm_timing{[] {
  const char* v = std::getenv("DYNACORE_MATMUL_TIMING");
  return v != nullptr && v[0] == '1';
}()};
struct MatmulShapeKey {
  MatmulKernelId id;
  DType weight;
  int64_t n, k, m;
  bool operator<(const MatmulShapeKey& o) const {
    return std::tie(id, weight, n, k, m) < std::tie(o.id, o.weight, o.n, o.k, o.m);
  }
};
std::mutex g_mm_shape_mu;
std::map<MatmulShapeKey, std::pair<uint64_t, uint64_t>> g_mm_shapes;  // calls, ns (timing on)
std::map<MatmulShapeKey, bool> g_mm_explained;                         // explain-once set (under g_mm_shape_mu)

bool matmul_explain_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DYNACORE_MATMUL_EXPLAIN");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// ns < 0: timing off for this call.
void record_matmul(MatmulKernelId id, const MatmulShape& s, int64_t ns) {
  const size_t i = static_cast<size_t>(id);
  g_mm_calls[i].fetch_add(1, std::memory_order_relaxed);
  g_mm_rows[i].fetch_add(static_cast<uint64_t>(s.m), std::memory_order_relaxed);
  g_mm_macs[i].fetch_add(static_cast<uint64_t>(s.m) * static_cast<uint64_t>(s.n) * static_cast<uint64_t>(s.k), std::memory_order_relaxed);
  if (ns >= 0) {
    g_mm_ns[i].fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mm_shape_mu);
    auto& v = g_mm_shapes[MatmulShapeKey{id, s.weight, s.n, s.k, s.m}];
    ++v.first;
    v.second += static_cast<uint64_t>(ns);
  }
}

// DYNACORE_MATMUL_EXPLAIN=1: the first call of each (kernel, shape) logs the
// selection and the higher-priority kernels it passed over.
void explain_matmul_once(const MatmulShape& s, const MatmulSelection& sel) {
  {
    std::lock_guard<std::mutex> lock(g_mm_shape_mu);
    if (!g_mm_explained.emplace(MatmulShapeKey{sel.id, s.weight, s.n, s.k, s.m}, true).second) return;
  }
  std::fprintf(stderr, "dynacore matmul: %s [%lld x %lld] m=%lld -> %s\n", std::string(dtype_name(s.weight)).c_str(),
               static_cast<long long>(s.n), static_cast<long long>(s.k), static_cast<long long>(s.m), sel.explain().c_str());
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
size_t grain_for(size_t n, int threads, size_t min_grain, int32_t chunks_per_thread) {
  return std::max(min_grain, n / (static_cast<size_t>(threads) * static_cast<size_t>(chunks_per_thread)));
}

// Grain for a decode-shaped matmul (KernelPlan::gemv_min_chunk_bytes): the
// same number of chunks for every thread, as many as fit min_bytes of weights
// each (at least 1, at most chunks_per_thread). A chunk count just above the
// thread count would leave some threads with twice the work of the others.
size_t gemv_grain(size_t n, size_t item_bytes, int threads, int32_t chunks_per_thread, int64_t min_bytes) {
  if (min_bytes <= 0 || item_bytes == 0) return grain_for(n, threads, 1, chunks_per_thread);
  const auto t = static_cast<size_t>(std::max(threads, 1));
  const size_t per_thread =
      std::clamp<size_t>(n * item_bytes / t / static_cast<size_t>(min_bytes), 1, static_cast<size_t>(std::max(chunks_per_thread, 1)));
  return std::max<size_t>(1, (n + t * per_thread - 1) / (t * per_thread));
}

// x * sigmoid(x) with e = e^-|x| from exp_nonpos (no libm call, so loops of
// it vectorize; relative error ~1e-7): x / (1 + e) for x >= 0, x e / (1 + e)
// below.
inline float silu(float x) {
  const float e = exp_nonpos(-std::fabs(x));
  return x * (x >= 0.0f ? 1.0f : e) / (1.0f + e);
}
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

CpuDevice::CpuDevice(ThreadPool& pool, CpuIsa isa)
    : pool_(pool), k_(make_cpu_kernels(isa)), name_("CPU/" + std::string(isa_name(k_.isa))) {
  int8_accelerated_ = k_.int8_simd;  // AVX2, or NEON with the dot-product instructions
  // qmm only where a SIMD tier exists (the scalar reference is for tests);
  // DYNACORE_ISA=generic keeps the portable kernels everywhere.
  if (k_.isa != CpuIsa::kGeneric && qmm::accelerated()) qmm_ = &qmm::best_kernels();
  if (qmm_ != nullptr) name_ += std::string("+qmm/") + qmm_->name;
  if (qmm_ != nullptr) sme_ = qmm::sme_gemm();
  if (sme_ != nullptr) name_ += std::string("+") + sme_->name;
}

void CpuDevice::qmm_quantize(qmm::ActBuffer& buf, const TensorView& x, qmm::ActMode mode, bool paired) {
  const int64_t m = rows(x), k = cols(x);
  buf.prepare(mode, m, k, paired);
  const float* xp = x.data_as<const float>();
  const int64_t xs = x.rank() == 1 ? k : x.stride(0) / static_cast<int64_t>(sizeof(float));
  if (m >= 16) {
    pool_.parallel_for(static_cast<size_t>(m), 4, [&](size_t b, size_t e) {
      buf.quantize_rows(xp, xs, static_cast<int64_t>(b), static_cast<int64_t>(e), *qmm_);
    });
  } else {
    buf.quantize_rows(xp, xs, 0, m, *qmm_);
  }
}

const std::byte* CpuDevice::find_qmm_packed(const void* w, int64_t n, int64_t k) const {
  const auto it = qmm_packed_.find(w);
  return it != qmm_packed_.end() && it->second.n == n && it->second.k == k ? it->second.data.data() : nullptr;
}

void CpuDevice::qmm_columns(const qmm::Acts& a, bool gemm, const TensorView& w, const std::byte* packed, const TensorView& y,
                            const float* bias, int64_t n0, int64_t n1) const {
  const DType wt = w.dtype();
  const int64_t rb = dtype_row_bytes(wt, a.k);
  float* yp = y.data_as<float>();
  const int64_t ys = y.rank() == 1 ? cols(y) : y.stride(0) / static_cast<int64_t>(sizeof(float));
  // Packed copy: whole 4-row groups of [n0, n1) from it (n0 is a multiple of
  // 16 here), the rows past the last group from the original.
  if (const std::byte* pk = packed; pk != nullptr) {
    const int64_t g0 = n0 / 4, g1 = std::min(n1, rows(w) / 4 * 4) / 4;
    if (g1 > g0) {
      if (gemm && qmm_->gemm_packed != nullptr) {
        thread_local qmm::GemmScratch scratch;
        qmm_->gemm_packed(wt, pk, a.k, g0, g1, a, yp, ys, bias, scratch);
      } else {
        for (int64_t r0 = 0; r0 < a.m; r0 += qmm::kGemvMaxRows) {
          qmm_->gemv_packed(wt, pk, a.k, g0, g1, a, r0, static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, a.m - r0)), yp, ys,
                            bias);
        }
      }
      n0 = 4 * g1;
    }
    if (n0 >= n1) return;
  }
  if (gemm) {
    thread_local qmm::GemmScratch scratch;
    qmm_->gemm(wt, w.data(), rb, n0, n1, a, yp, ys, bias, scratch);
    return;
  }
  for (int64_t r0 = 0; r0 < a.m; r0 += qmm::kGemvMaxRows) {
    qmm_->gemv(wt, w.data(), rb, n0, n1, a, r0, static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, a.m - r0)), yp, ys,
               bias);
  }
}

Result<std::shared_ptr<Storage>> CpuDevice::allocate(size_t bytes) { return Storage::allocate_host(bytes); }

void CpuDevice::copy(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

bool CpuDevice::supports_weight_type(DType type) const { return k_.vec_dot_for(type) != nullptr; }

void CpuDevice::embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) {
  const int64_t dim = cols(table);
  const int64_t row_bytes = dtype_row_bytes(table.dtype(), dim);
  const auto* base = static_cast<const std::byte*>(table.data());
  for (size_t i = 0; i < ids.size(); ++i) {
    k_.dequant_for(table.dtype())(base + ids[i] * row_bytes, row_ptr<float>(out, static_cast<int64_t>(i)), dim);
  }
}

void CpuDevice::download(const TensorView& src, std::span<float> dst) {
  const int64_t r = rows(src), c = cols(src);
  assert(dst.size() >= static_cast<size_t>(r * c));
  for (int64_t i = 0; i < r; ++i) {
    std::memcpy(dst.data() + i * c, row_ptr<const float>(src, i), static_cast<size_t>(c) * sizeof(float));
  }
}

void CpuDevice::fill(const TensorView& x, float value) {
  for (int64_t i = 0; i < rows(x); ++i) std::fill_n(row_ptr<float>(x, i), cols(x), value);
}

void CpuDevice::gather_rows(const TensorView& src, std::span<const int32_t> idx, const TensorView& dst) {
  const auto bytes = static_cast<size_t>(cols(src)) * sizeof(float);
  for (size_t i = 0; i < idx.size(); ++i) {
    std::memcpy(row_ptr<float>(dst, static_cast<int64_t>(i)), row_ptr<const float>(src, idx[i]), bytes);
  }
}

void CpuDevice::scatter_add_rows(const TensorView& src, std::span<const int32_t> idx, std::span<const float> weights,
                                  const TensorView& dst) {
  const int64_t c = cols(src);
  for (size_t i = 0; i < idx.size(); ++i) {
    float* d = row_ptr<float>(dst, idx[i]);
    const float* s = row_ptr<const float>(src, static_cast<int64_t>(i));
    const float w = weights[i];
    for (int64_t j = 0; j < c; ++j) d[j] += w * s[j];
  }
}

namespace {
bool q6_packed_enabled();  // defined below (DD-082 opt-in switch)
}  // namespace

bool CpuDevice::prepack_allowed(int64_t bytes) {
  // Memory control (DD-078): DYNACORE_Q4_REPACK=off | auto (default) | always.
  // auto packs only while at least 3 GiB of RAM stay available, checked per
  // tensor: packing reads the weights, so the model's own pages become
  // resident as this runs and available memory falls.
  const char* mode = std::getenv("DYNACORE_Q4_REPACK");
  const std::string_view m = mode != nullptr ? std::string_view(mode) : std::string_view("auto");
  if (m == "off" || m == "0") return false;
  if (m != "always" && m != "1") {
    constexpr int64_t kReserve = int64_t{3} << 30;
    if (memory_info().available_bytes - bytes < kReserve) {
      ++prepack_stats_.skipped;
      return false;
    }
  }
  return true;
}

void CpuDevice::prepack_weight(const TensorView& w, bool int16_activations) {
  // qmm: the packed (4-row interleaved) copy for its decode kernels where the
  // format has one; the other qmm formats read the original layout and need
  // no copy. DYNACORE_QMM_PACK=0 keeps every weight in its file layout.
  if (qmm_ != nullptr && KernelPlan::defaults().qmm && qmm::supports(w.dtype())) {
    static const bool pack_on = [] {
      const char* v = std::getenv("DYNACORE_QMM_PACK");
      return !(v != nullptr && v[0] == '0');
    }();
    const int64_t n = rows(w), k = cols(w);
    if (!pack_on || qmm_->gemv_packed == nullptr || !qmm::packable(w.dtype(), k) || n < 4 || qmm_packed_.count(w.data()) != 0) {
      return;
    }
    const auto bytes = static_cast<int64_t>(qmm::packed_bytes(w.dtype(), n, k));
    if (!prepack_allowed(bytes)) return;
    const int64_t t0 = now_ns();
    QmmPacked p;
    p.n = n;
    p.k = k;
    p.data.resize(static_cast<size_t>(bytes));
    const int64_t groups = n / 4, rb = dtype_row_bytes(w.dtype(), k);
    pool_.parallel_for(static_cast<size_t>(groups), 16, [&](size_t begin, size_t end) {
      qmm::pack_groups(w.dtype(), w.data(), rb, k, static_cast<int64_t>(begin), static_cast<int64_t>(end), p.data.data());
    });
    ++prepack_stats_.tensors;
    prepack_stats_.source_bytes += n * rb;
    prepack_stats_.packed_bytes += bytes;
    prepack_stats_.ms += static_cast<double>(now_ns() - t0) * 1e-6;
    qmm_packed_.emplace(w.data(), std::move(p));
    // The qmm kernels read only the packed copy from here on, so the file
    // pages of the original stop counting toward resident memory (they fault
    // back in from the file if another path reads them).
    // DYNACORE_QMM_RELEASE_SOURCE=0 keeps them resident.
    static const bool release_on = [] {
      const char* v = std::getenv("DYNACORE_QMM_RELEASE_SOURCE");
      return !(v != nullptr && v[0] == '0');
    }();
    if (release_on) {
      prepack_stats_.released_bytes += static_cast<int64_t>(release_file_pages(w.data(), static_cast<size_t>(n * rb)));
    }
    return;
  }
  if (!int8_accelerated_ || KernelPlan::defaults().q4_repack_min_rows <= 0) return;
  const int64_t n = rows(w), k = cols(w);
  if (n % 8 != 0 || k % 256 != 0) return;
  // Interleaved Q6_K, only for int16-activation weights (DD-082): its only
  // kernel is the int16 one.
  if (w.dtype() == DType::kQ6_K) {
    if (!int16_activations || !q6_packed_enabled() || k_.dot_q16_x8_q6_K == nullptr || packed6_.count(w.data()) != 0) {
      return;
    }
    const int64_t row_bytes = dtype_row_bytes(DType::kQ6_K, k), nb = k / 256, groups = n / 8;
    const int64_t bytes = groups * nb * static_cast<int64_t>(sizeof(BlockQ6_Kx8));
    if (!prepack_allowed(bytes)) return;
    const int64_t t0 = now_ns();
    Packed6 p;
    p.n = n;
    p.k = k;
    p.blocks.resize(static_cast<size_t>(groups * nb));
    pool_.parallel_for(static_cast<size_t>(groups), 1, [&](size_t begin, size_t end) {
      for (size_t g = begin; g < end; ++g) {
        repack_q6_K_x8(w.data(), row_bytes, static_cast<int64_t>(g) * 8, k, p.blocks.data() + static_cast<int64_t>(g) * nb);
      }
    });
    ++prepack_stats_.tensors;
    prepack_stats_.source_bytes += n * row_bytes;
    prepack_stats_.packed_bytes += bytes;
    prepack_stats_.ms += static_cast<double>(now_ns() - t0) * 1e-6;
    packed6_.emplace(w.data(), std::move(p));
    return;
  }
  // Interleaved Q4_K for multi-row decode (DD-078): only where the kernel
  // exists and the shape groups into 8 rows of whole super-blocks.
  if (w.dtype() != DType::kQ4_K || k_.dot_q8_sx_x8_q4_K == nullptr || packed_.count(w.data()) != 0) return;
  const int64_t row_bytes = dtype_row_bytes(DType::kQ4_K, k), nb = k / 256, groups = n / 8;
  const int64_t bytes = groups * nb * static_cast<int64_t>(sizeof(BlockQ4_Kx8));
  if (!prepack_allowed(bytes)) return;
  const int64_t t0 = now_ns();
  Packed p;
  p.n = n;
  p.k = k;
  p.blocks.resize(static_cast<size_t>(groups * nb));
  pool_.parallel_for(static_cast<size_t>(groups), 1, [&](size_t begin, size_t end) {
    for (size_t g = begin; g < end; ++g) {
      repack_q4_K_x8(w.data(), row_bytes, static_cast<int64_t>(g) * 8, k, p.blocks.data() + static_cast<int64_t>(g) * nb);
    }
  });
  ++prepack_stats_.tensors;
  prepack_stats_.source_bytes += n * row_bytes;
  prepack_stats_.packed_bytes += static_cast<int64_t>(p.blocks.size() * sizeof(BlockQ4_Kx8));
  prepack_stats_.ms += static_cast<double>(now_ns() - t0) * 1e-6;
  packed_.emplace(w.data(), std::move(p));
}

uint64_t CpuDevice::experimental_rows8_dispatches() { return g_packed_rows8_dispatches.load(std::memory_order_relaxed); }
uint64_t CpuDevice::experimental_int16_prefill_calls() { return g_int16_prefill_calls.load(std::memory_order_relaxed); }

void CpuDevice::int8_rows(const Int8Kernel& ik, const std::byte* wbase, int64_t w_row_bytes, const ActBlockQ8* act,
                          int64_t m, int64_t k, int64_t n0, int64_t n1, const TensorView& y, const float* bias) const {
  const int64_t nb = k / kActQ8Block;
  // EXPERIMENTAL (KernelPlan::packed_int8_8rows, off by default): more than
  // 4 rows on an interleaved copy go through 8-row tiles, one pass over each
  // packed group instead of two. Same per-row arithmetic as the 4-row tiles.
  if (ik.packed != nullptr && ik.rows8 && m > kDotRowsMax && k_.dot_q8_sx_x8_q4_K_rows8 != nullptr) {
    g_packed_rows8_dispatches.fetch_add(1, std::memory_order_relaxed);
    const ActBlockQ8* xr8[kPackedRowsMax];
    float out[8 * kPackedRowsMax];
    const int64_t sb = k / 256;
    for (int64_t g = n0 / 8; g < n1 / 8; ++g) {
      for (int64_t r0 = 0; r0 < m; r0 += kPackedRowsMax) {
        const int mm = static_cast<int>(std::min<int64_t>(kPackedRowsMax, m - r0));
        for (int t = 0; t < mm; ++t) xr8[t] = act + (r0 + t) * nb;
        k_.dot_q8_sx_x8_q4_K_rows8(ik.packed + g * sb, xr8, mm, k, out);
        for (int t = 0; t < mm; ++t) {
          float* yr = row_ptr<float>(y, r0 + t) + 8 * g;
          for (int i = 0; i < 8; ++i) yr[i] = out[8 * t + i] + (bias ? bias[8 * g + i] : 0.0f);
        }
      }
    }
    return;
  }
  const ActBlockQ8* xr[kDotRowsMax];
  if (ik.packed != nullptr) {
    float out[8 * kDotRowsMax];
    const int64_t sb = k / 256;
    for (int64_t g = n0 / 8; g < n1 / 8; ++g) {
      for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
        const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
        for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
        k_.dot_q8_sx_x8_q4_K(ik.packed + g * sb, xr, mm, k, out);
        for (int t = 0; t < mm; ++t) {
          float* yr = row_ptr<float>(y, r0 + t) + 8 * g;
          for (int i = 0; i < 8; ++i) yr[i] = out[8 * t + i] + (bias ? bias[8 * g + i] : 0.0f);
        }
      }
    }
    return;
  }
  float out[kDotRowsMax];
  for (int64_t r = n0; r < n1; ++r) {
    const float br = bias ? bias[r] : 0.0f;
    for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
      const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
      for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
      ik.dot(wbase + r * w_row_bytes, xr, mm, k, out);
      for (int t = 0; t < mm; ++t) row_ptr<float>(y, r0 + t)[r] = out[t] + br;
    }
  }
}

namespace {
// DYNACORE_Q16_PACKED=0 keeps int16 matmuls on the row kernel (A/B switch).
// Packed Q6_K int16 (DD-082) is opt-in: DYNACORE_Q6_PACKED=1. It saves ~2% of
// a 4-user decode step but did not move end-to-end tok/s outside noise, so by
// default nothing is packed for it and int16 Q6_K stays on the row kernel.
bool q6_packed_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DYNACORE_Q6_PACKED");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
bool q16_packed_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DYNACORE_Q16_PACKED");
    return !(v != nullptr && v[0] == '0');
  }();
  return on;
}
}  // namespace

const BlockQ6_Kx8* CpuDevice::find_packed6(const void* w, int64_t n, int64_t k) const {
  const auto it = packed6_.find(w);
  return it != packed6_.end() && it->second.n == n && it->second.k == k ? it->second.blocks.data() : nullptr;
}

const BlockQ4_Kx8* CpuDevice::find_packed(const void* w, int64_t n, int64_t k) const {
  const auto it = packed_.find(w);
  return it != packed_.end() && it->second.n == n && it->second.k == k ? it->second.blocks.data() : nullptr;
}

CpuDevice::Int8Kernel CpuDevice::int8_from(const MatmulSelection& sel, DType wt, const void* w, int64_t n, int64_t k,
                                           int64_t m) const {
  switch (sel.id) {
    case MatmulKernelId::kInt8PrefillX8: {
      // Prefill-sized batches on the interleaved copy (DD-078, opt-in); 8-row
      // tiles as the plan allows (int8_rows).
      Int8Kernel ik{k_.quantize_act_sx, k_.dot_q8_sx_rows_for(wt), ActFormat::kSubscaled, find_packed(w, n, k)};
      ik.rows8 = plan_.packed_int8_8rows && m > kDotRowsMax && k_.dot_q8_sx_x8_q4_K_rows8 != nullptr &&
                 (plan_.matmul_disabled & matmul_kernel_bit(MatmulKernelId::kInt8SubscaledX8Rows8)) == 0;
      return ik;
    }
    case MatmulKernelId::kInt8SubscaledX8Rows8:
    case MatmulKernelId::kInt8SubscaledX8: {
      Int8Kernel ik{k_.quantize_act_sx, k_.dot_q8_sx_rows_for(wt), ActFormat::kSubscaled, find_packed(w, n, k)};
      ik.rows8 = sel.id == MatmulKernelId::kInt8SubscaledX8Rows8;
      return ik;
    }
    case MatmulKernelId::kInt8Subscaled: return {k_.quantize_act_sx, k_.dot_q8_sx_rows_for(wt), ActFormat::kSubscaled};
    case MatmulKernelId::kInt8Superblock: return {k_.quantize_act_sb, k_.dot_q8_sb_rows_for(wt), ActFormat::kSuperblock};
    case MatmulKernelId::kInt8Per32: return {k_.quantize_act, k_.dot_q8_rows_for(wt), ActFormat::kPer32};
    default: return {};
  }
}

CpuDevice::Int8Kernel CpuDevice::int8_kernel(DType wt, int64_t m, int64_t k, const void* w, int64_t n) const {
  // The int8 family of the matmul registry (DD-053 / 075 / 077 / 078).
  const MatmulShape shape{m, n, k, wt, true};
  return int8_from(select_matmul_int8(shape, plan_, matmul_caps(wt, w, n, k, m)), wt, w, n, k, m);
}

void CpuDevice::matmul_many(std::span<const MatmulJob> jobs) {
  // Grouped execution (P11, DD-060): every job's output rows are cut into
  // chunks of 16 and all chunks of all jobs run in ONE parallel region; each
  // chunk uses its job's own path (int8 rows kernel for <= int8 max rows,
  // fused dequantize-dot for single rows, dequantized 4-row panels through the
  // GEMM tile otherwise). MoE layers issue one call per projection, so a step
  // no longer pays a fork/join per expert per projection.
  if (jobs.empty()) return;
  if (jobs.size() == 1) {
    matmul(jobs[0].x, jobs[0].w, jobs[0].bias, jobs[0].y);
    return;
  }
  if (sme_ != nullptr) {
    // Prefill-sized jobs on the SME panel gemm (one region for all of them);
    // the rest below.
    thread_local std::vector<MatmulJob> sme_jobs, rest;
    sme_jobs.clear();
    rest.clear();
    for (const MatmulJob& j : jobs) {
      const int64_t m = rows(j.x), k = cols(j.x);
      const MatmulShape shape{m, rows(j.w), k, j.w.dtype(), true};
      const MatmulSelection sel = select_matmul(shape, plan_, matmul_caps(j.w.dtype(), j.w.data(), rows(j.w), k, m));
      if (sel.id == MatmulKernelId::kSmeGemm) {
        record_matmul(sel.id, shape, -1);
        sme_jobs.push_back(j);
      } else {
        rest.push_back(j);
      }
    }
    if (!sme_jobs.empty()) {
      const std::vector<MatmulJob> others = rest;  // the recursive call below reuses the thread_locals
      sme_matmul(sme_jobs);
      if (!others.empty()) matmul_many(others);
      return;
    }
  }
  constexpr int64_t kChunk = 16;
  enum class Path : uint8_t { kInt8, kFused, kPanel, kQmm };
  struct JobPlan {
    Path path;
    int64_t act_offset;       // into act_q8_ (int8 path); qmm: index into qmm_acts_
    bool shares_act = false;  // reuses an earlier job's quantized activations
    Int8Kernel int8{};
    bool qmm_gemm = false;
    qmm::ActMode qmm_mode = qmm::ActMode::kInt8;
    const std::byte* qmm_packed = nullptr;
  };
  thread_local std::vector<JobPlan> plans;
  thread_local std::vector<size_t> first_chunk;  // prefix sums of per-job chunk counts
  plans.clear();
  first_chunk.assign(1, 0);
  int64_t act_blocks = 0;
  size_t qmm_buffers = 0;
  bool decode_shaped = true;  // every job has at most qmm::kGemvMaxRows rows
  size_t weight_bytes = 0;
  for (const MatmulJob& j : jobs) {
    const int64_t m = rows(j.x), k = cols(j.x);
    const DType wt = j.w.dtype();
    decode_shaped = decode_shaped && m <= qmm::kGemvMaxRows;
    weight_bytes += static_cast<size_t>(dtype_row_bytes(wt, k) * rows(j.w));
    // Rows may be strided (MoE slices of a wider buffer): quantization works
    // row by row and the GEMM tile takes a row stride, so neither needs dense x.
    JobPlan p{Path::kFused, 0, false};
    const MatmulShape shape{m, rows(j.w), k, wt, true};
    if (qmm_ != nullptr) {
      const MatmulSelection sq = select_matmul(shape, plan_, matmul_caps(wt, j.w.data(), rows(j.w), k, m));
      if (sq.id == MatmulKernelId::kQmmGemm || sq.id == MatmulKernelId::kQmmGemv) {
        record_matmul(sq.id, shape, -1);
        p.path = Path::kQmm;
        p.qmm_gemm = sq.id == MatmulKernelId::kQmmGemm;
        p.qmm_mode = sq.act == MatmulActFormat::kInt16 ? qmm::ActMode::kInt16 : qmm::ActMode::kInt8;
        p.qmm_packed = find_qmm_packed(j.w.data(), rows(j.w), k);
        // Jobs reading the same activations (Q/K/V, gate/up) share one buffer.
        for (size_t prev = 0; prev < plans.size(); ++prev) {
          const MatmulJob& pj = jobs[prev];
          if (plans[prev].path == Path::kQmm && !plans[prev].shares_act && plans[prev].qmm_mode == p.qmm_mode &&
              plans[prev].qmm_gemm == p.qmm_gemm && pj.x.data() == j.x.data() && rows(pj.x) == m && cols(pj.x) == k &&
              pj.x.stride(0) == j.x.stride(0)) {
            p.act_offset = plans[prev].act_offset;
            p.shares_act = true;
            break;
          }
        }
        if (!p.shares_act) p.act_offset = static_cast<int64_t>(qmm_buffers++);
        plans.push_back(p);
        first_chunk.push_back(first_chunk.back() + static_cast<size_t>((rows(j.w) + kChunk - 1) / kChunk));
        continue;
      }
    }
    const MatmulSelection s8 = select_matmul_int8(shape, plan_, matmul_caps(wt, j.w.data(), rows(j.w), k, m));
    const Int8Kernel ik8 = int8_from(s8, wt, j.w.data(), rows(j.w), k, m);
    record_matmul(ik8.dot != nullptr ? s8.id
                  : (m >= plan_.expand_min_rows && wt != DType::kF32) ? MatmulKernelId::kF32ExpandGemm
                                                                      : MatmulKernelId::kF32RowDot,
                  shape, -1);
    if (const Int8Kernel& ik = ik8; ik.dot != nullptr) {
      p = {Path::kInt8, act_blocks, false, ik};
      // Jobs reading the same activations (Q/K/V, gate/up) share one quantized
      // copy when they also use the same activation format.
      bool shared = false;
      for (size_t prev = 0; prev < plans.size() && !shared; ++prev) {
        const MatmulJob& pj = jobs[prev];
        if (plans[prev].path == Path::kInt8 && plans[prev].int8.format == ik.format &&
            pj.x.data() == j.x.data() && rows(pj.x) == m && cols(pj.x) == k && pj.x.stride(0) == j.x.stride(0)) {
          p.act_offset = plans[prev].act_offset;
          p.shares_act = true;
          shared = true;
        }
      }
      if (!shared) act_blocks += m * (k / kActQ8Block);
    } else if (m >= plan_.expand_min_rows && wt != DType::kF32) {
      p.path = Path::kPanel;
    }
    plans.push_back(p);
    first_chunk.push_back(first_chunk.back() + static_cast<size_t>((rows(j.w) + kChunk - 1) / kChunk));
  }
  // Quantize the int8 and qmm jobs' activations once, before the region.
  if (qmm_acts_.size() < qmm_buffers) qmm_acts_.resize(qmm_buffers);
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    if (plans[ji].path != Path::kQmm || plans[ji].shares_act) continue;
    qmm_quantize(qmm_acts_[static_cast<size_t>(plans[ji].act_offset)], jobs[ji].x, plans[ji].qmm_mode, plans[ji].qmm_gemm);
  }
  act_q8_.resize(static_cast<size_t>(act_blocks));
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    if (plans[ji].path != Path::kInt8 || plans[ji].shares_act) continue;
    const int64_t m = rows(jobs[ji].x), k = cols(jobs[ji].x), nb = k / kActQ8Block;
    for (int64_t r = 0; r < m; ++r) {
      plans[ji].int8.quantize(row_ptr<const float>(jobs[ji].x, r), act_q8_.data() + plans[ji].act_offset + r * nb, k);
    }
  }
  const size_t total = first_chunk.back();
  const std::vector<size_t>& offsets = first_chunk;  // read by the workers below
  const std::vector<JobPlan>& job_plans = plans;
  const ActBlockQ8* act = act_q8_.data();
  const size_t grain = decode_shaped ? gemv_grain(total, weight_bytes / std::max<size_t>(total, 1), pool_.size(),
                                                  plan_.matmul_chunks_per_thread, plan_.gemv_min_chunk_bytes)
                                     : grain_for(total, pool_.size(), 1, plan_.matmul_chunks_per_thread);
  pool_.parallel_for(total, grain, [&](size_t begin, size_t end) {
    thread_local std::vector<float> panel;
    for (size_t c = begin; c < end; ++c) {
      const size_t ji = static_cast<size_t>(std::upper_bound(offsets.begin(), offsets.end(), c) - offsets.begin()) - 1;
      const MatmulJob& j = jobs[ji];
      const int64_t m = rows(j.x), k = cols(j.x), n = rows(j.w);
      const DType wt = j.w.dtype();
      const int64_t w_row_bytes = dtype_row_bytes(wt, k);
      const auto* wbase = static_cast<const std::byte*>(j.w.data());
      const int64_t n0 = static_cast<int64_t>(c - offsets[ji]) * kChunk, n1 = std::min(n, n0 + kChunk);
      switch (job_plans[ji].path) {
        case Path::kQmm: {
          // Chunks start at multiples of kChunk (16): whole qmm row groups.
          qmm_columns(qmm_acts_[static_cast<size_t>(job_plans[ji].act_offset)].acts(), job_plans[ji].qmm_gemm, j.w,
                      job_plans[ji].qmm_packed, j.y, nullptr, n0, n1);
          break;
        }
        case Path::kInt8: {
          // Chunks start at multiples of kChunk (16), so packed groups align.
          int8_rows(job_plans[ji].int8, wbase, w_row_bytes, act + job_plans[ji].act_offset, m, k, n0, n1, j.y,
                    nullptr);
          break;
        }
        case Path::kFused: {
          const VecDotFn vec_dot = k_.vec_dot_for(wt);
          for (int64_t r = n0; r < n1; ++r) {
            for (int64_t i = 0; i < m; ++i) {
              row_ptr<float>(j.y, i)[r] = vec_dot(wbase + r * w_row_bytes, row_ptr<const float>(j.x, i), k);
            }
          }
          break;
        }
        case Path::kPanel: {
          const DequantFn dequant = k_.dequant_for(wt);
          constexpr int64_t kPanel = 4;
          panel.resize(static_cast<size_t>(kPanel * k));
          const int64_t y_stride = j.y.stride(0) / static_cast<int64_t>(sizeof(float));
          for (int64_t j0 = n0; j0 < n1; j0 += kPanel) {
            const int nr = static_cast<int>(std::min<int64_t>(kPanel, n1 - j0));
            for (int r = 0; r < nr; ++r) dequant(wbase + (j0 + r) * w_row_bytes, panel.data() + r * k, k);
            k_.gemm_panel(panel.data(), nr, j.x.data_as<const float>(), j.x.stride(0) / static_cast<int64_t>(sizeof(float)),
                          m, k, j.y.data_as<float>() + j0, y_stride, /*accumulate=*/false);
          }
          break;
        }
      }
      if (j.bias != nullptr) {
        const float* bias = j.bias->data_as<const float>();
        for (int64_t i = 0; i < m; ++i) {
          float* yr = row_ptr<float>(j.y, i);
          for (int64_t r = n0; r < n1; ++r) yr[r] += bias[r];
        }
      }
    }
  });
}

MatmulCaps CpuDevice::matmul_caps(DType wt, const void* w, int64_t n, int64_t k, int64_t m) const {
  (void)m;
  MatmulCaps c;
  c.int8_accelerated = int8_accelerated_;
  c.packed_q4 = wt == DType::kQ4_K && w != nullptr && find_packed(w, n, k) != nullptr;
  c.packed_q6 = wt == DType::kQ6_K && w != nullptr && find_packed6(w, n, k) != nullptr;
  c.q16_packed_enabled = q16_packed_enabled();
  c.q6_packed_enabled = q6_packed_enabled();
  c.quantize_act = k_.quantize_act != nullptr;
  c.quantize_act_sb = k_.quantize_act_sb != nullptr;
  c.quantize_act_sx = k_.quantize_act_sx != nullptr;
  c.quantize_act16 = k_.quantize_act16 != nullptr;
  c.dot_q8_rows = k_.dot_q8_rows_for(wt) != nullptr;
  c.dot_q8_sb_rows = k_.dot_q8_sb_rows_for(wt) != nullptr;
  c.dot_q8_sx_rows = k_.dot_q8_sx_rows_for(wt) != nullptr;
  c.dot_q8_sx_x8_rows8 = k_.dot_q8_sx_x8_q4_K_rows8 != nullptr;
  c.dot_q16_x8_q4 = k_.dot_q16_x8_q4_K != nullptr;
  c.dot_q16_x8_q6 = k_.dot_q16_x8_q6_K != nullptr;
  c.dot_q16_x8_q4_rows8 = k_.dot_q16_x8_q4_K_rows8 != nullptr;
  c.dot_q16_x8_q6_rows8 = k_.dot_q16_x8_q6_K_rows8 != nullptr;
  c.dot_q16s_x8_q4_rows8 = k_.dot_q16s_x8_q4_K_rows8 != nullptr;
  c.split_act16 = k_.split_act16 != nullptr;
  c.dot_q16_rows = k_.dot_q16_rows_for(wt) != nullptr;
  if (qmm_ != nullptr && qmm::supports(wt) && k % qmm::kBlock == 0) {
    c.qmm_int8 = true;
    c.qmm_int16 = qmm_->int16;
    c.qmm_gemm = qmm_->gemm != nullptr;
    c.sme_f32 = sme_ != nullptr;
  }
  return c;
}

void CpuDevice::sme_matmul(std::span<const MatmulJob> jobs) {
  constexpr int64_t P = qmm::kSmePanel;
  // Activation panels: one set per distinct x (Q/K/V and gate/up share theirs).
  thread_local std::vector<size_t> x_of;    // per job: index of its x in xs
  thread_local std::vector<size_t> x_base;  // per distinct x: offset into sme_x_
  thread_local std::vector<size_t> xs;      // per distinct x: its first job
  x_of.assign(jobs.size(), 0);
  x_base.clear();
  xs.clear();
  size_t floats = 0;
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    const MatmulJob& j = jobs[ji];
    size_t found = xs.size();
    for (size_t q = 0; q < xs.size(); ++q) {
      const MatmulJob& o = jobs[xs[q]];
      if (o.x.data() == j.x.data() && rows(o.x) == rows(j.x) && cols(o.x) == cols(j.x) && o.x.stride(0) == j.x.stride(0)) {
        found = q;
        break;
      }
    }
    if (found == xs.size()) {
      xs.push_back(ji);
      x_base.push_back(floats);
      floats += static_cast<size_t>((rows(j.x) + P - 1) / P * P * cols(j.x));
    }
    x_of[ji] = found;
  }
  if (sme_x_.size() < floats) sme_x_.resize(floats);
  // The workers read the caller's vectors through these (a thread_local named
  // inside the lambdas would be the worker's own, empty, instance).
  const std::vector<size_t>& job_x = x_of;
  const std::vector<size_t>& base = x_base;
  const std::vector<size_t>& first_job = xs;
  // Pack: one work item per (distinct x, 32-row panel).
  thread_local std::vector<size_t> pack_first;
  pack_first.assign(1, 0);
  for (size_t q : xs) pack_first.push_back(pack_first.back() + static_cast<size_t>((rows(jobs[q].x) + P - 1) / P));
  const std::vector<size_t>& pack_items = pack_first;
  pool_.parallel_for(pack_items.back(), 1, [&](size_t begin, size_t end) {
    for (size_t it = begin; it < end; ++it) {
      const size_t q = static_cast<size_t>(std::upper_bound(pack_items.begin(), pack_items.end(), it) - pack_items.begin()) - 1;
      const TensorView& x = jobs[first_job[q]].x;
      const int64_t m0 = static_cast<int64_t>(it - pack_items[q]) * P, k = cols(x);
      sme_->pack_x(x.data_as<const float>(), x.stride(0) / static_cast<int64_t>(sizeof(float)), rows(x), k, m0,
                   sme_x_.data() + base[q] + static_cast<size_t>(m0 * k));
    }
  });
  // Multiply: one work item per (job, 32-column weight panel).
  thread_local std::vector<size_t> first;
  thread_local std::vector<const std::byte*> packed;
  first.assign(1, 0);
  packed.assign(jobs.size(), nullptr);
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    const MatmulJob& j = jobs[ji];
    packed[ji] = find_qmm_packed(j.w.data(), rows(j.w), cols(j.w));
    first.push_back(first.back() + static_cast<size_t>((rows(j.w) + P - 1) / P));
  }
  const std::vector<size_t>& items = first;
  const std::vector<const std::byte*>& job_packed = packed;
  pool_.parallel_for(items.back(), 1, [&](size_t begin, size_t end) {
    thread_local std::vector<float> wp;  // this worker's weight panel
    for (size_t it = begin; it < end; ++it) {
      const size_t ji = static_cast<size_t>(std::upper_bound(items.begin(), items.end(), it) - items.begin()) - 1;
      const MatmulJob& j = jobs[ji];
      const int64_t m = rows(j.x), k = cols(j.x), n = rows(j.w), n0 = static_cast<int64_t>(it - items[ji]) * P;
      const int cols_here = static_cast<int>(std::min<int64_t>(P, n - n0));
      if (wp.size() < static_cast<size_t>(k * P)) wp.resize(static_cast<size_t>(k * P));
      qmm::weight_panel_f32(j.w.dtype(), j.w.data(), dtype_row_bytes(j.w.dtype(), k), job_packed[ji], n, k, n0, wp.data());
      const float* xp = sme_x_.data() + base[job_x[ji]];
      float* y = j.y.data_as<float>();
      const int64_t ys = j.y.stride(0) / static_cast<int64_t>(sizeof(float));
      for (int64_t m0 = 0; m0 < m; m0 += P) {
        sme_->block(xp + m0 * k, wp.data(), k, y + m0 * ys + n0, ys, static_cast<int>(std::min<int64_t>(P, m - m0)), cols_here);
      }
      if (j.bias != nullptr) {
        const float* b = j.bias->data_as<const float>() + n0;
        for (int64_t r = 0; r < m; ++r) {
          float* yr = y + r * ys + n0;
          for (int c = 0; c < cols_here; ++c) yr[c] += b[c];
        }
      }
    }
  });
}

int CpuDevice::external_for(const MatmulShape& s, const void* w) const {
  for (size_t i = 0; i < external_.size(); ++i) {
    if (external_[i].eligible(s, w)) return static_cast<int>(i);
  }
  return -1;
}

Status CpuDevice::register_external_matmul(ExternalMatmulKernel kernel) {
  if (kernel.name.empty()) return InvalidArgument("external matmul kernel needs a name");
  if (!kernel.eligible || !kernel.run) return InvalidArgument("external matmul kernel '" + kernel.name + "' needs eligible and run");
  for (const ExternalMatmulKernel& e : external_) {
    if (e.name == kernel.name) return Status(StatusCode::kAlreadyExists, "external matmul kernel '" + kernel.name + "' already registered");
  }
  external_.push_back(std::move(kernel));
  return Status::Ok();
}

MatmulSelection CpuDevice::plan_matmul(const TensorView& x, const TensorView& w, bool explain) const {
  const int64_t m = rows(x), k = cols(x), n = rows(w);
  const MatmulShape shape{m, n, k, w.dtype(), x.stride(0) == k * static_cast<int64_t>(sizeof(float))};
  MatmulCaps caps = matmul_caps(w.dtype(), w.data(), n, k, m);
  caps.external_eligible = plan_.external_matmul && external_for(shape, w.data()) >= 0;
  return select_matmul(shape, plan_, caps, explain);
}

std::vector<MatmulKernelStats> CpuDevice::matmul_stats() {
  std::vector<MatmulKernelStats> out;
  for (size_t i = 0; i < kMatmulKernelCount; ++i) {
    out.push_back({static_cast<MatmulKernelId>(i), g_mm_calls[i].load(std::memory_order_relaxed), g_mm_rows[i].load(std::memory_order_relaxed),
                   g_mm_macs[i].load(std::memory_order_relaxed), g_mm_ns[i].load(std::memory_order_relaxed)});
  }
  return out;
}

std::vector<MatmulShapeStats> CpuDevice::matmul_shape_stats() {
  std::lock_guard<std::mutex> lock(g_mm_shape_mu);
  std::vector<MatmulShapeStats> out;
  out.reserve(g_mm_shapes.size());
  for (const auto& [key, v] : g_mm_shapes) {
    out.push_back({key.id, key.weight, key.n, key.k, key.m, v.first, v.second});
  }
  return out;
}

void CpuDevice::reset_matmul_stats() {
  for (size_t i = 0; i < kMatmulKernelCount; ++i) {
    g_mm_calls[i].store(0, std::memory_order_relaxed);
    g_mm_rows[i].store(0, std::memory_order_relaxed);
    g_mm_macs[i].store(0, std::memory_order_relaxed);
    g_mm_ns[i].store(0, std::memory_order_relaxed);
  }
  std::lock_guard<std::mutex> lock(g_mm_shape_mu);
  g_mm_shapes.clear();
}

void CpuDevice::set_matmul_timing(bool on) { g_mm_timing.store(on, std::memory_order_relaxed); }
bool CpuDevice::matmul_timing() { return g_mm_timing.load(std::memory_order_relaxed); }

void CpuDevice::matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) {
  const int64_t m = rows(x), k = cols(x), n = rows(w);
  assert(cols(w) == k && cols(y) == n && rows(y) == m);
  const DType wt = w.dtype();
  const MatmulShape shape{m, n, k, wt, x.stride(0) == k * static_cast<int64_t>(sizeof(float))};
  // Unsafe metadata never reaches a kernel (it was a debug-only assert).
  if (const Status st = validate_matmul_shape(shape); !st.ok() || cols(w) != k || cols(y) != n || rows(y) != m) {
    std::fprintf(stderr, "dynacore: invalid matmul (x %lld x %lld, w %lld x %lld, y %lld x %lld): %s\n",
                 static_cast<long long>(m), static_cast<long long>(k), static_cast<long long>(n),
                 static_cast<long long>(cols(w)), static_cast<long long>(rows(y)), static_cast<long long>(cols(y)),
                 st.ok() ? "shape mismatch" : st.message().c_str());
    std::abort();
  }
  MatmulCaps caps = matmul_caps(wt, w.data(), n, k, m);
  int ext = -1;
  if (plan_.external_matmul && !external_.empty()) {
    ext = external_for(shape, w.data());
    caps.external_eligible = ext >= 0;
  }
  const bool explain = matmul_explain_enabled();
  const MatmulSelection sel = select_matmul(shape, plan_, caps, explain);
  if (explain) explain_matmul_once(shape, sel);
  const bool timing = g_mm_timing.load(std::memory_order_relaxed);
  const int64_t t0 = timing ? now_ns() : 0;
  run_matmul(sel, ext, x, w, bias, y);
  record_matmul(sel.id, shape, timing ? now_ns() - t0 : -1);
}

void CpuDevice::run_matmul(const MatmulSelection& sel, int ext, const TensorView& x, const TensorView& w, const TensorView* bias,
                           const TensorView& y) {
  const int64_t m = rows(x), k = cols(x), n = rows(w);
  const DType wt = w.dtype();
  const int64_t w_row_bytes = dtype_row_bytes(wt, k);
  const auto* wbase = static_cast<const std::byte*>(w.data());
  const float* b = bias ? bias->data_as<const float>() : nullptr;
  const VecDotFn vec_dot = k_.vec_dot_for(wt);
  const DequantFn dequant = k_.dequant_for(wt);
  const auto dot = k_.dot_f32;

  switch (sel.id) {
    case MatmulKernelId::kExternal: {
      external_[static_cast<size_t>(ext)].run(x, w, bias, y);
      return;
    }
    case MatmulKernelId::kSmeGemm: {
      const MatmulJob job{x, w, y, bias};
      sme_matmul(std::span<const MatmulJob>(&job, 1));
      return;
    }
    case MatmulKernelId::kQmmGemm:
    case MatmulKernelId::kQmmGemv: {
      const bool gemm = sel.id == MatmulKernelId::kQmmGemm;
      const qmm::ActMode mode = sel.act == MatmulActFormat::kInt16 ? qmm::ActMode::kInt16 : qmm::ActMode::kInt8;
      if (qmm_acts_.empty()) qmm_acts_.resize(1);
      qmm_quantize(qmm_acts_[0], x, mode, gemm);
      const qmm::Acts& a = qmm_acts_[0].acts();
      // Work items of 16 output rows (4 qmm row groups).
      constexpr int64_t kUnit = 16;
      const auto items = static_cast<size_t>((n + kUnit - 1) / kUnit);
      const std::byte* packed = find_qmm_packed(w.data(), n, k);
      const size_t grain = gemm ? grain_for(items, pool_.size(), 1, plan_.matmul_chunks_per_thread)
                                : gemv_grain(items, static_cast<size_t>(kUnit * dtype_row_bytes(w.dtype(), k)), pool_.size(),
                                             plan_.matmul_chunks_per_thread, plan_.gemv_min_chunk_bytes);
      pool_.parallel_for(items, grain, [&](size_t begin, size_t end) {
        qmm_columns(a, gemm, w, packed, y, b, static_cast<int64_t>(begin) * kUnit, std::min<int64_t>(n, static_cast<int64_t>(end) * kUnit));
      });
      return;
    }
    case MatmulKernelId::kInt8PrefillX8:
    case MatmulKernelId::kInt8SubscaledX8Rows8:
    case MatmulKernelId::kInt8SubscaledX8:
    case MatmulKernelId::kInt8Subscaled:
    case MatmulKernelId::kInt8Superblock:
    case MatmulKernelId::kInt8Per32: {
      // Decode-shaped batches: int8 activations + integer dot against the packed
      // weights, each weight block unpacked once for up to 4 rows (DD-053).
      // Rows are quantized one by one, so strided x (MoE slices) is fine and gets
      // the same path as in matmul_many.
      const Int8Kernel ik = int8_from(sel, wt, w.data(), n, k, m);
      const int64_t nb = k / kActQ8Block;
      act_q8_.resize(static_cast<size_t>(m * nb));
      for (int64_t r = 0; r < m; ++r) ik.quantize(row_ptr<const float>(x, r), act_q8_.data() + r * nb, k);
      const ActBlockQ8* act = act_q8_.data();
      // Work items: weight rows, or groups of 8 rows for the packed copy.
      const int64_t unit = ik.packed != nullptr ? 8 : 1, items = n / unit;
      pool_.parallel_for(static_cast<size_t>(items),
                         grain_for(static_cast<size_t>(items), pool_.size(), unit == 1 ? 4 : 1, plan_.matmul_chunks_per_thread),
                         [&](size_t begin, size_t end) {
        int8_rows(ik, wbase, w_row_bytes, act, m, k, static_cast<int64_t>(begin) * unit, static_cast<int64_t>(end) * unit, y, b);
      });
      return;
    }
    case MatmulKernelId::kInt16SplitPrefillX8: {
      // Split-byte int16 tiles (DynaLM Next): the same int16 codes as the
      // int16 prefill tiles below, split q = 256 * hi + lo; bit-identical
      // results (test_matmul_registry).
      g_int16_prefill_calls.fetch_add(1, std::memory_order_relaxed);
      const void* pk = find_packed(w.data(), n, k);
      const size_t group_bytes = sizeof(BlockQ4_Kx8) * static_cast<size_t>(k / 256);
      const int64_t nb = k / kActQ8Block;
      act_q16_.resize(static_cast<size_t>(m * nb));
      act_q16s_.resize(static_cast<size_t>(m * nb));
      for (int64_t r = 0; r < m; ++r) k_.quantize_act16(row_ptr<const float>(x, r), act_q16_.data() + r * nb, k);
      k_.split_act16(act_q16_.data(), act_q16s_.data(), m * nb);
      const ActBlockQ16S* act = act_q16s_.data();
      const CpuKernels::DotQ16SX8Fn tile = k_.dot_q16s_x8_q4_K_rows8;
      const int64_t groups = n / 8;
      pool_.parallel_for(static_cast<size_t>(groups), grain_for(static_cast<size_t>(groups), pool_.size(), 1, plan_.matmul_chunks_per_thread),
                         [&](size_t begin, size_t end) {
        float out[8 * kPackedRowsMax];
        const ActBlockQ16S* xr[kPackedRowsMax];
        for (size_t g = begin; g < end; ++g) {
          for (int64_t r0 = 0; r0 < m; r0 += kPackedRowsMax) {
            const int mm = static_cast<int>(std::min<int64_t>(kPackedRowsMax, m - r0));
            for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
            tile(static_cast<const std::byte*>(pk) + g * group_bytes, xr, mm, k, out);
            for (int t = 0; t < mm; ++t) {
              float* yr = row_ptr<float>(y, r0 + t) + 8 * static_cast<int64_t>(g);
              for (int i = 0; i < 8; ++i) yr[i] = out[8 * t + i] + (b ? b[8 * g + i] : 0.0f);
            }
          }
        }
      });
      return;
    }
    case MatmulKernelId::kInt16PrefillX8: {
      // EXPERIMENTAL (KernelPlan::int16_prefill_max_rows, off by default):
      // prefill-sized matmuls (more rows than any decode kernel takes) on an
      // interleaved copy with enough output rows: int16 activations in 8-row
      // tiles instead of fp32 expand. Small or unpacked weights keep expand.
      const void* pk = nullptr;
      CpuKernels::DotQ16X8Fn tile = nullptr;
      size_t group_bytes = 0;
      if (wt == DType::kQ4_K) {
        pk = find_packed(w.data(), n, k);
        tile = k_.dot_q16_x8_q4_K_rows8;
        group_bytes = sizeof(BlockQ4_Kx8) * static_cast<size_t>(k / 256);
      } else {
        pk = find_packed6(w.data(), n, k);
        tile = k_.dot_q16_x8_q6_K_rows8;
        group_bytes = sizeof(BlockQ6_Kx8) * static_cast<size_t>(k / 256);
      }
      g_int16_prefill_calls.fetch_add(1, std::memory_order_relaxed);
      const int64_t nb = k / kActQ8Block;
      act_q16_.resize(static_cast<size_t>(m * nb));
      for (int64_t r = 0; r < m; ++r) k_.quantize_act16(row_ptr<const float>(x, r), act_q16_.data() + r * nb, k);
      const ActBlockQ16* act = act_q16_.data();
      const int64_t groups = n / 8;
      pool_.parallel_for(static_cast<size_t>(groups), grain_for(static_cast<size_t>(groups), pool_.size(), 1, plan_.matmul_chunks_per_thread),
                         [&](size_t begin, size_t end) {
        float out[8 * kPackedRowsMax];
        const ActBlockQ16* xr[kPackedRowsMax];
        for (size_t g = begin; g < end; ++g) {
          for (int64_t r0 = 0; r0 < m; r0 += kPackedRowsMax) {
            const int mm = static_cast<int>(std::min<int64_t>(kPackedRowsMax, m - r0));
            for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
            tile(static_cast<const std::byte*>(pk) + g * group_bytes, xr, mm, k, out);
            for (int t = 0; t < mm; ++t) {
              float* yr = row_ptr<float>(y, r0 + t) + 8 * static_cast<int64_t>(g);
              for (int i = 0; i < 8; ++i) yr[i] = out[8 * t + i] + (b ? b[8 * g + i] : 0.0f);
            }
          }
        }
      });
      return;
    }
    case MatmulKernelId::kInt16X8: {
      // int16 activations (DD-076): inputs too outlier-heavy for int8 (the FFN
      // down projection), when the plan asks for it. A Q4_K weight with an
      // interleaved copy uses it (DD-081) up to int8_superblock_max_rows rows: it
      // stays ahead of fp32 expand there, the row kernel only up to 4.
      const void* packed16 = nullptr;
      CpuKernels::DotQ16X8Fn dot16x8 = nullptr;
      size_t group16_bytes = 0;
      if (wt == DType::kQ4_K) {
        packed16 = find_packed(w.data(), n, k);
        dot16x8 = k_.dot_q16_x8_q4_K;
        group16_bytes = sizeof(BlockQ4_Kx8) * static_cast<size_t>(k / 256);
      } else {
        packed16 = find_packed6(w.data(), n, k);
        dot16x8 = k_.dot_q16_x8_q6_K;
        group16_bytes = sizeof(BlockQ6_Kx8) * static_cast<size_t>(k / 256);
      }
      const int64_t nb = k / kActQ8Block;
      act_q16_.resize(static_cast<size_t>(m * nb));
      for (int64_t r = 0; r < m; ++r) k_.quantize_act16(row_ptr<const float>(x, r), act_q16_.data() + r * nb, k);
      const ActBlockQ16* act = act_q16_.data();
      const int64_t groups = n / 8;
      pool_.parallel_for(static_cast<size_t>(groups), grain_for(static_cast<size_t>(groups), pool_.size(), 1, plan_.matmul_chunks_per_thread),
                         [&](size_t begin, size_t end) {
        float out[8 * kDotRowsMax];
        const ActBlockQ16* xr[kDotRowsMax];
        for (size_t g = begin; g < end; ++g) {
          for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
            const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
            for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
            dot16x8(static_cast<const std::byte*>(packed16) + g * group16_bytes, xr, mm, k, out);
            for (int t = 0; t < mm; ++t) {
              float* yr = row_ptr<float>(y, r0 + t) + 8 * static_cast<int64_t>(g);
              for (int i = 0; i < 8; ++i) yr[i] = out[8 * t + i] + (b ? b[8 * g + i] : 0.0f);
            }
          }
        }
      });
      return;
    }
    case MatmulKernelId::kInt16Rows: {
      const DotQ16RowsFn dot16 = k_.dot_q16_rows_for(wt);
      const int64_t nb = k / kActQ8Block;
      act_q16_.resize(static_cast<size_t>(m * nb));
      for (int64_t r = 0; r < m; ++r) k_.quantize_act16(row_ptr<const float>(x, r), act_q16_.data() + r * nb, k);
      const ActBlockQ16* act = act_q16_.data();
      pool_.parallel_for(static_cast<size_t>(n), grain_for(static_cast<size_t>(n), pool_.size(), 4, plan_.matmul_chunks_per_thread),
                         [&](size_t begin, size_t end) {
        float out[kDotRowsMax];
        const ActBlockQ16* xr[kDotRowsMax];
        for (size_t j = begin; j < end; ++j) {
          const std::byte* src = wbase + static_cast<int64_t>(j) * w_row_bytes;
          const float bj = b ? b[j] : 0.0f;
          for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
            const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
            for (int t = 0; t < mm; ++t) xr[t] = act + (r0 + t) * nb;
            dot16(src, xr, mm, k, out);
            for (int t = 0; t < mm; ++t) row_ptr<float>(y, r0 + t)[j] = out[t] + bj;
          }
        }
      });
      return;
    }
    case MatmulKernelId::kF32ExpandGemm: {
      // Prefill path: panels of kPanel weight rows are expanded to fp32 once and
      // multiplied against all activation rows by the register-blocked kernel.
      constexpr int64_t kPanel = 4;
      const size_t panels = static_cast<size_t>((n + kPanel - 1) / kPanel);
      const int64_t y_stride = y.stride(0) / static_cast<int64_t>(sizeof(float));
      const auto* xp = x.data_as<const float>();
      auto* yp = y.data_as<float>();
      // K-blocking (gemm_k_block > 0): slices of gemm_k_block columns, so a slice of
      // all activation rows stays cache-resident while a thread sweeps its
      // panels; partial sums accumulate into y. Slices start on multiples of
      // 256, so every block format dequantizes slice by slice.
      const int64_t kc = (plan_.gemm_k_block > 0 && plan_.gemm_k_block < k) ? plan_.gemm_k_block : k;
      pool_.parallel_for(panels, grain_for(panels, pool_.size(), 1, plan_.matmul_chunks_per_thread), [&](size_t begin, size_t end) {
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
    case MatmulKernelId::kF32RowDot:
    case MatmulKernelId::kCount:
      break;
  }

  // Few activation rows (decode): fused dequantize-dot straight from the
  // packed weights. Many rows with strided x: expand each weight row once and
  // reuse it for every activation row.
  const bool expand = wt != DType::kF32 && m >= plan_.expand_min_rows;
  pool_.parallel_for(static_cast<size_t>(n), grain_for(static_cast<size_t>(n), pool_.size(), 4, plan_.matmul_chunks_per_thread),
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

void CpuDevice::rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) {
  const int64_t m = rows(x), d = cols(x);
  const float* wv = weight.data_as<const float>();
  // Several short rows per task (per-head q/k norms are rows of head_dim).
  const auto grain = static_cast<size_t>(std::max<int64_t>(1, 8192 / std::max<int64_t>(d, 1)));
  pool_.parallel_for(static_cast<size_t>(m), grain, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      const float* xr = row_ptr<const float>(x, static_cast<int64_t>(r));
      float* yr = row_ptr<float>(y, static_cast<int64_t>(r));
      double ss = 0;  // double: long rows of large activations
      int64_t i0 = 0;
#if defined(__aarch64__)
      float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0);
      for (; i0 + 4 <= d; i0 += 4) {
        const float32x4_t v = vld1q_f32(xr + i0);
        const float64x2_t lo = vcvt_f64_f32(vget_low_f32(v)), hi = vcvt_high_f64_f32(v);
        s0 = vfmaq_f64(s0, lo, lo);
        s1 = vfmaq_f64(s1, hi, hi);
      }
      ss = vaddvq_f64(vaddq_f64(s0, s1));
#endif
      for (; i0 < d; ++i0) ss += static_cast<double>(xr[i0]) * xr[i0];
      const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + eps));
      for (int64_t i = 0; i < d; ++i) yr[i] = xr[i] * inv * wv[i];
    }
  });
}

void CpuDevice::layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
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

void CpuDevice::rope(const TensorView& x, int32_t num_heads, int32_t head_dim, std::span<const int32_t> positions,
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
        const float c = std::cos(theta) * rope.attn_factor, s = std::sin(theta) * rope.attn_factor;
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

void CpuDevice::kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                          std::span<const int32_t> row_seq, std::span<const KvLayerView> kv_views) {
  // Rows write disjoint KV slots (one position each), so prompt chunks split
  // across the pool; decode rows stay on the calling thread.
  const size_t nrows = positions.size();
  const size_t grain = nrows >= 64 ? 16 : nrows;
  pool_.parallel_for(nrows, std::max<size_t>(1, grain), [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
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
        } else if (g.dtype == DType::kQ8_0) {  // DD-074
          quant::quantize_q8_0(ks, static_cast<quant::BlockQ8_0*>(kv.k_at(ko)), g.head_dim / quant::kQK);
          quant::quantize_q8_0(vs, static_cast<quant::BlockQ8_0*>(kv.v_at(vo)), g.head_dim_v / quant::kQK);
        } else {
          std::memcpy(static_cast<float*>(kv.k) + ko, ks, sizeof(float) * static_cast<size_t>(g.head_dim));
          std::memcpy(static_cast<float*>(kv.v) + vo, vs, sizeof(float) * static_cast<size_t>(g.head_dim_v));
        }
      }
    }
  });
}

// Unnormalized attention over key positions [t0, t1) of one (row, head):
// acc = sum_t exp(s_t - mx) * v_t, with mx = max_t s_t and sum = sum_t exp(s_t - mx).
// A full softmax is acc / sum; partials over disjoint ranges merge exactly
// with log-sum-exp rescaling (split-K / flash-decoding).
// GQA-grouped attention (DD-054): one pass over [t0, t1) of one KV head serves
// all `G` query heads that share it. Each K/V vector is fetched from memory
// once and reused from L1 for the group, instead of once per query head. Per
// head, the arithmetic (the same dot/axpy primitives, positions in ascending
// order) is exactly what a per-head pass computes, so results are identical.
// Covers heads h0 .. h0 + count - 1 of the group (a sub-group when there are
// fewer (row, KV head) units than threads). acc: count * head_dim_v floats
// (head gi at acc + gi * hdv); mx/sum: count entries.
void CpuDevice::attend_group(const AttentionParams& p, size_t r, int32_t kvh, int32_t group, int32_t h0,
                              int32_t count, int64_t t0, int64_t t1, float* acc, float* mx_out, double* sum_out) {
  const KvGeometry& g = *p.kv.front().geom;
  const int32_t hd = g.head_dim, hdv = g.head_dim_v;
  const KvLayerView& kv = p.kv[static_cast<size_t>(p.row_seq[r])];
  const float* q0 = row_ptr<const float>(p.q, static_cast<int64_t>(r)) + (static_cast<int64_t>(kvh) * group + h0) * hd;
  group = count;  // from here on, the heads this call covers
  const bool f32 = g.dtype == DType::kF32;
  const auto len = static_cast<size_t>(t1 - t0);
  thread_local std::vector<float> scores;  // [group][len]
  scores.resize(static_cast<size_t>(group) * len);

  // KV positions are contiguous within a block of one head: walk block runs
  // and hand each run to the block kernels (DD-056). Results depend only on
  // the row's own data (never on the batch), so batching stays invariant.
  const int32_t bs = g.block_size;
  // With several heads sharing the KV head, an fp16 run is converted to fp32
  // once and every head reads the fp32 copy (one conversion per group, not
  // one per head).
  // q8_0 KV (DD-074) is always converted: there are no q8 attention kernels.
  const bool q8 = g.dtype == DType::kQ8_0;
  // fp16 KV with grouped kernels that read fp16 directly: no conversion.
  const bool f16_heads = !f32 && !q8 && group > 1 && plan_.grouped_attention && k_.attn_scores_heads_f16 != nullptr &&
                         k_.attn_accum_heads_f16 != nullptr;
  const bool convert = !f32 && !f16_heads && (group > 1 || q8);
  const DequantFn to_f32 = k_.dequant_for(q8 ? DType::kQ8_0 : DType::kF16);
  thread_local std::vector<float> kv_f32;
  if (convert) kv_f32.resize(static_cast<size_t>(bs) * std::max(hd, hdv));
  for (int64_t t = t0; t < t1;) {
    const int64_t n = std::min<int64_t>(t1 - t, bs - t % bs);
    const int64_t off = kv.k_offset(t, kvh);
    if (f16_heads) {
      k_.attn_scores_heads_f16(static_cast<const uint16_t*>(kv.k_at(off)), n, hd, q0, group, p.scale,
                               scores.data() + static_cast<size_t>(t - t0), static_cast<int64_t>(len));
      t += n;
      continue;
    }
    if (convert) to_f32(kv.k_at(off), kv_f32.data(), n * hd);
    if ((f32 || convert) && group > 1 && plan_.grouped_attention) {  // all heads per k load (DD-066)
      k_.attn_scores_heads_f32(convert ? kv_f32.data() : static_cast<const float*>(kv.k) + off, n, hd, q0, group,
                               p.scale, scores.data() + static_cast<size_t>(t - t0), static_cast<int64_t>(len));
      t += n;
      continue;
    }
    for (int32_t gi = 0; gi < group; ++gi) {
      const float* q = q0 + static_cast<int64_t>(gi) * hd;
      float* s = scores.data() + static_cast<size_t>(gi) * len + static_cast<size_t>(t - t0);
      if (f32 || convert) {
        k_.attn_scores_f32(convert ? kv_f32.data() : static_cast<const float*>(kv.k) + off, n, hd, q, p.scale, s);
      } else {
        k_.attn_scores_f16(static_cast<const uint16_t*>(kv.k) + off, n, hd, q, p.scale, s);
      }
    }
    t += n;
  }
  for (int32_t gi = 0; gi < group; ++gi) {
    float* s = scores.data() + static_cast<size_t>(gi) * len;
    float mx = -INFINITY;
    for (size_t i = 0; i < len; ++i) {
      if (p.softcap > 0) s[i] = p.softcap * std::tanh(s[i] / p.softcap);
      mx = std::max(mx, s[i]);
    }
    mx_out[gi] = mx;
    sum_out[gi] = 0;
  }
  std::fill(acc, acc + static_cast<size_t>(group) * hdv, 0.0f);
  thread_local std::vector<float> w;  // softmax numerators of one block run, [group][bs]
  w.resize(static_cast<size_t>(group) * static_cast<size_t>(bs));
  for (int64_t t = t0; t < t1;) {
    const int64_t n = std::min<int64_t>(t1 - t, bs - t % bs);
    const int64_t off = kv.v_offset(t, kvh);
    if (convert) to_f32(kv.v_at(off), kv_f32.data(), n * hdv);
    if ((f32 || convert || f16_heads) && group > 1 && plan_.grouped_attention) {  // all heads per v load (DD-066)
      for (int32_t gi = 0; gi < group; ++gi) {
        const float* s = scores.data() + static_cast<size_t>(gi) * len + static_cast<size_t>(t - t0);
        float* wg = w.data() + static_cast<size_t>(gi) * static_cast<size_t>(bs);
        const float m = mx_out[gi];
        for (int64_t j = 0; j < n; ++j) wg[j] = exp_nonpos(s[j] - m);  // vectorizes
        for (int64_t j = 0; j < n; ++j) sum_out[gi] += wg[j];
      }
      if (f16_heads) {
        k_.attn_accum_heads_f16(static_cast<const uint16_t*>(kv.v_at(off)), n, hdv, w.data(), group, bs, acc);
      } else {
        k_.attn_accum_heads_f32(convert ? kv_f32.data() : static_cast<const float*>(kv.v) + off, n, hdv, w.data(), group,
                                bs, acc);
      }
      t += n;
      continue;
    }
    for (int32_t gi = 0; gi < group; ++gi) {
      const float* s = scores.data() + static_cast<size_t>(gi) * len + static_cast<size_t>(t - t0);
      const float m = mx_out[gi];
      for (int64_t j = 0; j < n; ++j) w[static_cast<size_t>(j)] = exp_nonpos(s[j] - m);  // vectorizes
      for (int64_t j = 0; j < n; ++j) sum_out[gi] += w[static_cast<size_t>(j)];
      float* a = acc + static_cast<int64_t>(gi) * hdv;
      if (f32 || convert) {
        k_.attn_accum_f32(convert ? kv_f32.data() : static_cast<const float*>(kv.v) + off, n, hdv, w.data(), a);
      } else {
        k_.attn_accum_f16(static_cast<const uint16_t*>(kv.v) + off, n, hdv, w.data(), a);
      }
    }
    t += n;
  }
}

void CpuDevice::attention(const AttentionParams& p) {
  const KvGeometry& g0 = *p.kv.front().geom;
  const size_t m = p.positions.size();
  if (sme_ != nullptr && plan_.sme_attention_min_rows > 0 && m >= static_cast<size_t>(plan_.sme_attention_min_rows) &&
      g0.head_dim_v % qmm::kSmePanel == 0 && (p.num_heads / g0.num_kv_heads) * qmm::kSmePanel <= 2048) {
    // Prompt chunks (runs of consecutive positions of one sequence) through
    // the SME tiles; every other row through the per-row path below.
    thread_local std::vector<SmeRun> runs;
    thread_local std::vector<size_t> rest;
    runs.clear();
    rest.clear();
    for (size_t r = 0; r < m;) {
      size_t e = r + 1;
      while (e < m && p.row_seq[e] == p.row_seq[r] && p.positions[e] == p.positions[e - 1] + 1) ++e;
      if (e - r >= static_cast<size_t>(plan_.sme_attention_min_rows)) {
        runs.push_back({r, e});
      } else {
        for (size_t i = r; i < e; ++i) rest.push_back(i);
      }
      r = e;
    }
    if (!runs.empty()) {
      const std::vector<SmeRun> run_list = runs;
      const std::vector<size_t> rest_rows = rest;
      attention_sme(p, run_list);
      if (!rest_rows.empty()) attention_rows(p, rest_rows);
      return;
    }
  }
  attention_rows(p, {});
}

namespace {

// panel[d * kSmePanel + j] = src[j * dim + d] for j < n rows of `dim` floats
// (panel already offset to the first row's column).
void transpose_into_panel(const float* src, int64_t n, int32_t dim, float* panel) {
  constexpr int64_t P = qmm::kSmePanel;
  int64_t j = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (; j + 4 <= n; j += 4) {
    int32_t d = 0;
    for (; d + 4 <= dim; d += 4) {
      const float32x4_t r0 = vld1q_f32(src + (j + 0) * dim + d), r1 = vld1q_f32(src + (j + 1) * dim + d);
      const float32x4_t r2 = vld1q_f32(src + (j + 2) * dim + d), r3 = vld1q_f32(src + (j + 3) * dim + d);
      const float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1), t2 = vtrn1q_f32(r2, r3), t3 = vtrn2q_f32(r2, r3);
      const float64x2_t u0 = vreinterpretq_f64_f32(t0), u1 = vreinterpretq_f64_f32(t1);
      const float64x2_t u2 = vreinterpretq_f64_f32(t2), u3 = vreinterpretq_f64_f32(t3);
      vst1q_f32(panel + (d + 0) * P + j, vreinterpretq_f32_f64(vtrn1q_f64(u0, u2)));
      vst1q_f32(panel + (d + 1) * P + j, vreinterpretq_f32_f64(vtrn1q_f64(u1, u3)));
      vst1q_f32(panel + (d + 2) * P + j, vreinterpretq_f32_f64(vtrn2q_f64(u0, u2)));
      vst1q_f32(panel + (d + 3) * P + j, vreinterpretq_f32_f64(vtrn2q_f64(u1, u3)));
    }
    for (; d < dim; ++d) {
      for (int64_t q = 0; q < 4; ++q) panel[d * P + j + q] = src[(j + q) * dim + d];
    }
  }
#endif
  for (; j < n; ++j) {
    for (int32_t d = 0; d < dim; ++d) panel[d * P + j] = src[j * dim + d];
  }
}

}  // namespace

void CpuDevice::attention_sme(const AttentionParams& p, std::span<const SmeRun> runs) {
  constexpr int64_t P = qmm::kSmePanel;
  constexpr int64_t KB = 4 * P;  // keys per softmax step (one O round trip through ZA per step)
  // Upper bound for the shared K/V panels of one slab of KV heads (one head
  // per slab can exceed it at very long contexts).
  constexpr size_t kPanelBytes = size_t{32} << 20;
  const KvGeometry& g = *p.kv.front().geom;
  const int32_t hd = g.head_dim, hdv = g.head_dim_v, nkv = g.num_kv_heads, group = p.num_heads / nkv;
  const bool q8 = g.dtype == DType::kQ8_0, f32 = g.dtype == DType::kF32;
  const DequantFn to_f32 = k_.dequant_for(q8 ? DType::kQ8_0 : DType::kF16);
  // Keys each run reads: [klo, khi), panels of nblk 32-key blocks.
  struct RunKeys {
    int64_t klo, khi, nblk;
    size_t offset;  // of this run's panels in the slab, for KV head 0 of the slab
  };
  std::vector<RunKeys> rk(runs.size());
  size_t per_head = 0;  // floats of every run's panels for one KV head
  for (size_t ri = 0; ri < runs.size(); ++ri) {
    const int64_t first = p.positions[runs[ri].begin], last = p.positions[runs[ri].end - 1];
    RunKeys& k = rk[ri];
    k.klo = p.sliding_window > 0 ? std::max<int64_t>(0, first - p.sliding_window + 1) : 0;
    k.khi = last + 1;
    k.nblk = (k.khi - k.klo + P - 1) / P;
    k.offset = per_head;
    per_head += static_cast<size_t>(k.nblk * P) * static_cast<size_t>(hd + hdv);
  }
  const int32_t slab = static_cast<int32_t>(std::clamp<size_t>(kPanelBytes / (per_head * sizeof(float) + 1), 1, static_cast<size_t>(nkv)));
  sme_kv_panels_.resize(per_head * static_cast<size_t>(slab));
  float* panels = sme_kv_panels_.data();
  // Layout per (KV head of the slab, run): K blocks kp[blk][d * 32 + j], then
  // V by 32-dim chunk vp[c][key][32] (keys contiguous for the PV product).
  auto k_panel = [&](int32_t hs, size_t ri, int64_t blk) {
    return panels + static_cast<size_t>(hs) * per_head + rk[ri].offset + static_cast<size_t>(blk * P * hd);
  };
  auto v_panel = [&](int32_t hs, size_t ri, int32_t c) {
    return panels + static_cast<size_t>(hs) * per_head + rk[ri].offset + static_cast<size_t>(rk[ri].nblk * P * hd) +
           static_cast<size_t>(c) * static_cast<size_t>(rk[ri].nblk * P * P);
  };

  struct Unit {
    size_t run;
    int32_t kvh;
    int64_t a;  // first row of the block, relative to the run
  };
  struct Fill {
    size_t run;
    int32_t kvh;
    int64_t blk;
  };
  std::vector<Unit> units;
  std::vector<Fill> fills;
  for (int32_t h0 = 0; h0 < nkv; h0 += slab) {
    const int32_t h1 = std::min(nkv, h0 + slab);
    // Pass 1: K/V of the slab's heads into the panels, one task per 32-key block.
    fills.clear();
    for (size_t ri = 0; ri < runs.size(); ++ri) {
      for (int32_t h = h0; h < h1; ++h) {
        for (int64_t blk = 0; blk < rk[ri].nblk; ++blk) fills.push_back({ri, h, blk});
      }
    }
    pool_.parallel_for(fills.size(), 4, [&](size_t begin, size_t end) {
      thread_local std::vector<float> kv_row;
      kv_row.resize(static_cast<size_t>(g.block_size) * std::max(hd, hdv));
      for (size_t fi = begin; fi < end; ++fi) {
        const Fill& f = fills[fi];
        const KvLayerView& kv = p.kv[static_cast<size_t>(p.row_seq[runs[f.run].begin])];
        const int32_t hs = f.kvh - h0;
        const int64_t t0 = rk[f.run].klo + f.blk * P, nk = std::min<int64_t>(P, rk[f.run].khi - t0);
        float* kp = k_panel(hs, f.run, f.blk);
        for (int64_t t = t0; t < t0 + nk;) {
          const int64_t n = std::min<int64_t>(t0 + nk - t, g.block_size - t % g.block_size);
          const int64_t ko = kv.k_offset(t, f.kvh), vo = kv.v_offset(t, f.kvh);
          const float* kr = f32 ? static_cast<const float*>(kv.k) + ko : kv_row.data();
          if (!f32) to_f32(kv.k_at(ko), kv_row.data(), n * hd);
          transpose_into_panel(kr, n, hd, kp + (t - t0));
          const float* vr = f32 ? static_cast<const float*>(kv.v) + vo : kv_row.data();
          if (!f32) to_f32(kv.v_at(vo), kv_row.data(), n * hdv);
          for (int32_t c = 0; c < hdv / P; ++c) {
            float* vp = v_panel(hs, f.run, c) + (t - rk[f.run].klo) * P;
            for (int64_t j = 0; j < n; ++j) std::memcpy(vp + j * P, vr + j * hdv + c * P, P * sizeof(float));
          }
          t += n;
        }
        for (int64_t j = nk; j < P; ++j) {  // padding keys: zero (and masked)
          for (int32_t d = 0; d < hd; ++d) kp[d * P + j] = 0.0f;
          for (int32_t c = 0; c < hdv / P; ++c) std::fill_n(v_panel(hs, f.run, c) + (t0 - rk[f.run].klo + j) * P, P, 0.0f);
        }
      }
    });

    // Pass 2: (run, KV head, block of 32 query rows), the longest contexts first.
    units.clear();
    for (size_t ri = 0; ri < runs.size(); ++ri) {
      const int64_t len = static_cast<int64_t>(runs[ri].end - runs[ri].begin);
      for (int64_t a = 0; a < len; a += P) {
        for (int32_t h = h0; h < h1; ++h) units.push_back({ri, h, a});
      }
    }
    std::stable_sort(units.begin(), units.end(), [&](const Unit& x, const Unit& y) {
      return p.positions[runs[x.run].begin + static_cast<size_t>(x.a)] > p.positions[runs[y.run].begin + static_cast<size_t>(y.a)];
    });
    pool_.parallel_for(units.size(), 1, [&](size_t begin, size_t end) {
      thread_local std::vector<float> qp, st, o, mrow, lrow;
      qp.resize(static_cast<size_t>(group) * hd * P);
      st.resize(static_cast<size_t>(KB * P));
      o.resize(static_cast<size_t>(group) * P * hdv);
      mrow.resize(static_cast<size_t>(group) * P);
      lrow.resize(static_cast<size_t>(group) * P);
      for (size_t wi = begin; wi < end; ++wi) {
        const Unit& u = units[wi];
        const RunKeys& keys = rk[u.run];
        const int32_t hs = u.kvh - h0;
        const size_t r0 = runs[u.run].begin + static_cast<size_t>(u.a);
        const int rows = static_cast<int>(std::min<int64_t>(P, static_cast<int64_t>(runs[u.run].end - r0)));
        const int64_t pos0 = p.positions[r0], pos1 = p.positions[r0 + static_cast<size_t>(rows) - 1];
        // Q panels, one per head of the group: qp[gi][d * 32 + i], pre-scaled.
        for (int32_t gi = 0; gi < group; ++gi) {
          float* qg = qp.data() + static_cast<size_t>(gi) * hd * P;
          for (int i = 0; i < P; ++i) {
            if (i < rows) {
              const float* q = row_ptr<const float>(p.q, static_cast<int64_t>(r0) + i) + (static_cast<int64_t>(u.kvh) * group + gi) * hd;
              for (int32_t d = 0; d < hd; ++d) qg[d * P + i] = q[d] * p.scale;
            } else {
              for (int32_t d = 0; d < hd; ++d) qg[d * P + i] = 0.0f;
            }
          }
        }
        std::fill(o.begin(), o.end(), 0.0f);
        std::fill(mrow.begin(), mrow.end(), -INFINITY);
        std::fill(lrow.begin(), lrow.end(), 0.0f);
        // This block's keys, from the panel block holding its first key.
        const int64_t lo = p.sliding_window > 0 ? std::max<int64_t>(keys.klo, pos0 - p.sliding_window + 1) : 0;
        const int64_t hi = pos1 + 1;
        for (int64_t t0 = keys.klo + (lo - keys.klo) / P * P; t0 < hi; t0 += KB) {
          const int64_t nk = std::min<int64_t>(KB, hi - t0);
          const int nsub = static_cast<int>((nk + P - 1) / P);  // 32-key blocks in this step
          const int64_t depth = nsub * P;
          const int64_t blk0 = (t0 - keys.klo) / P;
          for (int32_t gi = 0; gi < group; ++gi) {
            // S^T[j][i] = k_j . q_i for the step's keys (keys as rows: the k-major P panel of PV).
            const float* qg = qp.data() + static_cast<size_t>(gi) * hd * P;
            for (int b = 0; b < nsub; ++b) sme_->block(k_panel(hs, u.run, blk0 + b), qg, hd, st.data() + b * P * P, P, P, P);
            float* mg = mrow.data() + static_cast<size_t>(gi) * P;
            float* lg = lrow.data() + static_cast<size_t>(gi) * P;
            // Online softmax over this step, lane i = query row i: every inner
            // loop runs over the 32 contiguous rows (vectorizes).
            int32_t jlo[P], jhi[P];
            for (int i = 0; i < P; ++i) {
              const int64_t qpos = pos0 + i;
              jhi[i] = i < rows ? static_cast<int32_t>(std::clamp<int64_t>(qpos - t0 + 1, 0, nk)) : 0;
              jlo[i] = p.sliding_window > 0 ? static_cast<int32_t>(std::clamp<int64_t>(qpos - p.sliding_window + 1 - t0, 0, depth)) : 0;
            }
            if (p.softcap > 0) {
              for (int64_t j = 0; j < depth * P; ++j) st[static_cast<size_t>(j)] = p.softcap * std::tanh(st[static_cast<size_t>(j)] / p.softcap);
            }
            float mx[P], sum[P], base[P], alpha[P];
            for (int i = 0; i < P; ++i) {
              mx[i] = mg[i];
              sum[i] = 0.0f;
            }
            // Every key of the step precedes every query row and no window:
            // nothing to mask (rows past `rows` are never stored).
            const bool full = p.sliding_window <= 0 && t0 + depth <= pos0 + 1;
            if (full) {
              for (int64_t j = 0; j < depth; ++j) {
                const float* srow = st.data() + j * P;
                for (int i = 0; i < P; ++i) mx[i] = std::max(mx[i], srow[i]);
              }
            } else {
              for (int64_t j = 0; j < depth; ++j) {
                float* srow = st.data() + j * P;
                const auto jj = static_cast<int32_t>(j);
                for (int i = 0; i < P; ++i) {
                  const float v = jj >= jlo[i] && jj < jhi[i] ? srow[i] : -INFINITY;
                  srow[i] = v;
                  mx[i] = std::max(mx[i], v);
                }
              }
            }
            bool rescale = false;
            for (int i = 0; i < P; ++i) {
              // No valid key yet: weights 0, nothing to rescale.
              base[i] = mx[i] == -INFINITY ? 0.0f : mx[i];
              alpha[i] = mx[i] == -INFINITY ? 1.0f : mg[i] == -INFINITY ? 0.0f : exp_nonpos(mg[i] - mx[i]);
              rescale |= alpha[i] != 1.0f;
            }
            if (full) {
              for (int64_t j = 0; j < depth; ++j) {
                float* srow = st.data() + j * P;
                for (int i = 0; i < P; ++i) {
                  const float e = exp_nonpos(srow[i] - base[i]);
                  srow[i] = e;
                  sum[i] += e;
                }
              }
            } else {
              for (int64_t j = 0; j < depth; ++j) {
                float* srow = st.data() + j * P;
                for (int i = 0; i < P; ++i) {
                  const float v = srow[i];
                  const float e = v == -INFINITY ? 0.0f : exp_nonpos(v - base[i]);
                  srow[i] = e;
                  sum[i] += e;
                }
              }
            }
            for (int i = 0; i < P; ++i) {
              lg[i] = lg[i] * alpha[i] + sum[i];
              mg[i] = mx[i];
            }
            float* og = o.data() + static_cast<size_t>(gi) * P * hdv;
            if (rescale) {
              for (int i = 0; i < rows; ++i) {
                const float a = alpha[i];
                if (a == 1.0f) continue;
                float* orow = og + static_cast<int64_t>(i) * hdv;
                for (int32_t d = 0; d < hdv; ++d) orow[d] *= a;
              }
            }
            for (int32_t c = 0; c < hdv / P; ++c) {
              sme_->block_acc(st.data(), v_panel(hs, u.run, c) + (t0 - keys.klo) * P, depth, og + c * P, hdv, rows, P);
            }
          }
        }
        for (int32_t gi = 0; gi < group; ++gi) {
          const float* og = o.data() + static_cast<size_t>(gi) * P * hdv;
          const float* lg = lrow.data() + static_cast<size_t>(gi) * P;
          for (int i = 0; i < rows; ++i) {
            float* out = row_ptr<float>(p.out, static_cast<int64_t>(r0) + i) + (static_cast<int64_t>(u.kvh) * group + gi) * hdv;
            const float inv = lg[i] > 0 ? 1.0f / lg[i] : 0.0f;
            for (int32_t d = 0; d < hdv; ++d) out[d] = og[static_cast<int64_t>(i) * hdv + d] * inv;
          }
        }
      }
    });
  }
}

void CpuDevice::attention_rows(const AttentionParams& p, std::span<const size_t> sel) {
  const KvGeometry& g = *p.kv.front().geom;  // all sequences share the pool geometry
  const int32_t hdv = g.head_dim_v;
  const int32_t nkv = g.num_kv_heads;
  const int32_t group = p.num_heads / nkv;
  // `sel`: the rows to compute (empty = all); index i below is the i-th of them.
  const size_t m = sel.empty() ? p.positions.size() : sel.size();
  auto row_of = [&](size_t i) { return sel.empty() ? i : sel[i]; };
  // Work unit: one query row x one KV head (its whole query-head group).
  const size_t units = m * static_cast<size_t>(nkv);
  auto range_of = [&](size_t r, int64_t& lo, int64_t& hi) {  // r: a row of p (not an index into sel)
    const int64_t pos = p.positions[r];
    lo = p.sliding_window > 0 ? std::max<int64_t>(0, pos - p.sliding_window + 1) : 0;
    hi = pos + 1;
  };
  int64_t longest = 0;
  for (size_t i = 0; i < m; ++i) {
    int64_t lo, hi;
    range_of(row_of(i), lo, hi);
    longest = std::max(longest, hi - lo);
  }

  // Split-K when the units alone cannot occupy the pool (decode) and contexts
  // are long enough to amortize the merge. Strategy from the planner
  // (DD-051); without a plan, the same rule per call.
  const int64_t kChunk = plan_.attention_chunk;
  const AttentionStrategy strategy = p.sliding_window > 0 ? plan_.attention_window : plan_.attention_full;
  const bool split = strategy == AttentionStrategy::kSplitK ||
                     (strategy == AttentionStrategy::kAuto &&
                      attention_should_split(static_cast<int64_t>(units), longest, pool_.size(), plan_.attention_chunk));
  if (!split) {
    // Fewer units than threads (small batches): split each group's query heads
    // into sub-groups so every thread has work; heads within a sub-group still
    // share each K/V fetch.
    const auto threads = static_cast<size_t>(pool_.size());
    const int32_t subgroups =
        units >= threads ? 1 : std::min(group, static_cast<int32_t>((threads + units - 1) / units));
    const int32_t per = (group + subgroups - 1) / subgroups;
    pool_.parallel_for(units * static_cast<size_t>(subgroups), 1, [&](size_t begin, size_t end) {
      thread_local std::vector<float> mx;
      thread_local std::vector<double> sum;
      mx.resize(static_cast<size_t>(group));
      sum.resize(static_cast<size_t>(group));
      for (size_t task = begin; task < end; ++task) {
        const size_t unit = task / static_cast<size_t>(subgroups);
        const auto h0 = static_cast<int32_t>(task % static_cast<size_t>(subgroups)) * per;
        const int32_t count = std::min(per, group - h0);
        if (count <= 0) continue;
        const size_t r = row_of(unit / static_cast<size_t>(nkv));
        const auto kvh = static_cast<int32_t>(unit % static_cast<size_t>(nkv));
        int64_t lo, hi;
        range_of(r, lo, hi);
        // The group's heads are adjacent in the output row.
        float* out = row_ptr<float>(p.out, static_cast<int64_t>(r)) + (static_cast<int64_t>(kvh) * group + h0) * hdv;
        attend_group(p, r, kvh, group, h0, count, lo, hi, out, mx.data(), sum.data());
        for (int32_t gi = 0; gi < count; ++gi) {
          const auto inv = static_cast<float>(1.0 / sum[static_cast<size_t>(gi)]);
          float* o = out + static_cast<int64_t>(gi) * hdv;
          for (int32_t i = 0; i < hdv; ++i) o[i] *= inv;
        }
      }
    });
    return;
  }

  // Pass 1: every (unit, chunk) computes the group's partials into scratch:
  // per head, acc[hdv] followed by max and sum.
  const auto chunks = static_cast<size_t>((longest + kChunk - 1) / kChunk);
  const size_t stride = static_cast<size_t>(hdv) + 2;
  const size_t job_stride = static_cast<size_t>(group) * stride;
  split_scratch_.resize(units * chunks * job_stride);
  float* scratch = split_scratch_.data();
  pool_.parallel_for(units * chunks, 1, [&](size_t begin, size_t end) {
    thread_local std::vector<float> acc, mx;
    thread_local std::vector<double> sum;
    acc.resize(static_cast<size_t>(group) * hdv);
    mx.resize(static_cast<size_t>(group));
    sum.resize(static_cast<size_t>(group));
    for (size_t job = begin; job < end; ++job) {
      const size_t unit = job / chunks, c = job % chunks;
      const size_t r = row_of(unit / static_cast<size_t>(nkv));
      const auto kvh = static_cast<int32_t>(unit % static_cast<size_t>(nkv));
      int64_t lo, hi;
      range_of(r, lo, hi);
      const int64_t t0 = lo + static_cast<int64_t>(c) * kChunk, t1 = std::min(hi, t0 + kChunk);
      float* parts = scratch + job * job_stride;
      if (t0 >= t1) {  // this row's context is shorter than the longest
        for (int32_t gi = 0; gi < group; ++gi) {
          parts[gi * stride + static_cast<size_t>(hdv)] = -INFINITY;
          parts[gi * stride + static_cast<size_t>(hdv) + 1] = 0.0f;
        }
        continue;
      }
      attend_group(p, r, kvh, group, 0, group, t0, t1, acc.data(), mx.data(), sum.data());
      for (int32_t gi = 0; gi < group; ++gi) {
        float* part = parts + gi * stride;
        std::copy_n(acc.data() + static_cast<size_t>(gi) * hdv, hdv, part);
        part[hdv] = mx[static_cast<size_t>(gi)];
        part[hdv + 1] = static_cast<float>(sum[static_cast<size_t>(gi)]);
      }
    }
  });
  // Pass 2: merge each head's partials with log-sum-exp rescaling.
  const size_t pairs = units * static_cast<size_t>(group);
  pool_.parallel_for(pairs, 1, [&](size_t begin, size_t end) {
    for (size_t pair = begin; pair < end; ++pair) {
      const size_t unit = pair / static_cast<size_t>(group);
      const auto gi = static_cast<int32_t>(pair % static_cast<size_t>(group));
      const size_t r = row_of(unit / static_cast<size_t>(nkv));
      const auto kvh = static_cast<int32_t>(unit % static_cast<size_t>(nkv));
      float* out = row_ptr<float>(p.out, static_cast<int64_t>(r)) + (static_cast<int64_t>(kvh) * group + gi) * hdv;
      float gmax = -INFINITY;
      for (size_t c = 0; c < chunks; ++c) {
        gmax = std::max(gmax, scratch[(unit * chunks + c) * job_stride + gi * stride + static_cast<size_t>(hdv)]);
      }
      double total = 0;
      std::fill(out, out + hdv, 0.0f);
      for (size_t c = 0; c < chunks; ++c) {
        const float* part = scratch + (unit * chunks + c) * job_stride + gi * stride;
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

namespace {

// Elementwise work over [m, d] as (row, column chunk) tasks: a decode step has
// one row, so splitting columns is what spreads it over the pool.
constexpr int64_t kEltChunk = 2048;
// Elementwise ops below this many values run on the calling thread.
constexpr int64_t kParallelElems = 1 << 16;

template <typename Fn>
void for_row_chunks(ThreadPool& pool, int64_t m, int64_t d, Fn&& fn) {
  const int64_t chunks = (d + kEltChunk - 1) / kEltChunk;
  const auto tasks = static_cast<size_t>(m * chunks);
  if (tasks <= 1) {
    if (m == 1) fn(int64_t{0}, int64_t{0}, d);
    return;
  }
  pool.parallel_for(tasks, 1, [&](size_t begin, size_t end) {
    for (size_t t = begin; t < end; ++t) {
      const int64_t r = static_cast<int64_t>(t) / chunks, c = static_cast<int64_t>(t) % chunks;
      fn(r, c * kEltChunk, std::min(d, (c + 1) * kEltChunk));
    }
  });
}

template <typename ActFn>
void act_mul_span(ActFn f, const float* g, const float* u, float* o, int64_t n) {
  for (int64_t i = 0; i < n; ++i) o[i] = f(g[i]) * u[i];
}

#if defined(__aarch64__)
// exp_nonpos on 4 lanes (the same reduction and polynomial).
inline float32x4_t exp_nonpos4(float32x4_t x) {
  x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
  const float32x4_t big = vdupq_n_f32(12582912.0f);
  const float32x4_t n = vsubq_f32(vfmaq_f32(big, x, vdupq_n_f32(1.44269504088896341f)), big);
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
  r = vfmsq_f32(r, n, vdupq_n_f32(1.428606765330187e-06f));
  float32x4_t p = vdupq_n_f32(1.0f / 720);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 120), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  const int32x4_t bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(bits));
}

// silu(g) * u, 4 lanes at a time (silu above, vectorized).
void silu_mul_span(const float* g, const float* u, float* o, int64_t n) {
  int64_t i = 0;
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    const float32x4_t x = vld1q_f32(g + i);
    const float32x4_t e = exp_nonpos4(vnegq_f32(vabsq_f32(x)));
    const float32x4_t num = vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.0f)), one, e);
    const float32x4_t sx = vdivq_f32(vmulq_f32(x, num), vaddq_f32(one, e));
    vst1q_f32(o + i, vmulq_f32(sx, vld1q_f32(u + i)));
  }
  for (; i < n; ++i) o[i] = silu(g[i]) * u[i];
}
#endif

}  // namespace

void CpuDevice::act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) {
  const int64_t m = rows(gate), d = cols(gate);
  for_row_chunks(pool_, m, d, [&](int64_t r, int64_t c0, int64_t c1) {
    const float* gr = row_ptr<const float>(gate, r) + c0;
    const float* ur = row_ptr<const float>(up, r) + c0;
    float* o = row_ptr<float>(out, r) + c0;
    // One loop per activation (no switch per element) so each vectorizes.
    switch (act) {
#if defined(__aarch64__)
      case Activation::kSilu: silu_mul_span(gr, ur, o, c1 - c0); break;
#else
      case Activation::kSilu: act_mul_span(silu, gr, ur, o, c1 - c0); break;
#endif
      case Activation::kGelu: act_mul_span(gelu, gr, ur, o, c1 - c0); break;
      case Activation::kGeluTanh: act_mul_span(gelu_tanh, gr, ur, o, c1 - c0); break;
    }
  });
}

void CpuDevice::matmul_gated(Activation act, const TensorView& x, const TensorView& w_gate, const TensorView& w_up,
                             const TensorView& out, const TensorView& gate_scratch, const TensorView& up_scratch) {
  const int64_t m = rows(x), k = cols(x), n = rows(w_gate);
  if (qmm_ != nullptr && rows(w_up) == n) {
    const MatmulShape sg{m, n, k, w_gate.dtype(), true}, su{m, n, k, w_up.dtype(), true};
    const MatmulSelection qg = select_matmul(sg, plan_, matmul_caps(sg.weight, w_gate.data(), n, k, m));
    const MatmulSelection qu = select_matmul(su, plan_, matmul_caps(su.weight, w_up.data(), n, k, m));
    const bool qmm_g = qg.id == MatmulKernelId::kQmmGemm || qg.id == MatmulKernelId::kQmmGemv || qg.id == MatmulKernelId::kSmeGemm;
    const bool qmm_u = qu.id == MatmulKernelId::kQmmGemm || qu.id == MatmulKernelId::kQmmGemv || qu.id == MatmulKernelId::kSmeGemm;
    if (qmm_g && qmm_u && qg.id == MatmulKernelId::kQmmGemv && qu.id == MatmulKernelId::kQmmGemv && qg.act == qu.act) {
      // Decode-sized: both projections per 16-column chunk, the activation
      // applied while the chunk is in L1 (no gate/up round trip).
      record_matmul(qg.id, sg, -1);
      record_matmul(qu.id, su, -1);
      if (qmm_acts_.empty()) qmm_acts_.resize(1);
      qmm_quantize(qmm_acts_[0], x, qg.act == MatmulActFormat::kInt16 ? qmm::ActMode::kInt16 : qmm::ActMode::kInt8, false);
      const qmm::Acts& a = qmm_acts_[0].acts();
      constexpr int64_t kUnit = 16;
      const auto items = static_cast<size_t>((n + kUnit - 1) / kUnit);
      const int64_t gb = dtype_row_bytes(w_gate.dtype(), k), ub = dtype_row_bytes(w_up.dtype(), k);
      const std::byte* pg = find_qmm_packed(w_gate.data(), n, k);
      const std::byte* pu = find_qmm_packed(w_up.data(), n, k);
      const size_t grain = gemv_grain(items, static_cast<size_t>(kUnit * (gb + ub)), pool_.size(), plan_.matmul_chunks_per_thread,
                                      plan_.gemv_min_chunk_bytes);
      pool_.parallel_for(items, grain, [&](size_t begin, size_t end) {
        float g[qmm::kGemvMaxRows * kUnit], u[qmm::kGemvMaxRows * kUnit];
        for (size_t it = begin; it < end; ++it) {
          const int64_t n0 = static_cast<int64_t>(it) * kUnit, nn = std::min<int64_t>(kUnit, n - n0);
          const auto* wg = static_cast<const std::byte*>(w_gate.data()) + n0 * gb;
          const auto* wu = static_cast<const std::byte*>(w_up.data()) + n0 * ub;
          for (int64_t r0 = 0; r0 < m; r0 += qmm::kGemvMaxRows) {
            const int rr = static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, m - r0));
            // Row r0 + t of the tile lands at g[t * kUnit + j] (y offset by -r0 rows).
            if (pg != nullptr && pu != nullptr && nn == kUnit) {
              // Packed copies: groups n0 / 4 .. + 4, outputs at columns 4 g (offset by -n0).
              qmm_->gemv_packed(w_gate.dtype(), pg, k, n0 / 4, n0 / 4 + kUnit / 4, a, r0, rr, g - r0 * kUnit - n0, kUnit, nullptr);
              qmm_->gemv_packed(w_up.dtype(), pu, k, n0 / 4, n0 / 4 + kUnit / 4, a, r0, rr, u - r0 * kUnit - n0, kUnit, nullptr);
            } else {
              qmm_->gemv(w_gate.dtype(), wg, gb, 0, nn, a, r0, rr, g - r0 * kUnit, kUnit, nullptr);
              qmm_->gemv(w_up.dtype(), wu, ub, 0, nn, a, r0, rr, u - r0 * kUnit, kUnit, nullptr);
            }
            for (int t = 0; t < rr; ++t) {
              float* o = row_ptr<float>(out, r0 + t) + n0;
              for (int64_t j = 0; j < nn; ++j) o[j] = apply_act(act, g[t * kUnit + j]) * u[t * kUnit + j];
            }
          }
        }
      });
      return;
    }
    if (qmm_g && qmm_u) {
      // Prefill-sized: both projections in one parallel region, then the activation.
      const MatmulJob jobs[2] = {{x, w_gate, gate_scratch, nullptr}, {x, w_up, up_scratch, nullptr}};
      matmul_many(jobs);
      act_mul(act, gate_scratch, up_scratch, out);
      return;
    }
  }
  const MatmulShape sg{m, n, k, w_gate.dtype(), true}, su{m, rows(w_up), k, w_up.dtype(), true};
  const MatmulSelection selg = select_matmul_int8(sg, plan_, matmul_caps(sg.weight, w_gate.data(), n, k, m));
  const MatmulSelection selu = select_matmul_int8(su, plan_, matmul_caps(su.weight, w_up.data(), su.n, k, m));
  const Int8Kernel ig = int8_from(selg, sg.weight, w_gate.data(), n, k, m), iu = int8_from(selu, su.weight, w_up.data(), su.n, k, m);
  const DotQ8RowsFn dot_gate = ig.dot, dot_up = iu.dot;
  // Fused only where both projections take the int8 decode path (DD-053) with
  // one activation format; otherwise the unfused sequence.
  if (!(dot_gate && dot_up && ig.format == iu.format && rows(w_up) == n)) {
    Device::matmul_gated(act, x, w_gate, w_up, out, gate_scratch, up_scratch);
    return;
  }
  record_matmul(selg.id, sg, -1);
  record_matmul(selu.id, su, -1);
  const int64_t nb = k / kActQ8Block;
  act_q8_.resize(static_cast<size_t>(m * nb));
  for (int64_t r = 0; r < m; ++r) ig.quantize(row_ptr<const float>(x, r), act_q8_.data() + r * nb, k);
  const ActBlockQ8* actq = act_q8_.data();
  const int64_t g_row = dtype_row_bytes(w_gate.dtype(), k), u_row = dtype_row_bytes(w_up.dtype(), k);
  const auto* gbase = static_cast<const std::byte*>(w_gate.data());
  const auto* ubase = static_cast<const std::byte*>(w_up.data());
  if (ig.packed != nullptr && iu.packed != nullptr) {
    // Interleaved copies (DD-078): 8 output channels per work item.
    const int64_t groups = n / 8, sb = k / 256;
    pool_.parallel_for(static_cast<size_t>(groups), grain_for(static_cast<size_t>(groups), pool_.size(), 1, plan_.matmul_chunks_per_thread),
                       [&](size_t begin, size_t end) {
      float g[8 * kDotRowsMax], u[8 * kDotRowsMax];
      const ActBlockQ8* xr[kDotRowsMax];
      for (size_t gi = begin; gi < end; ++gi) {
        for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
          const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
          for (int t = 0; t < mm; ++t) xr[t] = actq + (r0 + t) * nb;
          k_.dot_q8_sx_x8_q4_K(ig.packed + static_cast<int64_t>(gi) * sb, xr, mm, k, g);
          k_.dot_q8_sx_x8_q4_K(iu.packed + static_cast<int64_t>(gi) * sb, xr, mm, k, u);
          for (int t = 0; t < mm; ++t) {
            float* o = row_ptr<float>(out, r0 + t) + 8 * static_cast<int64_t>(gi);
            for (int i = 0; i < 8; ++i) o[i] = apply_act(act, g[8 * t + i]) * u[8 * t + i];
          }
        }
      }
    });
    return;
  }
  pool_.parallel_for(static_cast<size_t>(n), grain_for(static_cast<size_t>(n), pool_.size(), 4, plan_.matmul_chunks_per_thread),
                     [&](size_t begin, size_t end) {
    float g[kDotRowsMax], u[kDotRowsMax];
    const ActBlockQ8* xr[kDotRowsMax];
    for (size_t j = begin; j < end; ++j) {
      for (int64_t r0 = 0; r0 < m; r0 += kDotRowsMax) {
        const int mm = static_cast<int>(std::min<int64_t>(kDotRowsMax, m - r0));
        for (int t = 0; t < mm; ++t) xr[t] = actq + (r0 + t) * nb;
        dot_gate(gbase + static_cast<int64_t>(j) * g_row, xr, mm, k, g);
        dot_up(ubase + static_cast<int64_t>(j) * u_row, xr, mm, k, u);
        for (int t = 0; t < mm; ++t) row_ptr<float>(out, r0 + t)[j] = apply_act(act, g[t]) * u[t];
      }
    }
  });
}

void CpuDevice::activation(Activation act, const TensorView& x, const TensorView& out) {
  const int64_t m = rows(x), d = cols(x);
  for_row_chunks(pool_, m, d, [&](int64_t r, int64_t c0, int64_t c1) {
    const float* xr = row_ptr<const float>(x, r) + c0;
    float* o = row_ptr<float>(out, r) + c0;
    for (int64_t i = 0; i < c1 - c0; ++i) o[i] = apply_act(act, xr[i]);
  });
}

void CpuDevice::add(const TensorView& a, const TensorView& b, const TensorView& y) {
  const int64_t m = rows(a), d = cols(a);
  auto span = [&](int64_t r, int64_t c0, int64_t c1) {
    const float* ar = row_ptr<const float>(a, r);
    const float* br = row_ptr<const float>(b, r);
    float* yr = row_ptr<float>(y, r);
    for (int64_t i = c0; i < c1; ++i) yr[i] = ar[i] + br[i];
  };
  if (m * d < kParallelElems) {  // decode: a parallel region costs more than the work
    for (int64_t r = 0; r < m; ++r) span(r, 0, d);
    return;
  }
  for_row_chunks(pool_, m, d, span);
}

void CpuDevice::scale(const TensorView& x, float s) {
  const int64_t m = rows(x), d = cols(x);
  for (int64_t r = 0; r < m; ++r) {
    float* xr = row_ptr<float>(x, r);
    for (int64_t i = 0; i < d; ++i) xr[i] *= s;
  }
}

void CpuDevice::softcap(const TensorView& x, float cap) {
  const int64_t m = rows(x), d = cols(x);
  for (int64_t r = 0; r < m; ++r) {
    float* xr = row_ptr<float>(x, r);
    for (int64_t i = 0; i < d; ++i) xr[i] = cap * std::tanh(xr[i] / cap);
  }
}

}  // namespace dynacore

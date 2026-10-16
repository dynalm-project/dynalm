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

#include "dynacore/base/fast_exp.h"
#include "dynacore/base/timer.h"
#include "dynacore/hardware/cpu_info.h"
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

CpuDevice::CpuDevice(ThreadPool& pool, CpuIsa isa)
    : pool_(pool), k_(make_cpu_kernels(isa)), name_("CPU/" + std::string(isa_name(k_.isa))) {
  int8_accelerated_ = k_.isa == CpuIsa::kAvx2;  // NEON integer kernels: not written yet
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
  constexpr int64_t kChunk = 16;
  enum class Path : uint8_t { kInt8, kFused, kPanel };
  struct JobPlan {
    Path path;
    int64_t act_offset;       // into act_q8_ (int8 path)
    bool shares_act = false;  // reuses an earlier job's quantized activations
    Int8Kernel int8{};
  };
  thread_local std::vector<JobPlan> plans;
  thread_local std::vector<size_t> first_chunk;  // prefix sums of per-job chunk counts
  plans.clear();
  first_chunk.assign(1, 0);
  int64_t act_blocks = 0;
  for (const MatmulJob& j : jobs) {
    const int64_t m = rows(j.x), k = cols(j.x);
    const DType wt = j.w.dtype();
    // Rows may be strided (MoE slices of a wider buffer): quantization works
    // row by row and the GEMM tile takes a row stride, so neither needs dense x.
    JobPlan p{Path::kFused, 0, false};
    const MatmulShape shape{m, rows(j.w), k, wt, true};
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
  // Quantize the int8 jobs' activations once, before the region.
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
  pool_.parallel_for(total, grain_for(total, pool_.size(), 1, plan_.matmul_chunks_per_thread), [&](size_t begin, size_t end) {
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
  return c;
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

void CpuDevice::kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
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
      } else if (g.dtype == DType::kQ8_0) {  // DD-074
        quant::quantize_q8_0(ks, static_cast<quant::BlockQ8_0*>(kv.k_at(ko)), g.head_dim / quant::kQK);
        quant::quantize_q8_0(vs, static_cast<quant::BlockQ8_0*>(kv.v_at(vo)), g.head_dim_v / quant::kQK);
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
  const bool convert = !f32 && (group > 1 || q8);
  const DequantFn to_f32 = k_.dequant_for(q8 ? DType::kQ8_0 : DType::kF16);
  thread_local std::vector<float> kv_f32;
  if (convert) kv_f32.resize(static_cast<size_t>(bs) * std::max(hd, hdv));
  for (int64_t t = t0; t < t1;) {
    const int64_t n = std::min<int64_t>(t1 - t, bs - t % bs);
    const int64_t off = kv.k_offset(t, kvh);
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
    if ((f32 || convert) && group > 1 && plan_.grouped_attention) {  // all heads per v load (DD-066)
      for (int32_t gi = 0; gi < group; ++gi) {
        const float* s = scores.data() + static_cast<size_t>(gi) * len + static_cast<size_t>(t - t0);
        float* wg = w.data() + static_cast<size_t>(gi) * static_cast<size_t>(bs);
        const float m = mx_out[gi];
        for (int64_t j = 0; j < n; ++j) wg[j] = exp_nonpos(s[j] - m);  // vectorizes
        for (int64_t j = 0; j < n; ++j) sum_out[gi] += wg[j];
      }
      k_.attn_accum_heads_f32(convert ? kv_f32.data() : static_cast<const float*>(kv.v) + off, n, hdv, w.data(), group,
                              bs, acc);
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
  const KvGeometry& g = *p.kv.front().geom;  // all sequences share the pool geometry
  const int32_t hdv = g.head_dim_v;
  const int32_t nkv = g.num_kv_heads;
  const int32_t group = p.num_heads / nkv;
  const size_t m = p.positions.size();
  // Work unit: one query row x one KV head (its whole query-head group).
  const size_t units = m * static_cast<size_t>(nkv);
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
        const size_t r = unit / static_cast<size_t>(nkv);
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
      const size_t r = unit / static_cast<size_t>(nkv);
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
      const size_t r = unit / static_cast<size_t>(nkv);
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

}  // namespace

void CpuDevice::act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) {
  const int64_t m = rows(gate), d = cols(gate);
  for_row_chunks(pool_, m, d, [&](int64_t r, int64_t c0, int64_t c1) {
    const float* gr = row_ptr<const float>(gate, r) + c0;
    const float* ur = row_ptr<const float>(up, r) + c0;
    float* o = row_ptr<float>(out, r) + c0;
    // One loop per activation (no switch per element) so each vectorizes.
    switch (act) {
      case Activation::kSilu: act_mul_span(silu, gr, ur, o, c1 - c0); break;
      case Activation::kGelu: act_mul_span(gelu, gr, ur, o, c1 - c0); break;
      case Activation::kGeluTanh: act_mul_span(gelu_tanh, gr, ur, o, c1 - c0); break;
    }
  });
}

void CpuDevice::matmul_gated(Activation act, const TensorView& x, const TensorView& w_gate, const TensorView& w_up,
                             const TensorView& out, const TensorView& gate_scratch, const TensorView& up_scratch) {
  const int64_t m = rows(x), k = cols(x), n = rows(w_gate);
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
  for (int64_t r = 0; r < m; ++r) {
    const float* ar = row_ptr<const float>(a, r);
    const float* br = row_ptr<const float>(b, r);
    float* yr = row_ptr<float>(y, r);
    for (int64_t i = 0; i < d; ++i) yr[i] = ar[i] + br[i];
  }
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

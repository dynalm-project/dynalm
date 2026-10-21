#pragma once

// Internal to the ggml device (ggml_device.cpp: buffers, running segments,
// registry; ggml_graph.cpp: lowering recorded calls to a ggml graph).
// Not part of DynaCore's API: ggml headers stay private to src/gpu.

#include "dynacore/gpu/ggml_device.h"

#if DYNACORE_HAS_GGML


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "dynacore/device/segment.h"
#include "dynacore/hardware/cpu_info.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpu.h"  // the CPU backend is part of every ggml build
#if DYNACORE_HAS_METAL
#include "ggml-metal.h"
#endif
#if DYNACORE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif
#if DYNACORE_HAS_CUDA
#include "ggml-cuda.h"
#endif


namespace dynacore::ggml_detail {


using Clock = std::chrono::steady_clock;
inline int64_t ns_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
}

inline bool to_ggml_type(DType t, ggml_type& out) {
  switch (t) {
    case DType::kF32: out = GGML_TYPE_F32; return true;
    case DType::kF16: out = GGML_TYPE_F16; return true;
    case DType::kBF16: out = GGML_TYPE_BF16; return true;
    case DType::kI32: out = GGML_TYPE_I32; return true;
    case DType::kQ4_0: out = GGML_TYPE_Q4_0; return true;
    case DType::kQ4_1: out = GGML_TYPE_Q4_1; return true;
    case DType::kQ5_0: out = GGML_TYPE_Q5_0; return true;
    case DType::kQ5_1: out = GGML_TYPE_Q5_1; return true;
    case DType::kQ8_0: out = GGML_TYPE_Q8_0; return true;
    case DType::kQ2_K: out = GGML_TYPE_Q2_K; return true;
    case DType::kQ3_K: out = GGML_TYPE_Q3_K; return true;
    case DType::kQ4_K: out = GGML_TYPE_Q4_K; return true;
    case DType::kQ5_K: out = GGML_TYPE_Q5_K; return true;
    case DType::kQ6_K: out = GGML_TYPE_Q6_K; return true;
    case DType::kIQ4_NL: out = GGML_TYPE_IQ4_NL; return true;
    case DType::kIQ4_XS: out = GGML_TYPE_IQ4_XS; return true;
    default: return false;
  }
}

// The prepared-graph workspace cap (LRU eviction beyond it): 1 GiB, or
// DYNACORE_GPU_GRAPH_MB. Read once.
inline int64_t graph_workspace_cap() {
  static const int64_t cap = [] {
    const char* v = std::getenv("DYNACORE_GPU_GRAPH_MB");
    const long mb = v != nullptr ? std::strtol(v, nullptr, 10) : 0;
    return (mb > 0 ? static_cast<int64_t>(mb) : 1024) << 20;
  }();
  return cap;
}

// A ggml context for tensor metadata only (no data).
struct CtxDeleter {
  void operator()(ggml_context* c) const { ggml_free(c); }
};
using CtxPtr = std::unique_ptr<ggml_context, CtxDeleter>;
inline CtxPtr meta_ctx(size_t tensors, size_t graph_nodes = 0) {
  ggml_init_params p{};
  p.mem_size = ggml_tensor_overhead() * tensors + (graph_nodes ? ggml_graph_overhead_custom(graph_nodes, false) : 0) + 4096;
  p.mem_buffer = nullptr;
  p.no_alloc = true;
  return CtxPtr(ggml_init(p));
}

// One backend buffer handed to the runtime (allocate / upload). `base` is the
// address the runtime sees (its tensors point into [base, base + size));
// `real` is the backend's own address for it. They are equal on Metal. Other
// providers do not give each buffer a unique address (ggml-vulkan returns
// one placeholder for all), so the runtime gets a reserved, inaccessible
// address range per buffer instead (HandleRange) and `real` is used only to
// bind ggml tensors.
struct BufferInfo {
  ggml_backend_buffer_t buf = nullptr;
  std::byte* base = nullptr;
  std::byte* real = nullptr;
  size_t size = 0;
  std::byte* real_at(const std::byte* p) const { return real + (p - base); }
};

// A reserved range of virtual addresses with no access: unique handles for
// device buffers that cannot overlap host memory or each other. A host
// access through one faults instead of reading unrelated memory.
class HandleRange {
 public:
  static std::byte* reserve(size_t bytes) {
#if defined(_WIN32)
    void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    return static_cast<std::byte*>(p);
#else
    void* p = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<std::byte*>(p);
#endif
  }
  static void release(std::byte* p, size_t bytes) {
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
  }
};

// Buffer table shared with the Storage keep-alive handles, so a storage freed
// after the device still releases its buffer safely.
struct BufferTable {
  std::map<uintptr_t, BufferInfo> by_base;
  int64_t weight_bytes = 0, buffer_bytes = 0;
  int64_t mapped_bytes = 0;  // weights used in place from a host file mapping (no copy)
  // Bumped whenever a buffer is freed. Prepared graphs bind buffers, and a
  // new buffer can reuse a freed one's address (same signature), so every
  // prepared graph is dropped when this changes.
  uint64_t free_epoch = 0;
};

// Exact identity of an fp32 2-D activation view: base pointer, rows, cols,
// row stride (bytes).
struct ViewKey {
  const std::byte* p = nullptr;
  int64_t rows = 0, cols = 0, stride = 0;
  friend bool operator==(const ViewKey&, const ViewKey&) = default;
  const std::byte* end() const { return p + (rows - 1) * stride + cols * 4; }
  bool contiguous() const { return stride == cols * 4 || rows == 1; }
  // The same contiguous bytes under another [rows, cols] split (e.g. Q as
  // [rows, heads x head_dim] and as [rows x heads, head_dim]).
  bool same_range(const ViewKey& o) const {
    return p == o.p && contiguous() && o.contiguous() && rows * cols == o.rows * o.cols;
  }
  // Whether the two views share an element. Column slices of one row buffer
  // (Q/K/V in qkv) interleave in address range without sharing elements, so
  // views of equal stride compare rows and columns exactly; others by range.
  bool overlaps(const ViewKey& o) const {
    if (!(p < o.end() && o.p < end())) return false;
    if (stride != o.stride || stride <= 0) return true;
    const ViewKey& a = p <= o.p ? *this : o;
    const ViewKey& b = p <= o.p ? o : *this;
    const int64_t d = b.p - a.p, row = d / stride, col = d % stride;  // b starts at (row, col) of a
    if (col + b.cols * 4 > stride || a.cols * 4 > stride) return true;  // rows wrap: be conservative
    const bool rows_meet = row < a.rows;                                // b's first row lies within a's rows
    const bool cols_meet = col < a.cols * 4;                            // and its columns start inside a's
    // b's columns [col, col + 4 cols) against a's [0, 4 cols); rows [row, row + b.rows) against [0, a.rows).
    return rows_meet && cols_meet;
  }
};

inline int64_t rows_of(const TensorView& t) { return t.rank() > 1 ? t.dim(0) : 1; }
inline int64_t cols_of(const TensorView& t) { return t.dim(t.rank() - 1); }
inline ViewKey key_of(const TensorView& t) {
  const int64_t cols = cols_of(t), rows = rows_of(t);
  return ViewKey{static_cast<const std::byte*>(t.data()), rows, cols, t.rank() > 1 ? t.stride(0) : cols * 4};
}

// Dynamic per-step inputs (docs/gpu-backend.md, section 4).
enum class InKind : uint8_t {
  kIds,        // embedding token ids: I32 [m]
  kPositions,  // rope positions: I32 [m]
  kRows,       // gather rows: I32 [n]
  kKvIndex,    // kv_store row indices: I64 [nkv * m]
  kGatherIdx,  // attention K/V row gather of one sequence: I32 [nkv * L]
  kMask,       // attention mask of one sequence: F16 [L, rows]
  kPagedMask,  // in-place paged attention: F16 [cells, rows], the block tables as a mask
  kConst,      // constant floats copied at build time (rope frequency factors)
};

struct Input {
  InKind kind;
  ggml_tensor* t = nullptr;
  int32_t call = 0;  // representative call
  int32_t seq = 0;   // sequence (kGatherIdx, kMask)
  int32_t r0 = 0, n = 0, span = 0;  // rows of the sequence, padded span L
  std::vector<int32_t> members;     // other calls with identical source data
  std::vector<float> constant;      // kConst
};

struct Prepared {
  std::vector<uint64_t> key;
  CtxPtr ctx, ctx_in;
  ggml_cgraph* graph = nullptr;
  ggml_backend_buffer_t in_buf = nullptr;
  ggml_gallocr_t galloc = nullptr;
  std::vector<Input> inputs;
  int64_t workspace_bytes = 0;
  uint64_t last_use = 0;  // GgmlDevice::use_tick_ at the last run (LRU eviction)
  ~Prepared() {
    if (galloc) ggml_gallocr_free(galloc);
    if (in_buf) ggml_backend_buffer_free(in_buf);
  }
};

// Per sequence: its rows in the batch (contiguous) and its padded KV span.
struct SeqRows {
  int32_t seq = 0, r0 = 0, n = 0, ctx = 0, span = 0;
  // Slot-major: the most KV cells any one row attends to, padded to the
  // bucket (the sparse flash-attention bound; 0 = dense over the span).
  int32_t kv_max = 0;
};

class GgmlDevice final : public Device, public SegmentExecutor {
 public:
  GgmlDevice(ggml_backend_dev_t dev, ggml_backend_t backend, DeviceType type, const GgmlDeviceOptions& opts)
      : dev_(dev), backend_(backend), type_(type), opts_(opts), table_(std::make_shared<BufferTable>()) {
    const char* provider = type == DeviceType::kMetal    ? "Metal/"
                           : type == DeviceType::kVulkan ? "Vulkan/"
                           : type == DeviceType::kCuda   ? "CUDA/"
                           : type == DeviceType::kCpu    ? "ggml-CPU/"
                                                         : "GPU/";
    name_ = std::string(provider) + ggml_backend_dev_description(dev);
    buft_ = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const char* w = std::getenv("DYNACORE_GPU_WEIGHTS");
    // Unique per-buffer addresses from the backend: Metal (shared memory) and
    // the ggml CPU backend (host memory). Elsewhere buffers get handle ranges, and weights are copied (mapping a
    // host range under a handle is not supported).
    real_addresses_ = type == DeviceType::kMetal || type == DeviceType::kCpu;
    zero_copy_weights_ = real_addresses_ && props.caps.buffer_from_host_ptr &&
                         !(w != nullptr && std::string_view(w) == "copy");
  }
  ~GgmlDevice() override {
    // DYNACORE_GPU_STATS=1: where the device's time went (diagnostics).
    if (const char* v = std::getenv("DYNACORE_GPU_STATS"); v != nullptr && v[0] == '1' && stats_.segments > 0) {
      const auto per = [&](int64_t ns) { return static_cast<double>(ns) / 1e3 / static_cast<double>(stats_.segments); };
      std::fprintf(stderr,
                   "[%s] %lld segments: per segment %.1f us host inputs, %.1f us graph compute (submit+wait), "
                   "%.1f us download; %lld graph builds (%.1f ms total), %lld reuses, %lld evictions, %lld errors; "
                   "%lld nodes in the last graph; %lld weights mapped in place; %lld sparse attention calls; "
                   "%zu prepared graphs, %.0f MiB workspace now, %.0f MiB at most\n",
                   name_.c_str(), static_cast<long long>(stats_.segments), per(stats_.input_ns), per(stats_.compute_ns),
                   per(stats_.download_ns), static_cast<long long>(stats_.graph_builds),
                   static_cast<double>(stats_.build_ns) / 1e6, static_cast<long long>(stats_.graph_reuses),
                   static_cast<long long>(stats_.graph_evictions), static_cast<long long>(stats_.errors),
                   static_cast<long long>(stats_.graph_nodes), static_cast<long long>(mapped_weights_),
                   static_cast<long long>(stats_.sparse_attention), graphs_.size(),
                   static_cast<double>(memory_stats().workspace_bytes) / 1048576.0,
                   static_cast<double>(peak_workspace_) / 1048576.0);
    }
    graphs_.clear();
    repacked_.clear();
    for (ggml_backend_buffer_t b : repack_bufs_) ggml_backend_buffer_free(b);
    ggml_backend_free(backend_);
    if (threadpool_ != nullptr) ggml_threadpool_free(threadpool_);  // after the backend that uses it
  }

  // ggml-cpu: a persistent worker pool for every graph (owned from here on).
  void adopt_threadpool(ggml_threadpool* tp) { threadpool_ = tp; }

  // --- Device: identity and memory ---
  std::string_view name() const override { return name_; }
  DeviceLoc device() const override { return DeviceLoc{type_, 0}; }
  bool host_accessible() const override { return false; }
  SegmentExecutor* segment_executor() override { return this; }
  // Slot-major: attention reads the paged pool in place. Head-major: the
  // stage-1 gather path (kept for A/B measurement).
  KvLayout preferred_kv_layout() const override { return KvLayout::kSlotMajor; }
  bool supports_kv_layout(KvLayout) const override { return true; }

  Result<std::shared_ptr<Storage>> allocate(size_t bytes) override { return new_buffer(bytes, /*weight=*/false); }

  Result<Tensor> upload(const Tensor& host) override {
    const auto bytes = static_cast<size_t>(host.view().span_bytes());
    // Zero-copy weights (docs/gpu-backend.md, section 0): a tensor borrowed
    // from a read-only file mapping (GGUF mmap) is wrapped in place as a
    // shared Metal buffer instead of being copied, so the weights are not
    // held twice. The buffer's keep-alive holds the host tensor, hence the
    // mapping, for as long as the GPU can read it. Owned tensors (converted or
    // dequantized at load) are copied. DYNACORE_GPU_WEIGHTS=copy copies all.
    // ggml's CPU backend asserts (aborts) on a pointer that is not 32-byte
    // aligned instead of failing; such tensors are copied.
    const bool mappable = type_ != DeviceType::kCpu || reinterpret_cast<uintptr_t>(host.data()) % 32 == 0;
    if (host.storage() && !host.storage()->owned() && zero_copy_weights_ && mappable && bytes > 0) {
      ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev_, host.data(), bytes, bytes);
      if (buf != nullptr) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto* p = static_cast<std::byte*>(host.data());
        auto storage = register_buffer(buf, p, p, bytes, /*weight=*/false, host);
        ++mapped_weights_;
        return Tensor::from_storage(storage, TensorView(host.data(), host.view().layout(), device()));
      }
      // Not mappable (alignment / limits): fall back to a copy.
    }
    ENGINE_ASSIGN_OR_RETURN(auto storage, new_buffer(bytes, /*weight=*/true));
    ENGINE_RETURN_IF_ERROR(write_bytes(static_cast<std::byte*>(storage->data()), host.data(), bytes));
    return Tensor::from_storage(storage, TensorView(storage->data(), host.view().layout(), device()));
  }

  void copy(void* dst, const void* src, size_t bytes) override {
    std::vector<std::byte> tmp(bytes);
    Status st = read_bytes(static_cast<const std::byte*>(src), tmp.data(), bytes);
    if (st.ok()) st = write_bytes(static_cast<std::byte*>(dst), tmp.data(), bytes);
    note(st);
  }

  void synchronize() override { ggml_backend_synchronize(backend_); }

  // ggml-cpu: matmul weights in ggml's repack buffer (CPU_REPACK). ggml
  // rearranges them at upload into the interleaved layouts its int8 GEMM / GEMV
  // kernels need (Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL, Q8_0: 8x4 with dotprod, 8x8
  // with i8mm on ARM; 8x8 on AVX2), as llama.cpp does by default. The graph
  // then binds matmul weights to the copy (GraphBuilder::weight); everything
  // else (embedding lookups of a tied table) keeps reading the original.
  // A 3-D expert tensor [experts, rows, cols] (contiguous experts) is repacked
  // whole for mul_mat_id (GraphBuilder::expert_weight), as llama.cpp does.
  // Costs one copy of the matmul weights in RAM. DYNACORE_GGML_REPACK=0: off.
  void prepack_weight(const TensorView& w, bool) override {
    if (type_ != DeviceType::kCpu || (w.rank() != 2 && w.rank() != 3) ||
        repacked_.count(static_cast<const std::byte*>(w.data()))) {
      return;
    }
    static const bool on = [] {
      const char* v = std::getenv("DYNACORE_GGML_REPACK");
      return v == nullptr || std::string_view(v) != "0";
    }();
    ggml_type type;
    if (!on || !to_ggml_type(w.dtype(), type)) return;
    ggml_backend_buffer_type_t repack = repack_buffer_type();
    const int r = w.rank();
    const int64_t cols = w.dim(r - 1), rows = w.dim(r - 2), experts = r == 3 ? w.dim(0) : 1;
    const size_t row = ggml_row_size(type, cols);
    if (repack == nullptr || static_cast<size_t>(w.stride(r - 2)) != row) return;
    if (r == 3 && static_cast<size_t>(w.stride(0)) != row * static_cast<size_t>(rows)) return;
    if (!repack_ctx_) repack_ctx_ = meta_ctx(8192);
    ggml_tensor* t = ggml_new_tensor_3d(repack_ctx_.get(), type, cols, rows, experts);
    if (t == nullptr) return;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(repack, ggml_backend_buft_get_alloc_size(repack, t));
    if (buf == nullptr) return;
    if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS || t->extra == nullptr) {
      ggml_backend_buffer_free(buf);  // no repacked kernel for this type / shape
      return;
    }
    ggml_backend_tensor_set(t, w.data(), 0, ggml_nbytes(t));  // the repack happens here
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    repack_bufs_.push_back(buf);
    repacked_[static_cast<const std::byte*>(w.data())] = t;
    ++prepack_.tensors;
    prepack_.source_bytes += static_cast<int64_t>(ggml_nbytes(t));
  }
  PrepackStats prepack_stats() const override { return prepack_; }
  // The repacked copy of a matmul weight at `p` with ggml shape [cols, rows,
  // experts] (experts 1: a 2-D weight), or null.
  ggml_tensor* repacked(const std::byte* p, int64_t cols, int64_t rows, int64_t experts = 1) const {
    auto it = repacked_.find(p);
    if (it == repacked_.end()) return nullptr;
    const ggml_tensor* t = it->second;
    return t->ne[0] == cols && t->ne[1] == rows && t->ne[2] == experts ? it->second : nullptr;
  }

  void download(const TensorView& src, std::span<float> dst) override {
    const ViewKey k = key_of(src);
    const auto t0 = Clock::now();
    if (k.stride == k.cols * 4) {
      note(read_bytes(k.p, dst.data(), static_cast<size_t>(k.rows * k.cols * 4)));
    } else {
      for (int64_t r = 0; r < k.rows; ++r) {
        note(read_bytes(k.p + r * k.stride, dst.data() + r * k.cols, static_cast<size_t>(k.cols * 4)));
      }
    }
    stats_.download_bytes += k.rows * k.cols * 4;
    stats_.download_ns += ns_since(t0);
  }

  bool supports_weight_type(DType type) const override {
    ggml_type t;
    return to_ggml_type(type, t);
  }

  Status take_error() override {
    Status st = error_;
    error_ = Status::Ok();
    return st;
  }

  MemoryStats memory_stats() const override {
    MemoryStats m;
    m.weight_bytes = table_->weight_bytes;  // copies only; mapped weights are not device copies
    m.buffer_bytes = table_->buffer_bytes;
    for (const auto& [h, g] : graphs_) m.workspace_bytes += g->workspace_bytes;
    m.workspace_limit_bytes = graph_workspace_cap();
    if (!real_addresses_) {  // Vulkan / CUDA: KV and copied weights live in device memory
      size_t free = 0, total = 0;
      ggml_backend_dev_memory(dev_, &free, &total);
      if (total > 0) m.free_bytes = static_cast<int64_t>(free);
    } else if (type_ == DeviceType::kMetal) {
      // Unified memory, but capped: ggml-metal reports total =
      // recommendedMaxWorkingSetSize and free = total - currentAllocatedSize.
      size_t free = 0, total = 0;
      ggml_backend_dev_memory(dev_, &free, &total);
      if (total > 0) {
        m.working_set_bytes = static_cast<int64_t>(total);
        m.allocated_bytes = static_cast<int64_t>(total - std::min(free, total));
      }
    }
    return m;
  }

  // --- Device ops: only through a RecordingDevice ---
  void embedding(const TensorView&, std::span<const int32_t>, const TensorView&) override { direct(); }
  void matmul(const TensorView&, const TensorView&, const TensorView*, const TensorView&) override { direct(); }
  bool supports_moe_ffn() const override { return true; }
  void moe_ffn(const MoeFfnParams&) override { direct(); }
  void rms_norm(const TensorView&, const TensorView&, float, const TensorView&) override { direct(); }
  void layer_norm(const TensorView&, const TensorView&, const TensorView*, float, const TensorView&) override {
    direct();
  }
  void rope(const TensorView&, int32_t, int32_t, std::span<const int32_t>, const RopeConfig&, const float*) override {
    direct();
  }
  void kv_store(const TensorView&, const TensorView&, std::span<const int32_t>, std::span<const int32_t>,
                std::span<const KvLayerView>) override {
    direct();
  }
  void attention(const AttentionParams&) override { direct(); }
  void act_mul(Activation, const TensorView&, const TensorView&, const TensorView&) override { direct(); }
  void activation(Activation, const TensorView&, const TensorView&) override { direct(); }
  void add(const TensorView&, const TensorView&, const TensorView&) override { direct(); }
  void scale(const TensorView&, float) override { direct(); }
  void softcap(const TensorView&, float) override { direct(); }
  void fill(const TensorView&, float) override { direct(); }
  void gather_rows(const TensorView&, std::span<const int32_t>, const TensorView&) override { direct(); }
  void scatter_add_rows(const TensorView&, std::span<const int32_t>, std::span<const float>,
                        const TensorView&) override {
    direct();
  }

  // --- SegmentExecutor ---
  Status run_segment(std::span<const DeviceCall* const> calls, std::span<const uint64_t> signature) override;
  SegmentStats segment_stats() const override { return stats_; }

 private:
  friend class GraphBuilder;

  void direct() {
    note(Unsupported("the " + name_ + " device runs only compiled (recorded) steps; use --execution compiled"));
  }
  void note(const Status& st) {
    if (!st.ok() && error_.ok()) error_ = st;
  }

  Result<std::shared_ptr<Storage>> new_buffer(size_t bytes, bool weight) {
    // Host memory (ggml-cpu): an allocation beyond physical RAM can succeed
    // lazily (overcommit) and the clear below then gets the process killed.
    if (type_ == DeviceType::kCpu) {
      const int64_t ram = memory_info().total_bytes;
      if (ram > 0 && bytes > static_cast<uint64_t>(ram)) {
        return OutOfMemory(name_ + ": cannot allocate " + std::to_string(bytes) + " bytes (more than physical memory)");
      }
    }
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft_, std::max<size_t>(bytes, 1));
    if (buf == nullptr) {
      return OutOfMemory(name_ + ": cannot allocate " + std::to_string(bytes) + " bytes of device memory");
    }
    if (weight) {
      ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else {
      // Zeroed: attention reads masked KV cells inside partially masked
      // chunks, and a NaN bit pattern there would poison the softmax (0 * NaN).
      ggml_backend_buffer_clear(buf, 0);
    }
    auto* real = static_cast<std::byte*>(ggml_backend_buffer_get_base(buf));
    std::byte* handle = real;
    if (!real_addresses_) {
      handle = HandleRange::reserve(std::max<size_t>(ggml_backend_buffer_get_size(buf), 1));
      if (handle == nullptr) {
        ggml_backend_buffer_free(buf);
        return OutOfMemory(name_ + ": cannot reserve an address range for a device buffer");
      }
    }
    return register_buffer(buf, handle, real, bytes, weight, Tensor());
  }

  // Records a backend buffer and hands it out as Storage; freeing the Storage
  // frees the buffer (and drops `host`, a mapped tensor kept alive with it).
  // `weight` counts device-memory weight copies; mapped weights count as
  // mapped bytes (no device copy).
  std::shared_ptr<Storage> register_buffer(ggml_backend_buffer_t buf, std::byte* base, std::byte* real, size_t bytes,
                                           bool weight, Tensor host) {
    const bool mapped = host.storage() != nullptr;
    BufferInfo info{buf, base, real, mapped ? bytes : ggml_backend_buffer_get_size(buf)};
    const bool handle = base != real;
    std::shared_ptr<BufferTable> table = table_;
    table->by_base[reinterpret_cast<uintptr_t>(info.base)] = info;
    int64_t BufferTable::*counter = mapped ? &BufferTable::mapped_bytes : weight ? &BufferTable::weight_bytes : &BufferTable::buffer_bytes;
    (*table).*counter += static_cast<int64_t>(info.size);
    std::shared_ptr<const void> keep(info.base, [table, counter, info, host, handle](const void*) {
      table->by_base.erase(reinterpret_cast<uintptr_t>(info.base));
      ++table->free_epoch;
      (*table).*counter -= static_cast<int64_t>(info.size);
      ggml_backend_buffer_free(info.buf);
      if (handle) HandleRange::release(info.base, std::max<size_t>(info.size, 1));
      (void)host;  // released after the buffer
    });
    return Storage::borrow(info.base, bytes, device(), std::move(keep));
  }

  // The buffer holding [p, p + bytes).
  const BufferInfo* find(const std::byte* p, size_t bytes) const {
    auto it = table_->by_base.upper_bound(reinterpret_cast<uintptr_t>(p));
    if (it == table_->by_base.begin()) return nullptr;
    --it;
    const BufferInfo& b = it->second;
    if (p < b.base || p + bytes > b.base + b.size) return nullptr;
    return &b;
  }

  // A tensor of `bytes` I8 at `p`, for byte transfers.
  Status with_bytes(const std::byte* p, size_t bytes, const std::function<void(ggml_tensor*)>& f) {
    const BufferInfo* b = find(p, bytes);
    if (b == nullptr) return InvalidArgument(name_ + ": address is not device memory of this backend");
    CtxPtr ctx = meta_ctx(1);
    ggml_tensor* t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, static_cast<int64_t>(bytes));
    if (ggml_backend_tensor_alloc(b->buf, t, b->real_at(p)) != GGML_STATUS_SUCCESS) {
      return Internal(name_ + ": cannot bind a transfer tensor");
    }
    f(t);
    return Status::Ok();
  }
  Status write_bytes(std::byte* dst, const void* src, size_t bytes) {
    if (bytes == 0) return Status::Ok();
    return with_bytes(dst, bytes, [&](ggml_tensor* t) { ggml_backend_tensor_set(t, src, 0, bytes); });
  }
  Status read_bytes(const std::byte* src, void* dst, size_t bytes) {
    if (bytes == 0) return Status::Ok();
    return with_bytes(src, bytes, [&](ggml_tensor* t) { ggml_backend_tensor_get(t, dst, 0, bytes); });
  }

  Result<std::unique_ptr<Prepared>> build(std::span<const DeviceCall* const> calls, std::vector<uint64_t> key,
                                          const std::vector<std::vector<SeqRows>>& seqs);
  Status fill_inputs(Prepared& g, std::span<const DeviceCall* const> calls,
                     const std::vector<std::vector<SeqRows>>& seqs);

  ggml_backend_dev_t dev_;
  bool zero_copy_weights_ = false;
  bool real_addresses_ = true;  // buffers keep the backend address (Metal); else handle ranges
  int64_t mapped_weights_ = 0;
  ggml_backend_t backend_;
  ggml_threadpool* threadpool_ = nullptr;
  ggml_backend_buffer_type_t buft_ = nullptr;
  DeviceType type_;
  GgmlDeviceOptions opts_;
  std::string name_;
  std::shared_ptr<BufferTable> table_;
  std::unordered_map<uint64_t, std::unique_ptr<Prepared>> graphs_;
  uint64_t graphs_epoch_ = 0;
  uint64_t use_tick_ = 0;
  // ggml-cpu repacked matmul weights (prepack_weight): by source address.
  CtxPtr repack_ctx_;
  std::vector<ggml_backend_buffer_t> repack_bufs_;
  std::unordered_map<const std::byte*, ggml_tensor*> repacked_;
  PrepackStats prepack_{};
  ggml_backend_buffer_type_t repack_buffer_type() const {
    using get_extra_t = ggml_backend_buffer_type_t* (*)(ggml_backend_dev_t);
    auto* fn = reinterpret_cast<get_extra_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev_), "ggml_backend_dev_get_extra_bufts"));
    if (fn == nullptr) return nullptr;
    for (ggml_backend_buffer_type_t* b = fn(dev_); b != nullptr && *b != nullptr; ++b) {
      if (std::string_view(ggml_backend_buft_name(*b)) == "CPU_REPACK") return *b;
    }
    return nullptr;
  }
  int64_t peak_workspace_ = 0;  // most workspace bytes held by prepared graphs at once (diagnostics)
  SegmentStats stats_;
  Status error_;
  std::vector<std::byte> host_in_;  // staging for dynamic inputs
};

// Slot-major (in-place) attention: one entry covering every row, whose span is
// the pool cells [0, span) that hold any block the batch attends to, padded to
// `bucket` (capped at the pool) so a decode stream reuses one graph per bucket.
inline Result<std::vector<SeqRows>> paged_span(const DeviceCall& c, int32_t bucket, DeviceType type) {
  const AttentionParams& p = c.ap;
  const KvGeometry& g = *p.kv[0].geom;
  const int32_t bs = g.block_size;
  int64_t cells = 0;
  for (size_t r = 0; r < p.positions.size(); ++r) {
    const auto s = static_cast<size_t>(p.row_seq[r]);
    if (s >= p.kv.size()) return InvalidArgument("attention: row sequence out of range");
    const std::span<const int32_t> table = p.kv[s].block_table;
    const auto blocks = static_cast<size_t>(p.positions[r] / bs + 1);
    if (blocks > table.size()) return InvalidArgument("attention: position has no KV block");
    for (size_t b = 0; b < blocks; ++b) {
      if (table[b] < 0 || table[b] >= g.num_blocks) return InvalidArgument("attention: block id out of range");
      cells = std::max<int64_t>(cells, (static_cast<int64_t>(table[b]) + 1) * bs);
    }
  }
  const int64_t pool_cells = static_cast<int64_t>(g.num_blocks) * bs;
  const int64_t span = std::min(pool_cells, (cells + bucket - 1) / bucket * bucket);
  // Sparse attention bound: a row attends to positions [lo, pos] of its own
  // sequence only, so at most max(pos - lo + 1) of the span's cells. With
  // several sequences in the pool that is far below the span.
  int64_t most = 0;
  for (size_t r = 0; r < p.positions.size(); ++r) {
    const int64_t pos = p.positions[r];
    const int64_t lo = p.sliding_window > 0 ? std::max<int64_t>(0, pos - p.sliding_window + 1) : 0;
    most = std::max(most, pos - lo + 1);
  }
  const int64_t kv_max = (most + bucket - 1) / bucket * bucket;
  SeqRows all{-1, 0, static_cast<int32_t>(p.positions.size()), static_cast<int32_t>(cells), static_cast<int32_t>(span)};
  // Sparse K/V bound, per step (sparse_kv_bound, ggml_device.h): always-on cost
  // one user 10% on Metal (7.13 -> 7.87 ms/token, 1.5B, 2026-10-18), so kAuto
  // takes it only when the span far exceeds the context (a cache-filled pool).
  all.kv_max = sparse_kv_bound(span, kv_max, gpu_sparse_mode(), type);
  return std::vector<SeqRows>{all};
}

// Whether two calls feed an input of `kind` the same data (inputs are shared
// by such calls; the sharing is re-checked on every run).
inline bool same_source(InKind kind, const DeviceCall& a, const DeviceCall& b, int32_t seq) {
  auto tables_equal = [&](bool only_seq) {
    if (a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
      if (only_seq && static_cast<int32_t>(i) != seq) continue;
      if (!std::equal(a.kv[i].block_table.begin(), a.kv[i].block_table.end(), b.kv[i].block_table.begin(),
                      b.kv[i].block_table.end())) {
        return false;
      }
    }
    return true;
  };
  switch (kind) {
    case InKind::kIds:
    case InKind::kPositions:
    case InKind::kRows: return a.ints0 == b.ints0;
    case InKind::kKvIndex: return a.ints0 == b.ints0 && a.ints1 == b.ints1 && tables_equal(false);
    case InKind::kGatherIdx: return a.ints0 == b.ints0 && a.ints1 == b.ints1 && tables_equal(true);
    case InKind::kMask: return a.ints0 == b.ints0 && a.ints1 == b.ints1 && a.ap.sliding_window == b.ap.sliding_window;
    case InKind::kPagedMask:
      return a.ints0 == b.ints0 && a.ints1 == b.ints1 && a.ap.sliding_window == b.ap.sliding_window && tables_equal(false);
    case InKind::kConst: return a.freq == b.freq && a.rope.dim == b.rope.dim;
  }
  return false;
}

// Head-major (gather) attention: contiguous row ranges per sequence index, and
// each sequence's padded KV span.
inline Result<std::vector<SeqRows>> seq_rows(const DeviceCall& c, int32_t bucket) {
  const std::vector<int32_t>& pos = c.ints0;
  const std::vector<int32_t>& rs = c.ints1;
  std::vector<SeqRows> out;
  for (size_t r = 0; r < rs.size(); ++r) {
    if (out.empty() || out.back().seq != rs[r]) {
      for (const SeqRows& s : out) {
        if (s.seq == rs[r]) return Unsupported("batch rows of one sequence are not contiguous");
      }
      out.push_back(SeqRows{rs[r], static_cast<int32_t>(r), 0, 0, 0});
    }
    SeqRows& s = out.back();
    ++s.n;
    s.ctx = std::max(s.ctx, pos[r] + 1);
  }
  for (SeqRows& s : out) s.span = (s.ctx + bucket - 1) / bucket * bucket;
  return out;
}

}  // namespace dynacore::ggml_detail

#endif  // DYNACORE_HAS_GGML

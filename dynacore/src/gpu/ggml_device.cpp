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
#include <unordered_map>
#include <vector>

#include "dynacore/device/segment.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#if DYNACORE_HAS_METAL
#include "ggml-metal.h"
#endif
#if DYNACORE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif
#if DYNACORE_HAS_CUDA
#include "ggml-cuda.h"
#endif

namespace dynacore {
namespace {

using Clock = std::chrono::steady_clock;
int64_t ns_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
}

bool to_ggml_type(DType t, ggml_type& out) {
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
    default: return false;
  }
}

// A ggml context for tensor metadata only (no data).
struct CtxDeleter {
  void operator()(ggml_context* c) const { ggml_free(c); }
};
using CtxPtr = std::unique_ptr<ggml_context, CtxDeleter>;
CtxPtr meta_ctx(size_t tensors, size_t graph_nodes = 0) {
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

int64_t rows_of(const TensorView& t) { return t.rank() > 1 ? t.dim(0) : 1; }
int64_t cols_of(const TensorView& t) { return t.dim(t.rank() - 1); }
ViewKey key_of(const TensorView& t) {
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
  ~Prepared() {
    if (galloc) ggml_gallocr_free(galloc);
    if (in_buf) ggml_backend_buffer_free(in_buf);
  }
};

// Per sequence: its rows in the batch (contiguous) and its padded KV span.
struct SeqRows {
  int32_t seq = 0, r0 = 0, n = 0, ctx = 0, span = 0;
};

class GgmlDevice final : public Device, public SegmentExecutor {
 public:
  GgmlDevice(ggml_backend_dev_t dev, ggml_backend_t backend, DeviceType type, const GgmlDeviceOptions& opts)
      : dev_(dev), backend_(backend), type_(type), opts_(opts), table_(std::make_shared<BufferTable>()) {
    const char* provider = type == DeviceType::kMetal    ? "Metal/"
                           : type == DeviceType::kVulkan ? "Vulkan/"
                           : type == DeviceType::kCuda   ? "CUDA/"
                                                         : "GPU/";
    name_ = std::string(provider) + ggml_backend_dev_description(dev);
    buft_ = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const char* w = std::getenv("DYNACORE_GPU_WEIGHTS");
    // Unique per-buffer addresses from the backend: Metal (shared memory).
    // Elsewhere buffers get handle ranges, and weights are copied (mapping a
    // host range under a handle is not supported).
    real_addresses_ = type == DeviceType::kMetal;
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
                   "%lld nodes in the last graph; %lld weights mapped in place\n",
                   name_.c_str(), static_cast<long long>(stats_.segments), per(stats_.input_ns), per(stats_.compute_ns),
                   per(stats_.download_ns), static_cast<long long>(stats_.graph_builds),
                   static_cast<double>(stats_.build_ns) / 1e6, static_cast<long long>(stats_.graph_reuses),
                   static_cast<long long>(stats_.graph_evictions), static_cast<long long>(stats_.errors),
                   static_cast<long long>(stats_.graph_nodes), static_cast<long long>(mapped_weights_));
    }
    graphs_.clear();
    ggml_backend_free(backend_);
  }

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
    if (host.storage() && !host.storage()->owned() && zero_copy_weights_ && bytes > 0) {
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
    if (!real_addresses_) {  // Vulkan / CUDA: KV and copied weights live in device memory
      size_t free = 0, total = 0;
      ggml_backend_dev_memory(dev_, &free, &total);
      if (total > 0) m.free_bytes = static_cast<int64_t>(free);
    }
    return m;
  }

  // --- Device ops: only through a RecordingDevice ---
  void embedding(const TensorView&, std::span<const int32_t>, const TensorView&) override { direct(); }
  void matmul(const TensorView&, const TensorView&, const TensorView*, const TensorView&) override { direct(); }
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
  ggml_backend_buffer_type_t buft_ = nullptr;
  DeviceType type_;
  GgmlDeviceOptions opts_;
  std::string name_;
  std::shared_ptr<BufferTable> table_;
  std::unordered_map<uint64_t, std::unique_ptr<Prepared>> graphs_;
  uint64_t graphs_epoch_ = 0;
  SegmentStats stats_;
  Status error_;
  std::vector<std::byte> host_in_;  // staging for dynamic inputs
};

// Slot-major (in-place) attention: one entry covering every row, whose span is
// the pool cells [0, span) that hold any block the batch attends to, padded to
// `bucket` (capped at the pool) so a decode stream reuses one graph per bucket.
Result<std::vector<SeqRows>> paged_span(const DeviceCall& c, int32_t bucket) {
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
  return std::vector<SeqRows>{SeqRows{-1, 0, static_cast<int32_t>(p.positions.size()), static_cast<int32_t>(cells),
                                      static_cast<int32_t>(span)}};
}

// Whether two calls feed an input of `kind` the same data (inputs are shared
// by such calls; the sharing is re-checked on every run).
bool same_source(InKind kind, const DeviceCall& a, const DeviceCall& b, int32_t seq) {
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
Result<std::vector<SeqRows>> seq_rows(const DeviceCall& c, int32_t bucket) {
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

// Builds one ggml graph for a recorded segment. Values flow through ggml
// nodes; each DynaLM view written in the segment is copied back to its buffer
// at the end (so memory reads in later segments see it), and any read of
// memory written through a different, non-derivable view is refused.
class GraphBuilder {
 public:
  GraphBuilder(GgmlDevice& d, Prepared& g, std::span<const DeviceCall* const> calls,
               const std::vector<std::vector<SeqRows>>& seqs)
      : d_(d), g_(g), calls_(calls), seqs_(seqs) {}

  Status build() {
    const size_t n_calls = calls_.size();
    const size_t max_nodes = std::max<size_t>(1024, n_calls * 48);
    g_.ctx = meta_ctx(max_nodes * 2, max_nodes);
    g_.ctx_in = meta_ctx(n_calls * 4 + 16);
    ctx_ = g_.ctx.get();
    g_.graph = ggml_new_graph_custom(ctx_, max_nodes, false);
    for (size_t i = 0; i < n_calls; ++i) {
      ci_ = static_cast<int32_t>(i);
      ENGINE_RETURN_IF_ERROR(lower(*calls_[i]));
    }
    // Write back every view written in this segment, in write order.
    for (const auto& [k, t] : live_) ENGINE_RETURN_IF_ERROR(materialize(k, t));
    for (ggml_tensor* t : pools_written_) ggml_build_forward_expand(g_.graph, t);
    // Every node must be supported by the backend: no silent CPU fallback.
    for (int i = 0; i < ggml_graph_n_nodes(g_.graph); ++i) {
      ggml_tensor* n = ggml_graph_node(g_.graph, i);
      if (!ggml_backend_supports_op(d_.backend_, n)) {
        return Unsupported(d_.name_ + " does not support op " + std::string(ggml_op_desc(n)) + " (" +
                           ggml_type_name(n->type) + ") at " + std::to_string(n->ne[0]) + "x" +
                           std::to_string(n->ne[1]) + "x" + std::to_string(n->ne[2]));
      }
    }
    return Status::Ok();
  }

 private:
  using M = DeviceCall::M;

  // --- values -------------------------------------------------------------------
  // A leaf bound to runtime memory at `p` with the given type and shape.
  Result<ggml_tensor*> bind(const std::byte* p, ggml_type type, int64_t ne0, int64_t ne1, size_t nb1) {
    const auto key = std::make_tuple(p, static_cast<int>(type), ne0, ne1, nb1);
    if (auto it = leaves_.find(key); it != leaves_.end()) return it->second;
    ggml_tensor* t = ggml_new_tensor_2d(ctx_, type, ne0, ne1);
    t->nb[1] = nb1;
    t->nb[2] = t->nb[3] = nb1 * static_cast<size_t>(ne1);
    const size_t bytes = ggml_nbytes(t);
    const BufferInfo* b = d_.find(p, bytes);
    if (b == nullptr) return InvalidArgument(d_.name_ + ": tensor memory is not owned by this device");
    if (ggml_backend_tensor_alloc(b->buf, t, b->real_at(p)) != GGML_STATUS_SUCCESS) {
      return Internal(d_.name_ + ": cannot bind tensor memory");
    }
    leaves_.emplace(key, t);
    return t;
  }

  // Weight or constant vector: any supported dtype, rank 1 or 2, dense rows.
  Result<ggml_tensor*> weight(const TensorView& w) {
    ggml_type type;
    if (!to_ggml_type(w.dtype(), type)) return Unsupported("weight type " + std::string(dtype_name(w.dtype())));
    if (w.rank() > 2) return Unsupported("weights of rank > 2 (MoE expert tensors)");
    const int64_t ne0 = cols_of(w), ne1 = rows_of(w);
    const size_t row = ggml_row_size(type, ne0);
    const size_t nb1 = w.rank() > 1 ? static_cast<size_t>(w.stride(0)) : row;
    if (nb1 < row) return Unsupported("weight rows overlap");
    return bind(static_cast<const std::byte*>(w.data()), type, ne0, ne1, nb1);
  }

  // The current value of an fp32 activation view.
  Result<ggml_tensor*> read(const TensorView& v) {
    if (v.dtype() != DType::kF32) return Unsupported("activation dtype " + std::string(dtype_name(v.dtype())));
    const ViewKey k = key_of(v);
    for (const auto& [lk, t] : live_) {
      if (lk == k) return t;
    }
    // A sub-view (rows and/or columns) of a value written in this segment.
    for (const auto& [lk, t] : live_) {
      if (lk.stride != k.stride || k.p < lk.p || !lk.overlaps(k)) continue;
      const int64_t off = k.p - lk.p;
      const int64_t r0 = off / lk.stride, c0 = (off % lk.stride) / 4;
      if ((off % lk.stride) % 4 == 0 && r0 + k.rows <= lk.rows && c0 + k.cols <= lk.cols) {
        return ggml_view_2d(ctx_, t, k.cols, k.rows, t->nb[1], static_cast<size_t>(r0) * t->nb[1] + c0 * 4);
      }
    }
    for (const ViewKey& w : written_) {
      if (w.overlaps(k)) {
        return Unsupported("a step reads memory written through a different view (aliasing): call " +
                           std::to_string(ci_) + " (" + call_name(calls_[static_cast<size_t>(ci_)]->m) + ") reads " +
                           describe(k) + ", written as " + describe(w));
      }
    }
    return bind(k.p, GGML_TYPE_F32, k.cols, k.rows, static_cast<size_t>(k.stride));
  }

  // `t` ([cols, rows] or reshapeable to it) becomes the value of view `v`.
  Status write(const TensorView& v, ggml_tensor* t) {
    const ViewKey k = key_of(v);
    if (ggml_nelements(t) != k.rows * k.cols) return Internal("value shape does not match its view");
    if (t->ne[0] != k.cols || t->ne[1] != k.rows || t->ne[2] != 1 || t->ne[3] != 1) {
      if (!ggml_is_contiguous(t)) t = ggml_cont(ctx_, t);
      t = ggml_reshape_2d(ctx_, t, k.cols, k.rows);
    }
    // Overlapping values written earlier: write them back first, then forget
    // them (a later read of what they covered is refused or re-derived).
    for (auto it = live_.begin(); it != live_.end();) {
      if (!(it->first == k) && it->first.overlaps(k)) {
        ENGINE_RETURN_IF_ERROR(materialize(it->first, it->second));
        it = live_.erase(it);
      } else {
        ++it;
      }
    }
    bool found = false;
    for (auto& [lk, lt] : live_) {
      if (lk == k) {
        lt = t;
        found = true;
      }
    }
    if (!found) live_.emplace_back(k, t);
    written_.push_back(k);
    return Status::Ok();
  }

  Status materialize(const ViewKey& k, ggml_tensor* t) {
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * dst, bind(k.p, GGML_TYPE_F32, k.cols, k.rows, static_cast<size_t>(k.stride)));
    ggml_build_forward_expand(g_.graph, ggml_cpy(ctx_, t, dst));
    return Status::Ok();
  }

  // A dynamic input tensor, shared by calls whose source data are identical.
  ggml_tensor* input(InKind kind, ggml_type type, int64_t ne0, int64_t ne1, int32_t seq = 0, int32_t r0 = 0,
                     int32_t n = 0, int32_t span = 0) {
    for (Input& in : g_.inputs) {
      if (in.kind == kind && in.seq == seq && in.r0 == r0 && in.n == n && in.span == span &&
          in.t->ne[0] == ne0 && in.t->ne[1] == ne1 && same_source(kind, *calls_[in.call], *calls_[ci_], seq)) {
        in.members.push_back(ci_);
        return in.t;
      }
    }
    Input in;
    in.kind = kind;
    in.t = ggml_new_tensor_2d(g_.ctx_in.get(), type, ne0, ne1);
    ggml_set_input(in.t);
    in.call = ci_;
    in.seq = seq;
    in.r0 = r0;
    in.n = n;
    in.span = span;
    g_.inputs.push_back(std::move(in));
    return g_.inputs.back().t;
  }

  // --- lowering -----------------------------------------------------------------
  Status lower(const DeviceCall& c) {
    switch (c.m) {
      case M::kEmbedding: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * table, weight(c.a));
        ggml_tensor* ids = input(InKind::kIds, GGML_TYPE_I32, static_cast<int64_t>(c.ints0.size()), 1);
        return write(c.b, ggml_get_rows(ctx_, table, ids));
      }
      case M::kMatmul: return matmul(c.a, c.b, c.has_c ? &c.c : nullptr, c.d);
      case M::kMatmulMany:
        for (const Device::MatmulJob& j : c.jobs) ENGINE_RETURN_IF_ERROR(matmul(j.x, j.w, j.bias, j.y));
        return Status::Ok();
      case M::kRmsNorm:
      case M::kLayerNorm: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * w, weight(c.b));
        if (!ggml_is_contiguous_rows(x)) x = ggml_cont(ctx_, x);
        ggml_tensor* y = c.m == M::kRmsNorm ? ggml_rms_norm(ctx_, x, c.f) : ggml_norm(ctx_, x, c.f);
        y = ggml_mul(ctx_, y, w);
        if (c.has_c) {
          ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, weight(c.c));
          y = ggml_add(ctx_, y, b);
        }
        return write(c.d, y);
      }
      case M::kRope: return rope(c);
      case M::kKvStore: return kv_store(c);
      case M::kAttention: return attention(c);
      case M::kActMul: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * g, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * u, read(c.b));
        if (!ggml_is_contiguous(g)) g = ggml_cont(ctx_, g);
        if (!ggml_is_contiguous(u)) u = ggml_cont(ctx_, u);
        ggml_tensor* y = c.act == Activation::kSilu   ? ggml_swiglu_split(ctx_, g, u)
                         : c.act == Activation::kGelu ? ggml_geglu_erf_split(ctx_, g, u)
                                                      : ggml_geglu_split(ctx_, g, u);
        return write(c.d, y);
      }
      case M::kActivation: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
        ggml_tensor* y = c.act == Activation::kSilu   ? ggml_silu(ctx_, x)
                         : c.act == Activation::kGelu ? ggml_gelu_erf(ctx_, x)
                                                      : ggml_gelu(ctx_, x);
        return write(c.d, y);
      }
      case M::kAdd: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * a, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, read(c.b));
        return write(c.d, ggml_add(ctx_, a, b));
      }
      case M::kScale: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        return write(c.a, ggml_scale(ctx_, x, c.f));
      }
      case M::kSoftcap: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        return write(c.a, ggml_scale(ctx_, ggml_tanh(ctx_, ggml_scale(ctx_, x, 1.0f / c.f)), c.f));
      }
      case M::kFill: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
        return write(c.a, ggml_fill(ctx_, x, c.f));
      }
      case M::kGather: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * src, read(c.a));
        ggml_tensor* rows = input(InKind::kRows, GGML_TYPE_I32, static_cast<int64_t>(c.ints0.size()), 1);
        return write(c.d, ggml_get_rows(ctx_, src, rows));
      }
      case M::kScatterAdd: return Unsupported("scatter_add_rows (mixture-of-experts layers) on the GPU");
    }
    return Internal("unknown recorded call");
  }

  Status matmul(const TensorView& xv, const TensorView& wv, const TensorView* bv, const TensorView& yv) {
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(xv));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * w, weight(wv));
    if (w->ne[0] != x->ne[0]) return Internal("matmul inner dimensions differ");
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
    ggml_tensor* y = ggml_mul_mat(ctx_, w, x);
    // fp32 accumulation (DD-053): Vulkan and CUDA otherwise may accumulate
    // matrix-matrix products in fp16; Metal always uses fp32.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (bv != nullptr) {
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, weight(*bv));
      y = ggml_add(ctx_, y, b);
    }
    return write(yv, y);
  }

  Status rope(const DeviceCall& c) {
    if (c.rope.style == RopeStyle::kNone) return Status::Ok();
    float freq_scale = 1.0f;
    if (c.rope.scaling == RopeScaling::kLinear) {
      freq_scale = 1.0f / c.rope.scaling_factor;
    } else if (c.rope.scaling != RopeScaling::kNone) {
      return Unsupported("RoPE scaling other than none/linear (YaRN, Llama 3) on the GPU");
    }
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
    const int64_t m = static_cast<int64_t>(c.ints0.size());
    ggml_tensor* x3 = ggml_reshape_3d(ctx_, x, c.i1, c.i0, m);
    ggml_tensor* pos = input(InKind::kPositions, GGML_TYPE_I32, m, 1);
    ggml_tensor* factors = nullptr;
    if (c.freq != nullptr) {
      factors = input(InKind::kConst, GGML_TYPE_F32, c.rope.dim / 2, 1);
      Input& in = g_.inputs.back();
      if (in.t == factors && in.constant.empty()) in.constant.assign(c.freq, c.freq + c.rope.dim / 2);
    }
    const int mode = c.rope.style == RopeStyle::kHalfSplit ? GGML_ROPE_TYPE_NEOX : 0;
    ggml_tensor* y = ggml_rope_ext(ctx_, x3, pos, factors, c.rope.dim, mode, static_cast<int>(c.rope.original_context),
                                   c.rope.freq_base, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    return write(c.a, ggml_reshape_2d(ctx_, y, static_cast<int64_t>(c.i0) * c.i1, m));
  }

  // The current K or V pool tensor of a layer. Head-major: rows of head_dim
  // elements (one per block, head, slot). Slot-major: one row per cell, all
  // KV heads of one token (n_kv_heads * head_dim elements).
  Result<ggml_tensor*> pool(const KvLayerView& kv, bool is_v) {
    const KvGeometry& g = *kv.geom;
    ggml_type type;
    if (g.dtype != DType::kF16 && g.dtype != DType::kF32 && g.dtype != DType::kQ8_0) {
      return Unsupported("KV cache dtype " + std::string(dtype_name(g.dtype)) + " on the GPU (f16, f32, q8_0)");
    }
    to_ggml_type(g.dtype, type);
    const auto* base = static_cast<const std::byte*>(is_v ? kv.v : kv.k);
    if (auto it = pool_now_.find(base); it != pool_now_.end()) return it->second;
    const int64_t dim = is_v ? g.head_dim_v : g.head_dim;
    const bool slot = g.layout == KvLayout::kSlotMajor;
    const int64_t ne0 = slot ? dim * g.num_kv_heads : dim;
    const int64_t rows = static_cast<int64_t>(g.num_blocks) * g.block_size * (slot ? 1 : g.num_kv_heads);
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * t, bind(base, type, ne0, rows, ggml_row_size(type, ne0)));
    pool_now_[base] = t;
    return t;
  }

  Status kv_store(const DeviceCall& c) {
    if (c.kv.empty()) return Status::Ok();
    for (const KvLayerView& v : c.kv) {
      if (v.k != c.kv[0].k || v.v != c.kv[0].v) return Unsupported("kv_store across several KV pools");
    }
    const KvGeometry& g = *c.kv[0].geom;
    const bool slot = g.layout == KvLayout::kSlotMajor;
    const int64_t m = static_cast<int64_t>(c.ints0.size());
    // Slot-major: one row (cell) per token; head-major: one row per token and head.
    ggml_tensor* idx = input(InKind::kKvIndex, GGML_TYPE_I64, slot ? m : g.num_kv_heads * m, 1);
    for (int which = 0; which < 2; ++which) {
      const bool is_v = which == 1;
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * src, read(is_v ? c.b : c.a));
      if (!ggml_is_contiguous(src)) src = ggml_cont(ctx_, src);
      const int64_t dim = is_v ? g.head_dim_v : g.head_dim;
      ggml_tensor* rows = slot ? ggml_reshape_2d(ctx_, src, dim * g.num_kv_heads, m)
                               : ggml_reshape_2d(ctx_, src, dim, g.num_kv_heads * m);
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * dst, pool(c.kv[0], is_v));
      ggml_tensor* out = ggml_set_rows(ctx_, dst, rows, idx);
      pool_now_[static_cast<const std::byte*>(is_v ? c.kv[0].v : c.kv[0].k)] = out;
      pools_written_.push_back(out);
    }
    return Status::Ok();
  }

  Status attention(const DeviceCall& c) {
    const AttentionParams& p = c.ap;
    if (p.kv.empty()) return Internal("attention without KV");
    const KvGeometry& g = *p.kv[0].geom;
    for (const KvLayerView& v : p.kv) {
      if (v.k != p.kv[0].k || v.v != p.kv[0].v) return Unsupported("attention across several KV pools");
    }
    const int32_t hd = g.head_dim, hdv = g.head_dim_v, nkv = g.num_kv_heads, nh = p.num_heads;
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * q, read(p.q));
    if (!ggml_is_contiguous(q)) q = ggml_cont(ctx_, q);
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * kpool, pool(p.kv[0], false));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * vpool, pool(p.kv[0], true));
    const std::vector<SeqRows>& seqs = seqs_[static_cast<size_t>(ci_)];
    if (g.layout == KvLayout::kSlotMajor) {
      // In-place paged attention: K/V are strided views of the pool's first
      // `span` cells ([dim, cells, kv_heads]); the mask admits, per row, exactly
      // the cells of its own sequence at positions [lo, pos] (block tables,
      // causality, window). Keys are stored after RoPE, so attention over
      // physical cell order is exact up to summation order. Fully masked
      // chunks are skipped by the kernel; nothing is gathered or copied.
      const int64_t m = static_cast<int64_t>(p.positions.size()), L = seqs.front().span;
      ggml_tensor* qs = ggml_permute(ctx_, ggml_reshape_3d(ctx_, q, hd, nh, m), 0, 2, 1, 3);  // [hd, m, nh]
      ggml_tensor* ks = ggml_view_3d(ctx_, kpool, hd, L, nkv, kpool->nb[1], ggml_row_size(kpool->type, hd), 0);
      ggml_tensor* vs = ggml_view_3d(ctx_, vpool, hdv, L, nkv, vpool->nb[1], ggml_row_size(vpool->type, hdv), 0);
      ggml_tensor* mask = input(InKind::kPagedMask, GGML_TYPE_F16, L, m, -1, 0, static_cast<int32_t>(m), static_cast<int32_t>(L));
      ggml_tensor* o = ggml_flash_attn_ext(ctx_, qs, ks, vs, mask, p.scale, 0.0f, p.softcap);  // [hdv, nh, m]
      ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);  // fp32 softmax/accumulation, as on Metal
      return write(p.out, ggml_reshape_2d(ctx_, o, static_cast<int64_t>(hdv) * nh, m));
    }
    ggml_tensor* out = nullptr;
    for (const SeqRows& s : seqs) {
      const int64_t n = s.n, L = s.span;
      ggml_tensor* qs = ggml_view_2d(ctx_, q, q->ne[0], n, q->nb[1], static_cast<size_t>(s.r0) * q->nb[1]);
      qs = ggml_permute(ctx_, ggml_reshape_3d(ctx_, ggml_cont(ctx_, qs), hd, nh, n), 0, 2, 1, 3);  // [hd, n, nh]
      ggml_tensor* gidx = input(InKind::kGatherIdx, GGML_TYPE_I32, nkv * L, 1, s.seq, s.r0, s.n, s.span);
      ggml_tensor* ks = ggml_reshape_3d(ctx_, ggml_get_rows(ctx_, kpool, gidx), hd, L, nkv);
      ggml_tensor* vs = ggml_reshape_3d(ctx_, ggml_get_rows(ctx_, vpool, gidx), hdv, L, nkv);
      ggml_tensor* mask = input(InKind::kMask, GGML_TYPE_F16, L, n, s.seq, s.r0, s.n, s.span);
      ggml_tensor* o = ggml_flash_attn_ext(ctx_, qs, ks, vs, mask, p.scale, 0.0f, p.softcap);  // [hdv, nh, n]
      ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
      o = ggml_reshape_2d(ctx_, o, static_cast<int64_t>(hdv) * nh, n);
      out = out == nullptr ? o : ggml_concat(ctx_, out, o, 1);
    }
    return write(p.out, out);
  }

  static std::string describe(const ViewKey& k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "[%lldx%lld stride %lld @%p]", static_cast<long long>(k.rows),
                  static_cast<long long>(k.cols), static_cast<long long>(k.stride), static_cast<const void*>(k.p));
    return buf;
  }
  static const char* call_name(M m) {
    switch (m) {
      case M::kEmbedding: return "embedding";
      case M::kMatmul: return "matmul";
      case M::kMatmulMany: return "matmul_many";
      case M::kRmsNorm: return "rms_norm";
      case M::kLayerNorm: return "layer_norm";
      case M::kRope: return "rope";
      case M::kKvStore: return "kv_store";
      case M::kAttention: return "attention";
      case M::kActMul: return "act_mul";
      case M::kActivation: return "activation";
      case M::kAdd: return "add";
      case M::kScale: return "scale";
      case M::kSoftcap: return "softcap";
      case M::kFill: return "fill";
      case M::kGather: return "gather_rows";
      case M::kScatterAdd: return "scatter_add_rows";
    }
    return "?";
  }

  GgmlDevice& d_;
  Prepared& g_;
  std::span<const DeviceCall* const> calls_;
  const std::vector<std::vector<SeqRows>>& seqs_;
  ggml_context* ctx_ = nullptr;
  int32_t ci_ = 0;
  std::vector<std::pair<ViewKey, ggml_tensor*>> live_;  // values written in this segment, in write order
  std::vector<ViewKey> written_;
  std::map<std::tuple<const std::byte*, int, int64_t, int64_t, size_t>, ggml_tensor*> leaves_;
  std::map<const std::byte*, ggml_tensor*> pool_now_;  // latest K/V pool value per layer
  std::vector<ggml_tensor*> pools_written_;
};

Result<std::unique_ptr<Prepared>> GgmlDevice::build(std::span<const DeviceCall* const> calls, std::vector<uint64_t> key,
                                                    const std::vector<std::vector<SeqRows>>& seqs) {
  auto g = std::make_unique<Prepared>();
  g->key = std::move(key);
  GraphBuilder b(*this, *g, calls, seqs);
  ENGINE_RETURN_IF_ERROR(b.build());
  g->in_buf = ggml_backend_alloc_ctx_tensors(g->ctx_in.get(), backend_);
  if (g->in_buf == nullptr && !g->inputs.empty()) return OutOfMemory(name_ + ": cannot allocate graph inputs");
  g->galloc = ggml_gallocr_new(buft_);
  if (g->galloc == nullptr || !ggml_gallocr_alloc_graph(g->galloc, g->graph)) {
    return OutOfMemory(name_ + ": cannot allocate the graph workspace");
  }
  g->workspace_bytes = static_cast<int64_t>(ggml_gallocr_get_buffer_size(g->galloc, 0)) +
                       (g->in_buf ? static_cast<int64_t>(ggml_backend_buffer_get_size(g->in_buf)) : 0);
  // Constant inputs are written once.
  for (Input& in : g->inputs) {
    if (in.kind == InKind::kConst) ggml_backend_tensor_set(in.t, in.constant.data(), 0, ggml_nbytes(in.t));
  }
  return g;
}

Status GgmlDevice::fill_inputs(Prepared& g, std::span<const DeviceCall* const> calls,
                               const std::vector<std::vector<SeqRows>>& seqs) {
  (void)seqs;
  for (Input& in : g.inputs) {
    if (in.kind == InKind::kConst) continue;
    const DeviceCall& c = *calls[static_cast<size_t>(in.call)];
    // Members were grouped by identical source data when the graph was built;
    // a step that breaks that assumption fails instead of using wrong inputs.
    for (int32_t mi : in.members) {
      const DeviceCall& o = *calls[static_cast<size_t>(mi)];
      if (!same_source(in.kind, c, o, in.seq)) return Internal(name_ + ": calls sharing a graph input disagree");
    }
    const size_t bytes = ggml_nbytes(in.t);
    host_in_.resize(bytes);
    switch (in.kind) {
      case InKind::kIds:
      case InKind::kPositions:
      case InKind::kRows:
        std::memcpy(host_in_.data(), c.ints0.data(), bytes);
        break;
      case InKind::kKvIndex: {
        const KvGeometry& geo = *c.kv[0].geom;
        auto* idx = reinterpret_cast<int64_t*>(host_in_.data());
        for (size_t r = 0; r < c.ints0.size(); ++r) {
          const KvLayerView& kv = c.kv[static_cast<size_t>(c.ints1[r])];
          const int32_t pos = c.ints0[r];
          const auto bi = static_cast<size_t>(pos / geo.block_size);
          if (bi >= kv.block_table.size()) return InvalidArgument("kv_store: position has no KV block");
          const int64_t block = kv.block_table[bi];
          if (block < 0 || block >= geo.num_blocks) return InvalidArgument("kv_store: block id out of range");
          if (geo.layout == KvLayout::kSlotMajor) {
            idx[r] = block * geo.block_size + pos % geo.block_size;
            continue;
          }
          for (int32_t h = 0; h < geo.num_kv_heads; ++h) {
            idx[r * static_cast<size_t>(geo.num_kv_heads) + static_cast<size_t>(h)] =
                (block * geo.num_kv_heads + h) * geo.block_size + pos % geo.block_size;
          }
        }
        break;
      }
      case InKind::kGatherIdx: {
        const KvLayerView& kv = c.ap.kv[static_cast<size_t>(in.seq)];
        const KvGeometry& geo = *kv.geom;
        int32_t ctx = 0;
        for (int32_t r = in.r0; r < in.r0 + in.n; ++r) ctx = std::max(ctx, c.ap.positions[static_cast<size_t>(r)] + 1);
        if (ctx > in.span) return Internal("attention span exceeds its bucket");
        auto* idx = reinterpret_cast<int32_t*>(host_in_.data());
        for (int32_t h = 0; h < geo.num_kv_heads; ++h) {
          for (int32_t t = 0; t < in.span; ++t) {
            int32_t row = 0;  // padding: any valid row, masked out
            if (t < ctx) {
              const auto bi = static_cast<size_t>(t / geo.block_size);
              if (bi >= kv.block_table.size()) return InvalidArgument("attention: position has no KV block");
              row = (kv.block_table[bi] * geo.num_kv_heads + h) * geo.block_size + t % geo.block_size;
            }
            idx[static_cast<size_t>(h) * static_cast<size_t>(in.span) + static_cast<size_t>(t)] = row;
          }
        }
        break;
      }
      case InKind::kMask: {
        auto* mask = reinterpret_cast<ggml_fp16_t*>(host_in_.data());
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        const int32_t win = c.ap.sliding_window;
        for (int32_t i = 0; i < in.n; ++i) {
          const int32_t pos = c.ap.positions[static_cast<size_t>(in.r0 + i)];
          const int32_t lo = win > 0 ? std::max(0, pos - win + 1) : 0;
          for (int32_t t = 0; t < in.span; ++t) {
            mask[static_cast<size_t>(i) * static_cast<size_t>(in.span) + static_cast<size_t>(t)] =
                t >= lo && t <= pos ? zero : ninf;
          }
        }
        break;
      }
      case InKind::kPagedMask: {
        auto* mask = reinterpret_cast<ggml_fp16_t*>(host_in_.data());
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::fill(mask, mask + static_cast<size_t>(in.n) * static_cast<size_t>(in.span), ninf);
        const int32_t win = c.ap.sliding_window;
        for (int32_t i = 0; i < in.n; ++i) {
          const KvLayerView& kv = c.ap.kv[static_cast<size_t>(c.ap.row_seq[static_cast<size_t>(i)])];
          const int32_t bs = kv.geom->block_size;
          const int32_t pos = c.ap.positions[static_cast<size_t>(i)];
          const int32_t lo = win > 0 ? std::max(0, pos - win + 1) : 0;
          ggml_fp16_t* row = mask + static_cast<size_t>(i) * static_cast<size_t>(in.span);
          for (int32_t t = lo; t <= pos; ++t) {
            const int64_t cell = static_cast<int64_t>(kv.block_table[static_cast<size_t>(t / bs)]) * bs + t % bs;
            if (cell >= in.span) return Internal("paged attention: cell outside its span");
            row[cell] = zero;
          }
        }
        break;
      }
      case InKind::kConst: break;
    }
    ggml_backend_tensor_set(in.t, host_in_.data(), 0, bytes);
    stats_.input_bytes += static_cast<int64_t>(bytes);
  }
  return Status::Ok();
}

Status GgmlDevice::run_segment(std::span<const DeviceCall* const> calls, std::span<const uint64_t> signature) {
  ++stats_.segments;
  // Graph key: the recorder's structural signature plus what it leaves out
  // but the graph depends on: per-sequence row splits and padded KV spans.
  std::vector<uint64_t> key(signature.begin(), signature.end());
  std::vector<std::vector<SeqRows>> seqs(calls.size());
  for (size_t i = 0; i < calls.size(); ++i) {
    const DeviceCall& c = *calls[i];
    if (c.m != DeviceCall::M::kAttention) continue;
    if (c.ap.kv.empty()) {
      ++stats_.errors;
      return Internal("attention without KV");
    }
    auto rows = c.ap.kv[0].geom->layout == KvLayout::kSlotMajor ? paged_span(c, opts_.kv_bucket)
                                                                 : seq_rows(c, opts_.kv_bucket);
    if (!rows.ok()) {
      ++stats_.errors;
      return rows.status();
    }
    seqs[i] = std::move(*rows);
    key.push_back(0xA77Eull << 32 | i);
    for (const SeqRows& s : seqs[i]) {
      key.push_back(static_cast<uint64_t>(s.seq) << 32 | static_cast<uint32_t>(s.r0));
      key.push_back(static_cast<uint64_t>(s.n) << 32 | static_cast<uint32_t>(s.span));
    }
  }
  key.push_back(static_cast<uint64_t>(opts_.kv_bucket));
  uint64_t h = 1469598103934665603ull;
  for (uint64_t w : key) h = (h ^ w) * 1099511628211ull;

  if (table_->free_epoch != graphs_epoch_) {  // a bound buffer may be gone
    stats_.graph_evictions += static_cast<int64_t>(graphs_.size());
    graphs_.clear();
    graphs_epoch_ = table_->free_epoch;
  }
  Prepared* g = nullptr;
  if (auto it = graphs_.find(h); it != graphs_.end() && it->second->key == key) {
    g = it->second.get();
    ++stats_.graph_reuses;
  } else {
    const auto t0 = Clock::now();
    if (static_cast<int32_t>(graphs_.size()) >= opts_.max_graphs) {
      stats_.graph_evictions += static_cast<int64_t>(graphs_.size());
      graphs_.clear();
    }
    auto built = build(calls, std::move(key), seqs);
    stats_.build_ns += ns_since(t0);
    if (!built.ok()) {
      ++stats_.errors;
      return built.status();
    }
    ++stats_.graph_builds;
    g = built->get();
    // DYNACORE_GPU_STATS=2: the op mix of each new graph (diagnostics).
    if (const char* v = std::getenv("DYNACORE_GPU_STATS"); v != nullptr && v[0] == '2') {
      std::map<std::string, int> ops;
      for (int i = 0; i < ggml_graph_n_nodes(g->graph); ++i) ++ops[ggml_op_desc(ggml_graph_node(g->graph, i))];
      std::string line;
      for (const auto& [op, n] : ops) line += " " + op + "=" + std::to_string(n);
      std::fprintf(stderr, "[%s] graph %lld nodes:%s\n", name_.c_str(), static_cast<long long>(ggml_graph_n_nodes(g->graph)),
                   line.c_str());
    }
    graphs_[h] = std::move(*built);
  }
  const auto t1 = Clock::now();
  if (Status st = fill_inputs(*g, calls, seqs); !st.ok()) {
    ++stats_.errors;
    return st;
  }
  stats_.input_ns += ns_since(t1);
  const auto t2 = Clock::now();
  const ggml_status gs = ggml_backend_graph_compute(backend_, g->graph);
  stats_.compute_ns += ns_since(t2);
  stats_.graph_nodes = ggml_graph_n_nodes(g->graph);
  if (gs != GGML_STATUS_SUCCESS) {
    ++stats_.errors;
    return Internal(name_ + ": graph execution failed (ggml status " + std::to_string(static_cast<int>(gs)) + ")");
  }
  return Status::Ok();
}

}  // namespace

namespace {

// The ggml backend registry of a provider; null when it is not compiled in.
ggml_backend_reg_t provider_reg(DeviceType type) {
  switch (type) {
#if DYNACORE_HAS_METAL
    case DeviceType::kMetal: return ggml_backend_metal_reg();
#endif
#if DYNACORE_HAS_VULKAN
    case DeviceType::kVulkan: return ggml_backend_vk_reg();
#endif
#if DYNACORE_HAS_CUDA
    case DeviceType::kCuda: return ggml_backend_cuda_reg();
#endif
    default: return nullptr;
  }
}

}  // namespace

bool ggml_gpu_compiled() { return true; }

bool ggml_gpu_compiled(DeviceType type) { return provider_reg(type) != nullptr; }

std::string ggml_revision() { return DYNACORE_GGML_REVISION; }

Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType type, const GgmlDeviceOptions& opts) {
  ggml_backend_reg_t reg = provider_reg(type);
  if (reg == nullptr) return Unsupported("this GPU provider is not built in (ENABLE_METAL / ENABLE_VULKAN / ENABLE_CUDA)");
  const size_t count = ggml_backend_reg_dev_count(reg);
  size_t index = 0;
  if (const char* v = std::getenv("DYNACORE_GPU_INDEX"); v != nullptr && *v != '\0') {
    char* end = nullptr;
    const unsigned long long i = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0') return InvalidArgument(std::string("DYNACORE_GPU_INDEX=") + v + " is not a device index");
    index = static_cast<size_t>(i);
  }
  if (index >= count) {
    return Unsupported(std::string("no ") + ggml_backend_reg_name(reg) + " GPU device " + std::to_string(index) + " (" +
                       std::to_string(count) + " found)");
  }
  ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, index);
  ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
  if (backend == nullptr) return Unsupported(std::string("cannot initialize GPU device ") + ggml_backend_dev_name(dev));
  return std::unique_ptr<Device>(std::make_unique<GgmlDevice>(dev, backend, type, opts));
}

}  // namespace dynacore

#else  // !DYNACORE_HAS_GGML

namespace dynacore {

bool ggml_gpu_compiled() { return false; }

bool ggml_gpu_compiled(DeviceType) { return false; }

std::string ggml_revision() { return {}; }

Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType, const GgmlDeviceOptions&) {
  return Unsupported("this build has no GPU backend (configure with -DENABLE_METAL, -DENABLE_VULKAN or -DENABLE_CUDA)");
}

}  // namespace dynacore

#endif

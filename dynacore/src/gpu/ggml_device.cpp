#include "ggml_device_internal.h"

#include "dynacore/hardware/cpu_info.h"

#if DYNACORE_HAS_GGML

namespace dynacore {
namespace ggml_detail {

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
    auto rows = c.ap.kv[0].geom->layout == KvLayout::kSlotMajor ? paged_span(c, opts_.kv_bucket, type_)
                                                                 : seq_rows(c, opts_.kv_bucket);
    if (!rows.ok()) {
      ++stats_.errors;
      return rows.status();
    }
    seqs[i] = std::move(*rows);
    for (const SeqRows& s : seqs[i]) stats_.sparse_attention += s.kv_max > 0 ? 1 : 0;
    key.push_back(0xA77Eull << 32 | i);
    for (const SeqRows& s : seqs[i]) {
      key.push_back(static_cast<uint64_t>(s.seq) << 32 | static_cast<uint32_t>(s.r0));
      key.push_back(static_cast<uint64_t>(s.n) << 32 | static_cast<uint32_t>(s.span));
      key.push_back(static_cast<uint64_t>(static_cast<uint32_t>(s.kv_max)));
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
    // Full: drop the least recently used graph, not all of them. Clearing
    // the cache also dropped the hot decode graphs, rebuilt on the next step
    // (v0.1.7, 8 users: 33 evictions for 71 builds). A colliding hash slot is
    // replaced below.
    if (static_cast<int32_t>(graphs_.size()) >= opts_.max_graphs && graphs_.find(h) == graphs_.end()) {
      auto lru = graphs_.begin();
      for (auto gi = graphs_.begin(); gi != graphs_.end(); ++gi) {
        if (gi->second->last_use < lru->second->last_use) lru = gi;
      }
      graphs_.erase(lru);
      ++stats_.graph_evictions;
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
    // Workspace cap: each prepared graph keeps its own activation buffers (about
    // 86 MiB per graph for an 8B model). 26 of them (2.2 GiB) on top of 8.5 GB of
    // weights and 8 GB of KV made Metal fail steps (out of memory) at 16 users.
    // Drop least recently used graphs (never the one about to run) past the cap.
    const int64_t cap = graph_workspace_cap();
    for (int64_t ws = memory_stats().workspace_bytes; ws > cap && graphs_.size() > 1;) {
      auto lru = graphs_.end();
      for (auto gi = graphs_.begin(); gi != graphs_.end(); ++gi) {
        if (gi->first != h && (lru == graphs_.end() || gi->second->last_use < lru->second->last_use)) lru = gi;
      }
      ws -= lru->second->workspace_bytes;
      graphs_.erase(lru);
      ++stats_.graph_evictions;
    }
    peak_workspace_ = std::max(peak_workspace_, memory_stats().workspace_bytes);
  }
  g->last_use = ++use_tick_;
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

}  // namespace ggml_detail

using ggml_detail::CtxPtr;
using ggml_detail::meta_ctx;


namespace {

// ggml's own log: warnings and errors only, so a normal run shows DynaLM's
// log, not every ggml device-initialization line. DYNACORE_GPU_LOG=1 shows all.
void quiet_ggml_log(ggml_log_level level, const char* text, void*) {
  thread_local ggml_log_level last = GGML_LOG_LEVEL_INFO;  // CONT continues the previous line
  if (level != GGML_LOG_LEVEL_CONT) last = level;
  if (last >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

void install_ggml_log() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* v = std::getenv("DYNACORE_GPU_LOG");
    if (v == nullptr || v[0] != '1') ggml_log_set(quiet_ggml_log, nullptr);
  });
}

// The ggml backend registry of a provider; null when it is not compiled in.
// Calling it initializes the provider (device discovery).
ggml_backend_reg_t provider_reg(DeviceType type) {
  install_ggml_log();
  switch (type) {
    case DeviceType::kCpu: return ggml_backend_cpu_reg();
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

// Whether `dev` supports representative instances of the ops every model
// step uses: matmul with f32 / f16 / q8_0 / q4_K weights, flash attention
// over f16 K/V with an f16 mask, RMS norm, RoPE, row store and gather.
Status check_core_ops(ggml_backend_dev_t dev) {
  CtxPtr ctx = meta_ctx(32);
  ggml_context* c = ctx.get();
  const int64_t k = 256, n = 64, rows = 8, hd = 64, heads = 4, cells = 256;
  ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, rows);
  std::vector<std::pair<const char*, ggml_tensor*>> ops;
  for (ggml_type t : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
    ops.emplace_back(ggml_type_name(t), ggml_mul_mat(c, ggml_new_tensor_2d(c, t, k, n), x));
  }
  ggml_tensor* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, rows, heads);
  ggml_tensor* kv_k = ggml_new_tensor_3d(c, GGML_TYPE_F16, hd, cells, heads);
  ggml_tensor* kv_v = ggml_new_tensor_3d(c, GGML_TYPE_F16, hd, cells, heads);
  ggml_tensor* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, cells, rows);
  ggml_tensor* fa = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kv_k, kv_v, mask, 0.125f, 0.0f, 0.0f);
  (void)ggml_prec_set_acc(fa, GGML_PREC_F32);
  ops.emplace_back("flash attention", fa);
  ops.emplace_back("rms_norm", ggml_rms_norm(c, x, 1e-6f));
  ggml_tensor* qr = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, heads, rows);
  ops.emplace_back("rope", ggml_rope_ext(c, qr, ggml_new_tensor_1d(c, GGML_TYPE_I32, rows), nullptr, static_cast<int>(hd),
                                         GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
  ggml_tensor* pool = ggml_new_tensor_2d(c, GGML_TYPE_F16, hd, cells);
  ops.emplace_back("set_rows", ggml_set_rows(c, pool, ggml_new_tensor_2d(c, GGML_TYPE_F32, hd, rows),
                                             ggml_new_tensor_1d(c, GGML_TYPE_I64, rows)));
  ops.emplace_back("get_rows", ggml_get_rows(c, ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, k, n),
                                             ggml_new_tensor_1d(c, GGML_TYPE_I32, rows)));
  for (const auto& [what, t] : ops) {
    if (!ggml_backend_dev_supports_op(dev, t)) {
      return Unsupported(std::string(ggml_backend_dev_description(dev)) + " cannot run " + ggml_op_desc(t) + " (" + what +
                         ") on the GPU; this GPU is not supported (use --backend cpu)");
    }
  }
  return Status::Ok();
}

// A GPU backend is built in (ggml alone may be here for the ggml-cpu device).
bool ggml_gpu_compiled() {
#if DYNACORE_HAS_METAL || DYNACORE_HAS_VULKAN || DYNACORE_HAS_CUDA
  return true;
#else
  return false;
#endif
}

// Compile-time only: answering must not initialize a GPU (dynalm --version).
bool ggml_gpu_compiled(DeviceType type) {
  switch (type) {
    case DeviceType::kCpu: return true;
#if DYNACORE_HAS_METAL
    case DeviceType::kMetal: return true;
#endif
#if DYNACORE_HAS_VULKAN
    case DeviceType::kVulkan: return true;
#endif
#if DYNACORE_HAS_CUDA
    case DeviceType::kCuda: return true;
#endif
    default: return false;
  }
}

std::string ggml_revision() { return DYNACORE_GGML_REVISION; }

std::vector<std::string> ggml_gpu_device_names(DeviceType type) {
  std::vector<std::string> out;
  if (type == DeviceType::kCpu) return out;
  ggml_backend_reg_t reg = provider_reg(type);
  if (reg == nullptr) return out;
  for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
    out.emplace_back(ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, i)));
  }
  return out;
}

bool ggml_cpu_usable_here() {
#if defined(__x86_64__) || defined(_M_X64)
  const CpuFeatures& f = cpu_info().features;
  return f.avx2 && f.fma && f.f16c;
#else
  return true;
#endif
}

Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType type, const GgmlDeviceOptions& opts) {
  // ggml-cpu is compiled for the AVX2 tier on x86-64 (cmake/Ggml.cmake): on a
  // CPU without it the first graph would die of SIGILL. Every caller (the
  // device registry, tests, benchmarks) gets a clean refusal instead.
  if (type == DeviceType::kCpu && !ggml_cpu_usable_here()) {
    return Unsupported("the ggml-cpu backend needs a CPU with AVX2, FMA and F16C");
  }
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
  // The device must run the core ops of a transformer step. A GPU can start
  // and still lack them (ggml-metal needs simdgroup reductions, which
  // paravirtual / Apple5-class GPUs do not have); refusing here lets
  // `--backend auto` fall back to the CPU instead of failing the first request.
  if (Status st = check_core_ops(dev); !st.ok()) return st;
  ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
  if (backend != nullptr && type == DeviceType::kCpu && opts.threads > 0) ggml_backend_cpu_set_n_threads(backend, opts.threads);
  if (backend == nullptr) return Unsupported(std::string("cannot initialize GPU device ") + ggml_backend_dev_name(dev));
  auto d = std::make_unique<ggml_detail::GgmlDevice>(dev, backend, type, opts);
  // ggml-cpu: one persistent worker pool, as llama.cpp attaches. Without it
  // ggml starts and joins its threads for every graph, i.e. every token;
  // the pool's workers spin briefly between graphs, then sleep.
  // DYNACORE_GGML_THREADPOOL=0: the per-graph threads (A/B).
  const char* tpv = std::getenv("DYNACORE_GGML_THREADPOOL");
  if (type == DeviceType::kCpu && opts.threads > 0 && !(tpv != nullptr && tpv[0] == '0')) {
    ggml_threadpool_params tpp = ggml_threadpool_params_default(opts.threads);
    if (ggml_threadpool* tp = ggml_threadpool_new(&tpp)) {
      ggml_backend_cpu_set_threadpool(backend, tp);
      d->adopt_threadpool(tp);
    }
  }
  return std::unique_ptr<Device>(std::move(d));
}

}  // namespace dynacore

#else  // !DYNACORE_HAS_GGML

namespace dynacore {

bool ggml_gpu_compiled() { return false; }

bool ggml_gpu_compiled(DeviceType) { return false; }

std::string ggml_revision() { return {}; }

std::vector<std::string> ggml_gpu_device_names(DeviceType) { return {}; }

bool ggml_cpu_usable_here() { return false; }

Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType, const GgmlDeviceOptions&) {
  return Unsupported("this build has no GPU backend (configure with -DENABLE_METAL, -DENABLE_VULKAN or -DENABLE_CUDA)");
}

}  // namespace dynacore

#endif

// Sparse K/V policy for slot-major attention (ggml_device.h). Pure, and
// defined in every build so it can be unit-tested anywhere.
namespace dynacore {

GpuSparseMode gpu_sparse_mode() {
  static const GpuSparseMode mode = [] {
    const char* v = std::getenv("DYNACORE_GPU_SPARSE");
    if (v == nullptr) return GpuSparseMode::kAuto;
    const std::string_view s(v);
    return s == "1" ? GpuSparseMode::kAlways : s == "0" ? GpuSparseMode::kNever : GpuSparseMode::kAuto;
  }();
  return mode;
}

int32_t sparse_kv_bound(int64_t span, int64_t kv_max, GpuSparseMode mode, DeviceType type) {
  if (kv_max <= 0 || kv_max >= span) return 0;  // nothing to skip
  switch (mode) {
    case GpuSparseMode::kNever: return 0;
    case GpuSparseMode::kAlways: return static_cast<int32_t>(kv_max);
    case GpuSparseMode::kAuto:
      return type == DeviceType::kMetal && kv_max <= kSparseMaxKv && span >= kSparseSpanRatio * kv_max
                 ? static_cast<int32_t>(kv_max)
                 : 0;
  }
  return 0;
}

}  // namespace dynacore

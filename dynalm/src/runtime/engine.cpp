#include "runtime/engine.h"

#include "dynacore/ir/passes.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/gpu/ggml_device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "dynacore/base/timer.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/thread_qos.h"
#include "runtime/autotune.h"
#include "runtime/sequence.h"
#include "runtime/text_stream.h"
#include "common/core.h"

namespace dynalm {

std::string_view stream_finish_name(StreamFinish f) {
  switch (f) {
    case StreamFinish::kNone: return "none";
    case StreamFinish::kStop: return "stop";
    case StreamFinish::kLength: return "length";
    case StreamFinish::kCancelled: return "cancelled";
    case StreamFinish::kError: return "error";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// RequestStream

void RequestStream::push(StreamEvent ev) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (final_pushed_) return;  // nothing after the final event
    final_pushed_ = ev.done;
    events_.push_back(std::move(ev));
  }
  cv_.notify_one();
}

bool RequestStream::next(StreamEvent& out, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mu_);
  if (final_popped_) return false;
  if (!cv_.wait_for(lock, timeout, [&] { return !events_.empty(); })) return false;
  out = std::move(events_.front());
  events_.pop_front();
  if (out.done) final_popped_ = true;
  return true;
}

bool RequestStream::finished() const {
  std::lock_guard<std::mutex> lock(mu_);
  return final_pushed_;
}

void RequestStream::cancel() {
  if (cancel_fn_) cancel_fn_();
}

// ---------------------------------------------------------------------------
// Engine

namespace {

// Per-request streaming state, owned by the scheduler callback.
struct RequestState {
  std::shared_ptr<RequestStream> stream;
  TextStreamer streamer;
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
  bool ended = false;  // final event already pushed (e.g. a stop string matched)

  RequestState(const Tokenizer& tok, std::vector<std::string> stops) : streamer(tok, std::move(stops)) {}
};

StreamFinish map_finish(const RequestEvent& ev) {
  switch (ev.status) {
    case SequenceStatus::kFinished: return ev.reason == FinishReason::kLength ? StreamFinish::kLength : StreamFinish::kStop;
    case SequenceStatus::kCancelled: return StreamFinish::kCancelled;
    default: return StreamFinish::kError;
  }
}

}  // namespace

int64_t auto_kv_tokens(const ModelConfig& c, DType kv_dtype, int64_t weight_bytes, int64_t available_ram) {
  constexpr int64_t kBlock = 16, kMax = 65536;
  if (available_ram <= 0) return std::min<int64_t>(c.context_length, 16384);
  const int64_t per_token = std::max<int64_t>(1, c.kv_bytes_per_token(kv_dtype));
  const int64_t budget = std::max<int64_t>(0, available_ram - weight_bytes) / 2;
  const int64_t floor = std::min<int64_t>(c.context_length, 2048);
  const int64_t tokens = std::clamp<int64_t>(budget / per_token, floor, std::max(floor, kMax));
  return std::max(kBlock, tokens / kBlock * kBlock);
}

int64_t device_kv_budget(int64_t working_set, int64_t allocated, int64_t weight_bytes, int64_t workspace_reserve,
                         int64_t scratch_bytes) {
  if (working_set <= 0) return -1;
  const int64_t usable = working_set / 100 * (100 - kWorkingSetMarginPct);
  const int64_t reserve = workspace_reserve > 0 ? workspace_reserve : kDefaultWorkspaceReserve;
  return std::max<int64_t>(0, usable - std::max<int64_t>(0, allocated) - std::max<int64_t>(0, weight_bytes) - reserve -
                                  std::max<int64_t>(0, scratch_bytes));
}

int64_t kv_tokens_within(const ModelConfig& c, DType kv_dtype, int64_t budget) {
  const int64_t block = kv_geometry_for(c, kv_dtype, 16, 16).total_bytes();
  return block > 0 && budget > 0 ? budget / block * 16 : 0;
}

Status check_memory_fit(int64_t weight_bytes, int64_t kv_bytes, int64_t kv_per_token, int64_t total_ram,
                        int64_t device_free, int64_t kv_budget) {
  auto gb = [](int64_t b) { return std::to_string(b / (1 << 20) / 1024) + "." + std::to_string(b / (1 << 20) % 1024 * 10 / 1024) + " GiB"; };
  if (kv_budget >= 0 && kv_bytes > kv_budget) {
    // A capped device (Metal working set): the error must come here, at load,
    // not as a GPU out-of-memory failure in the first large step.
    const int64_t fit = kv_per_token > 0 ? kv_budget / kv_per_token / 16 * 16 : 0;
    std::string m = "not enough GPU working set: the KV cache needs " + gb(kv_bytes) + " and " + gb(kv_budget) +
                    " is left for it after the model (" + gb(weight_bytes) + "), the graph workspace and a " +
                    std::to_string(kWorkingSetMarginPct) + "% margin";
    m += fit >= 16 ? "; use --ctx " + std::to_string(fit) + " or less, or --kv q8_0 (half the KV memory)"
                   : "; this model does not fit on the GPU, use --cpu or a smaller or more quantized model";
    return ResourceExhausted(m);
  }
  auto check = [&](int64_t limit, const char* where) -> Status {
    if (limit <= 0 || weight_bytes + kv_bytes <= limit) return Status::Ok();
    const int64_t fit = kv_per_token > 0 ? std::max<int64_t>(0, limit - weight_bytes) / kv_per_token / 16 * 16 : 0;
    std::string m = std::string("not enough ") + where + ": the model needs " + gb(weight_bytes) + " and the KV cache " +
                    gb(kv_bytes) + ", " + gb(weight_bytes + kv_bytes) + " in all; the limit is " + gb(limit);
    m += fit >= 16 ? "; use --ctx " + std::to_string(fit) + " or less, or --kv q8_0 (half the KV memory)"
                   : "; this model does not fit, use a smaller or more quantized model";
    return ResourceExhausted(m);
  };
  if (device_free >= 0) ENGINE_RETURN_IF_ERROR(check(device_free / 100 * 95, "GPU memory"));
  return check(total_ram / 10 * 9, "memory");
}

namespace {

// Maps every weight page into the process before the first request. Weights
// are memory-mapped, so otherwise the first forward pass takes one page fault
// per 4 KiB page (~600k for a 2.4 GB model), serially inside the kernels, even
// when the file is already in the OS cache; a cold file is also read page by
// page. Touching the pages from all workers at load turns that into parallel,
// one-time work. Skipped when the weights do not fit in available memory
// (touching would only evict other pages) or with DYNALM_PREFAULT=0.
void prefault_weights(const TensorRegistry& weights, int64_t weight_bytes, ThreadPool& pool) {
  if (const char* v = std::getenv("DYNALM_PREFAULT"); v && std::strcmp(v, "0") == 0) return;
  if (weight_bytes > memory_info().available_bytes) return;
  struct Range {
    const volatile unsigned char* p;
    size_t n;
  };
  constexpr size_t kChunk = size_t{2} << 20;
  std::vector<Range> chunks;
  weights.for_each([&](const Tensor& t) {
    const auto* p = static_cast<const volatile unsigned char*>(t.data());
    const auto n = static_cast<size_t>(t.view().span_bytes());
    for (size_t off = 0; off < n; off += kChunk) chunks.push_back({p + off, std::min(kChunk, n - off)});
  });
  pool.parallel_for(chunks.size(), 1, [&](size_t begin, size_t end) {
    unsigned sink = 0;
    for (size_t i = begin; i < end; ++i) {
      for (size_t off = 0; off < chunks[i].n; off += 4096) sink += chunks[i].p[off];
      sink += chunks[i].p[chunks[i].n - 1];
    }
    static_cast<void>(sink);
  });
}

}  // namespace

std::string_view execution_mode_name(ExecutionMode m) {
  return m == ExecutionMode::kCompiled ? "compiled" : "reference";
}

Result<ExecutionMode> parse_execution_mode(std::string_view s) {
  if (s == "reference" || s == "auto") return ExecutionMode::kReference;
  if (s == "compiled") return ExecutionMode::kCompiled;
  return InvalidArgument("unknown execution mode '" + std::string(s) + "' (reference, compiled)");
}

Status apply_backend_choice(std::string_view choice, EngineOptions& o) {
  o.backend_auto = choice == "auto" || choice == "gpu";
  o.backend_require_gpu = choice == "gpu";
  o.cpu_ggml = choice == "auto" || choice == "cpu";
  if (o.backend_auto || choice == "native-cpu") {
    o.backend = DeviceKind::kCpu;
    return Status::Ok();
  }
  auto k = parse_device_kind(choice);
  if (!k.ok()) {
    return InvalidArgument("unknown backend '" + std::string(choice) +
                           "' (auto, gpu, cpu, native-cpu, ggml-cpu, metal, vulkan, cuda)");
  }
  o.backend = *k;
  return Status::Ok();
}

Result<std::unique_ptr<Engine>> Engine::create(EngineOptions opts) {
  const bool choose = opts.backend_auto || (opts.backend == DeviceKind::kCpu && opts.cpu_ggml);
  if (!choose) return create_on(std::move(opts));
  // Pick devices (runtime/device_select.h), then open them in order.
  ENGINE_ASSIGN_OR_RETURN(auto probe, load_model(opts.model_path));
  const DeviceRequest req{opts.backend, opts.backend_auto, opts.backend_require_gpu, opts.cpu_ggml, opts.kv_dtype};
  const std::vector<DeviceKind> compiled = compiled_devices();
  const DevicePlan plan =
      plan_devices(probe->config, req, compiled, dynacore::ggml_gpu_compiled(dynacore::DeviceType::kCpu));
  probe.reset();
  if (!plan.error.ok()) return plan.error;
  if (!plan.note.empty()) LOG_INFO("Device: {}", plan.note);
  opts.backend_auto = opts.backend_require_gpu = opts.cpu_ggml = false;
  Status last = Status::Ok();
  for (DeviceKind k : plan.candidates) {
    EngineOptions o = opts;
    o.backend = k;
    auto r = create_on(std::move(o));
    if (r.ok()) return r;
    last = r.status();
    LOG_WARN("Device: {} failed to start ({})", device_kind_name(k), r.status().message());
  }
  if (!plan.native_cpu_last) return Unsupported("--gpu: no GPU could start (" + last.message() + "); use --cpu");
  opts.backend = DeviceKind::kCpu;
  return create_on(std::move(opts));
}

Result<std::unique_ptr<Engine>> Engine::create_on(EngineOptions opts) {
  request_full_speed_process();  // DD-052
  std::unique_ptr<Engine> e(new Engine());
  ENGINE_ASSIGN_OR_RETURN(e->model_, load_model(opts.model_path));
  const ModelConfig& c = e->model_->config;
  if (opts.backend != DeviceKind::kCpu) ENGINE_RETURN_IF_ERROR(gpu_model_support(c, opts.kv_dtype, opts.backend));
  // Default threads: every physical core, except for ggml's CPU kernels on a
  // hybrid CPU: they split work evenly and meet at barriers, so efficiency cores
  // hold back the step (M4 Pro, 1.5B decode: 12 threads 81 tok/s, 8 P-cores 138).
  // DynaCore's pool hands out chunks dynamically and keeps all cores.
  int threads = opts.threads > 0 ? opts.threads : cpu_info().physical_cores;
  if (opts.threads <= 0 && opts.backend == DeviceKind::kGgmlCpu && cpu_info().performance_cores > 0) {
    threads = cpu_info().performance_cores;
  }
  // A thread count measured by `dynalm autotune --run` on this machine and
  // build, when the caller left threads on auto (runtime/autotune.h).
  if (opts.threads <= 0) {
    if (const int32_t t = tuned_threads(opts.backend, c, e->model_->quantization); t > 0) {
      LOG_INFO("Threads: {} (tuned, {})", t, tune_cache_path());
      threads = t;
    }
  }
  e->pool_ = std::make_unique<ThreadPool>(threads);
  e->kind_ = opts.backend;
  prefault_weights(e->model_->weights, e->model_->weight_bytes, *e->pool_);
  ENGINE_ASSIGN_OR_RETURN(e->backend_, create_device(opts.backend, *e->pool_));
  Device* device = e->backend_.get();
  // A device that runs whole recorded steps (the GPU, docs/gpu-backend.md)
  // is never driven op by op: its steps always go through the recorder.
  const bool segment_device = e->backend_->segment_executor() != nullptr;
  if (segment_device && opts.execution != ExecutionMode::kCompiled) {
    LOG_INFO("Execution: compiled (required by the {} device)", e->backend_->name());
    opts.execution = ExecutionMode::kCompiled;
  }
  // Device defaults (explicit settings always win). Solo-prefill steps carry
  // no decode rows, so a larger GPU chunk cannot stall running streams; mixed
  // steps keep the DD-079 decode protection.
  if (opts.scheduler.solo_prefill_chunk < 0) opts.scheduler.solo_prefill_chunk = segment_device ? kGpuSoloPrefillChunk : 0;
  if (segment_device && opts.device_scheduler_defaults && opts.scheduler.policy == SchedulerPolicy::kBalanced) {
    opts.scheduler.prefill_token_budget = kGpuMixedPrefillBudget;
    opts.scheduler.max_prefill_chunk = kGpuMaxPrefillChunk;
    opts.scheduler.decode_latency_target = kGpuDecodeLatencyTarget;
    opts.scheduler.decode_protect_predictive = kGpuDecodePredictive;
    opts.scheduler.protected_prefill_min = kGpuProtectedPrefillMin;
  }
  if (opts.max_batch_tokens <= 0) {
    opts.max_batch_tokens = std::max({256, opts.scheduler.prefill_token_budget + opts.scheduler.decode_token_budget + 1,
                                      opts.scheduler.solo_prefill_chunk});
  }
  // int8 activation rows (DD-053) on DynaCore's CPU kernels. ARM64: up to
  // kNeonInt8Rows, where the SDOT row kernels beat the fp32 expand GEMM at
  // prefill sizes too (M4 Pro, Qwen2.5-1.5B, 401-token prompt: 2394 -> 1835 ms;
  // the per-row arithmetic is decode's, prefill chunks of 32/64: perplexity
  // +0.04%, mean KL 0.0012). x86: the plan default (4), the measured AVX2 crossover.
  int32_t int8_rows = opts.int8_decode_rows;
#if defined(__aarch64__) || defined(_M_ARM64)
  if (int8_rows < 0 && opts.backend == DeviceKind::kCpu) int8_rows = kNeonInt8Rows;
#endif
  if (opts.execution == ExecutionMode::kCompiled) {
    auto rec = std::make_unique<dynacore::ir::RecordingDevice>(*e->backend_, dynacore::ir::RecordingDevice::Mode::kDeferred);
    if (!segment_device) {  // the CPU compiler pipeline; a segment device plans its own graphs
      KernelPlan base = KernelPlan::defaults();
      if (int8_rows >= 0) base.int8_decode_max_rows = int8_rows;
      dynacore::ir::CompileOptions co;
      co.cost = dynacore::ir::CostModel::from(base, threads);
      rec->set_planner(dynacore::ir::make_planner(co));
    }
    device = rec.get();
    e->compiled_ = std::move(rec);
  }
  // Auto KV capacity from free host memory, and from free device memory on a
  // GPU with its own memory (weights are copied there too).
  int64_t available = memory_info().available_bytes;
  const Device::MemoryStats dev_mem = device->memory_stats();
  if (const int64_t dev_free = dev_mem.free_bytes; dev_free >= 0) {
    available = available > 0 ? std::min(available, dev_free) : dev_free;
  }
  // A device with a working-set cap (Metal): host free RAM says nothing about
  // it, so the KV cache also gets at most what the cap leaves.
  const int64_t scratch = estimate_memory(c, 0, 0, opts.kv_dtype, opts.max_batch_tokens).activation_bytes;
  const int64_t kv_budget = device_kv_budget(dev_mem.working_set_bytes, dev_mem.allocated_bytes,
                                             e->model_->weight_bytes, dev_mem.workspace_limit_bytes, scratch);
  int64_t kv_tokens = opts.kv_tokens > 0 ? opts.kv_tokens
                                         : auto_kv_tokens(c, opts.kv_dtype, e->model_->weight_bytes, available);
  if (opts.kv_tokens <= 0 && kv_budget >= 0) {
    const int64_t fit = kv_tokens_within(c, opts.kv_dtype, kv_budget);
    if (fit >= std::min<int64_t>(kMinAutoKvTokens, c.context_length) && fit < kv_tokens) {
      LOG_INFO("KV cache: {} tokens (the GPU working set leaves {} MiB for it; the free RAM would allow {})", fit,
               kv_budget >> 20, kv_tokens);
      kv_tokens = fit;
    }
  }
  KvGeometry kv_geom = kv_geometry_for(c, opts.kv_dtype, 16, kv_tokens);
  // The device's KV layout (paged_kv.h): head-major on the CPU, slot-major on
  // the GPU (attention reads the paged pool in place). DYNALM_KV_LAYOUT=head
  // selects the GPU's gather path instead, for A/B measurement only.
  kv_geom.layout = device->preferred_kv_layout();
  if (const char* kl = std::getenv("DYNALM_KV_LAYOUT"); kl != nullptr && *kl != '\0') {
    const std::string_view v(kl);
    if (v == "head" || v == "slot") {
      kv_geom.layout = v == "head" ? KvLayout::kHeadMajor : KvLayout::kSlotMajor;
    } else {
      LOG_WARN("ignoring DYNALM_KV_LAYOUT={} (head | slot)", v);
    }
  }
  {
    // Memory admission: refuse a KV cache that cannot fit instead of failing
    // (or swapping) mid-run. Within physical RAM but above what is available
    // now (other apps, file cache): a warning; macOS frees file cache on demand.
    const int64_t kv_need = static_cast<int64_t>(kv_geom.total_bytes());
    const int64_t dev_free = device->memory_stats().free_bytes;
    ENGINE_RETURN_IF_ERROR(check_memory_fit(e->model_->weight_bytes, kv_need, c.kv_bytes_per_token(opts.kv_dtype),
                                            memory_info().total_bytes, dev_free, kv_budget));
    if (const int64_t avail = memory_info().available_bytes; avail > 0 && e->model_->weight_bytes + kv_need > avail) {
      LOG_WARN("Memory: the model and KV cache need {} MiB; {} MiB is available now (other apps may be swapped out)",
               (e->model_->weight_bytes + kv_need) >> 20, avail >> 20);
    }
  }
  ENGINE_ASSIGN_OR_RETURN(e->kv_, KvBlockPool::create(kv_geom, *device));
  ENGINE_ASSIGN_OR_RETURN(e->transformer_, Transformer::create(c, e->model_->weights, *device, opts.max_batch_tokens,
                                                                  std::min<int64_t>(kv_tokens, c.context_length)));
  ENGINE_RETURN_IF_ERROR(device->take_error());
  if (int8_rows >= 0) {
    KernelPlan base = KernelPlan::defaults();
    base.int8_decode_max_rows = int8_rows;
    e->transformer_->set_kernel_base(base);
  }
  if (!opts.draft_model_path.empty()) {
    // The draft: same device, same KV dtype, its own KV pool sized like the target's.
    ENGINE_ASSIGN_OR_RETURN(e->draft_model_, load_model(opts.draft_model_path));
    const ModelConfig& dc = e->draft_model_->config;
    if (opts.backend != DeviceKind::kCpu) ENGINE_RETURN_IF_ERROR(gpu_model_support(dc, opts.kv_dtype, opts.backend));
    KvGeometry dg = kv_geometry_for(dc, opts.kv_dtype, 16, kv_tokens);
    dg.layout = device->preferred_kv_layout();
    // Both models and both KV caches against the working set as measured
    // before the target's KV pool existed (dev_mem): nothing counted twice.
    ENGINE_RETURN_IF_ERROR(check_memory_fit(
        e->model_->weight_bytes + e->draft_model_->weight_bytes,
        static_cast<int64_t>(kv_geom.total_bytes() + dg.total_bytes()),
        c.kv_bytes_per_token(opts.kv_dtype) + dc.kv_bytes_per_token(opts.kv_dtype), memory_info().total_bytes,
        device->memory_stats().free_bytes,
        device_kv_budget(dev_mem.working_set_bytes, dev_mem.allocated_bytes,
                         e->model_->weight_bytes + e->draft_model_->weight_bytes, dev_mem.workspace_limit_bytes,
                         scratch + estimate_memory(dc, 0, 0, opts.kv_dtype, opts.max_batch_tokens).activation_bytes)));
    ENGINE_ASSIGN_OR_RETURN(e->draft_kv_, KvBlockPool::create(dg, *device));
    ENGINE_ASSIGN_OR_RETURN(e->draft_transformer_,
                            Transformer::create(dc, e->draft_model_->weights, *device, opts.max_batch_tokens));
    ENGINE_RETURN_IF_ERROR(device->take_error());
    ENGINE_ASSIGN_OR_RETURN(e->drafter_, BatchDrafter::create(*e->draft_transformer_, *e->draft_kv_,
                                                              *e->draft_model_->tokenizer, *e->model_->tokenizer));
    if (opts.scheduler.spec_k <= 0) opts.scheduler.spec_k = 4;
    LOG_INFO("Speculative decoding: draft model {} ({}), up to {} drafts per stream",
             dc.name.empty() ? dc.architecture : dc.name, e->draft_model_->quantization, opts.scheduler.spec_k);
  }
  e->scheduler_ = std::make_unique<Scheduler>(*e->transformer_, *e->kv_, *e->model_->tokenizer, opts.scheduler);
  if (e->drafter_) e->scheduler_->set_draft_model(e->drafter_.get());
  e->core_ = std::make_shared<Core>();
  e->core_->sched = e->scheduler_.get();
  e->thread_ = std::thread([eng = e.get()] { eng->loop(); });
  return e;
}

Engine::~Engine() {
  {
    std::lock_guard<std::mutex> lock(wake_mu_);
    stop_ = true;
  }
  wake_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  // Compiled execution: run recorded work while the buffers it references
  // (transformer scratch, KV) still exist, after the scheduler thread (the
  // device's only other user) has stopped.
  if (compiled_) (void)static_cast<dynacore::ir::RecordingDevice&>(*compiled_).flush();
  if (core_) {  // null when create() failed part-way (e.g. a corrupt model file)
    std::lock_guard<std::mutex> lock(core_->mu);
    core_->sched = nullptr;  // streams' cancel() becomes a no-op
  }
  // Requests still in flight end as cancelled; the scheduler's destructor
  // releases their KV.
  for (auto& weak : live_) {
    if (auto st = weak.lock()) {
      StreamEvent ev;
      ev.done = true;
      ev.finish = StreamFinish::kCancelled;
      st->push(std::move(ev));
    }
  }
}

void Engine::loop() {
  request_full_speed_thread();  // the scheduler thread also runs kernel chunks (DD-052)
  pool_->pin_caller();
  // Throughput window: restarts whenever the engine wakes from idle.
  int64_t window_start = 0;
  uint64_t window_gen = 0, window_prefill = 0, prefill_total = 0;
  double gen_rate = 0, prefill_rate = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(wake_mu_);
      wake_cv_.wait(lock, [&] { return stop_ || !scheduler_->idle(); });
      if (stop_) return;
    }
    if (const bool want = want_profiling_.load(std::memory_order_relaxed); want != profiling_) {
      profiling_ = want;
      transformer_->set_profiling(want);
      pool_->set_stats_enabled(want);
    }
    const uint64_t gen_before = scheduler_->stats().tokens_generated;
    const int64_t t0 = now_ns();
    if (window_start == 0) window_start = t0;
    scheduler_->step();
    const int64_t t1 = now_ns();
    step_ms_.observe(static_cast<double>(t1 - t0) * 1e-6);

    EngineStats st;
    st.scheduler = scheduler_->stats();
    window_gen += st.scheduler.tokens_generated - gen_before;
    window_prefill += static_cast<uint64_t>(st.scheduler.last_prefill_rows);
    prefill_total += static_cast<uint64_t>(st.scheduler.last_prefill_rows);
    if (const double secs = static_cast<double>(t1 - window_start) * 1e-9; secs >= 1.0) {
      gen_rate = static_cast<double>(window_gen) / secs;
      prefill_rate = static_cast<double>(window_prefill) / secs;
      window_start = t1;
      window_gen = window_prefill = 0;
    }
    if (scheduler_->idle()) {  // going to sleep: rates are no longer current
      gen_rate = prefill_rate = 0;
      window_start = 0;
      window_gen = window_prefill = 0;
    }
    st.generation_tok_s = gen_rate;
    st.prefill_tok_s = prefill_rate;
    st.prefill_tokens = prefill_total;
    if (const PrefixCache* pc = scheduler_->prefix_cache()) st.prefix = pc->stats();
    st.kv_blocks_total = kv_->num_blocks();
    st.kv_blocks_used = kv_->used_blocks();
    st.tokenize_ms = static_cast<double>(tokenize_ns_.load(std::memory_order_relaxed)) * 1e-6;
    st.tokenized_requests = tokenized_.load(std::memory_order_relaxed);
    st.profiling = profiling_;
    if (compiled_) {
      const auto& cs = static_cast<const dynacore::ir::RecordingDevice&>(*compiled_).stats();
      st.compiled = true;
      st.compiled_segments = cs.segments;
      st.compiled_cache_hits = cs.cache_hits;
      st.compiled_cache_misses = cs.cache_misses;
      st.compiled_fallbacks = cs.fallbacks;
      st.compiled_device_calls = cs.planned_steps;
      st.compiled_recorded_calls = cs.ops;
    }
    if (profiling_) {
      st.forward = transformer_->profile();
      st.pool = pool_->stats();
    }
    std::lock_guard<std::mutex> lock(stats_mu_);
    stats_ = st;
  }
}

int64_t Engine::kv_capacity_tokens() const {
  return static_cast<int64_t>(kv_->num_blocks()) * kv_->geometry().block_size;
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(stats_mu_);
  return stats_;
}

Result<std::shared_ptr<RequestStream>> Engine::submit(std::vector<TokenId> tokens, const GenerateParams& params) {
  if (params.max_tokens <= 0) return InvalidArgument("max_tokens must be > 0");
  auto stream = std::make_shared<RequestStream>();
  auto state = std::make_shared<RequestState>(*model_->tokenizer, params.stop);
  state->stream = stream;
  state->prompt_tokens = static_cast<int32_t>(tokens.size());
  std::weak_ptr<Core> weak_core = core_;

  Request r;
  r.prompt = std::move(tokens);
  r.stop = StopParams{params.max_tokens, params.stop_at_eog};
  r.priority = params.priority;
  r.sampling = params.sampling;
  r.timeout_ms = params.timeout_ms;
  // Runs on the scheduler thread.
  r.on_event = [state, weak_core](const RequestEvent& ev) {
    if (state->ended) return;
    if (!ev.finished) {
      ++state->completion_tokens;
      TextStreamer::Delta d = state->streamer.push(ev.token);
      if (!d.text.empty() || d.stopped) {
        StreamEvent out;
        out.text = std::move(d.text);
        if (d.stopped) {
          out.done = true;
          out.finish = StreamFinish::kStop;
          out.prompt_tokens = state->prompt_tokens;
          out.completion_tokens = state->completion_tokens;
          state->ended = true;
          if (auto core = weak_core.lock()) {
            std::lock_guard<std::mutex> lock(core->mu);
            if (core->sched) core->sched->cancel(ev.request_id);  // stop generating
          }
        }
        state->stream->push(std::move(out));
      }
      return;
    }
    StreamEvent out;
    out.text = state->streamer.finish();
    out.done = true;
    out.finish = map_finish(ev);
    out.error = ev.error;
    out.prompt_tokens = state->prompt_tokens;
    out.completion_tokens = state->completion_tokens;
    state->ended = true;
    state->stream->push(std::move(out));
  };

  const uint64_t id = scheduler_->submit(std::move(r));
  stream->set_id(id);
  stream->set_cancel_fn([weak_core, id] {
    if (auto core = weak_core.lock()) {
      std::lock_guard<std::mutex> lock(core->mu);
      if (core->sched) core->sched->cancel(id);  // the request is active, so the loop is awake
    }
  });
  {
    std::lock_guard<std::mutex> lock(wake_mu_);
    live_.erase(std::remove_if(live_.begin(), live_.end(), [](const auto& w) { return w.expired(); }), live_.end());
    live_.push_back(stream);
  }
  wake_cv_.notify_one();
  return stream;
}

Result<std::shared_ptr<RequestStream>> Engine::generate_text(std::string_view prompt, bool parse_special,
                                                             const GenerateParams& params) {
  std::vector<TokenId> tokens;
  const int64_t t0 = now_ns();
  ENGINE_RETURN_IF_ERROR(model_->tokenizer->encode(prompt, /*add_special=*/true, parse_special, tokens));
  tokenize_ns_.fetch_add(now_ns() - t0, std::memory_order_relaxed);
  tokenized_.fetch_add(1, std::memory_order_relaxed);
  return submit(std::move(tokens), params);
}

Result<std::shared_ptr<RequestStream>> Engine::generate_chat(std::span<const ChatMessage> messages,
                                                             const GenerateParams& params) {
  if (!model_->chat_template) return Unsupported("model has no recognized chat template");
  ENGINE_ASSIGN_OR_RETURN(std::string text, model_->chat_template->apply(messages, /*add_generation_prompt=*/true));
  return generate_text(text, /*parse_special=*/true, params);
}

}  // namespace dynalm

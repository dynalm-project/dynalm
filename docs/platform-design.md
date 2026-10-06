# DynaLM + DynaCore: Platform Design

Status: **design, not yet implemented** (DD-068). Written 2026-10-06 against commit `f0d870a`.

This document covers the 14 deliverables of the DynaLM/DynaCore platform spec. It starts from
the engine as it is today, not from a blank page. DynaLM already loads GGUF and SafeTensors,
batches continuously, uses a paged KV cache with a radix prefix cache, runs an OpenAI server,
and selects SIMD kernels at runtime. All of that was built and measured over 29 phases and
P1–P16. The spec's Phases 1–3 are therefore done. What it adds is a **hard boundary**: a
reusable low-level runtime (DynaCore) that the LLM platform (DynaLM) sits on, enforced by the
build instead of by convention.

The work is mostly extraction, not new code. Section 2 lists the four places where today's code
crosses the future boundary. Fixing those four is the first milestone (R0).

---

## 1. Product architecture

```
                              USER
                                │
         ┌──────────────────────┼───────────────────────┐
         │ dynalm CLI           │ OpenAI HTTP API       │ (later) C/C++ SDK
         └──────────────────────┼───────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────────┐
│ DynaLM  (libdynalm)                                                │
│                                                                    │
│  server/api ─► Engine facade ─► RequestManager ─► Scheduler        │
│                    │                                  │            │
│              Tokenizer/ChatTemplate         KV manager + PrefixCache
│                    │                                  │            │
│              Sampler/Speculative ◄── Generator ◄── StepPlanner     │
│                                          │                         │
│  Loaders (GGUF, SafeTensors/HF) ─► TensorRegistry + ModelConfig    │
│                                          │                         │
│                              Architecture adapters ─► Transformer  │
└──────────────────────────────────────────┬─────────────────────────┘
                                           │  dynacore:: public API only
┌──────────────────────────────────────────▼─────────────────────────┐
│ DynaCore  (libdynacore)                                            │
│                                                                    │
│  Device (op interface) ──► Kernel dispatch (chosen once at create) │
│     │                              │                               │
│  Tensor/DType/Layout      Quant formats (block layouts, dequant)   │
│     │                              │                               │
│  Memory (aligned, mmap-backed Storage, device alloc, pools)        │
│     │                                                              │
│  Exec (ThreadPool, QoS, parallel_for)   Hardware (cpuid, caches,   │
│                                          hybrid cores, GPU probe)  │
│                                                                    │
│  Backends: cpu/{generic, avx2, avx_vnni, avx512, amx, neon}  cuda/ │
└────────────────────────────────────────────────────────────────────┘
```

Responsibilities in one line each:

- **DynaCore** executes tensor operations as fast as the hardware allows. It knows tensors,
  dtypes, quant blocks, paged-KV *layout*, threads and ISAs. It does not know models, files,
  tokens, requests or HTTP.
- **DynaLM** turns requests into tensor operations. It owns every policy: which sequences run,
  how KV blocks are allocated and shared, which tokens are sampled, and what the user sees.

The split keeps one rule from the existing engine (DD-045, DD-051). A *planner* above the
boundary decides policy per step. The *device* below the boundary executes it and never
re-derives it.

---

## 2. Where the code crosses the boundary today

Measured with an include scan of `src/` (all other modules are already clean):

| # | Violation | Why it is wrong | Fix in R0 |
|---|---|---|---|
| V1 | `backends/backend.h` includes `model_ir/model_config.h` | The op interface takes `RopeConfig` and `Activation`, which are defined next to the model config | Move `RopeConfig`, `RopeStyle`, `RopeScaling`, `Activation`, `NormType` to `dynacore/ops.h`. `ModelConfig` includes them from there. |
| V2 | `backends/backend.h` includes `kv_cache/kv_layout.h` | Attention kernels need the physical paged-KV layout | Move `KvGeometry` and `KvLayerView` to `dynacore/paged_kv.h`. The layout is a kernel concern. Allocation, block tables, refcounts and eviction stay in DynaLM. |
| V3 | `backends/backend_registry.h` and `cpu_backend.h` include `runtime/thread_pool.h` | The thread pool is an execution primitive, not a runtime policy | Move `ThreadPool` (and `thread_qos`) to `dynacore/exec/`. |
| V4 | `execution/batch_planner.h` mixes a model view (`SeqBatch`, `ModelConfig`) with kernel heuristics (split-K rule, int8 crossover) | Kernel selection belongs to DynaCore; batch composition belongs to DynaLM | Split it. DynaCore gets `KernelPlan plan_kernels(const StepShape&, const HardwareProfile&)`. `StepShape` holds only numbers: rows, decode rows, max context, KV heads, window. DynaLM's `StepPlanner` builds `StepShape` from `SeqBatch`. |

Two smaller moves complete the split:

- `quant/gptq_awq.*` repacks AutoGPTQ/AutoAWQ checkpoint layouts. Those are file-format
  conventions, so this code moves to `dynalm/loader/`. `quant_formats` and `dequant` stay in
  DynaCore.
- `common/status.h` (`Status`/`Result`) and `common/platform.h` become `dynacore/base/`. Both
  libraries use them; DynaCore is the lower one, so it owns them.

`dtype.h` already says "the enum is engine-owned: file formats map to it". The GGML block types
are layouts, not GGUF knowledge, so they stay in DynaCore unchanged.

---

## 3. Monorepo structure

The spec suggests `core/dynacore/` and `runtime/dynalm/`. This design flattens that to two
top-level library directories. Each has a public `include/<name>/` directory. The include path
is the enforcement mechanism: the `dynacore` target never receives `dynalm/include` on its
include path, so `#include <dynalm/...>` inside DynaCore is a compile error, not a review comment.

```
dynalm/                          (repo root, product name)
├── CMakeLists.txt               top-level: options, add_subdirectory(dynacore dynalm apps ...)
├── CMakePresets.json
├── cmake/                       compiler flags, ISA helper, version, packaging
│
├── dynacore/                    ── libdynacore: low-level inference runtime
│   ├── include/dynacore/        PUBLIC API: the only headers DynaLM may include
│   │   ├── base/                status.h, platform.h, function_ref.h, log.h (sink hook)
│   │   ├── hw.h                 HardwareInfo, Isa, detect()
│   │   ├── dtype.h  tensor.h    DType, Shape, Layout, TensorView, Tensor
│   │   ├── memory.h             Storage, aligned alloc, MappedRegion, Arena
│   │   ├── quant.h              block formats, quantize/dequantize rows
│   │   ├── ops.h                RopeConfig, Activation, NormType, AttentionParams
│   │   ├── paged_kv.h           KvGeometry, KvLayerView (layout only)
│   │   ├── kernel_plan.h        KernelPlan, StepShape, plan_kernels()
│   │   ├── device.h             Device: the op interface (today's Backend)
│   │   ├── exec.h               ThreadPool, parallel_for, QoS
│   │   └── version.h
│   ├── src/
│   │   ├── hw/                  cpu_info, isa, perf_counters, process_stats
│   │   ├── memory/  tensor/  quant/  exec/  plan/
│   │   └── backends/
│   │       ├── cpu/             cpu_device.cpp + dispatch table
│   │       │   ├── generic/     reference kernels (always built)
│   │       │   ├── avx2/        per-file -mavx2 -mfma -mf16c
│   │       │   ├── avx_vnni/    (R3, only if measured to win)
│   │       │   ├── avx512/  amx/  neon/
│   │       └── cuda/            (R3, ENABLE_CUDA)
│   ├── tests/                   unit tests, link dynacore only
│   └── benchmarks/              kernel microbenchmarks, link dynacore only
│
├── dynalm/                      ── libdynalm: LLM inference + serving platform
│   ├── include/dynalm/          engine.h, request.h, config.h, metrics.h, version.h
│   ├── src/
│   │   ├── loader/              gguf/, safetensors/, hf/, gptq_awq, model_source, mapped files
│   │   ├── model/               model_config, tensor_registry, architectures/, transformer
│   │   ├── tokenizer/  chat_template/
│   │   ├── kv/                  KV manager (pool, block tables, refcount), prefix_cache/
│   │   ├── scheduler/  planner/ request manager, continuous batching, StepPlanner
│   │   ├── sampling/            sampler, speculative
│   │   ├── engine/              Engine facade, Generator, Sequence, text_stream
│   │   ├── api/                 JSON, OpenAI schema (no HTTP)
│   │   ├── server/              cpp-httplib binding (libdynalm_server)
│   │   ├── config/  logging/  metrics/
│   │   └── bench/               loadgen, analysis (used by `dynalm benchmark`)
│   ├── tests/                   unit + integration tests
│   └── benchmarks/              end-to-end benchmarks (decode, prefix, scheduler, ...)
│
├── apps/
│   └── dynalm/                  main.cpp + one file per command → `dynalm` executable
│
├── tests/
│   ├── integration/             tiny models end to end, through the CLI and HTTP
│   ├── data/                    fixtures, goldens
│   └── boundary/                check_boundary.py (see §3.1)
├── tools/                       ab_summary.py, perf_sweep.sh, model fixture generators, docker/
├── scripts/                     install.sh, install.ps1
├── docker/                      Dockerfile.cpu, Dockerfile.cuda
├── docs/
│   ├── dynacore/                API, kernels, dispatch, quantization
│   └── dynalm/                  CLI, config, server, architecture support
├── examples/                    C++ embedding example, curl/OpenAI-client examples
└── README.md LICENSE NOTICE CHANGELOG.md
```

The spec's `tools/dynalm-cli/` becomes `apps/dynalm/`. A shipped binary is a product, not a
tool. `tools/` keeps developer scripts only.

### 3.1 Boundary enforcement

Three checks, each cheap:

1. **Include paths.** `dynacore` has only `dynacore/include` and `dynacore/src` on its path.
2. **Link graph.** `dynacore` links nothing from the repo. CMake rejects a cycle if anyone adds one.
3. **Banned-name scan.** `tests/boundary/check_boundary.py` runs in CI. It fails if any file
   under `dynacore/` mentions `gguf`, `safetensors`, `tokeniz`, `chat`, `http`, `openai`,
   `request`, `llama`, `qwen`, `gemma` or `phi` outside comments. This catches knowledge leaks
   that do not need an include, such as a model-specific branch hidden in a kernel.

---

## 4. Module dependency graph

Arrows mean "may include / link". Anything not shown is forbidden.

```
apps/dynalm ──► dynalm_server ──► libdynalm ──► libdynacore ──► OS, libc, (CUDA runtime)
                     │                │
                cpp-httplib     (no third-party deps)
```

Inside DynaCore (lower rows never include upper rows):

```
device (backends/*)          ← kernels + dispatch
  ├── plan                   ← plan_kernels(StepShape, HardwareProfile)
  ├── ops, paged_kv          ← parameter types only
  ├── quant
  ├── tensor ── dtype
  ├── memory
  ├── exec (ThreadPool)
  ├── hw
  └── base (Status, platform, log sink)
```

Inside DynaLM:

```
server ─► api ─► engine ─► scheduler ─► kv ─► (dynacore)
                   │          │
                   │          └► planner ─► model ─► (dynacore)
                   ├► sampling
                   ├► tokenizer, chat_template
                   └► loader ─► model (TensorRegistry, ModelConfig)
config, logging, metrics: leaf utilities, usable by all of DynaLM
```

The existing per-layer rules (from `docs/architecture.md`) carry over unchanged:

| Module | Must not know about |
|---|---|
| scheduler | model architecture, file format, ISA |
| kv | file format, model family, ISA |
| model / adapters | GGUF or SafeTensors (sees only TensorRegistry + ModelConfig), SIMD |
| api / server | kernels, KV layout, scheduler internals |
| **all of dynacore** | everything in DynaLM |

---

## 5. DynaCore API

### 5.1 Design choice: a `Device` object, not global free functions

The spec sketches `Tensor dynacore::gemm(const Tensor& A, const Tensor& B)`. This design does
not use that form, for two measured reasons:

- **No allocation in the hot path.** A function that returns a new `Tensor` allocates per call.
  A decode step on Qwen3-4B issues about 250 ops. The engine writes into activation buffers
  allocated once at model build. Ops take an output view.
- **No hidden global state.** One process can hold a CPU device and, later, a CUDA device, plus
  a memory-guarded test device (DD-045). Free functions on a global context make that and the
  guarded test impossible.

The ops are already the right granularity: coarse (whole matmul, whole batched attention), so
one virtual call per op costs nothing measurable. The `Device` interface is today's `Backend`
with the V1–V3 types moved. Its contract is already proven by every model and runtime path.

### 5.2 Headers

```cpp
// dynacore/hw.h
namespace dynacore {
enum class Isa : uint8_t { kGeneric, kNeon, kAvx2, kAvxVnni, kAvx512, kAmx };
struct HardwareInfo {
  std::string cpu_brand, vendor;
  int32_t physical_cores, logical_cores, performance_cores, efficiency_cores;
  int64_t l1d_bytes, l2_bytes, l3_bytes, total_ram, available_ram;
  struct { bool avx2, fma, f16c, avx_vnni, avx512f, avx512_vnni, avx512_bf16, amx_tile, amx_int8, neon, dotprod; } cpu;
  std::vector<GpuInfo> gpus;   // empty unless a GPU backend is compiled in
  Isa best_isa;                // best tier that is both compiled AND supported
};
const HardwareInfo& hardware();  // detected once (cpuid + XCR0 OS-state checks), then cached
}
```

Every feature check in the codebase goes through `hardware()`. Kernels never call `cpuid`.
Dispatch reads `best_isa` once when a device is created.

```cpp
// dynacore/device.h
namespace dynacore {
enum class DeviceKind : uint8_t { kCpu, kCuda, kHip, kMetal, kVulkan };

struct DeviceOptions {
  int32_t threads = 0;                // 0 = auto (P-cores first on hybrid CPUs)
  std::optional<Isa> force_isa;       // tests and A/B runs only
};

class Device {
 public:
  virtual ~Device() = default;
  virtual std::string_view name() const = 0;          // "cpu/avx2", "cuda/sm_89"
  virtual DeviceKind kind() const = 0;

  // memory
  virtual Result<std::shared_ptr<Storage>> allocate(size_t bytes) = 0;
  virtual Result<Tensor> upload(const Tensor& host) = 0;        // CPU: zero-copy
  virtual void download(const TensorView& src, std::span<float> dst) = 0;
  virtual void copy(void* dst, const void* src, size_t bytes) = 0;
  virtual void synchronize() = 0;
  virtual bool host_accessible() const = 0;
  virtual bool supports_weight_type(DType t) const = 0;

  // per-step decisions (from plan_kernels); never re-derived inside kernels
  virtual void set_kernel_plan(const KernelPlan& plan) = 0;

  // ops: outputs are caller-owned views; nothing here allocates
  virtual void embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) = 0;
  virtual void matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) = 0;
  virtual void matmul_many(std::span<const MatmulJob> jobs) = 0;
  virtual void rms_norm(const TensorView& x, const TensorView& w, float eps, const TensorView& y) = 0;
  virtual void layer_norm(const TensorView& x, const TensorView& w, const TensorView* b, float eps, const TensorView& y) = 0;
  virtual void rope(const TensorView& x, int32_t heads, int32_t head_dim, std::span<const int32_t> pos,
                    const RopeConfig& rope, const float* freq_factors) = 0;
  virtual void kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> pos,
                        std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) = 0;
  virtual void attention(const AttentionParams& p) = 0;    // paged, GQA-grouped, causal/window, softcap
  virtual void act_mul(Activation a, const TensorView& gate, const TensorView& up, const TensorView& out) = 0;
  virtual void activation(Activation a, const TensorView& x, const TensorView& out) = 0;
  virtual void add(const TensorView& a, const TensorView& b, const TensorView& y) = 0;
  virtual void scale(const TensorView& x, float s) = 0;
  virtual void softcap(const TensorView& x, float cap) = 0;
  virtual void fill(const TensorView& x, float v) = 0;
  virtual void gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) = 0;
  virtual void scatter_add_rows(const TensorView& src, std::span<const int32_t> rows,
                                std::span<const float> w, const TensorView& dst) = 0;
};

Result<std::unique_ptr<Device>> create_device(DeviceKind kind, const DeviceOptions& opts = {});
std::vector<DeviceKind> compiled_devices();
}
```

`matmul` is the GEMM/GEMV entry point. The device picks GEMV-style fused dequantize-dot for few
rows and the blocked fp32 GEMM for many rows. The `KernelPlan` sets the crossover. A caller never
names a kernel.

```cpp
// dynacore/kernel_plan.h — per-step kernel selection, numbers in, decisions out
struct StepShape {
  int32_t rows, decode_rows, prefill_rows, sequences;
  int32_t max_context;        // longest attention span of any row
  int64_t attention_work;     // sum of spans
  int32_t num_kv_heads, sliding_window;
  bool has_window_layers;
};
struct HardwareProfile { int32_t threads, physical_cores, performance_cores, efficiency_cores;
                         int64_t l2_bytes, llc_bytes; Isa isa; };
KernelPlan plan_kernels(const StepShape& s, const HardwareProfile& hw, const KernelPlan& base);
```

`KernelPlan` keeps its current fields (`expand_min_rows`, `gemm_k_block`,
`int8_decode_max_rows`, attention strategies, `grouped_attention`, ...). Each field is a measured
decision with a DD number. The `DYNALM_*` tuning overrides keep working. DynaCore reads them
through a `KernelPlan::from_env(prefix)` hook, so its own code never hardcodes the product name.

```cpp
// dynacore/tensor.h — inference-sized, no autograd, no broadcasting engine
enum class DType : uint8_t { kF32, kF16, kBF16, kI8, kI32, kQ4_0, kQ4_1, kQ5_0, kQ5_1, kQ8_0,
                             kQ8_1, kQ2_K, kQ3_K, kQ4_K, kQ5_K, kQ6_K, kQ8_K /* ... */ };
struct Device { DeviceKind type; int32_t index; };      // a location, not the op interface
struct TensorLayout { Shape shape; Strides strides; DType dtype; };  // rank ≤ 4
class TensorView { void* data; TensorLayout layout; DeviceLoc device; /* row(), slice(), ... */ };
class Tensor     { std::shared_ptr<Storage> storage; TensorView view; };  // owns or maps memory
```

A rename is needed: the location struct is `DeviceLoc` so `Device` can name the op interface.

```cpp
// dynacore/quant.h — pluggable by table, not by inheritance
struct QuantTraits { DType type; int32_t block_elems; int32_t block_bytes; bool has_int8_dot;
                     void (*dequant_row)(const void*, float*, int64_t);
                     void (*quant_row)(const float*, void*, int64_t); };   // nullptr: load-only
const QuantTraits& quant_traits(DType t);
```

A new format is one `QuantTraits` entry plus its kernels in each backend that accelerates it.
Backends without a kernel for the format fall back to `dequant_row` + fp32. Model code sees
only `DType`.

```cpp
// dynacore/exec.h
class ThreadPool { public: explicit ThreadPool(int n, QosPolicy p = QosPolicy::kPerfCoresFirst);
  void parallel_for(int64_t n, FunctionRef<void(int64_t begin, int64_t end, int worker)> fn, int64_t chunks = 0);
  int size() const; ThreadPoolStats stats() const; };
```

```cpp
// dynacore/base/log.h — DynaCore never prints; DynaLM installs the sink
using LogSink = void (*)(LogLevel, std::string_view);
void set_log_sink(LogSink);
```

### 5.3 Compatibility rules

- DynaCore and DynaLM ship together and link statically, so there is **no ABI promise**.
- **Source compatibility** of `dynacore/include/` holds within a DynaCore minor version. A
  breaking change bumps the minor version (0.x) or major version (≥1.0) and is listed in
  `docs/dynacore/CHANGELOG.md`.
- `DYNACORE_VERSION_{MAJOR,MINOR,PATCH}` live in `dynacore/version.h`. DynaLM checks a minimum
  with `static_assert`. `dynalm doctor` prints both versions.
- New ops are added as virtual functions with a default implementation built from existing ops.
  For example, a fused `rms_norm_matmul` defaults to `rms_norm` then `matmul`. Old devices keep
  compiling, and new devices override what they accelerate.

---

## 6. DynaLM API

```cpp
// dynalm/engine.h — the embedding surface (CLI, server and SDK all use this)
namespace dynalm {
struct EngineOptions {
  std::string model;                 // path, HF dir, or registry name "qwen3:4b"
  dynacore::DeviceKind device = dynacore::DeviceKind::kCpu;   // "auto" resolves at load
  int32_t threads = 0;               // 0 = auto
  int32_t context_length = 0;        // 0 = model default, clamped by memory
  SchedulerOptions scheduler;        // max_concurrent, max_batch_tokens, chunk size, policy
  KvOptions kv;                      // dtype, block_size, memory budget, prefix cache kind
  SpeculativeOptions speculative;    // adaptive by default (DD-062)
};

class Engine {
 public:
  static Result<std::unique_ptr<Engine>> create(const EngineOptions& opts);
  const ModelInfo& model() const;
  // Thread-safe. The returned stream yields tokens as the scheduler produces them.
  Result<RequestHandle> submit(GenerationRequest req);
  void cancel(RequestId id);
  EngineStats stats() const;          // queue depth, active, KV usage, tok/s, TTFT/ITL histograms
  void shutdown(std::chrono::milliseconds drain);   // stop admitting, finish or cancel, join
};

struct GenerationRequest {
  std::variant<std::string, std::vector<ChatMessage>, std::vector<int32_t>> input;
  SamplingParams sampling;            // temperature, top_k/p, min_p, penalties, seed, logit_bias, grammar (later)
  int32_t max_tokens; std::vector<std::string> stop;
  int32_t priority = 0; std::optional<std::chrono::milliseconds> deadline;
};

class RequestHandle {                 // pull-based; the server turns it into SSE
 public:
  std::optional<StreamEvent> next(std::chrono::milliseconds wait);   // text delta / finish / error
  RequestId id() const;
};
}
```

Internal interfaces (not installed headers, but stable enough to test against):

| Interface | Owns | Talks to DynaCore through |
|---|---|---|
| `ModelLoader` → `LoadedModel{TensorRegistry, ModelConfig, TokenizerData}` | format parsing, mmap, repacking | `dynacore::Tensor`, `MappedRegion`, `quant_traits` |
| `ModelArchitecture` (Llama, Qwen, Gemma, Phi, DeepSeek/MoE) | names → roles, config flags | none (pure metadata) |
| `Transformer::forward(SeqBatch[], logits)` | layer loop, MoE routing | `Device` ops |
| `KvManager` | block pool, block tables, refcounts, COW, eviction | `Device::allocate`, `KvLayerView` |
| `PrefixCache` (radix default, hash) | prefix sharing, LRU | none |
| `Scheduler` | admission, continuous batching, chunked prefill, priority, preemption | none |
| `StepPlanner` | `SeqBatch[]` → `StepShape` | `plan_kernels()` |
| `Sampler`, `Speculative` | token choice, draft/verify | none (host math on logits) |

---

## 7. CLI specification

One binary, `dynalm`. Commands that exist today are marked ✅. Renames keep the old name as an
alias for one minor release.

| Command | Purpose | Status |
|---|---|---|
| `dynalm run <model> [-p PROMPT]` | One prompt, or interactive chat without `-p` | ✅ (`chat` alias) |
| `dynalm serve <model> [--host --port --api-key ...]` | OpenAI-compatible server | ✅ |
| `dynalm pull <model>` | Download GGUF/HF into the model store | ✅ (URL/HF repo); add `name:tag` registry |
| `dynalm models [ls\|rm NAME]` | List or remove local models | ✅ as `list`/`rm`; rename |
| `dynalm inspect <model>` | Metadata, tensors, quant mix, memory estimate | ✅ |
| `dynalm benchmark <model>` | Load, prefill, decode, aggregate tok/s, TTFT, ITL p50/95/99, RSS, concurrency sweep | ✅ |
| `dynalm doctor` | System report for bug reports | ✅ as `info`; extend and rename |
| `dynalm config [show\|path\|init]` | Effective config with source of each value | new |
| `dynalm version` | DynaLM + DynaCore versions, build flags, compiled ISAs | extend |

Model references resolve in this order: existing file path, then local store name, then
registry `name:tag`, then HF repo id. `qwen3:4b` maps through a small curated table
(`dynalm/src/loader/registry.json`) to an HF repo and a default quant (`Q4_K_M`). The engine
has no network service of its own.

`dynalm doctor` output (target):

```
DynaLM System Report
────────────────────────────
DynaLM    0.6.0  (commit f0d870a, MSVC 19.44, LTO off)
DynaCore  0.6.0  (compiled ISAs: generic avx2)

CPU       12th Gen Intel Core i7-1255U   2 P-cores + 8 E-cores, 12 threads
Caches    L1d 48 KiB  L2 1.25 MiB  L3 12 MiB
RAM       15.7 GiB total, 1.9 GiB available        ⚠ low: models above ~1.5 GiB will page
ISA       AVX2 ✓  FMA ✓  F16C ✓  AVX-VNNI ✓  AVX-512 ✗  AMX ✗
GPU       none detected (CUDA backend not compiled)
Device    cpu/avx2   threads 10 (P-cores first)
Clock     effective 2.1 GHz under load (base 1.7)  power plan: Balanced
Models    3 in C:\Users\...\.dynalm\models (4.1 GiB)
Status    Ready ✓
```

`doctor` also prints *actionable* warnings: low available RAM, a power plan that throttles,
AVX-VNNI present but not compiled, Smart App Control blocking unsigned binaries. `--json`
emits the same data for issue templates.

---

## 8. Configuration specification

Precedence (unchanged from today): built-in defaults < config file < `DYNALM_*` environment <
command line.

The spec asks for YAML. Adding a YAML library would be the project's first runtime dependency
besides cpp-httplib. Instead, DynaLM accepts a **strict YAML subset**: two levels of maps,
scalars, `#` comments. A hand-written parser of about 200 lines handles it and rejects anything
else with a line number. Each `section.key` maps to an existing long option. The current flat
`key = value` format stays supported.

```yaml
# ~/.dynalm/config.yaml  (or --config PATH, or DYNALM_CONFIG)
model:
  path: qwen3:4b                 # file, store name, name:tag, or HF repo

runtime:
  device: auto                   # auto | cpu | cuda
  threads: auto                  # auto = P-cores (+ E-cores if they measured faster)
  context_length: auto           # model max, clamped so KV fits kv_cache.memory
  mlock: false

scheduler:
  continuous_batching: true
  max_concurrent_requests: 32
  max_batch_tokens: 512          # per step; prefill is chunked to fit
  prefill_chunk: 256
  policy: fcfs                   # fcfs | priority
  queue_timeout_ms: 30000

kv_cache:
  dtype: f16                     # f32 | f16 | (q8_0, R2)
  block_size: 16
  memory: auto                   # bytes or "auto" = fraction of free RAM after weights
  prefix_cache: radix            # radix | hash | off

sampling:
  temperature: 0.7
  top_p: 0.95
  max_tokens: 1024

speculative:
  mode: adaptive                 # off | adaptive | always

server:
  host: 127.0.0.1                # 0.0.0.0 only inside containers (DYNALM_HOST)
  port: 8000
  api_key_file: ""               # empty = no auth
  max_request_bytes: 1048576
  max_prompt_tokens: auto
  request_timeout_ms: 600000

logging:
  level: info                    # error | warn | info | debug
  format: text                   # text | json
```

Every `auto` resolves at startup and is printed once at `info` level. `dynalm config show`
prints the same resolved table with the source of each value.

---

## 9. Installation architecture

One product, one install. DynaCore is a static library inside the `dynalm` binary, so there is
nothing separate to install and no library version to mismatch.

| Platform | Channel | Contents |
|---|---|---|
| Linux x86-64/arm64 | `curl -fsSL https://dynalm.ai/install.sh \| sh` (`scripts/install.sh` exists) | one static-runtime `dynalm` binary to `~/.local/bin`, SHA256 verified against the release manifest |
| Windows x64 | `install.ps1` (exists) now; signed MSI later | `dynalm.exe` with static CRT, UTF-8 manifest, PATH entry |
| macOS arm64 | `brew install dynalm` (tap formula built from the release tarball) | universal NEON build |
| Containers | `dynalm/dynalm:cpu`, `dynalm/dynalm:cuda` | binary + `/models` volume, non-root user, `DYNALM_HOST=0.0.0.0` |

Decisions:

- **One binary per platform, all ISAs inside.** Runtime dispatch picks the tier, so there are
  no per-CPU downloads. CUDA is the only split. `dynalm-cuda` is a separate archive and image
  because the CUDA runtime is about 500 MB and most users run CPU.
- **Windows signing is a release requirement.** Smart App Control blocks unsigned freshly
  built executables on this very development machine. An unsigned installer would hit end
  users the same way.
- **Model store** at `~/.dynalm/models` (`%USERPROFILE%\.dynalm\models` on Windows),
  overridable with `DYNALM_MODELS`. Images never bake models in; users mount them.

Docker images:

```
docker/Dockerfile.cpu   ubuntu:24.04 build → distroless/cc runtime, multi-arch (amd64, arm64)
docker/Dockerfile.cuda  nvidia/cuda:12.x-devel build → nvidia/cuda:12.x-runtime, amd64
ENTRYPOINT ["dynalm"]   CMD ["serve", "/models/model.gguf"]
HEALTHCHECK  GET /health
```

---

## 10. Build architecture

```
Targets                Output                  Links
dynacore               libdynacore.a/.lib      (none from repo)
dynalm_runtime         libdynalm.a/.lib        dynacore
dynalm_server          libdynalm_server.a      dynalm_runtime, cpp-httplib (pinned, SHA256-checked)
dynalm                 dynalm / dynalm.exe     dynalm_server (or dynalm_runtime if server off)
dynacore_tests         ctest label "core"      dynacore
dynalm_tests           ctest label "lm"        dynalm_runtime (+ server)
*_bench                bench executables       per layer
```

The library target is `dynalm_runtime` with `OUTPUT_NAME dynalm` because CMake target names must
not collide with the `dynalm` executable.

Options. The old `ENABLE_*` names stay as deprecated aliases for one release.

| Option | Default | Effect |
|---|---|---|
| `DYNALM_BUILD_TESTS` | ON | unit + integration tests |
| `DYNALM_BUILD_BENCHMARKS` | ON | benchmark executables |
| `DYNALM_BUILD_SERVER` | ON | HTTP server (needs cpp-httplib) |
| `DYNALM_CORE_ONLY` | OFF | configure only `dynacore` + its tests (proves independence) |
| `DYNALM_ENABLE_AVX2` / `_AVX_VNNI` / `_AVX512` / `_AMX` / `_NEON` | per arch | compile that kernel tier (per-file ISA flags, DD-003) |
| `DYNALM_ENABLE_CUDA` | OFF | CUDA device (R3) |
| `DYNALM_LTO` | OFF | IPO for release (DD-063) |
| `DYNALM_STATIC_RUNTIME` | OFF (ON in release CI) | static CRT / libstdc++ |
| `DYNALM_SANITIZE` | none | `address`, `undefined`, `thread` |

`DYNALM_CORE_ONLY=ON` is a CI job. It builds and tests DynaCore without the `dynalm/`
directory present on the include path. If DynaCore ever needs DynaLM, this job fails.

Presets keep the current set (`msvc-release`, `linux-release`, `asan`, `tsan`, ...) and add
`core-only`.

---

## 11. Runtime lifecycle

```
startup        parse CLI → merge config (file < env < args) → install log sink → print resolved config
hardware       dynacore::hardware(): cpuid/XCR0, caches, hybrid topology, RAM, GPU probe (once, cached)
device         create_device(kind, threads) → dispatch table bound to best compiled ISA; ThreadPool pinned P-cores first
model load     resolve reference → loader (GGUF mmap / SafeTensors) → TensorRegistry + ModelConfig
               → architecture adapter → device.upload (CPU: zero-copy) → pre-fault weight pages (DD-064)
memory         activation buffers sized for max_batch_tokens (allocated once) → KV budget = min(config,
               free RAM − weights − headroom) → KvManager block pool → clamp context_length to fit
scheduler      Engine starts one scheduler thread; HTTP worker threads only enqueue and read streams
admission      tokenize + chat template on the HTTP thread → limits checked (prompt tokens, max_tokens,
               queue depth) → 429/503 early, before any compute → prefix-cache lookup reserves shared blocks
step loop      scheduler picks sequences (decode first, then chunked prefill up to max_batch_tokens)
               → StepPlanner: SeqBatch[] → StepShape → plan_kernels → device.set_kernel_plan
               → Transformer::forward (one device op stream for the whole batch)
               → sampler / speculative verify → append tokens → text_stream detokenizes incrementally
streaming      RequestHandle queue → SSE chunks; stop strings and think-filter applied before emit
completion     release KV blocks (refcount; prefix blocks stay cached for reuse) → record TTFT/ITL/tokens
shutdown       SIGINT/SIGTERM → stop admission (503) → drain for `drain_ms` → cancel rest → join scheduler
               → join ThreadPool → unmap weights
```

The scheduler thread is the only thread that calls the device. That keeps the device single-
threaded at the API level, and it gets parallelism from the pool inside each op. A GPU device
uses the same rule with one stream.

---

## 12. Performance architecture

The primary metric is **aggregate output tok/s at a target concurrency**. TTFT, ITL p95 and peak
RSS are guardrails, not goals: a change may not move any guardrail by more than its noise band.

The loop (unchanged, now written down as policy): profile → hypothesis → prototype behind a
`KernelPlan` field or env override → correctness test → A/B → end-to-end → keep or revert with a
DD entry either way. DD-065 and DD-066 are examples of this loop ending in "neutral, kept as
reference path" or "no kernel needed". A negative result is recorded like a positive one.

Tooling per layer:

| Layer | Tool | Measures |
|---|---|---|
| DynaCore kernels | `dynacore/benchmarks/bench_matmul`, `bench_attn`, `bench_quant` | GFLOP/s, GB/s vs. roofline per ISA tier |
| DynaCore dispatch | `force_isa` option | same op, every compiled tier, same run |
| DynaLM step | `bench_batch_decode`, `bench_small_m`, `bench_prefix` | step ms by phase, rows |
| End to end | `dynalm benchmark --concurrency 1,4,16` | aggregate tok/s, TTFT, ITL p50/95/99, RSS |
| A/B | `tools/ab_summary.py` (interleaved runs, median + spread, effective-clock check from DD-066) | keep/revert verdict |
| Comparison | `tools/bench_ollama.sh` | same model, same quant, same prompt set vs. Ollama/llama.cpp |

Rules that come from past mistakes on this machine:

- Interleave A and B runs. The i7-1255U's clock varies with temperature. Report the effective
  clock next to every number (DD-066).
- Never call a kernel win an end-to-end win. DD-066 made attention 4–22% faster and end-to-end
  decode neutral.
- Microbenchmarks pick candidates. Only `dynalm benchmark` decides.

Regression gate: CI stores a baseline per runner (`results/baselines/`). A PR that drops
aggregate tok/s by more than 3% on the tiny-model suite fails. `tools/compare_baselines.sh`
exists for this. Real-model numbers are tracked on a schedule, not per PR, because CI runners
are noisy.

---

## 13. Testing architecture

| Layer | Suite | Examples (existing → new location) |
|---|---|---|
| DynaCore unit | `dynacore/tests` | `test_tensor`, `test_dtype`, `test_quant`, `test_cpu_info`, `test_kernels` (every op × every compiled ISA vs. generic reference, ULP/rel-error bounds), `test_device_backend` (memory-guarded device) |
| DynaCore dispatch | `dynacore/tests/test_dispatch` (new) | `force_isa` each tier; `best_isa` never picks an uncompiled or unsupported tier |
| DynaLM unit | `dynalm/tests` | `test_gguf`, `test_safetensors`, `test_gptq_awq`, `test_model_ir`, `test_architectures`, `test_tokenizer` (goldens from HF tokenizers), `test_scheduler`, `test_kv_cache`, `test_prefix_cache`, `test_sampling`, `test_speculative`, `test_config`, `test_json` |
| Correctness | `test_models`, `test_int8_decode` | tiny generated models vs. `tools/ref_model.py` logits; perplexity/KL contract for int8 (≤ +1% PPL, ≤ 0.0025 nats) |
| Integration | `tests/integration` | `test_runtime`, `test_batching`, `test_streaming`, `test_server` (real HTTP, SSE, cancel mid-stream, 429/503), CLI smoke |
| Boundary | `tests/boundary` | `check_boundary.py`, `core-only` build |
| Robustness | `test_hardening` + ASan/UBSan/TSan presets | truncated/hostile GGUF, oversize requests, concurrent cancel |
| Regression | `test_compat` | old config keys, CLI aliases, OpenAI response schema snapshots |
| Performance | benchmarks only, never in ctest | §12 |

Test models are generated (`tools/make_tiny_models.py`), so CI needs no downloads. Real-model
checks (Qwen2.5-0.5B, SmolLM2-135M) run on demand via `tools/fetch_models.sh`.

---

## 14. Roadmap

The spec's Phases 1–3 are complete in the current engine. This roadmap continues from there.
Each step ends the way all phases did: builds, tests pass, benchmarks run, docs updated, DD
entry written.

```
R0 boundary ──► R1 product surface ──┐
     │                               ├──► R3 hardware ──► R4 IR ──► R5 DSL
     └────────► R2 performance ──────┘
```

**R0 — Extract DynaCore (no behavior change).**
Fix V1–V4. Move files into `dynacore/` and `dynalm/`. Split `engine_core` into
`dynacore` + `dynalm_runtime`. Rename namespaces `engine` → `dynacore` / `dynalm`, with a
temporary `namespace engine = ...` alias so the move and the rename land as separate commits.
Add `check_boundary.py` and the `core-only` preset.
*Exit:* all tests green, and an A/B on Qwen2.5-0.5B shows aggregate tok/s within noise (±2%).
This is a pure move, so any slowdown is a bug, such as lost inlining across the new library
boundary.

**R1 — Product surface.**
`doctor` (extends `info`), `models` (renames `list`/`rm`), `config show|path|init`, YAML-subset
config, `name:tag` registry for `pull`/`run`, `version` with both versions.
*Exit:* the spec's UX path `install → doctor → pull qwen3:4b → run qwen3:4b` works on Windows
and Linux.

**R2 — Performance (open items, ordered by expected tok/s gain).**
1. Q8_0 KV cache. It halves KV bandwidth at long context and doubles concurrent sequences per
   GiB. Gate: perplexity contract.
2. Packed weight layouts (interleaved panels for the GEMV path). DD-054 deferred this; reopen it
   with the batch-decode benchmark.
3. Prefill on hybrid cores: P/E work split for GEMM (`matmul_chunks_per_thread`, DD-067
   in progress).

*Exit per item:* end-to-end aggregate tok/s gain beyond noise, or a DD entry recording why not.

**R3 — Hardware backends, only when measured to win.**
- **AVX-VNNI** first. The development CPU (i7-1255U) supports it and the int8 decode path
  (DD-053) uses `maddubs`-based dot products today. VNNI's `vpdpbusd` replaces three
  instructions with one. Expected: int8 GEMV faster. Gate: end-to-end decode.
- AVX-512 / AMX: need hardware this project does not have locally. Build them only with CI or
  cloud runners that can A/B them.
- CUDA: implements `Device` (contract in `docs/gpu-backend.md`). Paged attention and quantized
  GEMV first. Gate: beats CPU on aggregate tok/s for the same model on the same box.

**R4 — DynaCore IR (see §15).** Recording device → IR → fusion passes → replay.

**R5 — DynaCore DSL.** Only after R4 has run in production for one release and its op set has
stopped changing.

---

## 15. IR and DSL: designed now, built later

The key observation: **the `Device` interface is already the IR seam.** DynaLM's forward pass is
a straight sequence of device op calls with fixed shapes per step type. So the IR can enter
without touching DynaLM:

```
Phase R4a  RecordingDevice : Device      wraps the real device, records each op call as an IR node
                                           (op kind, input/output TensorView ids, params, dtype)
Phase R4b  passes over the recorded graph  fusion (rms_norm+matmul, act_mul+matmul, rope+kv_store),
                                           buffer liveness → arena offsets, layout choice for weights
Phase R4c  ReplayDevice / CompiledStep   per (step phase, row bucket): record once, optimize once,
                                           replay with new data pointers each step
                                           (CUDA graphs fit the same slot)
```

IR node kinds map 1:1 to `Device` ops at first: `Embedding, MatMul, MatMulMany, RmsNorm,
LayerNorm, Rope, KvWrite, Attention(KvRead), ActMul, Activation, Add, Scale, Softcap, Fill,
Gather, ScatterAdd`. Fused nodes are added only when a pass produces them and a kernel exists.
The spec's higher-level nodes (`MLP`, `QuantizedWeight`, `Batch`) are not separate ops.
`QuantizedWeight` is a dtype on a tensor, `Batch` is the row dimension plus `row_seq`, and `MLP`
is a pattern that a fusion pass matches.

The DSL is a text front end for the same IR. Its job is to let a kernel author describe a fused
op, like the spec's `attention { mode = GQA ... }` block, and have a generator emit the
per-ISA variants. It targets kernel authors, not end users, and it reuses the IR's node set and
type system. Nothing in R0–R3 depends on it.

What R0–R3 must do to keep this possible:

- Every op goes through `Device`. No DynaLM code touches raw buffers on the hot path.
- Ops carry all parameters explicitly (no hidden state other than `KernelPlan`).
- Activation buffers come from one arena with stable offsets, so a liveness pass can replace
  the hand-written allocation later.

---

## 16. Production concerns

Owned by DynaLM. DynaCore has none of these.

| Concern | Mechanism (✅ exists) |
|---|---|
| Request limits | ✅ max body bytes, concurrency cap → 503 when overloaded; add max prompt tokens and per-key rate limits |
| Timeouts | ✅ per-request timeout (504), socket read/write timeouts, client disconnect → cancel and free KV |
| Auth | new: bearer API key from `api_key_file` (constant-time compare) behind an `Authenticator` hook; off by default, server binds 127.0.0.1 |
| Logging | ✅ levels; add JSON format and request ids |
| Metrics | ✅ `/metrics` (Prometheus text): requests, active, queue depth, tokens, tok/s, TTFT/ITL histograms, KV blocks used/free, RSS; add CPU util and effective clock |
| Graceful shutdown | ✅ `begin_drain()`: 503 + `/health` draining, then cancel (§11) |
| Isolation | one model per process; scale with processes/containers, not in-process multi-tenancy |
| Resource limits | KV budget and activation arena are fixed at startup; no allocation grows with traffic |

---

## 17. Architecture decision records

Recorded in full as **DD-068** in `docs/design-decisions.md`. Summary:

| Decision | Choice | Reason |
|---|---|---|
| Same monorepo? | Yes | DynaCore has one consumer today. Separate repos would add version skew and cross-repo PRs for every op change, and gain nothing until a second consumer exists. Boundary is enforced by build + CI instead. |
| DynaCore form | Static library inside the binary | One-file install; LTO across the boundary; no ABI to keep stable. A shared-library option can come with the first external consumer. |
| API style | `Device` object with out-param ops | No hot-path allocation; several devices per process; guarded test device (§5.1). |
| Who owns scheduling? | DynaLM | Scheduling depends on requests, priorities, prefix sharing and SLOs. None of those are tensor concepts. |
| Who owns kernels and their selection? | DynaCore | Kernel choice depends on ISA, cache sizes and shapes. DynaLM passes numbers (`StepShape`), not intent. |
| Paged-KV layout | DynaCore (layout) / DynaLM (policy) | Attention kernels must know addressing; block allocation, sharing and eviction are policy. |
| Native CLI + Docker | Both | Desktop users want one binary; servers want an image. Both wrap the same binary. |
| CMake | Yes | Already in use, all three OSes, CUDA language support, presets. |
| YAML | Strict subset, hand-parsed | Avoid the first new runtime dependency for two-level config. |
| Delay compiler/IR | Until R4 | The op set is still changing (int8 paths, grouped attention, MoE). An IR frozen now would encode today's ops. The recording-device design means waiting costs no rewrite. |
| Rust? | No (2026-10-06) | Earlier decision: apply priorities to the C++ engine; language is not the bottleneck. |

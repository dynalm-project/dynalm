# DynaLM Architecture: How the Local LLM Engine Works

An overview of how DynaLM runs LLMs on a CPU: the platform layer (models, scheduler, server) on top of the DynaCore runtime (tensors, kernels, devices).

Status: all engine phases complete; DynaCore/DynaLM split done (R0, DD-068). Implemented parts are
marked ✅. The full platform design, roadmap and decision summary are in
[platform-design.md](platform-design.md).

## Two layers, one product

```
dynalm/  (namespace dynalm, library dynalm_runtime -> libdynalm, executable dynalm)
  CLI / API server        cli/ ✅, server/ ✅ cpp-httplib, api/ ✅ OpenAI + JSON, metrics/ ✅
        │
  Engine facade           runtime/engine ✅: scheduler thread + RequestStream streaming
        │
  Scheduler               scheduler/ ✅ continuous batching, chunked prefill, policies
        │
  KV policy               kv_cache/ ✅ paged pool + refcounted block tables, prefix_cache/ ✅ radix + hash
        │
  Model runtime           model/ ✅ adapters + Transformer, model_ir/ ✅ ModelConfig + TensorRegistry
        │                 execution/ ✅ BatchPlanner: SeqBatch[] -> StepShape
  Loaders, text           loader/ ✅ GGUF, SafeTensors/HF, GPTQ/AWQ repack; tokenizer/, chat_template/
════════╪═════════════════  dynacore:: public API only (dynacore/include/dynacore/...)
dynacore/ (namespace dynacore, library dynacore)
  Device                  device/device.h ✅ op interface, device_registry ✅, ops.h ✅ op parameters
        │
  Kernel selection        kernel/kernel_plan ✅ plan_kernels(StepShape, HardwareProfile)
        │
  CPU device              cpu/cpu_device ✅ + cpu_kernels table: generic ✅, avx2 ✅, neon ✅
        │
  Paged-KV layout         attention/paged_kv.h ✅ KvGeometry, KvLayerView (addressing only)
  Execution               execution/thread_pool ✅ (QoS, P-cores first)
  Tensor, memory, quant   tensor/ ✅ DType, TensorView; memory/ ✅ Storage, mmap; quantization/ ✅
  Hardware                hardware/ ✅ cpu_info, isa selection, perf counters, process stats
```

Compiled execution (`--execution compiled`, DD-072) inserts `dynacore/ir/RecordingDevice` between
the Transformer and the CPU device: ops are recorded as DynaCore IR, planned once per step shape
(cached), fused where measured to pay (Q/K/V groups, gated MLP on the decode path) and executed
on the same kernels. See [dynacore-compiler.md](dynacore-compiler.md).

DynaLM includes DynaCore headers as `"dynacore/<module>/<file>.h"` and sees the DynaCore names
unqualified through `dynalm/src/common/core.h`. DynaCore has only `dynacore/include` on its
include path, builds and tests alone (`cmake --preset core-only`), and is scanned by
`tests/boundary/check_boundary.py` for LLM-level concepts.

## Dependency rules

Arrows point downward only. A lower layer never includes a higher one.

| Layer | Must not know about |
|---|---|
| Scheduler | model architecture, file format, backend ISA |
| KV cache | file format, model family |
| Model runtime / adapters | GGUF (sees only the Tensor Registry + ModelConfig), SIMD |
| DynaCore (all of it) | model families, file formats, tokenizers, requests, HTTP, scheduling policy |
| API/server | kernels, SIMD, KV layout |

## Model loading path ✅

```
GGUF file ──► GGUF loader ─┐
HF dir ─────► SafeTensors + HF loader ─┴─► TensorRegistry + ModelConfig (IR) ─► ModelArchitecture adapter ─► runtime graph
```

Format conventions (Gemma's (1 + w) norms, Llama 3 RoPE factors, Llama Q/K row layout) are
handled inside the loaders. Adapters and the runtime see one IR (DD-040).

## CPU kernel dispatch

- The baseline binary uses no global `-march`/`/arch` flags, so it runs on any x86-64.
- SIMD variants are compiled per source file (`engine_set_isa()` in
  `cmake/EngineCompilerFlags.cmake`) and gated by `ENABLE_AVX2/AVX512/AMX`.
- At startup `dynacore/hardware/cpu_info` detects CPU features (cpuid + XCR0 OS-state
  checks) and `dynacore/hardware/isa` picks the best tier that is both **compiled** and
  **supported**: `amx > avx512 > avx2 > neon > generic`.
- The choice is made once. Kernels never re-check features per call.
- Future GPU backends add a device dimension beside this. They do not change the
  scheduler, model, or KV interfaces.
- Device memory is owned and touched only by the DynaCore device (`upload`, `allocate`, `copy`,
  `download`, plus gather/scatter/fill ops). A memory-guarded test backend enforces this
  for every model and runtime path (DD-045; contract in `docs/gpu-backend.md`).

## Forward pass (✅)

`Transformer::forward(tokens, positions, kv, block_table)`: embedding → per layer
[norm → QKV (fused or separate, +bias) → optional QK-norm → RoPE → kv_store → paged
attention → O proj → optional post-norm → residual → norm → gated/plain MLP →
optional post-norm → residual] → final norm on the last row → LM head → optional
soft-cap. Every branch is a `ModelConfig` flag.

MoE layers (DD-042) replace the MLP with: router matmul → per-row top-k → for each active
expert, one batched gated MLP over the rows routed to it (a 2-D slice of the 3-D expert
tensors) → weighted scatter-add → optional sigmoid-gated shared expert.

## Threads ✅

HTTP workers (cpp-httplib pool, max-active + 8) → `Engine::submit` → one scheduler
thread → persistent worker pool (`runtime/thread_pool`) for kernels. No thread is
created per request. HTTP workers only parse, submit and relay a `RequestStream`;
they never run model code (DD-034).

## Serving lifecycle ✅ (DD-039)

```
request ─► admission (≤ max-active, else 503) ─► validation (context, KV capacity: 400)
        ─► scheduler queue ─► running (preemptible) ─► finished | cancelled | timed out (504)
```

Each admitted request is owned by an RAII guard that returns its admission slot and
cancels the engine request on every exit path. A shutdown (SIGTERM or
`/admin/shutdown`) drains: new work gets 503 and in-flight requests finish within the
grace period. Configuration is merged as file < env < CLI (DD-038).

## Error model ✅

`Status` / `Result<T>` at every fallible boundary. No exceptions on the hot path.
One request failing must not take down the server (errors are per-request).
Engine status codes map to HTTP: 400 invalid or unsupported, 503 overloaded or
retryable, 504 deadline, 500 internal.

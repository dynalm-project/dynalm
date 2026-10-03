# Architecture

Status: **Phase 6** (single-sequence inference works end to end). This document describes the target architecture and marks
what exists today. Implemented parts are marked ✅; everything else is planned.

## Layering

```
CLI / API server            (cli/ ✅, server/ ✅ cpp-httplib, api/ ✅ OpenAI + JSON, metrics/ ✅)
      │
Engine facade               (runtime/engine ✅: scheduler thread + RequestStream streaming)
      │
Request manager / queue     (runtime/)
      │
Scheduler                   (scheduler/ ✅ continuous batching) — model-agnostic
      │
KV engine                   (kv_cache/ ✅ paged pool + refcounted block tables, prefix_cache/ ✅ radix (default) + hash, memory/)
      │
Model runtime               (model/, model_ir/ ✅)   — consumes ModelConfig only
      │
Kernel dispatch             (backends/backend.h ✅, CpuKernels table ✅)
      │
Backend                     (backends/cpu ✅ generic; avx2/avx512/amx in Phase 17; future GPU)
      │
Platform                    (platform/ ✅ CPU + memory detection, ISA selection)
```

Data: `dtype/` ✅, `tensor/` ✅, `memory/` ✅ (host allocator + Storage; pools later).

Text: `tokenizer/` ✅, `chat_template/` ✅ (neither knows about GGUF; the loader fills `TokenizerData`).

Cross-cutting: `common/` ✅ (Status/Result, platform macros, timer),
`logging/` ✅, `metrics/`, `config/`.

## Dependency rules

Arrows point downward only. A lower layer never includes a higher one.

| Layer | Must not know about |
|---|---|
| Scheduler | model architecture, file format, backend ISA |
| KV cache | file format, model family |
| Model runtime / adapters | GGUF (sees only the Tensor Registry + ModelConfig), SIMD |
| Backends | model families, requests |
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
- At startup `platform/cpu_info` detects CPU features (cpuid + XCR0 OS-state
  checks) and `platform/isa` picks the best tier that is both **compiled** and
  **supported**: `amx > avx512 > avx2 > neon > generic`.
- The choice is made once. Kernels never re-check features per call.
- Future GPU backends add a device dimension beside this. They do not change the
  scheduler, model, or KV interfaces.
- Device memory is owned and touched only by the backend (`upload`, `allocate`, `copy`,
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

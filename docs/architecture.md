# Architecture

Status: **Phase 6** (single-sequence inference works end to end). This document describes the target architecture and marks
what exists today. Implemented parts are marked ✅; everything else is planned.

## Layering

```
CLI / API server            (cli/ ✅ skeleton, server/, api/)
      │
Request manager / queue     (runtime/)
      │
Scheduler                   (scheduler/ ✅ continuous batching) — model-agnostic
      │
KV engine                   (kv_cache/ ✅ paged pool + refcounted block tables, prefix_cache/, memory/)
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

## Model loading path (GGUF parsing ✅, IR next)

```
GGUF file ──► GGUF loader ─┐
SafeTensors ► ST loader ───┴─► TensorRegistry + ModelConfig (IR) ─► ModelArchitecture adapter ─► runtime graph
```

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

## Forward pass (✅)

`Transformer::forward(tokens, positions, kv, block_table)`: embedding → per layer
[norm → QKV (fused or separate, +bias) → optional QK-norm → RoPE → kv_store → paged
attention → O proj → optional post-norm → residual → norm → gated/plain MLP →
optional post-norm → residual] → final norm on the last row → LM head → optional
soft-cap. Every branch is a `ModelConfig` flag.

## Threads (partially ✅)

API threads → request queue → one scheduler thread → persistent worker pool
(`runtime/thread_pool` ✅) for kernels. No thread creation per request.

## Error model ✅

`Status` / `Result<T>` at every fallible boundary. No exceptions on the hot path.
One request failing must not take down the server (errors are per-request).

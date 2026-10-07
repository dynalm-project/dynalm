# DynaCore Compiler Roadmap: Status and Audit

## Status after the compiler roadmap (2026-10-07)

| Roadmap item | State | Where |
|---|---|---|
| IR foundation, verifier, printer/parser, shape/type/layout | done | dynacore-ir.md, DD-071 |
| Recording device (trace + deferred), op-level profiler | done | `bench_op_trace` |
| Canonicalization, memory analysis, cost model, kernel selection | done | passes.h |
| Fusion (Q/K/V groups, gated MLP), decode-shaped only | done, measured | DD-072 |
| Compiled execution in DynaLM, fallback, plan cache (= kernel cache), metrics | done | `--execution compiled` |
| Benchmark framework (decode A/B, end to end, dynacorec) | done | compiler-benchmarks.md |
| Inference language, parser, language → IR, dynacorec | done | DD-073 |
| Tile selection, vectorization, prefetch | not built: no measured headroom (GEMVs at 88–96% of DRAM) | compiler-backends.md |
| AVX-VNNI / AVX-512 / AMX code generation | not built: VNNI measured slower (DD-058); no AVX-512/AMX hardware here | |
| CUDA backend | blocked: no NVIDIA GPU on this machine | gpu-backend.md |
| Runtime JIT | not needed: the plan cache specializes per shape without code generation | |
| q8_0 KV | done, opt-in (speed neutral, −47% memory) | DD-074 |

The rest of this page is the audit as taken before the work started.

# Starting state (2026-10-06)

This audit was taken before the IR/compiler work. It covers what exists, what is partial, and
what has not started. It also records the measurements that decide where a compiler can help.

## Completed

| Area | Where | Notes |
|---|---|---|
| DynaCore/DynaLM split | `dynacore/`, `dynalm/` | DD-068, DD-069. A ctest and CI job enforce the boundary; DynaCore builds alone (`core-only` preset). |
| Device abstraction | `dynacore/device/device.h` | Coarse ops with caller-owned outputs: matmul, matmul_many, attention (paged, GQA-grouped), rope, norms, activations, gather/scatter. |
| CPU device | `dynacore/cpu/` | generic, avx2 and neon kernel tables; int8 decode GEMV (DD-053); blocked fp32 GEMM (DD-036); grouped GQA attention (DD-066) |
| Kernel selection | `dynacore/kernel/kernel_plan.h` | `plan_kernels(StepShape, HardwareProfile)` picks attention split-K. Matmul path (int8 / fused / expand) by row-count thresholds. |
| Paged-KV layout | `dynacore/attention/paged_kv.h` | Block-addressed `KvLayerView` |
| Hardware detection | `dynacore/hardware/` | cpuid + XCR0, hybrid topology, caches, OS, GPU driver probe |
| Quantized formats | `dynacore/quantization/` | Q4_0/1, Q5_0/1, Q8_0, Q2–Q6_K: dequantize and dot |
| Platform features | `dynalm/` | Continuous batching, chunked prefill, radix prefix cache, speculative decoding, MoE, OpenAI server, metrics, registry names, doctor, YAML config (DD-070) |
| A/B tooling | `bench_batch_decode`, `tools/ab_summary.py`, `bench_decode_matmul`, `bench_profile` | Interleaved repetitions and effective-clock reporting |

## Partially completed

| Area | State |
|---|---|
| Kernel registry | Selection exists (`CpuKernels` function table per ISA, `KernelPlan` thresholds), but no registry keyed by (op, dtype, shape, ISA) and no cost model. |
| Memory planning | Activation buffers are allocated once per model at the maximum batch size. There is no lifetime analysis or reuse. |
| Fusion | Two model-level fusions exist: fused QKV / gate-up when the checkpoint stores them fused, and `matmul_many` for MoE experts. There is no automatic fusion. |
| VNNI | Detected. A prototype int8 GEMM was measured slower than fp32 (DD-058), so no VNNI kernels ship. |
| GPU | Contract only (`docs/gpu-backend.md`, DD-045). Detection works; no device is built. |

## Not started

The IR (types, graph, builder, verifier, printer, parser), the recording device, compiler
passes, the cost model, tile selection, code generation, the inference language and
`dynacorec`, the compiled execution mode in DynaLM, the kernel cache, the CUDA device and
runtime specialization.

## Needs redesign

Nothing blocks the compiler. Two existing designs shape it:

- The forward pass already expresses itself as a stream of `Device` ops. A recording device
  turns that stream into IR without touching model code (platform-design.md §15).
- `KernelPlan` is per step, not per op. A compiler that selects kernels per op needs
  per-op plans. The IR carries them as attributes, and the CPU device accepts a plan per call.

## Where the time goes (measured, i7-1255U, 10 threads)

Qwen2.5-1.5B Q4_K_M, decode at about 300 tokens of context (`bench_profile`). Step time is
82–86 ms.

| Section | Share of decode step |
|---|---|
| MLP up/gate + down (matmul) | 60% |
| LM head (matmul, 151936 × 1536 Q6_K) | 13% |
| QKV (matmul) | 12–13% |
| attention output (matmul) | 5% |
| attention | 4% |
| norms, RoPE, activation, KV store | < 4% |

Bandwidth:

| Measurement | GB/s |
|---|---|
| DRAM read ceiling (`tools/proto/read_bw.cpp`, 10 threads, AVX2 streaming) | 18.4 |
| Best single GEMV at M = 1 (`bench_decode_matmul`; Q4_K/Q6_K, int8 path) | 14.0–17.6 (76–96% of ceiling) |
| End-to-end decode (1.1 GB of weights per token / 84 ms) | ~13 (70% of ceiling) |

Consequences for the compiler:

1. **Large GEMVs are near the hardware limit.** A generated GEMV kernel cannot be much faster
   than ~5–10% on the big matrices. Claims beyond that would be noise.
2. **The 30% end-to-end gap is between and around the big kernels.** In the model, QKV costs
   0.36 ms per layer. The same three matmuls cost 0.12 ms in isolation, a 3× gap, and it is
   unchanged with int8 off. The small K/V projections (256 rows) are latency-bound. Per-op
   timing of the real op stream is the first job of the recording device.
3. **Elementwise fusion is worth at most the < 4% those ops take**, plus whatever dispatch and
   barrier cost goes with them. That is worth doing only when measured.

The compiler roadmap therefore starts with the IR, plus a recording device that times every
op. That measurement decides which optimizations to build.

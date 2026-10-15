# DynaCore: The Low-Level CPU Inference Runtime Behind DynaLM

DynaCore is the low-level inference runtime under DynaLM. It holds:

- tensors and memory;
- hardware detection;
- devices and kernels;
- quantized formats;
- the paged-KV layout;
- the IR, compiler and language.

It knows nothing about LLMs as products: no model families, file formats, tokenizers,
requests or HTTP. The `core-only` build and its CI job enforce this.

| Module (`dynacore/include/dynacore/...`) | Contents |
|---|---|
| `base/` | `Status`/`Result`, platform macros, `FunctionRef`, timers, fast exp |
| `hardware/` | CPU features (cpuid + XCR0), hybrid P/E topology, caches, ISA selection, OS info, GPU probe via the driver, perf counters, process stats |
| `tensor/` | `DType` (f32/f16/bf16/int and GGML block formats), shapes, layouts, `TensorView`, `Tensor` |
| `memory/` | aligned host memory, `Storage`, memory-mapped files |
| `quantization/` | block layouts, dequantization, Q8_0 quantization |
| `execution/` | `ThreadPool` (P-cores first, dynamic chunks, stats) |
| `kernel/` | `KernelPlan`, `plan_kernels(StepShape, HardwareProfile)`, the CPU matmul kernel registry (`matmul_registry.h`) |
| `attention/` | paged-KV geometry and addressing (`KvGeometry`, `KvLayerView`); f32, f16 and q8_0 |
| `device/` | the `Device` op interface (matmul, matmul_many, matmul_gated, attention, rope, norms, ...), device registry |
| `cpu/` | `CpuDevice` and the kernel tables (generic, AVX2, NEON) |
| `ir/` | IR, text form, verifier, recording device, compiler passes, graph executor ([dynacore-ir.md](dynacore-ir.md), [dynacore-compiler.md](dynacore-compiler.md)) |
| `lang/` | the DynaCore language ([dynacore-language.md](dynacore-language.md)) |

The version is in `dynacore/version.h` (`DYNACORE_VERSION_STRING`).

- The public headers are source-compatible within a minor version while the major version
  is 0.
- There is no ABI promise, because DynaCore is linked statically.

Tools:

- `dynacorec`: the compiler driver;
- benchmarks: `bench_kernels`, `bench_matmul`, `bench_quant`, `bench_tensor`,
  `bench_thread_pool`, `bench_elementwise`.

Tuning knobs for experiments (not for production):

- `DYNACORE_GEMM_KC`, `DYNACORE_MATMUL_EXPAND_MIN`, `DYNACORE_INT8_DECODE_ROWS`;
- `DYNACORE_MATMUL_CHUNKS`, `DYNACORE_ATTN_GROUPED`, `DYNACORE_PIN_THREADS`.

## Matmul kernel registry

`CpuDevice::matmul` asks the registry (`dynacore/kernel/matmul_registry.h`) which kernel runs each `y = x · wᵀ`.

**How selection works:**
- The choice depends only on the call's shape (rows M, N, K, weight type, row stride), the step's `KernelPlan`, and what this CPU tier provides. That includes whether an interleaved copy of the weight exists.
- Candidates are tried in a fixed priority order, and the first eligible kernel wins.
- There is no timing-based tuning, so the same shape on the same machine always gets the same kernel.
- Every kernel has a descriptor: name, weight types, row rule, ISA, packing, activation arithmetic, accuracy class and default state. List them with `matmul_kernels()`.

**Fallback:**
- The fp32 kernels (`f32_expand_gemm`, `f32_row_dot`) accept every supported shape, so a matmul always has a kernel.
- On CPUs without AVX2 (or with `DYNACORE_ISA=generic`), every integer path is ineligible and selection falls back to fp32.
- A matmul with invalid metadata (non-positive dims, K not a multiple of the weight block, unsupported weight type) stops the process with a message before any kernel runs.

**Diagnostics:**

| switch / interface | what it shows |
|---|---|
| `DYNACORE_MATMUL_EXPLAIN=1` | logs to stderr, once per (kernel, shape): the selected kernel and the reason each higher-priority kernel was passed over (`disabled`, `not_enabled`, `rows`, `alignment`, `weight_type`, `no_kernel`, `no_packed`, `size_gate`, `not_dense`) |
| `DYNACORE_MATMUL_TIMING=1` | adds wall time per kernel and per shape. Two clock reads per matmul; off by default |
| `dynacore_matmul_calls_total{kernel="..."}`, `dynacore_matmul_rows_total{kernel="..."}` | in `GET /metrics`: calls and activation rows per kernel |
| benchmark JSON (`dynalm benchmark --out`) | `diag.matmul_kernels`: calls, rows, MACs (and ms with timing on) per kernel for each point |

**Disabling kernels:**
- `DYNACORE_MATMUL_DISABLE=name,name` removes kernels from selection; it falls through to the next eligible kernel.
- `DYNACORE_MATMUL_DISABLE=all-accelerated` keeps only the fp32 reference arithmetic.
- Unknown names are reported and ignored. The fp32 kernels cannot be disabled.

**Experimental CPU features (off by default, opt-in per process):**

| setting | effect | status |
|---|---|---|
| `DYNACORE_INT16_PREFILL_ROWS=64` | int16 activations for prefill matmuls of 9–64 rows on large Q4_K weights, instead of fp32 | passes the DD-053 accuracy contract on Qwen2.5-1.5B and Qwen3-4B; opt-in |
| `DYNACORE_INT16_PREFILL_SPLIT=1` (with the line above; needs AVX-VNNI) | the same int16 arithmetic computed as split bytes with `vpdpbusd`: bit-identical results, faster on P-cores | experimental; see the CHANGELOG for the measured conditions |
| `DYNACORE_DECODE_WIDE_ROWS=16` | decode-only steps of 9–16 rows keep the int8 / int16 decode kernels instead of fp32. Each row gets the arithmetic it gets at ≤ 8 rows (bit-identical per row). Above 8 rows the int8 kernel uses its 8-row tiles: one pass over each packed weight group per 8 rows. `DYNACORE_WIDE_ROWS8=0` keeps the 4-row tiles. Clamped to 0..16. Has no effect at ≤ 8 concurrent decoders | experimental; measured results in the CHANGELOG |

Leaving these unset gives the default kernels, unchanged from 0.1.3.

**External kernels:** `CpuDevice::register_external_matmul` (an extension point, for example for generated kernels) is consulted only while `KernelPlan::external_matmul` is set. Nothing in DynaLM registers one.

**Tests:** `test_matmul_registry` (built with `ENABLE_BENCHMARKS`, no model files) checks:
- selection against the pre-registry dispatch;
- every kernel's numerics;
- the bit-identity claims;
- the generic-tier fallback, diagnostics, external kernels, and concurrent devices.

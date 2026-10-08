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
| `kernel/` | `KernelPlan`, `plan_kernels(StepShape, HardwareProfile)` |
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

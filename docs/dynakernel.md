# DynaKernel IR v0

DynaKernel is the kernel-level layer of DynaCore. It sits below DynaCore IR
(`dynacore/ir`, DD-071) and the DynaCore language (`dynacore/lang`, DD-073).
Those describe what a forward step computes: SSA values with symbolic shapes,
inference ops (attention, rope, kv_write), and a kernel name chosen by passes.
DynaKernel describes the kernel invocations themselves: concrete shapes,
caller-owned buffers, and the numerical contract each kernel must honor. A
future native backend or tuner needs that information to pick, verify or
generate an implementation for one shape.

```
ProgramBuilder ──► Program ──► verify() ──► Backend::check() ──► Backend::run(Program, Bindings)
 (typed values,    (values +    (structure,    (can this target       │ verify + check again,
  matmul ops)       ops, in      shapes,        honor each op's        │ validate bindings
                    order)       dtypes,        contract?)             │ (layout, alignment, overlap)
                                 layouts,                              ▼
                                 dependencies)                 ReferenceBackend
                                                               own CpuDevice ──► Device::matmul
                                                               (existing kernels: fused dot,
                                                                fp32 panel GEMM, int8 rows)
```

## What is implemented

| Piece | Where | Notes |
|---|---|---|
| Values | `dynacore/include/dynacore/dynakernel/program.h` (`Value`, `ValueRole`) | dtype, shape and byte strides are a `dynacore::TensorLayout`, the same layout kernels receive. Quantized dtypes keep their GGUF block layout (`DType`, `dtype_row_bytes`). Required alignment, and a role: activation, weight, bias or result |
| Ops | `program.h` (`Op`, `OpKind`, `MatMulAttrs`) | `kMatMul`: `y[M,N] = x[M,K] · w[N,K]ᵀ (+ bias[N])`, fp32 accumulation, an activation-format contract (`auto` or `f32`). Other kinds are declared only (quantize, dequantize, bias_add, activation, rms_norm, attention, fused_group) |
| Program | `program.h` (`Program`) | values and ops in execution order. Destination passing: each op writes one declared result value |
| Builder | `program.h` (`ProgramBuilder`) | typed construction of dense values and matmuls (`matmul`, `matmul_into` for strided destinations). `finish()` returns the verified program or the first diagnostic |
| Text form | `to_text()` in `dynacore/src/dynakernel/program.cpp` | deterministic, one line per value and op |
| Verifier | `verify_all()` / `verify()` in `program.cpp` | see below |
| Backend boundary | `dynacore/include/dynacore/dynakernel/backend.h` (`Backend`, `TargetDesc`, `Bindings`) | `check()` says what a target can run; `run()` executes over caller buffers |
| Reference backend | `dynacore/src/dynakernel/reference_backend.cpp` (`ReferenceBackend`) | lowers matmul to `CpuDevice::matmul` on a device it owns |
| Tests | `dynacore/tests/test_dynakernel.cpp` (target `test_dynakernel`) | construction, verifier cases, execution against a double-precision reference (f32 and Q8_0 weights; strided x/y through the fused, int8 and panel paths), rejections (bindings, alignment, capacity, overlap), and defaults. Run: `build/msvc-release/dynacore/test_dynakernel.exe`; set `DYNACORE_ISA=generic` for the generic tier |

### Supported in v0

- Op: matmul only (lowered from graph `matmul` / `quantized_matmul`, see below).
- x and y: f32, rank 2. The innermost dimension is contiguous; row strides are free, so strided views such as column slices of a wider buffer work.
- Weights: any dtype `dequant_supported()` accepts and the CPU device has a kernel for (`f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_K` … `q6_K`). Dense row-major in GGUF block order.
- Bias: optional, dense f32 `[N]`.
- Accumulation: f32.
- Activation format: `auto` runs the backend's verified default dispatch, the same `KernelPlan::defaults()` as production. `f32` turns off every reduced-precision activation path (int8 per-32, sub-scaled, super-block, packed Q4_K, int16). `int8` and `int16` cannot be requested: the reference backend returns `kUnsupported` rather than forcing a kernel outside the planner's validated row limits.

## Verifier guarantees

`verify_all()` reports every problem, and `verify()` returns the first one. Each message names the op and value: `op 0 (matmul 'q'): x has 64 columns but w has 128 (K mismatch)`. Nothing is repaired. The verifier checks:

- Values: dtype valid, rank 1–4, positive dims, non-negative strides. The innermost stride equals the packed element stride (0 for quantized). A quantized innermost dim is a whole number of blocks. Outer strides do not overlap inner extents, and for scalar dtypes they are whole elements (kernels index rows as `T* + stride / sizeof(T)`). Alignment is 0 or a power of two.
- Ops: the kind is defined in v0. Declared-only kinds give `kUnsupported` (`Diagnostic::unsupported`), and their destination is not also reported as never written. Every operand and output id exists. Results are read only after an op wrote them (op order is execution order). Each result is written exactly once, and every result is written by some op.
- Matmul: roles are right (x activation or result, w weight, bias bias, y result). Ranks are 2/2/2. x and y are f32. The weight dtype has a defined dequantization and the weight is dense. K matches, y is `[M, N]`, and `M,N,K` attributes agree with the operands (they are not recomputed). The bias is a dense f32 `[N]`. Accumulation is f32. The output is not an operand.

`ReferenceBackend::check()` runs `verify()` first, so it is safe on unverified input. `run()` calls `check()`, then validates bindings before any kernel runs:

- Every value is bound to a non-null view equal to its declared dtype, shape and strides.
- The pointer meets the larger of the declared alignment and `natural_alignment(dtype)`: 4 for f32, 2 for f16 and fp16-scaled GGUF blocks, 4 for Q8_K.
- A result buffer does not overlap any other bound buffer.
- Bound with a capacity (`bind(v, view, bytes)`) or a `Tensor` (capacity from its `Storage`), the layout's byte span fits the buffer. A bare `TensorView` carries no size and is trusted.

Ownership: `Bindings` never own memory. Buffers must stay valid until `run()` returns; results are written in place, and nothing is allocated or resized. `Device::matmul` has no error channel, so every error `run()` reports comes from validation before execution. Once kernels start, they run to completion.

Concurrency: `run()` is not thread-safe (the device keeps per-call scratch). The `ThreadPool` must not run another device's work at the same time, so do not hand the backend a live engine's pool.

## Reference backend

`ReferenceBackend(ThreadPool&, CpuIsa)` owns a `CpuDevice`. Before each matmul it sets that device's kernel plan from the op's contract (`plan_for()`), then calls `Device::matmul`. The CPU kernels choose the path exactly as in production: the fused dot for one row, int8 rows within the plan's limits, fp32 panel GEMM otherwise. Nothing is generated, approximated or substituted. A model's device never sees these plans: the backend has its own device, and `test_defaults_untouched` checks this.

Differences from a model's device:

- The backend does not `prepack_weight`, so Q4_K interleaved copies (DD-078) are not used. Those kernels compute the same contract on a different layout.
- `auto` follows `KernelPlan::defaults()`, including its `DYNACORE_*` environment overrides.

## Lowering from graph IR

`dk::lower(const ir::Graph&, const LowerOptions&)` (`dynacore/include/dynacore/dynakernel/lower.h`, `dynacore/src/dynakernel/lower.cpp`) turns a DynaCore IR graph into one DynaKernel `Program`. It returns a `LowerResult` with the program, the graph→kernel value and op maps, and structured `LowerIssue`s, each with a kind, graph op, graph value and message.

Lowering is all-or-nothing. Any issue means no program at all, never a partial one. The graph must first pass `ir::verify_all`.

| Graph construct | Lowered as | Otherwise |
|---|---|---|
| `matmul`, `quantized_matmul` | one `kMatMul`, in graph order | — |
| `x` | f32 tensor that is an `input` (role activation) or a lowered matmul's result (role result) | f16/bf16 `x` is `kUnsupportedType`: the graph allows it, but the kernels read x as f32, and DynaKernel has no cast op |
| `w` | `weight` leaf: f32/f16/bf16 row-major, or a quantized dtype in GGUF blocks (`kBlocked(block)` or row-major) | a computed right-hand side is `kUnsupportedType` |
| `bias` | f32 `input`/`weight` leaf `[N]`, role bias | non-f32 or computed bias is `kUnsupportedType` |
| Layouts | `row_major`; `strided(p)` on f32 tensors becomes `strides[0] = 4p` | `col_major`, `packed`, `paged` are `kUnsupportedType` |
| Symbolic dims | bound from `LowerOptions::symbols` | unbound or non-positive symbol is `kUnboundSymbol` (never defaulted) |
| Unused leaves (index arrays, KV caches, other weights) | not lowered; need no binding | — |
| Any other op (`rmsnorm`, `act_mul`, `rope`, `attention`, …) | — | `kUnsupportedOp`; the whole graph is rejected |
| Lowered program fails `dk::verify` | — | `kInternal` (a lowering bug) |

`LowerResult::status()` is `kUnsupported` when every issue is an unsupported op or type, else `kInvalidArgument`. The activation contract (`LowerOptions::act`, default `auto`) applies to every lowered matmul.

Equivalence (`dynacore/tests/test_dynakernel_lowering.cpp`): for the same bound leaves, `GraphExecutor` (op order) and `lower` + `ReferenceBackend` (`auto`) are bit-identical, because both call `CpuDevice::matmul` with `KernelPlan::defaults()`. The test also shows the comparison is sensitive: an `f32` contract changes the bits on AVX2, where the int8 path exists. Shapes are checked against a double-precision reference for f32 and Q8_0 weights. Q4_K/Q6_K accuracy is covered separately (next section).

## Q4_K / Q6_K accuracy

`dynacore/tests/test_dynakernel_quant.cpp` (target `test_dynakernel_quant`).

- **Reference:** an independent double-precision decoder of the GGUF Q4_K/Q6_K layouts, written from the format, not shared with the loader. It gives the exact value of each stored weight; `y_ref = Σ x·w` is accumulated in double.
- **Weights:** random block bytes (every code and scale pattern), fp16 super-scales of both signs, and edge blocks: zero scale, saturated 6-bit scales/mins (Q4_K), int8 scale −128 (Q6_K).
- **Activations:** smooth rows, an outlier channel (±40), tiny values, a zero 32-block, large mixed signs, an all-zero row.
- **Shapes:** K ∈ {256, 768, 1280, 2304} (1–9 super-blocks; 1280 and 2304 cross `gemm_k_block` = 1024), N ∈ {1, 5, 33}, M ∈ {1, 2, 3, 4, 5, 8, 9, 17}, each with the `f32` and `auto` contracts.
- **Dequantizers:** the loader (`dequantize_row`), the CPU tier's own dequantizer and gguf-py fixtures (`DK_QUANT_FIXTURES`, made by `tools/make_quant_fixtures.py`) are each checked against the exact decode.

Bounds are derived per element, not fitted. With `u = 2⁻²⁴`, `T = |d·sc·q| + |dmin·m|` and `S = Σ|x|·T`:

- fp32 paths: `|e| ≤ (K+4)·u·S` (2u·T for the fp32 weight value, K·u·S for accumulation).
- int8 paths add `Σ (h/2)·|w|`, where h is the activation rounding step of the path's quantizer: per-32 `amax32/127`; sub-scaled `u_s·m_b`, with `u_s = amax256/(127·128)` and `m_b = ⌈amax32/(127·u_s)⌉`.

The int8 paths are also required to change the bits relative to `f32`, which is evidence that they ran.

Measured on the i7-1255U (2026-10-10). Max rel is relative to the case's largest |y_ref|; err/bound is the largest ratio over all elements.

| Tier | dtype | Path (rows) | Cases | max \|err\| | max rel | err/bound |
|---|---|---|---|---|---|---|
| AVX2 | q4_K | fused dot (M=1, f32) | 12 | 7.1e-05 | 3.6e-07 | 0.002 |
| AVX2 | q4_K | fp32 panel (M≥2 f32; M≥9 auto) | 108 | 3.5e-03 | 1.5e-06 | 0.009 |
| AVX2 | q4_K | int8 per-32 (M=1 auto) | 12 | 2.22 | 0.016 | 0.108 |
| AVX2 | q4_K | int8 sub-scaled (M=2–8 auto) | 60 | 62.4 | 0.042 | 0.338 |
| AVX2 | q6_K | fused dot | 12 | 9.7e-05 | 3.3e-07 | 0.002 |
| AVX2 | q6_K | fp32 panel | 108 | 1.7e-03 | 6.9e-07 | 0.007 |
| AVX2 | q6_K | int8 per-32 | 12 | 1.44 | 0.029 | 0.136 |
| AVX2 | q6_K | int8 sub-scaled | 60 | 49.6 | 0.055 | 0.339 |
| generic | q4_K | fused dot / fp32 panel | 24 / 168 | 7.0e-05 / 5.5e-03 | 3.7e-07 / 1.2e-06 | 0.002 / 0.016 |
| generic | q6_K | fused dot / fp32 panel | 24 / 168 | 7.2e-05 / 2.7e-03 | 3.8e-07 / 1.5e-06 | 0.002 / 0.013 |

The int8 errors are activation rounding, as designed (DD-053/DD-077). The worst cases are the outlier and ×20 rows. The loader, AVX2 and generic dequantizers and gguf-py all equal the exact decode bit for bit on the test blocks and the fixtures.

### Prepacked (interleaved) Q4_K

`ReferenceBackend::prepack_weight(view, capacity)` is opt-in. It makes the DD-078 interleaved 8-row copy of a Q4_K weight on the backend's own device, through the existing `CpuDevice::prepack_weight`.

- **Validated:** dtype q4_K (packed q6_K serves int16 activations only), dense `[N, K]`, `N % 8 == 0`, non-null, natural alignment, and the byte capacity when given. The device must really have made a copy (`PrepackStats::tensors`). Otherwise the call fails with the device's reason: memory budget, `DYNACORE_Q4_REPACK=off`, or no kernel on this tier (generic).
- **Explicit:** a copy is used for a bound weight only when the binding has the registered data pointer, dtype and shape. Every other op runs with `q4_repack_min_rows = 0`, which only stops the device from attaching any packed copy. So an ordinary GGUF weight is never treated as packed, even at an address the device has a copy for.
- **Lifetime:** the device owns the copy until the backend is destroyed; `CpuDevice` has no release. `release_packed(data)` ends the registration before the source is freed or reused, and a released address cannot be packed again on that backend. While registered, the source bytes are FNV-1a fingerprinted at packing time and re-checked on every run (`RunOptions::check_packed`, default on). A changed source fails with "interleaved copy is stale".
- **Reached:** after packing, the test zeroes the source bytes and runs with checking off. Rows 2–8 still match the original weights within the sub-scaled bound, so the result came from the copy alone. Rows 1 and 9 (per-32 int8 and fp32 panel, which read the source) give exactly 0.
- **Accuracy** (AVX2, K ∈ {256, 1280, 2304}, N ∈ {8, 24, 64}, M ∈ {2, 3, 4, 5, 8}, 45 cases, same bound as the sub-scaled path): max |err| 40.7, max rel 0.031, err/bound 0.247. The unpacked sub-scaled kernel gives the same maxima but different bits in all 45 cases (another summation order). A grouped run with a packed weight equals the single packed run.

Not reachable through `ReferenceBackend`, so untested here:

- super-block activations (DD-075: off in the defaults, which `plan_for` uses);
- int16 activations (DD-076: model-set for FFN-down only);
- the generic tier's int8 kernels (`CpuDevice` uses int8 only where accelerated, i.e. AVX2).

## Grouped execution

`ReferenceBackend::run(p, b, RunOptions{.group_independent = true})` runs maximal runs of consecutive, independent matmuls with the same contract as one `Device::matmul_many`, like `GraphExecutor` does for a planned matmul group. `plan_calls()` shows the partition: a dependency or a contract change starts a new call, and a group of one is a plain `matmul`.

It is not the same computation as one `matmul` per op:

| Case | Grouped vs single |
|---|---|
| int8 paths; single-row fused dot | bit-identical (same kernels) |
| quantized panel, K ≤ `gemm_k_block` (1024) | bit-identical |
| quantized panel, K > 1024 | different summation order (`matmul` K-blocks, `matmul_many` does not); within the fp32 bound |
| f32 weights, M ≥ 2 | `matmul`: panel GEMM; `matmul_many`: per-row dot; within the fp32 bound |
| strided x | `matmul`: per-row dots; `matmul_many`: panel GEMM with a row stride |

Tested: grouped and single runs both stay within the reference bounds. Outputs are not swapped. A planned `GraphExecutor` run (one matmul group of Q4_K/Q6_K/Q4_K) is bit-identical to the lowered grouped run. Validation runs before any kernel in both modes, so a rejected run (overlap, short capacity, unsupported contract) writes no output, including the outputs of ops before the failing one.

## Generated C++ backend (codegen stage 0)

The first code generator emits C++ for one constrained kernel family, which the existing toolchain compiles ahead of time. There is no JIT, LLVM or MLIR.

**Contract** (`dynacore/include/dynacore/dynakernel/codegen.h`): `y[M,N] = x[M,K]·w[N,K]ᵀ (+ bias[N])`.

- w: Q4_K in its GGUF block layout, dense rows, not prepacked.
- x, y: fp32 with contiguous rows and any whole-element row stride.
- bias: optional fp32 `[N]`.
- M ∈ 1..8, K a positive multiple of 256, N > 0, all fixed per kernel, with the bias variant part of the specialization.
- fp32 activations and fp32 accumulation. Each weight value is `(d·sc)·q − (dmin·m)` in fp32, the loader's `dequantize_q4_K` arithmetic, accumulated in k order.

Everything else is rejected before execution (`validate_spec`, `GeneratedBackend::check`): M = 9, a shape or bias variant missing from the table, non-Q4_K weights, the int8/int16 contracts, unverified programs. Bindings use the shared `dk::validate_bindings`: layout, alignment, capacity, overlap.

**What it is:** new portable scalar C++, one function template instantiated per specialization. It does not wrap any DynaCore kernel, and it is not tuned; no performance claim is made. `GeneratedBackend` owns no `Device` and can only call generated function pointers. It splits N into 8-column chunks across the `ThreadPool`; each output element is computed by one thread in fixed k order, so results don't depend on the thread count.

**Build flow** (`dynacore/CMakeLists.txt`, tests only):

1. `dkgen` (`dynacore/tools/dkgen.cpp`) calls `generate_q4k_source(q4k_test_specs())`. It accepts built-in spec sets only, no user source.
2. It writes `<build>/dynacore/dynakernel_gen/dk_generated_q4k.cpp`, rewriting it only when the text changes. That unit holds 56 kernels: M = 1–8 × (N, K) ∈ {(8, 256), (33, 256), (33, 256)+bias, (16, 768), (24, 1280), (24, 1280)+bias, (7, 2304)}.
3. CMake compiles it into the static library `dynakernel_generated`, linked only by `test_dynakernel_codegen`.
4. The unit exports `dk_generated_q4k_kernels`, `dk_generated_q4k_fingerprint` and `dk_generated_q4k_version` (C linkage). Each kernel has its own call counter.

**Tests** (`dynacore/tests/test_dynakernel_codegen.cpp`):

- Generation is deterministic; the linked unit's fingerprint equals a fresh generation's, and a rerun of `dkgen` reproduces the file byte for byte.
- Execution proof: the selected kernel's counter moves once per column chunk while no other kernel's moves, and a table whose entry is swapped for a stub makes the backend output the stub's values.
- Accuracy: every one of the 56 specializations against the independent double-precision Q4_K decode, with bound `(K+5)·u·(S+|bias|)`. The reference backend's f32, per-32, sub-scaled and prepacked paths run on the same data, each within its own bound.
- Strided x/y with bias, 1 vs 4 threads bit-identical, and the rejections.

Measured on the i7-1255U (2026-10-10), err/bound as the largest ratio over all elements:

| Implementation | Tier | Cases | max \|err\| | max rel | err/bound |
|---|---|---|---|---|---|
| generated (fp32) | any | 56 | 0.011 | 1.6e-06 | 0.022 |
| reference f32 contract | AVX2 | 56 | 0.0022 | 6.1e-07 | 0.006 |
| reference int8 per-32 | AVX2 | 7 | 2.13 | 0.012 | 0.114 |
| reference int8 sub-scaled | AVX2 | 49 | 58 | 0.022 | 0.214 |
| reference packed q4_K | AVX2 | 28 | 58 | 0.022 | 0.184 |
| reference f32 / auto (both fp32) | generic | 112 | 0.0057 | 6.7e-07 | 0.013 |

The generated kernel is never bit-identical to the reference f32 paths (0 of 56): its sequential k-order sum differs from their SIMD/panel order. It has the largest fp32 error of the three, at 2% of the bound.

### AVX2 variant

`generate_q4k_source(specs, CodegenIsa::kAvx2, unit)` emits the same family with AVX2 + FMA intrinsics:

- Per output column, each 32-value sub-block is widened to 4 × 8 fp32 lanes.
- `w = fma(d·sc, q, −(dmin·m))`.
- Every row accumulates `fma(w, x, acc)` in 8 lanes, and the lanes are summed in a fixed tree at the end.

The contract and error bound are the same as the scalar kernels; the summation order differs.

- **Build:** the unit (`dk_generated_<unit>_avx2.cpp`) is the only generated file compiled with `engine_set_isa(AVX2 …)`. It contains `#error` without `__AVX2__`; `compile_commands.json` shows `/arch:AVX2` only on it and the two existing AVX2 kernel files.
- **Runtime guard:** a `GeneratedBackend` over an AVX2 table runs only when `select_isa()` picked the AVX2 tier, which honors `DYNACORE_ISA=generic`. Otherwise `runnable()`, `check()` and `run()` return `kUnsupported` and nothing executes. A table mixing variants is `kInvalidArgument`.
- **Tests:** the whole codegen suite runs on both tables (provenance, execution proof, accuracy, strides/threads, rejections). On the generic tier the test asserts the AVX2 table is refused, its output untouched and zero AVX2 calls made.
- **Accuracy** (56 specializations): max |err| 0.0030, max rel 7.5e-07, err/bound 0.006, versus 0.022 for scalar (shorter lane sums). Bit-identical to the reference f32 path in 11 of 56 cases and to scalar in 0 of 56.

### Microbenchmark (`bench_dynakernel_q4k`)

Model-free. It compares gen_scalar, gen_avx2, ref_f32 (fused dot or fp32 panel), ref_int8 (per-32 or sub-scaled) and ref_int8_pk (prepacked) on the same synthetic Q4_K weights and inputs. Shapes are Qwen2.5-1.5B-sized: N×K ∈ {1536×1536, 8960×1536, 1536×8960}, M ∈ {1, 2, 4, 8}.

Every output is verified against the double-precision reference within its path's bound before timing, and must be reproduced bit for bit after timing. Samples of about 20 ms are interleaved across implementations, after 3 warm-up rounds; 15 samples per point. It reports median, p10, p90, spread, GFLOP/s and weight GB/s.

Results from 2026-10-10 on the i7-1255U (MSVC 19.44, Release). Two repetitions per configuration agreed. Threads=1 runs used `DYNACORE_PIN_THREADS=1` (median spread 12%). Threads=10 runs were unpinned (median spread 21–28%); a pinned 10-thread run had tens-of-ms stalls and was excluded.

| | gen_avx2 ÷ gen_scalar | gen_avx2 ÷ ref_f32 (same fp32 numerics) | fastest overall (int8 numerics) |
|---|---|---|---|
| M = 1 | 0.24–0.30 | 1.4–1.5 (ref fused `vec_dot_q4_K` faster) | ref_int8 per-32, 1.5–1.8× ref_f32 |
| M = 2–4 | 0.18–0.33 | 0.81–1.16; parity within noise; gen_avx2 ahead only for some 1-thread shapes with K = 8960 | ref_int8_pk, 2.8–5.7× ref_f32 |
| M = 8 | 0.09–0.20 | 1.0 (1 thread) to 1.2–1.35 (10 threads) | ref_int8_pk, 1.9–2.8× ref_f32 |

Correction (2026-10-10): the int8 figures in this table were first reported from a misparsed CSV. The `numerics` label "int8 act, …" contains a comma, so those rows shifted one column and p10 was read as p50. The fp32 rows were unaffected. The values above are re-derived from the same raw files with comma-aware parsing, and the CSV writer now quotes text fields.

Raw outputs: `D:\cpu-llama-snapshots\bench-dynakernel-q4k-*.txt|csv` and `bench-dk-*.txt|csv`.

**Conclusion:** the generated AVX2 kernel is a large step over the scalar kernel. It does not beat the existing hand-written fp32 paths, which are as fast or faster for every M, and it is far slower than the int8 paths. That gap is the activation-precision choice, not codegen.

## INT8 Q4_K: audit of the existing kernels (2026-10-10)

All AVX2 kernels are in `dynacore/cpu/avx2/kernels_avx2.cpp`; the dispatch is `CpuDevice::int8_kernel` / `int8_rows` in `dynacore/cpu/cpu_device.cpp`.

**What the benchmark labels run** (KernelPlan defaults; `DYNACORE_VNNI` unset):

| Label | M | Activations | Dot kernel |
|---|---|---|---|
| `ref_int8` | 1 | per-32, `quantize_act` | `dot_rows_q4_K` (GGUF rows) |
| `ref_int8` | 2–8 | sub-scaled, `quantize_act_sx` | `dot_rows_sx_q4_K` (GGUF rows) |
| `ref_int8_pk` | 2–8 | sub-scaled | `dot_x8_sx_q4_K` (interleaved `BlockQ4_Kx8`, DD-078) |

The VNNI variant `dot_x8_sx_q4_K_vnni` (`kernels_avxvnni.cpp`) is registered only with `DYNACORE_VNNI=1`, so the benchmark did not use it. `int8_rows` calls every int8 kernel with at most 4 rows (`kDotRowsMax`). For M = 5–8 it makes two passes, re-reading and re-unpacking the weights for each.

**Activation quantization.**

- **Per-32** (`quantize_act`): per 32 values, `d = amax/127`, codes `round_nearest_even(x/d)` (`_mm256_round_ps`), packed with saturation to int8. Since `|x/d| ≤ 127`, codes are in [−127, 127] and the saturation never triggers. The code sum is kept for the weight-minimum term.
- **Sub-scaled** (`quantize_act_sx`, DD-077): per 256 values, `u = amax256/(127·128)`. Per 32-block, `m = clamp(⌈amax32/(127·u)⌉, 1, 128)`, codes `round(x/(u·m))` in [−127, 127]. It stores `u`, `m[8]` and `ms[b] = u·m_b·Σcodes_b`. One unit scale per super-block lets the integer sub-block scale absorb `m`.

**Weights.** Q4_K value `d·sc_j·q − dmin·mn_j`, with 6-bit `sc` and `mn` unpacked by `get_scale_min_k4`. `BlockQ4_Kx8` (`repack.h`) holds 8 rows × 256 values. `d` and `dmin` are pre-converted to fp32, `sc`/`mn` unpacked as `[sub-block][row]`, and codes laid out as `qs[jp][c][4r+t]`, so 32 bytes = 4 values × 8 rows.

**Accumulation** (`dot_x8_sx_q4_K`).

- **Per sub-block j and row:** 8 × `maddubs(codes u8, broadcast 4 activation bytes s8)` are added in int16. Each lane is at most 8·2·15·127 = 30480 < 2¹⁵, so there is no saturation.
- **Scaling:** one `madd` with `sc_j·m_j` (≤ 63·128 = 8064, int16) gives one int32 lane per weight row. The int32 sum over half a super-block (4 sub-blocks) is at most 4·2·30480·8064 < 1.97·10⁹ < 2³¹. It is converted to float once per half, because a Q4_K dot is a difference of nearly cancelling terms and rounding per sub-block hurt accuracy (DD-078).
- **Minimum term:** `−Σ_j (dmin·mn_j)·ms_j`, by FMA in fp32.
- **Output:** `acc += float(int32 sum)·(d·u)`; the 8 lanes are the 8 rows' outputs, so no horizontal sum is needed.

The row kernel `dot_rows_sx_q4_K` uses the same integer scheme on GGUF rows: 8 lanes, then a horizontal sum.

**Error contract.** The only approximation is activation rounding, at most half a step `h` per value (the step is `u·m_b`, or `amax32/127` per-32). Hence `|e| ≤ Σ(h/2)·|w| + (K+5)·u·S` (the fp32 term covers the float conversions and min/scale products). The reference paths measured err/bound 0.11–0.25.

**Reuse decision.**

- **Reused unchanged:** the activation quantizer `quantize_act_sx` (from the AVX2 `CpuKernels` table, a pure function), the `BlockQ4_Kx8` format and `repack_q4_K_x8`, and `get_scale_min_k4`. Reusing them gives identical activation codes and an identical weight layout, so any difference is the dot kernel alone, and outputs are comparable bit for bit.
- **Written independently:** the dot kernel, specialized at compile time for M (1–8, all rows in one weight pass, with no 4-row split), N and K. Weight unpacking is shared by all M rows. It keeps the same int16/int32 overflow invariants and one float conversion per half super-block.

## Generated INT8 kernel: contract (prototype)

- **Shape:** `y[M,N] = x[M,K]·w[N,K]ᵀ (+ bias)`, with M 1–8, N a multiple of 8, K a multiple of 256, all fixed per specialization.
- **Weights:** Q4_K, used through a `BlockQ4_Kx8` copy owned by the backend (`GeneratedInt8Backend::prepack_weight`, the same validation and fingerprint rules as the reference). An op whose weight was not prepacked is rejected, never silently run another way.
- **Activations:** sub-scaled int8, quantized per call with `quantize_act_sx`. The op's contract must allow reduced precision (`auto` or `int8`); `f32` is rejected.
- **Accumulation:** int16 inside a sub-block, int32 over half a super-block, one fp32 conversion per half, then the fp32 minimum term. This is the same error bound as the sub-scaled reference path.
- **ISA:** AVX2 (`maddubs`/`madd`). The unit is compiled with the per-file AVX2 flags and runs only when `select_isa()` picked AVX2. On the generic tier the backend refuses (`kUnsupported`), and callers keep the reference backend. There is no scalar int8 variant in this prototype.

## Generated INT8 kernel: results (prototype)

**Code.**

- Generator: `generate_q4k_int8_source` in `dynacore/src/dynakernel/codegen.cpp`.
- Backend: `GeneratedInt8Backend` in `generated_backend.{h,cpp}`.
- Units: `dk_generated_<unit>_int8_avx2.cpp`, the only int8 generated files, compiled with `engine_set_isa(AVX2)`.
- Tests: `dynacore/tests/test_dynakernel_int8gen.cpp`, with shared reference helpers in `dynacore/tests/q4k_reference.h`.
- Benchmark: `bench_dynakernel_q4k --int8-only`.

**Tests.** 40 specializations: M 1–8 × {8×256, 16×768, 24×1280, 24×1280+bias, 64×2304}, each with edge blocks and structured activations.

- **Bound:** every element is within the sub-scaled bound of the independent double-precision decode. err/bound ≤ 0.184 (the reference sub-scaled paths reach the same maximum), max rel 0.016.
- **Bits:** 0 of 35 bit-identical to `ref_int8_pk`, because the float accumulation order differs; it converts once per half super-block where the reference sums both halves first. 0 of 35 identical to `ref_int8` rows either.
- **Execution proof:** the call counter moves once per group chunk, and a stubbed table entry is what executes.
- **Tier guard:** on the generic tier the backend is refused (`kUnsupported`), nothing is written, and zero generated calls are made.
- **Rejections:** `f32` contract, M 9, N 33, missing shape, Q6_K, weight not prepacked, stale source, released copy, and prepack validation.
- **Other:** strided x/y, bias and the explicit `int8` contract; 1 and 4 threads give identical bits.

**Benchmark** (i7-1255U, MSVC 19.44 Release, 2026-10-10). `--int8-only`: 15 interleaved samples of about 20 ms after 3 warm-up rounds. Every output was verified before timing and reproduced bit for bit after. Two repetitions per configuration agreed.

- **Configurations:** 1 thread pinned (median spread 10%); 10 threads unpinned (median 15%).
- **Speedup** = reference p50 ÷ gen_int8 p50, same synthetic weights and inputs:

| M | vs ref_int8_pk (same codes, same layout, same numerics), 1 thread | 10 threads | vs ref_int8 (GGUF rows) |
|---|---|---|---|
| 1 | — (production uses per-32 at M = 1) | — | 3.9–4.4× (1 t), 2.3–2.6× (10 t); **different activation format** |
| 2 | 1.02–1.10 | 0.93–1.05 | 1.75–3.35 |
| 4 | 1.07–1.14 | 0.97–1.03 | 1.56–2.44 |
| 8 | 1.13–1.27 | 1.07–1.14 | 1.81–2.77 |

**Reading.**

- At M = 8 the generated kernel is consistently faster than the hand-written packed kernel: 13–27% at 1 thread, 7–14% at 10 threads. The likely cause is one weight pass for all 8 rows, where `CpuDevice` runs two 4-row passes (`kDotRowsMax`); that is inferred, not separately measured.
- At M = 2–4 it is at parity at 10 threads and 2–14% faster at 1 thread.
- At M = 1 the gain over `ref_int8` mixes layout and activation format: sub-scaled vs per-32. Production deliberately keeps one row on per-32, and model-level accuracy (DD-053 perplexity/KL) for sub-scaled single-row decode has not been evaluated, so this is not a like-for-like win.
- All shapes fit in the 12 MB LLC, and the benchmark re-runs the same weights, so results are cache-resident. Real decode streams weights from DRAM, where these gaps would likely shrink; not measured.

### DRAM-sized validation (`bench_dynakernel_q4k --dram`)

A separate configuration; the default cases are unchanged. It compares `ref_int8_pk` and `gen_int8` at M = 2, 4, 8 on the three bench shapes.

**Cache control.**

- Each implementation gets R distinct weight copies, so its packed working set is at least max(4 × LLC, 64 MB). That is 48 copies (64 MB) for 1536×1536 and 9 copies (70 MB) for 8960×1536 and 1536×8960, against a 12 MB LLC.
- Every timed call uses the next copy round-robin, so a copy is touched again only after more than 3 × LLC of other weights has streamed through.
- Each implementation reads its own packed copies: the device's for the reference, the backend's own for the generated kernel. The activations (a few KB) stay cached, as in decode.

**Evidence that it works.** In the same interleaved rounds, a "hot" control runs each kernel on copy 0 only. Every row also reports weight GB/s as a share of a 256 MB streaming-read probe run on the same pool: 16.0–16.3 GB/s at 1 thread, 22.6–22.8 GB/s at 10 threads.

- Hot rows reach 121–248% of the probe at 10 threads M = 2–4, so they are served from cache.
- DRAM rows stay at or below 92% of it.

**Method.** Same as the default mode: every copy's output is verified against the double-precision reference within the sub-scaled bound before timing, and the first and last copy are re-checked bit for bit after timing. There are 3 warm-up rounds and 15 samples of about 20 ms (whole rotations). Two repetitions per configuration were run: 1 thread pinned, 10 threads unpinned.

**Results** (2026-10-10, i7-1255U). Speedup = `ref_int8_pk` p50 ÷ `gen_int8` p50, for both repetitions. err/bound ≤ 0.139 for both kernels, identical per case.

| M | 10 threads: speedup | 10 threads: weight GB/s (% of probe) | 1 thread: speedup | 1 thread: % of probe |
|---|---|---|---|---|
| 2 | 0.98–1.09 | 18–21 (78–92%) | 0.99–1.09 | 75–83% |
| 4 | 0.95–1.09 | 17–20 (74–89%) | 1.04–1.12 | 57–64% |
| 8 | 1.07–1.14 | 12–16 (52–72%) | 1.09–1.34 | 33–44% |

Run-to-run medians agreed within about 10%. Most per-row p10–p90 spreads were 10–30%; the shortest call (M = 2, 1536×1536, about 75 µs at 10 threads) spread 99–115%, so it is noise-level.

**Reading.**

- M = 2 and 4 at 10 threads are bandwidth-bound: both kernels are near the measured read bandwidth and at parity. The cache-resident M = 2–4 gains do not carry over to DRAM.
- M = 8 stays compute-bound (52–72% of bandwidth), and the generated kernel keeps a 7–14% (10 threads) and 9–34% (1 thread) advantage.

**Limits.**

- There are no hardware cache-miss counters on this Windows setup. DRAM residency is inferred from the rotation size, the hot/DRAM contrast and the bandwidth ratio, not measured per access.
- The probe is a plain summing read, not a hardware peak.
- Hardware prefetching is active, as it would be in decode.
- One machine, three shapes, synthetic weights, no model run.

## Experimental 8-row packed INT8 kernel (hand-written, opt-in)

**Question.** Does the generated kernel's M = 8 gain come from code generation, or from the hand-written path's 4-row tiling?

**Change**, off by default:

- `KernelPlan::packed_int8_8rows` (env `DYNACORE_PACKED_8ROWS=1`; `RunOptions::experimental_packed_8rows` for DynaKernel runs).
- With it on, `CpuDevice::int8_rows` sends more than 4 rows on an interleaved Q4_K copy through `CpuKernels::dot_q8_sx_x8_q4_K_rows8`. That is a new entry point in `kernels_avx2.cpp` instantiating the existing `dot_x8_sx_q4_K_m<M>` template for M up to 8 (`kPackedRowsMax`), in 8-row tiles plus a tail.
- The existing 4-row entry, its dispatch, the packing format and activation quantization are unchanged.
- The AVX-VNNI registration clears the 8-row entry (there is no VNNI variant), so `DYNACORE_VNNI=1` keeps the 4-row VNNI path.
- `matmul_gated` (fused gate/up) still uses 4-row tiles.
- Evidence counter: `CpuDevice::experimental_rows8_dispatches()`.

**Correctness** (`dynacore/tests/test_packed_8rows.cpp`). Rows never interact in the kernel, so 8-row results must equal the 4-row ones bit for bit, and they do in every case:

- M 1–8 on 4 shapes with bias (32 cases);
- strided x/y with M = 7;
- M = 12 and 16 tiles on the opt-in packed prefill path;
- grouped `matmul_many`.

Also checked:

- **Bounds:** both are within the sub-scaled bound of the independent decode (err/bound ≤ 0.300, identical).
- **Dispatch:** the 8-row path runs only when enabled and M > 4. Odd N (12, 33) is not packable and falls back. VNNI stands down (0 dispatches).
- **Generic tier:** the flag changes nothing.
- **Env opt-in:** `DYNACORE_PACKED_8ROWS=1` works.

**DRAM benchmark** (`bench_dynakernel_q4k --dram`, which adds `ref_int8_pk8` with its own packed copies). Same method and copies as before: 2 repetitions × {1 thread pinned, 10 threads unpinned}; median spread 11% and 15%. Speedup = `ref_int8_pk` p50 ÷ variant p50:

| M | ref_int8_pk8, 10 t | gen_int8, 10 t | ref_int8_pk8, 1 t | gen_int8, 1 t | 8-row dispatches |
|---|---|---|---|---|---|
| 2 | 0.99–1.11 | 0.98–1.06 | 0.97–1.09 | 1.01–1.09 | 0 (falls back) |
| 4 | 0.95–1.09 | 0.96–1.09 | 0.97–1.02 | 1.03–1.08 | 0 (falls back) |
| 8 | 1.08–1.12 | 1.08–1.16 | 1.04–1.19 | 1.11–1.32 | > 0 (all timed calls) |

At M = 8 with 10 threads the 8-row hand-written kernel matches the generated one (pk8 ÷ gen 0.97–1.04). With 1 thread the generated kernel stays 6–12% ahead. M = 2/4 are bandwidth-bound (77–93% of the read probe at 10 threads) and at parity.

**Conclusion.** Most of the generated kernel's multi-threaded M = 8 advantage is the single weight pass, which the hand-written kernel gets from this small change. The remaining single-thread gap is attributed to the generated kernel's compile-time K / one conversion per half super-block, not separately measured.

**Status.** Experimental, off by default. Production use would need end-to-end validation: M ≥ 5 happens only with 5+ concurrent decode streams, and `matmul_gated` is not covered.

## LLVM native backend (prototype, opt-in)

DynaKernel programs compiled to native code by LLVM at run time. Files: `dynakernel/llvm_jit.h` (IR emission, runtime, JIT), `dynakernel/llvm_backend.h` (`LlvmInt8Backend`), `src/dynakernel/llvm_ir.cpp` (in `dynacore`, text only), `src/dynakernel/llvm_jit.cpp` and `llvm_backend.cpp` (library `dynakernel_llvm`, linked only by `test_dynakernel_llvm` and `bench_dynakernel_llvm`). Nothing in DynaCore's dispatch or DynaLM constructs it.

**Pipeline.** `LlvmInt8Backend::prepare(program)`:

1. `verify()` + `check()`: matmul, Q4_K weight, contract auto/int8, f32 accumulation, M 1..8, K % 256 == 0, N % 8 == 0.
2. Each op becomes a `Q4KMatmulSpec`, and `emit_q4k_int8_llvm_ir` writes textual LLVM IR for it.
3. `LLVMParseIRInContext`, `LLVMVerifyModule`, then `LLVMRunPasses("default<O3>")` with a target machine for the host CPU and features.
4. ORC LLJIT compiles the module (one resource tracker per module, so a failed lookup removes everything the module added).

`run()` prepares what is missing and then executes through `GeneratedInt8Backend`, because the compiled function has the generated INT8 kernels' ABI. Weight packing (BlockQ4_Kx8 copies), `quantize_act_sx`, the 8-row-group work split, binding validation and packed-source fingerprints are therefore the existing tested code. Only the dot-product kernel is LLVM-generated.

**Dependency.** LLVM's C API library (`LLVM-C.dll`; `libLLVM.so` elsewhere, untested) is loaded at run time from `DYNACORE_LLVM_C` or the loader path. About 50 entry points are declared locally, so the build needs no LLVM headers or libraries and is unchanged without LLVM. Requirements: LLVM ≥ 17 (opaque-pointer IR), an x86-64 host, and every entry point present. LLVM 21 removed `LLVMOrcThreadSafeContextGetContext`, so it would be reported unavailable. Tested with the official `clang+llvm-20.1.8-x86_64-pc-windows-msvc.tar.xz` (sha256 `f229769f…b107f`, from which only `bin/LLVM-C.dll` is needed). Without the library every entry point returns `kUnsupported` and callers keep `ReferenceBackend`. The library is never unloaded.

**Lowerings** (`LlvmLowering`). The integer dot product is written three ways:

| lowering | IR | target | selected instructions (M=4, per super-block) |
|---|---|---|---|
| `portable` | zext/sext, `mul`, shuffles, `add` | any x86-64 | `vpmullw` 544, `vpmaddwd` 32 (LLVM does not recover `vpmaddubsw`) |
| `avx2` | `llvm.x86.avx2.pmadd.ub.sw`, `pmadd.wd` | host, `-avxvnni` | `vpmaddubsw` 256, `vpmaddwd` 32 |
| `avxvnni` | `llvm.x86.avx512.vpdpbusd.256` | host | `vpdpbusd` 256 (VEX), `vpmulld` |

Instruction counts come from the generated assembly (`kernel_assembly`), not from assumptions. Compiled for the x86-64 baseline, `portable` uses xmm only (no ymm/zmm or VEX multiply-adds), and `llvm.fma` becomes `fmaf` calls resolved from the process. All lowerings keep the C++ kernel's fp32 sequence (explicit `llvm.fma`, no fast-math or contraction) and form the same exact int32 half-super-block sums. Results are bit-identical to the generated C++ INT8 kernel: 40/40 test specs per lowering, plus the SSE2 baseline build.

**ISA safety.**

- Targets follow `select_isa`. Under the AVX2 tier the backend uses the host CPU/features with `-avx512f`. On any other tier it uses `x86-64` with no extra features, and `LlvmInt8Backend` refuses on non-AVX2 tiers anyway (the sub-scaled quantizer is an AVX2 kernel).
- `avxvnni` additionally needs `cpu_info().avx_vnni` and LLVM's `+avxvnni`.
- LLVM *terminates the process* when IR uses intrinsics its target lacks. Found by running the test with `DYNACORE_ISA=generic`: `LLVM ERROR: Do not know how to split the result of this operator!`. `llvm_lowering_compatible` therefore refuses such IR before LLVM sees it, and `compile_kernel` / `kernel_assembly` apply that check.

**Tests** (`test_dynakernel_llvm`; `DK_TEST_EXPECT_LLVM=1|0` asserts that LLVM is present or absent):

- IR determinism and rejections.
- LLVM syntax and verifier errors as statuses.
- Rollback of a failed lookup; duplicate symbols refused.
- Incompatible lowerings refused.
- Instruction-set inspection.
- Bit-exactness against the generated kernel and the double-precision bound for 40 specs × 3 lowerings.
- Direct call of the SSE2 build.
- Contract rejections (f32, M 9, N 33, q6_K, unpacked, stale weight).
- Native call counter.
- Strided x/y with bias; 1 == 4 threads.
- 20 create/compile/run/destroy cycles: live JITs back to 0, private bytes +0.0–0.5 MB.

Results (2026-10-11):

| configuration | checks |
|---|---|
| AVX2 tier with LLVM | 15,889 |
| `DYNACORE_ISA=generic` with LLVM | 216 |
| no LLVM | 197 |
| no LLVM + generic | 197 |

**Compile cost** (once per kernel, before any timed run; `bench_dynakernel_llvm`):

| M | portable | avx2 | avxvnni |
|---|---|---|---|
| 1 | 25–42 ms | 20–25 ms | 16–21 ms |
| 4 | 85–109 ms | 54–66 ms | 36–48 ms |
| 8 | 178–225 ms | 114–140 ms | 79–101 ms |

Optimization (O3) and code generation take about equal shares; parsing and verification together take at most 7.4 ms. One lowering for 4 projection shapes × M 1..8 is roughly 2–3 s of compilation per process.

**Microbenchmark.** `bench_dynakernel_llvm`, i7-1255U (2 P + 8 E cores), Qwen2.5-1.5B projection shapes, weights cache-resident (≤ 7.7 MB), 15–21 interleaved samples. Every output was checked against the double-precision decode before and after timing, and LLVM == generated C++ bit for bit wherever both exist. Raw data: `results/dynakernel-llvm/bench-20261011-084712.*` (unpinned, 1 and 10 threads) and `bench-pinned-20261011-084837.*` (`DYNACORE_PIN_THREADS=1`, 1 thread).

Single thread, p50 speedup of `llvm_avx2` (range over the 3 shapes, both runs):

| M | vs DynaCore packed (`ref_pk`) | vs generated C++ (same algorithm, MSVC) | `llvm_avxvnni` vs `ref_pk` |
|---|---|---|---|
| 2 | 1.10–1.37× | 1.04–1.25× | 1.18–1.38× |
| 4 | 1.09–1.36× | 1.00–1.15× | 1.19–1.39× |
| 5 | 1.07–1.26× | (no C++ kernel) | 1.17–1.43× |
| 8 | 1.14–1.24× | 1.05–1.17× | 1.11–1.27× |

At M 5–8, `ref_pk` is the experimental 8-row path; at M 2–4 it is the production packed path. DynaCore's own opt-in VNNI kernel (`ref_pk_vnni`, DD-080) measured 0.87–0.98× of `ref_pk`, while LLVM's VNNI lowering beats LLVM's AVX2 lowering at M 4–5 on all three shapes in both runs (mixed at M 2 and 8).

M = 1: all sub-scaled packed kernels (`gen_int8` included) are 3.4–4.7× faster than `ref_pk`. That is not an LLVM effect: at M = 1 DynaCore uses per-32 int8 on GGUF rows (sub-scale needs ≥ 2 rows; `q4_repack_min_rows` = 2), a different numerical path.

10 threads (unpinned): `llvm_avx2` vs `ref_pk` is 0.74–0.97× on 1536 × 1536 (M ≥ 2) and 1.02–1.16× on the two larger shapes. Small-shape samples are noisy (p10–p90 spread up to 290 %). The single-thread kernel gain mostly disappears once threading and memory dominate, and `GeneratedInt8Backend` (hence the LLVM path) quantizes activations on one thread.

`portable` runs at 0.48–0.63× of `ref_pk` at M ≥ 2: generic vector IR is not enough, and the explicit intrinsics are what make the LLVM kernels competitive.

**Not shown by this benchmark:** end-to-end inference speed, DRAM-bound behaviour, multi-threaded wins on small shapes, and other CPUs.

### Phase 2: multithread scaling (2026-10-11)

**Method.** `bench_dynakernel_llvm --breakdown [--dram]` times each stage of one call in isolation, with interleaved samples on the same pool:

| component | what it times |
|---|---|
| `ref_pk` | DynaCore packed path |
| `llvm` | `LlvmInt8Backend::run` |
| `exec` | `GeneratedInt8Backend::run` on the same compiled kernel |
| `quant` | serial `quantize_act_sx` over the M rows |
| `kernel` | the compiled kernel alone in one parallel region |
| `region` | an empty region with the same chunking |
| `checks` | the executor's check + `validate_bindings` |
| `prepare` | `LlvmInt8Backend::prepare` |

`--dram` rotates over weight copies (≥ max(4 × LLC, 64 MB) packed). Every configuration ran 3 times. The tables give the median of the per-run p50s and the [min–max] of the per-run ratios. `DYNACORE_PIN_THREADS=1` now also pins the calling thread (`ThreadPool::pin_caller`, P-core 0), which is where all work runs at 1 thread. Raw data: `results/dynakernel-llvm/breakdown-*`, `pq-*` (before the fast path) and `fp-*` (after); aggregation scripts `agg_breakdown.py` and `compare_fastpath.py`.

**Root cause** (pinned, cache-resident, 10 threads, before the fix; µs):

| shape (N × K), M | ref_pk | llvm | exec | quant | kernel | region | prepare |
|---|---|---|---|---|---|---|---|
| 1536 × 1536, 2 | 27.2 | 34.2 | 27.5 | 1.5 | 21.4 | 7.8 | 4.7 |
| 1536 × 1536, 8 | 79.5 | 84.2 | 73.7 | 7.1 | 61.7 | 7.8 | 5.6 |
| 1536 × 8960, 8 | 459.2 | 427.6 | 430.0 | 47.1 | 361.9 | 8.0 | 6.2 |
| 8960 × 1536, 8 | 398.0 | 372.7 | 380.5 | 8.4 | 348.4 | 15.9 | 6.3 |

- **Serial quantization** is 0.4–3 % of a call at 1 thread and 1.4–11 % at 10 threads. It is largest at K = 8960, but it is not the main reason LLVM lost to `ref_pk`, which also quantizes serially (`CpuDevice::matmul`).
- **The main reason** was `LlvmInt8Backend::run` calling `prepare()` on every call: verify, contract checks, a copy and scan of LLVM's feature string, and spec validation that builds strings. That is 4–6 µs, after which `exec` repeats the verify/contract/binding checks (~2 µs). At small 10-thread shapes `llvm` was 1.12–1.28 × `exec`, while `exec` was already at or below `ref_pk`.
- **The kernel itself scales:** kernel-only is 1.16–1.34 × faster than `ref_pk` at 10 threads.
- **A parallel region costs 6.5–16 µs** of fork/join at 10 threads, so an extra region pays only for enough work.

**Change 1 (opt-in experiment): parallel activation quantization.** `RunOptions::experimental_parallel_quant`, with `parallel_quant_min_superblocks` (default 96), applies to `GeneratedInt8Backend` and therefore `LlvmInt8Backend`.

- `quantize_act_sx` processes each 256-value super-block independently. The experiment splits the M × K/256 super-blocks into (row, super-block range) chunks, one parallel region before the kernel's, on the existing pool: no nesting and no new threads.
- Serial when off, at 1 thread, or below the threshold.
- Codes and outputs are identical: 180 test runs bit-identical across 4 shapes (strided rows, chunks spanning rows), 1/4/10 threads, thresholds 0/96/off, and both executors.

Results (`llvm_pq / llvm`, 10 threads, same run). Below the threshold both slots run the serial code, so those 9 points per run set are a control for slot-order bias: in hot mode `llvm_pq` runs right after `llvm` on the same packed weights.

| run set | control: median [range] | K = 8960, M 4 / 5 / 8 (parallel) |
|---|---|---|
| hot, pinned (before / after fast path) | 0.977 [0.93–1.00] / 0.985 [0.95–1.00] | 0.88 / 0.89 / 0.89 and 0.92 / 0.89 / 0.88 |
| hot, unpinned | 0.962 [0.93–0.98] / 0.963 [0.95–1.00] | 0.75 / 0.82 / 0.83 and 0.71 / 0.73 / 0.81 |
| DRAM, pinned | 0.991 [0.97–1.05] / 1.004 [0.97–1.05] | 0.96 / 0.94 / 0.89 and 0.95 / 0.92 / 0.90 |
| DRAM, unpinned | 0.991 [0.98–1.02] | 0.84 / 0.84 / 0.81 |

- Net of the control median, the gain is 4–11 % pinned and 13–26 % unpinned, repeated in both independent run sets.
- Forcing it at every size (threshold 0) is mixed at K = 1536 (0.83–1.04 at M 2–5). At K = 8960 M = 2 (70 super-blocks) it gives 0.86–0.94, so a threshold near 64 would probably pay off, but that was not the tested default.
- At 1 thread the path is off; the ratios of 0.94–1.03 there are the noise floor.
- **Decision:** a repeatable improvement at 10 threads for large K, with no 1-thread or accuracy cost, so it is kept, still opt-in.

**Change 2: `LlvmInt8Backend::run` fast path.**

- When every op already has a compiled kernel, `run()` goes straight to the executor. The executor still verifies the program, checks contract, tier and table, and validates bindings, and `prepare()`'s extra checks held at compile time.
- A test runs a compiled shape under an f32 contract, and a program that fails verification, through the fast path: both are refused with nothing written.
- `llvm / exec` at small 10-thread shapes went from 1.12–1.28 to 0.93–1.03.

**After both** (median of 3 runs; ratio to `ref_pk`, < 1 = faster):

| 10 threads | llvm (serial quant) | llvm + parallel quant |
|---|---|---|
| hot, pinned: 1536 × 1536 | 0.91–0.99 | 0.90–0.94 |
| hot, pinned: 1536 × 8960 | 0.90–0.94 | 0.80–0.91 |
| hot, pinned: 8960 × 1536 | 0.86–0.89 | 0.85–0.89 |
| hot, unpinned: all shapes | 0.80–0.92 | 0.66–0.89 |
| DRAM, pinned: all shapes | 0.94–1.02 | 0.87–1.03 |

1 thread, pinned: `llvm` 0.76–0.88 hot and 0.77–0.86 DRAM. Unpinned 1-thread runs vary up to ±12 % between runs (the caller lands on P- or E-cores).

With weights in DRAM at 10 threads, every implementation converges to roughly the same time: 8960 × 1536 takes ~380–470 µs for 7.8 MB, about 17–21 GB/s. There the kernel no longer matters, and only less weight traffic (or fewer passes over it) would help.

### Phase 3: memory bandwidth (2026-10-11)

**Methodology audit.**

- `bench_dynakernel_q4k --dram` rotates over distinct random weight copies (≥ max(4 × LLC, 64 MB) packed), with separate packed copies per implementation. It times the full backend call (quantization, checks, region), computes GB/s as packed bytes per call, and compares against a best-of-5 256 MB streaming read probe.
- Phase 2's `--breakdown --dram` rotates the same way over byte-identical copies, so it times stages but not GB/s against a common reference.
- Neither benchmark flushes caches or uses streaming stores. Rotation alone keeps the working set out of the LLC.

**New `bench_dynakernel_llvm --bandwidth`** (kernel only):

- DynaCore's packed kernel (`dot_q8_sx_x8_q4_K`, and `_rows8` for M 5–8, called as `CpuDevice::int8_rows` calls them), the generated C++ kernel and the LLVM kernels run on **one shared set of packed copies** and identical pre-quantized activations, in one parallel region with the executor's grain on the same pool. No quantization, checks, allocation or packing is timed.
- `traverse` reads the same bytes with the same region and chunking but does no compute: the achievable rate for this access pattern.
- Each kernel runs in DRAM mode (rotation) and hot mode (copy 0), interleaved within the same rounds.
- Every copy's output, the hot output and all LLVM variants are compared before and after timing. LLVM variants must equal the control bit for bit; DynaCore, generated and LLVM outputs must be within the double-precision bound.
- **Byte model:** GB/s = packed weight bytes per call (N/8 · K/256 · 1216) / p50. This is an *estimate of application traffic*: activations (cache-resident, ~0.3 KB per 256 values per row) and outputs (M·N·4) are not counted, and `BlockQ4_Kx8` is 5.6 % larger than GGUF Q4_K.
- **No hardware counters:** `PerfCounters` uses Linux `perf_event_open` only, so actual DRAM traffic and cache misses are not measured.
- The read probe is the median of 10 passes over 256 MB.
- Raw data: `results/dynakernel-llvm/bw-*`, aggregated by `agg_bandwidth.py` and `agg_candidates.py`.

**Baseline** (5 runs alternating pinned and unpinned; pinned medians shown). Read probe: 15.9 GB/s at 1 thread, 22.1 GB/s at 10 threads.

| 10 threads, DRAM | DynaCore | generated | LLVM | traverse | LLVM / traverse |
|---|---|---|---|---|---|
| M 2 (3 shapes) | 20.3–21.0 GB/s | 20.4–21.4 | 20.1–20.6 | 20.7–21.0 | 0.97–0.99 |
| M 4 | 20.1–20.2 | 20.1–20.5 | 19.4–19.9 | 20.7–21.4 | 0.91–0.95 |
| M 5 | 19.3–20.5 | – | 18.9–19.2 | 20.5–21.5 | 0.89–0.92 |
| M 8 | 16.0–16.3 | 15.8–16.9 | 15.8–16.6 | 20.6–20.9 | 0.77–0.80 |

- **M ≤ 5 at 10 threads is bandwidth-saturated** for all three kernels: 84–93 % of the read probe, ≥ 0.89 of traversal. LLVM's cache-resident advantage does not survive: it is 1–6 % below DynaCore at M 4–5.
- **M = 8 at 10 threads is compute-bound.** The hot kernel time (415–450 µs for 7.8 MB) is above the DRAM traversal time (~375 µs).
- **1 thread:** M = 2 reaches the probe rate. M ≥ 4 is compute-bound; there LLVM moves 13–20 % more GB/s than DynaCore.

**Why M = 8 is compute-bound.**

| | P-core, 1 thread | E-core, 1 thread |
|---|---|---|
| LLVM avx2, M = 8, hot (8960 × 1536) | 1077 µs | 3103 µs |
| VNNI lowering vs AVX2 | 0.80–0.94 (faster) | 1.11–1.21 (slower) |

- Effective clocks from Windows counters: P-core 4.4 GHz at 1 thread, but 3.0 GHz (P) and 2.3 GHz (E) with 10 threads busy (15 W part).
- With the work spread over P- and E-cores under that power limit, compute per weight byte, not DRAM, sets the M = 8 time.

**Candidates** (opt-in, one factor at a time, compared with the LLVM control in the same run, < 1 = faster; pinned, 3 runs, 12 points per cell, median [range]):

| candidate | 1 thread DRAM | 1 thread hot | 10 threads DRAM | 10 threads hot |
|---|---|---|---|---|
| prefetch +1216 B (`LlvmKernelOptions::prefetch_bytes`) | 0.954 [0.85–0.98] | 1.011 [0.99–1.03] | 0.973 [0.92–1.04] | 1.014 [0.97–1.05] |
| prefetch +2432 B | 0.947 [0.84–0.97] | 1.016 [1.00–1.07] | 0.997 [0.94–1.02] | 1.052 [1.02–1.12] |
| prefetch +4864 B | 0.955 [0.83–0.99] | 1.021 | 1.014 [0.96–1.08] | 1.061 |
| prefetch +9728 B | 1.060 | 1.053 | 1.068 | 1.095 |
| 8 chunks per thread (default 32) | 1.003 | 0.999 | 1.002 [0.97–1.07] | 1.041 |
| 128 chunks per thread | 1.005 | 1.000 | 0.997 [0.93–1.09] | 1.024 |
| 2 chunks per thread (contiguous) | 1.004 | 1.000 | 1.080 [1.01–1.18] | 1.231 [1.10–1.29] |
| VNNI lowering (M 4/5/8) | 0.939 [0.90–0.96] | 0.953 [0.84–1.02] | 1.024 [0.97–1.08] | 1.068 [1.01–1.15] |

- **Prefetch:** a repeatable gain only for single-thread DRAM, where one core's outstanding misses limit it (it even exceeds the scalar 1-thread read probe). There is no repeatable gain at 10 threads, where the stream is already saturated, and it costs 1–10 % cache-resident. **Not accepted**; it stays an opt-in option (default 0: IR and symbols unchanged, bit-identical when enabled).
- **Chunk ordering and partitioning:** the default (32 chunks per thread, dynamic claiming) is as good as or better than the alternatives. Contiguous per-thread ranges lose 8–23 % to imbalance between P- and E-cores. **Not accepted.**
- **Temporary buffers and memory movement:** the kernel path has none to remove. The executor reuses its activation buffer and writes outputs in place.
- **Synchronization:** one region costs 6.5–16 µs (Phase 2). That is ~10 % of a 1536 × 1536 DRAM call at 10 threads and 2–3 % of the 7.8 MB shapes. Cutting it means grouping several matmuls into one region, not a kernel change.

**Conclusion.** On this machine the LLVM kernel is not leaving DRAM bandwidth unused at 10 threads for M ≤ 5, and at M = 8 the limit is power-limited compute on hybrid cores. Next: fewer bytes per weight in the packed layout. `BlockQ4_Kx8` stores d/dmin as fp32 and the scales and mins as whole bytes: 1216 B per 8 × 256 values against GGUF's 1152 B.

### Phase 4: compact packed layouts (2026-10-11)

**Layouts** (`dynakernel/compact_layout.h`; opt-in through `LlvmKernelOptions::layout`, which also selects the layout `GeneratedInt8Backend::prepack_weight` builds):

| layout | d, dmin | scales / mins | bytes per 8 × 256 | vs `x8` |
|---|---|---|---|---|
| `x8` | fp32 | unpacked bytes | 1216 | (BlockQ4_Kx8, unchanged default) |
| `h` | fp16 | unpacked bytes | 1184 | −2.6 % |
| `s` | fp32 | GGUF 6-bit `[12][8]` | 1184 | −2.6 % |
| `hs` | fp16 | GGUF 6-bit | 1152 | −5.3 % |

- Every layout stores exactly the `x8` values. fp16 d/dmin are GGUF's own values and are widened exactly by `fpext <8 x half>` (vcvtph2ps). The 6-bit fields decode with `get_scale_min_k4`'s arithmetic on 8 rows at once. So results are bit-identical.
- The fp16 layouts need F16C (`llvm_kernel_compatible`, `LlvmInt8Backend::runnable`).
- The generated C++ kernels read `x8` only.

**Correctness** (`test_dynakernel_llvm`, now 577,698 checks):

- 540,672 element decodes per layout are bit-identical to `BlockQ4_Kx8`, including d = 0, saturated 6-bit scales/mins, and fp16 subnormal / −0 / max scales.
- 360 layout × lowering × spec backend runs (all 40 INT8 test specs, portable/avx2/avxvnni) are bit-identical to `x8` and within the double-precision bound.
- fp16 layouts are refused without F16C, while `s` compiles for the x86-64 baseline.
- The layout is fixed once weights are packed; stale-weight detection and release still work.

**Benchmark.** `bench_dynakernel_llvm --layouts x8,h,s [--calls]`. Each layout has its own DRAM-sized packed copies (≥ 64 MB) and a cache-resident copy, measured as:

- `k_*`: kernel only;
- `trav_*`: traversal of the layout's bytes;
- `call_*`: full `LlvmInt8Backend::run`, including quantization and checks.

`x8` is the control in every run. Raw data: `results/dynakernel-llvm/lay-hs1-*` (x8, h, s; 5 runs alternating pinned and unpinned) and `lay-h2-*` (x8, h; 4 runs); aggregated by `agg_layouts.py`. Ratios are time / x8 in the same run, median over 12 points [range of point medians]; "slower in all runs" counts points where every run was slower.

| | 1 thread, DRAM | 1 thread, hot | 10 threads, DRAM | 10 threads, hot |
|---|---|---|---|---|
| `h`, full call (9 runs) | 0.971 [0.94–0.98] | 0.990 [0.98–1.02] | 0.967 [0.945–0.991] | 1.001 [0.97–1.03] |
| `h`, kernel (9 runs) | 0.967 [0.95–1.00] | 0.991 | 0.965 [0.94–1.00] | 0.997 [0.98–1.01] |
| `s`, full call (5 runs) | 0.982 [0.95–1.04] | 1.025 [0.99–1.06], 6/12 slower in all runs | 0.976 [0.92–1.02] | 1.041 [0.99–1.08], 9/12 slower in all runs |
| `s`, kernel (5 runs) | 0.998 [0.87–1.15] | 1.034 | 0.978 [0.97–1.04] | 1.051 [1.01–1.10], 8/12 slower in all runs |
| traversal, h and s | 0.96–0.97 | – | 0.96–0.97 | – |

- **Packing** (one full matrix, single thread): `h` 0.20 / 1.33 / 1.83 ms and `s` 0.19 / 1.29 / 1.64 ms, against `x8` 0.21 / 1.42 / 1.81 ms (1536², 1536 × 8960, 8960 × 1536). That is equal or faster, being fewer bytes with no fp16 → fp32 conversion.
- **Compile time:** unchanged (60–150 ms per kernel).

**Decision.**

- `h` is **retained (opt-in)** for **DRAM-resident weights** (working set ≫ LLC) at M 2–8, 1 or 10 threads. It gives 3–4 % per call and kernel, about the byte saving. The gain holds at 10 threads (M 2–5 bandwidth-bound points 0.92–1.00), is ≈ neutral at M = 8 (compute-bound), and shows no repeatable regression cache-resident (no point slower in all 9 runs).
- `s` is **not retained**: unpacking the 6-bit fields costs 3–5 % cache-resident, more than its DRAM gain.
- `hs` was **not tested**, because `s` was not promising on its own.

**MLIR.** Not used. It would be justified once DynaKernel needs transformations over loop nests rather than one fixed kernel template: tiling and packing search across shapes and cache levels, fusion of quantize → matmul → bias/activation chains, or retargeting one description to AVX-512/AMX/NEON/GPU. The LLVM C API cannot express those structurally, and this prototype emits the loop nest directly.

## Phase 6: prefill-sized matmuls (M ≈ 33–40) without accuracy regression (2026-10-12)

**Execution path (from the source).**

- The scheduler (`dynalm/src/scheduler/scheduler.h`: `max_prefill_chunk = 32`, `prefill_token_budget = 64`) batches a prefill chunk with the decode rows into one forward pass, and every dense projection is one `CpuDevice::matmul`.
- Above 8 rows, `int8_kernel` returns nothing, because `q4_repack_max_rows = 0` keeps the DD-078 int8 prefill path off: it fails DD-053 on Qwen3-4B.
- The int16 branch is gated by `int16_decode_max_rows`, which only FFN-down sets, and only to 4.
- So a 33–40-row matmul runs fp32 expand: dequantize 4 weight rows to fp32, then a 4×3-tile FMA GEMM (`gemm_panel`) with K-blocks of 1024.

**DD-053 gate** (CHANGELOG, P3: perplexity within +1 %, mean KL ≤ 0.0025 against fp32). The prefill form is `BENCH_PREFILL_ONLY=1 bench_int8_accuracy <model> [threads]`: 32- and 64-token chunks, the default `KernelPlan` against an fp32 plan. The fp32 reference now also switches off `int16_prefill_max_rows` and `q4_repack_max_rows`, so an alternative path can never become the reference.

**Baseline** (`bench_decode_matmul`, real weights cycled through all layers, 10 threads; ms):

| shape | expand | packed int8 (DD-078, fails DD-053) | int16, existing 4-row tiles |
|---|---|---|---|
| Qwen2.5-1.5B ffn_gate 8960 × 1536, M 33–40 | 2.83–3.41 | 0.69× expand | 2.75–2.81× expand |
| Qwen2.5-1.5B ffn_down Q6_K 1536 × 8960 | 2.88–3.49 | 1.36–1.40× | 2.24–2.54× |
| Qwen3-4B ffn_gate 9728 × 2560 | 6.45–8.11 | 0.64× | — |

fp32 expand reaches ~325 GFLOP/s on ffn_gate. That is roughly two thirds of an estimated fp32 peak at the clocks measured under load (P 3.0 GHz, E 2.3 GHz). So expand has at most ~25–30 % headroom, and even int8 is only ~1.5× faster on Q4_K.

**Candidate (opt-in): int16 activations in 8-row tiles.**

- `KernelPlan::int16_prefill_max_rows` (default 0; `DYNACORE_INT16_PREFILL_ROWS`) applies to matmuls of 9..N rows on a packed Q4_K weight (Q6_K only with `DYNACORE_Q6_PACKED=1`) with N ≥ `q4_repack_min_out` and N·K ≥ `int16_prefill_min_weights` (default 4 M elements).
- They use `quantize_act16` (the DD-076 accuracy class) and the existing int16 x8 kernels in 8-row tiles (`dot_q16_x8_q{4,6}_K_rows8`): half the weight unpacking of the 4-row tiles, same per-row arithmetic.
- On AVX-VNNI CPUs the tiles use `vpdpwssd` instead of `madd_epi16` + `add_epi32`. The integer results are exact, so the output is bit-identical. `DYNACORE_INT16_PREFILL_VNNI=0` keeps the AVX2 tiles.
- `/metrics`: `dynacore_int16_prefill_max_rows`, `dynacore_int16_prefill_matmuls_total`.

**Accuracy (DD-053 prefill gate):**

| configuration | Qwen2.5-1.5B ppl / mean KL | Qwen3-4B ppl / mean KL |
|---|---|---|
| default | +0.000 % / 0 | +0.000 % / 0 |
| int16 prefill, Q4_K + Q6_K | −0.003 % / 0.000000 | +0.013 % / 0.000001 |
| int16 prefill, recommended (Q4_K, size gate) | +0.003 % / 0.000000 | −0.012 % / 0.000001 |
| int8 prefill (DD-078), for reference | +0.172 % / 0.000361 | **+3.358 % / 0.001858 (fails)** |

These are 32-token chunks; 64-token chunks agree (int16 at most +0.037 %; int8 +2.93 % on Qwen3-4B).

**Speed** (time ÷ expand in the same run; between-run drift reached 30 % from heat soak, so only same-run ratios are used):

| shape (M 33–40) | 4-row int16 | 8-row AVX2 | 8-row VNNI (3–5 runs, medians) |
|---|---|---|---|
| Qwen2.5-1.5B ffn_gate/up | 2.8× | 0.92–1.08 | 0.962–0.966, every run faster |
| Qwen2.5-1.5B ffn_down Q4_K | 2.9× | 0.98–1.08 | 0.931–0.954 |
| Qwen2.5-1.5B attn_q 1536 × 1536 | 2.6× | 0.98–1.44 | 1.03–1.09 (slower; below the size gate) |
| Qwen3-4B attn_q / attn_out / ffn_gate / ffn_down Q4_K | — | 0.95–1.07 | 0.90–0.96, every run faster (12/12) |
| Qwen3-4B attn_k 1024 × 2560 | — | 1.02–1.25 | 1.00–1.04 (below the size gate) |
| every Q6_K matrix | 2.2–3.4× | 1.16–1.42 | 1.07–1.44 (slower; keep `DYNACORE_Q6_PACKED` off) |

The fp32 tuning knobs (`DYNACORE_GEMM_KC`, `DYNACORE_MATMUL_CHUNKS`) gave no usable signal under that drift; the defaults show no sign of being wrong. A rewritten fp32 micro-kernel was not attempted.

**Projected effect** (M = 36, gated policy, per-layer matmul time from the medians): about −3 % of a prefill step's matmul time on Qwen2.5-1.5B and about −5 % on Qwen3-4B. The repeatable kernel gain is real (4–10 % on the large Q4_K matrices) but far smaller than the TTFT goal (−30 % to reach llama.cpp). The remaining gap is the activation precision class: llama.cpp prefills with int8 activations, which DD-053 rejects for DynaLM on Qwen3-4B.

**Tests.** `test_int16_prefill` (320,146 checks):

- bit-identical to the 4-row int16 path at 9..64 rows (Q4_K, Q6_K);
- VNNI tiles == AVX2 tiles at 1–8 rows;
- within an int16 activation bound of the exact Q4_K decode;
- strided x/y with bias; 1 == 4 threads;
- gating: ≤ 8 rows, N < 1024, unpacked weights, above the row cap, below the size gate;
- the dispatch counter;
- on the generic tier it stands down.

All existing DynaCore/DynaKernel suites pass, and `test_dynakernel_quant` passes with the knob forced on.

## Concurrency and ownership

- `ReferenceBackend` owns its `CpuDevice`; the `ThreadPool` is borrowed and must outlive it.
- `run()` is not thread-safe (device scratch).
- `ThreadPool` runs one parallel region at a time. Before this change, a second region (fn calling `parallel_for` on its own pool, or another thread calling it) overwrote the in-flight job, guarded only by an `assert`. `test_thread_pool` with the guard removed shows the corruption: indices not visited exactly once.
- Now an atomic ownership flag (`in_region_`, claimed by `exchange`) makes the check and the claim one step. A region that finds the pool busy never touches the in-flight job's state and never waits. It runs `fn` serially on its own thread in chunks of at most `grain` (same result) and increments `misuse_count()`.
- Inline regions (`n <= grain`, or a 1-thread pool) touch no shared job state and are unchanged.
- Cost on the normal path: one atomic exchange and one store per region; not benchmarked.
- `ReferenceBackend::run` compares `misuse_count()` before and after. If another region found the pool busy during the run, it returns `kInternal` ("thread pool contract violated …"). The outputs are complete (computed serially), but the shared-pool use is reported, not hidden.
- Still unsupported: a pool shared with a running engine. The guard turns it from a data race into serialized execution plus a reported error.

### Check counts per tier

`test_dynakernel_quant` prints checks per section. AVX2 and generic differ by design:

- `grouped_vs_single` (2681 vs 2680): bitwise grouped == single is asserted only where it must hold. On generic, K=1280 with M=3 is a quantized fp32 panel with K > `gemm_k_block`, where `matmul` (K-sliced) and `matmul_many` differ. Its elements are still bound-checked.
- `prepack_validation` (21 vs 11) and `packed_kernel` (13632 vs 0): there is no interleaved kernel on generic, so the test verifies the rejection and stops.
- Every other section has identical counts.

Graph-layer gaps found while writing the lowering (not changed here):

- `GraphExecutor::run_op` passes f16/bf16 `x` to `CpuDevice::matmul`, which reads `x` as `float*`. Matmul bias dtype is never checked.
- `GraphExecutor::allocate` ignores `Layout` (every value is dense), and `GraphExecutor::bind` checks only dtype and rank.

## Not in v0

Not routed from DynaLM: nothing in the inference path builds or runs DynaKernel programs. There is no optimizer and no tuner. Code generation (generated C++ and the LLVM backend) exists only as opt-in prototypes used by tests and benchmarks. Lowering covers matmul graphs only.

## Where future pieces belong

- **Target capabilities** (ISA, integer-dot support, threads, cache sizes): `TargetDesc` and the backend. The program never names an ISA.
- **Kernel and tile choice, packing, fusion of bias/activation**: backend lowering. The program states only the contract: shapes, layouts, accumulation and activation format.
- **Kernel constraints** (K multiple of 32 or 256, row limits, packed layouts): backend `check()`, as `kUnsupported` with a reason. Shape-specific eligibility stays there, never in the verifier.
- **Measured costs**: a future tuner keys on `(TargetDesc, op contract, M/N/K bucket, weight dtype)`. Only programs that pass `verify()` and `check()` and match the reference backend numerically become candidates.

## Next backend: is MLIR a good fit?

MLIR is a reasonable target for the code-generation stage, with caveats. The matmul contract maps to `linalg.matmul` / `linalg.generic` on memrefs with explicit strides. Tiling, vectorization and lowering to LLVM for AVX2 / AVX-VNNI are existing passes (`transform` dialect, `vector`, `x86vector`). Block-quantized GGUF weights are the hard part: no upstream dialect understands Q4_K super-blocks. Dequantization would be expressed as `linalg.generic` bodies over `i8`/`i4` loads, or as a custom dialect. Measured parity with the hand-written int8 kernels is not guaranteed.

Cost on Windows/MSVC:

- The official LLVM Windows installers do not ship MLIR libraries, so expect a source build (or a vcpkg/conda package, to be verified). A source build of `mlir` + `llvm` (Release, X86 target only) takes tens of minutes and several GB, and must use the same CRT (`/MD`) as DynaCore.
- An AOT path keeps MLIR out of the runtime: a build-time tool emits object files or C for chosen shapes, and DynaCore links them. A JIT (`ExecutionEngine`) would add LLVM to the shipped binary.
- Smaller first step: emit C++ for one matmul shape from a verified program, compile it with the existing toolchain, and compare it against `ReferenceBackend`. This tests the backend boundary before taking on the MLIR dependency.

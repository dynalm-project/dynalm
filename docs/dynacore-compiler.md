# DynaCore compiler

The DynaCore compiler turns recorded IR into an execution plan for a device (DD-072). Today
its backend is the existing C++ kernel library: the compiler decides **which calls** run,
how they are **grouped**, and under **which kernel plan**. Stage 1 of
[the native code generation strategy](platform-design.md) is implemented and measured.
Stages 2 and 3 (generated intrinsic kernels, native code) are not built.
[compiler-backends.md](compiler-backends.md) explains why.

```
DynaLM Transformer
      │  Device ops (unchanged model code)
      ▼
RecordingDevice (deferred)            dynacore/ir/recording_device.h
      │  record calls + structural signature          (~50 µs per decode step)
      ▼
sync point (download / synchronize / copy / allocate)
      │
      ├── plan cache hit ─────────────────────────────┐   (every decode step after the first)
      │                                               │
      ▼ miss                                          │
build IR ─► verify (tests) ─► canonicalize ─►         │
  select_kernels ─► plan_execution (fusion) ─►        │
  call-level plan, cached                             │
      │                                               │
      ▼                                               ▼
execute on the inner device: one call per step (op, matmul group, gated matmul)
      │
      ▼
CpuDevice kernels (generic / AVX2 / NEON)
```

## Using it

| Where | How |
|---|---|
| CLI | `dynalm run qwen3:4b --execution compiled`, `dynalm serve ... --execution compiled`, `dynalm benchmark ... --execution compiled` |
| Config | `runtime.execution: compiled` |
| C++ | `EngineOptions::execution = ExecutionMode::kCompiled` |

The default is `reference`. The engine wraps its device in a `RecordingDevice` with the
compiler pipeline as planner. Nothing else in DynaLM changes, and DynaLM never sees IR.

## Pipeline (`dynacore/ir/passes.h`)

Every pass is a plain function with its own tests.

| Pass | Does | Notes |
|---|---|---|
| `verify` | Full IR check | On in tests (`CompileOptions::verify`), off in production (costs time per miss) |
| `canonicalize` | Rewrites quantized `matmul` → `qmatmul` and `attention` ↔ `gqa` by head grouping | No dead-code elimination: a segment's results (logits) are read by the host after the sync point, which the graph cannot see |
| `analyze_memory` | Activation bytes, peak live bytes, distinct buffers, greedy reuse plan | Analysis only: DynaLM owns its scratch |
| `select_kernels` | Annotates each op with the CPU kernel it will run: `cpu.int8_dot.q4_K` (M ≤ the op's `int8_rows`), `cpu.panel_gemm.*` (M ≥ expand_min_rows), `cpu.fused_dot.*`, `cpu.gqa_grouped` | Uses the per-op `int8_rows` recorded from the kernel plan in effect |
| `plan_execution` | Fusion; see below | Never reorders across a data dependence; never fuses across different kernel plans |

### Fusions

| Fusion | Pattern | Executes as | Why it pays |
|---|---|---|---|
| Shared-input group | Back-to-back matmuls reading the same activation value, none consuming another's result (Q/K/V) | One `matmul_many` region; activations quantized once for all jobs; bias per job | K/V projections of 256 rows are latency-bound (51–53 µs each in isolation for 0.2 MB, 4 GB/s); one region removes two fork/joins |
| Gated MLP | `matmul(x, Wg)`, `matmul(x, Wu)`, `act_mul(gate, up)` with gate/up used only by the act_mul | One `matmul_gated` call: both dot products per output column, the activation applied in the epilogue | Removes the separate act_mul (84 µs per layer in the model: reading values just written by other cores), one region, and the gate/up writes |

Both fusions compute **bit-identical** results. They use the same dot-product kernels and the
same activation function, in the same order per element. The tests require exact equality
of logits for all 11 tiny architectures.

## Plan cache

- A segment's **signature** covers, for every call: method, buffer pointers, shapes, strides,
  dtypes, scalar parameters, the count of positions, sequences and jobs, and the kernel plan's
  content.
- It does **not** cover the values that change every step: positions, token ids, block
  tables.
- Decode steps of one batch size share a signature. After the first step a segment costs one
  signature (64-bit FNV over the words, then an exact word compare). No IR construction and
  no planning happen.
- The cache holds 256 entries. Misses compile.
- Per-step overhead measured on Qwen2.5-1.5B (452 calls per step):
  - recording: 1.25 ms per step, cut to ~50 µs by the cache refactor;
  - planning: 0.3 ms on a miss, 0 on a hit.

## Safety

- **Fallback.** A planner error, an invalid plan or an unknown pattern runs the recorded calls
  in their original order. A valid plan must:
  - cover every call exactly once;
  - give fused steps the right call kinds;
  - run each fused region under one kernel plan.

  `RecordingStats::fallbacks` counts fallbacks; tests require 0.
- **Kernel plans per call.** `set_kernel_plan` is not a sync point. Each call keeps the plan in
  effect when it was recorded, and execution re-applies plans as needed. The FFN down
  projection runs without int8 activations (DD-053) inside a compiled segment exactly as in
  reference mode.
- **Device memory.** The recording device reports `host_accessible() == false`, so the model
  reads results only through `download`, which is a sync point.

## Debugging

| Tool | Shows |
|---|---|
| `bench_op_trace <model> [threads] [steps] --dump step.ir` | IR of a decode step, per-op wall time, time by op kind |
| `bench_compiled <model> ... ref,ir,group,gated,all` | Step-interleaved A/B of the variants, logits accuracy, recording/planning cost, calls per op |
| `dynacorec` | See [dynacore-language.md](dynacore-language.md) |
| `RecordingDevice::set_segment_observer` | Every compiled segment with its plan |

## Results

See [compiler-benchmarks.md](compiler-benchmarks.md).

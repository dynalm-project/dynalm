# DynaCore optimization log

Every compiler optimization follows the same loop: profile → hypothesis → prototype →
correctness → A/B → end to end → keep or revert.

- Results are on an i7-1255U (2 P + 8 E cores, 10 threads) with 16 GB of RAM, using
  Qwen2.5-1.5B-Instruct Q4_K_M unless noted.
- Noise between repetitions is ±5–10% on this laptop. Every comparison below therefore
  interleaves variants, step by step or run by run.

## 1. Where decode time goes (`bench_op_trace`, trace mode)

One decode step at ~256 tokens of context: 452 device calls, 79 ms.

| Op | ms/step | µs/call | Share | Note |
|---|---|---|---|---|
| ffn gate/up `q4_K[8960×1536]` | 26.6 | 475 | 36% | 16 GB/s, near DRAM ceiling |
| lm_head `q6_K[151936×1536]` | 10.7 | 10686 | 15% | 17 GB/s |
| ffn_down `q6_K[1536×8960]` | 10.7 | 761 | 15% | |
| attn_output + Q `q4_K[1536×1536]` | 7.1 | 127 | 10% | 10 GB/s |
| gqa | 2.5 | 90 | 3% | |
| act_mul | 2.4 | 84 | 3% | **anomaly**: 23 µs in isolation |
| K/V `q4_K/q6_K[256×1536]` | 2.9 | 51–53 | 4% | **latency-bound**: 4 GB/s |
| rope, rmsnorm, kv_write, add | 0.9 | 2–7 | 1% | |

The DRAM ceiling is 18.4 GB/s (`tools/proto/read_bw.cpp`).

- **Large matmuls.** They already run at 85–96% of the ceiling. No kernel rewrite can gain
  much there.
- **The rest.** The remaining waste is in latency-bound small ops and in data that crosses
  cores. `act_mul` reads gate/up values that 10 other cores just wrote: 84 µs in place
  against 23 µs isolated.

## 2. Activation kernels (kept: column chunking; reverted: polynomial exp)

- **Hypothesis A:** `act_mul` is slow because `std::exp` per element is scalar.
  - Test: an `exp_nonpos` polynomial SiLU.
  - Result: slower than MSVC's vectorized `std::exp` (1×1536: 8.6 vs 5.6 µs; 64×8960: 739 vs
    517 µs).
  - **Reverted.**
- **Hypothesis B:** a decode row runs on one thread.
  - Change: split columns into 2048-wide chunks over the pool.
  - Result (`bench_elementwise`):

    | Shape | Before | After |
    |---|---|---|
    | 1×8960 | 33 µs | 14 µs |
    | 4×8960 | 79 µs | 33 µs |
    | 64×8960 | 517 µs | 402–437 µs |

  - **Kept.**
- **In the model,** `act_mul` stayed ~88 µs: its cost is reading another core's freshly
  written lines. Fusion attacks that (section 4).

## 3. Recording overhead (plan cache)

The first deferred implementation built IR for every call. A/B (`bench_compiled`, 4
interleaved rounds):

| Item | Before cache | After cache |
|---|---|---|
| recording | 1.25 ms/step | ~50 µs/step |
| planning | 0.3 ms/step | 0 after the first step |
| IR-only variant vs reference | +2.4 to +11.6% | ±2% (noise) |

Design: record calls cheaply, sign the segment structurally, and cache the call-level plan
(dynacore-compiler.md). Decode steps share one signature.

A second issue found on the way: `set_kernel_plan` was a sync point. The FFN down projection
switches plans twice per layer, which made 57 segments per step. Recording each call's plan
fixed it: one segment per step.

## 4. Fusion (kept for decode-shaped matmuls)

`bench_compiled`, step-interleaved, 40 steps per variant per round, 4 rounds. Change in mean
step time against the reference CPU device:

| Variant | 1 sequence | 4 sequences |
|---|---|---|
| IR only (record + cached plan, no fusion) | +0.3, −0.0, +1.8, +0.2% | +2.1, −0.3, −0.4, −1.2% |
| all fusions (Q/K/V group + gated MLP) | −4.1, −7.7, −8.6, −9.1% | −1.6, −4.3, −5.8, −6.0% |

- **Accuracy:** maximum logit difference 0 for every variant. The same kernels are combined
  differently, and greedy outputs are byte-identical.
- **Prefill.** The first end-to-end run grouped prefill-shaped matmuls too. At concurrency 4
  and 8, mixed steps then lost 4–7%: `matmul_many` uses 16-row panels without the K-blocked
  GEMM that `matmul` uses for large M.
- **Rule:** fuse only matmuls the kernel selector places on the int8 decode path (M ≤ the
  op's `int8_rows`). For prefill shapes the compiled plan equals reference execution.

## 5. End to end (`dynalm benchmark`)

Results and the comparison table are in [compiler-benchmarks.md](compiler-benchmarks.md).

## Ideas measured and not pursued

- **Generated GEMV kernels:** the large GEMVs are at 88–96% of DRAM bandwidth already
  (compiler-backends.md).
- **VNNI int8 GEMM:** slower than fp32 with per-block scales (DD-058).
- **Elementwise fusion beyond act_mul:** rope, norms and adds total about 1% of the step.
- **Memory planning:** the analysis is available (`dynacorec --memory`). A decode layer's
  activations are ~160 KiB and already reuse fixed scratch, so there is nothing to win at
  runtime.

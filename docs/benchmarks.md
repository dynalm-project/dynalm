# Benchmarks

All numbers: i7-1255U (2P+8E), 16 GB, Windows 11, MSVC 19.44 Release (`msvc-release`),
laptop on AC power, unless stated otherwise. Units are ns/op unless stated.

## Phase 0 — foundation primitives (`bench_foundation`)

| benchmark | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| `now_ns()` | 25.7 | 18.9 | 37.9 | 42.6 | 80.5 |
| `LOG_DEBUG` (disabled) | 0.32 | 0.30 | 0.40 | 0.40 | 0.40 |
| `Result<int>` ok path | 0.59 | 0.60 | 0.60 | 0.70 | 1.00 |

Takeaways:
- Timestamps (QueryPerformanceCounter) cost about 20–40 ns. That's negligible per token
  (tokens take milliseconds), but per-op timing inside kernels would be too costly, so
  kernel profiling must be sampled or compiled out.
- Disabled debug logs cost almost nothing, so debug logging can stay in scheduler code.
- The error type's OK path costs almost nothing (DD-002).

## Phase 1 — tensor/dtype (`bench_tensor`)

| benchmark | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| fp16→fp32 ×4096 (scalar) | 6019 | 5130 | 8310 | 9990 | 13360 |
| fp32→fp16 ×4096 (scalar) | 7208 | 6650 | 8080 | 8910 | 12980 |
| bf16→fp32 ×4096 | 593 | 580 | 640 | 640 | 720 |
| host_alloc+free 4 KiB | 48 | 46 | 51 | 54 | 91 |
| host_alloc+free 1 MiB | 7211 | 6918 | 8590 | 9534 | 12784 |
| TensorView select+slice | 83 | 77 | 96 | 125 | 222 |

Takeaways:
- Scalar fp16 costs 1.25 ns/elem. F16C (Phase 17) should make it about 10× faster.
  Until then, hot paths should avoid converting fp16 per element.
- 1 MiB allocations cost about 7 µs, with a long tail. This confirms that per-token or
  per-step buffers must come from preallocated arenas and pools, never `host_alloc`.
- Building a view costs 77 ns. That's fine at graph-build time; kernels index directly.

## Phase 2 — GGUF loader (`bench_loader models/SmolLM2-135M-Instruct-f16.gguf`)

Model: SmolLM2-135M-Instruct f16, 258 MiB, 272 tensors, 33 KV (49,152 tokens, 48,900 merges).

| benchmark | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| `GgufFile::open` (warm page cache) | 768.6 µs | 748.6 µs | 855.6 µs | 1020.5 µs | 1108.0 µs |
| decode `tokenizer.ggml.tokens` (49k strings) | 113.6 µs | 112.6 µs | 124.3 µs | 132.5 µs | 212.9 µs |
| `load_tensor` × 272 (zero-copy) | 40.2 µs | 38.5 µs | 39.4 µs | 47.6 µs | 93.9 µs |

Most of the open time is validating the vocab and merges string arrays. Weights are not
touched at load time.

## Phase 3 — Model IR

Nothing here is performance-critical: IR construction happens once per model load, and
role lookups are O(1) array indexing. `engine inspect` on SmolLM2-135M (open + config +
estimate) runs at process-startup speed. No benchmark was added.

## Phase 4 — tokenizer (`bench_tokenizer models/SmolLM2-135M-Instruct-f16.gguf`)

32 KiB of mixed prose, code, numbers, CJK and emoji → 13,985 tokens. Vocab 49,152;
tokenizer build 17.8 ms.

| benchmark | mean | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| encode 32 KiB | 2.51 ms | 2.15 ms | 3.15 ms | 5.25 ms | 6.37 ms |
| decode 13,985 tokens | 228 µs | 217 µs | 258 µs | 329 µs | 360 µs |

15.3 MB/s encode (6.5 M tokens/s) and 15.5 ns/token decode. A 50K-token prompt
tokenizes in about 8 ms, far below prefill cost, so the tokenizer is not a bottleneck.

## Phase 6 — baseline end-to-end (generic kernels)

`engine run SmolLM2-135M-Instruct-f16.gguf -p "Write a short story about a robot learning to paint." -n 64`
(chat template → 41 prompt tokens), f16 weights, f16 KV, generic kernels (no SIMD, no GEMM tiling).
Linux container (gcc 13, -O3) on WSL2, same i7-1255U. Smart App Control blocked the Windows binary.

| threads | prefill tok/s | TTFT ms | decode tok/s | ITL p50 | ITL p90 | ITL p99 |
|---|---|---|---|---|---|---|
| 1 | 14.7 | 2787 | 6.6 | 148.5 | 160.1 | 202.5 |
| 2 | 25.5 | 1607 | 11.9 | 83.5 | 87.4 | 105.0 |
| 4 | 31.4 | 1306 | 16.3 | 60.9 | 64.1 | 79.0 |
| 6 | 34.2 | 1198 | 19.2 | 51.6 | 56.6 | 62.4 |
| 8 | 27.5 | 1491 | 23.3 | 42.4 | 45.9 | 65.2 |
| **10** | 32.9 | 1245 | **27.0** | **36.6** | 39.1 | 55.6 |
| 12 (SMT) | 31.9 | 1284 | 21.6 | 40.6 | 69.1 | 120.3 |

Takeaways:
- Decode scales to the 10 physical cores. SMT (12 threads) hurts throughput and especially
  the tail (p99 120 ms). The default of physical cores (DD-004) is confirmed.
- Prefill is the weak spot: 33 tok/s is barely faster per token than decode. The generic
  matmul computes one scalar dot product per (row, output), with fp16 conversion and no
  tiling, so prefill is compute-bound. First targets for Phases 17/18: SIMD dot
  products with F16C, and a tiled GEMM for m > 1.

## Phase 8 — quantized execution

### Reference dequantization throughput (`bench_quant`, one 4096-element row, scalar)

| type | p50 ns | p99 ns | ns/elem | GB/s of source |
|---|---|---|---|---|
| f16 | 3829 | 9297 | 0.935 | 2.14 |
| bf16 | 259 | 333 | 0.063 | 31.67 |
| q8_0 | 482 | 1349 | 0.118 | 9.03 |
| q4_0 | 1559 | 3332 | 0.381 | 1.48 |
| q5_0 | 3284 | 11019 | 0.802 | 0.86 |
| q2_K | 1876 | 2486 | 0.458 | 0.72 |
| q3_K | 2971 | 4471 | 0.725 | 0.59 |
| q4_K | 576 | 674 | 0.141 | 4.00 |
| q5_K | 763 | 883 | 0.186 | 3.69 |
| q6_K | 3494 | 5250 | 0.853 | 0.96 |

### End to end (`engine run ... -n 64 -t 10`, same prompt as Phase 6, Linux container)

| model | weights | prefill tok/s | decode tok/s | ITL p50 / p90 / p99 ms |
|---|---|---|---|---|
| SmolLM2-135M | f16 | 23.0 | 18.5 | 45.7 / 69.4 / 140.3 |
| SmolLM2-135M | Q8_0 | 29.3 | **42.3** | 21.9 / 34.6 / 45.0 |
| Qwen2.5-0.5B | f16 | 11.1 | 5.8 | 164.8 / 190.8 / 259.9 |
| Qwen2.5-0.5B | Q8_0 | 14.2 | **16.5** | 55.4 / 79.4 / 100.6 |
| Qwen2.5-0.5B | Q4_K_M (mostly Q5_0) | 19.3 | 8.5 | 113.4 / 132.8 / 176.4 |

Takeaways:
- With the Phase 8 matmul (expand each weight row to fp32, then dot), decode is bound by
  **conversion arithmetic, not memory**. Scalar fp16 conversion (0.94 ns/elem) is slower
  than Q8_0 (0.12), so Q8_0 models decode 2–3× faster than their f16 originals.
- Q4_K_M files are mostly Q5_0 and Q6_K here, and their bit unpacking is the slowest,
  so smaller files aren't faster yet.
- Phase 17 priorities follow directly: F16C conversion, then fused integer quantized dot
  products that never materialize fp32 weights.
- Measurement noise: SmolLM2 f16 decode measured 27.0 tok/s in Phase 6 and 18.5 here with
  identical code paths for f16. On this laptop, thermal state and background load move
  results by up to ~30%. Phase 21's benchmark framework will pin runs and report
  repeated-run variance.

## Phase 9 — decode latency vs context (`bench_decode_context`, SmolLM2-135M Q8_0, 10 threads, f16 KV)

| context | p50 ms | p90 ms | p99 ms | tok/s |
|---|---|---|---|---|
| 64 | 14.81 | 17.33 | 18.40 | 67.5 |
| 256 | 17.63 | 22.07 | 22.78 | 56.7 |
| 1024 | 39.65 | 45.47 | 47.30 | 25.2 |
| 2048 | 59.28 | 66.51 | 77.82 | 16.9 |
| 4096 | 93.20 | 106.39 | 126.91 | 10.7 |

Latency grows linearly, about 19 µs per context token. KV is appended in place (one row
per step, never copied), so the growth comes from attention reads. The analysis
explains why it's this steep:
- Decode attention is parallelized only over query heads (9 jobs on 10 threads). Each
  job walks its head's entire context serially.
- K/V are fp16 and converted element by element in scalar code (~0.9 ns/elem), for
  30 layers × 4096 tokens × 128 dims per head.

That's about 2.5 ms per layer at 4K, which matches the measured +78 ms.
Phase 17/18 plan: split-K ("flash-decoding") attention that parallelizes over KV
blocks and merges partial softmaxes, plus F16C conversion. The target is a nearly flat
curve up to several thousand tokens.

## Phase 10 — paged KV

### Pool operations (`bench_kv`, SmolLM2-135M geometry: 30 layers, 3 kv heads, 64 dims, f16, 16-token blocks)

| operation | mean ns | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| allocate + release (1 thread) | 13.0 | 12.5 | 13.6 | 13.6 | 21.2 |
| allocate + release (4 threads hammering) | 451 | 424 | 844 | 1013 | 1311 |
| clone 256-block table + release | 2372 | 2277 | 2525 | 2635 | 3366 |
| copy-on-write 1 block (360 KiB) | 6450 | 6146 | 6788 | 7319 | 11619 |

### Block size sweep (`bench_decode_context ... 10 <bs>`, SmolLM2-135M Q8_0, p50 ms per decode token)

| block size | ctx 256 | ctx 2048 | ctx 4096 |
|---|---|---|---|
| 8 | 16.6 | 49.3 | 80.3 |
| 16 | 19.1 | 48.7 | 79.9 |
| 32 | 19.0 | 49.4 | 80.5 |
| 64 | 19.6 | 48.4 | 79.0 |

Block size has no measurable effect on attention speed today: scalar fp16 compute
dominates, and the per-token block-table lookup is noise. See DD-024 for the decision.

## Phase 11 — concurrent sequences (`bench_batch_decode`, 10 threads, 128-token context each)

One step = one decode token for each of N sequences, in a single batched forward pass.

| seqs | SmolLM2-135M Q8_0 step p50 ms | p99 | aggregate tok/s | Qwen2.5-0.5B Q8_0 step p50 ms | p99 | aggregate tok/s |
|---|---|---|---|---|---|---|
| 1 | 14.6 | 16.8 | 68.3 | 51.6 | 80.0 | 19.4 |
| 2 | 20.9 | 22.1 | 95.5 | 68.4 | 79.8 | 29.3 |
| 4 | 32.2 | 34.2 | 124.3 | 106.4 | 127.7 | 37.6 |
| 8 | 52.9 | 54.4 | **151.3** | 206.7 | 227.1 | 38.7 |
| 16 | 128.4 | 187.2 | 124.6 | 363.2 | 452.5 | 44.1 |
| 32 | 243.7 | 309.5 | 131.3 | 697.4 | 777.3 | **45.9** |

Takeaways:
- Batching pays off (2.2× SmolLM2 at 8 sequences, 2.4× Qwen2.5 at 32) because weights
  are converted and read once per step for all rows.
- It saturates early because the generic matmul is compute-bound: a scalar dot product
  per (row, output), after expanding each weight row to fp32. With SIMD and integer dot
  products (Phase 17) and a blocked GEMM that keeps weight tiles in cache across rows
  (Phase 18), aggregate throughput should keep scaling well past 8 sequences.
- Per-sequence ITL grows with batch size (14.6 → 52.9 ms at 8). The Phase 13 scheduler
  must balance throughput against ITL (decode token budget).

## Phase 12 — continuous batching (`bench_scheduler`, SmolLM2-135M Q8_0, 10 threads)

N requests submitted at once; 64-token prompts, 64 generated tokens each; token budget
256 per step, FCFS admission, decode rows scheduled before prefill.

| requests | wall s | aggregate tok/s | TTFT p50 ms | p90 | p99 | ITL p50 ms | p90 | p99 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.14 | 29.9 | 1232 | 1232 | 1232 | 13.2 | 18.1 | 33.2 |
| 4 | 4.39 | 58.3 | 955 | 956 | 956 | 49.6 | 91.7 | 155.6 |
| 8 | 5.35 | **95.7** | 968 | 2042 | 2042 | 52.0 | 58.4 | 116.4 |
| 16 | 12.07 | 84.9 | 2990 | 3995 | 4202 | 117.8 | 191.8 | **1003.4** |
| 32 | 23.96 | 85.5 | 6037 | 9941 | 10614 | 222.2 | 279.1 | **1455.5** |

Takeaways:
- Continuous batching raises aggregate throughput 3.2× at 8 concurrent requests.
- **Prefill/decode interference is the dominant latency problem.** With 16+ requests, a
  step that admits new prompts carries up to 256 prefill rows, which take about 1 s with
  the generic kernels, and every decoding sequence waits (ITL p99 1.0–1.5 s). Phase 13
  (separate prefill and decode budgets) and Phase 14 (chunked prefill) target this directly.
- TTFT is dominated by slow prefill (≈30 prompt tok/s generic). Phase 17/18 kernels
  address the absolute numbers.
- The 1-request row reports the first prefill including cold page faults of the mmapped
  weights (TTFT 1.2 s for 64 tokens).

## Phase 13 — prefill/decode budgets (`bench_scheduler <model> 10 <prefill_budget>`, SmolLM2-135M Q8_0)

Decode budget 64 rows; 64-token prompts, 64 generated tokens.

| prefill budget | reqs | tok/s | TTFT p50 ms | TTFT p99 | ITL p50 ms | ITL p90 | ITL p99 |
|---|---|---|---|---|---|---|---|
| 192 | 8 | 85.5 | 1600 | 2203 | 58.4 | 70.5 | 603.2 |
| 192 | 16 | 78.6 | 2486 | 4645 | 129.6 | 189.5 | 890.5 |
| 192 | 32 | 76.3 | 6729 | 12370 | 244.1 | 323.0 | 1226.2 |
| **64** | 8 | 68.4 | 1373 | 2850 | 69.5 | 126.1 | 376.0 |
| **64** | 16 | 73.0 | 2893 | 6238 | 130.1 | 383.7 | 441.0 |
| **64** | 32 | 75.6 | 6252 | 14747 | 244.5 | 489.3 | 760.6 |
| 32 | 8 | 70.8 | 1455 | 3222 | 68.8 | 187.8 | 245.0 |
| 32 | 16 | 68.8 | 3230 | 7554 | 126.2 | 273.3 | 328.3 |
| 32 | 32 | 72.9 | 7608 | 18971 | 264.8 | 389.0 | 481.1 |

A smaller prefill budget trades TTFT for ITL tail latency: ITL p99 drops up to 2.5×
(1226 → 481 ms at 32 requests), while TTFT p99 rises about 50% and throughput dips about 5%.
Default: 64 (see DD-027). Both budgets are configuration, and the AutoTuner (Phase 22) can pick
them per model and workload.

## Phase 14 — chunked prefill (`bench_long_prompt`, SmolLM2-135M Q8_0, 10 threads)

8 requests are decoding when a 2048-token prompt arrives. The table shows decoder ITL
measured during the long prefill, and the long request's TTFT.

| config | ITL p50 ms | ITL p90 | ITL p99 | long TTFT s |
|---|---|---|---|---|
| unchunked (whole prompt in one step) | 36710 | 36711 | 36711 | 36.71 |
| prefill budget 256, chunk 256 | 4132 | 7592 | 7592 | 36.88 |
| prefill budget 64, chunk 64 | 1147 | 1822 | 2114 | 38.85 |
| prefill budget 32, chunk 32 | 624 | 1028 | 1086 | 41.19 |

Without chunking a long prompt freezes every active conversation for its entire prefill
(37 s here). 32-token chunks bound the stall to about 1 s p99 for +12% TTFT on the long
request. The remaining per-step cost is 32 prefill rows attending over up to 2K context
with scalar kernels (Phase 17/18).

Head-of-line blocking (test `ChunkCapPreventsHeadOfLineBlocking`): with a per-sequence
chunk cap of 16, a 4-token prompt submitted behind a 120-token prompt gets its first
token in step 1 instead of step 4.

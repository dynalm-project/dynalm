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
role lookups are O(1) array indexing. `dynalm inspect` on SmolLM2-135M (open + config +
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

`dynalm run SmolLM2-135M-Instruct-f16.gguf -p "Write a short story about a robot learning to paint." -n 64`
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

### End to end (`dynalm run ... -n 64 -t 10`, same prompt as Phase 6, Linux container)

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

## Phase 15 — prefix hash cache (`bench_prefix`, SmolLM2-135M Q8_0, 10 threads)

A warm-up request with a 512-token system prompt, then 15 requests arriving together, each with
the same system prompt + 16 unique tokens, 16 generated. Prefill budget 128, chunk 128.

| prefix cache | wall s | rows computed | TTFT p50 ms | TTFT p90 | TTFT p99 | hit rate |
|---|---|---|---|---|---|---|
| off | 65.00 | 8145 | 33004 | 59819 | 64298 | 0% |
| **on** | **5.78** | **465** | **1430** | **2840** | **2840** | 94% |

17.5× fewer rows, 11× less wall time and 23× lower median TTFT. Outputs are bit-identical to
uncached runs (tests). Limitation: requests arriving in the same step as the *first* request
with a new prefix can't reuse it, because blocks are cached once computed. In-flight dedup is
a candidate for the radix cache (Phase 16).

## Phase 16 — radix vs hash prefix cache (`bench_prefix`, SmolLM2-135M Q8_0, 10 threads)

Same workload as Phase 15, but the system prompt is 500 tokens (not a multiple of the
16-token block).

| cache | wall s | rows computed | TTFT p50 ms | TTFT p90 | TTFT p99 | hit rate |
|---|---|---|---|---|---|---|
| off | 57.41 | 7965 | 28137 | 53097 | 56798 | 0% |
| hash | 6.36 | 525 | 2827 | 3447 | 3448 | 90.8% |
| **radix** | **5.42** | **465** | **1373** | **2692** | **2692** | 91.0% |

Lookup latency for an 8192-token fully cached prompt (512 blocks):

| cache | p50 ns | p99 ns |
|---|---|---|
| hash | 63699 | 73333 |
| radix | 25975 | 38963 |

Radix reuses the partially matching block (500 = 31×16 + 4), so it computes 11% fewer rows
and halves median TTFT. Its lookups are 2.5× faster (a per-block child lookup instead of
per-token chained mixing). Radix is the default.

## Phase 17 — CPU SIMD (AVX2 + FMA + F16C)

### Fused dequantize-dot per weight type (`bench_kernels`, one 4096-element row)

| type | generic ns | AVX2 ns | speedup | AVX2 GB/s of weights |
|---|---|---|---|---|
| f32 | 484 | 127 | 3.8× | 129.2 (L1-resident) |
| f16 | 3433 | 277 | 12.4× | 29.6 |
| bf16 | 854 | 537 | 1.6× | 15.3 |
| q8_0 | 1595 | 234 | 6.8× | 18.6 |
| q4_0 | 1662 | 351 | 4.7× | 6.6 |
| q5_0 | 3096 | 432 | 7.2× | 6.5 |
| q4_K | 2062 | 412 | 5.0× | 5.6 |
| q5_K | 1238 | 1267 | 1.0× (no AVX2 kernel yet) | 2.2 |
| q6_K | 3661 | 406 | 9.0× | 8.3 |

(An earlier run fed f32/bf16 rows of tiny byte patterns, which are denormal floats and
pathologically slow. The benchmark now uses real values.)

### End to end (`dynalm run ... -n 64 -t 10`, Linux container) — Phase 8 → Phase 17

| model | weights | prefill tok/s | decode tok/s (was) | ITL p50 / p99 ms |
|---|---|---|---|---|
| SmolLM2-135M | f16 | 28.4 | **53.1** (18.5) | 17.2 / 30.1 |
| SmolLM2-135M | Q8_0 | 51.6 | **95.3** (42.3) | 9.9 / 25.2 |
| Qwen2.5-0.5B | f16 | 12.7 | **16.3** (5.8) | 56.2 / 138.1 |
| Qwen2.5-0.5B | Q8_0 | 15.9 | **22.9** (16.5) | 43.2 / 68.1 |
| Qwen2.5-0.5B | Q4_K_M | 23.5 | **28.6** (8.5) | 30.0 / 101.7 |

### Decode vs context (`bench_decode_context`, SmolLM2 Q8_0, f16 KV) — p50 ms/token

| context | Phase 9 | Phase 17 |
|---|---|---|
| 64 | 14.8 | 16.3 |
| 256 | 17.6 | 10.9 |
| 1024 | 39.7 | 15.0 |
| 2048 | 59.3 | 17.4 |
| 4096 | 93.2 | 27.0 |

### Batched decode (`bench_batch_decode`, SmolLM2 Q8_0) — aggregate tok/s

| seqs | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Phase 11 | 68.3 | 95.5 | 124.3 | 151.3 | 124.6 | 131.3 |
| Phase 17 | 100.9 | 184.1 | 258.7 | 326.2 | **355.1** | 302.6 |

Takeaways:
- Decode is 1.4–3.4× faster. Qwen2.5 Q8_0 decode (22.9 tok/s × 675 MB ≈ 15.5 GB/s) is now
  close to this laptop's practical memory bandwidth under WSL: bandwidth-bound, as decode
  should be.
- f16 KV attention via F16C made long-context decode 3.4× faster at 4K.
- Prefill barely moved (it runs the same per-row dots for every activation row). The first
  Phase 18 target is a tiled GEMM that keeps weight tiles in cache across rows, followed by
  int8 activation quantization (AVX-VNNI is available) and split-K decode attention.

## Phase 18 — kernel optimization

### Where time goes (`bench_profile`, Qwen2.5-0.5B Q8_0, 10 threads, before GEMM tiling)

| op | prefill 256 tokens | decode (32 steps) |
|---|---|---|
| qkv | 5.6% | 10.3% |
| attention | 3.2% | 4.1% |
| attn_out | 4.3% | 5.7% |
| mlp_up (gate+up) | 44.0% | 36.2% |
| mlp_down | 40.2% | 19.2% |
| lm_head | 0.5% | 22.1% |
| norm + rope + act + kv_store + embed | 2.3% | 2.5% |

Matmuls dominate both phases. Elementwise ops are under 6% in total, so fusing them can't pay off
(DD-033). Decode reads 675 MB per token in 36 ms (≈18.7 GB/s): bandwidth-bound.

### Matmul (`bench_matmul`, m=256, Q8_0)

- Register-blocked GEMM panel (4 weight rows × 2 activation rows, 8 accumulators): 56–81
  GFLOPS single-thread, **~290 GFLOPS on 10 threads** for the gate/up shape (k=896).
- K-blocking on the down shape (k=4864, n=896), 10 threads:

| K-slice | Q8_0 GFLOPS | f32 GFLOPS |
|---|---|---|
| off | 172.8 | 188.9 |
| 512 | 240.4 | 198.8 |
| **1024 (default)** | **254.8** | **252.9** |
| 2048 | 173.3 | 205.5 |

### Warm prefill, 256 tokens (`bench_profile`)

| model | Phase 17 | Phase 18 |
|---|---|---|
| SmolLM2-135M Q8_0 | ~493 tok/s | **912 tok/s** |
| Qwen2.5-0.5B Q8_0 | 132 tok/s | **277 tok/s** |
| Qwen2.5-0.5B Q4_K_M | — | **278 tok/s** |

Qwen prefill sustains about 257 GFLOPS end to end (≈1 GFLOP per token), near this laptop's
practical fp32 FMA peak. Earlier `dynalm run` prefill numbers (16–50 tok/s) were dominated by
cold page faults of the memory-mapped weights on the first forward pass, not by compute.

### Split-K decode attention (`bench_decode_context`, SmolLM2 Q8_0) — p50 ms/token

| context | Phase 17 | Phase 18 |
|---|---|---|
| 256 | 10.9 | 9.9 |
| 1024 | 15.0 | 12.8 |
| 2048 | 17.4 | 15.4 |
| 4096 | 27.0 | 22.9 |

The remaining long-context cost is memory traffic: about 94 MB of K/V per token at 4K, and each
GQA KV head is read once per query head (3×). Grouped GQA attention is the next step (see ROADMAP.md).

### End to end through the scheduler (`bench_scheduler`, SmolLM2 Q8_0, budgets 64/64)

| requests | Phase 13 tok/s | Phase 18 tok/s | TTFT p50 ms (was) | ITL p99 ms (was) |
|---|---|---|---|---|
| 1 | — | 38.5 | 939 | 62.2 |
| 4 | — | 177.1 | 137 | 77.4 |
| 8 | 68.4 | **246.3** | 317 (1373) | 82.6 (376.0) |
| 16 | 73.0 | **316.4** | 613 (2893) | 90.9 (441.0) |
| 32 | 75.6 | **314.2** | 1341 (6252) | 200.7 (760.6) |

Faster prefill shrinks the per-step stall, so throughput, TTFT and the ITL tail all improve at
the same time: 4× throughput and 4.7× lower TTFT at 16 requests.

## Phase 21 — serving benchmarks (`dynalm benchmark`, DD-037)

All rows come from the same closed-loop client (`dynalm benchmark --url`, streaming
`/v1/completions`, `ignore_eos`, temperature 0). Every request has a unique prompt of exact
length, and prompts are never reused across points. Each point runs 2 × concurrency requests
(at least 4) after one warm-up request. Model: Qwen2.5-0.5B-Instruct Q4_K_M; 10 threads; the
i7-1255U laptop. Servers run one at a time in the same `dynalm-dev`-style Docker environment
(`tools/compare_baselines.sh`); llama.cpp is `ghcr.io/ggml-org/llama.cpp:server` with
`-t 10 -tb 10 -c 32768 -np 16`. Latencies are in ms. A laptop under sustained load throttles,
so differences under ~10% are noise.

### Prompt 128, output 128

| target | conc | out tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPOT p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|
| engine | 1 | **27.3** | **501** | 549 | 30.3 | 87.6 | 31.1 | 5104 |
| llama.cpp | 1 | 25.5 | 647 | 792 | 28.6 | 109.3 | 30.6 | 5540 |
| engine | 4 | 51.5 | **1260** | 1840 | 53.0 | 268.0 | 65.7 | 10578 |
| llama.cpp | 4 | 53.3 | 2808 | 2974 | 46.2 | 125.4 | 51.2 | 9798 |
| engine | 16 | **77.7** | **1953** | 11589 | 127.2 | 520.6 | 185.7 | 35549 |
| llama.cpp | 16 | 61.9 | 11512 | 12371 | 150.4 | 388.8 | 163.3 | 33108 |

### Prompt 512, output 128

| target | conc | out tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPOT p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|
| engine | 1 | 15.1 | **2658** | 3260 | 34.6 | 114.3 | 37.2 | 11405 |
| llama.cpp | 1 | 15.1 | 3012 | 3446 | 33.4 | 140.7 | 35.7 | 9746 |
| engine | 4 | **28.6** | **5401** | 10684 | 56.5 | 407.5 | 93.2 | 22815 |
| llama.cpp | 4 | 24.4 | 12297 | 13922 | 54.0 | 207.0 | 71.7 | 21475 |
| engine | 16 | **33.3** | **8058** | 50001 | 392.5 | **638.1** | 377.3 | 102124 |
| llama.cpp | 16 | 27.8 | 24251 | 53155 | 162.9 | 12574.6 | 377.7 | 101120 |

How to read this:
- **Throughput and TTFT.** Throughput is at parity or better in every cell. TTFT p50 is
  1.1–3× lower under concurrency, because chunked prefill admits new requests within one
  step.
- **Inter-token latency.** llama.cpp has the lower ITL p50, because it decodes without
  interleaving prefill. It pays for that with multi-second stalls: an ITL p99 of 12.6 s at
  c=16 / 512, where every running stream waits for a newcomer's whole prompt. We spread the
  prefill over steps (64-token budget), so each step is slower but no stream stalls (p99
  638 ms). This is the TTFT ↔ ITL tradeoff from DD-027 and DD-028. TPOT ends up equal at c=16.
- **Remaining gap.** At c=4, decode TPOT is 66 vs 51 ms (prompt 128). The next kernel steps
  are a multi-row fused decode kernel and GQA-grouped attention (see ROADMAP.md).

### The regression the first run exposed (DD-036)

The first baseline run (before DD-036, `results/baselines-pre-dd036.jsonl`) had prompts reused
across points, which flattered both servers' TTFT on later points. It showed us at
**34.5 vs 64.4 out tok/s at c=4**: batched decode re-read every weight per sequence.
Expanding weights from 2 rows, plus AVX2 row dequantization, fixed it:

| Q4_K_M, in-process, prompt 128 / output 64 | fused (old m < 4) | expand ≥ 4 | expand ≥ 2 (new) |
|---|---|---|---|
| c=2 ITL p50 ms | 47.1–48.3 | 46.4–46.7 | **34.8–37.5** |
| c=4 ITL p50 ms | 87.1–91.0 | 48.4–49.8 | **42.7–47.0** |

Row dequantization, 4096 elements (`bench_kernels`), in ns:

| type | generic | AVX2 | speedup |
|---|---|---|---|
| f16 | 10819 | 348 | 31.1× |
| q8_0 | 1235 | 667 | 1.9× |
| q4_0 | 4419 | 1080 | 4.1× |
| q5_0 | 9043 | 1794 | 5.0× |
| q4_K | 1594 | 1397 | 1.1× |
| q6_K | 11178 | 1146 | 9.8× |

### Ollama 0.34.2 (native Windows, same GGUF via `ollama create`; `tools/bench_ollama.sh`)

| conc | prompt | out tok/s | mean output tokens | TTFT p50 | ITL p50 | E2E p99 |
|---|---|---|---|---|---|---|
| 1 | 128 | 10.3 | 13.5 | 976 | 25.3 | 1320 |
| 4 | 128 | 20.5 | 41.8 | 4781 | 25.0 | 8193 |
| 1 | 512 | 16.7 | 122.5 | 3690 | 28.4 | 7633 |
| 4 | 512 | 17.0 | 128 | 26271 | 27.1 | 30115 |

These rows are not directly comparable:
- **Output length.** Ollama ignores `ignore_eos`, so outputs are shorter and vary.
- **Environment.** It runs natively, outside the Docker VM, with its own thread count.
- **Parallelism.** It ran with its default `OLLAMA_NUM_PARALLEL`. TTFT growing with
  concurrency while ITL stays flat indicates requests were served one at a time.

The comparable signal is single-stream ITL: 25–28 ms natively vs our 30–35 ms and
llama.cpp's 29–33 ms in the container. Some of that difference is likely the VM boundary.

## Phase 22 — hardening overhead

Hardening adds two atomic operations per HTTP request (admission slot), one shared guard
per request, and two clock reads plus one histogram update per scheduler step (steps take
≥ 10 ms). In-process decode, Qwen2.5-0.5B Q4_K_M, 10 threads, prompt 128 / output 64, two
alternating repetitions:

| ITL p50 ms | Phase 21 (DD-036 sweep) | Phase 22 |
|---|---|---|
| c=2 | 34.8–37.5 | 32.8–37.0 |
| c=4 | 42.7–47.0 | 40.7–47.5 |

No measurable change; run-to-run spread on this laptop is ±10%.

Lifecycle checks in the container:
- `dynalm stop` drains and exits 0.
- SIGTERM with one request in flight finishes it (96 tokens, 0 errors), then exits 0.
- AUTO KV picked 65536 tokens (0.75 GB) with 5.85 GB free.

## Phase 23 — SafeTensors (BF16) vs GGUF (F16), same model

SmolLM2-135M-Instruct, 10 threads, in-process `dynalm benchmark`, prompt 128 / output 64:

| format | conc | out tok/s | ITL p50 ms | TTFT p50 ms |
|---|---|---|---|---|
| GGUF F16 | 1 | 41.8 | 19.4 | 181 |
| SafeTensors BF16 | 1 | 41.3 | 19.6 | 175 |
| GGUF F16 | 4 | 91.9 | 28.7 | 397 |
| SafeTensors BF16 | 4 | **120.0** | **23.1** | 363 |

Row dequantization, 4096 elements (`bench_kernels`): BF16 generic 280 ns → AVX2 149 ns (1.9×).
F16 needs a real conversion (F16C, 144 ns); BF16 is a shift. That is why BF16 batches decode
faster than F16. Single-stream decode is memory-bound and the same for both.

Load time (`dynalm run`, warm page cache): GGUF 58 ms, SafeTensors 170 ms. The difference is
parsing the 2 MB `tokenizer.json`; the weights are memory-mapped in both cases.

Accuracy: max |logit diff| 1.7e-5 between the BF16 checkpoint and its F16 GGUF conversion. The
tiny fixtures are bit-exact (0) for 6 of 7 architectures, and Llama differs by 1.1e-6
(RoPE pairing order).

## Phase 24 — GPTQ / AWQ checkpoints (repacked at load)

Qwen2.5-0.5B-Instruct, 10 threads, in-process `dynalm benchmark`, prompt 128 / output 64:

| checkpoint | runs as | load ms | c=1 out tok/s | c=1 ITL p50 ms | c=4 out tok/s | c=4 ITL p50 ms |
|---|---|---|---|---|---|---|
| GGUF Q4_K_M | Q4_K/Q5_0/Q6_K | 162 | 25.4 | 29.0 | 46.2 | 49.8 |
| HF GPTQ-Int4 (sym, g128) | Q4_0 (bit-exact) | 2052 | 24.2 | 31.0 | 46.3 | 50.2 |
| HF AWQ (zero point, g128), generic Q4_1 | Q4_1 | 2357 | 11.0 | 77.6 | 32.2 | 84.7 |
| HF AWQ, with AVX2 Q4_1 (this phase) | Q4_1 | 2357 | **23.4** | **33.4** | **49.5** | **50.2** |

Kernels, one 4096-element row (`bench_kernels`): Q4_1 dot 2533 → 487 ns (5.2×); Q4_1 dequant
1707 → 537 ns (3.2×).

Load time is spent unpacking and repacking every linear layer on one thread (once per
layer). GGUF loads are memory-maps only.

## Phase 25 — mixture of experts (Granite-3.1-1B-A400M: 32 experts, 8 active, 1.3B total / 0.4B active)

Where decode time went (`bench_profile`, Q8_0, 10 threads, ms per token):

| version | total | expert matmuls | note |
|---|---|---|---|
| per-expert `matmul` calls | 59.7 | 42.4 | 576 small matmuls per token, each its own thread-pool dispatch |
| `matmul_many` (one parallel region per layer for all active experts) | **29.1** | 17.7 | ≈ 290 MB of expert weights per token at ≈ 16 GB/s |

Prefill, 256 tokens: 1.5 s → 0.98 s. Experts with many rows still use the GEMM path, one expert
at a time.

End to end (in-process `dynalm benchmark`, prompt 128 / output 64, 10 threads), against a dense
model with a similar active size:

| model | quant | c=1 ITL p50 ms | c=4 ITL p50 ms | c=4 out tok/s |
|---|---|---|---|---|
| Granite-3.1-1B-A400M (MoE) | Q8_0 | **28.1** | 59.8 | 39.4 |
| Qwen2.5-0.5B (dense) | Q8_0 | 31.6 | 51.8 | 46.1 |
| Granite-3.1-1B-A400M (MoE) | Q4_K_M | 34.5 | 64.6 | 35.9 |
| Qwen2.5-0.5B (dense) | Q4_K_M | 30.4 | 48.7 | 46.9 |

How to read this:
- **Single stream.** One sequence touches only 8 of 32 experts per layer, so a 1.3B-parameter
  MoE decodes as fast as a 0.5B dense model.
- **Batched decode.** Different sequences route to different experts, so up to all 32 are read
  per step. MoE batching therefore gains less than dense batching.
- **Q4_K_M vs Q8_0.** Granite's Q4_K_M is slower than its Q8_0 at one stream: expert matrices
  are small (512 × 1024), and the Q4_K fused dot does more work per byte than Q8_0.
- **Cold start.** The first request after start-up page-faults the memory-mapped weights:
  16 s TTFT through the Docker bind mount, versus 0.8 s warm (TODO: prefetch at load).

## Phase 26 — sampling cost (`bench_sampling`, 151,936-entry row, flat N(0, 3²) logits)

| config | first version p50 µs | final p50 µs |
|---|---|---|
| greedy | 440 | **70** |
| temperature 1 | 661 | 580 |
| top_k 40 | 1180 | 844 |
| top_p 0.9 | 2789 | 878 |
| min_p 0.05 | 285 | 167 |
| top_k 40 + top_p 0.95 + repetition 1.1 | 1073 | 843 |

The row copy alone is 15 µs. Flat synthetic logits are the worst case for top-p, since the
nucleus is large. Worst case is under 1 ms per token, 2–3% of a Qwen2.5-0.5B decode step.

## Phase 27 — speculative decoding (`bench_speculative`, greedy, 96 new tokens, k = 4, 10 threads)

Target Qwen2.5-1.5B-Instruct Q4_K_M; draft model Qwen2.5-0.5B-Instruct Q4_K_M (same vocabulary).
The "quote" prompt asks the model to repeat a paragraph; the "open" prompt asks for a story.

| prompt | drafter | decode tok/s | speed-up | acceptance | tokens / target pass | output = plain greedy |
|---|---|---|---|---|---|---|
| quote | none | 13.8 | 1.00× | – | 1.00 | – |
| quote | n-gram | **21.4** | **1.55×** | 75% | 1.94 | yes |
| quote | 0.5B model | 11.4 | 0.83× | 75% | 3.96 | yes |
| open | none | 12.7 | 1.00× | – | 1.00 | – |
| open | n-gram | 12.3 | 0.96× | 11% | 1.03 | yes |
| open | 0.5B model | 10.0 | 0.78× | 47% | 2.79 | yes |

Same benchmark with Qwen2.5-0.5B Q8_0 as the target: n-gram gives 1.64× on "quote"
(90% acceptance) and 1.09× on "open".

How to read this:
- **Prompt lookup.** It is a clear win when the output repeats the input (edits, quoting,
  code refactors) and costs nothing measurable otherwise. It needs no extra model or memory.
- **Draft model.** It loses here, even with almost 4 tokens per target pass. A 0.5B draft is
  only about 2.4× cheaper per token than a 1.5B target on this machine (both are
  bandwidth-bound: 0.47 vs 1.1 GB). The draft's 4 sequential passes, plus a 5-row
  verification pass that costs more than a 1-row decode (expand path, DD-036), eat the gain.
  Draft models pay off when the cost ratio is around 10× or more (for example 0.5B drafting
  for 7B), or once the multi-row fused decode kernel (see ROADMAP.md) makes verification nearly free.

## P1 — measurement baseline (`tools/perf_sweep.sh full`, DD-050)

- **Machine:** i7-1255U (2P + 8E, 12 threads), 16 GB, Windows 11, native `dynalm.exe` (MSVC 19.44),
  10 compute threads, AC power, Balanced plan.
- **Not an idle box:** Docker/WSL and a browser were running, and only 1.7 GB of RAM was free.
- **Bandwidth ceiling:** measured DRAM read ceiling 18.5–20.9 GB/s. Hardware counters are
  unavailable (Windows).
- **Data:** raw points in `results/p1-baseline.jsonl`, graphs in [docs/perf/p1/](perf/p1/), and the
  environment in `docs/perf/p1/environment.txt`.
- **Workload:** prompt 128 / output 64 tokens unless stated.

**Aggregate output tok/s vs concurrency** (the program's headline metric):

| model | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 18.6 | 27.7 | 32.8 | 40.1 | 48.0 | 52.4 | 54.7 |
| Qwen2.5-0.5B Q8_0 | 22.0 | | | 67.5 | | | |
| Qwen2.5-0.5B F16 | 16.6 | | | 66.9 | | | |
| Qwen2.5-1.5B Q4_K_M | 9.8 | | 16.6 | | 22.2 | | |
| Qwen3-4B Q4_K_M | 3.1 | | 4.6 | | | | |
| Granite-3.1-1B-A400M Q4_K_M (MoE) | 34.3 | | 40.6 | | 58.5 | | 60.6 |

**Where the time goes, Qwen2.5-0.5B Q4_K_M:**

| c | decode rows/step | decode step ms | est. GB/s (of ~19) | ITL p50 / p99 ms | class |
|---|---|---|---|---|---|
| 1 | 1.0 | 45.4 | 8.7 | 45.8 / 63.9 | MIXED |
| 8 | 7.0 | 109.5 | 3.7 | 112.6 / 439.4 | COMPUTE_BOUND |
| 16 | 12.6 | 149.9 | 2.8 | 168.2 / 574.3 | COMPUTE_BOUND |
| 64 | 26.2 | 183.3 | 2.4 | 554.4 / 769.0 | COMPUTE_BOUND |

Per-op share of forward time, roughly the same at every concurrency:

| op | mlp_up | mlp_down | lm_head | qkv | attn_out | attention | everything else |
|---|---|---|---|---|---|---|---|
| share | 37–41% | 17–22% | 11–16% | 8–14% | 5–6% | 5–6% | < 10% |

**Findings** (in priority order for the next phases):

1. **Batched decode is compute-bound on K-quant unpacking, not on memory.**
   - From 1 to 26 rows per step, modelled bandwidth falls from 8.7 to 2.4 GB/s.
   - The same 7-row step costs 109.5 ms with Q4_K_M weights but 60.0 ms with Q8_0 and 65.1 ms
     with F16.
   - The difference is dequantizing Q4_K/Q6_K into fp32 panels for the multi-row (expand) path.
   - Matmuls are about 84% of forward time. This is the P3 (multi-row decode) / P4 (weight
     packing) target.
2. **Aggregate throughput flattens early:** 40 tok/s at c=8, 55 at c=64. At c=64 decode steps carry
   only 26 rows on average: prompt chunks (prefill budget 64/step, chunk cap 32) share the steps,
   and TTFT p50 reaches 35 s. Scheduling policy (P12) decides how much of each step goes to
   prefill.
3. **CPU utilization stays at 57–60% for Q4_K at every concurrency** (77–79% for Q8_0/F16).
   Thread-pool tail wait is 7–16% of forward time, plus serial scheduler work (0.2 → 2.5 ms per
   step for sampling and stream delivery as rows grow): P6 and P10.
4. **Context length costs single-stream decode:** 45 → 61 → 76 ms per step from 128 → 2048 → 4096
   tokens of context (attention): P5, P7.
5. **MoE is dispatch-heavy:** Granite runs 1,292–2,125 parallel regions per step against 313 for a
   dense model of similar depth: P11, P6.
6. **Single-stream Q4_K_M decode is near the bandwidth limit** (MEMORY_BOUND for 1.5B, 4B and
   Granite at c=1). Gains there come from moving fewer bytes (packing, KV), not from compute.

**Correction to the P1 baseline (found by P1's own diagnostics, DD-052).** The baseline process was
power-throttled by Windows (EcoQoS). The same engine ran its decode forward pass 1.6× faster inside
`dynalm run`. After opting compute threads out of throttling (`results/p1-qos.jsonl`):

| Qwen2.5-0.5B Q4_K_M | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|
| output tok/s, P1 baseline | 18.6 | 27.7 | 32.8 | 40.1 | 48.0 | 52.4 | 54.7 |
| output tok/s, with DD-052 | **31.5** | **45.5** | **54.2** | **71.1** | **78.9** | **86.7** | **88.5** |
| decode step ms (rows) | 26.2 (1) | 33.2 (2) | 47.9 (3.7) | 59.2 (7) | 88.6 (12.6) | 101.5 (21) | 101.0 (26.1) |
| class | MEMORY | MIXED | MIXED | COMPUTE | COMPUTE | COMPUTE | COMPUTE |

Every later phase compares against this corrected curve. The other P1 findings (relative costs of
K-quant unpacking, scheduling, context length and MoE dispatch) still hold, and are re-measured as
each phase runs.

## P2 — execution planner (DD-051)

- **Correctness:** outputs are bit-identical with and without the planner (`test_execution`:
  llama, gemma2 with sliding-window layers, qwen2), and the split-K and per-pair strategies agree
  when forced. 385/385 tests pass.
- **Cost:** all scheduler planning (cancellation, admission, prefix lookup, KV reservation, batch
  building, planning) takes 0.007 ms/step at c=1 and at most 0.09 ms/step at c=32, against 30–290 ms
  of forward time: performance-neutral.

## P3 (part 1) — decode matmul micro-kernel (`bench_decode_matmul`, Qwen2.5-0.5B Q4_K_M weights)

- **Method:** `bench_decode_matmul` times each matmul path for M = 1–32 on the model's real tensors,
  cycling through every layer's copy so the weights come from DRAM, as in a real step. A first
  version timed one cache-hot tensor and overstated throughput 2–3×; it was replaced.
- **The 4 weight rows × 3 activation rows micro-kernel:** 12 accumulators, 12 FMAs per 7 loads
  against 8 per 6 for the 4×2 tile. Each output keeps the same FMA chain, so results are
  bit-identical.
- **A/B:** median of 3 runs each, 10 threads, expand path. The 4×3 runs went first, so warm-up
  slightly favours them:

| tensor | type | M=3 | M=6 | M=12 | M=24 |
|---|---|---|---|---|---|
| ffn_gate [4864, 896] | q5_0 | −11.9% | −11.0% | −6.4% | −9.2% |
| ffn_down [896, 4864] | q6_K | −26.4% | 0.0% | −2.8% | −23.7% |
| lm_head [151936, 896] | q8_0 | −17.1% | −21.5% | −27.2% | −24.4% |

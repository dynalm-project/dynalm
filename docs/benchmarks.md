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

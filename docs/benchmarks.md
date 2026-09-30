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

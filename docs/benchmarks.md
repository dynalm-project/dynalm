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

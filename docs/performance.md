# Performance

Principles: profile before optimizing; report P50/P90/P95/P99, never only
averages; record the hardware and model for every number.

## Reference machine

| | |
|---|---|
| CPU | Intel Core i7-1255U (Alder Lake-U; 2 P-cores + 8 E-cores, 12 threads) |
| ISA | AVX2, FMA, F16C, AVX-VNNI. No AVX-512 |
| RAM | 16 GB |
| OS | Windows 11 Home (26200) |

Notes: This is a mobile part with hybrid cores and a low power limit. Throughput
depends on the power profile and on thermals. Barrier-synchronized kernels run
at E-core speed unless the work is partitioned unevenly or restricted to P-cores
(see DD-004).

## Current profile (Phase 6, generic kernels)

- Decode: 27 tok/s on SmolLM2-135M f16 at 10 threads. At 256 MiB of weights per token,
  that's about 7 GB/s effective, well below the ~50 GB/s memory bandwidth, so the
  generic kernels are compute/latency-bound rather than bandwidth-bound.
- Prefill: 33 tok/s. There is no GEMM blocking, and every weight row is re-converted
  from fp16 per call.
- Planned order (Phase 17/18): AVX2+F16C dot kernels → quantized dot kernels → tiled
  GEMM for prefill → attention over contiguous block runs → operator fusion where
  measured.

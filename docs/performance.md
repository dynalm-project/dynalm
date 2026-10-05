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

## Current profile (Phase 21)

Qwen2.5-0.5B Q4_K_M at 10 threads, served over HTTP (docs/benchmarks.md, Phase 21):

- **Single stream:** about 30 ms per token decode, and roughly 200 tok/s prefill through the
  server (TTFT 2.7 s at 512 tokens). This is at parity with llama.cpp in the same container.
- **Under concurrency:** throughput is at parity or up to 25% ahead of llama.cpp, and TTFT is
  2–3× lower (chunked prefill).
- **Gap:** per-step decode cost at 4–16 sequences (TPOT 66 vs 51 ms at c=4).
- **Matmul path selection (DD-036):** batched matmuls expand weight panels from 2 rows
  (fused dequantize-dot only for m=1). Dequantization is AVX2 for the common formats.
- **Next levers, in expected-gain order:**
  - a multi-row fused decode kernel (weights decoded once per batch, no fp32 panel);
  - GQA-grouped decode attention;
  - int8 activations with VNNI for prefill (DD-033).

## Performance program (P1–P16)

The goal is the most **aggregate output tokens/s under concurrency** on the same hardware and
model, with correctness, p99 and memory under control. The phases and their status are listed
in [ROADMAP.md](../ROADMAP.md). Every optimization starts from a measured bottleneck and reports
Before / After / Delta for the same workload.

### How to measure (P1)

```sh
tools/perf_sweep.sh full results/<name>       # matrix -> sweep.jsonl, report.md, graphs/*.svg
dynalm benchmark <model> --concurrency 1,2,4,8,16,32,64 --prompt 128 --output 64
dynalm benchmark <model> --prompt-mix 64,512,2048 --concurrency 8,32
```

`dynalm benchmark` measures each in-process point as follows.

**Client side:**
- aggregate and per-request output tok/s;
- TTFT, ITL, TPOT and E2E at P50/P90/P95/P99;
- CPU utilization and peak RSS.

**Engine side** (accumulated by the scheduler on every step, a few clock reads each):

| Field | Meaning |
|---|---|
| `plan` | Cancellation, admission and batch building, which include `prefix_lookup` and `kv_reserve` |
| `forward` | Split into decode-only, prefill-only and mixed steps |
| `sample` | Sampling and stop/EOS checks |
| `emit` | Stream delivery, including detokenization |
| `prefix_insert` | Offering completed blocks to the prefix cache |
| `tokenize` | Prompt tokenization on the caller threads |
| mean queue wait | Time from submission to admission |
| steps by composition | Decode-only / prefill-only / mixed counts, mean decode rows per step, mean decode-step time |

**Profiling** (switched on for every benchmark point; costs about 1 µs per forward op):
- per-op forward time (`embed … lm_head`);
- thread-pool regions per step and mean region length;
- the caller's **tail wait** at region ends, i.e. synchronization and imbalance;
- worker sleeps.

**Counters:**
- Context switches and page faults, from OS process accounting.
- Hardware counters: cycles, instructions (IPC), LLC references/misses, L1D misses, branch
  misses, CPU migrations. These come from Linux `perf_event_open`, opened per thread. On
  Windows, macOS and VMs without a virtual PMU (e.g. WSL2) they read `null`, and the report says
  why.
- Mean CPU clock during the run (Linux sysfs / `/proc/cpuinfo`; Windows
  `CallNtPowerInformation`, which on many laptops reports a nominal rather than live clock).

**Memory bandwidth:**
- A measured DRAM read ceiling: all threads stream a 256 MiB buffer, best of 3 passes.
- A **modelled** decode traffic estimate: weights read once per step (untied input embedding
  only for gathered rows; MoE only the expected touched experts, 1 − (1 − k/E)^rows), plus each
  row's KV at the mean context, divided by the measured decode-step time.
- It is labelled as an estimate everywhere. Real DRAM counters need uncore PMU access.

### Bottleneck classes (DD-050)

Rules are checked in this priority order. The first that fires is the primary class; all are
listed with evidence.

| Class | Rule |
|---|---|
| IO_BOUND | Major page faults read ≥ 50 MB/s from disk |
| SYNCHRONIZATION_BOUND | Caller tail wait ≥ 25% of forward time and regions average < 50 µs |
| LOAD_IMBALANCED | Caller tail wait ≥ 25% and regions average ≥ 50 µs (uneven work, not hand-off cost) |
| DISPATCH_BOUND | ≥ 150 regions per step averaging < 20 µs, or scheduler work outside the forward pass ≥ 20% of step time |
| MEMORY_BOUND | Decode ≥ 50% of forward time and modelled bandwidth ≥ 60% of the measured ceiling |
| CACHE_BOUND | LLC miss ratio ≥ 30% while DRAM use < 60% (needs hardware counters) |
| COMPUTE_BOUND | Prefill ≥ 50% of forward time with bandwidth < 50% and no synchronization signal; or IPC ≥ 2; or decode ≥ 50% with bandwidth < 35% and nothing else firing (dequantization/arithmetic) |
| MIXED | No rule fires |

Thresholds are deliberately round numbers. They are a triage aid that points at the next thing
to measure, not a verdict, and each optimization report quotes the underlying figures, not just
the class.

# DynaCore compiler benchmarks

```
DynaCore Compiler Benchmark
──────────────────────────────────────────────────────────────
Model: Qwen2.5-1.5B-Instruct Q4_K_M      CPU: Intel i7-1255U, 10 threads, AVX2
dynalm benchmark, prompt 128, output 64, 4 runs per mode (ABBA order)

                     output tok/s           ITL p50        TTFT p50
                 c=1     c=4     c=8      c=1    c=4      c=1     c=4
reference       9.07   15.36   18.27     81.0  152.4     1919    4162
compiled        9.79   15.43   18.51     72.4  144.1     1858    4044
change         +7.9%   +0.5%   +1.3%   -10.6%  -5.4%    -3.2%   -2.8%
```

Raw data: `results/compiler-e2e-reference.jsonl` and `results/compiler-e2e-compiled.jsonl`.

## Reading it

**Single stream (c=1): faster.**

- Output tok/s went up by 7.9%, and the per-token latency (ITL p50) went down by 10.6%.
- Paired by run, the four gains were +14.2%, −2.1%, +4.5% and +14.1%. Noise on this laptop
  is ±5–10%, so individual runs vary widely.

**Concurrency 4 and 8: neutral.**

- Aggregate throughput there is dominated by prefill and mixed steps. The compiler
  deliberately leaves prefill-shaped matmuls unfused (DD-072).
- The decode-step gain remains visible in ITL p50 (−5% at c=4) but is diluted in aggregate
  tok/s.
- ITL p99 at c≥4 (~1 s) comes from prefill chunks in the same steps in both modes.

## Decode steps in isolation (`bench_compiled`)

- Setup: step-interleaved, so every variant sees the same thermal state; 40 steps per
  variant, 4 rounds; context 256. Data: `results/compiler-bench_compiled-ab.csv`.

| Variant | 1 sequence | 4 sequences |
|---|---|---|
| reference (C++ device) | 0 | 0 |
| IR only (record + cached plan) | ±2% | ±2% |
| Q/K/V group + gated MLP | −4.1 / −7.7 / −8.6 / −9.1% | −1.6 / −4.3 / −5.8 / −6.0% |

## One layer in the DynaCore language (`dynacorec --benchmark`)

| Graph | Unplanned | Compiled | Change |
|---|---|---|---|
| `examples/dynacore/decoder_layer.dyna`, M=1 | 2709.7 µs | 2368.2 µs | −12.6% |
| same, M=64 (nothing fused by rule) | 30764.5 µs | 30705.1 µs | −0.2% |

## Accuracy

- `bench_compiled`: the maximum absolute logit difference is 0 for every variant.
- `test_compiled`: generation and logits are bit-identical for all 11 tiny architectures.
- `dynacorec --benchmark`: outputs are bit-identical.

The fusions reuse the same dot-product kernels and activation, so no tolerance was needed or
loosened.

## The rejected version (kept for the record)

The first end-to-end run also grouped prefill-shaped matmuls:
`results/compiler-e2e-v0-*.jsonl`.

| c | reference | compiled |
|---|---|---|
| 4 | 15.67–16.81 tok/s | 15.34–16.77 tok/s |
| 8 | 18.63–19.31 tok/s | 18.30–19.21 tok/s |

Medians were −4% to −7%. The cause is that `matmul_many`'s 16-row panels lack the K-blocked
GEMM. The rule now fuses only matmuls on the int8 decode path.

## Reproduce

```sh
build/msvc-release/dynalm/benchmarks/bench_compiled models/qwen2.5-1.5b-instruct-q4_k_m.gguf 10 256 1,4 4 ref,ir,group,gated,all 40
for x in reference compiled compiled reference; do
  dynalm -q benchmark models/qwen2.5-1.5b-instruct-q4_k_m.gguf --execution $x --concurrency 1,4,8 \
    --prompt 128 --output 64 --out e2e_$x.jsonl
done
dynacorec examples/dynacore/decoder_layer.dyna --benchmark
```

## Status

- `--execution compiled` is opt-in. It is recommended for single-stream use: chat and agents
  with one user.
- It becomes the default once a larger model (Qwen3-4B) and another CPU confirm the
  single-stream gain without regressions.

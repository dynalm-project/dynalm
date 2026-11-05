# DynaLM Benchmark Results: Speed, Memory and Accuracy

DynaLM against llama.cpp's server on the same GGUF model files, measured with the same client
(`dynalm benchmark --url`). Above 1× means DynaLM is faster. How the measurements are made:
[benchmarking.md](benchmarking.md).

## Apple Silicon CPU (Apple M4 Pro)

| Workload | DynaLM vs llama.cpp |
|---|---|
| Qwen2.5-1.5B, 4 users: total tokens/s | **1.95×** |
| Qwen2.5-1.5B, 2,048-token prompt: whole request | **1.78×** |
| Qwen2.5-1.5B, 8 users: total tokens/s | **1.68×** |
| Qwen3-4B, 2,048-token prompt: first token | **2.26×** |
| Qwen3-4B, one user: output tokens/s | **1.46×** |
| Qwen3-4B, 2,048-token prompt: time between tokens | **1.23×** |
| Qwen2.5-1.5B, 8 users: 95th-percentile time between tokens | **1.19×** |
| Qwen2.5-1.5B, one user: output tokens/s | **1.08×** |
| **Average (geometric mean)** | **1.53×** |

| Peak server memory | DynaLM | Less than llama.cpp |
|---|---|---|
| Qwen2.5-1.5B | 1,508 MiB | 2.73× |
| Qwen3-4B | 3,944 MiB | 2.89× |

- Against Ollama: 1.53× on the Qwen3-4B workloads.
- Across 19 model shapes (architecture, size and quantization): first token on a 2,048-token
  prompt 1.9× faster, 4 users 1.5× faster, one user 1.2× faster, with 2.5× less memory
  ([perf/cpu-model-sweep.md](perf/cpu-model-sweep.md)).

Setup: Apple M4 Pro (12-core CPU), 24 GB, on AC power. Qwen2.5-1.5B and Qwen3-4B (Q4_K_M);
prompts of 128, 512 and 2,048 tokens; 1, 4 and 8 users; 128 output tokens; 5 rounds, medians.
llama.cpp commit `9a16f1dbe`, built with Apple Accelerate. Details:
[perf/cpu-suite-v1.md](perf/cpu-suite-v1.md), [perf/cpu-experiments.md](perf/cpu-experiments.md).

## x86-64 CPU (AMD EPYC, AVX2)

| Model, one user | Time per output token: DynaLM / llama.cpp | DynaLM vs llama.cpp |
|---|---|---|
| Qwen3-4B with EAGLE-3 | 90.7 / 127.6 ms | **1.41×** |
| Qwen3-8B with EAGLE-3 | 154.2 / 206.9 ms | **1.34×** |
| Qwen2.5-1.5B, no draft head | 39.2 / 45.0 ms | **1.15×** |
| Qwen3-4B, no draft head | 93.7 / 109.5 ms | **1.17×** |
| Qwen3-8B, no draft head | 160.5 / 172.3 ms | **1.07×** |

DynaLM 0.6.0 with `--mode exact`; the EAGLE-3 rows add `--spec-eagle` with the AngelSlim Qwen3-4B
and Qwen3-8B heads and up to 3 drafted tokens per step.

Setup: a 4-core AMD EPYC (Zen 2) cloud server, 16 GB, Ubuntu 24.04. Q4_K_M models, 512-token
prompt, 128 output tokens. llama.cpp b11695 (`llama-server`, 4 threads). Details and raw data:
[perf/x86-zen2-2026-11.md](perf/x86-zen2-2026-11.md).

## Apple Silicon GPU (Metal, Apple M4 Pro)

| Workload | DynaLM vs llama.cpp |
|---|---|
| 16 users: slowest time between tokens (99th percentile), 8B models | **7–10× shorter** |
| 16 users: first token (median) | **2.1–3.3× sooner** |
| One user | about the same speed |

Setup: Qwen2.5-1.5B, Qwen3-4B and Qwen3-8B (Q4_K_M) and Llama-3.1-8B (Q8_0); 512-token prompts,
128 output tokens, 2,048 tokens of context per user; 3 rounds, medians. llama.cpp commit
`9a16f1dbe`.

## Accuracy

DynaLM's rule (DD-053): perplexity at most 1% above the fp32 reference and mean KL divergence at
most 0.0025.

| Check | Result |
|---|---|
| Apple M4 Pro CPU, Qwen2.5-1.5B and Qwen3-4B | same next token as llama.cpp at every position, mean KL ≤ 4e-7 |
| 45-model sweep, Apple M4 Pro CPU | same next token as llama.cpp on 94–99% of positions (llama.cpp against itself: 94–100%) |
| AMD EPYC, AVX2 kernels, Qwen2.5-1.5B | perplexity +0.20%, mean KL 0.0011 against fp32 |
| AMD EPYC, AVX2 kernels, Qwen3-4B | perplexity +0.01%, mean KL 0.0005 against fp32 |
| GPU (Metal, Vulkan) | within the rule against the CPU reference |

EAGLE-3 speculative decoding checks every drafted token with the full model, so the output is the
same as without it.

## Measure your own machine

```bash
dynalm cpu-report            # DynaLM against llama.cpp on this CPU, writes a shareable report
dynalm benchmark qwen2.5:1.5b
```

To compare with another engine, run it as a server on the same GGUF file and measure both with
the same client:

```bash
dynalm benchmark qwen2.5:1.5b --url http://127.0.0.1:8080 --concurrency 1,4,16 \
  --prompt 512 --output 128 --warmup-requests 1 --model-name <name the server expects>
```

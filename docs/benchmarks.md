# DynaLM Benchmark Results: Speed, Memory and Accuracy

> **Historical results, withdrawn as current claims.** These were measured on a development build
> that became DynaLM 0.1.5 (GPU path as 0.1.6). DynaLM has changed since, and a later end-to-end run
> did not reproduce the multi-user time-to-first-token advantage. They no longer back any claim in
> the README and are kept, unchanged below, for reference until the re-measurement for this
> release replaces them.

Every number here was measured; nothing is estimated. The method is in
[benchmarking.md](benchmarking.md).

## Setup

| Item | Value |
|---|---|
| Machine | Apple M4 Pro (12-core CPU: 8 performance + 4 efficiency; 16-core GPU), 24 GB unified memory |
| OS | macOS 26.7 (build 25G229) |
| Power | On AC power for every run. The Energy Mode setting for AC is "Automatic" (`pmset powermode 0`). It was not logged during the runs; this is the machine's setting. |
| Model | Qwen2.5-1.5B-Instruct, Q4_K_M, official GGUF `qwen2.5-1.5b-instruct-q4_k_m.gguf`, SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| DynaLM | Development build of the Metal backend that became 0.1.5. Its GPU code path matches 0.1.6. ggml 0.27.0 (`a05c992`). |
| DynaLM settings | One user: `dynalm serve <model> --backend metal --ctx 4096`. Several users: the same with `--ctx 16384`. Default scheduler (balanced, GPU profile). |
| llama.cpp | Commit `9a16f1dbe4722a93ab75a2d15e29aa71ef764c7b`, `llama-server` built from source with Metal; it uses the same ggml 0.27.0, so both engines run the same Metal kernels |
| llama.cpp settings | One user: `-ngl 99 -c 4096 -np 1 -fa auto`. Several users: `-ngl 99 -c 262144 -np 16 -fa auto` (16 slots of 16,384 tokens) |
| Ollama | 0.40.5, run on Metal (`ollama ps`: 100% GPU); see "Not compared" |
| Client | The same load generator for every engine: `dynalm benchmark --url` |
| Requests | Tokenizer-exact prompts, 128 output tokens with `ignore_eos`, greedy. One warm-up request per point. One user: 4 requests per point; several users: 32 per point. |
| Rounds | 3, with the engine order rotated each round |

**Verdict rule:**
- "better" or "slower" only when all rounds agree beyond ±10% and both engines varied ≤ 10%;
- "parity" within ±10%;
- otherwise "inconclusive".

## One user

| Prompt | Result |
|---|---|
| 128 tokens | time to first token 0.92× of llama.cpp's speed; 7.1 vs 6.4 ms per token: llama.cpp about 10% faster |
| 512 to 3,584 tokens | inconclusive: both engines varied 10–25% between rounds |

## Several users (512-token prompts, 128 output tokens)

Medians over 3 rounds, DynaLM / llama.cpp:

| Metric | 4 users | 8 users | 16 users |
|---|---|---|---|
| Output tokens/s | 155 / 164 (parity) | 163 / 198 (slower) | 161 / 244 (slower) |
| First token, median | 588 / 1,135 ms (better) | 590 / 1,544 ms (better) | 715 / 2,499 ms (inconclusive) |
| First token, 95th percentile | 1,157 / 1,388 ms (better) | 3,792 / 2,945 ms (slower) | 10,942 / 5,483 ms (slower) |
| Gap between tokens, median | 12.6 / 12.3 ms (parity) | 14.3 / 15.1 ms (parity) | 25.4 / 22.1 ms (slower) |
| Gap between tokens, 99th percentile | 128 / 26 ms (slower) | 144 / 26 ms (slower) | 153 / 1,256 ms (better) |
| Whole request, median | 3,366 / 2,826 ms (slower) | 5,982 / 4,620 ms (slower) | 11,352 / 8,234 ms (slower) |

**In short:**
- DynaLM usually starts answering sooner.
- llama.cpp finishes requests sooner, and delivers more tokens per second at 8 or more users.

## Memory (peak footprint)

| Load | DynaLM | llama.cpp | Comparable? |
|---|---|---|---|
| One user | 389–393 MB | 1,113–1,117 MB | Yes: both hold 4,096 tokens of KV cache |
| 16 users | 883–897 MB | 8,841–8,899 MB | **No**: llama.cpp reserved 16 × 16,384 tokens of cache; DynaLM shares 16,384 |

DynaLM uses the model file in place (memory-mapped) instead of copying it to the GPU.

## Accuracy (GPU against the CPU reference)

The rule: perplexity change ≤ 1% and mean KL divergence ≤ 0.0025.

| Model | GPU | Decode | Prompt reading | Greedy match |
|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | Metal | +0.011% | +0.150% | 4 of 4 prompts |
| Qwen2.5-1.5B Q4_K_M | Vulkan (MoltenVK) | +0.015% | −0.010% | 4 of 4 prompts |
| Llama-3.2-1B Q4_K_M | Metal | −0.004% | −0.112% | 4 of 4 prompts |

All pass, and the GPU results are deterministic run to run.

## CPU speed (unverified)

There is **no fair, repeatable CPU comparison yet**. The CPU numbers seen so far were single
runs, on battery in Low Power Mode, with other apps using the machine. Do not quote them. A
CPU benchmark against llama.cpp's CPU mode is on the task list.

## Not compared

- **Ollama:** it ran on the GPU, but its results are not comparable. It stops answers early
  (it ignores `ignore_eos`), and it re-templated the prompt (512 → 541 tokens).
- **Not measured yet:**
  - CUDA on NVIDIA GPUs, and Vulkan on discrete AMD, Intel or NVIDIA GPUs;
  - larger models (7B and up).

## Reproduce

Run DynaLM and the other engine as servers on the same GGUF file. Then measure both with
the same client:

```bash
dynalm benchmark qwen2.5:1.5b --url http://127.0.0.1:8080 --concurrency 1,4,8,16 \
  --prompt 512 --output 128 --warmup-requests 1 --model-name <name the server expects>
```

To measure DynaLM alone on your machine: `dynalm benchmark qwen2.5:1.5b`.

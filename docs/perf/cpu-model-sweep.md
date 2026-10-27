# CPU model sweep: DynaLM 0.3.0 vs llama.cpp

**Status: in progress.** 28 of 45 models done (27 measured, 1 could not be
downloaded); speed for 19 of 27 model shapes. This page is updated as the sweep continues.

One machine: Apple M4 Pro, 24 GiB, macOS 27.0.1, AC power. DynaLM 0.3.0 (release binary), llama.cpp `9a16f1dbe` (CPU build with Apple
Accelerate, the same build as [CPU suite v1](cpu-suite-v1.md)). Same GGUF file for both engines. Measured 2026-10-26/27.

## Summary

- **Every model that downloaded ran correctly** (27 of 27): it loaded, matched llama.cpp's next
  token on 94.4–98.9 % of positions (median 97.1 %), and generated text.
- **First token on a 2,048-token prompt: 1.91× faster** than llama.cpp on average
  (1.23–2.67×; faster on 18 of 18 models).
- **Output tokens per second, one user: 1.21×** (0.77–1.66×; faster on 15 of 19).
  Slower on: Qwen2.5 0.5B Instruct, SmolLM2 1.7B Instruct, SmolLM2 360M Instruct, Gemma 3 1B IT.
- **Total tokens per second, 4 users: 1.50×** (0.94–2.08×; faster on 18 of 19).
  Slower on: SmolLM2 1.7B Instruct.
- **Peak memory: 2.5× less** (1.6–4.6×; less on all 19).
- **Where DynaLM is worse:** the slowest token gaps (p99) are longer than llama.cpp's on 14 of 19 models
  (Qwen2.5 0.5B Instruct, Llama 3.2 1B Instruct, Qwen3 0.6B, Qwen2.5 1.5B Instruct, SmolLM2 1.7B Instruct, TinyLlama 1.1B Chat, SmolLM2 360M Instruct, Qwen2.5 3B Instruct, Qwen3 4B, Llama 3.2 3B Instruct, Gemma 3 1B IT, Gemma 3 4B IT, DeepSeek Coder 1.3B Instruct, Mistral 7B Instruct v0.3), mostly the small ones. The median gap is shorter or equal on 14 of 19.

Averages are geometric means over the measured models. One machine only; other CPUs use ggml's kernels by
default and were not measured.

## Method (fixed before the first result)

- **Accuracy, every model.** llama.cpp's `llama-perplexity` scores the wikitext-2 test set (10 chunks of
  512 tokens, second half of each chunk scored, CPU: `-ngl 0 -dev none`, 8 threads) and writes its
  log-probabilities; DynaLM's `bench_ppl` (default CPU path, `CPU/neon+qmm/neon_i8mm+sme_f32` on this
  machine) scores the same tokens against them. Reported: top-1 agreement, mean KL divergence and both
  perplexities. For scale: llama.cpp against itself, with only its batch size changed, agreed on
  94–100 % on the seven models tested on 2026-10-20. This measures agreement with llama.cpp, not
  answer quality. The 100 % reported for CPU suite v1 came from a different llama.cpp reference run;
  this method gives 95.3 % and 97.7 % on the same two files.
- **Generation, every model.** `dynalm run --cpu --raw -n 32 --temp 0` must print text.
- **Speed, one model per architecture, size and quantization** (models listed under "same shape" share
  it and were not measured separately): `tools/bench/baseline.sh`, DynaLM against `llama-server`, CPU,
  2 rounds with engine order varied, 8 threads, 4,096 tokens of context per user, 128 output tokens
  (`ignore_eos`), unique prompts. Points: one user with a 512-token prompt (output tokens/s, token gaps),
  one user with a 2,048-token prompt (first token), 4 users with 512-token prompts (total tokens/s).
  Every server start waited for AC power and a 1-minute load average below 3. Downloads were paused
  during speed runs. Values are the median of the 2 rounds.

## Accuracy

| # | Model | File | Top-1 vs llama.cpp | Mean KL | PPL DynaLM | PPL llama.cpp |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5 0.5B Instruct | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 97.65 % | 0.0016 | 17.661 | 17.735 |
| 2 | Llama 3.2 1B Instruct | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 97.41 % | 0.0017 | 16.513 | 16.523 |
| 3 | Qwen3 0.6B | `Qwen3-0.6B-Q8_0.gguf` | 97.33 % | 0.0014 | 23.779 | 23.724 |
| 4 | Qwen2.5 1.5B Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 95.29 % | 0.0068 | 12.077 | 12.188 |
| 5 | Qwen3 1.7B | `Qwen3-1.7B-Q8_0.gguf` | 97.61 % | 0.0018 | 18.482 | 18.494 |
| 6 | SmolLM2 1.7B Instruct | `smollm2-1.7b-instruct-q4_0.gguf` | 94.78 % | 0.0055 | 12.212 | 12.317 |
| 7 | Qwen2.5 Coder 1.5B Instruct | `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` | 96.12 % | 0.0034 | 16.396 | 16.420 |
| 8 | TinyLlama 1.1B Chat | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 97.10 % | 0.0024 | 25.325 | 25.395 |
| 9 | SmolLM2 360M Instruct | `SmolLM2-360M-Instruct.Q4_K_M.gguf` | 96.55 % | 0.0025 | 16.072 | 16.099 |
| 10 | Qwen2.5 3B Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | 94.98 % | 0.0101 | 10.410 | 10.528 |
| 11 | Qwen3 4B | `Qwen3-4B-Q4_K_M.gguf` | 97.69 % | 0.0031 | 18.626 | 18.741 |
| 12 | Qwen3 4B Instruct 2507 | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 96.82 % | 0.0045 | 12.368 | 12.405 |
| 13 | Llama 3.2 3B Instruct | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 98.31 % | 0.0013 | 12.469 | 12.522 |
| 14 | Gemma 3 1B IT | `google_gemma-3-1b-it-Q4_K_M.gguf` | 97.33 % | 0.0037 | 34.152 | 34.325 |
| 15 | Gemma 3 4B IT | `google_gemma-3-4b-it-Q4_K_M.gguf` | 96.51 % | 0.0102 | 19.994 | 19.692 |
| 24 | Qwen2.5 Coder 0.5B Instruct | `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | 98.27 % | 0.0009 | 39.603 | 39.603 |
| 25 | Qwen2.5 Coder 3B Instruct | `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` | 95.69 % | 0.0068 | 12.179 | 12.362 |
| 28 | DeepSeek Coder 1.3B Instruct | `deepseek-coder-1.3b-instruct.Q4_K_M.gguf` | 96.94 % | 0.0024 | 34.695 | 34.595 |
| 16 | Qwen2.5 7B Instruct | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 96.51 % | 0.0050 | 7.286 | 7.317 |
| 17 | Qwen3 8B | `Qwen3-8B-Q4_K_M.gguf` | 97.14 % | 0.0052 | 11.619 | 11.638 |
| 18 | Llama 3.1 8B Instruct | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 98.12 % | 0.0014 | 8.809 | 8.821 |
| 19 | Mistral 7B Instruct v0.3 | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | 97.73 % | 0.0015 | 7.608 | 7.617 |
| 20 | Gemma 2 9B IT | `gemma-2-9b-it-Q4_K_M.gguf` | 98.86 % | 0.0013 | 10.400 | 10.433 |
| 21 | Llama 3 8B Instruct | `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | 98.39 % | 0.0014 | 10.585 | 10.592 |
| 23 | DeepSeek R1 Distill Llama 8B | `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf` | 94.43 % | 0.0356 | 16.717 | 16.313 |
| 26 | Qwen2.5 Coder 7B Instruct | `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` | 97.65 % | 0.0016 | 10.802 | 10.812 |
| 29 | DeepSeek Coder 6.7B Instruct | `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` | 96.75 % | 0.0041 | 20.812 | 21.021 |

Not measured: #22 DeepSeek R1 Distill Qwen 7B (download failed (curl exit 56)).

## Speed

DynaLM / llama.cpp; in brackets how many times better DynaLM is (above 1× = DynaLM better).

| Model | Same shape | Output tok/s, 1 user | First token ms, 2,048 prompt | Total tok/s, 4 users | Token gap p50 ms | Token gap p99 ms | Peak memory GiB |
|---|---|---|---|---|---|---|---|
| Qwen2.5 0.5B Instruct | Qwen2.5 Coder 0.5B Instruct | 139.5 / 175.2 (0.80×) | 834 / 1,343 (1.61×) | 278.9 / 134.4 (2.08×) | 5.1 / 3.4 | 12.5 / 4.3 | 0.62 / 1.18 (1.90×) |
| Llama 3.2 1B Instruct | — | 125.7 / 104.5 (1.20×) | 1,581 / 2,929 (1.85×) | 182.3 / 118.0 (1.54×) | 4.7 / 5.0 | 7.8 / 5.1 | 1.08 / 2.91 (2.69×) |
| Qwen3 0.6B | — | 158.1 / 145.3 (1.09×) | 1,096 / 1,805 (1.65×) | 254.9 / 221.6 (1.15×) | 4.3 / 4.7 | 7.1 / 4.9 | 1.22 / 5.52 (4.52×) |
| Qwen2.5 1.5B Instruct | Qwen2.5 Coder 1.5B Instruct | 103.3 / 77.9 (1.33×) | 2,449 / 4,241 (1.73×) | 128.6 / 85.1 (1.51×) | 5.4 / 6.3 | 9.6 / 6.5 | 1.34 / 3.04 (2.27×) |
| Qwen3 1.7B | — | 73.2 / 65.6 (1.12×) | 2,330 / 3,801 (1.63×) | 116.1 / 99.1 (1.17×) | 9.4 / 9.8 | 14.1 / 14.3 | 2.37 / 6.98 (2.95×) |
| SmolLM2 1.7B Instruct | — | 73.9 / 77.1 (0.96×) | 3,153 / 3,870 (1.23×) | 91.5 / 97.8 (0.94×) | 6.9 / 7.0 | 17.5 / 7.7 | 1.96 / 7.88 (4.02×) |
| TinyLlama 1.1B Chat | — | 140.2 / 103.9 (1.35×) | — / — (—) | 193.2 / 117.3 (1.65×) | 3.8 / 4.7 | 6.8 / 4.8 | 0.84 / 1.90 (2.26×) |
| SmolLM2 360M Instruct | — | 147.3 / 182.3 (0.81×) | 875 / 1,499 (1.71×) | 276.8 / 203.7 (1.36×) | 4.8 / 3.1 | 11.5 / 4.5 | 0.51 / 1.99 (3.90×) |
| Qwen2.5 3B Instruct | Qwen2.5 Coder 3B Instruct | 53.6 / 38.9 (1.38×) | 4,093 / 9,080 (2.22×) | 71.0 / 42.5 (1.67×) | 10.7 / 11.6 | 18.3 / 12.9 | 2.34 / 5.09 (2.18×) |
| Qwen3 4B | Qwen3 4B Instruct 2507 | 38.9 / 28.9 (1.35×) | 5,703 / 12,329 (2.16×) | 50.3 / 31.0 (1.62×) | 14.7 / 15.8 | 27.2 / 22.2 | 3.24 / 9.12 (2.81×) |
| Llama 3.2 3B Instruct | — | 53.1 / 38.6 (1.38×) | 4,060 / 9,210 (2.27×) | 70.0 / 41.3 (1.69×) | 11.0 / 12.0 | 18.8 / 12.3 | 2.62 / 7.76 (2.96×) |
| Gemma 3 1B IT | — | 74.4 / 96.6 (0.77×) | 1,648 / 2,024 (1.23×) | 131.9 / 120.3 (1.10×) | 9.6 / 5.9 | 23.2 / 9.5 | 1.01 / 2.27 (2.25×) |
| Gemma 3 4B IT | — | 41.5 / 31.9 (1.30×) | 5,049 / 10,130 (2.01×) | 54.1 / 34.8 (1.55×) | 14.0 / 14.9 | 24.0 / 17.4 | 3.18 / 8.24 (2.59×) |
| DeepSeek Coder 1.3B Instruct | — | 88.7 / 83.3 (1.06×) | 2,013 / 3,770 (1.87×) | 127.2 / 91.6 (1.39×) | 7.2 / 6.3 | 14.0 / 6.5 | 1.84 / 8.40 (4.57×) |
| Qwen2.5 7B Instruct | Qwen2.5 Coder 7B Instruct | 24.0 / 14.5 (1.66×) | 8,930 / 20,445 (2.29×) | 30.3 / 17.5 (1.73×) | 23.0 / 28.7 | 54.2 / 156.5 | 5.00 / 9.04 (1.81×) |
| Qwen3 8B | — | 23.4 / 15.3 (1.53×) | 9,699 / 22,531 (2.32×) | 25.1 / 15.6 (1.61×) | 24.8 / 26.8 | 37.8 / 73.6 | 5.63 / 9.63 (1.71×) |
| Llama 3.1 8B Instruct | Llama 3 8B Instruct, DeepSeek R1 Distill Llama 8B | 23.9 / 15.6 (1.53×) | 9,475 / 22,603 (2.39×) | 28.7 / 16.5 (1.74×) | 24.2 / 26.0 | 36.0 / 44.9 | 5.48 / 9.85 (1.80×) |
| Mistral 7B Instruct v0.3 | — | 23.2 / 16.8 (1.38×) | 9,685 / 22,265 (2.30×) | 29.5 / 17.3 (1.71×) | 24.3 / 23.9 | 52.4 / 34.2 | 4.95 / 8.97 (1.81×) |
| Gemma 2 9B IT | — | 17.2 / 10.8 (1.59×) | 12,651 / 33,817 (2.67×) | 22.1 / 12.3 (1.80×) | 34.6 / 36.1 | 56.7 / 93.5 | 7.23 / 11.53 (1.59×) |

Notes: TinyLlama 1.1B Chat: 2,048-token prompt exceeds the model's 2,048-token context: not measurable in either engine.

Data: [data/cpu-model-sweep.json](data/cpu-model-sweep.json).

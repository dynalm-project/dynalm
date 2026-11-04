## DynaLM CPU report

- **CPU:** DO-Premium-AMD (sse4.2 avx2 fma f16c), 4 threads
- **OS:** Linux Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic (x86_64), RAM 15.6 GB
- **DynaLM:** 0.5.0; **llama.cpp:** llama.cpp b11695 (pinned release, sha256 7e77d0c9f4982658…)
- **Workload:** prompt 512, 128 output tokens, 2 round(s), medians
- **Run:** **uncontrolled** (see notes): compare with care
- uncontrolled: machine load stayed above 1.00
- 1.5b llama.cpp round 1: server did not start
- 1.5b llama.cpp round 2: server did not start
- 4b llama.cpp round 1: server did not start
- 4b llama.cpp round 2: server did not start
- 8b llama.cpp round 1: server did not start
- 8b llama.cpp round 2: server did not start

### 1.5b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 11.1 | – | 5854 ms | 44.8 / 51.5 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 9.0 | – | 7373 ms | 53.2 / 61.4 ms |
| DynaLM default (ggml kernels) | 4 | 11.7 | – | 14174 ms | 113.2 / 617.7 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 12.8 | – | 16501 ms | 83.2 / 540.2 ms |

### 4b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 3.9 | – | 16179 ms | 127.6 / 155.4 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 3.5 | – | 20438 ms | 128.1 / 142.1 ms |
| DynaLM default (ggml kernels) | 4 | 4.1 | – | 47877 ms | 331.2 / 1720.5 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 4.8 | – | 44247 ms | 206.9 / 1505.9 ms |

### 8b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 2.4 | – | 28667 ms | 194.4 / 221.5 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 2.0 | – | 34692 ms | 222.2 / 243.2 ms |
| DynaLM default (ggml kernels) | 4 | 2.5 | – | 64833 ms | 499.4 / 2770.0 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 2.8 | – | 75235 ms | 348.5 / 2534.2 ms |

### Agreement with llama.cpp (same tokens)

| Path | Model | Top-1 | Mean KL | PPL (llama.cpp) | Verdict |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1.5b | – | – | – | not run: llama-perplexity failed (log: /root/.dynalm/cpu-report/logs/llama-perplexity.log) |
| DynaLM, DynaCore kernels (--mode exact) | 1.5b | – | – | – | not run: llama-perplexity failed (log: /root/.dynalm/cpu-report/logs/llama-perplexity.log) |

Pass: top-1 ≥ 94 % and perplexity ≤ llama.cpp's + 1 %.

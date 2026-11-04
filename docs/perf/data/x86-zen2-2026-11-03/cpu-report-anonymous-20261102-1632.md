## DynaLM CPU report

- **CPU:** DO-Premium-AMD (sse4.2 avx2 fma f16c), 4 threads
- **OS:** Linux Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic (x86_64), RAM 15.6 GB
- **DynaLM:** 0.5.0; **llama.cpp:** llama.cpp b11695 (pinned release, sha256 7e77d0c9f4982658…)
- **Workload:** prompt 512, 128 output tokens, 2 round(s), medians
- **Run:** **uncontrolled** (see notes): compare with care
- llama.cpp version: 0.7.0-dev (build 11695, commit 9a16f1dbe)
- uncontrolled: machine load stayed above 1.00
- 1.5b llama.cpp round 1: server did not start
- 1.5b dynacore round 2: server did not start

### 1.5b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 11.1 | 0.91× | 5811 ms | 44.9 / 50.2 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 9.2 | 0.76× | 7176 ms | 51.8 / 60.2 ms |
| llama.cpp (llama-server, CPU) | 1 | 12.2 | – | 5599 ms | 38.1 / 43.2 ms |
| DynaLM default (ggml kernels) | 4 | 12.1 | 0.81× | 13589 ms | 107.2 / 604.3 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 12.9 | 0.86× | 16648 ms | 82.1 / 537.6 ms |
| llama.cpp (llama-server, CPU) | 4 | 15.0 | – | 21153 ms | 99.7 / 112.9 ms |

### 4b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 4.0 | 0.85× | 15817 ms | 129.3 / 151.5 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 3.5 | 0.75× | 20261 ms | 126.5 / 137.7 ms |
| llama.cpp (llama-server, CPU) | 1 | 4.7 | – | 14823 ms | 97.1 / 110.2 ms |
| DynaLM default (ggml kernels) | 4 | 4.2 | 0.77× | 38557 ms | 328.2 / 1676.6 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 4.8 | 0.88× | 44431 ms | 207.0 / 1500.7 ms |
| llama.cpp (llama-server, CPU) | 4 | 5.5 | – | 59100 ms | 264.9 / 305.3 ms |

### 8b

| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1 | 2.4 | 0.89× | 28003 ms | 190.2 / 215.6 ms |
| DynaLM, DynaCore kernels (--mode exact) | 1 | 2.1 | 0.75× | 34086 ms | 216.5 / 237.3 ms |
| llama.cpp (llama-server, CPU) | 1 | 2.7 | – | 26310 ms | 156.6 / 181.2 ms |
| DynaLM default (ggml kernels) | 4 | 2.6 | 0.81× | 71198 ms | 499.6 / 2714.2 ms |
| DynaLM, DynaCore kernels (--mode exact) | 4 | 2.9 | 0.91× | 73570 ms | 334.0 / 2406.8 ms |
| llama.cpp (llama-server, CPU) | 4 | 3.2 | – | 106502 ms | 414.7 / 469.2 ms |

### Agreement with llama.cpp (same tokens)

| Path | Model | Top-1 | Mean KL | PPL (llama.cpp) | Verdict |
|---|---|---|---|---|---|
| DynaLM default (ggml kernels) | 1.5b | 100.00 % | 0.00000 | 12.1785 (12.1785) | pass |
| DynaLM, DynaCore kernels (--mode exact) | 1.5b | 95.61 % | 0.00668 | 12.0766 (12.1785) | pass |

Pass: top-1 ≥ 94 % and perplexity ≤ llama.cpp's + 1 %.

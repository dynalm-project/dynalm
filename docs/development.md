# Development

## Requirements

- CMake ≥ 3.24, Ninja
- Windows: MSVC Build Tools 2022 (C++ workload). Build from a *Developer
  PowerShell / x64 Native Tools* prompt, or call `vcvars64.bat` first.
- Linux: gcc ≥ 13 or clang ≥ 17 (both need `<format>`)
- No GPU is required. GPU flags (`ENABLE_CUDA`, …) default to OFF and are not
  implemented yet.

## Build

```sh
cmake --preset msvc-release        # or linux-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Sanitizers: `msvc-asan`, `linux-asan-ubsan`, `linux-tsan`.

Without presets:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build
```

## Linux build and sanitizers (Docker)

```sh
bash tools/linux.sh linux-release      # gcc 13, -Wall -Wextra -Wshadow, zero warnings
bash tools/linux.sh linux-asan-ubsan   # fatal on first finding
bash tools/linux.sh linux-tsan         # runs with ASLR disabled (see DD-014)
```

This is the reference gate: every phase must pass all three.

## Windows Smart App Control

When Smart App Control is in enforce mode, Windows may block freshly linked, unsigned
test or benchmark executables ("An Application Control policy has blocked this file").
This shows up as `Error running test executable` during gtest discovery. It's a host
security policy, and the build doesn't try to work around it. Use the Linux container
as the test gate, or turn Smart App Control off in Windows Security if you choose to.

## Test models

`bash tools/fetch_models.sh` downloads SmolLM2-135M (f16, Q8_0) and Qwen2.5-0.5B
(Q8_0, Q4_K_M) into `models/` (git-ignored). Real-model tests pick them up
automatically and are skipped when the files are absent. Tokenizer golden files come
from `tools/gen_tokenizer_golden.py` (needs `pip install tokenizers` and the model's
HF `tokenizer.json`).

## Feature flags

| Flag | Default | Meaning |
|---|---|---|
| `ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_AMX` | ON / OFF / OFF | compile those CPU kernel tiers (selected at runtime) |
| `ENABLE_CUDA` / `HIP` / `METAL` / `VULKAN` | OFF | future; ON is a configure error |
| `ENABLE_SERVER` / `ENABLE_TESTS` / `ENABLE_BENCHMARKS` | ON | build components |
| `ENABLE_ASAN` / `ENABLE_UBSAN` / `ENABLE_TSAN` | OFF | sanitizers |

## Run

```sh
build/msvc-release/src/engine info
build/msvc-release/benchmarks/bench_foundation
```

## Conventions

- `namespace engine`. Files are `snake_case`, types `PascalCase`, functions `snake_case`.
- Errors: `Status` / `Result<T>`. No exceptions on hot paths.
- SIMD intrinsics only in `src/backends/cpu/<isa>/*.cpp`.
- No heap allocation per token. No logging per token above DEBUG.
- Every phase ends with a green build, tests, benchmarks, updated docs, and a commit.

## Serving

```sh
engine serve models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
curl http://127.0.0.1:8000/metrics
```

Endpoints: `GET /health`, `GET /v1/models`, `GET /metrics` (Prometheus text),
`POST /v1/chat/completions`, `POST /v1/completions` (both support `stream`).
Sampling parameters are accepted, but decoding is greedy until Phase 26.
The `"ignore_eos": true` request extension (also accepted by llama.cpp) disables stopping on
end-of-generation tokens. It is used for fixed-length benchmarking.

## Benchmarking

```sh
# In-process: concurrency x prompt x output sweep, JSON lines appended to --out.
engine benchmark models/qwen2.5-0.5b-instruct-q4_k_m.gguf -t 10 \
  --concurrency 1,4,16 --prompt 128,512 --output 128 --out results/run.jsonl
# Any OpenAI-compatible server. The model file supplies the tokenizer used to size prompts.
engine benchmark models/qwen2.5-0.5b-instruct-q4_k_m.gguf --url http://127.0.0.1:8000
# Head to head with llama.cpp in identical containers, then render tables.
bash tools/compare_baselines.sh qwen2.5-0.5b-instruct-q4_k_m.gguf 10 results/baselines.jsonl
PYTHONUTF8=1 python tools/bench_report.py results/baselines.jsonl
```

Tuning knobs for experiments (not for production): `ENGINE_GEMM_KC` (GEMM K-slice) and
`ENGINE_MATMUL_EXPAND_MIN` (rows from which matmul expands weights, DD-036).

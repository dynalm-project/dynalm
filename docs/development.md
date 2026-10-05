# Development

## Requirements

- CMake ≥ 3.24, Ninja
- Windows: MSVC Build Tools 2022 (C++ workload). Build from a *Developer
  PowerShell / x64 Native Tools* prompt, or call `vcvars64.bat` first.
- Linux: gcc ≥ 13 or clang ≥ 17 (both need `<format>`)
- macOS: Xcode command line tools (Apple clang 15+) and `brew install cmake ninja`
- No GPU is required. GPU flags (`ENABLE_CUDA`, …) default to OFF and are not
  implemented yet.

## Build

```sh
cmake --preset msvc-release        # or linux-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Presets: `msvc-release`, `linux-release`, `linux-clang-release`, `macos-release`.
Sanitizers: `msvc-asan`, `linux-asan-ubsan`, `linux-tsan`.

## Platforms and CPU kernels

| Platform | Kernels | Verified by |
|---|---|---|
| Linux x86-64 | generic + AVX2 (runtime-selected) | full suite under gcc, clang, ASAN/UBSAN, TSAN (locally and in CI) |
| Linux ARM64 (Graviton, Ampere, Raspberry Pi 5) | generic + NEON | full suite in an emulated arm64 container; natively in CI (`ubuntu-24.04-arm`) |
| macOS Apple Silicon / Intel | NEON / AVX2 | CI job (`macos-14`): build, full suite, install script; runs once the repository is on GitHub. Not built on a Mac yet |
| Windows x64 | generic + AVX2 | MSVC build, install script, native inference run (locally); full suite in the CI job (Smart App Control blocks local test executables) |

SIMD options follow the target architecture automatically: AVX2/AVX-512/AMX exist only on
x86-64, and NEON only on ARM64 (DD-046).

## Install and packages

- `scripts/install.sh` (Linux, macOS) and `scripts/install.ps1` (Windows) build a release
  binary with a static C/C++ runtime (`DYNALM_STATIC_RUNTIME=ON`) and install it with
  `cmake --install`.
- `cpack` in a build directory produces `dynalm-<version>-<os>-<arch>.tar.gz` / `.zip`.
- The `release` workflow does this for all four platforms on a version tag.
- `Dockerfile` builds a two-stage image (amd64 or arm64). The server listens on `0.0.0.0`
  in the container through `DYNALM_HOST`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:
- Linux x86-64 (gcc, clang, ASAN/UBSAN, TSAN);
- Linux ARM64;
- Windows (MSVC);
- macOS (Apple Silicon);
- the install scripts on all three operating systems;
- the Docker image build.

Real-model tests skip themselves there, because model files are not in the repository.

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
| `ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_AMX` | ON / OFF / OFF | x86-64 kernel tiers (selected at runtime; forced OFF on other CPUs) |
| `ENABLE_NEON` | ON | ARM64 kernel tier (forced OFF on other CPUs) |
| `DYNALM_STATIC_RUNTIME` | OFF | static C/C++ runtime for self-contained release binaries |
| `ENABLE_CUDA` / `HIP` / `METAL` / `VULKAN` | OFF | future; ON is a configure error |
| `ENABLE_SERVER` / `ENABLE_TESTS` / `ENABLE_BENCHMARKS` | ON | build components |
| `ENABLE_ASAN` / `ENABLE_UBSAN` / `ENABLE_TSAN` | OFF | sanitizers |

## Run

```sh
build/msvc-release/src/dynalm info
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
dynalm serve models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
curl http://127.0.0.1:8000/metrics
```

Endpoints: `GET /health` (503 `draining` during shutdown), `GET /v1/models`,
`GET /metrics` (Prometheus text), `POST /v1/chat/completions`, `POST /v1/completions` (both
support `stream`), and `POST /admin/shutdown` (loopback only; `--disable-admin` turns it off).

### Model formats

Every command that takes a model accepts a GGUF file or a Hugging Face model directory
(SafeTensors):

```sh
dynalm inspect models/st/qwen2.5-0.5b-instruct     # config, dtypes, RAM estimate, support
dynalm run models/st/smollm2-135m-instruct -p "Hi"
dynalm serve models/st/qwen2.5-0.5b-instruct --port 8000
PYTHONUTF8=1 python tools/make_tiny_hf.py tests/data   # regenerate HF test fixtures
```

### Speculative decoding (single sequence)

```sh
dynalm run models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "..." --spec ngram          # prompt lookup
dynalm run models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "..." --spec models/qwen2.5-0.5b-instruct-q4_k_m.gguf --spec-k 4
bench_speculative <target.gguf> [draft.gguf|-] [threads] [k]
```

Greedy output matches plain greedy decoding; with sampling, outputs follow the target's
distribution (DD-044).

### Configuration

Options come from three sources; later ones win: config file < `DYNALM_<OPTION>`
environment < command line. `threads`, `ctx`, `batch` and `http-threads` accept `auto`. With
`ctx auto` (the default), the KV cache uses half of the free RAM left after the weights, capped
at 65536 tokens (DD-038).

```ini
# engine.conf — keys are the long option names
model = models/qwen2.5-0.5b-instruct-q4_k_m.gguf
host = 0.0.0.0
port = 8000
threads = auto
ctx = auto
max-active = 64          # concurrent requests before 503 + Retry-After
request-timeout = 600    # seconds, queued + generating
shutdown-timeout = 30    # drain time on SIGTERM / dynalm stop
```

```sh
DYNALM_PORT=9000 dynalm serve --config engine.conf --threads 8
dynalm pull <owner>/<repo>/<file>.gguf   # download a GGUF from Hugging Face (resumable, pre-checked)
dynalm pull <link> --check               # only check whether DynaLM supports it (256 KiB read)
dynalm rm <model> [-y]                   # delete a downloaded model (GGUF, .part, or HF directory)
dynalm list models          # GGUF files with architecture, quantization, context, support status
dynalm stop --port 9000     # drain in-flight requests, then exit (alias: unload)
```

Status codes: 400 for invalid or unsupported requests (including prompt + max_tokens larger
than the context or the KV cache), 503 for overload, draining or a retryable resource limit,
504 for a request timeout, and 500 otherwise. Error bodies use the OpenAI shape
`{"error":{"message","type"}}`.
Sampling (DD-043): `temperature` (default 1.0, as in OpenAI; `serve --temperature`), `top_p`,
`seed`, `presence_penalty`, `frequency_penalty`, plus the extensions `top_k`, `min_p`,
`repetition_penalty` and `repeat_last_n`. A request with a seed reproduces exactly, also
across platforms. `dynalm run` is greedy unless `--temp` is given.
The `"ignore_eos": true` request extension (also accepted by llama.cpp) disables stopping on
end-of-generation tokens. It is used for fixed-length benchmarking.

## Benchmarking

```sh
# In-process: concurrency x prompt x output sweep, JSON lines appended to --out.
dynalm benchmark models/qwen2.5-0.5b-instruct-q4_k_m.gguf -t 10 \
  --concurrency 1,4,16 --prompt 128,512 --output 128 --out results/run.jsonl
# Any OpenAI-compatible server. The model file supplies the tokenizer used to size prompts.
dynalm benchmark models/qwen2.5-0.5b-instruct-q4_k_m.gguf --url http://127.0.0.1:8000
# Head to head with llama.cpp in identical containers, then render tables.
bash tools/compare_baselines.sh qwen2.5-0.5b-instruct-q4_k_m.gguf 10 results/baselines.jsonl
PYTHONUTF8=1 python tools/bench_report.py results/baselines.jsonl
```

Tuning knobs for experiments (not for production): `DYNALM_GEMM_KC` (GEMM K-slice) and
`DYNALM_MATMUL_EXPAND_MIN` (rows from which matmul expands weights, DD-036).
`DYNALM_INT8_DECODE_ROWS` sets the largest row count that uses int8 activations (DD-053).
A sweep on the development laptop found the defaults best (DD-063).
Configure with `-DDYNALM_LTO=ON` for link-time optimization. It is off by default because it showed no measured gain.

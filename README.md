# DynaLM

**A fast, CPU-first LLM inference engine.** Run Llama, Qwen, Gemma, Phi, Mistral and
mixture-of-experts models on an ordinary CPU, from the terminal or as an OpenAI-compatible
server. Written in C++20 with no Python at runtime.

```sh
dynalm run  models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "What is the capital of France?"
dynalm serve models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8000
```

## Why DynaLM

- **Fast on CPUs.**
  - Hand-written AVX2 (x86-64) and NEON (ARM64 / Apple Silicon) kernels.
  - Matches or beats llama.cpp's throughput on the same machine, with 2–3× faster time to
    first token under load.
- **Built for many users.**
  - Continuous batching, so new requests join running ones.
  - Paged KV cache, chunked prompts, and reuse of shared prompt prefixes.
- **Opens what you already have.**
  - GGUF files (llama.cpp / Ollama), Hugging Face SafeTensors folders, and GPTQ / AWQ
    checkpoints.
  - Quantized formats from Q2_K to Q8_0, plus F16 and BF16.
- **Drop-in API.** OpenAI-compatible `/v1/chat/completions` and `/v1/completions` with
  streaming, so existing OpenAI clients work by changing the base URL.
- **Production basics.** Graceful shutdown, overload protection, timeouts, Prometheus
  metrics, and configuration from a file, environment variables or flags.

## Install

| Platform | Command | Status |
|---|---|---|
| Linux (x86-64, ARM64) | `./scripts/install.sh` | ✅ tested (in CI) |
| macOS (Apple Silicon, Intel) | `./scripts/install.sh` | ✅ tested (Apple Silicon, in CI) |
| Windows 10/11 (x64) | `powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -AddToPath` | ✅ tested |
| Docker (amd64, arm64) | `docker build -t dynalm .` | ✅ tested |

You need a C++20 compiler and CMake ≥ 3.24 (on Windows: Visual Studio 2022 Build Tools with
the C++ workload). The installers build a release binary and put `dynalm` in `~/.local/bin`
(Linux/macOS) or `%LOCALAPPDATA%\Programs\DynaLM\bin` (Windows). Release archives for every
platform are produced by the `release` workflow when a version tag is pushed.

```sh
git clone <this repository> && cd dynalm
./scripts/install.sh            # or scripts\install.ps1 on Windows
dynalm info                     # your CPU, its SIMD features and the kernels DynaLM will use
```

Docker:

```sh
docker build -t dynalm .
docker run --rm -p 8000:8000 -v "$PWD/models:/models" dynalm serve /models/model.gguf
```

## Use

```sh
dynalm pull Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf   # download into ./models
dynalm rm qwen2.5-0.5b-instruct-q4_k_m        # delete a downloaded model (asks first; -y skips)
dynalm list                                   # models in ./models (GGUF and Hugging Face folders)
dynalm inspect models/model.gguf              # architecture, quantization, memory estimate, support
dynalm run models/model.gguf -p "Hi" --temp 0.7 --top-p 0.9
dynalm serve models/model.gguf --port 8000    # OpenAI-compatible server
dynalm stop                                   # drain in-flight requests and exit
dynalm benchmark models/model.gguf --concurrency 1,4,16
```

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
print(client.chat.completions.create(model="any", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

## Getting models

Models are not part of the repository. Download a `.gguf` file from Hugging Face with `dynalm pull`, using
its link (page or download link) or the short form `<owner>/<repo>/<file>.gguf`:

```sh
dynalm pull https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

- It checks the architecture first and refuses models DynaLM cannot run, before downloading gigabytes.
- Interrupted downloads resume; gated models (e.g. Llama) need their license accepted and `HF_TOKEN` set.
- Pick `Q4_K_M` for a good size/quality balance, `Q8_0` for higher quality.
- `pull` uses `curl` (built into Windows 10+, macOS and Linux).

## Supported models

Llama 1–3 (incl. Mistral, SmolLM, DeepSeek-distilled), Qwen 2 / 2.5 / 3, Gemma 1 / 2 / 3, Phi-3,
IBM Granite, and mixture-of-experts models: Mixtral, Qwen-MoE, Granite-MoE.

Good starting points:

| RAM | Model | Command |
|---|---|---|
| any | Qwen3-0.6B | `dynalm pull Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` |
| any | Llama-3.2-1B | `dynalm pull bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| 8 GB | Qwen3-4B | `dynalm pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` |
| 8 GB | Gemma-3-4B | `dynalm pull unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf` |
| 16 GB | Qwen2.5-Coder-7B | `dynalm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| 16 GB | Llama-3.1-8B | `dynalm pull bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |

**Not supported yet:**
- Qwen3.5 / Qwen3.8 (`qwen35`), LFM2, full DeepSeek-V3/R1.
- IQ-quantized files.

`dynalm pull` refuses these before downloading. The full list, with sizes, test status and caveats, is in
[docs/model-support.md](docs/model-support.md).

## Documentation

- [docs/development.md](docs/development.md): building, testing, configuration, serving.
- [docs/architecture.md](docs/architecture.md): how the engine is put together.
- [docs/benchmarks.md](docs/benchmarks.md): measured performance, including head-to-head
  runs against llama.cpp.
- [docs/design-decisions.md](docs/design-decisions.md): every major design choice, with
  evidence.
- [docs/gpu-backend.md](docs/gpu-backend.md): the GPU backend contract (GPU support is
  designed, not implemented yet).
- [CHANGELOG.md](CHANGELOG.md) and [TODO.md](TODO.md).

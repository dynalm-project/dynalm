# DynaLM: fast local LLM inference on CPU

[![CI](https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml/badge.svg)](https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C)

**DynaLM is an open-source LLM inference engine that runs large language models on an ordinary CPU, with
no GPU and no Python.**
- Runs **Llama, Qwen, Gemma, Phi, Mistral, Granite** and **mixture-of-experts** models (Mixtral, Qwen-MoE).
- Loads GGUF files (the format used by llama.cpp and Ollama), Hugging Face SafeTensors, and GPTQ/AWQ
  checkpoints.
- Use it from the terminal, or as an **OpenAI-compatible API server** for local and self-hosted AI.
- Works on Linux, macOS (including Apple Silicon) and Windows.

```sh
dynalm pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf        # download a model from Hugging Face
dynalm run   models/Qwen3-4B-Q4_K_M.gguf -p "What is the capital of France?"
dynalm serve models/Qwen3-4B-Q4_K_M.gguf --port 8000       # OpenAI-compatible server
```

## Why DynaLM

- **Fast on CPUs.**
  - Hand-written AVX2 (x86-64) and NEON (ARM64 / Apple Silicon) kernels.
  - On the same laptop, it matches or beats llama.cpp's throughput, with up to 3× lower time to first
    token under concurrent load ([benchmarks](docs/benchmarks.md)).
- **Built for many users at once.**
  - Continuous batching, so new requests join running ones.
  - Paged KV cache, chunked prefill, and a radix prefix cache that reuses shared prompts.
- **Opens the models you already have.**
  - GGUF, Hugging Face SafeTensors folders, GPTQ and AWQ.
  - Q2_K–Q8_0, F16 and BF16.
  - `dynalm pull` downloads from Hugging Face and refuses unsupported models before downloading.
- **Drop-in OpenAI API.** `/v1/chat/completions` and `/v1/completions` with streaming. Existing OpenAI
  SDK clients, LangChain or LlamaIndex apps work by changing the base URL.
- **Production basics.**
  - Graceful shutdown, overload protection (503), timeouts.
  - Prometheus metrics.
  - Configuration from a file, environment variables or flags.
- **Advanced decoding.** Top-k, top-p, min-p, repetition penalties, seeded sampling, and speculative
  decoding (prompt lookup or a draft model).
- **Small and dependency-free.** One binary, C++20, no Python or CUDA at runtime.

## Install

| Platform | Command | Status |
|---|---|---|
| Linux (x86-64, ARM64) | `./scripts/install.sh` | ✅ tested in CI |
| macOS (Apple Silicon, Intel) | `./scripts/install.sh` | ✅ tested in CI (Apple Silicon) |
| Windows 10/11 (x64) | `powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -AddToPath` | ✅ tested in CI |
| Docker (amd64, arm64) | `docker build -t dynalm .` | ✅ tested in CI |

You need a C++20 compiler and CMake ≥ 3.24.
- On macOS: `xcode-select --install && brew install cmake ninja`.
- On Windows: Visual Studio 2022 Build Tools with the C++ workload.

The installers build a release binary and put `dynalm` in `~/.local/bin` (Linux/macOS) or
`%LOCALAPPDATA%\Programs\DynaLM\bin` (Windows).

```sh
git clone https://github.com/dynalm-project/dynalm.git && cd dynalm
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
dynalm pull <owner>/<repo>/<file>.gguf        # download a GGUF from Hugging Face (resumable)
dynalm pull <link> --check                    # will DynaLM run it? (reads 256 KiB, saves nothing)
dynalm list                                   # models in ./models, with a support status for each
dynalm rm <model>                             # delete a downloaded model (asks first; -y skips)
dynalm inspect models/model.gguf              # architecture, quantization, memory estimate
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

## Supported models

Llama 1–3 (incl. Mistral, SmolLM, DeepSeek-R1-Distill), Qwen 2 / 2.5 / 3 (dense and MoE), Gemma 1 / 2 / 3,
Phi-3, IBM Granite, and mixture-of-experts models: Mixtral, Qwen-MoE, Granite-MoE.

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

## How DynaLM compares

| | DynaLM | llama.cpp | Ollama | vLLM |
|---|---|---|---|---|
| Primary target | CPU | CPU + GPU | CPU + GPU (llama.cpp inside) | GPU |
| Language | C++20 | C/C++ | Go + C/C++ | Python + CUDA |
| GGUF models | ✅ | ✅ | ✅ | partial |
| Hugging Face SafeTensors / GPTQ / AWQ | ✅ (CPU) | via conversion | via import | ✅ (GPU) |
| OpenAI-compatible server | ✅ | ✅ | ✅ | ✅ |
| Continuous batching + paged KV cache | ✅ | partial | via llama.cpp | ✅ |
| GPU support | not yet (designed) | ✅ | ✅ | ✅ |

DynaLM focuses on serving many users well on CPU-only machines: laptops, cloud CPU instances, ARM servers
and edge devices. If you have a GPU, llama.cpp, Ollama or vLLM will be faster today.

## FAQ

**How do I run an LLM locally on a CPU without a GPU?**
Install DynaLM, download a quantized GGUF model with `dynalm pull`, then use `dynalm run` to chat or
`dynalm serve` for an API. A 4B model in Q4_K_M needs about 3 GB of RAM.

**Is DynaLM an alternative to Ollama or llama.cpp?**
Yes, for CPU inference. It reads the same GGUF files and offers an OpenAI-compatible server. Its scheduler
(continuous batching, chunked prefill, prefix caching) is built for concurrent users.

**Can I use DynaLM with the OpenAI Python SDK, LangChain or LlamaIndex?**
Yes. Point the client's base URL at `http://127.0.0.1:8000/v1`. Any API key works.

**Does DynaLM run on Apple Silicon (M1/M2/M3/M4) and Raspberry Pi?**
It builds and passes its tests on Apple Silicon and ARM64 Linux in CI, using NEON kernels. A Raspberry Pi
4/5 running a 64-bit OS uses the same ARM64 build. Small models such as Qwen3-0.6B or Llama-3.2-1B fit in
its memory, though we have not measured speed on a Pi yet.

**Which quantization should I choose?**
Choose `Q4_K_M` for the best balance of size and quality, or `Q8_0` for near-original quality at twice the
size. IQ quantizations are not supported yet.

**How much RAM do I need?**
About the size of the model file, plus a little for the context: 1 GB for 1B models, 3 GB for 4B, 5–6 GB
for 7–8B in Q4_K_M. `dynalm inspect <model>` prints an estimate.

**Does it support GPUs?**
Not yet. The backend interface is designed and tested for GPUs (see [docs/gpu-backend.md](docs/gpu-backend.md)),
but only the CPU backend is implemented.

**Is it free for commercial use?**
Yes. DynaLM is licensed under Apache-2.0. The models you run have their own licenses.

## Documentation

- [docs/model-support.md](docs/model-support.md): every supported model, with download commands.
- [docs/development.md](docs/development.md): building, testing, configuration, serving.
- [docs/architecture.md](docs/architecture.md): how the engine is put together.
- [docs/benchmarks.md](docs/benchmarks.md): measured performance, including comparisons with llama.cpp and
  Ollama.
- [docs/performance.md](docs/performance.md) and [docs/quantization.md](docs/quantization.md).
- [docs/design-decisions.md](docs/design-decisions.md): every major design choice, with evidence.
- [docs/gpu-backend.md](docs/gpu-backend.md): the GPU backend contract.
- [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md).

## Contributing

Contributions are welcome: bug reports, model requests, benchmarks on new hardware, and pull requests. See
[CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately ([SECURITY.md](SECURITY.md)).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party components. If you use DynaLM in
research, please cite it ([CITATION.cff](CITATION.cff)).

<!-- Keywords: local LLM, CPU inference, LLM inference engine, run LLM without GPU, llama.cpp alternative,
Ollama alternative, vLLM alternative for CPU, GGUF runner, OpenAI-compatible API, self-hosted AI, offline AI,
private AI, edge AI, AVX2, ARM NEON, Apple Silicon, quantization, Q4_K_M, continuous batching,
paged attention, KV cache, speculative decoding, mixture of experts, Llama 3, Qwen3, Gemma 3, Phi-3,
Mistral, C++ LLM. -->

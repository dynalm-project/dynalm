<p align="center">
  <img src="public/light.png" alt="DynaLM - fast, private local LLM engine for CPU and GPU with an OpenAI-compatible server" width="620">
</p>

<div align="center">

# DynaLM: Run AI Models Locally on Your CPU or GPU

### A fast, private, open-source way to run Llama, Qwen, Gemma, DeepSeek and Phi on your own computer. One small program. No Python. No cloud. Works offline.

[![CI Status](https://img.shields.io/github/actions/workflow/status/dynalm-project/dynalm/ci.yml?branch=main&label=CI&logo=github&style=flat-square)](https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/dynalm-project/dynalm?color=3b82f6&label=Release&style=flat-square)](https://github.com/dynalm-project/dynalm/releases/latest)
[![License](https://img.shields.io/badge/License-Apache--2.0-22c55e.svg?style=flat-square)](LICENSE)
[![Platforms](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-64748b?style=flat-square)](docs/installation.md)
[![CPU](https://img.shields.io/badge/CPU-AVX2%20%7C%20AVX--512%20%7C%20NEON-f97316?style=flat-square)](docs/architecture.md)
[![GPU](https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan%20%7C%20CUDA-8b5cf6?style=flat-square)](docs/gpu.md)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-a855f7?style=flat-square)](docs/api.md)
[![Private](https://img.shields.io/badge/Privacy-100%25%20Offline-success?style=flat-square)](SECURITY.md)

<p align="center">
  <a href="#what-is-dynalm"><b>What is it?</b></a> •
  <a href="#quick-start"><b>Quick start</b></a> •
  <a href="#install"><b>Install</b></a> •
  <a href="#where-it-runs"><b>Where it runs</b></a> •
  <a href="#use-your-gpu"><b>GPU</b></a> •
  <a href="#models"><b>Models</b></a> •
  <a href="#speed"><b>Speed</b></a> •
  <a href="#api-server"><b>API server</b></a> •
  <a href="#faq"><b>FAQ</b></a>
</p>

</div>

---

<!--
SEO keywords: local LLM, run LLM locally, run AI on my computer, offline AI, private AI, Ollama alternative, llama.cpp alternative, LLM on CPU, LLM on GPU, Apple Silicon LLM, Metal LLM, Vulkan LLM, CUDA LLM, AMD GPU LLM, Intel Arc LLM, Windows on ARM LLM, Snapdragon LLM, GGUF runner, OpenAI compatible local server, local ChatGPT alternative, Qwen local, Llama local, DeepSeek R1 local.
AEO answer: DynaLM is a free, open-source program that runs large language models on your own computer, on the CPU or the GPU, fully offline. Install it with one command, then run `dynalm run qwen2.5:1.5b` to chat, or `dynalm serve` for an OpenAI-compatible API.
-->

## What is DynaLM?

**DynaLM is a free program that runs AI chat models on your own computer.** It works on your
CPU and on your GPU, and it needs no internet once the model is downloaded.

- **One small program.** It needs no Python, PyTorch or Docker, and no background service.
- **Private.** Your prompts never leave your machine. There is no telemetry.
- **Familiar commands.** `pull`, `run` and `serve`, like Ollama.
- **Works with your tools.** It has a built-in OpenAI-compatible API, so the OpenAI SDK,
  LangChain, LlamaIndex, Open WebUI, Continue.dev and Cursor connect to it directly.
- **Uses less memory.** On an Apple M4 Pro, DynaLM serves a 1.5B model in **about 3× less
  memory** than llama.cpp's server: 390 MB against 1,113 MB.
- **Built for many users.** Continuous batching, a paged KV cache and prefix sharing keep
  answers coming when several people use it at once.

**License:** Apache-2.0, free for personal and commercial use.

---

## Quick start

```bash
# 1. Install (Linux and macOS)
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh

# 2. Download a model and chat with it
dynalm pull qwen2.5:1.5b
dynalm run qwen2.5:1.5b
```

| You want to... | Run this |
|---|---|
| Ask one question | `dynalm run qwen2.5:1.5b -p "Explain DNS in two sentences"` |
| Start an API server | `dynalm serve qwen2.5:1.5b` (http://127.0.0.1:8000/v1) |
| Check your computer | `dynalm doctor` |
| See what is installed | `dynalm models` |

---

## Install

| Your computer | How to install |
|---|---|
| **Linux or macOS** | `curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh \| sh` |
| **Windows 10 / 11 (Intel or AMD)** | PowerShell: `irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 \| iex` |
| **Windows 11 on ARM (Snapdragon)** | PowerShell: `powershell -File scripts\install.ps1 -FromSource` (builds on your PC) |
| **Docker** | `docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models:/models" ghcr.io/dynalm-project/dynalm:cpu serve /models/model.gguf --allow-unauthenticated` |

**Notes:**
- No administrator rights are needed.
- Every download is checked with SHA-256.
- Open a new terminal afterwards, then run `dynalm doctor`.

More detail, including manual downloads and building from source: [docs/installation.md](docs/installation.md).

---

## Where it runs

**Status labels:**
- ✅ **Stable:** ready for everyday use.
- 🧪 **Preview:** works and passes our test suite. Feedback is welcome.

### CPU

| System | Chips | Status |
|---|---|---|
| macOS | Apple Silicon M1, M2, M3, M4 (all Pro / Max / Ultra) | ✅ Stable |
| macOS | Intel Macs | ✅ Stable |
| Linux x86-64 | Intel and AMD (AVX2, AVX-512, AMX) | ✅ Stable |
| Linux ARM64 | Raspberry Pi 4/5, AWS Graviton, Ampere | ✅ Stable |
| Windows 10 / 11 x64 | Intel and AMD | ✅ Stable |
| Windows 11 on ARM | Snapdragon X and other ARM64 PCs | 🧪 Preview |
| Docker | linux/amd64, linux/arm64 | ✅ Stable |

Older CPUs without AVX2 still work, using portable code. `dynalm doctor` shows what your CPU uses.

### GPU

| GPU | Works on | Status |
|---|---|---|
| **Apple Metal** | Macs with Apple Silicon (M1 to M4) | 🧪 Preview |
| **Vulkan** | AMD Radeon, Intel Arc / Iris, NVIDIA, Qualcomm Adreno (Windows and Linux) | 🧪 Preview |
| **NVIDIA CUDA** | NVIDIA GeForce, RTX and data-center GPUs (Windows and Linux) | 🧪 Preview |

---

## Use your GPU

**Step 1: build DynaLM with your GPU switched on.** One option is enough:

```bash
cmake --preset macos-metal  && cmake --build --preset macos-metal    # Mac (Apple Silicon)
cmake --preset linux-vulkan && cmake --build --preset linux-vulkan   # AMD / Intel / NVIDIA on Linux
cmake --preset linux-cuda   && cmake --build --preset linux-cuda     # NVIDIA on Linux
```

On Windows, use `msvc-vulkan` or `msvc-cuda`. What each one needs is in [docs/gpu.md](docs/gpu.md).

**Step 2: just run it.** DynaLM picks the device for you:

| Setting | What happens |
|---|---|
| `--backend auto` (default) | Uses the GPU when it can run the model, otherwise the CPU. The log says which one and why. |
| `--backend cpu` | Always the CPU |
| `--backend metal` / `vulkan` / `cuda` | Always that GPU. It stops with a clear message if the model can't run there. |

You can also set it once in `~/.dynalm/config.yaml` (`runtime: device: cpu`) or with
`DYNALM_BACKEND=cpu`.

**Models that run on the GPU today:** the Qwen2.5 family and other models of the same design.
For Qwen3, Gemma 3, Llama 3.x and mixture-of-experts models, `auto` uses the CPU for now.
Full guide: [docs/gpu.md](docs/gpu.md).

---

## Models

Download by name with `dynalm pull <name>`, or use any `.gguf` file.

| Name | Size | RAM needed | Good for |
|---|---:|---:|---|
| `smollm2:135m` | 145 MB | 1 GB | testing, tiny devices |
| `gemma3:270m` | 292 MB | 1 GB | quick answers |
| `qwen2.5:0.5b` | 491 MB | 1 GB | fast chat on small PCs (runs on GPU too) |
| `qwen3:0.6b` | 639 MB | 1 GB | small "thinking" model |
| `gemma3:1b` | 806 MB | 2 GB | general chat |
| `llama3.2:1b` | 808 MB | 2 GB | fast chat on laptops |
| `granite3.1-moe:1b` | 822 MB | 2 GB | efficient mixture-of-experts |
| `qwen2.5:1.5b` | 1.1 GB | 2 GB | **best start**: good quality, fast, runs on GPU too |
| `qwen3:1.7b` | 1.8 GB | 2.5 GB | step-by-step reasoning |
| `llama3.2:3b` | 2.0 GB | 3–4 GB | writing and coding |
| `phi3.5:3.8b` | 2.4 GB | 4 GB | maths and code |
| `qwen3:4b` | 2.5 GB | 4 GB | best quality in the list |

**Other ways to get a model:**

```bash
dynalm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf   # any Hugging Face GGUF
dynalm run ./my-model.gguf                                                          # a file on disk
dynalm models --available                                                          # the full catalog
```

**Supported:**
- **Families:** Llama 1–3.2, Qwen 2 / 2.5 / 3 (dense and MoE), Gemma 1–3, Phi-3 / 3.5,
  Mistral, Mixtral, IBM Granite, and the DeepSeek-R1 distills.
- **File formats:** GGUF, Hugging Face SafeTensors, GPTQ and AWQ.
- **Quantizations:** Q2_K to Q8_0, F16, BF16 and F32. `Q4_K_M` is recommended.

---

## Speed

All numbers come from our own repeatable tests. Method: [docs/benchmarking.md](docs/benchmarking.md).

**Apple M4 Pro GPU (Metal), Qwen2.5-1.5B Q4_K_M, compared with llama.cpp's server on the same
file:**

| What we measured | Result |
|---|---|
| Memory, one user | **390 MB vs 1,113 MB**: about 3× less |
| First word (typical), 4 users at once | **1.9–2.1× faster** |
| First word (typical), 8 users at once | **2.5× faster** |
| Speed per token, one user | about 7 ms per token, within about 10% of llama.cpp |
| Against DynaLM's own CPU mode | about 4× faster replies, 11–16× faster prompt reading |

Where llama.cpp is still ahead: total tokens per second with many users at once (8 or more).
This is our next focus.

**Laptop CPU (Intel Core i7-1255U):** `qwen2.5:1.5b` about 13 tokens/s, `qwen3:4b` about
6 tokens/s.

**Measure your own machine:** `dynalm benchmark qwen2.5:1.5b`.

---

## Coming from Ollama?

| Ollama | DynaLM |
|---|---|
| `ollama pull qwen3:4b` | `dynalm pull qwen3:4b` |
| `ollama run qwen3:4b` | `dynalm run qwen3:4b` |
| `ollama run qwen3:4b "hi"` | `dynalm run qwen3:4b -p "hi"` |
| `ollama list` | `dynalm models` |
| `ollama rm qwen3:4b` | `dynalm models rm qwen3:4b` |
| `ollama show qwen3:4b` | `dynalm inspect qwen3:4b` |
| `ollama serve` | `dynalm serve qwen3:4b` |

**In a chat:**

| Command | What it does |
|---|---|
| `/bye` | leave |
| `/clear` | forget the conversation |
| `/system ...` | set a system prompt |
| `/think on\|off` | Qwen3 reasoning on or off |
| `/set temp 0.7` | change a setting |
| `/stats on` | show speed after each reply |

All commands and options: [docs/cli.md](docs/cli.md).

---

## API server

```bash
dynalm serve qwen2.5:1.5b          # http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
reply = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "Give me three tips for clean code."}],
)
print(reply.choices[0].message.content)
```

| Endpoint | Use |
|---|---|
| `POST /v1/chat/completions` | chat, with streaming |
| `POST /v1/completions` | plain text completion |
| `GET /v1/models` | list models |
| `GET /health` | health check |
| `GET /metrics` | Prometheus metrics |

**Connect your tools:** in Open WebUI, LangChain, LlamaIndex, LiteLLM, Continue.dev or Cursor,
set the OpenAI base URL to `http://127.0.0.1:8000/v1`.

**Sharing on a network:**

```bash
openssl rand -hex 32 > ~/.dynalm/api-key
dynalm serve qwen2.5:1.5b --host 0.0.0.0 --api-key-file ~/.dynalm/api-key
```

Clients then send `Authorization: Bearer <key>`. For internet access, put a TLS reverse proxy
in front. Details: [docs/deployment.md](docs/deployment.md) and [docs/security.md](docs/security.md).

---

## FAQ

### How do I run an AI model on my own computer?
Install DynaLM with one command, then run `dynalm run qwen2.5:1.5b`. It downloads the model
once and then works fully offline.

### Do I need a GPU?
No. DynaLM runs well on normal CPUs. If you have a GPU (Apple Silicon, AMD, Intel, or NVIDIA),
DynaLM can use it for faster answers.

### Does DynaLM work on Apple Silicon Macs?
Yes. It runs on M1, M2, M3 and M4 Macs, on the CPU and on the GPU through Metal.

### Does it work with AMD or Intel graphics cards?
Yes, through Vulkan. Build with `-DENABLE_VULKAN=ON`, or use the `linux-vulkan` or
`msvc-vulkan` preset.

### Does it work with NVIDIA GPUs?
Yes, through CUDA (`-DENABLE_CUDA=ON`) or through Vulkan.

### Does it run on Windows on ARM (Snapdragon laptops)?
Yes. Install from source with `scripts\install.ps1 -FromSource`. It uses the fast ARM (NEON)
code.

### Is DynaLM an Ollama alternative?
Yes. It uses the same kind of commands (`pull`, `run`, `serve`), the same GGUF model files, and
an OpenAI-compatible API. It ships as one small program with no background service.

### How does DynaLM compare to llama.cpp?
On an Apple M4 Pro, DynaLM uses about 3× less memory and typically gives the first word 2× faster
when several people use it at once. Its single-user speed is within about 10% of llama.cpp's.
llama.cpp still has higher total throughput with many users.

### Is it private?
Yes. Nothing is sent anywhere. DynaLM only goes online when you run `dynalm pull`.

### How much memory do I need?
About the model file size plus 1 GB. A 1.5B model needs about 2 GB, and a 4B model about 4 GB.

### Can I choose CPU or GPU myself?
Yes: `--backend cpu`, `--backend metal`, `--backend vulkan` or `--backend cuda`. You can also
set `runtime: device:` in `~/.dynalm/config.yaml`.

---

## Build from source

```bash
git clone https://github.com/dynalm-project/dynalm.git && cd dynalm
cmake --preset linux-release   # or macos-release, msvc-release, macos-metal, linux-vulkan, linux-cuda
cmake --build --preset linux-release
bash tools/ci/run_tests.sh build/linux-release
```

**You need:**
- a C++20 compiler (GCC 13+, Clang 17+, Apple Clang 15+, or Visual Studio 2022);
- CMake 3.24 or newer, or 3.28 or newer for GPU builds.

All options: [docs/build.md](docs/build.md).

---

## Documentation

| Guide | What is inside |
|---|---|
| [Installation](docs/installation.md) | every platform, Docker, building |
| [GPU guide](docs/gpu.md) | Metal, Vulkan, CUDA: setup and status |
| [Command line](docs/cli.md) | every command and option |
| [API](docs/api.md) | the OpenAI-compatible server |
| [Configuration](docs/configuration.md) | `config.yaml` and environment variables |
| [Deployment](docs/deployment.md) | running a server safely |
| [Security](docs/security.md) | threat model and protections |
| [Architecture](docs/architecture.md) | how DynaLM is built |
| [Benchmarking](docs/benchmarking.md) | how we measure speed |
| [Roadmap](ROADMAP.md) | what comes next |
| [Changelog](CHANGELOG.md) | what changed |

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Search topics

**Local AI:**
- local LLM, run AI offline, private AI;
- Ollama alternative, llama.cpp alternative, local ChatGPT alternative;
- self-hosted LLM server.

**Hardware:**
- CPU: AVX2, AVX-512, NEON;
- Apple Silicon M1–M4 with Metal;
- AMD Radeon and Intel Arc with Vulkan, NVIDIA with CUDA;
- Snapdragon Windows on ARM, Raspberry Pi.

**Models:**
- Qwen 2.5, Qwen 3, Llama 3.2, Gemma 3;
- DeepSeek-R1 Distill, Phi-3.5, Mistral, Granite.

**Formats:** GGUF, SafeTensors, GPTQ, AWQ, Q4_K_M.

**Integrations:**
- OpenAI-compatible API;
- LangChain, LlamaIndex, LiteLLM;
- Open WebUI, Continue.dev, Cursor.

---

## License and citation

DynaLM is open source under the [Apache License 2.0](LICENSE). Third-party notices are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). To cite DynaLM, see
[CITATION.cff](CITATION.cff).

<p align="center"><sub>Built for local, private and open AI.</sub></p>

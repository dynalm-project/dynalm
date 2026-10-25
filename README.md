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
[![CPU](https://img.shields.io/badge/CPU-AVX2%20%7C%20AVX--512%20%7C%20NEON-f97316?style=flat-square)](docs/installation.md)
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
SEO keywords: local LLM, run LLM locally, run AI on my computer, offline AI, private AI, Ollama alternative, llama.cpp alternative, LLM on CPU, LLM on GPU, Apple Silicon LLM, Metal LLM, Vulkan LLM, CUDA LLM, AMD GPU LLM, Intel GPU LLM, NVIDIA GPU LLM, Windows on ARM LLM, GGUF runner, OpenAI compatible local server, local ChatGPT alternative, Qwen local, Llama local, DeepSeek R1 local.
AEO answer: DynaLM is a free, open-source program that runs large language models on your own computer, on the CPU or the GPU, fully offline. Install it with one command, then run `dynalm run qwen2.5:1.5b` to chat, or `dynalm serve` for an OpenAI-compatible API.
-->

## What is DynaLM?

**DynaLM is a free program that runs AI chat models on your own computer.** It works on your
CPU and on your GPU, and it needs no internet once the model is downloaded.

- **One small program.** It needs no Python, PyTorch or Docker, and no background service.
- **Private.** Your prompts never leave your machine. There is no telemetry.
- **Familiar commands.** `pull`, `run` and `serve`, like Ollama.
- **Works with your tools.** It has a built-in OpenAI-compatible API (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, streaming), so tools written for the OpenAI API, such as the
  OpenAI SDK, LangChain, LlamaIndex, Open WebUI, Continue.dev and Cursor, are designed to connect to it.
  For this release only DynaLM's own client and test suite are verified against it.
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
| **Windows 10 / 11 (x64 and ARM64)** | PowerShell: `irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 \| iex` |
| **Docker** | `docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models:/models" ghcr.io/dynalm-project/dynalm:cpu serve /models/model.gguf --allow-unauthenticated` |

**Which build you get:**
- **Macs with Apple Silicon:** the GPU version (Metal), always.
- **Everything else:** the CPU version. On Linux x86-64 and Windows x64, add `--gpu` (`-Gpu` on
  Windows) for the experimental Vulkan version: `curl -fsSL .../install.sh | sh -s -- --gpu`.

**Notes:**
- No administrator rights are needed.
- Every download is checked with SHA-256.
- The installer adds DynaLM to your `PATH` (skip with `--no-modify-path`, or `-NoPath` on Windows).
  Open a new terminal afterwards, then run `dynalm doctor`.

More detail, including manual downloads and building from source: [docs/installation.md](docs/installation.md).

---

## Where it runs

**Status labels** (this release):
- ✅ **Verified:** checked on real hardware with real models for this release: the full test
  suite plus token-by-token parity with llama.cpp on eight models (Apple M4 Pro, macOS 26).
- ☑️ **CI-tested:** built, tested with the model-free test suite and checked with a fresh install in CI
  on that platform; not yet checked with real models for this release.
- 🧪 **Preview:** implemented and covered by model-free tests; not yet checked with real models for
  this release. Feedback is welcome.
- ⚗️ **Experimental:** built in CI, tested only on a software GPU or not on a GPU at all.

### CPU

| System | Chips | Status |
|---|---|---|
| macOS | Apple Silicon | ✅ Verified |
| macOS | Intel Macs | ☑️ CI-tested (under Rosetta) |
| Linux x86-64 | Intel and AMD (AVX2, AVX-512, AMX) | ☑️ CI-tested |
| Linux ARM64 | 64-bit ARM boards and servers | ☑️ CI-tested |
| Windows 10 / 11 x64 | Intel and AMD | ☑️ CI-tested |
| Windows 11 on ARM | ARM64 PCs | ☑️ CI-tested |
| Docker | linux/amd64, linux/arm64 | 🧪 Preview |

Older CPUs without AVX2 still work, using portable code. `dynalm doctor` shows what your CPU uses.

### GPU

| GPU | Works on | Status |
|---|---|---|
| **Apple Metal** | Apple Silicon Macs | ✅ Verified (M4 Pro) |
| **Vulkan** | AMD, Intel, NVIDIA and Qualcomm GPUs (Windows and Linux) | ⚗️ Experimental (CI: software GPU, lavapipe) |
| **NVIDIA CUDA** | NVIDIA GPUs (Windows and Linux) | ⚗️ Experimental (CI: build only) |

---

## Use your GPU

**There is nothing to set up.** DynaLM uses the GPU first, and the CPU when the GPU can't run
the model:

```bash
dynalm run qwen2.5:1.5b          # GPU first (default)
dynalm run qwen2.5:1.5b --cpu    # use the CPU
dynalm run qwen2.5:1.5b --gpu    # back to the GPU (stops with a clear message if it can't)
```

**Check what it uses:** the first lines show `Device: compiled/Metal/...` (or `Vulkan/...`) for
the GPU, and `Device: CPU/...` for the CPU.

**Make a choice permanent:** set `runtime: device: cpu` in `~/.dynalm/config.yaml`, or
`DYNALM_BACKEND=cpu`. `--gpu` switches back for one run.

**Models that run on the GPU today:** the Qwen2.5 family and other models of the same design.
Llama 3.x, Qwen3, Gemma 3 and mixture-of-experts models (Granite-MoE, Mixtral, Qwen-MoE) run on
the GPU too (MoE: Metal and Vulkan). For other cases `auto` uses the CPU for now.
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
| `llama3.2:1b` | 808 MB | 2 GB | fast chat on laptops (runs on GPU too) |
| `granite3.1-moe:1b` | 822 MB | 2 GB | efficient mixture-of-experts |
| `qwen2.5:1.5b` | 1.1 GB | 2 GB | **best start**: good quality, fast, runs on GPU too |
| `qwen3:1.7b` | 1.8 GB | 2.5 GB | step-by-step reasoning |
| `llama3.2:3b` | 2.0 GB | 3–4 GB | writing and coding (runs on GPU too) |
| `phi3.5:3.8b` | 2.4 GB | 4 GB | maths and code |
| `qwen3:4b` | 2.5 GB | 4 GB | best quality in the list |

**Other ways to get a model:**

```bash
dynalm pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf   # any Hugging Face GGUF
dynalm run ./my-model.gguf                                                          # a file on disk
dynalm models --available                                                          # the full catalog
```

**Supported:**
- **Families:** Llama 1–3.2, Qwen 2 / 2.5 / 3 (dense and MoE), Gemma 1–3, Phi-3 / 3.5 / 4-mini,
  Mistral, Mixtral, IBM Granite, and the DeepSeek-R1 distills.
- **File formats:** GGUF, Hugging Face SafeTensors, GPTQ and AWQ.
- **Quantizations:** Q2_K to Q8_0, IQ4_NL, IQ4_XS, F16, BF16 and F32. `Q4_K_M` is recommended.
- **Not supported yet:** the other i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S),
  TQ1_0 / TQ2_0 and MXFP4. Such a file is refused at load with the tensor and type named.

---

## Speed

### CPU (DynaLM 0.3.0, CPU suite v1)

Measured on one Apple M4 Pro (24 GB, AC power) against llama.cpp's server (commit `9a16f1dbe`, CPU
build native with Apple Accelerate) and Ollama, on the same GGUF files: Qwen2.5-1.5B and Qwen3-4B
(Q4_K_M). Prompts of 128, 512 and 2,048 tokens; 1, 4 and 8 users; 128 output tokens; 4,096 tokens
of context per user; 5 rounds. The workloads were fixed before any measurement
([docs/perf/cpu-suite-v1.md](docs/perf/cpu-suite-v1.md)). Ratio per workload = median over the
rounds; above 1 means DynaLM is better.

| CPU workload | vs llama.cpp |
|---|---|
| 1.5B, one user, 128-token prompt: output tokens/s | 1.08× |
| 1.5B, one user, 2,048-token prompt: whole request | 1.78× |
| 1.5B, 4 users: total tokens/s | 1.95× |
| 1.5B, 8 users: total tokens/s | 1.68× |
| 1.5B, 8 users: 95th-percentile token gap | 1.19× |
| 4B, one user, 512-token prompt: output tokens/s | 1.46× |
| 4B, one user, 2,048-token prompt: first token | 2.26× |
| 4B, one user, 2,048-token prompt: token gap | 1.23× |
| **Geometric mean** | **1.53×** |

Peak server memory: 1,508 MiB (1.5B) and 3,944 MiB (4B), 2.73× and 2.89× less than llama.cpp.
Against Ollama: 1.53× on the three 4B workloads (on 1.5B Ollama stops early despite `ignore_eos`,
so those are not comparable). Next-token agreement with llama.cpp's logits: 100%, mean KL ≤ 4e-7.
These are DynaLM's own CPU kernels, the default CPU path on Apple M4 class CPUs (SME); other CPUs
use ggml's kernels by default and were not measured. The project's 2× target is not met. Details:
[docs/perf/cpu-experiments.md](docs/perf/cpu-experiments.md).

### Metal (DynaLM 0.2.0)

Measured on one machine (Apple M4 Pro, 24 GB, macOS 26.7, AC power) against llama.cpp's server
(commit `9a16f1dbe`) on the same GGUF files: Qwen2.5-1.5B, Qwen3-4B and Qwen3-8B (Q4_K_M) and
Llama-3.1-8B (Q8_0); 512-token prompts, 128 output tokens, 2,048 tokens of context per user,
3 rounds, both engines checked to run on the intended backend. Ratios: DynaLM ÷ llama.cpp for
throughput, llama.cpp ÷ DynaLM for latencies, so above 1 means DynaLM is better; values are medians
over the 3 rounds, ranges across the four models.

**Confidence:** a difference counts as confirmed only when all 3 rounds agree beyond ±10% and both
engines varied by 10% or less between rounds. llama.cpp's multi-user results varied by up to about
30% between rounds, so several differences below are **not confirmed** (the direction was
consistent, the size is uncertain).

| Metal | Result | Confirmed? |
|---|---|---|
| One user | about the same speed (throughput 0.98–1.21×, time per token 0.96–1.24×) | parity on the 8B models; the others inconclusive |
| 16 users: total tokens per second | DynaLM 0.71–0.78× (llama.cpp ahead) | yes, 3 of 4 models |
| 16 users: slowest token gaps (99th percentile) | 7–10× shorter with DynaLM | yes for the 8B models, not for 1.5B / 4B |
| 16 users: first token | 2.1–3.3× sooner with DynaLM (median) | no, on all four models |
| 4 users: 99th-percentile token gaps | longer with DynaLM (0.19–0.30×) | no (consistent direction) |

On Metal DynaLM is not faster than llama.cpp overall: it trades total throughput for first-token
time and worst-case gaps at high concurrency. Metal was not re-measured for 0.3.0, which does not
change the Metal code. Method: [docs/benchmarking.md](docs/benchmarking.md). Earlier results (DynaLM
0.1.5–0.1.6), now withdrawn, stay in [docs/benchmarks.md](docs/benchmarks.md) for reference.

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
DynaLM uses it automatically for faster answers.

### Does DynaLM work on Apple Silicon Macs?
Yes. It runs on Apple Silicon Macs, on the CPU and on the GPU through Metal.

### Does it work with AMD or Intel graphics cards?
Experimentally, through Vulkan: install with `--gpu` (`-Gpu` on Windows). Vulkan is tested in
CI on a software GPU only.

### Does it work with NVIDIA GPUs?
Experimentally, through Vulkan (install with `--gpu`) or CUDA (build from source,
[docs/gpu.md](docs/gpu.md)).

### Does it run on Windows on ARM laptops?
Yes. The normal Windows installer detects ARM64 and installs the native ARM build.

### Is DynaLM an Ollama alternative?
Yes. It uses the same kind of commands (`pull`, `run`, `serve`), the same GGUF model files, and
an OpenAI-compatible API. It ships as one small program with no background service.

### How does DynaLM compare to llama.cpp?
On Apple M4 class CPUs DynaLM runs its own CPU kernels: 1.53× llama.cpp on CPU suite v1, with
2.7–2.9× less memory. On Metal, and on other CPUs by default, both run ggml's kernels. DynaLM's
output is checked against llama.cpp's token by token (teacher-forced parity). See [Speed](#speed).

### Is it private?
Yes. Nothing is sent anywhere. DynaLM only goes online when you run `dynalm pull`.

### How much memory do I need?
About the model file size plus 1 GB. A 1.5B model needs about 2 GB, and a 4B model about 4 GB.

### Can I choose CPU or GPU myself?
Yes. Add `--cpu` or `--gpu` to any command, or set `runtime: device: cpu` in
`~/.dynalm/config.yaml`. The default is GPU first.

### Can I trade speed for accuracy?
Yes, with `--mode`. `auto` (the default) picks the fastest validated path; on the CPU that is
DynaLM's own kernels on Apple M4 class CPUs and ggml's kernels (llama.cpp-level accuracy)
elsewhere. `--mode exact` keeps DynaLM's own accuracy contract (perplexity within +1%, DD-053) on
every device; on CPUs without SME it is slower. `--mode safe` uses
plain fp32 CPU kernels, for debugging. `dynalm autotune` shows what each mode runs on your machine;
`dynalm autotune --run <model>` picks the best CPU thread count and remembers it
([docs/cli.md](docs/cli.md#execution-modes-and-autotune)).

---

## Known limitations

- **Platform status for this release:** verified on macOS on Apple Silicon (CPU and Metal), on one
  machine (M4 Pro). Linux and Windows are CI-tested (model-free tests in CI); Vulkan and CUDA are
  Experimental. Models that a GPU cannot run yet fall back to the CPU under `--backend auto`;
  `--gpu` reports why instead of falling back.
- **16 users on Metal:** total throughput is 0.71–0.78× llama.cpp's (worst-case token gaps are
  much shorter, confirmed on the 8B models; see [Speed](#speed)).
- **4 users:** the 99th-percentile gap between tokens is longer than llama.cpp's (0.19–0.30×,
  consistent in every round but not confirmed by the verdict rule): new prompts are processed in
  chunks alongside the running streams.
- **CPU:** measured on one Apple M4 Pro only. Single-user decode is 1.08–1.46× llama.cpp; with 8
  users on Qwen3-4B, token gaps reach about 1 s while new 512-token prompts are processed. Other
  CPUs (Apple M1–M3, x86-64, Linux ARM64) have not been measured.
- **Memory limits are enforced at load.** A KV cache that does not fit the GPU working set is
  refused with the largest `--ctx` that fits. Example: Llama-3.1-8B Q8_0 with 16 users × 4,096
  tokens on a 24 GB Mac (8.0 GiB of KV, 6.9 GiB available) is refused; `--ctx 57024` or
  `--kv q8_0` runs. Granite MoE on the CPU uses about 0.7 GB more for its repacked expert weights.
- **Long-running Metal servers:** sparse attention keeps a cache-filled KV pool fast for contexts up
  to 4,096 tokens; longer contexts still slow down as the prefix cache fills the pool.
- **Tested hardware for the newest changes.** The ARM int8 kernels, runtime ARM feature detection,
  AutoTune, IQ4 support, the GPT-4o tokenizer and the Phi-3 / Phi-4 fixes in the changelog's
  Unreleased section were measured on an Apple M4 Pro (macOS, CPU, Metal, Vulkan via MoltenVK)
  only. Other platforms run the model-free test suite in CI; their speed and accuracy with these
  changes are not measured yet.
- **Windows on ARM:** MSVC builds have neither ggml's CPU kernels nor the ARM dot-product
  kernels, so the CPU runs DynaLM's portable fp32 kernels (correct, slower).
- **Linux ARM64 packages:** ggml's CPU kernels are built for the baseline ARMv8.0 so they run on
  every ARM64 CPU, which leaves out their dot-product and repacked kernels (ggml selects those at
  compile time). DynaLM's own kernels pick dot product at run time. A source build with
  `-DDYNALM_GGML_NATIVE=ON` gets ggml's fast kernels for that machine.
- **x86-64 without AVX2** (old CPUs, some emulators): ggml's CPU kernels are not used; DynaLM's
  portable kernels run instead.
- **CUDA:** built and unit-tested in CI without a GPU; not measured on NVIDIA hardware by us.
- **AutoTune tunes the CPU thread count only.** On Windows it cannot read the power and load state,
  so `dynalm autotune --run` stores a result only with `--force`.
- **Tokenizers:** byte-level BPE files need a pre-tokenizer DynaLM implements (GPT-2, Llama 3,
  Qwen2, StarCoder/SmolLM, GPT-4o families); others are refused at load. Old GGUF files without a
  pre-tokenizer tag use the GPT-2 split.
- **Chat templates** are recognized by family (no Jinja engine, see docs/security.md). An
  unrecognized one makes chat mode fail with "model has no recognized chat template"; `--raw`
  sends the text as it is.

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
| [Benchmark results](docs/benchmarks.md) | measured speed, memory and accuracy, with limits |
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
- Apple Silicon with Metal;
- AMD and Intel GPUs with Vulkan, NVIDIA GPUs with CUDA;
- Windows on ARM, ARM64 Linux.

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

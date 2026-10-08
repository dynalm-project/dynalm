<p align="center">
  <img src="public/light.png" alt="DynaLM" width="620">
</p>

<div align="center">

  <h3>Fast, Lightweight Local LLM Inference Engine for Modern CPUs</h3>

  <p>
    <b>Run Llama, Qwen, Gemma, Phi, Mistral, and Granite models directly on your CPU — No GPU required, no Python runtime, 100% private and offline.</b>
  </p>

  <p align="center">
    <a href="https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/dynalm-project/dynalm/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI"></a>
    <a href="https://github.com/dynalm-project/dynalm/releases/latest"><img src="https://img.shields.io/github/v/release/dynalm-project/dynalm?color=3b82f6&label=Version&style=flat-square" alt="Latest Release"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-22c55e.svg?style=flat-square" alt="License"></a>
    <a href="docs/architecture.md"><img src="https://img.shields.io/badge/Language-C%2B%2B20-00599C?style=flat-square&logo=c%2B%2B" alt="C++20"></a>
    <a href="docs/installation.md"><img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-64748b?style=flat-square" alt="Platforms"></a>
    <a href="docs/api.md"><img src="https://img.shields.io/badge/API-OpenAI%20Compatible-a855f7?style=flat-square" alt="OpenAI Compatible"></a>
    <a href="docs/architecture.md"><img src="https://img.shields.io/badge/Hardware-AVX2%20%7C%20NEON-f97316?style=flat-square" alt="CPU Kernels"></a>
  </p>

  <p align="center">
    <a href="#-quick-start"><b>Quick Start</b></a> •
    <a href="#1-installation"><b>Installation</b></a> •
    <a href="#2-supported-models"><b>Supported Models</b></a> •
    <a href="#3-cli-reference"><b>CLI Reference</b></a> •
    <a href="#4-local-openai-compatible-api-server"><b>API Server</b></a> •
    <a href="#6-cpu-performance--speed-optimization"><b>Performance Guide</b></a> •
    <a href="#8-comparison-with-other-engines"><b>Comparison</b></a> •
    <a href="#10-developer-guide--architecture"><b>Docs</b></a>
  </p>

</div>

---

### ✨ Key Features

| Capability | What It Delivers |
| :--- | :--- |
| ⚡ **Hardware-Tuned CPU Kernels** | Native AVX2 vectorization for x86_64 (Intel & AMD) and NEON acceleration for ARM64 & Apple Silicon (M1/M2/M3/M4). |
| 🔒 **100% Private & Local** | Air-gapped execution with zero cloud calls, zero telemetry, and no accounts required. Prompts and tokens never leave your machine. |
| 📦 **Zero-Dependency Native Binary** | Single C++20 binary executable. No Python virtualenvs, no Torch/CUDA runtime packages, and no driver headaches. |
| 🌐 **Drop-In OpenAI API Server** | Built-in high-performance HTTP server with streaming completions (`/v1/chat/completions`), continuous batching, and paged KV cache. |
| 🔄 **Broad Ecosystem Compatibility** | Direct support for GGUF files (llama.cpp & Ollama ecosystems), Hugging Face SafeTensors directories, and GPTQ / AWQ checkpoints. |
| 💻 **Universal Cross-Platform** | Native one-line automated installers for Linux, macOS, and Windows. |

---

## 🚀 Quick Start

Get from zero to interactive AI chat in seconds:

```bash
# 1. Download an optimized model (~2.5 GB)
dynalm pull qwen3:4b

# 2. Launch an interactive terminal chat
dynalm run qwen3:4b
```

> **Tip:** You can also ask single questions directly without entering interactive chat:
> ```bash
> dynalm run qwen3:4b -p "Explain what a CPU cache is in one sentence."
> ```

---

## 1. Installation

DynaLM provides automated, non-admin installers for Linux, macOS, and Windows.

### Linux & macOS

Open your terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh
```

### Windows (PowerShell)

Open PowerShell and execute:

```powershell
irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex
```

### Verify Your Installation

Open a **new** terminal session and verify your setup:

```bash
dynalm doctor
```

The diagnostics report will check your CPU vector extensions, available RAM, and verify ready status (`Status: Ready`).

<details>
<summary><b>Installation Details & Locations</b></summary>

The automated installer:
1. Detects your CPU architecture and OS platform.
2. Verifies download integrity using cryptographic SHA-256 checksums.
3. Installs DynaLM in your user space (no root or administrator permissions needed).

**Binary Locations:**
- **Linux & macOS:** `~/.local/bin/dynalm`
- **Windows:** `%LOCALAPPDATA%\Programs\DynaLM\bin\dynalm.exe`

For manual downloads, building from source, or Docker setup, refer to [docs/installation.md](docs/installation.md).

</details>

---

## 2. Supported Models

### Ready-to-Use Catalog

DynaLM includes shortcuts for leading open-weight LLMs. Run `dynalm pull <name>` or `dynalm run <name>`:

| Model Identifier | Download Size | Free RAM Needed | Best For |
| :--- | :--- | :--- | :--- |
| `smollm2:135m` | ~145 MB | ~1 GB | Fast unit testing and minimal environments |
| `gemma3:270m` | ~292 MB | ~1 GB | Lightweight, compact Google model |
| `qwen2.5:0.5b` | ~491 MB | ~1 GB | High-speed, low-footprint reasoning |
| `qwen3:0.6b` | ~639 MB | ~1 GB | Compact thinking model with reasoning scratchpad |
| `gemma3:1b` | ~806 MB | ~2 GB | General-purpose lightweight Google model |
| `llama3.2:1b` | ~808 MB | ~2 GB | Ultra-fast Meta instruct model |
| `granite3.1-moe:1b` | ~822 MB | ~2 GB | IBM Mixture-of-Experts (MoE) efficiency |
| `qwen2.5:1.5b` | ~1.1 GB | ~2 GB | Optimal balance of speed and generation quality |
| `qwen3:1.7b` | ~1.8 GB | ~2.5 GB | Mid-sized model with advanced chain-of-thought |
| `llama3.2:3b` | ~2.0 GB | ~3–4 GB | High-quality Meta model for complex reasoning |
| `phi3.5:3.8b` | ~2.4 GB | ~4 GB | Microsoft reasoning & coding tuned model |
| `qwen3:4b` | ~2.5 GB | ~4 GB | **Recommended:** Premier quality & reasoning for CPU |

> **Aliases:** Short names like `qwen3`, `llama`, `llama:3b`, `gemma`, `gemma:270m`, and `phi` are automatically resolved. Run `dynalm models --available` to see all supported presets.

---

### Run Any GGUF from Hugging Face

DynaLM can directly resolve and stream GGUF models hosted on [Hugging Face](https://huggingface.co):

```bash
# Pull directly via Hugging Face repository path
dynalm pull bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Verify model compatibility before downloading full weights (downloads only 256 KB)
dynalm pull <url-or-repo> --check

# Execute local GGUF files directly
dynalm run ./path/to/any-model.gguf
```

---

### Architecture & Format Compatibility

| Supported Architectures | In Development / Planned |
| :--- | :--- |
| **Llama**: 1, 2, 3, 3.1, 3.2 (also Mistral, SmolLM, DeepSeek-R1-Distill) | Qwen 3.5 / 3.8 |
| **Qwen**: 2, 2.5, 3 (including MoE variants) | LFM2 |
| **Gemma**: 1, 2, 3 (text generation) | Full-parameter DeepSeek-V3 / R1 671B |
| **Phi**: 3 / 3.5 | Legacy IQ quantization variants (IQ2, IQ3, IQ4) |
| **Mixtral & Granite**: IBM Granite and Granite-MoE | GPU inference acceleration ([Roadmap](ROADMAP.md)) |

- **Weight Formats:** GGUF (`.gguf`), Hugging Face SafeTensors folders (`.safetensors`), and GPTQ/AWQ checkpoints.
- **Quantization Types:** `Q2_K` through `Q8_0`, `F16`, `BF16`, and `F32`. We recommend `Q4_K_M` for the optimal performance-to-size trade-off.

---

### Memory Requirements Guide

Estimated free RAM needed based on model parameter count:

| Parameter Scale | Typical Quantization (`Q4_K_M`) | Minimum System RAM |
| :---: | :---: | :---: |
| **< 1 Billion** | ~300 MB – 650 MB | **1 GB** |
| **1 – 2 Billion** | ~800 MB – 1.2 GB | **2 GB** |
| **3 – 4 Billion** | ~2.0 GB – 2.5 GB | **4 GB** |
| **7 – 8 Billion** | ~4.5 GB – 5.5 GB | **8 GB** |

To inspect exact model memory requirements, run:
```bash
dynalm inspect <model-name-or-path>
```

---

## 3. CLI Reference

### Primary Commands

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `dynalm run <model>` | Start an interactive terminal chat session | `dynalm run qwen3:4b` |
| `dynalm run <model> -p "..."` | Generate a single completion and exit | `dynalm run qwen3:4b -p "Hello"` |
| `dynalm chat <model>` | Alias for interactive chat mode | `dynalm chat llama:3b` |
| `dynalm serve <model>` | Start a local OpenAI-compatible HTTP server | `dynalm serve qwen3:4b --port 8000` |
| `dynalm stop` | Gracefully shut down background server | `dynalm stop` |
| `dynalm pull <target>` | Download a model from registry or Hugging Face | `dynalm pull gemma3:1b` |
| `dynalm models` | List all locally cached models | `dynalm models` |
| `dynalm models --available` | Browse available catalog presets | `dynalm models --available` |
| `dynalm models rm <model>` | Delete a locally cached model (`-y` to skip confirmation) | `dynalm models rm gemma3:1b` |
| `dynalm inspect <model>` | Display architecture, tensor sizes, and memory specs | `dynalm inspect qwen3:4b` |
| `dynalm doctor` | System check (CPU features, RAM, binary health) | `dynalm doctor` |
| `dynalm config show` | Display active configuration values and sources | `dynalm config show` |
| `dynalm config init` | Generate a default `~/.dynalm/config.yaml` file | `dynalm config init` |
| `dynalm benchmark <model>` | Benchmark token generation throughput & latency | `dynalm benchmark qwen3:4b` |
| `dynalm version` | Display version information | `dynalm --version` |
| `dynacorec <file>` | Low-level DynaCore kernel compiler ([docs](docs/dynacore-language.md)) | `dynacorec layer.dyna --dump-kernels` |

> **Global flags:** Use `-q` (quiet), `-v` (verbose), or `--log-level debug` before the command (e.g., `dynalm -v run qwen3:4b`).

---

### In-Chat Slash Commands

Inside an interactive chat session (`dynalm run` or `dynalm chat`), use these control commands:

| Command | Action |
| :--- | :--- |
| `/bye` | Exit the chat session |
| `/clear` | Clear the ongoing conversation context |
| `/system <text>` | Set runtime system prompt (e.g., `/system You are a succinct coding assistant`) |
| `/think on` \| `/think off` | Toggle Qwen3 reasoning scratchpad (turn `off` for faster, concise responses) |
| `/set <key> <val>` | Dynamically update parameters (`temp`, `top_p`, `top_k`, `min_p`, `max_tokens`, `seed`) |
| `/show` | Display active model metadata and sampling configuration |
| `/stats on` \| `/stats off` | Display tokens/second generation metrics after each response |
| `/help` | Print in-chat help menu |

---

### Execution Flags (`run` & `serve`)

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `-t, --threads <N>` | CPU worker threads | Number of physical cores |
| `-c, --ctx <N>` | Context window size in tokens | `4096` (`run`) / auto-sized (`serve`) |
| `-n, --max-tokens <N>` | Maximum output tokens per reply | `128` (single prompt) / `2048` (chat) |
| `--temp <float>` | Sampling temperature (creativity) | `0.0` (with `-p`) / `0.8` (chat) |
| `--top-p, --top-k, --min-p` | Advanced sampling controls | Standard model defaults |
| `--kv <type>` | KV cache precision (`f16`, `q8_0`) | `f16` (use `q8_0` to save ~50% cache RAM) |
| `--execution compiled` | DynaCore compiled execution backend | `reference` |
| `--host <address>` | Server host address (`serve`) | `127.0.0.1` |
| `--port <port>` | Server listening port (`serve`) | `8000` |
| `--max-active <N>` | Concurrent in-flight request limit | `64` |

*For complete CLI documentation, view [docs/cli.md](docs/cli.md).*

---

## 4. Local OpenAI-Compatible API Server

Start the lightweight, multi-threaded inference server:

```bash
dynalm serve qwen3:4b --port 8000
```

DynaLM serves an OpenAI-compliant REST API. Point any tool, framework, or client directly to `http://127.0.0.1:8000/v1`:

### Using cURL

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "How do CPU vector instructions work?"}],
    "stream": true
  }'
```

### Using Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused"  # DynaLM requires no auth
)

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Give me a quick high-five in words."}]
)

print(response.choices[0].message.content)
```

### Supported Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat completions with SSE streaming support |
| `/v1/completions` | `POST` | Plain text raw completions |
| `/v1/models` | `GET` | List currently loaded active model(s) |
| `/health` | `GET` | Server liveness and readiness probe |
| `/metrics` | `GET` | Prometheus-compatible inference latency and memory metrics |

*Supports seamless integration with LangChain, LlamaIndex, LiteLLM, Open WebUI, and Continue.dev. See [docs/api.md](docs/api.md).*

---

## 5. Configuration & Storage

DynaLM is zero-configuration by default. To persist custom defaults, initialize a configuration file:

```bash
dynalm config init    # Writes ~/.dynalm/config.yaml
dynalm config show    # Inspects effective settings and source hierarchy
```

```yaml
runtime:
  threads: auto
  context_length: 8192

kv_cache:
  dtype: q8_0

server:
  host: 0.0.0.0
  port: 8000
```

> **Precedence Hierarchy:** Command-line flags > Environment variables (e.g. `DYNALM_PORT=9000`) > Configuration file (`config.yaml`) > Built-in defaults.

### Storage Directory Layout

| Data | Path | Environment Override |
| :--- | :--- | :--- |
| **Model Weights** | `~/.dynalm/models` (Windows: `%USERPROFILE%\.dynalm\models`) | `DYNALM_MODELS_DIR` |
| **User Settings** | `~/.dynalm/config.yaml` | `DYNALM_CONFIG` |
| **Installed Binary** | `~/.local/bin/dynalm` (Windows: `%LOCALAPPDATA%\Programs\DynaLM\bin`) | Standard system `PATH` |

*DynaLM also automatically inspects a `./models` directory in the current working directory.*

---

## 6. CPU Performance & Speed Optimization

LLM inference on a CPU is primarily **memory-bandwidth bound** during token generation: every decoded token requires transferring model weights through memory channels.

### Optimization Strategies

1. **Pick the Right Model Scale:** For rapid responses, prefer smaller or MoE architectures (`qwen2.5:1.5b` or `llama3.2:1b`) over larger checkpoints.
2. **Quantization Selection:** Use `Q4_K_M` for optimal inference throughput with minimal perplexity degradation.
3. **Turn Off Chain-of-Thought for Short Queries:** With reasoning models like Qwen3, use `/think off` to bypass lengthy reasoning tokens when concise answers are preferred.
4. **Leverage the Server Mode:** Keep the model weights resident in RAM with `dynalm serve` to eliminate startup and initialization latencies.
5. **Enable Compiled Execution:** Pass `--execution compiled` to engage specialized DynaCore kernel pipelines for an additional ~8% single-stream throughput gain.
6. **Ensure Adequate Free RAM:** Keep sufficient free physical memory to avoid page swapping, which drastically degrades CPU throughput.
7. **Serving several users at once:** DynaLM keeps an extra interleaved copy of Q4_K weights for batched decoding (2-8 requests at a time), about +5% of the Q4_K weight size per tensor packed. It is skipped tensor by tensor when it would leave less than 3 GiB of RAM free. Control it with `DYNACORE_Q4_REPACK=off|auto|always`. `--policy latency` shortens the pauses running streams see while a new long prompt is read; `--policy throughput` turns that protection off.

### Measured Throughput (Reference Laptop: Intel Core i7-1255U)

| Model Scale | Quantization | Generation Speed |
| :---: | :---: | :---: |
| **1.5 Billion** (`qwen2.5:1.5b`) | `Q4_K_M` | **~13.0 tokens/sec** |
| **4.0 Billion** (`qwen3:4b`) | `Q4_K_M` | **~6.0 tokens/sec** |

---

## 7. Troubleshooting

| Symptom | Probable Cause | Recommended Resolution |
| :--- | :--- | :--- |
| `dynalm: command not found` | PATH variable not reloaded | Open a new terminal window or add `~/.local/bin` to your shell `PATH`. |
| Windows: "An Application Control policy has blocked this file" | Windows Smart App Control | Permit execution in Windows Security settings, or compile directly from source. |
| Slow generation / high RAM warning in `dynalm doctor` | Insufficient physical memory | Switch to a smaller model (`1b` or `1.5b`) and close high-memory background applications. |
| `"architecture ... is not supported"` | Unsupported model family | Check [Supported Model Families](#architecture--format-compatibility) or open a feature request. |
| ISA or kernel CPU exception | Unsupported vector instructions | Run with `DYNACORE_ISA=generic dynalm run ...` to force the portable fallback path. |

> To report an issue, run `dynalm doctor` and paste the sanitized diagnostic output in a [GitHub Issue](https://github.com/dynalm-project/dynalm/issues).

---

## 8. Comparison with Other Engines

| Capability | DynaLM | Ollama | llama.cpp | vLLM |
| :--- | :---: | :---: | :---: | :---: |
| **Target Architecture** | **CPU-First** | CPU & GPU | CPU & GPU | GPU Clusters |
| **GGUF Format Support** | **Native** | Native | Native | Partial |
| **SafeTensors / GPTQ / AWQ** | **Native CPU** | Import required | Conversion required | GPU-only |
| **OpenAI-Compatible REST API** | **Built-in** | Built-in | Built-in | Built-in |
| **Continuous Batching & Paged KV** | **Native** | Via llama.cpp | Partial | Native |
| **Direct Model Registry (`pull model:size`)** | **Yes** | Yes | No | No |
| **Runtime & Dependencies** | **Single C++20 Binary** | Go + C++ | C / C++ | Python + CUDA |
| **GPU Acceleration** | Planned | Yes | Yes | Yes |

* **Choose DynaLM** when deploying on CPU-based hardware: consumer laptops, desktops without discrete GPUs, cost-effective cloud CPU instances, or edge ARM devices.
* **Choose Ollama, llama.cpp, or vLLM** if you have dedicated high-end GPU hardware available today.

---

## 9. Frequently Asked Questions

<details>
<summary><b>How do I run an LLM locally without a discrete GPU?</b></summary>
<br>
Install DynaLM and run <code>dynalm run qwen3:4b</code>. It automatically downloads the 4B parameter model and begins interactive generation using optimized vector instructions on your CPU.
</details>

<details>
<summary><b>Can I run Llama 3, Qwen 3, or Gemma 3 on my computer?</b></summary>
<br>
Yes. Simply run <code>dynalm run llama3.2:3b</code>, <code>dynalm run qwen3:4b</code>, or <code>dynalm run gemma3:1b</code>. Any standard GGUF model hosted on Hugging Face can also be loaded via <code>dynalm pull &lt;owner&gt;/&lt;repo&gt;/&lt;file&gt;.gguf</code>.
</details>

<details>
<summary><b>Is DynaLM an alternative to Ollama or llama.cpp?</b></summary>
<br>
Yes, specifically engineered for CPU inference. It provides familiar commands (<code>pull</code>, <code>run</code>, <code>serve</code>) and supports the same GGUF model ecosystem while featuring an independent C++ engine with custom AVX2/NEON vector kernels and high-concurrency continuous batching.
</details>

<details>
<summary><b>Can I connect LangChain, LlamaIndex, or the OpenAI SDK?</b></summary>
<br>
Yes. Launch the background daemon with <code>dynalm serve &lt;model&gt;</code> and configure your client to connect to <code>http://127.0.0.1:8000/v1</code>.
</details>

<details>
<summary><b>Does DynaLM run on Apple Silicon (M1/M2/M3/M4) and Raspberry Pi?</b></summary>
<br>
Yes. macOS ARM64 Apple Silicon is natively supported and validated in CI. Linux ARM64 (such as Raspberry Pi 4/5 with a 64-bit OS) runs using native NEON vectorization.
</details>

<details>
<summary><b>Are my prompts private? Does DynaLM phone home?</b></summary>
<br>
Completely private. DynaLM contains zero telemetry and zero tracking. The internet is only accessed when you explicitly request a model download using <code>dynalm pull</code>.
</details>

<details>
<summary><b>Is DynaLM free for commercial use?</b></summary>
<br>
Yes. DynaLM is open-source under the Apache-2.0 license. Downloaded model weights are governed by their respective author licenses (Meta Llama, Google Gemma, Qwen, etc.).
</details>

---

## 10. Developer Guide & Architecture

DynaLM is architected in two clean layers:
1. **DynaLM:** High-level model loader, CLI, request scheduler, and OpenAI-compatible server.
2. **DynaCore:** Low-level high-performance tensor execution engine, memory manager, and compiler.

| Documentation Guide | Focus Area |
| :--- | :--- |
| 🛠️ [docs/build.md](docs/build.md) | Source build instructions and CI workflows |
| 🏛️ [docs/architecture.md](docs/architecture.md) | Architectural overview of DynaLM and DynaCore |
| 📦 [docs/dynalm.md](docs/dynalm.md) & [docs/dynacore.md](docs/dynacore.md) | Model runtimes, tensor structures, and memory arenas |
| ⚙️ [docs/dynacore-compiler.md](docs/dynacore-compiler.md) | DynaCore intermediate representation (IR) and JIT compiler |
| 📝 [docs/dynacore-language.md](docs/dynacore-language.md) | The `.dyna` kernel specification domain-specific language |
| 📋 [CHANGELOG.md](CHANGELOG.md) & [ROADMAP.md](ROADMAP.md) | Release history, migration guides, and future plans |

### Building from Source

```bash
cmake --preset release
cmake --build --preset release
```

Contributions are enthusiastically welcomed! Please review our [Contribution Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md). For security reports, refer to [SECURITY.md](SECURITY.md).

---

## 📄 License & Attribution

DynaLM is released under the **[Apache License 2.0](LICENSE)**. Third-party components and attributions are documented in [NOTICE](NOTICE).

If you use DynaLM in academic research or production infrastructure, please cite our project using [CITATION.cff](CITATION.cff).

<p align="center">
  <sub>Built with ❤️ for local, private, and open-source AI computing.</sub>
</p>

# DynaLM

[![CI](https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml/badge.svg)](https://github.com/dynalm-project/dynalm/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

**Run AI chat models on your own computer, using only the CPU.**

- No GPU, no Python, no account, no internet needed once a model is downloaded.
- One small program for Linux, macOS (Intel and Apple Silicon) and Windows.
- Chat in the terminal, or run a local server that works with any app built for the OpenAI API.

```sh
dynalm pull qwen3:4b      # download a model
dynalm run qwen3:4b       # chat with it
```

---

## 1. Install

**Linux or macOS:** open a terminal and run:

```sh
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh
```

**Windows:** open PowerShell and run:

```powershell
irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex
```

Then **open a new terminal** and check that it works:

```sh
dynalm doctor
```

The last line should say `Status: Ready`.

The installer:

1. Picks the right download for your computer.
2. Checks the file is genuine, using its SHA-256 checksum.
3. Installs it just for your user, without admin rights.

Where it installs:

- Linux and macOS: `~/.local/bin`
- Windows: `%LOCALAPPDATA%\Programs\DynaLM\bin`

Other ways to install (manual download, from source, Docker): [docs/installation.md](docs/installation.md).

## 2. Your first chat

```sh
dynalm pull qwen3:4b     # downloads about 2.5 GB, once
dynalm run qwen3:4b      # starts a chat; type /bye to leave
```

Ask one question without entering chat mode:

```sh
dynalm run qwen3:4b -p "Explain what a CPU is in one sentence."
```

## 3. Models

### Ready-to-use names

These short names work with `dynalm pull` and `dynalm run`:

| Name | Download size | Good for |
|---|---|---|
| `smollm2:135m` | 145 MB | testing; very fast, basic answers |
| `gemma3:270m` | 292 MB | tiny Google model |
| `qwen2.5:0.5b` | 491 MB | small and quick |
| `qwen3:0.6b` | 639 MB | small model that "thinks" before answering |
| `gemma3:1b` | 806 MB | small Google model |
| `llama3.2:1b` | 808 MB | small Meta model |
| `granite3.1-moe:1b` | 822 MB | IBM "mixture of experts" model |
| `qwen2.5:1.5b` | 1.1 GB | good balance of speed and quality |
| `qwen3:1.7b` | 1.8 GB | mid-size thinking model |
| `llama3.2:3b` | 2.0 GB | Meta; good quality |
| `phi3.5:3.8b` | 2.4 GB | Microsoft |
| `qwen3:4b` | 2.5 GB | best quality in this list |

Short forms also work: `qwen3`, `llama`, `llama:3b`, `gemma`, `gemma:270m` and `phi`. Run
`dynalm models --available` to see this list on your machine, with what is already downloaded.

**Where they come from:** every name points to one file on [Hugging Face](https://huggingface.co),
the public site where AI models are shared. They come from the official Qwen account and from
the well-known `unsloth` and `bartowski` accounts. Each model keeps its maker's license
(Gemma, Llama, Qwen, ...).

### Any other model

DynaLM runs most GGUF files, the same format llama.cpp and Ollama use. Give `dynalm pull` a
Hugging Face path or link instead of a name:

```sh
dynalm pull bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
dynalm pull <link> --check     # only check whether DynaLM can run it (downloads 256 KB)
dynalm run ./path/to/any-model.gguf
```

### Which model families work

| Works | Not yet |
|---|---|
| Llama 1, 2, 3, 3.1, 3.2 (also Mistral, SmolLM, DeepSeek-R1-Distill) | Qwen 3.5 / 3.8 |
| Qwen 2, 2.5, 3, including their MoE versions | LFM2 |
| Gemma 1, 2, 3 (text) | full DeepSeek-V3 / R1 |
| Phi-3 / 3.5 | files with "IQ" quantization (IQ2, IQ3, IQ4 ...) |
| Mixtral, IBM Granite and Granite-MoE | |

`dynalm pull` refuses unsupported models before downloading them.

**File formats:** GGUF; Hugging Face model folders (`.safetensors`); GPTQ and AWQ checkpoints.

**Quantizations:** Q2_K to Q8_0, F16, BF16 and F32. Pick `Q4_K_M` for the best balance. `Q8_0`
is closer to the original but twice the size.

### How much memory do I need?

About the size of the model file, plus a little for the conversation:

| Model size | Example | Free RAM needed |
|---|---|---|
| under 1B | `qwen3:0.6b` | about 1 GB |
| 1–2B | `llama3.2:1b`, `qwen2.5:1.5b` | about 2 GB |
| 3–4B | `qwen3:4b`, `llama3.2:3b` | about 3–4 GB |
| 7–8B | Llama-3.1-8B (Q4_K_M) | about 6 GB |

`dynalm inspect <model>` prints an estimate for any model.

## 4. All commands

| Command | What it does | Example |
|---|---|---|
| `dynalm run <model>` | Chat in the terminal | `dynalm run qwen3:4b` |
| `dynalm run <model> -p "text"` | Answer one question and exit | `dynalm run qwen3:4b -p "Hi"` |
| `dynalm chat <model>` | Same as `run` without `-p` | `dynalm chat llama:3b` |
| `dynalm serve <model>` | Start a local OpenAI-compatible server | `dynalm serve qwen3:4b --port 8000` |
| `dynalm stop` | Stop a running server gracefully | `dynalm stop` |
| `dynalm pull <name or link>` | Download a model | `dynalm pull gemma3:1b` |
| `dynalm models` | List downloaded models | `dynalm models` |
| `dynalm models --available` | List the names you can pull | `dynalm models --available` |
| `dynalm models rm <name>` | Delete a downloaded model (asks first; `-y` skips) | `dynalm models rm gemma3:1b` |
| `dynalm inspect <model>` | Show model details and memory needed | `dynalm inspect qwen3:4b` |
| `dynalm doctor` | Check your computer and the install; paste this into bug reports | `dynalm doctor` |
| `dynalm config` | Show your settings and where each comes from | `dynalm config show` |
| `dynalm config init` | Create a starter settings file | `dynalm config init` |
| `dynalm benchmark <model>` | Measure speed (tokens per second, response times) | `dynalm benchmark qwen3:4b` |
| `dynalm version` | Show the version | `dynalm --version` |
| `dynacorec <file>` | Developer tool: the DynaCore compiler ([docs](docs/dynacore-language.md)) | `dynacorec layer.dyna --dump-kernels` |

`<model>` can be a name (`qwen3:4b`), a `.gguf` file, or a Hugging Face model folder.

Global flags go before the command:

- `-q`: quiet, errors only.
- `-v`: verbose.
- `--log-level debug`: more detail still.

Example: `dynalm -q run qwen3:4b`.

### Commands inside a chat

| Type | Effect |
|---|---|
| `/bye` | leave the chat |
| `/clear` | forget the conversation so far |
| `/system <text>` | set the assistant's instructions, e.g. `/system Answer like a pirate` |
| `/think on` or `/think off` | turn Qwen3's thinking step on or off; off gives faster, shorter replies |
| `/set temp 0.7` | change a setting: `temp`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `max_tokens`, `seed` |
| `/show` | show the model and the current settings |
| `/stats on` or `/stats off` | show or hide speed numbers after each reply |
| `/help` | list these commands |

### Useful options for `run` and `serve`

| Option | Meaning | Default |
|---|---|---|
| `-n 256` / `--max-tokens 256` | longest reply, in tokens | 128 for one question; 2048 in chat |
| `--temp 0.7` | creativity: 0 always picks the most likely word, higher is more varied | 0 with `-p`; 0.8 in chat |
| `--top-p`, `--top-k`, `--min-p` | other ways to limit word choice | |
| `--seed 42` | same answer every time | random |
| `--system "text"` | assistant instructions (`run`) | |
| `-t 8` / `--threads 8` | CPU threads to use | number of physical cores |
| `-c 8192` / `--ctx 8192` | how much conversation the model remembers, in tokens | 4096 (`run`); sized from free RAM (`serve`) |
| `--kv q8_0` | store the conversation memory in half the RAM, at the same speed | `f16` |
| `--execution compiled` | optimized execution mode; up to about 8% faster for one user | `reference` |
| `--host 0.0.0.0` / `--port 8000` | where the server listens (`serve`) | `127.0.0.1:8000` |
| `--max-active 64` | requests the server handles at once before it answers "busy" | 64 |

Type `dynalm run` or `dynalm serve` with no model to see every option. The full reference is in
[docs/cli.md](docs/cli.md).

## 5. Use it from other apps (OpenAI-compatible API)

Start the server:

```sh
dynalm serve qwen3:4b
```

Any program that talks to the OpenAI API can now use your local model. Set its "base URL" to
`http://127.0.0.1:8000/v1`; any API key works.

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen3:4b", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

| Address | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat; add `"stream": true` to get the reply word by word |
| `POST /v1/completions` | plain text completion |
| `GET /v1/models` | the model being served |
| `GET /health` | is the server ready |
| `GET /metrics` | numbers for monitoring tools (Prometheus) |

This works with the OpenAI SDKs, LangChain, LlamaIndex and other OpenAI-compatible apps. Details:
[docs/api.md](docs/api.md).

## 6. Settings

You don't need any settings: DynaLM picks threads, memory and everything else automatically.
To change the defaults, create a settings file:

```sh
dynalm config init       # writes ~/.dynalm/config.yaml with comments
dynalm config show       # shows every setting and where it comes from
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

Priority: command-line options beat environment variables (`DYNALM_PORT=9000`), which beat the
file. All keys: [docs/configuration.md](docs/configuration.md).

### Where things are stored

| What | Where |
|---|---|
| Downloaded models | `~/.dynalm/models` (Windows: `C:\Users\<you>\.dynalm\models`); change with `DYNALM_MODELS_DIR` |
| Settings | `~/.dynalm/config.yaml` |
| Program | `~/.local/bin` (Windows: `%LOCALAPPDATA%\Programs\DynaLM\bin`) |

A `models` folder in the current directory is searched too.

## 7. Speed tips

On a CPU, every word the model writes means reading the whole model from memory. So speed
depends mostly on model size and your RAM speed.

- **Use a smaller model** for faster replies: `qwen2.5:1.5b` instead of `qwen3:4b`, or `Q4_K_M`
  instead of `Q8_0`.
- **Keep enough free RAM.** If the model doesn't fit, it is read from disk and gets very slow.
  Close large apps first.
- **Use `/think off`** with Qwen3 for short answers. The model skips its long thinking step.
- **Use `dynalm serve`** when you ask many questions: the model stays loaded.
- **Try `--execution compiled`.** On a 1.5B model it gave about 8% more tokens per second for a
  single user. Bigger models and many users see little change.

Rough speed on a laptop CPU (Intel i7-1255U): about 13 tokens per second with a 1.5B model, and
about 6 with a 4B model.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `dynalm: command not found` | open a new terminal; or add `~/.local/bin` to your `PATH` |
| Windows: "An Application Control policy has blocked this file" | Windows Smart App Control blocks unsigned programs; allow it in Windows Security, or build from source |
| Very slow, or "only X GiB RAM available" in `dynalm doctor` | use a smaller model and close other programs |
| "architecture ... is not supported" | that model family isn't supported yet (see section 3) |
| Anything else | run `dynalm doctor` and include its output in an [issue](https://github.com/dynalm-project/dynalm/issues) |

To force the simplest CPU code path, for example to rule out a CPU feature problem, run
`DYNACORE_ISA=generic dynalm run ...`.

## 9. FAQ

**Is this like Ollama or llama.cpp?**
Yes, for CPUs. It reads the same GGUF files and has an OpenAI-compatible server. It is built to
serve several users at once on a CPU-only machine. If you have a GPU, Ollama, llama.cpp or vLLM
will be faster, because DynaLM is CPU-only for now.

**Does it work on Apple Silicon or a Raspberry Pi?**
Yes, Apple Silicon (M1 to M4) is supported and tested. A Raspberry Pi 4/5 with a 64-bit OS uses
the same ARM64 build. Small models such as `qwen3:0.6b` or `llama3.2:1b` fit; speed there has not
been measured yet.

**Does it need the internet?**
Only to download a model. After that everything runs offline, and nothing leaves your computer.

**Is it free for commercial use?**
Yes, under the Apache-2.0 license. Each model has its own license from its maker.

## 10. For developers

| Read | About |
|---|---|
| [docs/build.md](docs/build.md) | building from source, tests, CI |
| [docs/architecture.md](docs/architecture.md), [docs/dynalm.md](docs/dynalm.md), [docs/dynacore.md](docs/dynacore.md) | how it is built: DynaLM (models, server) on top of DynaCore (low-level engine) |
| [docs/dynacore-compiler.md](docs/dynacore-compiler.md), [docs/dynacore-ir.md](docs/dynacore-ir.md), [docs/dynacore-language.md](docs/dynacore-language.md) | the DynaCore compiler, its IR and the `.dyna` language |
| [docs/compiler-benchmarks.md](docs/compiler-benchmarks.md), [docs/dynacore-optimization.md](docs/dynacore-optimization.md) | measured performance |
| [docs/design-decisions.md](docs/design-decisions.md) | every major design choice, with evidence |
| [docs/platform-design.md](docs/platform-design.md), [docs/compiler-backends.md](docs/compiler-backends.md), [docs/gpu-backend.md](docs/gpu-backend.md) | overall design, backends, GPU plans |
| [CHANGELOG.md](CHANGELOG.md), [ROADMAP.md](ROADMAP.md) | what changed, what's next |

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues
privately: see [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE). Third-party parts are listed in [NOTICE](NOTICE). To cite DynaLM,
use [CITATION.cff](CITATION.cff).

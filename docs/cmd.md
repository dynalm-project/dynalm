# DynaLM command reference

Every `dynalm` command, with all options and examples. Run `dynalm help`, or any command without
arguments, to see the same help in the terminal.

**Quick start (like Ollama):**

```sh
dynalm pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf     # download a model
dynalm chat models/Qwen3-4B-Q4_K_M.gguf                 # chat with it (Ctrl+D or /bye to exit)
```

| Ollama | DynaLM |
|---|---|
| `ollama pull qwen3:4b` | `dynalm pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` |
| `ollama run qwen3:4b` | `dynalm chat models/Qwen3-4B-Q4_K_M.gguf` (or `dynalm run` without `-p`) |
| `ollama run qwen3:4b "question"` | `dynalm run models/Qwen3-4B-Q4_K_M.gguf -p "question"` |
| `ollama list` | `dynalm list` |
| `ollama rm qwen3:4b` | `dynalm rm Qwen3-4B-Q4_K_M` |
| `ollama show qwen3:4b` | `dynalm inspect models/Qwen3-4B-Q4_K_M.gguf` |
| `ollama serve` | `dynalm serve models/Qwen3-4B-Q4_K_M.gguf` (one model per server) |
| `ollama stop` | `dynalm stop` |

---

## Global options

```text
dynalm [--log-level trace|debug|info|warn|error|off] <command> [args]
dynalm help | --help
```

`--log-level` controls the `[INFO]` lines. Use `--log-level warn` to hide them.

---

## Models

### `dynalm pull`: download a model

```text
dynalm pull <link> [-o DIR] [--force] [--check]
```

| Option | Meaning |
|---|---|
| `<link>` | A Hugging Face file link (page or download link), the short form `<owner>/<repo>/<file>.gguf`, or any http(s) URL to a `.gguf` |
| `-o DIR` | Where to save (default `$DYNALM_MODELS_DIR` or `./models`) |
| `--check` | Only check whether DynaLM can run it (reads 256 KiB, saves nothing) |
| `--force` | Download even if it looks unsupported, or download again |

```sh
dynalm pull https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf
dynalm pull bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
dynalm pull unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf --check
```

- Unsupported architectures are refused before the download starts.
- Interrupted downloads resume when you run the same command again.
- Gated models (e.g. official Llama): accept the license on Hugging Face, then set `HF_TOKEN`.
- Needs `curl`, which is built into Windows 10+, macOS and Linux.

Which models to pick: [model-support.md](model-support.md).

### `dynalm list`: show downloaded models

```text
dynalm list [DIR]
```

Lists GGUF files, Hugging Face model folders and unfinished downloads in `DIR` (default
`$DYNALM_MODELS_DIR` or `./models`). For each it shows architecture, quantization, size, context
length, and whether DynaLM can run it. Nothing is loaded into RAM.

### `dynalm rm`: delete models

```text
dynalm rm <model>... [-y] [--dir DIR]       (alias: dynalm delete)
```

```sh
dynalm rm Qwen3-4B-Q4_K_M                  # asks "delete ... ? [y/N]"
dynalm rm modelA modelB -y                 # no question
dynalm rm st/qwen2.5-0.5b-instruct         # a Hugging Face folder
```

Deletes only GGUF files, `.gguf.part` downloads and Hugging Face model folders. Anything else is refused.

### `dynalm inspect`: model details

```text
dynalm inspect <model> [--metadata] [--tensors]
```

Shows architecture, layers, context length, quantization by tensor type, the RAM estimate and whether it is
supported. `--metadata` adds every GGUF key; `--tensors` lists every tensor. Works on GGUF files and
Hugging Face folders.

---

## Chat and generation

### `dynalm chat`: interactive chat (like `ollama run`)

```text
dynalm chat <model> [options]
dynalm run  <model> [options]          (same thing: run without -p)
```

```text
$ dynalm --log-level warn chat models/Qwen3-4B-Q4_K_M.gguf
Chatting with Qwen3 4B. Type /help for commands, /bye to exit.
>>> /think off
Thinking off.
>>> My name is Sam. What is 2+3?
5, Sam.
>>> What is my name?
Your name is Sam.
>>> /bye
```

The model is loaded once and the conversation is remembered. Follow-up messages start quickly because
DynaLM reuses the earlier conversation instead of reprocessing it. When the conversation outgrows the
context (`-c`), the oldest exchanges are dropped.

**Commands inside the chat:**

| Command | What it does |
|---|---|
| `/help` | List the commands |
| `/bye`, `/exit`, `/quit` | Leave (also Ctrl+D, or Ctrl+Z then Enter on Windows) |
| `/clear` | Forget the conversation |
| `/system TEXT` | Set the system message and start over. `/system` alone shows it |
| `/think on` / `/think off` | Qwen3 reasoning on/off (off = direct answers, much faster) |
| `/set temp 0.7` | Temperature (0 = always the most likely word) |
| `/set top_p 0.9`, `/set top_k 40`, `/set min_p 0.05` | Sampling filters |
| `/set repeat_penalty 1.1` | Discourage repetition |
| `/set max_tokens 4096` | Longest reply |
| `/set seed 42` | Reproducible replies |
| `/show` | Model, settings, messages in history |
| `/stats on` / `/stats off` | Print speed after each reply |
| `"""` | Start a multi-line message; finish it with another `"""` |
| Ctrl+C | Stop the current reply (the chat keeps going) |

Chat uses the same options as `run`, below. Its defaults are conversational: temp 0.8, top-k 40, top-p 0.9,
repeat-penalty 1.1, and up to 2048 tokens per reply. Any sampling flag you pass replaces them.

### `dynalm run`: one prompt, one answer

```text
dynalm run <model> -p <prompt> [options]
```

| Option | Meaning |
|---|---|
| `-p, --prompt TEXT` | The prompt (without it, `run` starts a chat) |
| `-n, --max-tokens N` | Tokens to generate (default 128; chat: 2048) |
| `--system TEXT` | System message |
| `--chat` / `--raw` | Use the model's chat template (default) / send the text as is |
| `--stop TEXT` | Stop when this text appears (repeatable; it is not printed) |
| `-t, --threads N` | Compute threads (default: physical cores) |
| `-c, --ctx N` | Context / KV cache size in tokens (default 4096) |
| `--batch N` | Max tokens per forward pass (default 256) |
| `--kv f16\|f32` | KV cache precision (default f16) |
| `--backend cpu` | Compute backend (only `cpu` is built today) |
| `--no-stream` | Print the answer at the end instead of word by word |

Sampling (default for `run`: greedy, i.e. always the most likely token):

| Option | Meaning |
|---|---|
| `--temp T` | Temperature (0 = greedy; 0.7–1.0 for more varied text) |
| `--top-k K`, `--top-p P`, `--min-p P` | Limit the choice to the most likely tokens |
| `--repeat-penalty R`, `--presence-penalty P`, `--frequency-penalty F` | Discourage repetition |
| `--repeat-last-n N` | How many recent tokens the penalties look at |
| `--seed S` | Reproducible output |

Speculative decoding (faster answers that are identical to normal decoding):

| Option | Meaning |
|---|---|
| `--spec ngram` | Draft from text already in the prompt (good for summaries and code edits) |
| `--spec models/small.gguf` | Use a small model with the same vocabulary as the draft model |
| `--spec-k K` | Tokens drafted per step (default 4) |

```sh
dynalm run models/Qwen3-4B-Q4_K_M.gguf -p "Explain recursion /no_think"
dynalm run models/model.gguf -p "Write a haiku" --temp 0.9 --seed 7
dynalm run models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "..." --spec models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

---

## Server (OpenAI-compatible API)

### `dynalm serve`

```text
dynalm serve <model> [options]
```

| Option | Meaning |
|---|---|
| `--config FILE` | Read options from a file (`key = value` lines) |
| `--host ADDR` | Bind address (default `127.0.0.1`; `0.0.0.0` = reachable from the network) |
| `--port N` | Port (default 8000) |
| `--model-id NAME` | Name reported by `/v1/models` (default: file name) |
| `-t, --threads N\|auto` | Compute threads |
| `-c, --ctx N\|auto` | KV cache capacity in tokens (auto: sized from free RAM) |
| `--batch N\|auto` | Max tokens per forward pass |
| `--kv f16\|f32` | KV cache precision |
| `--max-active N` | Concurrent requests before answering 503 (default 64) |
| `--http-threads N\|auto` | HTTP worker threads |
| `--max-tokens N` | Default `max_tokens` per request (default 1024) |
| `--temperature T` | Default temperature when a request omits it (default 1.0) |
| `--request-timeout S` | Per-request timeout in seconds, 0 = none (default 600) |
| `--shutdown-timeout S` | Time to finish in-flight requests on stop (default 30) |
| `--disable-admin` | Turn off `POST /admin/shutdown` (used by `dynalm stop`) |

Every option can also be set as `DYNALM_<OPTION>` (e.g. `DYNALM_PORT=9000`) or in the config file.
Precedence: command line > environment > file.

### `dynalm stop` (alias `unload`)

```text
dynalm stop [--host 127.0.0.1] [--port 8000]
```

Asks the running server to finish its current requests and exit.

### HTTP endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat (OpenAI format, `"stream": true` for streaming) |
| `POST /v1/completions` | Plain text completion |
| `GET /v1/models` | The served model |
| `GET /health` | `200` when ready, `503` while draining |
| `GET /metrics` | Prometheus metrics (requests, TTFT, inter-token latency, tokens) |
| `POST /admin/shutdown` | Graceful stop (loopback only; what `dynalm stop` calls) |

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"temperature":0.7}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="any", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

Request fields: `messages` / `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`stop`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `stream`.

---

## Diagnostics

| Command | What it shows |
|---|---|
| `dynalm version` | Version, compiler, built SIMD kernels |
| `dynalm info` | CPU, cores, SIMD features, RAM, and the kernel tier DynaLM will use |
| `dynalm benchmark <model> [options]` | Load test: throughput and P50–P99 latency |

`dynalm benchmark` options:

| Option | Meaning |
|---|---|
| `--concurrency LIST` | e.g. `1,4,16` (default `1,4`) |
| `--prompt LIST` | Prompt lengths in tokens (default `128,512`) |
| `--output LIST` | Output lengths (default `128`) |
| `--requests N` | Requests per point (default 2 × concurrency, at least 4) |
| `--url http://HOST:PORT` | Benchmark a running server (DynaLM, llama.cpp, vLLM, Ollama…) instead of in-process |
| `--model-name NAME` | `model` field sent with `--url` |
| `-t N`, `-c N` | Threads and KV capacity (in-process) |
| `--out FILE` | Append results as JSON lines |

---

## Environment variables

| Variable | Meaning |
|---|---|
| `DYNALM_MODELS_DIR` | Default models folder for `pull`, `list`, `rm` |
| `DYNALM_CONFIG` | Config file for `serve` |
| `DYNALM_<OPTION>` | Any `serve` option, e.g. `DYNALM_PORT`, `DYNALM_HOST`, `DYNALM_THREADS` |
| `HF_TOKEN` | Hugging Face token for gated models (`pull`) |

## Tips

- **Faster answers:**
  - Smaller model or `Q4_K_M` quantization.
  - `/think off` (Qwen3).
  - Keep the model in free RAM: close Docker/WSL and heavy apps.
  - `dynalm chat` or `dynalm serve` keep the model loaded between questions.
- **Hide the `[INFO]` lines:** `dynalm --log-level warn chat ...`.
- **Building and testing DynaLM:** [development.md](development.md).

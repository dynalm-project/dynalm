# Deploying DynaLM: Local Use, Trusted Networks and Gateways

How to run `dynalm serve` safely in each supported mode, what to monitor, and how to stop it.
Security background: [security.md](security.md). Options: [configuration.md](configuration.md),
[cli.md](cli.md). API: [api.md](api.md).

## Choose a mode

| Mode | Use when | Command |
|---|---|---|
| Local | one user on this machine | `dynalm serve qwen2.5:1.5b` (binds `127.0.0.1:8000`) |
| Trusted network | other machines on a private network must reach it | `DYNALM_API_KEY=$(cat key) dynalm serve qwen2.5:1.5b --host 0.0.0.0` |
| Behind a gateway | a reverse proxy terminates TLS and authenticates users | bind the private interface the proxy reaches, with `--allow-unauthenticated` only if the proxy is the sole path to DynaLM |

**Direct public-internet exposure is not supported.** There is no TLS, and no per-client
rate limiting.

### API keys

```sh
umask 077
openssl rand -hex 32 > ~/.dynalm/api-key     # or any secret of >= 16 characters
dynalm serve qwen2.5:1.5b --host 0.0.0.0 --api-key-file ~/.dynalm/api-key
```

- **Where the key can come from:**
  - `--api-key-file FILE` (first line), recommended;
  - `DYNALM_API_KEY` (it is visible to the process environment).
- **The key never goes on the command line,** where other users could see it in `ps`.
- **Clients send** `Authorization: Bearer <key>`. OpenAI SDKs do this when you set their
  `api_key`.
- **What it covers:** every endpoint except `GET /health`.
- **Other tools:** `dynalm stop` and `dynalm benchmark --url` read `DYNALM_API_KEY`.
- **Key length:** shorter than 16 characters is refused.

### Reverse proxy (TLS)

Terminate TLS in the proxy and forward to DynaLM on a private address. Requirements:

- forward `Content-Type: application/json` unchanged (other types get 415);
- disable response buffering for `text/event-stream` (streaming);
- set the proxy's read timeout above DynaLM's `--request-timeout` (600 s by default);
- if DynaLM is bound to loopback behind a local proxy, the proxy must send a loopback `Host`
  header (for example nginx `proxy_set_header Host localhost;`). Requests with other Host
  names are refused (DNS-rebinding protection).

## Resource limits

| Setting | Default | Effect |
|---|---|---|
| `--max-active` | 64 | concurrent completions; beyond it 503 + `Retry-After` |
| `--request-timeout` | 600 s | per request, queue + generation (504) |
| `--max-tokens` | 1024 | default `max_tokens` when a request omits it |
| `--ctx` | auto (from free RAM) | KV capacity; requests whose prompt + `max_tokens` exceed it get 400 |
| body size | 8 MiB | 413 beyond it |
| `--kv` | f16 | `q8_0` halves KV memory |

## Health, metrics and logs

- **`GET /health`:** 200 `{"status":"ok"}`, or 503 `{"status":"draining"}` while shutting down.
  Use it for liveness and readiness; it needs no key.
- **`GET /metrics`:** Prometheus text — request counts, TTFT, ITL, TPOT and E2E histograms,
  scheduler and matmul kernel counters. It needs the key when one is set. It never contains
  prompts or keys.
- **Logs** go to stderr. Prompts and generated text are not logged. Control characters in
  logged strings are escaped. `-q` keeps errors only.

## Stopping

- **Clean stop:** `SIGTERM`, `Ctrl-C` or `dynalm stop` (`POST /admin/shutdown`, loopback only).
- **What happens:**
  1. New requests get 503, and `/health` reports `draining`.
  2. In-flight requests get `--shutdown-timeout` (30 s by default) to finish.
  3. The process exits with code 0.
- **Turning off remote stopping:** `--disable-admin` turns off the HTTP stop.

## GPU (Metal, Vulkan, CUDA; preview)

- **Requirements:** a build with `-DENABLE_METAL=ON` (Apple Silicon), `-DENABLE_VULKAN=ON`
  (AMD, Intel, NVIDIA, Qualcomm) or `-DENABLE_CUDA=ON` (NVIDIA). Setup: [gpu.md](gpu.md).
- **Choosing the device:** `--backend auto` (the default) uses the GPU when the model runs on
  it, and the CPU otherwise. It logs `Device: auto selects the CPU: <reason>` when it falls
  back. `--backend cpu`, `metal`, `vulkan` or `cuda` forces one device. On machines with
  several GPUs, `DYNACORE_GPU_INDEX=N` picks one.
- **Logs to expect:**
  - startup logs `Device: compiled/<Metal|Vulkan|CUDA>/<GPU>` and `Execution: compiled`;
  - with a fixed GPU backend, an unsupported model is refused at startup with a named error;
    it is not silently run on the CPU.
- **Supported:** dense models, f16 / f32 / q8_0 KV.
- **Scheduling on the GPU:**
  - The default (`balanced`) GPU profile favours throughput and typical latency. Measured at
    4–16 users against the CPU-tuned defaults: +20–45% tok/s and much lower TTFT and ITL p50,
    with a higher ITL p99 (up to about 175 ms). It is still below llama.cpp's server in
    multi-user throughput (0.66–0.94×).
  - Use `--policy latency` when token-to-token tail latency matters more than throughput.
- **GPU limitations:** MoE models, YaRN or Llama 3 RoPE scaling, or Qwen3-style per-head Q/K
  norm currently run on CPU. See [gpu.md](gpu.md).

## Checks before going live

```sh
dynalm doctor                                   # CPU, kernel self-test, memory
curl -s localhost:8000/health
curl -s -H "Authorization: Bearer $(cat ~/.dynalm/api-key)" localhost:8000/v1/models
python3 tools/ci/smoke_api.py "$(command -v dynalm)" <model.gguf>   # full CLI + API smoke test
```

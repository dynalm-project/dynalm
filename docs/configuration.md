# DynaLM Configuration: Settings File, Environment Variables and Defaults

DynaLM runs with no configuration: `dynalm serve qwen3:4b` picks the device, the thread
count, the KV cache size and the batching limits from the machine. Configuration exists to
override those choices.

## Sources and precedence

```
built-in defaults  <  config file  <  DYNALM_* environment variables  <  command-line options
```

- **Config file:** `--config FILE`. Otherwise `$DYNALM_CONFIG`. Otherwise
  `~/.dynalm/config.yaml` (`%USERPROFILE%\.dynalm\config.yaml` on Windows), but only when it
  exists.
- **Environment:** `DYNALM_<OPTION>`, with `-` written as `_`. Example:
  `DYNALM_HTTP_THREADS=32`, `DYNALM_THREADS=8`.
- `dynalm config show` prints every key, its effective value, and where the value came from.
  `dynalm config init` writes a commented starter file.

## YAML file

```yaml
model:
  path: qwen3:4b             # model for `dynalm serve` / `dynalm run` when none is given

runtime:
  device: auto               # auto | cpu
  threads: auto              # auto = physical cores
  context_length: 8192       # KV cache tokens; auto = sized from free RAM
  batch_tokens: 256          # max tokens per forward pass
  int8_decode: 4             # int8 activations for matmuls of <= N rows; 0 = off (DD-053)

scheduler:
  max_concurrent_requests: 32
  policy: balanced           # balanced | latency | throughput (DD-061)

kv_cache:
  dtype: f16                 # f16 | f32 | q8_0 (half the memory, same speed; DD-074)

sampling:
  temperature: 0.7
  max_tokens: 1024

server:
  host: 0.0.0.0
  port: 8000
  http_threads: auto
  request_timeout: 600       # seconds; 0 = none
  shutdown_timeout: 30       # seconds of draining on stop
  admin: true                # POST /admin/shutdown (used by `dynalm stop`)
```

The file is a **strict YAML subset**, parsed by about 150 lines of code with no YAML
library (DD-070):

- Two levels only: `section:` followed by indented `key: value` lines.
- `#` comments are allowed. Values may be wrapped in `"..."` or `'...'`.
- The parser rejects lists, tabs, deeper nesting and multi-line values. Each error names the
  line it is on.
- An unknown key is an error, with its line number. This catches typos such as
  `runtime.thread`.
- A documented key that the running command does not use is skipped. For example,
  `dynalm run` ignores `server.port`, so one file can serve every command.
- A top-level `key: value` line may name a command-line option directly (`threads: 8`).

The older flat format, with one `key = value` per line using option names, still works.

| Key | Option | Default |
|---|---|---|
| model.path | `<model>` / `--model` | — |
| model.id | `--model-id` | file name, or the registry name |
| runtime.device | `--backend` | auto (cpu) |
| runtime.threads | `--threads` | auto |
| runtime.context_length | `--ctx` | auto |
| runtime.batch_tokens | `--batch` | 256 |
| runtime.int8_decode | `--int8-decode` | 4 |
| runtime.execution | `--execution` (reference, compiled) | reference |
| scheduler.max_concurrent_requests | `--max-active` | 64 |
| scheduler.policy | `--policy` | balanced |
| kv_cache.dtype | `--kv` (f16, f32, q8_0) | f16 |
| sampling.temperature | `--temperature` | 1.0 (serve), 0.8 (interactive chat) |
| sampling.max_tokens | `--max-tokens` | 1024 (serve), 2048 (interactive chat) |
| server.host | `--host` | 127.0.0.1 |
| server.port | `--port` | 8000 |
| server.http_threads | `--http-threads` | auto |
| server.request_timeout | `--request-timeout` | 600 |
| server.shutdown_timeout | `--shutdown-timeout` | 30 |
| server.admin | `--disable-admin` when false | true |

## Logging

`dynalm -q ...` prints errors only. `dynalm -v ...` is verbose. `--log-level` accepts
`quiet`, `normal`, `verbose`, `debug` and `trace`, as do the older `error`, `warn`, `info` and
`off`. `DYNALM_LOG_LEVEL` sets the default.

## Model store

- Downloads go to `$DYNALM_MODELS_DIR`. Otherwise they go to `~/.dynalm/models`.
- `./models` is searched too.

See [cli.md](cli.md) for model names.

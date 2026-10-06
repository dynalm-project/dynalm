# HTTP API

`dynalm serve <model>` exposes an OpenAI-compatible API. Clients built for OpenAI work by
pointing them at `http://HOST:PORT/v1`.

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat completion with the model's chat template; `"stream": true` for server-sent events |
| `POST /v1/completions` | Plain text completion |
| `GET /v1/models` | The served model. The id is the registry name (`qwen3:4b`), `--model-id`, or the file name. |
| `GET /health` | `200` when ready, `503` with `draining` during shutdown |
| `GET /metrics` | Prometheus text format |
| `POST /admin/shutdown` | Graceful stop. Loopback only; this is what `dynalm stop` calls. `--disable-admin` turns it off. |

## Requests

Fields: `messages` (chat) or `prompt` (completions), `max_tokens`, `temperature`, `top_p`,
`top_k`, `min_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty`, `stream`, `stream_options.include_usage`.

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true,"temperature":0.7}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
r = client.chat.completions.create(model="qwen3:4b", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

**Streaming** sends `data: {chat.completion.chunk}` events with `delta.content`. The last
chunk carries `finish_reason` (`stop`, `length`), then `data: [DONE]`. If the client
disconnects mid-stream, the request is cancelled and its KV blocks are freed.

## Limits and errors

| Condition | Status |
|---|---|
| Body larger than the limit | 413 |
| Malformed JSON or invalid parameters | 400 (`invalid_request_error`) |
| More concurrent requests than `--max-active` | 503 (`overloaded_error`) |
| Server draining | 503 |
| Request exceeded `--request-timeout` | 504 (`timeout_error`) |

Error bodies follow OpenAI's `{"error": {"message", "type", "code"}}` shape.

## Metrics

| Group | Series |
|---|---|
| Requests | `dynalm_requests_total`, `_failed_total`, `_cancelled_total`, `_rejected_total`, `_timed_out_total`; gauge `dynalm_requests_active` |
| Tokens | `dynalm_prompt_tokens_total`, `dynalm_generation_tokens_total`, `dynalm_prefill_tokens_total`, `dynalm_tokens_processed_total`; gauges for tokens per second |
| Latency | TTFT and inter-token latency histograms (p50/p95/p99 derivable) |
| KV cache | blocks used/total, prefix-cache hits |
| Process | RSS |
| Compiled execution (DD-072) | `dynalm_execution_compiled`; and with it on: `dynalm_compiled_segments_total`, `_plan_cache_hits_total`, `_plan_cache_misses_total`, `_fallbacks_total`, `_recorded_calls_total`, `_device_calls_total` |

Server options and configuration keys: [cli.md](cli.md), [configuration.md](configuration.md).

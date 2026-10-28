# DynaLM OpenAI-Compatible API Reference: Self-Hosted Local LLM Server

<!--
SEO Keywords: OpenAI compatible API server, local LLM server, self-hosted LLM API, /v1/chat/completions, streaming SSE completions, private offline API, LangChain local model, LlamaIndex local server, Python OpenAI SDK, CPU LLM inference server.
AEO Direct Answer: DynaLM serves an OpenAI-compatible HTTP API on your own machine (CPU or GPU, no Python), exposing /v1/chat/completions with Server-Sent Events (SSE) streaming on port 8000.
-->

`dynalm serve <model>` starts a lightweight, OpenAI-compatible HTTP server on your machine. It runs on the CPU or the GPU ([gpu.md](gpu.md)). Any software client, SDK, or framework built for OpenAI (such as LangChain, LlamaIndex, LiteLLM, Open WebUI, Continue.dev, or Cursor) works seamlessly out of the box by setting the base URL to `http://HOST:PORT/v1` (default: `http://127.0.0.1:8000/v1`).

---

## 📡 Supported REST Endpoints

| Method and Path | Endpoint Category | Functionality & Specifications |
| :--- | :---: | :--- |
| `POST /v1/chat/completions` | Inference | OpenAI-compliant chat completions with the model's chat template; `"stream": true` provides Server-Sent Events (SSE) chunk streaming. |
| `POST /v1/completions` | Inference | Plain-text raw prompt completion with sampling controls. |
| `GET /v1/models` | Registry | List active served model metadata (`id` reflects catalog name like `qwen3:4b`, `--model-id`, or filename). |
| `GET /health` | Health Check | Kubernetes / Docker liveness probe (`200 OK` when ready, `503` with `draining` during graceful shutdown). |
| `GET /metrics` | Observability | Prometheus text-format metrics covering TTFT, tokens/sec, KV cache blocks, and process RSS memory. |
| `POST /admin/shutdown` | Management | Graceful server stop (restricted to loopback `127.0.0.1`, `Content-Type: application/json`; used by `dynalm stop`). |

---

## 📝 Request Schema & Sampling Parameters

Supported JSON payload fields:
- `messages` (required for chat): Array of `{"role": "system"|"user"|"assistant", "content": "..."}` objects.
- `prompt` (required for completions): Raw text prompt string.
- `stream` (boolean): `true` for streaming tokens via Server-Sent Events (SSE).
- `temperature` (float): Randomness / creativity control (default: `0.7`).
- `top_p` (float): Nucleus sampling cutoff.
- `top_k` (int): Top-K sampling cutoff.
- `min_p` (float): Minimum probability threshold relative to the most likely token.
- `max_tokens` (int): Maximum output tokens to generate.
- `seed` (int): Deterministic RNG seed for reproducible generation.
- `stop` (string | array): Stop sequences that halt generation.
- `presence_penalty`, `frequency_penalty`, `repetition_penalty`: Penalties against repeated tokens.
- `stream_options.include_usage` (boolean): Emits token usage statistics in the final chunk.
- `response_format` (object): Structured output, see below.

### Structured output (`response_format`)

The answer is forced to be valid JSON, and optionally to match a JSON Schema. Tokens that would
break the JSON are never chosen, and generation ends as soon as the document is complete.

| `response_format` | Output |
| :--- | :--- |
| `{"type": "text"}` (default) | Free text |
| `{"type": "json_object"}` | Any JSON object |
| `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` | JSON matching the schema |
| `{"type": "json_object", "schema": {...}}` | Same as `json_schema` (llama.cpp's form) |

```python
response = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "Capital of France, as JSON with city and country."}],
    response_format={"type": "json_schema", "json_schema": {"name": "place", "schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"], "additionalProperties": False}}},
)
# '{\n  "city": "Paris",\n  "country": "France"\n}'
```

**Supported JSON Schema subset:**
- `type`: one type, or a list of types;
- `properties`, `required`, and `additionalProperties: false`;
- `items`, `enum` and `const`;
- `anyOf` / `oneOf`, when the alternatives start with different characters (for example a string
  or `null`).

Annotations (`title`, `description`, `default`, `examples`, `$schema`) are ignored. Any other
keyword (`pattern`, `minLength`, `minimum`, `$ref`, `format`, ...) is refused with a 400 error
naming it, never silently ignored.

**Behaviour:**
- **Key order:** object keys may come in any order, each at most once.
- **Whitespace:** the document starts at its first character, and at most 32 whitespace
  characters may come in a row.
- **Length limit:** if `max_tokens` runs out first, the answer stops mid-document with
  `finish_reason: "length"`, as with OpenAI. Give it enough tokens.
- **Speculative decoding:** structured requests do not use it.
- **Speed:** on an Apple M4 Pro, Qwen2.5-0.5B, the speed was the same as without structured
  output.
- **Prompting:** say in the prompt that you want JSON. The constraint guarantees valid JSON, but
  the model writes better JSON when it is told to.

---

## 💻 Client Integration Examples

### 1. cURL (Streaming Server-Sent Events)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [
      {"role": "system", "content": "You are a concise, helpful programming assistant."},
      {"role": "user", "content": "How do I optimize CPU cache locality in C++?"}
    ],
    "stream": true,
    "temperature": 0.7
  }'
```

### 2. Python (`openai` Official SDK)

```python
from openai import OpenAI

# Connect to local DynaLM server (no API key required on localhost)
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused"
)

stream = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Explain AVX2 SIMD in two sentences."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 3. LangChain Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused",
    model="qwen3:4b",
    temperature=0.2
)

response = llm.invoke("Summarize the benefits of CPU inference for private AI.")
print(response.content)
```

### 4. LlamaIndex Integration

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    api_base="http://127.0.0.1:8000/v1",
    api_key="unused",
    model="qwen3:4b",
    is_chat_model=True
)

response = llm.complete("Explain paged attention.")
print(response)
```

### 5. Node.js / TypeScript (`openai` npm package)

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://127.0.0.1:8000/v1",
  apiKey: "unused",
});

const response = await openai.chat.completions.create({
  model: "qwen3:4b",
  messages: [{ role: "user", content: "What is continuous batching?" }],
});

console.log(response.choices[0].message.content);
```

---

## ⚡ Streaming Behavior & Cancellation

When `"stream": true` is requested:
1. The server transmits `data: {chat.completion.chunk}` events containing incremental token updates in `choices[0].delta.content`.
2. The terminating chunk provides `finish_reason` (`stop`, `length`), followed by `data: [DONE]\n\n`.
3. **Mid-stream cancellation:** If the client disconnects before generation finishes, DynaLM immediately halts token generation and recycles the assigned paged KV cache blocks back to the memory pool.

---

## 🛑 Limits, Error Handling & HTTP Status Codes

| Condition | Status Code | Error Shape (`error.type`) | Resolution |
| :--- | :---: | :--- | :--- |
| Request body exceeds maximum size | `413 Payload Too Large` | `invalid_request_error` | Reduce prompt size or increase limit |
| Malformed JSON or invalid parameter types | `400 Bad Request` | `invalid_request_error` | Validate JSON structure and field types |
| In-flight requests exceed `--max-active` | `503 Service Unavailable` | `overloaded_error` | Implement client backoff or increase `--max-active` |
| Server is draining during graceful shutdown | `503 Service Unavailable` | `draining` | Re-route traffic to another instance |
| Request duration exceeds `--request-timeout` | `504 Gateway Timeout` | `timeout_error` | Reduce `max_tokens` or increase timeout limit |
| Missing or incorrect Bearer authentication token | `401 Unauthorized` | `authentication_error` | Supply valid `Authorization: Bearer <token>` |

Error responses follow OpenAI's standard structure:
```json
{
  "error": {
    "message": "Model 'invalid-name' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

---

## 📊 Prometheus Metrics Endpoint (`/metrics`)

DynaLM exposes fine-grained inference and systems metrics in Prometheus text format:

| Metric Group | Series Names | Description |
| :--- | :--- | :--- |
| **Request Counts** | `dynalm_requests_total`, `_failed_total`, `_cancelled_total`, `_rejected_total`, `_timed_out_total` | Monotonic counters of request outcomes |
| **Active Requests** | `dynalm_requests_active` | Current in-flight requests running through the scheduler |
| **Token Counters** | `dynalm_prompt_tokens_total`, `dynalm_generation_tokens_total`, `dynalm_prefill_tokens_total` | Volume of tokens processed and decoded |
| **Throughput** | `dynalm_tokens_per_second` | Real-time token generation velocity |
| **Latency Histograms** | `dynalm_ttft_seconds_bucket`, `dynalm_inter_token_latency_seconds_bucket` | Time-to-First-Token and per-token decoding latency (p50/p95/p99) |
| **Paged KV Cache** | `dynalm_kv_blocks_used`, `dynalm_kv_blocks_total`, `dynalm_prefix_cache_hits_total` | Memory block pool utilization and prefix cache efficiency |
| **Process Memory** | `dynalm_process_resident_memory_bytes` | Real Resident Set Size (RSS) memory consumption |
| **Compiled Backend** | `dynalm_execution_compiled`, `dynalm_compiled_segments_total`, `_plan_cache_hits_total` | DynaCore IR compiler execution diagnostics |

---

## 🔒 Security & Access Control

- **Bearer Token Authentication:** Provide `--api-key-file <path>` or set `DYNALM_API_KEY=<token>` to enforce authorization on all endpoints except `/health`.
- **DNS Rebinding Protection:** When binding loopback (`127.0.0.1`), DynaLM strictly verifies the `Host` header against `localhost`, `127.0.0.1`, or `[::1]` (rejecting foreign hosts with `403 Forbidden`).
- **Content-Type Enforcement:** All POST requests require `Content-Type: application/json` (rejecting invalid formats with `415 Unsupported Media Type`).

For container and network deployment best practices, refer to [security.md](security.md) and [deployment.md](deployment.md).

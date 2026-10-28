# DynaLM Security: Threat Model, Controls and Deployment Boundaries

What DynaLM protects, which inputs it treats as untrusted, the controls in the code, and the
deployments those controls support. Report vulnerabilities as described in
[SECURITY.md](../SECURITY.md).

## Supported deployments

| Mode | Bind | Authentication | Status |
|---|---|---|---|
| **Local** (one user, one machine) | `127.0.0.1` (default) | none needed; Host-header and JSON-only checks block browser-driven attacks | supported |
| **Trusted LAN / service** | a non-loopback `--host` | **required**: `--api-key-file` or `DYNALM_API_KEY`; serve refuses to start without one | supported, with the limits below |
| **Behind an authenticating gateway** | non-loopback, `--allow-unauthenticated` | delegated to the gateway (TLS, auth, rate limits) | supported only when the network path to DynaLM is restricted to the gateway |
| **Public internet, directly** | — | — | **not supported**: no TLS, no per-client rate limiting, a single shared key |

Binding, CORS and a reverse proxy do not replace authentication. A non-loopback bind without
a key is refused unless `--allow-unauthenticated` is given explicitly, and serve then logs a
warning.

## Assets

- **The host:** CPU/GPU time and memory, because a request costs compute.
- **The service:** availability; `/admin/shutdown` stops it.
- **Request data:** prompts and generated text, which may be sensitive.
- **Secrets:** the API key, and `HF_TOKEN` (used by `dynalm pull`).
- **The filesystem:** the model store and its parent directories.

## Trust boundaries and untrusted inputs

| Input | Trust | Entry point |
|---|---|---|
| HTTP requests | untrusted (any client that can reach the port; any web page in the user's browser for loopback) | `dynalm/src/server/server.cpp`, `dynalm/src/api/openai.cpp`, `dynalm/src/api/json.cpp` |
| Model files (GGUF, SafeTensors/HF) | untrusted (downloaded from the internet) | `dynalm/src/loader/**` |
| Download links (`dynalm pull`) | untrusted text given on the command line | `dynalm/src/cli/pull.cpp` |
| Config file and environment | trusted (the operator's) | `dynalm/src/config/config.cpp` |
| Chat templates in GGUF metadata | untrusted; **never executed**: DynaLM recognizes template families by pattern and renders them with built-in code (no Jinja engine) | `dynalm/src/chat_template/chat_template.cpp` |

## Controls in the code

**HTTP server** (`dynalm/src/server/server.{h,cpp}`):

| Threat | Control |
|---|---|
| Remote use of the API | Loopback bind by default. A non-loopback bind needs an API key (`load_api_key`, at least 16 characters; several keys with `--api-keys-file`, `load_api_keys`). Every endpoint except `GET /health` then requires `Authorization: Bearer <key>` for one of the keys, compared in constant time against every key. Keys are never logged or echoed. |
| A web page driving the local API (CSRF), including `POST /admin/shutdown` | POST bodies must be `application/json` (415 otherwise). A browser cannot send that cross-origin without a CORS preflight, and the server never answers preflights with CORS headers. |
| DNS rebinding (a remote name resolving to 127.0.0.1) | On a loopback bind, the `Host` header must be `localhost`, `127.x` or `[::1]` (403 otherwise). |
| Shutting the server down | `/admin/shutdown` is loopback-only, needs JSON, needs the key when one is set, and can be turned off (`--disable-admin`). |
| Resource exhaustion | Body limit 8 MiB (413); `max_tokens` validated; at most `--max-active` concurrent completions, then 503 with `Retry-After`; per-request timeout (default 600 s); HTTP read/write timeouts 30 s; JSON nesting depth limit; prompt length checked against KV capacity. |
| Stuck clients | Streaming requests are cancelled, and their KV freed, when the client disconnects. |
| Information disclosure | Errors raised while generating are reported by category ("internal error", "unsupported", ...); details go to the server log. Request-validation errors keep their text. Error bodies never contain model paths. |
| Log injection | Control characters in log messages (which can carry model metadata) are escaped. |

**Model files:**
- **GGUF parser** (`loader/gguf/gguf.cpp`): bounds-checks counts against the remaining bytes, guards overflow in the dims product, and checks extents, alignment, duplicates and nesting depth.
- **Tensor layouts** (`dynacore/src/tensor/tensor.cpp`): computed with overflow checks.
- **SafeTensors** (`loader/safetensors/safetensors.cpp`): checks the header length, caps the header size and guards shape overflow.
- **Arithmetic on weights:** NaN and Inf from a corrupt model's weights cannot reach a float→int conversion (`fast_exp.h`, `quantize_act`, `quantize_q8_0`).
- **Fuzzing:** `dynalm/tests/test_fuzz_parsers` mutation-fuzzes all three parsers (GGUF, SafeTensors, request JSON) and runs a forward step on every mutated model that still loads. It runs in CI under ASan+UBSan.

**Downloads** (`dynalm pull`):
- the saved file name is restricted to `[A-Za-z0-9._-]`, does not start with `.`, and must end in `.gguf`;
- the first 256 KiB are checked before the full download;
- `HF_TOKEN` is sent only to `https://huggingface.co/` URLs, through a mode-0600 header file, never on the command line.

**Prompts and output** are not stored or logged by default. `--request-log` and
`--step-trace` (benchmark only) record token counts and timings, not text.

## Findings from the 0.1.4+GPU review (October 2026)

| ID | Severity | Finding | Status |
|---|---|---|---|
| S1 | High (for any non-loopback deployment) | No authentication, while the example configuration bound `0.0.0.0` | Fixed: API key, and a non-loopback bind without a key is refused |
| S2 | Medium | A web page could POST to the loopback server (simple cross-origin requests), including `/admin/shutdown`, which passed the loopback check | Fixed: JSON-only POST |
| S3 | Medium | DNS rebinding against the loopback server | Fixed: Host allowlist on loopback binds |
| S4 | Low | Internal error details (device messages with memory addresses) returned to clients | Fixed: category messages; details logged |
| S5 | Low | Undefined behaviour (NaN → int) from a corrupt model's weights | Fixed (found by the fuzzer under UBSan) |
| S6 | Low | Signed overflow in tensor stride computation for a GGUF tensor with a zero dimension and huge others | Fixed (found by the fuzzer under UBSan) |
| S7 | Low | Log injection through control characters in model metadata | Fixed |
| S8 | Low | `dynalm pull` derived the file name from the link without rejecting `\\` (a path separator on Windows) | Fixed: strict file-name allowlist |
| S9 | High (network deployments) | The pinned cpp-httplib v0.18.3 is in the affected range of published advisories, among them GHSA-xm2j-vfr9-mg9m (critical, internal header shadowing), GHSA-wpc6-j37r-jcx7 (critical, header CRLF injection) and GHSA-px83-72rx-v57c / GHSA-qjmq-h3cc-qv6w / GHSA-xjhg-gf59-p92h (high, unbounded memory). DynaLM's loopback admin check reads the socket address field, not the shadowable header, but the DoS and injection issues apply to any exposed server. In addition, a header already present in the build tree was never re-verified against its hash | Fixed: upgraded to v0.61.0 (SHA-256 pinned); an existing header is re-hashed and replaced if it does not match |
| S10 | Medium | Memory exhaustion from model metadata: a GGUF declaring `block_count` 67,108,866 with 2 layers of tensors passed validation (`dynalm inspect`: "Valid: yes") and sized per-layer state from it before the tensors were checked (peak footprint 162 GB in the fuzzer) | Fixed: hard limits on every metadata dimension that sizes allocations (`ModelConfig::check_limits`), and `block_count` may not exceed the file's tensor count; found by the 20,000-iteration fuzz run, with a regression test |

Every fix has a regression test in `dynalm/tests/test_server.cpp`, `dynalm/tests/test_fuzz_parsers.cpp`
or the sanitizer CI jobs.

## Known residual risks (accepted for the documented deployments)

- **No TLS.** Use a TLS-terminating reverse proxy for anything beyond a trusted network.
- **No per-client rate limiting.** Several keys (`--api-keys-file`) give each client its own
  key, but `--max-active` bounds concurrency globally. Per-client limits belong in the gateway.
- **Unauthenticated loopback.** `/v1/models`, `/metrics` and the completion endpoints are open
  to local processes when no key is set (`--disable-metrics` turns `/metrics` off). Any local user can use the model. Set a key on
  multi-user machines.
- **Untrusted model files.** The parsers are hardened and fuzzed, but not formally verified.
  The Metal backend reads weights through ggml, whose own parsing of DynaLM-provided tensors is
  limited to the layouts DynaLM validates. Load models from sources you trust.
- **Memory exhaustion by a model file** is bounded by the file size (tensors must lie inside
  the file) and the configured KV capacity.
- **Dependencies:**
  - cpp-httplib (SHA-256) and ggml (commit) are pinned.
  - Advisories were checked through the GitHub security-advisory API (cpp-httplib) and OSV (by
    ggml commit) in October 2026. OSV returned nothing for either, but GitHub listed 23
    cpp-httplib advisories (S9). A scanner's silence is not evidence of safety.
  - Re-check both before each release. ggml has no published advisories at the pinned commit
    as of the check.

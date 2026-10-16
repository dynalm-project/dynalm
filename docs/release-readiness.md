# DynaLM Release Readiness: Gates and Evidence

The release gates for the first DynaLM release that includes the Metal GPU backend, and the
evidence for each, as of 2026-10-16.

- **Branch:** `feat/gpu-backend-design`, from the 0.1.4 release candidate `44a9851`;
  uncommitted.
- **Machine:** an Apple Silicon Mac, on AC power for all benchmarks.
- **Raw evidence:** kept outside the repository, in the session's scratchpad
  (`evidence/`, `bench/`). Each benchmark directory has a manifest with executable, model and
  engine hashes.

Statuses: **PASS** (ran, acceptance met) · **FAIL** (reproducible failure) · **INCONCLUSIVE**
(ran, evidence insufficient or too noisy) · **NOT RUN** (not executed; reason given) ·
**BLOCKED** (unresolved prerequisite).

## Scope of this release candidate

| Component | Status |
|---|---|
| CPU inference (Linux, macOS, Windows; x86-64, arm64) | Supported; default; behaviour unchanged from 0.1.4 |
| Metal GPU inference (macOS Apple Silicon, `-DENABLE_METAL=ON`, `--backend metal`) | **Experimental**: correct and validated, but single-user parity with llama.cpp is not reached and multi-user throughput is below llama.cpp |
| HTTP server, local (loopback) | Supported |
| HTTP server, trusted network with an API key / behind an authenticating gateway | Supported (see `docs/security.md`) |
| HTTP server, direct public internet | **Not supported** (no TLS, no per-client rate limits) |

## Gate A: correctness and stability (PASS)

| ID | Check | Status | Evidence |
|---|---|---|---|
| A1 | Model-free tests, CPU-only build (`tools/ci/run_tests.sh build/macos-release`) | PASS | 17/17 programs (`suite2-macos-release.log`) |
| A2 | Model-free tests, Metal build, on a real Metal device | PASS | 17/17 programs; `test_ggml_device` 283,568 checks on Metal (`suite2-macos-metal.log`) |
| A3 | GPU vs CPU fp32 reference within DD-053 (`bench_gpu_accuracy`, Qwen2.5-1.5B Q4_K_M) | PASS | Decode ppl +0.011%, KL 0.000003; prefill ppl +0.150%, KL 0.000179; 4/4 prompts greedy-identical for 64 tokens; deterministic (`accuracy-final.log`); q8_0 KV ppl −0.45% / −0.02% (`accuracy-slot-q8_0.log`) |
| A4 | In-place paged attention vs reference: partial blocks, lengths, scattered blocks, shared prefixes and copy-on-write, concurrency | PASS | `test_ggml_device` (3 sequences, f16/f32/q8_0, window, softcap, GQA); model-level: 2 sequences in one call vs alone, max Δlogit 0.013; fork with copy-on-write vs scratch 0.013, original unaffected |
| A5 | Concurrency: no cross-request leakage; cancellation; deadlines; errors do not corrupt later requests; solo prefill never stalls a decoding stream | PASS | `test_engine` CPU (exact equality, concurrent == alone, prefix cache == no cache) and Metal (lengths, determinism, trace) |
| A6 | Sanitizers: ASan+UBSan and TSan on the full model-free suite; ASan+UBSan with Metal | PASS | `final-linux-asan-ubsan.log`, `final-linux-tsan.log` (17/17 each); ASan+Metal `test_ggml_device`, `test_engine`, `test_server` pass. Four defects found and fixed on the way (S5, S6, S10, the NaN→int UB) |
| A7 | GPU resource lifetime: stale buffers, buffer reuse after free, mapped weights kept alive, clean exit | PASS | `test_ggml_device` (free-epoch regression, mapped-weights test with the file deleted); found and fixed: a stale-graph use-after-free in stage 1 |
| A8 | CPU path unchanged | PASS | Greedy output byte-identical to the 0.1.4 baseline build; `dynalm doctor` OK; `check_isa.py` OK; core-only build OK |

## Gate B: security (PASS for the documented deployments)

| ID | Check | Status | Evidence |
|---|---|---|---|
| B1 | Threat model and controls documented | PASS | `docs/security.md`, `docs/deployment.md` |
| B2 | Loopback default; non-loopback needs a key unless explicitly overridden | PASS | `serve` refuses `--host 0.0.0.0` without a key; `test_server` |
| B3 | API key authentication (constant time, never logged or echoed) | PASS | `test_server`: 401 without, with a wrong key, with non-bearer auth; `/health` open; admin needs the key |
| B4 | Browser-driven localhost attacks (DNS rebinding, simple cross-origin POST, `/admin/shutdown`) | PASS | `test_server`: Host allowlist (403), JSON-only POST (415) |
| B5 | Request limits (body, max_tokens, context, concurrency, timeouts) | PASS | `test_server` (413, 400, 503 + Retry-After); scheduler rejects prompt + max_tokens beyond context or KV |
| B6 | Malformed and fuzzed model files and request JSON fail safely | PASS | `test_fuzz_parsers`: 20,000 GGUF + 20,000 SafeTensors + 80,000 JSON mutations under ASan+UBSan, clean |
| B7 | Error responses do not leak paths or internal state | PASS | `client_message` categories; `test_server` checks that the model path never appears in error bodies |
| B8 | Secret scan of the tree and the package | PASS | `tools/ci/scan_secrets.py`: 0 findings in the tree (planted secrets detected in its self-check); package: see E2 |
| B9 | Dependencies pinned, advisories checked, licences, SBOM | PASS | cpp-httplib upgraded v0.18.3 → v0.61.0 (S9), SHA-256 pinned and re-verified; ggml commit-pinned; GitHub advisory and OSV checks recorded; `THIRD_PARTY_LICENSES.md` carries the exact upstream texts; CycloneDX SBOM (`tools/ci/sbom.py`) |

Findings S1–S10 and their fixes are listed in `docs/security.md`: two High, three Medium, five
Low; all fixed. Accepted residual risks (no TLS, one shared key, no per-client rate limits,
unauthenticated loopback by default, model files not formally verified) are documented there.

## Gate C: performance (measured; targets partly met; no release claim of parity)

| ID | Check | Status | Evidence |
|---|---|---|---|
| C1 | Reproducible single-user comparison vs llama.cpp Metal (same GGUF, same ggml 0.27.0 kernels, same client, 3 paired rotated rounds, AC power) | INCONCLUSIVE / PARTLY PARITY | Release sweep: at 3,584 tokens PARITY on every metric. Final sweep: 128 tokens TTFT 0.92×, decode ITL 7.1 vs 6.4 ms (about 10% slower; PARITY-OR-SLOWER); 512–3,584 INCONCLUSIVE (both engines varied 10–25% in that session). DynaLM's own variation ≤ 2% in quiet rounds. The ±2% target was not reached for llama.cpp |
| C2 | GPU prefill policy (solo-prefill chunk 256) | PASS | Paired sweep of 6 chunk sizes × 3 rounds + recheck: single-user TTFT 1.56–1.66× faster, ITL unchanged, no 4-user regression; CPU unchanged |
| C3 | Weight memory: duplicate copy removed | PASS | Zero-copy mmap weights: peak footprint 1,354 → 291 MiB (`dynalm run`); server single-user 383 MiB vs llama.cpp 1,113 MiB |
| C4 | In-place paged attention: no full-cache gather; long-context gain | PASS | No gather on the optimized path (`kv_store` via `set_rows`, attention reads the pool view). Against the stage-1 gather build at 3,584 tokens: decode ITL 15.2 → 7.9 ms (1.92×), TTFT 1.75× (release sweep, 3 rounds, ≤ 4% spread) |
| C5 | Concurrency vs llama.cpp server (4/8/16 users, 512-token prompts) | FAIL against the throughput target | With the GPU scheduling profile, at 4 / 8 / 16 users: output tok/s 0.94× / 0.82× / 0.66× llama.cpp; TTFT p50 better at 4 and 8 users (1.9–2.6×), INCONCLUSIVE at 16; TTFT p95 better at 4 users, worse at 8 and 16 (10.9 vs 5.5 s at 16); ITL p50 at parity at 4 and 8 users, slower at 16; ITL p99 128 / 144 ms vs llama.cpp 26 ms (4 / 8 users), 153 vs 1,256 ms at 16; TPOT p50 and E2E p50 slower at every point (E2E 0.73–0.85×). The 2–2.5× throughput aspiration is **not met**. Full table: `docs/gpu-backend.md` section 0, "Release measurements" |
| C6 | Ollama measured separately, backend verified | RAN; NOT COMPARABLE | Ollama 0.40.5 on its own server (127.0.0.1:11435, private model store), the same GGUF, the same client, 3 rounds. Backend verified: `ollama ps` shows `100% GPU` with the `llamacpp` runner; its log shows `library=Metal`. Every request ended early (a mean of 13–74 of the 128 requested tokens; Ollama ignores `ignore_eos`), and the prompt was re-templated despite a raw `TEMPLATE` (512 → 541 tokens). No result is ranked against Ollama. Its single-user ITL (6.4 ms) equals llama.cpp's, which it embeds |

**Bottleneck analysis** (docs/gpu-backend.md, section 0):

- **Single-user decode:** host overhead is under 1% of a step (input upload 1 µs, logits
  download 30 µs, graph build reused). The graph's op mix matches llama.cpp's for Qwen2.
  The ~10% gap is inside GPU execution.
- **Open hypotheses:** per-tensor Metal buffers (339) against llama.cpp's single mapping, and
  kernel-variant selection.
- **Multi-user:** mixed prefill/decode steps dominate (about 0.8 ms per prompt token at
  64-row granularity). The GPU profile recovered +20–47% over the CPU-tuned defaults.
  llama.cpp batches whole prompts (higher throughput, much worse tails at 16 users).

## Gate D: automation and CI

| ID | Check | Status | Evidence |
|---|---|---|---|
| D1 | CI covers all platforms, sanitizers, core-only; new tests included | PASS (configuration) | `.github/workflows/ci.yml`: build matrix (Linux gcc/clang/arm64, macOS arm64/x86_64-Rosetta, Windows MSVC), core-only, ASan+UBSan, TSan, package, docker; the new tests run through `tools/ci/run_tests.sh` |
| D2 | Metal job; GPU tests report whether a real device ran | PASS (configuration) | `macos-metal` job prints "Metal GPU tests RAN" or "did NOT run … NOT RUN" |
| D3 | Security job | PASS (configuration) | Secret scan, SBOM, 20,000-iteration fuzz under ASan+UBSan |
| D4 | Hosted CI run on this branch | NOT RUN | Requires a push, which is not authorized. Every job's commands were run locally on macOS arm64 except the Linux, Windows and Docker jobs |
| D5 | Windows / Linux builds of the new code | NOT RUN | No Windows or Linux machine available here; portability issues found by inspection were fixed (no `getpid`/`unistd.h` in tests; `<iterator>`) |

## Gate E: packaging and operations

| ID | Check | Status | Evidence |
|---|---|---|---|
| E1 | Package builds from the verified source, installs into a fresh prefix, and the installed binary passes the CLI + API smoke test | PASS (CPU and Metal) | `tools/ci/package.sh` with `DYNALM_SMOKE_MODEL` (Qwen2.5-1.5B Q4_K_M). CPU: 18/18 smoke checks against the packaged binary, `doctor` OK, ISA selection OK (`package-cpu.log`). Metal (`-DENABLE_METAL=ON`, `DYNALM_BACKEND=metal`): 18/18 against a fresh extract of the archive; both server logs report `Device: compiled/Metal/<GPU>` (`smoke-pkg-metal.log`) |
| E2 | Package contents, licences, checksums, SBOM, secret scan | PASS (after one fix) | Both archives: the content allowlist check passes (bin/dynalm, bin/dynacorec, share/doc only; licences, NOTICE and THIRD_PARTY_LICENSES present); SHA-256 file verifies; secret scan 0 findings in 24 files; CycloneDX SBOM lists cpp-httplib 0.61.0 (CPU) and cpp-httplib 0.61.0 + ggml 0.27.0 (Metal). **Found and fixed:** the first Metal package also shipped ggml's headers, static libraries and CMake files, because ggml's install rules ran. `cmake/Ggml.cmake` now declares ggml `EXCLUDE_FROM_ALL` (and requires CMake ≥ 3.28 for Metal builds); the rebuilt package passes the allowlist |
| E3 | Deployment, health, monitoring and shutdown documented | PASS | `docs/deployment.md` (modes, API keys, reverse proxy, limits, `/health`, `/metrics`, logs, drain and stop, Metal notes, go-live checks) |
| E4 | Smoke test matches each device's documented defaults | PASS (after one fix) | The Metal package first scored 17/18: the smoke test expected the CPU scheduler budgets, while Metal uses its documented GPU profile (prefill budget 256, max batch 321). The check now expects the exact budgets for the device in use. CPU and Metal both pass 18/18 |

Notes:

- The CPU and Metal archives share the name `dynalm-macos-arm64.tar.gz`. Release packaging
  must give the Metal build a distinct name if both are published.
- The archives were built from the uncommitted tree. Release packages must be rebuilt from
  the tagged commit.

## Gate F: release decision

| ID | Check | Status | Evidence |
|---|---|---|---|
| F1 | Version, CHANGELOG, documentation and package agree | PASS | Version 0.1.4 (CMake, `--version`); the CHANGELOG `[Unreleased]` section lists every change since 0.1.4; docs updated (`gpu-backend.md`, `security.md`, `deployment.md`, `api.md`, `cli.md`, `configuration.md`, `installation.md`, README). The release version number is a release-time decision |
| F2 | Limitations and unsupported configurations listed | PASS | `docs/gpu-backend.md` (section 0, "Still open"), `docs/security.md` (residual risks), `docs/deployment.md` |

### Recommendation

**READY WITH DOCUMENTED LIMITATIONS**, as a limited production beta of the **CPU path** with
the hardened server, with the **Metal backend shipped as experimental**:

- The CPU path is unchanged and validated.
- The server's security defaults and fixes are tested.
- The Metal path is correct (DD-053, sanitizers, concurrency, prefix sharing) and much
  faster than the CPU and than stage 1. It is not at single-user parity with llama.cpp and
  is below llama.cpp in multi-user throughput, so it must not be presented as faster than
  llama.cpp or as production-ready.

Before a **public release**:

1. Push the branch and get the hosted CI green on all platforms (D4/D5), including the
   Windows, Linux and Docker builds of the new code.
2. Decide the release version (0.1.5 or 0.2.0) and date the CHANGELOG section.
3. Re-check dependency advisories (cpp-httplib, ggml) at release time.
4. Decide whether release packages for macOS include Metal (`-DENABLE_METAL=ON`). The Metal
   package passed locally (E1/E2). If it ships, give it a distinct archive name and rebuild it
   from the tagged commit.
5. Keep the public claim to: correct, experimental Metal backend; measured numbers as in this
   report.

# Changelog

## [Unreleased]

CPU-only builds behave as before. GPU backends (Metal, Vulkan, CUDA) are opt-in at build
time. **In a GPU build, `--backend auto` (the default) now selects the GPU** when the model
runs on it. Otherwise it selects the CPU and logs why. `--backend cpu` and
`--backend metal` are always obeyed. **One security default changed:** `serve` refuses a
non-loopback `--host` without an API key (see Security).

### Vulkan and CUDA GPU backends, Windows on ARM (preview)
- **Vulkan** (preview; `-DENABLE_VULKAN=ON`, `--backend vulkan`): AMD, Intel, NVIDIA and
  Qualcomm GPUs through the same ggml device as Metal.
  - Passes the GPU device suite, the engine tests, DD-053 accuracy and the API smoke test
    (validated through MoltenVK).
  - CI runs it on Mesa lavapipe.
- **CUDA** (preview; `-DENABLE_CUDA=ON`, `--backend cuda`): the same device layer. CI builds
  it.
- **Shared GPU changes:**
  - Matmul and attention request fp32 precision on every GPU (DD-053).
  - Buffers get unique handle addresses on providers that do not provide them.
  - Auto KV capacity respects free device memory.
  - `DYNACORE_GPU_INDEX` selects a GPU.
  - `ENABLE_HIP` is refused with a pointer to Vulkan.
  - A GPU that cannot run the core ops (for example a virtualized GPU without the
    required features) is refused at startup. `auto` then uses the CPU instead of
    failing on the first request.
- **Windows on ARM** (preview):
  - presets `msvc-release` (ARM64 prompt) and `windows-arm64` (clang-cl);
  - CI build, test and package jobs on `windows-11-arm`;
  - a `dynalm-windows-arm64.zip` release asset;
  - the installer detects ARM64 from the system registry (an emulated x64 PowerShell reports
    AMD64).
- **Portability fix:** `TensorLayout::contiguous` no longer uses `__builtin_mul_overflow`,
  which MSVC lacks.

### Metal GPU backend (preview; docs/gpu.md, docs/gpu-backend.md)
- **Device choice:** `--backend auto|cpu|metal` on `run`, `serve` and `benchmark`;
  `runtime.device` in the config file; or `DYNALM_BACKEND`.
  - `auto` checks the model at startup (`gpu_model_support`). Unsupported models fall back to
    the CPU: MoE, per-head Q/K norm (Qwen3, Gemma 3), YaRN or Llama 3 RoPE scaling, and KV
    types other than f16, f32 and q8_0.
  - `auto` also falls back to the CPU when the GPU fails to start. It logs
    `Device: auto selects the CPU: <reason>`.
  - `--backend metal` with an unsupported model now fails at startup, not on the first
    request.
- **Execution:** each forward step runs on the GPU as one ggml graph (pinned ggml v0.27.0,
  `a05c992`, MIT; Metal shaders embedded and compiled at run time, so no Xcode is needed).
  - The step is recorded by DynaIR and handed over whole (`SegmentExecutor`, `DeviceCall`).
  - Prepared graphs are cached by structure and padded KV span; only token ids, positions, KV
    indices and masks are uploaded per step.
- **Paged KV read in place:** the GPU KV pool uses a slot-major block layout
  (`KvLayout::kSlotMajor`) and flash attention reads it directly.
  - The block tables, causality and sliding window become a mask; fully masked chunks are
    skipped by the kernel.
  - Nothing is gathered or copied. Prefix sharing and copy-on-write work unchanged.
  - The stage-1 gather path remains for A/B (`DYNALM_KV_LAYOUT=head`).
  - f16, f32 and q8_0 KV are supported.
- **Zero-copy weights:** weights borrowed from the GGUF mmap are wrapped in place as shared
  Metal buffers. Peak memory footprint for Qwen2.5-1.5B Q4_K_M: 1,354 → 291 MiB.
  `DYNACORE_GPU_WEIGHTS=copy` restores copies.
- **GPU prefill default:** steps without decode rows use 256-token prompt chunks
  (`kGpuSoloPrefillChunk`); `--solo-prefill-chunk` overrides it. CPU defaults are unchanged.
  - **Single user:** paired sweep of chunks 0–512, 3 rounds, Apple Silicon Mac on AC. TTFT 1.56× / 1.66× /
    1.64× faster at 128 / 512 / 2,048 prompt tokens, ITL unchanged.
  - **4 users:** no change.
  - **Why 256:** 512 was marginally faster for one user, but noisier, with a small 4-user
    ITL regression.
- **Accuracy:** within DD-053 against the CPU fp32 reference — decode perplexity +0.011%,
  prefill +0.150%, greedy-identical on the reference prompts; deterministic.
- **Not supported on the GPU (named errors, no CPU fallback):** MoE models, YaRN or Llama 3
  RoPE scaling, Qwen3-style per-head Q/K norm.

### Security (docs/security.md)
- **API key authentication:** `--api-key-file FILE` or `DYNALM_API_KEY`, at least 16
  characters. Bearer authentication applies to every endpoint except `GET /health`, compared in
  constant time and never logged. `dynalm stop` and `dynalm benchmark --url` send the key.
- **Non-loopback binds:** `serve` refuses a non-loopback `--host` without a key, unless
  `--allow-unauthenticated` is given (for deployments behind an authenticating gateway).
- **Browser attacks on localhost:**
  - on a loopback bind, requests need a loopback `Host` header (DNS-rebinding protection,
    403);
  - POST bodies must be `application/json` (415), so web pages cannot drive the API or
    `/admin/shutdown` with simple cross-origin requests.
- **Error messages:** errors raised while generating are reported to clients by category;
  details go to the log.
- **Fixed, found by fuzzing under UBSan:**
  - undefined behaviour on NaN weights (`fast_exp`, int8 activation and q8_0 quantization);
  - signed overflow in tensor strides for a GGUF tensor with a zero dimension.
- **Logs:** control characters in log messages are escaped (log injection through model
  metadata).
- **Model metadata limits:** dimensions that size allocations are bounded
  (`ModelConfig::check_limits`; for example at most 1,024 layers), and a GGUF `block_count`
  may not exceed the file's tensor count. A fuzzed file that declared 67 million layers
  previously exhausted memory.
- **cpp-httplib upgraded from v0.18.3 to v0.61.0:** v0.18.3 is in the affected range of
  published advisories, including critical header shadowing and CRLF injection, and high
  unbounded-memory issues. A cached header that does not match the pinned SHA-256 is now
  fetched again instead of being trusted.
- **`dynalm pull`:** the saved file name must be a plain name (`[A-Za-z0-9._-]`), so a link
  cannot write outside the model store on Windows.

### Tests and CI
- **New model-free tests on a synthetic GGUF model** (`dynalm/tests/tiny_gguf.h`), in
  `tools/ci/run_tests.sh`:
  - `test_engine`: determinism, concurrent == alone, prefix cache, cancellation, deadlines,
    the solo-prefill non-starvation trace, bounded memory; CPU and Metal;
  - `test_server`: access control, limits, overload, disconnect, drain;
  - `test_fuzz_parsers`: GGUF, SafeTensors and JSON mutation fuzzing.
- **`test_ggml_device`:** every GPU op against the CPU, paged attention in both layouts,
  failure paths, buffer reuse, zero-copy weights.
- **`bench_gpu_accuracy`:** model-based DD-053, determinism, multi-sequence and
  copy-on-write checks.
- **CI jobs:**
  - `macos-metal`: reports whether a real Metal device ran the GPU tests;
  - `security`: secret scan, CycloneDX SBOM, 20,000-iteration fuzzing under ASan+UBSan.
- **Package checks:** `package.sh` checks the package contents and licences, secret-scans the
  archive and writes an SBOM.

### Removed
- Internal benchmark campaign scripts: `tools/next_sweep*.ps1`, `tools/phase8_sweep.ps1`,
  `tools/phase9_sweep.ps1`, `tools/pk8_validation.ps1`, `tools/pk8_analyze.py`,
  `tools/e2e_engines.ps1` and `tools/e2e_engines_analyze.py`. `docs/benchmarking.md` now
  describes paired runs with `dynalm benchmark` and `tools/bench_analyze.py` directly.
- Unused fixture generators for the test suites removed in 0.1.3 (`tools/make_tiny_*.py`,
  `tools/gen_tokenizer_golden.py`) and the dev-container helper (`tools/linux.sh`,
  `tools/docker/Dockerfile.dev`).

## [0.1.4] - 2026-10-15

Default behaviour is unchanged from 0.1.3: with no new environment variables set, the same
kernels run and greedy output tokens are identical (verified against the 0.1.3-based reference
build).

### CPU matmul kernel registry (DynaCore)
- `CpuDevice::matmul` now asks an explicit registry (`dynacore/kernel/matmul_registry.h`)
  which kernel runs. Selection is deterministic: shape, `KernelPlan` and CPU tier only. The
  priority order is the one of the previous dispatch: `test_matmul_registry` checks 554,190
  selections against a copy of it.
- **Diagnostics:**
  - `DYNACORE_MATMUL_EXPLAIN=1` logs each kernel choice and why higher-priority kernels were
    not eligible;
  - `DYNACORE_MATMUL_TIMING=1` adds time per kernel and shape;
  - `/metrics` adds `dynacore_matmul_calls_total{kernel}` and `dynacore_matmul_rows_total{kernel}`;
  - `dynalm benchmark` JSON adds `diag.matmul_kernels`.
- `DYNACORE_MATMUL_DISABLE=name,...|all-accelerated` removes kernels from selection;
  `all-accelerated` leaves the fp32 reference arithmetic.
- A matmul with invalid shape metadata now stops with an error message instead of reaching a
  kernel (it was a debug-only assert). The DynaIR verifier applies the same shape check.
- Extension point for external kernels (`CpuDevice::register_external_matmul`), used only when
  `KernelPlan::external_matmul` is set; nothing registers one by default.

### Experimental CPU features (opt-in; off by default)
- **Split-byte int16 prefill:** `DYNACORE_INT16_PREFILL_ROWS=64 DYNACORE_INT16_PREFILL_SPLIT=1`
  (AVX-VNNI). It computes the opt-in int16 prefill path's exact integer dot products with
  `vpdpbusd` on split activation bytes.
  - **Accuracy:** bit-identical to that path in tests and in greedy tokens. DD-053 on
    Qwen2.5-1.5B (perplexity +0.003% / +0.005%) and Qwen3-4B (−0.012% / −0.003%), mean
    KL ≤ 0.000001.
  - **Measured** on an x86-64 hybrid laptop CPU, Q4_K_M models, 10 threads, paired runs with
    predeclared gates, each against the default:

  | workload | change | verdict |
  |---|---|---|
  | Qwen3-4B, single 2048-token prompt | TTFT −13.6% | reliable PASS |
  | Qwen2.5-1.5B, single 2048-token prompt | TTFT −11 to −13% | baseline varied 2.3%, inconclusive by rule |
  | Qwen2.5-1.5B open loop (Poisson 0.15 req/s, mixed 128 / 2048-token prompts) | output tok/s +2%, short-request TTFT p95 −21%, E2E p99 −12%, ITL p99 −13% | PASS |
  | 16 concurrent 128-token requests | output tok/s +17%, TPOT −15% | PASS |
  | same, ITL p99 | one round +3.3% | inconclusive |
  | Qwen2.5-1.5B, 4 users, 512-token prompts (release campaign, 3 sessions) | TTFT −14.5% (every pair −13.9 to −15.5%), output tok/s +11.9% | inconclusive by rule (baseline ITL p99 varied 13%) |
  | Qwen2.5-1.5B, 4 users, 128-token prompts | output tok/s +4.3% | PASS (no regression) |
  | 4 users, with `--policy latency` (both models, 6 pairs each) | ITL p99 −48%, TTFT −11 to −15%, output tok/s +5 to +7% | inconclusive by rule (baseline tok/s varied 9–10%) |

  - These figures hold for the measured machine and these workloads only; they are not a general
    guarantee.
  - On E-cores the AVX-VNNI unit runs integer dot products at half the fp32 FMA rate, which
    caps the gain.
- **Wide decode:** `DYNACORE_DECODE_WIDE_ROWS=16`.
  - In decode-only steps of 9–16 rows, the batch planner keeps the int8 / int16 decode kernels
    instead of fp32. Each row gets the arithmetic it gets at ≤ 8 rows today (bit-identical per
    row, tested). The value is clamped to 0..16.
  - Above 8 rows it uses the 8-row int8 tiles, so each packed weight group is read once per
    8 rows instead of once per 4. `DYNACORE_WIDE_ROWS8=0` keeps the 4-row tiles.
  - Release campaign (3 sessions, gates declared in advance), Qwen2.5-1.5B:
    - output tok/s +11.0% at 12 users and +11.9% at 16 users (PASS);
    - together with split-byte prefill: +19.7% and +22.5% (PASS);
    - but at 12 users TTFT p50 rose 45% (FAIL; mean TTFT about +7% in a traced run). This is
      why wide decode stays opt-in.
  - Qwen3-4B moved the same way (+4 to +5% tok/s alone, +14 to +19% with split), but those
    results are inconclusive: the baseline varied too much on the measured machine.
  - The 8-row tiles make decode steps 2–4% faster than the 4-row tiles.
  - It has no effect at 8 or fewer concurrent requests.
- **Tail latency at a few users:**
  - Split-byte prefill together with `--policy latency` cut ITL p99 by about 48% at 4 users,
    without the TTFT and throughput cost that `--policy latency` has on its own (+13 to +16%
    TTFT, −6 to −10% tok/s).
  - The verdict is inconclusive by the campaign's rule, so this remains an experimental
    combination, not a default.

### Also new since 0.1.3 (experimental, opt-in)
- **CPU kernels:**
  - int16 prefill tiles for 9–N-row matmuls on large Q4_K weights: `DYNACORE_INT16_PREFILL_ROWS`,
    using `vpdpwssd` on AVX-VNNI CPUs (`DYNACORE_INT16_PREFILL_VNNI=0` for the AVX2 tiles);
  - interleaved Q6_K for int16 activations: `DYNACORE_Q6_PACKED=1`;
  - 8-row int8 tiles for 5–8 decode rows: `DYNACORE_PACKED_8ROWS=1`.
  - Results are identical to the paths they replace, or within the DD-053 contract. All are off
    by default.
- **Scheduler:**
  - `--solo-prefill-chunk N` / `--solo-prefill-split`: a larger prompt chunk in steps without
    decode rows;
  - `--prefill-order fifo|srpt` with `--prefill-aging-ms`: shortest-remaining-prompt first.
    Known defect: after aging, a long prompt takes the whole mixed-step budget, so keep FIFO
    (the default) for latency-sensitive use;
  - `--step-trace FILE` (benchmark): per-step and per-request scheduler trace (queue, prefill
    and decode times, rows per step).
- **Load generator** (`dynalm benchmark`):
  - open-loop arrivals: `--arrival-rate`, `--seed`, `--arrival-trace`, `--prompt-mix`;
  - per-request logs: `--request-log`;
  - uncounted full-size warm-up: `--warmup-requests`;
  - per-prompt-length latency percentiles.
- **Thread pool:** a parallel region requested while another is in flight (reentrant or from a
  second thread) is detected, run serially on the caller and counted (`misuse_count()`),
  instead of racing on the running region's state.
- **DynaKernel prototype** (`dynacore/dynakernel/`, `dkgen`; LLVM backend loaded at run time
  through `DYNACORE_LLVM_C`): kernel IR, lowering, generated C++ and LLVM backends for
  experiments. It is not linked into `dynalm` and is not a production acceleration.

### Tools
- `tools/compare_engines.{ps1,py}`:
  - `-DynalmExe` / `--dynalm` selects the exact executable under test. Its path, version and
    SHA-256 are recorded in `manifest.json`, `report.md` and every result row. It is never
    replaced by a local build.
  - The tool no longer stops processes it did not start. Busy ports fail that engine's
    preflight, and Ollama runs as the script's own server on port 11435.
  - Ollama gets the same per-user context.
  - The report shows the prompt length each engine reported. Ollama 0.40.1 wraps
    `/v1/completions` prompts in its chat template (128 → 157 tokens on Qwen2.5) and ignored
    the raw `TEMPLATE` requested by the script.
  - Rows below 95% of the requested output tokens are marked not comparable and are not ranked.
    Ollama ignores `ignore_eos`, and Qwen2.5 stops early there: mean 13–74 tokens for a
    128-token target.
- New:
  - `tools/next_sweep.ps1`: paired benchmark runner. It records a quiet check, git state and
    per-run hashes.
  - `tools/bench_analyze.py`: paired analysis with predeclared gates. It counts runs, points,
    pairs and gate evaluations separately, and reports session-level stability and bootstrap
    intervals.
  - `docs/benchmarking.md`: the supported ways to benchmark.
  - Tool tests: `tools/test_compare_engines.py`, `tools/test_bench_analyze.py`.
  - `tools/ci/smoke_api.py`: CLI and HTTP API smoke test against a model.
  - `dynacore/benchmarks/bench_ffn_kernels.cpp`: per-core kernel rates.

### CI
- The model-free test programs and the tool tests now run in CI (`tools/ci/run_tests.sh`).
  Covered jobs: Windows (MSVC), Linux (GCC, Clang, ARM64), macOS (ARM64, x86-64 under
  Rosetta), the core-only boundary build, and the ASan+UBSan and TSan builds.
- Pushes to `release/**` branches run CI.
- New model-free test `test_batch_planner` (wide-decode planning).

### Fixed
- An empty tuning environment variable (for example `DYNALM_PROTECT_MIN=`) was read as 0.
  For `DYNALM_PROTECT_MIN` and `DYNALM_DECODE_TARGET` that cut the prompt rows of a mixed step
  to one: at 4 concurrent users TTFT rose from about 2 s to about 20 s. Empty values are now
  ignored for all `DYNACORE_*` planner variables. `DYNALM_DECODE_PROTECT`, `DYNALM_DECODE_TARGET`
  and `DYNALM_PROTECT_MIN` are validated: invalid values are ignored, with a warning in the log.
  The benchmark runner (`tools/next_sweep.ps1`) also restored unset variables as empty strings;
  it now removes them and checks the environment after each run.
- `test_dynakernel_quant` could fail intermittently. After releasing a packed weight, it packed
  new weights on the same backend, which by design refuses an address it released earlier. When
  the heap reused the freed buffer, the test failed. The accuracy section now uses its own
  backend.
- `dynalm serve` logged the requested scheduler budgets, not the ones in effect, with
  `--policy latency|throughput`.
- Packages now include `THIRD_PARTY_LICENSES.md` with the cpp-httplib MIT license text, which
  is compiled into `dynalm`.
- `NOTICE` no longer describes GoogleTest as a current build dependency.

## [0.1.3] - 2026-10-09

### Performance: repacked multi-row decode (DD-081..082)
- Repacked Q4_K layout with int16 activations (`dot_q16_x8_q4_K`, DD-081) for
  multi-row decode on FFN down projections.
- Repacked Q6_K x8 layout (`BlockQ6_Kx8`, `repack_q6_K_x8`, `dot_q16_x8_q6_K`,
  DD-082) with int16 activations, completing packed multi-row decode for
  mixed Q4_K/Q6_K architectures (such as Q4_K_M).
- Automatic prepacking of down projections with int16 activations in
  `Transformer::create`.
- `bench_decode_matmul` supports `BENCH_INT16` benchmarking across original and
  repacked K-quant layers.

### Removed
- Removed deprecated `examples/dynacore/decoder_layer.dyna`.

### Removed
- The unit, smoke and boundary test suites (`tests/`, `dynacore/tests`,
  `dynalm/tests`) and the googletest dependency. CI now builds every platform
  and checks the binaries with `dynalm --version`, `dynalm doctor` (DynaCore
  kernel self-test) and `tools/ci/check_isa.py`; the `core-only` build still
  guards the DynaCore/DynaLM boundary.
- Internal design and benchmark documents and the benchmark result files;
  `tools/compare_engines.{ps1,py}` remains for comparisons with llama.cpp and
  Ollama.

### Fixed
- `/show` in chat reports `thinking n/a` for models without a thinking mode.

### Performance: multi-row kernels and latency (DD-078..080)
- Interleaved Q4_K copy for 2-8-row decode (DD-078): +7.6% at 4 concurrent
  requests and +13.2% at 8 on Qwen2.5-1.5B; single-request decode unchanged.
  Memory-guarded at load (`DYNACORE_Q4_REPACK=off|auto|always`, default auto:
  skip tensors when less than 3 GiB would stay free). Integer prefill through
  the same kernel is opt-in (`DYNACORE_Q4_REPACK_MAX_ROWS=128`): faster, but
  it fails the perplexity limit on Qwen3-4B.
- Decode-protected adaptive prefill (DD-079): ITL p99 -43% at 4 users and -38%
  at 8 on Qwen2.5-1.5B for <= 1.5% throughput (`--policy throughput` keeps the
  old behaviour; `DYNALM_DECODE_PROTECT=0` turns it off).
- AVX-VNNI kernel (DD-080): bit-identical, not faster on Alder Lake; opt-in with
  `DYNACORE_VNNI=1`.
- `bench_int8_accuracy` scores prefill-shaped chunks; `bench_decode_matmul`
  compares original and packed layouts (`BENCH_CHECK`, `BENCH_HEAVY`).

### Performance: concurrent decode (DD-075..077)
- Sub-scaled int8 activations for 2-8-row K-quant matmuls (DD-077): +16% at 4
  concurrent requests on Qwen2.5-1.5B and +12.5% on Qwen3-4B (in-process, 3
  interleaved repeats), within the accuracy contract; single-request decode
  unchanged. `DYNACORE_INT8_SUBSCALE=0` restores the previous path.
- int16 activations for the FFN down projection (DD-076), replacing fp32 at
  1-4 rows with equal accuracy. `DYNACORE_INT16_FFN_DOWN=0` turns it off.
- Super-block int8 activations (DD-075) are available as an experiment only
  (`DYNACORE_INT8_SUPERBLOCK`); they fail the accuracy contract.
- Tools: `tools/compare_engines.{ps1,py}` (DynaLM vs llama.cpp vs Ollama).
- Benchmarks: per-thread utilization in `dynalm benchmark` diagnostics,
  `bench_alloc` (allocations per decode step), `BENCH_HOT=1` and an M list
  for `bench_decode_matmul`, int8/int16 kernel table in `bench_kernels`.

## [0.1.1] - 2026-10-07

### Fixed
- `dynalm run`, `serve` and `inspect` now accept a bare model file name that includes its
  extension, such as `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`, exactly as `dynalm models` prints
  it. The file is looked up in `~/.dynalm/models` and `./models`. Before, such a name failed
  with "model file not found".

## [0.1.0] - 2026-10-04

### Cross-platform CPU release
- Release archives `dynalm-{linux-x86_64,linux-arm64,macos-arm64,macos-x86_64}.tar.gz` and
  `dynalm-windows-x86_64.zip`, plus `SHA256SUMS`. Every archive is built, tested, installed
  through the installer and smoke-tested before it is published (`tools/ci/package.sh`).
- `scripts/install.sh` and `scripts/install.ps1` download the release for the detected
  platform, verify its SHA-256 and install per user. The old build path is now
  `--from-source` / `-FromSource`.
- `dynalm doctor` has a new layout. It adds the install path, `dynacorec` presence and a
  DynaCore kernel self-test (selected ISA vs generic).
- `DYNACORE_ISA=generic|avx2|neon|...` forces a kernel tier. An unavailable tier falls back
  with a note.
- Smoke tests (ctest label `smoke`): version, doctor, inference with the best tier and with
  generic, compiled execution, dynacorec, and `serve` with the OpenAI endpoints. All run on a
  committed tiny model, offline. ctest labels: `dynacore`, `dynalm`, `boundary`, `smoke`.
- Fixed: `--execution compiled` crashed when an engine was destroyed before its first forward
  pass, for example after `dynalm run` failed early.

### DynaCore compiler and compiled execution (DD-071..073)
- DynaCore IR (inference-typed SSA: quantized weights, paged KV, gqa), verifier, text form,
  recording device, passes (kernel selection, Q/K/V grouping, gated-MLP fusion), plan cache.
- `--execution compiled` / `runtime.execution: compiled`. On Qwen2.5-1.5B Q4_K_M at c=1:
  +7.9% output tok/s and -10.6% ITL p50. Neutral at c>=4 and on Qwen3-4B (about 1-2%).
  Bit-identical outputs. Opt-in.
- DynaCore language (`.dyna`) and `dynacorec`: dump IR, the optimized IR and kernels, run
  memory analysis and benchmarks.
- New Device op `matmul_gated`. `MatmulJob` takes an optional bias. `act_mul` uses the whole
  pool for single rows. `/metrics` reports compiled-execution counters.

### q8_0 KV cache (DD-074)
- `--kv q8_0` halves KV memory, within the accuracy contract. Decode speed is unchanged, so
  f16 stays the default.

### DD-067 validated
- 32 matmul chunks per thread (was 8): neutral to slightly positive on decode and prefill;
  kept.

### Product surface: names, doctor, models, config (R1, DD-070)
- Model names: `dynalm run qwen3:4b`, `dynalm serve llama:3b`, `dynalm pull gemma:270m`. A
  built-in registry maps 12 names to tested GGUF downloads. `run`, `serve` and `benchmark`
  download a named model on first use. `/v1/models` reports the name.
- Model store: `~/.dynalm/models`, or `$DYNALM_MODELS_DIR` when set (previously `./models`,
  which is still searched).
- `dynalm doctor [--json]` replaces `info` (`info` remains an alias). It reports OS, CPU, ISA,
  RAM, GPU (probed through the NVIDIA driver, no SDK needed), device, threads, store and
  config, with actionable warnings.
- `dynalm models` replaces `list`, and `list` remains an alias. It shows NAME, FORMAT, SIZE,
  LOCATION, QUANTIZATION and STATUS. `--available` lists the registry names.
- Config files may be a two-level YAML subset (`runtime.threads`, `server.port`, ...).
  `~/.dynalm/config.yaml` applies automatically when present. New commands:
  `dynalm config show|path|init`. `run` now reads the config and `DYNALM_*` too.
- `dynalm version` / `--version` print the DynaLM and DynaCore versions. Logging takes
  `-q` / `-v`, the level names `quiet|normal|verbose|debug|trace`, and `DYNALM_LOG_LEVEL`.
- `dynalm inspect` prints the parameter count.

### DynaCore / DynaLM split (R0, DD-068, DD-069)
- The engine is now two libraries:
  - `dynacore`: tensors, memory, devices, kernels, hardware. It knows no LLM concepts.
  - `dynalm_runtime`: models, scheduler, KV policy, API, CLI.
- A ctest and a CI job enforce the boundary. A `core-only` preset builds DynaCore alone.
- The binary is built at `build/<preset>/bin/dynalm`.
- The experiment knobs are renamed `DYNALM_*` → `DYNACORE_*`: `GEMM_KC`,
  `MATMUL_EXPAND_MIN`, `INT8_DECODE_ROWS`, `MATMUL_CHUNKS`, `ATTN_GROUPED` and `PIN_THREADS`.
- Throughput is unchanged within noise (`results/r0-split-ab.csv`).

### Grouped GQA attention (DD-066)
- Query heads sharing a KV head are scored and accumulated in one pass. Attention is 4–22% faster
  at most measured points; end-to-end aggregate tok/s is unchanged within noise (attention is
  5–8% of a decode step). `DYNALM_ATTN_GROUPED=0` selects the per-head path.
- Benchmark diagnostics on Windows report the effective CPU clock (PDH `% Processor
  Performance`), not the constant base clock (the CPU's rated base clock).
- `bench_batch_decode` runs in-process old/new A/B with alternating order and a free-memory
  floor; new `bench_attn_accuracy` and `tools/ab_summary.py`.

### Faster first token (DD-064)
- Weight pages are mapped in parallel at load instead of faulting in during the first request:
  Qwen3-4B first-request TTFT 874–891 → 477–508 ms, Gemma-3-270M 127–133 → 36–41 ms
  (`DYNALM_PREFAULT=0` disables it).
- `dynalm run` prints prefill forward time next to the decode step time.

### Performance program P14 (DD-063)
- **`-DDYNALM_LTO=ON`** build option for link-time optimization (off by default: no measured gain on
  this CPU). A sweep of the kernel-plan settings found the defaults best; no tuning cache added.

### Performance program P13 (DD-062)
- **Adaptive speculative decoding** (default for `dynalm run --spec`): k is chosen per round from
  {0, 3, K} by measured output tok/s, so speculation switches off when it is slower and back on
  when drafts start paying. Qwen2.5-1.5B: ngram drafting 11.5–15.2 → 15.3–17.1 tok/s (plain
  15.7–17.5); a 0.5B draft keeps 19–20 tok/s on repetitive text. `--spec-fixed` keeps the old
  behaviour. The stats line reports the chosen k and the number of passes without drafts.

### Performance program P12 (DD-061)
- **`--policy balanced|latency|throughput`** (serve, benchmark). At c=64: throughput policy TTFT p50
  21 s → 3.8 s at the same tok/s, with ITL 340 → 616 ms; latency policy ITL 186 ms, TTFT 35 s.
- **`benchmark --prefill-budget/--decode-budget/--chunk/--batch`** for scheduler experiments.

### Performance program P11 (DD-060)
- **Grouped MoE execution:** `matmul_many` runs every expert in one parallel region, each with its
  own path (int8 rows, fused, or GEMM panel), including strided rows. Granite-MoE: 1,857 → 313
  regions per step; +7% tok/s at c=4, +5% at c=16. Accuracy within the contract (perplexity +0.27%).
- `matmul` takes the int8 path for strided rows too (same numerics alone or grouped).

### Performance program P10 (DD-059)
- **`CpuInfo::core_first_cpu`:** the first logical CPU of each physical core, P-cores first
  (Windows, Linux).
- **Opt-in pinning:** `DYNALM_PIN_THREADS=1` binds compute threads one per core. Off by default:
  equal or slower on Windows in A/B runs.
- **Thread-count sweep:** one thread per physical core is best (10 threads: 30.2 / 63.8 tok/s at
  c=1 / c=8; adding SMT siblings is slower).

### Performance program P8 (DD-057)
- **Prefill GEMM measured at 300–360 GFLOP/s** (~70–80% of this laptop's practical fp32 peak). The
  M-blocking hypothesis was refuted; no change. `bench_decode_matmul` now also covers M = 64–256.

### Performance program P7 (DD-056)
- **Block-run attention kernels** (`attn_scores_*`, `attn_accum_*` for generic, AVX2 and NEON). Each
  fp16 run is converted once per query group, and softmax uses the vectorized `exp_nonpos` (now
  `common/fast_exp.h`). Qwen2.5-0.5B attention at 4K context: 9.7 → 6.6 ms per decode step.
- **The planner sizes split-K chunks** to fill whole thread waves: 1K-context attention −15%.
- `bench_decode_context` prints per-op time per decode step.
- Block size kept at 16; Q8 KV deferred (attention moves ~7.6 GB/s, so it is not bandwidth-bound).

### Performance program P6 (DD-055)
- **`bench_thread_pool`** measures fork/join: 2.5 µs p50 at 10 threads, ~2% of a 0.5B decode step.
- **The thread pool locks and notifies only when a worker sleeps:** a seq_cst handshake; region
  cost −16%, TSAN-clean.

### Performance program P4–P5 (DD-054)
- **GQA-grouped attention:** each (row, KV head) task serves its whole query-head group, so K/V
  is fetched once per group; heads split into sub-groups when tasks are fewer than threads.
  Results are identical. Qwen2.5-0.5B 4K-context decode step 40.2 → 31.7 ms (−21%), TTFT −13%.
- **P4 weight packing evaluated and deferred:** with the throttling fix and int8 decode, Q4_K_M is
  the fastest format at c=8/16, so K-quant unpacking is no longer a bottleneck.

### Performance program P3: multi-row decode (DD-053)
- **int8-activation decode kernels** (AVX2; scalar references) for Q8_0, Q4_0, Q5_0, Q4_K and Q6_K:
  integer dot products against packed weights, each block unpacked once for up to 4 rows.
  - Default for matmuls of ≤ 4 rows, except the FFN down projection.
  - Speed: 17–19 GB/s at M=1–2. Qwen2.5-1.5B Q4_K_M serving +14% (c=1), +24% (c=2), +14% (c=4).
  - Accuracy contract: perplexity within +1% and mean KL ≤ 0.0025 on 4 models.
  - `--int8-decode N` / `EngineOptions::int8_decode_rows` / `DYNALM_INT8_DECODE_ROWS`; 0 restores
    bit-exact batch invariance.
- **New tools:** `bench_int8_accuracy` (perplexity, KL, top-1 and greedy agreement against fp32,
  with a summation-order noise floor) and an `int8` column in `bench_decode_matmul`.
- `Transformer::set_kernel_base`, and a per-site int8 switch for the down projection.

### Performance program P2 + throttling fix
- **Execution planner (DD-051):** `src/execution/BatchPlanner` decides per step the phase, shape and
  `KernelPlan`. The `KernelPlan` holds the matmul expand threshold, GEMM K-block, and the attention
  strategy for full and sliding-window layers. Results are bit-identical, at ≤ 0.09 ms/step of
  planning.
- **Compute threads opt out of OS power throttling (DD-052):** Windows EcoQoS, and the macOS QoS
  class. A quiet process such as `dynalm serve` or `dynalm benchmark` had run its forward pass
  1.6× slower. Qwen2.5-0.5B Q4_K_M serving throughput: 18.6→31.5 tok/s at c=1, 40.1→71.1 at c=8,
  54.7→88.5 at c=64.
- **AVX2 GEMM micro-kernel:** a 4×3 tile (12 accumulators) for batched decode, 3–27% faster on
  real tensors and bit-identical. `bench_decode_matmul` measures decode matmuls on a model's real,
  DRAM-resident weights.
- `dynalm run` prints the engine's mean decode-step forward time.
- The benchmark clock sampler and profiling run only with diagnostics.

### Performance program P1: measurement (DD-050)
- **Scheduler step accounting** (always on): plan, prefix lookup, KV reservation, forward split by
  decode-only / prefill-only / mixed steps, sampling, emit, prefix insert, and mean decode rows per step.
  Prompt tokenization is timed in the engine.
- **`Engine::set_profiling`:** per-op forward time plus thread-pool region count, region time, the
  caller's tail wait and worker sleeps, all published in `EngineStats`.
- **`platform/perf_counters`:**
  - per-thread Linux hardware counters (cycles, instructions, LLC, L1D, branch misses, migrations);
  - OS context switches and page faults;
  - the CPU clock.
  Unavailable counters read `null`, with the reason (Windows, macOS, VMs without a PMU).
- **`bench/analysis`:** a measured DRAM read ceiling, a modelled decode-traffic estimate, and a
  rule-based bottleneck classifier (COMPUTE / MEMORY / CACHE / SYNCHRONIZATION / DISPATCH /
  LOAD_IMBALANCED / IO / MIXED).
- **`dynalm benchmark`:**
  - `--prompt-mix` for mixed prompt lengths;
  - `--no-diag`;
  - a per-point diagnostics line, and a `diag` object in the JSON lines.
- **`tools/bench_report.py --svg DIR`:** the eight standard graphs (standard library only), plus a
  diagnostics table.
- **`tools/perf_sweep.sh quick|full`:** the measurement matrix (concurrency 1–64, context up to 4K,
  mixed prompts, F16/Q8_0/Q4_K_M, 0.5B / 1.5B / 4B dense, MoE).
- `ROADMAP.md` lists the performance program P1–P16.

### Interactive chat
- `dynalm chat <model>` (and `dynalm run <model>` without `-p`): an Ollama-style multi-turn chat.
  - The model is loaded once. History is re-sent each turn and the prefix cache reuses it, so a follow-up
    starts in about 0.25 s (Qwen2.5-1.5B).
  - The oldest exchanges are dropped when the context is full.
  - Commands: `/help`, `/bye`, `/clear`, `/system`, `/think on|off` (Qwen3), `/set temp|top_p|top_k|min_p|
    repeat_penalty|max_tokens|seed`, `/show`, `/stats`, `"""` for multi-line input; Ctrl+C stops a reply.
  - Conversational defaults (temp 0.8, top-k 40, top-p 0.9, repeat-penalty 1.1, 2048 tokens per reply).
  - UTF-8 console input on Windows.
  - `ThinkFilter` hides the empty `<think></think>` block Qwen3 emits with reasoning off and streams real
    thinking live (unit-tested across chunk splits).
- `docs/cmd.md`: every command, option, chat command, HTTP endpoint and environment variable, plus a table of
  Ollama equivalents.

### Fixes
- **Windows: emoji and non-English output.** Model output (UTF-8) showed as garbage in consoles using a legacy
  code page; an emoji printed as "ƒÿè". `dynalm` now switches the console to UTF-8 while it runs and restores
  the previous code page on exit.
- **Windows: UTF-8 everywhere else.** An embedded manifest makes UTF-8 the process code page (Windows 10
  1903+), so non-English prompts in `-p` and non-ASCII file paths arrive intact.
- **Docker image.** The build copies `LICENSE` and `NOTICE`, which the install step now packages.
- **Docs.** A README FAQ on CPU speed (memory bandwidth, RAM pressure, `serve`, Qwen3 `/no_think`).

### Open source
- Licensed under Apache-2.0 (`LICENSE`, `NOTICE` with third-party components).
- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue templates (bug, model request), a
  pull request template, `CITATION.cff` and `llms.txt`.
- README rewritten for new users: FAQ, comparison with llama.cpp / Ollama / vLLM, badges.
- `TODO.md` became `ROADMAP.md`. Removed references to the internal build specification.
- CMake homepage URL fixed. Packages and the Docker image carry the license and OCI labels.

### Model downloads
- `dynalm pull <link>` downloads a GGUF into `./models` (or `-o DIR`, `$DYNALM_MODELS_DIR`) from a Hugging Face
  file link (page or download form), the short form `<owner>/<repo>/<file>.gguf`, or any http(s) URL.
  - Reads the first 256 KiB first and refuses unsupported architectures before the full download (`--force` overrides).
  - Resumable (`<file>.part`), retried, gated models via `HF_TOKEN`; verifies every tensor type afterwards.
  - Uses the system `curl` (DD-047).
  - `--check` only checks support (reads 256 KiB, saves nothing).
- `dynalm rm <model>... [-y]` (alias `delete`) deletes downloaded models by the name `dynalm list` shows (the
  `.gguf` extension is optional) or by path.
  - Asks before deleting unless `-y` is given.
  - Deletes only GGUF files, partial downloads and Hugging Face model directories, never other files or folders.
- `dynalm list` names the blocker (`unsupported architecture 'qwen35'` or `unsupported tensor types: iq3_s, ...`)
  and shows partial downloads.
- CI: all 11 jobs green (Linux x64 gcc/clang/ASAN/TSAN, Linux ARM64, macOS, Windows, three installers, Docker).

### DynaLM: name, installs, CI, NEON
- Renamed to **DynaLM**:
  - the command `engine` → `dynalm`;
  - environment variables `ENGINE_*` → `DYNALM_*` (`DYNALM_CONFIG`, `DYNALM_MODELS_DIR`,
    `DYNALM_<OPTION>`, `DYNALM_GEMM_KC`, `DYNALM_MATMUL_EXPAND_MIN`);
  - Prometheus metrics `engine_*` → `dynalm_*`;
  - API `owned_by: "dynalm"`.
- Installers: `scripts/install.sh` (Linux, macOS) and `scripts/install.ps1` (Windows). CMake
  install rules, CPack archives, `DYNALM_STATIC_RUNTIME`, a two-stage `Dockerfile`
  (amd64/arm64).
- CI: `.github/workflows/ci.yml`:
  - Linux x86-64 gcc/clang/ASAN/TSAN;
  - Linux ARM64;
  - Windows MSVC;
  - macOS Apple Silicon;
  - install scripts and Docker.

  `release.yml` builds packages on tags.
- ARM64 NEON kernel tier (Apple Silicon, ARM64 Linux, Windows on ARM). SIMD options are now
  architecture-aware.
- macOS platform support: CPU topology (P/E cores), memory, process stats.
- Presets `linux-clang-release` and `macos-release`. The dev image includes clang and builds
  for arm64.
- The cpp-httplib download retries, and `DYNALM_HTTPLIB_HEADER` supports offline builds.
- README rewritten (DD-046).

### Phase 28 — GPU backend architecture (no GPU code)
- Backend memory contract (DD-045): `upload`, `download`, `host_accessible`, all scratch and
  KV through `allocate`, KV copy-on-write through `copy`. New data-movement ops `fill`,
  `gather_rows` and `scatter_add_rows` replace the Transformer's direct memory access (logits
  gather, MoE permutation and combine); the CPU path stays zero-copy.
- `DeviceType`: CUDA/HIP/Metal/Vulkan/Simulated. `backends/backend_registry`: `BackendKind`,
  `create_backend` (CPU built; GPU kinds give a clear "not built"), `EngineOptions::backend`,
  `--backend` on `run`/`serve`.
- `docs/gpu-backend.md`: the interface contract and a CUDA implementation sketch, with spec
  §47 extension points mapped to code.
- `test_device_backend`: a memory-guarded (`mprotect`) simulated device backend runs every
  architecture, the scheduler with prefix-cache COW, and speculative decoding with results
  bit-identical to CPU; a death test shows any host access faults.

### Phase 27 — Speculative decoding
- `SeqBatch::logits_last`: one forward pass returns logits for the last N tokens of a sequence.
- `Sampler::sample_speculative`: verification with the target's own sampling pipeline (greedy:
  argmax match; sampling: accept with p(d), else resample without d), so outputs follow the
  target distribution exactly (DD-044).
- `runtime/speculative`: `Drafter` interface, `NgramDrafter` (prompt lookup), `ModelDrafter`
  (same-vocabulary small model with KV rollback), and `SpeculativeGenerator`
  (propose → one-pass verify → accept → `KvBlockTable::truncate` rollback).
- `engine run --spec ngram|DRAFT.gguf --spec-k K`; `bench_speculative`.
- Tests: `test_speculative` (greedy exactness with three drafters, distribution preservation,
  KV rollback, vocabulary check).

### Phase 26 — Advanced sampling
- `sampling/sampler`: `SamplingParams` (temperature, top_k, top_p, min_p,
  repetition/frequency/presence penalties, penalty window, seed) and a per-sequence `Sampler`.
  Histogram-pruned exact top-k/top-p, a vectorized argmax, and a portable xoshiro256** RNG and
  exp, so seeds reproduce across platforms; no per-token allocations (DD-043).
- Scheduler entries own their sampler; invalid parameters fail the request cleanly.
- API: real sampling for OpenAI fields plus `top_k`, `min_p`, `repetition_penalty`,
  `repeat_last_n`; default temperature 1.0 (`serve --temperature`). The "greedy only" caveat
  is gone.
- `engine run`: `--temp --top-k --top-p --min-p --repeat-penalty --presence-penalty
  --frequency-penalty --repeat-last-n --seed` (default greedy).
- `bench_sampling`; tests `test_sampling` plus API/HTTP cases.

### Phase 25 — MoE support
- Generic Transformer MoE (DD-042):
  - host-side routing (softmax, top-k, optional renormalization);
  - each active expert runs once per batch on its gathered rows, with a weighted scatter-add;
  - optional shared expert with a sigmoid gate.
- Families:
  - Mixtral (GGUF `llama` + experts, HF `mixtral`);
  - Qwen2-MoE / Qwen1.5-MoE and Qwen3-MoE (GGUF `qwen2moe`/`qwen3moe`, HF `qwen2_moe`/`qwen3_moe`);
  - IBM Granite and Granite-MoE (`granite`, `granitemoe`), whose embedding, attention,
    residual and logit multipliers are now ModelConfig scalars.
- New tensor roles for shared experts.
- GGUF: reads `expert_shared_feed_forward_length` and `expert_weights_norm`.
- HF: per-expert tensors are stacked into 3-D at load, and Granite's fused `input_linear` is
  split without a copy. Models mixing dense and MoE layers, and GPTQ/AWQ MoE checkpoints, are
  rejected clearly.
- `tools/ref_model.py`: MoE and Granite multipliers, written from the HF modeling code.
- `tools/make_tiny_models.py` / `make_tiny_hf.py`: four tiny MoE fixtures, as GGUF and HF.
- `Backend::matmul_many` (CPU: one parallel region for many small matmuls): Granite-MoE decode
  59.7 → 29.1 ms/token.
- Granite chat template (`<|start_of_role|>`), including its default system prompt and turn
  separator, read from the template text.
- Profiler ops `moe_route`, `moe_experts`, `moe_gather_scatter`.
- Tests: MoE goldens, HF↔GGUF equivalence and the compat suite for all four; a Granite-MoE
  real-model golden; Granite template vs jinja2 rendering; `matmul_many` vs `matmul`.

### Phase 24 — GPTQ/AWQ
- `quant/gptq_awq`: `PackedScheme`, exact GPTQ (v1/v2, 4/8-bit, act-order) and AWQ (GEMM,
  4-bit) unpacking and reference dequantization; load-time repack to Q4_0 / Q8_0 (bit-exact for
  symmetric weights), Q4_1 (asymmetric) or F16 (act-order) (DD-041).
- `hf::read_quantization` (clear errors for bitsandbytes, fp8, other packings);
  `hf::split_packed_name`; the HF loader gathers packed components across shards, validates
  dtypes and shapes, and repacks them.
- `engine inspect` reports the packed scheme; the load summary reports the repack targets.
- AVX2 Q4_1 dot product and dequantization (5.2× / 3.2×): AWQ decode 77.6 → 33.4 ms/token.
- Fix: `~Engine` dereferenced a null control block when `Engine::create` failed part-way
  (found by UBSAN on a truncated download); regression test `Hardening.FailedCreateCleansUp`.
- Verified on Qwen's official Qwen2.5-0.5B-Instruct GPTQ-Int4 (→ Q4_0) and AWQ (→ Q4_1).
- `tools/make_tiny_quant_hf.py` + five packed tiny fixtures with NumPy references.
- Tests: `test_gptq_awq` (config parsing, hand-packed layouts, corrupt `g_idx`, five variants
  vs reference weights and logits).

### Phase 23 — SafeTensors
- `loader/safetensors`: memory-mapped, zero-copy SafeTensors reader with a strict header validator.
- `loader/hf`: Hugging Face directory loading (single or sharded); `config.json` → ModelConfig
  (llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3_text, phi3); HF tensor names → roles;
  Gemma (1 + w) norm folding and Llama 3 RoPE factors at load; `tokenizer.json` → TokenizerData
  (byte-level BPE and SentencePiece-style BPE); chat template from `tokenizer_config.json`.
- `ModelConfig::qk_rows_interleaved`: the Llama adapter picks the RoPE style from the weight layout
  (GGUF permuted vs HF original), with no weight copies.
- `engine inspect` and `engine list` understand HF directories; `inspect` reports real support status.
- AVX2 BF16 row dequantization (1.9×); BF16 checkpoints decode as fast as F16.
- `tools/make_tiny_hf.py`: HF exports of the tiny fixtures (committed in `tests/data/hf_tiny_*`).
- Tests: `test_safetensors` (parser hardening, config/names/RoPE factors, bit-exact GGUF↔HF
  equivalence for 7 architectures, sharding, tokenizer.json vs GGUF ids, real SmolLM2/Qwen2.5).

### Phase 22 — Production hardening
- Configuration (DD-038): `config/` merges a config file (`--config`/`ENGINE_CONFIG`),
  `ENGINE_<OPTION>` env vars and CLI flags; `auto` values; KV capacity sized from free RAM by
  default (`auto_kv_tokens`).
- Server (DD-039): admission limit `--max-active` (503 + Retry-After), HTTP workers default to
  max-active + 8, default request timeout `--request-timeout` (504), engine-error → HTTP status
  mapping, graceful drain on SIGINT/SIGTERM with `--shutdown-timeout`, `/health` reports
  `draining`, loopback-only `POST /admin/shutdown` (`--disable-admin`). Requests are owned by
  an RAII guard, which fixes a leaked active-request gauge (and an uncancelled request) when
  a client vanished before streaming began.
- Scheduler: requests larger than the KV pool are rejected at submission.
- Metrics: `engine_scheduler_step_ms`, `engine_tpot_ms`, `engine_generation_tokens_per_second`,
  `engine_prefill_tokens_per_second`, `engine_prefill_tokens_total`,
  `engine_kv_cache_capacity_tokens`, `engine_requests_rejected_total`,
  `engine_requests_timed_out_total`, `engine_queue_latency_ms_max`.
- CLI: `engine list [dir]`, `engine stop|unload [--host --port]`. The structured startup
  summary now includes RAM required and the KV size.
- Tests: `test_config`, `test_compat` (compatibility suite for every tiny architecture plus real
  models when present), `test_hardening` (abuse/leak/shutdown), and 7 new `test_server` cases.

### Phase 21 — Benchmark framework
- `bench/loadgen`: closed-loop load generator. Exact-length prompts, unique per request and
  per point, fixed output lengths, a warm-up request. Reports TTFT/ITL/TPOT/E2E at
  P50/P90/P95/P99, tokens/s, errors, and peak RSS / CPU use for in-process runs.
- Targets: in-process engine, or any OpenAI-compatible server over HTTP (`bench/http_target`,
  streaming SSE with usage).
- `engine benchmark <model> [--concurrency --prompt --output --requests --url --model-name
  -t -c --out]`; JSON-lines output.
- `platform/process_stats`: RSS, peak RSS and CPU seconds (Windows, Linux).
- API: `ignore_eos` request extension (needed for fixed-length benchmarking).
- `tools/bench_report.py` (Markdown tables) and `tools/compare_baselines.sh` (engine vs
  llama.cpp `llama-server`, same container environment).
- Fix found by the baselines (DD-036): AVX2 row dequantization for Q8_0/Q4_0/Q5_0/Q4_K/Q6_K,
  and batched matmuls expand weights from 2 rows (was 4). Q4_K_M decode ITL at 4
  concurrent requests drops from ~89 ms to ~45 ms.
- `bench_kernels` dequant section; `bench_matmul` m sweep.
- Tests: percentiles, prompt lengths/uniqueness, run_point accounting, no cross-point prefix reuse.

### Phase 20 — OpenAI-compatible API
- `api/json`: strict JSON parser (depth/size limits, surrogate pairs) and serializer.
- `api/openai`: chat/completions request parsing and validation, response/chunk/usage/error
  builders, `/v1/models`.
- `metrics/`: lock-free counters, gauges, histograms with Prometheus text output.
- `server/`: cpp-httplib server (pinned, SHA256-verified). Endpoints `/health`, `/v1/models`,
  `/metrics`, `/v1/chat/completions`, `/v1/completions` with SSE streaming
  (`stream_options.include_usage`); client disconnect cancels the request; TTFT/ITL/E2E
  histograms; KV, prefix-cache and scheduler gauges.
- `engine serve <model> [--host --port --threads --ctx --batch --kv --http-threads --max-tokens]`.
- Engine exposes an observability snapshot (`EngineStats`).
- Tests: JSON (incl. 20k-case fuzz), API parsing/format, HTTP end-to-end.

### Phase 19 — Streaming
- `runtime/text_stream`: `TextStreamer`, UTF-8-safe deltas with stop strings that are
  never emitted (minimal holdback).
- `runtime/engine`: `Engine` facade (model, pool, backend, KV, scheduler, scheduler thread),
  `generate_text` / `generate_chat` → `RequestStream` (thread-safe event queue: deltas, final
  event with finish reason, error, token usage), consumer cancellation, safe shutdown.
- `engine run` streams through the Engine (`--stop` added); reports TTFT and inter-delta latency.
- Tests: streamer behaviour; engine incremental streaming, stop strings, concurrent clients,
  cancellation, errors, shutdown.

### Phase 18 — Kernel optimization
- Forward-pass profiler (`Transformer::set_profiling`, `ForwardProfile`, `bench_profile`).
- Prefill GEMM: AVX2 4×2 register-blocked `gemm_panel` (+ generic) over expanded weight
  panels; K-blocking with 1024-wide slices (`ENGINE_GEMM_KC` override).
- Split-K decode attention with log-sum-exp merging (`attend_range` partials).
- Benchmarks: `bench_matmul` (threads × shapes), `bench_kernels` gemm section, `bench_profile`.
- Tests: gemm_panel in every tier incl. accumulate; split-K vs naive (scrambled blocks, mixed
  lengths, sliding window).
- Prefill 2× faster (Qwen2.5-0.5B 277 tok/s, SmolLM2 912 tok/s warm). Scheduler at 16 requests:
  4× throughput, 4.7× lower TTFT, 4.9× lower ITL p99.

### Phase 17 — CPU SIMD optimization
- `backends/cpu/cpu_kernels.h`: `CpuKernels` table (dot/axpy f32 and f16, per-dtype fused
  `vec_dot` and `dequant`), generic tier (fused scalar kernels + chunked fallback for every
  type) and AVX2/FMA/F16C tier (f32, f16, bf16, Q8_0, Q4_0, Q5_0, Q4_K, Q6_K), with runtime
  selection. Per-file ISA flags only.
- CpuBackend matmul: fused path for decode, expand-once path for prefill. Attention reads f16
  KV through F16C dot/axpy.
- `test_kernels`: every tier × 13 weight types vs the dequantize reference.
- Batched-vs-sequential test now uses a float-rounding tolerance (DD-031).
- Decode 1.4–3.4× faster; 4K-context decode 3.4×; batched decode 355 tok/s aggregate.

### Phase 16 — Radix prefix cache
- `PrefixCache` is now an interface: `make_hash_prefix_cache` (Phase 15) and
  `make_radix_prefix_cache` (block-edged radix tree with token-granular partial-block reuse
  via copy, LRU leaf eviction). `SchedulerConfig::prefix_cache_kind` (radix by default).
- Fixed admission bookkeeping so a partial (private) block is inserted once it fills.
- Tests run against both implementations; a radix-specific partial-copy test.
- `bench_prefix`: radix 11% fewer rows, 2× lower TTFT p50, 2.5× faster lookups than hash.

### Phase 15 — Prefix hash cache
- `prefix_cache/PrefixCache`: block-granular, chained verified hashes, LRU leaf-first eviction,
  optional capacity, stats (lookups, hit/eligible tokens, hit rate, inserted/evicted/cached blocks).
- Scheduler: lookup on admission (`SequenceState::adopt_prefix`), insert completed blocks
  after each step, evict cache before preemption; `enable_prefix_cache`,
  `prefix_cache_max_blocks`.
- Tests: longest-prefix match, collision-proof verification, never freed while referenced,
  LRU leaf-first eviction, capacity; scheduler reuse is exact with exact row accounting;
  cache yields under KV pressure.
- `bench_prefix`: 512-token shared system prompt → 11× wall, 23× TTFT p50, 94% hit rate.

### Phase 14 — Chunked prefill
- `SchedulerConfig::max_prefill_chunk` (default 32): per-sequence cap on prefill rows per
  step, so prompts prefill side by side (no head-of-line blocking).
- Test: a short prompt behind a long one gets its first token in step 1 (vs 4 unchunked);
  outputs identical either way.
- `bench_long_prompt`: decoder ITL p99 during a 2048-token prefill 36.7 s → 1.09 s.

### Phase 13 — Prefill/decode scheduler
- `SchedulerConfig`: `decode_token_budget`, `prefill_token_budget`, `max_running`.
- Decode rows scheduled first with least-recently-served rotation; prefill chunks by
  priority, then admission order.
- `Request::priority` (higher first, FCFS within a level) and `Request::timeout_ms`
  (kDeadlineExceeded for queued or running requests; KV released).
- Stats: prefill/decode rows per step, queue latency (total/max), timeouts.
- Tests: budget bounds, decode continuity during large prefill, fairness, priority order,
  deadlines, queue latency; outputs identical to isolated runs.
- Budget sweep: ITL p99 up to 2.5× lower (DD-027).

### Phase 12 — Continuous batching
- `scheduler/Scheduler`: iteration-level continuous batching. Thread-safe submit/cancel
  via a per-step handoff; admission control on KV; decode-first batch building within a
  token budget; recompute preemption under KV pressure; per-request callbacks
  (token events + a final event with status, reason and error).
- `SequenceState::reset_for_recompute` (preemption).
- Errors isolated per request (validation up front; model errors fail only that batch).
- Tests: outputs identical to isolated runs under staggered arrivals, token-budget
  chunking, cancellation, preemption and concurrent submitters; no KV leaks.
- `bench_scheduler`: 3.2× aggregate throughput at 8 requests; prefill/decode
  interference quantified (ITL p99 > 1 s at 16+ requests) for Phase 13/14.

### Phase 11 — Concurrent sequences
- `Transformer::forward_batch(SeqBatch...)`: rows from many sequences in one pass; per-row
  positions and sequence index; logits only for requested rows.
- Backend `attention`/`kv_store` address KV per row's sequence (`row_seq` + per-sequence views).
- Tests: mixed decode/prefill batches bit-identical to sequential runs (Llama, Gemma-3, Phi-3);
  batch validation.
- `bench_batch_decode`: aggregate decode throughput vs concurrency (up to 2.4×).

### Phase 10 — Paged KV
- `KvBlockPool` (renamed from KvCache): atomic per-block refcounts, allocate/retain/release,
  copy_block; thread-safe.
- `KvBlockTable` (renamed from KvSequence): reserve, truncate (rollback), clone (fork),
  append_shared (prefix cache), make_writable (copy-on-write); movable.
- `SequenceState::reserve_kv` makes the target range writable (copy-on-write aware).
- Tests: refcount lifecycle, clone sharing, copy-on-write isolation on real K/V contents,
  exhaustion/recovery, a 4-thread stress test (TSAN).
- `bench_kv`; block-size sweep → 16-token blocks (DD-024).

### Phase 9 — KV cache v1
- `runtime/sequence`: `SequenceState` (id, tokens, computed position, block table, stop
  params, status machine prefill/decode/finished/cancelled/error, finish reason, KV bytes).
  KV is released on every terminal transition.
- Generator rewritten around `SequenceState` (compute pending → sample), reporting the finish reason.
- KV sizing helpers: `kv_geometry_for(config, dtype, block, tokens)`, `kv_tokens_for_budget`.
- `bench_decode_context`: decode latency vs context 64–4096 (attention-bound; see benchmarks.md).

### Phase 8 — Quantized GGUF execution
- `quant/quant_formats`: GGML block layouts + reference dequantization for Q4_0, Q4_1,
  Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. Wired into
  `dequantize_row`, so matmul and embedding run every format.
- Fixtures from gguf-py (`tools/make_quant_fixtures.py`): all 10 dequantizers match.
- `tools/ref_model.py` accepts quantized weights (gguf-py dequantization).
- Goldens: SmolLM2 Q8_0 (1e-5 vs the Q8_0 reference) and Qwen2.5-0.5B Q4_K_M (mixed
  Q4_K/Q5_0/Q6_K/Q8_0) match their references. Gemma-3-270M f16 and Qwen2.5-0.5B f16
  real goldens pass. Gemma-3 SPM tokenizer golden passes (262K vocab).
- `bench_quant`; per-format end-to-end numbers in docs/benchmarks.md.

### Phase 7 — Qwen / Mistral / Gemma / Phi / DeepSeek adapters
- Adapters: Qwen (`qwen2`, `qwen3`), Gemma (`gemma`, `gemma2`, `gemma3`), Phi (`phi3`).
  Mistral and dense DeepSeek run via the Llama/Qwen adapters (DD-021).
- `ModelArchitecture::prepare_weights` hook + `TensorRegistry::take` (Phi-3 fused gate|up).
- Per-layer local RoPE (`ModelConfig::rope_local`) for Gemma-3 sliding layers; Gemma-2/3
  sliding-window patterns; 27B query scaling.
- `tools/ref_model.py`: generalized NumPy reference (all families, lazy fp32 upcast).
- `tools/make_tiny_models.py`: committed tiny GGUFs + fixtures for all 7 architectures.
- Real-model goldens: Qwen2.5-0.5B f16 and Gemma-3-270M f16 (optional); Gemma-3 SPM
  tokenizer golden.

### Phases 5 + 6 — Llama model and basic CPU execution
(One commit: the Llama adapter can only be verified by executing it. The golden test covers both.)
- `model/architecture`: `ModelArchitecture` adapter interface, registry, and shared
  standard-decoder shape validation. `LlamaArchitecture` covers Llama/Mistral/DeepSeek-LLM/SmolLM GGUFs.
- `loader/model_loader`: file → validated `LoadedModel` (config, weights, adapter, tokenizer, chat template).
- `backends/backend.h`: backend-independent op interface. `CpuBackend` with generic kernels
  (matmul f32/f16/bf16, RMSNorm/LayerNorm, RoPE interleaved/half-split with linear scaling and
  freq factors, paged kv_store, GQA attention with soft-cap and sliding window, SiLU/GELU).
- `model/transformer`: generic decoder forward. Weights resolved once, scratch preallocated,
  logits only for the last token.
- `kv_cache/`: block-based KV storage (f32/f16) + per-sequence block tables.
- `runtime/thread_pool`: persistent spin-then-sleep workers, dynamic chunking.
- `runtime/generator`: chunked prefill + greedy decode with TTFT/ITL stats. `sampling/`: greedy.
- `engine run <model> -p ...`: streamed chat/raw generation with stats.
- Golden tests vs an independent NumPy reference (`tools/ref_llama.py`).

### Phase 4 — Tokenizer / chat template
- `tokenizer/`: `Tokenizer` with byte-level BPE (rank merges, O(n log n)) and
  SentencePiece BPE (score merges, byte fallback); special-token splitting (control
  tokens only with `parse_special`); EOG set from metadata plus known terminators;
  `Utf8Buffer` for streaming-safe decoding.
- Hand-written pre-tokenizers (GPT-2, Llama-3, Qwen2, StarCoder/SmolLM) over generated
  Unicode tables (`tools/gen_unicode_tables.py`).
- `chat_template/`: family detection from Jinja source; ChatML (with default system
  extraction), Llama-3, Llama-2, Mistral, Gemma, Phi-3, DeepSeek-V2/3.
- `loader/gguf/gguf_tokenizer`: GGUF → format-neutral `TokenizerData`.
- Golden tests: exact match with HF `tokenizers` for SmolLM2 and Qwen2.5 (38 cases each).
- Linux container gate (`tools/linux.sh`): gcc 13 with zero warnings; ASAN+UBSAN (fatal); TSAN.
- `bench_tokenizer`: 15 MB/s encode, 15.5 ns/token decode.

### Phase 3 — Model IR
- `model_ir/ModelConfig`: family-neutral hyperparameters (GQA, head dims, RoPE incl.
  scaling, norms, activation, biases, QK-norm, sandwich norms, soft-capping, sliding
  window, MoE) with `validate()`, KV bytes/token and a memory estimate.
- `model_ir/TensorRegistry`: weights by `(TensorRole, layer)`; duplicate and range checks.
- `loader/gguf/gguf_model`: `<arch>.*` keys → ModelConfig; `blk.N.*` names → roles;
  unmapped tensors reported; `general.file_type` labels.
- `engine inspect` now shows the model config, quantization, estimated RAM and backend.
- 12 new tests (config validation, registry, name parsing, synthetic and real models).

### Phase 2 — GGUF loader
- `loader/mapped_file`: read-only mmap (Win32 / POSIX) with prefetch hint.
- `loader/gguf`: GGUF v2/v3 parser; zero-copy metadata, arrays and tensors; hardened
  against malformed input; maps GGML type IDs to engine DTypes (IQ*/TQ*/MXFP4 are
  recognized but unsupported).
- `engine inspect <model> [--metadata] [--tensors]`.
- 12 new tests (synthetic GGUF builder, truncation at every byte, random corruption,
  optional real-model test via `ENGINE_TEST_MODEL`). `bench_loader`.
- `tools/fetch_models.sh`: resumable download of the dev test models.

### Phase 1 — Tensor + dtype system
- `dtype/`: engine-owned `DType` with GGML-compatible block geometry; exact fp16/bf16 conversion.
- `memory/`: aligned host allocation with global stats and an OOM test hook; owned or borrowed `Storage`.
- `tensor/`: `TensorShape`, `TensorLayout` (byte strides), non-owning `TensorView`
  (reshape/slice/select/transpose without copies), owning `Tensor`.
- 18 new tests (exhaustive fp16/bf16 round trips, slow-reference fp16 rounding, view semantics,
  quantized restrictions, OOM propagation). `bench_tensor`.

### Phase 0 — Project foundation
- CMake build with presets (MSVC release/debug/ASAN; Linux release/ASAN+UBSAN/TSAN).
- Feature flags: `ENABLE_{CUDA,HIP,METAL,VULKAN}` (future phases; ON is an error),
  `ENABLE_{AVX2,AVX512,AMX}`, `ENABLE_{SERVER,TESTS,BENCHMARKS}`, sanitizers.
- Per-file ISA compile flags. No global arch flags.
- `common/`: `Status`, `Result<T>`, propagation macros, platform macros, monotonic timer.
- `logging/`: leveled, line-atomic logging. Disabled levels skip argument evaluation.
- `platform/`: CPU feature detection (cpuid + XCR0), hybrid P/E-core topology,
  cache sizes, RAM; ISA tier selection (compiled ∩ supported).
- CLI: `engine version`, `engine info`. Later-phase commands report their phase.
- GoogleTest suite and a percentile benchmark harness (`bench_foundation`).
- Docs: architecture, design decisions, development, performance, model support, quantization.

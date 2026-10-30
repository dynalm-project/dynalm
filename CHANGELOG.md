# Changelog

## [Unreleased]

### Added
- `dynalm cpu-report`: measures DynaLM's CPU paths (default, the other kernel family, each
  DynaCore tier the CPU has) against a pinned llama.cpp release on Qwen2.5-1.5B, Qwen3-4B and
  Qwen3-8B, checks each path's agreement with llama.cpp on wikitext-2, and writes a JSON and
  Markdown report for issue #31. `--quick` runs the 1.5B model for one round.
- `dynalm cpu-report` prints its progress in seven numbered, timestamped steps: what it detected
  about the CPU (family, cores, features, default kernels, threads), what it plans to run, each
  download, each accuracy score and each speed measurement, with the log file for every server.

## [0.4.0] - 2026-10-28

Structured JSON output, several API keys, and checksum-verified model downloads. The engine's
speed and defaults are unchanged from 0.3.2.

### Added
- Structured output: `response_format` `json_object` and `json_schema` (OpenAI), and
  llama.cpp's `json_object` + `schema`. Only tokens that keep the answer valid JSON, matching the
  schema, are chosen, and generation ends with the document. JSON Schema subset: types,
  properties, required, `additionalProperties: false`, items, enum, const, and anyOf/oneOf with
  alternatives that start differently. Other keywords are refused with an error, never ignored.
  On an Apple M4 Pro (Qwen2.5-0.5B, CPU and Metal), throughput is unchanged.

### Security
- `serve --api-keys-file FILE`: several API keys, one per line (`#` comments allowed), so each
  user or app can have its own key and one can be revoked without changing the others.
  `--api-key-file` still reads the first line only.
- `serve --disable-metrics` (`server.metrics: false`): turns off `GET /metrics`.
- `dynalm pull` checks a file from Hugging Face against the SHA-256 Hugging Face publishes for it
  (`X-Linked-ETag`) before moving it into the model store. A corrupt or altered download is
  deleted and the pull fails. Files from other servers are not checked, and pull says so.

### Documentation
- `--max-active` above 64: the scheduler generates for at most 64 requests at once; the others
  wait in its queue.

## [0.3.2] - 2026-10-28

A performance patch for ARM CPUs: faster Q5_0 and Q5_1 weights. Results are bit-identical to 0.3.1.

### Performance
- ARM (NEON) unpacking for Q5_0 and Q5_1 weights in the CPU matmul kernels, for both decode and
  prompt processing. These formats used the scalar path before. llama.cpp's Q4_K_M files use Q5_0
  for tensors whose row length is not a multiple of 256 (Qwen2.5-0.5B, SmolLM2-360M, Gemma-3-1B).
  On Qwen2.5-0.5B Q4_K_M, 8 threads, Apple M4 Pro: 66 to 107 output tokens/s against 0.3.1's
  engine (A/B on battery, Low Power Mode).

## [0.3.1] - 2026-10-27

A patch release: one display fix and new benchmark documentation. The engine, its kernels and its
defaults are unchanged from 0.3.0.

### Fixed
- `dynalm --version` printed `cpu path: ggml kernels` on CPUs where DynaLM's own kernels are the
  default CPU path (SME CPUs such as Apple M4). It now reports the path the engine uses:
  `DynaLM kernels (default on this CPU)` there, ggml's kernels elsewhere. `dynalm doctor` and the
  `Device:` log line were already correct.

### Documentation
- CPU model sweep (`docs/perf/cpu-model-sweep.md`, in progress): accuracy against llama.cpp for
  27 GGUF models from 0.36B to 9B parameters and speed for 19 model shapes, on one Apple M4 Pro.

## [0.3.0] - 2026-10-26

DynaLM's own CPU engine (no ggml in the CPU path) and the CPU performance work measured on the
fixed CPU suite v1 (docs/perf/cpu-suite-v1.md; development measurements and the controlled runs:
docs/perf/cpu-experiments.md). Controlled result on an Apple M4 Pro (AC power, 5 rounds): 1.53×
geometric mean vs llama.cpp over the 8 workloads, peak memory 2.73× (1.5B) and 2.89× (4B) lower;
the 2× target is not met.

### Added
- **Strict native mode:** `DYNALM_NATIVE_ONLY=ON` (presets `macos-native`, `linux-native`) builds
  no ggml / llama.cpp code; `tools/ci/check_native.sh` proves it for a build.
- **qmm CPU kernels** (`dynacore/cpu/qmm`): int8 / int16 per-32 activations (the DD-053 / DD-076
  accuracy classes), NEON dot-product gemv and i8mm gemm, packed 4-row Q4_K / Q6_K copies.
- NEON grouped-head attention kernels (were scalar on ARM).
- **SME CPU tier** (Apple M4 and other CPUs with FEAT_SME and 512-bit streaming vectors): fp32
  panel gemm for prefill-sized matmuls (`sme_f32_gemm`, reference accuracy class) and flash
  attention for prompt chunks; `DYNACORE_SME=0` turns it off.
- **CPU suite v1** (`tools/bench/cpu_suite.py`, `docs/perf/cpu-suite-v1.md`): fixed CPU workloads,
  geometric mean vs llama.cpp / Ollama, memory metrics, and a keep / reject A/B rule.
- `tools/bench/baseline.sh`: `REQUIRE_AC=0` for development runs (marked not controlled).

### Changed
- **Default CPU path on SME CPUs (Apple M4 class):** `--backend cpu` and `auto` without a GPU use
  DynaLM's own CPU kernels there, also in builds with ggml's CPU kernels (`--backend ggml-cpu`
  still selects those). Other CPUs keep ggml's kernels as the default until they are measured.
  `dynalm doctor` reports the default.
- DynaCore's qmm CPU kernels default to the performance cores on hybrid CPUs (as ggml's do; CPU
  suite v1 was measured that way). `--threads` overrides.
- Release workflow: the macOS arm64 package is built on `macos-15` and the build stops if the
  SME tier does not compile.
- `--execution auto` (new default): compiled on GPUs and on CPUs with qmm kernels.
- On ARM CPUs with qmm, the native int8 path covers prefill too (no fp32 weight expansion).
- CPU memory: after a weight is packed, the file pages of the original are dropped from resident
  memory (`DYNACORE_QMM_RELEASE_SOURCE=0` keeps them); the CPU prefix cache keeps at most a
  quarter of the KV pool when `prefix_cache_max_blocks` is not set.
- CPU scheduling where the SME prefill runs: 512-token solo-prefill chunks and prefill bursts
  (decoding streams wait while prompts prefill, at most 2 s; `DYNALM_PREFILL_BURST_MS`).
- CPU decode attention on fp16 KV reads fp16 directly in the grouped kernels; prompt-chunk
  elementwise ops (add, rms_norm, KV writes, SiLU) run vectorized and in parallel. SiLU uses the
  polynomial exp (relative error ~1e-7).
- CPU decode matmuls split the weights into equal per-thread chunks of at least 2 MiB
  (`KernelPlan::gemv_min_chunk_bytes`, `DYNACORE_GEMV_CHUNK_KB`) instead of 32 small chunks per
  thread: decode steps 15–21 % faster on an M4 Pro, same output.

## [0.2.0] - 2026-10-22

Stable release of the 0.2.0-beta.1 code; no code changes since the beta. Verified on macOS on
Apple Silicon (one Apple M4 Pro, CPU and Metal); the Linux and Windows builds are CI-tested
(model-free tests in CI), Vulkan and CUDA experimental. This release also attaches the Vulkan
archives (experimental) and updates the Docker `:cpu` image. Release notes:
`.github/release-notes/v0.2.0.md`.

## [0.2.0-beta.1] - 2026-10-22

Beta pre-release: verified on macOS on Apple Silicon (one Apple M4 Pro, CPU and Metal); the Linux
and Windows builds are CI-tested (model-free tests in CI), Vulkan and CUDA experimental. Release notes:
`.github/release-notes/v0.2.0-beta.1.md`.

### Changed (release process)
- **The version has one source:** `project(VERSION)` plus `DYNALM_VERSION_SUFFIX` in
  `CMakeLists.txt` (`dynalm --version`: `0.2.0-beta.1`; also the SBOM and `CITATION.cff`).
  `tools/ci/version.py` prints it and checks a tag against it; `run_tests.sh` and the release
  packaging check the built and the installed binary's version.
- **Release workflow:** a tag must match the source version; the release is created as a draft for
  review, only when every package job passed (a failed platform no longer gets left out silently);
  a pre-release tag (`-alpha.N`, `-beta.N`, `-rc.N`) is marked pre-release and never Latest, leaves
  out the experimental Vulkan builds, and does not run the Docker image job (the `:cpu` image is
  for stable tags only); notes come from `.github/release-notes/<tag>.md`. Test:
  `tools/test_release.py`.

### Changed (compatibility)
- **`--backend native-cpu` is not part of this release.** It, and `runtime.device: native-cpu` /
  `DYNALM_BACKEND=native-cpu` from configurations written for earlier versions, now stop with an
  error that names the replacement; nothing is redirected silently. DD-053 accuracy on the CPU is
  `--mode exact` (DynaLM's own CPU kernels, unchanged), which also replaces
  `dynalm autotune --run --backend native-cpu`. The same kernels still serve `--mode safe` and the
  CPU fallback for models ggml's CPU kernels cannot run.

Performance and compatibility work: CPU int8 kernels, AutoTune, GPU scheduling, model fixes.
The numbers below were measured on one machine (Apple M4 Pro, macOS); the full end-to-end
benchmark is still pending. Methods and tools (`tools/bench/ab.sh`, `baseline.sh`, `bench_ppl`):
docs/benchmarking.md.

### Changed (performance)
- **Metal: DynaLM's GPU graphs are reordered for concurrency**, with ggml's own pass
  (`graph_optimize`, as llama.cpp's scheduler runs it), once per cached graph and before its memory
  is allocated. Before, ggml-metal put a barrier before 98% of DynaLM's dispatches. Barriers per
  decode step: Qwen2.5-0.5B 487 → 365 (llama.cpp 338), Qwen3-8B 688 → 581 (llama.cpp 506).
  Decode at one user (paired A/B, 8 pairs, 95% CI): 0.5B +7.5% [+6.4, +8.0], 8B +1.3%
  [+0.5, +1.8]; no regression in TTFT, ITL p99 or 4/16-user throughput; DD-053 PASS on six models
  (including Granite MoE and Gemma 3). `DYNACORE_GPU_GRAPH_OPTIMIZE=0` turns it off.
- **Metal: fewer false barriers (patched ggml).** ggml-metal treated two memory ranges that only
  touch as overlapping and put a barrier between independent kernels. DynaLM now builds its pinned
  ggml v0.27.0 with `cmake/patches/ggml-metal-mem-ranges-touching.patch` (a one-line fix, applied by
  `cmake/GgmlPatch.cmake`; recorded in the SBOM and NOTICE). Barrier placement only, not
  arithmetic. Decode at one user (paired A/B, 8 pairs, M4 Pro): Qwen2.5-0.5B ITL p50 −2.7%
  [−4.2, −1.8]; Qwen3-8B inconclusive. DD-053 PASS on six models. Test: `test_ggml_metal_ranges`.

### Changed (benchmarks)
- **`tools/bench/model_matrix.py` measures DynaLM and llama.cpp alike** (`bench_method`
  `d512-r5`): decode of 64 tokens at depth 512 in both (`llama-bench -d 512`; before, llama.cpp's
  tg128 ran at depth 0 against DynaLM's decode after a 512-token prompt), 5 repetitions after a
  warm-up in both (was 3 and `-r 2`), means with standard deviations (new `*_sd` columns),
  flash attention on in both, and llama.cpp's CPU run with op offload off (`-nopo 1`).
  `--bench-paths` selects the speed runs. Older results still render, labelled with their method.

### Added
- **Exact-mode CPU speed in the model matrix and the end-to-end report:** with `--bench`,
  `tools/bench/model_matrix.py` also measures `--backend cpu --mode exact` (DynaLM's
  DD-053 kernels) per model, one user and 4 users, against llama.cpp's CPU (columns
  `exact_prompt_tok_s`, `exact_decode_tok_s`, `exact_4user_tok_s`, appended). `matrix_summary.py`
  and `e2e_report.py` show it and still read older results. Adds about 0.5-1 h to an e2e run.
- **`bench_ppl`**: perplexity over a text file, scored as llama.cpp's `llama-perplexity` does,
  on any device (`BENCH_DEVICE=native|ggml-cpu|metal|vulkan|cuda`), for direct accuracy comparisons
  with llama.cpp (docs/benchmarking.md).
- **Sampling tests:** temperature, top-k, top-p, min-p, penalties, seeds and invalid parameters
  (`test_engine`; checked against deliberate errors in top-k, the nucleus cut and temperature).
- **Teacher-forced parity with llama.cpp** (`tools/bench/llama_parity.py`, `bench_ppl`
  `BENCH_KLD_REF` / `BENCH_KLD_BASE` / `BENCH_UBATCH` / `BENCH_FRESH_KV`): both engines score the
  same token ids; reports top-1 agreement, KL, both perplexities, near-ties and the first
  divergence per model and path, checks each run's settings and backend from its log (a run on the
  wrong backend is a setup mismatch, exit status 2), and repeats each engine against itself. It
  measures implementation agreement with llama.cpp, not task quality. Test: `tools/test_llama_parity.py`.
- **AutoTune** (`dynalm autotune`, `--mode`):
  - `dynalm autotune` reports the CPU features detected at run time, the built-in GPU backends,
    the kernels of each CPU path, what each mode runs and the tuned settings.
  - `--mode auto|performance|exact|safe` for `run`, `chat`, `serve` and `benchmark` (config key
    `runtime.mode`). `exact` keeps DD-053 accuracy everywhere (never ggml's 8-bit CPU kernels);
    `safe` runs the fp32 CPU reference kernels only. Contradicting `--backend` choices are errors.
  - `dynalm autotune --run MODEL` measures CPU thread counts (bounded by `--budget`, median of 3)
    and stores the fastest in a versioned cache (`~/.dynalm/autotune.json`). The cache is keyed by
    the CPU, features, cores, OS, version, compiler, ggml revision and binary; any change ignores
    it. A run on battery, under load or thermally limited is not stored (unless `--force`, then
    marked UNRELIABLE). Used only when `--threads` is not given; `DYNALM_AUTOTUNE=0` ignores it;
    `--clear` deletes it.
- **ARM int8 and int16 kernels for DynaCore's CPU path** (`--mode exact`), with SDOT
  (`vdotq_s32`) for Q8_0, Q4_0, Q4_K and Q6_K, 1-4 rows (DD-053), and int16 (DD-076, opt-in).
  Before, ARM ran these matmuls in fp32. Tested against an exact reference by `test_int_kernels`.
- **Runtime ARM feature detection**: dot product, i8mm, fp16, bf16, SVE, SME, read from the OS
  (sysctl on macOS, HWCAP on Linux, `IsProcessorFeaturePresent` on Windows) instead of compiler
  macros. `dynalm doctor` lists them. The SDOT kernels are built with `+dotprod` in their own file
  and only selected when the CPU has it.
- **Qwen3 runs on the GPU** (dense models). Per-head Q/K norm is one GPU op per layer. Qwen3-4B
  Q4_K_M passes DD-053 on Metal and Vulkan, with 4/4 greedy continuations identical:
  - Metal: decode +0.25% perplexity, prefill mean KL 0.0011;
  - Vulkan: decode +0.07%, prefill +0.09%, mean KL 0.000014.

- **Gemma 3 runs on the GPU.** Gemma 3 1B passes DD-053 on Metal and Vulkan, with 4/4 greedy
  continuations identical.
- **Mixture-of-experts on the GPU (Metal and Vulkan):** routing runs on the device in one op
  (`Device::moe_ffn`): router, softmax, top-k, expert-indexed matmuls (`mul_mat_id`) with fp32
  accumulation, the weighted sum, and the optional sigmoid-gated shared expert (Qwen2-MoE). The
  host round trip per layer is gone.
  - Granite 3.1 MoE 1B passes DD-053 on both. Metal: decode KL 0.000003, prefill +0.13%.
    Vulkan: decode +0.11%, prefill +0.15%, KL 0.000056.
  - The routed and shared-expert math matches a host reference within 1e-4 (new device test).
    No Qwen2-MoE file was run: the smallest is ~9 GB.
  - CUDA keeps MoE on the CPU until checked.
  - Accuracy tool: MoE batching checks (two sequences, fork vs scratch) are gated on mean KL
    ≤ 0.0025, with the greedy choice reported, not gated, as DD-053 reports greedy agreement.
    The old dense-model rule (max logit diff < 0.5) does not fit MoE. Granite's top-k routing
    has near-ties: the 8th and 9th experts are within 0.2% of the probability mass in ~12k
    row-layer choices, so a rounding difference between row counts swaps an expert. Metal had
    passed the old rule at 0.49; under the new report, one row's greedy choice differs at KL
    0.00066.
- **Sparse GPU attention (opt-in, `DYNACORE_GPU_SPARSE=1`):** each row reads only its own
  sequence's KV (ggml's sparse flash attention) instead of the shared span. Measured:
  - one user −10% (7.13 → 7.87 ms/token, 1.5B);
  - 8B at 16 users no change;
  - its bound in the graph key multiplied the prepared graphs (part of the memory problem
    above).

  So it is off by default.
- **Faster CPU by default: ggml's CPU kernels** (8-bit activations, as in llama.cpp; about 4×
  the native kernels in the v0.1.7 baseline). They are used for `--cpu`, and for `auto` when
  there is no GPU.
  - **Accuracy is llama.cpp-level, not DD-053.** Qwen2.5-1.5B: mean KL 0.023 against the fp32
    reference; llama.cpp's own CPU is 0.007 against its Metal path.
  - `--mode exact` keeps DynaLM's own DD-053 kernels.
  - Models the ggml kernels cannot run (MoE, Gemma 3) fall back to the native kernels.
  - Built when CMake >= 3.28 and not MSVC on ARM64 (`-DENABLE_GGML_CPU=OFF` to opt out). On
    x86-64 they need AVX2/FMA/F16C, else the native kernels run.
  - `dynalm --version` shows the CPU path.
- **Speculative decoding in the server with a draft model (`dynalm serve --spec qwen2.5:0.5b`).**
  - A small model with the same vocabulary drafts for every stream at once (`BatchDrafter`):
    catch-up passes, then k − 1 one-token passes, batched across streams, on the same device
    with its own KV cache, which counts toward the memory check.
  - Tested: identical text with the target itself as draft and with an unrelated model.
  - Qwen2.5-1.5B + 0.5B on Metal: 4 streams, 8/8 requests complete, 227 of 346 drafts accepted
    (66%).
  - `/metrics` and `dynalm benchmark` (`diag`) report drafted and accepted tokens.
- **Speculative decoding in the server (`dynalm serve --spec ngram [--spec-k K]`).**
  - Every decoding stream drafts up to K tokens by prompt lookup. The target scores them in the
    stream's own batch row, in the same continuous-batching step as the other streams.
  - Output equals plain decoding, greedy or sampled (exact verification). Rejected drafts' KV is
    rolled back.
  - A stream whose acceptance falls below 25% stops drafting for 32 steps, then tries again.
  - Tested: identical greedy text with 4 streams batched together (CPU); completion and
    accepted drafts on the GPU.
  - Speed not benchmarked yet: `dynalm benchmark --spec ngram` A/B.
- **Fast CPU path: ggml's repacked int8 kernels and performance-core threads.** Measured on an
  M4 Pro against llama.cpp `-t 8` on the same files, back to back:
  - Qwen2.5 1.5B prompt 295 → 573 tok/s (llama.cpp 549), decode 81 → 144 tok/s (166);
  - Qwen3 4B prompt 197 (185), decode 53.5 (67.7).
  - **Repack:** matmul weights are copied into ggml's `CPU_REPACK` buffer
    (`Device::prepack_weight`), so ggml runs its interleaved dotprod / i8mm GEMM and GEMV
    kernels. The copy is used only by matmuls; embedding lookups of tied tables read the
    original. Costs one extra copy of the matmul weights in RAM. `DYNACORE_GGML_REPACK=0` turns
    it off.
  - **Threads:** without `--threads`, ggml's CPU kernels use the performance cores only (8 of
    8P+4E). ggml splits work evenly and syncs at barriers, so the efficiency cores held each
    step back (decode 81 → 138 tok/s from this alone).
  - `-DDYNALM_GGML_NATIVE=ON` builds ggml for the build machine's exact CPU. It measured no gain
    here (the repacked dotprod kernels already run), so packages stay portable.
- **Memory admission:** a KV cache that cannot fit is refused at startup. The limits are 90%
  of RAM, or 95% of free GPU memory. The error names the largest `--ctx` that fits and
  `--kv q8_0`. Before, the server started and failed or swapped mid-run.
- **Release provenance:** release archives carry signed build attestations
  (`gh attestation verify`), and the Docker image carries provenance and an SBOM. See
  SECURITY.md.
- **`tools/bench/ab.sh`:** paired A/B runs of two builds (or one build under two settings) for
  `tools/bench_analyze.py`. Binaries are copied first, and every run waits for AC power, Energy
  Mode Automatic and a quiet machine. **`tools/bench/baseline.sh`:** the controlled DynaLM /
  llama.cpp / Ollama comparison behind the Phase 0 baseline.
- `bench_gpu_accuracy`: `BENCH_GPU=ggml-cpu` and `BENCH_PREFILL_CHUNKS=...`.
- **Model matrix with safetensors and speed** (`tools/bench/model_matrix.py`; the `.sh` wraps
  it, since macOS ships bash 3.2):
  - `hf:<owner>/<repo>` entries fetch `config.json`, the tokenizer and every safetensors shard
    through the Hugging Face API, after a `model_type` pre-screen and a size limit.
  - `--bench` adds DynaLM in-process speed (GPU, CPU; 1 and 4 users) and `llama-bench` on the
    same GGUF with ratios.
  - `tools/bench/models-50.txt`: 50 models in both formats, with 13 GGUF links
    corrected (wrong file names, split files, missing repos; each noted in the list).
  - `tools/bench/models-51-80.txt`: 30 more models (7-35B) in both formats, with the GGUF files
    resolved from repo pages and searches.
  - Safetensors downloads take only the indexed shards. Mistral repos' duplicate
    `consolidated.safetensors` had doubled their size.
- **`tools/bench/model_matrix.sh`:** a correctness sweep over a model list (default
  `tools/bench/models-matrix.txt`: 43 models, 12 architectures, Q2_K to F16 and IQ4_XS, two MoE).
  - Per model: support check, download, generation on `auto` / `cpu` / exact mode with
    cross-path text agreement, and the DD-053 check with `--accuracy`.
  - Downloads are deleted after each row unless `--keep`.
  - `--check-only` validates the whole list in ~2 min: all 43 links resolve, 42 supported,
    OLMoE unsupported (kept as a negative test).
- **`tools/bench/e2e.sh`:** unattended end-to-end runs. It snapshots the binaries, then runs the
  model matrix (tier 1: 27 entries across six providers; correctness, DD-053 and speed vs
  llama-bench on GPU and CPU), the engine comparison (`baseline.sh --quick`) and the A/B against
  v0.1.7 (GPU and CPU gates). `tools/bench/e2e_report.py` writes one `REPORT.md`. Speed runs are
  gated on AC power and a quiet machine (`model_matrix.py --gate`).
- **`tools/bench/baseline.sh --quick`:** about 45 min instead of 4+ h. 2 rounds, fewer prompt
  sizes, user counts and requests.
- **`tools/bench/report.py`:** turns `baseline.sh` runs into the markdown tables for
  docs/benchmarks.md: setup, run conditions, errors, medians ± spread, verdicts. Tested by
  `tools/test_bench_report.py` in CI. Publishing steps: docs/benchmarking.md.

### Changed
- **DynaLM's ARM64 CPU kernels use int8 activations up to 64 rows** (was 4), so prompts use the
  SDOT kernels too. M4 Pro, Qwen2.5-1.5B, 401-token prompt in exact mode: 2394 → 1835 ms. The
  per-row arithmetic is that of decode (DD-053), and prefill chunks of 32 and 64 measure perplexity
  +0.04%, mean KL 0.0012. x86 keeps 4 (the AVX2 crossover). `--int8-decode N` overrides it.
- **ggml-cpu keeps one worker pool** (`ggml_backend_cpu_set_threadpool`), as llama.cpp does.
  Before, ggml started and joined its threads for every graph, so once per token. Decode ITL on
  Qwen2.5-1.5B dropped from 6.0 to 5.9 ms (on battery, alternating runs). On AC power, back to back
  with llama-server on the CPU: ITL p50 6.2 vs 6.0 ms, prompt (128 tokens) 212 vs 211 ms.
  `DYNACORE_GGML_THREADPOOL=0` restores the old behaviour.
- **The ggml device is split by role.** `ggml_device_internal.h` holds the shared types,
  buffers and the device class; `ggml_graph.cpp` lowers recorded calls to a ggml graph (about
  500 lines); `ggml_device.cpp` runs segments and holds the registry. Behaviour is unchanged:
  the same 569,718 device checks pass.
- **Device selection is one pure, tested policy** (`runtime/device_select.{h,cpp}`,
  `plan_devices`). Every combination of request (auto / --gpu / cpu / exact / named),
  model and build is unit-tested without a GPU. `Engine::create` opens the planned devices in
  order.
- **GPU scheduling: the v0.1.7 profile stays the default; a predictive controller is opt-in**
  (`DYNALM_DECODE_PREDICTIVE=1 DYNALM_DECODE_TARGET=4`). The predictive controller sizes each
  mixed step from a learned per-row cost. Measured on an M4 Pro (2026-10-18, A/B against v0.1.7
  and traces):
  - ITL p99 −39% to −52% at 4-16 users;
  - but ITL p50 3-4× worse at 16 users (1.5B: 26.6 → 83.5 ms; 8B: 80 → 310 ms) and −7%
    throughput at 8B, because it pins every mixed step at the protected minimum.
  - Re-measured 2026-10-20 at target 8×: the longest mixed step −17% to −44% and ITL p99 −5% to
    −18% at 4-16 users, but Llama-3.1-8B Q8_0 at 16 users −6% tok/s with ITL p50 +13%, and TTFT
    p99 +2% to +14%. It stays opt-in.
- **GPU graph cache:** when full, it now drops the least recently used graph instead of all
  graphs. Prepared graphs are also capped at 1 GiB of workspace (`DYNACORE_GPU_GRAPH_MB`).
  - Each graph holds its own activation buffers (~86 MiB for an 8B model).
  - Without the cap, 26 graphs (2.2 GiB) plus 8.5 GB of weights and 8 GB of KV made Metal fail
    steps with "Insufficient Memory" at 16 users (8B: 20 → 50 tok/s once fixed).

### Fixed
- **Metal slowed down on long-running servers.** Attention over the paged KV pool covered every
  cell up to the highest block a step used; once the prefix cache held most of the pool, new
  requests got high block ids and every step attended over nearly the whole pool (Qwen2.5-1.5B,
  16 users, 65,536-token pool: typical token gap 21 ms on a fresh server, 136 ms with a full
  cache). Metal now uses ggml's sparse attention automatically for steps whose span is at least
  32× the longest context (up to 4,096 tokens): 23 ms, +54% tok/s, fresh servers unchanged, Metal
  parity unchanged on eight models. `DYNACORE_GPU_SPARSE=0` restores the previous dense-only
  behaviour, `=1` forces sparse attention. Not fixed for contexts above 4,096 tokens (ggml's sparse
  path does not cover them) or for Vulkan, CUDA and ggml-cpu. Test: `test_ggml_device`.
- **Benchmarks reused prompts across client processes.** `dynalm benchmark` numbered prompts from
  0 in every process, so `tools/bench/baseline.sh`'s later points repeated earlier prompts and hit
  both engines' prompt caches (llama.cpp processed 1 token of 512). New `--prompt-base N`;
  `baseline.sh` gives every point its own range. Multi-user results measured before this fix are
  invalid. `baseline.sh` also runs llama.cpp on the CPU with `-dev none`, checks each server's
  backend from its log, randomizes and balances the engine order, adds cool-downs and open-loop
  points (`OPEN_RATES`), and records build flags.
- **The fast CPU path (ggml-cpu) used the GPU's scheduling profile.** ggml-cpu runs whole graphs
  like the GPUs, so it received their mixed-step budgets (prompt chunks of up to 128 rows in steps
  that also carry other users' tokens); on the CPU those steps took about 11× a decode step. The
  GPU profile now applies only to GPU devices. Measured (M4 Pro, ggml-cpu, 4 users, 512-token
  prompts, 3 shuffled rounds): Qwen2.5-1.5B 74.2 → 85.5 tok/s, ITL p99 196 → 82 ms; Qwen3-4B
  26.4 → 28.9 tok/s, ITL p99 557 → 260 ms. Single-user speed unchanged.
- **Granite (MoE) on the fast CPU path disagreed with llama.cpp** (93.8-94.4% top-1): the expert
  weights were not repacked for ggml's `mul_mat_id` as llama.cpp does, so other kernels ran. They
  are now repacked; teacher-forced top-1 against llama.cpp 99.96-100%, identical perplexity.
  Prefill +26% (898 → 1,135 tok/s, Granite-3.1-1B-A400M, M4 Pro); peak memory +696 MiB
  (the repacked copy of the expert weights).
- **Gemma on the fast CPU path disagreed with llama.cpp** (96.9-97.1% top-1): attention scaled
  after the dot product, llama.cpp scales the query first, and the f16 rounding differs. The CPU
  path now uses llama.cpp's order (Gemma 1/2/3 only); top-1 100.00%, identical perplexity. Both
  orders are equally accurate against an f32 reference; this is a parity fix.
- **Metal ran out of GPU memory with large KV caches** (e2e-20261019-0022: Llama-2-7B,
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`, model + KV 20,275 MiB). Automatic KV sizing used
  host free RAM only; Metal reported no limit. The Metal device now reports its working set
  (`recommendedMaxWorkingSetSize`, 18,186 MiB on a 24 GB M4 Pro) and current allocation. Automatic
  KV sizing caps the cache at what that leaves after a 10% margin, the weights, the 1 GiB graph
  workspace and the activation scratch (`device_kv_budget`), each counted once. The cap uses the
  KV pool's own block arithmetic (`kv_tokens_within`), so the admission check never refuses an
  automatically sized cache. Below 1,024 tokens it does not shrink further: the GPU is refused at
  load, `--backend auto` starts on the CPU and `--gpu` names the largest `--ctx` that fits. An
  explicit `--ctx` that cannot fit fails the same way, instead of failing on the GPU mid-run.
- **Support checks ignored the tokenizer** (e2e: DeepSeek-Coder-1.3B safetensors was reported
  supported from its architecture, 2.5 GB downloaded, then refused at its pre-tokenizer):
  - `dynalm inspect` reports architecture, tokenizer and weights separately, the tokenizer by the
    load path's own code, and says `Supported: YES` only when all three pass;
  - `gguf_support_status` (`dynalm models`, `pull`) includes the tokenizer, and `dynalm models`
    checks the tokenizer of safetensors folders too;
  - `dynalm pull --check` checks the tokenizer when its keys are within the 256 KiB it reads;
  - the model matrix fetches only the small config and tokenizer files first and skips the weights
    of an unsupported tokenizer.

  DeepSeek-Coder's `Split+Split+Split+Split+Digits+ByteLevel` pre-tokenizer
  (`[\r\n]`, `\s?\p{L}+`, `\s?\p{P}+`, `[一-龥ࠀ-一가-퟿]+`, individual digits) is still unsupported;
  it needs a Unicode punctuation class and sequential HF split semantics.
- **GPU A/B plans could not reach a verdict:**
  - the plan tested a predictive scheduler absent from both builds;
  - 1 session × 3 rounds without warm-up, where round 1 ran ~10% faster for both arms.

  Fixes:
  - `plans/gpu-regression.json` checks what changed (no regression, single-user equivalence);
  - `phase1-gpu.json` is for the opt-in predictive profile within one build;
  - `ab.sh` adds `--warmup-rounds` (default 1, not analyzed) and defaults to 2 sessions × 4 rounds;
  - with `"require_ci": true`, `bench_analyze.py` needs the 95% CI to clear the threshold for a
    PASS and calls fewer than 4 pairs INCONCLUSIVE (only stricter).
- **`dynalm doctor` predated GPU support:** on a Metal build it said "GPU: none detected (not used by
  this build)" and warned "this release runs on the CPU only". It now lists the backends built in
  (DynaLM's CPU kernels and their tier, ggml's CPU kernels, Metal / Vulkan / CUDA with the devices
  each reports, without initializing them), the device `--backend auto` tries first, both default
  thread counts and the AutoTune cache state (also in `--json`: `gpu_backends`, `ggml_cpu`,
  `ggml_threads`, `autotune`).
- **`--mode safe --gpu` switched to the CPU silently;** it is now an error, like `--mode safe
  --backend metal`. A bad `--mode` value names the valid ones.
- **`dynalm autotune --run` stored unverified results** where it cannot read the power and load
  state (Windows); they now need `--force` and are marked UNRELIABLE.
- **Notices:** NOTICE and THIRD_PARTY_LICENSES.md said ggml is in GPU builds only; CPU builds include
  it by default too (`ENABLE_GGML_CPU`). The Unicode License V3 for the Unicode Character Database
  (the tokenizer tables) is added.
- **Llama 3 GGUFs without `tokenizer.ggml.add_bos_token` ran without BOS** (e.g. bartowski's
  Llama-3.2 files), in raw text and in chat (the template's BOS is left to the tokenizer). As
  llama.cpp does, the Llama 3 pre-tokenizer ids and Tekken now default to BOS. Wikitext-2
  perplexity, 20 chunks of 512: Llama-3.2-1B IQ4_XS 16.567 → 16.409 (llama.cpp 16.435);
  Llama-3.1-8B Q8_0 on Metal 8.1455 → 8.1993 (llama.cpp 8.1993).
- **Phi-3 / Phi-3.5 on the GPU and on ggml's CPU kernels** failed with "a step reads memory
  written through a different view (aliasing)". RoPE writes Q inside the fused Q|K|V value; the
  graph builder now keeps the rest of that value (K, V) live instead of dropping it. A test runs a
  fused-QKV model and compares it with separate Q, K, V weights on native CPU, ggml-cpu, Metal and Vulkan.
- **Phi-3 / Phi-4 LongRoPE was ignored** (`rope_factors_short/long` were "not used by the
  engine"). As llama.cpp does, DynaLM now applies the short factors, or the long ones when a
  sequence can exceed the original context. It also applies the attention factor
  `rope.scaling.attn_factor` inside RoPE (`RopeConfig::attn_factor`, on cos and sin), so only the
  rotated dims are scaled; this matters for Phi-4-mini's partial RoPE (96 of 128 dims). Wikitext-2
  perplexity, 20 chunks of 512:
  - Phi-3.5-mini: llama.cpp 7.1421; DynaLM 7.1144 on Metal (7.7083 before);
  - Phi-4-mini: llama.cpp 11.4255 on Metal and 11.5078 on CPU; DynaLM 11.4256 on Metal,
    11.4217 on Vulkan, 11.4245 native and 11.5252 on ggml-cpu (12.18 with the factor on the scores).
- **Phi-4-mini's chat template** (`'<|' + role + '|>'`, no newlines) was not recognized; it is now
  the `phi4` format.
- **IQ4_NL and IQ4_XS models** were refused at load. They now run on every device: ggml's kernels
  on the GPU and ggml-cpu, dequantization on DynaLM's CPU kernels. They are checked against ggml's
  matmul on ggml-cpu, Metal and Vulkan.
- **Phi-4-mini (pre-tokenizer `gpt-4o`)** was refused. The GPT-4o (o200k) split is implemented as the
  HF regex, combining marks included; `llama4` uses it too. The Unicode tables are regenerated
  from UnicodeData.txt 15.0.0 (`tools/gen_unicode_tables.py --ucd`).
- **Chat models without a recognized template** (TheBloke's Llama-2-chat and DeepSeek-Coder GGUFs)
  failed in chat mode. DeepSeek-Coder's `### Instruction:` template is now rendered natively. A file
  with no template uses the format its name implies (`*llama-2*chat*`, `*mistral*instruct*`,
  `*deepseek-coder*instruct*`) and logs it; base models stay unsupported for chat.
- **`tools/bench/ab.sh`** marked every CPU run "not quiet": its after-run check counted the load
  the benchmark itself creates. It now re-checks only power and power mode.
- **ggml without OpenMP.** An OpenMP ggml build needs libgomp (Linux) or vcomp140.dll (Windows)
  at run time. The Docker image failed with `libgomp.so.1: cannot open shared object file`; ggml's
  own thread pool is used now. The Docker build installs git to fetch the pinned ggml.
- **Benchmark client:** a stream that ends with an `{"error": ...}` event now counts as a failed
  request. Before, it counted as a success.
- **ggml-cpu device:** an unaligned host tensor is copied, and a buffer larger than physical
  RAM is refused. Before, ggml aborted on the first and was OOM-killed on the second.
- **CI:** "Metal/Vulkan GPU tests ran" now requires that provider by name in the test summary.
  The always-available ggml-cpu provider could otherwise have passed it.
- `test_solo_prefill` expected the pre-v0.1.5 `solo_prefill_chunk` default of 0. It is now -1,
  the device's default.

## [0.1.7] - 2026-10-17

Pre-launch fixes: reliable downloads and accurate docs.

### Fixed
- **`dynalm pull` no longer hangs on a stalled connection.**
  - A download slower than 1 KB/s for 60 s, or a connection that cannot connect within 30 s,
    now ends the attempt.
  - DynaLM then resumes the partial file, up to 10 attempts. curl's own retry restarted from
    the first byte, so it is no longer used for the download.
  - Before this, a silent connection made `pull` wait forever with no message.
- **SECURITY.md:** it said the server has no authentication (API keys exist, and network
  binds require one) and named `dynalm rm` (it is `dynalm models rm`).

### Docs
- **New `docs/benchmarks.md`** with every measured number:
  - the exact setup: Apple M4 Pro, 24 GB, macOS 26.7, AC power, model checksum, engine
    versions and flags;
  - where llama.cpp is faster;
  - CPU speed marked unverified.
- **README:**
  - removed two speed claims no measurement backs: an x86-64 laptop CPU figure, and
    "4× faster than the CPU mode";
  - the speed section now lists where llama.cpp is ahead.
- **Llama 3.x GGUF is listed as running on the GPU.** Llama 3.2 1B passed the GPU accuracy
  check (perplexity −0.004% decode, −0.112% prefill, 4/4 greedy identical, deterministic).

## [0.1.6] - 2026-10-17

GPU first, with nothing to set up: the normal installers now deliver GPU builds.

### GPU by default
- **Downloads with the GPU included:**
  - the macOS Apple Silicon package includes Metal;
  - new `dynalm-linux-x86_64-vulkan` and `dynalm-windows-x86_64-vulkan` packages.
- **The installers choose the build:** they install the Vulkan package when the machine has a
  Vulkan driver and a GPU, otherwise the CPU package. Older releases without a Vulkan package
  fall back to the CPU package.
- **Installer flags:** `install.sh --cpu` / `--gpu` and `install.ps1 -Cpu` / `-Gpu`.
- **From source:** `install.sh --from-source` enables every GPU toolkit it finds (Metal,
  Vulkan, CUDA).
- **Run-time flags:** `--cpu` and `--gpu` on `run`, `serve` and `benchmark`;
  `runtime.device: gpu` in the config file.
  - `--gpu` requires a GPU and stops with the reason instead of using the CPU.
  - `--gpu` also overrides a saved `device: cpu` for one run.
  - The default (`auto`) is GPU first, with CPU fallback.
- **`dynalm --version`** lists the devices built in (for example `devices: cpu metal`).
- **Release workflow:**
  - the toolchain install retries;
  - publish downloads only the platform archives.
- **CI:** package jobs for the Metal and Vulkan variants. `install-check` installs the Vulkan
  build with `--gpu`.

## [0.1.5] - 2026-10-17

0.1.4 was prepared but never published as a GitHub release. Its changes (the [0.1.4] section
below) ship in 0.1.5 together with everything listed here.

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

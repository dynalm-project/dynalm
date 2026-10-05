# Changelog

## [Unreleased]

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
- ARM64 NEON kernel tier (Apple Silicon, Graviton, Windows on ARM). SIMD options are now
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

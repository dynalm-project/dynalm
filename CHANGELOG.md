# Changelog

## [Unreleased]

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
- Tests: `test_config`, `test_compat` (spec §38 suite for every tiny architecture plus real
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

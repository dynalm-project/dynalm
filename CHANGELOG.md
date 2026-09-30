# Changelog

## [Unreleased]

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

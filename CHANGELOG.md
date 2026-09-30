# Changelog

## [Unreleased]

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

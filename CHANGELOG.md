# Changelog

## [Unreleased]

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

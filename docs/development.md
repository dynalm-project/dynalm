# Development

## Requirements

- CMake ≥ 3.24, Ninja
- Windows: MSVC Build Tools 2022 (C++ workload). Build from a *Developer
  PowerShell / x64 Native Tools* prompt, or call `vcvars64.bat` first.
- Linux: gcc ≥ 13 or clang ≥ 17 (both need `<format>`)
- No GPU is required. GPU flags (`ENABLE_CUDA`, …) default to OFF and are not
  implemented yet.

## Build

```sh
cmake --preset msvc-release        # or linux-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Sanitizers: `msvc-asan`, `linux-asan-ubsan`, `linux-tsan`.

Without presets:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build
```

## Feature flags

| Flag | Default | Meaning |
|---|---|---|
| `ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_AMX` | ON / OFF / OFF | compile those CPU kernel tiers (selected at runtime) |
| `ENABLE_CUDA` / `HIP` / `METAL` / `VULKAN` | OFF | future; ON is a configure error |
| `ENABLE_SERVER` / `ENABLE_TESTS` / `ENABLE_BENCHMARKS` | ON | build components |
| `ENABLE_ASAN` / `ENABLE_UBSAN` / `ENABLE_TSAN` | OFF | sanitizers |

## Run

```sh
build/msvc-release/src/engine info
build/msvc-release/benchmarks/bench_foundation
```

## Conventions

- `namespace engine`. Files are `snake_case`, types `PascalCase`, functions `snake_case`.
- Errors: `Status` / `Result<T>`. No exceptions on hot paths.
- SIMD intrinsics only in `src/backends/cpu/<isa>/*.cpp`.
- No heap allocation per token. No logging per token above DEBUG.
- Every phase ends with a green build, tests, benchmarks, updated docs, and a commit.

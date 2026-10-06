# Build

DynaLM is a CMake project with two libraries and two executables (DD-068).

| Target | Output | Links |
|---|---|---|
| `dynacore` | `libdynacore` (static) | nothing from the repository |
| `dynalm_runtime` | `libdynalm` (static) | `dynacore` |
| `dynalm_server` | `libdynalm_server` (static) | `dynalm_runtime`, cpp-httplib (pinned, SHA256-checked header) |
| `dynalm` | `build/<preset>/bin/dynalm` | `dynalm_server` (or `dynalm_runtime` with the server off) |
| `dynacorec` | `build/<preset>/bin/dynacorec` | `dynacore` (the DynaCore compiler driver, DD-073) |

```sh
cmake --preset msvc-release          # linux-release, linux-clang-release, macos-release
cmake --build --preset msvc-release
ctest --preset msvc-release
```

Windows needs a shell with MSVC on the PATH: a Developer PowerShell, or `tools\dev.cmd <command>`
in this repository.

## Presets

| Preset | Purpose |
|---|---|
| `msvc-release`, `linux-release`, `linux-clang-release`, `macos-release` | release builds with tests and benchmarks |
| `msvc-asan`, `linux-asan-ubsan`, `linux-tsan` | sanitizers |
| `core-only` | DynaCore alone (`DYNALM_CORE_ONLY=ON`): proves DynaCore builds and tests without `dynalm/` |

## Options

| Option | Default | Effect |
|---|---|---|
| `ENABLE_TESTS`, `ENABLE_BENCHMARKS`, `ENABLE_SERVER` | ON | components |
| `ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_AMX` | ON / OFF / OFF | x86-64 kernel tiers (selected at run time; off on other CPUs) |
| `ENABLE_NEON` | ON | ARM64 kernel tier |
| `ENABLE_CUDA` / `HIP` / `METAL` / `VULKAN` | OFF | future devices; ON is a configure error |
| `DYNALM_CORE_ONLY` | OFF | configure DynaCore only |
| `DYNALM_STATIC_RUNTIME` | OFF (ON in release builds) | static C/C++ runtime |
| `DYNALM_LTO` | OFF | link-time optimization (DD-063) |
| `ENABLE_ASAN` / `ENABLE_UBSAN` / `ENABLE_TSAN` | OFF | sanitizers |

## Tests

- `ctest` runs these suites:
  - DynaCore: foundation, tensor, quant, kernels, IR, language;
  - DynaLM: loader, tokenizer, architectures, runtime, scheduler, KV, prefix cache, server,
    compiled execution, registry/config, ...;
  - the boundary check `boundary.dynacore`.
- Real-model tests skip themselves when `models/*.gguf` are absent (`tools/fetch_models.sh`).

## Boundary

- `tests/boundary/check_boundary.py` fails if anything under `dynacore/` includes a
  non-DynaCore header or names an LLM-platform concept (model families, GGUF, tokenizer, HTTP,
  scheduler, the product name).
- The CI job `boundary` runs the check and the `core-only` build.

More: [development.md](development.md) (platforms, CI, sanitizers, Smart App Control).

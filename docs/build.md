# Build DynaLM from Source (CMake on Linux, macOS and Windows)

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
build/msvc-release/bin/dynalm doctor   # DynaCore kernel self-test
```

Windows needs a shell with MSVC on the PATH: a Developer PowerShell, or `tools\dev.cmd <command>`
in this repository.

## Presets

| Preset | Purpose |
|---|---|
| `msvc-release`, `linux-release`, `linux-clang-release`, `macos-release` | release builds with benchmarks |
| `msvc-asan`, `linux-asan-ubsan`, `linux-tsan` | sanitizers |
| `core-only` | DynaCore alone (`DYNALM_CORE_ONLY=ON`): proves DynaCore builds without `dynalm/` |

## Options

| Option | Default | Effect |
|---|---|---|
| `ENABLE_BENCHMARKS`, `ENABLE_SERVER` | ON | components |
| `ENABLE_AVX2` / `ENABLE_AVX512` / `ENABLE_AMX` | ON / OFF / OFF | x86-64 kernel tiers (selected at run time; off on other CPUs) |
| `ENABLE_NEON` | ON | ARM64 kernel tier |
| `ENABLE_METAL` | OFF | Metal GPU backend (Apple Silicon); preset `macos-metal` |
| `ENABLE_VULKAN` | OFF | Vulkan GPU backend (AMD, Intel, NVIDIA, Qualcomm; macOS through MoltenVK). Needs the Vulkan headers, loader, `glslc` and SPIRV-Headers. Presets `linux-vulkan`, `msvc-vulkan`, `macos-vulkan` |
| `ENABLE_CUDA` | OFF | CUDA GPU backend (NVIDIA). Needs the CUDA toolkit; `CMAKE_CUDA_ARCHITECTURES` picks GPU generations. Presets `linux-cuda`, `msvc-cuda` |
| `ENABLE_HIP` | OFF | not implemented (configure error); use `ENABLE_VULKAN` for AMD GPUs |

GPU backends can be combined, and all of them need CMake 3.28 or newer. `--backend auto`
tries the built-in GPUs in this order: CUDA, Metal, Vulkan.

**Windows on ARM (Snapdragon):** open an *ARM64* Developer Command Prompt and run
`cmake --preset msvc-release`, or `cmake --preset windows-arm64` to build with clang-cl.
Both use the NEON kernels.
| `DYNALM_CORE_ONLY` | OFF | configure DynaCore only |
| `DYNALM_STATIC_RUNTIME` | OFF (ON in release builds) | static C/C++ runtime |
| `DYNALM_LTO` | OFF | link-time optimization (DD-063) |
| `ENABLE_ASAN` / `ENABLE_UBSAN` / `ENABLE_TSAN` | OFF | sanitizers |

## Checking a build

- `dynalm --version` prints the build and the kernel tiers compiled in.
- `dynalm doctor` (or `--json`) reports the CPU, the kernel tier in use and memory, and runs a
  DynaCore kernel self-test (selected tier against the generic kernels).
- `tools/ci/check_isa.py <dynalm>` checks that the selected tier matches the CPU and that the
  generic fallback works (`DYNACORE_ISA=generic`).
- `tools/ci/package.sh` builds the release archive, installs it through the installer into an
  empty prefix, and runs these checks on the installed binaries.
- The `core-only` preset builds DynaCore without `dynalm/`, so any dependency of DynaCore on
  DynaLM fails to compile; the CI job `boundary` runs it.

## Test executables

With `ENABLE_BENCHMARKS=ON` (the default in every preset), the build also produces
self-checking test programs. Each exits non-zero on the first failed check and prints
`all ... tests passed (N checks)` otherwise. They are not registered with CTest.

`bash tools/ci/run_tests.sh build/<preset>` runs every model-free test program and the tool
tests; CI runs it on every platform and under the sanitizers. Tests whose kernels the CPU lacks
(AVX2 / AVX-VNNI) skip themselves.

| Test | Location | Needs |
|---|---|---|
| `test_matmul_registry` | `build/<preset>/dynacore/` | nothing: matmul kernel selection against the pre-registry dispatch, every kernel's numerics, bit-identity of the split-byte and wide-decode paths, generic-tier fallback, diagnostics, external kernels |
| `test_int16_prefill`, `test_packed_8rows`, `test_thread_pool`, `test_dynakernel*` | `build/<preset>/dynacore/` | nothing |
| `test_batch_planner` | `build/<preset>/dynalm/benchmarks/` | nothing: the wide-decode switch applies only to decode-only steps of ≤ N rows and selects the integer decode kernels |
| `python tools/test_bench_analyze.py` | — | Python 3: the paired A/B analyzer ([benchmarking.md](benchmarking.md)) |
| `test_loadgen_openloop`, `test_prefill_priority`, `test_solo_prefill` | `build/<preset>/dynalm/benchmarks/` | a GGUF model as the first argument (for example `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`) |
| `python tools/test_compare_engines.py` | — | Python 3: checks of the engine-comparison report (no servers) |
| `python tools/ci/smoke_api.py <dynalm> <model.gguf>` | — | a model: CLI errors, `serve` start, the OpenAI-compatible endpoints, malformed requests, `/metrics`, graceful shutdown |

Accuracy of activation-quantized paths is checked with
`build/<preset>/dynalm/benchmarks/bench_int8_accuracy <model.gguf> [threads]` against the DD-053
contract (perplexity change ≤ +1%, mean KL ≤ 0.0025). Set `BENCH_PREFILL_ONLY=1
BENCH_PREFILL_CHUNKS=32,64` for the prefill paths, with the path's environment switches set
(see [dynacore.md](dynacore.md)).

## Platforms

| Platform | Kernels | Verified by |
|---|---|---|
| Linux x86-64 | generic + AVX2 | CI: gcc, clang, ASAN/UBSAN, TSAN |
| Linux ARM64 | generic + NEON | CI (`ubuntu-24.04-arm`) |
| macOS Apple Silicon | NEON | CI (`macos-14`) |
| Windows x64 | generic + AVX2 | MSVC build locally; CI |

## Models for local runs

`bash tools/fetch_models.sh` downloads SmolLM2-135M and Qwen2.5-0.5B into `models/`, which is
git-ignored.

## Windows Smart App Control

- In enforce mode, Smart App Control can block freshly linked, unsigned executables
  ("An Application Control policy has blocked this file").
- The build does not work around this host policy. Rely on the CI jobs, or change the policy
  yourself.

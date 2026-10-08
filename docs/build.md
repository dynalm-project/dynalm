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
| `ENABLE_CUDA` / `HIP` / `METAL` / `VULKAN` | OFF | future devices; ON is a configure error |
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

# Installation

You install DynaLM only. DynaCore is a static library inside the `dynalm` binary, so there is
nothing separate to install.

## Quick start

### Linux and macOS

```sh
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh
dynalm doctor
dynalm pull qwen3:4b
dynalm run qwen3:4b
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex
# open a new terminal so the PATH change applies
dynalm doctor
dynalm pull qwen3:4b
dynalm run qwen3:4b
```

What the installers do:

1. Detect your OS and CPU architecture. An unsupported platform is an error, never a wrong
   binary.
2. Download `dynalm-<os>-<arch>` and `SHA256SUMS` from the latest GitHub release, or from
   `--version v0.1.0` / `-Version v0.1.0`.
3. Verify the archive's SHA-256. A mismatch or a missing checksum stops the install.
4. Install `dynalm` and `dynacorec`:
   - Linux and macOS: `~/.local/bin` (change with `--prefix DIR`);
   - Windows: `%LOCALAPPDATA%\Programs\DynaLM\bin`, added to your user PATH (`-NoPath` to
     skip). No administrator rights are needed.
5. Run `dynalm --version`, and print PATH instructions if needed.

## Release archives

| Archive | Platform | Kernels selected at run time |
|---|---|---|
| `dynalm-linux-x86_64.tar.gz` | Linux x86-64, glibc ≥ 2.35 (Ubuntu 22.04+, Debian 12+, RHEL 9+) | AVX2, else generic |
| `dynalm-linux-arm64.tar.gz` | Linux ARM64, glibc ≥ 2.35 | NEON |
| `dynalm-macos-arm64.tar.gz` | macOS on Apple Silicon | NEON |
| `dynalm-macos-x86_64.tar.gz` | macOS on Intel | AVX2, else generic |
| `dynalm-windows-x86_64.zip` | Windows 10/11 x64 | AVX2, else generic |
| `SHA256SUMS` | SHA-256 of every archive | |

- **Layout:** `bin/dynalm`, `bin/dynacorec`, and `share/doc/dynalm/` (README, LICENSE,
  NOTICE, CHANGELOG, docs).
- **Self-contained:** the C/C++ runtime is linked statically, so no Visual C++
  redistributable and no libstdc++ version requirement.
- **CPU tiers:** a CPU without AVX2 runs the generic kernels; nothing fails. `dynalm doctor`
  shows which tier is in use.

Manual install:

```sh
curl -fLO https://github.com/dynalm-project/dynalm/releases/latest/download/dynalm-linux-x86_64.tar.gz
curl -fLO https://github.com/dynalm-project/dynalm/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing            # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
sh scripts/install.sh --archive dynalm-linux-x86_64.tar.gz --sha256sums SHA256SUMS
# or: tar xzf dynalm-linux-x86_64.tar.gz && ./dynalm-linux-x86_64/bin/dynalm doctor
```

```powershell
.\scripts\install.ps1 -Archive .\dynalm-windows-x86_64.zip -Sha256Sums .\SHA256SUMS
```

## From source

| Platform | Command | Needs |
|---|---|---|
| Linux, macOS | `sh scripts/install.sh --from-source` | C++20 compiler (GCC ≥ 13, Clang ≥ 17, Apple Clang 15), CMake ≥ 3.24, Ninja (optional) |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -FromSource` | Visual Studio 2022 Build Tools with the C++ workload |

Building, presets and tests: [build.md](build.md).

## Docker

- **CPU image:** `ghcr.io/<owner>/dynalm:cpu`, multi-arch amd64 and arm64, published by the
  release workflow. Build it locally with `docker build -t dynalm .`.

  ```sh
  docker run --rm -p 8000:8000 -v "$PWD/models:/models" ghcr.io/<owner>/dynalm:cpu \
    serve /models/qwen2.5-0.5b-instruct-q4_k_m.gguf
  ```

  - It runs as a non-root user.
  - Models are mounted at `/models`.
  - The server binds `0.0.0.0` inside the container.
- **CUDA image:** none is published. This release runs on the CPU only.

## Troubleshooting

- **`dynalm doctor`:** the first thing to run, and the output to paste into an issue.
  `dynalm doctor --json` gives the same in machine-readable form.
  - It reports:
    - version, platform and architecture;
    - CPU and its features;
    - the kernel tier in use;
    - threads and memory;
    - where `dynalm` is installed;
    - the model store.
  - It also runs a DynaCore kernel self-test. `Status: Ready` means all of this checked out.
- **Force the portable kernels:** `DYNACORE_ISA=generic dynalm ...`, to rule out a SIMD
  problem.
- **Windows Smart App Control:** it can block executables that are not signed ("An
  Application Control policy has blocked this file"). Official signing is not in place yet.
- **macOS:** archives downloaded with `curl` (as the installer does) are not quarantined. If
  you downloaded one in a browser, run
  `xattr -d com.apple.quarantine ~/.local/bin/dynalm ~/.local/bin/dynacorec`.

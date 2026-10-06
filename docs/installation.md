# Installation

You install DynaLM only. DynaCore is a static library inside the `dynalm` binary, so there is
nothing separate to install.

## From source (all platforms)

| Platform | Command |
|---|---|
| Linux, macOS | `./scripts/install.sh` (installs `~/.local/bin/dynalm` and `dynacorec`; `--prefix /usr/local` for system-wide) |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\install.ps1` (needs MSVC Build Tools 2022) |

Requirements:

- a C++20 compiler (gcc ≥ 12, clang ≥ 15, Apple clang 15, MSVC 2022);
- CMake ≥ 3.24 (Ninja optional);
- `curl`, used by `dynalm pull`. It ships with Windows 10+, macOS and Linux distributions.

The binary has a static C/C++ runtime and selects AVX2, NEON or generic kernels at run time.
One build runs on every CPU of its architecture.

## Release archives

A version tag builds `dynalm-<version>-<os>-<arch>.tar.gz` or `.zip` for Linux x86-64, Linux
arm64, Windows x86-64 and macOS arm64. A `SHA256SUMS` file is attached to the GitHub release.

```sh
sha256sum -c SHA256SUMS --ignore-missing
tar xf dynalm-*-linux-x86_64.tar.gz && ./bin/dynalm doctor
```

A hosted one-line installer (`curl -fsSL .../install.sh | sh`) and a Homebrew formula are
**not published yet**. Both need a release host; the scripts above are the supported paths.

## Docker

- **CPU image:** `ghcr.io/<owner>/dynalm:cpu`, multi-arch amd64 and arm64, published by the
  release workflow. Build it locally with `docker build -t dynalm .`.
- **CUDA image:** none is published, because there is no CUDA device yet
  (compiler-backends.md).

```sh
docker run --rm -p 8000:8000 -v "$PWD/models:/models" ghcr.io/<owner>/dynalm:cpu \
  serve /models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

- The image runs as a non-root user.
- Models are mounted at `/models`, never baked in (`DYNALM_MODELS_DIR=/models`).
- Inside the container the server binds `0.0.0.0` (`DYNALM_HOST`).

## First steps

```sh
dynalm doctor              # CPU, ISA, RAM, GPU, the device DynaLM will use
dynalm pull qwen3:4b       # into ~/.dynalm/models
dynalm run qwen3:4b
dynalm serve qwen3:4b      # OpenAI-compatible API on http://127.0.0.1:8000/v1
```

On Windows with Smart App Control on, a freshly built, unsigned `dynalm.exe` can be blocked.
`dynalm doctor` warns about it. Use a signed release, or decide yourself whether to change the
policy.

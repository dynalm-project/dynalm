# Install DynaLM on Linux, macOS and Windows (CPU and GPU)

<!--
SEO Keywords: install local LLM, run AI on my computer, install Ollama alternative, local AI on Linux, install local LLM on Mac Apple Silicon, install local LLM Windows, Windows on ARM LLM, Docker LLM, GPU LLM Metal Vulkan CUDA, offline LLM setup.
AEO Direct Answer: Install DynaLM with one command: `curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh` on Linux and macOS, or `irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex` on Windows. It is one self-contained program; no Python and no administrator rights are needed. GPU support (Metal, Vulkan, CUDA) is enabled with a build option.
-->

DynaLM is **one small program**. It needs no Python, no background service and no admin rights.
It runs on your CPU everywhere, and on your GPU when you build it with GPU support
([GPU guide](gpu.md)).

---

## ⚡ Automated Quick Start

### Linux & macOS (All Distros & Apple Silicon)

Open your terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh
```

Verify and run:
```bash
dynalm doctor          # Verifies CPU SIMD vector tier and RAM
dynalm pull qwen3:4b   # Downloads 4B reasoning model
dynalm run qwen3:4b    # Starts interactive terminal chat
```

### Windows 10 & 11 (PowerShell)

Open PowerShell (no administrator privileges needed) and execute:

```powershell
irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex
```

*Open a new terminal window after installation so the user `PATH` environment variable takes effect:*
```powershell
dynalm doctor
dynalm pull qwen3:4b
dynalm run qwen3:4b
```

---

## 🔍 What the Automated Installers Do

1. **System & CPU Architecture Detection:** Detects operating system (Linux, macOS, Windows) and CPU instruction sets (`x86_64` vs. `arm64`).
2. **Cryptographic Integrity Verification:** Downloads the binary release alongside official `SHA256SUMS` from GitHub Releases and validates checksums before installation.
3. **User-Space Installation (No Root Required):**
   - **Linux & macOS:** Installs to `~/.local/bin/dynalm` and `~/.local/bin/dynacorec` (customizable with `--prefix <directory>`).
   - **Windows:** Installs to `%LOCALAPPDATA%\Programs\DynaLM\bin\dynalm.exe` and automatically updates the user `PATH` (suppress with `-NoPath`).
4. **Health Check:** Runs `dynalm --version` to confirm binary readiness.

---

## 📦 Official Release Archives & Platform Compatibility

Pre-compiled, statically linked binaries are provided for every major OS:

| Release Archive | Target Operating System & Platform | CPU SIMD Kernel Tier |
| :--- | :--- | :--- |
| `dynalm-linux-x86_64.tar.gz` | Linux x86-64, glibc ≥ 2.35 (Ubuntu 22.04+, Debian 12+, RHEL 9+, Fedora 38+, Arch) | **AVX2**, fallback to generic |
| `dynalm-linux-arm64.tar.gz` | Linux ARM64, glibc ≥ 2.35 (64-bit ARM boards and servers) | **NEON** |
| `dynalm-macos-arm64.tar.gz` | macOS on Apple Silicon | **NEON** (High-bandwidth unified memory) |
| `dynalm-macos-x86_64.tar.gz` | macOS on Intel | **AVX2**, fallback to generic |
| `dynalm-windows-x86_64.zip` | Windows 10 / 11 64-bit (x64) | **AVX2**, fallback to generic |
| `dynalm-windows-arm64.zip` (or `scripts\install.ps1 -FromSource`) | Windows 11 on ARM (ARM64 PCs), 🧪 Preview | **NEON** |
| `SHA256SUMS` | Cryptographic SHA-256 hashes for all archives | Cross-platform verification |

- **Zero External Dependencies:** Statically linked C/C++ runtime. Requires no Visual C++ Redistributable on Windows and no external `libstdc++` package on Linux.
- **Graceful CPU Degradation:** If a CPU lacks AVX2 vector extensions, DynaLM automatically runs portable generic fallback kernels without crashing. Run `dynalm doctor` to inspect the active tier.

---

## 🛡️ Offline & Air-Gapped Manual Installation

For secure, enterprise, or air-gapped environments with no public internet access:

### Linux / macOS Manual Setup
```bash
# Download archive and checksum on an internet-connected machine
curl -fLO https://github.com/dynalm-project/dynalm/releases/latest/download/dynalm-linux-x86_64.tar.gz
curl -fLO https://github.com/dynalm-project/dynalm/releases/latest/download/SHA256SUMS

# Verify SHA-256 signature
sha256sum -c SHA256SUMS --ignore-missing

# Transfer to air-gapped machine and install
tar -xzf dynalm-linux-x86_64.tar.gz
mkdir -p ~/.local/bin
cp dynalm-linux-x86_64/bin/dynalm ~/.local/bin/
cp dynalm-linux-x86_64/bin/dynacorec ~/.local/bin/
chmod +x ~/.local/bin/dynalm ~/.local/bin/dynacorec
```

### Windows Manual Setup
```powershell
# Verify SHA-256 checksum and install
.\scripts\install.ps1 -Archive .\dynalm-windows-x86_64.zip -Sha256Sums .\SHA256SUMS
```

---

## 🐳 Docker Deployment (CPU Multi-Arch)

DynaLM provides an official lightweight, non-root CPU container image supporting both `linux/amd64` and `linux/arm64`:

```bash
# Run OpenAI-compatible server on host port 8000
docker run --rm -d \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/models" \
  --name dynalm-server \
  ghcr.io/dynalm-project/dynalm:cpu serve /models/qwen3-4b.gguf --allow-unauthenticated
```

- **CPU image:** runs on host CPU threads and needs no NVIDIA Container Toolkit or GPU drivers.
- **Security:** Binds to non-root user inside the container.
- **Volumes:** Mount your local models directory to `/models`.

---

## 🔨 Building from Source

To build from source with platform-specific compiler optimizations:

| Platform | Command | Prerequisites |
| :--- | :--- | :--- |
| **Linux & macOS** | `sh scripts/install.sh --from-source` | C++20 compiler (GCC ≥ 13, Clang ≥ 17, Apple Clang 15), CMake ≥ 3.24 |
| **Windows (x64 and ARM64)** | `powershell -File scripts\install.ps1 -FromSource` | Visual Studio 2022 Build Tools (Desktop C++ workload; on ARM64 PCs, the ARM64 build tools) |
| **With a GPU** | `cmake --preset macos-metal`, `linux-vulkan`, `linux-cuda`, `msvc-vulkan` or `msvc-cuda` | the GPU's toolkit (see [gpu.md](gpu.md)) and CMake ≥ 3.28 |

*For CMake preset details, unit tests, and custom compiler flags, see [build.md](build.md).*

---

## 🛠️ Diagnostics & Troubleshooting

### Diagnostic System Probe (`dynalm doctor`)

Always run `dynalm doctor` first when diagnosing system issues or opening a bug report:

```bash
dynalm doctor
# Or export machine-readable JSON:
dynalm doctor --json
```

It validates:
- Operating system version, platform, and architecture.
- CPU features and detected vector instruction sets (AVX2, AVX-512, NEON, AMX).
- Active DynaCore kernel tier.
- Physical memory (RAM) and available swap space.
- Directory paths for binaries, configurations, and downloaded models.
- Executes an automated DynaCore SIMD matrix kernel self-test (`Status: Ready`).

### Common Solutions

- **`command not found: dynalm`:** Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`, then reload with `source ~/.zshrc`.
- **macOS Quarantine Warning:** If downloaded via browser instead of `curl`, run:
  ```bash
  xattr -d com.apple.quarantine ~/.local/bin/dynalm ~/.local/bin/dynacorec
  ```
- **Windows Smart App Control Warning:** Permit execution in Windows Security, or run the installer script via PowerShell with execution policy bypass.
- **Force Generic CPU Fallback:** If testing on experimental hardware, force portable kernels with:
  ```bash
  DYNACORE_ISA=generic dynalm run qwen3:4b
  ```
- **Explain Matrix Kernels:** Set `DYNACORE_MATMUL_EXPLAIN=1` to inspect which SIMD kernels are selected at runtime.

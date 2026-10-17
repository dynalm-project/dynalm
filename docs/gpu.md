# Run Local AI Models on Your GPU with DynaLM: Metal, Vulkan and CUDA

<!--
SEO keywords: run LLM on GPU, local LLM GPU, Apple Silicon Metal LLM, Mac LLM GPU, AMD GPU LLM, Intel GPU LLM, Vulkan LLM, NVIDIA CUDA LLM, Ollama GPU alternative, llama.cpp GPU alternative, offline AI GPU.
AEO answer: DynaLM runs AI models on Apple Silicon GPUs (Metal), AMD, Intel, NVIDIA and Qualcomm GPUs (Vulkan), and NVIDIA GPUs (CUDA). The normal installer includes GPU support (Metal on Apple Silicon Macs, Vulkan on Linux and Windows PCs with graphics drivers), and DynaLM uses the GPU first, falling back to the CPU when needed. Switch with --cpu and --gpu.
-->

**Short answer:** install DynaLM normally and run it. You get the GPU version when your
computer can use it, and DynaLM uses the GPU first. Add `--cpu` to use the CPU, and `--gpu`
to switch back.

---

## Which GPUs work?

| Your GPU | Use | Systems | Status |
|---|---|---|---|
| Apple Silicon | **Metal** | macOS | 🧪 Preview |
| AMD | **Vulkan** | Windows, Linux | 🧪 Preview |
| Intel | **Vulkan** | Windows, Linux | 🧪 Preview |
| Qualcomm | **Vulkan** | Windows on ARM, Linux | 🧪 Preview |
| NVIDIA | **CUDA** or **Vulkan** | Windows, Linux | 🧪 Preview |

🧪 **Preview** means the backend works and passes DynaLM's test suite, and it is still being
tuned. Please report what you see on your hardware in
[GitHub Issues](https://github.com/dynalm-project/dynalm/issues).

---

## Step 1: install

```bash
curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh   # Linux, macOS
irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex       # Windows
```

| Your computer | You get |
|---|---|
| Mac with Apple Silicon | GPU version (Metal), always |
| Linux x86-64 or Windows x64 with a graphics driver | GPU version (Vulkan) |
| Anything else | CPU version |

**Choose yourself:**
- Linux / macOS: `curl -fsSL .../install.sh | sh -s -- --cpu`, or `-- --gpu`.
- Windows: `& ([scriptblock]::Create((irm .../install.ps1))) -Cpu`, or `-Gpu`.

---

## Step 2: run

```bash
dynalm run qwen2.5:1.5b          # GPU first (default)
dynalm run qwen2.5:1.5b --cpu    # use the CPU
dynalm run qwen2.5:1.5b --gpu    # use the GPU again (stops with a clear message if it can't)
```

The same flags work with `dynalm serve` and `dynalm benchmark`. The log shows the device:

```
Device: compiled/Metal/<your GPU>     <- GPU
Device: CPU/neon                      <- CPU
```

**Check what your copy supports:** `dynalm --version` lists `devices: cpu metal` (or
`cpu vulkan`).

---

## Build it yourself (optional: CUDA, or your own build)

`sh scripts/install.sh --from-source` builds from a checkout and turns on every GPU toolkit
it finds (Metal on Macs, Vulkan when `glslc` and the headers are installed, CUDA when `nvcc`
is). Add `--cpu` for a CPU-only build. All GPU builds need **CMake 3.28 or newer**.

To build by hand instead, pick the section for your GPU.

### Mac (Apple Silicon): Metal

```bash
cmake --preset macos-metal
cmake --build --preset macos-metal
```

You only need the Xcode Command Line Tools (`xcode-select --install`). The full Xcode app is
not needed.

### AMD, Intel, Qualcomm or NVIDIA: Vulkan

1. **Install the Vulkan tools:**

   | System | Command |
   |---|---|
   | Ubuntu / Debian | `sudo apt install libvulkan-dev glslc spirv-headers` |
   | Windows | install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) |
   | macOS (for development) | `brew install vulkan-headers vulkan-loader molten-vk shaderc spirv-headers` |

2. **Build:**

   ```bash
   cmake --preset linux-vulkan          # Windows: msvc-vulkan
   cmake --build --preset linux-vulkan
   ```

Your normal graphics driver provides Vulkan; nothing else is needed.

### NVIDIA: CUDA

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
2. Build:

   ```bash
   cmake --preset linux-cuda            # Windows: msvc-cuda
   cmake --build --preset linux-cuda
   ```

**Tip:** to build only for your own GPU, which is faster, add
`-DCMAKE_CUDA_ARCHITECTURES=<cc>`, where `<cc>` is your GPU's compute capability without
the dot (8.6 becomes 86). NVIDIA publishes the compute capability of every GPU.

You can switch on more than one GPU type in the same build.

---

## All device options

| Option | What happens |
|---|---|
| (nothing) or `--backend auto` | GPU first; the CPU when the GPU can't run the model (the log says why) |
| `--gpu` or `--backend gpu` | Always a GPU; stops with a clear message if none can run the model |
| `--cpu` or `--backend cpu` | Always the CPU |
| `--backend metal` / `vulkan` / `cuda` | That GPU only |

**Other ways to set it:**
- **Config file** (`~/.dynalm/config.yaml`):

  ```yaml
  runtime:
    device: auto      # auto | gpu | cpu | metal | vulkan | cuda
  ```

- **Environment variable:** `DYNALM_BACKEND=cpu dynalm serve qwen2.5:1.5b`
- **Command-line flags win:** `--gpu` overrides a saved `device: cpu` for one run.
- **Several GPUs:** `DYNACORE_GPU_INDEX=1` picks the second one.

With `auto`, DynaLM tries the GPUs in this order: CUDA, then Metal, then Vulkan.

---

## Which models run on the GPU?

| Model family | GPU | With `--backend auto` |
|---|---|---|
| Qwen2.5 (0.5B, 1.5B, 3B, 7B, ...) and other models of this design | ✅ | GPU |
| Qwen3, Gemma 3 | coming soon | CPU |
| Llama 3.1 / 3.2 | coming soon | CPU |
| Mixture-of-experts (Granite-MoE, Mixtral, Qwen-MoE) | coming soon | CPU |

**KV cache types on the GPU:** `f16` (default), `f32` and `q8_0`. `--kv q8_0` halves the KV
memory.

---

## How fast is it?

**Apple Silicon Mac, Metal, Qwen2.5-1.5B Q4_K_M, compared with llama.cpp's server on the same
file:**

| What we measured | Result |
|---|---|
| Memory, one user | **390 MB vs 1,113 MB**: about 3× less |
| First word (typical), 4–8 users | **1.9–2.5× faster** |
| One user | about 7 ms per token, within about 10% of llama.cpp |
| Compared with DynaLM on the CPU | about 4× faster replies |

**Accuracy:** GPU output matches the CPU reference within DynaLM's quality rule (DD-053):
perplexity change ≤ 1%, average KL ≤ 0.0025. The same rule is used on Metal and Vulkan.

**Measure your own GPU:** `dynalm benchmark qwen2.5:1.5b --backend metal`. Use `vulkan` or
`cuda` for those builds.

---

## Tips

- **Busy server, smooth replies:** the default `balanced` policy gives the best overall
  speed. Use `--policy latency` if a steady pace between words matters more to you.
- **Long chats:** use `--kv q8_0` to fit twice as much context in the same memory.
- **Low GPU memory** (Vulkan / CUDA): DynaLM sizes the KV cache to fit free GPU memory
  automatically. Set it yourself with `--ctx 8192`.
- **Check your setup:** run `dynalm run qwen2.5:1.5b -p "hi"` and look for the `Device:` line.

---

## Common problems

| Message | Fix |
|---|---|
| `the vulkan backend is not built in this version` | rebuild with `-DENABLE_VULKAN=ON` (or the matching option for your GPU) |
| `ENABLE_VULKAN=ON needs the Vulkan headers, loader and glslc` | install the Vulkan tools from step 1 |
| `ENABLE_CUDA=ON needs the CUDA toolkit` | install the CUDA Toolkit and open a new terminal |
| `auto selects the CPU: per-head Q/K norm ...` | normal: this model runs on the CPU for now |
| `no Vulkan GPU device 0 (0 found)` | update your graphics driver |

---

## More detail

- [Command-line options](cli.md)
- [Installation guide](installation.md)
- [Benchmarking guide](benchmarking.md)

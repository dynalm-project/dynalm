# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this is

DynaLM is a local LLM inference engine in C++20: one native program that runs GGUF and SafeTensors
models on the CPU or a GPU, with an OpenAI-compatible server. Apache-2.0.

It has two layers:

| Layer | Directory | Role |
|---|---|---|
| **DynaCore** | `dynacore/` | Tensors, devices, kernels, the IR compiler (`dynacorec`) |
| **DynaLM** | `dynalm/` | Loaders, tokenizer, transformer, scheduler, server, CLI (`dynalm`) |

**Execution paths:**
- **CPU, two kernel families:**
  - **DynaCore's own kernels (`qmm`):**
    - tiers `reference` / `neon_dot` / `neon_i8mm`, plus SME;
    - the default on SME CPUs (Apple M4 class; `native_cpu_default()`);
    - forced with `--mode exact`.
  - **ggml's CPU kernels** (pinned ggml): the default on every other CPU.
- **GPU:** Metal, Vulkan and CUDA through ggml.

## Layout

| Path | Contents |
|---|---|
| `dynacore/include`, `dynacore/src`, `dynacore/cpu` | Device API (`device/device.h`), CPU device, qmm kernels, ggml devices, IR |
| `dynalm/src/loader` | GGUF (split models too) and SafeTensors loaders, `model_loader.cpp` |
| `dynalm/src/model` | `transformer.cpp` (one generic decoder), `expert_cache.cpp` (MoE experts from disk) |
| `dynalm/src/runtime` | `engine.cpp` (device choice, memory checks), speculative decoding, `eagle3.cpp` |
| `dynalm/src/scheduler` | Continuous batching, prefix cache use, speculative drafts, preemption |
| `dynalm/src/prefix_cache` | Hash and radix prefix caches; `disk_prefix_cache.cpp` (`--prompt-cache`) |
| `dynalm/src/cli` | Commands: `run`, `chat`, `serve`, `pull`, `benchmark`, `cpu-report`, ... |
| `dynalm/src/bench` | Load generator, perplexity scorer, cpu-report plan and report |
| `dynalm/tests` | `test_engine.cpp` (most tests), `tiny_gguf.h` (synthetic GGUF models) |
| `tools/` | Python helpers and their offline tests (`tools/test_*.py`); `tools/bench/*` benchmark scripts |
| `docs/` | User docs; `docs/perf/` for measurements and design notes |

## Build and test

```sh
cmake --preset macos-metal          # macos-release, linux-release, msvc-release, ...
cmake --build build/macos-metal -j 10
bash tools/ci/run_tests.sh build/macos-metal   # every model-free test program (must all pass)
build/macos-metal/bin/test_engine              # the main engine tests alone
```

- `tools/ci/run_tests.sh` lists the C++ test programs and the `tools/test_*.py` scripts. Register a
  new Python test there.
- Tests use synthetic models from `dynalm/tests/tiny_gguf.h`, never downloads. The fixtures cover:
  - `TinyModelSpec`: dense qwen2 / qwen3;
  - `experts > 0`: qwen3moe;
  - `GgufWriter::split()`: split files.
- Before you report a test result, run the test and read its output.
- Check flaky behaviour by running a test several times in a loop.

## Rules of the codebase

**Errors and failures:**
- **Errors:** return `Status` / `Result<T>` (`ENGINE_RETURN_IF_ERROR`, `ENGINE_ASSIGN_OR_RETURN`).
  No exceptions in hot paths.
- **No fake implementations:** an unsupported case fails with a clear error. It never produces
  wrong output silently.

**Accuracy:**
- **DD-053 accuracy contract:** perplexity +1% at most and mean KL 0.0025 at most, measured against
  DynaLM's own fp32 reference.
- **Bit-identical features:** features that only move where data lives must give bit-identical
  outputs, and must have a test proving it. This covers the expert cache, split files, prompt
  cache on disk, KV spill, and speculative decoding with greedy sampling.
- **Batch-dependent rounding:** int8 activations make outputs depend on batching. Set
  `int8_decode_rows = 0` in tests that compare runs batched differently (DD-031).

**Performance claims:**
- Every speed claim needs before/after numbers from `dynalm benchmark` or `benchmarks/`.
- Measurements on battery or a busy machine are labelled uncontrolled.
- Never present an estimate as a measurement.

**Code and prose style:**
- **Style:** `.clang-format` (Google-based, 100 columns).
- **Comments:** explain why, in the surrounding density. Keep existing comments.
- **Docs, commit messages and PR descriptions:** plain, precise English, with measured facts only.
  Add a CHANGELOG `Unreleased` entry for user-visible changes.

## Release process

Releases happen only when the maintainer asks.

1. Open a PR that sets the version in `CMakeLists.txt` and `CITATION.cff`, adds the CHANGELOG
   section and `.github/release-notes/vX.Y.Z.md`. CI must be green.
2. Merge, then push an annotated tag `vX.Y.Z` (`tools/ci/version.py --check-tag` must pass).
3. `release.yml` builds a draft release.
4. Verify the draft:
   - the assets and `SHA256SUMS`;
   - `gh attestation verify`;
   - the binary's `--version`.
5. Publish the release.
6. Update the website repo (`dynalm-project/dynalm-website`) and check it with `npm run check`.

## Recent features (`feat/moe-ssd-experts`)

Plan, status and first measurements: `docs/perf/moe-ssd-plan.md`.

| Feature | Option | Where |
|---|---|---|
| MoE experts read from the model file into a RAM cache | `--experts-from-disk`, `--expert-cache SIZE` | `model/expert_cache.cpp`, `Transformer::moe_mlp`, `Engine::create` |
| Next-layer expert prediction | `DYNALM_EXPERT_PREFETCH=0` turns it off | `Transformer::moe_mlp` |
| Draft length sized by measured cost | on with `--experts-from-disk` | `choose_draft_length` (scheduler) |
| EAGLE-3 draft heads | `--spec-eagle DIR` | `runtime/eagle3.cpp`, scheduler `eagle_rows` / `draft_with_eagle` |
| Prompt KV cache on disk, spill on preemption | `--prompt-cache DIR`, `--prompt-cache-size` | `prefix_cache/disk_prefix_cache.cpp`, `Scheduler::preempt_one` |
| Split GGUF loading | give the first part | `gguf::open_parts`, `load_gguf` |
| MoE benchmark script | `tools/bench/moe_ssd_bench.py` | 30B / Mixtral / 235B (Q4_K_M) / DeepSeek-V3 |

**Limits:**
- The expert cache needs host routing, so it runs on DynaCore's CPU device only (not ggml).
- The prompt cache and EAGLE-3 need host-memory KV: the CPU only.
- DeepSeek-V3 / Kimi (`deepseek2`, MLA attention) are not supported.
- Compressing experts on disk was rejected: quantized weights compress by only about 2%.

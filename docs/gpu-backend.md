# DynaLM GPU Backend: Design for Metal First, CUDA Next

How DynaLM can run inference on a GPU without giving up what makes it different (continuous
batching, paged KV, prefix sharing): the layers, contracts, risks, and the smallest prototype
that proves or disproves the design.

Status: **experimental, implemented and validated on Apple M4 Pro** (section 0): in-place
paged attention, zero-copy weights, a GPU prefill policy, concurrency and prefix sharing
through DynaLM's own scheduler. It is opt-in at build time (`-DENABLE_METAL=ON`) and at run
time (`--backend metal`); CPU defaults are unchanged. Release status:
[release-readiness.md](release-readiness.md). Sections 1–13 are the original design. Where the implementation differs, section 0
says so and wins. Base: the 0.1.4 release candidate (`44a9851`), branch `feat/gpu-backend-design`.

Every statement is labelled:

| Label | Meaning |
|---|---|
| **[code]** | Verified by reading the source at `44a9851` (file and function given) |
| **[verified]** | Built, run or measured on the M4 Pro during stage 1 (section 0) |
| **[history]** | A measurement recorded in the repository (CHANGELOG, docs); not repeated on the M4 Pro |
| **[proposal]** | A design decision of this document; not implemented |
| **[assumption]** | Believed true but not verified here: ggml API details, Metal behaviour, hardware figures. Each one must be verified before the work that depends on it |

## 0. Implementation status (stages 1 and 2)

### Final decisions

| ID | Decision | Status |
|---|---|---|
| D1 | Standalone upstream ggml, tag `v0.27.0`, commit `a05c9922613c6a2230dbb1d0de679519ecafb4f0` (MIT). Fetched by `cmake/Ggml.cmake` with FetchContent at the full commit id (content-addressed, so it cannot drift); offline: `-DDYNALM_GGML_SOURCE_DIR=<checkout>` (version-checked) | **[verified]** fetched tree is at the pinned commit |
| D2 | Metal shaders embedded as source, compiled by the Metal framework at device creation (`GGML_METAL_EMBED_LIBRARY`). Works with the Command Line Tools only; full Xcode is not required | **[verified]** build and run without `xcrun metal` |
| D3 | Qwen2.5-1.5B-Instruct Q4_K_M, official `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (Apache-2.0, not gated), revision `91cad51`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes, SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` (matches the upstream etag) | **[verified]** |
| D4 | Stage 1 keeps DynaLM's **real paged KV pool** (16-token blocks, block tables, copy-on-write untouched). `kv_store` writes rows in place with `ggml_set_rows`; attention **gathers** each sequence's K/V rows (`ggml_get_rows`) into a contiguous span padded to 256 positions and masked, then runs `ggml_flash_attn_ext`. This is option A of section 7: correct for any block table, but it reads the KV about three times. It is temporary. It is not GPU-native paged attention | **[verified]** correct; traffic cost not yet measured at long contexts |
| D5 | `--backend metal` that cannot initialize, or cannot run the model, is an error at startup (never a silent CPU run). `--backend auto` (the CLI default) selects the GPU in a Metal build when `gpu_model_support` accepts the model and the device starts; otherwise it selects the CPU and logs the reason. CPU-only builds are unchanged | **[verified]** `test_engine` device-selection test (tiny Qwen2 → GPU, tiny Qwen3 → CPU, explicit metal + Qwen3 → Unsupported) |
| D6 | The DD-053 contract against the CPU fp32 reference (perplexity ≤ +1%, mean KL ≤ 0.0025 nats), plus GPU run-to-run determinism and a greedy-divergence report. Greedy token identity is reported, not gated: the CPU contract does not require it (DD-031 makes no batch-invariance promise for accelerated paths) | **[verified]** all pass, section 0 results |
| D7 | Work on `feat/gpu-backend-design` from the 0.1.4 release candidate | done; nothing committed |
| D8 | **More GPU providers, one device implementation.** `GgmlDevice` serves Metal, Vulkan and CUDA (`ENABLE_METAL` / `ENABLE_VULKAN` / `ENABLE_CUDA`), each chosen by its ggml registry; `DYNACORE_GPU_INDEX` picks a device. Providers other than Metal do not give each buffer a unique address (ggml-vulkan uses one placeholder for all), so the runtime gets a reserved no-access address range per buffer and the backend address is used only to bind ggml tensors; their weights are copied, not mapped. Matmul and attention request fp32 precision (`GGML_PREC_F32`): Vulkan's default fp16 accumulation failed DD-053 on prefill (mean KL 0.0056), and passes with it (0.000004). Auto KV capacity also respects free device memory on these providers | **[verified]** Vulkan on this Mac through MoltenVK: `test_ggml_device` 283,567 checks, `test_engine`, `bench_gpu_accuracy` PASS (decode +0.015%, prefill −0.010%, 4/4 greedy, deterministic), API smoke 18/18. **CUDA:** same device layer; CI builds it; hardware validation on NVIDIA GPUs is the next step |

### What was built [verified]

| Piece | Where |
|---|---|
| Public recorded call + segment interface: `DeviceCall` (was the private `RecordingDevice::Call`), `SegmentExecutor`, `SegmentStats` | `dynacore/include/dynacore/device/segment.h` |
| `Device` hooks with no-op defaults (CPU unchanged): `segment_executor()`, `take_error()` (errors of void/deferred ops), `memory_stats()` | `dynacore/include/dynacore/device/device.h` |
| `RecordingDevice::flush()` hands the whole recorded step to a segment-executing inner device (one call per step), keeps its error for `take_error()`; the CPU path (planner, plan cache, fusions) is untouched | `dynacore/src/ir/recording_device.cpp` |
| `GgmlDevice`: backend buffers for `allocate`/`upload`, byte transfers, segment → one ggml graph, prepared-graph cache, dynamic inputs, support check of every node (`ggml_backend_supports_op`) | `dynacore/include/dynacore/gpu/ggml_device.h`, `dynacore/src/gpu/ggml_device.cpp` |
| `create_device(kMetal)`, `compiled_devices()` | `dynacore/src/device/device_registry.cpp` |
| Engine: a segment-executing device forces compiled execution (logged); device errors after init; Transformer returns the device's step error after its sync point | `dynalm/src/runtime/engine.cpp`, `dynalm/src/model/transformer.cpp` |
| `--backend cpu\|metal` on `run`, `serve` (existing parsing) and `benchmark` (new); `runtime.device` text | `dynalm/src/cli/*.cpp`, `dynalm/src/config/config.cpp` |
| Build: `ENABLE_METAL` (no longer fatal), `cmake/Ggml.cmake`, preset `macos-metal` | `CMakeLists.txt`, `cmake/Ggml.cmake`, `dynacore/CMakeLists.txt`, `CMakePresets.json` |
| Tests: `test_ggml_device` (model-free, in `tools/ci/run_tests.sh`; skips without a GPU backend), `bench_gpu_accuracy` (model-based) | `dynacore/tests/`, `dynalm/benchmarks/` |

**How a step runs.**
1. The Transformer is unchanged. Its Device calls are recorded.
2. At the step's sync point (the logits `download`), `GgmlDevice::run_segment` builds or reuses one graph.
3. The graph key is the recorder's structural signature, plus each attention call's per-sequence row split and padded span.
4. Values flow through ggml nodes, keyed by exact DynaLM view.
5. Column and row sub-views of values written earlier are derived as ggml views.
6. A read of memory written through a different view that cannot be derived is refused as aliasing.
7. Every view written in the step is copied back to its buffer at the end, so a later step that reads memory sees it.
8. Token ids, positions, KV row indices, gather indices and masks are uploaded each step into per-graph input tensors, shared by calls with identical source data. That sharing is re-checked on every run.
9. Freeing any device buffer drops all prepared graphs: a graph binds buffers, and a new buffer can reuse a freed address.

**Supported on the GPU:**
- embedding: f16, bf16 and every GGUF type DynaLM loads, via `get_rows`;
- matmul and matmul_many with bias: same types, via `mul_mat`;
- RMSNorm, LayerNorm;
- RoPE: half-split or interleaved, partial dims, frequency factors, linear scaling;
- kv_store and attention: f16 or f32 KV, GQA, sliding window, softcap;
- act_mul and activation (SiLU, GELU-erf, GELU-tanh), add, scale, softcap, fill, gather_rows.

**Refused with a named error:**
- `scatter_add_rows`, which means MoE models;
- RoPE YaRN or Llama 3 scaling;
- q8_0 KV;
- rank-3 weights;
- multi-pool KV views;
- non-contiguous rows of one sequence in a batch;
- aliasing reads;
- direct (reference) op calls.

There is no CPU fallback inside a step.

### Environment facts [verified]

- Apple M4 Pro (8P + 4E), 24 GiB, macOS 26.7, Command Line Tools 26.6.
- CMake 4.4.4 and Ninja 1.13.2 were installed with Homebrew for this work.
- **SDK mismatch:** `xcrun` selects a leftover `MacOSX27.0.sdk` whose `.tbd` files the installed linker cannot read ("unknown architecture arm64e.x1-macos"). Every build here therefore sets `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`. Fixing the toolchain installation (reinstalling the Command Line Tools) is an alternative; it was not done.
- The ggml CPU backend is built as part of ggml but not used (`GGML_CPU` default). OpenMP is not found, and ggml warns but continues.

### Build and validation commands

```sh
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk   # this machine only
cmake --preset macos-metal && cmake --build --preset macos-metal         # fetches pinned ggml
bash tools/ci/run_tests.sh build/macos-metal                             # model-free, incl. test_ggml_device
build/macos-metal/bin/dynalm run qwen2.5:1.5b -p "..." --temp 0 --backend metal
build/macos-metal/dynalm/benchmarks/bench_gpu_accuracy ~/.dynalm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
DYNALM_BACKEND=metal python3 tools/ci/smoke_api.py build/macos-metal/bin/dynalm <model.gguf>
cmake --preset macos-release && cmake --build --preset macos-release     # CPU-only build, unchanged
```

### Gate A results: correctness [verified]

- **Model-free tests:** `tools/ci/run_tests.sh` passes 14 of 14 programs in both the CPU-only (`macos-release`) and Metal (`macos-metal`) builds. `test_ggml_device` runs 125,301 checks:
  - every lowered op against `CpuDevice` (f32, f16, q8_0, q4_K and q6_K weights at M = 1, 5 and 33, strided inputs);
  - paged attention: 3 sequences with scattered block tables, f16 and f32 KV, window and softcap, prefill then decode with graph reuse;
  - failure paths: direct calls, scatter_add, aliasing, q8_0 KV, a 64 TiB allocation returning OutOfMemory; the device stays usable afterwards;
  - buffer reuse after free.
- **CPU path:** the same greedy output as before the change, byte for byte. `dynalm doctor` is OK, `check_isa.py` passes, the core-only build passes, and `--backend metal` on a CPU-only build exits 1 with a clear message.
- **Model accuracy:** `bench_gpu_accuracy`, Qwen2.5-1.5B Q4_K_M, GPU against the CPU fp32 reference:

| Comparison | Perplexity change | Mean / max KL (nats) | Top-1 agreement | Max \|Δlogit\| | DD-053 |
|---|---|---|---|---|---|
| CPU noise floor: fp32 fused vs fp32 expand (decode) | +0.012% | 0.000000 / 0.000003 | 100% | 0.011 | — |
| GPU, decode (M = 1, 189 positions) | +0.011% | 0.000003 / 0.000018 | 100% | 0.045 | PASS |
| GPU, prefill chunks of 32 | +0.150% | 0.000179 / 0.000985 | 98.94% | 0.339 | PASS |
| GPU, prefill chunks of 64 | +0.150% | 0.000179 / 0.000985 | 98.94% | 0.339 | PASS |

  - **Greedy:** 4 of 4 chat-templated prompts are identical to the CPU for 64 tokens.
  - **Determinism:** the GPU decode pass run twice gives bit-identical logits.
  - **Two sequences in one batch** (different lengths, positions and blocks) give the same logits as each run alone (max diff 0.0).
  - **Graph reuse:** 664 segments, 34 graphs built, 0 errors.
  - **Why prefill differs more than decode:** prefill rows go through ggml's Metal matrix-matrix kernels, which round activations to half precision inside the tile [assumption from the kernel design, consistent with the measured gap]. That is a different precision class from DynaLM's fp32 activations. It is within DD-053 here, and it must be re-checked per model.
- **CLI and API:** `dynalm run --backend metal` loads, prefills, decodes, samples, streams and shuts down. `tools/ci/smoke_api.py` passes 18 of 18 with `DYNALM_BACKEND=metal`; both of its server logs report `Device: compiled/Metal/Apple M4 Pro`. That run includes 4 concurrent users decoding together, which completed correctly but was not measured.
- **Found and fixed during validation:**
  - Prepared graphs outlived a freed KV pool, and a new pool at the same address ran through the stale graph. This made 3 of 4 greedy prompts diverge at token 0. Fixed with the free epoch.
  - A false aliasing report on Q/K/V column slices. Fixed with an exact strided overlap test.
  - ggml-metal aborts (`GGML_ASSERT rsets`) if the process exits with a live device. The engine destroys its device before exit; tests and benchmarks must too.

### Gate B results: single-user benchmark against llama.cpp Metal [verified, with caveats]

> **Superseded.** This table measures the **stage-1** build (gather attention, copied weights,
> CPU prefill chunking) on **battery**. The current measurements on AC power are in
> "Release measurements" at the end of this section. The table is kept as the stage-1 record.

**Setup.**
- Both engines run as HTTP servers on the same GGUF file (SHA-256 above).
- llama.cpp `9a16f1dbe4722a93ab75a2d15e29aa71ef764c7b` (llama-server, `-ngl 99 -c 4096 -np 1 -fa auto`), built from source with Metal. It vendors the **same ggml 0.27.0**, so both engines use the same Metal kernels.
- llama-bench confirms backend `MTL` with all layers offloaded.
- DynaLM: `serve --backend metal --ctx 4096`, default scheduler.
- **Same client for both:** `dynalm benchmark --url` at concurrency 1:
  - tokenizer-exact unique prompts of 128, 512 and 2,048 tokens;
  - 128 output tokens with `ignore_eos`, greedy;
  - 1 warm-up and 4 measured requests per point.
- **Rounds:** 3 paired rounds with the arm order rotated. Every request produced exactly the requested prompt and output token counts on both engines.
- **Machine:** M4 Pro **on battery** (22% → 15% during the runs; it was not possible to plug in). Absolute numbers may be lower than on AC; both arms ran under the same conditions.
- **Verdict rule:**
  - "better" or "slower" only when all 3 rounds agree beyond ±5%;
  - INCONCLUSIVE when the rounds disagree or an arm varied more than 10% across rounds.

Medians over 3 rounds. The ratio is DynaLM's advantage (> 1 = DynaLM better), per round:

| Prompt | Metric | DynaLM | llama.cpp | Ratio per round | Verdict |
|---|---|---|---|---|---|
| 128 | TTFT p50 | 172 ms | 132 ms | 0.74 0.79 0.76 | slower |
| 128 | ITL p50 / p99 | 10.6 / 12.4 ms | 12.7 / 21.4 ms | 1.17–1.21 / 1.66–1.91 | DynaLM better |
| 128 | output tok/s | 84.3 | 69.9 | 1.19 1.21 1.20 | DynaLM better |
| 512 | TTFT p50 | 625 ms | 414 ms | 0.68 0.67 0.65 | slower |
| 512 | ITL p50 / p99 | 11.3 / 13.9 ms | 12.1 / 25.3 ms | 1.03–1.07 / 1.72–1.88 | parity / DynaLM better |
| 512 | output tok/s | 61.1 | 59.1 | 1.02 1.02 1.03 | parity |
| 2048 | TTFT p50 | 2,588 ms | 1,684 ms | 0.65 0.65 0.65 | slower |
| 2048 | ITL p50 | 15.4 ms | 15.2 ms | 0.88 1.03 0.97 | INCONCLUSIVE (llama.cpp spread 16%) |
| 2048 | TPOT p50 | 15.2 ms | 14.4 ms | 0.92 0.94 0.91 | slower |
| 2048 | output tok/s | 28.2 | 36.2 | 0.77 0.78 0.77 | slower (dominated by TTFT) |

Other measurements:

| Measurement | DynaLM Metal | llama.cpp Metal |
|---|---|---|
| Peak memory footprint (`/usr/bin/time -l`) | 1,250 MiB | 644 MiB |
| Max RSS | 2,290 MiB | 1,713 MiB |
| Model load (DynaLM log; weight copy to Metal buffers) | 240 ms warm (5.6 s on the first run of the binary, Metal shader compile) | not measured separately |

**Diagnostic runs (one round each; not part of the paired verdicts):**
- **DynaLM Metal with `--solo-prefill-chunk 512 --batch 1024`** (non-default, existing 0.1.4 option): TTFT 117 / 469 / 1,767 ms at 128 / 512 / 2,048 tokens. Same decode.
- **DynaLM CPU (NEON):** TTFT 1,475 / 6,188 / 28,921 ms; ITL p50 39–46 ms. Metal is about 3.7× faster at decode and about 11–16× faster at prefill than DynaLM's own CPU path.
- **llama-bench** on the same machine: pp512 1,234 ± 203 tok/s, tg128 73.9 ± 1.1 tok/s.

**Single-user parity is not reached.** Mixed result:
- **Decode on short and medium contexts: DynaLM is ahead.** ITL p50 is 3–20% lower, with much tighter tails (p99 1.7–1.9× lower) and +20% output tok/s at 128 tokens. The cause is not established. Both engines run the same ggml kernels, so the difference is in the engine around them (graph reuse, host work per token, server/streaming path). This needs per-step profiling before it is relied on, and it must be repeated on AC power.
- **Prefill / TTFT: DynaLM is 21–35% slower.** The diagnostic run points mostly at DynaLM's scheduler defaults, not the GPU path: 32-token prompt chunks (tuned for CPU, DD-027/DD-079) mean 64 graph runs for a 2,048-token prompt, against llama.cpp's 512-token ubatch. With 512-token solo chunks, DynaLM's TTFT is within 5–13% of llama.cpp, and better at 128 tokens (one round). The remaining gap is attributed, unverified, to per-step scratch write-back and the per-step K/V gather.
- **Long context decode (2,048): DynaLM is 6–9% slower per token.** This is consistent with option A's gather, which re-reads the whole K/V span every step and grows with context. It is the first measured sign of the stage-2 risk in section 7.
- **Memory: DynaLM uses about 2× llama.cpp's peak footprint**, because the weights are copied into Metal buffers while the GGUF mmap stays resident (section 0, "Not solved").
- **Not measured:**
  - CPU and GPU utilization: `powermetrics` needs root; not run;
  - per-step host↔device transfer bytes on the real model: `SegmentStats` has them but the server does not export them yet.

**Raw data:** `scratchpad/bench/run2/` of the session that produced this document: manifest with executable/model/llama.cpp hashes, per-request logs, server logs, client tables. These are temporary files and are not committed. Reproduce with the commands above and the session's `sweep.sh` / `analyze.py` logic (both engines as servers, `dynalm benchmark --url`, rotated paired rounds).

**Next optimizations, in measured order of impact (proposals):**
1. **GPU-aware prefill chunking.** Make the solo-prefill chunk the default for GPU devices. Measure paired, with the TTFT and ITL tail gates of `docs/benchmarking.md`.
2. **Zero-copy weights.** Measure `ggml_backend_dev_buffer_from_host_ptr` on the mmap against private copies; this halves the footprint if it holds.
3. **Avoid scratch write-back** for values that no later segment reads.
4. **Replace the gather (stage 2, section 7).**

### Stage 2 implemented: in-place paged attention, zero-copy weights, GPU prefill [verified]

**In-place paged attention** (replaces the stage-1 gather; option C of section 7, without a
custom kernel or a ggml patch):

- **Layout:** the GPU KV pool uses `KvLayout::kSlotMajor` (`dynacore/include/dynacore/attention/paged_kv.h`),
  `[block][slot][kv_head][dim]`, chosen by the Engine from `Device::preferred_kv_layout()`.
  - Every physical slot ("cell") is one row of all KV heads, so a layer's pool is one strided
    `[dim, cells, kv_heads]` view.
  - Blocks, block tables, refcounts, prefix sharing and copy-on-write are layout-independent
    and unchanged.
  - The CPU keeps head-major. `KvBlockPool::create` refuses a layout the device cannot run.
- **`kv_store`:** `ggml_set_rows` into the pool, one row per token (index = block × 16 + slot).
- **Attention:** one `ggml_flash_attn_ext` per layer for **all** sequences of the step,
  reading the pool view directly (`GgmlDevice` `attention`, `InKind::kPagedMask`).
  - The mask admits, per row, exactly its own sequence's cells for positions `[lo, pos]`:
    block tables, causality and the sliding window together.
  - Keys are stored after RoPE, so attention over physical cell order is exact up to
    summation order.
  - ggml-metal's mask pre-pass (`kernel_flash_attn_ext_blk`) skips fully masked chunks.
  - **No KV is gathered or copied on this path.**
- **Supporting rules:**
  - The span covers the cells of the highest block used, padded to 256 and capped at the pool,
    so the graph is reused within a bucket.
  - Device buffers are zeroed at allocation, so masked-but-read cells are never NaN.
  - The stage-1 gather path remains selectable for A/B (`DYNALM_KV_LAYOUT=head`).
- **KV types:** f16, f32 and **q8_0** (newly supported on the GPU).
- **Validation:**
  - `test_ggml_device` compares both layouts with the CPU, using 3 sequences with scattered
    blocks, partial blocks, GQA, window and softcap, and f16/f32/q8_0 KV;
  - `bench_gpu_accuracy`: DD-053 decode +0.011% / prefill +0.150% (f16), q8_0 KV −0.45% /
    −0.02%;
  - two sequences in one call vs alone: max Δlogit 0.013;
  - prefix fork with copy-on-write vs scratch: 0.013; the original sequence is unaffected;
  - `test_engine` on the GPU: concurrency, shared prefixes, cancellation, deadline.

**Zero-copy weights:**
- Weights borrowed from the GGUF mmap are wrapped in place
  (`ggml_backend_dev_buffer_from_host_ptr`; ggml page-aligns per tensor).
- Each buffer's keep-alive holds the host tensor, hence the mapping. A unit test deletes the
  file and drops every caller reference, and the GPU still reads correct data.
- Owned (converted or dequantized) tensors are still copied. `DYNACORE_GPU_WEIGHTS=copy` copies
  everything.
- **Measured** (`/usr/bin/time -l`, `dynalm run`, Qwen2.5-1.5B Q4_K_M): peak memory footprint
  1,354 → **291 MiB**, max RSS 2,393 → 1,331 MiB (RSS includes the clean mmap pages). Greedy
  output unchanged.

**GPU prefill policy:**
- With `SchedulerConfig::solo_prefill_chunk = -1` (the default), the Engine resolves to 0 on
  the CPU (unchanged) and `kGpuSoloPrefillChunk = 256` on a GPU device.
- Solo steps carry no decode rows by construction (verified by the scheduler-trace test in
  `test_engine`). Mixed steps keep the DD-079 decode protection.
- **Evidence:** in-process paired sweep, chunks 0/32/64/128/256/512, 3 rounds, AC power, plus
  a 4-round recheck of the one noisy point:
  - 1 user: TTFT 1.56× / 1.66× / 1.64× faster at 128 / 512 / 2,048 tokens, ITL unchanged;
  - 4 users: no change.
  - 512 was slightly faster for one user, but noisier, with a +3% ITL p50 at 4 users × 2,048
    tokens, so 256 was chosen.

**GPU scheduling profile** (the `balanced` policy on a GPU device; `kGpuMixedPrefillBudget`):

- **Finding.** With the CPU-tuned mixed-step limits (64 prompt tokens per step, 32 per
  sequence, DD-079 decode protection at target 6× / minimum 32), multi-user throughput on the
  GPU was 0.46–0.65× llama.cpp's (release sweep).
  - Profiling (`DYNACORE_GPU_STATS=1`) ruled out graph rebuilds (48.7 ms total over 547 steps)
    and the attention layout (gather and in-place both about 30 ms per step).
  - Mixed steps took 82% of forward time at about 0.8 ms per prompt token. A GPU step pays the
    full weight read whatever its row count, and DD-079 kept halving the budget to 32.
- **Profile:**
  - prompt budget 256 and per-sequence chunk 128 in mixed steps;
  - DD-079 on, but target 12× and minimum 64.
  - `--policy latency` and `--policy throughput` keep their CPU meaning. Explicit
    `--prefill-budget` / `--decode-budget` / `--chunk` replace the profile. DD-079 environment
    overrides still apply.
- **Evidence:** paired sweep of 4 profiles × concurrency 4/8/16, 2 rotated rounds, AC power.
  Against the old defaults:

  | Users | Output tok/s | TTFT p50 | ITL p50 | ITL p95 / p99 |
  |---|---|---|---|---|
  | 4 | +42% | 1.79× faster | 13.0 → 13.3 ms | 74 / 78 → 120 / 138 ms |
  | 8 | +46% | 1.54× faster | 90.4 → 15.4 ms | 103 / 104 → 126 / 128 ms |
  | 16 | +21% | 10× faster | 58.5 → 27.1 ms | 74 / 86 → 173 / 175 ms |

  - **The trade-off is real:** typical latency improves a lot and the tail gets worse.
  - **Alternatives measured:** a protection-off profile raised throughput further (+50–100%)
    but pushed p95/p99 to about 250 ms. It is available through `DYNALM_DECODE_PROTECT=0` with
    explicit budgets.

### Release measurements (stage 2, AC power) [verified]

**Setup:**
- Same model, client and verdict rule as the stage-1 Gate B. Measured on **AC power**.
- Three paired rounds with the arm order rotated.
- Single user: prompts of 128, 512, 2,048 and 3,584 tokens, `--ctx 4096`.
- Multi-user: 4, 8 and 16 users, 512-token prompts, 128 output tokens, 32 requests per point.
- DynaLM ran with `--ctx 16384` in the multi-user runs.
- llama.cpp ran with `-np 16 -c 262144`, that is 16 slots of 16,384 tokens each.

**Stage 2 against stage 1** (release sweep):
- TTFT is 1.56–1.77× faster.
- At 3,584 tokens, decode ITL fell from 15.2 to 7.9 ms (1.92×).
- These come from in-place attention plus the 256-token prefill chunk.

**Against llama.cpp, one user** (final sweep):

| Prompt | Result |
|---|---|
| 128 | TTFT 0.92×; ITL p50 7.1 vs 6.4 ms, about 10% slower per token |
| 512 to 3,584 | INCONCLUSIVE in the final sweep: both engines varied 10–25% between rounds. The earlier release sweep measured parity at 3,584 on every metric |

**Against llama.cpp, several users** (GPU scheduling profile). Each cell gives the medians
over 3 rounds as DynaLM / llama.cpp, then the verdict:

| Metric | 4 users | 8 users | 16 users |
|---|---|---|---|
| Output tok/s | 155 / 164, parity (0.94×) | 163 / 198, slower (0.82×) | 161 / 244, slower (0.66×) |
| TTFT p50 | 588 / 1,135 ms, better | 590 / 1,544 ms, better | 715 / 2,499 ms, INCONCLUSIVE (llama.cpp ±36%) |
| TTFT p95 | 1,157 / 1,388 ms, better | 3,792 / 2,945 ms, slower | 10,942 / 5,483 ms, slower |
| ITL p50 | 12.6 / 12.3 ms, parity | 14.3 / 15.1 ms, parity | 25.4 / 22.1 ms, slower |
| ITL p99 | 128 / 26 ms, slower | 144 / 26 ms, slower | 153 / 1,256 ms, better |
| TPOT p50 | 21.5 / 12.3 ms, slower | 41.5 / 24.1 ms, slower | 82.9 / 46.1 ms, INCONCLUSIVE |
| E2E p50 | 3,366 / 2,826 ms, slower | 5,982 / 4,620 ms, slower | 11,352 / 8,234 ms, slower |

DynaLM starts the first request sooner, but it finishes requests later. Its TTFT tail at
8–16 users is worse, because later requests wait for earlier prompts.

**Peak memory footprint** (`/usr/bin/time -l`, 3 rounds):

| Load | DynaLM | llama.cpp | Comparable? |
|---|---|---|---|
| One user | 389–393 MiB | 1,113–1,117 MiB | Yes: both servers hold 4,096 tokens of KV |
| 16 users | 883–897 MiB | 8,841–8,899 MiB | **No**: llama.cpp reserves 16 × 16,384 tokens of KV; DynaLM shares 16,384 tokens through its paged pool |

**Where the time goes:**
- **One user.**
  - Host work is under 1% of a step: input upload 1 µs, logits download 30 µs, graph reused.
  - The graph's op mix matches llama.cpp's for Qwen2 (631 kernels per token).
  - Of the fusions, only RMS_NORM + MUL fires. MUL_MAT + ADD fusion exists only in ggml-metal's
    matrix-matrix (prefill) kernels.
  - Mapped and copied weights give the same 7.0 ms per token.
  - The remaining gap is therefore inside GPU execution. Two hypotheses are open, and neither
    is measured:
    - 339 per-tensor Metal buffers, against llama.cpp's single mapping;
    - kernel-variant selection.
- **Several users.**
  - Mixed prefill and decode steps dominate.
  - llama.cpp batches whole prompts. That gives it more throughput, and much worse tails at
    16 users.
  - **The 2–2.5× multi-user throughput target is not met.**

**Ollama 0.40.5** ran on Metal (`ollama ps`: 100% GPU, llamacpp runner) with the same GGUF and
client. **No result is comparable**, for two reasons:
- requests stopped early, because Ollama ignores `ignore_eos`;
- the prompt was re-templated from 512 to 541 tokens.

Gate statuses and evidence for every row: [release-readiness.md](release-readiness.md), Gate C.

### Vulkan and CUDA status

- **Vulkan (preview; correctness validated through MoltenVK on Apple M4 Pro).**
  - Every check passes: the device suite, the engine tests, DD-053 accuracy, and the API
    smoke test.
  - MoltenVK translates Vulkan into Metal, so its speed (20.8 ms per token, against about
    8 ms on Metal) does not predict speed on discrete GPUs.
  - CI runs the GPU tests on Mesa's lavapipe, a software Vulkan device.
  - Next step: performance runs on AMD, Intel, NVIDIA and Qualcomm GPUs.
- **CUDA (preview).**
  - The same device code is used, through `ggml_backend_cuda_reg`.
  - CI compiles and links it (`linux-cuda-build`, compute capability 7.5).
  - **Hardware validation steps** on an NVIDIA machine:
    - `test_ggml_device`;
    - `bench_gpu_accuracy` (`BENCH_GPU=cuda`);
    - the smoke test (`DYNALM_BACKEND=cuda`).
  - Packages that ship CUDA depend on the NVIDIA CUDA runtime libraries under NVIDIA's
    licence.
- **Model coverage** is the same on every provider (`gpu_model_support`).

### Still open

- **Prefix sharing:** copy-on-write still flushes the step and copies the block through host
  memory (`GgmlDevice::copy`).
- **Concurrency:** many concurrent sequences make each attention call read the cells of every
  sequence's blocks in its span. Fully masked chunks are skipped, but partially masked ones
  are read.
- **Device-side sampling:** greedy argmax was not implemented; logits rows are downloaded
  (about 600 KB per row for Qwen2.5). Its share of step time was not measured separately.
- **Activation scratch** is written back to memory every step (correct, not minimal).
- **Not supported:** MoE models, YaRN or Llama 3 RoPE scaling, and Qwen3-style per-head Q/K
  norm (refused as aliasing).
- **Not done:** CUDA, autotuning, DynaKernel GPU kernels, speculative decoding in the batching
  scheduler.

## 1. Executive summary

- **Goal.** Run DynaLM models on Apple Metal (M4 Pro), then CUDA. Reach single-user parity
  with llama.cpp on the same GPU first. Then pursue multi-user throughput and tail latency as
  separate objectives. A 2–2.5× multi-user throughput gain over llama.cpp / Ollama on selected
  workloads is an **aspirational target**, not a prediction.
- **Approach.** Reuse ggml's GPU backends through their public graph API, behind a thin
  adapter in DynaCore. DynaLM keeps its own request lifecycle, scheduler, KV pool and prefix
  cache. DynaIR's deferred recording becomes the mandatory path from the Transformer's op
  stream to one backend graph per step.
- **The largest risk is paged attention**, not matmul. DynaLM's KV layout and block tables do
  not map onto ggml's attention operation (section 7). The first prototype may use a
  contiguous GPU KV layout to get end to end. That proves feasibility only. Stage 2 must show
  paged KV working with at least two concurrent users before the architecture is approved.
- **Smallest first prototype:** Qwen2.5-0.5B or -1.5B Q4_K_M, one user, Metal through a pinned
  ggml revision, DynaLM CLI end to end, greedy output validated against the CPU reference
  (section 9).

### Design principles

1. **The CPU path is the reference and does not change.** GPU work adds a device; it does not
   alter CPU kernels, defaults or numerics.
2. **One submission per step.** No per-op CPU↔GPU round trips. A step's ops become one backend
   graph; the host waits once, for the sampled tokens or logits.
3. **Two schedulers, two jobs.** DynaLM's `Scheduler` decides *which requests* are in a step.
   The backend graph scheduler decides *how that step's ops* run on the device. Neither does
   the other's job.
4. **No silent numerical changes.** A GPU result is accepted only against an explicit contract
   (section 8). An unsupported op is an error at load time, never a quiet CPU detour.
5. **Measure before claiming.** Unified memory, storage modes, graph reuse and kernel choices
   are measured on the target machine before they become defaults.
6. **Custom kernels are opt-in** until they pass correctness and benefit gates, exactly like
   the CPU kernel registry's accuracy classes (`dynacore/include/dynacore/kernel/matmul_registry.h`).

## 2. Current code findings

### Repository and environment

| Fact | Source |
|---|---|
| Design branch `feat/gpu-backend-design`, created from `origin/release/dynalm-release-candidate` @ `44a9851`; version 0.1.4 (`CMakeLists.txt`); contains the last CI fix `ef6fd28` | [code] git |
| `main` is v0.1.3 and lacks the 0.1.4 kernel registry, tests and thread-pool guard; GPU work must not start from it | [code] git |
| Machine: Apple M4 Pro (8 P + 4 E cores), 24 GiB unified memory, macOS 26.7, arm64 | [code] `sysctl`, `sw_vers` (read-only) |
| CMake and Ninja are **not installed**; the presets require Ninja and CMake ≥ 3.24 | [code] `command -v`, `CMakePresets.json`, `CMakeLists.txt` |
| Only the Command Line Tools are installed; `xcrun -f metal` fails, so there is **no offline Metal shader compiler**. The macOS SDK (with the Metal framework) is present | [code] `xcrun` (read-only) |
| No ggml source, submodule or fetch step exists in the repository. "GGML" appears only as the name of the block formats DynaCore implements itself | [code] `git grep`, `dynacore/include/dynacore/quantization/quant_formats.h` |
| No GPU measurement of DynaLM exists anywhere in the repository | [code] |

### Device layer (DynaCore)

- `Device` (`dynacore/include/dynacore/device/device.h`) is the only path from the model to
  hardware. It already has device-memory calls: `allocate`, `copy`, `synchronize`, `upload`,
  `download`, and `host_accessible()`. **[code]**
- Activations are fp32 row-major by contract ("activations are fp32 row-major [rows, cols]").
  **[code]**
- `create_device()` (`dynacore/src/device/device_registry.cpp`) parses `metal`, `cuda`, `hip`,
  `vulkan`. Anything but CPU returns `Unsupported` and points to `docs/gpu-backend.md`, which
  this document now provides. `compiled_devices()` returns `{kCpu}`. **[code]**
- `ENABLE_METAL`/`ENABLE_CUDA`/... are CMake options that make configuration fail when ON
  (`CMakeLists.txt`). **[code]**
- `Storage` (`dynacore/include/dynacore/memory/storage.h`) is a raw `void*` + size + `DeviceLoc`.
  `DeviceType::kMetal` and `kCuda` exist. Only CPU storage is allocated. **[code]**
- `TensorView`, `KvLayerView` and every op carry raw pointers. A GPU backend must map a
  pointer to (buffer, offset) or return pointers that are valid device addresses. **[code]**
- `prepack_weight` builds CPU-only interleaved copies (Q4_K×8, Q6_K×8). A GPU device must make
  it a no-op. **[code]** `dynacore/cpu/cpu_device.cpp` `CpuDevice::prepack_weight`

### Model runtime (DynaLM)

- `Engine::create` (`dynalm/src/runtime/engine.cpp`) creates the device. In `kCompiled` mode it
  wraps the device in a `RecordingDevice` (deferred). Then it creates the KV pool and the
  Transformer **on that wrapped device**. The default is `kReference` (direct calls). **[code]**
- `Transformer` already has the device-memory paths a GPU needs **[code]** `dynalm/src/model/transformer.cpp`:
  - `device_weight()` uploads when `!host_accessible()`;
  - `device_scratch()` allocates activations through the device;
  - logits go to a device scratch and are fetched with one `download` per step.
- **Mid-step host reads [code].** The MoE router downloads its logits inside the forward pass,
  and so does the shared-expert gate (`transformer.cpp`, the `host_accessible()` branches near
  the router and the sigmoid gate). Each is a sync point in the middle of a step. The dense
  models of the prototype do not hit them. MoE models would split a step into several graphs.
- `KvBlockPool::create` (`dynalm/src/kv_cache/kv_cache.cpp`) allocates one K and one V
  storage per layer through the device, in 16-token blocks (`engine.cpp` passes block size
  16). Copy-on-write (`make_writable` → `copy_block`) calls `Device::copy` per block. **[code]**
- `Scheduler::step` (`dynalm/src/scheduler/scheduler.cpp`) is synchronous **[code]**:
  1. plan (including KV reservation);
  2. `Transformer::forward_batch`;
  3. sample on the host for each row that wants logits;
  4. per-token callbacks;
  5. prefix-cache inserts.

  It already times each phase (`SchedulerStats::plan_ms`, `forward_ms`, `sample_ms`,
  `emit_ms`, `prefix_insert_ms`).
- The OpenAI API rejects `logprobs` / `top_logprobs` (`dynalm/src/api/openai.cpp`). Returning
  only sampled token ids from the device therefore breaks no API feature today. **[code]**
- Speculative decoding exists for a single sequence (`dynalm/src/runtime/speculative.h`:
  `NgramDrafter`, `ModelDrafter`, verify + `KvBlockTable::truncate` rollback). It is used by
  `dynalm run`, not by the batching scheduler. **[code]**

### DynaIR (compiled execution)

- `RecordingDevice` (`dynacore/include/dynacore/ir/recording_device.h`) in `kDeferred` mode
  **[code]**:
  - records each call with private copies of host arguments;
  - flushes at sync points (`synchronize`, `download`, `copy`, `allocate`, `upload`);
  - signs the segment structurally: op kinds, buffer pointers, shapes, strides, dtypes, kernel
    plan; not positions, token ids or block tables;
  - caches up to 256 plans and falls back to recorded order on any planner error.
- **What it executes [code]:** a cached plan is a list of *calls on the inner device*: one per
  op, `matmul_many` groups, `matmul_gated` (see `ExecStep`). It does **not** hand a whole graph
  to the device. A GPU device behind it would still receive hundreds of separate calls per
  step. Section 4 covers the change this needs.
- `GraphExecutor` (`dynacore/include/dynacore/ir/executor.h`) runs standalone IR on a Device.
  It reports softmax, mul, transpose, reshape, cast, reduce, broadcast and kv_read as
  unsupported. **[code]**
- `docs/dynakernel.md` records known graph-layer gaps **[code/history]**:
  - `GraphExecutor::run_op` passes f16/bf16 `x` to matmul as `float*`;
  - matmul bias dtype is unchecked;
  - `allocate` ignores `Layout`.
- Compiled mode is opt-in, measured on CPU only. Its tests require bit-identical logits on
  tiny architectures (`docs/dynacore-compiler.md`). **[history]**

### DynaKernel

- `dynacore/{include,src}/dynacore/dynakernel/` contains a kernel IR, lowering, generated C++
  and an LLVM JIT prototype (loaded at run time via `DYNACORE_LLVM_C`). They are linked only by
  tests and benchmarks (`dynacore/CMakeLists.txt`). **[code]**
- `CpuDevice::register_external_matmul` is an extension point that nothing registers, gated by
  `KernelPlan::external_matmul = false`. **[code]**
- It has no GPU target. **[code]**

### Recorded comparisons with llama.cpp (CPU, other machine)

| Result | Source |
|---|---|
| Throughput at parity or ahead of llama.cpp at every point; TTFT 2–3× lower under concurrency (Phase 21, Intel CPU) | **[history]** git log, Phase 21 commit |
| Single-prompt prefill about 30% slower than llama.cpp; the gap is attributed to llama.cpp's int8 activations, which DD-053 rejects on Qwen3-4B | **[history]** `docs/dynakernel.md` |
| All 0.1.4 opt-in kernel figures: i7-1255U, Windows, CPU | **[history]** `CHANGELOG.md` 0.1.4 |

None of these describe GPU behaviour or the M4 Pro.

## 3. Architecture

```
 HTTP / CLI ──► Engine (submit, RequestStream)                                dynalm/src/runtime
                   │
                   ▼
 ┌─────────────── DynaLM Scheduler (request level) ───────────────┐          dynalm/src/scheduler
 │ admission, KV reservation, prefix cache, batch = SeqBatch[]    │  decides WHICH rows run
 │ GPU-aware: device memory budget, shape buckets (proposal)      │
 └───────────────┬────────────────────────────────────────────────┘
                 │ forward_batch(batch, plan)
                 ▼
 Transformer (unchanged op stream: embedding, matmul, rope, kv_store, attention, ...)
                 │ Device calls
                 ▼
 DynaIR RecordingDevice (deferred) ── signature ──► plan cache                dynacore/ir
                 │ segment = whole step (proposal: GraphDevice hook)
                 ▼
 ┌──────────── DynaCore device contract ────────────┐                         dynacore/device
 │ capabilities · buffers · events · prepare/run    │
 └──────┬──────────────────────┬────────────────────┘
        │                      │
        ▼                      ▼
  CpuDevice (reference,   GgmlGraphDevice (adapter, proposal)                 dynacore/gpu/ggml
  fallback; unchanged)         │  IR segment ──► ggml_cgraph (cached per shape)
                               │  backend graph scheduler: HOW ops run
                    ┌──────────┴───────────┐
                    ▼                      ▼
             Metal provider          CUDA provider (later)                    pinned ggml backends
                    │
                    ▼  (stage 2, decision D4)
             custom paged-attention kernel / DynaKernel provider (opt-in only)
```

| Layer | Owns | Must not |
|---|---|---|
| DynaLM Scheduler | which requests and rows are in a step, KV block reservation, prefix sharing, admission against KV and device-memory budgets, fairness, preemption | choose kernels, know about ggml or Metal |
| Transformer | the model's op sequence, from `ModelConfig` | branch on GPU vendor |
| DynaIR (RecordingDevice + passes) | turning a step's calls into a graph; signature; plan cache; fusion; marking dynamic inputs | own device memory, decide request membership |
| DynaCore device contract | buffers, events, capability queries, prepare/run of a segment | know about requests or models |
| ggml adapter | IR → ggml graph translation, ggml buffers, mapping DynaCore pointers to ggml tensors | change DynaLM numerics without a contract |
| Backend graph scheduler (ggml) | op placement and execution within the step graph | add or remove requests, reorder steps |
| Metal / CUDA providers (ggml) | kernels | — |
| Custom kernel provider | an opt-in kernel for an op ggml lacks (paged attention) | become default without gates |
| CpuDevice | reference results, fallback at load time, model-free tests | change |

## 4. End-to-end flow and DynaIR compiled-execution requirements

### Flow of one step [proposal]

1. `Scheduler::step` builds `SeqBatch[]` exactly as today, with one added constraint: the
   prefill row count is rounded to a **shape bucket** (below) when the device asks for it.
2. `Transformer::forward_batch` issues its Device calls, which the `RecordingDevice` records.
3. At the step's single sync point (the logits or token `download`), the recorder:
   - computes the signature;
   - on a hit, reuses a **prepared backend graph**;
   - on a miss, builds IR, plans it, and asks the device to prepare a graph.
4. **Dynamic inputs** are uploaded into persistent per-plan input buffers: token ids,
   positions, `row_seq`, block tables (stage 2), attention masks or lengths, and the
   sampling selector.
5. The graph is submitted asynchronously, and the host waits on its completion event.
6. The host reads back either sampled token ids (device argmax) or logits rows (host sampler),
   then continues `Scheduler::step` as today.

### What DynaIR must gain [proposal]

| Need | Today **[code]** | Required |
|---|---|---|
| Whole-segment execution | `execute()` replays plan steps as inner-device calls | A device hook, e.g. `GraphDevice::prepare(const Graph&, const ExecPlan&) → PreparedGraph` and `run(PreparedGraph&, DynamicInputs)`. `RecordingDevice` uses it when the inner device implements it, and otherwise behaves as today (CPU path unchanged) |
| Dynamic inputs as graph inputs | Positions, ids and block tables are copied into each `Call` and folded into IR as `kConstant` (`index_const`) | They become named `kInput` values with fixed capacity, so one prepared graph serves every step with that signature |
| Attention span | Implicit: the KV length grows every decode step | Expressed as a bucketed capacity (for example a multiple of 256 tokens) plus a per-row length or mask input. Otherwise every decode step changes a tensor shape and misses the cache [assumption: ggml graphs have static shapes per build] |
| Shape buckets | Signature includes the exact row count | Decode: exact row counts 1..N (few values). Prefill: buckets (1, 2, 4, 8, 16, 32, 64, ...) with padded rows whose outputs are discarded. The padding cost against graph-build cost is a measured trade-off |
| Sync points | `allocate` and `copy` flush the segment | Allocation happens at load or warm-up, never inside a step. `copy_block` (copy-on-write) becomes a recorded op (`kCopy`), not a flush |
| Verification | `verify` is off in production | On for every cache miss in GPU builds (misses are rare once warm) |
| Op coverage | IR has `kSoftmax`, `kMul`, `kReshape`, ... that `GraphExecutor` cannot run | The adapter declares its supported set. Every op of a recorded step must be in it (section 8) |
| Known gaps | f16/bf16 `x` read as `float*`; bias dtype unchecked; `Layout` ignored (`docs/dynakernel.md`) | Fixed and tested before any GPU lowering relies on them |

**A GPU device requires compiled execution [proposal].** `Engine::create` forces `kCompiled`
for a non-host device. `--execution reference` with `--backend metal` is a configuration
error, except a debug mode that runs per op for bisecting a numerical mismatch.

**Production readiness [assumption].** Compiled mode has CPU tests and a plan cache, but it
has never driven a device with its own memory, events or asynchronous completion. Treat the
recorder, the signature and the fallback path as code that needs hardening, not as finished.

### Graph and buffer lifetimes [proposal]

- **Weights:** wrapped or uploaded once at load, owned by the device, alive for the Engine.
- **KV pool:** allocated once by `KvBlockPool::create`, never reallocated.
- **Activation workspace:** one arena per prepared graph shape, reused every step. ggml's
  graph allocator assigns offsets inside it [assumption: `ggml_gallocr` or the backend
  scheduler's reserve step].
- **Dynamic-input buffers:** fixed-capacity, persistent per prepared graph.
- **Prepared graphs:** an LRU cache keyed by signature, with a bounded count (like the
  256-entry plan cache). Evicting one frees its workspace only after its last completion
  event.
- **Staging:** at most one in-flight upload buffer and one readback buffer per step, sized
  for the largest bucket.

### Dependencies and completion [proposal]

- One completion event per submitted step. The host waits only on it.
- `Device::synchronize()` waits for all of the device's work.
- Copy-on-write block copies are ordered inside the step graph, before the first write to the
  copied block.
- Stage 1 has no host↔device overlap within a step. Overlap across steps is in section 6.

### Failure handling [proposal]

- Preparation failure (unsupported op, shape, allocation): the step is not submitted. That
  batch's sequences fail through the existing `e.seq->fail(st)` path in `Scheduler::step`.
  The device is marked unhealthy, and new requests get 503 until an operator restarts.
  There is no mid-request switch to CPU: the request's KV lives on the GPU, and recomputing
  it silently would change latency and numerics.
- Command execution failure (device error, timeout): treated the same way. The recorder
  discards the segment, and no partial outputs are read.
- Load-time failure (Metal unavailable, model op unsupported):
  - `--backend metal` stops with an error naming the op or reason;
  - `--backend auto` falls back to CPU and logs why, before any request is accepted.

## 5. ggml adapter design and dependency decisions

### Why ggml, and through which surface

ggml's Metal and CUDA backends are tuned for exactly the block formats DynaLM reads from GGUF
(Q4_K, Q6_K, Q8_0, ...). Reusing them is the fastest way to single-user parity.

Reuse means **using ggml's public graph API**:
- build a `ggml_cgraph` from ops such as `ggml_mul_mat`;
- allocate tensors in backend buffers;
- run the graph on a backend.

It does not mean calling ggml's Metal shaders or CUDA kernels individually. They are private
implementation details with no stable interface.

### Public API concepts used [assumption: verify all against the pinned revision]

| Concept | Purpose here |
|---|---|
| `ggml_init` with `no_alloc`, `ggml_new_tensor_*`, `ggml_view_*` | Describe tensors and views without allocating |
| `ggml_backend_dev_*` / `ggml_backend_*_init`, device and buffer types | Pick the Metal (later CUDA) device |
| `ggml_backend_alloc_ctx_tensors`, `ggml_gallocr_*` | Weights and graph workspaces in backend buffers |
| `ggml_backend_dev_buffer_from_host_ptr` (where the device supports it) | Candidate for wrapping the mmapped GGUF without a copy on Apple GPUs |
| `ggml_backend_tensor_set` / `_get` (and async variants) | Dynamic inputs and readback |
| `ggml_backend_graph_compute` / `_async`, `ggml_backend_synchronize`, `ggml_backend_event_*` | Submission and completion |
| `ggml_backend_sched_*` | Backend graph scheduler, used only for op placement inside one step |
| Ops: `ggml_get_rows`, `ggml_mul_mat`, `ggml_add`, `ggml_mul`, `ggml_rms_norm`, `ggml_norm`, `ggml_rope_ext` (freq factors), `ggml_soft_max_ext` (mask, scale), `ggml_flash_attn_ext`, `ggml_silu` / `ggml_gelu` (or fused GLU ops where present), `ggml_scale`, `ggml_tanh`, `ggml_cpy` / `ggml_set_rows`, `ggml_argmax`, `ggml_top_k` | Coverage of the dense transformer step and device-side greedy sampling |

### DynaIR op mapping (dense prototype) [proposal; ggml side is an assumption]

| DynaIR / Device op | ggml | Notes |
|---|---|---|
| `kEmbedding` | `ggml_get_rows` | Quantized embedding tables must be supported for the model's type |
| `kMatMul` / `kQuantizedMatMul` (+ bias) | `ggml_mul_mat` (+ `ggml_add`) | Weights stay in GGUF block layout; DynaCore's CPU repacks are not used |
| `matmul_many`, `matmul_gated` | separate `ggml_mul_mat` nodes (+ GLU) | The fusions are CPU plan steps; on GPU the graph keeps the nodes and ggml decides |
| `kRmsNorm` / `kLayerNorm` | `ggml_rms_norm` / `ggml_norm` with `ggml_mul` (+ `ggml_add`) | Gemma (1 + w) is already folded by the loader into the f32 vector **[code]** `docs/architecture.md` |
| `kRope` | `ggml_rope_ext` | Mode per `RopeConfig`. Llama Q/K row interleaving (`ModelConfig::qk_rows_interleaved`, **[code]** `dynalm/src/loader/gguf/gguf_model.cpp`) must match the ggml rope mode. Qwen2 does not use it |
| `kKvWrite` | `ggml_cpy` / `ggml_set_rows` into the KV view | Layout dependent (section 7) |
| `kAttention` / `kGqa` | `ggml_flash_attn_ext` or `mul_mat` + `soft_max_ext` + `mul_mat` | Needs a position-major KV view and a mask; softcap and sliding window by mask or op parameters |
| `kActMul`, `kActivation`, `kAdd`, `kScale`, `kSoftcap`, `kFill` | elementwise ops | Straightforward |
| `kGather` / `kScatterAdd` (MoE) | `ggml_get_rows`, `ggml_mul_mat_id`-style ops | Out of prototype scope; MoE also has mid-step host reads **[code]** |

### Layout and type mismatches

- **Dimension order:** ggml lists the innermost dimension first (`ne0` = columns). DynaCore
  `TensorView` is row-major `[rows, cols]` with byte strides. Translation is mechanical but
  must handle strided column views (`cols_of` is used for Q/K/V slices **[code]**
  `transformer.cpp`).
- **Activation precision:** DynaCore activations are fp32. ggml keeps fp32 activations for
  most ops and converts internally inside some kernels [assumption]. The prototype keeps fp32
  activations at graph boundaries. Any move to f16 activations needs a numerical contract.
- **KV layout:** incompatible as is (section 7).
- **Quantized types:** DynaCore's dtype enum and the GGUF type ids map one to one for the
  formats DynaLM supports [assumption: GGUF block layouts are ggml's by definition; verify
  the ggml type ids at the pinned revision].

### Memory ownership

- **DynaCore owns the lifetimes.** The adapter owns the ggml contexts, buffers and graphs, and
  releases them in reverse order on Engine destruction (after the scheduler thread stops, as
  `~Engine` already sequences **[code]**).
- **Pointer identity.** The signature keys on buffer pointers. The adapter keeps a map from
  each DynaCore `Storage::data()` pointer (and offsets inside it) to a ggml tensor view.
  Returned pointers must be unique and stable even when not host-dereferenceable.
- **Host dereference.** `host_accessible()` returns false for the GPU device, so the
  Transformer uploads and downloads explicitly **[code]**, even on unified memory. Zero-copy
  is an optimization inside the device, never an assumption in the runtime.

### Build, licensing, compatibility

- **Pinning.** ggml is MIT licensed [assumption: verify at the chosen revision]. Pin by commit
  and hash it, like cpp-httplib (`dynalm/src/CMakeLists.txt` checks a SHA-256 **[code]**).
  Include its license in `THIRD_PARTY_LICENSES.md` and `NOTICE`.
- **Source.** The candidates are the standalone `ggml` repository or ggml inside llama.cpp.
  This is decision **D1**.
- **Updates.** ggml's API changes often. A pin plus an adapter test suite is mandatory, and
  updates happen deliberately.
- **Boundary.** ggml headers are included only inside `dynacore/gpu/ggml/`. DynaLM never sees
  them, and the core-only build without GPU options stays dependency-free.
- **Shader build** (decision **D2**; compare with section 6):
  - *Runtime compilation*: embed the Metal shader source in the binary and compile it with the
    Metal framework at startup. This works with the Command Line Tools, costs startup time, and
    puts a compile failure at run time [assumption: ggml offers an embedded-library option].
  - *Offline metallib*: requires full Xcode (`xcrun metal`). Startup is faster and errors show
    at build time, but every developer machine and CI image needs Xcode.

## 6. Metal memory management and execution

### Measurements required before choosing defaults [proposal]

| Question | Measure |
|---|---|
| Weights: wrap the mmapped GGUF (shared storage) or copy to private storage? | Decode tok/s and prefill tok/s for each, plus resident memory (a private copy may double weight RAM while the mmap is resident) and load time |
| KV pool: shared or private storage? | Decode ITL at 1 and 4 users with 2k and 8k contexts |
| Is `buffer_from_host_ptr` usable for the whole mapping? | Alignment (page-aligned base, tensor offsets), the maximum buffer length (a large file may need several buffers), and correctness [assumption: ggml splits buffers above the device limit] |
| Working-set limit | Read the device's recommended working-set size at startup and report it in `dynalm doctor` |
| Graph build cost | Build time on a cache miss; misses per 1,000 steps for each bucket scheme |
| Host overhead per step | Recording, signature, input upload, submit, wait, readback |

Unified memory does not guarantee zero-copy, and private storage does not always win.
Both are measured, per buffer class.

### Memory budget and admission [proposal]

- **Budget:** weights + KV pool + the largest workspace + staging ≤ working-set limit − a
  safety margin (default 10%, configurable).
- **KV sizing:** `auto_kv_tokens` (`engine.cpp`) is sized from free system RAM today
  **[code]**. A GPU device sizes the KV pool from the device budget instead, and logs the
  numbers.
- **Admission:** the scheduler already admits against KV blocks **[code]**. Its limits come
  from the device budget.
- **Out of memory:** an allocation that fails at load is an error before serving starts.
  Workspaces are reserved during warm-up for every bucket, so no step allocates.

### Execution [proposal]

- **Step 1 (prototype):** synchronous. Submit, wait for the completion event, read back.
- **Step 2:** overlap host work across steps. After step N's tokens are read back, submit
  step N+1 first, then run step N's callbacks and prefix inserts (`emit_ms`,
  `prefix_insert_ms`) while the GPU works.
  - Decode of N+1 depends on N's sampled tokens, so true pipelining of decode needs
    device-resident token feedback. That is deferred until measured worthwhile.
- **Device sampling:**
  - Stage 1 appends `ggml_argmax` to the logits rows of greedy requests, and reads back one
    int32 per row instead of `vocab_size` floats. For Qwen2.5 (vocab ≈ 152k), that is about
    600 KB less readback per row [estimate].
  - Non-greedy requests keep the host `Sampler` on downloaded logits rows.
  - Top-k / top-p on device come later, behind the same contract (identical token for
    greedy; the distribution contract of `sampling/sampler.h` for sampling).

### Metrics [proposal]

| Metric | Where |
|---|---|
| Graph cache hits, misses, build ms, evictions | Like `RecordingStats` |
| Encode/submit ms, GPU execution ms (command-buffer GPU timestamps [assumption]), host wait ms | Per step |
| Bytes uploaded and downloaded per step; readback mode (tokens or logits) | Per step |
| Device memory: weights, KV, workspaces, staging; working-set limit | Startup and `/metrics` |
| Unsupported ops at load; fallbacks (must be 0); device errors | Counters |
| Existing scheduler times (`plan_ms`, `forward_ms`, ...) | Unchanged |

All of these go to `/metrics` and to the `dynalm benchmark` JSON `diag` block, next to the
existing `diag.matmul_kernels`.

## 7. Paged KV attention: trade-offs and risk

### How DynaLM's KV works today [code]

- `KvGeometry` / `KvLayerView` (`dynacore/include/dynacore/attention/paged_kv.h`):
  - per layer, a pool of `num_blocks` blocks of `block_size` = 16 tokens;
  - inside a block, K is `[kv_head][slot][head_dim]`, i.e. head-major, contiguous across a
    block's tokens for one head;
  - element types f32, f16, q8_0.
- A sequence maps logical blocks to physical ones through a block table. `KvLayerView::k_offset`
  computes `((block * n_kv_heads + head) * block_size + pos % block_size) * head_dim`.
- `KvBlockPool` (`dynalm/src/kv_cache/kv_cache.cpp`) refcounts blocks:
  - the radix or hash prefix cache shares full blocks across requests;
  - `make_writable` copies a shared block before it is written (copy-on-write);
  - speculative decoding rolls back with `truncate`.
- The CPU kernels follow `row_seq` → `KvLayerView` per row, so one batch mixes sequences
  with different block tables (`CpuDevice::kv_store`, `CpuDevice::attention`).

### Why it does not fit ggml attention as is

- `ggml_flash_attn_ext` (and the mul_mat + softmax form) expects, per sequence, K and V as
  tensors with uniform strides along the position dimension [assumption].
- DynaLM's positions are uniform only *inside* a block. Across blocks the stride jumps, and
  physical blocks are in arbitrary order.
- Batching several sequences in one attention call additionally needs a per-row view of a
  different block table.

### Options

Estimates use Qwen2.5-1.5B (28 layers, 2 KV heads, head_dim 128, f16 KV). That is about 28 KB
of K+V per token of context per sequence, so a 4k context means about 115 MB read per decode
step per sequence [estimate from `ModelConfig` values; head counts assumed from the public
model card]. Weights at Q4_K_M are roughly 1 GB [estimate].

| | A. Gather to contiguous | B. GPU layout / backend view | C. Custom paged-attention kernel |
|---|---|---|---|
| What | Each step, copy each sequence's blocks into a contiguous scratch K/V (an in-graph `ggml_get_rows` over a position-major pool, or `ggml_cpy` of views), then use ggml attention | Change the GPU KV representation: position-major blocks (`[block][slot][kv_head][dim]`), or one contiguous region per sequence with prefix sharing done by copying | A Metal kernel that reads the block table and walks blocks directly (flash-decoding style, with split-K for long contexts) |
| Memory traffic per step | ~3× the KV bytes (read, write, read again). At 4k, ~345 MB per sequence against ~115 MB; at 16 users, several GB per step, which dominates the weight traffic [estimate] | 1× when contiguous; sharing by copy adds copy traffic at admission | 1× the KV bytes, plus the small block table |
| Allocation / copy overhead | Scratch sized for the largest batch × context; per-step copy ops | Per-sequence regions fragment; copy-on-share costs memory | None beyond the pool |
| Prefix-cache semantics | Preserved (the pool stays paged) | Weakened: contiguous per-sequence regions cannot share blocks; position-major blocks keep sharing but still need a gather or kernel to attend | Preserved fully |
| Complexity | Low: uses only ggml ops | Medium; touches `paged_kv.h`, `kv_cache`, and the CPU path if layouts are shared | High: GQA, masks, softcap, sliding window, f16 and q8_0 KV, many lengths, numerics; and it must run inside or beside the ggml graph (below) |
| Accuracy risk | Low (ggml's validated attention) | Low | Medium: online softmax and split-K merges need their own tolerance tests against the CPU reference |
| CUDA implications | Same gather approach works | Same layout reused | A second kernel to write; established open-source paged-attention designs exist for reference |

**Integration constraint for C [assumption].** ggml's public API has no documented way to add
a custom GPU kernel as a node in a Metal or CUDA graph. The paths are:
1. split the step into ggml segments around attention, run our kernel on our own command queue
   with explicit event synchronization (needs access to the buffers' native handles), or
2. carry a minimal patch in the pinned ggml adding one op.

Both have costs. This is decision **D4**.

### Recommendation [proposal]

> **As implemented (section 0, D4):** stage 1 went further than planned. It keeps DynaLM's
> real head-major paged pool and uses option A, the gather, directly. No separate contiguous
> KV was needed, because `ggml_set_rows` and `ggml_get_rows` address the 16-token blocks row by
> row. The traffic cost of A at long contexts and many users is still the open question below.

- **Stage 1 (original plan)** uses a GPU-only, position-major, contiguous per-sequence KV for one user
  (option B without sharing), to get end to end with ggml attention.
- **Stage 2** keeps DynaLM's paged pool (position-major blocks on the GPU) and implements
  **A first, as the correctness baseline**. It measures A's traffic, then implements **C**
  behind an opt-in switch if A's cost is confirmed.

This explicitly accepts that **the first prototype's contiguous KV may have to be replaced by
custom paged attention** to keep DynaLM's serving advantages. Stage 1 proves the device path,
not the cache architecture.

## 8. Correctness, fallback and numerical-validation contracts

### Support checks [proposal]

- **At model load**, the adapter dry-runs a representative decode step and a prefill bucket
  through the recorder. Every recorded op, dtype and shape is checked against the adapter's
  declared support. Any miss is reported by name:
  - `--backend metal`: the error stops startup;
  - `--backend auto`: the error moves the Engine to CPU, before serving starts.
- **Mixed graphs.** The prototype accepts no partial graphs: all ops on the GPU, or the model
  does not run there.
- **Later splits.** An explicit planner may place a region on another backend only when its
  estimate shows the transfers pay. Any CPU region inside a ggml graph would run on **ggml's
  CPU backend, not DynaCore's CPU kernels**, which is a different numerical path. It needs
  its own validation entry and is reported in the metrics.

### Precision contracts

| Contract | Source | GPU requirement |
|---|---|---|
| DD-053: perplexity ≤ +1%, mean KL ≤ 0.0025 nats against the fp32 reference | **[code]** `kernel_plan.h`, `matmul_registry.h` | GPU logits against the CPU fp32 reference arithmetic (on 0.1.4: `DYNACORE_MATMUL_DISABLE=all-accelerated`) on the reference models |
| DD-031: batch invariance only with int8 off | **[code]** `engine.h` (`int8_decode_rows`) | The GPU makes no batch-invariance promise; the planner may choose different kernels per row count [assumption about ggml]. Documented, and measured by comparing 1-row and N-row results |
| Greedy token identity | — | **Not required** by DD-053 and not expected (different reduction orders). Required instead: (a) the DD-053 gate; (b) report the first diverging position against the CPU on a fixed prompt set; (c) the maximum and mean absolute logit difference at the first step stays within a tolerance fixed from the prototype's measured distribution and recorded in the test |
| Run-to-run determinism | — | Same binary, inputs and batch shape: identical tokens across 10 runs [proposal; verify ggml Metal's behaviour] |
| Sampling | **[code]** `sampling/sampler.h` | Device argmax equals host `sample_greedy` on the same logits (tie rule: lowest index, verified against the CPU code) |
| Custom kernels | **[code]** registry accuracy classes | New kernels enter as `kExperimental`, opt-in; default only after DD-053 and benchmark gates |

### Tests [proposal]

- **Model-free** (run in CI where a GPU exists, skipped elsewhere): every Device op on the GPU
  against `CpuDevice` on random inputs; dtype coverage; strided views; dynamic-input reuse;
  plan-cache reuse; failure injection (unsupported op, allocation failure).
- **Model-based:**
  - DD-053 perplexity and KL on Qwen2.5-0.5B/1.5B;
  - the greedy divergence report;
  - an end-to-end CLI and API smoke test (`tools/ci/smoke_api.py` with `--backend metal`).
- **Resource safety:** leak checks for buffers, graphs and events across 1,000 Engine
  create/destroy cycles. ASan on the host side; Metal validation layers enabled in debug runs.

## 9. Staged proof-of-concept plan

### Stage 1: single-user Metal feasibility

- **Scope:**
  - Qwen2.5-0.5B (or 1.5B) Q4_K_M, one user;
  - `dynalm run` and `dynalm serve` with `--backend metal`;
  - DynaLM's Engine and Scheduler unchanged in behaviour;
  - compiled/deferred DynaIR with the whole-segment hook;
  - GPU-only contiguous KV (section 7), f16;
  - ggml adapter at the pinned revision;
  - device argmax for greedy.
- **Not in scope:** paged KV, prefix cache on the GPU, concurrency above 1, MoE, CUDA,
  custom kernels.
- **Acceptance:**
  1. The model loads and generates through the CLI and the API.
  2. The DD-053 gate passes against the CPU fp32 reference, and the greedy divergence report
     is produced. Logit tolerances are recorded.
  3. llama.cpp's Metal backend is benchmarked on the same GGUF file (section 10).
  4. TTFT, prompt tok/s, TPOT/ITL p50/p95/p99, output tok/s and memory are measured.
  5. A report lists unsupported ops, every host↔device copy and sync per step, and any
     fallback.
  6. The CPU path still passes `tools/ci/run_tests.sh`, `dynalm doctor` and
     `tools/ci/check_isa.py`, with unchanged outputs.

A standalone Metal matmul benchmark does not count as this prototype.

### Stage 2: paged KV, concurrency (architecture gate)

- **Scope:**
  - DynaLM's `KvBlockPool` on the GPU with position-major blocks;
  - copy-on-write as a recorded op;
  - the prefix cache enabled;
  - option A gather attention, then option C if A's measured traffic confirms the estimate;
  - ≥ 2 concurrent users, then 4, 8 and 16;
  - shape buckets and graph reuse measured.
- **Acceptance:**
  - correct output for every concurrent stream (DD-053 per stream, and matching
    single-stream results within the stated tolerance);
  - prefix-cache hits produce the same tokens as misses;
  - a measured comparison of A against C (or A alone, with its traffic numbers);
  - multi-user benchmarks against llama.cpp server and Ollama.

**This stage decides whether the architecture is approved.**

### Stage 3: performance

Overlap across steps, top-k/top-p on device, graph-reuse tuning, shared vs private storage
defaults. Each change is gated by a paired A/B result (`docs/benchmarking.md`).

### Stage 4: CUDA

The same contracts with the ggml CUDA backend. See section 11.

## 10. Benchmark methodology and acceptance gates

### Fairness rules [proposal]

- **Same files:** the same GGUF file and SHA-256 for every engine; quantization Q4_K_M.
- **Same inputs:** tokenized prompts of the same length. Report the prompt length each engine
  saw, as `tools/compare_engines.py` does **[code]**.
- **Same settings:**
  - greedy (temperature 0), fixed seed;
  - same context per user;
  - fixed output length with EOS ignored. Ollama ignores `ignore_eos` **[history]**
    `docs/benchmarking.md`; rows below 95% of the requested tokens are not comparable;
  - same concurrency, warm-up requests excluded.
- **Backend confirmed in every run:**
  - llama.cpp: all layers offloaded (log line) and flash attention setting recorded;
  - Ollama: the GPU share it reports (for example `ollama ps`);
  - DynaLM: `backend_name()` in the benchmark JSON and `doctor`.
- **Machine state recorded:** power source, Low Power Mode off, other processes, temperature
  drift. Sessions are separated, and arm order rotates (`docs/benchmarking.md`).
- **Recorded per run:** executable path and SHA-256 for each engine, and the engine versions.

### Metrics

- TTFT;
- prompt processing rate;
- TPOT and ITL at p50, p95 and p99;
- aggregate output tok/s;
- end-to-end latency;
- prefix-cache hit rate;
- resident memory and device memory;
- GPU utilization where available [assumption: obtainable on macOS without special
  privileges].

### Workloads

| ID | Users | Prompt / output | Purpose |
|---|---|---|---|
| S1 | 1 | 128 / 256 | Single-user decode |
| S2 | 1 | 2048 / 128 | Single-user prefill (TTFT) |
| M1 | 4, 8, 16 | 512 / 128 | Multi-user throughput and ITL tails |
| M2 | 8, 16 | Shared 1,500-token system prompt + 200 unique / 128 | Prefix-cache benefit |
| SP | 1 | Code-editing and summarization prompts | Speculative decoding (separate track) |

### Gates (targets, not results)

| Phase | Gate |
|---|---|
| Stage 1 | Correctness gates of section 9; performance reported, no target |
| Single-user parity | S1 and S2 within ±10% of llama.cpp Metal (median of paired sessions, with a reliable reference by the `bench_analyze.py` rules) |
| Multi-user | M1 and M2: output tok/s and ITL p99 better than llama.cpp server and Ollama at ≥ 8 users. Aspirational: 2–2.5× on selected workloads, only claimed with reproducible paired evidence |
| Speculative decoding | Separately: identical greedy output, and a single-user speed-up measured on SP with acceptance rates reported. Compared with llama.cpp both with and without its own speculative mode |

Proposed targets stay in this document. Results go into the CHANGELOG with their raw data
location, as the 0.1.4 entries do.

## 11. CUDA compatibility contracts and deferred work

**Preserved for CUDA [proposal]:**
- the `Device` op set;
- the `GraphDevice` prepare/run hook;
- signature and dynamic-input semantics;
- buffer classes (weights, KV, workspace, staging) and completion events;
- capability queries;
- the support-check and failure rules;
- the precision contracts and test suites.

**Allowed to differ:**
- storage modes (discrete GPUs have no unified memory: weights are always uploaded);
- whether host memory is pinned;
- the use of CUDA graphs inside the ggml backend [assumption];
- multi-GPU placement;
- the paged-attention implementation (option C needs its own CUDA kernel).

**Deferred until Metal stage 2 passes:**
- CUDA build integration;
- CPU/GPU layer split for models larger than VRAM;
- cost-model tuning, autotuning and versioned tuning caches;
- a DynaKernel GPU provider;
- multi-GPU;
- HIP and Vulkan.

**Future contracts kept open:**
- a capability query that reports per-op support and dtypes;
- a provider registry, where an opt-in kernel provider can claim an op (paged attention) with
  an accuracy class, like `MatmulKernelDesc`;
- measured costs keyed by (device, op, shape bucket, dtype), consulted only for graph-level
  choices DynaLM actually makes: bucket sizes, splits, which provider claims an op.

There is no per-kernel optimizer for kernels ggml selects internally.

## 12. Risks, assumptions and decisions requiring approval

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Paged attention does not fit ggml | Serving advantages lost on the GPU | Stage 2 gate; options A and C; decision D4 |
| DynaIR recorder not ready for device graphs | Bugs in the critical path | Section 4 hardening list; verification on every miss; fallback counters must stay 0 |
| ggml API churn | Breakage on update | Pin plus hash; adapter test suite; deliberate updates |
| Graph rebuilds every step (dynamic shapes) | Host overhead kills decode | Attention-span buckets and masks; measure misses per 1,000 steps |
| Numerical divergence from the CPU | Confusing results, broken trust | DD-053 gate, divergence report, documented non-invariance |
| Memory pressure on 24 GiB unified memory | OOM, swapping | Device budget, warm-up reservation, admission limits |
| Shader toolchain (Command Line Tools only) | Build or startup failures | Decision D2; verify on this machine before stage 1 |
| No GPU in hosted CI runners [assumption] | GPU tests never run in CI | A local GPU test script plus CPU-only CI; tests skip without a device |
| Benchmark unfairness (Ollama templates, early EOS) | Misleading claims | Section 10 rules; `compare_engines.py` comparability marking |

### Rejected alternatives

- **Writing all Metal and CUDA kernels in-house first:** years of tuning to reach parity.
  Rejected for the first phases.
- **Calling ggml's internal kernels directly, outside its graph API:** no stable interface,
  and it breaks on every update. Rejected.
- **Per-op GPU dispatch through the existing `Device` calls (reference mode):** hundreds of
  submissions or syncs per step. Rejected except as a debug mode.
- **Silent per-op CPU fallback:** hides numerical and latency changes. Rejected.
- **Using ggml's backend scheduler for request scheduling:** mixes the two schedulers' jobs.
  Rejected.

### Decisions requiring approval

D1, D2, D3, D5, D6 and D7 are decided; see section 0. D4 (the stage-2 paged-attention path) is
still open.

| ID | Decision | Options |
|---|---|---|
| D1 | ggml source and exact pinned revision | Standalone `ggml` repository, or ggml from llama.cpp; vendored, or fetched with a hash |
| D2 | Metal shader build | Embedded source compiled at run time (works with the Command Line Tools) or offline metallib (requires full Xcode) |
| D3 | Prototype model | Qwen2.5-0.5B or 1.5B, Q4_K_M |
| D4 | Paged attention path for stage 2 | A only; A then C; and for C, a split graph with our own queue or a minimal ggml patch |
| D5 | Engine behaviour when Metal fails with `--backend auto` | Fall back to CPU with a log line (proposed), or refuse to start |
| D6 | Greedy-identity policy | Accept DD-053 + a divergence report (proposed), or require token identity on a golden prompt set (likely unattainable) |
| D7 | Where GPU work lands | `feat/gpu-backend-design` and follow-on branches from the 0.1.4 release candidate (proposed) |

### Build prerequisites for implementation (original plan; see section 0 for what was run)

```sh
brew install cmake ninja            # or official installers; CMake >= 3.24
xcode-select -p                     # Command Line Tools present today
xcrun -f metal                      # fails today; needed only for offline metallib (D2)
cmake --preset macos-release -DENABLE_METAL=ON      # once the option is implemented
cmake --build --preset macos-release
build/macos-release/bin/dynalm doctor               # should list the Metal device
bash tools/ci/run_tests.sh build/macos-release      # CPU tests unchanged
python3 tools/ci/smoke_api.py build/macos-release/bin/dynalm <model.gguf>
```

## 13. Files each phase would change

Proposed; none are changed by this document.

| Phase | Files |
|---|---|
| G0: dependency and build | `CMakeLists.txt` (`ENABLE_METAL` stops being fatal), `cmake/Ggml.cmake` (new: pinned fetch or vendored path, hash), `dynacore/CMakeLists.txt`, `CMakePresets.json` (a Metal preset), `THIRD_PARTY_LICENSES.md`, `NOTICE`, `docs/build.md`, `.github/workflows/ci.yml` (build-only Metal job) |
| G1: device contract | `dynacore/include/dynacore/device/device.h` (capabilities, events, GraphDevice hook), `dynacore/include/dynacore/device/device_registry.h`, `dynacore/src/device/device_registry.cpp` (`create_device`, `compiled_devices`), `dynacore/include/dynacore/memory/storage.h`, `dynacore/src/hardware/system_info.cpp` (Metal device in `gpu_info`) |
| G2: DynaIR lowering | `dynacore/include/dynacore/ir/recording_device.h`, `dynacore/src/ir/recording_device.cpp` (whole-segment execution, dynamic inputs, `copy` as an op), `dynacore/include/dynacore/ir/ir.h` (input and copy semantics), `dynacore/src/ir/passes.cpp`, `dynacore/src/ir/verifier.cpp`, `dynacore/src/ir/executor.cpp` (documented gaps) |
| G3: ggml adapter and Metal provider | New: `dynacore/include/dynacore/gpu/ggml_device.h`, `dynacore/src/gpu/ggml/ggml_adapter.cpp`, `dynacore/src/gpu/ggml/ggml_lowering.cpp`, `dynacore/src/gpu/metal/metal_device.cpp` (or `.mm`); `dynacore/CMakeLists.txt` |
| G4: runtime wiring | `dynalm/src/runtime/engine.h`, `dynalm/src/runtime/engine.cpp` (force compiled, device budget in `auto_kv_tokens`, backend auto), `dynalm/src/config/config.cpp` (`runtime.device: metal`), `dynalm/src/cli/run.cpp`, `dynalm/src/cli/serve.cpp`, `dynalm/src/cli/benchmark.cpp` (backend in JSON), `dynalm/src/diagnostics/doctor.cpp`, `dynalm/src/model/transformer.cpp` (logits or token readback, no prepack on GPU), `dynalm/src/metrics/metrics.cpp`, `dynalm/src/server/server.cpp` (`/metrics` fields), `dynalm/src/kv_cache/kv_cache.cpp` and `dynalm/src/runtime/sequence.cpp` (GPU KV geometry) |
| G5: device sampling | `dynalm/src/model/transformer.h`, `dynalm/src/model/transformer.cpp`, `dynalm/src/model/seq_batch.h`, `dynalm/src/scheduler/scheduler.cpp`, `dynalm/src/sampling/sampler.cpp` (tie-rule test) |
| G6: paged attention (stage 2) | `dynacore/include/dynacore/attention/paged_kv.h` (position-major layout option), `dynalm/src/kv_cache/kv_cache.h`, `dynalm/src/kv_cache/kv_cache.cpp`, `dynalm/src/prefix_cache/*` (unchanged semantics, tested), the adapter lowering, and new kernel sources if C is chosen |
| Tests | New: `dynacore/tests/test_gpu_device.cpp`, `dynacore/tests/test_ir_graph_lowering.cpp`, `dynalm/benchmarks/test_gpu_e2e.cpp` (model-based); `tools/ci/run_tests.sh`; `tools/ci/smoke_api.py` (`--backend`) |
| CUDA (later) | `dynacore/src/gpu/cuda/*` (new), `cmake/Ggml.cmake`, `CMakePresets.json`, CI |

## Recommendation: the smallest prototype

> **Outcome (section 0):** the stage-1 prototype was built and run through the real CLI and API.
> It differs from the plan below in two ways:
> - it uses DynaLM's paged KV with a gather, instead of a separate contiguous KV;
> - device argmax was deferred.
>
> Gate A (correctness) passed. Gate B (single-user parity with llama.cpp Metal) did **not**
> pass: decode is ahead on short contexts, prefill is behind, and long-context decode is
> slightly behind. Next: the GPU prefill chunk default, zero-copy weights, then stage 2.

**Build first:** stage 1 of section 9.
- One dense model: Qwen2.5-0.5B or 1.5B, Q4_K_M.
- One user, Metal through a pinned ggml.
- DynaIR deferred recording that hands the whole step to the device as one prepared,
  cached graph.
- Contiguous GPU KV.
- Device argmax.
- Driven by the real `dynalm run` and `dynalm serve`.

**It passes when:**
1. the model generates through the CLI and the API;
2. DD-053 holds against the CPU reference, and the divergence report and logit tolerances are
   recorded;
3. it is benchmarked against llama.cpp Metal on the same file;
4. every copy, sync and unsupported op is accounted for;
5. the CPU path is unchanged.

**Approve before implementation:** D1 (ggml source and revision), D2 (shader build),
D3 (model), D5 (auto fallback) and D7 (branch). D4 (paged attention path) and D6 (identity
policy) must be settled before stage 2.

If stage 1 works but stage 2 cannot keep paged KV and the prefix cache on the GPU at
acceptable cost, the multi-user goal needs a different plan. Stage 2 is the real test of this
design.

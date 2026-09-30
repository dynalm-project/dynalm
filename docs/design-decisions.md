# Design decisions

Format: Decision / Reason / Alternatives / Tradeoffs / Evidence.

---

## DD-001: C++20, CMake + Ninja, MSVC primary on Windows

- **Decision:** C++20. CMake ≥ 3.24 with presets. MSVC Build Tools 2022 is the
  Windows compiler; gcc/clang on Linux.
- **Reason:** The dev machine had no toolchain. MSVC is the host compiler nvcc
  requires on Windows (future CUDA), and it supports ASAN. C++20 gives
  `std::format`, `std::span`, concepts, and `starts_with` with no extra dependencies.
- **Alternatives:** LLVM-MinGW (clang, portable), C++23 (`std::expected`).
- **Tradeoffs:** MSVC has no UBSAN/TSAN, so those run under the Linux presets.
  C++23 support is still uneven across compilers.
- **Evidence:** n/a (toolchain choice).

## DD-002: Status/Result instead of exceptions for errors

- **Decision:** Fallible functions return `Status` or `Result<T>`. `ENGINE_RETURN_IF_ERROR`
  and `ENGINE_ASSIGN_OR_RETURN` propagate errors.
- **Reason:** Failure paths stay visible at subsystem boundaries. A bad request
  can't unwind through the scheduler. The OK path costs nothing: no allocation,
  just an empty string.
- **Alternatives:** exceptions; `std::expected` (C++23).
- **Tradeoffs:** More boilerplate than exceptions. `std::bad_alloc` from the STL can
  still throw; the memory manager will turn large allocation failures into
  `kOutOfMemory`.
- **Evidence:** `bench_foundation` "Result<int> ok path" (see docs/benchmarks.md).

## DD-003: Runtime ISA dispatch, no global arch flags

- **Decision:** Build a portable baseline. SIMD kernels get per-file arch flags.
  The best tier is chosen once at startup from cpuid + XCR0.
- **Reason:** One binary runs on every x86-64 CPU and still uses AVX2/AVX-512
  where present. Checking XCR0 avoids faults when the OS disabled a register state.
- **Alternatives:** `-march=native` builds; separate binaries per ISA.
- **Tradeoffs:** The kernel entry costs one indirect call (a function pointer set
  at startup). Per-file flags need care so inline functions from headers don't
  spread AVX2 code into generic TUs. Rule: SIMD code lives only in `.cpp`
  files under `backends/cpu/<isa>/`.
- **Evidence:** n/a until Phase 17 kernels exist.

## DD-004: Hybrid-core awareness from day one

- **Decision:** Detection reports P-cores and E-cores separately (Windows
  `EfficiencyClass`, Linux `cpu_core`/`cpu_atom` PMUs).
- **Reason:** On hybrid CPUs such as the i7-1255U (2P+8E), a barrier-synchronized
  GEMM runs at the pace of the slowest worker. AUTO thread selection needs this
  information to avoid putting heavy work on E-cores.
- **Alternatives:** use `hardware_concurrency()` only.
- **Tradeoffs:** Platform-specific code in `platform/`. It stays isolated there.
- **Evidence:** To be measured in Phase 6/17 (threads = P-cores vs. all cores).

## DD-005: GoogleTest via pinned FetchContent

- **Decision:** Use a system GTest if present, otherwise download v1.15.2 with a
  pinned SHA256.
- **Reason:** A standard, well-known framework. Pinning the hash makes builds
  reproducible.
- **Tradeoffs:** The first configure needs network access, or a pre-installed GTest.

## DD-006: Byte strides; quantized innermost dim is packed

- **Decision:** `TensorLayout` stores strides in bytes. For block-quantized dtypes the
  innermost dimension is always packed (stride recorded as 0). It can only be sliced on
  block boundaries and can't be transposed.
- **Reason:** A Q4_K row can't be addressed per element; addressing only makes sense
  per 256-element block. Byte strides handle scalar and block types the same way. This
  matches GGML's `nb[]`, so GGUF tensors map in with zero copies.
- **Alternatives:** element strides plus a special case for quantized types; separate
  tensor classes for quantized weights.
- **Tradeoffs:** Kernels must use `dtype_row_bytes()` rather than `n * sizeof(T)` for
  row sizes.

## DD-007: Non-owning TensorView for kernels; owning Tensor for lifetime

- **Decision:** `TensorView` (pointer + layout + device) is trivially copyable and has no
  refcount. `Tensor` = `shared_ptr<Storage>` + view. Kernels and the runtime's inner loops
  take views. Weights and KV pools hold Tensors.
- **Reason:** No atomic refcount traffic on hot paths. Ownership stays explicit at
  subsystem boundaries (the model owns weights, the KV pool owns blocks).
- **Alternatives:** a single refcounted tensor type (PyTorch-style).
- **Tradeoffs:** A view can dangle if it outlives its Tensor. The rule: views never
  outlive the call or step that created them.
- **Evidence:** `select+slice` p50 is 77 ns (bench_tensor). That's cheap for graph setup
  but not for per-element use. Kernels compute addresses directly.
- **Note:** The spec lists a separate `TensorMetadata` type. It isn't needed: dtype, shape
  and strides are in `TensorLayout`, and device and alignment are properties of the
  `Storage`. Adding the type would only add another layer.

## DD-008: Engine-owned DType; file type IDs live in loaders

- **Decision:** `DType` is the engine's own enum. GGML type IDs are mapped only inside
  `loader/gguf`. Types the engine can't run (IQ*, TQ*, MXFP4) still get their geometry
  checked, so the file is validated, and they're reported as unsupported.
- **Reason:** Keeps the core free of GGUF (spec §5). A SafeTensors loader maps its own
  dtype strings the same way.

## DD-009: Memory-map GGUF weights; zero-copy tensors

- **Decision:** `MappedFile` maps the whole file read-only. Every weight `Tensor` is a
  borrowed `Storage` over the mapping, holding the mapping as its keep-alive.
- **Reason:** Load time scales with metadata only (0.75 ms for a 258 MiB model, warm).
  Weights sit in the OS page cache, are shared between processes, and are never copied.
  That matters on a machine with about 1.7 GiB free.
- **Alternatives:** read the file into engine-allocated buffers; repack weights at load
  time into kernel-preferred layouts.
- **Tradeoffs:** The first inference touches cold pages (page-fault latency), and
  `MappedFile::prefetch` can warm them. If Phase 17/18 kernels want repacked layouts
  (e.g. interleaved rows for Q4_K GEMM), repacking will copy into owned memory, as a
  per-tensor opt-in with a measured benefit.
- **Evidence:** `bench_loader`: open 0.75 ms p50, `load_tensor` × 272 = 38 µs.

## DD-010: The GGUF parser treats every file as untrusted

- **Decision:** A bounds-checked reader; count sanity limits before reserving memory;
  checks for duplicate keys and tensors, alignment, and data extent; array nesting
  limited to depth 2.
- **Reason:** Model files come from the internet. A malformed file must produce
  `kCorrupt`, not undefined behavior.
- **Evidence:** A test truncates the file at every byte offset, and 300 random-corruption
  runs pass without crashes.

## DD-011: Model IR filled in two steps; weights addressed by role

- **Decision:** `ModelConfig` is format- and family-neutral. The format loader copies raw
  hyperparameters. The architecture adapter (Phase 5) sets family semantics (RoPE style,
  MLP type, norms, biases, soft-capping) and validates the result. Weights live in a
  `TensorRegistry` keyed by `(TensorRole, layer)`. Loaders translate their naming scheme
  into roles, and anything they can't map is reported rather than dropped silently.
- **Reason:** The same Llama adapter must run a GGUF or a SafeTensors checkpoint. Only the
  loaders know `blk.N.attn_q.weight` vs `model.layers.N.self_attn.q_proj.weight`. A flat
  array of slots makes lookup O(1) with no string hashing after load.
- **Alternatives:** adapters read GGUF keys directly (couples families to the format);
  string-keyed weight maps (slower, and typos only show up at runtime).
- **Tradeoffs:** Loaders need role tables, and new tensor kinds need a new enum value.
  That's deliberate: every weight the engine uses has a named meaning.
- **Precision:** `PrecisionConfig` keeps activation, accumulator and KV dtypes separate.
  Weight dtypes are per tensor (spec §6).

## DD-012: Hand-written pre-tokenizers, verified against HF

- **Decision:** Byte-level BPE pre-tokenization uses hand-written matchers, one per
  regex family (GPT-2, Llama-3, Qwen2, StarCoder/SmolLM). The Unicode L/N/White_Space
  tables are generated from `unicodedata`. Unknown `tokenizer.ggml.pre` ids are rejected.
- **Reason:** `std::regex` has no `\p{L}`/`\p{N}` and is slow. A wrong split silently
  changes token IDs, so unknown variants fail loudly instead of falling back.
- **Evidence:** Golden tests against HF `tokenizers` (tools/gen_tokenizer_golden.py)
  match exactly on 38 varied cases for SmolLM2 (49K vocab) and Qwen2.5 (151K vocab).
  Throughput is 15 MB/s encode (`bench_tokenizer`), so no word cache is needed yet.
- **Tradeoffs:** Each new regex family needs a matcher, a unit test and a golden file.

## DD-013: Chat templates by family detection, not Jinja (for now)

- **Decision:** Detect the template family from signature strings in the GGUF Jinja
  source and render it natively (ChatML, Llama-3, Llama-2, Mistral, Gemma, Phi-3,
  DeepSeek-V2/3). Default system prompts embedded in ChatML templates are extracted.
  BOS is never rendered; the tokenizer adds it per `add_bos`.
- **Reason:** A Jinja interpreter is a sizable subsystem, and the families above cover
  the Tier-1 models. Rendering is deterministic and testable.
- **Alternatives:** embed a Jinja engine (minja, jinja2cpp).
- **Tradeoffs:** Templates with custom logic (tools, reasoning toggles) render only
  their family's core format. Unrecognized templates return an error and the caller
  picks a format explicitly. Jinja support is tracked in TODO.

## DD-014: Linux container is the reference test environment

- **Decision:** `tools/linux.sh <preset>` builds and tests in an Ubuntu 24.04 / GCC 13
  container. Sanitizers run there: ASAN+UBSAN (fatal on first finding) and TSAN
  (with ASLR disabled via `setarch -R`, required by GCC 13's TSAN on kernels with
  32-bit mmap entropy).
- **Reason:** MSVC has no UBSAN or TSAN. Separately, Windows Smart App Control on the
  dev machine is in enforce mode and randomly blocks freshly linked unsigned test
  binaries. The Linux container gives a reliable gate without weakening host security.
- **Tradeoffs:** Docker is needed for the full gate. Windows builds remain for native
  benchmarking.

## DD-015: One generic Transformer; adapters hold no execution code

- **Decision:** A single `Transformer` runs every decoder family. Variation (RoPE style,
  norm type, biases, fused QKV/gate-up, QK-norm, sandwich norms, soft-capping, sliding
  windows, tied embeddings) is expressed as `ModelConfig` fields. Adapters only
  `configure()` and `validate()`.
- **Reason:** The spec requires no duplicated transformer code and a scheduler that
  doesn't know about model families. Adding Qwen/Gemma/Phi becomes a configuration
  task, and every kernel optimization benefits all families at once.
- **Alternatives:** a forward function per family (llama.cpp's early approach);
  a dynamic graph IR.
- **Tradeoffs:** Exotic architectures (MLA, hybrid SSM) will need new `ModelConfig`
  concepts or a second runtime. A dynamic graph can be added then if needed.

## DD-016: Coarse-grained Backend interface over TensorViews

- **Decision:** `Backend` exposes whole-batch ops (matmul, attention, rope, norms, kv_store,
  ...) that take `TensorView`s. `CpuBackend` implements them, with an inner
  `CpuKernels` table selected by ISA.
- **Reason:** A CUDA backend can implement the same ops with device pointers and no
  scheduler or model changes (spec §45). A virtual call per op is negligible:
  about 300 ops per token versus milliseconds of compute.
- **Tradeoffs:** Cross-op fusion must be expressed as new, fused ops (Phase 18),
  not discovered automatically.

## DD-017: Block-based KV layout from the first implementation

- **Decision:** KV is stored in fixed blocks, `[block][kv_head][slot][head_dim]` per layer,
  and addressed through a per-sequence block table (`KvLayerView`). The single-sequence
  runtime uses the same path with a trivial block table.
- **Reason:** Paged KV, prefix sharing and continuous batching (Phases 10–16) then change
  only block management, not attention kernels. Head-major blocks keep one head's keys
  contiguous across tokens, which is the decode access pattern.
- **Evidence:** A test runs attention over a scrambled block table [5, 2, 7] and matches
  a naive contiguous computation (GQA, f32/f16 KV, sliding window).
- **Tradeoffs:** One indirection per token per head in attention. It will be amortized
  per block in the optimized kernel (Phase 18).

## DD-018: Persistent spin-then-sleep pool with dynamic chunking

- **Decision:** `ThreadPool` keeps its workers alive and spins about 100 µs before
  sleeping. `parallel_for` hands out chunks from an atomic counter.
- **Reason:** A decode step launches hundreds of kernels, and waking threads through a
  condition variable on each launch would add milliseconds per token. Dynamic
  chunks let P-cores take more work than E-cores on hybrid CPUs.
- **Bug found:** Workers originally initialized their "seen epoch" with a fresh load. A
  job published before a worker's first instruction was then never run, and the caller
  spun forever. They now start from the construction epoch (0). Caught by the golden
  tests hanging, and covered by the TSAN run.

## DD-019: Golden logits from an independent NumPy implementation

- **Decision:** `tools/ref_llama.py` is a direct fp32 NumPy Llama forward pass with no KV
  cache, reading the same GGUF through the `gguf` package. Its top-32 logits, logit
  statistics and 8-token greedy continuation are committed as fixtures.
- **Reason:** An independent implementation is easy to audit against the model
  definition, and it catches layout errors (RoPE pairing, GQA mapping, tied embeddings)
  that self-consistency tests miss.
- **Evidence:** SmolLM2-135M: max |Δlogit| < 2e-3 over the top 32; greedy continuation
  identical with f32 and f16 KV; chunked prefill (batch 5) equals a single pass.

## DD-020: Tiny random-weight models as committed adapter fixtures

- **Decision:** `tools/make_tiny_models.py` writes one small GGUF per architecture
  (160–650 KB, f16, fixed seed) that exercises that family's features: QKV bias, QK-norm,
  decoupled head_dim, post-norms, soft-caps, sliding-window patterns, local RoPE base,
  fused QKV and gate|up. It also writes NumPy-reference logits and a greedy continuation
  for each. The models and fixtures are committed.
- **Reason:** Every adapter is verified numerically in CI with no downloads, including
  families whose smallest real checkpoint doesn't fit this machine (Phi-3 at 3.8B).
  Real f16 checkpoints (Qwen2.5-0.5B, Gemma-3-270M) add optional end-to-end checks.
- **Tradeoffs:** Random weights can't catch tokenizer or chat-template problems or
  realistic numeric ranges. The real-model and tokenizer golden tests cover those.
- **Evidence:** All 7 architectures match the reference (top-32 logits and 6-token greedy)
  on the first run, with chunked prefill (batch 5) over 12-token prompts that exceed the
  sliding windows.

## DD-021: Mistral and dense DeepSeek run through existing adapters

- **Decision:** No separate Mistral/DeepSeek adapter classes. Their GGUFs declare
  `general.architecture = llama` (Mistral 7B, DeepSeek-LLM/Coder) or `qwen2`
  (DeepSeek-R1-Distill-Qwen), and those adapters already describe them exactly.
  `deepseek`/`deepseek2` (MoE, MLA) come with Phase 25.
- **Reason:** The spec asks us not to duplicate transformer code or add abstraction
  without purpose. An adapter that only renames another adapter adds nothing.
- **Gaps:** The DeepSeek-LLM and DeepSeek-V3 pre-tokenizers aren't implemented yet, so
  those files fail with a clear error. Mistral v0.1's sliding window isn't in its GGUF
  metadata; llama.cpp ignores it too.

## DD-022: Quantized models are verified against a reference on the same weights

- **Decision:** For quantized checkpoints, the golden reference runs the NumPy forward
  pass on weights dequantized by gguf-py's independent implementation. Engine output
  must match that reference as tightly as in the f16 case. A separate quality check
  against the f16 reference asserts the same top-1 token and bounded logit error.
- **Reason:** A quantized model is a different model. Demanding identical greedy output
  versus f16 fails on legitimate late near-ties (SmolLM2 Q8_0 picks a different token
  than f16 after "Paris."), while a loose tolerance would hide real kernel bugs.
- **Evidence:** SmolLM2 Q8_0: max top-32 logit error 1e-5 vs the Q8_0 reference, 0.43 vs
  f16 (quantization noise). Qwen2.5-0.5B Q4_K_M (real Q4_K, Q5_0, Q6_K and Q8_0 tensors)
  matches its reference including the 8-token greedy continuation. Per-format
  dequantization matches gguf-py on fixtures for all 10 block types.

## DD-023: SequenceState with an explicit status machine

- **Decision:** Each generation stream is a `SequenceState`: tokens, prompt length,
  number of computed tokens, KV block table, stop parameters, and a status
  (waiting → prefill → decode → finished | cancelled | error). The loop computes
  `pending()` tokens when there are any, otherwise it samples. Every terminal
  transition releases KV blocks.
- **Reason:** The scheduler (Phases 11–14) needs exactly this: whether a sequence needs
  prefill or decode work, how much, and where its KV lives. It never needs to know
  the model. Making prefill/decode an explicit state (spec §22) instead of two loops
  lets chunked prefill and decode interleave later without new logic.
- **Evidence:** Unit tests cover every transition, including KV release on finish,
  cancel and error, and KV exhaustion. All goldens pass unchanged through the
  rewritten generator.

## DD-024: 16-token KV blocks; refcounted pool with a mutex-protected free list

- **Decision:** Blocks hold 16 tokens. Per-block refcounts are atomic. The free list is a
  LIFO vector under one short mutex. Block tables support clone (fork: shared blocks,
  retained), truncate (rollback), append_shared (prefix cache) and make_writable
  (copy-on-write before writing into a shared block).
- **Reason (block size):** The sweep over 8/16/32/64 shows no attention-speed difference
  (79–80 ms/token at 4K context), so the choice rests on memory and reuse. The
  expected waste is half a block per sequence: 8 tokens × 23 KB = 184 KB for SmolLM2
  f16 KV, about 12 MB at 64 sequences. Prefix reuse happens at block granularity, so
  smaller blocks reuse more but mean more table entries and more partial-block
  copy-on-write. 16 is the balance (and vLLM's default).
- **Reason (no lock-free list):** allocate+release costs 12.5 ns uncontended and 424 ns p50
  with 4 threads doing nothing else. Real demand is one allocation per 16 tokens per
  sequence, about 100/s even at 64 concurrent sequences, so the mutex costs roughly
  0.004% of one core. A lock-free free list would add ABA hazards and complexity for no
  measurable gain (spec §20: no lock-free code for appearance). Phase 11 re-measures
  under real concurrent load.
- **Invariant:** A shared block is never written. Writers only touch blocks made
  exclusive with make_writable, and the prefix cache shares only full blocks.
- **Evidence:** bench_kv and the block sweep above. A 4-thread alloc/clone/COW/release
  stress test is clean under TSAN, and a copy-on-write test verifies fork isolation on
  actual K/V contents.

## DD-025: Batched forward over flattened rows from many sequences

- **Decision:** `Transformer::forward_batch(seqs)` flattens every sequence's tokens into
  one row batch. Dense ops (matmul, norms, MLP) run on all rows together. Attention and
  kv_store take a per-row sequence index plus one KV view per sequence. Logits are
  computed only for rows that request them (gathered first). The single-sequence
  `forward` is a wrapper.
- **Reason:** This is the foundation of continuous batching. Decode and prefill rows
  from different requests share one pass, and each weight is read once per step.
  Rows are mathematically independent, so batching must not change results.
- **Alternatives:** padding sequences to a common length ([batch, seq] tensors) wastes
  compute on padding and complicates chunked prefill.
- **Evidence:** Mixed batches (decode rows + prefill chunks, with and without logits)
  are bit-identical to running each sequence alone, for Llama, Gemma-3 (sliding
  windows, local RoPE) and Phi-3 (fused projections). bench_batch_decode shows up to
  2.4× aggregate throughput.

## DD-026: Iteration-level continuous batching with recompute preemption

- **Decision:** `Scheduler::step()` runs one iteration: drain submissions and
  cancellations, admit waiting requests while KV allows (the whole pending prompt plus
  one block of headroom), build one batch within the token budget (decode rows first,
  then prefill in admission order), run `forward_batch`, sample, and fire callbacks.
  Under KV exhaustion, the newest running sequence is preempted: its KV is released and
  it's requeued at the front to recompute prompt + generated tokens.
  Submission and cancellation use a mutex-guarded handoff swapped once per step. The
  iteration itself holds no lock.
- **Reason:** Sequences join and leave at any iteration (spec §21). Preemption by
  recompute keeps requests alive under memory pressure without swap space. On CPU,
  recomputing a short prefix is cheap next to failing a request. The per-step handoff
  avoids a global scheduler mutex (spec §49).
- **Alternatives:** swap-out preemption (copy KV to a secondary store; unnecessary while
  KV and weights share host RAM); fail on exhaustion (poor UX); static batching.
- **Evidence:** Tests require every request's tokens to equal an isolated greedy run under
  staggered arrivals, budget-split prompts, cancellation, forced preemption (4 × 22
  tokens into 48 tokens of KV) and 4 concurrent submitters. Invalid requests fail
  individually while others complete. bench_scheduler: 3.2× aggregate throughput.

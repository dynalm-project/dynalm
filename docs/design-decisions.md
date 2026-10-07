# DynaLM Design Decisions (Architecture Decision Records)

Every major design choice in DynaLM, with the reason, the alternatives and the measurements behind it.

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
- **Reason:** Keeps the core free of GGUF. A SafeTensors loader maps its own
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
  Weight dtypes are per tensor.

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
  scheduler or model changes. A virtual call per op is negligible:
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
  the model. Making prefill/decode an explicit state instead of two loops
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
  measurable gain (no lock-free code for appearance). Phase 11 re-measures
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
- **Reason:** Sequences join and leave at any iteration. Preemption by
  recompute keeps requests alive under memory pressure without swap space. On CPU,
  recomputing a short prefix is cheap next to failing a request. The per-step handoff
  avoids a global scheduler mutex.
- **Alternatives:** swap-out preemption (copy KV to a secondary store; unnecessary while
  KV and weights share host RAM); fail on exhaustion (poor UX); static batching.
- **Evidence:** Tests require every request's tokens to equal an isolated greedy run under
  staggered arrivals, budget-split prompts, cancellation, forced preemption (4 × 22
  tokens into 48 tokens of KV) and 4 concurrent submitters. Invalid requests fail
  individually while others complete. bench_scheduler: 3.2× aggregate throughput.

## DD-027: Separate decode and prefill token budgets (default 64 / 64)

- **Decision:** Each step carries up to `decode_token_budget` decode rows (one per generating
  sequence, rotated least-recently-served-first) and up to `prefill_token_budget` prefill rows
  (priority, then admission order; prompts are chunked to fit). Waiting requests are ordered
  by priority, then arrival. Requests can carry a deadline, and expired ones fail with
  `kDeadlineExceeded` whether queued or running.
- **Reason:** Phase 12 measured prefill/decode interference as the dominant tail-latency
  cause (ITL p99 1.2 s). A prefill cap bounds how long any step can stall decoders.
- **Evidence:** Budget sweep in docs/benchmarks.md: 192 → 32 cuts ITL p99 2.5× for about 5%
  throughput and higher TTFT. 64 halves ITL p99 with moderate TTFT cost. Tests verify
  the budgets are respected, decoders keep running while a 40-token prompt is chunked,
  fair rotation under a small decode budget (each of 4 sequences gets exactly 11 tokens),
  strict priority order and deadline handling. Outputs match isolated runs throughout.
- **Tradeoffs:** Fixed budgets aren't optimal for every load. SLO-driven adaptive
  budgets are an AutoTuner item.

## DD-028: Chunked prefill with a per-sequence chunk cap (default 32)

- **Decision:** Prompts are prefilled in chunks bounded by the step's prefill budget, and
  each sequence takes at most `max_prefill_chunk` rows per step (default 32). Several
  prompts therefore prefill side by side.
- **Reason:** Spec §23: a huge prompt must not monopolize the CPU. The budget protects
  decoders (ITL). The per-sequence cap protects other *prefilling* requests from
  head-of-line blocking behind a long prompt (TTFT of short requests).
- **Evidence:** bench_long_prompt: decoder ITL p99 while a 2048-token prompt prefills is
  36.7 s unchunked vs 1.09 s with 32-token chunks (+12% TTFT for the long request).
  Test: a short prompt's first token arrives at step 1 instead of step 4 behind a
  120-token prompt. Chunking never changes outputs (tests compare against isolated runs).
- **Tradeoffs:** Smaller chunks mean more steps (per-step overhead) and a longer TTFT for
  long prompts. The AutoTuner can size chunks from measured per-row cost.

## DD-029: Block-granular prefix cache keyed by chained, verified hashes

- **Decision:** Cache full KV blocks under h_i = H(h_{i-1}, block tokens) (SplitMix64-based).
  Each entry stores its tokens and parent key, and lookups verify both, so a hash collision
  can't return wrong KV. The cache holds a pool reference per block. Eviction is LRU over
  leaf entries that only the cache references. The scheduler looks up on admission (always
  computing at least one prompt token), inserts newly completed blocks after each step, and
  evicts cached blocks before preempting live sequences.
- **Reason:** Spec §24–25. Block granularity reuses the paged KV directly: a hit is a
  refcount increment, never a copy. Full blocks are immutable once complete, so sharing
  needs no locking or copy-on-write. Leaf-first eviction keeps cached chains reachable.
- **Alternatives:** token-level radix tree (Phase 16); caching only prompt blocks (we also
  cache generated blocks, which multi-turn chat reuses).
- **Evidence:** Exact row accounting in tests (52 rows instead of 100 for 4 requests sharing
  16 tokens). A shared-reference test shows blocks are never freed while referenced. Under KV
  pressure there were 0 preemptions because the cache yielded. bench_prefix: 11× faster, 23×
  lower TTFT for a 512-token shared system prompt.
- **Tradeoffs:** Reuse granularity is 16 tokens. Same-step arrivals don't share in-flight
  prefill.

## DD-030: Radix tree over KV blocks with token-granular partial reuse (default)

- **Decision:** `PrefixCache` is an interface with two implementations. The radix cache is
  a tree whose edges are KV blocks (children keyed by a block-token hash, always verified).
  When a prompt diverges inside a cached block, the child with the longest common token
  prefix r is copied into a fresh private block, and r more tokens are reused. Eviction is
  LRU over leaves referenced only by the cache. The scheduler selects the implementation
  via `prefix_cache_kind`, and radix is the default.
- **Reason:** Spec §24 Stage 2. SGLang-style token-level radix trees assume token-granular
  KV. Our KV is paged, so the tree is block-granular, and the partial-block copy (6 µs)
  recovers token-granular reuse at the edge without a second KV layout.
- **Why the copy is correct:** The copied rows [0, r) were computed for exactly the same
  tokens at the same positions after the same prefix, and rows are batch-independent
  (bit-exact, DD-025). The copy is exclusive, so the sequence overwrites rows ≥ r freely.
  Tests: reuse of 18 of 20 tokens and 7 of 8 tokens produces outputs identical to isolated
  runs, and a partial copy's first r rows equal the source block.
- **Evidence:** bench_prefix (500-token unaligned prompt): 11% fewer rows, 2× lower TTFT p50
  than hash, and 2.5× faster lookups (26 µs vs 64 µs for 8K tokens).
- **Tradeoffs:** A partial hit costs one block copy and one extra block. When the pool
  has no free block the partial reuse is skipped (it's an optimization).

## DD-031: SIMD kernels keep fp32 activations; batch results match to float rounding

- **Decision:** Phase 17 kernels (AVX2/FMA/F16C) dequantize weights in registers and FMA
  with fp32 activations, accumulating in fp32. Only the summation order differs from
  the reference. The matmul uses fused dequantize-dot for small row counts (decode) and
  expands each weight row once for 4+ rows (prefill). Because the chosen path depends on
  batch size, batched outputs now match sequential ones to float rounding (~1e-6
  relative), no longer bit for bit (DD-025 tightened to a tolerance).
- **Reason:** Spec §37 requires reproducibility "within expected numerical tolerance".
  Bit-exact batch invariance would force one algorithm across decode and prefill, and it
  would conflict with the tiled GEMM planned next. Runs stay deterministic for the same
  schedule, and every golden test kept its original tolerance.
- **Evidence:** All 13 weight types checked per tier against the dequantize reference
  (test_kernels). All real-model goldens unchanged (SmolLM2 Q8_0 still 1e-5 vs its
  reference). Speedups in docs/benchmarks.md.
- **Tradeoffs:** Activation quantization (int8 dot products) would be faster still but
  changes numerics. It gets separate accuracy tests in Phase 18.

## DD-032: Register-blocked, K-blocked GEMM for prefill; split-K decode attention

- **Decision:**
  - Prefill matmul: expand panels of 4 weight rows to fp32 once, then run an AVX2 4×2
    microkernel (8 accumulators) over all activation rows. K is sliced into 1024-wide pieces
    so an activation slice stays cache-resident while a thread sweeps its panels; partial
    sums accumulate. The slice width is a backend setting (`DYNALM_GEMM_KC` overrides it,
    for experiments and the AutoTuner).
  - Decode attention: when (rows × heads) can't fill the pool and the context exceeds 512,
    split each context into 256-position chunks and merge the partial softmaxes with
    log-sum-exp rescaling.
- **Reason:** The profile showed matmuls at 84% of prefill, running at about 10 GFLOPS per
  core-equivalent. The per-row dot streamed every activation row once per weight row.
- **Evidence:** `bench_kernels` microkernel 56–81 GFLOPS single-thread. Prefill 2× faster
  (Qwen 132 → 277 tok/s, ≈257 GFLOPS sustained). The K-slice sweep picked 1024 (+47% on the
  down shape). Split-K: 4K-context decode 27.0 → 22.9 ms. Tests: gemm_panel vs reference in
  every tier, including accumulate mode; split-K vs naive softmax with scrambled blocks, mixed
  row lengths and a sliding window. All goldens unchanged.
- **Tradeoffs:** Expanded panels use 4×K floats of thread-local scratch. Split-K adds a merge
  pass and scratch of (pairs × chunks × (head_dim+2)) floats.

## DD-033: No elementwise operator fusion yet; int8/VNNI deferred with a plan

- **Decision:** Don't fuse RMSNorm/RoPE/activation into neighboring matmuls now. Defer int8
  activation quantization with AVX-VNNI.
- **Reason:** Spec §33: fuse only where measurements show meaningful gains. Norms, RoPE,
  activations and KV stores together are under 6% of prefill and 3% of decode, so the best
  case for fusion is a few percent at a large maintenance cost. Decode is bandwidth-bound
  (≈18.7 GB/s), so int8 compute doesn't help it. Prefill is now compute-bound at near-peak
  fp32 FMA throughput, so VNNI (2–4× int8 MACs) is the right next prefill lever. It changes
  numerics, though, and needs its own accuracy tests against the references (planned).
- **Evidence:** `bench_profile` breakdown (docs/benchmarks.md, Phase 18).

## DD-034: Engine facade with a dedicated scheduler thread and per-request event streams

- **Decision:** `Engine` owns the model, thread pool, backend, KV pool and scheduler, plus
  one scheduler thread that steps while there's work and sleeps on a condition variable
  otherwise. `generate_text/generate_chat` submit through the scheduler's handoff queue
  and return a `RequestStream`. Scheduler callbacks run `TextStreamer` (UTF-8-safe
  deltas, stop strings) and push events without blocking. Consumers pop with a timeout
  and may cancel. A weak-pointer control block keeps `cancel()` safe after shutdown, and
  streams still open at shutdown end with `kCancelled`.
- **Reason:** Spec §27–28 and §31: stream without buffering, and an HTTP thread → request
  queue → scheduler → runtime path where HTTP can never block the scheduler. A slow client
  only grows its own event queue.
- **Stop strings:** Text is held back only while it could still begin a stop string (the
  minimal suffix), so normal text flows immediately and a stop string is never emitted,
  even split across tokens. On a match, the stream ends with `stop` and the scheduler
  request is cancelled to free its KV.
- **Evidence:** Tests: UTF-8 never split; stop strings across tokens; minimal holdback;
  earliest of several stops; engine streams incrementally and matches the Generator;
  concurrent clients on 4 threads; consumer cancellation; per-request errors; streams
  outliving the engine. `dynalm run` now uses this path.
- **Tradeoffs:** Event queues are unbounded per request. A client that never reads holds
  at most its own request's text, bounded by `max_tokens`.

## DD-035: cpp-httplib for HTTP, an in-house JSON parser, greedy-only decoding until Phase 26

- **Decision:** The server uses cpp-httplib v0.18.3 as a single header, downloaded at
  configure time with SHA256 verification, compiled only when `ENABLE_SERVER=ON`, kept
  private to `server.cpp`, and included as SYSTEM so its warnings don't fail our build. JSON
  is a small in-house parser and serializer with depth and size limits. The OpenAI layer
  (`api/openai`) is pure functions, independent of HTTP.
- **Reason:** HTTP/1.1 parsing on a listening socket is security-critical and easy to get
  subtly wrong, so a widely used library beats new code there. JSON needs are narrow, and a
  small parser is easy to audit and fuzz (20k mutated documents per test run under
  ASAN/UBSAN). Spec §28: lightweight HTTP that never blocks the scheduler. Handlers only
  relay a `RequestStream`.
- **Compatibility choices:** Sampling fields (temperature, top_p, seed, penalties) are
  accepted so standard clients work, but decoding is greedy until Phase 26 (superseded by DD-043). A one-time
  warning is logged and the limitation is documented. Features we can't honor (tools,
  n>1, logprobs, non-text content, response_format) are rejected with 400 rather than
  ignored. Requests to a model without a chat template get a clear 400 on
  /v1/chat/completions, and /v1/completions still works.
- **Evidence:** test_server: parsing and validation, response and chunk shapes, an HTTP
  end-to-end test on an ephemeral port (health/models/metrics, non-streaming and SSE
  streaming equality, JSON errors without collateral damage, 6 concurrent clients, client
  disconnect), plus a manual run with Qwen2.5-0.5B ("What is 7 times 6?" → "42").

## DD-036: Batched matmuls expand weights from two rows up, with AVX2 row dequantization

- **Decision:** `CpuBackend::matmul` uses the fused dequantize-dot kernel only for a single
  activation row. From two rows up (`expand_min_rows_ = 2`, override
  `DYNALM_MATMUL_EXPAND_MIN`), it expands 4-row weight panels to fp32 once and runs the
  register-blocked GEMM. Row dequantization has AVX2 kernels for Q8_0, Q4_0, Q5_0, Q4_K
  and Q6_K (F16 already had one). They reuse the dot kernels' unpacking but store scaled
  floats.
- **Reason:** The first head-to-head run against llama.cpp (DD-037) showed us 1.9× slower
  at 4 concurrent requests, while matching it at 1. The fused kernel is per row: a decode
  batch of m sequences re-reads and re-decodes every weight m times, so m=4 cost about 4× a
  single row (Q4_K_M ITL 87–91 ms vs 27 ms). The expand path reads weights once per batch,
  but its scalar dequantization had made it slower than fused below m≈8.
- **Alternatives:** (a) Raise the threshold. That only trades which batch sizes are slow.
  (b) A multi-row fused kernel that dots each decoded weight block against up to 4
  activation rows in registers. It avoids the fp32 panel write and is the better long-term
  decode kernel, so it is listed in TODO. (c) int8 activations with VNNI (DD-033), deferred.
- **Evidence:**
  - `bench_kernels` dequant row speedups: Q6_K 9.8×, Q5_0 5.0×, Q4_0 4.1×, Q8_0 1.9×,
    Q4_K 1.1×.
  - End-to-end threshold sweep, `dynalm benchmark` in-process, Qwen2.5-0.5B Q4_K_M,
    10 threads, prompt 128, output 64. Two alternating repetitions, ITL p50 in ms:

    | rows (c) | fused | expand ≥ 4 | expand ≥ 2 |
    |---|---|---|---|
    | 2 | 47.1–48.3 | 46.4–46.7 | 34.8–37.5 |
    | 4 | 87.1–91.0 | 48.4–49.8 | 42.7–47.0 |

  - On Q8_0 at c=4, expand gives 48.9 vs 68 ms fused, and 75 vs 129 ms at c=8.
  - `test_kernels` checks every tier's dequant entry against the reference.
  - The batch tests (DD-031 tolerance) pass with the new threshold.
- **Tradeoffs:** Expanding writes an fp32 panel per weight panel (L1/L2 resident, 4 rows ×
  K). At m=2 the GEMM's 4×2 microkernel is half used. Q4_K dequantization gained little,
  because its generic code was already table-free, so Q4_K small batches gain mostly from
  reading weights once. Decode at m=1 is unchanged.

## DD-037: Benchmark framework: one closed-loop load generator for every target

- **Decision:** `dynalm benchmark` drives a closed-loop load generator (`bench/loadgen`).
  For each point (concurrency × prompt length × output length), C client threads issue
  requests back to back.
  - Prompts are generated to an exact token count, with a unique leading text per request,
    and are never reused across points.
  - Outputs are fixed length (`ignore_eos`).
  - Before each point, one uncounted warm-up request runs.
  - Reported per point: TTFT, inter-token latency (gap between streamed deltas), TPOT and
    end-to-end latency, as nearest-rank P50/P90/P95/P99 plus the mean. Also output and
    input tokens/s, error count, and, in-process, peak RSS and CPU utilization.
  - Targets: the engine in-process, or any OpenAI-compatible server (`--url`, streaming
    `/v1/completions` with `stream_options.include_usage`). The same client measures this
    engine's server, llama.cpp's `llama-server` and Ollama.
  - Results are JSON lines. `tools/bench_report.py` renders them, and
    `tools/compare_baselines.sh` runs the head-to-head in containers.
- **Reason:** Spec §21 asks for P50–P99 and baselines against llama.cpp and Ollama.
  Comparisons are only meaningful when every target sees the identical workload and
  measurement. Each tool's own benchmark measures different things (for example,
  llama-bench has no queueing or HTTP).
  - Fixed output lengths remove the "who stopped first" bias.
  - Unique prompts per request and per point keep prefix caches (ours and llama.cpp's
    `cache_prompt`) from turning a throughput test into a cache test. The first framework
    version reused prompt indices across points, and the second point's TTFT dropped to
    66 ms on 128-token prompts. A test now guards this.
- **Alternatives:**
  - An open-loop (Poisson arrival) generator. It is better for latency-under-load curves,
    but needs a target rate per system; closed loop at fixed concurrency is what both
    baselines' docs report. Open loop is listed in TODO.
  - External tools (vegeta, locust): no per-token streaming timing, and Python is not
    allowed on hot paths.
  - Server-reported timings: not comparable across implementations.
- **Tradeoffs:**
  - ITL is measured between streamed text deltas. A delta held back for UTF-8 or a stop
    string merges two tokens into one gap; TPOT (e2e − TTFT)/(n − 1) is reported alongside.
  - HTTP targets report no server RSS or CPU, because the client runs in its own container.
  - Ollama ignores `ignore_eos`, so its rows report the mean completion length actually
    produced.
  - Containers on a laptop share one thermally limited CPU, and servers run one at a time.
- **Evidence:** `test_bench`: percentile ranks, exact prompt lengths, unique prefixes,
  fixed output lengths through the scheduler, no cross-point prefix reuse. Results are in
  `docs/benchmarks.md` (Phase 21).

## DD-038: Layered configuration and RAM-based AUTO sizing

- **Decision:**
  - **Merging.** `dynalm serve` options come from a config file (`--config` or
    `DYNALM_CONFIG`; `key = value` lines), then `DYNALM_<OPTION>` environment variables,
    then the command line. Later sources win. `config/merge_config` concatenates the three
    sources into one argument list, and the command's existing parser consumes it, so every
    option works in every source without duplicated parsing.
  - **AUTO values.** `threads auto` means physical cores. `batch auto` means 256.
    `http-threads auto` means max-active + 8.
  - **KV auto (`ctx auto`, the default).** Use half of the RAM still available after the
    weights. Cap at 65536 tokens, never go below min(context, 2048), and round down to
    whole blocks. If RAM is unknown, fall back to min(context, 16384).
- **Reason:** Spec §42 asks for CLI, environment variables and a config file, plus an AUTO
  mode from cores, RAM, SIMD, model size and context. The old fixed min(context, 16384)
  default wasted capacity on large machines and could overcommit small ones.
  - Half of the free RAM leaves room for the page cache that backs the memory-mapped
    weights, and for other processes.
  - The 65536-token cap bounds the up-front allocation. At ~12 KiB per token (Qwen2.5-0.5B)
    that is 0.75 GB, enough for 32 concurrent 2K-token conversations.
- **Alternatives:**
  - A TOML/YAML parser: a dependency for a flat key/value need.
  - Per-option environment handling in each command: duplicated and easy to forget.
  - Growing the KV pool on demand: needs re-addressing of block tables. The pool is fixed,
    and sizing it is the AUTO decision.
- **Tradeoffs:**
  - MemAvailable is a point-in-time reading, so a later memory spike by another process is
    not foreseen.
  - The file format has no sections or quoting; values cannot contain `#`.
- **Evidence:**
  - `test_config`: parsing with line-numbered errors, file < env < CLI precedence, `auto`,
    and the AUTO KV policy (budget, cap, floor, unknown RAM, f32).
  - Container run: a 5.85 GB-free machine chose 65536 tokens (0.75 GB) for Qwen2.5-0.5B, and
    the startup log reports RAM required = weights + KV.

## DD-039: Server hardening: admission limit, spare HTTP workers, timeouts, graceful drain

- **Decision:**
  - **Admission limit.** At most `max_active` (default 64) completion requests are served
    at once. Further requests get 503 `overloaded_error` with `Retry-After: 1`.
  - **Spare workers.** The HTTP pool defaults to max_active + 8 workers. A streaming
    response occupies a worker for its whole lifetime, so the spare workers keep `/health`
    and `/metrics` answering under full load.
  - **Timeouts.** Every request gets a default timeout (600 s, queued plus generating),
    enforced by the scheduler's deadlines.
  - **Error mapping.** Engine errors map to HTTP status codes: 400 invalid or unsupported,
    503 resource exhausted or cancelled, 504 deadline, otherwise 500.
  - **Capacity check.** A request whose prompt + max_tokens exceeds the KV pool is
    rejected at submission, before any compute.
  - **Graceful drain.** On SIGINT/SIGTERM or `POST /admin/shutdown`, the server drains:
    new requests get 503, `/health` returns 503 `draining`, and in-flight requests run to
    completion for up to `--shutdown-timeout` (default 30 s). A second signal stops
    immediately; the remaining streams end as cancelled.
  - **Admin endpoint.** `/admin/shutdown` accepts loopback clients only, and
    `--disable-admin` removes it. `dynalm stop` is its client.
  - **Request ownership.** Each admitted request is owned by an RAII `Inflight` object held
    by the response. Its destructor returns the admission slot and cancels the engine
    request if it did not finish. That covers every exit path, including a client that
    disconnects before the streaming provider first runs; the old code leaked the active
    gauge, and never cancelled the request, on that path.
- **Reason:** Spec §29, §39 and §40.
  - Unbounded admission turns overload into unbounded latency and memory. Explicit 503s let
    load balancers retry elsewhere.
  - Before this change, with 16 HTTP workers, a 17th streaming client blocked behind the
    others, and so did health checks.
  - Draining lets rolling restarts happen without cutting off users mid-answer.
- **One model per process:** `dynalm unload` is the same as `dynalm stop`. Multi-model
  hosting means separate weights, KV pools and schedulers competing for the same cores and
  memory bandwidth. On a CPU that is better done with separate processes behind a router,
  whose failures are also isolated. This keeps the server small.
- **Alternatives:**
  - Queueing beyond `max_active` inside the server instead of 503: hides overload and grows
    TTFT without bound. The scheduler already queues up to `max_running`.
  - An async HTTP server so streams don't hold threads: cpp-httplib is thread-per-request,
    and 72 mostly idle threads cost little. Switching libraries would be a large change for
    no measured gain.
  - Token-authenticated admin: deferred. Loopback-only plus an off switch covers the
    single-host case. Behind a same-host reverse proxy, every client appears as loopback,
    so deployments should either use `--disable-admin` or block `/admin/` at the proxy.
- **Tradeoffs:** A fixed `max_active` is not adaptive to request size; KV pressure is handled
  by scheduler preemption beneath it.
- **Evidence:**
  - `test_server`: overload 503 with no leaked slot, health under overload, an 8-client
    admission storm (only 200 or 503, every slot returned), drain, loopback admin and
    disable, 504 on timeout with KV freed, Prometheus metric names, and status mapping.
  - `test_hardening`: 96 mixed requests (invalid, oversized, timeouts, cancellations) on a
    10-block KV pool. Every request ends exactly once, good requests succeed, no KV block
    leaks, and the engine stays usable. Oversized requests are rejected before compute.
    Engine destruction with in-flight streams is safe.
  - Container runs: SIGTERM with one request in flight let it finish (96 tokens, no error)
    and then exited 0; `dynalm stop` drained and exited.
  - The gate runs all of these under ASAN/UBSAN and TSAN.

## DD-040: SafeTensors and Hugging Face directories load into the same IR as GGUF

- **Decision:** `load_model` accepts a GGUF file, a Hugging Face model directory
  (`config.json`, `model.safetensors` or sharded `*.safetensors` with
  `model.safetensors.index.json`, `tokenizer.json`, and optionally `tokenizer_config.json` and
  `generation_config.json`), or one `.safetensors` file inside such a directory.
  - **Weight files.** `loader/safetensors` memory-maps each file. Tensors are zero-copy views,
    except that a tensor misaligned for its element size is copied. Headers are treated as
    hostile: size cap, offsets inside the data section, byte size equal to dtype × shape,
    no overlaps, and shard names that stay inside the directory.
  - **HF layer.** `loader/hf` translates HF conventions into exactly what the GGUF loader
    produces:
    - `config.json` to `ModelConfig`, with `model_type` mapped to the adapter id;
    - parameter names to `TensorRole`s (Gemma 2/3 "sandwich" norms are named differently);
    - `tokenizer.json` to `TokenizerData`;
    - the chat template from `tokenizer_config.json`.
  - **GGUF converter conventions applied at load.** llama.cpp's converter applies these
    offline; the HF loader applies them at load time:
    - **Gemma norm offset.** Gemma RMSNorm weights are folded to (1 + w) as F32 copies of
      the small norm vectors.
    - **Llama 3 RoPE scaling.** `rope_scaling` becomes `rope_freqs` factors, using the same
      formula as the converter.
    - **Llama Q/K layout.** Llama Q/K rows are *not* permuted at load; instead the loader
      records a format-neutral fact, `ModelConfig::qk_rows_interleaved`, from which the
      Llama adapter picks the RoPE style (interleaved for GGUF, half-split for HF). This
      avoids copying weights and keeps adapters free of format knowledge.
  - **Tokenizer models.** Byte-level BPE is supported when its pre-tokenizer matches one we
    implement (GPT-2, Llama 3, Qwen2 or SmolLM/StarCoder split; compared exactly).
    SentencePiece-style BPE with byte fallback (Gemma, Llama 2, Mistral) maps to the SPM
    tokenizer, with each piece's score = −(rank of the merge that creates it). Anything else
    gets a clear error.
  - **Weight dtypes.** F32, F16 and BF16 load directly. Integer and F8 tensors are rejected
    only if the engine would use them.
- **Reason:** Spec §5, phase 2: both loaders feed one Tensor Registry and Model IR, and the
  runtime must not depend on the format. Most models are published as HF SafeTensors first,
  and loading them directly skips a conversion step and its disk copy. Applying the
  converter's conventions inside the loader keeps a single runtime path, which is
  testable as exact equality.
- **Alternatives:**
  - Permuting Q/K rows at load: copies about 10% of the weights and loses zero-copy loading.
  - Folding the Gemma offset into the norm kernel: puts a format concern in the runtime.
  - Shelling out to `convert_hf_to_gguf.py`: needs Python and torch.
  - Implementing the HF Unigram and WordPiece tokenizer models: not used by the tier-1
    families.
- **Tradeoffs:**
  - BF16 and F16 weights are twice the size of Q8_0 and four times Q4. SafeTensors is the
    exact path; GGUF quantization remains the fast one.
  - Parsing `tokenizer.json` adds about 110 ms to load (SmolLM2: 170 vs 58 ms).
  - Multimodal Gemma 3 checkpoints (`model_type` "gemma3") are rejected; text-only ones load.
  - GPTQ/AWQ tensors are recognized only in Phase 24.
- **Evidence:** `test_safetensors`.
  - **Parser.** Valid files, and 10 kinds of hostile headers rejected.
  - **Config and names.** Config translation (Mistral window ignored, Qwen2
    `use_sliding_window: false`, Gemma 3 linear scaling, multimodal rejected), tensor
    names, and Llama 3 factors (1 → 8, monotonic).
  - **Tiny fixtures.** All 7 GGUF fixtures exported by `tools/make_tiny_hf.py` (Llama Q/K
    un-permuted, Gemma norms as w − 1) load with identical logits: max |diff| 0 for six
    architectures, 1.1e-6 for Llama (RoPE pairing order).
  - **Sharding.** A sharded copy with an index file is identical to the single file. A shard
    path escaping the directory is refused.
  - **Tokenizers.** `tokenizer.json` gives the same ids as the GGUF tokenizer for Gemma 3
    (262K, SPM path), Qwen2.5 and SmolLM2 on mixed text including emoji, CJK, whitespace
    runs and specials.
  - **Real checkpoints.** SmolLM2-135M SafeTensors (BF16) vs GGUF F16: identical tokenizer,
    max |logit diff| 1.7e-5, same argmax. `dynalm run` gives the same text.
  - **Qwen2.5-0.5B-Instruct.** SafeTensors loads, tokenizes identically to the official GGUF,
    and answers "What is 7 times 6?" with 42. The official GGUF is *not* a twin of the HF
    weights: its `general.version` is v0.1, and even its F32 norm vectors differ (by factors
    of 1.8–2.5×), so logits are not compared for that pair. Comparing with a converter-made
    twin is what proves layout equivalence, and the tiny fixtures and SmolLM2 do that.

## DD-041: GPTQ/AWQ are repacked at load into the block formats the kernels already run

- **Decision:** HF checkpoints with `quantization_config` of `quant_method` gptq (4 or 8 bit;
  `checkpoint_format` gptq/v1 or gptq_v2) or awq (4 bit, GEMM packing) are loaded by
  `quant/gptq_awq`. It unpacks codes and zero points exactly as AutoGPTQ/GPTQModel and AutoAWQ
  pack them:
  - GPTQ packs along the input dimension, and v1 stores zero − 1.
  - AWQ packs along the output dimension in the nibble order {0,2,4,6,1,3,5,7}.

  Each linear layer is converted once to an existing executable layout:

  | packed weights | engine layout | fidelity |
  |---|---|---|
  | 4-bit, zero = 8, contiguous groups of 32k | Q4_0 (d = scale, value d·(q − 8)) | bit-exact |
  | 8-bit, zero = 128, contiguous groups of 32k | Q8_0 (d = scale, q − 128) | bit-exact |
  | 4-bit asymmetric, contiguous groups of 32k | Q4_1 (d = scale, m = −scale·zero) | m rounded to fp16 |
  | act-order `g_idx`, other group sizes, 8-bit asymmetric | F16 | rounding to fp16 |

  Symmetry is verified on the actual zero points; the config's `sym` flag is not trusted.
  Other methods (bitsandbytes, fp8, compressed-tensors, …), 2/3-bit GPTQ and AWQ GEMV are
  rejected with a clear error.
- **Reason:** Spec §6 makes quantization a subsystem with format detection, scales, zero
  points, group size and layout, and keeps it out of model code. Repacking at load:
  - keeps the runtime unchanged: adapters and kernels only ever see `DType`s;
  - reuses AVX2 kernels that are already tested against references;
  - is exact for the most common published configuration (int4, symmetric, group 128, as in
    Qwen's official GPTQ-Int4 releases).

  GPTQ's scale is fp16, and Q4_0/Q8_0 store an fp16 `d`, so the mapping loses nothing.
- **Alternatives:**
  - Native GPTQ/AWQ kernels (int4 × fp32 with per-group zeros, or exllama-style).
    Potentially faster for asymmetric weights (no fp16 min rounding), but a second family of
    kernels to maintain and test, and the gain on CPU is unmeasured. This is the natural GPU
    path (Marlin-style) and can be added behind the same `PackedScheme` later.
  - Dequantizing everything to F16: simple, but 4× the memory.
  - Requantizing to K-quants: lossy twice.
- **Tradeoffs:**
  - Load time: each layer is unpacked once on the CPU (twice today, once to choose the
    target; this can be fused later).
  - Act-order models (`desc_act` with a permuted `g_idx`) run as F16, because 32-wide blocks
    cannot mix groups. Permuting input channels together with the previous layer's output
    could restore 4-bit storage; that is TODO.
  - Q4_1 needed AVX2 kernels to be competitive. They were added in this phase:
    `vec_dot_q4_1` = d·(q·x) + m·Σx, plus `dequant_q4_1`.
- **Evidence:** `test_gptq_awq`. `tools/make_tiny_quant_hf.py` packs the tiny Llama in five
  variants and writes an F32 reference computed independently in NumPy.
  - **Config.** GPTQ/AWQ parsing, and clear errors for bitsandbytes, fp8, 3-bit, AWQ GEMV
    and Marlin formats.
  - **Hand-built unpack.** A hand-packed layer for both layouts, and out-of-range `g_idx`
    rejected as corrupt.
  - **Fixtures:**

    | variant | layout | max \|w − ref\| | max \|logit − ref\| |
    |---|---|---|---|
    | GPTQ int4 sym g32 | Q4_0 | 0 | 0 |
    | GPTQ int8 sym g32 | Q8_0 | 0 | 0 |
    | GPTQ int4 asym g64 (v2) | Q4_1 | 2.4e-4 | 4.5e-3 |
    | AWQ int4 g32 | Q4_1 | 2.4e-4 | 3.7e-3 |
    | GPTQ int4 act-order g32 | F16 | 2.4e-4 | 1.5e-3 |

  - **Real checkpoints.** Qwen's official Qwen2.5-0.5B-Instruct-GPTQ-Int4 → Q4_0 ×168 and
    -AWQ → Q4_1 ×168. Both answer "7 × 6" with 42 and "capital of France" with Paris.
    Decode ITL p50, 10 threads: GPTQ 31.0 ms, AWQ 33.4 ms, GGUF Q4_K_M 29.0 ms.
  - **Load time.** Load is 2.0–2.4 s, against 0.16 s for GGUF (repacking on load). A first
    version took 5.3 s because it unpacked each layer twice with strided writes.

## DD-042: Mixture of experts in the generic Transformer: host routing, per-expert batched matmuls

- **Decision:** MoE is part of the one generic `Transformer`, configured by `MoeConfig`. There
  is no per-family code. A layer with expert tensors replaces its MLP with three steps:
  1. **Route.** Router logits = `matmul(xn, router)`. Each row takes a softmax over all
     experts and keeps the top-k (ties go to the lower expert id). The kept weights are
     renormalized when `normalize_topk` is set (Mixtral, Qwen3-MoE, Granite-MoE) and left
     as raw softmax probabilities otherwise (Qwen2-MoE).
  2. **Run experts.** Each *active* expert runs once on all rows routed to it: gather those
     rows, run gate/up matmuls on that expert's 2-D slice of the 3-D tensor, `act_mul`, then
     the down matmul. The result is scatter-added with the routing weights.
  3. **Shared expert.** Optional: a dense MLP on every row, scaled by
     σ(x · w_shared_router) (Qwen2-MoE).

  Expert tensors stay 3-D `[experts, rows, cols]` in the registry (zero-copy from GGUF). Per-expert
  2-D views are resolved once at init.
  - **Granite multipliers.** These are config scalars applied by the same code path:
    `embedding_scale`, `residual_scale` (each block's output before the residual add),
    `attn_scale` and `logit_scale`.
  - **Supported now:**
    - Mixtral (GGUF `llama` + experts; HF `mixtral`),
    - Qwen2-MoE / Qwen1.5-MoE (`qwen2moe` / `qwen2_moe`),
    - Qwen3-MoE (`qwen3moe` / `qwen3_moe`),
    - Granite / Granite-MoE (`granite`, `granitemoe`).
  - **HF layouts.** Per-expert HF tensors (`experts.N.w1`, `experts.N.gate_proj`, …) are
    stacked into the 3-D form at load. Granite's fused `input_linear` is split into
    gate/up views without copying.
- **Reason:** Spec §4 asks for a design where MoE fits in. Grouping rows by expert means
  prefill reads each active expert's weights once per batch, instead of once per token. For a
  single decode token, only k experts' weights are read, which is the whole point of MoE on a
  bandwidth-bound CPU.
- **Alternatives:**
  - **Dense evaluation of all experts, masked.** Simple, but reads every expert's weights:
    4× the bandwidth for Granite (32 experts, 8 active).
  - **A fused MoE kernel with expert-parallel threads.** A good next step for decode, where
    each expert's matmul is small. It would sit behind the same Layer data.
  - **llama.cpp-style `mul_mat_id`.** Equivalent to the per-expert batched matmul; a
    candidate for device backends.
- **Tradeoffs:**
  - Routing, gather and scatter run on the host CPU. Fine for the CPU backend; a GPU backend
    needs a route/gather/scatter op (Phase 28 design).
  - Decode would issue 3k small matmuls per layer (k = 8 for Granite). `Backend::matmul_many`
    runs all active experts' gate+up, then all down projections, in one parallel region each:
    2 dispatches per layer instead of 24. The CPU backend fuses jobs of ≤ 3 rows and falls
    back to per-job GEMM for larger ones.
  - Models that mix dense and MoE layers (`decoder_sparse_step` ≠ 1, `mlp_only_layers`,
    DeepSeek's dense first layers) are rejected with a clear error.
  - DeepSeek-V2/V3 additionally need MLA attention: not supported.
  - GPTQ/AWQ MoE checkpoints are rejected.
- **Evidence:**
  - **Tiny fixtures.** Four MoE fixtures (Mixtral-style, Qwen2-MoE with shared expert,
    Qwen3-MoE, Granite-MoE with multipliers) match the independent NumPy reference (prompt
    logits ≤ 2e-3, greedy continuation with chunked prefill identical). The reference
    `tools/ref_model.py` is written from the HF modeling code.
  - **Formats.** HF exports of the same fixtures load to identical logits: 0 for the Qwen
    MoEs, ≤ 1e-6 for Mixtral and Granite, from RoPE pairing order.
  - **Compatibility.** The suite (load, tokenizer, short and long generation, KV reuse,
    concurrent batching) passes for all four.
  - **Real model.** Granite-3.1-1B-A400M Q8_0 (32 experts, top-8) matches the NumPy reference
    golden: prompt logits within 5e-3, and 6 greedy tokens identical. The chat template
    renders identically to jinja2. It answers "capital of France" with Paris.
  - **Performance.** Decode is 59.7 → 29.1 ms/token with `matmul_many`. That is on par with
    dense Qwen2.5-0.5B Q8_0 (31.6 ms) at 2.6× the total parameters
    (docs/benchmarks.md, Phase 25).

## DD-043: Sampling pipeline: per-sequence Sampler, histogram pruning, portable RNG and exp

- **Decision:** Each request may carry `SamplingParams`: temperature, top_k, top_p, min_p,
  repetition/frequency/presence penalties with a context window, and a seed. A non-greedy
  request gets a `Sampler`, owned by its scheduler entry; greedy requests keep the old argmax
  path. Per token, on the logits row in place:
  1. **Penalties** over the last N context tokens. Counts come from sorting a reused copy of
     the window, with no hash map.
  2. **Greedy** (T ≤ 0 or top_k = 1): vectorized argmax, ties to the lowest id.
  3. **Temperature only**: one vectorized exp pass, then a walk to draw.
  4. **Otherwise**: one pass builds a 256-bin histogram of counts and probability mass over
     the top 40·T of logit range, and applies the min-p cut. This gives a conservative logit
     cutoff for top-k and top-p, so only the survivors are collected, sorted and used for the
     exact top-k, then the exact nucleus over the post-top-k / min-p distribution, then the draw.

  Order: penalties → top-k → temperature → min-p → top-p → draw (OpenAI applies temperature
  before top-p). The RNG is xoshiro256** seeded through splitmix64, and `exp` is our own
  polynomial. Both are exact functions of their input, so a seed reproduces the same tokens on
  Linux/gcc and Windows/MSVC (libm `exp` and `std::` distributions differ by platform).
- **Defaults:**
  - The HTTP API defaults to temperature 1.0, like OpenAI; `--temperature` changes it.
  - `dynalm run` defaults to greedy (a reproducible developer tool).
  - The in-process `GenerateParams` default is greedy.
  - Extensions accepted by llama.cpp/vLLM are parsed: `top_k`, `min_p`,
    `repetition_penalty`, `repeat_last_n`.
- **Reason:** Spec §26 asks for these modes, efficient sampling, and no temporary
  allocations per token. The Sampler's scratch (candidates, probabilities, history) is reused,
  so a warm Sampler allocates nothing.
- **Alternatives:**
  - A full sort of the 152k vocabulary per token for top-p: 2.8 ms worst case in the first
    version.
  - `std::nth_element` for top-k: 1.2 ms.
  - A dense per-token count array for penalties: 600 KB per sequence at 152k vocab.
- **Tradeoffs:**
  - Tokens below max − 40·T (relative probability < e⁻⁴⁰) are never sampled under top-k/top-p
    or min-p. That is negligible by construction; the temperature-only path keeps all tokens.
  - Worst case (flat synthetic logits, 152k vocab) is about 0.6–0.9 ms per token, 2–3% of a
    0.5B model's decode step.
- **Evidence:** `test_sampling`:
  - RNG output pinned to an independent Python implementation;
  - greedy and top_k = 1 equivalence;
  - empirical distributions (200k draws) within 0.006 of the exact softmax at T = 0.5, 1 and 1.7;
  - top-k / top-p / min-p support sets and renormalized ratios;
  - exact penalty arithmetic, including the window;
  - on a 152k vocabulary, every sampled token lies in the brute-force nucleus;
  - seed determinism, and parameter validation;
  - engine requests reproduce with a seed and fail cleanly on invalid parameters.

  `test_server`: API parsing of all fields, 400 on invalid values, a seeded HTTP request
  reproduces. `bench_sampling`: greedy 440 → 70 µs, top-p 2.8 → 0.9 ms, top-k 1.2 → 0.84 ms.

## DD-044: Speculative decoding: drafters, one-pass verification, KV rollback

- **Decision:** The runtime has the three pieces speculative decoding needs, plus a working
  single-sequence implementation built from them:
  1. **Multi-row logits.** `SeqBatch::logits_last` makes one forward pass score every drafted
     position. The batched forward already handles k + 1 rows of one sequence.
  2. **KV rollback.** `KvBlockTable::truncate` drops the K/V of rejected drafts and unmaps
     whole blocks.
  3. **Verification.** `Sampler::sample_speculative` uses the target's *own* sampling
     pipeline (penalties, top-k, top-p, min-p, temperature):
     - greedy accepts a draft iff it is the argmax;
     - sampling accepts draft d with probability p(d), otherwise draws from p without d.

     This is Leviathan et al.'s rule for a one-hot draft distribution, so outputs follow the
     target distribution exactly.

  Drafters implement one method, `propose(context, k)`:
  - `NgramDrafter` (prompt lookup) continues the latest earlier occurrence of the context's
    longest suffix (4 down to 2 tokens). It needs no model or memory.
  - `ModelDrafter` runs a small model with its own KV, greedy. Its vocabulary must match the
    target's token for token, which is checked at creation. It rolls its KV back to the prefix
    it shares with the accepted context. Drafting failures stop drafting, never the request.

  `SpeculativeGenerator` runs propose → verify (1 target pass) → accept a prefix, plus the
  target's correction or a bonus token → rollback. So each target pass yields 1..k+1 tokens.
  It is exposed as `dynalm run --spec ngram|DRAFT.gguf --spec-k K`, and measured by
  `bench_speculative`.
- **Reason:** Spec §47 asks for extension points for speculative decoding, draft models and
  KV rollback. On a CPU, decode is memory-bound: scoring k + 1 positions reads the weights
  once, costing little more than one token. Every accepted draft is a token for nearly free.
- **Not yet:** speculation inside the continuous-batching scheduler. The pieces are there: a
  decode entry contributing 1 + k rows, its own draft state, and rollback through
  `SequenceState`. The policy is the open question: with many concurrent sequences the batch
  already amortizes weight reads, so speculation helps mostly at low concurrency. Also
  deferred: tree drafts (Medusa/EAGLE), stochastic draft-model proposals (the rejection rule
  generalizes to max(0, p − q)), and an API switch.
- **Tradeoffs:**
  - Greedy speculative output equals plain greedy *up to float rounding*. Verification runs
    the multi-row matmul path, which accumulates in a different order than the one-row decode
    path (DD-031), so a near-tie could flip. The tiny-model tests show exact equality, and so
    does `bench_speculative` on Qwen2.5 (column "exact").
  - Prompt lookup only pays off when the output repeats the context.
  - A draft model costs its own decode; it pays off when it is several times cheaper than the
    target and agrees often.
- **Evidence:** `test_speculative`:
  - n-gram proposal logic;
  - greedy output identical to plain greedy with n-gram, self-draft (100% acceptance) and a
    different same-vocabulary model at k = 1, 3, 6;
  - all KV returned after rollbacks;
  - verification reproduces the target distribution within 0.005 (200k draws) for a
    high-probability, a low-probability and an impossible (outside top-k) draft, with an
    acceptance rate equal to p(draft);
  - seeded sampling reproduces;
  - mismatched draft vocabularies are rejected.

  `bench_speculative` results are in docs/benchmarks.md.

## DD-045: GPU-ready backend boundary, proven with a memory-guarded test backend

- **Decision:** Phase 28 makes the backend the only party that touches tensor memory, and
  adds what device backends need. No GPU code is written (spec: "Do not implement CUDA").
  - **Memory ops:**
    - `upload` places weights (zero-copy on CPU);
    - `allocate` provides all scratch and KV cache;
    - `copy` does KV copy-on-write (it was a raw `memcpy`);
    - `download` is the explicit device→host point (logits, MoE router logits, the
      shared-expert gate);
    - `host_accessible()` lets the CPU skip both upload and download, so the CPU path does
      not pay for the abstraction.
  - **Data-movement ops:** `fill`, `gather_rows` and `scatter_add_rows` replace the
    Transformer's direct `memset`/`memcpy` (logits-row gather, MoE token permutation and
    weighted combine).
  - **Kinds:** `DeviceType` and `BackendKind` carry CUDA/HIP/Metal/Vulkan.
    `create_backend(kind)` builds CPU and reports the others as not built. `EngineOptions`
    and `--backend` select the kind.
  - **Guide:** `docs/gpu-backend.md` is the implementation guide.
- **Reason:** Spec §45: a backend interface with allocate/free/memcpy/gemm/attention/
  rmsnorm/rope/synchronize, implemented by CpuBackend and later by CUDABackend. Spec §46:
  no CUDA-specific assumptions in the model or scheduler. Such a boundary is only real if
  something checks it. The guard works like this:
  - `GuardedBackend` (tests) allocates pages with `mprotect(PROT_NONE)` and unprotects them
    only inside its own ops.
  - Any direct host access by the runtime faults.
  - A death test proves the guard is armed.
- **Alternatives:**
  - Code review only: misses the next `memcpy` someone adds.
  - Implementing a real device backend now: excluded by the spec.
  - A wrapper that copies every tensor to "device" per op: proves nothing about access
    discipline.
- **Tradeoffs:**
  - Routing (MoE top-k) and sampling stay host-side. That costs one small download per layer
    or step on a GPU; fused device ops can replace them behind the same interfaces later.
  - Small per-step metadata (ids, positions, block tables) is passed as host spans, as
    launch arguments.
  - The CPU path is unchanged in cost: Granite-MoE decode 29.1 → 27.8 ms/token (noise), Qwen
    prefill unchanged.
- **Evidence:** `test_device_backend`:
  - registry behaviour (CPU built; GPU kinds give clear kUnsupported; unknown names
    rejected);
  - the guard faults on host access (death test);
  - all 11 tiny architectures (7 dense, 4 MoE) generate identical tokens and bit-identical
    logits on the guarded backend vs CPU;
  - continuous batching with shared prefixes (prefix cache, partial-block copy-on-write) and
    speculative decoding (multi-row logits, KV rollback) run on the guarded backend with
    results identical to CPU.

  352 tests pass.

## DD-046: DynaLM: name, three-OS installs, CI matrix, NEON kernels

- **Decision:** The project ships as **DynaLM**.
  - **Name.** The user-facing names are `dynalm`: the command, the version banner,
    `DYNALM_*` environment variables, `dynalm_*` Prometheus metrics, `owned_by` in the API,
    and the Docker images. Internal code names (the `engine::` namespace, `ENGINE_*` compile
    macros, library targets) are unchanged: renaming them touches every file and gains users
    nothing.
  - **Platforms.**
    - SIMD options follow the target CPU: AVX2/AVX-512/AMX only on x86-64, NEON only on
      ARM64. `-mavx2` would be a compile error on Apple Silicon.
    - macOS has its own platform code: CPU topology including P/E cores via `sysctl`, memory
      via Mach, process RSS via `task_info`.
    - A NEON kernel tier covers fp32/fp16/bf16 dot products, the GEMM panel, and Q8_0, Q4_0,
      Q4_1, Q5_0, Q4_K and Q6_K dot products and dequantization. Other types use the generic
      tier.
  - **Install.**
    - `scripts/install.sh` (Linux, macOS) and `scripts/install.ps1` (Windows, through
      Visual Studio's toolchain) build a release binary with a static C/C++ runtime
      (`DYNALM_STATIC_RUNTIME`) and run `cmake --install`.
    - CPack makes `dynalm-<ver>-<os>-<arch>` archives.
    - A two-stage `Dockerfile` (amd64/arm64) serves on `0.0.0.0` through `DYNALM_HOST`.
    - The one configure-time download (cpp-httplib, hash-pinned) now retries, and can come
      from a local file (`DYNALM_HTTPLIB_HEADER`).
  - **CI** (`.github/workflows/ci.yml`):
    - Linux x86-64 with gcc, clang, ASAN/UBSAN and TSAN;
    - native Linux ARM64;
    - Windows MSVC;
    - macOS on Apple Silicon;
    - the install scripts on all three operating systems;
    - the Docker build.

    `release.yml` packages all four platforms on a version tag.
- **Reason:** DynaLM targets installs on Linux, macOS and Windows. Without CI,
  only the developer's machine is ever tested. Without NEON, Apple Silicon (the most common
  Mac) would run only scalar kernels.
- **Alternatives:**
  - Prebuilt binaries only: needs CI and signing first.
  - A Homebrew formula or winget manifest: natural next steps once releases exist.
  - Renaming internal names too: churn, no user value.
- **Tradeoffs:**
  - Builds need network access once (cpp-httplib) unless the header is provided.
  - NEON performance is not benchmarked yet. Correctness runs on GitHub's ARM64 and Apple
    Silicon runners: the full test suite passes on both, plus the installers on all three OSes
    (CI, October 2026).
  - Windows Smart App Control may block freshly built, unsigned binaries on some machines.
- **Evidence:**
  - **Linux x86-64.** 352/352 tests with gcc 13 and with clang 18, a clean build
    (`-Wall -Wextra`, no warnings).
  - **Linux ARM64** (emulated: Docker + QEMU):
    - builds with `kernels: generic neon`;
    - `test_kernels` passes with the NEON tier checked against the dequantization
      reference for every type;
    - 339/340 tests pass. The exception is one real-model concurrency test whose 120 s
      request timeout expires under emulation (277 s run). The same model's single-sequence
      and KV-reuse tests pass on ARM64.
  - **Windows.** `install.ps1` built and installed a static-runtime `dynalm.exe`, which
    answered "capital of France" natively at 33.9 tok/s (Qwen2.5-0.5B Q4_K_M).
  - **Linux install.** `install.sh` installs into a fresh container.

## DD-047: `dynalm pull` downloads with the system curl and pre-checks the header

- **Decision:** `dynalm pull <link>` resolves Hugging Face links to download URLs, fetches the first 256 KiB
  with a range request, and refuses a GGUF whose `general.architecture` DynaLM has no adapter for. It then
  downloads to `<file>.part` with resume and retries, renames it into place, and checks every tensor type.
  Transfers run the system `curl` as a child process (no shell). `HF_TOKEN` is passed through a temporary
  owner-only header file, and only to huggingface.co.
- **Reason:**
  - Users copy model links from Hugging Face; without a pull command they needed curl, wget or Python.
  - A real case: an 11 GB `qwen35` GGUF was downloaded and then listed as unsupported. The header check
    now refuses it after 256 KiB.
  - `curl` ships with Windows 10 (1803+), macOS and every Linux distribution, and brings TLS, proxies,
    redirects, resume and retries.
- **Alternatives:**
  - cpp-httplib with OpenSSL: adds OpenSSL to every static build on three OSes; no resume or proxy logic.
  - libcurl linked in: a large build dependency for one command.
  - Platform APIs (WinHTTP, NSURLSession): three implementations to maintain.
  - `huggingface_hub` (Python): DynaLM has no Python at runtime.
- **Tradeoffs:**
  - Needs `curl` on PATH (a clear error says so if missing).
  - The pre-check sees only header metadata. Tensor types are listed after the tokenizer arrays, so
    IQ-quantized files of a supported architecture are only reported after the download.
  - Single files only. No repo browsing or sharded GGUFs yet (see ROADMAP.md).
- **Evidence:**
  - Windows, real Hugging Face:
    - the `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` IQ3_S link was refused in about 1 s
      ("architecture 'qwen35' is not supported");
    - a repo-only link got the hint to pick a file;
    - `SmolLM2-135M-Instruct-Q8_0.gguf` downloaded and verified "ok", and a second run skipped it;
    - a missing file reported the HTTP error with the gated-model hint.
  - Unit tests (`ModelSource.*`, 5): link forms, the header peek on truncated and HTML input, and the
    support verdict on synthetic GGUFs with an unknown architecture and an iq3_s tensor.

## DD-048: `dynalm rm` deletes only model artifacts, and asks first

- **Decision:** `dynalm rm <model>... [-y]` (alias `delete`) resolves a name the way `dynalm list` prints
  it (with or without `.gguf`, or a `.part`) under the models directory, or a path. It deletes only:
  - regular files ending in `.gguf` or `.gguf.part`;
  - directories that `hf::is_hf_model` recognizes.

  Anything else is refused. It asks `[y/N]` (EOF means no) unless `-y` is given. `dynalm list` now
  shows partial downloads so they can be found and removed. `dynalm pull --check` checks support
  without downloading.
- **Reason:**
  - Models are multi-GB, and users asked for a remove command to go with `pull`.
  - Accepting paths makes a typo potentially destructive, so the target type is checked first.
- **Alternatives:**
  - Delete any path: simpler, but `dynalm rm ~/Documents` must never work.
  - A model registry or manifest (Ollama-style blobs): DynaLM uses plain files users can see, so the
    filesystem is the registry.
- **Tradeoffs:**
  - On Windows, a model that a running server has mmapped cannot be deleted; the error suggests
    `dynalm stop`.
- **Evidence (Windows and Linux/gcc):**
  - "n" keeps the model; `-y` deletes it.
  - A `.gguf.part` is deleted by its base name.
  - An HF directory is deleted recursively.
  - `notes.txt` and a plain folder are refused (exit 1).
  - An unknown name gives "no model".
  - `pull --check` was used on 35 Hugging Face repositories:
    - Qwen3.5/3.8 (`qwen35`), Qwen3.8-Flash (`qwen4exp`) and LFM2 (`lfm2`) were refused;
    - Qwen3, Qwen3-MoE, Qwen2.5, Llama 3.x, Gemma 2/3, Phi-3/3.5, Mistral, Mixtral and
      DeepSeek-R1-Distill passed.
  - Downloaded and run end to end:
    - Qwen3-0.6B Q8_0: 22.1 tok/s, correct answer;
    - Llama-3.2-1B Q4_K_M: "The capital of France is Paris.", 13.2 tok/s with 1.6 GB free RAM.

## DD-049: Interactive chat re-sends the history and relies on the prefix cache

- **Decision:**
  - `dynalm chat` (and `dynalm run` without `-p`) is a read-eval-print loop on one in-process Engine.
  - Each turn sends system + history + the new message through `generate_chat`.
  - The history is trimmed from the oldest exchange when the rendered prompt plus 32 tokens would exceed
    the context.
  - Thinking (`<think>...</think>`) is shown, but dropped from the stored history.
  - Chat has its own defaults: temp 0.8, top-k 40, top-p 0.9, repeat-penalty 1.1 and 2048 tokens.
    One-shot `run` stays greedy.
- **Reason:**
  - Users compared DynaLM with `ollama run`. Its "one answer, then exit" model reloaded the weights for
    every question and kept no memory of the conversation.
  - Re-sending the whole conversation keeps chat stateless on the engine side, and the radix prefix cache
    (DD-031) makes it cheap. A follow-up with 101 prompt tokens gets its first token in 254 ms
    (Qwen2.5-1.5B Q4_K_M), because only the new message is prefilled.
- **Alternatives:**
  - A persistent sequence whose KV the chat appends to: faster in principle, but it needs a new
    engine API (pinned sequences), and the cache already captures most of the gain.
  - A chat client for a running `dynalm serve`: the right shape for a resident multi-model server, which
    is still a roadmap item. It can reuse this loop.
- **Tradeoffs:**
  - Trimming is by whole exchanges. A single message longer than the context is rejected with a hint
    to raise `-c`.
  - `/think` only toggles Qwen3, the family with a documented `/no_think` switch.
- **Evidence:**
  - Scripted sessions on Windows:
    - Qwen2.5-1.5B remembered a name across turns and forgot it after `/clear`;
    - `"""` multi-line input, `/set`, `/show`, `/stats` and unknown commands behaved as documented;
    - with Qwen3-4B, `/think off` gave direct answers with the empty think block hidden;
    - Gemma reports that it has no thinking switch.
  - `ThinkFilterTest` (3 tests) covers chunk splits, including a `</think>` split across deltas. That
    case was a bug the test found.
  - 360/360 Linux tests pass.

## DD-050: Measure before optimizing: in-engine accounting, counters and a bottleneck classifier

- **Decision:** Performance-program phase P1 adds measurement only. There are no optimizations.
  - **Always-on scheduler accounting:** a few clock reads per step cover plan, prefix lookup, KV
    reservation, forward (split by decode-only / prefill-only / mixed steps), sampling, emit and
    prefix insert. Tokenization is timed in the engine.
  - **Opt-in profiling** (`Engine::set_profiling`, applied on the scheduler thread): per-op
    forward time, plus thread-pool region count, region time, the caller's **tail wait** at
    region ends, and worker sleeps.
  - **`platform/perf_counters`:**
    - hardware counters opened **per thread** with `perf_event_open`, because inherited counters
      only fold into the parent when a thread exits, and the worker pool never exits;
    - OS accounting for context switches and faults;
    - the CPU clock.
  - **`bench/analysis`:**
    - a measured DRAM read ceiling;
    - a modelled bytes-per-decode-step (weights, MoE touched experts, KV);
    - a rule-based classifier with fixed thresholds and priority (docs/performance.md).
  - **Tooling:** `dynalm benchmark --prompt-mix`, per-point diagnostics in the JSON lines,
    `tools/bench_report.py --svg` (eight graphs, standard library only) and `tools/perf_sweep.sh`.
- **Reason:**
  - The program's objective is aggregate output tok/s, and the existing benchmark measured only
    the client side.
  - Without per-stage time, thread-pool behavior and a bandwidth reference, any optimization
    would be a guess. Example: the first probe showed 8-row decode steps costing 2.5× a 1-row
    step while using about 20% of memory bandwidth. That points at the expand path, not at DRAM.
- **Alternatives:**
  - External profilers only (perf, VTune, WPA): necessary for deep dives, but they don't attach
    to every benchmark point and aren't available everywhere.
  - Always-on per-op profiling: costs ~1 µs per op per step, so it is kept opt-in.
  - Real DRAM counters (uncore IMC): need root and bare metal. The model plus a measured ceiling
    works everywhere and is labelled as an estimate.
- **Tradeoffs:**
  - Hardware counters are unavailable on this development machine (Windows; WSL2 exposes no
    PMU), so CACHE_BOUND and IPC-based rules cannot fire here. They work on bare-metal Linux.
  - The classifier is triage, not proof: thresholds are round numbers, and reports quote the
    underlying figures.
  - Windows `CallNtPowerInformation` often reports a nominal clock.
- **Evidence:**
  - 16 unit tests: classifier rules, traffic model, counters, pool stats, the bandwidth probe,
    and an end-to-end diagnostics point.
  - 372/372 tests pass under linux-release and TSAN.
  - The baseline sweep (docs/benchmarks.md P1) produced its first actionable result: batched
    Q4_K_M decode is compute-bound on K-quant unpacking. A 7-row step takes 109.5 ms vs 60.0 ms
    for Q8_0, at 3.7 of 19 GB/s.

## DD-051: An execution planner owns every per-step execution decision

- **Decision:** A new layer, `src/execution/BatchPlanner`, sits between the scheduler and the model
  runtime.
  - For each step the scheduler calls `planner().plan(batch)`, which returns an `ExecutionPlan`:
    - the phase (decode / prefill / mixed);
    - rows, decode and prefill rows, logit rows;
    - the longest attention span, and the attention work (Σ span);
    - a backend-level `KernelPlan`.
  - `Transformer::forward_batch` hands the `KernelPlan` to `Backend::set_kernel_plan` before running.
  - `KernelPlan` lives in the backend layer, so backends never depend on the planner, the scheduler
    or model families. It carries:
    - the matmul expand threshold and GEMM K-block (which already existed);
    - attention strategy, decided separately for full-causal and sliding-window layers
      (`per_pair` / `split_k`);
    - the split-K chunk.
  - The planner gets the machine from a `HardwareProfile` (threads, physical/P/E cores, L2/LLC,
    ISA).
  - Direct callers (generator, speculative decoding, tests) pass no plan; the transformer then
    plans with the same planner.
  - A backend that never receives a plan uses `KernelPlan::defaults()`, where `kAuto` applies the
    same rule per call.
- **Reason:**
  - Before P2, these decisions were scattered inside kernels. Examples: the split-K rule was
    re-derived in every attention call; expand/GEMM thresholds came from backend members and
    environment variables.
  - The coming phases add phase- and shape-specific kernels (multi-row decode, GQA attention, a
    dynamic split) whose choice needs batch-level facts the kernels don't have: phase, row mix,
    per-step context distribution. One planner makes those choices testable and visible
    (`steps_split_attention`). Later phases (adaptive scheduling, autotuning) can then change
    them without touching kernels.
- **Alternatives:**
  - Keep heuristics in the backend: it cannot see the batch, only one op at a time.
  - Put the planning in the scheduler: that would leak hardware and kernel knowledge into a
    model-agnostic component.
  - A plan per op instead of per step: more flexible, but no current decision needs it.
- **Tradeoffs:**
  - One more struct per step (planning is a single pass over the sequences, microseconds).
  - The plan is computed for the whole step, so it cannot vary per layer except through the
    full/window split.
- **Evidence:** see P2 in docs/benchmarks.md (bit-identical outputs; neutral performance).

## DD-052: Compute threads opt out of OS power throttling

- **Decision:** `platform/thread_qos`.
  - `Engine::create` calls `request_full_speed_process()`. On Windows this sets `ProcessPowerThrottling`
    with `EXECUTION_SPEED` control and state 0, which opts out of EcoQoS.
  - Every thread-pool worker and the engine's scheduler thread call `request_full_speed_thread()`:
    - Windows: `ThreadPowerThrottling`, same opt-out.
    - macOS: `QOS_CLASS_USER_INITIATED`, eligible for performance cores.
    - Linux: no-op.
- **Reason:** Measurement found it (P1). The same engine code ran its decode forward pass at
  25.1–26.0 ms/step inside `dynalm run`, which streams output to a console, but at 37–45 ms inside
  `dynalm benchmark`, a quiet process. With 2 threads the gap was 39 vs 74 ms.
  - Ruled out one by one: profiling, the clock sampler, prompt length, KV size, stdout streaming
    and thread count.
  - Only the process's apparent activity differed. Windows 11 power-throttles processes it
    considers background, which moves them to efficiency cores and lower clocks. A server
    (`dynalm serve`) is exactly such a process.
- **Alternatives:**
  - Raise thread priority: doesn't affect EcoQoS placement, and can starve the UI.
  - Pin threads to P-cores: P10 work; on its own it is undone by throttling and loses E-core
    throughput.
  - Document "run in the foreground": not actionable for a server.
- **Tradeoffs:**
  - Higher power draw while generating. That's the intended trade for an inference engine, and
    idle workers still sleep after ~100 µs of spinning.
  - Linux has no equivalent hint; there, P10 handles placement.
- **Evidence:**
  - After the change, back-to-back runs give `run` 25.9/26.8 vs benchmark 25.8/25.4 ms per step
    at 10 threads, and 38.2/37.8 vs 39.4/38.0 at 2 threads.
  - Serving throughput, Qwen2.5-0.5B Q4_K_M, prompt 128 / output 64:

    | | c=1 | c=8 | c=16 | c=64 |
    |---|---|---|---|---|
    | tok/s before | 18.6 | 40.1 | 48.0 | 54.7 |
    | tok/s after | 31.5 | 71.1 | 78.9 | 88.5 |
    | change | +69% | +77% | +64% | +62% |

## DD-053: int8 activations for decode-shaped matmuls (≤ 4 rows), except the FFN down projection

- **Decision:**
  - **Activation format:** for a matmul with at most `int8_decode_max_rows` rows (default 4), the
    CPU backend quantizes each activation row into 32-value blocks: int8 codes, an fp32 scale, and
    the code sum.
  - **Kernels:** each weight row is computed with integer dot products against the packed weights
    (`maddubs`/`madd`, 32 multiply-adds per instruction). Each weight block is unpacked once and
    reused for up to 4 activation rows.
  - **Formats:** Q8_0, Q4_0, Q5_0, Q4_K and Q6_K have kernels.
    - The AVX2 tier accelerates them. The generic tier has scalar reference versions with the
      identical integer arithmetic.
    - Activation quantization is bit-identical across tiers (round-to-nearest-even).
    - Other formats keep the fp32 paths.
  - **Gating:** the path is used only on tiers with SIMD kernels (AVX2; NEON kernels are not written
    yet). The FFN down projection always keeps fp32 activations (`int8_ffn_down = false`).
  - **Control:** `--int8-decode N` (run/serve/benchmark), `EngineOptions::int8_decode_rows`, or
    `DYNALM_INT8_DECODE_ROWS`. 0 turns it off.
- **Reason:**
  - **P1/P3 measurements:** single-row fused fp32 kernels converted every weight to fp32 and
    reached only 7–14 GB/s on Q4_K layer weights, against a ~19 GB/s DRAM ceiling. Matmuls are
    ~84% of decode time.
  - **Integer dot speed:** 17–19 GB/s at M=1–2. It is the fastest path for every tensor up to M=4.
    From M=6 the fp32 expand/GEMM path is faster, which sets the default threshold.
  - **The `ffn_down` exclusion:** the gated activation feeding `ffn_down` carries outlier
    channels. With it quantized, Qwen3-4B perplexity rose 5.4%; without it, 0.9%.
- **Alternatives:**
  - Per-256 activation blocks (as llama.cpp's Q8_K): coarser, so less accurate with outliers.
  - Keep fp32 everywhere: leaves single-stream K-quant decode compute-bound.
  - Per-model calibration at load: costs seconds of startup.
  - VNNI (`dpbusd`): the P9 follow-up.
- **Tradeoffs:**
  - **Quantization error:** activations carry ~0.4% relative error per element, about 1000× the
    summation-order noise.
  - **Batch invariance:** a request decoded alone (1 row, int8) and inside a 6-row batch (fp32)
    can now produce different greedy text. Bit-for-bit batch invariance (DD-031) holds only with
    `--int8-decode 0`; the compatibility test runs that way. The default favours throughput, the
    program's objective.
  - **Large batches:** above 4 rows the int8 kernel loses to fp32 GEMM. A weight-row-tiled int8
    kernel and VNNI could extend it (P9).
- **Accuracy contract:**
  - Default only if, on every measured model, perplexity changes by at most +1% and mean
    KL(fp32 || int8) is at most 0.0025 nats.
  - Measured with `bench_int8_accuracy`: 189–213 positions of fixed English text, teacher-forced,
    plus a 65-token greedy run. "Noise floor" is the fp32 path with a different summation order.

    | model (Q4_K_M) | noise floor PPL | int8 (no exclusion) PPL | int8 except ffn_down: PPL / mean KL / top-1 / greedy identical |
    |---|---|---|---|
    | Qwen2.5-0.5B | −0.01% | −0.07% | −0.01% / 0.0021 / 98.4% / 65 of 65 |
    | Qwen2.5-1.5B | −0.01% | +0.72% | +0.54% / 0.0011 / 98.4% / 29 of 65 |
    | Granite-3.1-1B-A400M | −0.01% | +0.76% | +0.76% / 0.0019 / 99.1% / 65 of 65 |
    | Qwen3-4B | +0.06% | **+5.37%** | +0.90% / 0.0011 / 98.4% / 65 of 65 |
- **Evidence:** see P3 in docs/benchmarks.md.
  - Tests: 7 new (quantization bounds and tier identity; every format × M=1–4 against a
    dequantized reference; the backend path against fp32).
  - 392/392 pass under linux-release and ASAN/UBSAN.

## DD-054: GQA-grouped attention with head sub-groups; P4 weight packing deferred on evidence

- **Decision (P5):**
  - **Unit of attention work:** one (query row, KV head) pair, covering every query head that
    shares the KV head. Each K and V vector is fetched once and reused from L1 by the group,
    instead of being fetched again for each query head.
    - Per head, the arithmetic is exactly that of the former per-head pass: the same dot/axpy
      primitives over positions in ascending order. Results are identical.
  - **Small batches:** when there are fewer units than threads, each group's heads are divided
    into sub-groups, so every thread has work while heads within a sub-group still share fetches.
  - **Long contexts:** split-K as before, now over units. The planner's split rule counts
    (row, KV head) units (DD-051).
- **Reason:**
  - Measured: decode at 4K context spent ~17 ms/step more than at 128 tokens. Yet the KV it needs
    is ~50 MB (~2.5 ms at 20 GB/s).
  - Qwen2.5-0.5B has 7 query heads per KV head, and each query head re-read the same KV.
- **Alternatives:**
  - Convert K/V blocks to fp32 once per group: fewer conversions, but changes the summation and
    needs scratch.
  - Grouping only (no sub-groups): measured a +16% decode-step regression at 128 tokens, because
    one row gave 2 tasks for 10 threads.
- **Tradeoffs:**
  - Sub-groups re-read KV once per sub-group, which is cheap at the short contexts where they
    apply.
  - A dynamic split-K chunk (adapting chunk size to the context mix) is left for P7.
- **Evidence:**
  - Tests: all 393 pass, including golden references. `GqaAttention.OneRowMatchesNaiveInEveryLayout`
    covers sub-grouped, split-K and auto layouts with F16/F32 KV against a naive softmax.
  - Benchmarks: `results/p5-{before,after}-{c1,c8}.jsonl`.

    | Qwen2.5-0.5B Q4_K_M | decode step before → after | other |
    |---|---|---|
    | c=1, 128 tokens | 22.7 → 23.4 ms | noise |
    | c=1, 1,024 tokens | 25.1 → 25.3 ms | attention −12% |
    | c=1, 4,096 tokens | 40.2 → 31.7 ms (−21%) | TTFT 32.6 → 28.3 s (−13%) |
    | c=8, 1,024 tokens | 34.2 → 32.4 ms | +8% output tok/s |

- **P4 (weight packing): evaluated, deferred.** P1 had shown batched Q4_K_M decode far slower than
  Q8_0 (109.5 vs 60.0 ms at 7 rows), which suggested repacking K-quants at load. Re-measured after
  DD-052 and DD-053:

  | format | c=8 tok/s | c=16 tok/s |
  |---|---|---|
  | Q4_K_M | 89.7 | 94.2 |
  | Q8_0 | 76.0 | 82.1 |
  | F16 | 72.0 | 82.2 |

  At c=16 all formats take ~83–85 ms per step, so the cost is the fp32 batched arithmetic, not
  K-quant unpacking. The P1 gap was mostly the power-throttling artifact. A packed layout would
  not address a measured bottleneck now, so it is deferred and re-evaluated with P9 (int8/VNNI for
  larger batches), where a packed int8 layout may pay off.

## DD-055: Decode synchronization: measured small; wake-ups only for sleeping workers

- **Decision (P6):** `ThreadPool::parallel_for` publishes a job with a sequentially consistent
  epoch increment. It takes the mutex and notifies only when the new `sleepers_` count says a
  worker is blocked.
  - Workers announce themselves before re-checking the epoch under the lock (a two-flag
    handshake), so no wake-up is lost.
  - Fusing the q/k/v and gate/up matmuls into shared regions was evaluated and not done.
- **Reason:** `bench_thread_pool` (10 threads, i7-1255U) measured the actual fork/join cost.
  - Back-to-back regions, the case inside a forward pass: 2.5 µs p50 (p99 66 µs, OS preemption).
  - With 207 regions per Qwen2.5-0.5B decode step, that is ~0.5 ms of a 23 ms step (~2%).
  - The thread-pool tail wait reported by P1 (5–10% of forward time) comes mostly from uneven
    P-core/E-core progress and preemption (P10 territory), not from dispatch.
- **Evidence:**
  - Region cost 2.5 → 2.1 µs p50 (−16%); small body 3.0 → 2.4 µs. End to end that is ≤ 0.1 ms
    per step, below this laptop's run-to-run noise.
  - TSAN: 393/393.
- **Alternatives:**
  - Region fusion (q/k/v, gate/up): it would save ~48 regions × ~2 µs ≈ 0.1 ms per step, and
    needs the int8 path wired into `matmul_many`. Not justified by the measurement.
  - Lock-free work stealing: no measured contention to remove.
- **Tradeoffs:** none measurable. The handshake relies on seq_cst ordering, which is documented
  next to the code.

## DD-056: KV traversal by block runs; per-group fp16 conversion; vector exp; dynamic split-K; Q8 KV deferred

- **Decision (P7):**
  - **Block-run traversal:** attention walks the KV in runs of contiguous positions (one block of
    one head). Per run:
    - one kernel call computes all scores (`attn_scores_f16/f32`);
    - one call accumulates all weighted values (`attn_accum_f16/f32`, accumulator held in
      registers);
    - this replaces a function call, an offset computation and a conversion per position and head.
  - **Shared conversion:** when several query heads share the KV head, each fp16 run is converted
    to fp32 once and all heads read the copy.
  - **Vector exp:** softmax weights use `exp_nonpos`, moved to `common/fast_exp.h`. It is the
    sampler's libm-free exp (~1e-7 relative error) and vectorizes.
  - **Dynamic split-K:** the planner sizes the chunks so (row, KV head) units × chunks fill whole
    waves of the pool, never below 256-token chunks (spec §8).
  - **Block size:** stays at 16 tokens.
  - **Q8 KV:** deferred.
- **Reason:** `bench_decode_context` with per-op timing showed attention growing to 9.7 ms per
  decode step at 4K context (Qwen2.5-0.5B), about 5 GB/s effective against a 20 GB/s ceiling.
  Every other op stayed flat and at bandwidth.
- **Evidence (decode step at 4K context, two runs each):**

  | step | attention, ms/step | whole step, ms |
  |---|---|---|
  | before P7 | 9.7 | 32.8 |
  | block runs | 7.1 | — |
  | + shared conversion + vector exp | 6.6 | 29.2–29.5 |
  | + dynamic chunks | 6.8–6.9 (4K), 2.1–2.4 at 1K (was 2.6–2.8) | 29.6–30.0 |

  - Block size 16 vs 64: 33.9 vs 32.0 ms at 4K (~5%, near noise), identical elsewhere. 16-token
    blocks keep prefix-cache reuse finer, so 16 stays.
  - Q8 KV would halve KV bytes, but attention now moves ~7.6 GB/s: it is not bandwidth-bound, so a
    lossy format (with its accuracy risk) is not justified yet.
  - 393/393 tests pass, including golden references and the naive-softmax GQA tests.
- **Tradeoffs:**
  - The shared conversion and `exp_nonpos` change attention numerics by float rounding against
    the per-head fp16 path. Results still depend only on a row's own data, so batch invariance
    (DD-031) holds.
  - fp32 conversion scratch is 16 × head_dim floats per thread.

## DD-057: Prefill fp32 GEMM is near practical peak; no P8 change; the lever is int8 (P9)

- **Decision (P8):** no change to the prefill GEMM. The hypothesis that it lacked M-blocking was
  tested and refuted.
- **Evidence:**
  - `bench_profile` (Qwen2.5-0.5B Q4_K_M, 256-token prefill, 10 threads): 639 ms; FFN matmuls are
    76% of it. Per op, over all 24 layers:

    | op | time | GFLOP | GFLOP/s |
    |---|---|---|---|
    | gate+up | 304 ms | 107 | ~350 |
    | down | 180 ms | 53.5 | ~300 |
    | qkv | 39 ms | 12.7 | ~320 |
    | attn_out | 32 ms | 9.9 | ~310 |

  - `bench_decode_matmul` with DRAM-resident weights: the expand GEMM scales from ~290 GFLOP/s at
    32 rows to ~330 at 64 and ~360 at 256 rows (`ffn_gate` 0.99 / 1.71 / 6.22 ms). Activations
    are not re-streamed from L3 in a way that hurts at large M.
  - A first reading of ~15 GFLOP/s was an arithmetic slip (×24 layers omitted). It is recorded
    here because it nearly motivated the wrong optimization.
- **Reason:**
  - On this 15 W i7-1255U, practical all-core fp32 FMA throughput under sustained load is roughly
    450 GFLOP/s: 2 P-cores at 32 FLOP/cycle and 8 E-cores at 16 FLOP/cycle, at ~3 GHz sustained.
    The 4×3 tile reaches ~70–80% of that.
  - Remaining fp32 headroom (tile shape, packing) is ~10–20%, while AVX-VNNI (`vpdpbusd`, 32 int8
    multiply-adds per instruction) can roughly double matmul throughput. That is P9.
- **Alternatives:** larger register tiles, or packed fp32 weight panels kept at load. Both are
  bounded by the same FMA ceiling.

## DD-058: P9 int8/VNNI GEMM prototype is slower than fp32; not adopted

- **Decision:** no AVX-VNNI GEMM for prefill or for batched decode above 4 rows. The int8 decode
  kernels for ≤ 4 rows (DD-053) stay.
- **Evidence:** `tools/proto/vnni_gemm_proto.cpp`, single thread, Q8_0-shaped data, M=64,
  N=1536, K=896. Three runs:
  - fp32 path (dequantized 4-row panel + 4×3 FMA tile): 1.44–1.53 ms, 115–122 GFLOP/s.
  - int8 `vpdpbusd` tile (3 weight × 4 activation rows, sign trick): 1.72–1.93 ms, 91–102 GFLOP/s.
  - So int8 is 0.75–0.89× the fp32 speed, with a max difference of 4e-5 of a 147 range.
- **Reason:** every 32-value block carries its own weight and activation scale. Each (weight row,
  activation row) pair therefore needs a convert + scale + FMA per block on top of the `vpdpbusd`,
  about as many instructions as the 4 fp32 FMAs it replaces. The fp32 path amortizes
  dequantization over all M rows.
- **Alternatives:** formats with one scale per row (per-channel int8 weights, a per-token
  activation scale) would let int32 sums run across a whole row. That needs a requantized weight
  copy (memory) and new accuracy validation. Recorded for the future, not done.

## DD-059: CPU topology: one thread per physical core (default kept); pinning is opt-in

- **Decision (P10):**
  - The default stays at one compute thread per physical core, letting the OS place them.
  - Topology detection now records each physical core's first logical CPU, performance cores
    first (`CpuInfo::core_first_cpu`; Windows `GetLogicalProcessorInformationEx`, Linux sysfs).
  - `DYNALM_PIN_THREADS=1` binds the scheduler thread to core 0 and worker i to core i + 1
    (`pin_current_thread`: Windows `SetThreadAffinityMask`, Linux `pthread_setaffinity_np`;
    macOS has no hard affinity).
  - Power-throttling opt-out (DD-052) stays the main placement lever on hybrid CPUs.
- **Evidence:** Qwen2.5-0.5B Q4_K_M, prompt 128 / output 48, i7-1255U (2 P + 8 E cores,
  12 threads).
  - Thread count (output tok/s, c=1 / c=8): 2 → 21.6 / 42.3, 4 → 25.8 / 49.7,
    6 → 27.5 / 54.2, 8 → 28.5 / 60.0, **10 → 30.2 / 63.8**, 12 (adds SMT siblings) → 28.6 / 61.3.
  - Pinning, three back-to-back A/B pairs (c=1; c=8):
    - off: 30.6, 30.3, 28.9; 78.3, 64.1, 63.9;
    - on: 25.2, 30.2, 26.0; 56.3, 63.9, 60.0.

    Never better: the dynamic chunk grabbing already balances P- and E-cores, and pinning stops
    the OS from moving a worker off a core busy with interrupts or other processes.
- **Reason for keeping the option:** on a dedicated bare-metal Linux server with isolated cores
  the trade-off may differ. That could not be measured on this machine.
- **NUMA:** not applicable to this single-socket machine. Per-node memory placement is left for
  multi-socket hardware.

## DD-060: Grouped MoE expert execution: one region per projection, per-expert path

- **Decision (P11):** `CpuBackend::matmul_many` runs all jobs (one per active expert) in a single
  parallel region, for any job sizes.
  - **Chunks:** every job's output rows are cut into 16-row chunks. Each chunk takes its job's
    path:
    - int8 rows kernel for ≤ `int8_decode_max_rows` rows (activations quantized once, before the
      region);
    - fused dequantize-dot for a single row;
    - dequantized 4-row panels through the GEMM tile otherwise.
  - **Strided rows:** rows may be strided (MoE slices). The int8 path quantizes row by row and the
    panel path passes the row stride. `matmul` now also accepts strided rows on its int8 path, so a
    job gets the same numerics whether it runs alone or grouped.
  - **Down projection:** MoE expert down projections get the same int8 exclusion as the dense FFN
    (DD-053).
- **Reason:**
  - **Before:** the old fused region applied only when every expert had ≤ 3 rows. Otherwise each
    expert's gate, up and down ran as separate `matmul` calls, which meant 1,292–1,857 parallel
    regions per decode step for Granite-3.1-1B-A400M (32 experts, 8 active).
  - **Why that hurt:** each expert matrix is small (512 × 1024) with ~4 rows, so fork/join cost
    and idle threads dominated.
- **Evidence:** Granite-3.1-1B-A400M Q4_K_M, prompt 128 / output 64,
  `results/p11-{before,after}.jsonl`.

  | | c=1 | c=4 | c=16 |
  |---|---|---|---|
  | regions per step | 323 → 223 | 1,292 → 313 | 1,857 → 313 |
  | decode step, ms | 18.5 → 18.3 | 42.5 → 39.0 | 108.9 → 99.6 |
  | output tok/s | 36.7 → 37.2 | 56.8 → 60.7 (+7%) | 66.1 → 69.6 (+5%) |
  | TTFT p50, ms | 547 → 519 | 1,077 → 1,051 | 1,811 → 1,614 |

  - Accuracy (`bench_int8_accuracy`; int8 now reaches expert gate/up): perplexity +0.27% (was
    +0.76%), mean KL 0.0014, top-1 99.5%, greedy identical for 65 of 65 tokens.
  - Tests: 394/394, including `GroupedMatmul.EveryPathMatchesIndividualMatmuls` (jobs of 1/3/5/9
    rows, strided input and output, int8 on and off).
- **Tradeoffs:**
  - One region means all experts share one dynamic chunk queue. Very uneven expert loads balance
    automatically, at 16-row granularity.
  - MoE prefill with large expert batches now uses whole-k panels without K-blocking. These are
    fine at expert widths (≤ a few thousand); dense layers keep the K-blocked path.

## DD-061: Scheduler policies: explicit latency/throughput presets; adaptive budget rejected

- **Decision (P12):** `SchedulerConfig::policy` (`--policy` on serve and benchmark):
  - `balanced`: the default, unchanged (64 prompt + 64 decode tokens per step, 32-token chunks);
  - `latency`: 32-token prompt budget;
  - `throughput`: 128-token prompt budget.

  `benchmark` also exposes `--prefill-budget`, `--decode-budget`, `--chunk` and `--batch` for
  experiments. An adaptive rule (double the prompt budget while ≥ 8 or ≥ 16 prompts are pending)
  was implemented, measured and removed.
- **Reason:** on this CPU the aggregate output rate is saturated at c ≥ 16. Budgets mostly move
  waiting between the first token (TTFT) and the gaps between tokens (ITL). Different deployments
  want different trade-offs, so it is an explicit choice.
- **Evidence:** Qwen2.5-0.5B Q4_K_M, prompt 128 / output 64, two runs each,
  `results/p12-*.jsonl`. Run-to-run noise on this laptop is ±15%.

  | c=64 | output tok/s | TTFT p50 | ITL p50 |
  |---|---|---|---|
  | latency | 85–86 | 35 s | 186 ms |
  | balanced (default) | 93–96 | 21 s | 340 ms |
  | throughput | 95–96 | 3.8 s | 616 ms |
  | adaptive (≥ 8 pending) | 94–96 | 5.7–6.0 s | 497–509 ms |

  At c=16, balanced gives TTFT p50 1.0–1.3 s, latency 0.97–1.04 s (ITL p50 174 ms) and
  throughput 2.2 s. The adaptive rule doubled c=16 TTFT p50 (2.1–3.3 s): closed-loop bursts put
  ≥ 16 prompts in flight, and splitting each step among more prompts finishes each one later.
  Its c=64 gain is available explicitly as `throughput`.
- **Alternatives:** a feedback controller targeting a TTFT/ITL SLO. It needs per-deployment
  targets and a quieter benchmark machine to tune without overfitting noise.
- **Tests:** `SchedulerPolicyTest.ParseAndSameOutputUnderEveryPolicy`. Policies change
  scheduling only: greedy output is identical. 395/395.

## DD-062: Adaptive speculation: choose k (including off) from measured output tok/s

- **Decision (P13):** `SpeculativeOptions::adaptive` (default on; `dynalm run --spec-fixed` turns
  it off). A `SpecController` picks k for every verification round.
  - **Arms:** k ∈ {0, min(3, K), K}, where K is `--spec-k`. k = 0 is plain decoding: the drafter
    is not called. k = 3 keeps verification at 4 rows, the int8 decode path (DD-053).
  - **Rate:** each arm keeps exponentially weighted tokens emitted and milliseconds per round,
    with drafting included and α = 0.25. Its rate is their ratio.
  - **Schedule:**
    - three warm-up rounds per arm, then the best arm runs;
    - the least recently measured other arm is re-probed for 3 rounds, first after 16 rounds;
    - the gap doubles up to 128 while probes confirm the best arm, and resets to 16 when a probe
      changes it.
  - **Catch-up rounds:** the first drafting round after plain rounds is not measured. A model
    drafter spends it prefilling the tokens it skipped, which is a switching cost, not the arm's
    rate.
- **Reason:**
  - Measured before this change, speculation lost in most cases on this CPU, even with good
    drafts. A verification of k + 1 = 5 rows leaves the int8 path, a draft model costs a forward
    per drafted token, and the target's single-row decode is already fast.
  - The spec says to judge speculation on final output tok/s and to disable it when it loses.
    The best k depends on the text (acceptance), so it has to be chosen at run time.
- **Evidence:**
  - Setup: Qwen2.5-1.5B-Instruct Q4_K_M target, greedy, 128 new tokens, 10 threads.
    "story" = "Write a long story about a river.", "list" = "List the numbers from 1 to 100
    separated by commas."
  - Two runs per cell, from two separate sweeps (`results/p13-speculation-run{1,2}.txt`).
    Laptop noise is about ±5% at c = 1.

  | output tok/s | plain | ngram fixed k=4 | ngram adaptive | 0.5B draft fixed k=4 | 0.5B draft adaptive |
  |---|---|---|---|---|---|
  | story | 16.6–17.5 | 14.5–15.2 | 16.2–17.1 | 12.1–12.7 | 13.9–15.0 |
  | list | 15.7–16.4 | 11.5–11.9 | 15.3–15.8 | 19.2–20.2 | 19.0–19.9 |

  - **Ngram:** adaptive recovers plain speed (fixed k loses 10–28%).
  - **0.5B draft, list** (100% acceptance): adaptive keeps the speed-up, +18–24% over plain,
    within 0–3% of fixed k.
  - **0.5B draft, story** (~55% acceptance): adaptive turns speculation off for most rounds
    (62–70 of 83–90 passes run plain). It still trails plain by 10–17% at 128 tokens and by
    4–9% at 512 tokens (14.5–15.2 vs 15.8–16.0; fixed 12.8–13.0). What remains is warm-up,
    probes and the draft's catch-up prefill at each probe.
  - **Before tuning:** a first version without backoff and without excluding catch-up rounds
    gave 13.8–14.4 on that cell.
  - **QoS:** `run --spec` now applies the same full-speed QoS request as the Engine (DD-052). This
    made no measurable difference here and is kept for consistency.
- **Accuracy:** with greedy sampling every accepted token is the target's own argmax. Output
  equals plain decoding up to batch invariance. The target sees 1-, 4- or 5-row verifications,
  and int8 is used only up to 4 rows (DD-053). So on quantized models long greedy outputs can
  diverge after a near-tie, as fixed k already could. `Speculative.AdaptiveGreedyIsExact` checks
  token identity on the fp32 test models with both drafters and k ∈ {1, 4, 6}.
  `SpecController.*` tests the controller's choices with synthetic timings, including switching
  off and back on when acceptance changes.
- **Alternatives:**
  - choosing k from acceptance alone (a cost model): it needs per-machine costs for drafting
    and verification, which the measured rate already includes;
  - keeping a model drafter in step during plain rounds, so probes need no catch-up: this costs
    a draft forward per token, on every token;
  - a finer k grid: each arm costs warm-up and probe rounds.
- **Tradeoffs:**
  - Exploration has a floor cost: about 10 measured warm-up rounds plus 3-round probes.
  - The rate is wall-clock, so other load on the machine shifts the choice. That is intended:
    the goal is delivered tok/s.

## DD-063: LTO available but off; PGO deferred; no autotuning cache

- **Decision (P14):**
  - **LTO:** `-DDYNALM_LTO=ON` turns on interprocedural optimization through CMake's
    `CheckIPOSupported` (MSVC `/GL` + `/LTCG`, GCC/Clang `-flto`). It is **off by default**.
  - **PGO:** not adopted, because it is not measured (see Evidence).
  - **Autotuning cache:** none. The existing environment overrides (`DYNALM_GEMM_KC`,
    `DYNALM_MATMUL_EXPAND_MIN`, `DYNALM_INT8_DECODE_ROWS`) remain the way to tune a new machine.
- **Reason:** the spec adopts an optimization only when measurement shows a gain. Neither LTO nor
  any tuned setting beat the default build and the default plan here beyond noise.
- **Evidence:**
  - **LTO** (MSVC, Qwen2.5-0.5B Q4_K_M, prompt 128 / output 64, three alternating runs,
    `results/p14-{release,lto}.jsonl`):
    - output tok/s at c=1: 32.6–35.4 default vs 30.7–33.2 LTO;
    - at c=8: 71.4–81.0 vs 71.5–73.7.

    Why no gain: the hot loops are SIMD kernels in their own translation units, reached through
    the ISA dispatch table (function pointers), so cross-module inlining cannot reach them.
    Decode is bandwidth-bound (DD-050).
  - **Tuning sweep** (Qwen2.5-0.5B Q4_K_M, prompt 512 / output 64, two runs,
    `results/p14-tune.jsonl`), output tok/s at c=1 / c=8:

    | setting | c=1 | c=8 |
    |---|---|---|
    | defaults (KC 1024, expand from 2 rows, int8 ≤ 4 rows) | 17.2–17.5 | 27.1–28.3 |
    | `DYNALM_GEMM_KC=512` | 16.0–16.7 | 23.2–25.7 |
    | `DYNALM_GEMM_KC=2048` | 17.1–17.9 | 27.0–27.2 |
    | `DYNALM_GEMM_KC=0` (no K-blocking) | 17.9–18.0 | 26.9–27.2 |
    | `DYNALM_MATMUL_EXPAND_MIN=4` | 17.5–18.1 | 27.9–28.0 |
    | `DYNALM_INT8_DECODE_ROWS=8` | 16.8–17.7 | 27.5–27.6 |

    Only KC=512 is clearly worse. A cache would store the defaults on this machine, and its value
    on other CPUs cannot be checked here.
  - **PGO:** a GCC `-fprofile-generate` / `-fprofile-use` trial in the Linux container crashed the
    Docker VM during the training run (about 2 GB of host memory available). It was not repeated
    on this machine. It is deferred to a machine with more memory or CI; the expected gain is
    small for the same reason as LTO.
- **Alternatives:** a `dynalm tune` command that sweeps these settings and stores the best per
  CPU model. Worth building once a second machine shows different optima.
- **Tradeoffs:** LTO roughly doubles link time and memory, for no measured benefit, which is why
  it is off by default.

## DD-064: Pre-fault weight pages at load (cold-start TTFT)

- **Decision:** `Engine::create` touches every weight page once (one read per 4 KiB) from all pool
  workers, right after the pool starts. It is skipped when the weights exceed available memory,
  or with `DYNALM_PREFAULT=0`. `TensorRegistry::for_each` provides the ranges, so only tensors
  the engine uses are touched.
- **Reason:**
  - **The cost:** weights are memory-mapped. The first forward pass took one page fault per page,
    about 600k for Qwen3-4B Q4_K_M, serially inside the kernels. This happened even when the file
    was already in the OS cache. On a cold file, `FILE_FLAG_RANDOM_ACCESS` also turns off
    read-ahead, so pages came from disk one by one.
  - **How it showed:** the first request's TTFT. `dynalm run` reported 0.8–0.9 s for 9–17 prompt
    tokens on Qwen3-4B, and 2.4 s on a cold Gemma-3-270M.
  - **How it was found:** the benchmark warms up before measuring, so it never saw this cost. The
    new `[engine] prefill … ms` field of `run` showed TTFT = prefill forward time, which ruled
    out the streamer.
  - **Existing helper:** `MappedFile::prefetch` existed but had no caller. It only stages pages
    in the OS cache; it does not map them.
- **Evidence:** `dynalm run -p "Hi /no_think" -n 4`, three alternating runs per setting, times
  in ms.

  | | load | TTFT | load + TTFT |
  |---|---|---|---|
  | Qwen3-4B Q4_K_M, before | 89–123 | 874–891 | ~980 |
  | Qwen3-4B Q4_K_M, after | 382–414 | 477–508 | ~880 |
  | Gemma-3-270M F16 (warm cache), before | 66–73 | 127–133 | ~195 |
  | Gemma-3-270M F16 (warm cache), after | 126–135 | 36–41 | ~166 |

  - **One-shot `run`:** time to first token improves 10–15% end to end.
  - **`serve` and `benchmark`:** loading happens once at startup, so each process's first
    request gets the full TTFT drop, 45% on Qwen3-4B and ~70% on Gemma.
  - **Not measured:** a cold-cache start with pre-faulting. Evicting the OS file cache needs
    tools that are not available here.
- **Alternatives:**
  - `PrefetchVirtualMemory` / `MADV_WILLNEED` alone: the pages still fault on first touch.
  - `MAP_POPULATE`: Linux only, and serial.
  - Large pages: need a privilege on Windows.
  - Reading weights into private memory: doubles memory when the file is cached.
- **Tradeoffs:**
  - Load takes longer: +300 ms for a 2.4 GB model.
  - The weights count toward the working set from the start. They would after the first request
    anyway.

## DD-065: No dedicated small-batch prefill kernel: short prompts already run at the chip's GEMM rate

- **Decision:** prefill steps of 5–64 rows keep the existing path, which dequantizes 4-row weight
  panels to fp32 once and runs the 4×3 FMA tile. No new small-M kernel was added.
  `bench_small_m` stays as a diagnostic.
- **Reason:** a short-prompt prefill on Qwen3-4B Q4_K_M takes ~480 ms for 11 tokens, about 3.4× a
  decode step, so it looked like a small-batch weakness. Measurement says otherwise.
  - **Per-row cost:** a fit of `bench_decode_matmul` (`ffn_gate`, 9728×2560, Q4_K, 10 threads)
    is 0.86 ms fixed plus 0.216 ms per row:

    | rows | 6 | 12 | 24 | 64 |
    |---|---|---|---|---|
    | ms | 2.16 | 3.63 | 5.87 | 14.66 |

    The fixed part is reading the weights once (one decode row alone takes 0.7–1.0 ms). Each
    extra row adds 50 MFLOP at about 230 GFLOP/s.
  - **Large batches are no faster:** a 512-token prefill of the same model runs at about
    184 GFLOP/s (22.3 s, 22.7 tok/s). Small batches are not worse per row than large ones.
    Prefill cost ≈ one decode step + about 30 ms per prompt token, and that per-token cost is
    whole-chip arithmetic throughput.
  - **The tile is already efficient:** `bench_small_m` splits one thread's share (243 panels).
    Dequantization is a fixed ~0.28 ms whatever the row count. The tile adds ~0.045–0.05 ms per
    row, about 100–106 GFLOP/s on one core, roughly 70% of a P-core's FMA peak.
  - **Remaining gap:** this is single-core speed vs speed with all cores busy (2 P-cores and
    8 E-cores, which run 256-bit FMA at half rate). A new tile does not change that.
- **Alternatives:**
  - int8/VNNI small-M kernels: already measured at 0.75–0.89× fp32 (DD-058), because of
    per-block scales.
  - An outer-product microkernel layout: same FMA-to-load ratio as the 4×3 tile, so the
    expected gain is a few percent at most.
  - Formats with one scale per row: the open option from DD-058.
- **Measurement note:** the Windows clock query in the benchmark diagnostics reported exactly
  1367 MHz on every run, including prefill-heavy ones. It looks like a fixed value, not a live
  clock, so throttling cannot be confirmed or ruled out from it.

## DD-066: Grouped GQA attention kernels: attention 4–22% faster, end-to-end NEUTRAL

- **Decision:**
  - **Kernels:** `attn_scores_heads_f32` and `attn_accum_heads_f32` (generic, and AVX2 templated
    on up to 8 heads) process all query heads that share a KV head in one pass.
    - Scores: each K vector is loaded once and FMA'd into one accumulator per head. The head
      sums are reduced by one `hadd` tree.
    - Accumulation: each V vector is loaded once, with one register accumulator per head.
    - `attend_group` uses them whenever several heads share the converted fp32 run (f16 KV
      with group > 1, or f32 KV).
  - **Switch:** `KernelPlan::grouped_attention`, default on; `DYNALM_ATTN_GROUPED=0` turns it
    off. It exists for A/B measurement, and the per-head path remains the reference.
  - **Diagnostics changed alongside:**
    - `cpu_current_mhz()` on Windows now reads the effective clock through PDH
      (`% Processor Performance` × nominal frequency). `CallNtPowerInformation` returned a
      constant per-core-type base clock: 1700 MHz on P-cores, 1200 MHz on E-cores, mean 1367,
      regardless of load. Under full decode load the effective clocks were about 2.39 GHz on
      P-cores and 1.34 GHz on E-cores (turbo maxima 4.7 / 3.5 GHz).
    - `bench_batch_decode` was rewritten for in-process A/B with alternating order. It also
      reports memory and stops below a free-memory floor.
    - New: `bench_attn_accuracy` and `tools/ab_summary.py`.
- **End-to-end classification: NEUTRAL.** Attention is reproducibly faster. Aggregate output
  tok/s does not separate from run-to-run noise at any measured point. The change is kept
  because it is correct to rounding, never slower beyond noise, and shrinks the attention share
  that grows with context × concurrency. No throughput gain is claimed.
- **Evidence (Qwen2.5-1.5B Q4_K_M):** 10 threads, f16 KV, 24 timed decode steps plus 12
  profiled steps per measurement. 3 repetitions per point, old/new alternating with the order
  flipped each repetition. Docker Desktop was running (~3 GB), which leaves 3–4 GB available.
  Data: `results/dd066-ab.csv`, `results/dd066-ab-summary.txt`. Medians, old → new:

  | context × seqs | attention ms/step | aggregate tok/s | ITL p50 | ITL p99 |
  |---|---|---|---|---|
  | 512 × 1 | 4.5 → 5.3 (+19%, ~) | 12.2 → 11.7 (−4.1%, ~) | +8.0% ~ | −6.6% ~ |
  | 512 × 4 | 12.6 → 9.8 (−22%, separated) | 25.0 → 23.8 (−4.8%, ~) | +2.0% ~ | +66% ~ (one outlier run) |
  | 512 × 8 | 18.2 → 16.9 (−7%, separated) | 30.1 → 31.6 (+5.0%, ~) | −3.0% ~ | −22% ~ |
  | 512 × 16 | 31.1 → 26.4 (−15%, separated) | 38.4 → 40.9 (+6.5%, ~) | −7.5% ~ | +5.0% ~ |
  | 512 × 32 | 55.2 → 50.1 (−9%, ~) | 49.7 → 52.1 (+4.8%, ~) | −6.3% ~ | −6.8% ~ |
  | 2K × 1 | 9.1 → 7.2 (−22%, ~) | 12.3 → 12.0 (−2.4%, ~) | +1.6% ~ | +0.9% ~ |
  | 2K × 4 | 25.5 → 22.0 (−14%, separated) | 24.6 → 24.1 (−2.0%, ~) | −0.1% ~ | +4.3% ~ |
  | 2K × 8 | 51.8 → 45.5 (−12%, separated) | 29.6 → 30.1 (+1.7%, ~) | +0.1% ~ | −6.4% ~ |
  | 2K × 16 | 109.2 → 104.5 (−4%, ~) | 30.3 → 33.3 (+9.9%, ~) | −11.6% separated | −8.6% ~ |

  "~" means the old and new [min, max] ranges overlap. Attention is only 5–8% of a decode step
  (see the profile below), so even a 20% attention gain is ≤ 2% of the step. That is below
  this laptop's run-to-run noise.
  - **Not measured:** 2K × 32 and all 4K points did not fit beside the other workloads in the
    available memory.
  - **First attempts:** two earlier runs (a full matrix, then one under memory pressure) were
    stopped by the OS and by the benchmark's memory floor. Their single repetitions disagreed
    in sign at several points and are not used.
- **Accuracy:** `results/dd066-accuracy.txt`, `bench_attn_accuracy`. Prefill and 32
  teacher-forced decode positions, then 48 greedy tokens. Models: SmolLM2-135M (3 heads per KV
  head, dim 64), Qwen2.5-0.5B (7, 64), Gemma-3-270M (4, 256, sliding window), Qwen2.5-1.5B
  (6, 128), Qwen3-4B (4, 128). Contexts 64–3072.
  - **Without int8 activations** (F16 Gemma; `DYNALM_INT8_DECODE_ROWS=0` for the others):
    rounding-level everywhere. Mean KL ≤ 1.1e-6, top-1 and top-5 agreement 100%, greedy
    continuation identical (48/48).
  - **With int8 decode** (the default for quantized models), small differences are amplified
    by the int8 activation rounding.
    - **Noise floor:** the same amplification appears between two pre-DD-066 runs that differ
      only in float ordering (per-pair vs split-K attention). Qwen2.5-0.5B at 64 tokens
      diverges in greedy after 2 tokens in both comparisons. Qwen3-4B at 64 tokens changes
      perplexity by +2.7% in the floor comparison alone.
    - **Worst DD-066 point:** Qwen3-4B at 512 tokens, mean KL 4.2e-3 vs 9.4e-4 for the floor.
      The same point with int8 off is 1.1e-6.
    - **Conclusion:** the int8 path's sensitivity to float ordering is a pre-existing property,
      tracked separately. No tolerance was loosened.
  - **Kernel tests:** `GqaAttention.GroupedMatchesPerHeadAndNaive` compares grouped, per-head
    and naive double-precision attention at the existing 1e-4 tolerance. Coverage:
    - both ISAs (AVX2 and generic) and f32/f16 KV;
    - head dims 64/128/72 (72 takes the non-multiple-of-8 fallback);
    - 1–9 heads per KV head (9 needs two passes);
    - three rows from two sequences with scrambled block tables.

    Grouped vs per-head max difference < 1e-5. 386/386 tests pass, none skipped or unbuilt.
- **Post-DD-066 decode profile** (`results/dd066-profile-512*.csv`): Qwen2.5-1.5B, 512 context,
  median of 2 runs, ms/step (share):

  | op | M=1 | M=4 | M=8 | M=16 | M=32 |
  |---|---|---|---|---|---|
  | MLP up (gate+up) | 28.1 (35%) | 61.1 (38%) | 96.7 (40%) | 158.6 (42%) | 263.1 (42%) |
  | MLP down | 21.8 (28%) | 40.1 (25%) | 51.6 (21%) | 81.6 (21%) | 135.9 (22%) |
  | LM head | 11.4 (14%) | 17.0 (10%) | 33.2 (14%) | 50.0 (13%) | 80.0 (13%) |
  | QKV | 7.4 (9%) | 14.0 (9%) | 18.1 (8%) | 26.4 (7%) | 38.8 (6%) |
  | attention | 4.0 (5%) | 9.6 (6%) | 15.2 (6%) | 25.7 (7%) | 48.7 (8%) |
  | attention out | 3.4 (4%) | 7.4 (5%) | 10.5 (4%) | 16.9 (4%) | 26.8 (4%) |
  | norm + RoPE + act + KV store | 3.2 (4%) | 13.0 (8%) | 16.1 (7%) | 21.1 (6%) | 31.9 (5%) |
  | step p50 / aggregate tok/s | 80.8 / 11.5 | 150.9 / 26.1 | 238.2 / 33.3 | 378.4 / 42.1 | 609.0 / 52.1 |

  - **Share of each step:** matmuls are ~90%. In the serving path (`dynalm benchmark`, c = 1–16),
    sampling is 0.3–0.7% of wall time, and planning + emission + prefix cache are < 0.1%.
  - **Thread pool:** tail wait is 7–14% of region time, and CPU utilization is 0.76–0.86.
- **Alternatives:** none needed. The per-head path stays available as the reference.

## DD-067: 32 matmul chunks per thread (was 8): P/E-core tail, neutral to slightly positive

- **Decision:** `KernelPlan::matmul_chunks_per_thread` defaults to 32 (`DYNACORE_MATMUL_CHUNKS`
  overrides it). Output columns of a parallel matmul are cut into about 32 dynamically
  claimed chunks per thread, so fast P-cores take over work that slow E-cores would otherwise
  finish last.
- **Evidence:** Qwen2.5-1.5B Q4_K_M on the i7-1255U (2 P + 8 E cores), 10 threads.
  - **Decode** (`bench_batch_decode`, context 512, 4 repetitions interleaved, data
    `results/dd067-chunks-decode-ab.csv`):

    | Sequences | 8 chunks | 32 chunks | Paired range |
    |---|---|---|---|
    | 1 | 11.8 tok/s | 12.3 tok/s | −0.8% to +21% |
    | 4 | 25.6 | 25.7 | −9.5% to +2.4% |
    | 16 | 43.2 | 43.5 | −2.9% to +7.5% |

    The thread-pool tail-wait share falls slightly (1 seq: 0.216 → 0.207; 4 seq: 0.191 →
    0.171).
  - **Prefill** (512-token prompt, TTFT p50, 3 runs each, alternating): 8 chunks gave 7781,
    7836 and 7783 ms; 32 chunks gave 7588, 7825 and 7707 ms (about −1%).
    Data: `results/dd067-chunks*-prefill.jsonl`.
  - **Classification: NEUTRAL to slightly positive.** No regression at any point, so the
    default is kept. No end-to-end gain is claimed.
- **P/E placement alternatives:** measured in DD-059.
  - Thread counts: 2 threads (P-cores only) gave 21.6 tok/s and 10 threads (P + E) 30.2 at
    c=1. Using E-cores helps.
  - Pinning was never better on Windows.
  - A P-core-only prefill would idle 8 cores, so it is not pursued.

## DD-068: Split the engine into DynaCore (runtime library) and DynaLM (platform), same repo

- **Decision:**
  - `engine_core` becomes two static libraries: `dynacore` (tensors, dtypes, quant formats,
    memory, thread pool, hardware detection, devices and kernels) and `dynalm_runtime`
    (loaders, models, tokenizer, KV policy, scheduler, sampling, engine, API, server).
    Both stay in this repository and ship inside the one `dynalm` binary.
  - The dependency runs one way, `dynalm → dynacore`. Three checks enforce it: separate public
    include directories, a `core-only` CI build, and `tests/boundary/check_boundary.py`, which
    rejects model, file-format and HTTP names under `dynacore/`.
  - The `Backend` interface becomes `dynacore::Device` unchanged in shape: coarse ops with
    caller-owned outputs, chosen over allocating free functions (`gemm(A, B) -> Tensor`).
  - Paged-KV *layout* (`KvGeometry`, `KvLayerView`) moves to DynaCore because attention kernels
    address it. KV *policy* (pool, block tables, refcounts, prefix sharing, eviction) stays in
    DynaLM.
  - The batch planner splits: DynaLM reduces `SeqBatch[]` to a numeric `StepShape`, and
    DynaCore's `plan_kernels()` turns that into a `KernelPlan`.
  - The future IR enters as a recording `Device` that captures the op stream, so it needs no
    DynaLM rewrite.
- **Reason:** the platform spec asks for a reusable inference runtime with no LLM knowledge.
  An include scan found four crossings in today's code: `backend.h` includes
  `model_ir/model_config.h` and `kv_cache/kv_layout.h`, backends include `runtime/thread_pool.h`,
  and the planner mixes model views with kernel heuristics. Every other module is already clean,
  so the split is a move, not a rewrite.
- **Alternatives:**
  - Two repositories: rejected while DynaCore has one consumer. It would add version skew and
    cross-repo changes for every new op.
  - A shared `libdynacore`: deferred until an external consumer exists. Static linking keeps a
    one-file install and LTO across the boundary.
  - YAML library for config: rejected in favor of a strict two-level subset parser, to avoid a
    new runtime dependency.
- **Gate for the implementation (R0):** all tests pass, and aggregate output tok/s on
  Qwen2.5-0.5B is within noise (±2%) of the pre-split binary in an interleaved A/B.
- **Full design:** [platform-design.md](platform-design.md).

## DD-069: R0 done: DynaCore extracted with no measurable throughput change

- **Decision:** the split designed in DD-068 is implemented. Details that differ from, or add
  to, the design:
  - **Names.** The op interface is `dynacore::Device` (`device/device.h`), with
    `CpuDevice`, `DeviceKind` and `create_device`. The memory-location struct that used to be
    called `Device` is now `DeviceLoc`. The design called the header `ops.h`; the parameter
    types live in `device/ops.h`.
  - **Layout.** Public headers are `dynacore/include/dynacore/<module>/`, where `<module>` is
    base, tensor, memory, hardware, quantization, execution, kernel, device, attention or cpu.
    Sources are in `dynacore/src/` and `dynacore/cpu/{generic,avx2,neon}`. DynaLM keeps its
    module directories under `dynalm/src/` and includes them as before (`"scheduler/..."`).
    It has no separate public include directory yet, because nothing outside the repository
    links it.
  - **Namespaces.** `dynacore` and `dynalm`. `dynalm/src/common/core.h` holds the single
    `using namespace ::dynacore` inside `namespace dynalm`, so DynaLM code was not rewritten.
  - **Planner (V4).** DynaCore's `plan_kernels(StepShape, HardwareProfile, base)` owns the
    attention strategy and the split-K chunking. DynaLM's `BatchPlanner` keeps phase and row
    accounting.
  - **Moves not in the design.** `MappedFile` moved to DynaCore memory, since generic mmap is
    not format knowledge. The GPTQ/AWQ repacker moved to `dynalm/src/loader`.
  - **Tuning environment variables.** Renamed `DYNALM_*` to `DYNACORE_*`: `GEMM_KC`,
    `MATMUL_EXPAND_MIN`, `INT8_DECODE_ROWS`, `MATMUL_CHUNKS`, `ATTN_GROUPED`, `PIN_THREADS`.
    DynaCore reads them, so it must not carry the product name. They are experiment knobs,
    documented as not for production, so no compatibility aliases were kept.
  - **Enforcement.** The `boundary.dynacore` ctest and the CI job `boundary`
    (`tests/boundary/check_boundary.py` plus `cmake --preset core-only`).
  - **Paths.** The executable moved to `build/<preset>/bin/dynalm`. Test fixtures moved to
    `dynalm/tests/data`, and the quant fixtures to `dynacore/tests/data/quant`.
- **Evidence:** 387/387 tests pass (386 before, plus the boundary test). Core-only: 69/69.
  - **Setup.** Throughput gate on Qwen2.5-0.5B Q4_K_M: the pre-split binary (commit 107745d)
    against the post-split binary, using `bench_batch_decode` at context 512 with
    10 threads, f16 KV and 32 timed steps. There were 4 repetitions, process-level and
    interleaved, with the order flipped each repetition. Data: `results/r0-split-ab.csv`.

    | concurrent sequences | before (median agg tok/s) | after | difference | spread of single runs |
    |---|---|---|---|---|
    | 1  | 26.5  | 28.9  | +8.9% | 23.1–31.6 |
    | 4  | 78.4  | 77.8  | −0.8% | 67.2–81.4 |
    | 16 | 126.1 | 128.2 | +1.7% | 116.0–134.1 |

  - **Classification: NEUTRAL.** Every difference lies inside the run-to-run spread on this
    thermally limited laptop CPU. The compiled kernels are identical (the move changed no
    function bodies), so nothing points to a real change. No speed-up is claimed. Resolving a
    ±2% gate would need many more repetitions than the noise allows here. The medians show
    no regression.

## DD-070: Model names through a small built-in registry; YAML-subset config; doctor

- **Decision:**
  - **Model references.** A model reference is a path, or a registry name `family:size`.
    - The registry is a compiled-in table in `dynalm/src/registry/`. It holds 12 entries.
      Each entry gives a direct GGUF URL (HTTP 200 checked when added), the quantization and
      the approximate size.
    - There are aliases such as `llama:3b` and `gemma:270m`. A bare family name picks a
      default size.
    - `run`, `serve` and `benchmark` pull a missing named model, as Ollama does. `inspect`
      does not pull.
  - **Store.** Models live in `~/.dynalm/models` or `$DYNALM_MODELS_DIR`. `./models` is
    searched as well.
  - **Config.** The config file accepts a strict two-level YAML subset, parsed by hand.
    - A documented key maps to one existing command-line option, so precedence and
      validation stay in one place. `--threads` and `runtime.threads` cannot drift apart.
    - Unknown keys are errors with a line number.
    - A known key that the running command does not use is skipped, so one file serves both
      `run` and `serve`.
    - `~/.dynalm/config.yaml` applies when present. Tests that inject an environment never
      read it.
  - **Doctor.** `doctor` replaces `info`. GPU detection loads the NVIDIA driver library at
    run time (`cuInit`, `cuDeviceGet*`), so a CPU-only build still reports a usable GPU, and no
    CUDA SDK is needed to build.
- **Reason:** this is the platform spec's user path `install → doctor → pull qwen3:4b → run
  qwen3:4b` (platform-design.md R1).
  - A remote registry service would need hosting and trust decisions, and would add a network
    call to every resolution. A table that ships with the binary is versioned and testable.
  - YAML via a library would be the first new runtime dependency after cpp-httplib.
- **Alternatives:**
  - Ollama-style manifests and blobs: rejected. GGUF files stay as plain files that users can
    see and reuse with other tools.
  - Auto-pulling in `inspect`: rejected. Inspecting must stay a cheap, read-only command.
- **Evidence:**
  - Unit tests: `test_registry` covers names, aliases, resolution and every YAML-subset rule.
  - Manual end-to-end runs:
    - `run smollm2:135m` with a YAML config;
    - `serve smollm2:135m` with the port taken from YAML;
    - `/v1/models` reports `id: smollm2:135m`;
    - SSE streaming works;
    - `stop` drains the server.

## DD-071: DynaCore IR: inference-typed SSA graph, captured by a recording Device

- **Decision:**
  - DynaCore has an IR (`dynacore/ir/`, [dynacore-ir.md](dynacore-ir.md)). It is an
    execution-ordered list of ops over SSA values.
    - **Types:** kind (tensor, weight, kv, index), any DType including block-quantized
      formats, a static or symbolic shape, a layout (row-major, strided, blocked, packed,
      paged) and a device.
    - **Ops:** first-class inference ops: `qmatmul`, `gqa` (head grouping checked), `kv_write`
      on paged caches, `rope`, norms, `act_mul`.
    - **Typing:** one rule per op (`infer_type`), shared by the builder, the parser, the
      verifier and the recorder.
    - **Text form:** round-trips.
    - **Verifier:** checks SSA order, types, in-place aliasing and an optional memory budget.
  - The IR enters through a `RecordingDevice` that implements `Device`, as platform-design.md
    §15 planned. DynaLM's model code is unchanged.
    - **Trace mode** builds the graph and times every op. The op-level profile comes from it.
    - **Deferred mode** is compiled execution (DD-072).
- **Reason:** the compiler needs a representation that knows inference, not a general tensor
  graph. "This weight is q4_K in 256-element blocks", "these 12 query heads share 2 KV heads"
  and "this cache is paged in 16-token blocks" are exactly the facts kernel selection and
  fusion use.
  - Recording keeps one source of truth for the forward pass. A second, hand-written model
    graph would drift from the Transformer.
- **Alternatives:**
  - MLIR/LLVM: rejected for now. It is a large dependency, and nothing here needs it yet
    (DD-073 explains why no code generation).
  - Building IR inside DynaLM: rejected. DynaLM would depend on compiler internals.
- **Evidence:**
  - All 11 tiny architectures (dense, MoE, Gemma softcap and window, Phi) produce IR that
    verifies, and whose text form round-trips (`test_compiled`).
  - IR unit tests and parser fuzzing: `dynacore_test_ir`.

## DD-072: Compiled execution: cached plans, decode-shaped fusion, C++ kernels as backend

- **Decision:**
  - `ExecutionMode::kCompiled` (`--execution compiled`, `runtime.execution: compiled`) wraps
    the device in a deferred `RecordingDevice`. At each sync point the segment is planned and
    executed on the CPU device.
    - **Plan cache:** keyed by the segment's structural signature, which leaves out
      positions, token ids and block tables. Decode steps after the first cost one hash and no
      planning.
    - **Pipeline:** canonicalize → select_kernels (per-op int8 limit recorded from the
      kernel plan) → plan_execution.
  - **Two fusions,** only for matmuls on the int8 decode path:
    1. Shared-input groups (Q/K/V) run as one `matmul_many`. `MatmulJob` gained an optional
       bias, and jobs reading the same input share the int8-quantized activations.
    2. Gated MLP (gate, up, act_mul) runs as one `matmul_gated`. This is a new Device op: the
       default is the unfused sequence; the CPU version fuses on the int8 path.
  - **Kernel plans:** `set_kernel_plan` is no longer a sync point under recording. Each call
    carries its plan, and execution re-applies it.
  - **Fallback:** an invalid plan or a planner error runs the calls in recorded order.
  - **Default:** reference mode stays the default until the end-to-end measurements below
    make the case.
  - **Elementwise kernels:** `act_mul` and `activation` now split columns across the pool
    when rows are few (1×8960: 33 → 14 µs).
- **Reason:** the op profile located the waste outside the big GEMVs, which run at 85–96% of
  DRAM bandwidth.
  - Latency-bound K/V projections: 4 GB/s.
  - `act_mul` reading freshly written lines from other cores: 84 µs in the model against
    23 µs isolated.
  - One fork/join per op.
  - Grouping and gated fusion remove regions and the cross-core traffic. They compute the
    same values with the same kernels, so the results are bit-identical.
- **Evidence:** step-interleaved decode A/B (`bench_compiled`, 4 rounds) on Qwen2.5-1.5B
  Q4_K_M, 10 threads.

  | Variant | 1 sequence | 4 sequences |
  |---|---|---|
  | IR only | ±2% | ±2% |
  | all fusions | −4 to −9% step time | −2 to −6% step time |

  - Logits differ by 0. All architectures are bit-exact (`test_compiled`).
  - The first end-to-end run also fused prefill-shaped matmuls and lost 4–7% at concurrency
    4/8. Hence the int8-path-only rule.
  - End-to-end results: [compiler-benchmarks.md](compiler-benchmarks.md).
- **Alternatives:**
  - Fusing at prefill shapes: measured slower. `matmul_many`'s 16-row panels lack the
    K-blocked GEMM.
  - A polynomial-exp SiLU: slower than vectorized `std::exp`, reverted.
  - Rebuilding IR every step: 1.25 ms/step overhead, replaced by the plan cache.

## DD-073: DynaCore language and dynacorec; no code generation yet

- **Decision:** a small dataflow language for inference graphs (`dynacore/lang/`,
  [dynacore-language.md](dynacore-language.md)).
  - It has typed declarations (quantized weights, paged KV, symbolic rows), expressions
    (`@` matmul with bias folding, `+`, `*`), inference builtins (rmsnorm, rope, kv_write,
    attention→gqa, swiglu, ...) and a `schedule` block of optimizer permissions
    (`fuse gated`, `group shared_input`).
  - It compiles to IR, so it shares the typing rules, verifier and optimizer.
  - `dynacorec` dumps IR, the optimized IR, the kernel plan and the memory analysis. It also
    benchmarks unplanned against compiled execution with a bit-exactness check.
    `GraphExecutor` lowers standalone IR to Device calls.
- **Not built, on evidence:**
  - Kernel-body syntax (tiles, vectors, loads), generated intrinsics and native code. The
    hand-written decode GEMVs reach 88–96% of the measured DRAM ceiling, and VNNI int8 GEMM
    measured slower (DD-058).
  - Prefetch and tile directives are rejected rather than accepted and ignored.
  - A CUDA backend: no NVIDIA GPU on this machine (`dynalm doctor`). The IR and plans are
    device-neutral; see compiler-backends.md.
- **Evidence:**
  - `dynacorec examples/dynacore/decoder_layer.dyna --benchmark`: 32 ops become 12 device
    calls. At M=1 the compiled plan is 12.6% faster than one call per op, and outputs are
    bit-identical. At M=64 nothing fuses and timing is unchanged.
  - `dynacore_test_lang` covers compilation, every error class with its location, fuzzing,
    and exact execution.

## DD-074: q8_0 KV cache: available as `--kv q8_0`, not the default (speed neutral)

- **Decision:** the KV cache accepts `q8_0` besides f16 and f32 (`--kv q8_0`,
  `kv_cache.dtype: q8_0`).
  - **Format:** 32-value blocks with one fp16 scale, quantized in `kv_store`. Attention
    dequantizes each block run to fp32 through the existing convert path.
  - **Constraint:** head dimensions must be multiples of 32.
  - **Sizing:** `KvGeometry` byte math now counts whole blocks (`bytes_of`,
    `k_block_bytes`).
  - f16 stays the default.
- **Evidence:** Qwen2.5-1.5B Q4_K_M, 10 threads.
  - **Accuracy** (`bench_kv_accuracy`): text prefilled into both caches, then the last 256
    positions decoded token by token.

    | Context | Perplexity change | Mean KL | Max KL | Top-1 agreement | Greedy, 64 tokens |
    |---|---|---|---|---|---|
    | 512 | +0.12% | 1.6e-4 nats | 0.012 | 99.6% | identical |
    | 2048 | +0.07% | 1.5e-4 | 0.011 | 100% | identical |

    Both are within the DD-053 contract (≤ +1% perplexity, ≤ 0.0025 mean KL).
  - **Memory:** 1.06 bytes per value against 2, so 7.4 vs 14.0 MiB for 512 tokens (−47%).
  - **Speed** (`bench_batch_decode`, process-level ABBA, aggregate tok/s, f16 → q8_0):

    | Context, sequences | f16 | q8_0 |
    |---|---|---|
    | 1K, 1 | 12.45 | 12.2 |
    | 1K, 4 | 24.0 | 24.85 |
    | 4K, 1 | 11.2 | 10.95 |
    | 4K, 4 | 20.5 | 21.45 |

    **Classification: NEUTRAL.** Attention is ≤ 23% of a step even at 4K × 4, and
    dequantizing costs about what the halved KV reads save. Data:
    `results/dd074-kv-q8-decode-ab.csv`.
- **Why not the default:** throughput is the primary metric, and q8 KV does not raise it per
  step.
  - It does raise capacity. `auto` context sizing gives about 1.9× the tokens per GiB, which
    matters on RAM-limited machines and for many concurrent sequences.
  - It is offered for those cases, and documented in configuration.md.
- **Alternatives:**
  - A q8 attention kernel (integer q·k dot): not built. The attention share is too small for
    it to pay at the measured contexts.
  - Revisit at 16K+ contexts, or when KV memory limits concurrency.

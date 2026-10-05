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

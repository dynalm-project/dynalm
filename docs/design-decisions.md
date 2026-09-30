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

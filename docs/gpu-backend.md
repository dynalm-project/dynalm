# GPU backend architecture

Status: **designed and enforced by tests; no GPU backend is implemented** (spec §45–§47, DD-045).
This page is the contract a CUDA, HIP, Metal or Vulkan backend implements, and it lists what
the rest of the engine already guarantees.

## What the runtime guarantees today

1. **The model never dereferences device memory.** Every tensor byte the Transformer, KV cache,
   prefix cache, scheduler or speculative decoder touches goes through a `Backend` op.
   - **How this is enforced:** `test_device_backend` runs every supported architecture (dense
     and MoE), continuous batching with prefix-cache copy-on-write, and speculative decoding
     on `GuardedBackend`, a test backend whose memory is `mprotect(PROT_NONE)` except inside
     its own ops.
   - **What failure looks like:** a stray host access is a SIGSEGV.
   - **What must hold:** results equal the CPU backend bit for bit.
2. **Weights are placed by the backend.** `Backend::upload(host_tensor)` returns the device
   copy. CPU returns the same tensor, so loading stays zero-copy from the memory-mapped file.
3. **Scratch and KV cache are backend allocations.** `Backend::allocate` provides them, and KV
   copy-on-write uses `Backend::copy`.
4. **Host-side decisions use small downloads:**
   - final logits (sampling runs on the host);
   - MoE router logits, `m × experts` floats per layer (top-k routing on the host);
   - the shared-expert gate, `m` floats.
5. **No CUDA (or other vendor) types leak** into the model, scheduler or KV interfaces.
   `DeviceType` and `BackendKind` already carry the GPU kinds; `create_backend` reports them
   as "not built".

## The interface (`src/backends/backend.h`)

| group | ops | notes |
|---|---|---|
| memory | `allocate`, `copy`, `upload`, `download`, `synchronize`, `host_accessible` | `download` is the sync point for sampling |
| GEMM | `matmul` (+bias), `matmul_many` | weights in any `DType` the backend supports (`supports_weight_type`); `matmul_many` = all active experts of an MoE layer in one launch |
| attention | `kv_store`, `attention` | paged KV: `KvLayerView` = base pointers + per-sequence block table; rows of a batch belong to different sequences (`row_seq`); sliding window and soft-cap are parameters |
| elementwise | `rms_norm`, `layer_norm`, `rope` (+ per-pair frequency factors), `act_mul`, `activation`, `add`, `scale`, `softcap`, `fill` | fp32 activations, row-major `[rows, cols]`, row strides allowed |
| data movement | `embedding`, `gather_rows`, `scatter_add_rows` | logits-row gather; MoE token permutation |

Small host arrays passed to ops (token ids, positions, row→sequence maps, block tables, MoE row
indices and weights, RoPE frequency factors) are per-step metadata. A device backend copies
them with the launch, or caches them; the tables are constant.

## Implementing CUDA (sketch)

- **Memory:** `cudaMallocAsync` from a per-backend stream-ordered pool. `upload` happens once
  at load. Quantized blocks are kept byte-identical (Q4_0/Q8_0/Q4_K/...), so the same
  dequantize-in-kernel GEMMs apply.
- **GEMM:**
  - decode (m ≤ 8): fused dequantize-dot kernels, one warp per output row;
  - prefill: tensor-core GEMM on dequantized tiles;
  - `matmul_many`: one grouped-GEMM launch (llama.cpp's `mul_mat_id`, Marlin-style for
    GPTQ/AWQ).
- **Attention:** a paged decode kernel reading `block_table` (FlashInfer-style), and
  FlashAttention for prefill. The KV layout `[block][kv_head][slot][dim]` maps directly.
- **Graphs:** decode steps of a fixed batch shape can be captured as CUDA graphs. The
  scheduler's per-step token budgets bound the shapes.
- **Sampling:** stays on the host for now (one `download` per step, `vocab × rows` floats). A
  device top-k/top-p op can replace it behind `Sampler` later.
- **Streams:** one per backend; `synchronize` before `download` returns.

## Extension points already in place (spec §47)

| feature | where |
|---|---|
| speculative decoding, draft models, KV rollback | `runtime/speculative`, `SeqBatch::logits_last`, `KvBlockTable::truncate` (DD-044) |
| MoE / expert parallelism | `matmul_many` over experts; routing results are explicit host arrays that an expert-parallel backend can partition (DD-042) |
| GPU/CPU KV tiers, remote KV | the KV pool is allocated through a `Backend`; blocks are opaque ids with refcounts (prefix cache, COW) |
| prefill/decode disaggregation | scheduler budgets separate prefill and decode rows (DD-027/028) |
| tensor / pipeline parallelism | not started: would split `Transformer` layers or matmuls across several `Backend` instances |
| multimodal | not started |

## Build flags

`ENABLE_CUDA`, `ENABLE_HIP`, `ENABLE_METAL` and `ENABLE_VULKAN` exist and default to OFF.
Turning one on is a configure error until its backend exists, so a build never silently
lacks the requested backend. `dynalm info` lists the built backends, and
`--backend cuda` fails with a clear message.

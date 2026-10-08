# DynaCore IR: An Intermediate Representation for LLM Inference

The internal format DynaLM uses to describe one step of an LLM (weights, attention, KV cache) so a compiler can optimize it.

The DynaCore IR is an inference-specific intermediate representation (DD-071). A graph is a
list of operations in execution order over SSA values.

The IR describes **what** a forward step computes, in inference terms: quantized weights,
paged KV caches, grouped-query attention, RoPE, norms. Backends decide **how** it runs:
kernels, tiles and threads. The IR is not a general compute graph and has no control flow.

Code: `dynacore/include/dynacore/ir/` (`ir.h`, `text.h`, `verifier.h`, `passes.h`,
`recording_device.h`).

## Values and types

Every value has a full type. Its parts:

| Part | Values |
|---|---|
| kind | `tensor` (activations), `weight` (parameters), `kv` (one layer's K/V cache), `index` (int32 host arrays: token ids, positions, row→sequence, rows) |
| element type | any DynaCore `DType`: `f32`, `f16`, `bf16`, `i8`, ..., and the block formats `q4_0`, `q8_0`, `q4_K`, `q6_K`, ... |
| shape | static sizes or symbols (`M` = rows of the step) |
| layout | `row_major` (default), `col_major`, `strided(n)`, `blocked(n)`, `packed(n)`, `paged(n)` |
| device | CPU (default) or `@gpu` |

Examples:

```
tensor<f32>[M,1536]               activations, M rows
tensor<f32>[1,256] strided(2048)  K columns written into a wider QKV buffer
weight<q4_K>[1536,8960] blocked(256)
kv<f16>[64,2,16,128] paged(16)    64 blocks x 2 KV heads x 16 tokens x 128 dims
index<i32>[M]                     positions
```

**Quantization is part of the type.** `weight<q4_K>` is not `tensor<f32>`. Kernel selection and
the verifier both see:

- the block size (`blocked(256)`);
- the bytes per block (`Type::bytes`);
- that K must hold whole blocks.

**In-place ops are SSA.** `rope`, `scale`, `softcap`, `fill`, `kv_write` and `scatter_add`
produce a new value that shares the operand's buffer (`Value::buffer`). Passes then see true
data flow, while execution still writes in place. Other aliasing, such as a residual
`add` into an operand, is printed as `alias=N`.

## Operations

| Group | Ops |
|---|---|
| Leaves | `input`, `weight`, `kv_cache`, `constant` |
| Linear algebra | `matmul` (f16/f32 weight), `qmatmul` (block-quantized weight, the canonical form), `embedding` |
| Attention | `attention` (heads == kv_heads), `gqa` (heads % kv_heads == 0, heads > kv_heads), `kv_write`, `kv_read`, `rope` |
| Norms, elementwise | `rmsnorm`, `layernorm`, `softmax`, `activation`, `act_mul` (SwiGLU/GeGLU), `add`, `mul`, `scale`, `softcap`, `fill` |
| Data movement | `gather`, `scatter_add`, `transpose`, `reshape`, `cast`, `reduce`, `broadcast` |

**GQA and paged KV are first-class.**

- `gqa` carries `heads`, `kv_heads` and `head_dim`.
- Its KV operand has the `paged(block_size)` layout.
- The verifier rejects:
  - a grouping that does not divide (`GQA requires query_heads % kv_heads == 0 (query_heads = 30, kv_heads = 4)`);
  - a cache whose KV-head count or head_dim differs from the attributes;
  - a `block_size` attribute that differs from the cache layout.

`infer_type(kind, operand types, attrs)` derives each result type: shape, dtype and layout.
The builder, the parser, the recording device and the verifier all use it, so there is one
definition of every op's typing rule.

## Text form

```
graph @decode {
  %0 = input "x" : tensor<f32>[M,1536]
  %1 = weight "blk.0.attn_q" : weight<q4_K>[1536,1536] blocked(256)
  %2 = kv_cache "blk.0" : kv<f16>[64,2,16,128] paged(16)
  %3 = constant "positions" [7] : index<i32>[1]
  %4 = qmatmul %0, %1 {int8_rows=4} : tensor<f32>[M,1536]  // kernel=cpu.int8_dot.q4_K fusion=0
  %5 = gqa %4, %2, %3, %3 {block_size=16, head_dim=128, heads=12, kv_heads=2, scale=0.0883883} : tensor<f32>[M,1536]
}
```

- `print_graph` and `parse_graph` round-trip: `parse(print(g))` prints identically. This is
  tested for every supported architecture.
- The parser re-infers each result type and rejects a declared type that the operands cannot
  produce.
- Errors carry `ir:line:col`. Input is capped at 64 MiB.
- A fuzz test feeds corrupted programs to the parser and the verifier; neither may crash.

## Verifier

`verify(g)` checks:

- operands are defined before use (op order is execution order);
- every result type matches `infer_type`, allowing a result to be placed in rows of a wider
  buffer (strided);
- leaf kinds match their op;
- in-place ops really share their operand's buffer;
- value producer indices are current;
- optionally, that weights + KV + the largest activation fit `memory_budget`.

`verify_all` returns every error rather than only the first.

## Builder

```cpp
Graph g("layer");
Builder b(g);
ValueId x  = b.input("x", tensor_type(DType::kF32, {Dim::symbol("M"), Dim::of(4096)}));
ValueId wq = b.weight("wq", weight_type(DType::kQ4_K, {Dim::of(4096), Dim::of(4096)}));
ValueId kv = b.kv_cache("kv0", kv_type(DType::kF16, 512, 8, 16, 128));
ValueId q  = b.rope(b.matmul(x, wq), pos, 32, 128);
ValueId a  = b.attention(q, b.kv_write(kv, k, v, pos, seq), pos, seq, 32, 8, 128, 0.0884);  // emits gqa
if (!b.status().ok()) ...  // first typing error, with the op and the reason
```

## Where IR comes from

DynaLM never builds IR by hand. Its `Transformer` issues `Device` ops, and a
`RecordingDevice` turns that stream into IR . The recording device
has two modes:

- **Trace mode** builds the graph of one forward pass and times every op
  (`bench_op_trace`).
- **Deferred mode** is the compiled execution path described in
  [dynacore-compiler.md](dynacore-compiler.md).

# DynaCore Language (.dyna) and the dynacorec Compiler

A small language for describing LLM inference graphs, and `dynacorec`, the tool that compiles, inspects and benchmarks them.

A small language for **inference graphs** (DD-073). A program compiles to DynaCore IR, then
goes through the same optimizer and backend as compiled execution.

The language has no loops, classes, exceptions or general control flow. A program is a
dataflow graph whose types carry what the compiler optimizes on:

- quantized weight formats;
- paged KV caches;
- head grouping;
- shapes with symbolic batch rows.

The author states what is computed and which optimizations are allowed. The compiler picks
kernels, groups calls and fuses.

```
// examples/dynacore/decoder_layer.dyna (abridged)
graph decoder_layer(M = 1) {
  config hidden = 1536, heads = 12, kv_heads = 2, head_dim = 128, ff = 8960

  input  x    : tensor<f32>[M, hidden]
  input  pos  : index[M]
  input  seq  : index[M]
  weight wq   : q4_K[hidden, hidden]
  weight bq   : f32[hidden]
  ...
  kv cache    : kv<f16>[64, kv_heads, 16, head_dim]

  xn = rmsnorm(x, attn_norm, eps = 1e-6)
  q  = xn @ wq + bq                       // matmul with bias
  q  = rope(q, pos, heads, head_dim, base = 1000000)
  cache = kv_write(cache, k, v, pos, seq)
  a  = attention(q, cache, pos, seq, heads, kv_heads, head_dim)   // lowers to gqa
  h  = x + a @ wo
  y  = h + swiglu(hn @ w_gate, hn @ w_up) @ w_down
  output y

  schedule {
    fuse gated
    group shared_input
  }
}
```

## Syntax

| Element | Form |
|---|---|
| Graph | `graph NAME(SYM = default, ...) { statements }`. Parameters are symbolic dimensions such as `M`, the rows of a step. |
| Constants | `config a = 1536, b = a * 2`. Integer or float constant expressions; `sqrt()` is allowed. |
| Declarations | `input NAME : tensor<f32>[M, hidden]`, `input NAME : index[M]`, `weight NAME : q4_K[N, K]` (or `weight<q4_K>[...]`), `kv NAME : kv<f16>[blocks, kv_heads, block_size, head_dim]` |
| Assignment | `name = expr`. Rebinding a name is allowed (SSA underneath); assigning to an input or weight is an error. |
| Expressions | `x @ W` (matmul, W is [out, in]); `x @ W + b` (bias folded into the matmul); `a + b` (add); `a * b` (elementwise product); calls; parentheses |
| Outputs | `output y, z` |
| Schedule | `schedule { fuse gated|none ; group shared_input|none }` |
| Comments | `//` or `#` to end of line |

Dimensions are products of integers and config names (`kv_heads * head_dim`), or a single
symbolic parameter.

## Builtins

| Call | IR |
|---|---|
| `rmsnorm(x, w, eps=1e-6)`, `layernorm(x, w[, b], eps=1e-5)` | `rmsnorm`, `layernorm` |
| `rope(x, positions, heads, head_dim, base=10000, style=half_split|interleaved, dim=head_dim)` | `rope` (in place) |
| `kv_write(cache, k, v, positions, row_seq)` | `kv_write`. The result is the updated cache; rebind it: `cache = kv_write(cache, ...)`. |
| `attention(q, cache, positions, row_seq, heads, kv_heads, head_dim, scale=1/sqrt(hd), softcap=, window=)` | `attention` when heads == kv_heads, otherwise `gqa` |
| `swiglu(gate, up)`, `geglu(gate, up)` | `act_mul` with silu / gelu_tanh |
| `silu(x)`, `gelu(x)`, `gelu_tanh(x)`, `softmax(x)`, `scale(x, s)`, `softcap(x, cap)`, `embedding(table, ids)` | the matching ops |

## Errors

Every error names `file:line:col`. Semantic errors come from the IR typing rules, so the
language, the IR text form and recorded graphs agree on what is valid. Examples:

```
decoder_layer.dyna:3:7: undefined name 'xn'
t.dyna:3:1: qmatmul: weight rows must hold whole q4_K blocks (256 elements), K = 8
t.dyna:6:6: gqa: GQA requires query_heads % kv_heads == 0 (query_heads = 30, kv_heads = 4)
t.dyna:3:2: unknown schedule directive 'prefetch weights' (supported: fuse gated|none, group shared_input|none)
```

The parser and the lowering are fuzzed in `test_lang` (truncations and byte corruptions);
neither may crash.

## dynacorec

```
dynacorec <file.dyna | file.ir> [--graph NAME] [--set M=4] [--dump-ir] [--dump-optimized-ir]
          [--dump-kernels] [--memory] [--emit FILE] [--benchmark] [--iters N] [--threads N] [--no-fuse]
```

```
$ dynacorec examples/dynacore/decoder_layer.dyna --dump-kernels --memory
   0  op           %16=rmsnorm[cpu.rmsnorm]
   1  matmul_many  %17=qmatmul[cpu.int8_dot.q4_K] %18=qmatmul[cpu.int8_dot.q4_K] %19=qmatmul[cpu.int8_dot.q6_K]
   ...
   9  matmul_gated %27=qmatmul[cpu.int8_dot.q4_K] %28=qmatmul[cpu.int8_dot.q4_K] %29=act_mul[cpu.act_mul]
memory (M=1): 15 activation values, 162 KiB; peak live 111 KiB; 13 buffers 155 KiB; with reuse 119 KiB

$ dynacorec examples/dynacore/decoder_layer.dyna --benchmark
benchmark (CPU/avx2, 10 threads, M=1): unplanned 2709.7 us, compiled 2368.2 us, -12.6%; outputs bit-identical
```

- `--benchmark` runs the graph on this CPU through `GraphExecutor`. Unbound weights get a
  constant byte pattern; activations are random.
- It interleaves one call per op ("unplanned") with the compiled plan, and requires
  bit-identical outputs. The exit code is 3 if they differ.
- `.ir` files in the [IR text form](dynacore-ir.md) are accepted as input too.

## What the language is not (yet)

- **Kernel bodies.** There are no tiles, vectors, loads or stores. A kernel-level sublanguage
  is useful only once the compiler generates kernels (stage 2,
  [compiler-backends.md](compiler-backends.md)). Measurements show the hand-written GEMVs at
  88–96% of DRAM bandwidth on this machine, so there is no case for it yet.
- **Prefetch and tile directives.** They are rejected rather than silently ignored, because
  the optimizer does not act on them.
- **Runtime JIT.** Not built. Compiled execution already specializes per shape through the
  plan cache, at no code-generation cost.

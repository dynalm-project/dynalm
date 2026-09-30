# Quantization

Quantization is its own subsystem (`src/quant/`). Model code never contains
format-specific logic; it sees `DType`s, and the backend decides how to compute
with them.

Separate concepts, tracked independently per tensor/op:

| Concept | Today | Later |
|---|---|---|
| weight dtype | per tensor: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K | IQ*, GPTQ/AWQ (Phase 24) |
| activation dtype | F32 | Q8_0/Q8_K-quantized activations for integer dot products (Phase 17/18) |
| accumulator dtype | F32 | int32 inside integer dot products |
| KV dtype | F16 (default) or F32 | Q8 KV |

GGUF "file types" such as Q4_K_M mix block types across tensors (for example Q4_K for
most weights, Q6_K for some, F32 norms). Kernels only ever see the block type of the
tensor at hand.

## Layers

| Layer | File | Role |
|---|---|---|
| Block geometry | `dtype/dtype.cpp` | elements/bytes per block for every DType |
| Block layouts + reference dequant | `quant/quant_formats.{h,cpp}` | GGML-compatible `BlockQ*` structs; scalar `dequantize_q*` (the correctness oracle) |
| Row conversion | `quant/dequant.{h,cpp}` | `dequantize_row(dtype, src, dst, n)` for all supported types |
| Compute | `backends/cpu/cpu_backend.cpp` | matmul: dequantize a weight row once, dot with every activation row |

## Correctness

- Dequantization for every block type is checked against **gguf-py's independent NumPy
  implementations** (`tools/make_quant_fixtures.py` → `tests/data/quant/*.bin|.f32`,
  test `Dequant.*`). Q8_K (internal activation format, no gguf-py dequantizer) is checked
  by hand-computed values.
- Scalar fp16/bf16 conversion is tested exhaustively over all 65,536 inputs.

## Performance status

Phase 8 is the correct baseline: a quantized weight row is expanded to fp32 once per
matmul and reused across activation rows. That's simple and exact, but it reads the
quantized bytes and then writes and re-reads fp32. Phase 17/18 add fused kernels that
skip the fp32 round trip: activations are quantized to Q8, then integer dot products
with AVX2 (`maddubs`/AVX-VNNI on this machine), with scales applied per block.
Every optimized kernel is tested against `dequantize_q*`.

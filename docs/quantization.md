# Quantization

Quantization is its own subsystem (`src/quant/`). Model code never contains
format-specific logic; it sees `DType`s, and the backend decides how to compute
with them.

Separate concepts, tracked independently per tensor/op:

| Concept | Today | Later |
|---|---|---|
| weight dtype | per tensor: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K; GPTQ/AWQ repacked into these at load | IQ*, native packed kernels |
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

## GPTQ and AWQ (Phase 24, DD-041)

`quant/gptq_awq.{h,cpp}` describes packed checkpoints with `PackedScheme` (method, bits,
group size, symmetry, act-order, GPTQ v1 zero offset), unpacks them exactly, and repacks each
linear layer into a layout the kernels run:

| packed weights | engine layout | fidelity |
|---|---|---|
| GPTQ int4, symmetric, groups of 32k | Q4_0 | bit-exact |
| GPTQ int8, symmetric, groups of 32k | Q8_0 | bit-exact |
| GPTQ/AWQ int4, asymmetric | Q4_1 | min rounded to fp16 (≤ 2.4e-4 on the fixtures) |
| act-order, other group sizes, 8-bit asymmetric | F16 | fp16 rounding |

Detection comes from `config.json` `quantization_config` (`hf::read_quantization`). Methods the
engine cannot run fail at load with a message naming the method. Weight, activation,
accumulator and KV types stay separate: a GPTQ model runs Q4_0 weights × fp32 activations →
fp32 accumulators, with f16 KV.

Fixtures: `tools/make_tiny_quant_hf.py` writes five packed variants of the tiny Llama, each
with an independent NumPy-dequantized F32 twin (`tests/data/hf_tiny_llama_*`).

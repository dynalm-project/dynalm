# Quantization

Planned for Phase 1 (dtype system) and Phase 8 (quantized execution).

Separate concepts, tracked independently per tensor/op:

| Concept | Example |
|---|---|
| weight dtype | Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16, F32 |
| activation dtype | F32 (initially), F16/BF16 later |
| accumulator dtype | F32 (int32 for int8 dot products) |
| KV dtype | F16 (default), F32 |

Initial GGUF targets: Q4_K_M, Q5_K_M, Q6_K, Q8_0. ("_M" mixes block types
across tensors. Kernels see the block types: Q4_K, Q6_K, and so on.)

## Implemented (Phase 1)

- `DType` covers F32, F16, BF16, I8/I16/I32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K–Q6_K and Q8_K,
  with block geometry matching GGML (`dtype/dtype.cpp`, unit-tested against the GGML struct sizes).
- Exact scalar fp16/bf16 conversion (`dtype/fp16.h`) with round-to-nearest-even, tested
  exhaustively over all 65,536 values. Generic backend: about 1.25 ns/elem (fp16→fp32).
- Not yet implemented: dequantization and quantized dot products (Phase 8).

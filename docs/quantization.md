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

Nothing is implemented yet.

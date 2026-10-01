# TODO

Phases run strictly in order. A phase is done only when it builds, its tests
pass, its benchmarks have run, the docs are updated, and it is committed.

- [x] Phase 0 — Project foundation
- [x] Phase 1 — Tensor + dtype system (DType incl. GGUF block types, TensorShape/Layout/View, aligned allocator)
- [x] Phase 2 — GGUF loader (mmap, metadata KV, tensor infos, `engine inspect`)
- [x] Phase 3 — Model IR (ModelConfig, TensorRegistry, tensor roles)
- [x] Phase 4 — Tokenizer / chat template (BPE + SentencePiece from GGUF metadata)
- [x] Phase 5 — Llama model
- [x] Phase 6 — Basic CPU execution (single sequence, greedy, `engine run`)
- [x] Phase 7 — Qwen / Mistral / Gemma / Phi / DeepSeek adapters
- [x] Phase 8 — Quantized GGUF execution
- [x] Phase 9 — KV cache
- [x] Phase 10 — Paged KV
- [x] Phase 11 — Concurrent sequences
- [x] Phase 12 — Continuous batching
- [x] Phase 13 — Prefill/decode scheduler
- [x] Phase 14 — Chunked prefill
- [x] Phase 15 — Prefix hash cache
- [x] Phase 16 — Radix cache
- [x] Phase 17 — CPU SIMD optimization
- [x] Phase 18 — Kernel optimization
- [x] Phase 19 — Streaming
- [x] Phase 20 — OpenAI API
- [x] Phase 21 — Benchmark framework
- [x] Phase 22 — Production hardening
- [x] Phase 23 — SafeTensors
- [x] Phase 24 — GPTQ/AWQ architecture preparation
- [ ] Phase 25 — MoE support
- [ ] Phase 26 — Advanced sampling
- [ ] Phase 27 — Speculative decoding preparation
- [ ] Phase 28 — GPU backend architecture

## Open items

- In-flight prefix dedup: requests arriving in the same step as the first request with a new prefix recompute it (Phase 15/16 limitation).

- int8 activation quantization + AVX-VNNI for prefill, with accuracy tests vs references (DD-033).
- GQA-grouped decode attention (read each KV head once per query group).
- Multi-row fused decode kernel (decode each weight block once, dot against up to 4 activation
  rows in registers) to replace the expand path for small batches (DD-036).
- GPTQ act-order: permute input channels (with the producing layer's outputs) to keep 4-bit storage (DD-041).
- Native packed int4 kernels behind PackedScheme (GPU path: Marlin-style); parallel repack at load (2 s for 0.5B).
- Open-loop (Poisson arrival) mode for the load generator (DD-037).
- Token-authenticated admin endpoint for deployments behind a same-host proxy (DD-039).
- CPU affinity / thread pinning / NUMA placement (spec §32), measured against the baselines.
- AVX2 Q5_K/Q2_K/Q3_K/Q5_1 fused kernels (they use the chunked fallback today).
- Cold-start TTFT: prefetch mmapped weights at load (first forward is page-fault bound).

- HF Unigram/WordPiece tokenizers; multimodal Gemma 3 checkpoints (text tower only).
- Jinja subset interpreter for chat templates that no family matches (DD-013).
- Pre-tokenizers for DeepSeek-LLM/V3 and Tekken (Mistral Nemo).
- YaRN and LongRoPE scaling (Qwen long-context, Phi-3-128k).

- Clang in the Linux container (gcc is covered).
- Measure P-core-only vs. all-core threading on hybrid CPUs (DD-004).

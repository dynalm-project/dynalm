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
- [ ] Phase 9 — KV cache
- [ ] Phase 10 — Paged KV
- [ ] Phase 11 — Concurrent sequences
- [ ] Phase 12 — Continuous batching
- [ ] Phase 13 — Prefill/decode scheduler
- [ ] Phase 14 — Chunked prefill
- [ ] Phase 15 — Prefix hash cache
- [ ] Phase 16 — Radix cache
- [ ] Phase 17 — CPU SIMD optimization
- [ ] Phase 18 — Kernel optimization
- [ ] Phase 19 — Streaming
- [ ] Phase 20 — OpenAI API
- [ ] Phase 21 — Benchmark framework
- [ ] Phase 22 — Production hardening
- [ ] Phase 23 — SafeTensors
- [ ] Phase 24 — GPTQ/AWQ architecture preparation
- [ ] Phase 25 — MoE support
- [ ] Phase 26 — Advanced sampling
- [ ] Phase 27 — Speculative decoding preparation
- [ ] Phase 28 — GPU backend architecture

## Open items

- Jinja subset interpreter for chat templates that no family matches (DD-013).
- Pre-tokenizers for DeepSeek-LLM/V3 and Tekken (Mistral Nemo).
- YaRN and LongRoPE scaling (Qwen long-context, Phi-3-128k).

- Get a small GGUF test model (~0.5B, e.g. Qwen2.5-0.5B-Instruct Q8_0/Q4_K_M) for Phases 2–8. Keep it out of git.
- Clang in the Linux container (gcc is covered).
- Measure P-core-only vs. all-core threading on hybrid CPUs (DD-004).

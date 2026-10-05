# Roadmap

DynaLM was built in 29 milestones (0–28), each finished only when it built, its tests passed, its
benchmarks ran, and its docs were updated. All are complete. The open items below are where help is
welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Completed milestones

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
- [x] Phase 25 — MoE support
- [x] Phase 26 — Advanced sampling
- [x] Phase 27 — Speculative decoding preparation
- [x] Phase 28 — GPU backend architecture

## Performance program (P1–P16)

Objective: maximize **aggregate output tokens/s under concurrency** on the same hardware and model,
while keeping numerical correctness, p99 and memory under control. Phases run in order. Each phase
reports Before / After / Delta from measurement (method: [docs/performance.md](docs/performance.md)).

- [x] P1 — Measurement: engine/scheduler timing, thread-pool and OS/hardware counters, bandwidth
  ceiling and traffic model, bottleneck classifier, sweep + 8 graphs (DD-050)
- [x] P2 — Execution planner: BatchPlanner → ExecutionPlan / KernelPlan, per-step decisions in one place (DD-051)
- [x] P3 — Multi-row decode: 4×3 fp32 GEMM tile; int8-activation integer-dot kernels for M ≤ 4 that
  unpack each weight block once per 4 rows (DD-053). Open: weight-row-tiled int8 for M > 4, NEON integer kernels
- [~] P4 — Weight packing: evaluated and deferred. After DD-052/053, Q4_K_M is the fastest format at c=8/16
  and every format costs the same per step, so unpacking is not the bottleneck (DD-054). Revisit with P9
- [x] P5 — GQA decode attention: one K/V fetch per query group, head sub-groups for small batches;
  4K-context decode −21% (DD-054). Dynamic split-K chunking moves to P7
- [x] P6 — Decode synchronization: measured fork/join at 2.5 µs (~2% of a step); wake-up lock and notify
  only when workers sleep (−16% region cost); region fusion not justified (DD-055)
- [x] P7 — KV: block-run attention kernels, per-group fp16 conversion, vector exp, dynamic split-K;
  4K-context attention 9.7 → 6.6 ms; block-size study (keep 16); Q8 KV deferred, not bandwidth-bound (DD-056)
- [ ] P8 — Prefill GEMM: tiled, cache-blocked
- [ ] P9 — INT8 activations / AVX-VNNI
- [ ] P10 — CPU topology: affinity, P/E-core placement, NUMA (power-throttling opt-out already done: DD-052)
- [ ] P11 — MoE: grouped expert execution
- [ ] P12 — Adaptive scheduler: THROUGHPUT_FIRST / BALANCED / LATENCY_FIRST
- [ ] P13 — Adaptive speculation (judged on output tok/s)
- [ ] P14 — LTO / PGO / autotuning cache
- [ ] P15 — AVX-512 / AMX (needs hardware or an emulator to verify)
- [ ] P16 — CUDA (after the CPU planner is stable; needs an NVIDIA GPU to verify)

## Open items

- In-flight prefix dedup: requests arriving in the same step as the first request with a new prefix recompute it (Phase 15/16 limitation).

- int8 activation quantization + AVX-VNNI for prefill, with accuracy tests vs references (DD-033).
- GQA-grouped decode attention (read each KV head once per query group).
- Multi-row fused decode kernel (decode each weight block once, dot against up to 4 activation
  rows in registers) to replace the expand path for small batches (DD-036).
- GPTQ act-order: permute input channels (with the producing layer's outputs) to keep 4-bit storage (DD-041).
- Native packed int4 kernels behind PackedScheme (GPU path: Marlin-style); parallel repack at load (2 s for 0.5B).
- MoE: mixed dense/MoE layer stacks (DeepSeek-MoE first-k-dense, Qwen mlp_only_layers);
  MLA attention for DeepSeek-V2/V3; OLMoE (full-width QK-norm); parallel gather/scatter and
  concurrent expert GEMMs for MoE prefill.
- Sampling: logprobs/top_logprobs in the API, n > 1, logit_bias, grammar/JSON-constrained decoding.
- Speculative decoding in the scheduler (1 + k decode rows per entry) and the API; tree drafts; stochastic draft proposals.
- GPU backends (CUDA first) per docs/gpu-backend.md; device top-k routing and sampling ops; tensor/pipeline parallelism.
- Open-loop (Poisson arrival) mode for the load generator (DD-037).
- Token-authenticated admin endpoint for deployments behind a same-host proxy (DD-039).
- CPU affinity / thread pinning / NUMA placement, measured against the baselines.
- AVX2 Q5_K/Q2_K/Q3_K/Q5_1 fused kernels (they use the chunked fallback today).
- Cold-start TTFT: prefetch mmapped weights at load (first forward is page-fault bound).

- HF Unigram/WordPiece tokenizers; multimodal Gemma 3 checkpoints (text tower only).
- Jinja subset interpreter for chat templates that no family matches (DD-013).
- Pre-tokenizers for DeepSeek-LLM/V3 and Tekken (Mistral Nemo).
- YaRN and LongRoPE scaling (Qwen long-context, Phi-3-128k).

- Measure NEON on real ARM hardware (CI runs it on GitHub's arm64 runners; all 11 jobs green).
- `qwen35` (Qwen3.5/3.8): Gated-DeltaNet layers need per-sequence recurrent state next to the paged KV cache
  (scheduler, prefix cache and speculative rollback all assume KV-only state), plus M-RoPE and the attention output gate.
- IQ quant kernels (iq1_s/m, iq2_xxs/xs/s, iq3_xxs/s, iq4_nl/xs): needed by most imatrix GGUFs of large models.
- `dynalm pull`: Hugging Face repo browsing (pick a quantization from a repo link), sharded GGUF (-00001-of-0000N).
- Homebrew formula / winget manifest / signed Windows releases once tagged releases exist.
- NEON kernels for Q5_K/Q2_K/Q3_K and int8 dot products (SDOT).
- Measure P-core-only vs. all-core threading on hybrid CPUs (DD-004).

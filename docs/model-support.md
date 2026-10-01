# Model support

| Family | Adapter | Status |
|---|---|---|
| Llama (incl. Mistral dense, DeepSeek-LLM, SmolLM via `llama` arch) | LlamaArchitecture | ✅ runs; golden-tested on SmolLM2-135M (f16) |
| Qwen2 / Qwen2.5 / Qwen3 (dense) | QwenArchitecture (`qwen2`, `qwen3`) | ✅ tiny-model golden; Qwen2.5-0.5B real golden |
| Mistral 7B (dense) | via LlamaArchitecture (GGUF arch `llama`) | ✅ same code path as Llama (DD-021) |
| Gemma / Gemma 2 / Gemma 3 (text) | GemmaArchitecture (`gemma`, `gemma2`, `gemma3`) | ✅ tiny-model golden; Gemma-3-270M real golden |
| Phi-3 / 3.5 / 4-mini (dense) | PhiArchitecture (`phi3`) | ✅ tiny-model golden (no real checkpoint fits the dev machine) |
| DeepSeek dense (LLM/Coder, R1-Distill) | via Llama / Qwen adapters | ⚠ R1-Distill-Qwen OK; DeepSeek-LLM needs its pre-tokenizer (TODO) |
| MoE (Mixtral, DeepSeek-MoE, Qwen-MoE) | — | Phase 25 |

Formats:
- GGUF v2/v3 ✅.
- Hugging Face SafeTensors ✅ (Phase 23, DD-040). Pass a model directory (`config.json` +
  `model.safetensors`, or sharded files with `model.safetensors.index.json` + `tokenizer.json`),
  or one `.safetensors` file inside it.
  - Weights: F32, F16, BF16.
  - `model_type`: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3_text, phi3.
  - Verified: SmolLM2-135M-Instruct (BF16), Qwen2.5-0.5B-Instruct (BF16), and every tiny
    fixture (bit-exact against GGUF).

GGUF tensor types the engine can load: f32, f16, bf16, i8/i16/i32, q4_0, q4_1, q5_0, q5_1,
q8_0, q8_1, q2_K–q6_K, q8_K. Recognized but unsupported: iq*, tq*, mxfp4, i64, f64.

Verified files: SmolLM2-135M-Instruct f16 (llama arch, 30 layers, 272 tensors).

## Tokenizers

| GGUF `tokenizer.ggml.model` / `pre` | Status |
|---|---|
| `gpt2` + gpt2/default, llama3/llama-bpe, qwen2, smollm/starcoder | ✅ (golden-tested: SmolLM2, Qwen2.5) |
| `llama` (SentencePiece) | ✅ (golden-tested: Gemma-3, 262K vocab) |
| deepseek-llm, deepseek-v3, tekken, others | not yet (clear error) |
| HF `tokenizer.json`: byte-level BPE (GPT-2 / Llama 3 / Qwen2 / SmolLM splits) | ✅ (ids equal to GGUF: Qwen2.5, SmolLM2) |
| HF `tokenizer.json`: SentencePiece-style BPE with byte fallback | ✅ (ids equal to GGUF: Gemma 3) |
| HF Unigram / WordPiece, other split regexes | not yet (clear error) |

Chat templates: ChatML, Llama-3, Llama-2, Mistral, Gemma, Phi-3, DeepSeek-V2/3.

Weight types that run today: f32, f16, bf16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K (so every common GGUF file type: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_*, Q2_K).
IQ* / TQ* / MXFP4 are recognized but not executable yet.

Verified real files: SmolLM2-135M f16 + Q8_0, Qwen2.5-0.5B f16 + Q4_K_M (golden),
Qwen2.5-0.5B Q8_0 (runs), Gemma-3-270M f16 (golden).

Not yet: YaRN / LongRoPE scaling (Qwen long-context configs, Phi-3-128k), Gemma-3 vision.

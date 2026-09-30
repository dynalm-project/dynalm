# Model support

| Family | Adapter | Status |
|---|---|---|
| Llama | LlamaAdapter | planned (Phase 5) |
| Qwen | QwenAdapter | planned (Phase 7) |
| Mistral | MistralAdapter | planned (Phase 7) |
| Gemma | GemmaAdapter | planned (Phase 7) |
| Phi | PhiAdapter | planned (Phase 7) |
| DeepSeek (dense) | DeepSeekAdapter | planned (Phase 7) |
| MoE (Mixtral, DeepSeek-MoE, Qwen-MoE) | — | Phase 25 |

Formats: GGUF v2/v3 ✅ (parsing and inspection, Phase 2), SafeTensors (Phase 23).

GGUF tensor types the engine can load: f32, f16, bf16, i8/i16/i32, q4_0, q4_1, q5_0, q5_1,
q8_0, q8_1, q2_K–q6_K, q8_K. Recognized but unsupported: iq*, tq*, mxfp4, i64, f64.

Verified files: SmolLM2-135M-Instruct f16 (llama arch, 30 layers, 272 tensors).

## Tokenizers

| GGUF `tokenizer.ggml.model` / `pre` | Status |
|---|---|
| `gpt2` + gpt2/default, llama3/llama-bpe, qwen2, smollm/starcoder | ✅ (golden-tested: SmolLM2, Qwen2.5) |
| `llama` (SentencePiece) | ✅ (synthetic tests; real-model golden pending a Llama-2/Mistral file) |
| deepseek-llm, deepseek-v3, tekken, others | not yet (clear error) |

Chat templates: ChatML, Llama-3, Llama-2, Mistral, Gemma, Phi-3, DeepSeek-V2/3.

No model runs yet.

# Model support

## Models you can download

Copy a line into `dynalm pull`, for example
`dynalm pull Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf`.
- To check any other GGUF first, run `dynalm pull <link> --check`. It reads 256 KiB and saves nothing.
- Rough rule: you need free RAM about the size of the file, plus a little for the context.

**Status legend:**
- **tested**: compared token by token against a reference implementation (golden test).
- **ran**: downloaded with `dynalm pull` and answered correctly on a test machine.
- **checked**: the architecture pre-check passed, and the family is covered by tests, but this exact file
  has not been run yet.

### Small (under 1 GB): any machine

| Model | `dynalm pull` argument | Size | Status |
|---|---|---|---|
| SmolLM2-135M | `bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf` | 0.14 GB | tested |
| Gemma-3-270M | `unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-F16.gguf` | 0.54 GB | tested |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf` | 0.49 GB | tested |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | 0.64 GB | ran |
| Llama-3.2-1B | `bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 0.81 GB | ran |
| Gemma-3-1B | `unsloth/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf` | 0.81 GB | checked |
| Granite-3.1-1B-A400M (MoE) | `bartowski/granite-3.1-1b-a400m-instruct-GGUF/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf` | 0.82 GB | tested (Q8_0) |

### Medium (1–3 GB): 8 GB RAM

| Model | `dynalm pull` argument | Size | Status |
|---|---|---|---|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.1 GB | ran |
| DeepSeek-R1-Distill-Qwen-1.5B | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | 1.1 GB | checked |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` | 1.8 GB | checked |
| Gemma-2-2B | `bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf` | 1.7 GB | checked |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf` | 2.1 GB | checked |
| Llama-3.2-3B | `bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 2.0 GB | checked |
| Phi-3-mini-4k | `microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf` | 2.4 GB | checked |
| Qwen3-4B | `Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` | 2.5 GB | checked |
| Gemma-3-4B (text only) | `unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf` | 2.5 GB | checked |

### Large (4–6 GB): 16 GB RAM

| Model | `dynalm pull` argument | Size | Status |
|---|---|---|---|
| Qwen2.5-Coder-7B | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4.7 GB | checked |
| DeepSeek-R1-Distill-Qwen-7B | `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 4.7 GB | checked |
| Mistral-7B-v0.3 | `bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | 4.4 GB | checked |
| Llama-3.1-8B | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.9 GB | checked |
| Qwen3-8B | `Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | 5.0 GB | checked |

### Very large (mixture of experts): 32 GB+ RAM

| Model | `dynalm pull` argument | Size | Status |
|---|---|---|---|
| Qwen3-30B-A3B | `unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf` | 18.6 GB | checked |
| Mixtral-8x7B | `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf` | 26 GB | checked |

**With caveats:**
- **Phi-3.5-mini** (`bartowski/Phi-3.5-mini-instruct-GGUF`) passes the pre-check, but it uses LongRoPE
  scaling, which is not implemented. Expect poor quality beyond short contexts.
- **Qwen2.5-7B-Instruct** (official repo): its Q4_K_M is split into two files, and `pull` does not handle
  split GGUFs yet. Use a single-file quantization or another uploader's repo.
- **Gated models** (e.g. the official `meta-llama/...` repos) need the license accepted on Hugging Face and
  `HF_TOKEN` set. The `bartowski/` and `unsloth/` mirrors above are not gated.

Hugging Face SafeTensors folders (`config.json` + `*.safetensors`) of the same families also load, including
GPTQ and AWQ. Download them with `huggingface-cli download <repo> --local-dir models/<name>`.

### Not supported (refused by `dynalm pull`)

| Models | Architecture | Why |
|---|---|---|
| Qwen3.5 / Qwen3.8 (all sizes and fine-tunes: 9B, 27B, abliterated, uncensored, MiMo-Distill, …) | `qwen35` | Hybrid Gated-DeltaNet + attention layers, M-RoPE |
| Qwen3.8-Flash-Next | `qwen4exp` | New architecture |
| LFM2 / LFM2.5 (Liquid AI) | `lfm2` | Convolution/attention hybrid |
| DeepSeek-V2 / V3 / R1 (full), DeepSeek-MoE, OLMoE | `deepseek2`, … | MLA attention, mixed dense/MoE layers |
| Image models (Qwen-Image, Stable Diffusion, …) and vision towers (mmproj) | — | Not text generation models |
| Any model whose GGUF uses IQ1–IQ4, TQ or MXFP4 types | — | Those quant kernels are not implemented. Pick a Q4_K_M / Q8_0 file |

## Families and adapters

| Family | Adapter | Status |
|---|---|---|
| Llama (incl. Mistral dense, DeepSeek-LLM, SmolLM via `llama` arch) | LlamaArchitecture | ✅ runs; golden-tested on SmolLM2-135M (f16) |
| Qwen2 / Qwen2.5 / Qwen3 (dense) | QwenArchitecture (`qwen2`, `qwen3`) | ✅ tiny-model golden; Qwen2.5-0.5B real golden |
| Mistral 7B (dense) | via LlamaArchitecture (GGUF arch `llama`) | ✅ same code path as Llama (DD-021) |
| Gemma / Gemma 2 / Gemma 3 (text) | GemmaArchitecture (`gemma`, `gemma2`, `gemma3`) | ✅ tiny-model golden; Gemma-3-270M real golden |
| Phi-3 / 3.5 / 4-mini (dense) | PhiArchitecture (`phi3`) | ✅ tiny-model golden (no real checkpoint fits the dev machine) |
| DeepSeek dense (LLM/Coder, R1-Distill) | via Llama / Qwen adapters | ⚠ R1-Distill-Qwen OK; DeepSeek-LLM needs its pre-tokenizer (TODO) |
| Mixtral (MoE) | LlamaArchitecture (GGUF `llama` + experts; HF `mixtral`) | ✅ tiny golden + HF/GGUF equivalence (DD-042) |
| Qwen2-MoE / Qwen1.5-MoE, Qwen3-MoE | QwenArchitecture (`qwen2moe`, `qwen3moe`) | ✅ tiny goldens (shared expert, raw vs renormalized gating) |
| IBM Granite / Granite-MoE | LlamaArchitecture (`granite`, `granitemoe`) | ✅ tiny golden; Granite-3.1-1B-A400M real golden |
| DeepSeek-MoE / V2 / V3, OLMoE | — | not yet (mixed dense/MoE layers, MLA attention; clear error) |
| Qwen3.5 / Qwen3.8 dense (`qwen35`) | — | not yet: hybrid Gated-DeltaNet (linear attention with recurrent SSM state) + gated full attention every 4th layer, M-RoPE, MTP head; published GGUFs also use IQ1–IQ4 types. `dynalm pull` refuses it before downloading |

Formats:
- GGUF v2/v3 ✅.
- Hugging Face SafeTensors ✅ (Phase 23, DD-040). Pass a model directory (`config.json` +
  `model.safetensors`, or sharded files with `model.safetensors.index.json` + `tokenizer.json`),
  or one `.safetensors` file inside it.
  - Weights: F32, F16, BF16, plus GPTQ (4/8-bit) and AWQ (4-bit GEMM) repacked at load (DD-041).
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

Chat templates: ChatML, Llama-3, Llama-2, Mistral, Gemma, Phi-3, DeepSeek-V2/3, Granite 3.x.

Weight types that run today: f32, f16, bf16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K (so every common GGUF file type: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_*, Q2_K).
IQ* / TQ* / MXFP4 are recognized but not executable yet.

Also run end to end via `dynalm pull`: Qwen3-0.6B Q8_0 (`qwen3`), Llama-3.2-1B-Instruct Q4_K_M (Llama 3 RoPE
factors). Architecture pre-check passed (not yet run): Qwen3-1.7B/4B/8B, Qwen3-30B-A3B, Qwen2.5-3B/7B/Coder-7B,
Llama-3.2-3B, Llama-3.1-8B, Gemma-3-1B/4B, Gemma-2-2B, Phi-3-mini-4k, Phi-3.5-mini, Mistral-7B-v0.3,
Mixtral-8x7B, DeepSeek-R1-Distill-Qwen-1.5B/7B.

Verified real files: SmolLM2-135M f16 + Q8_0, Qwen2.5-0.5B f16 + Q4_K_M (golden),
Qwen2.5-0.5B Q8_0 (runs), Gemma-3-270M f16 (golden).

Not yet: YaRN / LongRoPE scaling (Qwen long-context configs, Phi-3-128k), Gemma-3 vision.

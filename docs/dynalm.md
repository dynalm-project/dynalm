# DynaLM

DynaLM is the LLM inference and serving platform built on DynaCore. It owns every policy:
which model, which requests run, how KV blocks are shared, which tokens are sampled, and what
the user sees.

| Module (`dynalm/src/...`) | Contents |
|---|---|
| `loader/` | GGUF (mmap), SafeTensors / Hugging Face directories, GPTQ/AWQ repacking, model links |
| `model_ir/`, `model/` | `ModelConfig`, `TensorRegistry`, architecture adapters (Llama, Qwen 2/3, Gemma 1-3, Phi-3, Mistral, DeepSeek, MoE families), the `Transformer` forward pass |
| `tokenizer/`, `chat_template/` | BPE and SentencePiece, chat templates |
| `execution/` | `BatchPlanner`: `SeqBatch[]` becomes `StepShape`, which goes to DynaCore's `plan_kernels` |
| `kv_cache/`, `prefix_cache/` | paged KV pool with refcounted block tables and copy-on-write; radix and hash prefix caches |
| `scheduler/` | continuous batching, chunked prefill, policies (balanced, latency, throughput) |
| `sampling/`, `runtime/` | sampler, speculative decoding, `Engine` (scheduler thread, streams, execution mode), generator |
| `api/`, `server/` | JSON, OpenAI schema, HTTP server ([api.md](api.md)) |
| `config/`, `registry/`, `diagnostics/` | configuration sources and the YAML subset, model names and store, `doctor` |
| `cli/` | the `dynalm` commands ([cli.md](cli.md)) |
| `logging/`, `metrics/`, `bench/` | logs, Prometheus metrics, load generator |

## Supported models

- **Architectures:**
  - Llama 2/3.x;
  - Qwen 2/2.5/3, including the Qwen2/Qwen3 MoE models;
  - Mistral and Mixtral;
  - Gemma 1/2/3 (text);
  - Phi-3;
  - DeepSeek-R1 distills (Qwen/Llama based);
  - Granite 3.x and Granite MoE.
- **Formats:** GGUF with F32/F16/BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and Q2_K–Q6_K weights;
  Hugging Face SafeTensors folders; GPTQ and AWQ checkpoints (repacked at load).
- **Names:** `dynalm models --available` lists the tested downloads (`qwen3:4b`,
  `llama3.2:3b`, `gemma3:270m`, `granite3.1-moe:1b`, ...).
- **Any other GGUF:** check it with `dynalm pull <link> --check`, which reads 256 KiB and
  saves nothing.
- **Not supported yet:** Qwen3.5 (`qwen35`), LFM2, full DeepSeek-V3/R1, IQ-quantized files.
  `dynalm pull` refuses them before downloading.
- **RAM:** roughly the file size, plus the KV cache. `--kv q8_0` halves the KV cache.

Execution modes:

- `reference`: the Transformer calls the DynaCore CPU device directly.
- `compiled`: the same calls go through DynaCore's recording device and compiler (DD-072).

DynaLM never sees IR or kernels in either mode.

Further reading:

- architecture: [architecture.md](architecture.md)
- design: [platform-design.md](platform-design.md)
- decisions: [design-decisions.md](design-decisions.md)

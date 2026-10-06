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

Execution modes:

- `reference`: the Transformer calls the DynaCore CPU device directly.
- `compiled`: the same calls go through DynaCore's recording device and compiler (DD-072).

DynaLM never sees IR or kernels in either mode.

Further reading:

- architecture: [architecture.md](architecture.md)
- design: [platform-design.md](platform-design.md)
- decisions: [design-decisions.md](design-decisions.md)

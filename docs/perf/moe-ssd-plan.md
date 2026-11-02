# MoE models larger than RAM, on the CPU: plan

Status: phases 1, 2, 4, 5 and 6 implemented; phase 3 dropped on measurement. First measurements below.

## Goal

Run mixture-of-experts (MoE) models whose files are larger than RAM, on the CPU, with answers
bit-identical to the same model held entirely in RAM. Experts that are not in RAM are read from
the SSD when the router picks them.

Each token uses only a few experts per layer (Qwen3-30B-A3B: 8 of 128, about 3.3B of 30.5B
parameters), so the bytes read per token are a small part of the file.

## Baseline first: `tools/bench/moe_ssd_bench.py`

The script measures today's speed before any engine change:
- **Engines:** llama.cpp (`llama-bench`, reading the file through mmap) and DynaLM, both CPU only.
- **Models:**

  | Model | Size | Architecture |
  |---|---|---|
  | Qwen3-30B-A3B Q4_K_M | 18.6 GB | qwen3moe |
  | Mixtral-8x7B Q4_K_M | 26.4 GB | llama with experts |
  | Qwen3-235B-A22B Q4_K_M | 142.2 GB (3 parts) | qwen3moe |
  | DeepSeek-V3-0324 UD-IQ1_S | 186.2 GB | deepseek2; not supported by DynaLM |

- **Safety:** each download is checked against its SHA-256. A model is skipped when its download
  would leave less than 15 GB of disk free.

```sh
python3 tools/bench/moe_ssd_bench.py --dry-run                  # what it would do here
python3 tools/bench/moe_ssd_bench.py --download --models 30b,mixtral
python3 tools/bench/moe_ssd_bench.py --download --delete-after   # one large model at a time
```

On a 24 GB machine, Qwen3-30B fits in RAM, so it is the reference speed. Mixtral (26.4 GB) does
not fit, so it is a true from-SSD test. On a 16 GB machine, both run from the SSD.

## Phases

Each phase is a separate PR. A phase is kept only if its output is bit-identical to the all-in-RAM
run, and if it is measurably faster than the phase before.

0. **Baseline:** the script above.
1. **Expert cache:**
   - `--experts-from-disk [--expert-cache SIZE]`;
   - only the non-expert weights must fit in RAM (`check_memory_fit`);
   - a fixed-size RAM cache that keeps experts used both often and recently;
   - parallel reads that bypass the operating system's file cache (`F_NOCACHE`, `O_DIRECT` /
     `io_uring`, unbuffered I/O on Windows);
   - hook point: `Transformer::moe_mlp`, where the CPU path already knows the chosen experts
     before the matrix multiplications.
2. **Expert prediction:** while a layer computes, read the experts the next layer will probably
   need (its router applied to the current input).
3. **Compressed experts on disk: dropped.** Lossless compression, unpacked on the CPU, was to
   be kept only if measured at least 10% faster. Measured on Qwen3-8B Q4_K_M, 16 MiB samples at
   30%, 50% and 70% of the file (2026-11-02):
   - zlib level 6: 98.0–98.2% of the original size;
   - xz preset 1: 98.5–98.7%;
   - 4-bit values: entropy 3.90 bits of 4, so even an ideal coder saves at most 2.5%.

   Quantized weights are already close to random bits, so reading fewer bytes this way cannot
   reach the 10% bar.
4. **Shared reads across users: by design.** A batched step routes every row of every request
   first, then acquires each layer's experts once. An expert is read at most once per step,
   however many requests use it (test: one step of 4 sequences, reads <= experts per layer x
   layers).
5. **Speculative decoding:** prompt lookup, EAGLE-3 heads, and drafts that limit how many new
   experts they need. This extends `runtime/speculative.cpp`.
6. **Saved prompt caches on disk and KV cache spilled to SSD.** These apply to every model.

DeepSeek-V3 / Kimi (deepseek2, with MLA attention) need the architecture added first: separate
work.

## First measurements (2026-11-02, uncontrolled)

These are not results: an Apple M4 Pro (24 GB) on battery, with other apps (Docker) leaving about 5 GB
of RAM free, one round each. The model file had just been downloaded and hashed, so some expert
reads may have been served from the OS file cache rather than the SSD.

Qwen3-30B-A3B Q4_K_M (18.6 GB; 16.7 GB of experts), `dynalm benchmark --cpu`, 128-token
prompts, 64 output tokens:

| Setup | 1 user tok/s | 4 users tok/s | Experts resident when needed | GB read per token (1 user) |
|---|---|---|---|---|
| Whole model mapped (default) | 5.6 | 10.3 | – | – |
| `--expert-cache 4GB` | 10.9 | 18.4 | 97.0% | 0.240 |
| `--expert-cache 8GB` | 12.7 | 19.8 | 98.7% | 0.133 |
| `--expert-cache 12GB` | 10.3 | 23.1 | 99.6% | 0.046 |

- **The mapped model was the slowest** here: 18.6 GB of mapped weights with about 5 GB free
  means the OS keeps evicting and re-reading pages. The expert cache keeps a fixed working set.
- **Expert prediction**, 8 GB cache, 1 user, alternated off / on / off / on: 12.9, 14.2, 13.9 and
  14.7 tok/s. With prediction, waiting on reads fell from about 6.4 s to 2.6 s per run, and 98.7%
  of experts were resident vs 92.6% without it, for 16% more bytes read. A first, single A/B
  run had shown the opposite (14.1 off vs 12.7 on); the alternated runs did not repeat it.
- **Output** was identical, word for word, with the whole model in RAM and with a 4 GB cache.

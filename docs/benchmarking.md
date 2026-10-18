# Benchmarking DynaLM: CPU LLM Throughput, Latency & Engine Comparisons

<!--
SEO Keywords: LLM benchmark CPU, tokens per second CPU inference, llama.cpp vs Ollama vs DynaLM benchmark, TTFT CPU latency, time to first token, CPU LLM throughput.
AEO Direct Answer: DynaLM includes a native benchmark tool (`dynalm benchmark <model>`) measuring tokens/second throughput and Time-to-First-Token (TTFT) latency in-process or across HTTP servers.
-->

There are two supported ways to measure performance. Both record exactly which executable, ISA kernels, and model weights they ran.

| Question | Tool & Workflow |
|---|---|
| Is build / setting B faster than A on this machine? (paired A/B) | `dynalm benchmark` (runs) + `tools/bench_analyze.py` (analysis) |
| How does DynaLM compare with llama.cpp / Ollama? | `tools/compare_engines.ps1` (wraps `tools/compare_engines.py`) |

Both drive the same load generator, `dynalm benchmark`. In-process for A/B runs; over
HTTP with `--url` for engine comparisons. It produces fixed output lengths (EOS is ignored)
and unique prompts. It also writes per-request logs (`--request-log`), an optional scheduler
step trace (`--step-trace`), and per-point diagnostics (`diag.matmul_kernels`: which CPU
kernels ran).

## Paired A/B runs

1. **Write the plan first:** arms, workloads, and the gates in a JSON file (format: the
   docstring of `tools/bench_analyze.py`).
2. **Run each measurement session separately.** Run `dynalm benchmark` once per arm and round,
   alternating the arm order between sessions:

   ```sh
   dynalm benchmark <model> --concurrency 4 --prompt 512 --output 128 \
     --out results/my-campaign/raw/<workload>-s1r1-<arm>.jsonl \
     --request-log results/my-campaign/raw/<workload>-s1r1-<arm>.requests.jsonl
   ```

   - Add one record per run to `results/my-campaign/raw/runs.jsonl` (id, exit code, quiet
     check, executable SHA-256, environment; the format is in the `tools/bench_analyze.py`
     docstring).
   - Before every run, close other engines, benchmarks and tests, and keep the power source
     and plan constant.

3. **Analyze:**

   ```sh
   python tools/bench_analyze.py results/my-campaign/plan.json results/my-campaign/raw
   ```

   **Units are counted separately:**
   - a *run* is one benchmark process;
   - a *point* is one concurrency × prompt × output line of a run;
   - a *pair* is one (session, round) of two arms at one point;
   - a *gate evaluation* is one gate at one point.

   **Verdicts:**
   - Per gate evaluation: PASS, FAIL or INCONCLUSIVE.
   - Every evaluation at a point is INCONCLUSIVE when the reference arm varied more than the
     plan allows, across pairs or across session medians, or has fewer valid pairs than
     required.
   - A run or point is invalid, and named in the report, when:
     - its executable is not the planned one;
     - it failed, or had errors;
     - any request produced a different number of tokens than requested.

**On a 15 W laptop, expect 2–5% run-to-run variation.** Close other applications, keep the
power source and plan constant, and leave several minutes between sessions. A result that does
not clear its plan's reliability limits stays INCONCLUSIVE. Rerun with a new, predeclared
plan instead of changing the limits.

## Engine comparison

```powershell
powershell -ExecutionPolicy Bypass -File tools\compare_engines.ps1 -DynalmExe C:\path\to\dynalm.exe
```

- **Executable:**
  - `-DynalmExe` names the DynaLM executable under test; it is never replaced by a local build;
  - without it, the workspace build `build\msvc-release\bin\dynalm.exe` is used and reported
    as such;
  - each engine's path, version and SHA-256 go into `manifest.json`, `report.md` and every
    result row.
- **Fairness rules:**
  - every engine gets the same per-user context and thread count;
  - the report shows the prompt length each engine actually reported ("Prompt seen") and the
    mean completion length;
  - rows below 95% of the requested output (or with errors) are marked not comparable and are
    not ranked.
- **Process safety:**
  - Ollama runs as the script's own server on port 11435, so a running Ollama app is left
    alone;
  - a busy port fails that engine's preflight; the script never stops processes it did not
    start.
- **Known mismatch: Ollama.** Ollama ignores `ignore_eos`, so completions can end early. On
  `/v1/completions` it applies the model's chat template even when a raw `TEMPLATE` is set
  (seen with Ollama 0.40.1 on Qwen2.5: 128 → 157 prompt tokens, 13–74 of 128 output tokens).
  Such rows are not comparable.

## Accuracy against llama.cpp (perplexity)

`bench_ppl` scores a text file exactly as llama.cpp's `llama-perplexity` does: the same
tokenization, chunks of `ctx` tokens starting with BOS when the model uses one, and the second
half of each chunk scored. So the two numbers compare directly:

```bash
# wikitext-2 test set: test/.deps/llama.cpp/scripts/get-wikitext-2.sh
BENCH_DEVICE=metal build/macos-metal/dynalm/benchmarks/bench_ppl MODEL.gguf wiki.test.raw 512 20
llama-perplexity -m MODEL.gguf -f wiki.test.raw -c 512 --chunks 20 -ngl 0 -dev none
```

`BENCH_DEVICE`: `native` (DynaCore's CPU kernels, the default), `ggml-cpu`, `metal`, `vulkan`
or `cuda`. Each runs its default plan, as the engine does.

Example (Phi-3.5-mini Q4_K_M, 20 chunks of 512, M4 Pro):

| Engine | Perplexity |
|---|---|
| llama.cpp, CPU | 7.1421 ± 0.25 |
| DynaLM ggml-cpu / Metal / native CPU | 7.1267 / 7.1144 / 7.1153 |
| DynaLM before LongRoPE support (Metal) | 7.7083 |

Phi-4-mini Q4_K_M: llama.cpp 11.4255 on Metal and 11.5078 on CPU. DynaLM: 11.4256 on Metal,
11.4217 on Vulkan, 11.4245 native and 11.5252 on ggml-cpu. Llama-3.1-8B Q8_0 on Metal: both 8.1993.

Tokenizers compare the same way: `BENCH_TOKENS=1 bench_ppl MODEL.gguf FILE` prints DynaLM's token
ids, and llama-server's `POST /tokenize` (`"add_special": false`) gives llama.cpp's.

## Tests of the tools

| command | checks |
|---|---|
| `python tools/test_compare_engines.py` | comparability marking, ranking, executable selection, no appending to old results |
| `python tools/test_bench_analyze.py` | unit counts, invalid runs (exit code, wrong executable, early end, errors), missing metrics, unreliable references, the three gate rules, bootstrap intervals |

## Publishing results (docs/benchmarks.md)

Public numbers come only from these scripted, repeatable runs:

1. **Run the comparison** on a quiet machine (charger in, other apps closed):

   ```bash
   SERVER=<dynalm> CLIENT=<dynalm> LLAMA=<llama-server> LLAMA_COMMIT=<sha> \
     tools/bench/baseline.sh --out test/results/<name>/gpu --models 1.5b,8b --devices gpu
   ```

   The script copies every binary first, records checksums and versions in `manifest.txt`, and
   starts each server only on AC power, Energy Mode Automatic and a 1-minute load below 3
   (`conditions.log`). Rounds rotate the engine order.
2. **Make the report:** `python3 tools/bench/report.py test/results/<name>/gpu --out report.md`.
   Each cell is the median over rounds with its spread. Each comparison carries its verdict:
   better or slower only when every round agrees beyond ±10% and both engines vary ≤ 10%.
3. **Copy tables, not adjectives.** Keep "inconclusive" and "not comparable" as they are, and
   keep the manifest's setup. Say where the other engine is faster.
4. **To check one change**, use `tools/bench/ab.sh` (two DynaLM builds or settings, paired rounds),
   then `tools/bench_analyze.py` with a plan of pass/fail gates (`tools/bench/plans/`).

## Model matrix (does every model work, and how fast?)

A sweep over many models, separate from the published engine comparison:

```bash
DYNALM=build/macos-metal/bin/dynalm tools/bench/model_matrix.sh --out test/results/matrix \
  --list tools/bench/models-50.txt --check-only                       # ~4 min: support, links, sizes
DYNALM=build/macos-metal/bin/dynalm LLAMA_BENCH=<path>/llama-bench HF_TOKEN=<token> \
  tools/bench/model_matrix.sh --out test/results/matrix --list tools/bench/models-50.txt --accuracy --bench
```

**Lists:**
- `tools/bench/models-matrix.txt`: 43 GGUF files across architectures and quantizations.
- `tools/bench/models-50.txt`: 50 models, each as a GGUF file and as its Hugging Face
  safetensors repo (`hf:<owner>/<repo>`).
- `tools/bench/models-51-80.txt`: 30 more models (7-35B), the same two formats. Where a repo page
  or a search was given, the GGUF file was resolved from the repo's file list. Their
  full-precision safetensors are mostly above 16 GB: pass a larger `--max-gb` on a bigger
  machine.

For safetensors, only the shards named by `model.safetensors.index.json` are downloaded. Mistral
repos also ship a duplicate `consolidated.safetensors`, which is skipped.

**Per model:**
1. **Check.** GGUF: a 256 KiB support check. Safetensors: `config.json`'s `model_type`. Both
   check the size against `--max-gb` (default 16).
2. **Download.**
3. **Generate** 32 greedy tokens with `--backend auto` (GPU), `cpu` (fast kernels) and
   `native-cpu` (exact). Each path's text is compared with `native-cpu`'s (reported, not gated).
4. **`--accuracy`:** the DD-053 GPU accuracy check.
5. **`--bench`:**
   - DynaLM in-process on the GPU and the CPU: one user (prompt and decode tok/s) and 4 users;
   - with `LLAMA_BENCH`, `llama-bench` pp512 / tg128 on the same GGUF (GPU and CPU), with
     DynaLM ÷ llama.cpp ratios;
   - AC power and load are recorded with each.

   These are short single runs: a screen for failures and outliers. Published numbers come from
   `baseline.sh`.

Downloads are deleted after each model unless `--keep`. Gated repos (meta-llama, google, mpt)
need `HF_TOKEN` with the license accepted on Hugging Face.

**Output:** `results.tsv`, and `summary.md` with these sections: counts, "Needs attention" (each
failure and its reason), the support table, and the speed table.

**Check of `models-51-80.txt` (2026-10-18), 60 entries:**
- 28 pass;
- 13 too large at 16 GB (safetensors);
- 8 gated;
- architectures not supported yet: InternLM2, StarCoder2, DeepSeek-V2 (Coder V2 Lite),
  Command R, Gemma 4, Mistral 3 (Mistral Small 3.1 safetensors);
- Orca 2 has no safetensors.

**Check of `models-50.txt` (2026-10-18), 99 entries:**
- 61 pass;
- 10 gated;
- 3 too large as full-precision safetensors at 16 GB;
- GPT-Neo and OPT: no GGUF exists;
- architectures DynaLM does not support yet: OLMo 2, StarCoder2, StableLM, OpenELM, Falcon,
  InternLM2, MiniCPM3, GPT-NeoX (Pythia), MPT, GPT-Neo.

## How long the engine comparison takes

`tools/bench/baseline.sh` runs 3 engines × 3 rounds per model and device. The default set
(1.5B/8B on the GPU, 1.5B/4B on the CPU) is 36 server starts: about 1 h for the GPU half and
3+ h for the CPU half. A 2,048-token prompt on the 4B CPU model takes about a minute.

`--quick` takes about 45 minutes for the same set:
- 2 rounds;
- prompts 128 and 512;
- GPU users 4 and 16, CPU users 4;
- fewer requests per point.

Use it for a quick look. Use the full run for published numbers: with 2 rounds, fewer results
reach "better" or "slower".

## End to end (everything, unattended, one report)

```bash
OLD_DYNALM=<v0.1.7 dynalm> LLAMA_DIR=<llama.cpp build/bin> LLAMA_COMMIT=<sha> \
  caffeinate -i tools/bench/e2e.sh --out test/results/e2e-<name> [--tier2] [--skip matrix,engines,ab]
```

Phases, run one at a time:
1. **Snapshot** of every binary (checksums in `manifest.txt`).
2. **Model matrix:** `models-e2e-tier1.txt` (27 entries across Qwen, Meta, Google, Microsoft,
   Mistral AI and DeepSeek, quantization types and safetensors), plus `models-providers.txt`
   with `--tier2`. For each model: correctness on GPU / fast CPU / exact CPU, DD-053 GPU
   accuracy, and speed against llama-bench (GPU and CPU).
3. **Engine comparison:** `baseline.sh --quick`, DynaLM / llama.cpp / Ollama servers, 1-16 users,
   GPU and CPU.
4. **A/B against v0.1.7:** GPU (`plans/phase1-gpu.json`) and CPU (`plans/phase2-cpu.json`).
5. **Report:** `REPORT.md`, covering setup, the summary by provider, every model, the engine
   tables and the A/B gates.

Every speed measurement waits for AC power, Energy Mode Automatic and a quiet machine, and the
waits are logged. Tier 1 takes about 8-9 h on a 3 MB/s link, mostly downloads.

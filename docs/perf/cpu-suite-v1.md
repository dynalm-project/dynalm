# CPU suite v1 (fixed 2026-10-25, before any CPU-mission measurement)

The score for the CPU performance mission. Changing a workload, metric, model, prompt, output length,
context or concurrency after seeing results invalidates comparisons: make a v2 instead.

## Settings (every engine)

`tools/bench/baseline.sh --devices cpu --models 1.5b,4b --engines dynalm,llama,ollama` with its
defaults: prompts 128 / 512 / 2,048 at 1 user, 512 at 4 and 8 users, 128 output tokens, greedy,
`ignore_eos`, `CTX=4096` per user, 8 slots, `THREADS` = performance cores, unique prompts per point,
same prompts for every engine in a round, rotated engine order. Release results: 5 rounds,
AC power, Energy Mode Automatic (`REQUIRE_AC=1`, the default).

| Key | File | Quant |
|---|---|---|
| 1.5b | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Q4_K_M |
| 4b | `Qwen3-4B-Q4_K_M.gguf` | Q4_K_M |

## Workloads (score = geometric mean of C1..C8, per reference engine)

| # | Model | Users, prompt | Metric | What it stands for |
|---|---|---|---|---|
| C1 | 1.5b | 1, 128 | output tok/s | small model, short prompt, single user |
| C2 | 1.5b | 1, 2048 | whole request p50 (inverse) | small model, long prompt |
| C3 | 1.5b | 4, 512 | aggregate output tok/s | several users |
| C4 | 1.5b | 8, 512 | aggregate output tok/s | high concurrency |
| C5 | 1.5b | 8, 512 | gap between tokens p95 (inverse) | new prompts arriving while others decode |
| C6 | 4b | 1, 512 | output tok/s | medium model, single user |
| C7 | 4b | 1, 2048 | first token p50 (inverse) | medium model, long-prompt prefill |
| C8 | 4b | 1, 2048 | gap between tokens p50 (inverse) | decode over a long, grown KV cache |

Ratio per workload = DynaLM ÷ reference (latencies: reference ÷ DynaLM), per round, median over
rounds. A workload with a failed or short request (< 95 % of the output) in any round is "not
comparable" and the mean is reported as incomplete; nothing is estimated.

## Memory (Gate B)

| # | Metric |
|---|---|
| M1 | peak server RSS over the whole run, 1.5b (from `/usr/bin/time -l`, median over rounds) |
| M2 | the same, 4b |

Reported as reference ÷ DynaLM. Same model file, quantization, context, slots and KV precision (f16)
for every engine; peak RSS includes weights, KV cache, packed copies and workspaces. KV bytes per
token and incremental memory are reported beside it, not folded into it.

## Not in the mean

- Speculative decoding (`--spec`, prompt lookup) on copy-heavy prompts: separate configuration,
  only against a reference engine with an equivalent feature enabled, with the greedy output
  equality check.
- Phase 8 P1–P4 stay reported for continuity.

## Tools

```bash
python3 tools/bench/cpu_suite.py score <run dirs>              # vs llama.cpp / Ollama
python3 tools/bench/cpu_suite.py compare <baseline> <candidate>  # DynaLM A/B: keep / reject
```

A/B rule (`compare`): keep only when a workload improves by more than its noise band (the larger
round-to-round half-spread of the two runs) plus 3 %, no workload loses more than its noise band
plus 3 %, and peak RSS does not grow more than 5 %. Development A/B runs may use `REQUIRE_AC=0`
(marked "NOT controlled" in the manifest); they decide keep/reject between two DynaLM builds
measured back to back, never a result against a reference engine.

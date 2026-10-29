# `dynalm cpu-report`: CPU benchmark reports from testers (design)

Status: design, for review. Nothing here is implemented yet.

## Goal

DynaLM's own CPU kernels are measured only on one Apple M4 Pro, where they beat llama.cpp
(1.2–1.7×, docs/perf/cpu-model-sweep.md). On every other CPU, ggml's kernels are the default,
because nobody has measured DynaCore's kernels there.

`dynalm cpu-report` lets anyone measure that on their own machine, from the normal release
binary, with one command:

```sh
dynalm cpu-report                 # about 1-2 hours; writes cpu-report-<host>-<date>.json and .md
```

They attach the report to issue #31. The reports show, per CPU architecture, where DynaCore's
kernels win or lose against ggml and llama.cpp, and that decides what to optimize next. A
CPU's default changes to DynaCore's kernels only where the reports show a win: the rule already
used for Apple M4.

No per-architecture branches: one branch, one release. DynaLM already picks its kernels at run
time from the CPU's features (`cpu_info`), and experimental kernels go into main behind a
switch (as `DYNACORE_VNNI` does today), so the report can compare them in the same run.

## What it runs

1. **Machine.**
   - CPU model, core counts (performance and efficiency), cache sizes.
   - Every feature `cpu_info` detects: AVX2, FMA, AVX-VNNI, AVX-512 (F/BW/VL/VNNI/BF16), AMX,
     NEON, dotprod, i8mm, SVE, SME.
   - RAM, OS and version, power source (where the OS reports it), DynaLM version and build.
   - The same data as `dynalm doctor --json`.
2. **Preconditions.** As `tools/bench/baseline.sh` checks today:
   - on AC power (laptops);
   - 1-minute load below a limit before each point;
   - no other DynaLM or llama.cpp server on the ports used.

   When a check fails, the report is still written but marked `uncontrolled`. It is never
   silently mixed with controlled results.
3. **Models**, downloaded with `dynalm pull` (sha256-checked):

   | Model | File | Size |
   |---|---|---|
   | Qwen2.5-1.5B-Instruct | Q4_K_M | 1.1 GB |
   | Qwen3-4B | Q4_K_M | 2.5 GB |
   | Qwen3-8B | Q4_K_M | 5.0 GB |

   - **Skips:** the 8B is skipped when RAM is below 12 GB, and the report says so.
   - **Selection:** `--models 1.5b,4b` picks a subset.
4. **Engines and kernel paths.** All on the same machine, same files, same client:

   | Path | How |
   |---|---|
   | DynaLM, default | `dynalm serve --cpu` |
   | DynaLM, ggml CPU kernels | `--backend ggml-cpu` |
   | DynaLM, DynaCore kernels | `--backend cpu --mode exact`: forces DynaCore's kernels on any CPU (`autotune.cpp`) |
   | DynaLM, each DynaCore tier the CPU supports | One server per tier, with the tier's switch: `DYNACORE_QMM_TIER=neon_dot / neon_i8mm`, `DYNACORE_SME=0/1`, `DYNACORE_VNNI=0/1`. New tiers (AVX-512, AMX, SVE) are added here when they exist |
   | llama.cpp | A pinned llama.cpp release for the platform, downloaded and checked against a sha256 recorded in DynaLM's source; `llama-server -ngl 0` |

5. **Workload.** Fixed, so reports are comparable; the same as the M4 measurements:
   - prompt 512 tokens, 128 output tokens, `ignore_eos`;
   - 1 user and 4 users;
   - 2 rounds, engine order alternated between rounds;
   - threads: DynaLM's default (physical performance cores); llama.cpp gets the same count;
   - `dynalm benchmark --url` is the client for every server.
6. **Accuracy.** For every DynaLM path, on the 1.5B model: the same next token as llama.cpp
   (top-1 agreement) and the mean KL divergence over 10 × 512 tokens of a fixed text, the
   DD-053 check used for the M4 results. A path that fails it is reported as **failed**, whatever
   its speed. This needs `llama-perplexity --kl-divergence-base` (from the same llama.cpp release)
   and a perplexity scorer inside `dynalm` (today `bench_ppl`, which is not shipped; see Open
   questions).

## The report

- `cpu-report-<host>-<date>.json`: everything above, every measured point, raw.
- `cpu-report-<host>-<date>.md`: a short summary to paste into the issue:
  - the machine;
  - one table per model, with tokens/s, first token and token-gap p50/p99, per path;
  - the ratio of each DynaLM path to llama.cpp;
  - the accuracy results;
  - the controlled/uncontrolled verdict.

Prompts and outputs are not stored, only counts and timings (as `--request-log`). Host names
can be replaced by `--anonymous`.

## What it is not

- **Not a GPU benchmark.** CPU only.
- **Not a release gate.** One tester's machine decides nothing alone. Results go into
  docs/perf only after a maintainer reviews them, as the M4 results did.
- **No uploads.** The tester decides what to share.

## Coverage plan

| Architecture | Status | Machine |
|---|---|---|
| ARM SME (Apple M4) | Done (docs/perf/cpu-model-sweep.md) | M4 Pro |
| ARM NEON + dotprod + i8mm | Planned | Apple M3 (maintainer) |
| x86-64 AVX2 + AVX-VNNI, hybrid cores | Planned | Intel Core i7-1255U, Windows and WSL2 Linux (maintainer) |
| x86-64 AVX-512 + VNNI | Missing | AMD Zen 4/5 or cloud |
| Intel AMX | Missing | Xeon Sapphire Rapids+ (cloud) |
| ARM SVE, ARM Linux servers | Missing | AWS Graviton (cloud) |

## Open questions (for review)

1. **Perplexity scorer in `dynalm`.** Accuracy needs `bench_ppl`'s scorer in the shipped binary,
   for example `dynalm cpu-report` calling it internally (not a new user-facing command).
2. **llama.cpp download.** Pinning one llama.cpp release and its sha256 per platform in DynaLM's
   source adds a dependency to keep current. The alternative, `--llama PATH` to a llama.cpp the
   tester installed, is simpler but less comparable. Proposal: both, with the pinned download as
   the default.
3. **Run time.** 3 models × about 5 paths × 2 rounds × 2 user counts is about 1–2 hours on a
   laptop. Proposal: `--quick` (1.5B only, 1 round), about 15 minutes, for first reports.

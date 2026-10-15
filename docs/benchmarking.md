# Benchmarking DynaLM

There are two supported ways to measure performance. Both record exactly which executable and
model they ran.

| question | tool |
|---|---|
| Is build / setting B faster than A on this machine? (paired A/B) | `tools/next_sweep.ps1` (runs) + `tools/bench_analyze.py` (analysis) |
| How does DynaLM compare with llama.cpp / Ollama? | `tools/compare_engines.ps1` (wraps `tools/compare_engines.py`) |

Both drive the same load generator, `dynalm benchmark`. In-process for A/B runs; over
HTTP with `--url` for engine comparisons. It produces fixed output lengths (EOS is ignored)
and unique prompts. It also writes per-request logs (`--request-log`), an optional scheduler
step trace (`--step-trace`), and per-point diagnostics (`diag.matmul_kernels`: which CPU
kernels ran).

## Paired A/B runs

1. **Write the plan first:** arms, workloads, and the gates in a JSON file (format: the
   docstring of `tools/bench_analyze.py`).
2. **Run each measurement session separately.** Example (stage R2, the 0.1.4 release
   campaign):

   ```powershell
   $env:DYNALM_BASE_EXE = "C:\path\to\reference\dynalm.exe"
   $env:DYNALM_CAND_EXE = "C:\path\to\candidate\dynalm.exe"
   pwsh -NoProfile -Command "& 'tools\next_sweep.ps1' -Stage R2 -Session 1 -Out results\my-campaign\raw -Resume"
   ```

   - **Before every run, the runner waits for:**
     - at least 6 GB of free RAM;
     - no other dynalm, llama.cpp, Ollama, benchmark or test process. It never stops one.
   - It then idles for 60 s and requires total CPU below 10% for 5 s.
   - **Recorded per run** in `runs.jsonl`:
     - executable path and SHA-256, model SHA-256;
     - git HEAD and dirty-path count;
     - free RAM, the busiest background processes, power plan and source.
   - Arm order rotates with the session number.
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

## Tests of the tools

| command | checks |
|---|---|
| `python tools/test_compare_engines.py` | comparability marking, ranking, executable selection, no appending to old results |
| `python tools/test_bench_analyze.py` | unit counts, invalid runs (exit code, wrong executable, early end, errors), missing metrics, unreliable references, the three gate rules, bootstrap intervals |

## Older scripts

`tools/phase8_sweep.ps1`, `tools/phase9_sweep.ps1`, `tools/pk8_validation.ps1`,
`tools/e2e_engines.ps1` and their analyzers reproduce the measurements of earlier internal
phases. They are kept for reproducibility, not for new measurements.

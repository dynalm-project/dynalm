#!/usr/bin/env python3
"""REPORT.md for a tools/bench/e2e.sh run: setup, a per-provider summary, every
model's support / accuracy / speed against llama.cpp, the multi-user engine
comparison, and the A/B gates of this build against v0.1.7.

  python3 tools/bench/e2e_report.py <e2e run dir>

Only measured values appear; a missing measurement is "–". Ratios > 1 mean
DynaLM is faster. Speed rows of the model matrix are short single runs (a
screen); the engine comparison and the A/B are the controlled measurements.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROVIDERS = [("deepseek", "DeepSeek"), ("codellama", "Meta"), ("llama", "Meta"), ("qwen", "Alibaba (Qwen)"),
             ("gemma", "Google"), ("phi", "Microsoft"), ("orca", "Microsoft"), ("codestral", "Mistral AI"),
             ("mistral", "Mistral AI")]


def provider(model: str) -> str:
    m = model.lower()
    # A distill belongs to its maker (DeepSeek), whatever base it names.
    for k, v in PROVIDERS:
        if k in m:
            return v
    return "Other"


def num(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "", "-", "fail") else None
    except ValueError:
        return None


def ratio(a: str, b: str) -> float | None:
    x, y = num(a), num(b)
    return x / y if x and y else None


def fx(v: float | None, unit: str = "×") -> str:
    return f"{v:.2f}{unit}" if v is not None else "–"


def med(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return st.median(v) if v else None


def on_gpu(r: dict) -> bool:
    return r["auto"] == "ok" and not r["auto_device"].startswith(("CPU", "ggml-CPU"))


def matrix_section(out: Path) -> tuple[list[str], list[str]]:
    rows = []
    for tsv in sorted(out.glob("matrix-*/results.tsv")):
        rows += list(csv.DictReader(open(tsv), delimiter="\t"))
    if not rows:
        return ["## Models", "", "Not run.", ""], []
    ran = [r for r in rows if r["native"] != "-"]
    summary, L = [], ["## Models: support, accuracy and speed", ""]
    # Per provider.
    L += ["### By provider", "",
          "Speed: medians of the per-model ratios DynaLM ÷ llama.cpp (llama-bench on the same GGUF). "
          "Prompt = prompt tokens/s, decode = generated tokens/s (one user).", "",
          "| Provider | Models | Ran | GPU ok | CPU ok | Exact CPU ok | DD-053 PASS | GPU prompt | GPU decode | CPU prompt | CPU decode |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    provs = sorted({provider(r["model"]) for r in rows})
    for p in provs + ["All"]:
        rs = [r for r in rows if p == "All" or provider(r["model"]) == p]
        rr = [r for r in rs if r["native"] != "-"]
        acc = [r for r in rr if r["accuracy"] != "-"]
        L.append(f"| **{p}** | {len(rs)} | {len(rr)} | {sum(on_gpu(r) for r in rr)} | {sum(r['cpu'] == 'ok' for r in rr)} | "
                 f"{sum(r['native'] == 'ok' for r in rr)} | {sum(r['accuracy'].startswith('PASS') for r in acc)}/{len(acc)} | "
                 f"{fx(med([ratio(r['gpu_prompt_tok_s'], r['llama_gpu_pp512']) for r in rr]))} | "
                 f"{fx(med([ratio(r['gpu_decode_tok_s'], r['llama_gpu_tg128']) for r in rr]))} | "
                 f"{fx(med([ratio(r['cpu_prompt_tok_s'], r['llama_cpu_pp512']) for r in rr]))} | "
                 f"{fx(med([ratio(r['cpu_decode_tok_s'], r['llama_cpu_tg128']) for r in rr]))} |")
    allr = rows
    gd = med([ratio(r["gpu_decode_tok_s"], r["llama_gpu_tg128"]) for r in ran])
    cd = med([ratio(r["cpu_decode_tok_s"], r["llama_cpu_tg128"]) for r in ran])
    summary += [f"- **Models:** {len(allr)} entries; {len(ran)} downloaded and run. On the GPU: {sum(on_gpu(r) for r in ran)}. "
                f"Fast CPU path: {sum(r['cpu'] == 'ok' for r in ran)}. Exact CPU path: {sum(r['native'] == 'ok' for r in ran)}.",
                f"- **GPU accuracy (DD-053):** PASS {sum(r['accuracy'].startswith('PASS') for r in ran)} of "
                f"{sum(r['accuracy'] != '-' for r in ran)} checked.",
                f"- **Speed vs llama.cpp (median over models, one user):** GPU decode {fx(gd)}, CPU decode {fx(cd)} "
                f"(see Models; short single runs)."]
    # Problems.
    bad = [r for r in rows if r["supported"] != "yes" or "fail" in (r["auto"], r["cpu"], r["native"])
           or "timeout" in (r["auto"], r["cpu"], r["native"]) or r["accuracy"].startswith(("FAIL", "error"))]
    if bad:
        L += ["", "### Needs attention", "", "| Model | Provider | Format | Check | GPU | Fast CPU | Exact CPU | DD-053 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in bad:
            L.append(f"| `{r['model']}` | {provider(r['model'])} | {r['format']} | {r['supported']} | {r['auto']} | {r['cpu']} | "
                     f"{r['native']} | {r['accuracy']} |")
    # Every model.
    L += ["", "### Every model", "",
          "Tokens/s from `dynalm benchmark` in-process (one user: prompt = 512 / TTFT, decode = 1000 / ITL p50) and "
          "`llama-bench` (pp512, tg128), on the same file; ratio = DynaLM ÷ llama.cpp. \"= exact\": the 32-token greedy "
          "answer equals the native (DD-053) CPU path's (reported, not gated).", "",
          "| Provider | Model | Format / quant | GB | GPU | = exact | DD-053 | GPU prompt (llama, ratio) | GPU decode (llama, ratio) | "
          "CPU prompt (llama, ratio) | CPU decode (llama, ratio) | 4 users GPU / CPU | Conditions |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (provider(r["model"]), r["model"])):
        if r["native"] == "-":
            continue
        cell = lambda d, l: f"{r[d]} ({r[l]}, {fx(ratio(r[d], r[l]))})" if r[d] != "-" else "–"
        L.append(f"| {provider(r['model'])} | `{r['model']}` | {r['format']} {r['quant']} | {r['size_gb']} | "
                 f"{r['auto_device'] if on_gpu(r) else 'CPU'} | {r['auto_matches_native']}/{r['cpu_matches_native']} | {r['accuracy']} | "
                 f"{cell('gpu_prompt_tok_s', 'llama_gpu_pp512')} | {cell('gpu_decode_tok_s', 'llama_gpu_tg128')} | "
                 f"{cell('cpu_prompt_tok_s', 'llama_cpu_pp512')} | {cell('cpu_decode_tok_s', 'llama_cpu_tg128')} | "
                 f"{r['gpu_4user_tok_s']} / {r['cpu_4user_tok_s']} | {r['bench_conditions']} |")
    L += ["", "`= exact` columns: GPU / fast CPU. `DD-053`: GPU accuracy (perplexity change at decode).", ""]
    return L, summary


def engines_section(out: Path) -> tuple[list[str], list[str]]:
    runs = [d for d in (out / "engines" / "gpu", out / "engines" / "cpu") if d.is_dir()]
    if not runs:
        return ["## Engine comparison", "", "Not run.", ""], []
    spec = importlib.util.spec_from_file_location("bench_report", HERE / "report.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    L = ["## Engine comparison: DynaLM vs llama.cpp vs Ollama (servers, 1 to 16 users)", "",
         "`tools/bench/baseline.sh --quick`: the same client for every engine, rotated rounds, each server start gated on "
         "AC power and a quiet machine. Verdict: better/slower only when every round agrees beyond ±10% and both engines "
         "vary ≤ 10%.", ""]
    summary = []
    for d in runs:
        lines = rep.report(d)
        L += [("#" + l if l.startswith("#") else l) for l in lines]
        tok = [l for l in lines if l.startswith("| ")]
        better = sum("**better**" in l for l in tok)
        slower = sum("**slower**" in l for l in tok)
        summary.append(f"- **Engines ({d.name.upper()}):** DynaLM better in {better} comparisons, slower in {slower} "
                       f"(the rest parity / inconclusive / not comparable; see Engine comparison).")
    return L, summary


def ab_section(out: Path) -> tuple[list[str], list[str]]:
    L, summary = ["## This build vs v0.1.7 (A/B gates)", ""], []
    found = False
    for dev in ("gpu", "cpu"):
        md = out / "ab" / dev / "report" / "analysis.md"
        js = out / "ab" / dev / "report" / "analysis.json"
        if not md.exists():
            continue
        found = True
        L += [f"### {dev.upper()}", ""] + [("##" + l if l.startswith("#") else l) for l in md.read_text().splitlines()] + [""]
        try:
            res = json.load(open(js))
            ev = res.get("evaluations", [])
            counts = {}
            for e in ev:
                counts[e.get("verdict", "?")] = counts.get(e.get("verdict", "?"), 0) + 1
            summary.append(f"- **A/B {dev.upper()} (this build vs v0.1.7):** " +
                           ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + " gate evaluations.")
        except (OSError, ValueError):
            pass
    if not found:
        L += ["Not run.", ""]
    return L, summary


def main() -> int:
    out = Path(sys.argv[1])
    man = (out / "manifest.txt").read_text().strip() if (out / "manifest.txt").exists() else ""
    phases = (out / "phases.txt").read_text().strip() if (out / "phases.txt").exists() else ""
    m_lines, m_sum = matrix_section(out)
    e_lines, e_sum = engines_section(out)
    a_lines, a_sum = ab_section(out)
    conds = []
    for f in sorted(out.rglob("conditions.log")):
        t = f.read_text().splitlines()
        conds.append(f"{f.relative_to(out)}: {sum(' start:' in l for l in t)} starts, {sum(' waiting:' in l for l in t)} waits")
    L = ["# DynaLM end-to-end report", "", "## Setup", "", "```text", man, "```", "",
         "Phase status:", "", "```text", phases or "(none recorded)", "```", "",
         "## Summary", ""] + (m_sum + e_sum + a_sum or ["(no phase produced results)"]) + [
         "", "Run conditions (speed runs wait for AC power, Energy Mode Automatic and a 1-minute load below 3):", ""] + \
        [f"- {c}" for c in conds] + [""] + m_lines + e_lines + a_lines + [
         "## Method and limits", "",
         "- Model-matrix speed rows are short single runs per model: they find failures and outliers, not small differences.",
         "  DynaLM's prompt rate includes its request path (TTFT); llama-bench's pp512 is the bare forward pass.",
         "- The engine comparison and the A/B are the controlled measurements (rounds, rotation, gating, verdict rules).",
         "- Gated Hugging Face repos need HF_TOKEN; without it they are listed, not run.",
         "- Accuracy (DD-053) is checked on the GPU against DynaLM's fp32 CPU reference; the fast CPU path (ggml kernels) is "
         "at llama.cpp-level accuracy by design (CHANGELOG, Unreleased).",
         ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(["## Summary"] + m_sum + e_sum + a_sum))
    return 0


if __name__ == "__main__":
    sys.exit(main())

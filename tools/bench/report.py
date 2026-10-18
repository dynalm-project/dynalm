#!/usr/bin/env python3
"""Markdown report of a tools/bench/baseline.sh run (DynaLM vs llama.cpp vs Ollama).

  python3 tools/bench/report.py <run dir> [<run dir> ...] [--out FILE]

Each run dir holds manifest.txt, conditions.log, errors.txt (optional) and
<model>-<cpu|gpu>-<engine>-r<round>.jsonl result files (one JSON line per
point). The report carries the setup from the manifest, how many server
starts met the quiet-machine gate, every error, and per point the median
over rounds with each engine's spread, and DynaLM's ratio to each other
engine with a verdict:

  better / slower  every round beyond +/-10% the same way, both engines' spread <= 10%;
  parity           every round within +/-10%;
  inconclusive     otherwise;
  not comparable   a request failed or ended early (< 95% of the output tokens).

Ratios are oriented so > 1 means DynaLM is better (more tokens/s, less latency).
Nothing is estimated: a point missing on either side is left out.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ENGINES = ("dynalm", "llama", "ollama")
LABEL = {"dynalm": "DynaLM", "llama": "llama.cpp", "ollama": "Ollama"}
FILE = re.compile(r"^(?P<model>.+)-(?P<dev>cpu|gpu)-(?P<eng>dynalm|llama|ollama)-r(?P<round>\d+)\.jsonl$")
# name, getter, higher is better
METRICS = [
    ("Output tok/s", lambda d: d["output_tok_s"], True),
    ("First token p50 (ms)", lambda d: d["ttft_ms"]["p50"], False),
    ("First token p95 (ms)", lambda d: d["ttft_ms"]["p95"], False),
    ("Gap between tokens p50 (ms)", lambda d: d["itl_ms"]["p50"], False),
    ("Gap between tokens p99 (ms)", lambda d: d["itl_ms"]["p99"], False),
    ("Whole request p50 (ms)", lambda d: d["e2e_ms"]["p50"], False),
]


def spread(v: list[float]) -> float:
    m = st.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


def verdict(ratios: list[float], sa: float, sb: float, comparable: bool) -> str:
    if not comparable:
        return "not comparable"
    if len(ratios) >= 2 and all(0.9 <= x <= 1.1 for x in ratios):
        return "parity"
    if len(ratios) < 2 or sa > 10 or sb > 10:
        return "inconclusive"
    if all(x > 1.1 for x in ratios):
        return "**better**"
    if all(x < 0.9 for x in ratios):
        return "**slower**"
    return "inconclusive"


def load(run: Path) -> dict:
    points: dict = {}
    for f in sorted(run.glob("*-r*.jsonl")):
        if f.name.endswith(".requests.jsonl"):
            continue
        m = FILE.match(f.name)
        if not m:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            ok = (d.get("errors", 0) == 0 and d.get("completed") == d.get("requests")
                  and (d.get("mean_completion_tokens") or 0) >= 0.95 * (d.get("output_tokens") or 0))
            key = (m["model"], m["dev"], d["concurrency"], d["prompt_tokens"])
            points.setdefault(key, {}).setdefault(m["eng"], {})[int(m["round"])] = (d, ok)
    return points


def fmt(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def report(run: Path) -> list[str]:
    out = [f"## {run.name}", ""]
    man = run / "manifest.txt"
    if man.exists():
        out += ["Setup (manifest.txt):", "", "```text", *man.read_text(encoding="utf-8").strip().splitlines(), "```", ""]
    cond = run / "conditions.log"
    if cond.exists():
        lines = cond.read_text(encoding="utf-8").splitlines()
        starts = [l for l in lines if " start:" in l]
        waits = [l for l in lines if " waiting:" in l]
        out.append(f"Run conditions: {len(starts)} server starts, each on AC power, Energy Mode Automatic and a "
                   f"1-minute load below the limit ({len(waits)} waits logged).")
        out.append("")
    err = run / "errors.txt"
    if err.exists() and err.read_text(encoding="utf-8").strip():
        out += ["Errors:", "", *[f"- {l}" for l in err.read_text(encoding="utf-8").strip().splitlines()], ""]
    points = load(run)
    groups: dict = {}
    for (model, dev, conc, prompt), engines in points.items():
        groups.setdefault((model, dev), []).append((conc, prompt, engines))
    for (model, dev), rows in sorted(groups.items()):
        rows.sort()
        others = [e for e in ENGINES[1:] if any(e in r[2] for r in rows)]
        out += [f"### {model} on the {dev.upper()}", ""]
        for name, get, higher in METRICS:
            head = ["Users, prompt", *[LABEL[e] for e in ENGINES if any(e in r[2] for r in rows)],
                    *[f"vs {LABEL[o]}" for o in others]]
            out += [f"**{name}** (median over rounds, ± spread)", "", "| " + " | ".join(head) + " |",
                    "|" + "---|" * len(head)]
            for conc, prompt, E in rows:
                cells = [f"{conc}, {prompt}"]
                for e in ENGINES:
                    if not any(e in r[2] for r in rows):
                        continue
                    if e not in E:
                        cells.append("–")
                        continue
                    v = [get(E[e][r][0]) for r in sorted(E[e])]
                    flag = "" if all(E[e][r][1] for r in E[e]) else " (!)"
                    cells.append(f"{fmt(st.median(v))} ±{spread(v):.0f}%{flag}")
                for o in others:
                    if "dynalm" not in E or o not in E:
                        cells.append("–")
                        continue
                    rs = sorted(set(E["dynalm"]) & set(E[o]))
                    a = [get(E["dynalm"][r][0]) for r in rs]
                    b = [get(E[o][r][0]) for r in rs]
                    ratios = [(x / y if higher else y / x) for x, y in zip(a, b) if x and y]
                    comparable = all(E["dynalm"][r][1] and E[o][r][1] for r in rs)
                    cells.append(f"{st.median(ratios):.2f}× {verdict(ratios, spread(a), spread(b), comparable)}"
                                 if ratios else "–")
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
        out += ["(!) = a request failed or ended early in some round; such points are not compared.", ""]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    lines = ["# Benchmark report", "",
             "Generated by `tools/bench/report.py` from `tools/bench/baseline.sh` runs. Ratios > 1 mean "
             "DynaLM is better.", ""]
    for run in a.runs:
        if not run.is_dir():
            print(f"report: {run} is not a directory", file=sys.stderr)
            return 2
        lines += report(run)
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

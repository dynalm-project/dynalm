#!/usr/bin/env python3
"""CPU suite v1 (docs/perf/cpu-suite-v1.md): score and A/B compare.

  python3 tools/bench/cpu_suite.py score <run dir> [<run dir> ...]
      DynaLM vs llama.cpp and vs Ollama, per workload C1..C8 (median over
      rounds of the per-round ratio) and their geometric mean, plus the
      memory metrics M1..M2 (peak server RSS, median over rounds).
  python3 tools/bench/cpu_suite.py compare <baseline dir> <candidate dir>
      DynaLM vs DynaLM (an experiment): per workload the candidate/baseline
      ratio, the noise band (the larger round-to-round spread of the two),
      and a keep / reject verdict by the rules in the suite document.

Reads tools/bench/baseline.sh results through tools/bench/report.py's loader.
A workload missing on either side, or with a failed / short request in any
round, is "missing" / "not comparable", never estimated. Exit status: 0, or
for compare 0 = keep, 1 = reject, 2 = inconclusive / incomplete.
"""
from __future__ import annotations

import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402

# id, model, users, prompt tokens, metric, getter, higher is better.  Fixed: changing it after
# seeing results invalidates comparisons (version the suite instead).
SUITE = [
    ("C1", "1.5b", 1, 128, "output tok/s", lambda d: d["output_tok_s"], True),
    ("C2", "1.5b", 1, 2048, "whole request p50", lambda d: d["e2e_ms"]["p50"], False),
    ("C3", "1.5b", 4, 512, "aggregate output tok/s", lambda d: d["output_tok_s"], True),
    ("C4", "1.5b", 8, 512, "aggregate output tok/s", lambda d: d["output_tok_s"], True),
    ("C5", "1.5b", 8, 512, "gap between tokens p95", lambda d: d["itl_ms"]["p95"], False),
    ("C6", "4b", 1, 512, "output tok/s", lambda d: d["output_tok_s"], True),
    ("C7", "4b", 1, 2048, "first token p50", lambda d: d["ttft_ms"]["p50"], False),
    ("C8", "4b", 1, 2048, "gap between tokens p50", lambda d: d["itl_ms"]["p50"], False),
]
MEMORY = [("M1", "1.5b"), ("M2", "4b")]
RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)
SERVER = re.compile(r"(?P<model>.+)-cpu-(?P<eng>dynalm|llama|ollama)-r(?P<round>\d+)\.server\.log$")

# A/B rules: a change is kept only when its target improves beyond noise and nothing regresses.
MIN_GAIN = 0.03   # smallest improvement that counts, on top of the noise band
MAX_LOSS = 0.03   # largest tolerated loss on any other workload, on top of the noise band


def load(runs: list[Path]) -> dict:
    points: dict = {}
    for r in runs:
        for key, engines in report.load(r).items():
            if key[1] != "cpu":
                continue
            for eng, rounds in engines.items():
                points.setdefault((key[0], key[2], key[3]), {}).setdefault(eng, {}).update(rounds)
    return points


def peak_rss(runs: list[Path]) -> dict:
    """{(model, engine): [peak RSS MiB per round]} from the /usr/bin/time -l lines of the server logs."""
    out: dict = {}
    for r in runs:
        for f in sorted(r.glob("*.server.log")):
            m = SERVER.match(f.name)
            if not m:
                continue
            v = RSS.search(f.read_text(encoding="utf-8", errors="replace"))
            if v:
                out.setdefault((m["model"], m["eng"]), []).append(int(v.group(1)) / 1048576)
    return out


def values(E: dict, eng: str, get) -> tuple[list[float], bool]:
    rounds = E.get(eng, {})
    ok = bool(rounds) and all(v[1] for v in rounds.values())
    return [get(rounds[r][0]) for r in sorted(rounds) if get(rounds[r][0])], ok


def spread(xs: list[float]) -> float:
    return (max(xs) - min(xs)) / st.median(xs) / 2 if len(xs) > 1 else 0.0


def geomean(v: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in v) / len(v))


def score(runs: list[Path]) -> list[str]:
    P = load(runs)
    refs = report.ENGINES[1:]
    out = ["| Workload | Metric | " + " | ".join(f"vs {report.LABEL[e]}" for e in refs) + " |",
           "|---|---|" + "---|" * len(refs)]
    ratios: dict = {e: [] for e in refs}
    missing: dict = {e: [] for e in refs}
    for cid, model, users, prompt, name, get, higher in SUITE:
        E = P.get((model, users, prompt), {})
        cells = [f"{cid} {model} {users}u {prompt}p", name]
        for e in refs:
            rs = sorted(set(E.get("dynalm", {})) & set(E.get(e, {})))
            ok = bool(rs) and all(E["dynalm"][r][1] and E[e][r][1] for r in rs)
            rr = []
            for r in rs:
                x, y = get(E["dynalm"][r][0]), get(E[e][r][0])
                if x and y:
                    rr.append(x / y if higher else y / x)
            if not rs:
                cells.append("missing")
                missing[e].append(cid)
            elif not ok or not rr:
                cells.append("not comparable")
                missing[e].append(cid)
            else:
                m = st.median(rr)
                ratios[e].append(m)
                cells.append(f"{m:.2f}× (±{spread(rr) * 100:.0f}%)")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    for e in refs:
        v = ratios[e]
        note = f" (incomplete: {', '.join(missing[e])} missing)" if missing[e] else ""
        out.append(f"Geometric mean vs {report.LABEL[e]}: " +
                   (f"{geomean(v):.2f}× over {len(v)} of {len(SUITE)} workloads{note}" if v else "no data"))
    rss = peak_rss(runs)
    out += ["", "| Memory | Peak server RSS (MiB, median) | " + " | ".join(f"{report.LABEL[e]} ÷ DynaLM" for e in refs) + " |",
            "|---|---|" + "---|" * len(refs)]
    for mid, model in MEMORY:
        d = rss.get((model, "dynalm"))
        cells = [f"{mid} {model}", f"{st.median(d):,.0f}" if d else "missing"]
        for e in refs:
            o = rss.get((model, e))
            cells.append(f"{st.median(o) / st.median(d):.2f}× ({st.median(o):,.0f})" if d and o else "missing")
        out.append("| " + " | ".join(cells) + " |")
    return out


def compare(base: Path, cand: Path) -> tuple[list[str], int]:
    B, C = load([base]), load([cand])
    out = ["| Workload | Metric | Baseline | Candidate | Candidate ÷ baseline | Noise | Verdict |",
           "|---|---|---|---|---|---|---|"]
    gains, losses, incomplete = [], [], []
    for cid, model, users, prompt, name, get, higher in SUITE:
        b, bok = values(B.get((model, users, prompt), {}), "dynalm", get)
        c, cok = values(C.get((model, users, prompt), {}), "dynalm", get)
        if not b or not c or not bok or not cok:
            out.append(f"| {cid} | {name} | {'–' if not b else f'{st.median(b):.1f}'} | "
                       f"{'–' if not c else f'{st.median(c):.1f}'} | – | – | incomplete |")
            incomplete.append(cid)
            continue
        mb, mc = st.median(b), st.median(c)
        ratio = mc / mb if higher else mb / mc  # > 1: candidate better
        noise = max(spread(b), spread(c))
        if ratio > 1 + noise + MIN_GAIN:
            verdict = "better"
            gains.append(cid)
        elif ratio < 1 - noise - MAX_LOSS:
            verdict = "WORSE"
            losses.append(cid)
        else:
            verdict = "within noise"
        out.append(f"| {cid} | {name} | {mb:.1f} | {mc:.1f} | {ratio:.3f}× | ±{noise * 100:.0f}% | {verdict} |")
    rb, rc = peak_rss([base]), peak_rss([cand])
    for mid, model in MEMORY:
        b, c = rb.get((model, "dynalm")), rc.get((model, "dynalm"))
        if not b or not c:
            continue
        ratio = st.median(b) / st.median(c)
        verdict = "better" if ratio > 1.05 else "WORSE" if ratio < 0.95 else "same"
        if verdict == "WORSE":
            losses.append(mid)
        if verdict == "better":
            gains.append(mid)
        out.append(f"| {mid} | peak RSS MiB | {st.median(b):,.0f} | {st.median(c):,.0f} | {ratio:.3f}× | – | {verdict} |")
    if losses:
        decision, code = f"REJECT (regressions: {', '.join(losses)})", 1
    elif gains:
        decision, code = f"KEEP (better: {', '.join(gains)}" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else "") + ")", 0
    else:
        decision, code = "INCONCLUSIVE (no workload beyond noise)" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ""), 2
    return out + ["", f"Decision: {decision}"], code


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "score":
        print("\n".join(score([Path(a) for a in sys.argv[2:]])))
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        lines, code = compare(Path(sys.argv[2]), Path(sys.argv[3]))
        print("\n".join(lines))
        return code
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

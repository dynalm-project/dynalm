#!/usr/bin/env python3
"""Analyze a repeated engine comparison (tools/e2e_engines.ps1).

  python tools/e2e_engines_analyze.py results/e2e-engines-<ts> [--baseline DynaLM]

Reads every run*/results.jsonl (rows written by compare_engines.py) and the
run records. Per (engine, concurrency): median [min-max] over repetitions of
aggregate output tok/s, per-user generation speed (1000 / ITL p50), TTFT
p50/p95, ITL p50/p95/p99, E2E p50/p95; errors and completion counts.
Differences are relative to the baseline engine's median. A difference is
"separated" only when every repetition of one engine is on the same side of
every repetition of the other (ranges do not overlap); otherwise
"overlapping". With 3 repetitions this is evidence of a consistent
difference, not a significance test.

Checks (exit code 1 on any): engine exit codes, errors, completed ==
requests, and mean completion tokens == requested output. Engines whose
completions are shorter than requested (Ollama ignores ignore_eos) are
flagged: their throughput and E2E latency are not comparable.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

METRICS = [
    ("output tok/s", lambda r: r["output_tok_s"], True),
    ("per-user tok/s", lambda r: 1000.0 / r["itl_ms"]["p50"] if r["itl_ms"]["p50"] else 0.0, True),
    ("TTFT p50 ms", lambda r: r["ttft_ms"]["p50"], False),
    ("TTFT p95 ms", lambda r: r["ttft_ms"]["p95"], False),
    ("ITL p50 ms", lambda r: r["itl_ms"]["p50"], False),
    ("ITL p95 ms", lambda r: r["itl_ms"]["p95"], False),
    ("ITL p99 ms", lambda r: r["itl_ms"]["p99"], False),
    ("E2E p50 ms", lambda r: r["e2e_ms"]["p50"], False),
    ("E2E p95 ms", lambda r: r["e2e_ms"]["p95"], False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--baseline", default="DynaLM")
    args = ap.parse_args()
    root = Path(args.dir)
    problems, rows = [], {}
    for rec_path in sorted(root.glob("run*.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8-sig"))
        if rec.get("dry_run"):
            problems.append(f"{rec['name']}: dry-run record")
            continue
        if rec.get("exit_code", 0) != 0:
            problems.append(f"{rec['name']}: exit code {rec.get('exit_code')}")
        res = root / rec["name"] / "results.jsonl"
        if not res.exists():
            problems.append(f"{rec['name']}: no results.jsonl (engine failed? see {rec['name']}.log)")
            continue
        for line in res.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["_rep"] = rec["rep"]
            rows.setdefault((r["label"], int(r["concurrency"])), []).append(r)
    short = set()
    for (eng, c), rs in rows.items():
        for r in rs:
            tag = f"{eng} c{c} rep{r['_rep']}"
            if r["errors"]:
                problems.append(f"{tag}: {r['errors']} errors")
            if r["completed"] != r["requests"]:
                problems.append(f"{tag}: completed {r['completed']} of {r['requests']}")
            if round(r["mean_completion_tokens"]) < r["output_tokens"]:
                short.add((eng, c))
    out = [f"# Engine comparison: {root.name}", ""]
    pre = root / "preflight.json"
    if pre.exists():
        pf = json.loads(pre.read_text(encoding="utf-8-sig"))
        out.append(f"DynaLM sha256 {pf['dynalm']['sha256'][:16]}; workload {json.dumps(pf['workload'])}")
    out += ["", "## Checks", ""] + ([f"- FAIL: {p}" for p in problems] or ["- all runs completed without errors"])
    for eng, c in sorted(short):
        mean = st.mean(r["mean_completion_tokens"] for r in rows[(eng, c)])
        out.append(f"- NOT COMPARABLE: {eng} c{c} mean completion {mean:.0f} < {rows[(eng, c)][0]['output_tokens']} tokens "
                   "(ignore_eos not honored): tok/s and E2E are not comparable; TTFT and ITL are per token")
    for c in sorted({c for _, c in rows}):
        out += ["", f"## Concurrency {c}", "",
                "| metric | " + " | ".join(f"{e} (n={len(rows[(e, c)])})" for e in sorted({e for e, cc in rows if cc == c})) + " |",
                "|---|" + "---|" * len({e for e, cc in rows if cc == c})]
        engines = sorted({e for e, cc in rows if cc == c})
        base = rows.get((args.baseline, c))
        for name, f, higher in METRICS:
            cells = []
            for e in engines:
                v = [f(r) for r in rows[(e, c)]]
                cell = f"{st.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]"
                if base is not None and e != args.baseline:
                    b = [f(r) for r in base]
                    d = 100 * (st.median(v) - st.median(b)) / st.median(b) if st.median(b) else 0.0
                    sep = min(v) > max(b) or max(v) < min(b)
                    cell += f" ({d:+.1f}%, {'separated' if sep else 'overlapping'})"
                cells.append(cell)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Completion tokens (mean of means): " + ", ".join(
            f"{e} {st.mean(r['mean_completion_tokens'] for r in rows[(e, c)]):.0f}" for e in engines))
    out += ["", f"Percentages: engine median vs {args.baseline} median (+ = larger value; for latencies larger is worse). "
            "'separated': all repetitions of the two engines on opposite sides."]
    text = "\n".join(out) + "\n"
    (root / "e2e-analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

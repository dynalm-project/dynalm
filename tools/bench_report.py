#!/usr/bin/env python3
"""Render benchmark JSON lines (engine benchmark --out ...) as Markdown tables.

    PYTHONUTF8=1 python tools/bench_report.py results/*.jsonl > report.md

Rows from different targets (this engine in-process, this engine over HTTP,
llama.cpp llama-server, Ollama) with the same (prompt, output, concurrency)
are grouped together so they can be compared directly. Latencies are
P50/P99 (P90/P95 are in the JSON); never compare on means alone.
"""
import json
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r.setdefault("label", r["target"])
                    rows.append(r)
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = load(sys.argv[1:])
    hw = sorted({r["hardware"] for r in rows})
    models = sorted({r["model"] for r in rows})
    print(f"Hardware: {', '.join(hw)}  \nModel: {', '.join(models)}\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["prompt_tokens"], r["output_tokens"])].append(r)
    for (p, o), rs in sorted(groups.items()):
        print(f"### prompt {p} tokens, output {o} tokens\n")
        print("| target | conc | out tok/s | in tok/s | TTFT p50 ms | TTFT p99 | ITL p50 ms | ITL p99 | "
              "TPOT p50 | E2E p99 ms | RSS MB | errors |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda x: (x["concurrency"], x["label"])):
            rss = f"{r['rss_mb']:.0f}" if r.get("rss_mb") else "—"
            print(f"| {r['label']} | {r['concurrency']} | {r['output_tok_s']:.1f} | {r['input_tok_s']:.1f} | "
                  f"{r['ttft_ms']['p50']:.0f} | {r['ttft_ms']['p99']:.0f} | {r['itl_ms']['p50']:.1f} | "
                  f"{r['itl_ms']['p99']:.1f} | {r['tpot_ms']['p50']:.1f} | {r['e2e_ms']['p99']:.0f} | {rss} | "
                  f"{r['errors']} |")
        print()


if __name__ == "__main__":
    main()

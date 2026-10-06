#!/usr/bin/env python3
"""Summarize bench_batch_decode old/new A/B CSVs (DD-066 validation).

usage: ab_summary.py FILE.csv [FILE.csv ...]

Per (context, seqs): median / min / max over repetitions of aggregate tok/s,
step p50/p90/p99 (= ITL for every sequence), attention ms per step and CPU
utilization, for each variant, and the change of the medians (new vs old).
The change is marked '~' when the two variants' [min, max] ranges overlap,
i.e. when the repetitions do not separate them.
"""
import csv
import statistics
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline='', encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
    return rows


def main():
    rows = load(sys.argv[1:])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(int(r['context']), int(r['seqs']))][r['variant']].append(r)
    metrics = [('agg_tok_s', 'agg tok/s', True), ('step_p50_ms', 'ITL p50', False), ('step_p90_ms', 'ITL p90', False),
               ('step_p99_ms', 'ITL p99', False), ('attn_ms_step', 'attn ms', False), ('cpu_util', 'cpu util', None)]
    print('| ctx | seqs | metric | old median [min-max] | new median [min-max] | change | separated |')
    print('|---|---|---|---|---|---|---|')
    for (ctx, n) in sorted(by):
        v = by[(ctx, n)]
        if 'old' not in v or 'new' not in v:
            continue
        for key, label, higher_better in metrics:
            o = [float(r[key]) for r in v['old']]
            w = [float(r[key]) for r in v['new']]
            mo, mw = statistics.median(o), statistics.median(w)
            change = (mw / mo - 1) * 100 if mo else 0.0
            separated = max(o) < min(w) or max(w) < min(o)
            fmt = '{:.3f}' if key == 'cpu_util' else '{:.1f}'
            print(f"| {ctx} | {n} | {label} | {fmt.format(mo)} [{fmt.format(min(o))}-{fmt.format(max(o))}] | "
                  f"{fmt.format(mw)} [{fmt.format(min(w))}-{fmt.format(max(w))}] | {change:+.1f}% | "
                  f"{'yes' if separated else '~'} |")


if __name__ == '__main__':
    main()

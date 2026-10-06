#!/usr/bin/env python3
"""Summarize bench_batch_decode A/B CSVs.

usage: ab_summary.py [--base VARIANT] FILE.csv [FILE.csv ...]

Per (context, seqs) and variant: median / min / max over repetitions of
aggregate tok/s, step p50/p90/p99 (= ITL for every sequence), attention ms per
step, CPU utilization and thread-pool tail-wait share, and the change of the
medians against the base variant (default: "old" if present, else the first
variant seen). A change is marked '~' when the two [min, max] ranges overlap,
i.e. when the repetitions do not separate the variants.
"""
import csv
import statistics
import sys
from collections import defaultdict


def main():
    args = sys.argv[1:]
    base = None
    if args[:1] == ['--base']:
        base, args = args[1], args[2:]
    rows = []
    for p in args:
        with open(p, newline='', encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
    order = []
    for r in rows:
        if r['variant'] not in order:
            order.append(r['variant'])
    if base is None:
        base = 'old' if 'old' in order else order[0]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(int(r['context']), int(r['seqs']))][r['variant']].append(r)
    metrics = [('agg_tok_s', 'agg tok/s'), ('step_p50_ms', 'ITL p50'), ('step_p90_ms', 'ITL p90'),
               ('step_p99_ms', 'ITL p99'), ('attn_ms_step', 'attn ms'), ('cpu_util', 'cpu util'),
               ('tail_wait_share', 'tail wait')]
    print(f'| ctx | seqs | metric | variant | median [min-max] | change vs {base} | separated |')
    print('|---|---|---|---|---|---|---|')
    for (ctx, n) in sorted(by):
        v = by[(ctx, n)]
        if base not in v:
            continue
        for key, label in metrics:
            if key not in v[base][0]:
                continue
            b = [float(r[key]) for r in v[base]]
            fmt = '{:.3f}' if key in ('cpu_util', 'tail_wait_share') else '{:.1f}'
            for var in order:
                if var not in v:
                    continue
                w = [float(r[key]) for r in v[var]]
                mw = statistics.median(w)
                cell = f"{fmt.format(mw)} [{fmt.format(min(w))}-{fmt.format(max(w))}]"
                if var == base:
                    print(f'| {ctx} | {n} | {label} | {var} | {cell} | | |')
                    continue
                mb = statistics.median(b)
                change = (mw / mb - 1) * 100 if mb else 0.0
                separated = max(b) < min(w) or max(w) < min(b)
                print(f"| {ctx} | {n} | {label} | {var} | {cell} | {change:+.1f}% | {'yes' if separated else '~'} |")


if __name__ == '__main__':
    main()

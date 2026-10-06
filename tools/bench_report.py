#!/usr/bin/env python3
"""Render benchmark JSON lines (dynalm benchmark --out ...) as Markdown and SVG graphs.

    PYTHONUTF8=1 python tools/bench_report.py results/run.jsonl > report.md
    PYTHONUTF8=1 python tools/bench_report.py --svg results/graphs results/*.jsonl > report.md

Markdown: one table per (prompt, output) with throughput and P50/P99 latency
for every target, plus a diagnostics table (bottleneck class, decode step
cost, thread-pool and bandwidth figures) for in-process runs.

--svg DIR writes the eight standard graphs of the performance program
(DD-050) using only the standard library:
  1 aggregate output tok/s vs concurrency      5 output tok/s vs context length
  2 single-stream output tok/s                 6 est. memory bandwidth vs context length
  3 ITL p50/p99 vs concurrency                 7 CPU utilization vs concurrency
  4 TTFT p50 vs prompt length                  8 LLC miss ratio vs concurrency
Graphs whose data is absent (e.g. hardware counters on a VM) say so instead of
drawing an empty plot. Never compare on means alone.
"""
import json
import math
import os
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


def series_key(r):
    mix = r.get("prompt_mix")
    prompt = f"mix {'/'.join(map(str, mix))}" if mix else f"p{r['prompt_tokens']}"
    return f"{r['model']} · {r['label']} · {prompt} · o{r['output_tokens']}"


# --- markdown -----------------------------------------------------------------------


def markdown(rows):
    hw = sorted({r["hardware"] for r in rows})
    models = sorted({r["model"] for r in rows})
    out = [f"Hardware: {', '.join(hw)}  \nModel: {', '.join(models)}\n"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["prompt_tokens"], r["output_tokens"], tuple(r.get("prompt_mix", [])))].append(r)
    for (m, p, o, mix), rs in sorted(groups.items()):
        title = f"mixed prompts {'/'.join(map(str, mix))} (mean {p})" if mix else f"prompt {p} tokens"
        out.append(f"### {m}: {title}, output {o} tokens\n")
        out.append("| target | conc | out tok/s | in tok/s | TTFT p50 ms | TTFT p99 | ITL p50 ms | ITL p99 | "
                   "TPOT p50 | E2E p99 ms | CPU util | RSS MB | errors |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda x: (x["concurrency"], x["label"])):
            rss = f"{r['rss_mb']:.0f}" if r.get("rss_mb") else "—"
            cpu = f"{100 * r['cpu_util']:.0f}%" if r.get("cpu_util") else "—"
            out.append(f"| {r['label']} | {r['concurrency']} | {r['output_tok_s']:.1f} | {r['input_tok_s']:.1f} | "
                       f"{r['ttft_ms']['p50']:.0f} | {r['ttft_ms']['p99']:.0f} | {r['itl_ms']['p50']:.1f} | "
                       f"{r['itl_ms']['p99']:.1f} | {r['tpot_ms']['p50']:.1f} | {r['e2e_ms']['p99']:.0f} | {cpu} | "
                       f"{rss} | {r['errors']} |")
        out.append("")
        diag = [r for r in rs if r.get("diag")]
        if diag:
            out.append("| conc | bottleneck | decode step ms | rows/step | regions/step | tail wait | "
                       "sample+emit ms/step | est. GB/s | ceiling GB/s | MHz |")
            out.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(diag, key=lambda x: x["concurrency"]):
                d = r["diag"]
                steps = max(1, d["steps"])
                pool = d.get("pool", {})
                regions = f"{pool['regions'] / steps:.0f}" if pool.get("regions") is not None else "—"
                tail = (f"{100 * pool['tail_wait_ms'] / d['time_ms']['forward']:.0f}%"
                        if pool.get("tail_wait_ms") is not None and d["time_ms"]["forward"] > 0 else "—")
                host = (d["time_ms"]["sample"] + d["time_ms"]["emit"]) / steps
                bw = f"{d['est_decode_bw_gbs']:.1f}" if d.get("est_decode_bw_gbs") is not None else "—"
                peak = f"{d['peak_bw_gbs']:.1f}" if d.get("peak_bw_gbs") is not None else "—"
                mhz = f"{d['cpu_mhz']:.0f}" if d.get("cpu_mhz") is not None else "—"
                out.append(f"| {r['concurrency']} | {d['bottleneck']['primary']} | {d['decode_step_ms']:.1f} | "
                           f"{d['mean_decode_rows']:.1f} | {regions} | {tail} | {host:.2f} | {bw} | {peak} | {mhz} |")
            out.append("")
            for r in sorted(diag, key=lambda x: x["concurrency"]):
                for e in r["diag"]["bottleneck"]["evidence"]:
                    out.append(f"- c={r['concurrency']}: {e}")
            out.append("")
    return "\n".join(out)


# --- SVG ------------------------------------------------------------------------------

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_max(v):
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def line_chart(title, xlabel, ylabel, series, logx=False, note=None):
    """series: list of (name, [(x, y), ...]). Returns SVG text."""
    W, H, L, R, T, B = 900, 520, 80, 30, 50, 170
    pts = [p for _, s in series for p in s if p[1] is not None]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">{esc(title)}</text>']
    if not pts:
        svg.append(f'<text x="{W / 2}" y="{H / 2}" text-anchor="middle" fill="#666">'
                   f'{esc(note or "no data for this graph in the input")}</text></svg>')
        return "\n".join(svg)
    xs = sorted({p[0] for p in pts})
    xmin, xmax = min(xs), max(xs)
    ymax = nice_max(max(p[1] for p in pts) * 1.05)
    pw, ph = W - L - R, H - T - B

    def fx(x):
        if xmax == xmin:
            return L + pw / 2
        if logx:
            return L + pw * (math.log2(x) - math.log2(xmin)) / (math.log2(xmax) - math.log2(xmin))
        return L + pw * (x - xmin) / (xmax - xmin)

    def fy(y):
        return T + ph * (1 - y / ymax)

    for i in range(6):
        y = ymax * i / 5
        svg.append(f'<line x1="{L}" y1="{fy(y):.1f}" x2="{L + pw}" y2="{fy(y):.1f}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{L - 8}" y="{fy(y) + 4:.1f}" text-anchor="end">{y:g}</text>')
    for x in xs:
        svg.append(f'<line x1="{fx(x):.1f}" y1="{T + ph}" x2="{fx(x):.1f}" y2="{T + ph + 5}" stroke="#333"/>')
        svg.append(f'<text x="{fx(x):.1f}" y="{T + ph + 20}" text-anchor="middle">{x:g}</text>')
    svg.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="#333"/>')
    svg.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" stroke="#333"/>')
    svg.append(f'<text x="{L + pw / 2}" y="{T + ph + 42}" text-anchor="middle">{esc(xlabel)}</text>')
    svg.append(f'<text transform="translate(18 {T + ph / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    for i, (name, s) in enumerate(series):
        s = sorted(p for p in s if p[1] is not None)
        if not s:
            continue
        c = PALETTE[i % len(PALETTE)]
        dash = ' stroke-dasharray="6 4"' if "p99" in name else ""
        path = " ".join(f"{'M' if j == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for j, (x, y) in enumerate(s))
        svg.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        for x, y in s:
            svg.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="{c}"><title>{esc(name)}: '
                       f'{x:g} → {y:.2f}</title></circle>')
        ly = T + ph + 62 + 16 * i
        svg.append(f'<line x1="{L}" y1="{ly - 4}" x2="{L + 24}" y2="{ly - 4}" stroke="{c}" stroke-width="2"{dash}/>')
        svg.append(f'<text x="{L + 32}" y="{ly}">{esc(name)}</text>')
    if note:
        svg.append(f'<text x="{W - R}" y="{H - 8}" text-anchor="end" fill="#666">{esc(note)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def by_series(rows, xkey, ykey, filt=lambda r: True, name=series_key):
    groups = defaultdict(list)
    for r in rows:
        if not filt(r):
            continue
        y = ykey(r)
        if y is not None:
            groups[name(r)].append((xkey(r), y))
    return sorted(groups.items())


def diag_get(r, *path):
    v = r.get("diag")
    for p in path:
        if not isinstance(v, dict) or v.get(p) is None:
            return None
        v = v[p]
    return v


def write_graphs(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    conc = lambda r: r["concurrency"]  # noqa: E731
    ctx_name = lambda r: f"{r['model']} · {r['label']} · c={r['concurrency']} · o{r['output_tokens']}"  # noqa: E731
    fixed = lambda r: not r.get("prompt_mix")  # noqa: E731
    graphs = {
        "1_aggregate_tok_s_vs_concurrency.svg": line_chart(
            "Aggregate output tok/s vs concurrency", "concurrent requests", "output tokens / s",
            by_series(rows, conc, lambda r: r["output_tok_s"]), logx=True),
        "2_single_stream_tok_s.svg": line_chart(
            "Single-stream output tok/s (concurrency 1) vs prompt length", "prompt tokens", "output tokens / s",
            by_series(rows, lambda r: r["prompt_tokens"], lambda r: r["output_tok_s"],
                      lambda r: r["concurrency"] == 1 and fixed(r),
                      lambda r: f"{r['model']} · {r['label']} · o{r['output_tokens']}")),
        "3_itl_vs_concurrency.svg": line_chart(
            "Inter-token latency vs concurrency (solid p50, dashed p99)", "concurrent requests", "ms",
            by_series(rows, conc, lambda r: r["itl_ms"]["p50"], name=lambda r: series_key(r) + " p50") +
            by_series(rows, conc, lambda r: r["itl_ms"]["p99"], name=lambda r: series_key(r) + " p99"), logx=True),
        "4_ttft_vs_prompt_length.svg": line_chart(
            "TTFT p50 vs prompt length", "prompt tokens", "ms",
            by_series(rows, lambda r: r["prompt_tokens"], lambda r: r["ttft_ms"]["p50"], fixed, ctx_name)),
        "5_tok_s_vs_context_length.svg": line_chart(
            "Output tok/s vs context length", "prompt (context) tokens", "output tokens / s",
            by_series(rows, lambda r: r["prompt_tokens"], lambda r: r["output_tok_s"], fixed, ctx_name)),
        "6_bandwidth_vs_context_length.svg": line_chart(
            "Estimated decode memory bandwidth vs context length", "prompt (context) tokens", "GB/s (modelled)",
            by_series(rows, lambda r: r["prompt_tokens"], lambda r: diag_get(r, "est_decode_bw_gbs"), fixed,
                      ctx_name),
            note="bytes modelled per decode step (weights + KV) / measured decode-step time"),
        "7_cpu_util_vs_concurrency.svg": line_chart(
            "CPU utilization vs concurrency", "concurrent requests", "% of logical CPUs",
            by_series(rows, conc, lambda r: 100 * r["cpu_util"] if r.get("cpu_util") else None), logx=True),
        "8_llc_miss_ratio_vs_concurrency.svg": line_chart(
            "Last-level cache miss ratio vs concurrency", "concurrent requests", "% of LLC references",
            by_series(rows, conc, lambda r: (100 * diag_get(r, "counters", "llc_misses") /
                                             diag_get(r, "counters", "llc_references"))
                      if diag_get(r, "counters", "llc_references") else None), logx=True,
            note="hardware counters were unavailable for these runs (no PMU: Windows or a VM such as WSL2)"),
    }
    for name, svg in graphs.items():
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
            f.write(svg)
    return list(graphs)


def main():
    args = sys.argv[1:]
    svg_dir = None
    if len(args) >= 2 and args[0] == "--svg":
        svg_dir, args = args[1], args[2:]
    if not args:
        sys.exit(__doc__)
    rows = load(args)
    print(markdown(rows))
    if svg_dir:
        names = write_graphs(rows, svg_dir)
        print("\nGraphs:\n")
        for n in names:
            print(f"- [{n}]({os.path.join(svg_dir, n).replace(os.sep, '/')})")


if __name__ == "__main__":
    main()

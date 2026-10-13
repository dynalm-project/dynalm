#!/usr/bin/env python3
"""Analyze a paired validation (tools/pk8_validation.ps1): DYNACORE_PACKED_8ROWS
(default) or DYNACORE_INT16_PREFILL_ROWS (--feature int16prefill, or the
feature recorded in preflight.json).

  python tools/pk8_analyze.py results/pk8-validation-<ts> [--min-pairs 8] [--control-concurrency 1]

Reads every runNN-<base|exp>/ directory (results.jsonl from compare_engines.py
and the dynalm-metrics-*-c<N>-*.prom scrapes) plus the runNN-<cfg>.json run
records when present. The warm-up (run00-*) is ignored.

Checks (any failure makes the exit code 1):
  - every run: enabled gauge == configuration (0 base, 1 exp)
  - baseline: dispatch counter == 0; experimental: dispatch counter > 0
  - errors == 0, completed == requests, mean completion tokens == output length
  - server failed/rejected/timed-out counters == 0
  - every pair has one base and one exp run with the same concurrency points
  - run commands identical apart from --out (run records only)
  - pairs >= --min-pairs
  - --control-concurrency C (e.g. 1): points at concurrency C are a null
    control. Decode there never has more than 4 rows, so the 8-row kernel
    must NOT run in either configuration (dispatches == 0 for base and exp;
    the exp gauge must still read 1).

Pairing: the run records' "pair" field; without records, consecutive runs
(run1+run2, run3+run4, ...). Order inside a pair is kept for the order report.

Per concurrency and metric: baseline/experimental median, mean, min, max and
run-to-run spread; each pair's % difference (exp vs base); median paired
difference; how many pairs favour exp; exact two-sided sign test. A metric is
"improved"/"regressed" only when the sign test p <= 0.05 AND the median paired
difference exceeds the baseline's run-to-run coefficient of variation;
otherwise "inconclusive". Writes <dir>/pk8-analysis.md and pk8-pairs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

# (key, label, higher_is_better)
FEATURES = {
    "packed8rows": {"env": "DYNACORE_PACKED_8ROWS", "exp": "1", "gauge": "dynacore_packed_8rows_enabled",
                    "counter": "dynacore_packed_rows8_dispatches_total", "label": "8-row"},
    "int16prefill": {"env": "DYNACORE_INT16_PREFILL_ROWS", "exp": "64", "gauge": "dynacore_int16_prefill_max_rows",
                     "counter": "dynacore_int16_prefill_matmuls_total", "label": "int16 prefill"},
    "soloprefill": {"env": "DYNALM_SOLO_PREFILL_CHUNK", "exp": "128", "gauge": "dynalm_scheduler_solo_prefill_chunk",
                    "counter": "dynalm_scheduler_solo_prefill_steps_total", "label": "solo prefill"},
}
FEAT = dict(FEATURES["packed8rows"])  # set in main()

METRICS = [
    (("output_tok_s",), "output tok/s", True),
    (("ttft_ms", "p50"), "TTFT p50 ms", False),
    (("ttft_ms", "p95"), "TTFT p95 ms", False),
    (("ttft_ms", "p99"), "TTFT p99 ms", False),
    (("itl_ms", "p50"), "ITL p50 ms", False),
    (("itl_ms", "p95"), "ITL p95 ms", False),
    (("itl_ms", "p99"), "ITL p99 ms", False),
    (("e2e_ms", "p50"), "E2E p50 ms", False),
    (("e2e_ms", "p95"), "E2E p95 ms", False),
    (("e2e_ms", "p99"), "E2E p99 ms", False),
]


def step_stats(prom: dict) -> tuple:
    """(mean step ms, steps over 150 ms) from the dynalm_scheduler_step_ms histogram."""
    n = prom.get("dynalm_scheduler_step_ms_count", 0.0)
    mean = prom.get("dynalm_scheduler_step_ms_sum", 0.0) / n if n else float("nan")
    le150 = prom.get('dynalm_scheduler_step_ms_bucket{le="150"}')
    long_steps = n - le150 if le150 is not None else float("nan")
    return mean, long_steps
RUN_RE = re.compile(r"^run(\d+)-(base|exp)$")
PROM_RE = re.compile(r"dynalm-metrics-.*-c(\d+(?:_\d+)*)-\d{8}-\d{6}\.prom$")


def metric(row: dict, key: tuple) -> float:
    v = row
    for k in key:
        v = v[k]
    return float(v)


def parse_prom(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_runs(root: Path, problems: list) -> list:
    runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        m = RUN_RE.match(d.name)
        if not d.is_dir() or not m:
            continue
        num, cfg = int(m.group(1)), m.group(2)
        if num == 0:
            continue
        run = {"name": d.name, "num": num, "cfg": cfg, "dir": d, "points": {}, "record": None}
        rec = root / f"{d.name}.json"
        if rec.exists():
            run["record"] = json.loads(rec.read_text(encoding="utf-8-sig"))
        res = d / "results.jsonl"
        if not res.exists():
            problems.append(f"{d.name}: no results.jsonl")
            runs.append(run)
            continue
        for line in res.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                run["points"].setdefault(int(r["concurrency"]), {})["row"] = r
        for p in d.glob("dynalm-metrics-*.prom"):
            pm = PROM_RE.search(p.name)
            if pm:
                g = pm.group(1)
                c = int(g) if g.isdigit() else g
                run["points"].setdefault(c, {})["prom"] = parse_prom(p)
                run["points"][c]["prom_file"] = p.name
        runs.append(run)
    return runs


def check_run(run: dict, problems: list, control: frozenset = frozenset()) -> None:
    want_env = FEAT["exp"] if run["cfg"] == "exp" else "0"
    want = float(want_env)
    rec = run["record"]
    if rec is not None:
        env = rec.get("env", {}).get(FEAT["env"])
        if env != want_env:
            problems.append(f"{run['name']}: run record {FEAT['env']}={env!r}, expected {want_env!r}")
        if rec.get("dry_run"):
            problems.append(f"{run['name']}: dry-run record, no measurement")
        if rec.get("exit_code", 0) != 0:
            problems.append(f"{run['name']}: engine exit code {rec.get('exit_code')}")
    if not run["points"]:
        problems.append(f"{run['name']}: no concurrency points")
    for c, pt in sorted(run["points"].items(), key=lambda kv: str(kv[0])):
        tag = f"{run['name']} c{c}"
        row, prom = pt.get("row"), pt.get("prom")
        if row is None:
            problems.append(f"{tag}: no results row")
        else:
            if row["errors"] != 0:
                problems.append(f"{tag}: {row['errors']} errors")
            if row["completed"] != row["requests"]:
                problems.append(f"{tag}: completed {row['completed']} of {row['requests']}")
            if round(row["mean_completion_tokens"]) != row["output_tokens"]:
                problems.append(f"{tag}: mean completion {row['mean_completion_tokens']} != {row['output_tokens']}")
        if prom is None:
            problems.append(f"{tag}: no /metrics scrape")
            continue
        en = prom.get(FEAT["gauge"])
        disp = prom.get(FEAT["counter"])
        if en is None or disp is None:
            problems.append(f"{tag}: scrape lacks {FEAT['gauge']} / {FEAT['counter']} (old binary?)")
            continue
        if en != want:
            problems.append(f"{tag}: gauge {FEAT['gauge']} = {en:g}, expected {want:g}")
        if want == 0 and disp != 0:
            problems.append(f"{tag}: baseline dispatched the {FEAT['label']} path {disp:g} times")
        if want != 0 and c in control and disp != 0:
            problems.append(f"{tag}: control point dispatched the {FEAT['label']} path {disp:g} times")
        if want != 0 and c not in control and disp <= 0:
            problems.append(f"{tag}: experimental run never dispatched the {FEAT['label']} path")
        for k in ("dynalm_requests_failed_total", "dynalm_requests_rejected_total", "dynalm_requests_timed_out_total"):
            if prom.get(k, 0) != 0:
                problems.append(f"{tag}: {k} = {prom[k]:g}")


def normalize_command(cmd: str) -> str:
    return re.sub(r"--out \S+", "--out <run>", cmd)


def make_pairs(runs: list, problems: list) -> list:
    if runs and all(r["record"] is not None and "pair" in r["record"] for r in runs):
        groups: dict = {}
        for r in runs:
            groups.setdefault(int(r["record"]["pair"]), []).append(r)
        ordered = [groups[k] for k in sorted(groups)]
    else:
        ordered = [runs[i:i + 2] for i in range(0, len(runs), 2)]
    pairs = []
    for i, g in enumerate(ordered, 1):
        cfgs = sorted(r["cfg"] for r in g)
        if cfgs != ["base", "exp"]:
            problems.append(f"pair {i}: runs {[r['name'] for r in g]} are not one base + one exp")
            continue
        g = sorted(g, key=lambda r: r["num"])
        base = next(r for r in g if r["cfg"] == "base")
        exp = next(r for r in g if r["cfg"] == "exp")
        if set(base["points"]) != set(exp["points"]):
            problems.append(f"pair {i}: concurrency points differ")
        pairs.append({"pair": i, "order": "-".join(r["cfg"] for r in g), "base": base, "exp": exp})
    cmds = {normalize_command(r["record"]["command"]) for r in runs if r["record"] and "command" in r["record"]}
    if len(cmds) > 1:
        problems.append(f"run commands differ beyond --out: {sorted(cmds)}")
    return pairs


def summarize(values: list) -> dict:
    s = {"median": st.median(values), "mean": st.mean(values), "min": min(values), "max": max(values)}
    s["spread_pct"] = 100 * (s["max"] - s["min"]) / s["median"] if s["median"] else 0.0
    s["cv_pct"] = 100 * st.stdev(values) / s["mean"] if len(values) > 1 and s["mean"] else 0.0
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--control-concurrency", default="",
                    help="comma list of concurrencies that are null controls (no dispatches expected)")
    ap.add_argument("--feature", choices=sorted(FEATURES), default=None,
                    help="feature under test (default: preflight.json's, else packed8rows)")
    args = ap.parse_args()
    root = Path(args.dir).resolve()
    feature = args.feature
    pre0 = root / "preflight.json"
    pf0 = json.loads(pre0.read_text(encoding="utf-8-sig")) if pre0.exists() else {}
    if feature is None:
        feature = (pf0.get("feature") or {}).get("name", "packed8rows")
    FEAT.update(FEATURES[feature])
    if (pf0.get("feature") or {}).get("exp_value"):
        FEAT["exp"] = str(pf0["feature"]["exp_value"])
    problems: list = []
    runs = load_runs(root, problems)
    if not runs:
        print(f"no runNN-<base|exp> directories in {root}", file=sys.stderr)
        return 1
    for r in runs:
        check_run(r, problems, frozenset(int(c) for c in args.control_concurrency.split(",") if c.strip()))
    pairs = make_pairs(runs, problems)
    if len(pairs) < args.min_pairs:
        problems.append(f"{len(pairs)} complete pairs, fewer than --min-pairs {args.min_pairs}")

    md = [f"# Paired validation ({FEAT['label']}, {FEAT['env']}=0 vs {FEAT['exp']}): {root.name}", ""]
    md.append(f"Runs: {len(runs)} ({sum(r['cfg'] == 'base' for r in runs)} base, "
              f"{sum(r['cfg'] == 'exp' for r in runs)} exp); complete pairs: {len(pairs)}; "
              f"orders: {', '.join(p['order'] for p in pairs)}")
    pre = root / "preflight.json"
    if pre.exists():
        pf = json.loads(pre.read_text(encoding="utf-8-sig"))
        md.append(f"Binary sha256 {pf['dynalm']['sha256'][:16]}..., git {pf['git']['head'][:8]}, "
                  f"workload {json.dumps(pf['workload'])}")
    md += ["", "## Configuration checks", ""]
    md += [f"- FAIL: {p}" for p in problems] or ["- all checks passed"]

    md += ["", "## Dispatch counter and steps per run", "",
           f"| Run | Pair | c | gauge | {FEAT['label']} dispatches | sched steps | mean step ms | steps > 150 ms | gen tokens | requests ok |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    pair_of = {}
    for p in pairs:
        pair_of[p["base"]["name"]] = pair_of[p["exp"]["name"]] = p["pair"]
    for r in runs:
        for c, pt in sorted(r["points"].items(), key=lambda kv: str(kv[0])):
            pr, row = pt.get("prom", {}), pt.get("row", {})
            sm, sl = step_stats(pr)
            md.append(f"| {r['name']} | {pair_of.get(r['name'], '-')} | {c} | "
                      f"{pr.get(FEAT['gauge'], float('nan')):g} | "
                      f"{pr.get(FEAT['counter'], float('nan')):,.0f} | "
                      f"{pr.get('dynalm_scheduler_step_ms_count', float('nan')):g} | {sm:.1f} | {sl:g} | "
                      f"{pr.get('dynalm_generation_tokens_total', float('nan')):g} | "
                      f"{row.get('completed', '-')}/{row.get('requests', '-')} |")

    csv_rows = []
    concs = sorted({c for p in pairs for c in p["base"]["points"] if c in p["exp"]["points"]}, key=str)
    for c in concs:
        md += ["", f"## Concurrency {c}", ""]
        md += ["| Metric | base median | exp median | base min-max | exp min-max | base CV | exp CV | "
               "median exp gain | exp gain range | pairs favouring exp | sign p | verdict |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        per_pair_lines = ["", f"Per-pair % difference (exp vs base) at c{c}; + is better for exp:", "",
                          "| Pair | order | " + " | ".join(m[1] for m in METRICS) + " |",
                          "|---|---|" + "---|" * len(METRICS)]
        per_pair = {p["pair"]: [] for p in pairs}
        for key, label, higher in METRICS:
            b, e, diffs = [], [], []
            for p in pairs:
                rb = p["base"]["points"].get(c, {}).get("row")
                re_ = p["exp"]["points"].get(c, {}).get("row")
                if rb is None or re_ is None:
                    continue
                vb, ve = metric(rb, key), metric(re_, key)
                d = 100 * (ve - vb) / vb if vb else 0.0
                better = d if higher else -d  # + = exp better
                b.append(vb), e.append(ve), diffs.append(better)
                per_pair[p["pair"]].append(better)
                csv_rows.append({"concurrency": c, "pair": p["pair"], "order": p["order"], "metric": label,
                                 "base": vb, "exp": ve, "diff_pct": round(d, 3), "exp_better_pct": round(better, 3),
                                 "base_run": p["base"]["name"], "exp_run": p["exp"]["name"]})
            if not diffs:
                continue
            sb, se = summarize(b), summarize(e)
            wins = sum(d > 0 for d in diffs)
            losses = sum(d < 0 for d in diffs)
            pval = sign_test(wins, losses)
            med = st.median(diffs)
            if pval <= 0.05 and abs(med) > sb["cv_pct"]:
                verdict = "improved" if med > 0 else "regressed"
            else:
                verdict = "inconclusive"
            md.append(f"| {label} | {sb['median']:.2f} | {se['median']:.2f} | {sb['min']:.2f}-{sb['max']:.2f} | "
                      f"{se['min']:.2f}-{se['max']:.2f} | {sb['cv_pct']:.1f}% | {se['cv_pct']:.1f}% | "
                      f"{med:+.1f}% | {min(diffs):+.1f}%..{max(diffs):+.1f}% | {wins}/{len(diffs)} | "
                      f"{pval:.3f} | {verdict} |")
        for p in pairs:
            vals = per_pair[p["pair"]]
            if vals:
                per_pair_lines.append(f"| {p['pair']} | {p['order']} | " + " | ".join(f"{v:+.1f}%" for v in vals) + " |")
        md += per_pair_lines
        errs = [(p["pair"], p["base"]["points"].get(c, {}).get("row", {}).get("errors"),
                 p["exp"]["points"].get(c, {}).get("row", {}).get("errors")) for p in pairs]
        md += ["", "Errors (base/exp) per pair: " + ", ".join(f"{i}: {a}/{b}" for i, a, b in errs)]

    md += ["", "Verdict rule: improved/regressed only if the exact two-sided sign test p <= 0.05 and the "
           "median paired difference exceeds the baseline coefficient of variation; else inconclusive. "
           "With fewer than 6 pairs no result can reach p <= 0.05."]
    text = "\n".join(md) + "\n"
    (root / "pk8-analysis.md").write_text(text, encoding="utf-8")
    if csv_rows:
        with open(root / "pk8-pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print(text)
    print(f"written: {root / 'pk8-analysis.md'}, {root / 'pk8-pairs.csv'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

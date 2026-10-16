#!/usr/bin/env python3
"""Paired A/B analysis of `dynalm benchmark` runs.

  python tools/bench_analyze.py <plan.json> <raw dir> [--out DIR]

Inputs
- <raw dir>/runs.jsonl: one record per run (id, exit_code, quiet, exe_sha256,
  env), one per benchmark process;
- <raw dir>/<run id>.jsonl: the benchmark's JSON lines, one per point
  (concurrency x prompt x output);
- <raw dir>/<run id>.requests.jsonl: per-request log (optional but checked
  when present).

Run ids follow `<workload>-s<session>r<round>-<arm>`; a *pair* is one
(session, round) of two arms on the same workload point.

The plan (JSON) fixes everything before the data are seen:
  {
    "reference": "rc",                         # stability is judged on this arm
    "expected_exe": {"rc": "AD011D64", ...},   # sha256 prefixes; a mismatch invalidates the run
    "reliability": {"primary": {"output_tok_s": 2.0}, "tails": {"itl_p99": 5.0}, "min_pairs": 3},
    "gates": [
      {"id": "W1", "baseline": "rc", "candidate": "wide", "metric": "output_tok_s",
       "rule": "improve", "threshold": 3.0, "where": {"concurrency": [12, 16]}},
      {"id": "E1", "baseline": "orig", "candidate": "rc", "metric": "tpot_p50",
       "rule": "equivalent", "threshold": 2.0, "pair_threshold": 3.0}
    ]
  }
Rules (percent change of candidate vs baseline, per pair):
- improve:     PASS if every pair is at or beyond +threshold in the metric's good
               direction; FAIL if the median is not; else INCONCLUSIVE;
- no_regress:  PASS if the median is not worse than threshold and no pair is worse
               than pair_threshold; FAIL if the median is worse; else INCONCLUSIVE;
- equivalent:  PASS if |median| <= threshold and every |pair| <= pair_threshold;
               FAIL if |median| > threshold; else INCONCLUSIVE.
A gate evaluation is INCONCLUSIVE whenever the reference arm is unreliable at
that point: fewer than min_pairs valid pairs, or a spread (max - min) /
median above its limit across pairs or across session medians.

Units (the report counts each separately):
- run: one benchmark process (one runs.jsonl record);
- point: one (run, concurrency, prompt, output) result line;
- pair: one (session, round) with both arms valid at a point;
- gate evaluation: one gate at one point (a gate covers every matching point).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

RUN_ID = re.compile(r"^(?P<workload>.+)-s(?P<session>\d+)r(?P<round>\d+)-(?P<arm>[A-Za-z0-9_]+)$")
# metric -> (getter, higher is better)
METRICS = {
    "output_tok_s": (lambda j: j.get("output_tok_s"), True),
    "ttft_p50": (lambda j: (j.get("ttft_ms") or {}).get("p50"), False),
    "ttft_p95": (lambda j: (j.get("ttft_ms") or {}).get("p95"), False),
    "ttft_p99": (lambda j: (j.get("ttft_ms") or {}).get("p99"), False),
    "tpot_p50": (lambda j: (j.get("tpot_ms") or {}).get("p50"), False),
    "itl_p50": (lambda j: (j.get("itl_ms") or {}).get("p50"), False),
    "itl_p95": (lambda j: (j.get("itl_ms") or {}).get("p95"), False),
    "itl_p99": (lambda j: (j.get("itl_ms") or {}).get("p99"), False),
    "e2e_p50": (lambda j: (j.get("e2e_ms") or {}).get("p50"), False),
    "e2e_p99": (lambda j: (j.get("e2e_ms") or {}).get("p99"), False),
    "decode_step_ms": (lambda j: (j.get("diag") or {}).get("decode_step_ms"), False),
}


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def spread(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = st.median(v)
    return 100.0 * (max(v) - min(v)) / m if m else float("inf")


def bootstrap_ci(v: list[float], n: int = 4000, seed: int = 7) -> tuple[float, float] | None:
    """95% percentile-bootstrap interval of the median (None below 4 values)."""
    if len(v) < 4:
        return None
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(v, k=len(v))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def load(plan: dict, raw: Path):
    runs = {r["id"]: r for r in jl(raw / "runs.jsonl")}
    expected = plan.get("expected_exe", {})
    points: dict = {}  # (workload, conc, prompt, output) -> {(session, round, arm): point}
    run_status = []
    for rid, rec in sorted(runs.items()):
        m = RUN_ID.match(rid)
        if not m:
            continue
        arm = m["arm"]
        reasons = []
        if rec.get("exit_code") != 0:
            reasons.append(f"exit code {rec.get('exit_code')}")
        if rec.get("quiet") is not True:
            reasons.append("quiet check not passed")
        want = expected.get(arm)
        if want and not (rec.get("exe_sha256") or "").upper().startswith(want.upper()):
            reasons.append(f"executable {str(rec.get('exe_sha256'))[:16]} is not the planned {want}")
        lines = jl(raw / f"{rid}.jsonl")
        if not lines:
            reasons.append("no benchmark output")
        reqs = jl(raw / f"{rid}.requests.jsonl")
        run_status.append({"run": rid, "arm": arm, "valid": not reasons, "reasons": reasons, "points": len(lines)})
        for j in lines:
            key = (m["workload"], j.get("concurrency"), j.get("prompt_tokens"), j.get("output_tokens"))
            pr = []
            if j.get("errors", 0) != 0:
                pr.append(f"{j.get('errors')} errors")
            if j.get("completed") != j.get("requests"):
                pr.append(f"completed {j.get('completed')} of {j.get('requests')}")
            if j.get("mean_completion_tokens") is None or abs(j["mean_completion_tokens"] - (j.get("output_tokens") or 0)) > 1e-9:
                pr.append(f"mean completion {j.get('mean_completion_tokens')} != target {j.get('output_tokens')}")
            preq = [q for q in reqs if q.get("concurrency") == j.get("concurrency") and q.get("point_prompt_tokens") == j.get("prompt_tokens")]
            if reqs and len(preq) != j.get("requests"):
                pr.append(f"request log has {len(preq)} of {j.get('requests')} requests")
            if any(not q.get("ok") or q.get("completion_tokens") != j.get("output_tokens") for q in preq):
                pr.append("a request failed or ended early")
            points.setdefault(key, {})[(int(m["session"]), int(m["round"]), arm)] = {
                "valid": not reasons and not pr, "reasons": reasons + pr, "json": j, "run": rid}
    return run_status, points


def analyze(plan: dict, raw: Path) -> dict:
    rel = plan.get("reliability", {})
    primary, tails, min_pairs = rel.get("primary", {"output_tok_s": 2.0}), rel.get("tails", {}), rel.get("min_pairs", 3)
    ref = plan["reference"]
    run_status, points = load(plan, raw)
    evaluations = []
    reliability = {}
    for key, cells in sorted(points.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        # reference-arm reliability at this point
        refv = {(s, r): c for (s, r, a), c in cells.items() if a == ref and c["valid"]}
        lim = {**primary, **tails}
        stats = {}
        over = []
        for metric, limit in lim.items():
            vals = [METRICS[metric][0](c["json"]) for c in refv.values()]
            if any(v is None for v in vals):
                over.append(f"{metric} missing")
                continue
            by_session: dict = {}
            for (s, _), c in refv.items():
                by_session.setdefault(s, []).append(METRICS[metric][0](c["json"]))
            sess_meds = [st.median(v) for v in by_session.values()]
            sp_pairs, sp_sess = spread(vals), spread(sess_meds)
            stats[metric] = {"spread_pct": round(sp_pairs, 2), "session_spread_pct": round(sp_sess, 2), "sessions": len(by_session)}
            if sp_pairs > limit or sp_sess > limit:
                over.append(metric)
        reliable = len(refv) >= min_pairs and not over
        reliability[str(key)] = {"reference_pairs": len(refv), "reliable": reliable, "over": over, "stats": stats}
        workload, conc, prompt, output = key
        for g in plan.get("gates", []):
            where = g.get("where", {})
            if where.get("workload") and workload not in where["workload"]:
                continue
            if where.get("concurrency") and conc not in where["concurrency"]:
                continue
            if where.get("prompt") and prompt not in where["prompt"]:
                continue
            b, c = g["baseline"], g["candidate"]
            getter, higher = METRICS[g["metric"]]
            pairs = sorted({(s, r) for (s, r, a) in cells if a == b} & {(s, r) for (s, r, a) in cells if a == c})
            changes, missing = [], 0
            for (s, r) in pairs:
                cb, cc = cells[(s, r, b)], cells[(s, r, c)]
                if not (cb["valid"] and cc["valid"]):
                    continue
                vb, vc = getter(cb["json"]), getter(cc["json"])
                if vb is None or vc is None or not vb:
                    missing += 1
                    continue
                changes.append(100.0 * (vc - vb) / vb)
            good = (lambda x: x) if higher else (lambda x: -x)  # positive = better
            thr, pthr = g.get("threshold", 0.0), g.get("pair_threshold", g.get("threshold", 0.0))
            med = st.median(changes) if changes else None
            if not reliable:
                verdict = "INCONCLUSIVE"
                why = "reference arm unreliable: " + (", ".join(over) if over else f"{len(refv)} valid pairs < {min_pairs}")
            elif len(changes) < min_pairs:
                verdict, why = "INCONCLUSIVE", f"{len(changes)} valid pairs < {min_pairs}" + (f" ({missing} missing metric)" if missing else "")
            elif g["rule"] == "improve":
                if all(good(x) >= thr for x in changes):
                    verdict, why = "PASS", f"every pair >= {thr:+.1f}% better"
                elif good(med) < thr:
                    verdict, why = "FAIL", f"median {good(med):+.2f}% better < {thr:+.1f}%"
                else:
                    verdict, why = "INCONCLUSIVE", "median meets the threshold, a pair does not"
            elif g["rule"] == "no_regress":
                if good(med) >= -thr and all(good(x) >= -pthr for x in changes):
                    verdict, why = "PASS", f"median and every pair within -{thr}/-{pthr}%"
                elif good(med) < -thr:
                    verdict, why = "FAIL", f"median {good(med):+.2f}% worse than -{thr}%"
                else:
                    verdict, why = "INCONCLUSIVE", "median within, a pair beyond"
            elif g["rule"] == "equivalent":
                if abs(med) <= thr and all(abs(x) <= pthr for x in changes):
                    verdict, why = "PASS", f"|median| <= {thr}% and every |pair| <= {pthr}%"
                elif abs(med) > thr:
                    verdict, why = "FAIL", f"|median| {abs(med):.2f}% > {thr}%"
                else:
                    verdict, why = "INCONCLUSIVE", "median within, a pair beyond"
            else:
                raise ValueError(f"unknown rule {g['rule']}")
            ci = bootstrap_ci(changes)
            evaluations.append({"gate": g["id"], "workload": workload, "concurrency": conc, "prompt": prompt, "output": output,
                                "baseline": b, "candidate": c, "metric": g["metric"], "rule": g["rule"], "pairs": len(changes),
                                "changes_pct": [round(x, 2) for x in changes], "median_pct": None if med is None else round(med, 2),
                                "ci95_median_pct": None if ci is None else [round(ci[0], 2), round(ci[1], 2)],
                                "verdict": verdict, "reason": why})
    counts = {
        "runs": len(run_status), "runs_valid": sum(r["valid"] for r in run_status),
        "points": sum(len(c) for c in points.values()),
        "points_valid": sum(1 for c in points.values() for x in c.values() if x["valid"]),
        "workload_points": len(points),
        "gate_evaluations": len(evaluations),
        "verdicts": {v: sum(1 for e in evaluations if e["verdict"] == v) for v in ("PASS", "FAIL", "INCONCLUSIVE")},
    }
    invalid = [{"run": r["run"], "reasons": r["reasons"]} for r in run_status if not r["valid"]]
    invalid += [{"run": x["run"], "point": str(k), "reasons": x["reasons"]} for k, c in points.items() for x in c.values()
                if not x["valid"] and x["reasons"] and not any(r["run"] == x["run"] and not r["valid"] for r in run_status)]
    return {"counts": counts, "invalid": invalid, "reliability": reliability, "evaluations": evaluations}


def render(res: dict) -> str:
    c = res["counts"]
    v = c["verdicts"]
    out = ["# Paired benchmark analysis", "",
           f"- **Runs** (benchmark processes): {c['runs']} ({c['runs_valid']} valid).",
           f"- **Points** (one concurrency x prompt x output result of one run): {c['points']} ({c['points_valid']} valid), "
           f"over {c['workload_points']} distinct workload points.",
           f"- **Gate evaluations** (one gate at one workload point): {c['gate_evaluations']}: "
           f"{v['PASS']} PASS, {v['FAIL']} FAIL, {v['INCONCLUSIVE']} INCONCLUSIVE.", ""]
    if res["invalid"]:
        out += ["## Invalid runs / points", ""] + [f"- {i['run']}{(' ' + i['point']) if 'point' in i else ''}: {'; '.join(i['reasons'])}"
                                                    for i in res["invalid"]] + [""]
    out += ["## Gate evaluations", "", "| gate | workload | c | prompt | out | candidate vs baseline | metric | pairs | per-pair change % | median % | 95% CI of median | verdict | reason |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in res["evaluations"]:
        ci = "-" if e["ci95_median_pct"] is None else f"[{e['ci95_median_pct'][0]:+.2f}, {e['ci95_median_pct'][1]:+.2f}]"
        med = "-" if e["median_pct"] is None else f"{e['median_pct']:+.2f}"
        out.append(f"| {e['gate']} | {e['workload']} | {e['concurrency']} | {e['prompt']} | {e['output']} | {e['candidate']} vs {e['baseline']} | "
                   f"{e['metric']} | {e['pairs']} | {' '.join(f'{x:+.2f}' for x in e['changes_pct'])} | {med} | {ci} | {e['verdict']} | {e['reason']} |")
    out += ["", "## Reference-arm reliability per point", "", "| point | pairs | reliable | spreads (pairs / session medians) |", "|---|---|---|---|"]
    for k, r in res["reliability"].items():
        sp = ", ".join(f"{m} {s['spread_pct']:.2f}/{s['session_spread_pct']:.2f}% ({s['sessions']} sessions)" for m, s in r["stats"].items())
        out.append(f"| {k} | {r['reference_pairs']} | {'yes' if r['reliable'] else 'no: ' + ', '.join(r['over'])} | {sp} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("raw")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    res = analyze(plan, Path(a.raw))
    out = Path(a.out) if a.out else Path(a.raw).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    text = render(res)
    (out / "analysis.md").write_text(text, encoding="utf-8")
    c = res["counts"]
    print(f"runs {c['runs']} ({c['runs_valid']} valid) | points {c['points']} ({c['points_valid']} valid) | gate evaluations "
          f"{c['gate_evaluations']}: {c['verdicts']} -> {out / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

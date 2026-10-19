#!/usr/bin/env python3
"""Offline checks of tools/bench_analyze.py (synthetic runs; no models).

  python tools/test_bench_analyze.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_analyze", HERE / "bench_analyze.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAILED: {what}")
        sys.exit(1)


def point(tok_s, tpot=50.0, conc=16, out=128, mean=None, errors=0, completed=None, itl99=400.0):
    return {"concurrency": conc, "prompt_tokens": 128, "output_tokens": out, "requests": 8,
            "completed": 8 if completed is None else completed, "errors": errors,
            "mean_completion_tokens": out if mean is None else mean, "output_tok_s": tok_s,
            "ttft_ms": {"p50": 1000.0}, "tpot_ms": {"p50": tpot}, "itl_ms": {"p50": 60.0, "p99": itl99}, "e2e_ms": {"p99": 9000.0}}


def write(raw: Path, rid: str, points, exe="AAAA1111", exit_code=0, quiet=True, requests=True):
    with open(raw / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": rid, "exit_code": exit_code, "quiet": quiet, "exe_sha256": exe}) + "\n")
    (raw / f"{rid}.jsonl").write_text("\n".join(json.dumps(p) for p in points), encoding="utf-8")
    if requests:
        lines = []
        for p in points:
            for i in range(p["requests"]):
                done = p["mean_completion_tokens"] == p["output_tokens"]
                lines.append(json.dumps({"concurrency": p["concurrency"], "point_prompt_tokens": p["prompt_tokens"], "index": i,
                                         "ok": p["errors"] == 0, "completion_tokens": p["output_tokens"] if done else 1}))
        (raw / f"{rid}.requests.jsonl").write_text("\n".join(lines), encoding="utf-8")


PLAN = {"reference": "rc", "expected_exe": {"rc": "AAAA", "wide": "AAAA"},
        "reliability": {"primary": {"output_tok_s": 2.0}, "tails": {"itl_p99": 5.0}, "min_pairs": 3},
        "gates": [{"id": "W1", "baseline": "rc", "candidate": "wide", "metric": "output_tok_s", "rule": "improve", "threshold": 3.0},
                  {"id": "W2", "baseline": "rc", "candidate": "wide", "metric": "tpot_p50", "rule": "no_regress", "threshold": 3.0},
                  {"id": "E1", "baseline": "rc", "candidate": "wide", "metric": "itl_p99", "rule": "equivalent", "threshold": 2.0,
                   "pair_threshold": 3.0}]}

# 1. Clean improvement over 3 sessions x 1 round: counts and verdicts.
with tempfile.TemporaryDirectory() as t:
    raw = Path(t)
    for s, (rc, wide) in enumerate([(40.0, 46.0), (40.4, 46.5), (40.2, 46.2)], start=1):
        write(raw, f"C16-s{s}r1-rc", [point(rc)])
        write(raw, f"C16-s{s}r1-wide", [point(wide, tpot=44.0)])
    res = ba.analyze(PLAN, raw)
    c = res["counts"]
    check(c["runs"] == 6 and c["runs_valid"] == 6, "6 runs, all valid")
    check(c["points"] == 6 and c["workload_points"] == 1, "6 points over 1 workload point")
    check(c["gate_evaluations"] == 3, "3 gates x 1 point = 3 evaluations")
    ev = {e["gate"]: e for e in res["evaluations"]}
    check(ev["W1"]["verdict"] == "PASS" and ev["W1"]["pairs"] == 3, "improvement PASS with 3 pairs")
    check(ev["W2"]["verdict"] == "PASS", "TPOT better: no_regress PASS")
    check(ev["E1"]["verdict"] == "PASS", "equal ITL p99: equivalent PASS")
    text = ba.render(res)
    check("**Runs** (benchmark processes): 6" in text and "**Gate evaluations** (one gate at one workload point): 3" in text,
          "report names each unit")

# 2. Invalid results are excluded and named; too few pairs -> INCONCLUSIVE.
with tempfile.TemporaryDirectory() as t:
    raw = Path(t)
    write(raw, "C16-s1r1-rc", [point(40.0)])
    write(raw, "C16-s1r1-wide", [point(46.0, mean=97.5)])            # ended early
    write(raw, "C16-s2r1-rc", [point(40.1)])
    write(raw, "C16-s2r1-wide", [point(46.0)], exe="BBBB2222")         # wrong executable
    write(raw, "C16-s3r1-rc", [point(40.2)], exit_code=3)             # failed run
    write(raw, "C16-s3r1-wide", [point(46.0, errors=1)])
    res = ba.analyze(PLAN, raw)
    reasons = " ".join("; ".join(i["reasons"]) for i in res["invalid"])
    check("mean completion 97.5 != target 128" in reasons, "mismatched output length detected")
    check("not the planned AAAA" in reasons, "executable mismatch detected")
    check("exit code 3" in reasons and "1 errors" in reasons, "failed run and errors detected")
    check(res["counts"]["runs_valid"] == 4, "failed and mismatched-executable runs are invalid")
    check(all(e["verdict"] == "INCONCLUSIVE" for e in res["evaluations"]), "too few valid pairs -> INCONCLUSIVE")

# 3. Unstable reference across sessions -> INCONCLUSIVE even with a large effect.
with tempfile.TemporaryDirectory() as t:
    raw = Path(t)
    for s, (rc, wide) in enumerate([(40.0, 46.0), (42.0, 48.0), (40.1, 46.1)], start=1):
        write(raw, f"C16-s{s}r1-rc", [point(rc)])
        write(raw, f"C16-s{s}r1-wide", [point(wide)])
    res = ba.analyze(PLAN, raw)
    check(all(e["verdict"] == "INCONCLUSIVE" and "unreliable" in e["reason"] for e in res["evaluations"]), "5% reference spread -> INCONCLUSIVE")

# 4. Missing metric -> reference unreliable at that point.
with tempfile.TemporaryDirectory() as t:
    raw = Path(t)
    for s in (1, 2, 3):
        p = point(40.0)
        del p["itl_ms"]
        write(raw, f"C16-s{s}r1-rc", [p])
        write(raw, f"C16-s{s}r1-wide", [point(46.0)])
    res = ba.analyze(PLAN, raw)
    check(all(e["verdict"] == "INCONCLUSIVE" and "itl_p99 missing" in e["reason"] for e in res["evaluations"]), "missing metric -> INCONCLUSIVE")

# 5. FAIL and mixed verdicts; multi-point runs; where-filter.
with tempfile.TemporaryDirectory() as t:
    raw = Path(t)
    for s in (1, 2, 3):
        write(raw, f"HC-s{s}r1-rc", [point(40.0, conc=8), point(40.0, conc=16)])
        write(raw, f"HC-s{s}r1-wide", [point(40.4, conc=8), point(46.0 if s != 2 else 41.0, conc=16)])
    plan = dict(PLAN, gates=[dict(PLAN["gates"][0], where={"concurrency": [8, 16]})])
    res = ba.analyze(plan, raw)
    by_c = {e["concurrency"]: e for e in res["evaluations"]}
    check(res["counts"]["points"] == 12 and res["counts"]["workload_points"] == 2, "multi-point runs counted per point")
    check(by_c[8]["verdict"] == "FAIL", "+1% median vs +3% threshold -> FAIL")
    check(by_c[16]["verdict"] == "INCONCLUSIVE", "median passes but one pair does not -> INCONCLUSIVE")
    plan2 = dict(PLAN, gates=[dict(PLAN["gates"][0], where={"concurrency": [16]})])
    check(ba.analyze(plan2, raw)["counts"]["gate_evaluations"] == 1, "where-filter restricts points")

# 6. Bootstrap interval only with enough pairs.
check(ba.bootstrap_ci([1.0, 2.0, 3.0]) is None, "no interval below 4 pairs")
lo, hi = ba.bootstrap_ci([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
check(10.0 <= lo <= hi <= 15.0, "interval within the data range")

# 7. require_ci (opt-in): sample sufficiency and the interval, PASS only gets stricter.
CI_PLAN = {"reference": "a", "expected_exe": {"a": "AAAA", "b": "AAAA"},
           "reliability": {"primary": {"output_tok_s": 5.0}, "tails": {"itl_p99": 20.0}, "min_pairs": 3, "require_ci": True},
           "gates": [{"id": "T", "baseline": "a", "candidate": "b", "metric": "output_tok_s", "rule": "no_regress",
                      "threshold": 5.0, "pair_threshold": 10.0}]}


def pairs_run(raw, changes, base=100.0):
    for i, c in enumerate(changes):
        s, r = i // 4 + 1, i % 4 + 1
        write(raw, f"G-s{s}r{r}-a", [point(base)])
        write(raw, f"G-s{s}r{r}-b", [point(base * (1 + c / 100.0))])


with tempfile.TemporaryDirectory() as t:  # 3 clean pairs: PASS without an interval -> INCONCLUSIVE
    raw = Path(t)
    pairs_run(raw, [0.5, -0.5, 0.2])
    e = ba.analyze(CI_PLAN, raw)["evaluations"][0]
    check(e["verdict"] == "INCONCLUSIVE" and "too few for a 95% confidence interval" in e["reason"], "3 pairs: no CI")
    old = dict(CI_PLAN, reliability={k: v for k, v in CI_PLAN["reliability"].items() if k != "require_ci"})
    check(ba.analyze(old, raw)["evaluations"][0]["verdict"] == "PASS", "without require_ci: unchanged (PASS)")
with tempfile.TemporaryDirectory() as t:  # 8 clean pairs within +-1%: PASS, interval reported
    raw = Path(t)
    pairs_run(raw, [0.5, -0.5, 0.2, -0.2, 0.8, -0.8, 0.1, 0.0])
    e = ba.analyze(CI_PLAN, raw)["evaluations"][0]
    check(e["verdict"] == "PASS" and "95% CI" in e["reason"] and e["ci95_median_pct"] is not None, "8 pairs: PASS with CI")
with tempfile.TemporaryDirectory() as t:  # median -3% passes -5%, but the interval reaches below -5%
    raw = Path(t)
    pairs_run(raw, [-9.0, -8.0, -7.0, -6.0, 0.0, 0.0, 0.0, 0.0])
    e = ba.analyze(CI_PLAN, raw)["evaluations"][0]
    check(e["verdict"] == "INCONCLUSIVE" and "95% CI of the median reaches" in e["reason"], "CI crosses the threshold")
    old = dict(CI_PLAN, reliability={k: v for k, v in CI_PLAN["reliability"].items() if k != "require_ci"})
    check(ba.analyze(old, raw)["evaluations"][0]["verdict"] == "PASS", "the same data passes the median-only rule")
with tempfile.TemporaryDirectory() as t:  # a real regression still FAILs with require_ci
    raw = Path(t)
    pairs_run(raw, [-8.0, -9.0, -7.5, -8.5, -8.0, -9.5, -7.0, -8.2])
    check(ba.analyze(CI_PLAN, raw)["evaluations"][0]["verdict"] == "FAIL", "regression -> FAIL")
with tempfile.TemporaryDirectory() as t:  # round-1 drift in both arms (e2e-20261019-0022): reference unreliable
    raw = Path(t)
    for r, base in enumerate([100.0, 91.0, 91.5, 91.2], start=1):
        write(raw, f"G-s1r{r}-a", [point(base)])
        write(raw, f"G-s1r{r}-b", [point(base)])
    e = ba.analyze(CI_PLAN, raw)["evaluations"][0]
    check(e["verdict"] == "INCONCLUSIVE" and "reference arm unreliable" in e["reason"], "common drift -> unreliable reference")

print(f"all bench_analyze tests passed ({checks} checks)")

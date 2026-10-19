#!/usr/bin/env python3
"""Offline test of tools/bench/ab.sh's schedule: fake dynalm binaries, no model,
no benchmark.

Checks: warm-up runs go to DIR/warmup/ (never DIR/raw/, so the analyzer never
sees them), --warmup-rounds 0 runs none, the measured runs are
sessions x rounds x 2 arms with the arm order alternated every round, and
bench_analyze.py reads exactly the measured runs.
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_analyze", ROOT / "tools/bench_analyze.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)
checks = 0

FAKE = r"""#!/bin/sh
# Fake dynalm: --version, and `benchmark ... --concurrency C --out FILE --request-log FILE`.
[ "$1" = "--version" ] && { echo "DynaLM fake"; exit 0; }
out=""; req=""; c=1
while [ $# -gt 0 ]; do
  case "$1" in --out) out=$2; shift ;; --request-log) req=$2; shift ;; --concurrency) c=$2; shift ;; esac
  shift
done
n=$((c * 4 < 16 ? 16 : c * 4))
printf '{"concurrency": %s, "prompt_tokens": 8, "output_tokens": 4, "requests": %s, "completed": %s, "errors": 0, "mean_completion_tokens": 4, "output_tok_s": 100.0, "ttft_ms": {"p50": 10.0, "p95": 12.0}, "itl_ms": {"p50": 5.0, "p99": 6.0}}\n' "$c" "$n" "$n" >> "$out"
i=0; while [ $i -lt $n ]; do
  printf '{"concurrency": %s, "point_prompt_tokens": 8, "index": %s, "ok": true, "completion_tokens": 4}\n' "$c" "$i" >> "$req"
  i=$((i + 1))
done
echo "$(basename "$out")" >> "$ORDER_LOG"
"""


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        sys.exit(f"FAIL: {what}")


def ab(d: Path, *args):
    env = dict(os.environ, A=str(d / "fa"), B=str(d / "fb"), ALLOW_NOISY="1", ORDER_LOG=str(d / "order.log"))
    p = subprocess.run(["bash", str(ROOT / "tools/bench/ab.sh"), "--model", str(d / "model.gguf"), "--conc", "1,4",
                        "--prompt", "8", "--output", "4", *args], env=env, capture_output=True, text=True)
    check(p.returncode == 0, f"ab.sh failed: {p.stderr[-300:]}")


with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    for n in ("fa", "fb"):
        (d / n).write_text(FAKE)
        (d / n).chmod((d / n).stat().st_mode | stat.S_IEXEC)
    (d / "model.gguf").write_text("x")

    out = d / "run"
    ab(d, "--out", str(out), "--sessions", "2", "--rounds", "4", "--warmup-rounds", "1", "--workload", "w")
    raw_ids = [json.loads(l)["id"] for l in (out / "raw/runs.jsonl").read_text().splitlines()]
    warm_ids = [json.loads(l)["id"] for l in (out / "warmup/runs.jsonl").read_text().splitlines()]
    check(warm_ids == ["w-w1-a", "w-w1-b"], f"one warm-up per arm, in warmup/: {warm_ids}")
    check(len(raw_ids) == 2 * 4 * 2 and not any("-w1-" in i for i in raw_ids), "16 measured runs in raw/, no warm-up")
    order = []  # one line per concurrency: collapse each run's lines
    for x in (d / "order.log").read_text().split():
        if not order or order[-1] != x:
            order.append(x)
    check(order[:2] == ["w-w1-a.jsonl", "w-w1-b.jsonl"], "warm-up runs first")
    rounds = [raw_ids[i:i + 2] for i in range(0, len(raw_ids), 2)]
    firsts = [r[0][-1] for r in rounds]  # which arm ran first in each round
    for sess in (firsts[:4], firsts[4:]):  # alternates every round within a session (order = (s + r) % 2)
        check(all(sess[i] != sess[i + 1] for i in range(3)), f"arm order alternates within a session: {firsts}")
    check(firsts.count("a") == firsts.count("b"), "each arm runs first equally often")
    check("warmup rounds 1 (not analyzed)" in (out / "manifest.txt").read_text(), "manifest records the warm-up")

    plan = {"reference": "a", "reliability": {"primary": {"output_tok_s": 5.0}, "min_pairs": 3, "require_ci": True},
            "gates": [{"id": "E", "baseline": "a", "candidate": "b", "metric": "output_tok_s", "rule": "equivalent",
                       "threshold": 3.0, "pair_threshold": 5.0}]}
    res = ba.analyze(plan, out / "raw")
    check(res["counts"]["runs"] == 16, "the analyzer reads the 16 measured runs only")
    # ALLOW_NOISY=1 runs are recorded quiet=false: no verdict can come from them.
    check(res["counts"]["runs_valid"] == 0 and all(e["verdict"] == "INCONCLUSIVE" for e in res["evaluations"]),
          "unquiet (ALLOW_NOISY) runs give no verdict")
    # As if every run had been quiet: 8 pairs per point, an interval-backed verdict.
    runs = out / "raw/runs.jsonl"
    runs.write_text("".join(json.dumps(dict(json.loads(l), quiet=True)) + "\n" for l in runs.read_text().splitlines()))
    res = ba.analyze(plan, out / "raw")
    check(all(e["pairs"] == 8 and e["verdict"] == "PASS" and "95% CI" in e["reason"] for e in res["evaluations"]),
          "8 quiet pairs per point: a CI-backed verdict")

    out0 = d / "run0"
    ab(d, "--out", str(out0), "--sessions", "1", "--rounds", "1", "--warmup-rounds", "0")
    check(not (out0 / "warmup/runs.jsonl").exists(), "--warmup-rounds 0 runs no warm-up (BSD seq 1 0 would)")

print(f"all ab.sh schedule tests passed ({checks} checks)")

#!/usr/bin/env python3
"""Tests of tools/bench/report.py on synthetic baseline.sh results."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def point(tok_s, ttft, itl, ok=True):
    return {"concurrency": 4, "prompt_tokens": 512, "output_tokens": 128, "requests": 16,
            "completed": 16 if ok else 15, "errors": 0 if ok else 1, "mean_completion_tokens": 128 if ok else 120,
            "output_tok_s": tok_s, "ttft_ms": {"p50": ttft, "p95": ttft * 2}, "itl_ms": {"p50": itl, "p99": itl * 3},
            "e2e_ms": {"p50": ttft + 128 * itl}}


def run(points: dict) -> str:
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "gpu"
        run_dir.mkdir()
        (run_dir / "manifest.txt").write_text("machine: test\n")
        (run_dir / "conditions.log").write_text("10:00 a start: AC\n10:01 b start: AC\n")
        for name, p in points.items():
            (run_dir / name).write_text(json.dumps(p) + "\n")
        r = subprocess.run([sys.executable, str(ROOT / "tools/bench/report.py"), str(run_dir)],
                           capture_output=True, text=True, check=True)
        return r.stdout


checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAIL: {what}")
        sys.exit(1)


# DynaLM 20-25% faster in every round, both engines steady: better.
out = run({f"m-gpu-dynalm-r{r}.jsonl": point(120 + r, 100, 10) for r in (1, 2, 3)} |
          {f"m-gpu-llama-r{r}.jsonl": point(100, 100, 10) for r in (1, 2, 3)})
check("1.21× **better**" in out or "1.22× **better**" in out, "tok/s better")
check("parity" in out, "equal latency is parity")
check("2 server starts" in out, "conditions counted")
# A failed request makes the point not comparable, and the cell is flagged.
out = run({f"m-gpu-dynalm-r{r}.jsonl": point(100, 100, 10, ok=(r != 2)) for r in (1, 2, 3)} |
          {f"m-gpu-llama-r{r}.jsonl": point(100, 100, 10) for r in (1, 2, 3)})
check("not comparable" in out and "(!)" in out, "failure flagged")
# Noisy rounds (spread > 10%) and mixed directions: inconclusive.
out = run({"m-gpu-dynalm-r1.jsonl": point(130, 100, 10), "m-gpu-dynalm-r2.jsonl": point(90, 100, 10),
           "m-gpu-llama-r1.jsonl": point(100, 100, 10), "m-gpu-llama-r2.jsonl": point(100, 100, 10)})
check("inconclusive" in out, "noisy is inconclusive")
print(f"all bench report tests passed ({checks} checks)")

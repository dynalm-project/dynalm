#!/usr/bin/env python3
"""Offline tests of tools/bench/matrix_summary.py and tools/bench/e2e_report.py.

Synthetic model-matrix results, no models or engines:
  1. results.tsv from before the exact-mode speed columns still renders ("–");
  2. with exact_* columns, both reports show exact-mode CPU speed and its
     ratio to llama.cpp's CPU.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ["n", "model", "format", "supported", "size_gb", "file", "arch", "quant",
        "auto_device", "auto", "auto_tok_s", "cpu_device", "cpu", "cpu_tok_s", "native", "native_tok_s",
        "auto_matches_native", "cpu_matches_native", "accuracy",
        "gpu_prompt_tok_s", "gpu_decode_tok_s", "gpu_4user_tok_s",
        "cpu_prompt_tok_s", "cpu_decode_tok_s", "cpu_4user_tok_s",
        "llama_gpu_pp512", "llama_gpu_tg128", "llama_cpu_pp512", "llama_cpu_tg128", "bench_conditions", "seconds"]
EXACT = ["exact_prompt_tok_s", "exact_decode_tok_s", "exact_4user_tok_s"]
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        sys.exit(f"FAIL: {what}")


def row(**over):
    r = {c: "-" for c in BASE + EXACT}
    r.update(n="01", model="qwen2.5:1.5b", format="gguf", supported="yes", arch="qwen2", quant="Q4_K_M",
             auto_device="Metal/Apple_M4_Pro", auto="ok", cpu_device="ggml-CPU", cpu="ok", native="ok",
             auto_matches_native="yes", cpu_matches_native="yes", accuracy="PASS +0.01%",
             gpu_prompt_tok_s="3000.0", gpu_decode_tok_s="160.0", gpu_4user_tok_s="400.0",
             cpu_prompt_tok_s="600.0", cpu_decode_tok_s="160.0", cpu_4user_tok_s="300.0",
             llama_gpu_pp512="3200.0", llama_gpu_tg128="170.0", llama_cpu_pp512="640.0", llama_cpu_tg128="200.0",
             bench_conditions="AC,load=0.4", seconds="100")
    r.update(over)
    return r


def write(run: Path, rows, cols):
    m = run / "matrix-models-e2e-tier1"
    m.mkdir(parents=True)
    with open(m / "results.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")
    return m / "results.tsv"


def reports(run: Path, tsv: Path):
    py = sys.executable
    s = subprocess.run([py, "-B", str(ROOT / "tools/bench/matrix_summary.py"), str(tsv), str(run / "summary.md")],
                       capture_output=True, text=True)
    check(s.returncode == 0, f"matrix_summary failed: {s.stderr[-400:]}")
    e = subprocess.run([py, "-B", str(ROOT / "tools/bench/e2e_report.py"), str(run)], capture_output=True, text=True)
    check(e.returncode == 0, f"e2e_report failed: {e.stderr[-400:]}")
    return (run / "summary.md").read_text(), (run / "REPORT.md").read_text()


with tempfile.TemporaryDirectory() as d:
    # 1. Old results: no exact_* columns.
    old = Path(d) / "old"
    summary, report = reports(old, write(old, [row()], BASE))
    check("Exact CPU decode" in summary and "Exact CPU decode" in report, "exact columns shown for old results")
    check("exact-mode CPU decode –" in report, "old results: exact-mode speed reported as not measured")

    # 2. New results: exact mode measured at 0.48x llama.cpp's CPU decode (96 / 200), 0.37x prompt.
    new = Path(d) / "new"
    summary, report = reports(new, write(new, [row(exact_prompt_tok_s="236.8", exact_decode_tok_s="96.0",
                                                   exact_4user_tok_s="150.0")], BASE + EXACT))
    check("96.0 (200.0, 0.48×)" in summary, "summary: exact decode with llama.cpp CPU and ratio")
    check("236.8 (640.0, 0.37×)" in summary, "summary: exact prompt with llama.cpp CPU and ratio")
    check("exact-mode CPU decode 0.48×" in report, "report summary: median exact-mode ratio")
    check("400.0 / 300.0 / 150.0" in report, "report: 4 users GPU / CPU / exact")

print(f"all e2e report tests passed ({checks} checks)")

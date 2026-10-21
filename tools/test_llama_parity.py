#!/usr/bin/env python3
"""Offline checks of tools/bench/llama_parity.py's setup check (no models, no binaries).

  python tools/test_llama_parity.py

- A path is a parity result only when both engines' logs show the backend it needs: a Metal path
  whose llama.cpp ran on the CPU (a llama.cpp build without Metal: every layer "assigned to device
  CPU") is a setup mismatch, as is one whose log does not show the device, or a DynaLM run on the
  CPU. The CPU paths reject a GPU run the same way.
- Correct Metal and matched-CPU setups pass.
- Every path names its required backend.
The log lines are in the format llama-perplexity -lv 5 and bench_ppl print.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("llama_parity", HERE / "bench" / "llama_parity.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAILED: {what}")
        sys.exit(1)


def llama_log(device: str, layers: int = 3, matched: bool = False) -> str:
    lines = [f"0.00.215.526 D load_tensors: layer {i:3d} assigned to device {device}, is_swa = 0" for i in range(layers)]
    if matched:
        lines += ["0.00.300.000 I llama_context: n_ubatch      = 16", "0.00.300.001 D llama_context: n_seq=1"]
    return "\n".join(lines) + "\n"


def dynalm_log(device: str, matched: bool = False) -> str:
    t = f"model: /m/q.gguf (Q4_K_M), device {device}, 1024 tokens, 2 chunks of 512 (BOS no)\n"
    if matched:
        t += "scoring: ubatch 16, KV fresh pool per chunk\n"
    return t


def setup(llama: str, dynalm: str, ub, want):
    with tempfile.TemporaryDirectory() as tmp:
        l, d = Path(tmp) / "l.log", Path(tmp) / "d.log"
        l.write_text(llama)
        d.write_text(dynalm)
        return lp.setup(l, d, ub, want)


GPU, CPU = "compiled/Metal/Apple M4 Pro", "compiled/ggml-CPU/Apple M4 Pro"

# The false positive seen 2026-10-20: --paths metal with a CPU-only llama.cpp build.
r = setup(llama_log("CPU"), dynalm_log(GPU), None, "gpu")
check(not r["setup_ok"], "metal path, llama.cpp on the CPU: rejected")
check("llama.cpp ran on cpu" in r["setup_problems"] and "devices CPU" in r["setup_problems"], "the reason names the CPU device")
check(r["llama_backend"] == "cpu" and r["dynalm_backend"] == "gpu", "both backends reported")

r = setup(llama_log("MTL0"), dynalm_log(GPU), None, "gpu")
check(r["setup_ok"] and r["llama_devices"] == "MTL0", "metal path on Metal in both: accepted")

r = setup(llama_log("MTL0") + llama_log("CPU", 1), dynalm_log(GPU), None, "gpu")
check(r["setup_ok"] and r["llama_backend"] == "gpu", "partial offload with a GPU device counts as gpu")

r = setup("no device lines\n", dynalm_log(GPU), None, "gpu")
check(not r["setup_ok"] and "unknown backend" in r["setup_problems"], "llama.cpp device not in its log: rejected")

r = setup(llama_log("MTL0"), dynalm_log(CPU), None, "gpu")
check(not r["setup_ok"] and "DynaLM ran on cpu" in r["setup_problems"], "metal path, DynaLM on the CPU: rejected")

r = setup(llama_log("MTL0"), "no model line\n", None, "gpu")
check(not r["setup_ok"] and "DynaLM ran on an unknown backend" in r["setup_problems"], "DynaLM device unknown: rejected")

# Matched CPU path: the same check in the other direction, plus the existing method checks.
r = setup(llama_log("CPU", matched=True), dynalm_log(CPU, matched=True), 16, "cpu")
check(r["setup_ok"], f"cpu path on the CPU in both, ubatch 16, fresh KV: accepted ({r['setup_problems']})")
r = setup(llama_log("MTL0", matched=True), dynalm_log(CPU, matched=True), 16, "cpu")
check(not r["setup_ok"] and "llama.cpp ran on gpu" in r["setup_problems"], "cpu path, llama.cpp on Metal: rejected")

# No required backend (want None): the backend check is off, as before.
check(setup(llama_log("CPU"), dynalm_log(GPU), None, None)["setup_ok"], "want None: no backend check")

check(set(lp.WANT_BACKEND) == set(lp.PATHS), "every path names its required backend")
check(lp.WANT_BACKEND["metal"] == "gpu" and all(v == "cpu" for k, v in lp.WANT_BACKEND.items() if k != "metal"),
      "metal needs gpu, the CPU paths need cpu")

print(f"all llama_parity setup tests passed ({checks} checks)")

#!/usr/bin/env python3
"""CPU feature validation for CI: the kernel tier DynaLM selects must match
what the CPU reports, and the generic fallback must be selectable everywhere.

    python tools/ci/check_isa.py <path/to/dynalm>

Rules (DD-003, DD-046): arm64 -> neon; x86-64 with AVX2 (and an AVX2 build)
-> avx2; anything else -> generic. AVX2 is never required: a CPU or emulator
without it (e.g. x86-64 under Rosetta) must still run, on generic kernels.
"""
import json
import os
import subprocess
import sys


def doctor(exe, env=None):
    out = subprocess.run([exe, "doctor", "--json"], capture_output=True, text=True,
                         env={**os.environ, **(env or {})}, timeout=120)
    if out.returncode != 0:
        sys.exit(f"dynalm doctor exited {out.returncode}:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout)


def main():
    exe = os.path.abspath(sys.argv[1])
    d = doctor(exe)
    arch, isa, kernel = d["os"]["arch"], d["cpu"]["isa"], d["kernel"]
    compiled = d["compiled_isas"]
    if arch == "arm64":
        want = "neon" if "neon" in compiled and isa.get("NEON") else "generic"
    elif arch == "x86_64":
        want = "avx2" if "avx2" in compiled and isa.get("AVX2") and isa.get("FMA") and isa.get("F16C") else "generic"
    else:
        want = "generic"
    print(f"{d['os']['name']} {arch}: CPU '{d['cpu']['name']}', features {sorted(k for k, v in isa.items() if v)}, "
          f"compiled {compiled}, selected '{kernel}', expected '{want}'")
    if kernel != want:
        sys.exit(f"FAIL: selected kernel tier '{kernel}', expected '{want}'")
    if not d["dynacore_ok"]:
        sys.exit(f"FAIL: DynaCore self-test: {d['dynacore_detail']}")
    g = doctor(exe, {"DYNACORE_ISA": "generic"})
    if g["kernel"] != "generic" or not g["dynacore_ok"]:
        sys.exit(f"FAIL: generic fallback not selectable: {g['kernel']} / {g['dynacore_detail']}")
    bogus = doctor(exe, {"DYNACORE_ISA": "amx" if arch == "x86_64" else "avx2"})
    if bogus["kernel"] != want:
        sys.exit(f"FAIL: an unavailable DYNACORE_ISA must fall back to '{want}', got '{bogus['kernel']}'")
    print("ISA selection ok: best tier, forced generic, and unavailable-tier fallback")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline checks of tools/compare_engines.py (no servers, no models).

  python tools/test_compare_engines.py

- Rows below 95% of the requested output (or with errors) are "not comparable"
  and never ranked; comparable rows are ranked by output tok/s per workload.
- --dynalm must name an existing file (no silent fallback to another build).
- Existing results are never appended to.
- The report header carries every engine's executable path and SHA-256.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare_engines", HERE / "compare_engines.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAILED: {what}")
        sys.exit(1)


def row(label, mean, tok, errors=0, conc=1, prompt=128):
    return {"label": label, "model_file": "m.gguf", "concurrency": conc, "prompt_tokens": prompt, "output_tokens": 128,
            "mean_completion_tokens": mean, "output_tok_s": tok, "errors": errors, "ttft_ms": {"p50": 1.0},
            "itl_ms": {"p50": 1.0, "p99": 2.0}, "tpot_ms": {"p50": 1.0}, "e2e_ms": {"p99": 3.0}}


with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    res = out / "results.jsonl"
    rows = [row("DynaLM", 128, 17.0), row("Ollama", 74, 30.0), row("llama.cpp", 128, 17.4),
            row("DynaLM", 128, 36.9, conc=4), row("Ollama", 122, 40.0, conc=4), row("llama.cpp", 128, 38.9, errors=1, conc=4)]
    res.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {"created": "t", "host": {"cpu": "c", "os": "o"}, "threads": 10, "ctx_per_user": 2048, "requests": 0,
                "warmup_requests": 0, "ollama_raw_template": True,
                "engines": {"dynalm": {"path": "X:/next/dynalm.exe", "sha256": "ABCDEF0123456789FFFF", "version": "v"}},
                "models": [{"path": "m.gguf", "bytes": 1 << 30, "sha256": "1234"}]}
    text = ce.report(res, out, manifest)
    check("| Ollama | 1 | 128 | - | 128 | 74 | no | 30.00 | - |" in text, "short Ollama row not ranked")
    check("| llama.cpp | 1 | 128 | - | 128 | 128 | yes | 17.40 | 1 |" in text, "fastest comparable row ranked 1")
    check("| DynaLM | 1 | 128 | - | 128 | 128 | yes | 17.00 | 2 |" in text, "second comparable row ranked 2")
    check("| Ollama | 4 | 128 | - | 128 | 122 | yes | 40.00 | 1 |" in text, "row at >= 95% of the target is comparable")
    check("| llama.cpp | 4 | 128 | - | 128 | 128 | no | 38.90 | - |" in text, "row with errors is not comparable")
    check("X:/next/dynalm.exe" in text and "ABCDEF0123456789" in text, "executable path and hash in the report")
    check((out / "report.md").read_text(encoding="utf-8") == text, "report.md written")

    script = str(HERE / "compare_engines.py")
    r = subprocess.run([sys.executable, script, "--dynalm", str(out / "missing.exe"), "--model", "x.gguf", "--out", str(out / "a")],
                       capture_output=True, text=True)
    check(r.returncode == 1 and "does not exist" in r.stderr, "missing --dynalm refused")
    r = subprocess.run([sys.executable, script, "--model", "x.gguf", "--out", str(out / "b")], capture_output=True, text=True)
    check(r.returncode != 0, "--dynalm is required")
    r = subprocess.run([sys.executable, script, "--dynalm", sys.executable, "--model", "x.gguf", "--out", str(out)],
                       capture_output=True, text=True)
    check(r.returncode == 1 and "refusing to append" in r.stderr, "existing results never appended to")

print(f"all compare_engines tests passed ({checks} checks)")

#!/usr/bin/env python3
"""Offline tests of tools/bench/model_matrix.py's Hugging Face pre-check.

The pre-check fetches only the small files (never *.safetensors), runs
`dynalm inspect` on them and turns its Support section into the row's
"supported" value. A fake `dynalm` prints canned inspect output and the
download is replaced by a local stub: no network, no model, no binary.
"""
import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_matrix", ROOT / "tools/bench/model_matrix.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        sys.exit(f"FAIL: {what}")


SUPPORT = {
    "tokenizer": """Support:
  Architecture:  yes (llama)
  Tokenizer:     NO (tokenizer.json pre-tokenizer (Split+Split+Split+Split+Digits+ByteLevel) is not one the engine implements yet)
  Weights:       not present (only config and tokenizer files)
Supported:     NO (tokenizer)
""",
    "architecture": """Support:
  Architecture:  NO (no adapter for 'mamba')
  Tokenizer:     yes (byte-level BPE, pre-tokenizer default)
  Weights:       not present (only config and tokenizer files)
Supported:     NO (architecture)
""",
    "partial": """Support:
  Architecture:  yes (qwen2)
  Tokenizer:     yes (byte-level BPE, pre-tokenizer qwen2)
  Weights:       not present (only config and tokenizer files)
Supported:     PARTIAL (architecture and tokenizer supported; weights not present, not checked)
""",
}

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    # A fake dynalm that prints $FAKE_INSPECT: a .cmd on Windows (CreateProcess runs it), sh elsewhere.
    if os.name == "nt":
        fake = d / "dynalm.cmd"
        fake.write_text('@type "%FAKE_INSPECT%"\r\n')
    else:
        fake = d / "dynalm"
        fake.write_text('#!/bin/sh\ncat "$FAKE_INSPECT"\n')
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    (d / "logs").mkdir()
    fetched = []

    def fake_download(repo, files, root, token):
        fetched.append([f["path"] for f in files])
        t = root / repo.replace("/", "__")
        t.mkdir(parents=True, exist_ok=True)
        return "yes", t

    mm.hf_download = fake_download
    mm.say = lambda tag, msg: None
    m = SimpleNamespace(dynalm=str(fake), out=d, logs=d / "logs", token="")
    files = [{"path": "config.json", "size": 600}, {"path": "tokenizer.json", "size": 1_400_000},
             {"path": "model.safetensors", "size": 2_700_000_000}]
    for case, want in (("tokenizer", "no (tokenizer)"), ("architecture", "no (architecture)"), ("partial", "yes")):
        (d / f"{case}.txt").write_text(SUPPORT[case])
        os.environ["FAKE_INSPECT"] = str(d / f"{case}.txt")
        got = mm.Matrix.hf_precheck(m, "owner/repo", files, "01")
        check(got == want, f"{case}: {got!r} != {want!r}")
        check(not any(p.endswith(".safetensors") for p in fetched[-1]), f"{case}: weights fetched before the check")
        check(not (d / "hf-meta" / "owner__repo").exists(), f"{case}: the small files are removed after the check")

    # No verdict at all (inspect did not run, crashed or printed nothing): unverified, not "yes".
    (d / "empty.txt").write_text("")
    os.environ["FAKE_INSPECT"] = str(d / "empty.txt")
    check(mm.Matrix.hf_precheck(m, "owner/repo", files, "01") == "check-failed", "no verdict -> check-failed")

    # A failed small-file download is reported as such, without running inspect.
    mm.hf_download = lambda repo, files, root, token: ("gated", None)
    check(mm.Matrix.hf_precheck(m, "owner/repo", files, "01") == "gated", "download failure passed through")

print(f"all model matrix pre-check tests passed ({checks} checks)")

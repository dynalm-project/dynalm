#!/usr/bin/env python3
"""Offline tests of tools/bench/model_matrix.py's Hugging Face pre-check.

The pre-check fetches only the small files (never *.safetensors), runs
`dynalm inspect` on them and turns its Support section into the row's
"supported" value. A fake `dynalm` prints canned inspect output and the
download is replaced by a local stub: no network, no model, no binary.
"""
import importlib.util
import json
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

# The speed method (BENCH_METHOD "d512-r5"): both engines decode 64 tokens at depth 512, 5
# repetitions after a warm-up, flash attention on; llama.cpp's CPU run without op offload.
# `run` is replaced: the fakes write what dynalm benchmark / llama-bench would.
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "logs").mkdir()
    fake_lb = d / "llama-bench"
    fake_lb.write_text("")
    fake_lb.chmod(0o755)
    m = SimpleNamespace(dynalm="dynalm", out=d, logs=d / "logs", threads="8", llama_bench=str(fake_lb))
    m.llama_cmd = lambda ngl, model: mm.Matrix.llama_cmd(m, ngl, model)
    gpu, cpu = mm.Matrix.llama_cmd(m, 99, "m.gguf"), mm.Matrix.llama_cmd(m, 0, "m.gguf")
    for cmd in (gpu, cpu):
        flags = dict(zip(cmd[1::2], cmd[2::2]))
        check(flags.get("-p") == "512" and flags.get("-n") == "64" and flags.get("-d") == "0,512", f"depths: {cmd}")
        check(flags.get("-r") == "5" and flags.get("-fa") == "on" and "--no-warmup" not in cmd, f"reps, fa, warm-up: {cmd}")
    check(dict(zip(cpu[1::2], cpu[2::2])).get("-nopo") == "1" and "-nopo" not in gpu, "op offload off on the CPU only")
    calls = []

    def fake_run(cmd, log, timeout, stdout=None):
        calls.append(cmd)
        log.write_text("")
        if cmd[0] == str(fake_lb):
            res = [{"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 900.0, "stddev_ts": 9.0},
                   {"n_prompt": 0, "n_gen": 64, "n_depth": 0, "avg_ts": 110.0, "stddev_ts": 1.0},
                   {"n_prompt": 512, "n_gen": 0, "n_depth": 512, "avg_ts": 800.0, "stddev_ts": 8.0},
                   {"n_prompt": 0, "n_gen": 64, "n_depth": 512, "avg_ts": 100.0, "stddev_ts": 2.0}]
            stdout.write_text(json.dumps(res))
            return 0
        a = dict(zip(cmd[3::2], cmd[4::2]))  # dynalm benchmark MODEL --backend B ...
        c, n = int(a["--concurrency"]), int(a["--requests"])
        with open(a["--out"], "a") as f:
            f.write(json.dumps({"concurrency": c, "errors": 0, "requests": n, "completed": n, "output_tok_s": 300.0}) + "\n")
        with open(a["--request-log"], "a") as f:
            for i in range(n):  # decode: 63 tokens in 630 / 600 ms alternately -> 105, 100, 105, 100, 105 tok/s
                f.write(json.dumps({"concurrency": c, "ok": True, "completion_tokens": 64, "prompt_tokens": 512,
                                    "ttft_ms": 512.0, "e2e_ms": 512.0 + (630.0 if i % 2 else 600.0)}) + "\n")
        return 0

    real_run, mm.run = mm.run, fake_run
    got = mm.Matrix.llama(m, 99, "m.gguf", "01")
    check(got == ("900.0", "100.0", "9.00", "2.00", "110.0"), f"llama-bench: decode at depth 512, not 0: {got}")
    pp, tg, four, pp_sd, tg_sd = mm.Matrix.dyn_bench(m, "auto", "m.gguf", "01")
    check(pp == "1000.0" and tg == "103.0" and four == "300.0", f"dynalm: mean of the 5 requests: {pp} {tg} {four}")
    check(tg_sd == "2.74" and pp_sd == "0.00", f"dynalm: spread of the 5 requests: {tg_sd} {pp_sd}")
    one = next(c for c in calls if c[0] == "dynalm" and "1" == dict(zip(c[3::2], c[4::2]))["--concurrency"])
    oa = dict(zip(one[3::2], one[4::2]))
    check(oa["--prompt"] == "512" and oa["--output"] == "64" and oa["--requests"] == "5" and oa["--warmup-requests"] == "1",
          f"dynalm one-user point: {one}")
    mm.run = real_run

print(f"all model matrix pre-check tests passed ({checks} checks)")

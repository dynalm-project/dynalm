#!/usr/bin/env python3
"""CLI and HTTP API smoke test against a real model (release validation).

  python tools/ci/smoke_api.py <dynalm executable> <model.gguf> [--port 8191] [--threads N]

Checks, with the given executable only:
- `run` with a missing model file and with a file that is not GGUF fails with a
  non-zero exit code and an error message (no crash);
- `serve` starts and logs the scheduler budgets in effect (default policy and
  `--policy latency`);
- GET /health, GET /v1/models;
- POST /v1/completions returns text and usage; a streaming request ends with
  [DONE];
- malformed requests get 4xx answers (bad JSON, missing prompt, bad
  max_tokens), an unknown route gets 404;
- GET /metrics exposes the request counters and the matmul kernel counters;
- POST /admin/shutdown drains and the process exits with code 0.
Only the server processes this script starts are stopped.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

checks = 0
failures = []


def check(cond, what):
    global checks
    checks += 1
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def http(method, url, body=None, timeout=120, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_server(exe, model, port, threads, log_path, extra=()):
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([exe, "serve", model, "--port", str(port), "--threads", str(threads), "--ctx", "4096", *extra],
                            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 240
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc, False
        code, body = http("GET", f"http://127.0.0.1:{port}/health", timeout=2)
        if code == 200:
            return proc, True
        time.sleep(1)
    return proc, False


def stop_own(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--logs", default="")
    a = ap.parse_args()
    exe = str(Path(a.exe).resolve())
    logs = Path(a.logs) if a.logs else Path(tempfile.mkdtemp())
    logs.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{a.port}"
    print(f"executable: {exe}\nmodel: {a.model}\nlogs: {logs}")
    if not port_free(a.port):
        print(f"port {a.port} is in use; pick another with --port")
        return 2

    # 1. run: missing model, not-a-model file
    r = subprocess.run([exe, "run", str(logs / "does-not-exist.gguf"), "-p", "hi", "-n", "4"], capture_output=True, text=True, timeout=120)
    check(r.returncode != 0 and len((r.stderr + r.stdout).strip()) > 0, f"run with a missing model fails cleanly (exit {r.returncode})")
    junk = logs / "not-a-model.gguf"
    junk.write_bytes(b"this is not a gguf file" * 64)
    r = subprocess.run([exe, "run", str(junk), "-p", "hi", "-n", "4"], capture_output=True, text=True, timeout=120)
    check(r.returncode not in (0,) and r.returncode > 0 and len((r.stderr + r.stdout).strip()) > 0,
          f"run with a non-GGUF file fails cleanly (exit {r.returncode})")

    # 2. serve, default policy
    proc, up = start_server(exe, a.model, a.port, a.threads, logs / "serve.log")
    check(up, "serve starts and /health answers")
    if up:
        try:
            log = (logs / "serve.log").read_text(encoding="utf-8", errors="replace")
            check("balanced policy (prefill budget 64, decode budget 64" in log, "serve logs the effective scheduler budgets")
            code, body = http("GET", base + "/v1/models")
            check(code == 200 and "data" in body, "GET /v1/models")
            code, body = http("POST", base + "/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
            ok = code == 200
            usage = json.loads(body).get("usage", {}) if ok else {}
            check(ok and 0 < usage.get("completion_tokens", 0) <= 8, f"POST /v1/completions (status {code}, usage {usage})")
            code, body = http("POST", base + "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 8, "stream": True})
            check(code == 200 and "[DONE]" in body, f"streaming chat completion ends with [DONE] (status {code})")
            code, _ = http("POST", base + "/v1/completions", raw=b"{not json")
            check(400 <= code < 500, f"malformed JSON gets 4xx (status {code})")
            code, _ = http("POST", base + "/v1/completions", {"max_tokens": 4})
            check(400 <= code < 500, f"missing prompt gets 4xx (status {code})")
            code, _ = http("POST", base + "/v1/completions", {"prompt": "hi", "max_tokens": -5})
            check(400 <= code < 500, f"negative max_tokens gets 4xx (status {code})")
            code, _ = http("GET", base + "/no-such-route")
            check(code == 404, f"unknown route gets 404 (status {code})")
            code, body = http("GET", base + "/metrics")
            check(code == 200 and "dynalm_requests_total" in body, "GET /metrics has request counters")
            check('dynacore_matmul_calls_total{kernel="f32_expand_gemm"}' in body or 'dynacore_matmul_calls_total{kernel="int8_per32"}' in body,
                  "GET /metrics has the matmul kernel counters")
            (logs / "metrics.prom").write_text(body, encoding="utf-8")
            code, _ = http("POST", base + "/admin/shutdown", {})
            check(code in (200, 202, 204), f"POST /admin/shutdown accepted (status {code})")
            try:
                rc = proc.wait(90)
            except subprocess.TimeoutExpired:
                rc = None
            check(rc == 0, f"server drains and exits with code 0 (exit {rc})")
        finally:
            stop_own(proc)

    # 3. serve --policy latency: the effective budgets differ from the defaults
    proc, up = start_server(exe, a.model, a.port, a.threads, logs / "serve-latency.log", ("--policy", "latency"))
    check(up, "serve --policy latency starts")
    if up:
        try:
            log = (logs / "serve-latency.log").read_text(encoding="utf-8", errors="replace")
            check("latency policy (prefill budget 32" in log, "latency policy logs its effective prefill budget (32)")
            http("POST", base + "/admin/shutdown", {})
            try:
                proc.wait(90)
            except subprocess.TimeoutExpired:
                pass
        finally:
            stop_own(proc)

    print(f"{checks - len(failures)} of {checks} smoke checks passed")
    if failures:
        print("failed: " + "; ".join(failures))
        return 1
    print(f"all API smoke checks passed ({checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

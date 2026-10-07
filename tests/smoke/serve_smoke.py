#!/usr/bin/env python3
"""Server smoke test: `dynalm serve` with a tiny model, then the OpenAI-compatible
endpoints (models, completions, chat completions with and without streaming,
health, metrics) and a graceful `dynalm stop`. Standard library only.

    python tests/smoke/serve_smoke.py <path/to/dynalm> <tiny.gguf> [--execution compiled]
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exe, model, extra = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3:]
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([exe, "-q", "serve", model, "--port", str(port), "--model-id", "tiny"] + extra,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                print(proc.stdout.read())
                raise SystemExit(f"server exited early with {proc.returncode}")
            try:
                if http("GET", base + "/health", timeout=2)[0] == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit("server did not become healthy")

        status, body = http("GET", base + "/v1/models")
        assert status == 200 and json.loads(body)["data"][0]["id"] == "tiny", body

        status, body = http("POST", base + "/v1/completions", {"prompt": "hello", "max_tokens": 6, "temperature": 0})
        r = json.loads(body)
        assert status == 200 and r["choices"][0]["finish_reason"] in ("length", "stop"), body
        assert r["usage"]["completion_tokens"] >= 1, body

        status, body = http("POST", base + "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6})
        assert status == 200 and json.loads(body)["choices"][0]["message"]["role"] == "assistant", body

        status, body = http("POST", base + "/v1/completions",
                            {"prompt": "hello", "max_tokens": 6, "stream": True})
        events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
        assert status == 200 and events and events[-1] == "[DONE]", body
        assert any(json.loads(e)["choices"][0].get("text") for e in events[:-1]), body

        status, body = http("GET", base + "/metrics")
        assert status == 200 and "dynalm_generation_tokens_total" in body, body[:400]

        stop = subprocess.run([exe, "stop", "--port", str(port)], capture_output=True, text=True, timeout=60)
        assert stop.returncode == 0, stop.stderr
        proc.wait(timeout=60)
        assert proc.returncode == 0, f"server exit code {proc.returncode}"
        print(f"serve smoke ok ({' '.join(extra) or 'reference'})")
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())

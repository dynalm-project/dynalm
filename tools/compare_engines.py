#!/usr/bin/env python3
"""Head-to-head CPU benchmark: DynaLM vs llama.cpp (llama-server) vs Ollama.

Runs natively (no containers). For each model, each engine is started on its
own, serves the same GGUF file, and is measured by the same load generator
(`dynalm benchmark --url`, OpenAI-compatible HTTP, streaming, unique prompts,
uncounted warm-up). Only one server runs at a time.

  python tools/compare_engines.py \
      --dynalm build/msvc-release/bin/dynalm.exe \
      --llama-server path/to/llama-server.exe \
      --ollama ollama \
      --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf:1,4:128,512:128 \
      --threads 10 --out results/compare

A --model spec is FILE[:CONCURRENCY:PROMPTS:OUTPUTS] (lists comma-separated).
Pass --skip ENGINE to leave an engine out. Results: <out>/results.jsonl and a
Markdown table in <out>/report.md.

Caveats:
- Ollama ignores `ignore_eos`, so its completions can be shorter than asked;
  the report shows the mean completion length so this is visible.
- Ollama picks its own batching; the script sets num_thread and
  OLLAMA_NUM_PARALLEL to match the other engines.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORTS = {"dynalm": 8181, "llama.cpp": 8182, "ollama": 11434}


def get(url: str, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def wait_ready(url: str, ok, seconds: int, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        body = get(url)
        if body is not None and ok(body):
            return
        time.sleep(1)
    raise RuntimeError(f"{url} not ready after {seconds}s")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def parse_spec(spec: str):
    parts = spec.split(":")
    # Windows drive letters (C:/...) contain a colon.
    if len(parts) > 1 and len(parts[0]) == 1 and parts[1].startswith(("/", "\\")):
        parts = [parts[0] + ":" + parts[1]] + parts[2:]
    path = Path(parts[0]).resolve()
    conc = parts[1] if len(parts) > 1 else "1,4"
    prompts = parts[2] if len(parts) > 2 else "128,512"
    outputs = parts[3] if len(parts) > 3 else "128"
    return path, conc, prompts, outputs


def run_bench(args, model: Path, url: str, name: str, conc: str, prompts: str, outputs: str,
              label: str, results: Path) -> None:
    tmp = results.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    cmd = [args.dynalm, "benchmark", str(model), "--url", url, "--model-name", name,
           "--concurrency", conc, "--prompt", prompts, "--output", outputs, "--out", str(tmp), "--no-diag"]
    if args.requests:
        cmd += ["--requests", str(args.requests)]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(tmp, encoding="utf-8") as f, open(results, "a", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            r["label"] = label
            r["model_file"] = model.name
            out.write(json.dumps(r) + "\n")
    tmp.unlink(missing_ok=True)


def max_conc(conc: str) -> int:
    return max(int(c) for c in conc.split(","))


def bench_dynalm(args, model, conc, prompts, outputs, results, logs):
    port = PORTS["dynalm"]
    cmd = [args.dynalm, "serve", str(model), "--port", str(port), "--threads", str(args.threads),
           "--ctx", str(args.ctx * max_conc(conc))]
    log = open(logs / f"dynalm-{model.stem}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(f"http://127.0.0.1:{port}/health", lambda b: "ok" in b, 300, proc)
        run_bench(args, model, f"http://127.0.0.1:{port}", model.name, conc, prompts, outputs,
                  "DynaLM", results)
    finally:
        stop(proc)


def bench_llamacpp(args, model, conc, prompts, outputs, results, logs):
    port = PORTS["llama.cpp"]
    n = max_conc(conc)
    cmd = [args.llama_server, "-m", str(model), "--port", str(port), "-t", str(args.threads),
           "-c", str(args.ctx * n), "-np", str(n), "--no-webui"]
    log = open(logs / f"llamacpp-{model.stem}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(f"http://127.0.0.1:{port}/health", lambda b: '"ok"' in b, 300, proc)
        run_bench(args, model, f"http://127.0.0.1:{port}", model.name, conc, prompts, outputs,
                  "llama.cpp", results)
    finally:
        stop(proc)


def bench_ollama(args, model, conc, prompts, outputs, results, logs):
    port = PORTS["ollama"]
    n = max_conc(conc)
    # Restart Ollama so OLLAMA_NUM_PARALLEL applies.
    subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True) if os.name == "nt" else None
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True) if os.name == "nt" else None
    time.sleep(2)
    env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(n), OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_HOST=f"127.0.0.1:{port}")
    log = open(logs / f"ollama-{model.stem}.log", "w")
    proc = subprocess.Popen([args.ollama, "serve"], stdout=log, stderr=subprocess.STDOUT, env=env)
    name = "dynalm-bench-" + model.stem.lower().replace("_", "-").replace(".", "-")
    try:
        wait_ready(f"http://127.0.0.1:{port}/api/version", lambda b: "version" in b, 120, proc)
        modelfile = logs / f"Modelfile.{model.stem}"
        modelfile.write_text(f"FROM {model.as_posix()}\nPARAMETER num_thread {args.threads}\n"
                             f"PARAMETER num_ctx {args.ctx}\n", encoding="utf-8")
        subprocess.run([args.ollama, "create", name, "-f", str(modelfile)], check=True, env=env)
        run_bench(args, model, f"http://127.0.0.1:{port}", name, conc, prompts, outputs, "Ollama", results)
    finally:
        subprocess.run([args.ollama, "rm", name], capture_output=True, env=env)
        stop(proc)


def report(results: Path, out: Path) -> str:
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if l.strip()]
    lines = ["| Model | Engine | Users | Prompt | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) | Mean output tokens | Errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows.sort(key=lambda r: (r["model_file"], r["concurrency"], r["prompt_tokens"], r["label"]))
    for r in rows:
        lines.append(f"| {r['model_file']} | {r['label']} | {r['concurrency']} | {r['prompt_tokens']} | "
                     f"{r['output_tok_s']:.2f} | {r['ttft_ms']['p50']:.0f} | {r['itl_ms']['p50']:.1f} | "
                     f"{r['mean_completion_tokens']:.0f} | {r['errors']} |")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynalm", default=shutil.which("dynalm") or "dynalm")
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server")
    ap.add_argument("--ollama", default=shutil.which("ollama") or "ollama")
    ap.add_argument("--model", action="append", required=True, help="FILE[:CONC:PROMPTS:OUTPUTS]")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ctx", type=int, default=2048, help="context per user")
    ap.add_argument("--requests", type=int, default=0, help="requests per point (0 = benchmark default)")
    ap.add_argument("--skip", action="append", default=[], choices=["dynalm", "llama.cpp", "ollama"])
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    args.dynalm = str(Path(args.dynalm).resolve()) if Path(args.dynalm).exists() else args.dynalm

    engines = [("dynalm", bench_dynalm), ("llama.cpp", bench_llamacpp), ("ollama", bench_ollama)]
    for spec in args.model:
        model, conc, prompts, outputs = parse_spec(spec)
        if not model.is_file():
            print(f"model not found: {model}", file=sys.stderr)
            return 1
        for name, fn in engines:
            if name in args.skip:
                continue
            print(f"== {model.name}: {name}", flush=True)
            try:
                fn(args, model, conc, prompts, outputs, results, logs)
            except Exception as e:  # keep going with the other engines
                print(f"  {name} failed: {e}", file=sys.stderr, flush=True)
    print(report(results, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

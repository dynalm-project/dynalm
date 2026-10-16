#!/usr/bin/env python3
"""Head-to-head CPU benchmark: DynaLM vs llama.cpp (llama-server) vs Ollama.

Runs natively (no containers). For each model, each engine is started on its
own, serves the same GGUF file, and is measured by the same load generator
(`dynalm benchmark --url`, OpenAI-compatible HTTP, streaming, unique prompts,
uncounted warm-up). Only one server runs at a time, and only servers this
script started are ever stopped.

  python tools/compare_engines.py \
      --dynalm path/to/dynalm.exe \
      --llama-server path/to/llama-server.exe \
      --ollama ollama \
      --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf:1,4:128,512:128 \
      --threads 10 --out results/compare

A --model spec is FILE[:CONCURRENCY:PROMPTS:OUTPUTS] (lists comma-separated).
Pass --skip ENGINE to leave an engine out. Results: <out>/results.jsonl, the
provenance of every engine and model in <out>/manifest.json, per-request logs
in <out>/logs/, and a Markdown table in <out>/report.md.

Fairness rules (also stated in the report):
- The DynaLM executable is exactly --dynalm: its absolute path, version and
  SHA-256 are recorded in the manifest and in every result row.
- Every engine gets the same context per user (--ctx) and thread count.
- Ollama ignores `ignore_eos`, so generation can stop early at EOS, and it
  wraps /v1/completions prompts in the chat template. The script asks for a raw
  template (TEMPLATE {{ .Prompt }}); Ollama 0.40.1 ignored it on this endpoint
  (+29 prompt tokens on Qwen2.5). The report shows the prompt length each
  engine reported ("Prompt seen"). A row whose mean completion length is below
  95% of the requested output (or that has errors) is marked "not comparable"
  and is not ranked.
- A port already in use makes that engine fail its preflight; the script
  never stops a process it did not start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMPARABLE_FRACTION = 0.95


def get(url: str, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


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
    """Stops a server this script started (never anything else)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def version_of(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = (r.stdout + r.stderr).strip().splitlines()
        return " | ".join(l.strip() for l in text if l.strip())[:300]
    except Exception as e:
        return f"unavailable ({e})"


def exe_info(path: str, version_cmd: list[str]) -> dict:
    p = Path(path)
    if not p.is_file():
        found = shutil.which(path)
        p = Path(found) if found else p
    if not p.is_file():
        return {"path": path, "exists": False}
    return {"path": str(p.resolve()), "exists": True, "sha256": sha256(p), "bytes": p.stat().st_size,
            "version": version_of([str(p.resolve())] + version_cmd)}


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
              label: str, results: Path, engine: dict, model_info: dict) -> None:
    tmp = results.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    reqlog = results.parent / "logs" / f"requests-{label.replace('.', '')}-{model.stem}.jsonl"
    cmd = [args.dynalm, "benchmark", str(model), "--url", url, "--model-name", name,
           "--concurrency", conc, "--prompt", prompts, "--output", outputs, "--out", str(tmp), "--no-diag",
           "--request-log", str(reqlog)]
    if args.requests:
        cmd += ["--requests", str(args.requests)]
    if args.warmup_requests:
        cmd += ["--warmup-requests", str(args.warmup_requests)]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(tmp, encoding="utf-8") as f, open(results, "a", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            r["label"] = label
            r["model_file"] = model.name
            r["model_sha256"] = model_info.get("sha256")
            r["engine_path"] = engine.get("path")
            r["engine_sha256"] = engine.get("sha256")
            r["threads"] = args.threads
            r["ctx_per_user"] = args.ctx
            out.write(json.dumps(r) + "\n")
    tmp.unlink(missing_ok=True)


def max_conc(conc: str) -> int:
    return max(int(c) for c in conc.split(","))


def bench_dynalm(args, model, conc, prompts, outputs, results, logs, info, minfo):
    port = args.dynalm_port
    if not port_free(port):
        raise RuntimeError(f"port {port} is in use (another server?); not stopping it")
    cmd = [args.dynalm, "serve", str(model), "--port", str(port), "--threads", str(args.threads),
           "--ctx", str(args.ctx * max_conc(conc))]
    log = open(logs / f"dynalm-{model.stem}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(f"http://127.0.0.1:{port}/health", lambda b: "ok" in b, 300, proc)
        run_bench(args, model, f"http://127.0.0.1:{port}", model.name, conc, prompts, outputs,
                  "DynaLM", results, info["dynalm"], minfo)
        if args.dynalm_scrape_metrics:
            # Opt-in: the server's /metrics after the benchmark, before the server stops.
            body = get(f"http://127.0.0.1:{port}/metrics", timeout=10)
            path = results.parent / f"dynalm-metrics-{model.stem}-c{conc.replace(',', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.prom"
            path.write_text(body if body is not None else "# /metrics unavailable\n", encoding="utf-8")
            print(f"  DynaLM /metrics saved: {path}", flush=True)
    finally:
        stop(proc)


def bench_llamacpp(args, model, conc, prompts, outputs, results, logs, info, minfo):
    port = args.llama_port
    if not port_free(port):
        raise RuntimeError(f"port {port} is in use (another server?); not stopping it")
    n = max_conc(conc)
    cmd = [args.llama_server, "-m", str(model), "--port", str(port), "-t", str(args.threads),
           "-c", str(args.ctx * n), "-np", str(n), "--no-webui"]
    log = open(logs / f"llamacpp-{model.stem}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(f"http://127.0.0.1:{port}/health", lambda b: '"ok"' in b, 300, proc)
        run_bench(args, model, f"http://127.0.0.1:{port}", model.name, conc, prompts, outputs,
                  "llama.cpp", results, info["llama.cpp"], minfo)
    finally:
        stop(proc)


def bench_ollama(args, model, conc, prompts, outputs, results, logs, info, minfo):
    # The script's own `ollama serve` on its own port: a running Ollama (the
    # desktop app on 11434) is left alone. OLLAMA_NUM_PARALLEL applies to this
    # server only.
    port = args.ollama_port
    if not port_free(port):
        raise RuntimeError(f"port {port} is in use (another Ollama?); not stopping it -- pass --ollama-port")
    n = max_conc(conc)
    env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(n), OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_HOST=f"127.0.0.1:{port}")
    log = open(logs / f"ollama-{model.stem}.log", "w")
    proc = subprocess.Popen([args.ollama, "serve"], stdout=log, stderr=subprocess.STDOUT, env=env)
    name = "dynalm-bench-" + model.stem.lower().replace("_", "-").replace(".", "-")
    try:
        wait_ready(f"http://127.0.0.1:{port}/api/version", lambda b: "version" in b, 120, proc)
        modelfile = logs / f"Modelfile.{model.stem}"
        lines = [f"FROM {model.as_posix()}", f"PARAMETER num_thread {args.threads}",
                 # Same context per user as the other engines (num_ctx is per parallel slot).
                 f"PARAMETER num_ctx {args.ctx}"]
        if not args.ollama_chat_template:
            # The prompt as sent: the same prompt length as llama.cpp and DynaLM.
            lines.append('TEMPLATE """{{ .Prompt }}"""')
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run([args.ollama, "create", name, "-f", str(modelfile)], check=True, env=env)
        run_bench(args, model, f"http://127.0.0.1:{port}", name, conc, prompts, outputs, "Ollama", results,
                  info["ollama"], minfo)
    finally:
        subprocess.run([args.ollama, "rm", name], capture_output=True, env=env)
        stop(proc)


def comparable(r: dict) -> bool:
    return r["errors"] == 0 and r["mean_completion_tokens"] >= COMPARABLE_FRACTION * r["output_tokens"]


def prompt_seen(out: Path, r: dict) -> float | None:
    """Mean prompt tokens the engine reported (usage.prompt_tokens) for this point, from the request log."""
    log = out / "logs" / f"requests-{r['label'].replace('.', '')}-{Path(r['model_file']).stem}.jsonl"
    if not log.exists():
        return None
    vals = []
    for line in open(log, encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("concurrency") == r["concurrency"] and q.get("point_prompt_tokens") == r["prompt_tokens"] and q.get("prompt_tokens_reported", 0) > 0:
            vals.append(q["prompt_tokens_reported"])
    return sum(vals) / len(vals) if vals else None


def report(results: Path, out: Path, manifest: dict) -> str:
    if not results.exists():
        return "no results (every engine failed; see logs/)\n"
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if l.strip()]
    lines = ["# Engine comparison", "",
             f"Created {manifest['created']} on {manifest['host']['cpu']} ({manifest['host']['os']}); "
             f"threads {manifest['threads']}, context {manifest['ctx_per_user']} per user, "
             f"requests per point {manifest['requests'] or 'default'}, warm-up requests {manifest['warmup_requests']}, "
             f"Ollama template {'raw prompt' if manifest['ollama_raw_template'] else 'chat'}.", "",
             "| Engine | Executable | Version | SHA-256 |", "|---|---|---|---|"]
    for k, e in manifest["engines"].items():
        lines.append(f"| {k} | `{e.get('path')}` | {e.get('version', '-')} | {(e.get('sha256') or '-')[:16]} |")
    lines += ["", "| Model | Size | SHA-256 |", "|---|---|---|"]
    for m in manifest["models"]:
        lines.append(f"| {Path(m['path']).name} | {m.get('bytes', 0) / 2**30:.2f} GiB | {(m.get('sha256') or '-')[:16]} |")
    lines += ["", f"Rows marked *no* under Comparable produced less than {COMPARABLE_FRACTION:.0%} of the requested "
              "output tokens (or had errors): their throughput is not comparable and they are not ranked.", "",
              "*Prompt seen* is the mean prompt length the engine reported (usage.prompt_tokens); it differs from "
              "*Prompt* when the engine adds a template.", "",
              "| Model | Engine | Users | Prompt | Prompt seen | Target out | Mean out | Comparable | Output tok/s | Rank | TTFT p50 (ms) | ITL p50 (ms) | ITL p99 (ms) | TPOT p50 (ms) | E2E p99 (ms) | Errors |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows.sort(key=lambda r: (r["model_file"], r["concurrency"], r["prompt_tokens"], r["label"]))
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["model_file"], r["concurrency"], r["prompt_tokens"], r["output_tokens"]), []).append(r)
    for rs in groups.values():
        ranked = sorted([r for r in rs if comparable(r)], key=lambda r: -r["output_tok_s"])
        rank = {id(r): i + 1 for i, r in enumerate(ranked)}
        for r in rs:
            seen = prompt_seen(out, r)
            lines.append(f"| {r['model_file']} | {r['label']} | {r['concurrency']} | {r['prompt_tokens']} | "
                         f"{'-' if seen is None else f'{seen:.0f}'} | {r['output_tokens']} | "
                         f"{r['mean_completion_tokens']:.0f} | {'yes' if comparable(r) else 'no'} | {r['output_tok_s']:.2f} | "
                         f"{rank.get(id(r), '-')} | {r['ttft_ms']['p50']:.0f} | {r['itl_ms']['p50']:.1f} | {r['itl_ms']['p99']:.1f} | "
                         f"{r['tpot_ms']['p50']:.1f} | {r['e2e_ms']['p99']:.0f} | {r['errors']} |")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynalm", required=True, help="the DynaLM executable under test (also runs the load generator)")
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server")
    ap.add_argument("--ollama", default=shutil.which("ollama") or "ollama")
    ap.add_argument("--model", action="append", required=True, help="FILE[:CONC:PROMPTS:OUTPUTS]")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ctx", type=int, default=2048, help="context per user")
    ap.add_argument("--requests", type=int, default=0, help="requests per point (0 = benchmark default)")
    ap.add_argument("--warmup-requests", type=int, default=0, help="uncounted full-size requests before each point")
    ap.add_argument("--dynalm-port", type=int, default=8181)
    ap.add_argument("--llama-port", type=int, default=8182)
    ap.add_argument("--ollama-port", type=int, default=11435, help="port of the script's own `ollama serve`")
    ap.add_argument("--ollama-chat-template", action="store_true",
                    help="keep Ollama's chat template (longer prompts than the other engines; default: raw prompt)")
    ap.add_argument("--dynalm-scrape-metrics", action="store_true",
                    help="save DynaLM's /metrics to <out> after its benchmark, before stopping it (default: off)")
    ap.add_argument("--skip", action="append", default=[], choices=["dynalm", "llama.cpp", "ollama"])
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if (out / "results.jsonl").exists():
        print(f"refusing to append to existing results in {out}; pick a new --out", file=sys.stderr)
        return 1
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    dyn = Path(args.dynalm)
    if not dyn.is_file():
        print(f"--dynalm {args.dynalm} does not exist", file=sys.stderr)
        return 1
    args.dynalm = str(dyn.resolve())

    engines = [("dynalm", bench_dynalm), ("llama.cpp", bench_llamacpp), ("ollama", bench_ollama)]
    info = {"dynalm": exe_info(args.dynalm, ["--version"])}
    if "llama.cpp" not in args.skip:
        info["llama.cpp"] = exe_info(args.llama_server, ["--version"])
    if "ollama" not in args.skip:
        info["ollama"] = exe_info(args.ollama, ["--version"])
    specs = [parse_spec(s) for s in args.model]
    models = []
    for model, *_ in specs:
        if not model.is_file():
            print(f"model not found: {model}", file=sys.stderr)
            return 1
        models.append({"path": str(model), "bytes": model.stat().st_size, "sha256": sha256(model)})
    minfo = {m["path"]: m for m in models}
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
                "host": {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
                         "python": sys.version.split()[0]},
                "threads": args.threads, "ctx_per_user": args.ctx, "requests": args.requests,
                "warmup_requests": args.warmup_requests, "ollama_raw_template": not args.ollama_chat_template,
                "comparable_fraction": COMPARABLE_FRACTION, "engines": info, "models": models}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    d = info["dynalm"]
    print(f"DynaLM under test: {d['path']}  sha256 {d.get('sha256')}  ({d.get('version')})", flush=True)
    for model, conc, prompts, outputs in specs:
        for name, fn in engines:
            if name in args.skip:
                continue
            print(f"== {model.name}: {name}", flush=True)
            try:
                fn(args, model, conc, prompts, outputs, results, logs, info, minfo[str(model)])
            except Exception as e:  # keep going with the other engines
                print(f"  {name} failed: {e}", flush=True)
    print(report(results, out, manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())

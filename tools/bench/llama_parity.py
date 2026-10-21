#!/usr/bin/env python3
"""Numerical parity of DynaLM with llama.cpp: teacher-forced next-token agreement.

  python3 tools/bench/llama_parity.py --out DIR --text wiki.test.raw \
      [--model FILE ...] [--paths cpu,metal,exact] [--ctx 512] [--chunks 20] [--threads 8] \
      [--build build/macos-metal] [--llama-bin test/.deps/llama.cpp/build/bin] [--label portable] \
      [--greedy] [--floor] [--no-repeat]

Per model and path, both engines score the same token ids, as llama-perplexity does: the text
cut into chunks of --ctx tokens (BOS first when the model uses one), the second half of each
chunk scored.

  1. llama.cpp: llama-perplexity --kl-divergence-base writes its token ids and next-token
     log-probabilities (CPU: -ngl 0 -dev none; Metal: -ngl 99).
  2. DynaLM: bench_ppl with BENCH_KLD_REF scores those token ids (teacher-forced; its own
     tokenization of the text is only compared with them) and compares each position with
     llama.cpp's log-probabilities (BENCH_DEVICE: ggml-cpu, metal or native).

Method for the CPU paths (cpu, exact), the same on both sides and checked in the run logs:
one sequence (llama.cpp -b <ctx>: n_seq = 1), ubatches of 16 tokens (llama.cpp -ub 16, DynaLM
BENCH_UBATCH=16), a fresh KV cache per chunk (llama-perplexity clears its cache; bench_ppl makes a
new pool, BENCH_FRESH_KV), flash attention, f16 KV, the same thread count. Compare builds with the
same CPU features (--build and --llama-bin: both portable, or both native): ggml-cpu picks its
repacked kernels by the compiled CPU features (Q4_K: 8x8 with i8mm, 8x4 without), and the kernel
names both engines log are compared (DYNACORE_GPU_LOG=1; llama.cpp -lv 5). A row whose logged
setup differs is reported as a setup mismatch, not as a parity result.

Paths: cpu = DynaLM's fast CPU (ggml-cpu) against llama.cpp on the CPU with the same ggml-cpu
kernels (ubatch 16, below ggml-blas's threshold); cpu-blas = the same DynaLM run against llama.cpp's
default CPU setup (Accelerate BLAS for prompt batches: different arithmetic, reported); metal =
DynaLM on Metal against llama.cpp on Metal; exact = DynaLM's --mode exact CPU kernels against
llama.cpp's CPU (reported; not the release CPU path). Flash attention and an f16 KV cache in both.

Reported per model and path (DIR/parity.tsv, DIR/summary.md): top-1 agreement (target >= 99%
per model and path), top-5 overlap, KL(llama.cpp || DynaLM) mean and max (as llama-perplexity), max |delta p| of the
token that occurred, max |delta log p| of llama.cpp's top-1 token, both perplexities (llama.cpp's
from its own run), the
disagreements and how many are near-ties (llama.cpp's top-2 gap < 0.1 nats), the first
divergence (chunk, position, both tokens, llama.cpp's top-2 gap there) and whether DynaLM's
tokenizer gave the same ids.

What this measures: implementation agreement with llama.cpp on the same model file (numerical and
next-token parity), not task quality or factual accuracy; two engines can agree perfectly on a model
that answers wrongly.

Backend check: each path names the backend it needs (WANT_BACKEND: metal = gpu, the others = cpu),
and both engines' logs must show it (llama.cpp: the device every layer was assigned to; DynaLM: the
device line of bench_ppl). Otherwise the row is SETUP MISMATCH, never PASS or FAIL, and the run exits
with status 2 (also for any error row). Example: --paths metal with a llama.cpp build without Metal
runs llama.cpp on the CPU.

Repeatability (default; --no-repeat skips): DynaLM run again against its own first file, and
llama.cpp run again against its own reference, same settings.

--floor: llama.cpp against itself, its reference run repeated with ubatch 1 (gemv kernels
instead of gemm) on the same token ids ("llama.cpp vs itself"). On the CPU, ggml's 8-bit
activations make results depend on the kernel chosen by batch size: Qwen2.5-0.5B Q4_K_M, 2 chunks,
llama.cpp ubatch 1 vs 16 agreed on 97.06% of top-1 tokens (KLD 0.0023).

Cross-check: llama-perplexity --kl-divergence on DynaLM's file (llama.cpp re-runs on DynaLM's
token ids and prints its own "Same top p" and KL), in DIR/logs/<model>-<path>-llama-kld.log.

--greedy: free-running greedy text (separate metric) from `dynalm serve` and `llama-server`,
/v1/completions, temperature 0, raw prompt, the same prompts; reported as identical or as the
common prefix length in llama.cpp tokens.

No downloads, no benchmarks: model files must exist. Logs: DIR/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# path -> (DynaLM BENCH_DEVICE, llama.cpp run (its reference file is shared by paths with the same
# run), llama.cpp flags, DynaLM serve flags). llama.cpp's CPU reference runs ubatches of 16 tokens:
# below ggml-blas's 32-row minimum, so its matmuls use the same ggml-cpu kernels as DynaLM's fast
# CPU (this llama.cpp build has GGML_BLAS=ON: Accelerate's fp32 SGEMM for larger batches). cpu-blas
# compares with llama.cpp's default CPU setup instead (reported; different arithmetic).
CPU = ["-ngl", "0", "-dev", "none"]
PATHS = {
    "cpu": ("ggml-cpu", "cpu", CPU + ["-ub", "16"], ["--backend", "cpu"]),
    "cpu-blas": ("ggml-cpu", "cpu-blas", CPU, ["--backend", "cpu"]),
    "metal": ("metal", "metal", ["-ngl", "99"], ["--backend", "metal"]),
    "exact": ("native", "cpu", CPU + ["-ub", "16"], ["--backend", "cpu", "--mode", "exact"]),
}
# Paths scored with the matched CPU method: one sequence, ubatch 16 on both sides.
MATCHED = {"cpu": 16, "exact": 16}
# The backend each path must have run on, in both engines, as their logs show it (backends()).
WANT_BACKEND = {"cpu": "cpu", "cpu-blas": "cpu", "metal": "gpu", "exact": "cpu"}
GREEDY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which",
    "Q: What is 17 times 23?\nA:",
]


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd, log: Path, env=None, timeout=7200) -> int:
    with open(log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, **(env or {})), timeout=timeout).returncode


def same_top_p(log: Path) -> str:
    t = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(r"Same top p\s*:\s*([0-9.]+\s*±\s*[0-9.]+\s*%)", t)
    k = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+\s*±\s*[0-9.]+)", t)
    return f"{m.group(1) if m else '?'} (KLD {k.group(1) if k else '?'})"


def build_flags(build_dir: Path) -> dict:
    """ggml's CPU build options from a CMake cache (what decides the compiled kernels)."""
    c = build_dir / "CMakeCache.txt"
    t = c.read_text(errors="replace") if c.exists() else ""
    return {k: (m.group(1) if (m := re.search(rf"^{k}:BOOL=(\w+)", t, re.M)) else "?")
            for k in ("GGML_NATIVE", "GGML_LLAMAFILE", "GGML_BLAS", "GGML_CPU_REPACK")}


def kernels(text: str) -> dict:
    """Repacked weight kernels ggml-cpu logged ("repack tensor X with q4_K_8x8"), counted."""
    out: dict = {}
    for k in re.findall(r"repack tensor\s+\S*\s*with (\w+)", text):  # DynaLM's graph tensors have no names
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def backends(lt: str, dt: str) -> tuple[str, str, str]:
    """Where each engine actually ran, from its own log: "gpu", "cpu" or "?" (not found).

    llama.cpp (-lv 5): the device of every layer ("layer N assigned to device MTL0" / "CPU");
    any non-CPU device = gpu. DynaLM (bench_ppl): its "model: ..., device <name>," line;
    Metal / Vulkan / CUDA = gpu. Also returns llama.cpp's devices, for the report.
    """
    devs = sorted(set(re.findall(r"layer\s+\d+ assigned to device (\S+?),", lt)))
    llama = "?" if not devs else ("gpu" if any(not d.startswith("CPU") for d in devs) else "cpu")
    m = re.search(r"^model: .*?, device (.+?), \d+ tokens", dt, re.M)
    dyn = "?" if not m else ("gpu" if re.search(r"Metal|Vulkan|CUDA", m.group(1)) else "cpu")
    return llama, dyn, ",".join(devs) or "?"


def setup(llog: Path, dlog: Path, ub: int | None, want: str | None = None) -> dict:
    """The settings both runs actually used, from their logs; ok = they match the method.

    want ("cpu" / "gpu"): the backend the path requires. A run on another backend, or one whose
    log does not show where it ran, is a setup mismatch (never a parity result): e.g. a Metal
    path given a llama.cpp build without Metal silently runs llama.cpp on the CPU."""
    lt = llog.read_text(errors="replace") if llog.exists() else ""
    dt = dlog.read_text(errors="replace") if dlog.exists() else ""
    g = lambda pat, t: (m.group(1).strip() if (m := re.search(pat, t, re.M)) else "?")  # noqa: E731
    seqs = re.findall(r"n_seq=(\d+)", lt)
    r = {"llama_n_seq": seqs[-1] if seqs else "?", "llama_n_ubatch": g(r"n_ubatch\s*=\s*(\d+)", lt),
         "llama_cpu": g(r"CPU : (.*?)\s*\|?\s*$", lt), "llama_flash_attn": g(r"flash_attn\s*=\s*(\w+)", lt),
         "dynalm_ubatch": g(r"^scoring: ubatch (\d+)", dt), "dynalm_kv": g(r"^scoring: ubatch \d+, KV (.*)$", dt),
         "kernels_llama": kernels(lt), "kernels_dynalm": kernels(dt)}
    r["llama_backend"], r["dynalm_backend"], r["llama_devices"] = backends(lt, dt)
    problems = []
    if want is not None:
        for eng, got in (("llama.cpp", r["llama_backend"]), ("DynaLM", r["dynalm_backend"])):
            if got != want:
                problems.append(f"{eng} ran on {got if got != '?' else 'an unknown backend (not in its log)'}, "
                                f"path needs {want}" + (f" (devices {r['llama_devices']})" if eng == "llama.cpp" else ""))
    if ub is not None:
        if r["llama_n_seq"] != "1":
            problems.append(f"llama.cpp n_seq {r['llama_n_seq']}")
        if r["llama_n_ubatch"] != str(ub) or r["dynalm_ubatch"] != str(ub):
            problems.append(f"ubatch llama.cpp {r['llama_n_ubatch']} / DynaLM {r['dynalm_ubatch']}")
        if not r["dynalm_kv"].startswith("fresh"):
            problems.append(f"DynaLM KV {r['dynalm_kv']}")
        if r["kernels_llama"] != r["kernels_dynalm"]:
            problems.append("repacked kernels differ")
    r["setup_ok"] = not problems
    r["setup_problems"] = "; ".join(problems)
    return r


def parity(text: str) -> dict | None:
    m = re.search(r"^PARITY (\{.*\})$", text, re.M)
    return json.loads(m.group(1)) if m else None


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def post(url: str, body: dict, timeout=600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(url: str, proc, timeout=300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited ({proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become ready")


def greedy(args, model: Path, path: str, logs: Path) -> list[dict]:
    """Free-running greedy completions from both servers (one at a time)."""
    out = {}
    for eng, port in (("dynalm", args.dynalm_port), ("llama", args.llama_port)):
        if not port_free(port):
            raise RuntimeError(f"port {port} in use; not stopping it")
        if eng == "dynalm":
            cmd = [args.dynalm, "serve", model, "--port", port, "--ctx", "2048", *PATHS[path][3]]
        else:
            cmd = [args.llama_bin / "llama-server", "-m", model, "--port", port, "-c", "2048", "-np", "1",
                   "-fa", "on", "-t", args.threads, "--no-webui", *PATHS[path][2]]
        log = open(logs / f"{model.stem}-{path}-greedy-{eng}.log", "w")
        proc = subprocess.Popen(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_health(f"http://127.0.0.1:{port}/health", proc)
            out[eng] = [post(f"http://127.0.0.1:{port}/v1/completions",
                             {"prompt": p, "max_tokens": args.greedy_tokens, "temperature": 0, "top_k": 1,
                              "seed": 0, "stream": False})["choices"][0]["text"] for p in GREEDY_PROMPTS]
            if eng == "llama":  # llama.cpp tokens of both outputs, for the common prefix
                tok = lambda s: post(f"http://127.0.0.1:{port}/tokenize", {"content": s, "add_special": False})["tokens"]  # noqa: E731
                res = []
                for p, a, b in zip(GREEDY_PROMPTS, out["dynalm"], out["llama"]):
                    ta, tb = tok(a), tok(b)
                    k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                    res.append({"prompt": p, "identical": a == b, "common_prefix_tokens": k,
                                "llama_tokens": len(tb), "dynalm": a, "llama": b})
                return res
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", required=True, help="text file (wikitext-2: test/.deps/llama.cpp/scripts/get-wikitext-2.sh)")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--paths", default="cpu,metal,cpu-blas,exact")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--build", default=str(ROOT / "build/macos-metal"))
    ap.add_argument("--dynalm", default="", help="dynalm binary for --greedy (default: <build>/bin/dynalm)")
    ap.add_argument("--llama-bin", default=str(ROOT / "test/.deps/llama.cpp/build/bin"))
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--greedy-tokens", type=int, default=48)
    ap.add_argument("--dynalm-port", type=int, default=8191)
    ap.add_argument("--llama-port", type=int, default=8192)
    ap.add_argument("--target", type=float, default=99.0, help="top-1 target, metal and cpu-blas paths")
    ap.add_argument("--cpu-target", type=float, default=99.9, help="top-1 target, matched CPU paths (proposed)")
    ap.add_argument("--label", default="", help="name of this build pairing (e.g. portable, native)")
    ap.add_argument("--no-repeat", action="store_true", help="skip the repeatability runs")
    ap.add_argument("--floor", action="store_true", help="also run llama.cpp ubatch 1 vs its reference")
    a = ap.parse_args()
    a.llama_bin = Path(a.llama_bin)
    a.dynalm = a.dynalm or str(Path(a.build) / "bin/dynalm")
    out = Path(a.out)
    logs = out / "logs"
    kld = out / "kld"
    logs.mkdir(parents=True, exist_ok=True)
    kld.mkdir(exist_ok=True)
    bench_ppl = Path(a.build) / "dynalm/benchmarks/bench_ppl"
    lp = a.llama_bin / "llama-perplexity"
    for exe in (bench_ppl, lp):
        if not os.access(exe, os.X_OK):
            sys.exit(f"missing {exe}")
    models = [Path(m) for m in a.model] or sorted((Path.home() / ".dynalm/models").glob("*.gguf"))
    paths = [p for p in a.paths.split(",") if p]
    for p in paths:
        if p not in PATHS:
            sys.exit(f"unknown path {p} ({', '.join(PATHS)})")
    with open(out / "manifest.txt", "w") as f:
        git = subprocess.run(["git", "-C", ROOT, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--short"], capture_output=True, text=True).stdout
        llama = subprocess.run(["git", "-C", a.llama_bin.parents[1], "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
        flags = {"dynalm": build_flags(Path(a.build)), "llama.cpp": build_flags(a.llama_bin.parent)}
        f.write(f"label: {a.label}\nbuild flags: {json.dumps(flags)}\n")
        f.write(f"dynalm git: {git}\n{dirty}\nllama.cpp git: {llama}\nbench_ppl: {bench_ppl}\nllama-perplexity: {lp}\n"
                f"text: {a.text}\nctx {a.ctx}, chunks {a.chunks}, threads {a.threads}, paths {paths}\n"
                f"models:\n" + "".join(f"  {m}\n" for m in models))
    rows = []
    greedy_rows = []
    for m in models:
        for p in paths:
            dev, lrun, lflags, _ = PATHS[p]
            ub = MATCHED.get(p)
            if ub is not None:  # one sequence: llama-perplexity runs n_batch / n_ctx sequences at once
                lflags = lflags + ["-b", str(a.ctx)]
            tag = f"{m.stem}-{p}"
            row = {"model": m.name, "path": p, "target": a.cpu_target if ub is not None else a.target}
            lk = kld / f"{m.stem}-{lrun}-llama.kld"
            if not lk.exists():  # llama.cpp's tokens and log-probs (shared by paths with the same run)
                say(f"{tag}: llama.cpp ({' '.join(lflags)}) ...")
                rc = run([lp, "-m", m, "-f", a.text, "-c", a.ctx, "--chunks", a.chunks, "-fa", "on", "-t", a.threads,
                          *lflags, "-lv", "5", "--kl-divergence-base", lk], logs / f"{lk.stem}.log")
                if rc != 0:
                    lk.unlink(missing_ok=True)
                    row["error"] = f"llama-perplexity rc={rc} (logs/{lk.stem}.log)"
                    rows.append(row)
                    say(f"{tag}: {row['error']}")
                    continue
            say(f"{tag}: DynaLM ({dev}) ...")
            dk = kld / f"{tag}-dynalm.kld"
            dlog = logs / f"{tag}-dynalm.log"
            denv = {"BENCH_DEVICE": dev, "BENCH_KLD_REF": str(lk), "BENCH_KLD_BASE": str(dk), "BENCH_FRESH_KV": "1",
                    "DYNACORE_GPU_LOG": "1", **({"BENCH_UBATCH": str(ub)} if ub is not None else {})}
            rc = run([bench_ppl, m, a.text, a.ctx, a.chunks, a.threads], dlog, denv)
            par = parity(dlog.read_text(errors="replace"))
            if rc != 0 or not par:
                row["error"] = f"bench_ppl rc={rc} (logs/{dlog.name})"
                rows.append(row)
                say(f"{tag}: {row['error']}")
                continue
            row.update(par)
            row.update(setup(logs / f"{lk.stem}.log", dlog, ub, WANT_BACKEND[p]))
            # llama.cpp's perplexity from its own run (the 16-bit file would bias a recomputation).
            fe = re.search(r"Final estimate: PPL = ([0-9.]+)", (logs / f"{lk.stem}.log").read_text(errors="replace"))
            row["ppl_llama"] = float(fe.group(1)) if fe else float("nan")
            klog = logs / f"{tag}-llama-kld.log"  # cross-check: llama.cpp's own comparison on DynaLM's file
            run([lp, "-m", m, "-c", a.ctx, "-fa", "on", "-t", a.threads, *lflags,
                 "--kl-divergence-base", dk, "--kl-divergence"], klog)
            row["llama_same_top_p"] = same_top_p(klog)
            # llama.cpp against itself: its reference run repeated with ubatch 1 (other kernels:
            # gemv instead of gemm), on the same token ids. DynaLM cannot be expected to agree with
            # llama.cpp more closely than llama.cpp agrees with itself across batch sizes.
            if a.floor:
                flog = logs / f"{m.stem}-{lrun}-llama-self-ub1.log"
                if not flog.exists():
                    run([lp, "-m", m, "-c", a.ctx, "-fa", "on", "-t", a.threads,
                         *[x for x in lflags if x not in ("-ub", "16")], "-ub", "1",
                         "--kl-divergence-base", lk, "--kl-divergence"], flog)
                row["llama_self_ub1"] = same_top_p(flog)
            if not a.no_repeat:  # the same runs again: DynaLM against its first file, llama.cpp against its own
                rlog = logs / f"{tag}-dynalm-repeat.log"
                run([bench_ppl, m, a.text, a.ctx, a.chunks, a.threads], rlog,
                    {k: v for k, v in denv.items() if k != "BENCH_KLD_BASE"} | {"BENCH_KLD_REF": str(dk)})
                rp = parity(rlog.read_text(errors="replace"))
                row["dynalm_repeat"] = f"{rp['top1']:.3f} % (KLD {rp['kl_mean']:.7f})" if rp else "error"
                llrep = logs / f"{m.stem}-{lrun}-llama-repeat.log"
                if not llrep.exists():
                    run([lp, "-m", m, "-c", a.ctx, "-fa", "on", "-t", a.threads, *lflags,
                         "--kl-divergence-base", lk, "--kl-divergence"], llrep)
                row["llama_repeat"] = same_top_p(llrep)
            rows.append(row)
            say(f"{tag}: top-1 {row['top1']:.2f}% over {row['positions']}, KL {row['kl_mean']:.6f}, PPL {row['ppl_dynalm']:.4f} "
                f"vs {row['ppl_llama']:.4f}, tokenizer {'same' if row['tokenizer_first_mismatch'] < 0 else 'DIFFERS'}; "
                f"llama.cpp check: {row['llama_same_top_p']}; setup {'ok' if row['setup_ok'] else row['setup_problems']}")
            if a.greedy:
                try:
                    g = greedy(a, m, p, logs)
                    greedy_rows.append({"model": m.name, "path": p, "results": g})
                except Exception as ex:  # reported, not fatal
                    greedy_rows.append({"model": m.name, "path": p, "error": str(ex)})
    (out / "parity.json").write_text(json.dumps({"rows": rows, "greedy": greedy_rows}, indent=1))
    cols = ["model", "path", "positions", "top1", "top5_overlap", "kl_mean", "kl_max", "dp_max", "dlogp_top1_max",
            "ppl_dynalm", "ppl_llama", "disagreements", "near_ties", "median_margin", "first_div",
            "tokenizer_first_mismatch", "llama_same_top_p", "llama_self_ub1", "dynalm_repeat", "llama_repeat",
            "setup_ok", "setup_problems", "llama_backend", "dynalm_backend", "llama_devices", "llama_n_seq", "llama_n_ubatch", "dynalm_ubatch", "dynalm_kv",
            "llama_flash_attn", "llama_cpu", "kernels_dynalm", "kernels_llama", "error"]
    with open(out / "parity.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    L = ["# DynaLM vs llama.cpp: teacher-forced next-token parity" + (f" ({a.label})" if a.label else ""), "",
         f"Build flags: DynaLM {build_flags(Path(a.build))}; llama.cpp {build_flags(a.llama_bin.parent)}.", "",
         f"Text `{a.text}`, {a.chunks} chunks of {a.ctx} tokens, second half of each chunk scored (as llama-perplexity). "
         f"Target: top-1 agreement >= {a.cpu_target}% (matched CPU paths, proposed) or >= {a.target}% (others) per model and path. KL = KL(llama.cpp || DynaLM), from both files' "
         "16-bit log-probabilities. Margin = llama.cpp's log-prob gap between its two leading tokens at a disagreement.", "",
         "These numbers measure implementation agreement with llama.cpp on the same model file (numerical "
         "and next-token parity). They do not measure task quality or factual accuracy: two engines can agree "
         "perfectly on a model that answers wrongly. A row is a result only when both logs show the backend its "
         "path requires (\"Setup\"); otherwise it is SETUP MISMATCH and the run exits with status 2.", "",
         "| Model | Path | Positions | Top-1 | Target | Top-5 overlap | Mean KL | Max KL | Max Δp | PPL DynaLM | PPL llama.cpp | "
         "Disagreements (near-ties) | First divergence | Tokenizer | llama.cpp's own check | DynaLM repeat | llama.cpp repeat | "
         "Setup (from logs) | llama.cpp vs itself (ubatch 1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['model']} | {r['path']} | – | – | – | – | – | – | – | – | – | – | {r['error']} | – | – | – | – | – | – |")
            continue
        fd = r["first_div"]
        fds = "none" if fd["chunk"] < 0 else f"chunk {fd['chunk']} pos {fd['pos']}: {fd['llama']} vs {fd['dynalm']} (gap {fd['margin']:.4f})"
        tk = "same" if r["tokenizer_first_mismatch"] < 0 else f"differs at {r['tokenizer_first_mismatch']}"
        su = ("ok" if r["setup_ok"] else "MISMATCH: " + r["setup_problems"]) + \
            f" (ran on llama.cpp {r['llama_backend']} [{r['llama_devices']}] / DynaLM {r['dynalm_backend']}, " + \
            f"n_seq {r['llama_n_seq']}, ub {r['llama_n_ubatch']}/{r['dynalm_ubatch']}, KV {r['dynalm_kv']}, " + \
            ", ".join(f"{k} x{v}" for k, v in r["kernels_dynalm"].items()) + ")"
        verdict = "SETUP MISMATCH" if not r["setup_ok"] else ("PASS" if r["top1"] >= r["target"] else "FAIL")
        L.append(f"| {r['model']} | {r['path']} | {r['positions']} | {r['top1']:.2f}% | "
                 f"{verdict} (>= {r['target']}) | {r['top5_overlap']:.2f}% | {r['kl_mean']:.6f} | "
                 f"{r['kl_max']:.4f} | {r['dp_max']:.4f} | {r['ppl_dynalm']:.4f} | {r['ppl_llama']:.4f} | "
                 f"{r['disagreements']} ({r['near_ties']}) | {fds} | {tk} | {r['llama_same_top_p']} | "
                 f"{r.get('dynalm_repeat', '–')} | {r.get('llama_repeat', '–')} | {su} | {r.get('llama_self_ub1', '–')} |")
    if greedy_rows:
        L += ["", "## Free-running greedy (separate metric)", "",
              f"{len(GREEDY_PROMPTS)} prompts, {a.greedy_tokens} tokens, temperature 0, /v1/completions raw prompt.", "",
              "| Model | Path | Identical | Common prefix (llama.cpp tokens) |", "|---|---|---|---|"]
        for g in greedy_rows:
            if "error" in g:
                L.append(f"| {g['model']} | {g['path']} | error: {g['error']} | – |")
                continue
            L.append(f"| {g['model']} | {g['path']} | {sum(x['identical'] for x in g['results'])}/{len(g['results'])} | "
                     + ", ".join(f"{x['common_prefix_tokens']}/{x['llama_tokens']}" for x in g["results"]) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    say(f"done: {out / 'summary.md'}")
    bad = [f"{r['model']} {r['path']}" for r in rows if "error" in r or not r.get("setup_ok", False)]
    if bad:  # never a silent pass: these rows are not parity results
        say(f"INVALID (setup mismatch or error, see summary): {', '.join(bad)}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

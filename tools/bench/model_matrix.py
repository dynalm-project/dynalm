#!/usr/bin/env python3
"""Model matrix: does every model in a list load and generate on every device
path, do the paths agree, and (with --bench) how fast is each against
llama.cpp on the same file?

  DYNALM=build/macos-metal/bin/dynalm python3 tools/bench/model_matrix.py --out DIR \\
    [--list FILE] [--check-only] [--accuracy] [--bench] [--keep] [--max-gb 16] [--tokens 32]
  LLAMA_BENCH=<llama-bench>   with --bench: llama.cpp on the same GGUF (GPU and CPU)
  HF_TOKEN=<token>            gated repos (meta-llama, google)

List entries (one per line, '#' comments): a registry name, a local .gguf file
or model directory, a Hugging Face GGUF file (<owner>/<repo>/<file>.gguf or a
link), or hf:<owner>/<repo> for a safetensors model (config.json, tokenizer and
every weight shard are fetched). Default list: tools/bench/models-matrix.txt;
a 50-model list: tools/bench/models-50.txt.

Per model:
  1. check     GGUF: `dynalm pull --check` (256 KiB: architecture, and the
               tokenizer when its keys are in the header); safetensors: the HF
               file list and model_type, then only the small files (config,
               tokenizer.json) through `dynalm inspect` (architecture and tokenizer,
               by the loader's code), so an unsupported tokenizer costs no weight
               download. Download size (HTTP HEAD / HF API) against --max-gb.
               --check-only stops here (~2 s each).
  2. fetch     `dynalm pull`, or the repo files for hf:.
  3. generate  a fixed chat prompt, greedy, N tokens, with --backend auto (the
               GPU when it runs the model), cpu (ggml's fast CPU kernels) and
               exact (`--backend cpu --mode exact`: DynaCore's kernels, DD-053,
               the reference): device, success, tok/s, and whether each text
               equals exact's (the TSV keeps the column names native*)
               (reported, not gated: GPU and 8-bit kernels may choose
               differently at a near-tie).
  4. accuracy  (--accuracy, GGUF on a GPU) bench_gpu_accuracy: DD-053 PASS/FAIL.
  5. bench     (--bench; --bench-paths, default gpu,cpu,exact) `dynalm benchmark`
               in-process on the GPU (if auto used one), cpu (ggml's kernels) and
               exact (`--mode exact`, DynaLM's DD-053 kernels; compared with
               llama.cpp's CPU too), and llama-bench on the same GGUF, measured
               alike (BENCH_METHOD, recorded in the bench_method column):
                 - one user: a 512-token prompt, then 64 tokens, so decode runs
                   at depth 512-575 in both engines (llama-bench -d 512 -n 64);
                 - BENCH_REPS (5) measured repetitions after a warm-up in both
                   (DynaLM: --warmup-requests 1 and 5 requests; llama-bench: its
                   warm-up run and -r 5); each rate is the mean of the
                   repetitions, its standard deviation in the *_sd columns;
                 - DynaLM per request: prompt tok/s = prompt / TTFT, decode
                   tok/s = (tokens - 1) / (E2E - TTFT) (--request-log);
                 - flash attention on in both (-fa on: DynaLM's graphs always
                   use it); llama-bench's CPU run is -ngl 0 -nopo 1 (op offload
                   off: otherwise its prompt runs on the GPU); THREADS threads
                   (default: the performance cores, as DynaLM's ggml CPU path);
                 - 4 users: output tok/s (DynaLM only).
               AC power and load are recorded with each. A screen for outliers
               and failures; tools/bench/baseline.sh is the server comparison.
Downloads are deleted after their row unless --keep, so the sweep needs only
the largest model's disk space. Time limits per step: RUN_TIMEOUT (default
300 s) for generation; 1200 s for accuracy and bench; 7200 s for downloads.
Output: DIR/results.tsv, DIR/summary.md, DIR/logs/<n>-<step>.*
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The runtime paths measured, as dynalm options. "exact" is DynaCore's CPU kernels (DD-053), the
# reference the other texts are compared with; there is no public native-cpu backend in this release.
PATHS = {"auto": ["--backend", "auto"], "cpu": ["--backend", "cpu"], "exact": ["--backend", "cpu", "--mode", "exact"]}
PROMPT = "Write one sentence about the ocean."
COLS = ["n", "model", "format", "supported", "size_gb", "file", "arch", "quant",
        "auto_device", "auto", "auto_tok_s", "cpu_device", "cpu", "cpu_tok_s", "native", "native_tok_s",
        "auto_matches_native", "cpu_matches_native", "accuracy",
        "gpu_prompt_tok_s", "gpu_decode_tok_s", "gpu_4user_tok_s",
        "cpu_prompt_tok_s", "cpu_decode_tok_s", "cpu_4user_tok_s",
        "llama_gpu_pp512", "llama_gpu_tg128", "llama_cpu_pp512", "llama_cpu_tg128", "bench_conditions", "seconds",
        # Added after the first release of this file: appended, so column numbers above stay
        # (tools/bench/e2e_watch.sh reads them by position).
        "exact_prompt_tok_s", "exact_decode_tok_s", "exact_4user_tok_s",
        # Since bench_method "d512-r5": llama_*_pp512 is pp512 at depth 0 and llama_*_tg128 is
        # tg64 at depth 512 (names kept, so older results still render; an empty bench_method
        # means the old pp512 / tg128 at depth 0, -r 2). *_sd: standard deviation of the
        # repetitions, tok/s. llama_*_tg_d0: tg64 at depth 0 (the depth effect).
        "bench_method", "gpu_prompt_sd", "gpu_decode_sd", "cpu_prompt_sd", "cpu_decode_sd",
        "exact_prompt_sd", "exact_decode_sd", "llama_gpu_pp_sd", "llama_gpu_tg_sd", "llama_cpu_pp_sd",
        "llama_cpu_tg_sd", "llama_gpu_tg_d0", "llama_cpu_tg_d0"]
# The speed method (docstring, step 5). Change BENCH_METHOD with it.
BENCH_METHOD = "d512-r5"
BENCH_PROMPT, BENCH_GEN, BENCH_REPS = 512, 64, 5
# Hugging Face model_type values the safetensors loader accepts (keep in sync
# with dynalm/src/loader/hf/hf_model.cpp; the load step is the final word).
HF_MODEL_TYPES = {"llama", "mistral", "mixtral", "qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "gemma", "gemma2",
                  "gemma3_text", "phi3", "granite", "granitemoe"}
HF_FILES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt", "model.safetensors.index.json"}


def run(cmd: list[str], log: Path, timeout: int, stdout: Path | None = None) -> int:
    """Runs cmd with a time limit; stderr (and stdout unless given) to log. 124 = timed out."""
    with open(log, "w") as err, open(stdout, "w") if stdout else open(log, "a") as out:
        try:
            return subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124
        except OSError as e:
            err.write(f"cannot run {cmd[0]}: {e}\n")
            return 127


def hf_request(url: str, token: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {token}"} if token else {})
    return urllib.request.urlopen(req, timeout=60)


def gguf_url(entry: str) -> str:
    if entry.startswith("http"):
        return entry.replace("/blob/", "/resolve/")
    owner, repo, rest = entry.split("/", 2)
    return f"https://huggingface.co/{owner}/{repo}/resolve/main/{rest}"


def gguf_size_gb(entry: str, token: str) -> float | None:
    try:
        with hf_request(gguf_url(entry), token, "HEAD") as r:
            n = int(r.headers.get("Content-Length") or 0)
        return n / 2**30 if n else None
    except Exception:
        return None


def hf_files(repo: str, token: str) -> tuple[str, list[dict]]:
    try:
        with hf_request(f"https://huggingface.co/api/models/{repo}/tree/main", token) as r:
            files = json.load(r)
    except urllib.error.HTTPError as e:
        return ("gated" if e.code in (401, 403) else "link-error"), []
    except Exception:
        return "link-error", []
    want = [f for f in files if f.get("type") == "file" and (f["path"] in HF_FILES or f["path"].endswith(".safetensors"))]
    if not any(f["path"].endswith(".safetensors") for f in want):
        return "no-safetensors", want
    # Only the weights the model uses: the shards named by model.safetensors.index.json
    # when present (Mistral repos also ship a duplicate consolidated.safetensors).
    if any(f["path"] == "model.safetensors.index.json" for f in want):
        try:
            with hf_request(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json", token) as r:
                shards = set(json.load(r).get("weight_map", {}).values())
            want = [f for f in want if not f["path"].endswith(".safetensors") or f["path"] in shards]
        except urllib.error.HTTPError as e:
            return ("gated" if e.code in (401, 403) else "link-error"), want
        except Exception:
            return "link-error", want
    elif sum(f["path"].endswith(".safetensors") for f in want) > 1:
        want = [f for f in want if not f["path"].startswith("consolidated")]
    # Pre-screen by config.json's model_type (the loader's list,
    # dynalm/src/loader/hf/hf_model.cpp), so an unsupported model is not downloaded.
    try:
        with hf_request(f"https://huggingface.co/{repo}/resolve/main/config.json", token) as r:
            mt = json.load(r).get("model_type", "?")
    except urllib.error.HTTPError as e:
        return ("gated" if e.code in (401, 403) else "link-error"), want
    except Exception:
        return "link-error", want
    if mt not in HF_MODEL_TYPES:
        return f"no ({mt})", want
    return "yes", want


def hf_download(repo: str, files: list[dict], root: Path, token: str) -> tuple[str, Path | None]:
    d = root / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    try:
        for f in files:
            p = d / f["path"]
            if p.exists() and p.stat().st_size == f.get("size", -1):
                continue
            with hf_request(f"https://huggingface.co/{repo}/resolve/main/{f['path']}", token) as r, \
                    open(str(p) + ".part", "wb") as o:
                shutil.copyfileobj(r, o, 1 << 22)
            os.replace(str(p) + ".part", p)
    except urllib.error.HTTPError as e:
        return ("gated" if e.code in (401, 403) else "download-failed"), None
    except Exception:
        return "download-failed", None
    return "yes", d


def quiet(max_load: float) -> tuple[bool, str]:
    """(quiet, state): AC power, Energy Mode Automatic (macOS) and a 1-minute load below max_load."""
    load = os.getloadavg()[0]
    if platform.system() == "Darwin":
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        mode = re.findall(r"^\s*powermode\s+(\d+)", subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout, re.M)
        ac = "AC Power" in batt
        ok = ac and (not mode or mode[0] == "0") and load < max_load
        return ok, f"{'AC' if ac else 'battery'},powermode={mode[0] if mode else '?'},load={load:.1f}"
    return load < max_load, f"load={load:.1f}"


def wait_quiet(max_load: float, log: Path, tag: str) -> str:
    """Waits (logging every minute) until quiet(); returns the state it started in."""
    waited = 0
    while True:
        ok, state = quiet(max_load)
        if ok:
            with open(log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {tag} start: {state} (waited {waited}s)\n")
            return state
        if waited % 60 == 0:
            msg = f"{time.strftime('%H:%M:%S')} {tag} waiting: {state}"
            print("   " + msg, flush=True)
            with open(log, "a") as f:
                f.write(msg + "\n")
        time.sleep(10)
        waited += 10


def conditions() -> str:
    load = os.getloadavg()[0]
    if platform.system() == "Darwin":
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return f"{'AC' if 'AC Power' in batt else 'battery'},load={load:.1f}"
    return f"load={load:.1f}"


def say(tag: str, msg: str) -> None:
    """One progress line, timestamped (the live view: tools/bench/e2e_watch.sh)."""
    print(f"   {time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


class Matrix:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dynalm = a.dynalm
        self.acc = os.environ.get("ACC") or str(Path(a.dynalm).resolve().parent.parent / "dynalm/benchmarks/bench_gpu_accuracy")
        self.llama_bench = os.environ.get("LLAMA_BENCH", "")
        self.token = os.environ.get("HF_TOKEN", "")
        self.threads = os.environ.get("THREADS") or str(os.cpu_count() or 4)
        if platform.system() == "Darwin":
            p = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout.strip()
            self.threads = os.environ.get("THREADS") or p or self.threads
        self.timeout = int(os.environ.get("RUN_TIMEOUT", "300"))
        self.out = Path(a.out)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def generate(self, backend: str, model: str, tag: str) -> tuple[str, str, str, Path]:
        """('ok'|'fail'|'timeout', device, tok/s, text file)."""
        txt, log = self.logs / f"{tag}-{backend}.txt", self.logs / f"{tag}-{backend}.log"
        say(tag, f"generate: {' '.join(PATHS[backend])}, {self.a.tokens} tokens ...")
        code = run([self.dynalm, "run", model, *PATHS[backend], "-p", PROMPT, "-n", str(self.a.tokens), "--temp", "0",
                    "--ctx", "2048"], log, self.timeout, stdout=txt)
        text = log.read_text(errors="replace")
        devs = re.findall(r"Device: (?:compiled/)?([A-Za-z-]+/[^,\n]*)", text)
        tps = re.findall(r"generated \d+ tok, ([0-9.]+) tok/s", text)
        status = "timeout" if code == 124 else "ok" if code == 0 and txt.exists() and txt.stat().st_size > 0 else "fail"
        say(tag, f"generate {backend}: {status}" + (f" on {devs[-1]}, {tps[-1]} tok/s" if devs and tps else "") +
            (f" (see {log.name})" if status != "ok" else ""))
        return status, (devs[-1].replace(" ", "_") if devs else "-"), (tps[-1] if tps else "-"), txt

    def hf_precheck(self, repo: str, files: list[dict], tag: str) -> str:
        """Fetch only the small files (config, tokenizer), and let `dynalm inspect` check
        architecture and tokenizer with the loader's own code before any weights are downloaded."""
        meta = [f for f in files if not f["path"].endswith(".safetensors")]
        say(tag, f"check: tokenizer and config ({len(meta)} small files) ...")
        st, d = hf_download(repo, meta, self.out / "hf-meta", self.token)
        if d is None:
            return st
        log = self.logs / f"{tag}-precheck.log"
        run([self.dynalm, "inspect", str(d)], log, 120)
        text = log.read_text(errors="replace")
        shutil.rmtree(d, ignore_errors=True)
        verdicts = re.findall(r"^Supported:\s+(.+)$", text, re.M)
        if not verdicts:  # inspect did not run or printed no verdict: unverified, so not "yes"
            say(tag, f"check: no support verdict from dynalm inspect (see {log.name})")
            return "check-failed"
        verdict = verdicts[0]
        if verdict.startswith("NO"):
            for part in ("Architecture", "Tokenizer"):
                why = re.findall(rf"^\s+{part}:\s+NO \((.*)\)$", text, re.M)
                if why:
                    say(tag, f"check: {part.lower()} not supported: {why[0]}")
                    return f"no ({part.lower()})"
            return "no"
        return "yes"

    def dyn_bench(self, backend: str, model: str, tag: str) -> tuple[str, str, str, str, str]:
        """(prompt tok/s, decode tok/s, 4-user output tok/s, prompt sd, decode sd): BENCH_METHOD."""
        j = self.logs / f"{tag}-bench-{backend}.jsonl"
        rl = self.logs / f"{tag}-bench-{backend}.requests.jsonl"  # raw, one line per request
        log = self.logs / f"{tag}-bench-{backend}.log"
        j.unlink(missing_ok=True)
        rl.unlink(missing_ok=True)
        for c, req in ((1, BENCH_REPS), (4, 8)):
            say(tag, f"speed: DynaLM {' '.join(PATHS[backend])}, {c} user(s), {BENCH_PROMPT}-token prompt, "
                     f"{BENCH_GEN} tokens, {req} requests after 1 warm-up ...")
            run([self.dynalm, "benchmark", model, *PATHS[backend], "--concurrency", str(c), "--prompt", str(BENCH_PROMPT),
                 "--output", str(BENCH_GEN), "--requests", str(req), "--warmup-requests", "1", "--out", str(j),
                 "--request-log", str(rl)], Path(str(log) + f".c{c}"), 1200)
        rows = [json.loads(l) for l in j.read_text().splitlines() if l.strip()] if j.exists() else []
        ok = lambda d: d.get("errors", 1) == 0 and d.get("completed") == d.get("requests")
        one = next((d for d in rows if d["concurrency"] == 1 and ok(d)), None)
        four = next((d for d in rows if d["concurrency"] == 4 and ok(d)), None)
        reqs = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()] if rl.exists() else []
        solo = [r for r in reqs if r.get("concurrency") == 1 and r.get("ok") and r.get("completion_tokens") == BENCH_GEN]
        pps = [r["prompt_tokens"] * 1000 / r["ttft_ms"] for r in solo if r["ttft_ms"] > 0]
        tgs = [(r["completion_tokens"] - 1) * 1000 / (r["e2e_ms"] - r["ttft_ms"]) for r in solo if r["e2e_ms"] > r["ttft_ms"]]
        f = lambda v: f"{v:.1f}" if v else "-"
        sd = lambda v: f"{statistics.stdev(v):.2f}" if len(v) > 1 else "-"
        if one is None or len(tgs) != BENCH_REPS:  # a failed or short request: no rate rather than a partial one
            pps, tgs = [], []
        pp, tg = (statistics.mean(pps) if pps else 0), (statistics.mean(tgs) if tgs else 0)
        say(tag, f"speed {backend}: prompt {f(pp)} tok/s, decode {f(tg)} ± {sd(tgs)} tok/s, "
                 f"4 users {f(four['output_tok_s']) if four else '-'} tok/s")
        return f(pp), f(tg), (f(four["output_tok_s"]) if four else "-"), sd(pps), sd(tgs)

    def llama_cmd(self, ngl: int, model: str) -> list[str]:
        """llama-bench as BENCH_METHOD: pp512 and tg64, each at depth 0 and 512, -r 5 after its warm-up."""
        return [self.llama_bench, "-m", model, "-p", str(BENCH_PROMPT), "-n", str(BENCH_GEN), "-d", f"0,{BENCH_PROMPT}",
                "-r", str(BENCH_REPS), "-fa", "on", "-ngl", str(ngl), *(["-nopo", "1"] if ngl == 0 else []),
                "-t", self.threads, "-o", "json"]

    def llama(self, ngl: int, model: str, tag: str) -> tuple[str, str, str, str, str]:
        """(pp512 at depth 0, tg64 at depth 512, pp sd, tg sd, tg64 at depth 0), tok/s."""
        if not self.llama_bench or not os.access(self.llama_bench, os.X_OK):
            return "-", "-", "-", "-", "-"
        j = self.logs / f"{tag}-llama-{'gpu' if ngl else 'cpu'}.json"
        say(tag, f"speed: llama-bench on the {'GPU' if ngl else 'CPU'} (pp{BENCH_PROMPT}, tg{BENCH_GEN} at depth 0 "
                 f"and {BENCH_PROMPT}, -r {BENCH_REPS}) ...")
        run(self.llama_cmd(ngl, model), Path(str(j) + ".log"), 1800, stdout=j)
        try:
            e = json.loads(j.read_text())
            pick = lambda p, g, d: next(x for x in e if (x["n_prompt"], x["n_gen"], x.get("n_depth", 0)) == (p, g, d))
            pp, tg, tg0 = pick(BENCH_PROMPT, 0, 0), pick(0, BENCH_GEN, BENCH_PROMPT), pick(0, BENCH_GEN, 0)
            say(tag, f"llama.cpp {'GPU' if ngl else 'CPU'}: prompt {pp['avg_ts']:.1f} tok/s, decode at depth "
                     f"{BENCH_PROMPT} {tg['avg_ts']:.1f} ± {tg['stddev_ts']:.2f} tok/s (depth 0: {tg0['avg_ts']:.1f})")
            return (f"{pp['avg_ts']:.1f}", f"{tg['avg_ts']:.1f}", f"{pp['stddev_ts']:.2f}", f"{tg['stddev_ts']:.2f}",
                    f"{tg0['avg_ts']:.1f}")
        except Exception:
            return "fail", "fail", "-", "-", "-"

    def one(self, n: int, entry: str) -> dict:
        a, tag = self.a, f"{n:02d}"
        R = {c: "-" for c in COLS}
        R.update(n=tag, model=entry)
        path: Path | None = None
        downloaded = False
        if entry.startswith("hf:"):
            repo = entry[3:]
            R["format"] = "safetensors"
            say(tag, f"check: Hugging Face repo {repo} (files, model_type, size) ...")
            st, files = hf_files(repo, self.token)
            size = sum(f.get("size", 0) for f in files) / 2**30
            R["size_gb"] = f"{size:.1f}" if files else "-"
            if st == "yes" and size > a.max_gb:
                st = "too-large"
            if st == "yes":
                st = self.hf_precheck(repo, files, tag)
            R["supported"] = st
            say(tag, f"check: {st}, {R['size_gb']} GB")
            if st == "yes" and not a.check_only:
                say(tag, f"download: {len(files)} files, {R['size_gb']} GB ...")
                st, path = hf_download(repo, files, self.out / "hf", self.token)
                say(tag, f"download: {st}")
                R["supported"], downloaded = st, path is not None
        else:
            R["format"] = "gguf"
            if Path(entry).exists():
                R["supported"], path = "yes", Path(entry)
            else:
                say(tag, "check: architecture (256 KiB read) ...")
                code = run([self.dynalm, "pull", "--check", entry], self.logs / f"{tag}-check.log", 120)
                text = (self.logs / f"{tag}-check.log").read_text(errors="replace")
                m = re.search(r"architecture '([^']+)' is not supported", text)
                tok = "tokenizer not supported" in text  # dynalm pull --check: the loader's tokenizer rule
                R["supported"] = ("yes" if code == 0 else f"no ({m.group(1)})" if m else "no (tokenizer)" if tok
                                  else "link-error")
                if R["supported"] == "yes" and "/" in entry:
                    size = gguf_size_gb(entry, self.token)
                    if size is not None:
                        R["size_gb"] = f"{size:.1f}"
                        if size > a.max_gb:
                            R["supported"] = "too-large"
                say(tag, f"check: {R['supported']}" + (f", {R['size_gb']} GB" if R["size_gb"] != "-" else ""))
                if R["supported"] == "yes" and not a.check_only:
                    log = self.logs / f"{tag}-pull.log"
                    say(tag, "download: dynalm pull (progress: the .part file in the model store) ...")
                    if run([self.dynalm, "pull", entry], log, 7200) == 0:
                        text = log.read_text(errors="replace")
                        found = re.findall(r"(?:saved|found) (.+?\.gguf)", text)
                        path = Path(found[-1]) if found else None
                        downloaded = "already downloaded" not in text
                        say(tag, f"download: {'done' if downloaded else 'already in the store (kept after the run)'}: {path}")
                    else:
                        R["supported"] = "download-failed"
        if path is None or not path.exists() or a.check_only:
            return R
        m = str(path)
        inspect = self.logs / f"{tag}-inspect.log"
        run([self.dynalm, "inspect", m], inspect, 120)
        it = inspect.read_text(errors="replace")
        R["file"] = path.name
        R["arch"] = (re.findall(r"^Architecture:\s+(\S+)", it, re.M) or ["-"])[0]
        R["quant"] = (re.findall(r"^Quantization:\s+(\S+)", it, re.M) or ["-"])[0]
        if R["size_gb"] == "-":
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
            R["size_gb"] = f"{size / 2**30:.1f}"
        R["auto"], R["auto_device"], R["auto_tok_s"], ta = self.generate("auto", m, tag)
        R["cpu"], R["cpu_device"], R["cpu_tok_s"], tc = self.generate("cpu", m, tag)
        R["native"], _, R["native_tok_s"], tn = self.generate("exact", m, tag)
        same = lambda x, y: "yes" if x.exists() and y.exists() and x.stat().st_size and x.read_bytes() == y.read_bytes() else "no"
        if R["native"] == "ok":
            R["auto_matches_native"] = same(ta, tn) if R["auto"] == "ok" else "-"
            R["cpu_matches_native"] = same(tc, tn) if R["cpu"] == "ok" else "-"
        on_gpu = R["auto"] == "ok" and not R["auto_device"].startswith(("CPU", "ggml-CPU"))
        if a.accuracy and on_gpu and R["format"] == "gguf" and os.access(self.acc, os.X_OK):
            log = self.logs / f"{tag}-accuracy.log"
            say(tag, "accuracy: DD-053 GPU check against the fp32 CPU reference (1-3 min) ...")
            run([self.acc, m], log, 1200)
            t = log.read_text(errors="replace")
            res = re.findall(r"RESULT: ([A-Z]+)", t)
            dec = re.findall(r"reference: decode \(M = 1\)\]\n(?:.*\n){0,2}?.*change ([+-][0-9.]+%)", t)
            R["accuracy"] = f"{res[0] if res else 'error'} {dec[0] if dec else ''}".strip()
            say(tag, f"accuracy: {R['accuracy']}")
        if a.bench:
            say(tag, "speed: waiting for AC power, normal power mode and a quiet machine ..." if a.gate else "speed: start")
            R["bench_conditions"] = (wait_quiet(a.max_load, self.out / "conditions.log", tag) if a.gate else conditions())
            R["bench_method"] = BENCH_METHOD
            paths = a.bench_paths
            if on_gpu and "gpu" in paths:
                (R["gpu_prompt_tok_s"], R["gpu_decode_tok_s"], R["gpu_4user_tok_s"], R["gpu_prompt_sd"],
                 R["gpu_decode_sd"]) = self.dyn_bench("auto", m, tag)
            if R["cpu"] == "ok" and "cpu" in paths:
                (R["cpu_prompt_tok_s"], R["cpu_decode_tok_s"], R["cpu_4user_tok_s"], R["cpu_prompt_sd"],
                 R["cpu_decode_sd"]) = self.dyn_bench("cpu", m, tag)
            if R["native"] == "ok" and "exact" in paths:  # exact mode (--mode exact): DynaLM's own DD-053 CPU kernels
                (R["exact_prompt_tok_s"], R["exact_decode_tok_s"], R["exact_4user_tok_s"], R["exact_prompt_sd"],
                 R["exact_decode_sd"]) = self.dyn_bench("exact", m, tag)
            if R["format"] == "gguf":
                if a.gate:  # llama-bench under the same conditions as DynaLM's runs (state in conditions.log)
                    wait_quiet(a.max_load, self.out / "conditions.log", tag + " llama-bench")
                if "gpu" in paths:
                    (R["llama_gpu_pp512"], R["llama_gpu_tg128"], R["llama_gpu_pp_sd"], R["llama_gpu_tg_sd"],
                     R["llama_gpu_tg_d0"]) = self.llama(99, m, tag)
                if "cpu" in paths or "exact" in paths:
                    (R["llama_cpu_pp512"], R["llama_cpu_tg128"], R["llama_cpu_pp_sd"], R["llama_cpu_tg_sd"],
                     R["llama_cpu_tg_d0"]) = self.llama(0, m, tag)
        if downloaded and not a.keep:
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
            say(tag, f"deleted {path.name} (downloaded by this run)")
        return R


def entries(list_file: Path) -> list[str]:
    out = []
    for raw in list_file.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Model compatibility matrix (see the module docstring).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=str(HERE / "models-matrix.txt"))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--bench-paths", default="gpu,cpu,exact", type=lambda s: set(s.split(",")),
                    help="speed runs to make with --bench: any of gpu, cpu, exact")
    ap.add_argument("--gate", action="store_true",
                    help="before speed runs, wait for AC power, Energy Mode Automatic and load < --max-load")
    ap.add_argument("--max-load", type=float, default=3.0)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--max-gb", type=float, default=16.0)
    ap.add_argument("--dynalm", default=os.environ.get("DYNALM") or shutil.which("dynalm") or "")
    a = ap.parse_args()
    if not a.dynalm or not os.access(a.dynalm, os.X_OK):
        print("error: DYNALM=path to the dynalm binary (or --dynalm)", file=sys.stderr)
        return 1
    mx = Matrix(a)
    tsv = mx.out / "results.tsv"
    with open(tsv, "w") as f:
        f.write("\t".join(COLS) + "\n")
    todo = entries(Path(a.list))
    for n, entry in enumerate(todo, 1):
        t0 = time.time()
        print(f"== [{n:02d}/{len(todo):02d}] {time.strftime('%H:%M:%S')} {entry}", flush=True)
        R = mx.one(n, entry)
        R["seconds"] = str(int(time.time() - t0))
        with open(tsv, "a") as f:
            f.write("\t".join(str(R[c]) for c in COLS) + "\n")
        print(f"   {R['format']} {R['supported']} | auto {R['auto']} on {R['auto_device']} | cpu {R['cpu']} | "
              f"native {R['native']} | same as native: auto {R['auto_matches_native']}, cpu {R['cpu_matches_native']}"
              + (f" | gpu decode {R['gpu_decode_tok_s']} vs llama {R['llama_gpu_tg128']}" if a.bench else "")
              + f" | {R['seconds']}s", flush=True)
    subprocess.run([sys.executable, str(HERE / "matrix_summary.py"), str(tsv), str(mx.out / "summary.md"), a.dynalm])
    print(f"done: {mx.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

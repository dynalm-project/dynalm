#!/usr/bin/env python3
"""MoE models larger than RAM, on the CPU: baseline speed (docs/perf/moe-ssd-plan.md).

  python3 tools/bench/moe_ssd_bench.py --dry-run                 # the plan for this machine
  python3 tools/bench/moe_ssd_bench.py --download                # download what fits, then measure
  python3 tools/bench/moe_ssd_bench.py --models 30b,mixtral      # only these (downloaded ones)
  python3 tools/bench/moe_ssd_bench.py --download --delete-after # one large model on disk at a time

Models (fixed; GGUF from Hugging Face, each file checked against its SHA-256):
  30b      Qwen3-30B-A3B Q4_K_M        18.6 GB   qwen3moe   3.3B active
  mixtral  Mixtral-8x7B-Instruct Q4_K_M 26.4 GB  llama (MoE) 12.9B active
  235b     Qwen3-235B-A22B Q4_K_M      142.2 GB  qwen3moe   22B active (3 split files)
  671b     DeepSeek-V3-0324 UD-IQ1_S   186.2 GB  deepseek2  37B active (4 split files)

Engines, CPU only (no GPU), the same threads:
  llama.cpp  llama-bench -ngl 0 -dev none -lm mmap: weights are read from the file on demand,
             so a model larger than RAM runs from the SSD. --repack 0 for such models (repacking
             would copy every weight into RAM).
  dynalm     dynalm benchmark --backend cpu (in-process); prompt tok/s = prompt / time to first
             token, generate tok/s = 1000 / median gap between tokens (as llama-bench's pp and tg).
             Without the expert cache DynaLM refuses models larger than 90% of RAM; that is
             recorded as "refused", with its message.
  dynalm-disk  the same with --experts-from-disk (the expert cache, auto size): any model size.

Workload (fixed before any result): a model that fits in RAM: 512-token prompt, 128 generated
tokens, 2 repetitions. A model that does not: 128 and 32, 1 repetition (each token reads GBs).
Each run is wrapped in /usr/bin/time (macOS -l, Linux -v) for peak memory and disk reads.

Writes moe-ssd-<host>-<time>.json and .md to --out. Downloads go to --work (default
~/.dynalm/moe-bench); --delete-after removes only the files this script downloaded in this run.
A model whose download would leave less than 15 GB of disk free is skipped, never started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HF = "https://huggingface.co"
GB = 1_000_000_000
DISK_MARGIN = 15 * GB
RAM_FIT = 0.9  # as DynaLM's own check: weights + KV within 90 % of RAM

# id, label, repo, arch, active params, [(path in repo, bytes, sha256)]
MODELS = [
    ("30b", "Qwen3-30B-A3B Q4_K_M", "Qwen/Qwen3-30B-A3B-GGUF", "qwen3moe", "3.3B", [
        ("Qwen3-30B-A3B-Q4_K_M.gguf", 18556685824,
         "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5")]),
    ("mixtral", "Mixtral-8x7B-Instruct Q4_K_M", "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "llama (MoE)", "12.9B", [
        ("mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", 26441533376,
         "9193684683657e90707087bd1ed19fd0b277ab66358d19edeadc26d6fdec4f53")]),
    # Q4_K_M, not Q2_K: close to the original quality, and DynaCore has fast kernels for Q4_K / Q6_K
    # (none for Q2_K).
    ("235b", "Qwen3-235B-A22B Q4_K_M", "unsloth/Qwen3-235B-A22B-GGUF", "qwen3moe", "22B", [
        ("Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf", 49944699648,
         "a8bc43ded2ef4868ea31302ddbfa8cd45dbbeecb476f4372337f41fe51b8a1cb"),
        ("Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00003.gguf", 49930349824,
         "a1131375023518eb494c951708e79a3a82405e96ddc093c44994a96d0f001927"),
        ("Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00003-of-00003.gguf", 42279026016,
         "5d33367138d337305f8f12395f3f2ed850afa0935e98cfcf4d7247143599c500")]),
    ("671b", "DeepSeek-V3-0324 UD-IQ1_S", "unsloth/DeepSeek-V3-0324-GGUF", "deepseek2", "37B", [
        ("UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf", 48618624960,
         "a0a6642eb5f4cd378fdb2b637b89126a88e8ab2a510cf05224b5f300d983d98d"),
        ("UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00002-of-00004.gguf", 48496275072,
         "8faceb0370c0fa0ced2b410027955520c710d892dfa793297c0d8ec022375496"),
        ("UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00003-of-00004.gguf", 48496275072,
         "881849206d4ff8a562f5a2711f1c620abcbdfd2014dc912ff4c6f03d028b6fbf"),
        ("UD-IQ1_S/DeepSeek-V3-0324-UD-IQ1_S-00004-of-00004.gguf", 40627584480,
         "11fc8abfc5ef7051e2d9e8c5adb27408db0c80541fa5bd50d16021a8cd61dcff")]),
]
IN_RAM = {"prompt": 512, "gen": 128, "reps": 2}
FROM_SSD = {"prompt": 128, "gen": 32, "reps": 1}

T0 = time.monotonic()


def say(msg: str, detail: bool = False) -> None:
    s = int(time.monotonic() - T0)
    prefix = " " * 13 if detail else f"[+{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}] "
    print(prefix + msg, flush=True)


def model_bytes(m) -> int:
    return sum(f[1] for f in m[5])


def ram_bytes() -> int:
    if sys.platform == "darwin":
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def cpu_name() -> str:
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def fits_in_ram(size: int, ram: int) -> bool:
    return ram > 0 and size <= RAM_FIT * ram


def workload(size: int, ram: int) -> dict:
    return IN_RAM if fits_in_ram(size, ram) else FROM_SSD


def plan(ids: list[str], ram: int, free: int, have: dict[str, bool], download: bool, delete_after: bool):
    """Per model: 'run', 'download', or 'skip: why'. Smallest first; downloads must leave
    DISK_MARGIN free (with --delete-after, each model only needs room for itself)."""
    out = []
    left = free
    for m in sorted((m for m in MODELS if m[0] in ids), key=model_bytes):
        size = model_bytes(m)
        if have.get(m[0]):
            out.append((m, "run"))
        elif not download:
            out.append((m, "skip: not downloaded (use --download)"))
        elif left - size < DISK_MARGIN and delete_after:
            # Earlier models are deleted before it, and space may be freed meanwhile: decided then.
            out.append((m, f"download if {(size + DISK_MARGIN) / GB:.0f} GB are free by then "
                           f"({left / GB:.1f} GB free now)"))
        elif left - size < DISK_MARGIN:
            out.append((m, f"skip: needs {size / GB:.1f} GB of disk, {left / GB:.1f} GB free "
                           f"(keeps {DISK_MARGIN // GB} GB free)"))
        else:
            out.append((m, "download"))
            if not delete_after:
                left -= size
    return out


def parse_llama_bench(text: str) -> dict:
    """llama-bench -o json: the JSON array (logs may precede it). {'pp': tok/s, 'tg': tok/s}."""
    m = re.search(r"^\[", text, re.M)
    if not m or "]" not in text[m.start():]:
        return {}
    try:
        rows = json.loads(text[m.start():text.rindex("]") + 1])
    except json.JSONDecodeError:
        return {}
    r = {}
    for row in rows:
        if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0:
            r["pp"] = row["avg_ts"]
        elif row.get("n_gen", 0) > 0 and row.get("n_prompt", 0) == 0:
            r["tg"] = row["avg_ts"]
    return r


def parse_time(text: str) -> dict:
    """/usr/bin/time -l (macOS) or -v (Linux): peak RSS bytes and disk reads."""
    r = {}
    if m := re.search(r"(\d+)\s+maximum resident set size", text):  # macOS: bytes
        r["peak_rss"] = int(m.group(1))
    if m := re.search(r"Maximum resident set size \(kbytes\): (\d+)", text):  # Linux: KiB
        r["peak_rss"] = int(m.group(1)) * 1024
    if m := re.search(r"(\d+)\s+page faults", text):
        r["page_faults"] = int(m.group(1))
    if m := re.search(r"Major \(requiring I/O\) page faults: (\d+)", text):
        r["page_faults"] = int(m.group(1))
    if m := re.search(r"(\d+)\s+block input operations", text):
        r["block_inputs"] = int(m.group(1))
    if m := re.search(r"File system inputs: (\d+)", text):
        r["block_inputs"] = int(m.group(1))
    return r


def parse_dynalm(jsonl: str, prompt_tokens: int) -> dict:
    """dynalm benchmark --out (one JSON line per point), as llama-bench measures: prompt tok/s =
    prompt tokens / time to first token; generate tok/s = 1000 / median gap between tokens."""
    for line in jsonl.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ttft = (d.get("ttft_ms") or {}).get("p50")
        itl = (d.get("itl_ms") or {}).get("p50")
        return {"pp": prompt_tokens / (ttft / 1000) if ttft else None, "tg": 1000 / itl if itl else None,
                "ttft_ms": ttft}
    return {}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, path: str, size: int, sha: str, dest: Path, log: Path) -> str:
    """Downloads (resumable) and checks the SHA-256; '' when fine, else the error."""
    if dest.exists() and dest.stat().st_size == size:
        say(f"checking {dest.name} ({size / GB:.1f} GB)", True)
        if sha256_file(dest) == sha:
            say("already downloaded, sha256 matches", True)
            return ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/{repo}/resolve/main/{path}"
    say(f"downloading {url}", True)
    t = time.monotonic()
    with log.open("a") as lf:
        rc = subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-sS", "-o", str(dest), url],
                            stdout=lf, stderr=lf).returncode
    if rc != 0:
        return f"download failed (curl {rc}, log {log})"
    got = sha256_file(dest)
    if got != sha:
        dest.unlink(missing_ok=True)
        return f"sha256 mismatch for {path}: got {got}, expected {sha}"
    say(f"done in {time.monotonic() - t:.0f} s, sha256 matches", True)
    return ""


def timed(cmd: list[str], log: Path, timeout_s: int, env=None) -> tuple[int, str, str]:
    """Runs cmd under /usr/bin/time; returns (exit code, stdout, the time report)."""
    tflag = "-l" if sys.platform == "darwin" else "-v"
    wrap = ["/usr/bin/time", tflag] if Path("/usr/bin/time").exists() and sys.platform != "win32" else []
    with log.open("w") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        try:
            p = subprocess.run(wrap + cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            lf.write(f"timeout after {timeout_s} s\n")
            return -2, "", ""
        lf.write(p.stdout)
        lf.write(p.stderr)
    return p.returncode, p.stdout, p.stderr


def run_llama(bench: Path, model: Path, wl: dict, threads: int, in_ram: bool, log: Path, timeout_s: int) -> dict:
    cmd = [str(bench), "-m", str(model), "-p", str(wl["prompt"]), "-n", str(wl["gen"]), "-r", str(wl["reps"]),
           "-t", str(threads), "-ngl", "0", "-dev", "none", "-lm", "mmap", "-o", "json"]
    if not in_ram:
        cmd += ["--repack", "0"]
    env = dict(os.environ, LD_LIBRARY_PATH=str(bench.parent), DYLD_LIBRARY_PATH=str(bench.parent))
    rc, out, err = timed(cmd, log, timeout_s, env)
    r = {"status": "ok" if rc == 0 else ("timeout" if rc == -2 else f"failed (exit {rc})")}
    r.update(parse_llama_bench(out + "\n" + err))
    r.update(parse_time(err))
    if rc == 0 and "tg" not in r:
        r["status"] = "failed (no result in output)"
    return r


def run_dynalm(dynalm: str, model: Path, wl: dict, threads: int, log: Path, timeout_s: int,
               experts_from_disk: bool = False) -> dict:
    jsonl = log.with_suffix(".jsonl")
    jsonl.unlink(missing_ok=True)
    cmd = [dynalm, "benchmark", str(model), "--backend", "cpu", "-t", str(threads), "-c", "4096", "--concurrency", "1",
           "--prompt", str(wl["prompt"]), "--output", str(wl["gen"]), "--requests", str(max(1, wl["reps"])),
           "--warmup-requests", "1" if wl is IN_RAM else "0", "--out", str(jsonl)]
    if experts_from_disk:
        cmd.append("--experts-from-disk")
    rc, out, err = timed(cmd, log, timeout_s)
    r = {"status": "ok" if rc == 0 else ("timeout" if rc == -2 else f"failed (exit {rc})")}
    if rc not in (0, -2):
        if m := re.search(r"not enough memory[^\n]*", out + err):
            r["status"] = "refused: " + m.group(0)
        elif m := re.search(r"(unsupported|not supported)[^\n]*", out + err, re.I):
            r["status"] = "unsupported: " + m.group(0)
    if jsonl.exists():
        r.update(parse_dynalm(jsonl.read_text(), wl["prompt"]))
    r.update(parse_time(err))
    return r


def gguf_first(m, work: Path) -> Path:
    return work / m[2].replace("/", "__") / m[5][0][0]


def to_markdown(rep: dict) -> str:
    md = [f"## MoE larger-than-RAM baseline: {rep['cpu']}", "",
          f"- **RAM:** {rep['ram'] / GB:.1f} GB; **disk free at start:** {rep['disk_free'] / GB:.0f} GB; "
          f"**threads:** {rep['threads']}; CPU only",
          f"- **llama.cpp:** {rep['llama']}; **DynaLM:** {rep['dynalm_version']}",
          f"- **Workload:** in RAM {IN_RAM['prompt']}/{IN_RAM['gen']} tokens x{IN_RAM['reps']}; "
          f"from SSD {FROM_SSD['prompt']}/{FROM_SSD['gen']} x{FROM_SSD['reps']}", ""]
    md += ["| Model | Size | Fits in RAM | Engine | Prompt tok/s | Generate tok/s | Peak RSS | Disk reads | Status |",
           "|---|---|---|---|---|---|---|---|---|"]
    def num(v, fmt):
        return fmt % v if v is not None else "–"

    for r in rep["results"]:
        rss = f"{r['peak_rss'] / GB:.1f} GB" if r.get("peak_rss") else "–"
        reads = f"{r['block_inputs']:,}" if r.get("block_inputs") is not None else "–"
        md.append(f"| {r['label']} | {r['size'] / GB:.1f} GB | {'yes' if r['in_ram'] else 'no (SSD)'} | {r['engine']} | "
                  f"{num(r.get('pp'), '%.1f')} | {num(r.get('tg'), '%.2f')} | {rss} | {reads} | {r['status']} |")
    for s in rep["skipped"]:
        md.append(f"| {s['label']} | {s['size'] / GB:.1f} GB | – | – | – | – | – | – | {s['why']} |")
    if rep["notes"]:
        md += ["", "Notes:"] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(m[0] for m in MODELS))
    ap.add_argument("--download", action="store_true", help="download models that fit on disk")
    ap.add_argument("--delete-after", action="store_true", help="delete each model this run downloaded once measured")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="default: performance cores (macOS) or physical cores")
    ap.add_argument("--llama", default=str(Path.home() / ".dynalm/cpu-report/llama-b11695"),
                    help="a directory containing llama-bench (default: the build dynalm cpu-report downloads)")
    ap.add_argument("--dynalm", default=shutil.which("dynalm") or "dynalm")
    ap.add_argument("--engines", default="llama.cpp,dynalm,dynalm-disk")
    ap.add_argument("--work", default=str(Path.home() / ".dynalm/moe-bench"))
    ap.add_argument("--out", default=".")
    ap.add_argument("--timeout-min", type=int, default=120, help="per run")
    a = ap.parse_args()

    ids = [s for s in a.models.split(",") if s]
    unknown = [i for i in ids if i not in {m[0] for m in MODELS}]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)} (known: {', '.join(m[0] for m in MODELS)})", file=sys.stderr)
        return 1
    work = Path(a.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    ram = ram_bytes()
    free = shutil.disk_usage(work).free
    threads = a.threads
    if threads <= 0:
        threads = (int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                      text=True).stdout or 0) if sys.platform == "darwin" else 0) or os.cpu_count() or 4
    engines = [e for e in a.engines.split(",") if e]

    print("MoE larger-than-RAM baseline (CPU only)")
    say("[1/4] This machine")
    say(f"CPU: {cpu_name()}; RAM {ram / GB:.1f} GB (a model fits when <= {RAM_FIT:.0%} of it); threads {threads}", True)
    say(f"disk free: {free / GB:.0f} GB in {work}", True)

    have = {m[0]: all((work / m[2].replace("/", "__") / f[0]).exists() and
                      (work / m[2].replace("/", "__") / f[0]).stat().st_size == f[1] for f in m[5]) for m in MODELS}
    p = plan(ids, ram, free, have, a.download, a.delete_after)
    say("[2/4] Plan")
    for m, action in p:
        size = model_bytes(m)
        where = "in RAM" if fits_in_ram(size, ram) else "from SSD"
        wl = workload(size, ram)
        say(f"{m[0]:8s} {m[1]:30s} {size / GB:6.1f} GB, {m[3]}, {m[4]} active, runs {where} "
            f"({wl['prompt']}/{wl['gen']} tokens x{wl['reps']}): {action}", True)
    say(f"engines: {', '.join(engines)}", True)
    if a.dry_run:
        print("\n--dry-run: nothing downloaded or run")
        return 0

    bench = next(iter(sorted(Path(a.llama).expanduser().rglob("llama-bench*"))), None)
    if "llama.cpp" in engines and bench is None:
        print(f"llama-bench not found under {a.llama}; run `dynalm cpu-report --dry-run` once with the "
              "network, or pass --llama DIR", file=sys.stderr)
        return 1
    dyn_version = "not run"
    if "dynalm" in engines or "dynalm-disk" in engines:
        v = subprocess.run([a.dynalm, "--version"], capture_output=True, text=True)
        dyn_version = (v.stdout.splitlines() or ["unknown"])[0] if v.returncode == 0 else f"not found ({a.dynalm})"
    rep = {"cpu": cpu_name(), "ram": ram, "disk_free": free, "threads": threads, "os": platform.platform(),
           "llama": str(bench) if bench else "not run", "dynalm_version": dyn_version,
           "workloads": {"in_ram": IN_RAM, "from_ssd": FROM_SSD}, "results": [], "skipped": [], "notes": []}

    if sys.platform == "darwin" and "Battery Power" in subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True).stdout:
        rep["notes"].append("uncontrolled: ran on battery")
        say("WARNING: on battery; plug in the charger for comparable numbers", True)
    say("[3/4] Models and runs")
    for m, action in p:
        size = model_bytes(m)
        if action.startswith("skip"):
            rep["skipped"].append({"model": m[0], "label": m[1], "size": size, "why": action[6:]})
            continue
        downloaded = []
        say(f"{m[0]}: {m[1]}")
        # Disk again now: earlier models may have been deleted (--delete-after) or space freed.
        missing = sum(f[1] for f in m[5] if not ((work / m[2].replace("/", "__") / f[0]).exists() and
                                                 (work / m[2].replace("/", "__") / f[0]).stat().st_size == f[1]))
        free_now = shutil.disk_usage(work).free
        if missing and free_now - missing < DISK_MARGIN:
            why = (f"needs {missing / GB:.1f} GB more disk, {free_now / GB:.1f} GB free now "
                   f"(keeps {DISK_MARGIN // GB} GB free)")
            say(f"skipped: {why}", True)
            rep["skipped"].append({"model": m[0], "label": m[1], "size": size, "why": why})
            continue
        err = ""
        for path, nbytes, sha in m[5]:
            dest = work / m[2].replace("/", "__") / path
            existed = dest.exists() and dest.stat().st_size == nbytes
            err = fetch(m[2], path, nbytes, sha, dest, work / "logs" / "download.log")
            if err:
                break
            if not existed:
                downloaded.append(dest)
        if err:
            say(f"FAILED: {err}", True)
            rep["skipped"].append({"model": m[0], "label": m[1], "size": size, "why": err})
            continue
        model = gguf_first(m, work)
        in_ram = fits_in_ram(size, ram)
        wl = workload(size, ram)
        for eng in engines:
            log = work / "logs" / f"{m[0]}-{eng}.log"
            say(f"{eng}: {wl['prompt']}-token prompt, {wl['gen']} generated, x{wl['reps']} (log {log})", True)
            t = time.monotonic()
            if eng == "llama.cpp":
                r = run_llama(bench, model, wl, threads, in_ram, log, a.timeout_min * 60)
            elif eng in ("dynalm", "dynalm-disk"):
                r = run_dynalm(a.dynalm, model, wl, threads, log, a.timeout_min * 60, eng == "dynalm-disk")
            else:
                r = {"status": f"unknown engine {eng}"}
            r.update({"model": m[0], "label": m[1], "size": size, "in_ram": in_ram, "engine": eng,
                      "seconds": round(time.monotonic() - t, 1)})
            rep["results"].append(r)
            say(f"  {r['status']}: prompt {r.get('pp') or 0:.1f} tok/s, generate {r.get('tg') or 0:.2f} tok/s, "
                f"peak RSS {(r.get('peak_rss') or 0) / GB:.1f} GB, {r['seconds']:.0f} s", True)
        if a.delete_after and downloaded:
            for f in downloaded:
                f.unlink(missing_ok=True)
            say(f"deleted {len(downloaded)} file(s) downloaded for {m[0]} (--delete-after)", True)

    say("[4/4] Report")
    rep["notes"].append("A model that fits in RAM is read from the SSD only on its first run; later runs use "
                        "the operating system's file cache.")
    if sys.platform == "darwin":
        rep["notes"].append("macOS: 'Disk reads' counts block input operations from /usr/bin/time -l.")
    base = f"moe-ssd-{re.sub(r'[^A-Za-z0-9_-]', '-', socket.gethostname())}-{time.strftime('%Y%m%d-%H%M')}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{base}.json").write_text(json.dumps(rep, indent=2) + "\n")
    md = to_markdown(rep)
    (out / f"{base}.md").write_text(md)
    print("\n" + md)
    say(f"wrote {out / (base + '.json')} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

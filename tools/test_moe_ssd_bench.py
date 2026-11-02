#!/usr/bin/env python3
"""Offline checks of tools/bench/moe_ssd_bench.py (no models, no binaries).

  python tools/test_moe_ssd_bench.py

- The plan: smallest model first; a download that would leave less than 15 GB of disk free is
  skipped with its reason; with --delete-after each model only needs room for itself; a model
  already on disk always runs; without --download nothing is fetched.
- The workload follows the RAM fit (90 % of RAM, as DynaLM's own check).
- llama-bench JSON is found after log lines; /usr/bin/time reports parse on macOS and Linux;
  dynalm benchmark --out lines parse.
- Every model file has a 64-hex-digit SHA-256 and a size.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("moe_ssd_bench", HERE / "bench" / "moe_ssd_bench.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAILED: {what}")
        sys.exit(1)


GB = mb.GB
ids = ["30b", "mixtral", "235b", "671b"]
ram = 24 * 2**30

p = mb.plan(ids, ram, 112 * GB, {}, download=True, delete_after=False)
check([m[0] for m, _ in p] == ["30b", "mixtral", "235b", "671b"], "smallest first")
check([a for _, a in p][:2] == ["download", "download"], "30b and mixtral fit on 112 GB")
check(p[2][1].startswith("skip: needs 85.7 GB of disk, 67.0 GB free"), "235b skipped after the first two")
p = mb.plan(ids, ram, 112 * GB, {}, download=True, delete_after=True)
check([a for _, a in p][:3] == ["download"] * 3 and p[3][1].startswith("skip"), "--delete-after: 235b fits alone")
p = mb.plan(ids, ram, 1 * GB, {"235b": True}, download=False, delete_after=False)
check(dict((m[0], a) for m, a in p) == {"30b": "skip: not downloaded (use --download)",
                                         "mixtral": "skip: not downloaded (use --download)",
                                         "235b": "run", "671b": "skip: not downloaded (use --download)"},
      "on disk runs, the rest needs --download")

check(mb.workload(18_556_685_824, ram) is mb.IN_RAM, "30b fits in 24 GiB")
check(mb.workload(26_441_533_376, ram) is mb.FROM_SSD, "mixtral does not")
check(mb.workload(18_556_685_824, 16 * 2**30) is mb.FROM_SSD, "30b does not fit in 16 GiB")

log = 'load_backend: loaded CPU\n[\n  {"n_prompt": 128, "n_gen": 0, "avg_ts": 41.5},\n' \
      '  {"n_prompt": 0, "n_gen": 32, "avg_ts": 1.25}\n]\n'
check(mb.parse_llama_bench(log) == {"pp": 41.5, "tg": 1.25}, "llama-bench JSON after logs")
check(mb.parse_llama_bench("error: failed to load model\n") == {}, "no JSON")
mac = "  1.5 real\n  2147483648  maximum resident set size\n  812  page faults\n  9000  block input operations\n"
check(mb.parse_time(mac) == {"peak_rss": 2147483648, "page_faults": 812, "block_inputs": 9000}, "macOS time -l")
lin = "\tMaximum resident set size (kbytes): 1048576\n\tMajor (requiring I/O) page faults: 7\n\tFile system inputs: 64\n"
check(mb.parse_time(lin) == {"peak_rss": 1073741824, "page_faults": 7, "block_inputs": 64}, "Linux time -v")
d = mb.parse_dynalm('{"output_tok_s": 12.5, "ttft_ms": {"p50": 400.0}, "itl_ms": {"p50": 80.0}}\n', 128)
check(d == {"pp": 320.0, "tg": 12.5, "ttft_ms": 400.0}, "dynalm --out line, measured as llama-bench")

for m in mb.MODELS:
    for path, size, sha in m[5]:
        check(size > GB and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), f"{path} pinned")

print(f"all moe_ssd_bench tests passed ({checks} checks)")

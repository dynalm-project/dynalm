#!/usr/bin/env python3
"""summary.md of a tools/bench/model_matrix.py run.

  python3 tools/bench/matrix_summary.py <results.tsv> <summary.md> [<dynalm binary>]
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter


def num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ratio(a: str, b: str) -> str:
    x, y = num(a), num(b)
    return f"{x / y:.2f}×" if x and y else "–"


def main() -> int:
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    ver = []
    if len(sys.argv) > 3:
        ver = subprocess.run([sys.argv[3], "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    ran = [r for r in rows if r["native"] != "-"]
    ok = lambda r, k: r[k] == "ok"
    gpu = [r for r in ran if ok(r, "auto") and not r["auto_device"].startswith(("CPU", "ggml-CPU"))]
    benched = any(r["gpu_decode_tok_s"] != "-" or r["cpu_decode_tok_s"] != "-" for r in rows)
    L = ["# Model matrix", ""]
    if ver:
        L += ["```text", *ver, "```", ""]
    by_status = Counter(r["supported"] for r in rows)
    L += [f"- **Entries:** {len(rows)} ({sum(r['format'] == 'gguf' for r in rows)} GGUF, "
          f"{sum(r['format'] == 'safetensors' for r in rows)} safetensors)",
          "- **Check:** " + ", ".join(f"{k} {v}" for k, v in by_status.most_common()),
          f"- **Generated:** {len(ran)}. auto ok {sum(ok(r, 'auto') for r in ran)} (on a GPU {len(gpu)}), "
          f"cpu ok {sum(ok(r, 'cpu') for r in ran)}, native-cpu ok {sum(ok(r, 'native') for r in ran)}",
          f"- **Same text as native-cpu** (greedy; reported, not gated): auto "
          f"{sum(r['auto_matches_native'] == 'yes' for r in ran)}, cpu {sum(r['cpu_matches_native'] == 'yes' for r in ran)}"]
    acc = [r for r in rows if r["accuracy"] != "-"]
    if acc:
        L.append(f"- **GPU accuracy (DD-053):** PASS {sum(r['accuracy'].startswith('PASS') for r in acc)} of {len(acc)}")
    L.append("")
    fails = [r for r in rows if r["supported"] != "yes" or "fail" in (r["auto"], r["cpu"], r["native"])
             or "timeout" in (r["auto"], r["cpu"], r["native"])]
    if fails:
        L += ["## Needs attention", "", "| # | Model | Format | Check | auto | cpu | native-cpu |", "|---|---|---|---|---|---|---|"]
        for r in fails:
            L.append(f"| {r['n']} | `{r['model']}` | {r['format']} | {r['supported']} | {r['auto']} | {r['cpu']} | {r['native']} |")
        L += ["", "Reasons: `logs/<n>-check.log`, `<n>-auto.log`, `<n>-cpu.log`, `<n>-native-cpu.log`. "
              "`no (<arch>)`: architecture not supported; `link-error`: missing file or wrong link; "
              "`gated`: needs HF_TOKEN with the license accepted; `too-large`: above --max-gb.", ""]
    L += ["## Support", "",
          "| # | Model | Format | Arch | Quant | GB | auto: device, tok/s | cpu: device, tok/s | native-cpu tok/s | auto = native | cpu = native | DD-053 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']} | `{r['model']}` | {r['format']} | {r['arch']} | {r['quant']} | {r['size_gb']} | "
                 f"{r['auto']}: {r['auto_device']}, {r['auto_tok_s']} | {r['cpu']}: {r['cpu_device']}, {r['cpu_tok_s']} | "
                 f"{r['native']}: {r['native_tok_s']} | {r['auto_matches_native']} | {r['cpu_matches_native']} | {r['accuracy']} |")
    if benched:
        L += ["", "## Speed (short single runs: a screen, not a published benchmark)", "",
              "DynaLM: `dynalm benchmark` in-process. One user, 512-token prompt: prompt tok/s = 512 / TTFT p50, decode "
              "tok/s = 1000 / ITL p50. Four users: output tok/s. llama.cpp: `llama-bench` pp512 / tg128 on the same "
              "GGUF. Ratios > 1: DynaLM faster. Conditions: AC power and load at the start of each model's runs.", "",
              "| # | Model | GPU prompt (llama) | GPU decode (llama) | GPU 4 users | CPU prompt (llama) | CPU decode (llama) | CPU 4 users | Conditions |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["bench_conditions"] == "-":
                continue
            L.append(
                f"| {r['n']} | `{r['model']}` | {r['gpu_prompt_tok_s']} ({r['llama_gpu_pp512']}, {ratio(r['gpu_prompt_tok_s'], r['llama_gpu_pp512'])}) | "
                f"{r['gpu_decode_tok_s']} ({r['llama_gpu_tg128']}, {ratio(r['gpu_decode_tok_s'], r['llama_gpu_tg128'])}) | {r['gpu_4user_tok_s']} | "
                f"{r['cpu_prompt_tok_s']} ({r['llama_cpu_pp512']}, {ratio(r['cpu_prompt_tok_s'], r['llama_cpu_pp512'])}) | "
                f"{r['cpu_decode_tok_s']} ({r['llama_cpu_tg128']}, {ratio(r['cpu_decode_tok_s'], r['llama_cpu_tg128'])}) | {r['cpu_4user_tok_s']} | "
                f"{r['bench_conditions']} |")
    L += ["", "Per-step logs: `logs/`."]
    open(sys.argv[2], "w").write("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("- ")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

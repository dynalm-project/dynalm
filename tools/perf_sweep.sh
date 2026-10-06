#!/usr/bin/env bash
# Performance-program baseline sweep (DD-050): runs the measurement matrix and
# renders the report with the eight standard graphs.
#
#   tools/perf_sweep.sh [quick|full] [OUT_DIR]
#
# Env: DYNALM (binary, default: dynalm on PATH or build/*/bin/dynalm[.exe]),
#      MODELS_DIR (default models/). Models that are absent are skipped.
# Every point appends one JSON line to OUT_DIR/sweep.jsonl; OUT_DIR/report.md
# and OUT_DIR/graphs/*.svg are rewritten at the end. Run on an idle machine:
# background load, power plan and thermals all move the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
out="${2:-results/sweep-$(date +%Y%m%d-%H%M%S)}"
models="${MODELS_DIR:-models}"
bin="${DYNALM:-}"
if [[ -z "$bin" ]]; then
  for c in "$(command -v dynalm || true)" build/msvc-release/bin/dynalm.exe build/linux-release/bin/dynalm \
           build/macos-release/bin/dynalm; do
    if [[ -n "$c" && -x "$c" ]]; then bin="$c"; break; fi
  done
fi
[[ -x "$bin" ]] || { echo "dynalm binary not found (set DYNALM=...)" >&2; exit 1; }
mkdir -p "$out"
jsonl="$out/sweep.jsonl"

{
  echo "date: $(date -Iseconds)"
  echo "binary: $bin"
  "$bin" version
  "$bin" info
  echo "mode: $mode"
} > "$out/environment.txt" 2>&1

bench() {  # model-file, then benchmark args
  local m="$models/$1"
  shift
  if [[ ! -f "$m" ]]; then
    echo "skip: $m (absent)" | tee -a "$out/environment.txt"
    return
  fi
  echo "== $(basename "$m") $*"
  "$bin" --log-level warn benchmark "$m" --out "$jsonl" "$@" | tee -a "$out/console.txt"
}

small=qwen2.5-0.5b-instruct-q4_k_m.gguf
if [[ "$mode" == quick ]]; then
  bench "$small" --concurrency 1,4,16 --prompt 128 --output 64 -c 16384
  bench "$small" --concurrency 1 --prompt 512,2048 --output 64 --requests 2 -c 16384
  bench granite-3.1-1b-a400m-instruct-Q4_K_M.gguf --concurrency 1,8 --prompt 128 --output 64 -c 16384
else
  # 1. Small dense model: the full concurrency curve (the headline graph).
  bench "$small" --concurrency 1,2,4,8,16,32,64 --prompt 128 --output 64 -c 32768
  # 2. Context length at single stream and moderate concurrency.
  bench "$small" --concurrency 1 --prompt 512,2048,4096 --output 64 --requests 2 -c 32768
  bench "$small" --concurrency 8 --prompt 512,2048 --output 64 -c 65536
  # 3. Mixed prompt lengths under load.
  bench "$small" --concurrency 8,32 --prompt-mix 64,512,2048 --output 64 -c 65536
  # 4. Quantization formats of the same model.
  bench qwen2.5-0.5b-instruct-q8_0.gguf --concurrency 1,8 --prompt 128 --output 64 -c 16384
  bench qwen2.5-0.5b-instruct-fp16.gguf --concurrency 1,8 --prompt 128 --output 64 -c 16384
  # 5. Medium and larger dense models.
  bench qwen2.5-1.5b-instruct-q4_k_m.gguf --concurrency 1,4,16 --prompt 128 --output 64 -c 16384
  bench Qwen3-4B-Q4_K_M.gguf --concurrency 1,4 --prompt 128 --output 32 -c 8192
  # 6. Mixture of experts.
  bench granite-3.1-1b-a400m-instruct-Q4_K_M.gguf --concurrency 1,4,16,64 --prompt 128 --output 64 -c 32768
fi

PYTHONUTF8=1 python tools/bench_report.py --svg "$out/graphs" "$jsonl" > "$out/report.md"
echo "report: $out/report.md"

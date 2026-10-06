#!/usr/bin/env bash
# Head-to-head benchmark: this engine vs llama.cpp's llama-server, same model,
# same thread count, same container environment, same load generator (HTTP,
# streaming, ignore_eos for fixed output lengths).
#
#   bash tools/compare_baselines.sh [model.gguf] [threads] [out.jsonl]
#
# Requires the dynalm-dev image + linux-release build (tools/linux.sh) and
# ghcr.io/ggml-org/llama.cpp:server. Servers run one at a time (memory).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5-0.5b-instruct-q4_k_m.gguf}"
THREADS="${2:-10}"
OUT="${3:-results/baselines.jsonl}"
CONC="${CONC:-1,4,16}"
PROMPTS="${PROMPTS:-128,512}"
OUTPUTS="${OUTPUTS:-128}"
SRC="$(pwd -W 2>/dev/null || pwd)"
mkdir -p "$(dirname "$OUT")"

wait_health() {
  for _ in $(seq 1 120); do
    curl -s -m 2 "http://127.0.0.1:$1/health" | grep -q ok && return 0
    sleep 1
  done
  echo "server on port $1 did not become healthy" >&2
  return 1
}

bench() {  # $1 = port, $2 = label
  MSYS_NO_PATHCONV=1 docker run --rm --add-host=host.docker.internal:host-gateway \
    -v "$SRC:/src" -v dynalm-linux-build:/src/build dynalm-dev \
    /src/build/linux-release/bin/dynalm benchmark "/src/models/$MODEL" --url "http://host.docker.internal:$1" \
    --concurrency "$CONC" --prompt "$PROMPTS" --output "$OUTPUTS" --out "/src/$OUT.$2"
  # Label rows by target name for the report.
  python - "$OUT.$2" "$2" "$OUT" <<'PY'
import json, sys
src, label, dst = sys.argv[1:]
with open(src) as f, open(dst, "a") as out:
    for line in f:
        r = json.loads(line); r["label"] = label; out.write(json.dumps(r) + "\n")
PY
  rm -f "$OUT.$2"
}

echo "== DynaLM (threads $THREADS)"
docker rm -f bench-dynalm >/dev/null 2>&1 || true
MSYS_NO_PATHCONV=1 docker run -d --name bench-dynalm -p 127.0.0.1:8000:8000 -v "$SRC:/src" -v dynalm-linux-build:/src/build \
  dynalm-dev /src/build/linux-release/bin/dynalm --log-level warn serve "/src/models/$MODEL" --host 0.0.0.0 --port 8000 \
  -t "$THREADS" --ctx 32768 >/dev/null
wait_health 8000
bench 8000 "dynalm"
docker rm -f bench-dynalm >/dev/null

echo "== llama.cpp llama-server (threads $THREADS, 16 slots)"
docker rm -f bench-llamacpp >/dev/null 2>&1 || true
MSYS_NO_PATHCONV=1 docker run -d --name bench-llamacpp -p 127.0.0.1:8081:8080 -v "$SRC/models:/models" \
  ghcr.io/ggml-org/llama.cpp:server -m "/models/$MODEL" --host 0.0.0.0 --port 8080 -t "$THREADS" -tb "$THREADS" \
  -c 32768 -np 16 >/dev/null
wait_health 8081
bench 8081 "llama.cpp"
docker logs bench-llamacpp 2>&1 | grep -m1 -E "build|version" || true
docker rm -f bench-llamacpp >/dev/null

PYTHONUTF8=1 python tools/bench_report.py "$OUT"

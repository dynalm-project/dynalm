#!/usr/bin/env bash
# Benchmark a locally running Ollama (native install, port 11434) with the same
# load generator as tools/compare_baselines.sh. Register the same GGUF first:
#
#   printf 'FROM ./qwen2.5-0.5b-instruct-q4_k_m.gguf\n' > models/Modelfile.q4km
#   (cd models && ollama create cpu-llama-qwen05-q4km -f Modelfile.q4km)
#   bash tools/bench_ollama.sh [model.gguf] [ollama-model] [out.jsonl]
#
# Caveats (DD-037): Ollama runs natively, not in the benchmark container; it
# ignores ignore_eos (see mean_completion_tokens); its parallelism and thread
# count are its own defaults (OLLAMA_NUM_PARALLEL, physical cores).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5-0.5b-instruct-q4_k_m.gguf}"
OLLAMA_MODEL="${2:-cpu-llama-qwen05-q4km}"
OUT="${3:-results/ollama.jsonl}"
CONC="${CONC:-1,4,16}"
PROMPTS="${PROMPTS:-128,512}"
OUTPUTS="${OUTPUTS:-128}"
SRC="$(pwd -W 2>/dev/null || pwd)"
mkdir -p "$(dirname "$OUT")"
curl -s -m 3 http://127.0.0.1:11434/api/version >/dev/null || { echo "ollama is not running" >&2; exit 1; }
MSYS_NO_PATHCONV=1 docker run --rm --add-host=host.docker.internal:host-gateway \
  -v "$SRC:/src" -v dynalm-linux-build:/src/build dynalm-dev \
  /src/build/linux-release/src/dynalm benchmark "/src/models/$MODEL" --url "http://host.docker.internal:11434" \
  --model-name "$OLLAMA_MODEL" --concurrency "$CONC" --prompt "$PROMPTS" --output "$OUTPUTS" --out "/src/$OUT.tmp"
python - "$OUT.tmp" "$OUT" <<'PY'
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "a") as out:
    for line in f:
        r = json.loads(line); r["label"] = "ollama (native)"; out.write(json.dumps(r) + "\n")
PY
rm -f "$OUT.tmp"

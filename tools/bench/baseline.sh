#!/usr/bin/env bash
# Controlled engine comparison (Phase 0 baseline): DynaLM vs llama.cpp vs
# Ollama on the same GGUF files, measured by the same client.
#
#   tools/bench/baseline.sh --out DIR --models 1.5b,8b --devices gpu --engines dynalm,llama,ollama [--quick]
#
# Run time: about 1 h for the GPU half and 3+ h for the CPU half of the
# default 1.5B/8B GPU + 1.5B/4B CPU set (36 server starts: 3 engines x 3
# rounds x 4 model/device pairs; a 2,048-token prompt on a 4B CPU model takes
# ~1 min). --quick (about 45 min for the same set): 2 rounds, prompts 128 and
# 512, GPU users 4 and 16, CPU users 4, fewer requests per point. Its verdicts
# need every round to agree, so 2 rounds give fewer "better"/"slower" calls.
#
# Contamination controls:
#  - Every binary is copied into DIR/bin first and only the copies run, so a
#    rebuild during the run cannot change what is measured. Checksums and
#    versions go to DIR/manifest.txt.
#  - Before each server start: wait for AC power with Energy Mode Automatic
#    (pmset powermode 0) and a 1-minute load average below MAX_LOAD. Every
#    wait and the state at each start are logged (DIR/conditions.log).
#  - One server at a time; rounds rotate the engine order.
# Same settings for every engine: per-user context CTX, slots = the highest
# concurrency, THREADS compute threads, identical prompts and request counts.
# Requests that fail, or end early (fewer tokens than asked with ignore_eos),
# are recorded, not hidden (the client counts error events in the stream).
set -uo pipefail

SERVER=${SERVER:?SERVER=path to the DynaLM binary under test}
CLIENT=${CLIENT:?CLIENT=path to the DynaLM binary used as the load generator}
LLAMA=${LLAMA:-}
OLLAMA=${OLLAMA:-$(command -v ollama || true)}
MODELS_DIR=${MODELS_DIR:-$HOME/.dynalm/models}
THREADS=${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)}
CTX=${CTX:-4096}
MAX_LOAD=${MAX_LOAD:-3}
OUT=""; MODELS="1.5b"; DEVICES="gpu"; ENGINES="dynalm llama ollama"; ROUNDS=3
SINGLE_PROMPTS=128,512,2048; OUTPUT=128; MULTI_PROMPT=512
GPU_CONC=4,8,16; CPU_CONC=4,8
SINGLE_REQ=4; MULTI_MIN=16; MULTI_PER_USER=4

die() { echo "error: $*" >&2; exit 1; }
while (($#)); do
  case $1 in
    --out) OUT=$2; shift ;;
    --models) MODELS=${2//,/ }; shift ;;
    --devices) DEVICES=${2//,/ }; shift ;;
    --engines) ENGINES=${2//,/ }; shift ;;
    --rounds) ROUNDS=$2; shift ;;
    --quick) ROUNDS=2; SINGLE_PROMPTS=128,512; GPU_CONC=4,16; CPU_CONC=4; SINGLE_REQ=3; MULTI_MIN=8; MULTI_PER_USER=2 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ -n $OUT ]] || die "--out DIR is required"
[[ ! -e $OUT/manifest.txt ]] || die "$OUT already holds a run"

model_file() {
  case $1 in
    1.5b) echo "$MODELS_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" ;;
    4b) echo "$MODELS_DIR/Qwen3-4B-Q4_K_M.gguf" ;;
    8b) echo "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" ;;
    *) die "unknown model $1" ;;
  esac
}

mkdir -p "$OUT/bin"
cp "$SERVER" "$OUT/bin/dynalm-server" && cp "$CLIENT" "$OUT/bin/dynalm-client" || die "cannot copy binaries"
SRV=$OUT/bin/dynalm-server; CLI=$OUT/bin/dynalm-client
if [[ " $ENGINES " == *" llama "* ]]; then
  [[ -x $LLAMA ]] || die "LLAMA=path to llama-server is required for the llama engine"
  cp "$LLAMA" "$OUT/bin/llama-server"
  # llama-server loads its ggml libraries from beside the binary (rpath).
  cp "$(dirname "$LLAMA")"/*.dylib "$OUT/bin/" 2>/dev/null
  LLAMA=$OUT/bin/llama-server
fi
for m in $MODELS; do [[ -s $(model_file "$m") ]] || die "model $m missing: $(model_file "$m")"; done

{
  echo "date: $(date -u +%FT%TZ)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB, $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2" GPU cores"; exit}')"
  echo "os: macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "threads: $THREADS  ctx per user: $CTX  rounds: $ROUNDS  max load: $MAX_LOAD"
  echo "single: concurrency 1, prompts $SINGLE_PROMPTS, output $OUTPUT, $SINGLE_REQ requests, 1 warm-up"
  echo "multi: gpu concurrency $GPU_CONC / cpu $CPU_CONC, prompt $MULTI_PROMPT, output $OUTPUT, max($MULTI_MIN, $MULTI_PER_USER x concurrency) requests, 1 warm-up"
  echo "dynalm server: $(shasum -a 256 "$SRV" | cut -c1-16) $("$SRV" --version | tr '\n' ' ')"
  echo "dynalm client: $(shasum -a 256 "$CLI" | cut -c1-16) $("$CLI" --version | head -1)"
  [[ " $ENGINES " == *" llama "* ]] && echo "llama.cpp: $(shasum -a 256 "$LLAMA" | cut -c1-16) commit ${LLAMA_COMMIT:-unknown}"
  [[ " $ENGINES " == *" ollama "* ]] && echo "ollama: $("$OLLAMA" --version 2>&1 | tail -1)"
  for m in $MODELS; do echo "model $m: $(basename "$(model_file "$m")") $(shasum -a 256 "$(model_file "$m")" | cut -c1-16)"; done
} | tee "$OUT/manifest.txt"

# Waits for AC power, Energy Mode Automatic and a quiet machine; logs it.
gate() {  # tag
  local waited=0
  while :; do
    local batt mode load
    batt=$(pmset -g batt | head -1)
    mode=$(pmset -g | awk '/ powermode /{print $2}')
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if [[ $batt == *"AC Power"* && $mode == 0 ]] && awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l < m)}'; then
      echo "$(date +%T) $1 start: AC, powermode $mode, load $load (waited ${waited}s)" >> "$OUT/conditions.log"
      return
    fi
    ((waited % 60 == 0)) && echo "$(date +%T) $1 waiting: $batt | powermode $mode | load $load" | tee -a "$OUT/conditions.log"
    sleep 10; waited=$((waited + 10))
  done
}

wait_up() { for _ in $(seq 1 600); do curl -sf "$1" >/dev/null 2>&1 && return 0; kill -0 "$2" 2>/dev/null || return 1; sleep 0.5; done; return 1; }

stop_server() {  # time-pid
  local child; child=$(pgrep -P "$1" | head -1)
  [[ -n $child ]] && kill -INT "$child" 2>/dev/null
  for _ in $(seq 1 120); do kill -0 "$1" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$1" 2>/dev/null && [[ -n $child ]] && kill -KILL "$child" 2>/dev/null
  wait "$1" 2>/dev/null
}

run_one() {  # engine device model round
  local eng=$1 dev=$2 key=$3 r=$4 model; model=$(model_file "$key")
  local conc=$([[ $dev == gpu ]] && echo $GPU_CONC || echo $CPU_CONC)
  local n=${conc##*,} tag="$key-$dev-$eng-r$r" port=8460 pid url name oenv=()
  local log="$OUT/$tag.server.log"
  gate "$tag"
  case $eng in
    dynalm)
      /usr/bin/time -l "$SRV" serve "$model" --backend "$([[ $dev == gpu ]] && echo metal || echo cpu)" --port $port \
        --threads "$THREADS" --ctx $((CTX * n)) >"$log" 2>&1 &
      pid=$!; url=http://127.0.0.1:$port; name=$(basename "$model") ;;
    llama)
      /usr/bin/time -l "$LLAMA" -m "$model" -ngl "$([[ $dev == gpu ]] && echo 99 || echo 0)" -t "$THREADS" -c $((CTX * n)) -np "$n" \
        -fa auto --port $port --no-webui >"$log" 2>&1 &
      pid=$!; url=http://127.0.0.1:$port; name=default ;;
    ollama)
      port=11435
      oenv=(OLLAMA_HOST=127.0.0.1:$port OLLAMA_MODELS=$OUT/ollama-models OLLAMA_NUM_PARALLEL=$n OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m)
      env "${oenv[@]}" /usr/bin/time -l "$OLLAMA" serve >"$log" 2>&1 &
      pid=$!; url=http://127.0.0.1:$port; name=bench-$key-$dev ;;
  esac
  local health=$url/health; [[ $eng == ollama ]] && health=$url/api/version
  if ! wait_up "$health" "$pid"; then
    echo "$tag: server did not start: $(grep -m1 -iE 'unsupported|error' "$log")" | tee -a "$OUT/errors.txt"
    stop_server "$pid"; return
  fi
  if [[ $eng == ollama ]]; then
    { echo "FROM $model"; echo 'TEMPLATE "{{ .Prompt }}"'; echo "PARAMETER num_ctx $CTX"; echo "PARAMETER num_thread $THREADS"
      [[ $dev == cpu ]] && echo "PARAMETER num_gpu 0"; } >"$OUT/$tag.Modelfile"
    env "${oenv[@]}" "$OLLAMA" create "$name" -f "$OUT/$tag.Modelfile" >"$OUT/$tag.create.log" 2>&1 ||
      { echo "$tag: ollama create failed" | tee -a "$OUT/errors.txt"; stop_server "$pid"; return; }
  fi
  "$CLI" benchmark "$model" --url "$url" --model-name "$name" --concurrency 1 --prompt "$SINGLE_PROMPTS" --output $OUTPUT \
    --requests $SINGLE_REQ --warmup-requests 1 --request-log "$OUT/$tag.requests.jsonl" --out "$OUT/$tag.jsonl" >"$OUT/$tag.single.txt" 2>&1 ||
    echo "$tag: single-user client exit $?" | tee -a "$OUT/errors.txt"
  local c
  for c in ${conc//,/ }; do
    "$CLI" benchmark "$model" --url "$url" --model-name "$name" --concurrency "$c" --prompt $MULTI_PROMPT --output $OUTPUT \
      --requests $((c * MULTI_PER_USER < MULTI_MIN ? MULTI_MIN : c * MULTI_PER_USER)) --warmup-requests 1 --request-log "$OUT/$tag.requests.jsonl" --out "$OUT/$tag.jsonl" \
      >>"$OUT/$tag.multi.txt" 2>&1 || echo "$tag: c=$c client exit $?" | tee -a "$OUT/errors.txt"
  done
  [[ $eng == ollama ]] && env "${oenv[@]}" "$OLLAMA" rm "$name" >/dev/null 2>&1
  stop_server "$pid"
  sleep 5
}

read -r -a arms <<<"$ENGINES"
for m in $MODELS; do
  for d in $DEVICES; do
    for r in $(seq 1 "$ROUNDS"); do
      k=$(((r - 1) % ${#arms[@]})); order=("${arms[@]:$k}" "${arms[@]:0:$k}")
      for e in "${order[@]}"; do echo "== $(date +%T) $m $d r$r: $e"; run_one "$e" "$d" "$m" "$r"; done
    done
  done
done
echo "done: $OUT"

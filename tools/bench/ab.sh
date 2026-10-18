#!/usr/bin/env bash
# Paired A/B runs of two DynaLM builds (or one build under two settings) for
# tools/bench_analyze.py: the in-process engine (`dynalm benchmark`), the same
# workload points for both arms, arm order alternated every round.
#
#   A=<baseline dynalm> B=<candidate dynalm> tools/bench/ab.sh --out DIR --model FILE \
#     [--backend metal] [--conc 1,4,8,16] [--prompt 512] [--output 128] \
#     [--sessions 2] [--rounds 3] [--workload gpu512]
#   python3 tools/bench_analyze.py tools/bench/plans/phase1-gpu.json DIR/raw --out DIR/report
#
# Optional per-arm settings: A_ENV / B_ENV, space-separated NAME=VALUE pairs
# (e.g. B_ENV="DYNALM_DECODE_TARGET=12 DYNALM_DECODE_PREDICTIVE=0" runs the
# candidate binary with the earlier GPU scheduling profile).
#
# Contamination controls (as tools/bench/baseline.sh): the binaries are copied
# into DIR/bin and only the copies run; each run waits for AC power, Energy
# Mode Automatic (pmset powermode 0) and a 1-minute load below MAX_LOAD
# (macOS; elsewhere only the load is checked). A run that started in other
# conditions is recorded with quiet=false, and bench_analyze.py drops it.
# ALLOW_NOISY=1 starts runs without waiting (to check the setup only).
set -uo pipefail

A=${A:?A=path to the baseline dynalm binary}
B=${B:?B=path to the candidate dynalm binary}
A_ENV=${A_ENV:-}
B_ENV=${B_ENV:-}
MAX_LOAD=${MAX_LOAD:-3}
OUT=""; MODEL=""; BACKEND=metal; CONC=1,4,8,16; PROMPT=512; OUTPUT=128; SESSIONS=2; ROUNDS=3; WORKLOAD=""
die() { echo "error: $*" >&2; exit 1; }
while (($#)); do
  case $1 in
    --out) OUT=$2; shift ;;
    --model) MODEL=$2; shift ;;
    --backend) BACKEND=$2; shift ;;
    --conc) CONC=$2; shift ;;
    --prompt) PROMPT=$2; shift ;;
    --output) OUTPUT=$2; shift ;;
    --sessions) SESSIONS=$2; shift ;;
    --rounds) ROUNDS=$2; shift ;;
    --workload) WORKLOAD=$2; shift ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ -n $OUT && -n $MODEL ]] || die "--out DIR and --model FILE are required"
[[ -s $MODEL ]] || die "model not found: $MODEL"
[[ ! -e $OUT/raw/runs.jsonl ]] || die "$OUT already holds a run"
WORKLOAD=${WORKLOAD:-$BACKEND$PROMPT}
[[ $WORKLOAD =~ ^[A-Za-z0-9_.]+$ ]] || die "--workload: letters, digits, '_' and '.' only"

mkdir -p "$OUT/bin" "$OUT/raw"
cp "$A" "$OUT/bin/dynalm-a" && cp "$B" "$OUT/bin/dynalm-b" || die "cannot copy binaries"
sha() { shasum -a 256 "$1" | cut -c1-64; }
SHA_A=$(sha "$OUT/bin/dynalm-a"); SHA_B=$(sha "$OUT/bin/dynalm-b")
{
  echo "date: $(date -u +%FT%TZ)"
  if [[ $(uname) == Darwin ]]; then
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
    echo "os: macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  else
    echo "machine: $(uname -m), $(nproc) CPUs"; echo "os: $(uname -sr)"
  fi
  echo "model: $(basename "$MODEL") $(sha "$MODEL" | cut -c1-16)"
  echo "workload: $WORKLOAD backend $BACKEND conc $CONC prompt $PROMPT output $OUTPUT sessions $SESSIONS rounds $ROUNDS"
  echo "arm a: $SHA_A env [$A_ENV] $("$OUT/bin/dynalm-a" --version | head -1)"
  echo "arm b: $SHA_B env [$B_ENV] $("$OUT/bin/dynalm-b" --version | head -1)"
} | tee "$OUT/manifest.txt"
# The plan's expected_exe for this run (sha256 prefixes).
echo "{\"a\": \"${SHA_A:0:16}\", \"b\": \"${SHA_B:0:16}\"}" > "$OUT/expected_exe.json"

quiet_now() {  # prints the state; exit 0 when quiet
  local load mode batt
  load=$(uptime | awk -F'load averages?: ' '{print $2}' | awk -F'[ ,]+' '{print $1}')
  if [[ $(uname) == Darwin ]]; then
    batt=$(pmset -g batt | head -1); mode=$(pmset -g | awk '/ powermode /{print $2}')
    echo "$batt | powermode ${mode:-?} | load $load"
    [[ $batt == *"AC Power"* && ${mode:-0} == 0 ]] || return 1
  else
    echo "load $load"
  fi
  awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l < m)}'
}
gate() {
  local waited=0 state
  # ALLOW_NOISY=1: do not wait (smoke tests); such runs are recorded quiet=false.
  if [[ ${ALLOW_NOISY:-0} == 1 ]]; then
    echo "$(date +%T) $1 start without waiting: $(quiet_now)" >> "$OUT/conditions.log"
    return
  fi
  until state=$(quiet_now); do
    ((waited % 60 == 0)) && echo "$(date +%T) $1 waiting: $state" | tee -a "$OUT/conditions.log"
    sleep 10; waited=$((waited + 10))
  done
  echo "$(date +%T) $1 start: $state (waited ${waited}s)" >> "$OUT/conditions.log"
}

run_arm() {  # arm id
  local arm=$1 id=$2 exe envs sha quiet=true code=0 c
  exe=$OUT/bin/dynalm-$arm
  if [[ $arm == a ]]; then envs=$A_ENV; sha=$SHA_A; else envs=$B_ENV; sha=$SHA_B; fi
  gate "$id"
  for c in ${CONC//,/ }; do
    # shellcheck disable=SC2086  # envs is a list of NAME=VALUE words
    env $envs "$exe" benchmark "$MODEL" --backend "$BACKEND" --concurrency "$c" --prompt "$PROMPT" --output "$OUTPUT" \
      --requests $((c * 4 < 16 ? 16 : c * 4)) --warmup-requests 1 \
      --out "$OUT/raw/$id.jsonl" --request-log "$OUT/raw/$id.requests.jsonl" >>"$OUT/raw/$id.log" 2>&1 || code=$?
  done
  # Conditions changed during the run: power or power mode (not the load,
  # which the benchmark itself raises; it is checked before the start).
  if [[ $(uname) == Darwin ]]; then
    pmset -g batt | head -1 | grep -q 'AC Power' && [[ $(pmset -g | awk '/ powermode /{print $2}') == 0 ]] || quiet=false
  fi
  printf '{"id": "%s", "exit_code": %d, "quiet": %s, "exe_sha256": "%s", "env": "%s"}\n' \
    "$id" "$code" "$quiet" "$sha" "$envs" >> "$OUT/raw/runs.jsonl"
}

for s in $(seq 1 "$SESSIONS"); do
  for r in $(seq 1 "$ROUNDS"); do
    if (((s + r) % 2 == 0)); then order="a b"; else order="b a"; fi
    for arm in $order; do
      echo "== $(date +%T) session $s round $r: arm $arm"
      run_arm "$arm" "$WORKLOAD-s${s}r${r}-$arm"
    done
  done
done
echo "done: $OUT/raw (expected_exe for the plan: $(cat "$OUT/expected_exe.json"))"

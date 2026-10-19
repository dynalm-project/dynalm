#!/usr/bin/env bash
# End-to-end runs and benchmarks, unattended, one phase at a time (never two
# measurements at once), then one report:
#
#   tools/bench/e2e.sh --out test/results/e2e-<name> [--tier2] [--skip matrix,engines,ab]
#
#   1. snapshot  every binary is copied into OUT/bin first (DynaLM under test, its
#                accuracy tool, DynaLM v0.1.7 for the A/B, llama-server, llama-bench),
#                with checksums and versions in OUT/manifest.txt: a rebuild during
#                the run cannot change what is measured.
#   2. matrix    tools/bench/model_matrix.py on models-e2e-tier1.txt (--tier2: also
#                models-providers.txt): download, generate on auto / cpu / native-cpu,
#                cross-path agreement, DD-053 GPU accuracy, and speed (DynaLM in-process
#                vs llama-bench on the same GGUF: GPU, fast CPU and exact-mode CPU,
#                --backend native-cpu). Speed runs wait for AC
#                power, Energy Mode Automatic and a quiet machine (--gate).
#   3. engines   tools/bench/baseline.sh --quick: DynaLM vs llama.cpp vs Ollama servers,
#                1 to 16 users, GPU (1.5B, 8B) and CPU (1.5B, 4B), rotated rounds, gated.
#   4. ab        tools/bench/ab.sh: v0.1.7 vs this build on the same model, GPU
#                (plans/gpu-regression.json: no regression, single-user
#                equivalence) and CPU (plans/phase2-cpu.json: native vs ggml CPU
#                kernels); a discarded warm-up round, then 2 sessions x 4 rounds
#                (8 pairs per point, 95% CI); analyzed by tools/bench_analyze.py.
#   5. report    tools/bench/e2e_report.py -> OUT/REPORT.md.
#
# Inputs (defaults in parentheses):
#   DYNALM (build/macos-metal/bin/dynalm), ACC (its bench_gpu_accuracy),
#   OLD_DYNALM (a v0.1.7 dynalm; ab skipped if missing), LLAMA_DIR (dir with
#   llama-server and llama-bench), LLAMA_COMMIT, HF_TOKEN (gated models).
# Time (3 MB/s link, M4 Pro): matrix tier 1 ~7-8 h (mostly downloads; exact-mode speed
# adds ~0.5-1 h: the native CPU kernels are ~2-3x slower than ggml's), engines ~1 h,
# ab ~2 h (18 runs per device; was ~40 min with 1 session x 3 rounds, too few for a CI). A phase that fails is recorded and the next one runs.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=""; TIER2=0; SKIP=""
while (($#)); do
  case $1 in
    --out) OUT=$2; shift ;;
    --tier2) TIER2=1 ;;
    --skip) SKIP=",$2,"; shift ;;
    *) echo "error: unknown option $1" >&2; exit 1 ;;
  esac
  shift
done
[[ -n $OUT ]] || { echo "error: --out DIR is required" >&2; exit 1; }
[[ ! -e $OUT/manifest.txt ]] || { echo "error: $OUT already holds a run" >&2; exit 1; }
DYNALM=${DYNALM:-$ROOT/build/macos-metal/bin/dynalm}
ACC=${ACC:-$ROOT/build/macos-metal/dynalm/benchmarks/bench_gpu_accuracy}
OLD_DYNALM=${OLD_DYNALM:-}
LLAMA_DIR=${LLAMA_DIR:-}
MODELS_DIR=${MODELS_DIR:-$HOME/.dynalm/models}
skip() { [[ $SKIP == *",$1,"* ]]; }
mkdir -p "$OUT/bin"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/e2e.log"; }
phase_status() { echo "$1 $2" >> "$OUT/phases.txt"; log "phase $1: $2"; }

# --- 1. snapshot ----------------------------------------------------------------
cp "$DYNALM" "$OUT/bin/dynalm" || { echo "error: no DYNALM at $DYNALM" >&2; exit 1; }
[[ -x $ACC ]] && cp "$ACC" "$OUT/bin/bench_gpu_accuracy"
[[ -n $OLD_DYNALM && -x $OLD_DYNALM ]] && cp "$OLD_DYNALM" "$OUT/bin/dynalm-old"
if [[ -n $LLAMA_DIR ]]; then
  for f in llama-server llama-bench; do [[ -x $LLAMA_DIR/$f ]] && cp "$LLAMA_DIR/$f" "$OUT/bin/"; done
  cp "$LLAMA_DIR"/*.dylib "$LLAMA_DIR"/*.so* "$OUT/bin/" 2>/dev/null  # llama.cpp loads ggml from beside the binary
fi
BIN=$OUT/bin
sha() { shasum -a 256 "$1" | cut -c1-16; }
{
  echo "date: $(date -u +%FT%TZ)"
  if [[ $(uname) == Darwin ]]; then
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB," \
         "$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2" GPU cores"; exit}')"
    echo "os: macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  else
    echo "machine: $(uname -m), $(nproc) CPUs"; echo "os: $(uname -sr)"
  fi
  echo "dynalm: $(sha "$BIN/dynalm") $("$BIN/dynalm" --version | head -1) (git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo '+local changes'))"
  [[ -x $BIN/dynalm-old ]] && echo "dynalm (A/B baseline): $(sha "$BIN/dynalm-old") $("$BIN/dynalm-old" --version | head -1)"
  [[ -x $BIN/llama-server ]] && echo "llama.cpp: commit ${LLAMA_COMMIT:-unknown}, llama-server $(sha "$BIN/llama-server"), llama-bench $(sha "$BIN/llama-bench" 2>/dev/null)"
  command -v ollama >/dev/null && echo "ollama: $(ollama --version 2>&1 | tail -1)"
  echo "HF_TOKEN: $([[ -n ${HF_TOKEN:-} ]] && echo set || echo 'not set (gated models are reported, not run)')"
  echo "phases: matrix$( ((TIER2)) && echo '+tier2'), engines, ab (skipped: ${SKIP:-none})"
} | tee "$OUT/manifest.txt"
"$BIN/dynalm" --version > "$OUT/dynalm-version.txt"

# --- 2. matrix ------------------------------------------------------------------
if ! skip matrix; then
  log "phase matrix: start"
  lists=("$ROOT/tools/bench/models-e2e-tier1.txt"); ((TIER2)) && lists+=("$ROOT/tools/bench/models-providers.txt")
  status=ok
  for l in "${lists[@]}"; do
    name=$(basename "$l" .txt)
    DYNALM=$BIN/dynalm ACC=$BIN/bench_gpu_accuracy LLAMA_BENCH=$BIN/llama-bench \
      python3 "$ROOT/tools/bench/model_matrix.py" --out "$OUT/matrix-$name" --list "$l" --accuracy --bench --gate \
      >"$OUT/matrix-$name.log" 2>&1 || status="failed ($name)"
  done
  phase_status matrix "$status"
fi

# --- 3. engines -----------------------------------------------------------------
if ! skip engines; then
  log "phase engines: start"
  status=ok
  if [[ -x $BIN/llama-server ]]; then
    for set in "gpu 1.5b,8b" "cpu 1.5b,4b"; do
      read -r dev models <<<"$set"
      SERVER=$BIN/dynalm CLIENT=$BIN/dynalm LLAMA=$BIN/llama-server LLAMA_COMMIT=${LLAMA_COMMIT:-unknown} MODELS_DIR=$MODELS_DIR \
        bash "$ROOT/tools/bench/baseline.sh" --quick --out "$OUT/engines/$dev" --models "$models" --devices "$dev" \
        --engines "dynalm,llama$(command -v ollama >/dev/null && echo ,ollama)" >"$OUT/engines-$dev.log" 2>&1 ||
        status="failed ($dev)"
    done
  else
    status="skipped (no llama-server in LLAMA_DIR)"
  fi
  phase_status engines "$status"
fi

# --- 4. ab ----------------------------------------------------------------------
if ! skip ab; then
  log "phase ab: start"
  status=ok
  M=$MODELS_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf
  if [[ -x $BIN/dynalm-old && -s $M ]]; then
    # GPU: a regression guard (no GPU scheduling change exists between the two
    # builds; plans/gpu-regression.json). CPU: v0.1.7's native kernels vs the
    # ggml default. 2 sessions x 4 rounds after a discarded warm-up round: 8
    # pairs per point, enough for the 95% CI the plans require (require_ci).
    for set in "gpu metal 1,4,8,16 gpu-regression" "cpu cpu 1,4 phase2-cpu"; do
      read -r dev backend conc plan <<<"$set"
      A=$BIN/dynalm-old B=$BIN/dynalm bash "$ROOT/tools/bench/ab.sh" --out "$OUT/ab/$dev" --model "$M" \
        --backend "$backend" --conc "$conc" --prompt 512 --output 128 --sessions 2 --rounds 4 --warmup-rounds 1 \
        --workload "${dev}512" \
        >"$OUT/ab-$dev.log" 2>&1 || status="failed ($dev)"
      # The plan with this run's executables (sha256 prefixes) filled in.
      python3 - "$ROOT/tools/bench/plans/$plan.json" "$OUT/ab/$dev/expected_exe.json" "$OUT/ab/$dev/plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
try:
    plan["expected_exe"] = json.load(open(sys.argv[2]))
except OSError:
    pass
json.dump(plan, open(sys.argv[3], "w"), indent=2)
PY
      python3 "$ROOT/tools/bench_analyze.py" "$OUT/ab/$dev/plan.json" "$OUT/ab/$dev/raw" --out "$OUT/ab/$dev/report" \
        >>"$OUT/ab-$dev.log" 2>&1 || status="failed ($dev analysis)"
    done
  else
    status="skipped (needs OLD_DYNALM and $M)"
  fi
  phase_status ab "$status"
fi

# --- 5. report ------------------------------------------------------------------
python3 "$ROOT/tools/bench/e2e_report.py" "$OUT" && log "report: $OUT/REPORT.md"

#!/usr/bin/env bash
# Live view of a tools/bench/e2e.sh run, refreshed every few seconds:
#   tools/bench/e2e_watch.sh [RUN DIR]        (default: the newest test/results/e2e-*)
#   tools/bench/e2e_watch.sh RUN_DIR --once   (print once and exit)
# Shows: power and load, phase status, the current model and its last steps,
# downloads in progress (size and speed), finished models, and the newest
# lines of the engine / A/B phases. Read-only: it never touches the run.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:-$(ls -dt "$ROOT"/test/results/e2e-*/ 2>/dev/null | head -1)}
RUN=${RUN%/}
ONCE=0; [[ ${2:-} == --once ]] && ONCE=1
[[ -d $RUN ]] || { echo "no run directory (give one: tools/bench/e2e_watch.sh test/results/e2e-...)"; exit 1; }
STORE=${DYNALM_MODELS_DIR:-$HOME/.dynalm/models}
prev_bytes=0; prev_t=0

part_bytes() {  # total bytes of downloads in progress
  local t=0 f
  for f in "$STORE"/*.part "$RUN"/matrix-*/hf/*/*.part; do
    [[ -f $f ]] && t=$((t + $(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")))
  done
  echo $t
}

while :; do
  out=$(
    echo "DynaLM end-to-end run: $RUN"
    echo "now: $(date '+%F %T')   power: $(pmset -g batt 2>/dev/null | head -1 | sed "s/Now drawing from //; s/'//g")   load: $(uptime | awk -F'load averages?: ' '{print $2}')"
    echo
    echo "PHASES (done)"
    if [[ -s $RUN/phases.txt ]]; then sed 's/^/  /' "$RUN/phases.txt"; else echo "  (none finished yet)"; fi
    echo "  current: $(tail -1 "$RUN/e2e.log" 2>/dev/null | cut -d' ' -f3-)"
    echo
    mlog=$(ls -t "$RUN"/matrix-*.log 2>/dev/null | head -1)
    if [[ -n $mlog ]]; then
      echo "MODEL MATRIX ($(basename "$mlog" .log))"
      cur=$(grep '^== \[' "$mlog" | tail -1)
      echo "  model: ${cur#== }"
      echo "  last steps:"
      awk '/^== \[/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$mlog" | grep -v '^== \[' | tail -10 | sed 's/^ */    /'
      b=$(part_bytes); t=$(date +%s)
      if ((b > 0)); then
        speed=""
        ((prev_t > 0 && t > prev_t && b >= prev_bytes)) && speed=$(awk -v d=$((b - prev_bytes)) -v s=$((t - prev_t)) 'BEGIN{printf ", %.1f MB/s", d/s/1048576}')
        echo "  downloading: $(awk -v b="$b" 'BEGIN{printf "%.2f GB so far", b/1073741824}')$speed"
      fi
      prev_bytes=$b; prev_t=$t
      tsv=$(ls -t "$RUN"/matrix-*/results.tsv 2>/dev/null | head -1)
      if [[ -n $tsv ]]; then
        total=$(grep -cvE '^\s*(#|$)' "$ROOT/tools/bench/$(basename "$(dirname "$tsv")" | sed 's/^matrix-//').txt" 2>/dev/null)
        done_n=$(($(wc -l <"$tsv") - 1))
        echo
        echo "  finished: $done_n of ${total:-?} models"
        echo "  #   model                                          GPU   fast-CPU exact-CPU  DD-053        GPU decode (llama)   CPU decode (llama)"
        tail -n +2 "$tsv" | tail -8 | awk -F'\t' '{m=$2; if (length(m)>46) m="..." substr(m, length(m)-42);
          printf "  %-3s %-46s %-5s %-8s %-10s %-13s %-20s %s\n", $1, m, $10, $13, $15, $19, $21 " (" $27 ")", $24 " (" $29 ")"}'
      fi
      echo
    fi
    for p in engines-gpu engines-cpu ab-gpu ab-cpu; do
      [[ -s $RUN/$p.log ]] || continue
      echo "$(echo "$p" | tr 'a-z-' 'A-Z ') (last lines)"
      tail -4 "$RUN/$p.log" | sed 's/^/  /'
      echo
    done
    [[ -s $RUN/REPORT.md ]] && echo "REPORT: $RUN/REPORT.md"
  )
  ((ONCE)) && { echo "$out"; exit 0; }
  clear
  echo "$out"
  echo
  echo "(refreshing every 5 s; Ctrl-C to stop watching - the run continues)"
  sleep 5
done

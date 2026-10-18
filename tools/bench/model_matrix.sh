#!/usr/bin/env sh
# Model matrix: see tools/bench/model_matrix.py (it holds the whole runner; macOS
# ships bash 3.2, so the logic lives in Python 3, standard library only).
#   DYNALM=build/macos-metal/bin/dynalm tools/bench/model_matrix.sh --out DIR [--list FILE]
#     [--check-only] [--accuracy] [--bench] [--keep] [--max-gb 16] [--tokens 32]
exec python3 "$(dirname "$0")/model_matrix.py" "$@"

#!/usr/bin/env bash
# Builds a release package and validates it as a fresh install:
#   1. configure + build Release with the static C/C++ runtime (no tests)
#   2. cpack -> dist/dynalm-<os>-<arch>.{tar.gz,zip} + dist/<archive>.sha256
#   3. install it with the user-facing installer (scripts/install.sh or
#      scripts/install.ps1, --archive + checksum) into an empty prefix
#   4. run tests/smoke (CLI and server) against the INSTALLED binaries
#
#   tools/ci/package.sh [extra cmake args...]      (run from the repository root)
#
# Windows: run from a shell with MSVC on PATH (ilammy/msvc-dev-cmd in CI).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD="build/package"
DIST="dist"
rm -rf "$BUILD" "$DIST"
mkdir -p "$DIST"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) WIN=1 ;;
  *) WIN=0 ;;
esac
CXX_ARGS=()
if [ "$WIN" = 1 ]; then CXX_ARGS=(-DCMAKE_CXX_COMPILER=cl); fi

cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF -DENABLE_BENCHMARKS=OFF \
  -DDYNALM_STATIC_RUNTIME=ON ${CXX_ARGS[@]+"${CXX_ARGS[@]}"} "$@"
cmake --build "$BUILD"
(cd "$BUILD" && cpack)

shopt -s nullglob
archives=("$BUILD"/dynalm-*.tar.gz "$BUILD"/dynalm-*.zip)
shopt -u nullglob
[ "${#archives[@]}" -eq 1 ] || { echo "expected one archive from cpack, got: ${archives[*]:-none}" >&2; exit 1; }
ARCHIVE="${archives[0]}"
cp "$ARCHIVE" "$DIST/"
NAME="$(basename "$ARCHIVE")"
( cd "$DIST" && if command -v sha256sum >/dev/null; then sha256sum "$NAME"; else shasum -a 256 "$NAME"; fi ) > "$DIST/$NAME.sha256"
cat "$DIST/$NAME.sha256"

# Fresh install into an empty prefix through the real installer.
PREFIX="$(mktemp -d)/dynalm"
if [ "$WIN" = 1 ]; then
  WPREFIX="$(cygpath -w "$PREFIX")"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install.ps1 \
    -Archive "$(cygpath -w "$DIST/$NAME")" -Sha256Sums "$(cygpath -w "$DIST/$NAME.sha256")" -Prefix "$WPREFIX" -NoPath
  EXE="$PREFIX/bin/dynalm.exe"
  CEXE="$PREFIX/bin/dynacorec.exe"
else
  PLATFORM="${NAME#dynalm-}"
  PLATFORM="${PLATFORM%.tar.gz}"
  sh scripts/install.sh --archive "$DIST/$NAME" --sha256sums "$DIST/$NAME.sha256" --prefix "$PREFIX"     --platform "$PLATFORM"
  EXE="$PREFIX/bin/dynalm"
  CEXE="$PREFIX/bin/dynacorec"
fi

PY="$(command -v python3 || command -v python)"
MODEL="dynalm/tests/data/tiny_llama.gguf"
cmake -DDYNALM="$EXE" -DDYNACOREC="$CEXE" -DMODEL="$MODEL" -DDYNA=examples/dynacore/decoder_layer.dyna \
  -P tests/smoke/cli_smoke.cmake
"$PY" tests/smoke/serve_smoke.py "$EXE" "$MODEL"
"$PY" tests/smoke/serve_smoke.py "$EXE" "$MODEL" --execution compiled
"$PY" tools/ci/check_isa.py "$EXE"
"$EXE" doctor
echo "package ok: $DIST/$NAME"

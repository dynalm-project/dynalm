#!/usr/bin/env bash
# Builds a release package and validates it as a fresh install:
#   1. configure + build Release with the static C/C++ runtime (no tests)
#   2. cpack -> dist/dynalm-<os>-<arch>.{tar.gz,zip} + dist/<archive>.sha256
#   3. install it with the user-facing installer (scripts/install.sh or
#      scripts/install.ps1, --archive + checksum) into an empty prefix
#   4. check the INSTALLED binaries (version, doctor self-test, CPU features)
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

cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=OFF \
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

PY="$(command -v python3 || command -v python)"
# Package contents (docs/release-readiness.md, gate E2): licences present,
# nothing but the binaries and documentation, no secrets or model files.
"$PY" tools/ci/scan_secrets.py "$DIST/$NAME"
"$PY" - "$DIST/$NAME" <<'PYEOF'
import re, sys, tarfile, zipfile
p = sys.argv[1]
names = tarfile.open(p).getnames() if p.endswith(".tar.gz") else zipfile.ZipFile(p).namelist()
names = [n.rstrip("/") for n in names]
need = ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "README.md", "CHANGELOG.md"]
missing = [f for f in need if not any(n.endswith("/share/doc/dynalm/" + f) for n in names)]
allowed = re.compile(r"^dynalm-[a-z]+-[a-z0-9_]+(/bin(/(dynalm|dynacorec)(\.exe)?)?|/share(/doc(/dynalm(/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*)?)?)?)?$")
unexpected = [n for n in names if not allowed.match(n)]
for n in names: print("  ", n)
if missing or unexpected:
    sys.exit(f"package check failed: missing {missing}, unexpected {unexpected}")
print(f"package contents ok: {len(names)} entries, licences present")
PYEOF
SBOM_ARGS=()
for a in "$@"; do  # any GPU backend links ggml
  case "$a" in -DENABLE_METAL=ON|-DENABLE_VULKAN=ON|-DENABLE_CUDA=ON) SBOM_ARGS=(--ggml) ;; esac
done
"$PY" tools/ci/sbom.py ${SBOM_ARGS[@]+"${SBOM_ARGS[@]}"} > "$DIST/$NAME.cdx.json"
echo "SBOM: $DIST/$NAME.cdx.json"

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

"$EXE" --version
[ -f "$CEXE" ] || [ -f "$CEXE.exe" ] || { echo "dynacorec missing from the install"; exit 1; }
"$PY" tools/ci/check_isa.py "$EXE"
"$EXE" doctor
# Release gate: the full CLI + HTTP API smoke test against the installed
# binary, when a model is provided (not in CI: no model download there).
if [ -n "${DYNALM_SMOKE_MODEL:-}" ]; then
  "$PY" tools/ci/smoke_api.py "$EXE" "$DYNALM_SMOKE_MODEL"
fi
echo "package ok: $DIST/$NAME"

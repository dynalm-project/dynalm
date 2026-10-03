#!/usr/bin/env bash
# DynaLM installer for Linux and macOS: builds from source and installs `dynalm`.
#
#   ./scripts/install.sh                     # installs to ~/.local/bin/dynalm
#   ./scripts/install.sh --prefix /usr/local # system-wide (may need sudo)
#   ./scripts/install.sh --jobs 4 --no-server
#
# Needs: a C++20 compiler (gcc >= 12, clang >= 15 or Xcode command line
# tools), CMake >= 3.24 and, optionally, Ninja. The first configure downloads
# one pinned header (cpp-httplib) for the HTTP server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
JOBS=""
SERVER=ON

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-server) SERVER=OFF; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
case "$OS" in
  Linux) HINT="install them with your package manager, e.g. 'sudo apt install g++ cmake ninja-build'" ;;
  Darwin)
    HINT="run 'xcode-select --install' and 'brew install cmake ninja'"
    xcode-select -p >/dev/null 2>&1 || die "Xcode command line tools are missing: $HINT"
    ;;
  *) die "unsupported OS '$OS' (use scripts/install.ps1 on Windows)" ;;
esac

command -v cmake >/dev/null 2>&1 || die "cmake not found: $HINT"
CMAKE_VER="$(cmake --version | head -n1 | awk '{print $3}')"
if [[ "$(printf '%s\n3.24\n' "$CMAKE_VER" | sort -V | head -n1)" != "3.24" ]]; then
  die "cmake $CMAKE_VER is too old (need >= 3.24): $HINT"
fi
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 ||
  die "no C++ compiler found: $HINT"

GEN=()
if command -v ninja >/dev/null 2>&1; then GEN=(-G Ninja); fi
if [[ -z "$JOBS" ]]; then
  JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

BUILD="$ROOT/build/install-release"
say "Configuring DynaLM ($OS $(uname -m), cmake $CMAKE_VER)"
cmake -S "$ROOT" -B "$BUILD" "${GEN[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=OFF -DENABLE_BENCHMARKS=OFF -DENABLE_SERVER="$SERVER" \
  -DDYNALM_STATIC_RUNTIME=ON

say "Building with $JOBS jobs"
cmake --build "$BUILD" --parallel "$JOBS"

say "Installing to $PREFIX"
cmake --install "$BUILD" --prefix "$PREFIX"

"$PREFIX/bin/dynalm" version
say "Installed: $PREFIX/bin/dynalm"
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "Add it to your PATH, e.g.:  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile" ;;
esac
echo "Try:  dynalm info   |   dynalm run <model.gguf> -p \"Hello\"   |   dynalm serve <model.gguf>"

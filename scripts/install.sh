#!/usr/bin/env sh
# DynaLM installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.sh | sh
#   sh scripts/install.sh                          # latest release -> ~/.local/bin
#   sh scripts/install.sh --version v0.1.0         # a specific release
#   sh scripts/install.sh --archive dynalm-linux-x86_64.tar.gz [--sha256sums SHA256SUMS]
#   sh scripts/install.sh --prefix /opt/dynalm     # elsewhere (sudo only if you choose a system path)
#   sh scripts/install.sh --from-source            # build this checkout instead (C++20 compiler, CMake >= 3.24)
#   sh scripts/install.sh --platform macos-x86_64  # override detection (e.g. Intel build under Rosetta)
#
# Release mode downloads dynalm-<os>-<arch>.tar.gz and SHA256SUMS from the
# GitHub release, verifies the checksum, installs bin/dynalm and
# bin/dynacorec into <prefix>/bin (default ~/.local) and runs `dynalm --version`.
# Supported: linux-x86_64, linux-arm64, macos-arm64, macos-x86_64. Anything
# else stops with an error instead of installing a wrong binary.
# Environment: DYNALM_REPO (default dynalm-project/dynalm), DYNALM_PREFIX.
set -eu

REPO="${DYNALM_REPO:-dynalm-project/dynalm}"
PREFIX="${DYNALM_PREFIX:-$HOME/.local}"
VERSION="latest"
ARCHIVE=""
SUMS=""
FROM_SOURCE=0
FORCE_PLATFORM=""

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --sha256sums) SUMS="$2"; shift 2 ;;
    --from-source) FROM_SOURCE=1; shift ;;
    --platform) FORCE_PLATFORM="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

# --- platform -------------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  *) die "unsupported OS '$OS'. On Windows use scripts/install.ps1." ;;
esac
case "$ARCH" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) die "unsupported CPU architecture '$ARCH' (supported: x86_64, arm64). Try --from-source." ;;
esac
# A shell running under Rosetta reports x86_64 on Apple Silicon: install the native build.
if [ "$os" = macos ] && [ "$arch" = x86_64 ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
  arch=arm64
fi
PLATFORM="$os-$arch"
if [ -n "$FORCE_PLATFORM" ]; then
  case "$FORCE_PLATFORM" in
    linux-x86_64|linux-arm64|macos-arm64|macos-x86_64) ;;
    *) die "unknown --platform '$FORCE_PLATFORM' (linux-x86_64, linux-arm64, macos-arm64, macos-x86_64)" ;;
  esac
  [ "${FORCE_PLATFORM%%-*}" = "$os" ] || die "--platform $FORCE_PLATFORM cannot run on $OS"
  [ "$FORCE_PLATFORM" = "$PLATFORM" ] || say "Installing $FORCE_PLATFORM on a $PLATFORM machine (--platform)"
  PLATFORM="$FORCE_PLATFORM"
fi

# --- from source ------------------------------------------------------------------
if [ "$FROM_SOURCE" = 1 ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  [ -f "$ROOT/CMakeLists.txt" ] || die "--from-source needs a DynaLM checkout (no CMakeLists.txt in $ROOT)"
  command -v cmake >/dev/null 2>&1 || die "cmake not found (need >= 3.24)"
  GEN=""
  command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
  JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  BUILD="$ROOT/build/install-release"
  say "Building DynaLM from source ($PLATFORM)"
  # shellcheck disable=SC2086
  cmake -S "$ROOT" -B "$BUILD" $GEN -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=OFF \
    -DDYNALM_STATIC_RUNTIME=ON
  cmake --build "$BUILD" --parallel "$JOBS"
  cmake --install "$BUILD" --prefix "$PREFIX"
else
  # --- release archive ------------------------------------------------------------
  NAME="dynalm-$PLATFORM"
  TMP="$(mktemp -d 2>/dev/null || mktemp -d -t dynalm)"
  trap 'rm -rf "$TMP"' EXIT INT TERM
  fetch() {  # url dest
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O "$2" "$1"
    else
      die "need curl or wget to download"
    fi
  }
  if [ -z "$ARCHIVE" ]; then
    if [ "$VERSION" = latest ]; then
      BASE="https://github.com/$REPO/releases/latest/download"
    else
      BASE="https://github.com/$REPO/releases/download/$VERSION"
    fi
    say "Downloading $NAME.tar.gz ($VERSION)"
    fetch "$BASE/$NAME.tar.gz" "$TMP/$NAME.tar.gz" ||
      die "no $NAME.tar.gz in release '$VERSION' of $REPO (platform not published?). Try --from-source."
    fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS" || die "release '$VERSION' has no SHA256SUMS; refusing to install unverified"
    ARCHIVE="$TMP/$NAME.tar.gz"
    SUMS="$TMP/SHA256SUMS"
  fi
  [ -f "$ARCHIVE" ] || die "archive not found: $ARCHIVE"
  case "$(basename "$ARCHIVE")" in
    "$NAME.tar.gz") ;;
    *) die "$(basename "$ARCHIVE") is not the archive for this machine ($NAME.tar.gz)" ;;
  esac
  if [ -n "$SUMS" ]; then
    want="$(grep " \*\{0,1\}$NAME.tar.gz\$" "$SUMS" | awk '{print $1}')"
    [ -n "$want" ] || die "$NAME.tar.gz is not listed in $SUMS"
    if command -v sha256sum >/dev/null 2>&1; then
      have="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    else
      have="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
    fi
    [ "$want" = "$have" ] || die "SHA-256 mismatch for $NAME.tar.gz (expected $want, got $have)"
    say "Checksum verified"
  else
    say "No SHA256SUMS given: skipping checksum verification of a local archive"
  fi
  tar -xzf "$ARCHIVE" -C "$TMP"
  [ -x "$TMP/$NAME/bin/dynalm" ] || die "archive does not contain $NAME/bin/dynalm"
  say "Installing to $PREFIX"
  mkdir -p "$PREFIX/bin" "$PREFIX/share/doc"
  cp "$TMP/$NAME/bin/dynalm" "$PREFIX/bin/dynalm"
  [ -f "$TMP/$NAME/bin/dynacorec" ] && cp "$TMP/$NAME/bin/dynacorec" "$PREFIX/bin/dynacorec"
  if [ -d "$TMP/$NAME/share/doc/dynalm" ]; then
    rm -rf "$PREFIX/share/doc/dynalm"
    cp -R "$TMP/$NAME/share/doc/dynalm" "$PREFIX/share/doc/dynalm"
  fi
  chmod +x "$PREFIX/bin/dynalm" "$PREFIX/bin/dynacorec" 2>/dev/null || true
  # Downloads through a browser get quarantined on macOS; curl/tar do not, but clear it if present.
  if [ "$os" = macos ] && command -v xattr >/dev/null 2>&1; then
    xattr -d com.apple.quarantine "$PREFIX/bin/dynalm" "$PREFIX/bin/dynacorec" 2>/dev/null || true
  fi
fi

# --- verify -------------------------------------------------------------------------
"$PREFIX/bin/dynalm" --version || die "the installed binary does not run on this machine"
say "Installed: $PREFIX/bin/dynalm"
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "Add it to your PATH:  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile   (then open a new shell)" ;;
esac
echo "Next:  dynalm doctor   ->   dynalm pull qwen3:4b   ->   dynalm run qwen3:4b"

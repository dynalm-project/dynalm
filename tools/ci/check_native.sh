#!/usr/bin/env bash
# Verifies that a build is strictly native (DYNALM_NATIVE_ONLY=ON, docs/build.md):
# no ggml / llama.cpp code was fetched, compiled or linked.
#
#   tools/ci/check_native.sh <build dir>        (run from the repository root)
#
# Checks, each printed as ok / FAIL:
#   1. CMake cache: DYNALM_NATIVE_ONLY=ON and every ggml backend option OFF.
#   2. No ggml sources were fetched or built (_deps/ggml*, ggml targets).
#   3. The dynalm and dynacorec binaries link no ggml / llama library.
#   4. Their symbol tables hold no ggml / llama C symbols, defined or undefined
#      (the only "ggml" names allowed are DynaLM's own C++ functions in the
#      dynacore:: / dynalm:: namespaces: GGUF type names and the stubs that
#      report the ggml backends as not built).
# Exit status 0 only when every check passes.
set -uo pipefail
BUILD="${1:?usage: tools/ci/check_native.sh <build dir>}"
fail=0
ok() { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }

cache="$BUILD/CMakeCache.txt"
if [ ! -f "$cache" ]; then
  echo "FAIL  $cache not found"
  exit 1
fi
val() { grep -E "^$1:" "$cache" | head -1 | cut -d= -f2; }
[ "$(val DYNALM_NATIVE_ONLY)" = "ON" ] && ok "DYNALM_NATIVE_ONLY=ON" || bad "DYNALM_NATIVE_ONLY is '$(val DYNALM_NATIVE_ONLY)', not ON"
for opt in ENABLE_GGML_CPU ENABLE_METAL ENABLE_VULKAN ENABLE_CUDA; do
  v="$(val $opt)"
  [ "$v" = "OFF" ] || [ -z "$v" ] && ok "$opt=${v:-unset}" || bad "$opt=$v (a ggml backend)"
done

if compgen -G "$BUILD/_deps/ggml*" > /dev/null; then
  bad "ggml sources present in $BUILD/_deps"
else
  ok "no ggml sources fetched ($BUILD/_deps)"
fi
# Any generator: built ggml / llama libraries anywhere in the build tree.
libs_found="$(find "$BUILD" \( -name 'libggml*' -o -name 'libllama*' -o -name 'ggml*.lib' -o -name 'llama*.lib' \) 2>/dev/null | head -3)"
if [ -n "$libs_found" ]; then
  bad "ggml / llama libraries built: $(printf '%s\n' "$libs_found" | tr '\n' ' ')"
else
  ok "no ggml / llama library in the build tree"
fi

for exe in "$BUILD/bin/dynalm" "$BUILD/bin/dynacorec"; do
  [ -f "$exe" ] || { bad "$exe missing"; continue; }
  if command -v otool > /dev/null; then libs="$(otool -L "$exe")"; else libs="$(ldd "$exe" 2>/dev/null)"; fi
  if printf '%s\n' "$libs" | grep -qiE "ggml|llama"; then
    bad "$(basename "$exe") links a ggml / llama library"
  else
    ok "$(basename "$exe") links no ggml / llama library"
  fi
  # C symbols named ggml_* / llama_* (mangled C++ names of DynaLM's own
  # namespaced functions start with __ZN / _ZN and are excluded).
  syms="$(nm "$exe" 2>/dev/null | awk '{print $NF}' | grep -E '^_?_?(ggml|llama|gguf_)' || true)"
  if [ -n "$syms" ]; then
    bad "$(basename "$exe") has ggml / llama symbols: $(printf '%s\n' "$syms" | head -5 | tr '\n' ' ')"
  else
    ok "$(basename "$exe") has no ggml / llama C symbols"
  fi
  # Demangled names mentioning ggml must be DynaLM's own namespaced functions.
  foreign="$(nm -C "$exe" 2>/dev/null | grep -i ggml | grep -vE 'dynacore::|dynalm::' || true)"
  if [ -n "$foreign" ]; then
    bad "$(basename "$exe") has non-DynaLM ggml names: $(printf '%s\n' "$foreign" | head -3 | tr '\n' ' ')"
  else
    ok "$(basename "$exe"): every 'ggml' name is a dynacore:: / dynalm:: function (GGUF type names, not-built stubs)"
  fi
done
echo "$([ $fail -eq 0 ] && echo 'strict native: all checks passed' || echo 'strict native: FAILURES above')"
exit $fail

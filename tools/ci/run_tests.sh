#!/usr/bin/env bash
# Runs the model-free test programs of a build, then the offline tool tests.
#
#   tools/ci/run_tests.sh <build dir>        (run from the repository root)
#
# Each test program exits non-zero on its first failed check. Tests whose
# kernels the CPU lacks skip themselves (for example the AVX2 / AVX-VNNI
# kernel tests on ARM64 or under Rosetta), so the same list runs everywhere.
# Model-based tests (test_solo_prefill, test_prefill_priority,
# test_loadgen_openloop) need a GGUF model and are not run here; the
# dynalm/tests programs generate a tiny synthetic GGUF model themselves.
# GPU tests report "SKIP" or "NOT BUILT" in their last line when no Metal
# device or backend is available; that is not a GPU pass.
set -uo pipefail
BUILD="${1:?usage: tools/ci/run_tests.sh <build dir>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TESTS=(
  dynacore/test_matmul_registry
  dynacore/test_int16_prefill
  dynacore/test_packed_8rows
  dynacore/test_int_kernels
  dynacore/test_thread_pool
  dynacore/test_ggml_device
  dynacore/test_ggml_metal_ranges
  dynacore/test_dynakernel
  dynacore/test_dynakernel_lowering
  dynacore/test_dynakernel_quant
  dynacore/test_dynakernel_codegen
  dynacore/test_dynakernel_int8gen
  dynacore/test_dynakernel_llvm
  dynalm/benchmarks/test_batch_planner
  dynalm/tests/test_engine
  dynalm/tests/test_server
  dynalm/tests/test_fuzz_parsers
)
fail=0
ran=0
for t in "${TESTS[@]}"; do
  exe="$BUILD/$t"
  [ -f "$exe.exe" ] && exe="$exe.exe"
  if [ ! -f "$exe" ]; then
    # core-only builds have no dynalm/ tree
    if [[ "$t" == dynalm/* ]] && [ ! -d "$BUILD/dynalm" ]; then
      echo "skip  $t (not part of this build)"
      continue
    fi
    echo "FAIL  $t: executable missing ($exe)"
    fail=1
    continue
  fi
  out="$("$exe" 2>&1)"
  code=$?
  ran=$((ran + 1))
  if [ $code -eq 0 ]; then
    echo "ok    $t: $(printf '%s\n' "$out" | tail -n 1)"
  else
    echo "FAIL  $t (exit $code)"
    printf '%s\n' "$out" | tail -n 20
    fail=1
  fi
done
PY="$(command -v python3 || command -v python)"
for t in tools/test_compare_engines.py tools/test_bench_analyze.py tools/test_bench_report.py tools/test_e2e_report.py tools/test_model_matrix.py tools/test_ab.py tools/test_llama_parity.py tools/test_release.py; do
  out="$("$PY" "$t" 2>&1)"
  code=$?
  ran=$((ran + 1))
  if [ $code -eq 0 ]; then
    echo "ok    $t: $(printf '%s\n' "$out" | tail -n 1)"
  else
    echo "FAIL  $t (exit $code)"
    printf '%s\n' "$out" | tail -n 20
    fail=1
  fi
done
# The built binary reports the release version (CMakeLists.txt: project VERSION +
# DYNALM_VERSION_SUFFIX; tools/ci/version.py).
exe="$BUILD/bin/dynalm"; [ -f "$exe.exe" ] && exe="$exe.exe"
if [ -f "$exe" ]; then
  want="$("$PY" tools/ci/version.py)"; got="$("$exe" --version | awk 'NR==1{print $2}' | tr -d '\r')"
  ran=$((ran + 1))
  if [ "$got" = "$want" ]; then echo "ok    dynalm --version: $got"; else echo "FAIL  dynalm --version: '$got', expected '$want'"; fail=1; fi
fi
echo "$ran test programs run; $([ $fail -eq 0 ] && echo 'all passed' || echo 'FAILURES above')"
exit $fail

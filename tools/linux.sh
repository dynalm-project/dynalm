#!/usr/bin/env bash
# Configure, build and test a Linux preset inside the engine-dev container.
#   tools/linux.sh [preset]      (default: linux-release)
# The build tree lives in a Docker volume so it doesn't mix with Windows builds.
set -euo pipefail
cd "$(dirname "$0")/.."
preset="${1:-linux-release}"
docker image inspect engine-dev >/dev/null 2>&1 ||
  docker build -t engine-dev -f tools/docker/Dockerfile.dev tools/docker

# TSAN (gcc 13) cannot start under the 32-bit mmap ASLR entropy of recent
# kernels ("unexpected memory mapping"), so TSAN runs with ASLR disabled via
# setarch. That needs the personality() syscall, which Docker's default
# seccomp profile blocks; only this local, throwaway test container gets the
# exception.
run_prefix=""
docker_opts=()
if [[ "$preset" == *tsan* ]]; then
  run_prefix="setarch x86_64 -R"
  docker_opts=(--security-opt seccomp=unconfined)
fi

# MSYS_NO_PATHCONV keeps Git Bash from rewriting the container paths.
MSYS_NO_PATHCONV=1 docker run --rm "${docker_opts[@]}" \
  -v "$(pwd -W 2>/dev/null || pwd):/src" \
  -v engine-linux-build:/src/build \
  engine-dev bash -c "
    cmake --preset $preset -DENGINE_TEST_MODEL=/src/models/SmolLM2-135M-Instruct-f16.gguf \
      -DENGINE_TEST_MODEL_QWEN=/src/models/qwen2.5-0.5b-instruct-q8_0.gguf \
      -DENGINE_TEST_MODEL_QWEN_F16=/src/models/qwen2.5-0.5b-instruct-fp16.gguf \
      -DENGINE_TEST_MODEL_GEMMA=/src/models/gemma-3-270m-it-F16.gguf >/dev/null &&
    $run_prefix cmake --build --preset $preset 2>&1 | grep -E 'warning|error|FAILED' ;
    $run_prefix ctest --preset $preset --timeout 900 2>&1 | grep -E 'Failed|failed|passed|\*\*\*|ERROR|runtime error|Sanitizer'"

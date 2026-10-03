#!/usr/bin/env bash
# Downloads the small development/test models into models/ (git-ignored).
# Resumable: re-run to continue interrupted downloads.
#
#   tools/fetch_models.sh            # all
#   tools/fetch_models.sh smollm     # only SmolLM2-135M (llama arch, fastest)
#   tools/fetch_models.sh f16        # f16 Qwen2.5-0.5B + Gemma-3-270M (Phase 7 golden tests)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

SMOLLM=(
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf"
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
)
QWEN=(
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

# f16 variants used for golden tests of non-Llama adapters (Phase 7).
F16=(
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
  "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf"
)

case "${1:-all}" in
  smollm) urls=("${SMOLLM[@]}") ;;
  qwen) urls=("${QWEN[@]}") ;;
  f16) urls=("${F16[@]}") ;;
  *) urls=("${SMOLLM[@]}" "${QWEN[@]}" "${F16[@]}") ;;
esac

for u in "${urls[@]}"; do
  f="models/$(basename "$u")"
  echo "fetching $f"
  curl -sSL --fail --retry 3 -C - -o "$f" "$u"
done
ls -la models

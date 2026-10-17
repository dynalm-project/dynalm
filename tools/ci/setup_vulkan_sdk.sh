#!/usr/bin/env bash
# Installs the LunarG Vulkan SDK on a GitHub Actions runner (Linux x86-64 or
# Windows x64) for -DENABLE_VULKAN=ON builds: headers, loader, glslc and
# SPIRV-Headers. Exports VULKAN_SDK and the tool / library paths through
# GITHUB_ENV / GITHUB_PATH. Pinned so release builds are reproducible.
#
#   bash tools/ci/setup_vulkan_sdk.sh
set -euo pipefail
VER="${VULKAN_SDK_VERSION:-1.4.363.0}"
BASE="https://sdk.lunarg.com/sdk/download/$VER"

case "$(uname -s)" in
  Linux)
    dir="$RUNNER_TEMP/vulkansdk"
    mkdir -p "$dir"
    curl -fsSL --retry 5 -o "$dir/sdk.tar.xz" "$BASE/linux/vulkansdk-linux-x86_64-$VER.tar.xz"
    tar -xJf "$dir/sdk.tar.xz" -C "$dir"
    sdk="$dir/$VER/x86_64"
    echo "VULKAN_SDK=$sdk" >> "$GITHUB_ENV"
    echo "LD_LIBRARY_PATH=$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
    echo "$sdk/bin" >> "$GITHUB_PATH"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    exe="$RUNNER_TEMP/vulkansdk.exe"
    curl -fsSL --retry 5 -o "$exe" "$BASE/windows/vulkansdk-windows-X64-$VER.exe"
    sdk='C:\VulkanSDK'
    "$exe" --root "$sdk" --accept-licenses --default-answer --confirm-command install
    echo "VULKAN_SDK=$sdk" >> "$GITHUB_ENV"
    echo "$sdk\\Bin" >> "$GITHUB_PATH"
    ;;
  *) echo "setup_vulkan_sdk.sh: unsupported runner $(uname -s)" >&2; exit 1 ;;
esac
echo "Vulkan SDK $VER installed at $sdk"

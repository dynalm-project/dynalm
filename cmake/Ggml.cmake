# ggml: the GPU execution backend (docs/gpu-backend.md, decision D1).
#
# Pinned to one upstream commit of the standalone ggml project
# (https://github.com/ggml-org/ggml, MIT License). A git commit id names the
# exact tree, so a build can never pick up a newer revision silently. To
# update: change DYNALM_GGML_COMMIT and DYNALM_GGML_VERSION together, rebuild,
# run tools/ci/run_tests.sh and the GPU checks of docs/gpu-backend.md, and
# record the change in CHANGELOG.md.
#
# Offline builds: -DDYNALM_GGML_SOURCE_DIR=/path/to/ggml (a checkout of the
# same commit; its version is checked).
#
# GPU backends, each opt-in (any combination):
#   ENABLE_METAL   Apple Silicon. Shaders are embedded as source and compiled by
#                  the Metal framework at device creation (GGML_METAL_EMBED_LIBRARY,
#                  decision D2): the Command Line Tools suffice, no Xcode.
#   ENABLE_VULKAN  AMD / Intel / NVIDIA / Qualcomm GPUs with a Vulkan driver
#                  (macOS: through MoltenVK). Needs the Vulkan headers, loader
#                  and glslc (Vulkan SDK; brew: vulkan-headers vulkan-loader
#                  molten-vk shaderc spirv-headers).
#   ENABLE_CUDA    NVIDIA GPUs. Needs the CUDA toolkit (nvcc). Architectures:
#                  CMAKE_CUDA_ARCHITECTURES (ggml's default list when unset).

set(DYNALM_GGML_COMMIT a05c9922613c6a2230dbb1d0de679519ecafb4f0)  # tag v0.27.0
set(DYNALM_GGML_VERSION 0.27.0)
set(DYNALM_GGML_SOURCE_DIR "" CACHE PATH "Local ggml checkout of the pinned commit (offline builds)")

if(ENABLE_METAL AND NOT APPLE)
  message(FATAL_ERROR "ENABLE_METAL=ON: the Metal backend requires macOS")
endif()
if(CMAKE_VERSION VERSION_LESS 3.28)
  # FetchContent EXCLUDE_FROM_ALL keeps ggml's install rules out of the package.
  message(FATAL_ERROR "GPU backends (ENABLE_METAL/VULKAN/CUDA) and ENABLE_GGML_CPU need CMake >= 3.28 (this is ${CMAKE_VERSION})")
endif()
enable_language(C)
if(ENABLE_METAL)
  enable_language(OBJC)
endif()
if(ENABLE_VULKAN)
  find_package(Vulkan COMPONENTS glslc)
  if(NOT Vulkan_FOUND OR NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "ENABLE_VULKAN=ON needs the Vulkan headers, loader and glslc "
                        "(Vulkan SDK; macOS: brew install vulkan-headers vulkan-loader molten-vk shaderc spirv-headers)")
  endif()
endif()
if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "ENABLE_CUDA=ON needs the CUDA toolkit (nvcc not found)")
  endif()
endif()

# ggml's own options: only the requested GPU backends, static, portable,
# nothing extra built.
set(GGML_METAL ${ENABLE_METAL} CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ${ENABLE_METAL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_CUDA ${ENABLE_CUDA} CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
# ggml's own thread pool, not OpenMP: an OpenMP build needs libgomp (Linux) or
# vcomp140.dll (Windows) at run time, which the packages and the Docker image lack.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# ggml-cpu (the ggml-cpu device) on x86-64: the AVX2 tier, like DynaLM's own
# kernels; the device refuses CPUs without AVX2/FMA/F16C at creation.
if((CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64") AND ENABLE_AVX2)
  foreach(_isa AVX AVX2 FMA F16C BMI2)
    set(GGML_${_isa} ON CACHE BOOL "" FORCE)
  endforeach()
endif()
# ggml-cpu for this machine's exact CPU (-mcpu=native: i8mm, SME, ... on Apple
# M2+), as llama.cpp builds by default. OFF for release packages, which must
# run on every CPU of their platform.
option(DYNALM_GGML_NATIVE "Build ggml's CPU kernels for this machine's CPU (not portable)" OFF)
set(GGML_NATIVE ${DYNALM_GGML_NATIVE} CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(_dynalm_saved_shared ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

include(FetchContent)
if(DYNALM_GGML_SOURCE_DIR)
  file(STRINGS "${DYNALM_GGML_SOURCE_DIR}/CMakeLists.txt" _ggml_ver_lines REGEX "set\\(GGML_VERSION_(MAJOR|MINOR|PATCH)")
  string(REGEX MATCHALL "[0-9]+" _ggml_ver "${_ggml_ver_lines}")
  string(REPLACE ";" "." _ggml_ver "${_ggml_ver}")
  if(NOT _ggml_ver STREQUAL DYNALM_GGML_VERSION)
    message(FATAL_ERROR "DYNALM_GGML_SOURCE_DIR is ggml ${_ggml_ver}; this build pins ${DYNALM_GGML_VERSION} "
                        "(commit ${DYNALM_GGML_COMMIT})")
  endif()
  set(FETCHCONTENT_SOURCE_DIR_GGML "${DYNALM_GGML_SOURCE_DIR}")
endif()
FetchContent_Declare(ggml
  GIT_REPOSITORY https://github.com/ggml-org/ggml.git
  GIT_TAG        ${DYNALM_GGML_COMMIT}
  GIT_SHALLOW    FALSE
  SYSTEM            # ggml headers: their warnings are not ours
  EXCLUDE_FROM_ALL) # ggml's own install rules (headers, static libs) stay out of DynaLM packages
FetchContent_MakeAvailable(ggml)
set(BUILD_SHARED_LIBS ${_dynalm_saved_shared})

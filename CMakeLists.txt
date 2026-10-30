cmake_minimum_required(VERSION 3.24)

project(dynalm
  VERSION 0.5.0
  DESCRIPTION "DynaLM: CPU-first LLM inference engine"
  HOMEPAGE_URL "https://github.com/dynalm-project/dynalm"
  LANGUAGES CXX)
# The release version: PROJECT_VERSION (numeric, x.y.z) plus a pre-release
# suffix, "" for a stable release (e.g. "-beta.1" -> 0.2.0-beta.1). The one
# source of the version string: `dynalm --version` (version.h.in), the SBOM
# (tools/ci/sbom.py) and the release workflow's check against the tag
# (tools/ci/package.sh) all read it from here.
set(DYNALM_VERSION_SUFFIX "")
set(DYNALM_VERSION "${PROJECT_VERSION}${DYNALM_VERSION_SUFFIX}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
# Strict native mode (docs/build.md): the production
# inference path is DynaLM's own code only. Turns every ggml-based backend
# off (ggml-cpu, ggml Metal / Vulkan / CUDA) and refuses to configure if one
# is requested; ggml is then not fetched, built or linked.
option(DYNALM_NATIVE_ONLY "Build only DynaLM's native backends (no ggml / llama.cpp code)" OFF)

# GPU backends through pinned ggml (docs/gpu-backend.md); any combination.
option(ENABLE_CUDA   "CUDA GPU backend through pinned ggml (NVIDIA; needs the CUDA toolkit)" OFF)
option(ENABLE_HIP    "ROCm/HIP backend (not implemented)" OFF)
option(ENABLE_METAL  "Metal GPU backend through pinned ggml (Apple Silicon)" OFF)
option(ENABLE_VULKAN "Vulkan GPU backend through pinned ggml (needs the Vulkan SDK / glslc)" OFF)
if(ENABLE_HIP)
  message(FATAL_ERROR "ENABLE_HIP: the ROCm/HIP backend is not implemented; use ENABLE_VULKAN for AMD GPUs")
endif()
# ggml's CPU kernels: the default CPU path (`--cpu`, and `auto` without a GPU),
# with 8-bit activations as in llama.cpp (llama.cpp-level accuracy, ~4x the
# speed of the native kernels). `--mode exact` keeps DynaCore's own
# kernels (DD-053). Always part of a GPU build. Default ON where ggml's CPU
# backend builds: CMake >= 3.28, and not MSVC on ARM64 (ggml needs clang there).
set(_ggml_cpu_default ON)
if(CMAKE_VERSION VERSION_LESS 3.28 OR (MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$"))
  set(_ggml_cpu_default OFF)
endif()
if(DYNALM_NATIVE_ONLY)
  set(_ggml_cpu_default OFF)
  foreach(_g ENABLE_CUDA ENABLE_METAL ENABLE_VULKAN ENABLE_GGML_CPU)
    if(${_g})
      message(FATAL_ERROR "DYNALM_NATIVE_ONLY=ON: ${_g} builds a ggml backend; turn it off")
    endif()
  endforeach()
endif()
option(ENABLE_GGML_CPU "ggml's CPU kernels as the default CPU path (fetches pinned ggml)" ${_ggml_cpu_default})
if(NOT ENABLE_GGML_CPU)
  message(STATUS "ggml CPU kernels: off; the CPU uses DynaCore's native kernels")
endif()
if(ENABLE_METAL OR ENABLE_VULKAN OR ENABLE_CUDA OR ENABLE_GGML_CPU)
  set(DYNALM_WITH_GGML ON)
else()
  set(DYNALM_WITH_GGML OFF)
endif()

# CPU kernel variants compiled into the binary. Selection happens at runtime
# via CPU feature detection; compiling a variant never makes it mandatory.
option(ENABLE_AVX2   "Compile AVX2/FMA CPU kernels"      ON)
option(ENABLE_AVX512 "Compile AVX-512 CPU kernels"       OFF)
option(ENABLE_AMX    "Compile AMX CPU kernels"           OFF)

option(ENABLE_NEON   "Compile NEON CPU kernels (ARM64)"   ON)

# SIMD tiers follow the target architecture: AVX2/AVX-512/AMX exist only on
# x86-64, NEON only on ARM64 (Apple Silicon, ARM64 Linux, Windows on ARM). The
# per-file ISA flags (-mavx2, ...) would be compile errors elsewhere.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _dynalm_proc)
if(CMAKE_OSX_ARCHITECTURES)
  list(GET CMAKE_OSX_ARCHITECTURES 0 _dynalm_proc)
endif()
if(_dynalm_proc MATCHES "^(x86_64|amd64|x64)$")
  set(DYNALM_ARCH x86_64)
  set(ENABLE_NEON OFF)
elseif(_dynalm_proc MATCHES "^(aarch64|arm64)$")
  set(DYNALM_ARCH arm64)
  set(ENABLE_AVX2 OFF)
  set(ENABLE_AVX512 OFF)
  set(ENABLE_AMX OFF)
else()
  set(DYNALM_ARCH "${_dynalm_proc}")  # generic kernels only
  set(ENABLE_AVX2 OFF)
  set(ENABLE_AVX512 OFF)
  set(ENABLE_AMX OFF)
  set(ENABLE_NEON OFF)
endif()

# Self-contained binaries for packages: static C/C++ runtime (no Visual C++
# redistributable on Windows, no libstdc++ version dependency on Linux).
option(DYNALM_STATIC_RUNTIME "Link the C/C++ runtime statically (release packages)" OFF)
if(DYNALM_STATIC_RUNTIME)
  if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    add_link_options(-static-libstdc++ -static-libgcc)
  endif()
endif()

# Link-time optimization (P14, DD-063).
option(DYNALM_LTO "Link-time optimization (interprocedural) for release builds" OFF)
if(DYNALM_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)
  if(_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  else()
    message(WARNING "DYNALM_LTO=ON but the toolchain does not support IPO: ${_ipo_msg}")
  endif()
endif()

option(ENABLE_SERVER     "Build the HTTP server"   ON)
option(ENABLE_BENCHMARKS "Build benchmarks"        ON)

option(ENABLE_ASAN  "AddressSanitizer"            OFF)
option(ENABLE_UBSAN "UndefinedBehaviorSanitizer"  OFF)
option(ENABLE_TSAN  "ThreadSanitizer"             OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EngineCompilerFlags)
if(DYNALM_WITH_GGML)
  include(Ggml)
endif()

# Two libraries, one product (DD-068):
#   dynacore        low-level inference runtime (tensors, memory, devices, kernels)
#   dynalm_runtime  LLM platform on top of it (models, scheduler, KV policy, API)
# DynaCore never depends on DynaLM; the core-only build proves it.
option(DYNALM_CORE_ONLY "Configure only DynaCore" OFF)

add_subdirectory(dynacore)
if(NOT DYNALM_CORE_ONLY)
  add_subdirectory(dynalm)
endif()

# --- install + packages -------------------------------------------------------
#   cmake --install build --prefix ~/.local     -> bin/dynalm, share/doc/dynalm
#   cpack --config build/CPackConfig.cmake      -> dynalm-<os>-<arch>.zip/.tar.gz
#
# Archive names carry no version, so "latest" download links and the
# installers stay stable (scripts/install.sh, scripts/install.ps1); the
# version is in `dynalm --version` and in the release tag. <os> is linux,
# macos or windows; <arch> is x86_64 or arm64. Layout:
#   dynalm-<os>-<arch>/bin/{dynalm,dynacorec}
#   dynalm-<os>-<arch>/share/doc/dynalm/{README.md,LICENSE,NOTICE,CHANGELOG.md,docs/}
include(GNUInstallDirs)
install(TARGETS dynacorec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(NOT DYNALM_CORE_ONLY)
  install(TARGETS dynalm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES README.md CHANGELOG.md LICENSE NOTICE THIRD_PARTY_LICENSES.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR}/docs)
set(CPACK_PACKAGE_NAME dynalm)
set(CPACK_PACKAGE_VENDOR "DynaLM")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _dynalm_os)
if(_dynalm_os STREQUAL "darwin")
  set(_dynalm_os macos)
endif()
# GPU package variants: Vulkan / CUDA builds get a suffix so they sit next to
# the CPU archive (the installers pick one). Metal builds keep the plain name:
# every macOS arm64 package includes Metal (Metal is part of macOS).
set(_dynalm_variant "")
if(ENABLE_CUDA)
  string(APPEND _dynalm_variant "-cuda")
endif()
if(ENABLE_VULKAN)
  string(APPEND _dynalm_variant "-vulkan")
endif()
set(CPACK_PACKAGE_FILE_NAME "dynalm-${_dynalm_os}-${DYNALM_ARCH}${_dynalm_variant}")
if(WIN32)
  set(CPACK_GENERATOR ZIP)
else()
  set(CPACK_GENERATOR TGZ)
endif()
include(CPack)

message(STATUS "DynaLM ${DYNALM_VERSION}: build=${CMAKE_BUILD_TYPE} arch=${DYNALM_ARCH} "
               "avx2=${ENABLE_AVX2} avx512=${ENABLE_AVX512} amx=${ENABLE_AMX} neon=${ENABLE_NEON} "
               "metal=${ENABLE_METAL} vulkan=${ENABLE_VULKAN} cuda=${ENABLE_CUDA} server=${ENABLE_SERVER} bench=${ENABLE_BENCHMARKS} "
               "ggml=${DYNALM_WITH_GGML} native_only=${DYNALM_NATIVE_ONLY}")

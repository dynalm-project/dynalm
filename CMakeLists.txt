cmake_minimum_required(VERSION 3.24)

project(dynalm
  VERSION 0.1.0
  DESCRIPTION "DynaLM: CPU-first LLM inference engine"
  HOMEPAGE_URL "https://github.com/dynalm/dynalm"
  LANGUAGES CXX)

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
# Accelerator backends: future phases. Kept as explicit flags so that the
# build surface is stable; enabling one today is a configuration error.
option(ENABLE_CUDA   "CUDA backend (future phase)"   OFF)
option(ENABLE_HIP    "ROCm/HIP backend (future phase)" OFF)
option(ENABLE_METAL  "Metal backend (future phase)"  OFF)
option(ENABLE_VULKAN "Vulkan backend (future phase)" OFF)

# CPU kernel variants compiled into the binary. Selection happens at runtime
# via CPU feature detection; compiling a variant never makes it mandatory.
option(ENABLE_AVX2   "Compile AVX2/FMA CPU kernels"      ON)
option(ENABLE_AVX512 "Compile AVX-512 CPU kernels"       OFF)
option(ENABLE_AMX    "Compile AMX CPU kernels"           OFF)

option(ENABLE_NEON   "Compile NEON CPU kernels (ARM64)"   ON)

# SIMD tiers follow the target architecture: AVX2/AVX-512/AMX exist only on
# x86-64, NEON only on ARM64 (Apple Silicon, Graviton, Windows on ARM). The
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

option(ENABLE_SERVER     "Build the HTTP server"   ON)
option(ENABLE_TESTS      "Build unit tests"        ON)
option(ENABLE_BENCHMARKS "Build benchmarks"        ON)

option(ENABLE_ASAN  "AddressSanitizer"            OFF)
option(ENABLE_UBSAN "UndefinedBehaviorSanitizer"  OFF)
option(ENABLE_TSAN  "ThreadSanitizer"             OFF)

foreach(_gpu CUDA HIP METAL VULKAN)
  if(ENABLE_${_gpu})
    message(FATAL_ERROR "ENABLE_${_gpu}=ON: the ${_gpu} backend is a future phase and is not implemented.")
  endif()
endforeach()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EngineCompilerFlags)

add_subdirectory(src)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# --- install + packages -------------------------------------------------------
#   cmake --install build --prefix ~/.local     -> bin/dynalm, share/doc/dynalm
#   cpack --config build/CPackConfig.cmake      -> dynalm-<ver>-<os>-<arch>.zip/.tar.gz
include(GNUInstallDirs)
install(TARGETS dynalm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES README.md CHANGELOG.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR}/docs)
set(CPACK_PACKAGE_NAME dynalm)
set(CPACK_PACKAGE_VENDOR "DynaLM")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _dynalm_os)
set(CPACK_PACKAGE_FILE_NAME "dynalm-${PROJECT_VERSION}-${_dynalm_os}-${DYNALM_ARCH}")
if(WIN32)
  set(CPACK_GENERATOR ZIP)
else()
  set(CPACK_GENERATOR TGZ)
endif()
include(CPack)

message(STATUS "DynaLM ${PROJECT_VERSION}: build=${CMAKE_BUILD_TYPE} arch=${DYNALM_ARCH} "
               "avx2=${ENABLE_AVX2} avx512=${ENABLE_AVX512} amx=${ENABLE_AMX} neon=${ENABLE_NEON} "
               "server=${ENABLE_SERVER} tests=${ENABLE_TESTS} bench=${ENABLE_BENCHMARKS}")

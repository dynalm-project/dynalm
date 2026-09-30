cmake_minimum_required(VERSION 3.24)

project(engine
  VERSION 0.1.0
  DESCRIPTION "CPU-first LLM inference runtime"
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

message(STATUS "engine ${PROJECT_VERSION}: build=${CMAKE_BUILD_TYPE} "
               "avx2=${ENABLE_AVX2} avx512=${ENABLE_AVX512} amx=${ENABLE_AMX} "
               "server=${ENABLE_SERVER} tests=${ENABLE_TESTS} bench=${ENABLE_BENCHMARKS}")

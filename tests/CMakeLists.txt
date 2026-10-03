include(FetchContent)

# Prefer a system GoogleTest; fall back to a pinned download.
find_package(GTest 1.14 CONFIG QUIET)
if(NOT GTest_FOUND)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match MSVC /MD runtime
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# Optional real-model tests: skipped when the file is absent
# (fetch with tools/fetch_models.sh smollm).
set(ENGINE_TEST_MODEL "${PROJECT_SOURCE_DIR}/models/SmolLM2-135M-Instruct-f16.gguf"
    CACHE FILEPATH "GGUF model used by optional real-model tests")
set(ENGINE_TEST_MODEL_QWEN "${PROJECT_SOURCE_DIR}/models/qwen2.5-0.5b-instruct-q8_0.gguf"
    CACHE FILEPATH "Qwen GGUF used by optional real-model tests")
set(ENGINE_TEST_MODEL_QWEN_F16 "${PROJECT_SOURCE_DIR}/models/qwen2.5-0.5b-instruct-fp16.gguf"
    CACHE FILEPATH "Qwen f16 GGUF used by optional runtime golden tests")
set(ENGINE_TEST_MODEL_GEMMA "${PROJECT_SOURCE_DIR}/models/gemma-3-270m-it-F16.gguf"
    CACHE FILEPATH "Gemma-3 f16 GGUF used by optional golden tests")
set(ENGINE_TEST_MODEL_Q8 "${PROJECT_SOURCE_DIR}/models/SmolLM2-135M-Instruct-Q8_0.gguf"
    CACHE FILEPATH "Q8_0 GGUF of ENGINE_TEST_MODEL, for quantization accuracy tests")
set(ENGINE_TEST_MODEL_QWEN_Q4 "${PROJECT_SOURCE_DIR}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    CACHE FILEPATH "Qwen Q4_K_M GGUF (mixed Q4_K/Q5_0/Q6_K/Q8_0) for quantized golden tests")

set(ENGINE_TEST_MODEL_GRANITE_MOE "${PROJECT_SOURCE_DIR}/models/granite-3.1-1b-a400m-instruct-Q8_0.gguf"
    CACHE FILEPATH "Granite-3.1-1B-A400M (MoE, 32 experts / 8 active) Q8_0 GGUF for MoE golden tests")

function(engine_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE engine_core GTest::gtest_main)
  target_compile_definitions(${name} PRIVATE
    ENGINE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    ENGINE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    ENGINE_TEST_MODEL_PATH="${ENGINE_TEST_MODEL}"
    ENGINE_TEST_MODEL_QWEN_PATH="${ENGINE_TEST_MODEL_QWEN}"
    ENGINE_TEST_MODEL_QWEN_F16_PATH="${ENGINE_TEST_MODEL_QWEN_F16}"
    ENGINE_TEST_MODEL_GEMMA_PATH="${ENGINE_TEST_MODEL_GEMMA}"
    ENGINE_TEST_MODEL_Q8_PATH="${ENGINE_TEST_MODEL_Q8}"
    ENGINE_TEST_MODEL_QWEN_Q4_PATH="${ENGINE_TEST_MODEL_QWEN_Q4}"
    ENGINE_TEST_MODEL_GRANITE_MOE_PATH="${ENGINE_TEST_MODEL_GRANITE_MOE}")
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

engine_add_test(test_foundation
  test_status.cpp
  test_logging.cpp
  test_cpu_info.cpp)

engine_add_test(test_tensor
  test_dtype.cpp
  test_tensor.cpp)

engine_add_test(test_loader
  test_gguf.cpp
  test_model_ir.cpp)

engine_add_test(test_tokenizer
  test_tokenizer.cpp)

engine_add_test(test_runtime
  test_runtime.cpp)

engine_add_test(test_architectures
  test_architectures.cpp)

engine_add_test(test_quant
  test_quant.cpp)

engine_add_test(test_sequence
  test_sequence.cpp)

engine_add_test(test_kv_cache
  test_kv_cache.cpp)

engine_add_test(test_batching
  test_batching.cpp)

engine_add_test(test_scheduler
  test_scheduler.cpp)

engine_add_test(test_prefix_cache
  test_prefix_cache.cpp)

engine_add_test(test_kernels
  test_kernels.cpp)

engine_add_test(test_streaming
  test_streaming.cpp)

engine_add_test(test_json
  test_json.cpp)

if(ENABLE_SERVER)
  engine_add_test(test_server
    test_server.cpp)
  target_include_directories(test_server SYSTEM PRIVATE ${ENGINE_HTTPLIB_DIR})
  target_link_libraries(test_server PRIVATE engine_server)
endif()

engine_add_test(test_bench
  test_bench.cpp)

engine_add_test(test_config
  test_config.cpp)

engine_add_test(test_compat
  test_compat.cpp)

engine_add_test(test_hardening
  test_hardening.cpp)

engine_add_test(test_safetensors
  test_safetensors.cpp)

engine_add_test(test_gptq_awq
  test_gptq_awq.cpp)

engine_add_test(test_sampling
  test_sampling.cpp)

engine_add_test(test_speculative
  test_speculative.cpp)

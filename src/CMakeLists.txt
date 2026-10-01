configure_file(common/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/common/version.h @ONLY)

# engine_core: everything except the executable entry points. Tests,
# benchmarks, CLI and server all link against this one library.
add_library(engine_core STATIC
  api/json.cpp
  api/openai.cpp
  backends/cpu/avx2/kernels_avx2.cpp
  bench/loadgen.cpp
  backends/cpu/cpu_backend.cpp
  backends/cpu/generic/kernels_generic.cpp
  chat_template/chat_template.cpp
  common/status.cpp
  dtype/dtype.cpp
  kv_cache/kv_cache.cpp
  logging/log.cpp
  memory/host_memory.cpp
  memory/storage.cpp
  metrics/metrics.cpp
  model/arch_gemma.cpp
  model/arch_llama.cpp
  model/arch_phi.cpp
  model/arch_qwen.cpp
  model/architecture.cpp
  model/transformer.cpp
  model_ir/model_config.cpp
  model_ir/tensor_registry.cpp
  loader/gguf/gguf.cpp
  loader/gguf/gguf_model.cpp
  loader/gguf/gguf_tokenizer.cpp
  loader/mapped_file.cpp
  loader/model_loader.cpp
  platform/cpu_info.cpp
  prefix_cache/hash_prefix_cache.cpp
  prefix_cache/radix_prefix_cache.cpp
  platform/isa.cpp
  platform/process_stats.cpp
  quant/dequant.cpp
  quant/quant_formats.cpp
  runtime/engine.cpp
  runtime/generator.cpp
  runtime/sequence.cpp
  runtime/text_stream.cpp
  runtime/thread_pool.cpp
  sampling/sampler.cpp
  scheduler/scheduler.cpp
  tensor/tensor.cpp
  tokenizer/pretokenizer.cpp
  tokenizer/tokenizer.cpp
  tokenizer/unicode.cpp
)

target_include_directories(engine_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(engine_core PUBLIC engine_options)
target_compile_definitions(engine_core PUBLIC
  ENGINE_HAS_AVX2=$<BOOL:${ENABLE_AVX2}>
  ENGINE_HAS_AVX512=$<BOOL:${ENABLE_AVX512}>
  ENGINE_HAS_AMX=$<BOOL:${ENABLE_AMX}>)

# SIMD tiers: per-file ISA flags only (DD-003).
if(ENABLE_AVX2)
  engine_set_isa(AVX2 backends/cpu/avx2/kernels_avx2.cpp)
endif()

add_executable(engine cli/main.cpp cli/benchmark.cpp cli/inspect.cpp cli/run.cpp)
target_link_libraries(engine PRIVATE engine_core)

# HTTP server (cpp-httplib: pinned single header, verified by SHA256).
if(ENABLE_SERVER)
  set(_httplib_dir ${CMAKE_BINARY_DIR}/_deps/httplib)
  if(NOT EXISTS ${_httplib_dir}/httplib.h)
    file(DOWNLOAD https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h ${_httplib_dir}/httplib.h
         EXPECTED_HASH SHA256=a0a0c13dc086663863dbe6730a19716f7d3744904b6205496e8301750814dbd5
         TLS_VERIFY ON STATUS _dl)
    list(GET _dl 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(FATAL_ERROR "failed to download cpp-httplib: ${_dl}")
    endif()
  endif()
  add_library(engine_server STATIC server/server.cpp bench/http_target.cpp)
  # SYSTEM: third-party header warnings must not fail our -Wall build.
  target_include_directories(engine_server SYSTEM PRIVATE ${_httplib_dir})
  target_link_libraries(engine_server PUBLIC engine_core)
  if(WIN32)
    target_link_libraries(engine_server PRIVATE ws2_32)
  endif()
  target_sources(engine PRIVATE cli/serve.cpp)
  target_link_libraries(engine PRIVATE engine_server)
  target_compile_definitions(engine PRIVATE ENGINE_HAS_SERVER=1)
  set(ENGINE_HTTPLIB_DIR ${_httplib_dir} PARENT_SCOPE)
endif()

configure_file(common/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/common/version.h @ONLY)

# engine_core: everything except the executable entry points. Tests,
# benchmarks, CLI and server all link against this one library.
add_library(engine_core STATIC
  backends/cpu/avx2/kernels_avx2.cpp
  backends/cpu/cpu_backend.cpp
  backends/cpu/generic/kernels_generic.cpp
  chat_template/chat_template.cpp
  common/status.cpp
  dtype/dtype.cpp
  kv_cache/kv_cache.cpp
  logging/log.cpp
  memory/host_memory.cpp
  memory/storage.cpp
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
  quant/dequant.cpp
  quant/quant_formats.cpp
  runtime/generator.cpp
  runtime/sequence.cpp
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

add_executable(engine cli/main.cpp cli/inspect.cpp cli/run.cpp)
target_link_libraries(engine PRIVATE engine_core)

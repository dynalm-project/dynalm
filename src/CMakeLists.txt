configure_file(common/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/common/version.h @ONLY)

# engine_core: everything except the executable entry points. Tests,
# benchmarks, CLI and server all link against this one library.
add_library(engine_core STATIC
  chat_template/chat_template.cpp
  common/status.cpp
  dtype/dtype.cpp
  logging/log.cpp
  memory/host_memory.cpp
  memory/storage.cpp
  model_ir/model_config.cpp
  model_ir/tensor_registry.cpp
  loader/gguf/gguf.cpp
  loader/gguf/gguf_model.cpp
  loader/gguf/gguf_tokenizer.cpp
  loader/mapped_file.cpp
  platform/cpu_info.cpp
  platform/isa.cpp
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

add_executable(engine cli/main.cpp cli/inspect.cpp)
target_link_libraries(engine PRIVATE engine_core)

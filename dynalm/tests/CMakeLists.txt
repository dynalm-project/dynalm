# Model-free DynaLM tests: a tiny synthetic GGUF model (tiny_gguf.h) drives the
# loader, tokenizer, scheduler, engine and server on the CPU and, when built
# with ENABLE_METAL and a GPU is present, on Metal. No model download.
function(dynalm_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE dynalm_runtime)
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/dynalm)
endfunction()

dynalm_add_test(test_engine test_engine.cpp)

if(ENABLE_SERVER)
  dynalm_add_test(test_server test_server.cpp)
  target_link_libraries(test_server PRIVATE dynalm_server)
  target_include_directories(test_server SYSTEM PRIVATE ${ENGINE_HTTPLIB_DIR})
endif()
dynalm_add_test(test_fuzz_parsers test_fuzz_parsers.cpp)

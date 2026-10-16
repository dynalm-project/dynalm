# DynaLM: the LLM inference and serving platform, built on DynaCore (DD-068).
add_subdirectory(src)
if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
  add_subdirectory(tests)  # model-free tests on a synthetic GGUF (tests/tiny_gguf.h)
endif()

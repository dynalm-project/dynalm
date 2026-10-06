# DynaLM: the LLM inference and serving platform, built on DynaCore (DD-068).
add_subdirectory(src)
if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

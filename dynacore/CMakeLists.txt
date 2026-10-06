# DynaCore: the low-level inference runtime (DD-068).
#
# Knows tensors, dtypes, quantized block formats, memory, threads, hardware,
# devices and kernels. Must not know models, file formats, tokenizers,
# requests or HTTP: its include path holds only dynacore/include, and
# tests/boundary/check_boundary.py rejects LLM-level names under dynacore/.

add_library(dynacore STATIC
  src/base/status.cpp
  src/device/device_registry.cpp
  src/execution/thread_pool.cpp
  src/hardware/cpu_info.cpp
  src/hardware/isa.cpp
  src/hardware/perf_counters.cpp
  src/hardware/process_stats.cpp
  src/hardware/system_info.cpp
  src/hardware/thread_qos.cpp
  src/ir/executor.cpp
  src/ir/ir.cpp
  src/ir/passes.cpp
  src/ir/recording_device.cpp
  src/ir/text.cpp
  src/ir/verifier.cpp
  src/kernel/kernel_plan.cpp
  src/lang/lang.cpp
  src/memory/host_memory.cpp
  src/memory/mapped_file.cpp
  src/memory/storage.cpp
  src/quantization/dequant.cpp
  src/quantization/quant_formats.cpp
  src/tensor/dtype.cpp
  src/tensor/tensor.cpp
  cpu/cpu_device.cpp
  cpu/generic/kernels_generic.cpp
  cpu/avx2/kernels_avx2.cpp
  cpu/neon/kernels_neon.cpp
)
target_include_directories(dynacore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dynacore PUBLIC engine_options)
target_link_libraries(dynacore PRIVATE ${CMAKE_DL_LIBS})  # GPU driver probe (dlopen)
if(WIN32)
  target_link_libraries(dynacore PRIVATE pdh)  # effective CPU frequency (DD-066)
endif()
target_compile_definitions(dynacore PUBLIC
  ENGINE_HAS_AVX2=$<BOOL:${ENABLE_AVX2}>
  ENGINE_HAS_AVX512=$<BOOL:${ENABLE_AVX512}>
  ENGINE_HAS_AMX=$<BOOL:${ENABLE_AMX}>
  ENGINE_HAS_NEON=$<BOOL:${ENABLE_NEON}>)

# SIMD tiers: per-file ISA flags only (DD-003).
if(ENABLE_AVX2)
  engine_set_isa(AVX2 cpu/avx2/kernels_avx2.cpp)
endif()

# dynacorec: the DynaCore compiler driver (DD-073).
add_executable(dynacorec tools/dynacorec.cpp)
target_link_libraries(dynacorec PRIVATE dynacore)
set_target_properties(dynacorec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ENABLE_TESTS)
  function(dynacore_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE dynacore GTest::gtest_main)
    target_compile_definitions(${name} PRIVATE ENGINE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                                             DYNACORE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
  endfunction()
  dynacore_add_test(dynacore_test_foundation tests/test_status.cpp tests/test_cpu_info.cpp)
  dynacore_add_test(dynacore_test_tensor tests/test_dtype.cpp tests/test_tensor.cpp)
  dynacore_add_test(dynacore_test_quant tests/test_quant.cpp)
  dynacore_add_test(dynacore_test_kernels tests/test_kernels.cpp tests/test_int8_decode.cpp)
  dynacore_add_test(dynacore_test_ir tests/test_ir.cpp)
  dynacore_add_test(dynacore_test_lang tests/test_lang.cpp)
endif()

if(ENABLE_BENCHMARKS)
  foreach(_b elementwise kernels matmul quant tensor thread_pool)
    add_executable(bench_${_b} benchmarks/bench_${_b}.cpp)
    target_link_libraries(bench_${_b} PRIVATE dynacore)
    target_include_directories(bench_${_b} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  endforeach()
endif()

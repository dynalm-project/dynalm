# DynaCore: the low-level inference runtime (DD-068).
#
# Knows tensors, dtypes, quantized block formats, memory, threads, hardware,
# devices and kernels. Must not know models, file formats, tokenizers,
# requests or HTTP: its include path holds only dynacore/include, and

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
  src/quantization/repack.cpp
  src/tensor/dtype.cpp
  src/tensor/tensor.cpp
  cpu/cpu_device.cpp
  cpu/generic/kernels_generic.cpp
  cpu/avx2/kernels_avx2.cpp
  cpu/avx2/kernels_avxvnni.cpp
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
  engine_set_isa(AVXVNNI cpu/avx2/kernels_avxvnni.cpp)
endif()

# dynacorec: the DynaCore compiler driver (DD-073).
add_executable(dynacorec tools/dynacorec.cpp)
target_link_libraries(dynacorec PRIVATE dynacore)
set_target_properties(dynacorec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ENABLE_BENCHMARKS)
  foreach(_b elementwise kernels matmul quant tensor thread_pool)
    add_executable(bench_${_b} benchmarks/bench_${_b}.cpp)
    target_link_libraries(bench_${_b} PRIVATE dynacore)
    target_include_directories(bench_${_b} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  endforeach()
endif()

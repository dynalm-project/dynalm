# DynaCore: the low-level inference runtime (DD-068).
#
# Knows tensors, dtypes, quantized block formats, memory, threads, hardware,
# devices and kernels. Must not know models, file formats, tokenizers,
# requests or HTTP: its include path holds only dynacore/include, and

add_library(dynacore STATIC
  src/base/status.cpp
  src/device/device_registry.cpp
  src/dynakernel/codegen.cpp
  src/dynakernel/compact_layout.cpp
  src/dynakernel/generated_backend.cpp
  src/dynakernel/llvm_ir.cpp
  src/dynakernel/lower.cpp
  src/dynakernel/program.cpp
  src/dynakernel/reference_backend.cpp
  src/execution/thread_pool.cpp
  src/gpu/ggml_device.cpp
  src/gpu/ggml_graph.cpp
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
  src/kernel/matmul_registry.cpp
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
  cpu/neon/kernels_neon_dot.cpp
  cpu/qmm/qmm_reference.cpp
  cpu/qmm/qmm_neon_dot.cpp
  cpu/qmm/qmm_neon_i8mm.cpp
  cpu/qmm/qmm_sme.cpp
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

# GPU backend through pinned ggml (docs/gpu-backend.md). ggml headers stay
# private to src/gpu; DynaLM never sees them.
if(DYNALM_WITH_GGML)
  target_link_libraries(dynacore PRIVATE ggml ggml-base)
  # ggml-backend-impl.h (the backend's graph_optimize, src/gpu/ggml_graph.cpp).
  target_include_directories(dynacore SYSTEM PRIVATE $<TARGET_PROPERTY:ggml-base,SOURCE_DIR>)
  target_compile_definitions(dynacore PRIVATE DYNACORE_HAS_GGML=1 DYNACORE_GGML_REVISION="${DYNALM_GGML_COMMIT}")
  foreach(_gpu METAL VULKAN CUDA)
    string(TOLOWER ${_gpu} _lib)  # ggml-metal, ggml-vulkan, ggml-cuda
    if(ENABLE_${_gpu})
      target_link_libraries(dynacore PRIVATE ggml-${_lib})
      target_compile_definitions(dynacore PUBLIC DYNACORE_HAS_${_gpu}=1)
    endif()
  endforeach()
else()
  target_compile_definitions(dynacore PRIVATE DYNACORE_HAS_GGML=0)
endif()

# SIMD tiers: per-file ISA flags only (DD-003).
if(ENABLE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$" AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  engine_set_isa(NEONDOT cpu/neon/kernels_neon_dot.cpp cpu/qmm/qmm_neon_dot.cpp)
  engine_set_isa(NEONI8MM cpu/qmm/qmm_neon_i8mm.cpp)
  # SME needs a compiler that knows the flag and the ACLE SME intrinsics
  # (Apple clang 16+, LLVM 18+); otherwise the file compiles to its stub.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sme")
  check_cxx_source_compiles("#include <arm_sme.h>
    __arm_new(\"za\") __arm_locally_streaming void f(float* y) { svbool_t p = svptrue_b32(); svfloat32_t a = svdup_f32(1.0f);
      svmopa_za32_f32_m(0, p, p, a, a); svst1_hor_za32(0, 0, p, y); }
    int main() { return 0; }" DYNACORE_SME_COMPILES)
  unset(CMAKE_REQUIRED_FLAGS)
  if(DYNACORE_SME_COMPILES)
    engine_set_isa(SME cpu/qmm/qmm_sme.cpp)
  endif()
endif()
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
  # CpuDevice::attention at decode shapes vs threads and context.
  add_executable(bench_attention benchmarks/bench_attention.cpp)
  target_link_libraries(bench_attention PRIVATE dynacore)
  target_include_directories(bench_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  # qmm kernels vs CpuDevice::matmul on transformer shapes.
  add_executable(bench_qmm benchmarks/bench_qmm.cpp)
  target_link_libraries(bench_qmm PRIVATE dynacore)
  target_include_directories(bench_qmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  # Single-thread FFN prefill kernel throughput on a pinned CPU (DynaLM Next).
  add_executable(bench_ffn_kernels benchmarks/bench_ffn_kernels.cpp)
  target_link_libraries(bench_ffn_kernels PRIVATE dynacore)
  # DynaKernel IR v0 (docs/dynakernel.md): no model files needed.
  add_executable(test_dynakernel tests/test_dynakernel.cpp)
  target_link_libraries(test_dynakernel PRIVATE dynacore)
  add_executable(test_dynakernel_lowering tests/test_dynakernel_lowering.cpp)
  target_link_libraries(test_dynakernel_lowering PRIVATE dynacore)
  add_executable(test_dynakernel_quant tests/test_dynakernel_quant.cpp)
  target_link_libraries(test_dynakernel_quant PRIVATE dynacore)
  add_executable(test_packed_8rows tests/test_packed_8rows.cpp)
  target_link_libraries(test_packed_8rows PRIVATE dynacore)
  # Integer-activation kernels of every tier (DD-053 int8, DD-076 int16) vs an exact reference.
  add_executable(test_int_kernels tests/test_int_kernels.cpp)
  target_link_libraries(test_int_kernels PRIVATE dynacore)
  add_executable(test_int16_prefill tests/test_int16_prefill.cpp)
  target_link_libraries(test_int16_prefill PRIVATE dynacore)
  # Native quantized matmul kernels (qmm) of every compiled tier vs the reference.
  add_executable(test_qmm tests/test_qmm.cpp)
  target_link_libraries(test_qmm PRIVATE dynacore)
  # CpuDevice attention: the SME prompt-chunk path vs the per-row path.
  add_executable(test_cpu_attention tests/test_cpu_attention.cpp)
  target_link_libraries(test_cpu_attention PRIVATE dynacore)
  add_executable(test_thread_pool tests/test_thread_pool.cpp)
  target_link_libraries(test_thread_pool PRIVATE dynacore)
  # GPU device through pinned ggml (docs/gpu-backend.md); skips without a GPU backend.
  add_executable(test_ggml_device tests/test_ggml_device.cpp)
  target_link_libraries(test_ggml_device PRIVATE dynacore)
  # ggml-metal's concurrency check with cmake/patches (skips without Metal).
  add_executable(test_ggml_metal_ranges tests/test_ggml_metal_ranges.cpp)
  target_link_libraries(test_ggml_metal_ranges PRIVATE dynacore)
  if(ENABLE_METAL)
    target_link_libraries(test_ggml_metal_ranges PRIVATE ggml-metal ggml-cpu ggml-base)
    target_include_directories(test_ggml_metal_ranges SYSTEM PRIVATE $<TARGET_PROPERTY:ggml-metal,SOURCE_DIR>)
  endif()
  # CPU matmul kernel registry (DynaLM Next): selection, execution, diagnostics.
  add_executable(test_matmul_registry tests/test_matmul_registry.cpp)
  target_link_libraries(test_matmul_registry PRIVATE dynacore)
  # Generated C++ backend (docs/dynakernel.md): dkgen writes each unit into
  # the build tree at build time; only tests and benchmarks link them. The
  # AVX2 variant gets AVX2 flags on its generated file only (DD-003).
  add_executable(dkgen tools/dkgen.cpp)
  target_link_libraries(dkgen PRIVATE dynacore)
  set(DK_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/dynakernel_gen)
  function(dk_generated_library target set unit)
    set(_srcs)
    foreach(_isa scalar avx2)
      if(_isa STREQUAL "avx2" AND NOT ENABLE_AVX2)
        continue()
      endif()
      set(_suffix "")
      if(_isa STREQUAL "avx2")
        set(_suffix "_avx2")
      endif()
      set(_out ${DK_GEN_DIR}/dk_generated_${unit}${_suffix}.cpp)
      add_custom_command(
        OUTPUT ${_out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DK_GEN_DIR}
        COMMAND dkgen --out ${_out} --set ${set} --isa ${_isa}
        DEPENDS dkgen
        COMMENT "dkgen: generating DynaKernel ${unit} kernels (${_isa})"
        VERBATIM)
      if(_isa STREQUAL "avx2")
        engine_set_isa(AVX2 ${_out})
      endif()
      list(APPEND _srcs ${_out})
    endforeach()
    if(ENABLE_AVX2)
      # INT8 prototype (sub-scaled activations, interleaved Q4_K): AVX2 only.
      set(_out ${DK_GEN_DIR}/dk_generated_${unit}_int8_avx2.cpp)
      add_custom_command(
        OUTPUT ${_out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DK_GEN_DIR}
        COMMAND dkgen --out ${_out} --set ${set} --isa avx2 --kind int8
        DEPENDS dkgen
        COMMENT "dkgen: generating DynaKernel ${unit} int8 kernels (avx2)"
        VERBATIM)
      engine_set_isa(AVX2 ${_out})
      list(APPEND _srcs ${_out})
    endif()
    add_library(${target} STATIC ${_srcs})
    target_compile_definitions(${target} PUBLIC DK_GENERATED_HAS_AVX2=$<BOOL:${ENABLE_AVX2}>)
    target_link_libraries(${target} PRIVATE dynacore)
  endfunction()
  dk_generated_library(dynakernel_generated test q4k)
  dk_generated_library(dynakernel_generated_bench bench q4k_bench)
  add_executable(test_dynakernel_codegen tests/test_dynakernel_codegen.cpp)
  target_link_libraries(test_dynakernel_codegen PRIVATE dynacore dynakernel_generated)
  add_executable(test_dynakernel_int8gen tests/test_dynakernel_int8gen.cpp)
  target_link_libraries(test_dynakernel_int8gen PRIVATE dynacore dynakernel_generated)
  # LLVM native backend (prototype, docs/dynakernel.md): LLVM's C API library
  # is loaded at run time (DYNACORE_LLVM_C), so this builds without LLVM and
  # reports kUnsupported when the library is absent. Only tests and
  # benchmarks link it.
  add_library(dynakernel_llvm STATIC src/dynakernel/llvm_jit.cpp src/dynakernel/llvm_backend.cpp)
  target_link_libraries(dynakernel_llvm PUBLIC dynacore PRIVATE ${CMAKE_DL_LIBS})
  add_executable(test_dynakernel_llvm tests/test_dynakernel_llvm.cpp)
  target_link_libraries(test_dynakernel_llvm PRIVATE dynacore dynakernel_llvm dynakernel_generated)
  # Controlled microbenchmark: generated scalar / AVX2 vs reference paths.
  add_executable(bench_dynakernel_q4k benchmarks/bench_dynakernel_q4k.cpp)
  target_link_libraries(bench_dynakernel_q4k PRIVATE dynacore dynakernel_generated_bench)
  target_include_directories(bench_dynakernel_q4k PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  # LLVM-generated kernels against the reference, DynaCore and generated C++ paths.
  add_executable(bench_dynakernel_llvm benchmarks/bench_dynakernel_llvm.cpp)
  target_link_libraries(bench_dynakernel_llvm PRIVATE dynacore dynakernel_llvm dynakernel_generated_bench)
  target_include_directories(bench_dynakernel_llvm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

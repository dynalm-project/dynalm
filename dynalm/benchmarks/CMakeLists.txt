function(engine_add_benchmark name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE dynalm_runtime)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dynacore/benchmarks)
endfunction()

engine_add_benchmark(bench_foundation bench_foundation.cpp)
engine_add_benchmark(bench_loader bench_loader.cpp)
engine_add_benchmark(bench_tokenizer bench_tokenizer.cpp)
engine_add_benchmark(bench_decode_context bench_decode_context.cpp)
engine_add_benchmark(bench_kv bench_kv.cpp)
engine_add_benchmark(bench_batch_decode bench_batch_decode.cpp)
engine_add_benchmark(bench_scheduler bench_scheduler.cpp)
engine_add_benchmark(bench_long_prompt bench_long_prompt.cpp)
engine_add_benchmark(bench_prefix bench_prefix.cpp)
engine_add_benchmark(bench_profile bench_profile.cpp)
engine_add_benchmark(bench_sampling bench_sampling.cpp)
engine_add_benchmark(bench_speculative bench_speculative.cpp)
engine_add_benchmark(bench_decode_matmul bench_decode_matmul.cpp)
engine_add_benchmark(bench_int8_accuracy bench_int8_accuracy.cpp)
engine_add_benchmark(bench_small_m bench_small_m.cpp)
engine_add_benchmark(bench_attn_accuracy bench_attn_accuracy.cpp)
engine_add_benchmark(bench_op_trace bench_op_trace.cpp)
engine_add_benchmark(bench_compiled bench_compiled.cpp)
engine_add_benchmark(bench_kv_accuracy bench_kv_accuracy.cpp)
engine_add_benchmark(bench_alloc bench_alloc.cpp)
# Scheduler test for the experimental solo-prefill chunk (needs a model file:
# test_solo_prefill <model.gguf> [threads]).
engine_add_benchmark(test_solo_prefill test_solo_prefill.cpp)
# Load-generator test: closed / Poisson / trace arrivals against a fake target
# (test_loadgen_openloop <model.gguf>; uses only the tokenizer).
engine_add_benchmark(test_loadgen_openloop test_loadgen_openloop.cpp)
# Scheduler test for the experimental prefill order (srpt) and the scheduler trace
# (test_prefill_priority <model.gguf> [threads]).
engine_add_benchmark(test_prefill_priority test_prefill_priority.cpp)

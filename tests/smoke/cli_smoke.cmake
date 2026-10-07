# CLI smoke test: a fresh build (or a fresh install) can start, report its
# system, and run inference end to end, offline, in a few seconds.
#
#   cmake -DDYNALM=<path/to/dynalm> [-DDYNACOREC=<path/to/dynacorec>]
#         -DMODEL=<tiny.gguf> [-DDYNA=<file.dyna>] -P tests/smoke/cli_smoke.cmake
#
# Runs: --version, doctor (must report Ready and a passing DynaCore self-test),
# doctor --json, a short generation with the best CPU tier, the same with
# DYNACORE_ISA=generic (the fallback every CPU has), the same in compiled
# execution, and dynacorec on the example program (bit-exact check).

cmake_minimum_required(VERSION 3.24)

function(run_step name)
  cmake_parse_arguments(A "" "" "COMMAND;ENV;EXPECT" ${ARGN})
  if(A_ENV)
    set(cmd ${CMAKE_COMMAND} -E env ${A_ENV} ${A_COMMAND})
  else()
    set(cmd ${A_COMMAND})
  endif()
  execute_process(COMMAND ${cmd} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "smoke step '${name}' failed (exit ${rc})\n--- stdout\n${out}\n--- stderr\n${err}")
  endif()
  foreach(want IN LISTS A_EXPECT)
    string(FIND "${out}${err}" "${want}" pos)
    if(pos EQUAL -1)
      message(FATAL_ERROR "smoke step '${name}': output lacks '${want}'\n--- stdout\n${out}\n--- stderr\n${err}")
    endif()
  endforeach()
  message(STATUS "ok: ${name}")
endfunction()

if(NOT DYNALM OR NOT MODEL)
  message(FATAL_ERROR "usage: cmake -DDYNALM=... -DMODEL=... -P cli_smoke.cmake")
endif()

run_step(version COMMAND ${DYNALM} --version EXPECT "DynaLM" "DynaCore")
run_step(doctor COMMAND ${DYNALM} doctor EXPECT "DynaCore:      OK" "Status:        Ready")
run_step(doctor-json COMMAND ${DYNALM} doctor --json EXPECT "\"dynacore_ok\":true" "\"ready\":true")
run_step(run-best COMMAND ${DYNALM} -q run ${MODEL} -p "hello world" -n 8 --temp 0 --raw EXPECT "generated 8 tok")
run_step(run-generic ENV DYNACORE_ISA=generic
         COMMAND ${DYNALM} -q run ${MODEL} -p "hello world" -n 8 --temp 0 --raw EXPECT "generated 8 tok")
run_step(doctor-generic ENV DYNACORE_ISA=generic COMMAND ${DYNALM} doctor EXPECT "CPU / generic")
run_step(run-compiled COMMAND ${DYNALM} -q run ${MODEL} -p "hello world" -n 8 --temp 0 --raw --execution compiled
         EXPECT "generated 8 tok")
if(DYNACOREC AND DYNA)
  run_step(dynacorec COMMAND ${DYNACOREC} ${DYNA} --benchmark --iters 2 --threads 2 EXPECT "bit-identical")
endif()

# Applies cmake/patches/*.patch to the pinned ggml checkout, in name order
# (FetchContent PATCH_COMMAND, run in the ggml source dir; cmake/Ggml.cmake).
# Re-runnable: a patch that is already applied is skipped, one that applies
# neither way stops the build. --ignore-whitespace: a checkout with CRLF line
# endings (git core.autocrlf, Windows) still matches the LF patch.
#
#   cmake -DPATCH_DIR=<dir> -P GgmlPatch.cmake

find_package(Git REQUIRED)
file(GLOB _patches "${PATCH_DIR}/*.patch")
list(SORT _patches)
foreach(_p IN LISTS _patches)
  get_filename_component(_name "${_p}" NAME)
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace --reverse --check "${_p}"
                  RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
  if(_applied EQUAL 0)
    message(STATUS "ggml patch ${_name}: already applied")
    continue()
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${_p}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "ggml patch ${_name} does not apply to this ggml tree")
  endif()
  message(STATUS "ggml patch ${_name}: applied")
endforeach()

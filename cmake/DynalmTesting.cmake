# GoogleTest for the DynaCore and DynaLM test suites.
include(FetchContent)

# Prefer a system GoogleTest; fall back to a pinned download.
find_package(GTest 1.14 CONFIG QUIET)
if(NOT GTest_FOUND)
  if(DYNALM_STATIC_RUNTIME)  # match the MSVC runtime of the engine
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
  else()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

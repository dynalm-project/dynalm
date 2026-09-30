include(FetchContent)

# Prefer a system GoogleTest; fall back to a pinned download.
find_package(GTest 1.14 CONFIG QUIET)
if(NOT GTest_FOUND)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match MSVC /MD runtime
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

function(engine_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE engine_core GTest::gtest_main)
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

engine_add_test(test_foundation
  test_status.cpp
  test_logging.cpp
  test_cpu_info.cpp)

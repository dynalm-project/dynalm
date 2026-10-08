# Compiler configuration shared by every engine target.
#
# Rule: no global -march / /arch flags. The baseline binary must run on any
# x86-64 (or aarch64) CPU. SIMD variants are compiled per-source-file with
# engine_set_isa() and selected at runtime.

add_library(engine_options INTERFACE)

if(MSVC)
  target_compile_options(engine_options INTERFACE
    /W4 /permissive- /utf-8 /Zc:__cplusplus /Zc:preprocessor /EHsc /MP
    /wd4324  # structure padded due to alignment specifier (intentional)
    $<$<CONFIG:Release,RelWithDebInfo>:/O2 /Ob2 /Oi /GS->)
  target_compile_definitions(engine_options INTERFACE
    NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(engine_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
    # GCC 12/13 false positives on std::variant holding std::string (GCC PR 105562).
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
    $<$<CONFIG:Release,RelWithDebInfo>:-O3>)
  find_package(Threads REQUIRED)
  target_link_libraries(engine_options INTERFACE Threads::Threads)
endif()

# --- Sanitizers ------------------------------------------------------------
if(ENABLE_TSAN AND (ENABLE_ASAN OR ENABLE_UBSAN))
  message(FATAL_ERROR "TSAN cannot be combined with ASAN/UBSAN")
endif()
if(MSVC)
  if(ENABLE_ASAN)
    target_compile_options(engine_options INTERFACE /fsanitize=address /Zi)
    target_link_options(engine_options INTERFACE /INCREMENTAL:NO /DEBUG)
  endif()
  if(ENABLE_UBSAN OR ENABLE_TSAN)
    message(WARNING "UBSAN/TSAN are not supported by MSVC; use the Linux clang/gcc presets.")
  endif()
else()
  set(_san "")
  if(ENABLE_ASAN)
    list(APPEND _san address)
  endif()
  if(ENABLE_UBSAN)
    list(APPEND _san undefined)
  endif()
  if(ENABLE_TSAN)
    list(APPEND _san thread)
  endif()
  if(_san)
    list(JOIN _san "," _san_csv)
    # Sanitizer findings fail the test instead of printing and continuing.
    target_compile_options(engine_options INTERFACE -fsanitize=${_san_csv} -fno-sanitize-recover=all
                           -fno-omit-frame-pointer -g)
    target_link_options(engine_options INTERFACE -fsanitize=${_san_csv})
  endif()
endif()

# --- Per-file ISA flags ----------------------------------------------------
# engine_set_isa(<isa> <files...>) where isa is AVX2 | AVX512 | AMX.
function(engine_set_isa isa)
  if(MSVC)
    if(isa STREQUAL "AVX2" OR isa STREQUAL "AVXVNNI")
      set(_flags /arch:AVX2)
    elseif(isa STREQUAL "AVX512" OR isa STREQUAL "AMX")
      set(_flags /arch:AVX512)
    endif()
  else()
    if(isa STREQUAL "AVX2")
      set(_flags -mavx2 -mfma -mf16c)
    elseif(isa STREQUAL "AVXVNNI")
      set(_flags -mavx2 -mfma -mf16c -mavxvnni)
    elseif(isa STREQUAL "AVX512")
      set(_flags -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma -mf16c)
    elseif(isa STREQUAL "AMX")
      set(_flags -mavx512f -mavx512bw -mavx512vl -mamx-tile -mamx-int8 -mamx-bf16)
    endif()
  endif()
  if(NOT _flags)
    message(FATAL_ERROR "engine_set_isa: unknown ISA '${isa}'")
  endif()
  set_source_files_properties(${ARGN} PROPERTIES COMPILE_OPTIONS "${_flags}")
endfunction()

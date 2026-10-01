# Compiler diagnostics and sanitizers for project-owned C++ and CUDA.
#
# Included after third-party targets are declared, so the directory-scoped options below apply
# only to NInfer targets (src/, apps/, tests/, bench/). Policy and rationale:
# docs/maintainer/code-quality.md.

option(NINFER_WARNINGS_AS_ERRORS "Treat compiler warnings in project code as errors" ON)
set(NINFER_SANITIZE "" CACHE STRING
  "Host sanitizers for project code, e.g. address,undefined (use a separate build tree)")

# Each flag targets a defect class the builder toolchain (GCC 13, nvcc 13.1) reports reliably:
# shadowed locals, missing virtual destructors, hidden overloads, null dereference on a proven
# path, unannotated switch fallthrough, unhandled enumerators, format-string mismatch, unused
# code, and duplicated conditions or branches. Not enabled: conversion warnings (kernel index
# arithmetic is checked by clang-tidy's bugprone-implicit-widening-* instead), -Wdouble-promotion
# (host code only, where it fires on float varargs), and -Wmissing-field-initializers (aggregate
# initialization that relies on default member initializers is the intended idiom).
set(ninfer_host_warnings
  -Wall
  -Wextra
  -Wshadow
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wnull-dereference
  -Wimplicit-fallthrough
  -Wformat=2
  -Wmisleading-indentation
  -Wno-missing-field-initializers)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND ninfer_host_warnings -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
endif()
if(NINFER_WARNINGS_AS_ERRORS)
  list(APPEND ninfer_host_warnings -Werror)
endif()
list(JOIN ninfer_host_warnings "," ninfer_host_warnings_csv)

# nvcc front-end diagnostics: member-initializer order, kernels launched on the legacy default
# stream (which breaks stream ordering and CUDA Graph capture), and extended lambdas that
# capture `this` by pointer. Diagnostic numbers are displayed so a deliberate case can be
# suppressed precisely with `#pragma nv_diag_suppress <number>`.
set(ninfer_cuda_warnings
  --Wreorder
  --Wdefault-stream-launch
  --Wext-lambda-captures-this
  -Xcudafe=--display_error_number)
if(NINFER_WARNINGS_AS_ERRORS)
  list(APPEND ninfer_cuda_warnings --Werror=all-warnings)
endif()

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:${ninfer_host_warnings}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${ninfer_host_warnings_csv}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${ninfer_cuda_warnings}>")

if(NINFER_SANITIZE)
  set(ninfer_sanitize_flags -fsanitize=${NINFER_SANITIZE} -fno-omit-frame-pointer)
  list(JOIN ninfer_sanitize_flags "," ninfer_sanitize_flags_csv)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:${ninfer_sanitize_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${ninfer_sanitize_flags_csv}>")
  add_link_options(
    "$<$<LINK_LANGUAGE:C,CXX>:-fsanitize=${NINFER_SANITIZE}>"
    "$<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${NINFER_SANITIZE}>")
endif()

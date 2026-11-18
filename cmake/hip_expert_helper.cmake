# Opt-in gfx1201 HIP toolchain for the Qwen3.8-Flash-Next R9700 expert helper
# (docs/maintainer/qwen3.8-flash-next-expert-helper.md). Included from the top level only when
# NINFER_ENABLE_HIP_EXPERT_HELPER is ON, so the default build performs no HIP discovery.
#
# HIP translation units are compiled by ROCm's clang for gfx1201 alone and never include CUDA
# headers; CUDA translation units never include HIP headers. Shared contracts between the two
# sides are pure host headers.

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "HIP architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1201")
  message(FATAL_ERROR
    "The HIP expert helper supports only CMAKE_HIP_ARCHITECTURES=gfx1201; got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

# CMake rejects the hipcc wrapper as a HIP compiler: resolve ROCm's clang++ from ROCM_PATH (the
# dual-GPU builder sets it to /opt/rocm/core-10.0) before enabling the language.
if(NOT DEFINED CMAKE_HIP_COMPILER)
  find_program(NINFER_ROCM_CLANGXX NAMES clang++
    HINTS "$ENV{ROCM_PATH}/lib/llvm/bin" /opt/rocm/core-10.0/lib/llvm/bin /opt/rocm/llvm/bin
    NO_DEFAULT_PATH REQUIRED)
  set(CMAKE_HIP_COMPILER "${NINFER_ROCM_CLANGXX}" CACHE FILEPATH
    "ROCm clang++ for the gfx1201 expert helper")
endif()

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
# HIP objects contain complete device images and expose only host calls across translation units.
# Keep ordinary C++/CUDA consumers on their own linker: the HIP driver's archive bundling would
# otherwise pull unused helper kernels and runtimes into CPU-only executables.
set(CMAKE_HIP_LINKER_PREFERENCE_PROPAGATES OFF)

# Host-only HIP RAII uses the SDK's C++ headers/runtime without exporting their include paths or
# platform definitions to CUDA consumers. Link hip::host privately on that owning target.
find_package(hip REQUIRED CONFIG
  HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/cmake/hip")

# The project's host diagnostics (cmake/warnings.cmake) for HIP translation units, minus the
# GCC-only flags clang does not know.
set(ninfer_hip_warnings
  -Wall
  -Wextra
  -Wshadow
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wimplicit-fallthrough
  -Wformat=2
  -Wmisleading-indentation
  -Wno-missing-field-initializers)
if(NINFER_WARNINGS_AS_ERRORS)
  list(APPEND ninfer_hip_warnings -Werror)
endif()

# Applies the include roots, diagnostics and gfx1201 code object to a HIP target.
function(ninfer_hip_target target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
  set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES gfx1201)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${ninfer_hip_warnings}>)
endfunction()

# CPU grammar compilation/matching only. NInfer owns tokenization, GPU sampling,
# speculative state, and output publication; no Python or inference runtime is linked.
include(FetchContent)
FetchContent_Declare(ninfer_xgrammar_source
  GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
  GIT_TAG 4546988d1cf51670a10eafb8795d37e4a2f0d833 # v0.2.5.post1
  GIT_SUBMODULES 3rdparty/dlpack)
FetchContent_GetProperties(ninfer_xgrammar_source)
if(NOT ninfer_xgrammar_source_POPULATED)
  FetchContent_Populate(ninfer_xgrammar_source)
endif()

# Do not import upstream's Python-oriented top-level configuration or change
# NInfer's global compiler flags. This is the complete upstream C++ library lane.
set(grammar_root "${ninfer_xgrammar_source_SOURCE_DIR}")
# Keep downloaded source immutable. Build-copy patches provide exact unordered
# named properties and Qwen framing/key exclusion; both generation and final
# JSON validation use the same object constraints.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/xgrammar-qwen-framing.patch"
  "${CMAKE_CURRENT_LIST_DIR}/xgrammar-object-order.patch")
set(grammar_patch_dir "${CMAKE_CURRENT_BINARY_DIR}/ninfer_xgrammar_format")
file(MAKE_DIRECTORY "${grammar_patch_dir}")
configure_file("${grammar_root}/cpp/json_schema_converter_ext.cc"
               "${grammar_patch_dir}/json_schema_converter_ext.cc" COPYONLY)
configure_file("${grammar_root}/cpp/json_schema_converter.cc"
               "${grammar_patch_dir}/json_schema_converter.cc" COPYONLY)
find_program(NINFER_PATCH_EXECUTABLE patch REQUIRED)
execute_process(COMMAND "${NINFER_PATCH_EXECUTABLE}" --batch --fuzz=0 -p1 -i
  "${CMAKE_CURRENT_LIST_DIR}/xgrammar-qwen-framing.patch"
  WORKING_DIRECTORY "${grammar_patch_dir}" RESULT_VARIABLE grammar_patch_result
  ERROR_VARIABLE grammar_patch_error)
if(NOT grammar_patch_result EQUAL 0)
  message(FATAL_ERROR "Qwen grammar framing patch failed: ${grammar_patch_error}")
endif()
execute_process(COMMAND "${NINFER_PATCH_EXECUTABLE}" --batch --fuzz=0 -p1 -i
  "${CMAKE_CURRENT_LIST_DIR}/xgrammar-object-order.patch"
  WORKING_DIRECTORY "${grammar_patch_dir}" RESULT_VARIABLE grammar_patch_result
  ERROR_VARIABLE grammar_patch_error)
if(NOT grammar_patch_result EQUAL 0)
  message(FATAL_ERROR "Object grammar order patch failed: ${grammar_patch_error}")
endif()
add_library(ninfer_tool_grammar STATIC
  ${grammar_root}/cpp/compiled_grammar.cc
  ${grammar_root}/cpp/config.cc
  ${grammar_root}/cpp/earley_parser.cc
  ${grammar_root}/cpp/fsm.cc
  ${grammar_root}/cpp/fsm_builder.cc
  ${grammar_root}/cpp/grammar.cc
  ${grammar_root}/cpp/grammar_builder.cc
  ${grammar_root}/cpp/grammar_compiler.cc
  ${grammar_root}/cpp/grammar_functor.cc
  ${grammar_root}/cpp/grammar_matcher.cc
  ${grammar_root}/cpp/grammar_parser.cc
  ${grammar_root}/cpp/grammar_printer.cc
  ${grammar_patch_dir}/json_schema_converter.cc
  ${grammar_patch_dir}/json_schema_converter_ext.cc
  ${grammar_root}/cpp/lark_converter.cc
  ${grammar_root}/cpp/regex_converter.cc
  ${grammar_root}/cpp/structural_tag.cc
  ${grammar_root}/cpp/support/logging.cc
  ${grammar_root}/cpp/support/recursion_guard.cc
  ${grammar_root}/cpp/testing.cc
  ${grammar_root}/cpp/tokenizer_info.cc)
target_include_directories(ninfer_tool_grammar SYSTEM PUBLIC
  ${grammar_root}/include ${grammar_root}/3rdparty/dlpack/include)
target_include_directories(ninfer_tool_grammar SYSTEM PRIVATE ${grammar_root}/3rdparty/picojson)
target_include_directories(ninfer_tool_grammar SYSTEM PRIVATE ${grammar_root}/cpp)
target_compile_definitions(ninfer_tool_grammar PRIVATE XGRAMMAR_ENABLE_CPPTRACE=0)
target_link_libraries(ninfer_tool_grammar PRIVATE Threads::Threads)

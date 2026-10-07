find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Optional command prefix for every test, e.g. "compute-sanitizer;--tool;memcheck;--error-exitcode;1".
# Only test registration depends on it, so changing it re-generates CTest files without
# recompiling. ./scripts/run-unit-tests.sh --compute-sanitizer sets it.
set(NINFER_TEST_LAUNCHER "" CACHE STRING "Command prefix for every registered test")

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test
  COMMAND ${NINFER_TEST_LAUNCHER} $<TARGET_FILE:ninfer_public_api_test>)

function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${NINFER_TEST_LAUNCHER} $<TARGET_FILE:${name}>)
endfunction()

# Op tests carry the `kernel` label: they are the tests compute-sanitizer runs over.
function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_property(TEST ${name} APPEND PROPERTY LABELS kernel)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

# Registers `<name>_sanitizer`: the test's `--sanitizer` cases (tests/ops/sanitizer_scope.h),
# labelled `sanitizer` and `racecheck`, the sets the compute-sanitizer gates run.
function(ninfer_add_sanitizer_cases name)
  add_test(NAME ${name}_sanitizer
    COMMAND ${NINFER_TEST_LAUNCHER} $<TARGET_FILE:${name}> --sanitizer)
  set_property(TEST ${name}_sanitizer PROPERTY LABELS sanitizer racecheck)
endfunction()

ninfer_add_op_test(ninfer_dual_rmsnorm_test
  SOURCES ops/test_dual_rmsnorm.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_kv_projection_test
  SOURCES ops/linear/test_context_kv.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_normalized_rope_kv_append_test
  SOURCES ops/test_normalized_rope_kv_append.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_grouped_dynamic_conv_composite_test
  SOURCES ops/test_grouped_dynamic_conv_composite.cpp
  LIBRARIES ninfer_ops)
ninfer_add_sanitizer_cases(ninfer_grouped_dynamic_conv_composite_test)

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ninfer_media_decode)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_request_memory_test SOURCES test_request_memory.cpp)
ninfer_add_test(ninfer_admission_policy_test SOURCES test_admission_policy.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ninfer_add_test(ninfer_typical_cycle_test SOURCES test_typical_cycle.cpp)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_kv_ram_cache_test
  SOURCES test_kv_ram_cache.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_ram_cache_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_kv_disk_cache_test
  SOURCES test_kv_disk_cache.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_disk_cache_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_kv_ram_cache_perf_test
  SOURCES test_kv_ram_cache_perf.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_ram_cache_perf_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_kv_disk_cache_perf_test
  SOURCES test_kv_disk_cache_perf.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_disk_cache_perf_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_kv_ram_cache_opt_test
  SOURCES test_kv_ram_cache_opt.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_ram_cache_opt_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_kv_ram_cache_large_test
  SOURCES test_kv_ram_cache_large.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_kv_ram_cache_large_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_generation_recovery_test
  SOURCES targets/qwen3_6/test_generation_recovery.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_generation_recovery_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_recovery_executor_test
  SOURCES targets/qwen3_6/test_recovery_executor.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_recovery_executor_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_tool_grammar_test
  SOURCES targets/qwen3_6/test_tool_grammar.cpp
  LIBRARIES ninfer_engine ninfer_token_grammar)
target_include_directories(ninfer_qwen3_6_tool_grammar_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer_token_grammar)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_tool_mask_exchange_test
  SOURCES targets/qwen3_6/test_tool_mask_exchange.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_tool_mask_exchange_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_incremental_encode_test
  SOURCES targets/qwen3_6/test_incremental_host_encode.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_incremental_encode_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
find_package(Threads REQUIRED)
target_link_libraries(ninfer_qwen3_6_incremental_encode_test PRIVATE Threads::Threads)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_dflash_vision_capability_test
  SOURCES targets/qwen3_6/test_dflash_vision_capability.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_dflash_vision_capability_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_qwen3_6_adaptive_draft_test
  SOURCES targets/qwen3_6/test_adaptive_draft.cpp)
target_include_directories(ninfer_qwen3_6_adaptive_draft_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_score_index_test
  SOURCES targets/qwen3_6/test_score_index.cpp)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_recovery_kv_real_test
  SOURCES targets/qwen3_6_27b/test_recovery_kv_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_27b_recovery_kv_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
set_tests_properties(ninfer_qwen3_6_27b_recovery_kv_real_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 900)
ninfer_add_test(ninfer_qwen3_6_27b_ram_real_test
  SOURCES targets/qwen3_6_27b/test_engine_ram_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_27b_ram_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_27b_ram_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_cache_admission_real_test
  SOURCES targets/qwen3_6_27b/test_engine_cache_admission_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_cache_admission_real_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 300)
ninfer_add_test(ninfer_qwen3_6_27b_cache_interleavings_real_test
  SOURCES targets/qwen3_6_27b/test_engine_cache_interleavings_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_cache_interleavings_real_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 900)
foreach(backend mtp dflash)
  add_test(NAME ninfer_qwen3_6_27b_${backend}_cache_interleavings_real_test
    COMMAND ninfer_qwen3_6_27b_cache_interleavings_real_test --backend ${backend})
  set_tests_properties(ninfer_qwen3_6_27b_${backend}_cache_interleavings_real_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 900)
endforeach()
ninfer_add_test(ninfer_qwen3_6_27b_dflash_cache_cancel_real_test
  SOURCES targets/qwen3_6_27b/test_engine_dflash_cache_cancel_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_dflash_cache_cancel_real_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 900)
ninfer_add_test(ninfer_qwen3_6_27b_disk_real_test
  SOURCES targets/qwen3_6_27b/test_engine_disk_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_27b_disk_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_27b_disk_real_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME ninfer_qwen3_6_27b_disk_metadata_real_test
  COMMAND ninfer_qwen3_6_27b_disk_real_test --case metadata)
set_tests_properties(ninfer_qwen3_6_27b_disk_metadata_real_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 900)
ninfer_add_test(ninfer_qwen3_6_27b_context_checkpoint_real_test
  SOURCES targets/qwen3_6_27b/test_engine_context_checkpoint_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_6_27b_context_checkpoint_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_27b_context_checkpoint_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_output_constraint_real_test
  SOURCES targets/qwen3_6_27b/test_output_constraint_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_8_27b_output_constraint_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_dflash_real_test
  SOURCES targets/qwen3_6_27b/test_engine_dflash_real.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_8_27b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_logprobs_real_test
  SOURCES targets/qwen3_6_27b/test_engine_logprobs_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_8_27b_logprobs_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_dflash_capacity_real_test
  SOURCES targets/qwen3_6_27b/test_dflash_capacity_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_8_27b_dflash_capacity_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
set_tests_properties(ninfer_qwen3_8_27b_dflash_capacity_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_verify_score_real_test
  SOURCES targets/qwen3_6_27b/test_verify_score_real.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_qwen3_8_27b_verify_score_real_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
set_tests_properties(ninfer_qwen3_8_27b_verify_score_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_mtp_nvfp4_real_test
  SOURCES targets/qwen3_6_27b/test_engine_mtp_nvfp4_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_8_27b_mtp_nvfp4_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)

# Opt-in target diagnostic used by tools/parity/qwen3_6_27b/vision.py. It observes the exact
# production Vision schedule at every material numerical boundary and therefore is built with the
# tests, but it is not a standalone CTest (the Python driver supplies an artifact, prompt, and
# output directory).
add_executable(ninfer_qwen3_8_27b_vision_trace
  targets/qwen3_6_27b/vision_trace_real.cpp)
target_include_directories(ninfer_qwen3_8_27b_vision_trace PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_8_27b_vision_trace PRIVATE
  ninfer_engine ninfer_product_prompt_input ninfer_artifact ninfer_core CUDA::cudart)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_ram_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_ram_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_ram_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_cli_options_test
  SOURCES test_cli_options.cpp
          ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp)
target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps)
ninfer_add_test(ninfer_serve_system_prepend_real_test
  SOURCES test_serve_system_prepend_real.cpp
  LIBRARIES ninfer_serve ninfer_engine)
set_tests_properties(ninfer_serve_system_prepend_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_serve_metrics_test
  SOURCES test_serve_metrics.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_serve_metrics_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  a8_activation
  add_bias
  gelu
  silu_mul
  residual_add
  residual_rmsnorm
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  nll_from_logits
  token_logprobs
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  vision_attention
  sampling
  grouped_dynamic_conv
  dflash2_path_select
  scalar
  cast
  scatter
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_gqa_attention_test
  SOURCES ops/test_gqa_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_a4_activation_test
  SOURCES ops/test_a4_activation.cpp
  LIBRARIES ninfer_ops)
ninfer_add_sanitizer_cases(ninfer_a4_activation_test)
ninfer_add_op_test(ninfer_gqa_xattn_sort_test
  SOURCES ops/test_gqa_xattn_sort.cu
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_bidirectional_gqa_attention_test
  SOURCES ops/test_bidirectional_gqa_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_swa_test
  SOURCES ops/test_swa.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_prefix_test
  SOURCES ops/test_kv_cache_append_prefix.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_history_test
  SOURCES ops/test_gdn_history.cpp
  LIBRARIES ninfer_ops)
ninfer_add_sanitizer_cases(ninfer_gdn_history_test)
# The all-layer commit/count cadence belongs to memcheck. Racecheck exercises the
# record's shared-memory chain/tree paths once, keeping instrumentation bounded.
set_property(TEST ninfer_gdn_history_test_sanitizer PROPERTY LABELS sanitizer)
add_test(NAME ninfer_gdn_history_record_racecheck
  COMMAND ${NINFER_TEST_LAUNCHER} $<TARGET_FILE:ninfer_gdn_history_test> --racecheck)
set_property(TEST ninfer_gdn_history_record_racecheck PROPERTY LABELS sanitizer racecheck)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_ragged_prefix_test
  SOURCES ops/test_prepare_ragged_prefix.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_mtp_fc_test ops/test_mtp_fc.cpp)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a8_test ops/linear/test_nvfp4_a8.cpp)
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()

# Sanitizer set: the tests the compute-sanitizer gates run. It covers the Ops the qwen3.8-27b
# NVFP4 DFlash2 flow dispatches, text-only with NVFP4 KV. A test whose full sweep is too slow or
# too large under compute-sanitizer contributes its `--sanitizer` cases instead; the rest run
# unchanged. An Op added to that flow adds its test here.
#
# `sanitizer` (memcheck and initcheck) is the whole set. `racecheck` is the part that is also
# quick under racecheck, which tracks every shared-memory access; the GEMM-heavy linear and
# projection sweeps take 5 to 50 minutes each under it and stay outside that label.
ninfer_add_sanitizer_cases(ninfer_sampling_test)
ninfer_add_sanitizer_cases(ninfer_token_logprobs_test)
ninfer_add_sanitizer_cases(ninfer_speculative_round_test)
ninfer_add_sanitizer_cases(ninfer_gqa_attention_test)
ninfer_add_sanitizer_cases(ninfer_attn_input_proj_test)
ninfer_add_sanitizer_cases(ninfer_a8_activation_test)
ninfer_add_sanitizer_cases(ninfer_normalized_rope_kv_append_test)
ninfer_add_sanitizer_cases(ninfer_linear_kv_projection_test)
set_property(TEST ninfer_linear_kv_projection_test_sanitizer PROPERTY LABELS sanitizer)
set(ninfer_sanitizer_racecheck_tests
  ninfer_kv_cache_test
  ninfer_embedding_test
  ninfer_rmsnorm_test
  ninfer_dual_rmsnorm_test
  ninfer_gated_rmsnorm_test
  ninfer_rope_test
  ninfer_position_test
  ninfer_scalar_test
  ninfer_scatter_test
  ninfer_argmax_test
  ninfer_sigmoid_mul_test
  ninfer_silu_mul_test
  ninfer_residual_add_test
  ninfer_causal_conv1d_silu_test
  ninfer_gated_delta_net_test
  ninfer_gated_delta_net_replay_record_test
  ninfer_gdn_replay_fold_test
  ninfer_gdn_gating_proj_test
  ninfer_kv_cache_append_prefix_test
  ninfer_prepare_masked_block_test
  ninfer_prepare_ragged_prefix_test
  ninfer_grouped_dynamic_conv_test
  ninfer_swa_test
  ninfer_dflash2_path_select_test)
set(ninfer_sanitizer_memcheck_only_tests
  ninfer_gdn_input_proj_test
  ninfer_gdn_input_proj_conv_record_test
  ninfer_linear_nvfp4_a16_test
  ninfer_linear_nvfp4_a4_test
  ninfer_linear_nvfp4_a8_test
  ninfer_linear_bf16_a16_test
  ninfer_linear_w8_a16_test
  ninfer_linear_q4_a16_test
  ninfer_linear_add_nvfp4_test
  ninfer_linear_add_bf16_a16_test
  ninfer_linear_swiglu_nvfp4_test)
set_property(TEST ${ninfer_sanitizer_racecheck_tests}
  APPEND PROPERTY LABELS sanitizer racecheck)
set_property(TEST ${ninfer_sanitizer_memcheck_only_tests}
  APPEND PROPERTY LABELS sanitizer)

# Tests that run for minutes. `./scripts/run-unit-tests.sh --fast` skips them. The attention
# sanitizer cases stay slow because their host oracle dominates.
set_property(TEST
    ninfer_gqa_attention_test
    ninfer_gqa_attention_test_sanitizer
    ninfer_speculative_round_test
    ninfer_attn_input_proj_test
  APPEND PROPERTY LABELS slow)

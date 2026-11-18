# Qwen3.8-Flash-Next target (docs/maintainer/qwen3.8-flash-next-plan.md). The runtime library holds
# the target's own Program and is linked by ninfer_engine and by the parity driver; the package and
# its Engine adapter compile into ninfer_engine like every registered target.
add_library(ninfer_qwen3_8_flash_next STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ngram_table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/expert_residency.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/binder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/forward.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/dflash.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/kv_extents.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/ple_feeder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/program.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/state.cpp)
target_include_directories(ninfer_qwen3_8_flash_next PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/impl)
# LoadedModel retains the frontend resources in the shared Qwen3.6 frontend's resource type.
target_include_directories(ninfer_qwen3_8_flash_next PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
find_package(Threads REQUIRED)
target_link_libraries(ninfer_qwen3_8_flash_next
  PUBLIC ninfer_artifact ninfer_core ninfer_ops Threads::Threads)
ninfer_internal_includes(ninfer_qwen3_8_flash_next)

if(NINFER_ENABLE_HIP_EXPERT_HELPER)
  add_library(ninfer_flash_next_helper_hip STATIC impl/runtime/expert_helper.hip)
  ninfer_hip_target(ninfer_flash_next_helper_hip)
  target_include_directories(ninfer_flash_next_helper_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/impl)
  target_link_libraries(ninfer_flash_next_helper_hip PRIVATE ninfer_hip_core ninfer_hip_ops)
  target_sources(ninfer_qwen3_8_flash_next PRIVATE
    impl/runtime/expert_helper.cpp
    impl/runtime/expert_helper_placement.cpp
    impl/runtime/expert_helper_upload.cpp)
  # Every includer of Program/Forward must see the same optional member layout.
  target_compile_definitions(ninfer_qwen3_8_flash_next PUBLIC NINFER_ENABLE_HIP_EXPERT_HELPER)
  target_link_libraries(ninfer_qwen3_8_flash_next PRIVATE ninfer_flash_next_helper_hip)
endif()

target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/engine/engine_program.cpp)
target_include_directories(ninfer_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_engine PRIVATE ninfer_qwen3_8_flash_next)

if(BUILD_TESTING)
  # Teacher-forced parity and chunking self-consistency driver; tools/parity/qwen3_8_flash_next.
  add_executable(ninfer_qwen3_8_flash_next_parity
    ${PROJECT_SOURCE_DIR}/tools/parity/qwen3_8_flash_next/parity.cpp
    ${PROJECT_SOURCE_DIR}/tools/parity/qwen3_8_flash_next/npy.cpp)
  target_link_libraries(ninfer_qwen3_8_flash_next_parity PRIVATE ninfer_qwen3_8_flash_next ninfer_engine)
  target_include_directories(ninfer_qwen3_8_flash_next_parity PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
  ninfer_internal_includes(ninfer_qwen3_8_flash_next_parity)
endif()

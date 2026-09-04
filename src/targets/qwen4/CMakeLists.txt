# Unregistered architecture-verification package. It is intentionally not linked by ninfer_engine.
add_library(ninfer_qwen4_verifier STATIC verifier.cpp program.cpp)
ninfer_internal_includes(ninfer_qwen4_verifier)
target_link_libraries(ninfer_qwen4_verifier PUBLIC ninfer_artifact ninfer_ops)

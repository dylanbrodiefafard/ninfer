#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The launchers trust the wrapper's validation: (output_rows, input_rows) is an admitted problem
// whose output dtype matches y (FP32 or BF16 bits), tokens >= 1, and every pointer is a 16-byte
// aligned device address.
void bf16_projection_launch(const std::uint16_t* x, const std::uint16_t* weight, float* y,
                            std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens,
                            cudaStream_t stream);
void bf16_projection_launch(const std::uint16_t* x, const std::uint16_t* weight, std::uint16_t* y,
                            std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail

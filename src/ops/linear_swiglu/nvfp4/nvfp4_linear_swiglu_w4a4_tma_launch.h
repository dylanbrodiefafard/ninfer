#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_linear_swiglu_w4a4_tma_packed(
    const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
    const std::uint8_t* weight_codes, const std::uint8_t* weight_scales, __nv_bfloat16* diagnostic,
    std::uint8_t* codes, std::uint8_t* scales, std::int32_t tokens, float alpha,
    float output_divisor, cudaStream_t stream);

} // namespace ninfer::ops::detail

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

// The same projection publishing its [17408, tokens] output as the NVFP4 activation
// (ninfer/ops/a4_activation.h) of the down projection with output_input_scale_divisor, instead of
// BF16.
void launch_nvfp4_linear_swiglu_w4a4_tma_a4(const std::uint8_t* activation_codes,
                                            const std::uint8_t* activation_scales,
                                            const std::uint8_t* weight_codes,
                                            const std::uint8_t* weight_scales,
                                            std::uint8_t* output_codes, std::uint8_t* output_scales,
                                            float output_input_scale_divisor, std::int32_t tokens,
                                            float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail

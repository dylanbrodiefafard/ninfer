// Implements: include/ninfer/ops/grouped_dynamic_conv.h
#include "ops/launcher/grouped_dynamic_conv.h"

#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ops/common/math.h"
#include "ops/kernel/grouped_dynamic_conv.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_dynamic_conv_bf16_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream) {
    const std::int32_t n_rows = out.ne[0];
    const std::int32_t k_rows = x.ne[0];
    const std::int32_t cols   = x.ne[1];
    const dim3 grid(static_cast<unsigned int>(n_rows), static_cast<unsigned int>(cols));
    grouped_dynamic_conv_bf16_gemv_kernel<<<grid, kGroupedDynamicConvGemmBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata != nullptr ? weight.qdata : weight.payload),
        static_cast<__nv_bfloat16*>(out.data), n_rows, k_rows);
    CUDA_CHECK(cudaGetLastError());
}

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor& prepared,
                                         Tensor& finish_dynamic, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t batch  = hidden.ne[2];
    const std::int32_t cols   = tokens * batch;
    const int channel_blocks  = div_up(kGroupedDynamicConvHidden, kGroupedDynamicConvConvBlock);
    const dim3 grid(static_cast<unsigned int>(channel_blocks), static_cast<unsigned int>(cols));
    grouped_dynamic_conv_kernel<true>
        <<<grid, kGroupedDynamicConvConvBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(base_kernel.data),
            static_cast<const __nv_bfloat16*>(projection.data),
            static_cast<__nv_bfloat16*>(prepared.data),
            static_cast<__nv_bfloat16*>(finish_dynamic.data), tokens, batch);
    CUDA_CHECK(cudaGetLastError());
}

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& out,
                                        cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t batch  = hidden.ne[2];
    const std::int32_t cols   = tokens * batch;
    const int channel_blocks  = div_up(kGroupedDynamicConvHidden, kGroupedDynamicConvConvBlock);
    const dim3 grid(static_cast<unsigned int>(channel_blocks), static_cast<unsigned int>(cols));
    grouped_dynamic_conv_kernel<false>
        <<<grid, kGroupedDynamicConvConvBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(base_kernel.data),
            static_cast<const __nv_bfloat16*>(finish_dynamic.data),
            static_cast<__nv_bfloat16*>(out.data), nullptr, tokens, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

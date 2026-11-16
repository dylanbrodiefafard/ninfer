// Implements: include/ninfer/ops/grouped_dynamic_conv.h
#include "ops/launcher/grouped_dynamic_conv.h"

#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ops/common/math.h"
#include "ops/kernel/grouped_dynamic_conv.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

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

namespace {

// The wrapper admits D in {5120,2560}; each width compiles its own channel geometry.
template <bool kPrepare>
void launch_conv(const Tensor& hidden, const Tensor& base_kernel, const Tensor& dynamic_src,
                 Tensor& out, __nv_bfloat16* finish_dynamic, cudaStream_t stream) {
    const std::int32_t width  = hidden.ne[0];
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t batch  = hidden.ne[2];
    const std::int32_t cols   = tokens * batch;
    const int channel_blocks  = div_up(width, kGroupedDynamicConvConvBlock);
    const dim3 grid(static_cast<unsigned int>(channel_blocks), static_cast<unsigned int>(cols));
    const auto run = [&]<int Hidden>() {
        grouped_dynamic_conv_kernel<Hidden, kPrepare>
            <<<grid, kGroupedDynamicConvConvBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(hidden.data),
                static_cast<const __nv_bfloat16*>(base_kernel.data),
                static_cast<const __nv_bfloat16*>(dynamic_src.data),
                static_cast<__nv_bfloat16*>(out.data), finish_dynamic, tokens, batch);
    };
    switch (width) {
    case 5120:
        run.template operator()<5120>();
        break;
    case 2560:
        run.template operator()<2560>();
        break;
    default:
        throw std::invalid_argument("grouped_dynamic_conv: unregistered hidden width");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor& prepared,
                                         Tensor& finish_dynamic, cudaStream_t stream) {
    launch_conv<true>(hidden, base_kernel, projection, prepared,
                      static_cast<__nv_bfloat16*>(finish_dynamic.data), stream);
}

void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& out,
                                        cudaStream_t stream) {
    launch_conv<false>(hidden, base_kernel, finish_dynamic, out, nullptr, stream);
}

namespace {
template <int Hidden, int Block>
void launch_finish_residual_rmsnorm(const Tensor& hidden, const Tensor& base_kernel,
                                    const Tensor& finish_dynamic, Tensor& residual,
                                    const Tensor& weight, float epsilon, Tensor& out,
                                    cudaStream_t stream) {
    const auto columns =
        static_cast<unsigned int>(hidden.ne[1]) * static_cast<unsigned int>(hidden.ne[2]);
    grouped_dynamic_conv_finish_residual_rmsnorm_kernel<Hidden, Block>
        <<<columns, Block, 0, stream>>>(static_cast<const __nv_bfloat162*>(hidden.data),
                                        static_cast<const __nv_bfloat162*>(base_kernel.data),
                                        static_cast<const __nv_bfloat16*>(finish_dynamic.data),
                                        static_cast<__nv_bfloat162*>(residual.data),
                                        static_cast<const __nv_bfloat162*>(weight.data),
                                        static_cast<__nv_bfloat162*>(out.data), hidden.ne[1],
                                        epsilon);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void grouped_dynamic_conv_finish_residual_rmsnorm_launch(
    const Tensor& hidden, const Tensor& base_kernel, const Tensor& finish_dynamic, Tensor& residual,
    const Tensor& weight, float epsilon, Tensor& out, cudaStream_t stream) {
    // Both widths keep five BF16 pairs per thread.
    switch (hidden.ne[0]) {
    case 5120:
        launch_finish_residual_rmsnorm<5120, 512>(hidden, base_kernel, finish_dynamic, residual,
                                                  weight, epsilon, out, stream);
        return;
    case 2560:
        launch_finish_residual_rmsnorm<2560, 256>(hidden, base_kernel, finish_dynamic, residual,
                                                  weight, epsilon, out, stream);
        return;
    default:
        throw std::invalid_argument(
            "grouped_dynamic_conv_finish_residual_rmsnorm: unregistered hidden width");
    }
}

} // namespace ninfer::ops::detail

#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using M256N128S3 = Nvfp4W4a4TmaSchedule<256, 3, 1>;

template <class Geometry, class Schedule>
Nvfp4W4a4TmaDescriptors make_descriptors(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, std::int32_t tokens) {
    constexpr std::uint32_t kCodeColumns = 64;
    constexpr std::uint32_t kPairN       = Schedule::kBlockN / 2;
    constexpr std::uint64_t kWeightScaleBytes =
        static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;

    Nvfp4W4a4TmaDescriptors descriptors{};
    make_nvfp4_activation_descriptors<Geometry, Schedule::kBlockM>(descriptors, activation_codes,
                                                                   activation_scales, tokens);
    descriptors.b_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow, kCodeColumns,
        kPairN, CU_TENSOR_MAP_SWIZZLE_64B, "encode LinearSwiGLU weight codes TMA");
    descriptors.b_scales =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          16, kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                          "encode LinearSwiGLU weight scales TMA");
    return descriptors;
}

template <class Store>
void launch(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
            const std::uint8_t* weight_codes, const std::uint8_t* weight_scales, Store output,
            std::int32_t tokens, float alpha, cudaStream_t stream) {
    if (tokens <= 0) {
        throw std::invalid_argument("nvfp4 LinearSwiGLU TMA requires a positive token count");
    }

    using Geometry        = Nvfp4MlpGateUpGeometry;
    constexpr auto kernel = nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3, Store>;
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4LinearSwiGluTmaSharedStorage<M256N128S3>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const Nvfp4W4a4TmaDescriptors descriptors = make_descriptors<Geometry, M256N128S3>(
        activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr int kPairN = M256N128S3::kBlockN / 2;
    const dim3 grid((Geometry::kOutputRows / 2) / kPairN,
                    (tokens + M256N128S3::kBlockM - 1) / M256N128S3::kBlockM);
    kernel<<<grid, M256N128S3::kThreads, kSharedBytes, stream>>>(descriptors, alpha, output,
                                                                 tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch(activation_codes, activation_scales, weight_codes, weight_scales,
           Nvfp4SwiGluBf16Store{output}, tokens, alpha, stream);
}

void launch_nvfp4_linear_swiglu_w4a4_tma_a4(const std::uint8_t* activation_codes,
                                            const std::uint8_t* activation_scales,
                                            const std::uint8_t* weight_codes,
                                            const std::uint8_t* weight_scales,
                                            std::uint8_t* output_codes, std::uint8_t* output_scales,
                                            float output_input_scale_divisor, std::int32_t tokens,
                                            float alpha, cudaStream_t stream) {
    launch(activation_codes, activation_scales, weight_codes, weight_scales,
           Nvfp4SwiGluA4Store{output_codes, output_scales, output_input_scale_divisor}, tokens,
           alpha, stream);
}

} // namespace ninfer::ops::detail

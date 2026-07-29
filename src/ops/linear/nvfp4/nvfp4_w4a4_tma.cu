#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry     = Nvfp4LinearDecodeGeometry;
using TmaM256N128  = Nvfp4W4a4TmaSchedule<256, 3, 1>;
using LinearOutput = Nvfp4W4a4ContiguousOutput;

constexpr std::int32_t kQueryRows  = 6144;
constexpr std::int32_t kKeyRows    = 1024;
constexpr std::int32_t kGateRows   = 6144;
constexpr std::int32_t kKeyBegin   = kQueryRows;
constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

struct AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((kQueryRows % TmaM256N128::kBlockN) == 0);
static_assert((kKeyRows % TmaM256N128::kBlockN) == 0);
static_assert((kGateRows % TmaM256N128::kBlockN) == 0);

template <class Output>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                std::int32_t tokens, float alpha, Output output, cudaStream_t stream) {
    const Nvfp4W4a4TmaDescriptors descriptors =
        make_nvfp4_w4a4_tma_descriptors<Geometry, TmaM256N128::kBlockM>(
            activation_codes, activation_scales, weight_codes, weight_scales, tokens);
    constexpr std::size_t kSharedBytes = sizeof(Nvfp4W4a4TmaSharedStorage<TmaM256N128>);
    static const bool kConfigured      = [] {
        CUDA_CHECK(cudaFuncSetAttribute(nvfp4_w4a4_tma_kernel<Geometry, TmaM256N128, Output>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSharedBytes)));
        return true;
    }();
    (void)kConfigured;

    const dim3 grid(Geometry::kOutputRows / TmaM256N128::kBlockN, tokens / TmaM256N128::kBlockM);
    nvfp4_w4a4_tma_kernel<Geometry, TmaM256N128>
        <<<grid, TmaM256N128::kThreads, kSharedBytes, stream>>>(descriptors, alpha, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_w4a4_tma_linear(const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma(activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
               LinearOutput{output, Geometry::kOutputRows}, stream);
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream) {
    launch_tma(activation_codes, activation_scales, weight_codes, weight_scales, tokens, alpha,
               AttentionOutput{query, key, gate, value}, stream);
}

} // namespace ninfer::ops::detail

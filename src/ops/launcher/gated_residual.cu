#include "ops/launcher/gated_residual.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 2560;
constexpr int kBranches = 4;
constexpr int kFlat = kHidden * kBranches;
constexpr int kRank = 320;
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / kWarpSize;

__global__ __launch_bounds__(kBlock)
void normalize_kernel(const __nv_bfloat16* residual, const float* weight,
                      __nv_bfloat16* normalized) {
    const int branch = static_cast<int>(blockIdx.x);
    const int base = branch * kHidden;
    float local_sum = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += kBlock) {
        const float value = __bfloat162float(residual[base + d]);
        local_sum += value * value;
    }
    __shared__ float warp_sums[kWarps];
    __shared__ float inverse_rms;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum * (1.0F / static_cast<float>(kHidden)) + 1.0e-6F);
    }
    __syncthreads();
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += kBlock) {
        const float value = __bfloat162float(residual[base + d]);
        normalized[base + d] = __float2bfloat16_rn(value * inverse_rms * weight[base + d]);
    }
}

__global__ void activate_kernel(__nv_bfloat16* low_rank) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < kRank) {
        low_rank[row] = __float2bfloat16_rn(silu(__bfloat162float(low_rank[row]) * 0.25F));
    }
}

__global__ void mix_kernel(const __nv_bfloat16* normalized, const __nv_bfloat16* up_logits,
                           __nv_bfloat16* output) {
    const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= kHidden) { return; }
    float mixed = 0.0F;
#pragma unroll
    for (int branch = 0; branch < kBranches; ++branch) {
        const int index = branch * kHidden + d;
        mixed += sigmoid(__bfloat162float(up_logits[index])) *
                 __bfloat162float(normalized[index]);
    }
    output[d] = __float2bfloat16_rn(mixed * 0.25F);
}

__global__ __launch_bounds__(kBlock)
void write_kernel(const __nv_bfloat16* normalized, const float* weight,
                  __nv_bfloat16* write_scale) {
    const int branch = static_cast<int>(blockIdx.x);
    const auto* row = weight + static_cast<long long>(branch) * kFlat;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kFlat; k += kBlock) {
        sum = fmaf(__bfloat162float(normalized[k]), row[k], sum);
    }
    __shared__ float warp_sums[kWarps];
    sum = block_reduce_sum<kBlock>(sum, warp_sums);
    if (threadIdx.x == 0) {
        write_scale[branch] = __float2bfloat16_rn(2.0F * sigmoid(sum * 0.25F));
    }
}

__global__ void inject_kernel(const __nv_bfloat16* residual,
                              const __nv_bfloat16* block_output,
                              const __nv_bfloat16* write_scale,
                              __nv_bfloat16* output) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kFlat) { return; }
    const int branch = index / kHidden;
    const int d = index - branch * kHidden;
    const float value = __bfloat162float(residual[index]) +
                        __bfloat162float(write_scale[branch]) *
                            __bfloat162float(block_output[d]);
    output[index] = __float2bfloat16_rn(value);
}

} // namespace

void gated_residual_normalize_launch(const Tensor& residual, const Tensor& norm_weight,
                                     Tensor& normalized, cudaStream_t stream) {
    normalize_kernel<<<kBranches, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const float*>(norm_weight.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_activate_launch(Tensor& low_rank, cudaStream_t stream) {
    activate_kernel<<<(kRank + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(low_rank.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_mix_launch(const Tensor& normalized, const Tensor& up_logits, Tensor& x,
                               cudaStream_t stream) {
    mix_kernel<<<(kHidden + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(up_logits.data),
        static_cast<__nv_bfloat16*>(x.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_write_launch(const Tensor& normalized, const Tensor& write_weight,
                                 Tensor& write_scale, cudaStream_t stream) {
    write_kernel<<<kBranches, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const float*>(write_weight.data),
        static_cast<__nv_bfloat16*>(write_scale.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_inject_launch(const Tensor& residual, const Tensor& block_output,
                                  const Tensor& write_scale, Tensor& residual_out,
                                  cudaStream_t stream) {
    inject_kernel<<<(kFlat + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const __nv_bfloat16*>(write_scale.data),
        static_cast<__nv_bfloat16*>(residual_out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

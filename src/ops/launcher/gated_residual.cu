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

__device__ float stored_value(float value) { return value; }
__device__ float stored_value(__nv_bfloat16 value) { return __bfloat162float(value); }
template<class T> __device__ T store_value(float value) { return value; }
template<> __device__ __nv_bfloat16 store_value(float value) { return __float2bfloat16_rn(value); }

template<class Internal>
__global__ __launch_bounds__(kBlock)
void normalize_kernel(const __nv_bfloat16* residual, const float* weight,
                      Internal* normalized) {
    const int branch = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int base = token * kFlat + branch * kHidden;
    const int weight_base = branch * kHidden;
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
        normalized[base + d] =
            store_value<Internal>(value * inverse_rms * weight[weight_base + d]);
    }
}

template<class Internal>
__global__ void activate_kernel(Internal* low_rank, int count) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < count) {
        low_rank[row] = store_value<Internal>(silu(stored_value(low_rank[row]) * 0.25F));
    }
}

template<class Internal>
__global__ void mix_kernel(const Internal* normalized, const Internal* up_logits,
                           __nv_bfloat16* output) {
    const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int token = static_cast<int>(blockIdx.y);
    if (d >= kHidden) { return; }
    const int token_base = token * kFlat;
    float mixed = 0.0F;
#pragma unroll
    for (int branch = 0; branch < kBranches; ++branch) {
        const int index = token_base + branch * kHidden + d;
        mixed += sigmoid(stored_value(up_logits[index])) * stored_value(normalized[index]);
    }
    output[token * kHidden + d] = __float2bfloat16_rn(mixed * 0.25F);
}

template<class Internal>
__global__ __launch_bounds__(kBlock)
void write_kernel(const Internal* normalized, const float* weight,
                  __nv_bfloat16* write_scale) {
    const int branch = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const auto* row = weight + static_cast<long long>(branch) * kFlat;
    float sum = 0.0F;
    for (int k = static_cast<int>(threadIdx.x); k < kFlat; k += kBlock) {
        sum = fmaf(stored_value(normalized[token * kFlat + k]), row[k], sum);
    }
    __shared__ float warp_sums[kWarps];
    sum = block_reduce_sum<kBlock>(sum, warp_sums);
    if (threadIdx.x == 0) {
        write_scale[token * kBranches + branch] =
            __float2bfloat16_rn(2.0F * sigmoid(sum * 0.25F));
    }
}

__global__ void inject_kernel(const __nv_bfloat16* residual,
                              const __nv_bfloat16* block_output,
                              const __nv_bfloat16* write_scale,
                              __nv_bfloat16* output) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int token = static_cast<int>(blockIdx.y);
    if (index >= kFlat) { return; }
    const int branch = index / kHidden;
    const int d = index - branch * kHidden;
    const int token_base = token * kFlat;
    const float value = __bfloat162float(residual[token_base + index]) +
                        __bfloat162float(write_scale[token * kBranches + branch]) *
                            __bfloat162float(block_output[token * kHidden + d]);
    output[token_base + index] = __float2bfloat16_rn(value);
}

template<int N, int K>
__global__ void bf16_project_f32_kernel(const float* input, const __nv_bfloat16* weight, float* output) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int token = blockIdx.y;
    const int lane = threadIdx.x % 32;
    if (row >= N) { return; }
    float sum = 0;
    for (int column = lane; column < K; column += 32) {
        sum = fmaf(input[token*K+column], __bfloat162float(weight[row*K+column]), sum);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) { output[token*N+row] = sum; }
}

} // namespace

void gated_residual_bf16_project_f32_launch(const Tensor& x, const Weight& weight,
                                          Tensor& out, cudaStream_t stream) {
    if (weight.n == kRank) {
        bf16_project_f32_kernel<kRank,kFlat><<<dim3(kRank/4,x.ne[1]),128,0,stream>>>(
            static_cast<const float*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<float*>(out.data));
    } else {
        bf16_project_f32_kernel<kFlat,kRank><<<dim3(kFlat/4,x.ne[1]),128,0,stream>>>(
            static_cast<const float*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<float*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_normalize_launch(const Tensor& residual, const Tensor& norm_weight,
                                     Tensor& normalized, cudaStream_t stream) {
    if (normalized.dtype == DType::FP32) {
        normalize_kernel<<<dim3(kBranches,residual.ne[2]),kBlock,0,stream>>>(
            static_cast<const __nv_bfloat16*>(residual.data),static_cast<const float*>(norm_weight.data),
            static_cast<float*>(normalized.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    normalize_kernel<<<dim3(kBranches, residual.ne[2]), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const float*>(norm_weight.data),
        static_cast<__nv_bfloat16*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_activate_launch(Tensor& low_rank, cudaStream_t stream) {
    const int count = kRank * low_rank.ne[1];
    if (low_rank.dtype == DType::FP32) {
        activate_kernel<<<(count+kBlock-1)/kBlock,kBlock,0,stream>>>(static_cast<float*>(low_rank.data),count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    activate_kernel<<<(count + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(low_rank.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_mix_launch(const Tensor& normalized, const Tensor& up_logits, Tensor& x,
                               cudaStream_t stream) {
    if (normalized.dtype == DType::FP32) {
        mix_kernel<<<dim3((kHidden+kBlock-1)/kBlock,x.ne[1]),kBlock,0,stream>>>(
            static_cast<const float*>(normalized.data),static_cast<const float*>(up_logits.data),
            static_cast<__nv_bfloat16*>(x.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    mix_kernel<<<dim3((kHidden + kBlock - 1) / kBlock, x.ne[1]), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(up_logits.data),
        static_cast<__nv_bfloat16*>(x.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_write_launch(const Tensor& normalized, const Tensor& write_weight,
                                 Tensor& write_scale, cudaStream_t stream) {
    if (normalized.dtype == DType::FP32) {
        write_kernel<<<dim3(kBranches,write_scale.ne[1]),kBlock,0,stream>>>(
            static_cast<const float*>(normalized.data),static_cast<const float*>(write_weight.data),
            static_cast<__nv_bfloat16*>(write_scale.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    write_kernel<<<dim3(kBranches, write_scale.ne[1]), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const float*>(write_weight.data),
        static_cast<__nv_bfloat16*>(write_scale.data));
    CUDA_CHECK(cudaGetLastError());
}

void gated_residual_inject_launch(const Tensor& residual, const Tensor& block_output,
                                  const Tensor& write_scale, Tensor& residual_out,
                                  cudaStream_t stream) {
    inject_kernel<<<dim3((kFlat + kBlock - 1) / kBlock, residual.ne[2]), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const __nv_bfloat16*>(write_scale.data),
        static_cast<__nv_bfloat16*>(residual_out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

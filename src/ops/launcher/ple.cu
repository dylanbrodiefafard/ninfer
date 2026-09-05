#include "ops/launcher/ple.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr float kEpsilon = 1.0e-6F;

__device__ __forceinline__ std::uint16_t load_u16_le(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

__device__ __forceinline__ float iq4_nl_value(const std::uint8_t* block, int index) {
    constexpr int codebook[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code            = index < 16 ? packed & 0x0fU : packed >> 4U;
    return __half2float(__ushort_as_half(load_u16_le(block))) * static_cast<float>(codebook[code]);
}

__global__ void ple_iq4_nl_decode_kernel(const std::uint8_t* encoded, __nv_bfloat16* output,
                                         int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    const int row_index  = index / kPleRowWidth;
    const int d          = index % kPleRowWidth;
    const auto* row      = encoded + row_index * kPleIq4NlRowBytes;
    const auto* block    = row + (d / kPleIq4NlBlockValues) * kPleIq4NlBlockBytes;
    output[index]        = __float2bfloat16_rn(iq4_nl_value(block, d % kPleIq4NlBlockValues));
}

__global__ void ple_gate_kernel(const __nv_bfloat16* residual,
                                const __nv_bfloat16* projected_key,
                                const __nv_bfloat16* projected_value, const float* key_weight,
                                const float* query_weight, float* gated, int width) {
    __shared__ float q_reduce[256];
    __shared__ float k_reduce[256];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float gate;
    const int lane   = threadIdx.x;
    const int branch = blockIdx.x;
    const int token  = blockIdx.y;
    const int channel_base = kPleEmbeddingWidth * branch;
    const std::int64_t token_base = static_cast<std::int64_t>(kPleChannels) * token;
    float q_sum2 = 0.0F;
    float k_sum2 = 0.0F;
    for (int d = lane; d < kPleEmbeddingWidth; d += blockDim.x) {
        const int channel = channel_base + d;
        const float q = __bfloat162float(residual[channel + token_base]);
        const float k = __bfloat162float(projected_key[channel + token_base]);
        q_sum2 += q * q;
        k_sum2 += k * k;
    }
    q_reduce[lane] = q_sum2;
    k_reduce[lane] = k_sum2;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            q_reduce[lane] += q_reduce[lane + stride];
            k_reduce[lane] += k_reduce[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        inv_q = rsqrtf(q_reduce[0] / kPleEmbeddingWidth + kEpsilon);
        inv_k = rsqrtf(k_reduce[0] / kPleEmbeddingWidth + kEpsilon);
    }
    __syncthreads();
    float dot = 0.0F;
    for (int d = lane; d < kPleEmbeddingWidth; d += blockDim.x) {
        const int channel = channel_base + d;
        const float q = __bfloat162float(residual[channel + token_base]) * inv_q *
                        query_weight[channel];
        const float k = __bfloat162float(projected_key[channel + token_base]) * inv_k *
                        key_weight[channel];
        dot += q * k;
    }
    q_reduce[lane] = dot;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) { q_reduce[lane] += q_reduce[lane + stride]; }
        __syncthreads();
    }
    if (lane == 0) {
        const float raw = q_reduce[0] * 0.01976423537605237F; // 1/sqrt(2560)
        const float transformed = raw == 0.0F
                                      ? 0.0F
                                      : copysignf(sqrtf(fmaxf(fabsf(raw), 1.0e-6F)), raw);
        gate = 1.0F / (1.0F + expf(-transformed));
    }
    __syncthreads();
    for (int d = lane; d < kPleEmbeddingWidth; d += blockDim.x) {
        const int channel = channel_base + d;
        gated[channel + token_base] =
            gate * __bfloat162float(projected_value[d + static_cast<std::int64_t>(kPleEmbeddingWidth) * token]);
    }
}

__global__ void ple_conv_input_kernel(const float* gated, const float* norm_weight,
                                      __nv_bfloat16* current, int width) {
    __shared__ float reduce[256];
    __shared__ float inv_rms;
    const int lane   = threadIdx.x;
    const int branch = blockIdx.x;
    const int token  = blockIdx.y;
    const int channel_base = kPleEmbeddingWidth * branch;
    const std::int64_t token_base = static_cast<std::int64_t>(kPleChannels) * token;
    float sum2 = 0.0F;
    for (int d = lane; d < kPleEmbeddingWidth; d += blockDim.x) {
        const float value = gated[channel_base + d + token_base];
        sum2 += value * value;
    }
    reduce[lane] = sum2;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) { reduce[lane] += reduce[lane + stride]; }
        __syncthreads();
    }
    if (lane == 0) { inv_rms = rsqrtf(reduce[0] / kPleEmbeddingWidth + kEpsilon); }
    __syncthreads();
    for (int d = lane; d < kPleEmbeddingWidth; d += blockDim.x) {
        const int channel = channel_base + d;
        const float normalized = gated[channel + token_base] * inv_rms * norm_weight[channel];
        current[channel + token_base] = __float2bfloat16_rn(normalized);
    }
}

__device__ __forceinline__ float state_value(const __nv_bfloat16* old_state,
                                              const __nv_bfloat16* current, int channel,
                                              int logical_index) {
    return logical_index < 0
               ? __bfloat162float(old_state[channel + static_cast<std::int64_t>(kPleChannels) *
                                                        (logical_index + kPleConvHistory)])
               : __bfloat162float(current[channel + static_cast<std::int64_t>(kPleChannels) *
                                                     logical_index]);
}

__global__ void ple_conv_inject_kernel(const __nv_bfloat16* residual, const float* gated,
                                       const float* conv_weight,
                                       const __nv_bfloat16* old_state,
                                       const __nv_bfloat16* current, __nv_bfloat16* output,
                                       int width) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kPleChannels) { return; }
    for (int token = 0; token < width; ++token) {
        float conv = 0.0F;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            conv += conv_weight[static_cast<std::int64_t>(channel) * 4 + tap] *
                    state_value(old_state, current, channel, token - 9 + 3 * tap);
        }
        const float activated = conv / (1.0F + expf(-conv));
        const std::int64_t index = channel + static_cast<std::int64_t>(kPleChannels) * token;
        const float value = __bfloat162float(residual[index]) + gated[index] + activated;
        output[index] = __float2bfloat16_rn(value);
    }
}

__global__ void ple_state_update_kernel(const __nv_bfloat16* old_state,
                                        const __nv_bfloat16* current,
                                        __nv_bfloat16* new_state, int width) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kPleChannels) { return; }
    __nv_bfloat16 old[kPleConvHistory];
#pragma unroll
    for (int i = 0; i < kPleConvHistory; ++i) {
        old[i] = old_state[channel + static_cast<std::int64_t>(kPleChannels) * i];
    }
#pragma unroll
    for (int i = 0; i < kPleConvHistory; ++i) {
        const int sequence_index = width + i - kPleConvHistory;
        new_state[channel + static_cast<std::int64_t>(kPleChannels) * i] =
            sequence_index < 0
                ? old[sequence_index + kPleConvHistory]
                : current[channel + static_cast<std::int64_t>(kPleChannels) * sequence_index];
    }
}

} // namespace

void ple_iq4_nl_decode_rows_launch(const Tensor& device_rows, Tensor& embedding,
                                   cudaStream_t stream) {
    constexpr int block = 256;
    const int count = kPleHeads * kPleRowWidth * embedding.ne[2];
    ple_iq4_nl_decode_kernel<<<(count + block - 1) / block, block, 0, stream>>>(
        static_cast<const std::uint8_t*>(device_rows.data),
        static_cast<__nv_bfloat16*>(embedding.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void ple_gate_launch(const Tensor& residual, const Tensor& projected_key,
                     const Tensor& projected_value, const Tensor& key_norm_weight,
                     const Tensor& query_norm_weight, Tensor& gated, cudaStream_t stream) {
    ple_gate_kernel<<<dim3(kPleBranches, residual.ne[2]), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(projected_key.data),
        static_cast<const __nv_bfloat16*>(projected_value.data),
        static_cast<const float*>(key_norm_weight.data),
        static_cast<const float*>(query_norm_weight.data), static_cast<float*>(gated.data),
        residual.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_input_launch(const Tensor& gated, const Tensor& conv_norm_weight,
                           Tensor& current_state, cudaStream_t stream) {
    ple_conv_input_kernel<<<dim3(kPleBranches, gated.ne[1]), 256, 0, stream>>>(
        static_cast<const float*>(gated.data), static_cast<const float*>(conv_norm_weight.data),
        static_cast<__nv_bfloat16*>(current_state.data), gated.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_inject_launch(const Tensor& residual, const Tensor& gated,
                            const Tensor& conv_weight, const Tensor& old_state,
                            const Tensor& current_state, Tensor& out, cudaStream_t stream) {
    constexpr int block = 256;
    ple_conv_inject_kernel<<<(kPleChannels + block - 1) / block, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<const float*>(gated.data),
        static_cast<const float*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(old_state.data),
        static_cast<const __nv_bfloat16*>(current_state.data),
        static_cast<__nv_bfloat16*>(out.data), residual.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void ple_state_update_launch(const Tensor& old_state, const Tensor& current_state,
                             Tensor& new_state, cudaStream_t stream) {
    constexpr int block = 256;
    ple_state_update_kernel<<<(kPleChannels + block - 1) / block, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(old_state.data),
        static_cast<const __nv_bfloat16*>(current_state.data),
        static_cast<__nv_bfloat16*>(new_state.data), current_state.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

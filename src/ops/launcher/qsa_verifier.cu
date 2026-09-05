#include "ops/launcher/qsa_verifier.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cmath>

namespace ninfer::ops::detail {
namespace {

__global__ void bf16_project_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                    __nv_bfloat16* out, int rows, int columns, int width) {
    __shared__ float sums[256];
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= width) { return; }
    float partial = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        partial += __bfloat162float(weight[static_cast<std::int64_t>(row) * columns + column]) *
                   __bfloat162float(x[column + static_cast<std::int64_t>(columns) * token]);
    }
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { sums[threadIdx.x] += sums[threadIdx.x + stride]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row + static_cast<std::int64_t>(rows) * token] = __float2bfloat16_rn(sums[0]);
    }
}

__device__ __forceinline__ float rotated(float lo, float hi, int pair, bool upper,
                                         const std::int32_t* position) {
    const float inv = powf(1.0e7F, -static_cast<float>(pair) / 32.0F);
    float sine;
    float cosine;
    sincosf(static_cast<float>(position[pair % 3]) * inv, &sine, &cosine);
    return upper ? hi * cosine + lo * sine : lo * cosine - hi * sine;
}

__global__ void core_query_norm_rope_kernel(const __nv_bfloat16* raw,
                                             const std::int32_t* position,
                                             const float* norm_weight, __nv_bfloat16* output) {
    __shared__ float normalized[256];
    __shared__ float reduce[256];
    const int d    = threadIdx.x;
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int base = head * 512 + token * 12288;
    const float value = __bfloat162float(raw[base + d]);
    reduce[d] = value * value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (d < stride) { reduce[d] += reduce[d + stride]; }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduce[0] / 256.0F + 1.0e-6F);
    normalized[d] = value * inv_rms * norm_weight[d];
    __syncthreads();
    float result = normalized[d];
    if (d < 64) {
        const int pair = d & 31;
        result = rotated(normalized[pair], normalized[pair + 32], pair, d >= 32,
                         position + 3LL * token);
    }
    output[d + 256LL * (head + 24LL * token)] = __float2bfloat16_rn(result);
}

__global__ void core_key_norm_rope_kernel(const __nv_bfloat16* raw,
                                           const std::int32_t* position,
                                           const float* norm_weight, __nv_bfloat16* output) {
    __shared__ float normalized[256];
    __shared__ float reduce[256];
    const int d    = threadIdx.x;
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int base = head * 256 + token * 512;
    const float value = __bfloat162float(raw[base + d]);
    reduce[d] = value * value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (d < stride) { reduce[d] += reduce[d + stride]; }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduce[0] / 256.0F + 1.0e-6F);
    normalized[d] = value * inv_rms * norm_weight[d];
    __syncthreads();
    float result = normalized[d];
    if (d < 64) {
        const int pair = d & 31;
        result = rotated(normalized[pair], normalized[pair + 32], pair, d >= 32,
                         position + 3LL * token);
    }
    output[d + 256LL * (head + 2LL * token)] = __float2bfloat16_rn(result);
}

__global__ void output_gate_kernel(const __nv_bfloat16* attention,
                                   const __nv_bfloat16* raw_query_gate,
                                   __nv_bfloat16* gated, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * 24 * 256) { return; }
    const int token = index / (24 * 256);
    const int within = index % (24 * 256);
    const int head = within / 256;
    const int d    = within % 256;
    const float gate =
        __bfloat162float(raw_query_gate[token * 12288 + head * 512 + 256 + d]);
    const float value = __bfloat162float(attention[index]);
    gated[index] = __float2bfloat16_rn(value / (1.0F + expf(-gate)));
}

} // namespace

void qsa_bf16_project_launch(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    const int width = x.ne[1];
    bf16_project_kernel<<<dim3(weight.n, width), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, width);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_core_norm_rope_launch(const Tensor& raw_query_gate, const Tensor& raw_key,
                               const Tensor& position, const Tensor& query_norm,
                               const Tensor& key_norm, Tensor& query, Tensor& key,
                               cudaStream_t stream) {
    const int width = position.ne[1];
    core_query_norm_rope_kernel<<<dim3(24, width), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_query_gate.data),
        static_cast<const std::int32_t*>(position.data),
        static_cast<const float*>(query_norm.data), static_cast<__nv_bfloat16*>(query.data));
    CUDA_CHECK(cudaGetLastError());
    core_key_norm_rope_kernel<<<dim3(2, width), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_key.data),
        static_cast<const std::int32_t*>(position.data),
        static_cast<const float*>(key_norm.data), static_cast<__nv_bfloat16*>(key.data));
    CUDA_CHECK(cudaGetLastError());
}

void qsa_output_gate_launch(const Tensor& attention, const Tensor& raw_query_gate, Tensor& gated,
                            cudaStream_t stream) {
    const int width = attention.ne[2];
    const int elements = width * 24 * 256;
    output_gate_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(attention.data),
        static_cast<const __nv_bfloat16*>(raw_query_gate.data),
        static_cast<__nv_bfloat16*>(gated.data), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

#include "ops/launcher/gated_delta_net_layer.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden     = 2560;
constexpr int kQkHeads    = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadDim    = 128;
constexpr int kQkRows     = kQkHeads * kHeadDim;
constexpr int kValueRows  = kValueHeads * kHeadDim;
constexpr int kQkvRows    = 2 * kQkRows + kValueRows;
constexpr int kBlock      = 256;
constexpr int kWarps      = kBlock / kWarpSize;

__global__ __launch_bounds__(kBlock)
void control_kernel(const __nv_bfloat16* x, const float* a_weight, const float* b_weight,
                    const float* ssm_a, const float* dt_bias, float* g, float* beta, int tokens) {
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int head = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int token = static_cast<int>(blockIdx.y);
    if (head >= kValueHeads || token >= tokens) { return; }
    const auto* wa = a_weight + static_cast<std::int64_t>(head) * kHidden;
    const auto* wb = b_weight + static_cast<std::int64_t>(head) * kHidden;
    float a = 0.0F;
    float b = 0.0F;
    for (int d = lane; d < kHidden; d += kWarpSize) {
        const float value = __bfloat162float(x[static_cast<std::int64_t>(token) * kHidden + d]);
        a                 = fmaf(value, wa[d], a);
        b                 = fmaf(value, wb[d], b);
    }
    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
        const std::int64_t index = static_cast<std::int64_t>(token) * kValueHeads + head;
        g[index]                 = ssm_a[head] * softplus(a + dt_bias[head]);
        beta[index]              = sigmoid(b);
    }
}

__global__ void conv_kernel(const __nv_bfloat16* projected, const float* weight,
                            const __nv_bfloat16* state_in, __nv_bfloat16* state_out,
                            __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, int tokens) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= kQkvRows) { return; }
    float s0 = __bfloat162float(state_in[channel]);
    float s1 = __bfloat162float(state_in[kQkvRows + channel]);
    float s2 = __bfloat162float(state_in[2 * kQkvRows + channel]);
    const auto* channel_weight = weight + static_cast<std::int64_t>(channel) * 4;
    for (int token = 0; token < tokens; ++token) {
        const float p = __bfloat162float(
            projected[static_cast<std::int64_t>(token) * kQkvRows + channel]);
        float sum = fmaf(channel_weight[0], s0, 0.0F);
        sum       = fmaf(channel_weight[1], s1, sum);
        sum       = fmaf(channel_weight[2], s2, sum);
        sum       = fmaf(channel_weight[3], p, sum);
        const __nv_bfloat16 result = __float2bfloat16_rn(silu(sum));
        const std::int64_t token_base = static_cast<std::int64_t>(token) * kValueRows;
        if (channel < kQkRows) {
#pragma unroll
            for (int tile = 0; tile < 3; ++tile) {
                q[token_base + channel + tile * kQkRows] = result;
            }
        } else if (channel < 2 * kQkRows) {
            const int qk_channel = channel - kQkRows;
#pragma unroll
            for (int tile = 0; tile < 3; ++tile) {
                k[token_base + qk_channel + tile * kQkRows] = result;
            }
        } else {
            v[token_base + channel - 2 * kQkRows] = result;
        }
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
    for (int column = 0; column < 3; ++column) {
        const int source = tokens + column - 3;
        state_out[static_cast<std::int64_t>(column) * kQkvRows + channel] =
            source < 0
                ? state_in[static_cast<std::int64_t>(source + 3) * kQkvRows + channel]
                : projected[static_cast<std::int64_t>(source) * kQkvRows + channel];
    }
}

__global__ __launch_bounds__(kHeadDim)
void norm_sigmoid_kernel(const __nv_bfloat16* recurrent, const __nv_bfloat16* z,
                         const float* weight, __nv_bfloat16* output) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int d    = static_cast<int>(threadIdx.x);
    const int base = token * kValueRows + head * kHeadDim;
    const float value = __bfloat162float(recurrent[base + d]);
    __shared__ float warp_sums[kHeadDim / kWarpSize];
    const float sum = block_reduce_sum<kHeadDim>(value * value, warp_sums);
    __shared__ float inverse;
    if (d == 0) { inverse = rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F); }
    __syncthreads();
    output[base + d] = __float2bfloat16_rn(
        value * inverse * weight[d] * sigmoid(__bfloat162float(z[base + d])));
}

} // namespace

void gated_delta_net_layer_control_launch(const Tensor& x, const Tensor& a_weight,
                                          const Tensor& b_weight, const Tensor& ssm_a,
                                          const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                          cudaStream_t stream) {
    control_kernel<<<dim3((kValueHeads + kWarps - 1) / kWarps, x.ne[1]), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(a_weight.data),
        static_cast<const float*>(b_weight.data), static_cast<const float*>(ssm_a.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_net_layer_conv_launch(const Tensor& projected_qkv, const Tensor& conv_weight,
                                          const Tensor& conv_state_in, Tensor& conv_state_out,
                                          Tensor& q, Tensor& k, Tensor& v,
                                          cudaStream_t stream) {
    conv_kernel<<<(kQkvRows + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected_qkv.data),
        static_cast<const float*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data),
        projected_qkv.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_net_layer_norm_launch(const Tensor& recurrent, const Tensor& z,
                                       const Tensor& norm_weight, Tensor& normalized_gated,
                                       cudaStream_t stream) {
    norm_sigmoid_kernel<<<dim3(kValueHeads, recurrent.ne[2]), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(recurrent.data),
        static_cast<const __nv_bfloat16*>(z.data), static_cast<const float*>(norm_weight.data),
        static_cast<__nv_bfloat16*>(normalized_gated.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

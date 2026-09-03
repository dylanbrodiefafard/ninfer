#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/warp.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

template <int Block>
__launch_bounds__(Block) __global__
    void normalize_bf16_to_fp16_kernel(const __nv_bfloat162* x, __half2* out, std::int64_t rows) {
    static_assert(Block % ninfer::ops::kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / ninfer::ops::kWarpSize;
    constexpr int kPairsPerLane  = 2;
    constexpr int kPairsPerRow   = kStateDim / 2;
    const int lane               = static_cast<int>(threadIdx.x) & (ninfer::ops::kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / ninfer::ops::kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    __nv_bfloat162 values[kPairsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kPairsPerLane; ++k) {
        const int pair  = lane + k * ninfer::ops::kWarpSize;
        values[k]       = x[row_base + pair];
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    sum       = ninfer::ops::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + 1.0e-6f) : 0.0f;
    inv       = __shfl_sync(ninfer::ops::kFullWarpMask, inv, 0);
#pragma unroll
    for (int k = 0; k < kPairsPerLane; ++k) {
        const int pair       = lane + k * ninfer::ops::kWarpSize;
        const float2 xf      = __bfloat1622float2(values[k]);
        out[row_base + pair] = __floats2half2_rn(xf.x * inv, xf.y * inv);
    }
}

} // namespace

void launch_normalize_fp16(const Tensor& x, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::FP16 || x.ne[0] != kStateDim ||
        x.numel() != out.numel()) {
        throw std::invalid_argument(
            "gated_delta_net normalize: requires same-shaped BF16->FP16 D=128 tensors");
    }
    const std::int64_t rows      = out.numel() / kStateDim;
    constexpr int kBlock         = 512;
    constexpr int kWarpsPerBlock = kBlock / ninfer::ops::kWarpSize;
    const auto blocks =
        static_cast<unsigned int>(div_up(rows, static_cast<std::int64_t>(kWarpsPerBlock)));
    normalize_bf16_to_fp16_kernel<kBlock><<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(x.data), static_cast<__half2*>(out.data), rows);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    return chunked::workspace_bytes(value_heads, tokens);
}

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    const auto layout = chunked::compute_workspace_layout(v.ne[1], q.ne[2], q.dtype);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }

    const DeviceSpan backing{workspace, workspace_bytes};
    const Tensor g_cumsum = layout.g_cumsum.bind(backing);
    const Tensor W        = layout.W.bind(backing);
    const Tensor U        = layout.U.bind(backing);
    const Tensor v_new    = layout.v_new.bind(backing);
    const Tensor h_chunk  = layout.h_chunk.bind(backing);

    chunked::prepare_wy_wu_config prepare{};
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.k            = k.data;
    prepare.private_fp16 = k.dtype == DType::FP16;
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = W.data;
    prepare.U            = U.data;
    prepare.g_cumsum_out = static_cast<float*>(g_cumsum.data);
    prepare.stream       = stream;
    CUDA_CHECK(chunked::launch_prepare_wy_wu(prepare));

    chunked::state_passing_config state{};
    state.H_qk         = q.ne[1];
    state.H_v          = v.ne[1];
    state.L            = q.ne[2];
    state.W            = W.data;
    state.U            = U.data;
    state.k            = k.data;
    state.private_fp16 = k.dtype == DType::FP16;
    state.g_cumsum     = static_cast<const float*>(g_cumsum.data);
    state.state_in     = static_cast<const float*>(ssm_state_in.data);
    state.v_new        = v_new.data;
    state.h_chunk      = h_chunk.data;
    state.state_out    = static_cast<float*>(ssm_state_out.data);
    state.stream       = stream;
    CUDA_CHECK(chunked::launch_state_passing(state));

    chunked::chunk_output_config output{};
    output.H_qk         = q.ne[1];
    output.H_v          = v.ne[1];
    output.L            = q.ne[2];
    output.q            = q.data;
    output.k            = k.data;
    output.private_fp16 = q.dtype == DType::FP16;
    output.v_new        = v_new.data;
    output.g_cumsum     = static_cast<const float*>(g_cumsum.data);
    output.h_chunk      = h_chunk.data;
    output.attn_out     = static_cast<__nv_bfloat16*>(out.data);
    output.scale        = scale;
    output.stream       = stream;
    CUDA_CHECK(chunked::launch_output(output));
}

} // namespace ninfer::ops::detail::gated_delta_net

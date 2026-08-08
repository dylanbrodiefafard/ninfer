#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <int WarpsPerCta, int TokenTile>
__global__ __launch_bounds__(WarpsPerCta * 32) void nvfp4_gdn_snapshot_post_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_states, const std::int32_t* __restrict__ initial_slot,
    const std::int32_t* __restrict__ snapshot_base_slot, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ z, std::int32_t tokens) {
    static_assert((kNvfp4GdnSnapshotChannels % 32) == 0);

    struct Shared {
        float initial[3][32];
        float weight[4][32];
    };

    __shared__ Shared shared;

    const int warp                     = static_cast<int>(threadIdx.x) >> 5;
    const int lane                     = static_cast<int>(threadIdx.x) & 31;
    const int row                      = static_cast<int>(blockIdx.x) * 32 + lane;
    constexpr std::int64_t kSlotStride = static_cast<std::int64_t>(kNvfp4GdnSnapshotChannels) * 3;
    if (warp == 0) {
        const std::int64_t initial_base = static_cast<std::int64_t>(*initial_slot) * kSlotStride;
#pragma unroll
        for (int history = 0; history < 3; ++history) {
            shared.initial[history][lane] = __bfloat162float(
                conv_states[initial_base +
                            static_cast<std::int64_t>(history) * kNvfp4GdnSnapshotChannels + row]);
        }
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            shared.weight[tap][lane] = __bfloat162float(
                conv_weight[static_cast<std::int64_t>(tap) * kNvfp4GdnSnapshotChannels + row]);
        }
    }
    __syncthreads();

    const float w0 = shared.weight[0][lane];
    const float w1 = shared.weight[1][lane];
    const float w2 = shared.weight[2][lane];
    const float w3 = shared.weight[3][lane];

    __nv_bfloat16* output;
    std::int32_t output_rows;
    std::int32_t output_row;
    if (row < kNvfp4GdnSnapshotQueryRows) {
        output      = query;
        output_rows = kNvfp4GdnSnapshotQueryRows;
        output_row  = row;
    } else if (row < kNvfp4GdnSnapshotQueryRows + kNvfp4GdnSnapshotKeyRows) {
        output      = key;
        output_rows = kNvfp4GdnSnapshotKeyRows;
        output_row  = row - kNvfp4GdnSnapshotQueryRows;
    } else {
        output      = value;
        output_rows = kNvfp4GdnSnapshotValueRows;
        output_row  = row - kNvfp4GdnSnapshotQueryRows - kNvfp4GdnSnapshotKeyRows;
    }

    const int token_tiles = (tokens + TokenTile - 1) / TokenTile;
    for (int tile = warp; tile < token_tiles; tile += WarpsPerCta) {
        const int token0        = tile * TokenTile;
        const auto load_history = [&](int token) {
            if (token < 0) { return shared.initial[token + 3][lane]; }
            return __bfloat162float(
                projected[static_cast<std::int64_t>(token) * kNvfp4GdnSnapshotParentRows + row]);
        };
        float s0 = load_history(token0 - 3);
        float s1 = load_history(token0 - 2);
        float s2 = load_history(token0 - 1);

#pragma unroll
        for (int local_token = 0; local_token < TokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token < tokens) {
                const std::int64_t projected_base =
                    static_cast<std::int64_t>(token) * kNvfp4GdnSnapshotParentRows;
                const float p = __bfloat162float(projected[projected_base + row]);
                float conv    = fmaf(w0, s0, 0.0F);
                conv          = fmaf(w1, s1, conv);
                conv          = fmaf(w2, s2, conv);
                conv          = fmaf(w3, p, conv);
                output[static_cast<std::int64_t>(token) * output_rows + output_row] =
                    __float2bfloat16_rn(silu(conv));

                const std::int64_t snapshot_base =
                    static_cast<std::int64_t>(*snapshot_base_slot + token) * kSlotStride;
                conv_states[snapshot_base + row] = __float2bfloat16_rn(s1);
                conv_states[snapshot_base + kNvfp4GdnSnapshotChannels + row] =
                    __float2bfloat16_rn(s2);
                conv_states[snapshot_base + 2LL * kNvfp4GdnSnapshotChannels + row] =
                    __float2bfloat16_rn(p);

                if (row < kNvfp4GdnSnapshotZRows) {
                    z[static_cast<std::int64_t>(token) * kNvfp4GdnSnapshotZRows + row] =
                        projected[projected_base + kNvfp4GdnSnapshotChannels + row];
                }

                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }
    }
}

} // namespace

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& initial_slot,
                                    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                    Tensor& value, Tensor& z, cudaStream_t stream) {
    constexpr int kWarpsPerCta = 32;
    constexpr int kTokenTile   = 8;
    constexpr int kBlocks      = kNvfp4GdnSnapshotChannels / 32;
    nvfp4_gdn_snapshot_post_kernel<kWarpsPerCta, kTokenTile>
        <<<kBlocks, kWarpsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<const std::int32_t*>(snapshot_base_slot.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            projected.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

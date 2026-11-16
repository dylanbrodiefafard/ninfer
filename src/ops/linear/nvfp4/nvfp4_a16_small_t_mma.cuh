#pragma once

#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

namespace ninfer::ops::detail {

// Each CTA owns sixteen output rows and all live tokens (2..16). Eight warps split a K512
// group into K64 slices, widening E2M1 codes times their E4M3 scales exactly into BF16 MMA
// operands. Shared codes/scales use the weight stager's XOR-swizzled layout; each warp has a
// separate XOR-swizzled activation tile. The staging union becomes FP32 warp partials only
// after the final CTA barrier. Every asynchronous stage is committed, waited on and followed
// by a CTA barrier before consumption; a barrier also precedes reuse of either pipeline buffer.
// Registered geometries require N divisible by 16 and K divisible by 512. All per-CTA indices
// fit in 32 bits; persistent tensor row offsets use 64 bits. Inputs are 16-byte aligned.
struct Nvfp4A16SmallTMmaSchedule {
    static constexpr int kBlockN         = 16;
    static constexpr int kBlockK         = 512;
    static constexpr int kStages         = 2;
    static constexpr int kCodeRowBytes   = kBlockK / 2;
    static constexpr int kSegmentsPerRow = kCodeRowBytes / 16;
    static constexpr int kK64PerStage    = kBlockK / 64;
    static constexpr int kThreads        = 256;
};

template <class Geometry, int TileTokens>
__global__ __launch_bounds__(256, 2) void nvfp4_a16_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, int tokens, float inverse_weight_divisor,
    Nvfp4ContiguousOutput output) {
    using Schedule        = Nvfp4A16SmallTMmaSchedule;
    constexpr int kKWarps = 8;
    constexpr int kWarpK  = 64;
    constexpr int kPanels = TileTokens / 8;
    constexpr int kGroups = Geometry::kInputRows / Schedule::kBlockK;
    static_assert(TileTokens == 8 || TileTokens == 16);
    static_assert(Geometry::kInputRows % Schedule::kBlockK == 0);
    static_assert(Geometry::kOutputRows % Schedule::kBlockN == 0);

    union SharedStorage {
        struct {
            Nvfp4W4a8WeightStorage<Schedule> weights;
            alignas(16) __nv_bfloat16 activation[Schedule::kStages][kKWarps][TileTokens * kWarpK];
        } staging;

        float partials[kKWarps][kPanels][32][4];
    };

    __shared__ __align__(16) SharedStorage shared;
    const int tid             = static_cast<int>(threadIdx.x);
    const int warp            = tid / 32;
    const int lane            = tid & 31;
    const int gid             = lane / 4;
    const int lid             = lane & 3;
    const int row0            = static_cast<int>(blockIdx.x) * Schedule::kBlockN;
    const auto activation_col = [](int token, int col) { return col ^ ((token & 7) * 8); };
    const auto stage          = [&](int group) {
        const int slot = group % Schedule::kStages;
        stage_nvfp4_w4a4_weight<Geometry, Schedule, Nvfp4W4a4IdentityRows>(
            codes, scales, shared.staging.weights, slot, group, row0, {});
        for (int item = lane; item < TileTokens * (kWarpK / 8); item += 32) {
            const int token = item / (kWarpK / 8);
            const int col   = (item % (kWarpK / 8)) * 8;
            cp_async_zfill<16, Cache::cg>(
                &shared.staging.activation[slot][warp][token * kWarpK + activation_col(token, col)],
                x + static_cast<std::int64_t>(token < tokens ? token : 0) * Geometry::kInputRows +
                    group * Schedule::kBlockK + warp * kWarpK + col,
                token < tokens ? 16 : 0);
        }
        cp_commit();
    };
    float accumulators[kPanels][4] = {};
    stage(0);
    stage(1);
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        const int slot = group % Schedule::kStages;
        if (group + 1 < kGroups) {
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < kWarpK / 16; ++step) {
            const int scale_group  = warp * (kWarpK / 16) + step;
            const int code_col     = scale_group * 8 + lid;
            const auto load_weight = [&](int row, unsigned& lo, unsigned& hi) {
                const auto* row_codes =
                    shared.staging.weights.b_codes[slot] + row * Schedule::kCodeRowBytes;
                // BF16 MMA A fragments pair K=(2*lid, 2*lid+1) and (2*lid+8, 2*lid+9).
                // Assemble those code bytes before widening, matching ldmatrix's B order.
                const unsigned packed =
                    row_codes[nvfp4_w4a4_swizzled_byte<Schedule>(row, code_col)] |
                    (static_cast<unsigned>(
                         row_codes[nvfp4_w4a4_swizzled_byte<Schedule>(row, code_col + 4)])
                     << 8);
                const auto scale = nvfp4_scale_bf16x2(
                    shared.staging.weights
                        .b_scales[slot][row * (Schedule::kBlockK / 16) + scale_group]);
                nvfp4_codes_to_bf16x2(packed, scale, lo, hi);
            };
            unsigned a0, a1, a2, a3;
            load_weight(gid, a0, a2);
            load_weight(gid + 8, a1, a3);
#pragma unroll
            for (int panel = 0; panel < kPanels; ++panel) {
                const int token = panel * 8 + (lane & 7);
                const int col   = step * 16 + ((lane >> 3) & 1) * 8;
                unsigned b0, b1;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(&shared.staging.activation[slot][warp][token * kWarpK +
                                                                     activation_col(token, col)]));
                mma_bf16(accumulators[panel][0], accumulators[panel][1], accumulators[panel][2],
                         accumulators[panel][3], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
        if (group + Schedule::kStages < kGroups) { stage(group + Schedule::kStages); }
    }
#pragma unroll
    for (int panel = 0; panel < kPanels; ++panel) {
        store_vec(shared.partials[warp][panel][lane],
                  make_float4(accumulators[panel][0], accumulators[panel][1],
                              accumulators[panel][2], accumulators[panel][3]));
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int panel = 0; panel < kPanels; ++panel) {
            float sum[4] = {};
#pragma unroll
            for (int split = 0; split < kKWarps; ++split) {
                const float4 partial = load_vec<float4>(shared.partials[split][panel][lane]);
                sum[0] += partial.x;
                sum[1] += partial.y;
                sum[2] += partial.z;
                sum[3] += partial.w;
            }
            const int token = panel * 8 + lid * 2;
            if (token < tokens) {
                output.store(row0 + gid, token, sum[0] * inverse_weight_divisor);
                output.store(row0 + gid + 8, token, sum[2] * inverse_weight_divisor);
            }
            if (token + 1 < tokens) {
                output.store(row0 + gid, token + 1, sum[1] * inverse_weight_divisor);
                output.store(row0 + gid + 8, token + 1, sum[3] * inverse_weight_divisor);
            }
        }
    }
}

template <class Geometry>
void launch_nvfp4_a16_small_t_mma(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    const auto launch = [&]<int TileTokens>() {
        nvfp4_a16_small_t_mma_kernel<Geometry, TileTokens>
            <<<Geometry::kOutputRows / 16, Nvfp4A16SmallTMmaSchedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), x.ne[1],
                1.0F / weight.weight_scale_divisor,
                Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), weight.n});
        CUDA_CHECK(cudaGetLastError());
    };
    if (x.ne[1] <= 8) {
        launch.template operator()<8>();
    } else {
        launch.template operator()<16>();
    }
}

} // namespace ninfer::ops::detail

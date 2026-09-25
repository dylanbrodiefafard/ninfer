#pragma once

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"

namespace ninfer::ops::detail {

// Exact E2M1-code expansion; the stored block scale is applied to each K16
// partial in FP32, never rounded into an FP8 weight.
__device__ __forceinline__ unsigned nvfp4_codes_to_fp8(unsigned codes) {
    const unsigned magnitude = __byte_perm(0x3c383000U, 0x4c484440U, codes & 0x7777U);
    return magnitude | ((codes & 0x0008U) << 4) | ((codes & 0x0080U) << 8) |
           ((codes & 0x0800U) << 12) | ((codes & 0x8000U) << 16);
}

template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads, 1) void nvfp4_w4a8_mma_kernel(
    Fp8A8Workspace activation, const std::uint8_t* weight_codes,
    const std::uint8_t* weight_scales, int tokens, float inverse_weight_scale,
    Epilogue epilogue, Output output, RowPolicy rows = {}) {
    constexpr int BM = Schedule::kBlockM, BN = Schedule::kBlockN;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    constexpr int KT = Geometry::kInputRows / BK;
    constexpr int OS = BN + 8;
    constexpr bool fp32_tile = requires(Output o, float* tile) {
        o.template store_tile<Schedule>(tile, 0, 0);
    };
    // Projection staging is dead after the final K-loop barrier. Reuse it for
    // the FP32 convolution tile instead of reducing residency with another tile.
    union Staging {
        Nvfp4W4a4SharedStorage<Schedule> weights;
        float tile[BM * OS];
    };
    __shared__ Staging staging;
    auto& weights = staging.weights;
    __shared__ __align__(16) std::uint8_t inputs[S][BM * BK];
    __shared__ __align__(16) __nv_bfloat16 bf16_result[BM * OS];
    auto* result = [&] {
        if constexpr (fp32_tile) { return staging.tile; }
        else { return bf16_result; }
    }();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / Schedule::kWarpsN, wn = warp % Schedule::kWarpsN;
    const int token_begin = blockIdx.y * BM;
    const int row_begin = blockIdx.x * (PairRows ? BN / 2 : BN);
    const auto input_byte = [](int row, int column) {
        return row * BK + (column ^ ((row & (BK / 16 - 1)) * 16));
    };
    const auto stage_inputs = [&](int stage, int tile) {
        for (int task = tid; task < BM * BK / 16; task += Schedule::kThreads) {
            const int row = task / (BK / 16), column = (task % (BK / 16)) * 16;
            const int token = token_begin + row;
            const bool valid = token < tokens;
            cp_async_zfill<16, Cache::cg>(inputs[stage] + input_byte(row, column),
                activation.codes + static_cast<std::int64_t>(valid ? token : 0) * Geometry::kInputRows + tile * BK + column,
                valid ? 16 : 0);
        }
        stage_nvfp4_w4a4_weight<Geometry, Schedule, RowPolicy>(
            weight_codes, weight_scales, weights, stage, tile, row_begin, rows);
        cp_commit();
    };
    for (int stage = 0; stage < S; ++stage) { stage_inputs(stage, stage); }
    float accum[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    for (int tile = 0; tile < KT; ++tile) {
        const int stage = tile % S;
        if (tile + S <= KT) { cp_wait<S - 1>(); } else { cp_wait<0>(); }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < BK / 16; ++group) {
            unsigned a[Schedule::kMmaM][2];
#pragma unroll
            for (int m = 0; m < Schedule::kMmaM; ++m) {
                const int row = wm * Schedule::kWarpM + m * 16 + lane / 4;
                const int column = group * 16 + (lane & 3) * 4;
                a[m][0] = load_vec<unsigned>(inputs[stage] + input_byte(row, column));
                a[m][1] = load_vec<unsigned>(inputs[stage] + input_byte(row + 8, column));
            }
#pragma unroll
            for (int n = 0; n < Schedule::kMmaN; ++n) {
                const int row = wn * Schedule::kWarpN + n * 8 + lane / 4;
                const int column = group * 8 + (lane & 3) * 2;
                const unsigned packed = load_vec<std::uint16_t>(weights.b_codes[stage] + row * Schedule::kCodeRowBytes +
                    nvfp4_w4a4_swizzled_byte<Schedule>(row, column));
                const unsigned b = nvfp4_codes_to_fp8(packed);
                float scale[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int sr = wn * Schedule::kWarpN + n * 8 + (lane & 3) * 2 + j;
                    int offset;
                    if constexpr (RowPolicy::kContiguous && BN >= 32) {
                        constexpr int quartiles = BN >= 128 ? 4 : BN / 32;
                        offset = ((sr / 128 * Schedule::kK64PerStage + group / 4) * 32 + (sr & 31)) * quartiles * 4 +
                                 ((sr & 127) / 32) * 4 + group % 4;
                    } else {
                        offset = (sr * Schedule::kK64PerStage + group / 4) * 4 + group % 4;
                    }
                    scale[j] = decode_nvfp4_e4m3(weights.b_scales[stage][offset]);
                }
#pragma unroll
                for (int m = 0; m < Schedule::kMmaM; ++m) {
                    float partial[4] = {};
                    mma_fp8_e4m3_k16(partial, a[m][0], a[m][1], b);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        accum[m][n][j] = fmaf(partial[j], scale[j & 1], accum[m][n][j]);
                    }
                }
            }
        }
        __syncthreads();
        if (tile + S < KT) { stage_inputs(stage, tile + S); }
    }
    constexpr bool fp32 = requires(Output o) { o.store_fp32(0, 0, 0.0F); };
#pragma unroll
    for (int m = 0; m < Schedule::kMmaM; ++m) {
#pragma unroll
        for (int n = 0; n < Schedule::kMmaN; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int local_token = wm * Schedule::kWarpM + m * 16 + lane / 4 + (j / 2) * 8;
                const int token = token_begin + local_token;
                const int local_row = wn * Schedule::kWarpN + n * 8 + (lane & 3) * 2 + (j & 1);
                const int parent_row = rows.weight_row(row_begin, local_row);
                float value = 0;
                if (token < tokens) {
                    value = epilogue.apply(parent_row, token, accum[m][n][j] * (activation.scales[token] * inverse_weight_scale));
                }
                if constexpr (fp32_tile) {
                    result[local_token * OS + local_row] = value;
                } else if constexpr (fp32) {
                    if (token < tokens) { output.store_fp32(parent_row, token, value); }
                } else {
                    result[local_token * OS + local_row] = __float2bfloat16_rn(value);
                }
            }
        }
    }
    if constexpr (fp32_tile) {
        __syncthreads();
        output.template store_tile<Schedule>(result, row_begin, tokens);
    } else if constexpr (!fp32) {
        __syncthreads();
        constexpr int stored_rows = PairRows ? BN / 2 : BN;
        for (int task = tid; task < BM * stored_rows / 8; task += Schedule::kThreads) {
            const int local_token = task / (stored_rows / 8), row = task % (stored_rows / 8) * 8;
            const int token = token_begin + local_token;
            if (token < tokens) {
                const uint4 value = load_vec<uint4>(result + local_token * OS + row);
                if constexpr (PairRows) {
                    output.store_pair_vector(row_begin + row, token, value,
                        load_vec<uint4>(result + local_token * OS + stored_rows + row));
                } else { output.store_vector(row_begin + row, token, value); }
            }
        }
    }
}

template <class Geometry, class Epilogue, class Output,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false>
void launch_nvfp4_w4a8_mma(const Weight& weight, int tokens, Fp8A8Workspace workspace,
                           Epilogue epilogue, Output output, cudaStream_t stream, RowPolicy rows = {}) {
    const auto launch = [&]<int BM>() {
        // N=5120 with N64 launches only 80 CTAs on 170 SMs. Partition output rows
        // without splitting K or replaying weights; keep the ascending K16 FMAs.
        constexpr int BN = Geometry::kOutputRows == 5120 ? (BM == 16 ? 16 : 32) : 64;
        using Schedule = Nvfp4W4a4MmaSchedule<BM, BN, 256, 1, BN == 16 ? 2 : 4, 2, 1>;
        const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, (tokens + BM - 1) / BM);
        nvfp4_w4a8_mma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>
            <<<grid, Schedule::kThreads, 0, stream>>>(workspace,
                static_cast<const std::uint8_t*>(weight.qdata), static_cast<const std::uint8_t*>(weight.scales),
                tokens, 1.0F / weight.weight_scale_divisor, epilogue, output, rows);
    };
    if (tokens <= 16) { launch.template operator()<16>(); }
    else { launch.template operator()<32>(); }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

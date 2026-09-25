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

// A8 stages only the weight half of the W4A4 layout; FP8 activations use their own buffer.
template <class Schedule>
struct Nvfp4W4a8WeightStorage {
    alignas(16) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(16) std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockN * Schedule::kK64PerStage * 4];
};

// SwapAB places sixteen weight rows on MMA M and eight-token panels on N. Token panels past the
// CTA's valid tokens are skipped, and each weight code and scale is expanded once per 16 rows.
template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false, bool SwapAB = false>
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
        Nvfp4W4a8WeightStorage<Schedule> weights;
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
    const auto scale_offset = [](int sr, int group) {
        if constexpr (RowPolicy::kContiguous && BN >= 32) {
            constexpr int quartiles = BN >= 128 ? 4 : BN / 32;
            return ((sr / 128 * Schedule::kK64PerStage + group / 4) * 32 + (sr & 31)) * quartiles * 4 +
                   ((sr & 127) / 32) * 4 + group % 4;
        } else {
            return (sr * Schedule::kK64PerStage + group / 4) * 4 + group % 4;
        }
    };
    static_assert(!SwapAB || (Schedule::kWarpsM == 1 && Schedule::kWarpN % 16 == 0));
    constexpr int kSwapRows = Schedule::kWarpN / 16, kSwapPanels = BM / 8;
    float accum[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    float swap_accum[SwapAB ? kSwapRows : 1][SwapAB ? kSwapPanels : 1][4] = {};
    const int valid_panels = min(kSwapPanels, (tokens - token_begin + 7) / 8);
    for (int tile = 0; tile < KT; ++tile) {
        const int stage = tile % S;
        if (tile + S <= KT) { cp_wait<S - 1>(); } else { cp_wait<0>(); }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < BK / 16; ++group) {
            if constexpr (SwapAB) {
                unsigned b[kSwapPanels];
#pragma unroll
                for (int p = 0; p < kSwapPanels; ++p) {
                    if (p < valid_panels) {
                        b[p] = load_vec<unsigned>(inputs[stage] + input_byte(p * 8 + lane / 4, group * 16 + (lane & 3) * 4));
                    }
                }
#pragma unroll
                for (int r = 0; r < kSwapRows; ++r) {
                    const int row = wn * Schedule::kWarpN + r * 16 + lane / 4;
                    const int column = group * 8 + (lane & 3) * 2;
                    const unsigned a0 = nvfp4_codes_to_fp8(load_vec<std::uint16_t>(weights.b_codes[stage] +
                        row * Schedule::kCodeRowBytes + nvfp4_w4a4_swizzled_byte<Schedule>(row, column)));
                    const unsigned a1 = nvfp4_codes_to_fp8(load_vec<std::uint16_t>(weights.b_codes[stage] +
                        (row + 8) * Schedule::kCodeRowBytes + nvfp4_w4a4_swizzled_byte<Schedule>(row + 8, column)));
                    const float scale0 = decode_nvfp4_e4m3(weights.b_scales[stage][scale_offset(row, group)]);
                    const float scale1 = decode_nvfp4_e4m3(weights.b_scales[stage][scale_offset(row + 8, group)]);
#pragma unroll
                    for (int p = 0; p < kSwapPanels; ++p) {
                        if (p < valid_panels) {
                            float partial[4] = {};
                            mma_fp8_e4m3_k16(partial, a0, a1, b[p]);
                            swap_accum[r][p][0] = fmaf(partial[0], scale0, swap_accum[r][p][0]);
                            swap_accum[r][p][1] = fmaf(partial[1], scale0, swap_accum[r][p][1]);
                            swap_accum[r][p][2] = fmaf(partial[2], scale1, swap_accum[r][p][2]);
                            swap_accum[r][p][3] = fmaf(partial[3], scale1, swap_accum[r][p][3]);
                        }
                    }
                }
            } else {
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
                        scale[j] = decode_nvfp4_e4m3(weights.b_scales[stage][scale_offset(sr, group)]);
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
        }
        __syncthreads();
        if (tile + S < KT) { stage_inputs(stage, tile + S); }
    }
    constexpr bool fp32 = requires(Output o) { o.store_fp32(0, 0, 0.0F); };
    const auto emit = [&](int local_token, int local_row, float accumulated) {
        const int token = token_begin + local_token;
        const int parent_row = rows.weight_row(row_begin, local_row);
        float value = 0;
        if (token < tokens) {
            value = epilogue.apply(parent_row, token, accumulated * (activation.scales[token] * inverse_weight_scale));
        }
        if constexpr (fp32_tile) {
            result[local_token * OS + local_row] = value;
        } else if constexpr (fp32) {
            if (token < tokens) { output.store_fp32(parent_row, token, value); }
        } else {
            result[local_token * OS + local_row] = __float2bfloat16_rn(value);
        }
    };
    if constexpr (SwapAB) {
#pragma unroll
        for (int r = 0; r < kSwapRows; ++r) {
#pragma unroll
            for (int p = 0; p < kSwapPanels; ++p) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    emit(p * 8 + (lane & 3) * 2 + (j & 1), wn * Schedule::kWarpN + r * 16 + lane / 4 + (j / 2) * 8,
                         swap_accum[r][p][j]);
                }
            }
        }
    } else {
#pragma unroll
        for (int m = 0; m < Schedule::kMmaM; ++m) {
#pragma unroll
            for (int n = 0; n < Schedule::kMmaN; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    emit(wm * Schedule::kWarpM + m * 16 + lane / 4 + (j / 2) * 8,
                         wn * Schedule::kWarpN + n * 8 + (lane & 3) * 2 + (j & 1), accum[m][n][j]);
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
    // N=5120 with N64 launches only 80 CTAs on 170 SMs. Partition output rows without splitting
    // K or replaying weights, and deepen the pipeline instead: M16 N=5120 streams K512 over three
    // stages, other M16 and M32 N=5120 schedules stream K256 over three. Wider projections place
    // weight rows on MMA M (SwapAB); M32 keeps two stages. Every schedule keeps ascending K16 FMAs.
    const auto launch = [&]<int BM>() {
        constexpr bool narrow = Geometry::kOutputRows == 5120;
        constexpr int BN = narrow ? (BM == 16 ? 16 : 32) : 64;
        constexpr int BK = narrow && BM == 16 ? 512 : 256;
        constexpr int S  = narrow || BM == 16 ? 3 : 2;
        using Schedule = Nvfp4W4a4MmaSchedule<BM, BN, BK, 1, BN == 16 ? 2 : 4, S, 1>;
        const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN, (tokens + BM - 1) / BM);
        nvfp4_w4a8_mma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows, !narrow>
            <<<grid, Schedule::kThreads, 0, stream>>>(workspace,
                static_cast<const std::uint8_t*>(weight.qdata), static_cast<const std::uint8_t*>(weight.scales),
                tokens, 1.0F / weight.weight_scale_divisor, epilogue, output, rows);
    };
    if (tokens <= 16) { launch.template operator()<16>(); }
    else { launch.template operator()<32>(); }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

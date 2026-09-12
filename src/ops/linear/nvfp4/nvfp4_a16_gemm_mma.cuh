#pragma once

// Wide-T persistent NVFP4 weight x BF16 activation Tensor Core GEMM.
//
//   out[M,T] = decode_nvfp4(weight[M,K]) * x[K,T]
//
// A CTA stages one persistent packed-weight tile, decodes it once to a private BF16 MMA operand,
// and replays that tile across a BF16 activation panel. The represented public activation remains
// BF16 and the result is stored as BF16. No decoded weight is materialized outside the CTA.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ int nvfp4_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm, Cache WeightCache = Cache::cg,
          Cache ActivationCache = Cache::cg>
struct Nvfp4A16GemmSchedule {
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;

    static constexpr int kWarpsRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps       = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads     = kWarps * 32;
    static constexpr int kMmaRows     = kWarpRows / 16;
    static constexpr int kMmaTokens   = kWarpTokens / 8;
    static constexpr int kMmaK        = kBlockK / 16;
    static constexpr int kCodeBytes   = kBlockRows * kBlockK / 2;
    static constexpr int kScaleBytes  = kBlockRows * kBlockK / 16;
    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kCodeBytes + kScaleBytes;

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert((kBlockRows % kWarpRows) == 0 && (kBlockTokens % kWarpTokens) == 0);
    static_assert((kWarpRows % 16) == 0 && (kWarpTokens % 8) == 0);
    static_assert(kBlockK == 64 || kBlockK == 128);
    static_assert(kActivationStages == 1 || kActivationStages == 2);
    static_assert(kMinBlocksPerSm > 0);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Geometry, class Schedule, bool FullTokens>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void nvfp4_a16_gemm_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ weight_codes,
                               const std::uint8_t* __restrict__ weight_scales,
                               float inverse_weight_divisor, Nvfp4ContiguousOutput output,
                               std::int32_t tokens) {
    constexpr int M       = Geometry::kOutputRows;
    constexpr int K       = Geometry::kInputRows;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpTokens;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int KSUB    = Schedule::kMmaK;
    constexpr int THREADS = Schedule::kThreads;
    static_assert((M % BM) == 0);
    static_assert((K % BK) == 0);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* weight_shared     = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared = weight_shared + BM * BK;
    auto* code_shared       = reinterpret_cast<std::uint8_t*>(
        activation_shared + Schedule::kActivationStages * BN * BK);
    auto* scale_shared = code_shared + Schedule::kCodeBytes;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row_begin   = static_cast<int>(blockIdx.x) * BM;
    const int token_begin = static_cast<int>(blockIdx.y) * BN;

    float accumulators[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_col_offset = ((lane >> 3) & 1) << 3;

    const auto stage_activation = [&](int stage, int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int local_token = item / (BK / 8);
            const int k8          = item - local_token * (BK / 8);
            const int token       = token_begin + local_token;
            const int kk          = k8 * 8;
            auto* destination = &activation_shared[stage * BN * BK + local_token * BK +
                                                   nvfp4_a16_shared_col_64(local_token, kk)];
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
    };

    const auto stage_weight = [&](int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < Schedule::kCodeBytes / 16; item += THREADS) {
            const int local_row = item / (BK / 32);
            const int chunk     = item - local_row * (BK / 32);
            cp_async<16, Schedule::kWeightCache>(
                code_shared + local_row * (BK / 2) + chunk * 16,
                weight_codes + static_cast<std::int64_t>(row_begin + local_row) * (K / 2) +
                    k_begin / 2 + chunk * 16);
        }
#pragma unroll 1
        for (int item = tid; item < Schedule::kScaleBytes; item += THREADS) {
            const int local_row   = item / (BK / 16);
            const int local_group = item - local_row * (BK / 16);
            scale_shared[item] = weight_scales[nvfp4_scale_offset<Geometry>(
                row_begin + local_row, k_begin / 16 + local_group)];
        }
    };

    const auto widen_weight = [&]() {
        constexpr int kPairs = BM * BK / 2;
#pragma unroll 1
        for (int item = tid; item < kPairs; item += THREADS) {
            const int local_row = item / (BK / 2);
            const int pair      = item - local_row * (BK / 2);
            const int col       = pair * 2;
            // Every finite E2M1*E4M3 product is exactly representable in BF16. Keep the artifact
            // divisor out of this private operand cast and apply it once to the FP32 accumulator.
            const float scale =
                decode_nvfp4_e4m3(scale_shared[local_row * (BK / 16) + col / 16]);
            const float2 code = decode_nvfp4_e2m1x2(code_shared[local_row * (BK / 2) + pair]);
            auto* destination =
                &weight_shared[local_row * BK + nvfp4_a16_shared_col_64(local_row, col)];
            destination[0] = __float2bfloat16_rn(code.x * scale);
            destination[1] = __float2bfloat16_rn(code.y * scale);
        }
    };

    stage_activation(0, 0);
    stage_weight(0);
    cp_commit();

    constexpr int kTiles = K / BK;
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kActivationStages;
        cp_wait<0>();
        __syncthreads();

        widen_weight();
        __syncthreads();

        const int next = k_tile + 1;
        if (next < kTiles) {
            if constexpr (Schedule::kActivationStages == 2) { stage_activation(next & 1, next); }
            stage_weight(next);
            cp_commit();
        }

        unsigned a_fragments[2][MT][4];
        unsigned b_fragments[2][NT][2];
        const auto load_fragments = [&](int slot, int k_step) {
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                const int row = wm * WM + mma_row * 16 + a_row_offset;
                const int col = k_step * 16 + a_col_offset;
                ldmatrix_x4(a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                            a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                            smem_addr(&weight_shared[row * BK +
                                                     nvfp4_a16_shared_col_64(row, col)]));
            }
#pragma unroll
            for (int mma_token = 0; mma_token < NT; ++mma_token) {
                const int row = wn * WN + mma_token * 8 + b_inner_row;
                const int col = k_step * 16 + b_col_offset;
                ldmatrix_x2(b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1],
                            smem_addr(&activation_shared[stage * BN * BK + row * BK +
                                                         nvfp4_a16_shared_col_64(row, col)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) { load_fragments(slot ^ 1, k_step + 1); }
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
#pragma unroll
                for (int mma_token = 0; mma_token < NT; ++mma_token) {
                    mma_bf16(
                        accumulators[mma_row][mma_token][0], accumulators[mma_row][mma_token][1],
                        accumulators[mma_row][mma_token][2], accumulators[mma_row][mma_token][3],
                        a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                        a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                        b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1]);
                }
            }
        }

        if constexpr (Schedule::kActivationStages == 1) {
            if (next < kTiles) {
                __syncthreads();
                stage_activation(0, next);
                cp_commit();
            }
        }
    }

#pragma unroll
    for (int mma_row = 0; mma_row < MT; ++mma_row) {
        const int row0 = row_begin + wm * WM + mma_row * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int mma_token = 0; mma_token < NT; ++mma_token) {
            const int token0    = token_begin + wn * WN + mma_token * 8 + 2 * lid;
            const int token1    = token0 + 1;
            const float* values = accumulators[mma_row][mma_token];
            if constexpr (FullTokens) {
                output.store(row0, token0, values[0] * inverse_weight_divisor);
                output.store(row0, token1, values[1] * inverse_weight_divisor);
                output.store(row1, token0, values[2] * inverse_weight_divisor);
                output.store(row1, token1, values[3] * inverse_weight_divisor);
            } else {
                if (token0 < tokens) {
                    output.store(row0, token0, values[0] * inverse_weight_divisor);
                    output.store(row1, token0, values[2] * inverse_weight_divisor);
                }
                if (token1 < tokens) {
                    output.store(row0, token1, values[1] * inverse_weight_divisor);
                    output.store(row1, token1, values[3] * inverse_weight_divisor);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail

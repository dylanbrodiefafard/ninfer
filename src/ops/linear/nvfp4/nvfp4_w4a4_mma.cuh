#pragma once

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/a4_activation.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <int BlockM, int BlockN, int BlockK, int WarpsM, int WarpsN, int Stages,
          int MinBlocksPerSm>
struct Nvfp4W4a4MmaSchedule {
    static_assert(BlockM > 0 && (BlockM % 16) == 0);
    static_assert(BlockN > 0 && (BlockN % 8) == 0);
    static_assert(BlockK >= 64 && (BlockK % 64) == 0);
    static_assert(WarpsM > 0 && WarpsN > 0);
    static_assert((BlockM % WarpsM) == 0 && ((BlockM / WarpsM) % 16) == 0);
    static_assert((BlockN % WarpsN) == 0 && ((BlockN / WarpsN) % 8) == 0);
    static_assert(Stages >= 1 && Stages <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kBlockM         = BlockM;
    static constexpr int kBlockN         = BlockN;
    static constexpr int kBlockK         = BlockK;
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kWarpsN         = WarpsN;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarps          = WarpsM * WarpsN;
    // The M64N64 residual route at 113..128 tokens runs faster with ordinary stream ordering
    // on the RTX 5090; the other MMA schedules benefit from dependent launch setup.
    static constexpr bool kProgrammaticDependency = BlockM != 64 || BlockN != 64;

    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpM          = BlockM / WarpsM;
    static constexpr int kWarpN          = BlockN / WarpsN;
    static constexpr int kMmaM           = kWarpM / 16;
    static constexpr int kMmaN           = kWarpN / 8;
    static constexpr int kK64PerStage    = BlockK / 64;
    static constexpr int kCodeRowBytes   = BlockK / 2;
    static constexpr int kSegmentsPerRow = kCodeRowBytes / 16;
};

// The schedules behind the MMA values of Nvfp4W4a4Route: token tile x weight-row tile, K
// stages, and resident CTAs per SM (S3 at M32N64 runs two CTAs per SM).
using Nvfp4W4a4M32N64S3          = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 3, 2>;
using Nvfp4W4a4M32N64S4          = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 4, 1>;
using Nvfp4W4a4M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using Nvfp4W4a4M64N64            = Nvfp4W4a4MmaSchedule<64, 64, 256, 4, 2, 2, 1>;
using Nvfp4W4a4M64N128S3         = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 3, 1>;
using Nvfp4W4a4M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using Nvfp4W4a4M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <Nvfp4Problem Problem, Nvfp4W4a4Route Route, class Schedule, class Visitor>
bool nvfp4_w4a4_visit_reachable(Nvfp4W4a4Route route, const Visitor& visitor) {
    if constexpr (nvfp4_w4a4_route_reachable(Problem, Route)) {
        if (route == Route) {
            visitor.template operator()<Schedule>();
            return true;
        }
    }
    return false;
}

// Calls visitor.template operator()<Schedule>() with the schedule of an MMA route of Problem's
// table. Only the schedules that table can select are instantiated. Throws for any other route,
// including Nvfp4W4a4Route::Tma, which the caller dispatches to its TMA launcher first.
template <Nvfp4Problem Problem, class Visitor>
void visit_nvfp4_w4a4_mma_schedule(Nvfp4W4a4Route route, const Visitor& visitor) {
    using enum Nvfp4W4a4Route;
    const bool launched =
        nvfp4_w4a4_visit_reachable<Problem, M32N64S3, Nvfp4W4a4M32N64S3>(route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M32N64S4, Nvfp4W4a4M32N64S4>(route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M32N128, Nvfp4W4a4M32N128>(route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M64N64, Nvfp4W4a4M64N64>(route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M64N128S3, Nvfp4W4a4M64N128S3>(route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M128N128Pipelined, Nvfp4W4a4M128N128Pipelined>(
            route, visitor) ||
        nvfp4_w4a4_visit_reachable<Problem, M128N128Resident, Nvfp4W4a4M128N128Resident>(route,
                                                                                         visitor);
    if (!launched) {
        throw std::invalid_argument("nvfp4 W4A4: the route has no MMA schedule for this problem");
    }
}

struct Nvfp4W4a4MaterializedActivation {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
};

struct Nvfp4W4a4IdentityRows {
    static constexpr bool kContiguous = true;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

template <class Schedule>
struct Nvfp4W4a4SharedStorage {
    alignas(
        16) std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeRowBytes];
    alignas(
        16) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(
        16) std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockM * Schedule::kK64PerStage];
    alignas(16)
        std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockN * Schedule::kK64PerStage * 4];
};

template <class Schedule>
__device__ __forceinline__ int nvfp4_w4a4_swizzled_byte(int row, int logical_byte) {
    static_assert((Schedule::kSegmentsPerRow & (Schedule::kSegmentsPerRow - 1)) == 0);
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

// Byte offset of scale (token, group) in the tiled plane (ninfer/ops/a4_activation.h).
template <class Geometry>
__device__ __forceinline__ std::int64_t nvfp4_tiled_scale_offset(int token, int group) {
    return a4_scale_offset<Geometry::kInputRows / 16>(token, group);
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void
stage_nvfp4_w4a4_activation(Nvfp4W4a4MaterializedActivation source,
                            Nvfp4W4a4SharedStorage<Schedule>& shared, int stage, int k_tile,
                            int token_begin, int active_tokens) {
    constexpr int kCodeTasks = Schedule::kBlockM * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int token           = token_begin + row;
        const bool valid          = token < active_tokens;
        const int source_token    = valid ? token : 0;
        const int physical_byte   = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = source.codes +
                            static_cast<std::int64_t>(source_token) * Geometry::kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0);
    }

    // A stage's K tile spans kK64PerStage * 4 scale groups of one token, a contiguous run of the
    // tiled plane as long as it stays inside one kNvfp4ScaleTileGroups tile.
    static_assert(kNvfp4ScaleTileGroups % (Schedule::kK64PerStage * 4) == 0);
    constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM;
         row += Schedule::kThreads) {
        const int token        = token_begin + row;
        const bool valid       = token < active_tokens;
        const int source_token = valid ? token : 0;
        auto* destination      = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
        const auto* input =
            source.scales + nvfp4_tiled_scale_offset<Geometry>(source_token, k_tile * kScaleBytes);
        cp_async_zfill<kScaleBytes>(destination, input, valid ? kScaleBytes : 0);
    }
}

// Storage supplies b_codes/b_scales with the Nvfp4W4a4SharedStorage weight layout. Contiguous
// row policies preserve ordered 32-row scale quartiles, including the original plane offset.
template <class Geometry, class Schedule, class RowPolicy, Cache WeightCache = Cache::cg,
          class Storage>
__device__ __forceinline__ void stage_nvfp4_w4a4_weight(const std::uint8_t* __restrict__ codes,
                                                        const std::uint8_t* __restrict__ scales,
                                                        Storage& shared, int stage, int k_tile,
                                                        int row_begin, RowPolicy row_policy) {
    constexpr int kCodeTasks = Schedule::kBlockN * Schedule::kSegmentsPerRow;
    if constexpr (WeightCache == Cache::EvictFirst) {
        for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks;
             task += Schedule::kThreads) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int weight_row      = row_policy.weight_row(row_begin, row);
            const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_segment * 16);
            auto* destination =
                shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
            const auto* input = codes +
                                static_cast<std::int64_t>(weight_row) * Geometry::kCodeBytesPerRow +
                                k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
            cp_async_evict_first_16_noinline(destination, input);
        }
    } else {
        for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks;
             task += Schedule::kThreads) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int weight_row      = row_policy.weight_row(row_begin, row);
            const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_segment * 16);
            auto* destination =
                shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
            const auto* input = codes +
                                static_cast<std::int64_t>(weight_row) * Geometry::kCodeBytesPerRow +
                                k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
            if constexpr (WeightCache == Cache::ca) {
                cp_sync_cs_16(destination, input);
            } else {
                cp_async<16, Cache::cg>(destination, input);
            }
        }
    }

    if constexpr (RowPolicy::kContiguous && Schedule::kBlockN >= 32) {
        static_assert((Schedule::kBlockN % 32) == 0);
        constexpr int kScaleRowTiles     = Schedule::kBlockN >= 128 ? Schedule::kBlockN / 128 : 1;
        constexpr int kQuartilesPerTile  = Schedule::kBlockN >= 128 ? 4 : Schedule::kBlockN / 32;
        constexpr int kScaleBytesPerTask = kQuartilesPerTile * 4;
        constexpr int kScaleTasks        = kScaleRowTiles * Schedule::kK64PerStage * 32;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row_tile         = task / (Schedule::kK64PerStage * 32);
            const int remainder        = task - row_tile * Schedule::kK64PerStage * 32;
            const int local_k64        = remainder / 32;
            const int row_mod32        = remainder - local_k64 * 32;
            const int global_k64       = k_tile * Schedule::kK64PerStage + local_k64;
            const int global_row_begin = row_policy.weight_row(row_begin, row_tile * 128);
            const int persistent_tile  = global_row_begin / 128;
            const int first_quartile   = (global_row_begin & 127) / 32;
            auto* destination = shared.b_scales[stage] +
                                ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) *
                                    kScaleBytesPerTask;
            const auto* input = scales +
                                static_cast<std::int64_t>(
                                    persistent_tile * Geometry::kScaleTilesPerRow + global_k64) *
                                    512 +
                                row_mod32 * 16 + first_quartile * 4;
            cp_async<kScaleBytesPerTask>(destination, input);
        }
    } else {
        constexpr int kScaleTasks = Schedule::kBlockN * Schedule::kK64PerStage;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row             = task / Schedule::kK64PerStage;
            const int local_k64       = task - row * Schedule::kK64PerStage;
            const int weight_row      = row_policy.weight_row(row_begin, row);
            const int global_k64      = k_tile * Schedule::kK64PerStage + local_k64;
            const int persistent_tile = weight_row / 128;
            const int row_in_tile     = weight_row & 127;
            const int row_mod32       = row_in_tile & 31;
            const int row_quartile    = row_in_tile >> 5;
            auto* destination =
                shared.b_scales[stage] + (row * Schedule::kK64PerStage + local_k64) * 4;
            const auto* input = scales +
                                static_cast<std::int64_t>(
                                    persistent_tile * Geometry::kScaleTilesPerRow + global_k64) *
                                    512 +
                                row_mod32 * 16 + row_quartile * 4;
            cp_async<4>(destination, input);
        }
    }
}

template <class Geometry, class Schedule, class Epilogue, class OutputPolicy,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false,
          Cache WeightCache = Cache::cg>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_w4a4_mma_kernel(
    Nvfp4W4a4MaterializedActivation activation, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales, std::int32_t tokens, float alpha,
    Epilogue epilogue, OutputPolicy output, RowPolicy row_policy = {}) {
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);
    static_assert(!PairRows || (Schedule::kBlockN % 2) == 0);
    static_assert(!PairRows || ((Geometry::kOutputRows / 2) % (Schedule::kBlockN / 2)) == 0);

    __shared__ Nvfp4W4a4SharedStorage<Schedule> shared;
    int block_x = 0;
    int block_y = 0;
    nvfp4_raster_token_tiles_fastest(block_x, block_y);
    const int token_begin       = block_y * Schedule::kBlockM;
    constexpr int kRowsPerBlock = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    const int row_begin         = block_x * kRowsPerBlock;
    constexpr int kKTiles       = Geometry::kInputRows / Schedule::kBlockK;
    constexpr int kWaitGroups   = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;

    // Completion precedes reads and writes because caller-owned workspace can reuse an address
    // still read by the predecessor.
    if constexpr (Schedule::kProgrammaticDependency) { pdl::wait_for_dependencies(); }
#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_nvfp4_w4a4_activation<Geometry, Schedule>(activation, shared, stage, stage,
                                                            token_begin, tokens);
            stage_nvfp4_w4a4_weight<Geometry, Schedule, RowPolicy, WeightCache>(
                weight_codes, weight_scales, shared, stage, stage, row_begin, row_policy);
            cp_commit();
        }
    }

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    const int lane                                          = static_cast<int>(threadIdx.x) & 31;
    const int warp                                          = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m                                        = warp / Schedule::kWarpsN;
    const int warp_n                                        = warp - warp_m * Schedule::kWarpsN;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    const int active_m = nvfp4_active_token_fragments<Schedule::kMmaM>(
        token_begin + warp_m * Schedule::kWarpM, tokens);
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kStages;
        cp_wait<kWaitGroups>();
        __syncthreads();

        // The barriers stay outside the per-warp fragment dispatch, so every thread executes the
        // same aligned __syncthreads.
        nvfp4_with_active_token_fragments<Schedule::kMmaM>(active_m, [&]<int kActiveM>() {
#pragma unroll
            for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
                unsigned a_fragments[Schedule::kMmaM][4];
                unsigned b_fragments[Schedule::kMmaN][2];
                unsigned a_scales[Schedule::kMmaM];
                unsigned b_scales[Schedule::kMmaN];

#pragma unroll
                for (int mma_m = 0; mma_m < kActiveM; ++mma_m) {
                    const int row           = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                    const int logical_byte  = local_k64 * 32 + a_column_byte;
                    const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                    const auto* address =
                        shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                                a_fragments[mma_m][3], smem_addr(address));
                    const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                    a_scales[mma_m] =
                        shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
                }

#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                    const int row           = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                    const int logical_byte  = local_k64 * 32 + b_column_byte;
                    const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                    const auto* address =
                        shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                    ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                    const int scale_row = warp_n * Schedule::kWarpN + mma_n * 8 + sfb_row;
                    const std::uint8_t* scale_address;
                    if constexpr (RowPolicy::kContiguous) {
                        constexpr int kQuartilesPerTile =
                            Schedule::kBlockN >= 128 ? 4 : Schedule::kBlockN / 32;
                        constexpr int kScaleBytesPerTask = kQuartilesPerTile * 4;
                        const int row_tile               = scale_row / 128;
                        const int row_in_tile            = scale_row & 127;
                        const int row_mod32              = row_in_tile & 31;
                        const int row_quartile           = row_in_tile >> 5;
                        scale_address =
                            shared.b_scales[stage] +
                            ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) *
                                kScaleBytesPerTask +
                            row_quartile * 4;
                    } else {
                        scale_address = shared.b_scales[stage] +
                                        (scale_row * Schedule::kK64PerStage + local_k64) * 4;
                    }
                    b_scales[mma_n] = load_vec<unsigned>(scale_address);
                }

#pragma unroll
                for (int mma_m = 0; mma_m < kActiveM; ++mma_m) {
#pragma unroll
                    for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                        mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                       accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                       a_fragments[mma_m][0], a_fragments[mma_m][1],
                                       a_fragments[mma_m][2], a_fragments[mma_m][3],
                                       b_fragments[mma_n][0], b_fragments[mma_n][1],
                                       a_scales[mma_m], b_scales[mma_n]);
                    }
                }
            }
        });

        __syncthreads();
        const int next_k_tile = k_tile + Schedule::kStages;
        if (next_k_tile < kKTiles) {
            stage_nvfp4_w4a4_activation<Geometry, Schedule>(activation, shared, stage, next_k_tile,
                                                            token_begin, tokens);
            stage_nvfp4_w4a4_weight<Geometry, Schedule, RowPolicy, WeightCache>(
                weight_codes, weight_scales, shared, stage, next_k_tile, row_begin, row_policy);
        }
        cp_commit();
    }

    // Start dependent launch setup at the epilogue so waiting CTAs do not occupy resources
    // during the weight-streaming reduction. Consumers wait for all output stores to finish.
    if constexpr (Schedule::kProgrammaticDependency) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }
    const int accumulator_row = lane >> 2;
    const int accumulator_col = 2 * (lane & 3);
    // Fused GDN convolution consumes FP32 projection accumulators. Its output policy
    // bypasses the BF16 shared epilogue; all existing BF16 policies retain that path.
    constexpr bool kFp32Output = requires { output.store_fp32(0, 0, 0.0F); };
    static_assert(!kFp32Output || !PairRows);
    constexpr int kOutputStride = Schedule::kBlockN + 8;
    static_assert(sizeof(Nvfp4W4a4SharedStorage<Schedule>) >=
                  Schedule::kBlockM * kOutputStride * sizeof(__nv_bfloat16));
    auto* shared_output = reinterpret_cast<__nv_bfloat16*>(&shared);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = token_begin + warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
            const int local_row0  = warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_col;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            auto* destination0    = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token0 - token_begin) * kOutputStride + local_row0);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token1 - token_begin) * kOutputStride + local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);
            float value00         = accumulators[mma_m][mma_n][0] * alpha;
            float value01         = accumulators[mma_m][mma_n][1] * alpha;
            float value10         = accumulators[mma_m][mma_n][2] * alpha;
            float value11         = accumulators[mma_m][mma_n][3] * alpha;
            if (token0 < tokens) {
                value00 = epilogue.apply(parent_row0, token0, value00);
                value01 = epilogue.apply(parent_row1, token0, value01);
            }
            if (token1 < tokens) {
                value10 = epilogue.apply(parent_row0, token1, value10);
                value11 = epilogue.apply(parent_row1, token1, value11);
            }
            if constexpr (kFp32Output) {
                if (token0 < tokens) {
                    output.store_fp32(parent_row0, token0, value00);
                    output.store_fp32(parent_row1, token0, value01);
                }
                if (token1 < tokens) {
                    output.store_fp32(parent_row0, token1, value10);
                    output.store_fp32(parent_row1, token1, value11);
                }
            } else {
                *destination0 = __floats2bfloat162_rn(value00, value01);
                *destination1 = __floats2bfloat162_rn(value10, value11);
            }
        }
    }
    // An output policy with kGroupRows = 16 publishes whole 16-row groups (an NVFP4 activation of
    // the paired output, ninfer/ops/a4_activation.h) and writes the zero scales of the last
    // fragment's padding tokens.
    constexpr bool kGroupOutput = requires { OutputPolicy::kGroupRows; };
    if constexpr (kGroupOutput) {
        __syncthreads();
        static_assert(PairRows && OutputPolicy::kGroupRows == 16);
        constexpr int kStoredRows   = Schedule::kBlockN / 2;
        constexpr int kGroupsPerRow = kStoredRows / 16;
        static_assert(kStoredRows % 16 == 0);
        const int fragment_end = (tokens + 15) / 16 * 16;
        for (int task = static_cast<int>(threadIdx.x); task < Schedule::kBlockM * kGroupsPerRow;
             task += Schedule::kThreads) {
            const int token_local = task / kGroupsPerRow;
            const int group_local = task - token_local * kGroupsPerRow;
            const int token       = token_begin + token_local;
            const int row         = row_begin + group_local * 16;
            if (token >= tokens) {
                if (token < fragment_end) { output.store_padding_group(row, token); }
                continue;
            }
            const __nv_bfloat16* gate =
                shared_output + token_local * kOutputStride + group_local * 16;
            const __nv_bfloat16* up = gate + kStoredRows;
            output.store_pair_group(row, token, load_vec<uint4>(gate), load_vec<uint4>(gate + 8),
                                    load_vec<uint4>(up), load_vec<uint4>(up + 8));
        }
    } else if constexpr (!kFp32Output) {
        __syncthreads();
        constexpr int kStoredRows    = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
        constexpr int kVectorsPerRow = kStoredRows / 8;
        constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
        for (int task = static_cast<int>(threadIdx.x); task < kOutputVectors;
             task += Schedule::kThreads) {
            const int token_local = task / kVectorsPerRow;
            const int row_vector  = task - token_local * kVectorsPerRow;
            const int token       = token_begin + token_local;
            if (token < tokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
                if constexpr (PairRows) {
                    const uint4 paired = load_vec<uint4>(
                        shared_output + token_local * kOutputStride + kStoredRows + row_vector * 8);
                    output.store_pair_vector(row_begin + row_vector * 8, token, values, paired);
                } else {
                    output.store_vector(row_begin + row_vector * 8, token, values);
                }
            }
        }
    }
}

// Applies the schedule's measured stream-ordering profile to every projection epilogue. The
// ordinary route has no grid-dependency instructions and requires full predecessor completion.
// The caller supplies a validated grid and keeps buffers allocated until stream completion.
// CUDA launch failures propagate through cuda_check.
template <class Geometry, class Schedule, class Epilogue, class OutputPolicy,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false,
          Cache WeightCache = Cache::cg>
void launch_nvfp4_w4a4_mma(dim3 grid, cudaStream_t stream,
                           Nvfp4W4a4MaterializedActivation activation,
                           const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                           std::int32_t tokens, float alpha, Epilogue epilogue, OutputPolicy output,
                           RowPolicy row_policy = {}) {
    constexpr auto kernel = nvfp4_w4a4_mma_kernel<Geometry, Schedule, Epilogue, OutputPolicy,
                                                  RowPolicy, PairRows, WeightCache>;
    if constexpr (Schedule::kProgrammaticDependency) {
        CUDA_CHECK(pdl::launch_dependent(
            pdl::LaunchConfig{grid, dim3(Schedule::kThreads), 0, stream}, kernel, activation,
            weight_codes, weight_scales, tokens, alpha, epilogue, output, row_policy));
    } else {
        cudaLaunchConfig_t launch{};
        launch.gridDim  = grid;
        launch.blockDim = dim3(Schedule::kThreads);
        launch.stream   = stream;
        CUDA_CHECK(cudaLaunchKernelEx(&launch, kernel, activation, weight_codes, weight_scales,
                                      tokens, alpha, epilogue, output, row_policy));
    }
}

// One thread per (token, scale group) of the first round_up(tokens, 16) tokens. Threads of real
// tokens quantize 16 BF16 values to eight code bytes and one tiled-plane scale; threads of the
// last fragment's padding tokens write that token's zero scale, the only padding a TMA box loads
// (nvfp4_tma_loaded_rows). Codes need no padding: TMA zero-fills code rows past `tokens`.
template <class Geometry, int Threads>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_w4a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, float input_scale_divisor) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    static_assert((kGroupsPerRow % kNvfp4ScaleTileGroups) == 0);
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int token = task / kGroupsPerRow;
    const int group = task - token * kGroupsPerRow;
    // Padding writes share the same lifetime dependency as represented input reads.
    pdl::wait_for_dependencies();
    if (token >= tokens) {
        if (token < (tokens + 15) / 16 * 16) {
            scales[nvfp4_tiled_scale_offset<Geometry>(token, group)] = 0;
        }
        return;
    }

    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * Geometry::kInputRows + group * 16,
        input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[nvfp4_tiled_scale_offset<Geometry>(token, group)] = quantized.scale;
}

} // namespace ninfer::ops::detail

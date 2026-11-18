#include "ops/launcher/moe_expert_partition.h"

#include "core/device.h"
#include "ninfer/ops/moe_expert_record.h"
#include "ninfer/ops/moe_helper_contract.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kPartitionThreads = 256;
constexpr int kRoutes           = kMoeHelperRoutesPerToken;
constexpr int kRowUnits         = kMoeHidden / 8; // 16-byte units of one BF16 activation row
static_assert(kMoeHelperMaxTokens <= 2 * kWarpSize);

// The id moe_route promises ([0,512)), clamped so a violation still indexes the bitset.
__device__ __forceinline__ int clamped_expert(int id) {
    return min(max(id, 0), kMoeHelperExperts - 1);
}

__device__ __forceinline__ bool is_local(const std::uint32_t* local_experts, int expert) {
    return ((local_experts[expert / 32] >> (expert % 32)) & 1U) != 0;
}

// Grid 1, block 256. Phase 1: thread t < T (warps 0 and 1) decides whether token t participates;
// the two warp ballots, published in shared memory, give t its compact row (participants of
// lower t). It writes compact_map[t] and the inverse source[row] = t; thread 0 writes the count.
// Phase 2, after a block barrier: the block strides over the N compact rows' 320 16-byte
// activation units (each a coalesced 16-byte copy) and their 10 route entries.
__global__ void __launch_bounds__(kPartitionThreads) moe_expert_partition_kernel(
    const std::uint16_t* __restrict__ x, const std::int32_t* __restrict__ expert_ids,
    const float* __restrict__ route_weights, const std::uint32_t* __restrict__ local_experts,
    std::int32_t* __restrict__ compact_map, std::uint32_t* __restrict__ compact_count,
    std::uint16_t* __restrict__ compact_x, std::int32_t* __restrict__ compact_ids,
    float* __restrict__ compact_weights, int tokens) {
    __shared__ unsigned ballots[2];
    __shared__ int source[kMoeHelperMaxTokens];
    const int thread = static_cast<int>(threadIdx.x);
    const int lane   = thread % kWarpSize;
    const int warp   = thread / kWarpSize;

    bool participates = false;
    if (thread < tokens) {
#pragma unroll
        for (int j = 0; j < kRoutes; ++j) {
            participates |=
                !is_local(local_experts, clamped_expert(expert_ids[thread * kRoutes + j]));
        }
    }
    if (warp < 2) {
        const unsigned ballot = __ballot_sync(kFullWarpMask, participates);
        if (lane == 0) { ballots[warp] = ballot; }
    }
    __syncthreads();
    if (thread < tokens) {
        const unsigned below = (1U << lane) - 1U;
        const int row        = __popc(ballots[warp] & below) + (warp == 1 ? __popc(ballots[0]) : 0);
        compact_map[thread]  = participates ? row : -1;
        if (participates) { source[row] = thread; }
    }
    const int count = __popc(ballots[0]) + __popc(ballots[1]);
    if (thread == 0) { *compact_count = static_cast<std::uint32_t>(count); }
    __syncthreads();

    for (int unit = thread; unit < count * kRowUnits; unit += kPartitionThreads) {
        const int row    = unit / kRowUnits;
        const int offset = 8 * (unit % kRowUnits);
        store_vec(
            compact_x + static_cast<std::int64_t>(row) * kMoeHidden + offset,
            load_vec<uint4>(x + static_cast<std::int64_t>(source[row]) * kMoeHidden + offset));
    }
    for (int entry = thread; entry < count * kRoutes; entry += kPartitionThreads) {
        const int row          = entry / kRoutes;
        const int from         = source[row] * kRoutes + entry % kRoutes;
        const int expert       = clamped_expert(expert_ids[from]);
        const bool helper      = !is_local(local_experts, expert);
        compact_ids[entry]     = helper ? expert : kMoeHelperInactiveExpert;
        compact_weights[entry] = helper ? route_weights[from] : 0.0F;
    }
}

constexpr int kJoinThreads      = 128;
constexpr int kJoinRowsPerBlock = kJoinThreads * 4;
static_assert(kMoeHidden % kJoinRowsPerBlock == 0);

// Grid (2560 / 512, T), block 128: thread i of block (b, t) adds rows 512 b + 4 i .. +3 of token
// t's helper column (when it participates) to its local partial in FP32 and rounds each sum once
// to BF16 (float4 loads, 8-byte stores).
__global__ void __launch_bounds__(kJoinThreads)
    moe_expert_partial_join_kernel(const float* __restrict__ partial,
                                   const std::int32_t* __restrict__ compact_map,
                                   const float* __restrict__ helper, int helper_columns,
                                   std::uint16_t* __restrict__ y) {
    const int token = static_cast<int>(blockIdx.y);
    const int row =
        static_cast<int>(blockIdx.x) * kJoinRowsPerBlock + static_cast<int>(threadIdx.x) * 4;
    float4 sum = load_vec<float4>(partial + static_cast<std::int64_t>(token) * kMoeHidden + row);
    // A violation outside [-1, helper_columns) is clamped into it.
    const int column = min(max(compact_map[token], -1), helper_columns - 1);
    if (column >= 0) {
        const float4 remote =
            load_vec<float4>(helper + static_cast<std::int64_t>(column) * kMoeHidden + row);
        sum.x += remote.x;
        sum.y += remote.y;
        sum.z += remote.z;
        sum.w += remote.w;
    }
    const __nv_bfloat162 low  = __floats2bfloat162_rn(sum.x, sum.y);
    const __nv_bfloat162 high = __floats2bfloat162_rn(sum.z, sum.w);
    store_vec(y + static_cast<std::int64_t>(token) * kMoeHidden + row,
              make_uint2(*reinterpret_cast<const unsigned*>(&low),
                         *reinterpret_cast<const unsigned*>(&high)));
}

} // namespace

void moe_expert_partition_launch(const std::uint16_t* x, const std::int32_t* expert_ids,
                                 const float* route_weights, const std::uint32_t* local_experts,
                                 std::int32_t* compact_map, std::uint32_t* compact_count,
                                 std::uint16_t* compact_x, std::int32_t* compact_ids,
                                 float* compact_weights, std::int32_t tokens, cudaStream_t stream) {
    moe_expert_partition_kernel<<<1, kPartitionThreads, 0, stream>>>(
        x, expert_ids, route_weights, local_experts, compact_map, compact_count, compact_x,
        compact_ids, compact_weights, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_partial_join_launch(const float* partial, const std::int32_t* compact_map,
                                    const float* helper, std::int32_t helper_columns,
                                    std::uint16_t* y, std::int32_t tokens, cudaStream_t stream) {
    moe_expert_partial_join_kernel<<<dim3(kMoeHidden / kJoinRowsPerBlock,
                                          static_cast<unsigned>(tokens)),
                                     kJoinThreads, 0, stream>>>(partial, compact_map, helper,
                                                                helper_columns, y);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

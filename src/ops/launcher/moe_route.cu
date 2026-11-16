#include "ops/launcher/moe_route.h"

#include "core/device.h"
#include "ninfer/ops/moe_route.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarpsPerCta   = 4;
constexpr int kLogitsPerLane = kMoeExperts / kWarpSize;
static_assert(kMoeExperts % kWarpSize == 0);
static_assert(kMoeTopK <= kWarpSize);

// Whether candidate (value, id) ranks before (best_value, best_id): larger value first, the lower
// expert id among equal values.
__device__ __forceinline__ bool ranks_before(float value, int id, float best_value, int best_id) {
    return value > best_value || (value == best_value && id < best_id);
}

// Grid (ceil(T/4)), block 128: warp w routes token 4*blockIdx.x + w. Lane l holds logits
// l + 32 j (j = 0..15, coalesced loads). Each of the ten rounds takes the lane's best unselected
// logit, reduces (value, id) across the warp with the xor butterfly so every lane learns the
// winner, and the owning lane removes it (its id has lane id % 32 and slot id / 32). Lane k < 10
// then records round k's winner. The weights are exp(v_k - v_0) over their sum, v_0 being the
// maximum; lane 0 publishes the shared-expert scale. No shared memory or block barriers.
__global__ void __launch_bounds__(kWarpsPerCta* kWarpSize)
    moe_route_kernel(const float* __restrict__ logits, std::int32_t* __restrict__ expert_ids,
                     float* __restrict__ route_weights, float* __restrict__ shared_scale,
                     int tokens) {
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int token =
        static_cast<int>(blockIdx.x) * kWarpsPerCta + static_cast<int>(threadIdx.x) / kWarpSize;
    if (token >= tokens) { return; }
    const float* row = logits + static_cast<std::int64_t>(token) * kMoeControlRows;

    float values[kLogitsPerLane];
#pragma unroll
    for (int j = 0; j < kLogitsPerLane; ++j) { values[j] = row[j * kWarpSize + lane]; }

    float selected_value = 0.0F;
    int selected_id      = 0;
    float maximum        = 0.0F;
#pragma unroll 1
    for (int round = 0; round < kMoeTopK; ++round) {
        float best_value = -INFINITY;
        int best_id      = kMoeExperts;
#pragma unroll
        for (int j = 0; j < kLogitsPerLane; ++j) {
            const int id = j * kWarpSize + lane;
            if (ranks_before(values[j], id, best_value, best_id)) {
                best_value = values[j];
                best_id    = id;
            }
        }
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            const float other_value = __shfl_xor_sync(kFullWarpMask, best_value, offset);
            const int other_id      = __shfl_xor_sync(kFullWarpMask, best_id, offset);
            if (ranks_before(other_value, other_id, best_value, best_id)) {
                best_value = other_value;
                best_id    = other_id;
            }
        }
        if (round == 0) { maximum = best_value; }
        if (lane == best_id % kWarpSize) {
#pragma unroll
            for (int j = 0; j < kLogitsPerLane; ++j) {
                if (j == best_id / kWarpSize) { values[j] = -INFINITY; }
            }
        }
        if (lane == round) {
            selected_value = best_value;
            selected_id    = best_id;
        }
    }

    const float exponent = lane < kMoeTopK ? expf(selected_value - maximum) : 0.0F;
    const float total    = warp_sum(exponent);
    if (lane < kMoeTopK) {
        const std::int64_t out = static_cast<std::int64_t>(token) * kMoeTopK + lane;
        expert_ids[out]        = selected_id;
        route_weights[out]     = exponent / total;
    }
    if (lane == 0) { shared_scale[token] = sigmoid(row[kMoeSharedGateRow]); }
}

} // namespace

void moe_route_launch(const float* logits, std::int32_t* expert_ids, float* route_weights,
                      float* shared_scale, std::int32_t tokens, cudaStream_t stream) {
    const int blocks = (tokens + kWarpsPerCta - 1) / kWarpsPerCta;
    moe_route_kernel<<<blocks, kWarpsPerCta * kWarpSize, 0, stream>>>(
        logits, expert_ids, route_weights, shared_scale, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fp32_kernel(const float* __restrict__ q, const float* __restrict__ k,
                          const float* __restrict__ v, const float* __restrict__ g,
                          const float* __restrict__ beta, float* __restrict__ ssm_state,
                          float* __restrict__ out, std::int64_t T, head_map heads, float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);

    float* state_h = ssm_state + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float s_tile[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    __align__(16) float k_reg[kQkPerLane];
    load_qk_lane(k_reg, k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const float* v_t          = v + (t * heads.H_v + h_v) * kStateDim;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kDvPerWarp) { v_local = v_t[dv_base + lane]; }

#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            load_qk_lane(k_reg, k + ((t + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
        }

        __align__(16) float q_reg[kQkPerLane];
        load_qk_lane(q_reg, q + (t * heads.H_qk + h_qk) * kStateDim, dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kDvPerWarp) {
            out[(t * heads.H_v + h_v) * kStateDim + dv_base + lane] = attn_val * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                      dqk_base);
    }
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

template <bool NormalizeQK>
__device__ __forceinline__ void load_qk_lane_bf16(float (&reg)[kQkPerLane],
                                                  const __nv_bfloat16* base, int lane,
                                                  std::uint32_t dqk_base) {
    const Bf16x4Pack packed = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo         = bf16x2_to_float2(packed.pair[0]);
    const float2 hi         = bf16x2_to_float2(packed.pair[1]);
    reg[0]                  = lo.x;
    reg[1]                  = lo.y;
    reg[2]                  = hi.x;
    reg[3]                  = hi.y;

    if constexpr (NormalizeQK) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += reg[i] * reg[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { reg[i] *= inv; }
    }
}

// state_read / state_write are the read and write bases (fp32).
//   Spec (snapshot):    read from state_read[safe(*initial_slot)] slot, write per-token snapshots
//                       from *snapshot_base_slot.
//   non-spec:           read from state_read (caller-resolved view), write the final running state
//                       into state_write (slot-0 view). Passing state_read == state_write is the
//                       in-place form; distinct views let prefix-append prefill read a committed
//                       snapshot slot and publish the running state to slot 0.
template <bool Spec, bool NormalizeQK>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                          const float* __restrict__ beta, float* __restrict__ state_read,
                          float* __restrict__ state_write,
                          const std::int32_t* __restrict__ initial_slot,
                          const std::int32_t* __restrict__ snapshot_base_slot,
                          __nv_bfloat16* __restrict__ out, std::int64_t T, head_map heads,
                          float scale, std::int64_t state_slot_stride, std::int32_t slots) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);

    float* read_base = state_read;
    if constexpr (Spec) {
        const std::int32_t slot      = *initial_slot;
        const std::int32_t safe_slot = (slot >= 0 && slot < slots) ? slot : 0;
        read_base = state_read + static_cast<std::int64_t>(safe_slot) * state_slot_stride;
    }
    float* read_h = read_base + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float s_tile[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(s_tile[r], read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    __align__(16) float k_reg[kQkPerLane];
    load_qk_lane_bf16<NormalizeQK>(k_reg, k + static_cast<std::int64_t>(h_qk) * kStateDim, lane,
                                   dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const __nv_bfloat16* v_t  = v + (t * heads.H_v + h_v) * kStateDim;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kDvPerWarp) { v_local = __bfloat162float(v_t[dv_base + lane]); }

#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            load_qk_lane_bf16<NormalizeQK>(k_reg, k + ((t + 1) * heads.H_qk + h_qk) * kStateDim,
                                           lane, dqk_base);
        }

        __align__(16) float q_reg[kQkPerLane];
        load_qk_lane_bf16<NormalizeQK>(q_reg, q + (t * heads.H_qk + h_qk) * kStateDim, lane,
                                       dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kDvPerWarp) {
            out[(t * heads.H_v + h_v) * kStateDim + dv_base + lane] =
                __float2bfloat16(attn_val * scale);
        }

        if constexpr (Spec) {
            float* snapshot_h =
                state_write +
                static_cast<std::int64_t>(*snapshot_base_slot + t) * state_slot_stride +
                static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                store_qk_lane(s_tile[r],
                              snapshot_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                              dqk_base);
            }
        }
    }

    if constexpr (!Spec) {
        float* write_h = state_write + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(s_tile[r], write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                          dqk_base);
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net

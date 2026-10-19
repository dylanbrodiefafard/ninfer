#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {

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

// One CTA (single thread) per row compiles the packed tree into the step list documented on
// gated_delta_net_tree_schedule. It runs once per round; the serial walk over at most 16 columns
// is negligible beside the 48 record launches that reuse it.
__global__ void tree_schedule_kernel(const std::int32_t* parent_index,
                                     const std::int32_t* valid_columns, std::int32_t width,
                                     std::int32_t* schedule) {
    constexpr int kBranchSlots     = 3;
    constexpr int kResumeRoot      = 1;
    constexpr int kEmit            = 1 << 6;
    const int row                  = static_cast<int>(blockIdx.x);
    const std::int32_t* parent_row = parent_index + static_cast<std::int64_t>(row) * width;
    std::int32_t* out = schedule + static_cast<std::int64_t>(row) * kGdnTreeScheduleWords;
    int columns       = valid_columns != nullptr ? valid_columns[row] : width;
    columns           = columns < 1 ? 1 : (columns > width ? width : columns);
    int parents[kGdnTreeMaxColumns];
    int children[kGdnTreeMaxColumns];
    int slot_of[kGdnTreeMaxColumns];
    for (int c = 0; c < columns; ++c) {
        parents[c]  = parent_row[c];
        children[c] = 0;
        slot_of[c]  = -1;
    }
    for (int c = 1; c < columns; ++c) {
        if (parents[c] >= 0) { ++children[parents[c]]; }
    }
    unsigned free_slots = (1U << kBranchSlots) - 1U;
    int steps           = 0;
    for (int c = 0; c < columns; ++c) {
        const int parent = parents[c];
        int resume       = 0;
        if (parent < 0) {
            resume = kResumeRoot;
        } else if (parent != c - 1) {
            if (slot_of[parent] >= 0) {
                resume = 2 + slot_of[parent];
            } else {
                int path[kGdnTreeMaxColumns];
                int depth = 0;
                for (int n = parent; n >= 0 && depth < kGdnTreeMaxColumns; n = parents[n]) {
                    path[depth++] = n;
                }
                for (int i = depth - 1; i >= 0; --i) {
                    out[1 + steps++] = path[i] | ((i == depth - 1 ? kResumeRoot : 0) << 8);
                }
            }
        }
        if (parent >= 0 && --children[parent] == 0 && slot_of[parent] >= 0) {
            free_slots |= 1U << slot_of[parent];
        }
        int save = 0;
        if (children[c] >= 2 && free_slots != 0) {
            const int slot = __ffs(static_cast<int>(free_slots)) - 1;
            free_slots &= ~(1U << slot);
            slot_of[c] = slot;
            save       = 1 + slot;
        }
        out[1 + steps++] = c | ((resume | (save << 3) | kEmit) << 8);
    }
    out[0] = steps;
}

namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 80);
static_assert(alignof(GdnReplayFoldKernelRow) == 8);
static_assert(sizeof(GdnReplayFoldKernelRows) == 480);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

void launch_recurrent_fp32_fixed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                 const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                                 cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);

    recurrent_fp32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(k.data),
        static_cast<const float*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_state.data),
        static_cast<float*>(out.data), T, heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeQK>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_bf16_direct_kernel<NormalizeQK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const float*>(state_read.data),
        static_cast<float*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs, bool Batched, bool Masked>
void launch_recurrent_snapshot_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     Tensor& ssm_states, const Tensor& valid_columns,
                                     const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& out,
                                     cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), Batched ? static_cast<unsigned>(q.ne[3]) : 1U,
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const SnapshotAccess<Batched, Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_snapshot_kernel<NormalizeInputs, Batched, Masked><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, bool ParentIndexed, int NumWarps>
void launch_recurrent_record_warps(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots,
                                   const std::int32_t* tree_schedule, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    constexpr int kDv = NumWarps * kDvPerWarp;
    const auto heads  = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kDv));
    const dim3 block(kWarpSize, NumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked, ParentIndexed, NumWarps> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        tree_schedule,
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked, ParentIndexed, NumWarps><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, bool ParentIndexed>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots,
                                   const std::int32_t* tree_schedule, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    launch_recurrent_record_warps<Masked, ParentIndexed, kNumWarps>(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, tree_schedule,
        key_record, value_record, gate_record, out, stream);
}

template <class Geometry>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states,
                              const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                              cudaStream_t stream) {
    const FoldAccess<Geometry> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                           const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                           cudaStream_t stream) {
    launch_recurrent_fp32_fixed(q, k, v, g, beta, scale, ssm_state, out, stream);
}

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_direct_fixed<true>(q, k, v, g, beta, scale, ssm_state, ssm_state, out,
                                            stream);
    } else {
        launch_recurrent_direct_fixed<false>(q, k, v, g, beta, scale, ssm_state, ssm_state, out,
                                             stream);
    }
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_direct_fixed<true>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                            out, stream);
    } else {
        launch_recurrent_direct_fixed<false>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                             out, stream);
    }
}

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                               Tensor& out, cudaStream_t stream) {
    const bool dense_single = q.ne[3] == 1 && valid_columns.data == nullptr;
    if (dense_single && normalize_qk) {
        launch_recurrent_snapshot_fixed<true, false, false>(q, k, v, g, beta, scale, ssm_states,
                                                            valid_columns, initial_state_slots,
                                                            snapshot_base_slots, out, stream);
    } else if (dense_single) {
        launch_recurrent_snapshot_fixed<false, false, false>(q, k, v, g, beta, scale, ssm_states,
                                                             valid_columns, initial_state_slots,
                                                             snapshot_base_slots, out, stream);
    } else if (valid_columns.data == nullptr && normalize_qk) {
        launch_recurrent_snapshot_fixed<true, true, false>(q, k, v, g, beta, scale, ssm_states,
                                                           valid_columns, initial_state_slots,
                                                           snapshot_base_slots, out, stream);
    } else if (valid_columns.data == nullptr) {
        launch_recurrent_snapshot_fixed<false, true, false>(q, k, v, g, beta, scale, ssm_states,
                                                            valid_columns, initial_state_slots,
                                                            snapshot_base_slots, out, stream);
    } else if (normalize_qk) {
        launch_recurrent_snapshot_fixed<true, true, true>(q, k, v, g, beta, scale, ssm_states,
                                                          valid_columns, initial_state_slots,
                                                          snapshot_base_slots, out, stream);
    } else {
        launch_recurrent_snapshot_fixed<false, true, true>(q, k, v, g, beta, scale, ssm_states,
                                                           valid_columns, initial_state_slots,
                                                           snapshot_base_slots, out, stream);
    }
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream, const std::int32_t* tree_schedule) {
    const bool masked = valid_columns.data != nullptr;
    if (tree_schedule == nullptr) {
        if (!masked) {
            launch_recurrent_record_fixed<false, false>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                tree_schedule, key_record, value_record, gate_record, out, stream);
        } else {
            launch_recurrent_record_fixed<true, false>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                tree_schedule, key_record, value_record, gate_record, out, stream);
        }
        return;
    }
    if (!masked) {
        launch_recurrent_record_fixed<false, true>(
            q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, tree_schedule,
            key_record, value_record, gate_record, out, stream);
    } else {
        launch_recurrent_record_fixed<true, true>(
            q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, tree_schedule,
            key_record, value_record, gate_record, out, stream);
    }
}

void launch_tree_schedule(const std::int32_t* parent_index, const std::int32_t* valid_columns,
                          std::int32_t width, std::int32_t batch, std::int32_t* schedule,
                          cudaStream_t stream) {
    tree_schedule_kernel<<<static_cast<unsigned>(batch), 1, 0, stream>>>(
        parent_index, valid_columns, width, schedule);
    CUDA_CHECK(cudaGetLastError());
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry48x48>(records, states, rows, active_rows, stream);
        return;
    }
    if (records.spec.layers == FoldGeometry30x32::kLayers &&
        records.spec.qk_heads == FoldGeometry30x32::kQkHeads &&
        records.spec.value_heads == FoldGeometry30x32::kValueHeads &&
        records.spec.conv_channels == FoldGeometry30x32::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry30x32>(records, states, rows, active_rows, stream);
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace ninfer::ops::detail::gated_delta_net

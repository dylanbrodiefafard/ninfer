#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 80);
static_assert(alignof(GdnReplayFoldKernelRow) == 8);
static_assert(sizeof(GdnReplayFoldKernelRows) == 640);
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
                                   const std::int32_t* parent_index, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream, float* column_scratch) {
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
        parent_index,
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
        column_scratch,
    };
    if constexpr (ParentIndexed && NumWarps == kTreeNumWarps) {
        const std::size_t smem =
            static_cast<std::size_t>(q.ne[2]) * kDv * kStateDim * sizeof(float);
        recurrent_record_kernel<Masked, true, NumWarps><<<grid, block, smem, stream>>>(access);
    } else {
        recurrent_record_kernel<Masked, ParentIndexed, NumWarps>
            <<<grid, block, 0, stream>>>(access);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, bool ParentIndexed>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots,
                                   const std::int32_t* parent_index, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream, float* column_scratch) {
    if constexpr (ParentIndexed) {
        if (column_scratch != nullptr) {
            launch_recurrent_record_warps<Masked, true, kNumWarps>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                parent_index, key_record, value_record, gate_record, out, stream, column_scratch);
        } else {
            launch_recurrent_record_warps<Masked, true, kTreeNumWarps>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                parent_index, key_record, value_record, gate_record, out, stream, nullptr);
        }
    } else {
        launch_recurrent_record_warps<Masked, false, kNumWarps>(
            q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, parent_index,
            key_record, value_record, gate_record, out, stream, nullptr);
    }
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
                             Tensor& out, cudaStream_t stream, const std::int32_t* parent_index,
                             float* column_scratch) {
    const bool masked = valid_columns.data != nullptr;
    if (parent_index == nullptr) {
        if (!masked) {
            launch_recurrent_record_fixed<false, false>(q, k, v, g, beta, scale, ssm_states,
                                                        valid_columns, initial_state_slots,
                                                        parent_index, key_record, value_record,
                                                        gate_record, out, stream, nullptr);
        } else {
            launch_recurrent_record_fixed<true, false>(q, k, v, g, beta, scale, ssm_states,
                                                       valid_columns, initial_state_slots,
                                                       parent_index, key_record, value_record,
                                                       gate_record, out, stream, nullptr);
        }
        return;
    }
    if (!masked) {
        launch_recurrent_record_fixed<false, true>(q, k, v, g, beta, scale, ssm_states,
                                                   valid_columns, initial_state_slots, parent_index,
                                                   key_record, value_record, gate_record, out,
                                                   stream, column_scratch);
    } else {
        launch_recurrent_record_fixed<true, true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                                  initial_state_slots, parent_index, key_record,
                                                  value_record, gate_record, out, stream,
                                                  column_scratch);
    }
}

template <bool Masked, bool ParentIndexed>
void launch_recurrent_overlay_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, float scale,
                                    const Tensor& ssm_states, const Tensor& valid_columns,
                                    const Tensor& initial_state_slots, Tensor& out,
                                    float* overlay_states, const std::int32_t* parent_index,
                                    cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t live_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const std::int64_t overlay_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * v.ne[1];
    const OverlayAccess<Masked, ParentIndexed> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        overlay_states,
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        parent_index,
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        q.ne[3],
        live_slot_stride,
        overlay_slot_stride,
        scale,
    };
    recurrent_overlay_kernel<Masked, ParentIndexed><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_overlay(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, const Tensor& ssm_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              Tensor& out, float* overlay_states, const std::int32_t* parent_index,
                              cudaStream_t stream) {
    const bool masked = valid_columns.data != nullptr;
    if (parent_index == nullptr) {
        if (!masked) {
            launch_recurrent_overlay_fixed<false, false>(q, k, v, g, beta, scale, ssm_states,
                                                         valid_columns, initial_state_slots, out,
                                                         overlay_states, parent_index, stream);
        } else {
            launch_recurrent_overlay_fixed<true, false>(q, k, v, g, beta, scale, ssm_states,
                                                        valid_columns, initial_state_slots, out,
                                                        overlay_states, parent_index, stream);
        }
        return;
    }
    if (!masked) {
        launch_recurrent_overlay_fixed<false, true>(q, k, v, g, beta, scale, ssm_states,
                                                    valid_columns, initial_state_slots, out,
                                                    overlay_states, parent_index, stream);
    } else {
        launch_recurrent_overlay_fixed<true, true>(q, k, v, g, beta, scale, ssm_states,
                                                   valid_columns, initial_state_slots, out,
                                                   overlay_states, parent_index, stream);
    }
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

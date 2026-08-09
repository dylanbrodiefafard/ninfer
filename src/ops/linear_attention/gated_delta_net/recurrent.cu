#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

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

template <bool Snapshot, bool NormalizeQK, bool Batched = false, bool Masked = false>
void launch_recurrent_fixed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& state_read,
                            Tensor& state_write, const Tensor* valid_columns,
                            const Tensor* initial_state_slots, const Tensor* snapshot_base_slots,
                            Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), Batched ? static_cast<unsigned>(q.ne[3]) : 1U,
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * state_write.ne[2];
    const auto* valid =
        valid_columns == nullptr ? nullptr : static_cast<const std::int32_t*>(valid_columns->data);
    const auto* initial       = initial_state_slots == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(initial_state_slots->data);
    const auto* snapshot_base = snapshot_base_slots == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(snapshot_base_slots->data);

    recurrent_bf16_kernel<Snapshot, NormalizeQK, Batched, Masked><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(state_read.data),
        static_cast<float*>(state_write.data), valid, initial, snapshot_base,
        static_cast<__nv_bfloat16*>(out.data), T, heads, scale, state_slot_stride);
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
        launch_recurrent_fixed<false, true>(q, k, v, g, beta, scale, ssm_state, ssm_state, nullptr,
                                            nullptr, nullptr, out, stream);
    } else {
        launch_recurrent_fixed<false, false>(q, k, v, g, beta, scale, ssm_state, ssm_state, nullptr,
                                             nullptr, nullptr, out, stream);
    }
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_fixed<false, true>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                            nullptr, nullptr, nullptr, out, stream);
    } else {
        launch_recurrent_fixed<false, false>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                             nullptr, nullptr, nullptr, out, stream);
    }
}

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                               Tensor& out, cudaStream_t stream) {
    const bool dense_single = q.ne[3] == 1 && valid_columns.data == nullptr;
    if (dense_single && normalize_qk) {
        launch_recurrent_fixed<true, true>(q, k, v, g, beta, scale, ssm_states, ssm_states, nullptr,
                                           &initial_state_slots, &snapshot_base_slots, out, stream);
    } else if (dense_single) {
        launch_recurrent_fixed<true, false>(q, k, v, g, beta, scale, ssm_states, ssm_states,
                                            nullptr, &initial_state_slots, &snapshot_base_slots,
                                            out, stream);
    } else if (valid_columns.data == nullptr && normalize_qk) {
        launch_recurrent_fixed<true, true, true, false>(q, k, v, g, beta, scale, ssm_states,
                                                        ssm_states, nullptr, &initial_state_slots,
                                                        &snapshot_base_slots, out, stream);
    } else if (valid_columns.data == nullptr) {
        launch_recurrent_fixed<true, false, true, false>(q, k, v, g, beta, scale, ssm_states,
                                                         ssm_states, nullptr, &initial_state_slots,
                                                         &snapshot_base_slots, out, stream);
    } else if (normalize_qk) {
        launch_recurrent_fixed<true, true, true, true>(
            q, k, v, g, beta, scale, ssm_states, ssm_states, &valid_columns, &initial_state_slots,
            &snapshot_base_slots, out, stream);
    } else {
        launch_recurrent_fixed<true, false, true, true>(
            q, k, v, g, beta, scale, ssm_states, ssm_states, &valid_columns, &initial_state_slots,
            &snapshot_base_slots, out, stream);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net

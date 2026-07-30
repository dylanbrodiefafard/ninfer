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

template <bool Snapshot, bool NormalizeQK>
void launch_recurrent_fixed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& state_read,
                            Tensor& state_write, const Tensor* initial_slot, std::int32_t slots,
                            Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * state_write.ne[2];
    const auto* initial =
        initial_slot == nullptr ? nullptr : static_cast<const std::int32_t*>(initial_slot->data);

    recurrent_bf16_kernel<Snapshot, NormalizeQK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(state_read.data),
        static_cast<float*>(state_write.data), initial, static_cast<__nv_bfloat16*>(out.data), T,
        heads, scale, state_slot_stride, slots);
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
                                            1, out, stream);
    } else {
        launch_recurrent_fixed<false, false>(q, k, v, g, beta, scale, ssm_state, ssm_state, nullptr,
                                             1, out, stream);
    }
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_fixed<false, true>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                            nullptr, 1, out, stream);
    } else {
        launch_recurrent_fixed<false, false>(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out,
                                             nullptr, 1, out, stream);
    }
}

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t slots = ssm_states.ne[3];
    if (normalize_qk) {
        launch_recurrent_fixed<true, true>(q, k, v, g, beta, scale, ssm_states, ssm_states,
                                           &initial_slot, slots, out, stream);
    } else {
        launch_recurrent_fixed<true, false>(q, k, v, g, beta, scale, ssm_states, ssm_states,
                                            &initial_slot, slots, out, stream);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net

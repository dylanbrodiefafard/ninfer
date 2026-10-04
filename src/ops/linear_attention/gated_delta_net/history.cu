#include "ops/linear_attention/gated_delta_net/history.cuh"
#include "core/device.h"
#include "core/gdn_history.h"

namespace ninfer::ops::detail::gated_delta_net {
namespace {
HistoryPlanes planes(const GdnHistoryLayer& history) {
    return {static_cast<float*>(history.key.data),
            static_cast<float*>(history.innovation.data),
            static_cast<float*>(history.alpha.data),
            static_cast<std::int32_t*>(history.counts.data),
            static_cast<float*>(history.provisional_key.data),
            static_cast<float*>(history.provisional_innovation.data),
            static_cast<float*>(history.provisional_alpha.data),
            history.counts.ne[0],
            history.provisional_key.ne[2]};
}

HistoryPlanes planes(const GdnHistory& history) {
    return {static_cast<float*>(history.key.data),
            static_cast<float*>(history.innovation.data),
            static_cast<float*>(history.alpha.data),
            static_cast<std::int32_t*>(history.counts.data),
            static_cast<float*>(history.provisional_key.data),
            static_cast<float*>(history.provisional_innovation.data),
            static_cast<float*>(history.provisional_alpha.data),
            history.spec.slots,
            history.spec.width};
}

template <bool Masked, bool ParentIndexed>
void record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
            float scale, const Tensor& states, const Tensor& valid, const Tensor& slots,
            Tensor& key_record, Tensor& value_record, Tensor& gate_record, Tensor& out,
            const GdnHistoryLayer& history, cudaStream_t stream, const std::int32_t* schedule) {
    const RecordAccess<Masked, ParentIndexed, kNumWarps> base{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(states.data),
        static_cast<const std::int32_t*>(valid.data),
        static_cast<const std::int32_t*>(slots.data),
        schedule,
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        head_map::of(q.ne[1], v.ne[1]),
        q.ne[2],
        static_cast<std::int64_t>(128) * 128 * 48,
        scale};
    const HistoryRecordAccess<Masked, ParentIndexed> access{base, planes(history)};
    history_record_kernel<Masked, ParentIndexed>
        <<<dim3(48, static_cast<unsigned>(q.ne[3]), 8), dim3(32, 4), 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_history_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                           const Tensor& beta, float scale, const Tensor& states,
                           const Tensor& valid, const Tensor& slots, Tensor& key_record,
                           Tensor& value_record, Tensor& gate_record, Tensor& out,
                           const GdnHistoryLayer& history, cudaStream_t stream,
                           const std::int32_t* schedule) {
    if (schedule != nullptr) {
        if (valid.data != nullptr) {
            record<true, true>(q, k, v, g, beta, scale, states, valid, slots, key_record,
                               value_record, gate_record, out, history, stream, schedule);
        } else {
            record<false, true>(q, k, v, g, beta, scale, states, valid, slots, key_record,
                                value_record, gate_record, out, history, stream, schedule);
        }
    } else {
        if (valid.data != nullptr) {
            record<true, false>(q, k, v, g, beta, scale, states, valid, slots, key_record,
                                value_record, gate_record, out, history, stream, nullptr);
        } else {
            record<false, false>(q, k, v, g, beta, scale, states, valid, slots, key_record,
                                 value_record, gate_record, out, history, stream, nullptr);
        }
    }
}

void launch_history_commit(const GdnReplayRecords& records, const GdnHistory& history,
                           LinearAttentionStateAllLayersView states,
                           const GdnReplayFoldKernelRows& rows, std::int32_t batch,
                           bool force_flush, cudaStream_t stream) {
    const HistoryFold fold{
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
        rows};
    const auto storage = planes(history);
    history_commit_kernel<<<dim3(48, static_cast<unsigned>(batch), 48 * 8), dim3(32, 4), 0,
                            stream>>>(fold, storage, force_flush);
    CUDA_CHECK(cudaGetLastError());
    history_count_kernel<<<1, 32, 0, stream>>>(storage, rows, batch, force_flush);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::gated_delta_net

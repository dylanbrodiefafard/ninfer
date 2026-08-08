#include <ninfer/targets/qwen3_6/decoder_state.h>

namespace ninfer::targets::qwen3_6 {

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv =
        plan_kv_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                      spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_kv_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                      spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

KVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const KVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6

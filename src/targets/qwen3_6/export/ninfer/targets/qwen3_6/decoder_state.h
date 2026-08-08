#pragma once

#include "core/kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/layout.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

struct DecoderStateSpec {
    std::uint32_t full_attention_layers = 0;
    std::uint32_t mtp_layers            = 0;
    std::uint32_t capacity              = 0;
    std::int32_t kv_heads               = 0;
    std::int32_t attention_head_dim     = 0;
    DType kv_dtype                      = DType::BF16;
    std::int32_t kv_quant_group         = 0;
    bool enable_mtp                     = false;
    LinearAttentionStatePoolSpec linear_attention;
};

struct DecoderStateLayout {
    KVCacheLayout text_kv;
    std::optional<KVCacheLayout> mtp_kv;
    LinearAttentionStatePoolLayout linear_attention;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    KVCache text_kv;
    std::optional<KVCache> mtp_kv;
    LinearAttentionStatePool linear_attention;

    DecoderState() = default;
    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] KVCache* mtp_cache() noexcept;
    [[nodiscard]] const KVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6

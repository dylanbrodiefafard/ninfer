#pragma once

#ifndef NINFER_QWEN36_VARIANT
#    error "NINFER_QWEN36_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN36_RUNTIME_NS
#    error "NINFER_QWEN36_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/targets/qwen3_6/runtime.h>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using Variant                        = NINFER_QWEN36_VARIANT;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan    = qwen3_6::SequencePlan<Variant>;
using SequencePlanner = qwen3_6::SequencePlanner<Variant>;
using RequestBasePlan = qwen3_6::RequestBasePlan<Variant>;
using RequestPlan     = qwen3_6::RequestPlan<Variant>;
using Program         = qwen3_6::Program<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

// Auto verify width from k when --dflash-verify-width is omitted.
// 27B packed-tree product is k=7 W=12 (two GQA SmallT tiles). k=4 → W=5 chain and
// k=5 → W=6 chain stay one SmallT tile with no sibling columns. k>two_block_first
// stays chain W=k+1.
[[nodiscard]] inline constexpr std::uint32_t dflash_default_verify_width(std::uint32_t draft_window) {
    if constexpr (!DFlashConfig::tree_verify) {
        return draft_window + 1U;
    }
    if constexpr (DFlashConfig::two_block_first > 0) {
        if (draft_window > static_cast<std::uint32_t>(DFlashConfig::two_block_first)) {
            return draft_window + 1U;
        }
    }
    if (draft_window <= 5U) {
        return draft_window + 1U;
    }
    return static_cast<std::uint32_t>(DFlashConfig::verify_width);
}

[[nodiscard]] inline constexpr std::uint32_t dflash_verify_width(std::uint32_t draft_window,
                                                                std::uint32_t override_width = 0) {
    return override_width != 0 ? override_width : dflash_default_verify_width(draft_window);
}

// Packed-tree verify and GDN/KV path fold. Two-block k > two_block_first is chain
// verify: greedy accept does not write fold_path, so commit must keep path_length < 0.
// k<=5 with W=k+1 is chain (one SmallT tile, no sibling columns). k=4 W>k+1, and
// k=6..7, stay packed tree.
[[nodiscard]] inline constexpr bool dflash_uses_tree_verify(std::uint32_t draft_window,
                                                            std::uint32_t verify_width) {
    if constexpr (!DFlashConfig::tree_verify) {
        return false;
    }
    if constexpr (DFlashConfig::two_block_first > 0) {
        if (draft_window > static_cast<std::uint32_t>(DFlashConfig::two_block_first)) {
            return false;
        }
    }
    if (draft_window <= 5U && verify_width == draft_window + 1U) {
        return false;
    }
    return true;
}

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size,
                                                                std::uint32_t verify_width) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size, verify_width);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

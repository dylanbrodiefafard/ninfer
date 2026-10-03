#pragma once

#ifndef NINFER_QWEN36_VARIANT
#    error "NINFER_QWEN36_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN36_RUNTIME_NS
#    error "NINFER_QWEN36_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/targets/qwen3_6/runtime.h>
#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

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
// DFlash V1's per-topology graph allowance is large; it keeps one exchanging graph.
inline constexpr std::uint32_t kDFlashExchangeVariants =
    DFlashConfig::kind == qwen3_6::DFlashKind::DFlash2 ? qwen3_6::kGrammarExchangeVariants : 1U;

// Packed-tree verify width of the adaptive tree arm; 0 when the target has no tree route.
inline constexpr std::uint32_t kDFlashTreeVerifyWidth =
    static_cast<std::uint32_t>(DFlashConfig::tree_verify_width);
inline constexpr std::uint32_t kDFlashTreeVerifyMaxTokens =
    static_cast<std::uint32_t>(DFlashConfig::tree_verify_max_tokens);

// One captured DFlash round shape: draft window k, packed verify width, and the largest batch
// that runs it. W == k+1 is the chain; a wider W verifies a packed best-first tree. arm is the
// adaptive arm id (adaptive_tree_arm(k) for the adaptive tree arm, k otherwise).
struct DFlashRoundShape {
    std::uint32_t k            = 0;
    std::uint32_t verify_width = 0;
    std::uint32_t max_batch    = 0;
    std::uint32_t arm          = 0;
};

[[nodiscard]] inline constexpr bool dflash_uses_tree_verify(std::uint32_t k,
                                                            std::uint32_t verify_width) {
    return verify_width != k + 1U;
}

// Captured round shapes. Every captured k verifies its chain at every batch size, except that an
// explicit width pins the draft window's shape (W=k+1 is chain-only). Adaptive draft without an
// explicit width also captures the packed-tree arm of the full draft window for every batch whose
// packed extent fits kDFlashTreeVerifyMaxTokens.
[[nodiscard]] inline std::vector<DFlashRoundShape>
dflash_round_shapes(std::span<const std::uint32_t> captured_ks, std::uint32_t draft_window,
                    std::uint32_t override_width, std::uint32_t max_concurrency, bool adaptive) {
    std::vector<DFlashRoundShape> shapes;
    if (captured_ks.empty()) {
        shapes.push_back({draft_window, override_width != 0 ? override_width : draft_window + 1U,
                          max_concurrency, draft_window});
    } else {
        for (const std::uint32_t k : captured_ks) {
            shapes.push_back({k, override_width != 0 && k == draft_window ? override_width : k + 1U,
                              max_concurrency, k});
        }
    }
    if constexpr (kDFlashTreeVerifyWidth > 0) {
        if (adaptive && override_width == 0 && kDFlashTreeVerifyWidth > draft_window + 1U) {
            const std::uint32_t max_batch =
                std::min(max_concurrency, kDFlashTreeVerifyMaxTokens / kDFlashTreeVerifyWidth);
            if (max_batch > 0) {
                shapes.push_back({draft_window, kDFlashTreeVerifyWidth, max_batch,
                                  qwen3_6::adaptive_tree_arm(draft_window)});
            }
        }
    }
    return shapes;
}

// Adaptive arm ids a batch of this size can run, in captured order (chains, then the tree).
[[nodiscard]] inline std::vector<std::uint32_t>
dflash_batch_arms(std::span<const DFlashRoundShape> shapes, std::uint32_t batch_size) {
    std::vector<std::uint32_t> arms;
    for (const DFlashRoundShape& shape : shapes) {
        if (shape.max_batch >= batch_size) { arms.push_back(shape.arm); }
    }
    return arms;
}

// The captured shape of an arm.
[[nodiscard]] inline const DFlashRoundShape&
dflash_arm_shape(std::span<const DFlashRoundShape> shapes, std::uint32_t arm) {
    for (const DFlashRoundShape& shape : shapes) {
        if (shape.arm == arm) { return shape; }
    }
    throw std::logic_error("DFlash round shape for the selected arm was not captured");
}

// Storage / ReplaySSM / pending-features width: the widest captured shape.
[[nodiscard]] inline std::uint32_t
dflash_storage_verify_width(std::span<const DFlashRoundShape> shapes) {
    std::uint32_t width = 0;
    for (const DFlashRoundShape& shape : shapes) { width = std::max(width, shape.verify_width); }
    return width;
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

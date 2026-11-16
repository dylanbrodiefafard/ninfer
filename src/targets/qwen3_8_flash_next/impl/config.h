#pragma once

// Fixed Qwen3.8-Flash-Next text-model geometry (docs/maintainer/qwen3.8-flash-next-plan.md,
// "Verified model facts"). The Op headers own the geometry their contracts fix; this header names
// the model-level composition and re-states nothing an Op already declares.

#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next {

inline constexpr std::int32_t kHidden = ops::kHyperConnectionHidden;
inline constexpr std::int32_t kLayers = 48;
// Concurrent requests (lanes) one Program serves at most: the batched GDN, PLE and QSA Ops'
// row limit.
inline constexpr std::int32_t kMaxLanes = 12;
// Rows of the untied embedding and vocabulary head.
inline constexpr std::int32_t kVocabRows = 248320;
// Token ids the tokenizer defines, [0, kTokenDomain): sampling, greedy selection and log
// probabilities range over these rows only; head rows [kTokenDomain, kVocabRows) are padding.
inline constexpr std::int32_t kTokenDomain = 248077;
// U32 words of a token-eligibility bitset over the token domain (bit v of word v / 32).
inline constexpr std::int32_t kAllowedTokenWords = (kTokenDomain + 31) / 32;
inline constexpr float kRmsEpsilon               = 1.0e-6F;

// Every fourth layer (3, 7, ..., 47) is a QSA attention layer; the others are Gated DeltaNet.
inline constexpr std::int32_t kAttentionInterval = 4;
inline constexpr std::int32_t kQsaLayers         = kLayers / kAttentionInterval;
inline constexpr std::int32_t kGdnLayers         = kLayers - kQsaLayers;

[[nodiscard]] constexpr bool is_qsa_layer(std::int32_t layer) {
    return layer % kAttentionInterval == kAttentionInterval - 1;
}

// Index of a QSA layer among the QSA layers.
[[nodiscard]] constexpr std::int32_t qsa_index(std::int32_t layer) {
    return layer / kAttentionInterval;
}

// Index of a GDN layer among the GDN layers: the layer minus the QSA layers before it.
[[nodiscard]] constexpr std::int32_t gdn_index(std::int32_t layer) {
    return layer - (layer + 1) / kAttentionInterval;
}

static_assert(gdn_index(4) == 3 && gdn_index(46) == kGdnLayers - 1 && qsa_index(47) == 11);

// The PLE (n-gram) injection runs on the residual before layer 1's mixer read.
inline constexpr std::int32_t kPleLayer = 1;

// Gated DeltaNet: 16 query/key heads and 48 value heads of width 128 (value head h reads q/k head
// h / 3), a width-4 causal convolution over q|k|v, and a sigmoid output gate.
inline constexpr std::int32_t kGdnKeyHeads       = 16;
inline constexpr std::int32_t kGdnValueHeads     = 48;
inline constexpr std::int32_t kGdnHeadDim        = 128;
inline constexpr std::int32_t kGdnKeyDim         = kGdnKeyHeads * kGdnHeadDim;
inline constexpr std::int32_t kGdnValueDim       = kGdnValueHeads * kGdnHeadDim;
inline constexpr std::int32_t kGdnConvChannels   = 2 * kGdnKeyDim + kGdnValueDim;
inline constexpr std::int32_t kGdnProjectionRows = kGdnConvChannels + kGdnValueDim;
inline constexpr std::int32_t kGdnConvKernel     = 4;
inline constexpr std::int32_t kGdnConvState      = kGdnConvKernel - 1;
// Rows of the packed a|b gate projection consumed by gdn_gating_packed.
inline constexpr std::int32_t kGdnGateRows = 2 * kGdnValueHeads;

// QSA core attention: 24 query heads and 2 KV heads of width 256; the leading 64 features are
// rotated by text MRoPE (equal positions on all three axes) with theta 1e7.
inline constexpr std::int32_t kQsaQueryDim       = ops::kQsaQueryHeads * ops::kQsaHeadDim;
inline constexpr std::int32_t kQsaKvDim          = ops::kQsaKvHeads * ops::kQsaHeadDim;
inline constexpr std::int32_t kQsaProjectionRows = 2 * kQsaQueryDim + 2 * kQsaKvDim;
inline constexpr std::int32_t kQsaIndexQueryDim  = ops::kQsaIndexQueryHeads * ops::kQsaIndexHeadDim;
inline constexpr std::int32_t kRotaryDim         = 64;
inline constexpr std::int32_t kMropeAxes         = 3;
inline constexpr float kRopeTheta                = 1.0e7F;

// Mixer output projection [2560,6144] shared by the GDN and QSA layers.
inline constexpr std::int32_t kMixerOutputColumns = kGdnValueDim;
static_assert(kMixerOutputColumns == kQsaQueryDim);

// PLE projections of the decoded n-gram embedding: key [10240,2560] and value [2560,2560].
inline constexpr std::int32_t kPleKeyRows   = ops::kPleChannels;
inline constexpr std::int32_t kPleValueRows = ops::kPleHidden;


// Native DFlash2's adaptive domain K=0..3 uses four-column chains.
inline constexpr std::int32_t kDflashLayers   = 5;
inline constexpr std::int32_t kDflashFeatures = kDflashLayers * kHidden;
inline constexpr std::int32_t kDflashWindow   = 2048;
inline constexpr std::int32_t kDflashMaxK     = 3;
inline constexpr std::int32_t kDflashWidth    = kDflashMaxK + 1;
inline constexpr std::array<std::int32_t, kDflashLayers> kDflashFeatureLayers{5, 15, 25, 35, 45};
enum class DflashExecution : std::uint8_t { Disabled, Enabled };

static_assert(ops::kPleHidden == kHidden && ops::kMoeHidden == kHidden);
static_assert(kGdnProjectionRows == 16384 && kQsaProjectionRows == 13312);

} // namespace ninfer::targets::qwen3_8_flash_next

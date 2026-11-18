#pragma once

// Pure host constants of the Qwen3.8-Flash-Next expert-helper split, shared by the CUDA
// partition, local-partial and join Ops and the HIP helper Op
// (docs/maintainer/qwen3.8-flash-next-expert-helper.md). No device-runtime dependency.

#include <cstdint>

namespace ninfer::ops {

// Tokens of one helper round (one decode or verify round of up to twelve requests, four columns
// each), route entries per token (moe_route's top-k), and the routed experts of one layer.
inline constexpr std::int32_t kMoeHelperMaxTokens      = 48;
inline constexpr std::int32_t kMoeHelperRoutesPerToken = 10;
inline constexpr std::int32_t kMoeHelperExperts        = 512;

// Expert id of a compact route entry the helper does not own. Every id outside [0,512) is
// inactive; this is the value producers write.
inline constexpr std::int32_t kMoeHelperInactiveExpert = -1;

// A per-layer expert bitset is U32 [16]: bit e % 32 of word e / 32 is set when expert e is local
// (resident on or streamed to the RTX 5090) and clear when the helper owns it.
inline constexpr std::int32_t kMoeExpertMaskWords = kMoeHelperExperts / 32;

} // namespace ninfer::ops

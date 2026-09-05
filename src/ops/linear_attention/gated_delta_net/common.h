#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim  = 128;
inline constexpr std::int32_t kChunkSize = 64;
inline constexpr std::int32_t kFusedQNormalizationMinTokens = 448;
inline constexpr std::int32_t kFusedKNormalizationMinTokens = 512;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

[[nodiscard]] constexpr bool uses_fused_q_normalization(std::int32_t qk_heads,
                                                        std::int32_t value_heads,
                                                        std::int32_t tokens,
                                                        bool normalize_qk) noexcept {
    const std::int32_t full = (tokens / kChunkSize) * kChunkSize;
    return normalize_qk && qk_heads == 48 && value_heads == 48 &&
           full >= kFusedQNormalizationMinTokens;
}

[[nodiscard]] constexpr bool uses_fused_k_normalization(std::int32_t qk_heads,
                                                        std::int32_t value_heads,
                                                        std::int32_t tokens,
                                                        bool normalize_qk) noexcept {
    const std::int32_t full = (tokens / kChunkSize) * kChunkSize;
    return normalize_qk && qk_heads == 48 && value_heads == 48 &&
           full >= kFusedKNormalizationMinTokens;
}

} // namespace ninfer::ops::detail::gated_delta_net

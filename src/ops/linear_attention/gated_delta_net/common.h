#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kChunkTokens = 64;

[[nodiscard]] constexpr bool is_supported_head_dim(std::int64_t head_dim) noexcept {
    return head_dim == 16 || head_dim == 32 || head_dim == 64 || head_dim == 128;
}

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

} // namespace ninfer::ops::detail::gated_delta_net

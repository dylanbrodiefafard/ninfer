#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim  = 128;
inline constexpr std::int32_t kChunkSize = 64;

// Widths from which the multi-token Op runs the chunked WY path, the last chunk possibly
// partial; narrower widths run the sequential recurrence. The recurrence's cost grows with every
// token while one chunk costs about the same at any fill. In isolation (CUDA Graph replay) the two
// meet near 40 tokens at the 27B geometry, but eager prefill also pays the chunked path's extra
// normalization and stage launches: whole-model 27B prefill on the RTX 5090 breaks even at 60.
inline constexpr std::int32_t kChunkedMinTokens = 60;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

} // namespace ninfer::ops::detail::gated_delta_net

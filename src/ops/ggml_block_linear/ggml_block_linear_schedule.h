#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kGgmlBlockLinearAggregateTokens = 16;
inline constexpr std::int32_t kGgmlQ5KReplayTokens             = 256;

inline constexpr bool uses_qwen4_q5_k_weight_replay(QType format, std::int32_t n,
                                                     std::int32_t k, std::int32_t tokens) {
    return format == QType::GGML_Q5_K && n == 10240 && k == 2560 &&
           tokens >= kGgmlQ5KReplayTokens && tokens % kGgmlQ5KReplayTokens == 0;
}

} // namespace ninfer::ops::detail

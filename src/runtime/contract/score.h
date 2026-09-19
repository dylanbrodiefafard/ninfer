#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer {

// Positions [0, skip) fill KV but do not enter mean NLL. Default is n/2, matching llama.cpp's
// second-half window so every counted token has context. Always leaves at least one target.
[[nodiscard]] inline std::uint32_t resolve_score_skip(std::uint32_t prompt_tokens,
                                                      std::optional<std::uint32_t> skip_tokens) {
    if (prompt_tokens < 2) {
        throw std::invalid_argument("score requires at least two prompt tokens");
    }
    const std::uint32_t max_skip = prompt_tokens - 2;
    if (!skip_tokens.has_value()) { return std::min(prompt_tokens / 2, max_skip); }
    if (*skip_tokens > max_skip) {
        throw std::invalid_argument("score skip leaves no teacher-forced target");
    }
    return *skip_tokens;
}

// Decode executes the first `prefix` tokens with the prefill kernel, then T=1 for the rest.
// Prefix is at least one token and leaves at least one T=1 target.
[[nodiscard]] inline std::uint32_t resolve_decode_prefix(std::uint32_t prompt_tokens,
                                                         std::uint32_t skip) {
    if (prompt_tokens < 3) {
        throw std::invalid_argument("decode score requires at least three prompt tokens");
    }
    const std::uint32_t prefix = std::max(skip, 1u);
    if (prefix + 1 >= prompt_tokens) {
        throw std::invalid_argument("decode score skip leaves no T=1 target");
    }
    return prefix;
}

// Next-token ids for scored columns in one prefill chunk. Column `position` is scored iff
// skip <= position && position + 1 < ids.size().
[[nodiscard]] inline std::vector<std::int32_t>
prefill_chunk_targets(std::span<const TokenId> ids, std::uint32_t chunk_begin,
                      std::uint32_t chunk_tokens, std::uint32_t skip) {
    const auto n = static_cast<std::uint32_t>(ids.size());
    std::vector<std::int32_t> targets;
    if (chunk_tokens == 0 || chunk_begin >= n) { return targets; }
    const std::uint32_t available = std::min(chunk_tokens, n - chunk_begin);
    targets.reserve(available);
    for (std::uint32_t local = 0; local < available; ++local) {
        const std::uint32_t position = chunk_begin + local;
        if (position < skip) { continue; }
        if (position + 1 >= n) { break; }
        targets.push_back(ids[position + 1]);
    }
    return targets;
}

inline void record_score_nll(ScoreResult& result, double value) {
    if (!std::isfinite(value)) {
        ++result.non_finite;
        return;
    }
    if (result.tokens_scored == 0 || value > result.max_nll) { result.max_nll = value; }
    result.sum_nll += value;
    ++result.tokens_scored;
    if (value >= kScoreTerribleNll) { ++result.terrible_tokens; }
    result.token_nlls.push_back(static_cast<float>(value));
}

} // namespace ninfer

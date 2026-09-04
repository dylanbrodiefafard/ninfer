#pragma once

#include "ops/launcher/ngram_embedding.h"

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ std::int32_t ngram_euclidean_remainder(std::uint64_t mixed,
                                                                  std::int32_t modulus) {
    const std::int64_t signed_mixed = static_cast<std::int64_t>(mixed);
    std::int64_t remainder          = signed_mixed % static_cast<std::int64_t>(modulus);
    if (remainder < 0) { remainder += modulus; }
    return static_cast<std::int32_t>(remainder);
}

__global__ void ngram_row_ids_kernel(const std::int32_t* input_ids,
                                     const std::int32_t* valid_tokens,
                                     const std::int32_t* old_history,
                                     detail::NgramRowLaunchConfig config, std::int32_t* row_ids,
                                     std::int32_t width, std::int32_t requests) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(requests) * width * 16;
    if (index >= count) { return; }

    const std::int32_t head = static_cast<std::int32_t>(index % 16);
    const std::int64_t token_index = index / 16;
    const std::int32_t token = static_cast<std::int32_t>(token_index % width);
    const std::int32_t request = static_cast<std::int32_t>(token_index / width);
    if (token >= valid_tokens[request]) {
        row_ids[index] = -1;
        return;
    }

    const std::int64_t request_base = static_cast<std::int64_t>(request) * width;
    const std::int32_t current      = input_ids[request_base + token];
    const std::int32_t lag1 =
        token >= 1 ? input_ids[request_base + token - 1] : old_history[2 * request + 1];
    const std::int32_t preceding_lag1 =
        token >= 2 ? input_ids[request_base + token - 2]
                   : old_history[2 * request + static_cast<std::int32_t>(token != 0)];
    const std::int32_t lag2 =
        lag1 == config.eos_token_id ? config.eos_token_id : preceding_lag1;
    const std::uint64_t mixed2 =
        static_cast<std::uint64_t>(current) * config.multiplier[0] ^
        static_cast<std::uint64_t>(lag1) * config.multiplier[1];
    const std::uint64_t mixed =
        head < 8 ? mixed2 : mixed2 ^ static_cast<std::uint64_t>(lag2) * config.multiplier[2];
    row_ids[index] =
        config.offset[head] + ngram_euclidean_remainder(mixed, config.prime[head]);
}

__global__ void ngram_history_kernel(const std::int32_t* input_ids,
                                     const std::int32_t* valid_tokens,
                                     const std::int32_t* old_history, std::int32_t* new_history,
                                     std::int32_t width, std::int32_t requests) {
    const std::int32_t request = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (request >= requests) { return; }
    const std::int32_t valid = valid_tokens[request];
    const std::int64_t base  = static_cast<std::int64_t>(request) * width;
    if (valid == 0) {
        new_history[2 * request]     = old_history[2 * request];
        new_history[2 * request + 1] = old_history[2 * request + 1];
    } else if (valid == 1) {
        new_history[2 * request]     = old_history[2 * request + 1];
        new_history[2 * request + 1] = input_ids[base];
    } else {
        new_history[2 * request]     = input_ids[base + valid - 2];
        new_history[2 * request + 1] = input_ids[base + valid - 1];
    }
}

} // namespace ninfer::ops

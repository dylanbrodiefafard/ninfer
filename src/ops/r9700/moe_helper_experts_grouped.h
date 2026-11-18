#pragma once

// Private grouped-GEMM route of moe_helper_experts (ninfer/ops/moe_helper_experts.h) for calls
// whose max_tokens exceeds kMoeHelperMaxTokens. The public wrapper validates every operand and
// owns the workspace; this route only sizes its workspace region and enqueues its kernels.

#include "ninfer/ops/moe_helper_experts.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

/**
 * Bytes of the route's workspace region for every call with max_tokens <= `max_tokens`, where
 * kMoeHelperMaxTokens < max_tokens <= kMoeHelperExpertsMaxTokens. The region needs no
 * initialization: the route's first kernel resets the counters it uses.
 */
[[nodiscard]] std::size_t moe_helper_grouped_workspace_bytes(std::int32_t max_tokens);

/**
 * Enqueues the route on `stream` for validated `args` (kMoeHelperMaxTokens < args.max_tokens <=
 * capacity). `region` is a 256-byte aligned device region of moe_helper_grouped_workspace_bytes
 * (capacity) bytes on `device`, serving one call at a time in stream order. Throws
 * std::runtime_error when a launch fails.
 */
void moe_helper_experts_grouped(const MoeHelperExpertsArgs& args, std::byte* region,
                                std::int32_t capacity, int device, hipStream_t stream);

} // namespace ninfer::ops::detail

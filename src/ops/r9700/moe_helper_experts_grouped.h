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

/**
 * The A8 profile's quantization of the down-projection input, as the route enqueues it: for each
 * row r < min(*rows, capacity), capacity in [1, 10 kMoeHelperExpertsMaxTokens] (the route's
 * entries; 32-bit element offsets), of x (BF16 bits [capacity][640], 8-byte aligned), writes the
 * row's E4M3 codes to codes[r][0..640) (4-byte aligned) and its FP32 scale to scales[r] by the rule
 * MoeHelperE4m3Rows documents. `rows` (U32 [1]) is read on the device when the kernel executes;
 * rows at or past it are not written. Enqueued on `stream` of `device` (current); throws
 * std::runtime_error when the launch fails. The buffers must be disjoint.
 */
void moe_helper_quantize_rows(const std::uint16_t* x, const std::uint32_t* rows,
                              std::int32_t capacity, std::uint8_t* codes, float* scales, int device,
                              hipStream_t stream);

} // namespace ninfer::ops::detail

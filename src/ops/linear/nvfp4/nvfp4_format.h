#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

// Validates an NVFP4 weight's encoding, planes and weight divisor. Its input_scale_divisor is
// either 0 (the weight carries no activation divisor and runs only A16 routes) or finite and
// positive; routes that quantize activations obtain it through nvfp4_input_scale_divisor or
// nvfp4_w4a4_alpha.
Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

// The activation divisor an A4 route quantizes with. Throws std::invalid_argument for a weight
// without one (A16-only).
[[nodiscard]] float nvfp4_input_scale_divisor(const Weight& weight, const char* operation);

// The W4A4 dequantization multiplier 1 / (input divisor * weight divisor). Throws
// std::invalid_argument for a weight without an input divisor.
[[nodiscard]] float nvfp4_w4a4_alpha(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail

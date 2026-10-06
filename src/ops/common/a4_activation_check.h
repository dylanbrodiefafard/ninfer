#pragma once

// ninfer::ops::detail - host validation of an A4Activation at an Op boundary
// (include/ninfer/ops/a4_activation.h).

#include "ninfer/ops/a4_activation.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// Requires codes U8 [input_rows / 2, T] and scales U8 holding at least the tiled plane of T
// tokens, both contiguous, non-null and 16-byte aligned (the projections load them with 16-byte
// copies and TMA), and a positive finite input scale divisor. Returns T. Throws
// std::invalid_argument naming `op` on any mismatch.
inline std::int32_t validate_a4_activation(const A4Activation& activation, std::int32_t input_rows,
                                           const char* op) {
    const Tensor& codes  = activation.codes;
    const Tensor& scales = activation.scales;
    const auto fail      = [op](const char* what) {
        throw std::invalid_argument(std::string(op) + ": A4 activation " + what);
    };
    if (codes.dtype != DType::U8 || scales.dtype != DType::U8) { fail("must be U8 codes/scales"); }
    if (codes.ne[0] != input_rows / 2 || codes.ne[1] <= 0 || codes.ne[2] != 1 || codes.ne[3] != 1) {
        fail("codes must be [input_rows / 2, T]");
    }
    const std::int32_t tokens = codes.ne[1];
    if (scales.numel() < a4_scale_plane_bytes(input_rows, tokens)) {
        fail("scales must hold the tiled plane of T tokens");
    }
    if (!codes.is_contiguous() || !scales.is_contiguous()) { fail("must be contiguous"); }
    if (codes.data == nullptr || scales.data == nullptr) { fail("data must be non-null"); }
    if ((reinterpret_cast<std::uintptr_t>(codes.data) % 16) != 0 ||
        (reinterpret_cast<std::uintptr_t>(scales.data) % 16) != 0) {
        fail("codes and scales must be 16-byte aligned");
    }
    if (!(activation.input_scale_divisor > 0.0F) ||
        !std::isfinite(activation.input_scale_divisor)) {
        fail("input scale divisor must be positive and finite");
    }
    return tokens;
}

} // namespace ninfer::ops::detail

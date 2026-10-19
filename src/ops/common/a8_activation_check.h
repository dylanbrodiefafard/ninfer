#pragma once

// ninfer::ops::detail - host validation of an A8Activation at an Op boundary
// (include/ninfer/ops/a8_activation.h).

#include "ninfer/ops/a8_activation.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// Requires codes FP8_E4M3FN [input_rows, columns...] and scales FP32 [columns...], both contiguous
// and non-null, where the column dimensions are codes.ne[1..3]. Codes must be 16-byte aligned (the
// projections stage them with 16-byte copies); scales need only FP32 alignment, so a column slice
// of an activation remains valid. Returns the column count. Throws std::invalid_argument
// naming `op` on any mismatch.
inline std::int64_t validate_a8_activation(const A8Activation& activation, std::int32_t input_rows,
                                           const char* op) {
    const Tensor& codes  = activation.codes;
    const Tensor& scales = activation.scales;
    const auto fail      = [op](const char* what) {
        throw std::invalid_argument(std::string(op) + ": A8 activation " + what);
    };
    if (codes.dtype != DType::FP8_E4M3FN || scales.dtype != DType::FP32) {
        fail("must be FP8_E4M3FN codes with FP32 scales");
    }
    if (codes.ne[0] != input_rows) { fail("codes ne[0] must equal the input rows"); }
    if (scales.ne[0] != codes.ne[1] || scales.ne[1] != codes.ne[2] || scales.ne[2] != codes.ne[3] ||
        scales.ne[3] != 1) {
        fail("scales must hold one value per codes column");
    }
    if (!codes.is_contiguous() || !scales.is_contiguous()) { fail("must be contiguous"); }
    if (codes.data == nullptr || scales.data == nullptr) { fail("data must be non-null"); }
    if ((reinterpret_cast<std::uintptr_t>(codes.data) % 16) != 0 ||
        (reinterpret_cast<std::uintptr_t>(scales.data) % alignof(float)) != 0) {
        fail("codes must be 16-byte and scales FP32 aligned");
    }
    const std::int64_t columns = scales.numel();
    if (columns <= 0) { fail("must have at least one column"); }
    return columns;
}

} // namespace ninfer::ops::detail

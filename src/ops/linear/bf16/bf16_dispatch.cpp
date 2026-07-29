#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::uint64_t required_payload_bytes(const Weight& weight) {
    if (weight.n <= 0 || weight.k <= 0) {
        throw std::invalid_argument("bf16 linear: invalid weight shape");
    }
    const auto n = static_cast<std::uint64_t>(weight.n);
    const auto k = static_cast<std::uint64_t>(weight.k);
    if (n > std::numeric_limits<std::uint64_t>::max() / k ||
        n * k > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("bf16 linear: weight payload size overflow");
    }
    return n * k * sizeof(std::uint16_t);
}

void validate_bf16_weight(const Weight& weight) {
    constexpr std::int32_t kRows   = Bf16LinearDecodeGeometry::kOutputRows;
    constexpr std::int32_t kHidden = Bf16LinearDecodeGeometry::kInputRows;
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.ndim != 2 || weight.n != kRows || weight.k != kHidden || weight.shape[0] != kRows ||
        weight.shape[1] != kHidden || weight.padded_shape[0] != kRows ||
        weight.padded_shape[1] != kHidden ||
        weight.payload_bytes < required_payload_bytes(weight) || weight.high_plane_bytes != 0 ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.group_size != 0 ||
        weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument("bf16 linear: invalid contiguous weight");
    }
}

} // namespace

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("bf16 linear: only A16Only is supported");
    }
    if (x.ne[1] != 1) { throw std::invalid_argument("bf16 linear: unsupported shape or T"); }
    validate_bf16_weight(weight);
    launch_bf16_decode(x, weight, out, stream);
}

} // namespace ninfer::ops::detail

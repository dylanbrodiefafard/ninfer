// ninfer::ops - stream_mean wrapper: validation and launch.
#include "ninfer/ops/stream_mean.h"

#include "ops/launcher/stream_mean.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_16(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

} // namespace

void stream_mean(const Tensor& residual, Tensor& out, cudaStream_t stream) {
    if (residual.dtype != DType::BF16 || residual.ne[0] != kStreamMeanHidden ||
        residual.ne[1] != kStreamMeanStreams || residual.ne[2] <= 0 || residual.ne[3] != 1 ||
        !residual.is_contiguous() || !aligned_16(residual.data)) {
        throw std::invalid_argument(
            "stream_mean: residual must be contiguous 16-byte aligned BF16 [2560,4,T]");
    }
    const std::int32_t tokens           = residual.ne[2];
    constexpr std::int64_t kColumnBytes = std::int64_t{kStreamMeanHidden} * 2;
    if (out.dtype != DType::BF16 || out.ne[0] != kStreamMeanHidden || out.ne[1] != tokens ||
        out.ne[2] != 1 || out.ne[3] != 1 || out.nb[0] != 2 || out.nb[1] < kColumnBytes ||
        out.nb[1] % 16 != 0 || !aligned_16(out.data)) {
        throw std::invalid_argument("stream_mean: out must be 16-byte aligned BF16 [2560,T] with "
                                    "contiguous rows and a 16-byte multiple column stride");
    }
    // out spans (T-1) column strides plus one column; T * nb[1] cannot overflow for a tensor the
    // caller allocated, but the check keeps the interval arithmetic defined.
    if (out.nb[1] > std::numeric_limits<std::int64_t>::max() / tokens) {
        throw std::overflow_error("stream_mean: out span overflows int64");
    }
    const auto residual_begin = reinterpret_cast<std::uintptr_t>(residual.data);
    const auto residual_end   = residual_begin + residual.bytes();
    const auto out_begin      = reinterpret_cast<std::uintptr_t>(out.data);
    const auto out_end =
        out_begin + static_cast<std::uintptr_t>(out.nb[1] * (tokens - 1) + kColumnBytes);
    if (residual_begin < out_end && out_begin < residual_end) {
        throw std::invalid_argument("stream_mean: out must not overlap residual");
    }
    detail::stream_mean_launch(residual, out, stream);
}

} // namespace ninfer::ops

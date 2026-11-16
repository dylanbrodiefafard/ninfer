// ninfer::ops - bf16_projection wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/bf16_projection.h"

#include "ops/launcher/bf16_projection.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_contiguous(const Tensor& t) {
    return t.is_contiguous() && t.data != nullptr &&
           (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

} // namespace

bool bf16_projection_admits(std::int32_t output_rows, std::int32_t input_rows) {
    return input_rows == 2560 &&
           (output_rows == 96 || output_rows == 128 || output_rows == 512 || output_rows == 544 ||
            output_rows == 2560 || output_rows == 10240);
}

DType bf16_projection_output_dtype(std::int32_t output_rows) {
    // The QSA indexer query [512] and key [128] projections feed BF16 QSA inputs.
    return output_rows == 128 || output_rows == 512 ? DType::BF16 : DType::FP32;
}

void bf16_projection(const Tensor& x, const Tensor& weight, Tensor& y, cudaStream_t stream) {
    const std::int32_t input_rows  = weight.ne[0];
    const std::int32_t output_rows = weight.ne[1];
    const std::int32_t tokens      = x.ne[1];
    if (!bf16_projection_admits(output_rows, input_rows)) {
        throw std::invalid_argument("bf16_projection: unsupported weight shape");
    }
    if (weight.dtype != DType::BF16 || weight.ne[2] != 1 || weight.ne[3] != 1 ||
        !aligned_contiguous(weight)) {
        throw std::invalid_argument(
            "bf16_projection: weight must be aligned contiguous BF16 [N,K]");
    }
    if (tokens < 1 || x.dtype != DType::BF16 || x.ne[0] != input_rows || x.ne[2] != 1 ||
        x.ne[3] != 1 || !aligned_contiguous(x)) {
        throw std::invalid_argument(
            "bf16_projection: x must be aligned contiguous BF16 [K,T], T>0");
    }
    const DType output_dtype = bf16_projection_output_dtype(output_rows);
    if (y.dtype != output_dtype || y.ne[0] != output_rows || y.ne[1] != tokens || y.ne[2] != 1 ||
        y.ne[3] != 1 || !aligned_contiguous(y)) {
        throw std::invalid_argument(
            "bf16_projection: y must be aligned contiguous [N,T] of the projection's output dtype");
    }
    if (overlaps(y, x) || overlaps(y, weight)) {
        throw std::invalid_argument("bf16_projection: y must be disjoint from x and weight");
    }
    const auto* activation = static_cast<const std::uint16_t*>(x.data);
    const auto* matrix     = static_cast<const std::uint16_t*>(weight.data);
    if (output_dtype == DType::BF16) {
        detail::bf16_projection_launch(activation, matrix, static_cast<std::uint16_t*>(y.data),
                                       output_rows, input_rows, tokens, stream);
    } else {
        detail::bf16_projection_launch(activation, matrix, static_cast<float*>(y.data), output_rows,
                                       input_rows, tokens, stream);
    }
}

} // namespace ninfer::ops

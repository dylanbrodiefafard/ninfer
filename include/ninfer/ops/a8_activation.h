#pragma once

// ninfer::ops - row-scaled E4M3 activation: the A8 codec value passed between an activation
// producer that quantizes its own output and an NVFP4/A8 projection that consumes it.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

/**
 * Codec, for each column t of a represented BF16 activation v[K,T]:
 *
 *   scale[t]  = max_k |v[k,t]| / 448                (FP32; 0 when the column is all zero)
 *   codes[k,t] = E4M3FN_satfinite_rn(v[k,t] * (1 / scale[t]))   (0 when scale[t] == 0)
 *
 * represented as codes[k,t] * scale[t]. `1 / scale[t]` is the FP32 reciprocal, so the codec is an
 * exact function of the BF16 values and matches the projections' internal quantizer bit for bit.
 * `codes` is contiguous FP8_E4M3FN with ne[0] = K and the remaining dimensions equal to the
 * activation's columns; `scales` is contiguous FP32 with one value per column and the column
 * dimensions of `codes` shifted down by one. The value does not own storage.
 */
struct A8Activation {
    Tensor codes;
    Tensor scales;
};

namespace detail {

// 256-byte alignment matches the projections' own A8 scratch so a producer may write into either.
inline constexpr std::size_t kA8ActivationAlignment = 256;

} // namespace detail

/**
 * Allocates an [input_rows, tokens] A8 activation from `allocator` (a WorkspaceArena or its
 * WorkspaceLayoutBuilder simulation). input_rows must be a positive multiple of 32 and tokens
 * positive; violations throw std::invalid_argument.
 */
template <class Allocator>
[[nodiscard]] A8Activation allocate_a8_activation(Allocator& allocator, std::int32_t input_rows,
                                                  std::int32_t tokens) {
    if (input_rows <= 0 || (input_rows % 32) != 0 || tokens <= 0) {
        throw std::invalid_argument("A8 activation: invalid [K,T]");
    }
    return {
        allocator.alloc(DType::FP8_E4M3FN, {input_rows, tokens}, detail::kA8ActivationAlignment),
        allocator.alloc(DType::FP32, {tokens}, detail::kA8ActivationAlignment),
    };
}

/**
 * The columns [first, first + count) of a 2-D [K,T] activation, sharing its storage. The view stays
 * a valid A8 activation: codes advance by whole K-byte columns and scales by whole floats.
 */
[[nodiscard]] inline A8Activation a8_activation_columns(const A8Activation& activation,
                                                        std::int32_t first, std::int32_t count) {
    return {activation.codes.slice(1, first, count), activation.scales.slice(0, first, count)};
}

} // namespace ninfer::ops

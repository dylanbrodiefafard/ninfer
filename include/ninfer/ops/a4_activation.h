#pragma once

// ninfer::ops - NVFP4 activation: the A4 codec value passed between an activation producer that
// quantizes its own output and an NVFP4 W4A4 projection that consumes it.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

// The scale plane stores [kA4ScaleTileTokens tokens x kA4ScaleTileGroups groups] byte tiles, 4 KiB
// each, ordered token tile major then group tile; inside a tile scale (t, g) is at byte
// (t % 256) * 16 + g % 16. The plane covers whole token tiles.
inline constexpr std::int32_t kA4ScaleTileTokens = 256;
inline constexpr std::int32_t kA4ScaleTileGroups = 16;

/**
 * Codec, for each column t and 16-value group g of a represented BF16 activation v[K,T], with the
 * consuming weight's input scale divisor d:
 *
 *   amax        = max_{k in group g} |v[k,t]|
 *   scale[t,g]  = E4M3_satfinite_rn(fdiv_rn(d * amax, 6))
 *   codes[k,t]  = E2M1_satfinite_rn(fdiv_rn(v[k,t] * d, decode(scale[t,g])))   (0 when the scale
 *                 is 0)
 *
 * represented as codes[k,t] * decode(scale[t,g]) / d. The codec is an exact function of the BF16
 * values and d, and matches the projections' internal quantizer bit for bit. `codes` is
 * contiguous U8 [K/2, T]: byte k/2 of column t holds element k in its low and k+1 in its high
 * nibble. `scales` is contiguous U8 holding the tiled plane above, sized for whole token tiles;
 * a producer writes the scales of the real tokens and zero scales for the padding tokens of the
 * last 16-token fragment, and leaves the rest of the plane unspecified. The value does not own
 * storage.
 */
struct A4Activation {
    Tensor codes;
    Tensor scales;
    float input_scale_divisor = 0.0F;
};

// Widths from which each NVFP4 AllowA4 projection runs W4A4 and accepts an A4 activation; narrower
// widths keep the A16 route, whose input is BF16.
inline constexpr std::int32_t kA4AttnInputMinTokens     = 4;
inline constexpr std::int32_t kA4GdnInputMinTokens      = 3;
inline constexpr std::int32_t kA4MlpGateUpMinTokens     = 2;
inline constexpr std::int32_t kA4Residual6144MinTokens  = 5;
inline constexpr std::int32_t kA4Residual17408MinTokens = 3;

namespace detail {

inline constexpr std::size_t kA4ActivationAlignment = 256;

[[nodiscard]] constexpr std::int64_t a4_scale_plane_bytes(std::int32_t input_rows,
                                                          std::int32_t tokens) noexcept {
    const std::int64_t token_tiles =
        (static_cast<std::int64_t>(tokens) + kA4ScaleTileTokens - 1) / kA4ScaleTileTokens;
    return token_tiles * kA4ScaleTileTokens * (input_rows / 16);
}

} // namespace detail

/**
 * Allocates an [input_rows, tokens] A4 activation for a weight with `input_scale_divisor` from
 * `allocator` (a WorkspaceArena or its WorkspaceLayoutBuilder simulation). input_rows must be a
 * positive multiple of kA4ScaleTileGroups * 16 and tokens positive; violations throw
 * std::invalid_argument.
 */
template <class Allocator>
[[nodiscard]] A4Activation allocate_a4_activation(Allocator& allocator, std::int32_t input_rows,
                                                  std::int32_t tokens, float input_scale_divisor) {
    if (input_rows <= 0 || (input_rows % (kA4ScaleTileGroups * 16)) != 0 || tokens <= 0) {
        throw std::invalid_argument("A4 activation: invalid [K,T]");
    }
    const std::int64_t scale_bytes = detail::a4_scale_plane_bytes(input_rows, tokens);
    if (scale_bytes > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("A4 activation: scale plane exceeds a 32-bit extent");
    }
    return {
        allocator.alloc(DType::U8, {input_rows / 2, tokens}, detail::kA4ActivationAlignment),
        allocator.alloc(DType::U8, {static_cast<std::int32_t>(scale_bytes)},
                        detail::kA4ActivationAlignment),
        input_scale_divisor,
    };
}

} // namespace ninfer::ops

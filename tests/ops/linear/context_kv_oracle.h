#pragma once

#include "ops/quantized_weight.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::context_kv {

inline constexpr std::int32_t kHidden       = 5120;
inline constexpr std::int32_t kParentRows   = 6144;
inline constexpr std::int32_t kSelectedRows = 2048;
inline constexpr std::int32_t kFirstRow     = 4096;

inline std::uint8_t scale_word(std::int32_t row, std::int32_t group, std::uint32_t seed) {
    constexpr std::array<std::uint8_t, 12> scales{0x00, 0x01, 0x07, 0x08, 0x1d, 0x38,
                                                  0x51, 0x7e, 0x81, 0x88, 0xb8, 0xfe};
    const auto hash = quantized_weight::detail::mix64(
        static_cast<std::uint64_t>(row) * (kHidden / 16) + group + seed);
    return scales[hash % scales.size()];
}

// The oracle decodes the represented signed formats directly. It does not use production
// conversion helpers, staging casts, inverse-scale multiplication, or MMA reduction order.
inline double decode_code(std::uint8_t code) {
    constexpr std::array<double, 8> magnitudes{0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return std::copysign(magnitudes.at(code & 7U), (code & 8U) != 0 ? -1.0 : 1.0);
}

inline double decode_scale(std::uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int fraction = code & 7;
    if (exponent == 15 && fraction == 7) {
        throw std::invalid_argument("context KV oracle: NaN scale");
    }
    const double magnitude = exponent == 0
                                 ? std::ldexp(static_cast<double>(fraction), -9)
                                 : std::ldexp(static_cast<double>(8 + fraction), exponent - 10);
    return std::copysign(magnitude, (code & 128U) != 0 ? -1.0 : 1.0);
}

inline std::size_t scale_index(std::int32_t row, std::int32_t group) {
    return (static_cast<std::size_t>(row / 128) * (kHidden / 64) + group / 4) * 512 +
           static_cast<std::size_t>(row % 32) * 16 + static_cast<std::size_t>(row % 128 / 32) * 4 +
           group % 4;
}

inline quantized_weight::PackedWeight make_weight(std::uint32_t seed, float divisor) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = divisor;
    options.input_scale_divisor  = 3.5F;
    auto packed =
        quantized_weight::make_patterned_weight(QType::NVFP4, kParentRows, kHidden, seed, options);
    // Independent coordinate hash breaks the fixture's periodicity across the skipped 4096 rows.
    // Zero, subnormal, ordinary, signed and saturated scales exercise exact stored-value decode.
    // Walk physical storage in order independently of the logical decoder's index formula.
    std::size_t offset = packed.scale_plane_offset;
    for (std::int32_t row_tile = 0; row_tile < kParentRows / 128; ++row_tile) {
        for (std::int32_t group_tile = 0; group_tile < kHidden / 64; ++group_tile) {
            for (std::int32_t lane = 0; lane < 32; ++lane) {
                for (std::int32_t quartile = 0; quartile < 4; ++quartile) {
                    for (std::int32_t inner = 0; inner < 4; ++inner) {
                        packed.payload[offset++] = scale_word(row_tile * 128 + quartile * 32 + lane,
                                                              group_tile * 4 + inner, seed);
                    }
                }
            }
        }
    }
    return packed;
}

inline void verify_exact_codec(const std::vector<double>& logical, std::uint32_t seed,
                               float divisor) {
    for (std::int32_t selected = 0; selected < kSelectedRows; ++selected) {
        const auto original = kFirstRow + selected;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const auto code =
                static_cast<std::uint8_t>((static_cast<std::uint32_t>(original) * 13U +
                                           static_cast<std::uint32_t>(column) * 7U + seed) &
                                          15U);
            const auto expected =
                decode_code(code) * decode_scale(scale_word(original, column / 16, seed)) / divisor;
            if (logical[static_cast<std::size_t>(selected) * kHidden + column] != expected) {
                throw std::runtime_error("context KV oracle: exact codec mapping mismatch");
            }
        }
    }
}

inline std::vector<double> decode_selected(const quantized_weight::PackedWeight& packed) {
    std::vector<double> logical(static_cast<std::size_t>(kSelectedRows) * kHidden);
    for (std::int32_t selected = 0; selected < kSelectedRows; ++selected) {
        const auto original = kFirstRow + selected;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const auto byte =
                packed.payload[static_cast<std::size_t>(original) * kHidden / 2 + column / 2];
            const auto code = static_cast<std::uint8_t>((byte >> ((column % 2) * 4)) & 15U);
            const auto scale =
                packed.payload[packed.scale_plane_offset + scale_index(original, column / 16)];
            logical[static_cast<std::size_t>(selected) * kHidden + column] =
                decode_code(code) * decode_scale(scale) / packed.weight.weight_scale_divisor;
        }
    }
    return logical;
}

inline std::vector<double> decode_selected_row_split(const quantized_weight::PackedWeight& packed) {
    std::vector<double> logical(static_cast<std::size_t>(kSelectedRows) * kHidden);
    for (std::int32_t selected = 0; selected < kSelectedRows; ++selected) {
        for (std::int32_t column = 0; column < kHidden; ++column) {
            // The fixture's independent exact signed-code/FP16-scale oracle reads original
            // parent coordinates, without constructing or decoding any production row view.
            logical[static_cast<std::size_t>(selected) * kHidden + column] =
                quantized_weight::logical_weight_fp64(packed, kFirstRow + selected, column);
        }
    }
    return logical;
}

// Each complete dot accumulates sequentially in FP64 from the original BF16 values and exactly
// decoded stored weight. Eight columns share host weight reads without changing any dot's order.
inline std::vector<double> project(const std::vector<double>& weights,
                                   const std::vector<float>& activation) {
    if (weights.size() != static_cast<std::size_t>(kSelectedRows) * kHidden || activation.empty() ||
        activation.size() % kHidden != 0) {
        throw std::invalid_argument("context KV oracle: invalid logical matrices");
    }
    const auto tokens = activation.size() / kHidden;
    std::vector<double> output(static_cast<std::size_t>(kSelectedRows) * tokens);
    for (std::int32_t row = 0; row < kSelectedRows; ++row) {
        for (std::size_t first = 0; first < tokens; first += 8) {
            const auto active = std::min<std::size_t>(8, tokens - first);
            std::array<double, 8> sums{};
            for (std::int32_t column = 0; column < kHidden; ++column) {
                const auto weight = weights[static_cast<std::size_t>(row) * kHidden + column];
                for (std::size_t token = 0; token < active; ++token) {
                    sums[token] += weight * activation[(first + token) * kHidden + column];
                }
            }
            for (std::size_t token = 0; token < active; ++token) {
                output[(first + token) * kSelectedRows + row] = sums[token];
            }
        }
    }
    return output;
}

} // namespace ninfer::test::context_kv

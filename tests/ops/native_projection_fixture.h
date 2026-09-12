#pragma once

#include "ops/quantized_weight.h"

namespace ninfer::test {

inline bool native_projection_format(QType type) {
    return type == QType::NVFP4 || type == QType::FP8_E4M3FN_ROW_BF16S;
}

inline quantized_weight::PackedWeight native_projection_fixture(
    QType type, int rows, int columns, unsigned seed) {
    quantized_weight::PatternedWeightOptions options;
    if (type == QType::NVFP4) {
        options.weight_scale_divisor = 64.0F;
        options.input_scale_divisor = 1.0F;
    }
    auto result = quantized_weight::make_patterned_weight(type, rows, columns, seed, options);
    if (type == QType::FP8_E4M3FN_ROW_BF16S) {
        // Keep a nonlinear layer in its useful dynamic range while retaining all
        // signed code classes and row-scale variation in the represented fixture.
        for (int row = 0; row < rows; ++row) {
            const auto offset = result.scale_plane_offset + 2ULL * row;
            const auto bits = quantized_weight::detail::load_u16_le(result.payload, offset);
            quantized_weight::detail::store_u16_le(result.payload, offset, bits - (8U << 7U));
        }
    }
    return result;
}

inline std::vector<double> native_projection_oracle(
    const quantized_weight::PackedWeight& weight, const std::vector<double>& input) {
    std::vector<double> output(weight.weight.n);
    for (int row = 0; row < weight.weight.n; ++row) {
        double sum = 0;
        for (int column = 0; column < weight.weight.k; ++column) {
            sum += quantized_weight::logical_weight_fp64(weight, row, column) * input[column];
        }
        output[row] = sum;
    }
    return output;
}

// Exact sparse mathematical witnesses, with every stored scale equal to one.
inline quantized_weight::PackedWeight native_sparse_fixture(QType type, int rows, int columns) {
    quantized_weight::PatternedWeightOptions options;
    if (type == QType::NVFP4) {
        options.weight_scale_divisor = options.input_scale_divisor = 1.0F;
    }
    auto weight = quantized_weight::make_patterned_weight(type, rows, columns, 0, options);
    std::fill_n(weight.payload.begin(), weight.code_plane_bytes, 0);
    if (type == QType::NVFP4) {
        std::fill_n(weight.payload.begin() + weight.scale_plane_offset, weight.scale_plane_bytes, 0x38);
    } else {
        for (int row = 0; row < rows; ++row) {
            quantized_weight::detail::store_u16_le(weight.payload, weight.scale_plane_offset + row * 2ULL, 0x3f80);
        }
    }
    return weight;
}

inline void native_sparse_set(quantized_weight::PackedWeight& weight, int row, int column,
                               float value) {
    if (value != 1.0F && value != -1.0F && value != 0.5F) {
        throw std::invalid_argument("native sparse witness: unsupported exact coefficient");
    }
    if (weight.weight.qtype == QType::NVFP4) {
        const unsigned code = value == 0.5F ? 1U : value == 1.0F ? 2U : 10U;
        auto& byte = weight.payload[static_cast<std::size_t>(row) * weight.weight.k / 2 + column / 2];
        const int shift = 4 * (column & 1);
        byte = (byte & ~(15U << shift)) | (code << shift);
    } else {
        weight.payload[static_cast<std::size_t>(row) * weight.weight.k + column] =
            value == 0.5F ? 0x30 : value == 1.0F ? 0x38 : 0xb8;
    }
}

} // namespace ninfer::test

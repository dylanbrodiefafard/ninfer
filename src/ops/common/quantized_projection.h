#pragma once

#include "ninfer/ops/ggml_block_linear.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// Private composition glue: the enclosing Op still owns its finite weight domain,
// mathematical seams, workspace and output/state contract.
inline bool is_native_quantized_projection(QType type) {
    return type == QType::NVFP4 || type == QType::FP8_E4M3FN_ROW_BF16S;
}

inline void validate_native_projection(const Weight& weight, int rows, int columns,
                                       const char* operation) {
    if (weight.n != rows || weight.k != columns) {
        throw std::invalid_argument("quantized projection: wrong logical shape");
    }
    if (weight.qtype == QType::NVFP4) {
        (void)validate_nvfp4_weight(weight, operation);
    } else if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        (void)validate_fp8_weight(weight, operation);
    } else {
        throw std::invalid_argument("quantized projection: unsupported native format");
    }
}

inline std::size_t projection_workspace_bytes(QType type, int rows, int columns,
                                              int maximum_tokens) {
    return is_native_quantized_projection(type)
        ? linear_workspace_capacity_bytes(type, rows, columns, LinearPolicy::A16Only,
                                           1, maximum_tokens)
        : 0;
}

inline void quantized_projection(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    if (is_native_quantized_projection(weight.qtype)) {
        // Standalone A4/A8 qualification does not qualify repeated nonlinear use.
        linear(x, weight, out, LinearPolicy::A16Only, workspace, stream);
    } else {
        ggml_block_linear(x, weight, out, stream);
    }
}

} // namespace ninfer::ops::detail

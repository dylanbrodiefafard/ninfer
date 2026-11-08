#pragma once

#include "core/tensor.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t min_width,
                                                                    std::int32_t max_width);

[[nodiscard]] std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t batch_size,
                                                                  std::int32_t min_width,
                                                                  std::int32_t max_width);

// The FP8 GDN conv snapshot serves ordinary decode: one column per request (W=1) for B=1..8 on
// the fused T=1 GEMV. The FP8 conv record keeps A16 arithmetic for every verify width: short
// recurrent blocks amplify activation-quantization error through cancellation in the four-tap
// convolution (A8 exceeded the complete public-input FP64 criterion at B=2/3 W=5 and W=11).
inline constexpr std::int32_t kFp8GdnConvMaxBatch = 8;

// Record widths whose A16 projection and convolution fuse into one SmallT launch per request;
// the other widths materialize the BF16 projection first.
[[nodiscard]] constexpr bool fp8_gdn_conv_fused_width(std::int32_t width) noexcept {
    return width >= kFp8FirstSmallT && width <= kFp8LinearSmallTMax<Fp8GdnInputGeometry> &&
           (width <= 3 || width >= 7);
}

// Fused record widths the DFlash2 verify forms launch: a chain verifies k + 1 columns and a tree
// k + 2 through kDflash2TreeMaxWidth columns.
[[nodiscard]] constexpr bool fp8_gdn_fused_record_width(std::int32_t width, bool tree) noexcept {
    const bool verified = tree ? width >= kDflash2MinTreeWidth && width <= kDflash2TreeMaxWidth
                               : width >= kFp8FirstSmallT && width <= kDflash2MaxChainWidth;
    return verified && fp8_gdn_conv_fused_width(width);
}

// The W=1 snapshot of one request.
void fp8_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                    Tensor& value, Tensor& z, cudaStream_t stream);

// The fused record of one request at a width with fp8_gdn_fused_record_width(width, tree), where
// tree is whether parent_index is non-null; other widths throw std::invalid_argument.
void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream,
                                 const Tensor* parent_index);

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, cudaStream_t stream);

void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                             Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                             cudaStream_t stream, const Tensor* parent_index);

} // namespace ninfer::ops::detail

#pragma once

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The NVFP4 GDN conv snapshot serves ordinary decode, one column per request (W=1) for
// B=1..kNvfp4GdnSnapshotMaxBatch, on the fused T=1 GEMV under every policy.
inline constexpr std::int32_t kNvfp4GdnSnapshotMaxBatch = 8;

// The NVFP4 GDN conv record serves DFlash2 verification under AllowA8 (W4A8). A8 record launches
// whose W*B fits one M48 W4A8 tile write the conv record from the MMA epilogue; larger launches
// stage the FP32 projection and run the convolution separately.
inline constexpr std::int32_t kNvfp4GdnA8FusedRecordMaxWidth  = kDflash2TreeMaxWidth;
inline constexpr std::int32_t kNvfp4GdnA8FusedRecordMaxTokens = 48;

// Record widths per verify form: a chain verifies k + 1 columns and a tree k + 2 through
// kDflash2TreeMaxWidth columns, for a draft window k in [1, kDflash2MaxDraftTokens].
[[nodiscard]] constexpr bool nvfp4_gdn_record_width_registered(std::int32_t width,
                                                               bool tree) noexcept {
    return tree ? width >= kDflash2MinTreeWidth && width <= kNvfp4GdnA8FusedRecordMaxWidth
                : width >= kNvfp4FirstA8 && width <= kDflash2MaxChainWidth;
}

// Whether some batch B in [1, kDflash2MaxBatch] of a registered width runs the fused record
// (W*B <= kNvfp4GdnA8FusedRecordMaxTokens) with this W4A8 wide-path panel count.
[[nodiscard]] constexpr bool nvfp4_gdn_a8_fused_record_panels(std::int32_t width, bool tree,
                                                              int panels) noexcept {
    if (!nvfp4_gdn_record_width_registered(width, tree)) { return false; }
    for (std::int32_t batch = 1;
         batch <= kDflash2MaxBatch && width * batch <= kNvfp4GdnA8FusedRecordMaxTokens; ++batch) {
        if (nvfp4_w4a8_wide_panels(width * batch) == panels) { return true; }
    }
    return false;
}

// The staged A8 record projection covers launches wider than the fused tile, up to the widest
// registered width at the batch bound; these are the panel counts it selects.
[[nodiscard]] constexpr bool nvfp4_gdn_a8_staged_record_panels(int panels) noexcept {
    for (std::int32_t tokens = kNvfp4GdnA8FusedRecordMaxTokens + 1;
         tokens <= kNvfp4GdnA8FusedRecordMaxWidth * kDflash2MaxBatch; ++tokens) {
        if (nvfp4_w4a8_wide_panels(tokens) == panels) { return true; }
    }
    return false;
}

// The AllowA8 record of a BF16 [5120, W, B] input: quantizes it into workspace, then runs
// nvfp4_gdn_record_a8_input_launch.
void nvfp4_gdn_record_quantized_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& conv_record, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                       cudaStream_t stream, const std::int32_t* parent_index);

// The AllowA8 record of an already quantized [5120, width*batch] activation. parent_index is
// non-null exactly for tree verification. Throws std::invalid_argument for a width/tree pair
// outside nvfp4_gdn_record_width_registered or a batch outside [1, kDflash2MaxBatch].
void nvfp4_gdn_record_a8_input_launch(Fp8A8Workspace activation, std::int32_t width,
                                      std::int32_t batch, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                      cudaStream_t stream, const std::int32_t* parent_index);

// Throws std::invalid_argument unless width is 1 and batch_size is in
// [1, kNvfp4GdnSnapshotMaxBatch].
void nvfp4_gdn_snapshot_validate(std::int32_t width, std::int32_t batch_size);

[[nodiscard]] std::size_t nvfp4_gdn_record_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream);

} // namespace ninfer::ops::detail

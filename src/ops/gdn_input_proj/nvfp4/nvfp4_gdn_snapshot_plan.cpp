#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/layout.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline constexpr std::int32_t kNvfp4RecordChannels = 10240;

} // namespace

void nvfp4_gdn_snapshot_validate(std::int32_t width, std::int32_t batch_size) {
    if (width != 1 || batch_size <= 0 || batch_size > kNvfp4GdnSnapshotMaxBatch) {
        throw std::invalid_argument("nvfp4 gdn conv snapshot: registered only for W=1, B=1..12");
    }
}

std::size_t nvfp4_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens < kNvfp4FirstA8 || max_tokens < min_tokens ||
        max_tokens > kNvfp4GdnA8FusedRecordMaxWidth || batch_size <= 0 ||
        batch_size > kDflash2MaxBatch) {
        throw std::invalid_argument("nvfp4 gdn record workspace: invalid B/T domain");
    }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("nvfp4 gdn record workspace: the record is registered for A8");
    }
    // Every width up to max_tokens fuses when the widest launch fits the single M48 tile.
    if (max_tokens * batch_size <= kNvfp4GdnA8FusedRecordMaxTokens) {
        return fp8_a8_workspace_capacity_bytes(max_tokens * batch_size, 5120);
    }
    // Mirrors the launch order: the activation is quantized before the staged FP32 projection,
    // which the A8-input form allocates alone.
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_a8_workspace(layout, max_tokens * batch_size, 5120);
    (void)layout.alloc(DType::FP32, {kNvfp4RecordChannels, max_tokens, batch_size}, 256);
    return layout.peak_bytes(1);
}

} // namespace ninfer::ops::detail

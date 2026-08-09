#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout)
    : local(backing, layout.local), boundary_local(backing, layout.boundary_local),
      full(backing, layout.full), commit_count(layout.commit_count.bind(backing)),
      target_features(layout.target_features.bind(backing)),
      feature_positions(layout.feature_positions.bind(backing)) {
    if (local.layer_count() != DFlashConfig::local_layers ||
        boundary_local.layer_count() != DFlashConfig::local_layers ||
        local.capacity() != DFlashConfig::local_capacity ||
        boundary_local.capacity() != DFlashConfig::local_capacity || full.layers() != 1 ||
        full.max_context() != layout.full.max_context || full.pool().plane_count() != 2 ||
        local.num_kv_heads() != DFlashConfig::kv_heads ||
        boundary_local.num_kv_heads() != DFlashConfig::kv_heads ||
        local.head_dim() != DFlashConfig::head_dim ||
        boundary_local.head_dim() != DFlashConfig::head_dim ||
        full.pool().plane(0).dtype != DType::BF16 ||
        full.pool().plane(0).ne[0] != DFlashConfig::head_dim ||
        full.pool().plane(0).ne[1] != kPagedKVPageSize ||
        full.pool().plane(0).ne[3] != DFlashConfig::kv_heads) {
        throw std::invalid_argument("DFlash persistent cache layout is invalid");
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

qwen3_6::PagedKVCacheView
DFlashPersistentState::full_view(const PagedKVAllocation& allocation) const {
    return full.execution_view(allocation);
}

void DFlashPersistentState::save_boundary(cudaStream_t stream) {
    boundary_local.copy_from(local, stream);
}

void DFlashPersistentState::restore_boundary(cudaStream_t stream) {
    local.copy_from(boundary_local, stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

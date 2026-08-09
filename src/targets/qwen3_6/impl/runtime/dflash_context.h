#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlashPersistentState {
    CyclicKVCache local;
    CyclicKVCache boundary_local;
    qwen3_6::PagedKVCache full;
    Tensor commit_count;
    Tensor target_features;
    Tensor feature_positions;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView full_view(const PagedKVAllocation& allocation) const;
    void save_boundary(cudaStream_t stream);
    void restore_boundary(cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

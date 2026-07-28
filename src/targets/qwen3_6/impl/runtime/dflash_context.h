#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlashPersistentState {
    KVCache local;
    KVCache boundary_local;
    KVCache full;
    Tensor commit_count;
    Tensor target_features;
    Tensor feature_positions;

    DFlashPersistentState() = default;
    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] KVCacheLayerView full_layer() const;
    void save_boundary(cudaStream_t stream) const;
    void restore_boundary(cudaStream_t stream) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

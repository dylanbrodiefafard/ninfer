#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout)
    : local(backing, layout.local), staging_local(backing, layout.staging_local),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    const bool local_ok = local.layer_count() == DFlashConfig::local_layers &&
                          staging_local.layer_count() == DFlashConfig::local_layers &&
                          local.capacity() == DFlashConfig::local_capacity &&
                          staging_local.capacity() == DFlashConfig::local_capacity &&
                          local.num_kv_heads() == DFlashConfig::kv_heads &&
                          staging_local.num_kv_heads() == DFlashConfig::kv_heads &&
                          local.head_dim() == DFlashConfig::head_dim &&
                          staging_local.head_dim() == DFlashConfig::head_dim &&
                          staging_local.lane_capacity() == 1;
    if (!local_ok) { throw std::invalid_argument("DFlash persistent cache layout is invalid"); }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

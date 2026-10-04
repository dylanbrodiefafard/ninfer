#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer {

struct GdnHistorySpec {
    std::int32_t layers;
    std::int32_t slots;
    std::int32_t width;
    std::int32_t capacity;
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t key_dim;
    std::int32_t value_dim;
};

struct GdnHistoryLayout {
    GdnHistorySpec spec;
    TensorRegion key;
    TensorRegion innovation;
    TensorRegion alpha;
    TensorRegion counts;
    TensorRegion provisional_key;
    TensorRegion provisional_innovation;
    TensorRegion provisional_alpha;
};

[[nodiscard]] GdnHistoryLayout plan_gdn_history(LayoutBuilder& builder, const GdnHistorySpec& spec);

struct GdnHistoryLayer {
    Tensor key;                    // FP32 [key_dim,qk_heads,capacity,slots].
    Tensor innovation;             // FP32 [value_dim,value_heads,capacity,slots].
    Tensor alpha;                  // FP32 [value_heads,capacity,slots].
    Tensor counts;                 // I32 [slots], shared by all layers of each accepted sequence.
    Tensor provisional_key;        // FP32 [key_dim,qk_heads,width,active_rows].
    Tensor provisional_innovation; // FP32 [value_dim,value_heads,width,active_rows].
    Tensor provisional_alpha;      // FP32 [value_heads,width,active_rows].
};

// Non-owning planes over startup-reserved arena storage. The outer index is layer*slots+slot
// for retained planes and layer*slots+record_row for provisional planes. No plane is allocated
// by this object; mathematical interpretation and count publication belong to the closed Op.
struct GdnHistory {
    Tensor key;
    Tensor innovation;
    Tensor alpha;
    Tensor counts;
    Tensor provisional_key;
    Tensor provisional_innovation;
    Tensor provisional_alpha;
    GdnHistorySpec spec;

    GdnHistory(DeviceSpan backing, const GdnHistoryLayout& layout);
    [[nodiscard]] GdnHistoryLayer layer(std::int32_t index, std::int32_t row_begin,
                                        std::int32_t rows) const;
};

} // namespace ninfer

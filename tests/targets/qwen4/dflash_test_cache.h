#pragma once

#include "targets/qwen4/dflash.h"

namespace ninfer::test::qwen4 {

// Real page-pool owner for bounded component qualification. Production ownership lives in
// NativeDraftRuntime; tests reserve all their small cache pages explicitly before execution.
class DFlashTestCache {
    static PagedKVPoolLayout layout(int capacity,int slots) {
        LayoutBuilder b;
        PagedKVPoolSpec spec{pages_for_tokens(capacity)*std::uint32_t(slots),pages_for_tokens(capacity),
                            slots,PagedKVPlaneOrder::HeadMajor,{}};
        for(int i=0;i<10;++i) spec.planes.push_back({DType::BF16,256,2});
        return plan_paged_kv_pool(b,spec);
    }
    PagedKVPoolLayout layout_;
    DeviceBuffer backing_;
    PagedKVPool pool_;
    std::array<PagedKVAllocation,4> allocations_;
public:
    DFlashTestCache(int capacity,int slots,cudaStream_t stream)
        :layout_(layout(capacity,slots)),
         backing_(layout_.block_tables.region.offset+layout_.block_tables.region.bytes),
         pool_({backing_.p,backing_.bytes},layout_) {
        for(int i=0;i<slots;++i) {
            allocations_[i]=pool_.reserve(pages_for_tokens(capacity));
            allocations_[i].bind_row(i,stream);allocations_[i].materialize_tokens(capacity,stream);
        }
    }
    std::array<PagedKVBatchLayerView,5> views() const {
        std::array<PagedKVBatchLayerView,5> result;
        for(int i=0;i<5;++i) result[i]={.k_pages=pool_.plane(i*2),.v_pages=pool_.plane(i*2+1),
            .block_tables=pool_.block_tables(),.head_dim=256,.num_kv_heads=2,.dtype=DType::BF16};
        return result;
    }
};
} // namespace ninfer::test::qwen4

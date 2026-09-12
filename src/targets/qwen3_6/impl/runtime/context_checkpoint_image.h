#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/cache_cuda_event.h"
#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct ContextCheckpointHead {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    std::shared_ptr<PinnedHostBuffer> conv;
    std::shared_ptr<PinnedHostBuffer> recurrent;
    std::shared_ptr<PinnedHostBuffer> hidden;
    std::shared_ptr<PinnedHostBuffer> dflash;
    cudaEvent_t copies_done = nullptr;

    ContextCheckpointHead() = default;
    ContextCheckpointHead(const ContextCheckpointHead&)            = delete;
    ContextCheckpointHead& operator=(const ContextCheckpointHead&) = delete;
    ContextCheckpointHead(ContextCheckpointHead&& other) noexcept { *this = std::move(other); }
    ContextCheckpointHead& operator=(ContextCheckpointHead&& other) noexcept {
        if (this == &other) { return *this; }
        release();
        frontier     = other.frontier;
        hash         = other.hash;
        kind         = other.kind;
        conv         = std::move(other.conv);
        recurrent    = std::move(other.recurrent);
        hidden       = std::move(other.hidden);
        dflash       = std::move(other.dflash);
        copies_done  = other.copies_done;
        other.copies_done = nullptr;
        other.frontier    = 0;
        other.kind        = ContextCheckpointKind::Ladder;
        return *this;
    }
    ~ContextCheckpointHead() { release(); }

    inline static std::atomic<bool> fail_next_event_allocation{false};
    static void test_fail_next_copy_event_allocation() noexcept {
        fail_next_event_allocation.store(true, std::memory_order_release);
    }
    static bool test_copy_event_allocation_pending() noexcept {
        return fail_next_event_allocation.load(std::memory_order_acquire);
    }

    void prepare_copy_event() {
        if (copies_done == nullptr) {
            if (fail_next_event_allocation.exchange(false, std::memory_order_acq_rel)) {
                check_cache_cuda_event_allocation(cudaErrorMemoryAllocation);
            }
            create_cache_cuda_event(&copies_done, cudaEventDisableTiming);
        }
    }

    void wait_copies() const {
        if (copies_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(copies_done)); }
    }

    void release() noexcept {
        if (copies_done != nullptr) {
            (void)cudaEventSynchronize(copies_done);
            (void)cudaEventDestroy(copies_done);
            copies_done = nullptr;
        }
        conv.reset();
        recurrent.reset();
        hidden.reset();
        dflash.reset();
    }
};

// This cache keeps only its startup-reserved number of retired images. Dropping
// an excess image still waits for its final DMA owner through release().
inline void recycle_checkpoint_image(std::vector<ContextCheckpointHead>& pool,
                                     ContextCheckpointHead&& head) noexcept {
    if (pool.size() == pool.capacity()) {
        head.release();
        return;
    }
    head.frontier = 0;
    head.hash = {};
    head.kind = ContextCheckpointKind::Ladder;
    pool.push_back(std::move(head));
}

} // namespace ninfer::targets::qwen3_6::detail

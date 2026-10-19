#pragma once

#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/cache_cuda_event.h"
#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Page alignment of every checkpoint image region, so the disk tier's direct (O_DIRECT) state
// reads land in the images without a bounce buffer.
inline constexpr std::size_t kCheckpointImageAlignment = 4096;

// One region of the Program's checkpoint-image slab: pinned, prefaulted, and carved at startup.
// Non-owning; the slab outlives every head that views it.
class PinnedImage {
public:
    PinnedImage() noexcept = default;

    PinnedImage(void* data, std::size_t bytes) noexcept : data_(data), bytes_(bytes) {}

    [[nodiscard]] void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    explicit operator bool() const noexcept { return data_ != nullptr; }

private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
};

struct ContextCheckpointHead {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
    PinnedImage conv;
    PinnedImage recurrent;
    PinnedImage hidden;
    PinnedImage dflash;
    // Last asynchronous reader or writer of the images (D2H, H2D, or host-copy stream). Created
    // with the head at startup; every new owner of the images orders itself after it.
    cudaEvent_t copies_done = nullptr;

    ContextCheckpointHead()                                        = default;
    ContextCheckpointHead(const ContextCheckpointHead&)            = delete;
    ContextCheckpointHead& operator=(const ContextCheckpointHead&) = delete;

    ContextCheckpointHead(ContextCheckpointHead&& other) noexcept { *this = std::move(other); }

    ContextCheckpointHead& operator=(ContextCheckpointHead&& other) noexcept {
        if (this == &other) { return *this; }
        release();
        frontier       = other.frontier;
        hash           = other.hash;
        kind           = other.kind;
        conv           = std::exchange(other.conv, {});
        recurrent      = std::exchange(other.recurrent, {});
        hidden         = std::exchange(other.hidden, {});
        dflash         = std::exchange(other.dflash, {});
        copies_done    = std::exchange(other.copies_done, nullptr);
        other.frontier = 0;
        other.kind     = ContextCheckpointKind::Ladder;
        return *this;
    }

    ~ContextCheckpointHead() { release(); }

    void prepare_copy_event() {
        if (copies_done == nullptr) {
            create_cache_cuda_event(&copies_done, cudaEventDisableTiming);
        }
    }

    // Teardown only: waits for the images' last DMA or host-copy owner before the slab can go.
    void release() noexcept {
        if (copies_done != nullptr) {
            (void)cudaEventSynchronize(copies_done);
            (void)cudaEventDestroy(copies_done);
            copies_done = nullptr;
        }
        conv      = {};
        recurrent = {};
        hidden    = {};
        dflash    = {};
    }
};

// Every pooled head is carved at startup and returns here, so the pool never exceeds the
// capacity it reserved and recycling never allocates. Recycling does not wait: the next owner
// orders its first copy after copies_done.
inline void recycle_checkpoint_image(std::vector<ContextCheckpointHead>& pool,
                                     ContextCheckpointHead&& head) noexcept {
    if (pool.size() == pool.capacity()) {
        // Unreachable while every head comes from the pool; keeps recycling allocation-free.
        head.release();
        return;
    }
    head.frontier = 0;
    head.hash     = {};
    head.kind     = ContextCheckpointKind::Ladder;
    pool.push_back(std::move(head));
}

} // namespace ninfer::targets::qwen3_6::detail

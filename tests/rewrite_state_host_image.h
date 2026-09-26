#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace ninfer::test {

// Host rewrite-checkpoint image as the Qwen3.6 Program keeps it per lane: the GDN slot in its
// LinearAttentionStatePool host-image layout and, optionally, one CyclicKVCache lane image.
struct RewriteStateHostImage {
    std::vector<unsigned char> conv;
    std::vector<unsigned char> recurrent;
    std::vector<unsigned char> dflash;

    static RewriteStateHostImage sized(const LinearAttentionStatePool& gdn,
                                       const CyclicKVCache* cyclic = nullptr) {
        RewriteStateHostImage image;
        image.conv.assign(gdn.conv_host_image_bytes(), 0);
        image.recurrent.assign(gdn.recurrent_host_image_bytes(), 0);
        if (cyclic != nullptr) { image.dflash.assign(cyclic->lane_host_bytes(), 0); }
        return image;
    }

    void fill(unsigned char value) {
        std::fill(conv.begin(), conv.end(), value);
        std::fill(recurrent.begin(), recurrent.end(), value);
        std::fill(dflash.begin(), dflash.end(), value);
    }

    // Snapshot device slot `slot` (and `cyclic` lane `lane`) as a captured checkpoint image.
    static RewriteStateHostImage packed(const LinearAttentionStatePool& gdn, std::int32_t slot,
                                        const CyclicKVCache* cyclic = nullptr,
                                        std::int32_t lane = 0, cudaStream_t stream = nullptr) {
        RewriteStateHostImage image = sized(gdn, cyclic);
        gdn.pack_slot_to_host(slot, image.conv.data(), image.recurrent.data(), stream);
        if (cyclic != nullptr) { cyclic->copy_lane_to_host(lane, image.dflash.data(), stream); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return image;
    }

    // Install the image into device slot `slot` (and `cyclic` lane `lane`) so device-side checks
    // can compare a restored checkpoint.
    void unpack(LinearAttentionStatePool& gdn, std::int32_t slot, CyclicKVCache* cyclic = nullptr,
                std::int32_t lane = 0, cudaStream_t stream = nullptr) const {
        gdn.unpack_slot_from_host(slot, conv.data(), recurrent.data(), stream);
        if (cyclic != nullptr) { cyclic->copy_lane_from_host(dflash.data(), lane, stream); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] targets::qwen3_6::detail::RewriteStateHostSource source() const noexcept {
        return {conv.data(), recurrent.data(), dflash.empty() ? nullptr : dflash.data()};
    }

    [[nodiscard]] targets::qwen3_6::detail::RewriteStateHostTarget target() noexcept {
        return {conv.data(), recurrent.data(), dflash.empty() ? nullptr : dflash.data()};
    }
};

} // namespace ninfer::test

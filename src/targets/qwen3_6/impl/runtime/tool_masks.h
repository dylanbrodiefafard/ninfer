#pragma once

#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <ninfer/targets/qwen3_6/frontend.h>

#include <cuda_runtime_api.h>
#include <exception>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// Program-owned, startup-sized exchange. Its address and pinned/device buffers
// remain stable for every captured graph. The executor changes bindings only at
// synchronized round boundaries; the CUDA host node makes no CUDA API calls.
class ToolMaskExchange {
public:
    ToolMaskExchange(Tensor masks, Tensor sampling);
    void bind(std::span<const OutputSession* const> outputs,
              std::span<const ops::SamplingConfig> sampling);
    // Ordinary/prefill root sampling; called at a synchronized CPU boundary.
    [[nodiscard]] ops::SamplingConfig root(std::size_t row, cudaStream_t stream);
    // Called inside the speculative graph after ids/parents are constructed and
    // before target argmax/accept. The returned configs are target-only: draft
    // proposal sampling must continue to use its own unmasked configs.
    [[nodiscard]] const ops::SamplingConfig* enqueue(
        const Tensor& ids, const Tensor* parents, const Tensor& valid_columns,
        cudaStream_t stream);
    void rethrow_error() const;

private:
    static void CUDART_CB match(void* opaque) noexcept;
    void fill();
    [[nodiscard]] std::uint32_t* host_mask(std::size_t row) const;
    [[nodiscard]] const std::uint32_t* device_mask(std::size_t row) const;

    Tensor masks_;
    Tensor sampling_;
    const std::size_t width_;
    const std::size_t capacity_;
    PinnedHostBuffer host_masks_;
    PinnedHostBuffer host_sampling_;
    PinnedHostBuffer host_ids_;
    PinnedHostBuffer host_parents_;
    PinnedHostBuffer host_counts_;
    std::vector<const OutputSession*> outputs_;
    std::vector<ops::SamplingConfig> configs_;
    std::exception_ptr error_;
    bool tree_ = false;
};

} // namespace ninfer::targets::qwen3_6

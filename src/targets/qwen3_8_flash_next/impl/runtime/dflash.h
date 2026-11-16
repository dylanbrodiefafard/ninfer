#pragma once

#include "model_view.h"
#include "runtime/state.h"
#include "runtime/cuda_event.h"

#include "core/arena.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <span>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next {

// The target-owned five-layer drafter schedule. State owns its cyclic context and pending target
// features; this object owns startup-fixed transient capacity only. Model and State outlive it.
// Calls enqueue on the owning stream, including during graph capture. Inputs remain valid until
// stream completion; arguments obey the corresponding fixed-shape Op contracts. Errors throw.
class Dflash {
public:
    Dflash(const ModelView& model, State& state, cudaStream_t stream);
    ~Dflash();
    Dflash(const Dflash&)            = delete;
    Dflash& operator=(const Dflash&) = delete;

    // features [12800,W,B], positions [W,B], counts/lanes [B]. Appends exactly each row's prefix;
    // W<=2048. Context is projected independently for each layer, not propagated through it.
    void append(const Tensor& features, const Tensor& positions, const Tensor& counts,
                const Tensor& lanes);

    // Gathers each selected lane's fixed-width pending feature block and appends its counted
    // prefix. Metadata is supplied by the enclosing round's fixed ingress frame.
    void append_pending(const Tensor& positions, const Tensor& counts, const Tensor& lanes);

    void flush(std::span<const std::int32_t> lanes);
    void append_prefill(std::int32_t lane, const Tensor& features);
    void save_decode_features(std::span<const std::int32_t> lanes, const Tensor& features);
    void backup_context(const Tensor& slots, Tensor& keys, Tensor& values, std::int32_t rows);
    void restore_context(const Tensor& slots, const Tensor& keys, const Tensor& values,
                         std::int32_t row, std::int32_t rows);

    // width=K+1 in [2,4], anchors/frontiers/valid_columns/lanes [B], configs device [B].
    // Outputs drafts [K,B], selector_ids/selector_q [16,K,B]. Head padding is excluded.
    void propose(std::int32_t width, const Tensor& anchors, const Tensor& frontiers,
                 const Tensor& valid_columns, const Tensor& lanes,
                 const ops::SamplingConfig* configs, Tensor& drafts, Tensor& selector_ids,
                 Tensor& selector_q);

    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return work_.capacity() + transfers_.capacity() + inputs_.capacity();
    }

private:
    const ModelView& model_;
    const DflashWeights& weights_;
    State& state_;
    cudaStream_t stream_;
    WorkspaceArena work_;
    DeviceArena transfers_;
    Tensor proposal_residual_, head_logits_;

    struct HostInputs {
        alignas(16) std::array<std::int32_t, kMaxLanes> lanes, counts;
        alignas(16) std::array<std::int32_t, kDflashWindow> positions;
    };

    DeviceArena inputs_;
    Tensor input_frame_;
    PinnedHostBuffer input_staging_;
    CudaEvent inputs_uploaded_;
    [[nodiscard]] HostInputs& host_inputs();
    void upload_inputs();
};

} // namespace ninfer::targets::qwen3_8_flash_next

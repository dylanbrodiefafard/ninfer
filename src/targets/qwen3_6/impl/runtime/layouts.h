#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout = TensorRegion;

struct DFlashPersistentLayout {
    CyclicKVCacheLayout local;
    CyclicKVCacheLayout boundary_local;
    qwen3_6::PagedKVCacheLayout full;
    TensorLayout commit_count;
    TensorLayout target_features;
    TensorLayout feature_positions;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return local.payload_bytes() + boundary_local.payload_bytes() + full.payload_bytes();
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout boundary_hidden;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill    = 0;
    std::size_t ordinary_round  = 0;
    std::size_t mtp_prefill     = 0;
    std::size_t mtp_round       = 0;
    std::size_t dflash_context  = 0;
    std::size_t dflash_proposal = 0;
    std::size_t dflash_round    = 0;
    std::size_t vision_encode   = 0;
    std::size_t capacity        = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    int device          = 0;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_allowance_bytes            = 0;
    std::size_t device_reservation_bytes         = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

void validate_target_options(DeviceContext& device, const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl> plan_sequence_impl(DeviceContext& device,
                                                                   const EngineOptions& options,
                                                                   WeightsProfile weights_profile);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

using qwen3_6::PreparedPromptData;
using qwen3_6::PromptModality;

struct State {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    qwen3_6::PagedKVCacheView text_kv;
    qwen3_6::PagedKVCacheView mtp_kv;
    qwen3_6::PagedKVCacheView dflash_full_kv;
    DFlashPersistentState* dflash;
    LinearAttentionStatePool& linear_attention;
    qwen3_6::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    ProposalHead proposal_head;
    Tensor* tail_hidden;
    Tensor* boundary_hidden;
    std::int32_t linear_state_base                              = 0;
    std::int32_t linear_state_capacity                          = 0;
    const qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress         = nullptr;
    Tensor* continuation_hidden_store                           = nullptr;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

using GraphPrepare = std::function<void()>;

void configure_text_card(TextContext& card, const State& state);

[[nodiscard]] bool prefill_text(State& state, std::span<const TokenId> ids,
                                std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(State& state, std::span<const TokenId> ids,
                                                    std::optional<std::uint32_t> snapshot_boundary,
                                                    bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(State& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> snapshot_boundary, bool finalize_at_end);

struct MultimodalPrefillResult {
    bool mtp_prepared                = false;
    std::uint32_t final_chunk_tokens = 0;
    double vision_seconds            = 0.0;
};

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

[[nodiscard]] MultimodalPrefillResult
prefill_multimodal(State& state, const PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                   runtime::TransientRegion transient,
                   std::optional<std::uint32_t> snapshot_boundary, bool prepare_mtp,
                   const MtpBridgeInput* mtp_bridge);

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void target_verify(TextContext& card, State& state, const Tensor& ids, const Tensor& positions,
                   ops::GqaExecutionEnvelope envelope);
void speculative_verify_and_accept(State& state, TextContext& card, std::uint32_t draft_window,
                                   ops::GqaExecutionEnvelope target_envelope);
void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal, const Tensor* next_embedding = nullptr);

// Executes an exact one-token target step through the verify schedule. The resulting target hidden
// is in io.verify_hidden[:,0], the sampled token is in io.token, and the configured Linear
// Attention snapshot destination is the resulting state.
void warm_capture_ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                                 const GraphPrepare& prepare, DecodeGraphDefinition& definition);
void ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                    DecodeGraphExecutable* executable);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void warm_capture_ordinary_decode_batch(State& state, std::int32_t batch_size,
                                        ops::GqaExecutionEnvelope envelope,
                                        const GraphPrepare& prepare,
                                        DecodeGraphDefinition& definition);
void ordinary_decode_batch(State& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable);

// Executes one fixed-k MTP synchronization/proposal round around the common speculative
// verification transaction. The number of licensed tokens and their values are written to
// io.speculative.
void warm_capture_mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
                            const GraphPrepare& prepare, DecodeGraphDefinition& definition);
void mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
               DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(State& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(State& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_count,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_propose(State& state, std::uint32_t k, DFlashEnvelopes envelopes);
void warm_capture_dflash_initial_round(State& state, std::uint32_t k,
                                       ops::GqaExecutionEnvelope target_envelope,
                                       const GraphPrepare& prepare,
                                       DecodeGraphDefinition& definition);
void dflash_initial_round(State& state, std::uint32_t k, ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraphExecutable* executable);
void warm_capture_dflash_steady_round(State& state, std::uint32_t k, DFlashEnvelopes envelopes,
                                      ops::GqaExecutionEnvelope target_envelope,
                                      const GraphPrepare& prepare,
                                      DecodeGraphDefinition& definition);
void dflash_steady_round(State& state, std::uint32_t k, DFlashEnvelopes envelopes,
                         ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule

#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/layouts.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"
#include "targets/qwen3/impl/runtime/linear_state_slots.h"
#include "targets/qwen3/impl/runtime/prefill_schedule.h"
#include "targets/qwen3/impl/runtime/context_checkpoint.h"
#include "targets/qwen3/impl/runtime/prompt_embedding_staging.h"
#include "targets/qwen3/impl/runtime/vision_context.h"
#include "targets/qwen3/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/swa.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ninfer/types.h"

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {
// NOLINTNEXTLINE(misc-anonymous-namespace-in-header): single-TU fragment (runtime.hip)
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

// Conservative bounds for uncalibrated startup selectors. Every captured profile
// owns an immutable executable; runtime never instantiates or updates graphs.
constexpr std::size_t kR9700GraphExecutableBytes            = 4ULL * kMiB;
constexpr std::size_t kR9700PersistentDecodeDefinitionBytes = 6ULL * kMiB;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

struct TextAttentionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const bool context_checkpoint_staging = plan.features.mtp() || plan.features.dflash();
    const std::int32_t linear_state_slots =
        LinearStateSlots::state_slot_count(plan.max_concurrency, context_checkpoint_staging);
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3::plan_decoder_state(
        builder, qwen3::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads,
                     .attention_head_dim        = TextConfig::head_dim,
                     .value_group               = kR9700TextKVValueGroup,
                     .text_plane_layouts        = kR9700TextKVPlaneLayouts,
                     .mtp_plane_layouts         = kR9700TextKVPlaneLayouts,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     .linear_attention =
                         {
                             .layers         = TextConfig::gdn_layers(),
                             .conv_channels  = TextConfig::convolution_dim,
                             .conv_width     = TextConfig::gdn_conv_state_width,
                             .value_heads    = TextConfig::gdn_value_heads,
                             .value_head_dim = TextConfig::gdn_value_head_dim,
                             .key_head_dim   = TextConfig::gdn_key_head_dim,
                             .slot_count     = linear_state_slots,
                             .conv_dtype     = DType::BF16,
                         },
                 });
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{
                .layers          = TextConfig::gdn_layers(),
                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                .width = static_cast<std::int32_t>(plan.features.dflash() ? plan.dflash_verify_width
                                                                          : plan.draft_window + 1U),
                .conv_channels = TextConfig::convolution_dim,
                .qk_heads      = TextConfig::gdn_key_heads,
                .value_heads   = TextConfig::gdn_value_heads,
                .key_dim       = TextConfig::gdn_key_head_dim,
                .value_dim     = TextConfig::gdn_value_head_dim,
            });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            dflash.local = plan_cyclic_kv_cache(builder, DFlashConfig::local_layers,
                                                DFlashConfig::local_capacity,
                                                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                                                static_cast<std::int32_t>(plan.max_concurrency));
            dflash.staging_local = plan_cyclic_kv_cache(
                builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                DFlashConfig::kv_heads, DFlashConfig::head_dim, 1);
            if constexpr (DFlashConfig::full_layers > 0) {
                PagedKVPoolSpec full_pool{
                    .page_group_count      = physical_pages,
                    .logical_page_capacity = logical_pages,
                    .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                    .planes =
                        {
                            {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256,
                             PagedKVPlaneOrder::HeadMajor},
                            {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256,
                             PagedKVPlaneOrder::HeadMajor},
                        },
                };
                dflash.full = DFlashPersistentLayout::FullBF16Layout{
                    .storage     = plan_paged_kv_pool(builder, full_pool),
                    .layers      = 1,
                    .max_context = plan.capacity,
                    .kv_heads    = DFlashConfig::kv_heads,
                    .head_dim    = DFlashConfig::head_dim,
                };
            }
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {DFlashConfig::feature_rows, static_cast<std::int32_t>(plan.dflash_verify_width),
                 static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
            dflash.image_completion =
                add_tensor(builder, DType::I32, {2}, "DFlash activation image completion");
        }
    }

    out.round = qwen3::begin_round_state_layout(
        builder, qwen3::RoundStateSpec{.hidden         = TextConfig::hidden,
                                       .output_rows    = TextConfig::output_rows,
                                       .batch_capacity = plan.max_concurrency,
                                       .draft_window   = plan.draft_window,
                                       .dflash_verify_width =
                                           plan.features.dflash() ? plan.dflash_verify_width : 0U,
                                       .enable_mtp    = plan.features.mtp(),
                                       .enable_dflash = plan.features.dflash()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    // A chunk's prompt window carries one id past the chunk for the shifted MTP embedding.
    out.prompt_embedding_ids = checked_i32(static_cast<std::uint64_t>(effective_prefill_chunk) + 1U,
                                           "prompt embedding window exceeds int32");
    out.prompt_embedding_image =
        builder.add(static_cast<std::size_t>(PromptEmbeddingStaging::capacity_bytes(
                        Variant::token_embedding_qtype(plan.weights_profile), TextConfig::hidden,
                        out.prompt_embedding_ids)),
                    kArenaAlign, "prompt embedding staging");
    qwen3::complete_round_state_layout(builder, out.round);
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    const auto tool_width =
        static_cast<std::int32_t>(std::max({1U, plan.draft_window + 1U, plan.dflash_verify_width}));
    out.grammar_masks           = add_tensor(builder, DType::I32,
                                             {(TextConfig::token_domain + 31) / 32, tool_width,
                                              static_cast<std::int32_t>(plan.max_concurrency)},
                                             "token grammar token masks");
    out.grammar_sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "token grammar target sampling config");
    out.tail_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "tail hidden");
    out.rewrite_checkpoint_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "rewrite checkpoint hidden");
    if (context_checkpoint_staging) {
        out.staging_hidden = add_tensor(builder, DType::BF16, {TextConfig::hidden, 1},
                                        "context checkpoint staging hidden");
    }
    const std::uint64_t ordinary_tokens = plan.max_concurrency;
    const std::uint64_t mtp_tokens =
        static_cast<std::uint64_t>(plan.max_concurrency) * (plan.draft_window + 1U);
    const std::uint64_t dflash_tokens =
        static_cast<std::uint64_t>(plan.max_concurrency) *
        (plan.features.dflash() ? plan.dflash_verify_width : plan.draft_window + 1U);
    const std::uint32_t maximum_graph_tokens = static_cast<std::uint32_t>(
        checked_i32(std::max({ordinary_tokens, mtp_tokens, dflash_tokens}),
                    "selected linear graph width exceeds int32"));
    const std::size_t linear_execution_bytes = Variant::execution_state_capacity_bytes(
        plan.weights_profile, static_cast<std::uint32_t>(effective_prefill_chunk),
        maximum_graph_tokens);
    if (linear_execution_bytes != 0U) {
        out.linear_execution =
            builder.add(linear_execution_bytes, kArenaAlign, "serialized linear activation state");
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk         = static_cast<std::int32_t>(chunk_u32);
    const auto drafts        = static_cast<std::int32_t>(plan.draft_window);
    const auto verify        = drafts + 1;
    const auto dflash_verify = static_cast<std::int32_t>(
        plan.features.dflash() ? plan.dflash_verify_width : plan.draft_window + 1U);
    const bool dflash_tree = plan.features.dflash() &&
                             dflash_uses_tree_verify(plan.draft_window, plan.dflash_verify_width);
    const TextAttentionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                            std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };

    // The uniform verify sequences of a Text body when they are not its batch_size x max_width
    // panel (a mixed round's verify part beside the prefill owner's columns).
    struct VerifySequences {
        std::int32_t count = 0;
        std::int32_t width = 0;
    };

    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width, TextAttentionEnvelope envelope,
                                     bool tree_verify                 = false,
                                     VerifySequences verify_sequences = {}) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
        if (phase == qwen3::TextPhase::Prefill) {
            scratch(layout, Variant::text_prefill_attention_workspace_capacity_bytes(
                                max_width, envelope.max_visible_keys));
        } else
#endif
        {
            scratch(layout, Variant::full_attention_workspace_capacity_bytes(
                                max_width, envelope.max_visible_keys, tree_verify));
        }
        // The batched sequence launch holds one workspace slice per sequence; tree verification
        // attends one sequence at a time.
        const VerifySequences sequences =
            verify_sequences.count != 0 ? verify_sequences : VerifySequences{batch_size, max_width};
        if (!tree_verify) {
            scratch(layout, Variant::full_attention_sequences_workspace_capacity_bytes(
                                sequences.count, sequences.width, envelope.max_visible_keys));
        }
        (void)min_width;
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::ReplayRecord && plan.features.dflash() && batch_size > 1 &&
            !dflash_tree && Variant::gdn_record_fold_selected(last)) {
            (void)workspace_recipe::gdn_record_fold<TextConfig>(layout, dflash_verify, batch_size);
        }
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout, ops::gated_delta_net_workspace_capacity_bytes(last, true));
        }
        if (path == GdnWorkspacePath::ReplayRecord) {
            if (dflash_tree) {
                // Nested: gdn_mix scopes the fold alloc before gdn_normalized_output.
                scratch(layout, ops::gated_delta_net_replay_record_workspace_capacity_bytes(
                                    TextConfig::gdn_value_heads, batch_size, max_width));
            }
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width, TextAttentionEnvelope envelope,
                                 bool tree_verify = false, VerifySequences verify_sequences = {}) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope,
                        tree_verify, verify_sequences);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
        scratch(layout, Variant::mtp_fc_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   TextAttentionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, Variant::full_attention_workspace_capacity_bytes(
                            tokens, envelope.max_visible_keys, false));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_output_workspace_capacity_bytes(tokens, tokens));
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   TextAttentionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, true);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::full_attention_workspace_capacity_bytes(
                            1, text_envelope.max_visible_keys, false));
        scratch(layout, Variant::mtp_attention_output_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1, 1,
                chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                    1, chunk, text_envelope);
        // Shifted prompt embeddings, gathered by the prefill from its host-staged rows.
        matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
        if (plan.features.vision) {
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            target_body(target, aggregate, aggregate, qwen3::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout, Variant::full_attention_workspace_capacity_bytes(
                                    width, text_envelope.max_visible_keys, false));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_output_workspace_capacity_bytes(tokens, tokens));
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            const std::size_t mtp_peak = std::max({out.mtp_round, finish(target), finish(alignment),
                                                   finish(ar), finish(proposal), batch_accept});
            out.mtp_round              = mtp_peak;
            if (plan.adaptive_draft) {
                std::size_t panel_bytes = 0;
                for (const std::uint32_t k : plan.captured_ks) {
                    if (k == 0 || k >= plan.draft_window) { continue; }
                    const std::int32_t wk        = static_cast<std::int32_t>(k) + 1;
                    const std::int32_t compact_t = batch * wk;
                    WorkspaceLayoutBuilder panels;
                    matrix(panels, DType::I32, static_cast<std::int32_t>(k), batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::I32, wk, batch);
                    matrix(panels, DType::BF16, TextConfig::hidden, compact_t);
                    matrix(panels, DType::BF16, TextConfig::output_rows, compact_t);
                    matrix(panels, DType::BF16, TextConfig::hidden, compact_t);
                    panel_bytes = std::max(panel_bytes, finish(panels));
                    WorkspaceLayoutBuilder compact;
                    matrix(compact, DType::I32, static_cast<std::int32_t>(k), batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::I32, wk, batch);
                    matrix(compact, DType::BF16, TextConfig::hidden, compact_t);
                    matrix(compact, DType::BF16, TextConfig::output_rows, compact_t);
                    matrix(compact, DType::BF16, TextConfig::hidden, compact_t);
                    target_body(compact, compact_t, compact_t, qwen3::TextPhase::Verify,
                                GdnWorkspacePath::ReplayRecord, batch, wk, wk, text_envelope);
                    scratch(compact, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                         TextConfig::token_domain, static_cast<std::int32_t>(k),
                                         static_cast<std::int32_t>(k), batch, batch));
                    out.mtp_round = std::max(out.mtp_round, finish(compact));
                }
                // Compact panels stay live under nested mtp_forward (LLD Capture/run).
                if (panel_bytes != 0) {
                    out.mtp_round = std::max(
                        out.mtp_round, checked_add(mtp_peak, panel_bytes, "adaptive MTP compact"));
                }
            }
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const QType dflash_matrix_qtype = Variant::dflash_matrix_qtype(plan.weights_profile);
            // A projection without the serialized execution state quantizes into the arena.
            const auto dflash_linear_workspace = [&](std::int32_t tokens) {
                return std::max({ops::linear_workspace_capacity_bytes(dflash_matrix_qtype, tokens,
                                                                      DFlashConfig::hidden),
                                 ops::linear_workspace_capacity_bytes(dflash_matrix_qtype, tokens,
                                                                      DFlashConfig::query_size),
                                 ops::linear_workspace_capacity_bytes(dflash_matrix_qtype, tokens,
                                                                      DFlashConfig::intermediate)});
            };
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            // Sized at the widest captured chain; SWA split scratch is not monotone in width, so it
            // covers every narrower adaptive width at the same batch.
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                // Compact [W,B] I32 panels when tree verify stores a wider round-state width.
                matrix(layout, DType::I32, width, batch);
                matrix(layout, DType::I32, width, batch);
                matrix(layout, DType::I32, std::max(width - 1, 1), batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                const auto drafter_layers = [&] {
                    auto layers = layout.scope();
                    (void)workspace_recipe::dflash_layer_carry<DFlashConfig>(layout, tokens);
                    {
                        auto attention = layout.scope();
                        (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                        scratch(layout, ops::swa_workspace_capacity_bytes({0, plan.capacity}, 1,
                                                                          width, batch));
                        scratch(layout, dflash_linear_workspace(tokens));
                    }
                    {
                        auto mlp = layout.scope();
                        (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                        scratch(layout, dflash_linear_workspace(tokens));
                    }
                };
                drafter_layers();
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                }
                scratch(layout, ops::dflash2_path_select_workspace_capacity_bytes(
                                    dflash_matrix_qtype, drafts, drafts, batch));
                scratch(layout, static_cast<std::size_t>(ops::kDflash2PathSelectTopK) *
                                    static_cast<std::size_t>(drafts) *
                                    static_cast<std::size_t>(batch) *
                                    (sizeof(std::int32_t) + sizeof(float)));
                if constexpr (DFlashConfig::two_block_first > 0) { drafter_layers(); }
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = dflash_verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, dflash_verify, dflash_verify,
                            text_envelope, dflash_tree);
                const std::size_t chain_accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t tree_accept =
                    ops::speculative_accept_tree_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, dflash_verify, dflash_verify, batch, batch);
                const std::size_t accept   = std::max(chain_accept, tree_accept);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                const std::size_t dflash_peak =
                    std::max({out.dflash_round, finish(target), accept,
                              dflash_context_capacity(aggregate, true), proposal});
                out.dflash_round = dflash_peak;
                if (plan.adaptive_draft) {
                    std::size_t panel_bytes = 0;
                    for (const std::uint32_t k : plan.captured_ks) {
                        const auto wk = static_cast<std::int32_t>(
                            dflash_captured_verify_width(k, plan.dflash_verify_width));
                        if (wk >= dflash_verify) { continue; }
                        const std::int32_t compact_t = wk * batch;
                        WorkspaceLayoutBuilder extra;
                        matrix(extra, DType::BF16, DFlashConfig::feature_rows, compact_t);
                        matrix(extra, DType::I32, wk, batch);
                        const std::size_t extra_bytes = finish(extra);
                        out.dflash_round = std::max(out.dflash_round,
                                                    checked_add(dflash_peak, extra_bytes,
                                                                "adaptive DFlash prefix compact"));
                        WorkspaceLayoutBuilder panels;
                        matrix(panels, DType::I32, static_cast<std::int32_t>(k), batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::I32, wk, batch);
                        matrix(panels, DType::BF16, TextConfig::hidden, compact_t);
                        matrix(panels, DType::BF16, TextConfig::output_rows, compact_t);
                        panel_bytes = std::max(panel_bytes, finish(panels));
                        WorkspaceLayoutBuilder compact;
                        matrix(compact, DType::I32, static_cast<std::int32_t>(k), batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::I32, wk, batch);
                        matrix(compact, DType::BF16, TextConfig::hidden, compact_t);
                        matrix(compact, DType::BF16, TextConfig::output_rows, compact_t);
                        target_body(compact, compact_t, compact_t, qwen3::TextPhase::Verify,
                                    GdnWorkspacePath::ReplayRecord, batch, wk, wk, text_envelope,
                                    dflash_uses_tree_verify(k, static_cast<std::uint32_t>(wk)));
                        scratch(compact,
                                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                    TextConfig::token_domain, static_cast<std::int32_t>(k),
                                    static_cast<std::int32_t>(k), batch, batch));
                        out.dflash_round = std::max(out.dflash_round, finish(compact));
                    }
                    if (panel_bytes != 0) {
                        out.dflash_round =
                            std::max(out.dflash_round, checked_add(dflash_peak, panel_bytes,
                                                                   "adaptive DFlash compact"));
                    }
                }
                // Mixed round: the batch's live verify panels stay while one Text body of the
                // owner's slice plus the verify columns runs over them, with the GDN mixed front's
                // aggregate projections and record packing, then the owner's context append.
                if (plan.mixed_forward != 0 &&
                    batch < static_cast<std::int32_t>(plan.max_concurrency)) {
                    // The forward is always the configured width; the owner takes what `batch`
                    // live panels of at least two columns leave.
                    const auto mixed_t       = static_cast<std::int32_t>(plan.mixed_forward);
                    const std::int32_t owner = mixed_t - 2 * batch;
                    WorkspaceLayoutBuilder panels;
                    matrix(panels, DType::BF16, DFlashConfig::feature_rows, aggregate);
                    for (int i = 0; i < 10; ++i) {
                        matrix(panels, DType::I32, dflash_verify, batch);
                    }
                    matrix(panels, DType::BF16, TextConfig::hidden, aggregate);
                    matrix(panels, DType::BF16, TextConfig::output_rows, aggregate);
                    WorkspaceLayoutBuilder mixed;
                    matrix(mixed, DType::BF16, TextConfig::hidden, mixed_t);
                    matrix(mixed, DType::I32, mixed_t, 1);
                    matrix(mixed, DType::I32, mixed_t, 1);
                    matrix(mixed, DType::BF16, 2 * TextConfig::key_dim, mixed_t);
                    matrix(mixed, DType::BF16, 2 * TextConfig::value_dim, mixed_t);
                    matrix(mixed, DType::BF16, TextConfig::convolution_dim, aggregate);
                    matrix(mixed, DType::BF16, TextConfig::key_dim, aggregate);
                    matrix(mixed, DType::BF16, TextConfig::value_dim, aggregate);
                    matrix(mixed, DType::FP32, 2 * TextConfig::gdn_value_heads, aggregate);
                    target_body(mixed, 1, mixed_t, qwen3::TextPhase::Prefill,
                                GdnWorkspacePath::Prefill, 1, 1, mixed_t, text_envelope, false,
                                VerifySequences{batch, dflash_verify});
                    scratch(mixed, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                       TextConfig::token_domain, drafts, drafts, batch, batch));
                    scratch(mixed,
                            ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
                    out.dflash_mixed =
                        std::max({out.dflash_mixed,
                                  checked_add(finish(panels), finish(mixed), "mixed DFlash round"),
                                  dflash_context_capacity(owner, false)});
                }
            }
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit  = 32768;
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            merged, std::min(merged, kFrontendSegmentLimit), plan.weights_profile);
    }

    out.capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.dflash_mixed, out.vision_encode});
    return out;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,16]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.max_concurrency > kDFlashMaximumConcurrency) {
            throw std::invalid_argument(
                "DFlash supports max_concurrency 1..8; use MTP or no speculation above 8");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1," +
                                        std::to_string(kMaximumDFlashDraftTokens) + "]");
        }
        if constexpr (!DFlashConfig::tree_verify) {
            if (options.speculative.dflash_verify_width != 0 &&
                options.speculative.dflash_verify_width != options.speculative.draft_tokens + 1U) {
                throw std::invalid_argument(
                    "this DFlash target requires verify width equal to draft_tokens + 1");
            }
        }
        break;
    }
    if (options.speculative.adaptive_draft &&
        options.speculative.backend == SpeculativeBackend::None) {
        throw std::invalid_argument("adaptive draft requires speculative decoding");
    }
    if (!device.is_gfx1201()) {
        throw std::invalid_argument("Qwen3.8 R9700 runtime requires gfx1201 wave32");
    }
    qwen3::detail::validate_configured_context_checkpoint_marks(options.context_checkpoint_marks,
                                                                options.speculative.backend);
    validate_kv_system_prompt_options(options.kv_ram_capacity_bytes, options.kv_ram_system_prompts);
    validate_kv_disk_options(options.kv_ram_capacity_bytes, options.kv_disk_capacity_bytes,
                             options.kv_disk_location);
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                      = std::make_unique<SequencePlanImpl>();
    impl->weights_profile          = inputs.weights_profile;
    impl->capacity                 = inputs.capacity;
    impl->main_page_groups         = main_page_groups;
    impl->kv_capacity              = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency          = inputs.max_concurrency;
    impl->prefill_chunk            = inputs.prefill_chunk;
    impl->draft_window             = inputs.draft_window;
    impl->adaptive_draft           = inputs.adaptive_draft;
    impl->p_less_draft_temperature = inputs.p_less_draft_temperature;
    impl->captured_ks = qwen3::adaptive_draft_ks(inputs.speculative_backend, inputs.draft_window,
                                                 inputs.adaptive_draft);
    impl->dflash_verify_width =
        inputs.speculative_backend == SpeculativeBackend::DFlash
            ? dflash_storage_verify_width(impl->captured_ks, inputs.draft_window,
                                          inputs.dflash_verify_width)
            : 0U;
    impl->speculative_backend = inputs.speculative_backend;
    // The mixed forward width bounds every prefill step's forward while decode rows wait. A
    // mixed round's owner fills it beside the live verify panels, so it must exceed every other
    // lane's full panel. Automatic: kMixedForwardTokens (bounded by the chunk) when mixed rounds
    // exist, prefill-first otherwise.
    const std::uint64_t verify_columns =
        static_cast<std::uint64_t>(inputs.max_concurrency - 1U) * impl->dflash_verify_width;
    if (inputs.mixed_forward) {
        impl->mixed_forward = *inputs.mixed_forward;
    } else {
        const std::uint32_t forward =
            std::min(kMixedForwardTokens,
                     inputs.prefill_chunk - inputs.prefill_chunk % kMixedForwardAlignment);
        impl->mixed_forward = inputs.speculative_backend == SpeculativeBackend::DFlash &&
                                      inputs.max_concurrency > 1U && verify_columns < forward
                                  ? forward
                                  : 0U;
    }
    if (impl->mixed_forward != 0 &&
        (impl->mixed_forward % kMixedForwardAlignment != 0 ||
         impl->mixed_forward > inputs.prefill_chunk || impl->mixed_forward <= verify_columns)) {
        throw std::invalid_argument(
            "mixed_forward must be a multiple of " + std::to_string(kMixedForwardAlignment) +
            ", at most the prefill chunk (" + std::to_string(inputs.prefill_chunk) +
            "), and wider than the other lanes' verify columns (" + std::to_string(verify_columns) +
            ")");
    }
    impl->proposal_head            = inputs.proposal_head;
    impl->features                 = inputs.features;
    impl->use_device_graph         = inputs.use_device_graph;
    impl->device                   = inputs.device;
    impl->kv_ram_capacity_bytes    = inputs.kv_ram_capacity_bytes;
    impl->kv_ram_system_prompts    = inputs.kv_ram_system_prompts;
    impl->kv_disk_capacity_bytes   = inputs.kv_disk_capacity_bytes;
    impl->kv_disk_location         = inputs.kv_disk_location;
    impl->kv_disk_compress         = inputs.kv_disk_compress;
    impl->model_id                 = inputs.model_id;
    impl->weights_id               = inputs.weights_id;
    impl->artifact_tensor_identity = inputs.artifact_tensor_identity;
    impl->context_checkpoint_marks = inputs.context_checkpoint_marks;
    impl->persistent               = persistent_layout(*impl);
    impl->workspace                = build_workspace_plan(*impl);
    if (impl->features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit = 32768;
        const std::uint32_t merged = std::min(impl->capacity, kFrontendMergedLimit);
        impl->request_transient_capacity_bytes =
            schedule::VisionContext::output_transient_bytes(merged);
    }
    if (impl->use_device_graph) {
        std::size_t definitions         = 0;
        std::size_t lowered_definitions = 0;
        for (std::uint32_t batch = 1; batch <= impl->max_concurrency; ++batch) {
            const auto count_profiles = [&](const auto& profiles) {
                definitions += profiles.size();
                if (batch == 1U) lowered_definitions += profiles.size();
            };
            if (impl->speculative_backend == SpeculativeBackend::None) {
                count_profiles(ordinary_graph_profiles(impl->capacity));
            } else {
                for (const auto k : impl->captured_ks) {
                    if (impl->speculative_backend == SpeculativeBackend::Mtp) {
                        count_profiles(mtp_graph_profiles(impl->capacity, k));
                    } else {
                        count_profiles(dflash_graph_profiles(
                            impl->capacity, k, batch,
                            dflash_captured_verify_width(k, impl->dflash_verify_width)));
                    }
                }
            }
        }
        impl->graph_definition_count = definitions;
        impl->graph_executable_count = definitions;
        // Ordinary C1/capacity256 preparation measured 44 MiB on gfx1201 with eligible-logit
        // reporting (2026-10-18). Its two profiles reserve 20 MiB below; the remaining 24 MiB
        // is family residency. Reporting adds no captured graph nodes.
        const std::size_t family_bytes =
            (impl->speculative_backend == SpeculativeBackend::None  ? 24U
             : impl->speculative_backend == SpeculativeBackend::Mtp ? 46U
                                                                    : 48U) *
            kMiB;
        impl->graph_allowance_bytes =
            checked_add(checked_add(family_bytes,
                                    checked_mul(definitions, kR9700GraphExecutableBytes,
                                                "immutable graph executable allowance"),
                                    "graph family allowance"),
                        checked_mul(lowered_definitions, kR9700PersistentDecodeDefinitionBytes,
                                    "persistent decode allowance"),
                        "graph allowance with persistent decode");
        if (const auto calibrated = Variant::calibrated_dflash_graph_allowance_bytes(
                impl->weights_profile, impl->features, impl->capacity, impl->max_concurrency,
                impl->prefill_chunk, impl->draft_window, impl->adaptive_draft)) {
            impl->graph_allowance_bytes = std::min(impl->graph_allowance_bytes, *calibrated);
        }
        if (lowered_definitions != 0U) {
            // Cooperative graph dispatch adds shared runtime residency, not a new argument pool
            // per definition. C6 bulk preparation measured 80 MiB against the previous
            // 62-MiB inventory (2026-10-19); reserve 20 MiB for the additional graph residency.
            impl->graph_allowance_bytes = checked_add(impl->graph_allowance_bytes, 20U * kMiB,
                                                      "cooperative decode dispatch residency");
        }
    }
    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        checked_add(impl->graph_allowance_bytes,
                    checked_add(Variant::runtime_allocation_overhead_bound(impl->weights_profile),
                                Variant::runtime_driver_reservation_bytes(
                                    impl->weights_profile, impl->max_concurrency,
                                    impl->kv_disk_capacity_bytes != 0U),
                                "sequence driver and physical allocation bounds"),
                    "sequence physical allocation bound"),
        "sequence graph allowance");
    if (impl->kv_disk_capacity_bytes != 0) {
        const auto& text                  = impl->persistent.decoder.text_kv.storage.storage;
        const auto* backend               = impl->persistent.decoder.mtp_kv
                                                ? &impl->persistent.decoder.mtp_kv->storage.storage
                                                : nullptr;
        const std::size_t text_page_bytes = text.payload_bytes() / text.spec.page_group_count;
        const std::size_t backend_page_bytes =
            backend != nullptr ? backend->payload_bytes() / backend->spec.page_group_count : 0U;
        impl->device_reservation_bytes =
            checked_add(impl->device_reservation_bytes,
                        qwen3::detail::disk_restore_device_reservation_bytes(
                            std::max(text_page_bytes, backend_page_bytes), text.planes.size(),
                            backend != nullptr ? backend->planes.size() : 0U),
                        "KV disk restore device reservation");
    }
    return impl;
}

} // namespace

std::unique_ptr<SequencePlanImpl>
build_sequence_candidate_for_qualification(const SequencePlanningInputs& inputs,
                                           std::uint32_t main_page_groups) {
    return build_sequence_candidate(inputs, main_page_groups);
}

std::unique_ptr<qwen3::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    validate_target_options(device, options);

    SequencePlanningInputs inputs{
        .weights_profile          = weights_profile,
        .capacity                 = options.max_context,
        .max_concurrency          = options.max_concurrency,
        .prefill_chunk            = std::min(options.prefill_chunk, options.max_context),
        .mixed_forward            = options.mixed_forward,
        .draft_window             = options.speculative.draft_tokens,
        .dflash_verify_width      = options.speculative.dflash_verify_width,
        .adaptive_draft           = options.speculative.adaptive_draft,
        .p_less_draft_temperature = options.speculative.dflash_p_less_draft_temperature,
        .speculative_backend      = options.speculative.backend,
        .proposal_head            = options.speculative.proposal_head,
        .features                 = qwen3::startup_features(options),
        .use_device_graph         = options.use_device_graph,
        .device                   = options.device,
        .kv_ram_capacity_bytes    = options.kv_ram_capacity_bytes,
        .kv_ram_system_prompts    = options.kv_ram_system_prompts,
        .kv_disk_capacity_bytes   = options.kv_disk_capacity_bytes,
        .kv_disk_location         = options.kv_disk_location,
        .kv_disk_compress         = options.kv_disk_compress,
        .model_id                 = options.model_id,
        .weights_id               = options.weights_id,
        .artifact_tensor_identity = options.artifact_tensor_identity,
        .context_checkpoint_marks =
            qwen3::detail::resolved_prefill_context_marks(options.context_checkpoint_marks),
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
        .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
        .minimum_main_page_groups             = minimum_pages,
        .maximum_main_page_groups             = maximum_pages,
        .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
        .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS

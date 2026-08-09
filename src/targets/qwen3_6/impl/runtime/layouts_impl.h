#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

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

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    const std::size_t padding = (alignment - value % alignment) % alignment;
    return checked_add(value, padding, label);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const std::int32_t slots_per_sequence =
        LinearStateSlots::required_slot_count(plan.draft_window);
    const std::int32_t linear_state_slots =
        checked_i32(static_cast<std::uint64_t>(slots_per_sequence) * plan.max_concurrency,
                    "Linear Attention state slots exceed int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers = TextConfig::full_attention_layers(),
                     .mtp_layers            = TextConfig::mtp_layers,
                     .capacity              = plan.capacity,
                     .kv_heads              = TextConfig::kv_heads,
                     .attention_head_dim    = TextConfig::head_dim,
                     .kv_dtype              = plan.kv_dtype,
                     .kv_quant_group        = plan.kv_quant_group,
                     .enable_mtp            = plan.features.mtp(),
                     .kv_table_rows         = static_cast<std::int32_t>(plan.max_concurrency),
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
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            dflash.local          = plan_cyclic_kv_cache(builder, DFlashConfig::local_layers,
                                                         DFlashConfig::local_capacity,
                                                         DFlashConfig::kv_heads, DFlashConfig::head_dim);
            dflash.boundary_local = plan_cyclic_kv_cache(
                builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                DFlashConfig::kv_heads, DFlashConfig::head_dim);
            const std::uint32_t full_pages =
                (plan.capacity + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) /
                static_cast<std::uint32_t>(kPagedKVPageSize);
            PagedKVPoolSpec full_pool{
                .page_group_count      = full_pages,
                .logical_page_capacity = full_pages,
                .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                .plane_order           = PagedKVPlaneOrder::HeadMajor,
                .planes =
                    {
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                    },
            };
            dflash.full = qwen3_6::PagedKVCacheLayout{
                .pool        = plan_paged_kv_pool(builder, full_pool),
                .layers      = 1,
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = DType::BF16,
                .quant_group = 0,
            };
            dflash.commit_count = add_tensor(builder, DType::I32, {1}, "DFlash exact commit count");
            dflash.target_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash retained target features");
            dflash.feature_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash retained target positions");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden         = TextConfig::hidden,
                                         .output_rows    = TextConfig::output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .enable_mtp     = plan.features.mtp()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    out.tail_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "tail hidden");
    out.boundary_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "boundary hidden");
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
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

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
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, envelope, batch_size, min_width,
                            max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, bool snapshot,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (!snapshot) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, bool snapshot,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, snapshot, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto target_verify = [&](std::int32_t tokens, ops::GqaExecutionEnvelope envelope) {
        WorkspaceLayoutBuilder layout;
        matrix(layout, DType::I32, 1, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        target_body(layout, tokens, tokens, qwen3_6::TextPhase::Verify, true, 1, tokens, tokens,
                    envelope);
        return finish(layout);
    };

    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
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
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, false, 1, 1, chunk,
                text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, true, batch, 1, 1,
                    text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, false, 1, 1, chunk,
                    text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
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
            TextConfig::token_domain, drafts, drafts);
        out.mtp_round = std::max({target_verify(verify, text_envelope), accept, finish(mtp_batch),
                                  finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens) {
                WorkspaceLayoutBuilder layout;
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t tokens) {
                WorkspaceLayoutBuilder layout;
                (void)workspace_recipe::dflash_proposal<DFlashConfig>(layout, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes({0, plan.capacity}, tokens,
                                                                       tokens),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, tokens, tokens)));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::query_size, tokens, tokens));
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                        QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                        DFlashConfig::hidden, tokens, tokens));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::intermediate, tokens, tokens));
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts);
                }
                return finish(layout);
            };

            out.dflash_context  = dflash_context_capacity(chunk);
            out.dflash_proposal = dflash_proposal_capacity(verify);
            const std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts);
            out.dflash_round   = std::max({target_verify(verify, text_envelope), accept,
                                           dflash_context_capacity(verify), out.dflash_proposal});
            out.ordinary_round = std::max(out.ordinary_round, dflash_context_capacity(1));
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit  = 32768;
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            merged, std::min(merged, kFrontendSegmentLimit));
    }

    out.capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_proposal, out.dflash_round, out.vision_encode});
    return out;
}

} // namespace

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t context_pages =
        1U + (options.max_context - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    if (context_pages < options.max_concurrency) {
        throw std::invalid_argument(
            "max_context cannot provide one Paged KV page per concurrent request");
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
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> plan_sequence_impl(DeviceContext& device,
                                                     const EngineOptions& options,
                                                     WeightsProfile weights_profile) {
    validate_target_options(device, options);

    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = weights_profile;
    impl->capacity            = options.max_context;
    impl->max_concurrency     = options.max_concurrency;
    impl->prefill_chunk       = std::min(options.prefill_chunk, options.max_context);
    impl->draft_window        = options.speculative.draft_tokens;
    impl->speculative_backend = options.speculative.backend;
    impl->proposal_head       = options.speculative.proposal_head;
    impl->features            = qwen3_6::startup_features(options);
    impl->use_cuda_graph      = options.use_cuda_graph;
    impl->device              = options.device;
    impl->kv_dtype       = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
    impl->kv_quant_group = impl->kv_dtype == DType::I8 ? qwen3_6::kKvQuantGroup : 0;
    impl->persistent     = persistent_layout(*impl);
    impl->workspace      = build_workspace_plan(*impl);
    if (impl->features.vision) {
        constexpr std::size_t kFrontendMergedLimit = 32768;
        const std::size_t merged = std::min<std::size_t>(impl->capacity, kFrontendMergedLimit);
        impl->request_transient_capacity_bytes =
            align_up(checked_mul(checked_mul(static_cast<std::size_t>(VisionConfig::output_hidden),
                                             merged, "request transient elements"),
                                 dtype_size(DType::BF16), "request transient bytes"),
                     kArenaAlign, "request transient alignment");
    }
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        impl->graph_allowance_bytes = impl->speculative_backend == SpeculativeBackend::None
                                          ? checked_mul(12ULL * kMiB, impl->max_concurrency,
                                                        "ordinary exact-b graph allowance")
                                          : 12ULL * kMiB;
        if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            impl->graph_allowance_bytes = checked_add(impl->graph_allowance_bytes, 12ULL * kMiB,
                                                      "ordinary aligned graph allowance");
        }
        if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t mtp_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible =
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window;
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes =
                checked_add(impl->graph_allowance_bytes, mtp_allowance, "MTP graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::DFlash) {
            const auto class_allowance = [&](const std::vector<GraphExecutionProfile>& profiles) {
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible =
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL;
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            const std::size_t initial =
                class_allowance(dflash_initial_graph_profiles(impl->capacity, impl->draft_window));
            const std::size_t steady =
                class_allowance(dflash_steady_graph_profiles(impl->capacity, impl->draft_window));
            impl->graph_allowance_bytes = checked_add(
                checked_add(impl->graph_allowance_bytes, initial, "DFlash initial allowance"),
                steady, "DFlash steady allowance");
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

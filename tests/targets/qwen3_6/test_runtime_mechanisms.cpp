#include "core/layout.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/mtp_alignment.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"
#define NINFER_QWEN36_RUNTIME_NS mechanism_slots
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#undef NINFER_QWEN36_RUNTIME_NS
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/types.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_topology() {
    static_assert(q36::kHybridAttentionInterval == 4);
    static_assert(q36::full_attention_layers(64) == 16);
    static_assert(q36::gdn_layers(64) == 48);
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        expect(q36::is_full_attention_layer(layer) == ((layer + 1) % 4 == 0), "hybrid layer kind");
        if (q36::is_full_attention_layer(layer)) {
            expect(q36::full_attention_index(layer) == layer / 4, "full-attention index");
        } else {
            expect(q36::gdn_index(layer) == layer - layer / 4, "GDN index");
        }
    }
}

q36::DecoderStateSpec decoder_spec(ninfer::DType dtype, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 64,
        .kv_dtype                  = dtype,
        .kv_quant_group            = dtype == ninfer::DType::I8   ? q36::kKvQuantGroup
                                     : dtype == ninfer::DType::U8 ? q36::kKvNvfp4Group
                                                                  : 0,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
        .linear_attention =
            {
                .layers         = 3,
                .conv_channels  = 10,
                .conv_width     = 3,
                .value_heads    = 4,
                .value_head_dim = 5,
                .key_head_dim   = 6,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 =
        q36::plan_decoder_state(bf16_builder, decoder_spec(ninfer::DType::BF16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pool.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pool.spec.page_group_count == 5 &&
               bf16.text_kv.pool.spec.logical_page_capacity == 3 &&
               bf16.text_kv.pool.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(std::all_of(bf16.text_kv.pool.planes.begin(), bf16.text_kv.pool.planes.end(),
                       [](const ninfer::PagedKVPlaneLayout& plane) {
                           return plane.spec.dtype == ninfer::DType::BF16;
                       }),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.linear_attention.conv.size() == 3 && bf16.linear_attention.recurrent.size() == 3,
           "Linear Attention layer storage");
    expect(bf16.linear_attention.spec.slot_count == 4, "Linear Attention slot geometry");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 =
        q36::plan_decoder_state(int8_builder, decoder_spec(ninfer::DType::I8, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pool.planes.size() == 8 &&
               int8.text_kv.pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pool.planes.size() == 4 &&
               int8.mtp_kv->pool.spec.page_group_count == 4 &&
               int8.mtp_kv->pool.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    ninfer::LayoutBuilder nvfp4_builder;
    const q36::DecoderStateLayout nvfp4 =
        q36::plan_decoder_state(nvfp4_builder, decoder_spec(ninfer::DType::U8, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pool.planes.size() == 8 &&
               nvfp4.text_kv.pool.planes[0].spec.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pool.planes[0].spec.leading_extent == 32 &&
               nvfp4.text_kv.pool.planes[2].spec.dtype == ninfer::DType::FP8_E4M3FN &&
               nvfp4.text_kv.pool.planes[2].spec.leading_extent == 4,
           "NVFP4 Text KV has packed code and UE4M3 scale planes");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{
                     .hidden = 32, .output_rows = 128, .draft_window = 5, .enable_mtp = true});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder,
        q36::RoundStateSpec{
            .hidden = 32, .output_rows = 128, .draft_window = 15, .enable_dflash = true});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControl control = q36::build_vision_control(prompt);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].cu_seqlens == std::vector<std::int32_t>({0, 4}) &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].cu_seqlens == std::vector<std::int32_t>({0, 4, 8}) &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    resident.reserve(16);
    resident.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    append_text_token(original, 12, 4);
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_prefix_hash_and_dflash_gate() {
    q36::PreparedPromptData original = identity_prompt();
    const auto chain                 = q36::detail::prefix_hash_chain(original);
    expect(chain.size() == original.token_ids.size() + 1, "hash chain includes the empty prefix");

    q36::detail::ResidentPrefixIdentity resident;
    resident.assign(original);
    for (std::size_t count = 0; count <= original.token_ids.size(); ++count) {
        expect(q36::detail::prefix_hash_at(original.token_ids, resident, count) == chain[count],
               "prefix_hash_at matches prefix_hash_chain");
    }
    {
        const std::size_t e = std::min<std::size_t>(2, original.token_ids.size());
        const auto hash_e   = q36::detail::prefix_hash_at(original.token_ids, resident, e);
        std::vector<ninfer::TokenId> longer = original.token_ids;
        q36::PreparedPromptData assigned    = original;
        append_text_token(assigned, 99, 99);
        longer.push_back(99);
        resident.assign(assigned);
        expect(q36::detail::prefix_hash_at(longer, resident, e) == hash_e,
               "occupy ledger.assign of a longer prompt keeps the hash at rollback E");
        resident.assign(original);
    }

    q36::PreparedPromptData changed_token = original;
    changed_token.token_ids.back() += 1;
    const auto token_chain = q36::detail::prefix_hash_chain(changed_token);
    expect(token_chain[original.token_ids.size() - 1] == chain[original.token_ids.size() - 1] &&
               token_chain.back() != chain.back(),
           "token difference changes only hashes at and after the mutated token");

    q36::PreparedPromptData changed_type = original;
    changed_type.token_types[0]          = 1;
    expect(q36::detail::prefix_hash_chain(changed_type)[1] != chain[1],
           "token_type difference changes the hash chain");

    q36::PreparedPromptData changed_position = original;
    changed_position.positions[0] += 1;
    expect(q36::detail::prefix_hash_chain(changed_position).back() != chain.back(),
           "position-axis difference changes the hash chain");

    q36::PreparedPromptData changed_digest = identity_prompt(2);
    const auto digest_chain                = q36::detail::prefix_hash_chain(changed_digest);
    expect(digest_chain[2] == chain[2] && digest_chain[3] != chain[3],
           "completing vision item changes the hash at its end");

    expect(q36::detail::dflash_rewrite_checkpoint_ready(true, 16, 16),
           "RAM DFlash checkpoint with backend_image_present must not FullReset");
    expect(!q36::detail::dflash_rewrite_checkpoint_ready(false, 16, 16),
           "missing backend image FullResets a DFlash checkpoint view");
    expect(!q36::detail::dflash_rewrite_checkpoint_ready(true, 15, 16),
           "short DFlash context frontier FullResets a checkpoint view");
}

void test_prefill_context_marks() {
    using Path = ninfer::PrefixReusePath;
    expect(q36::detail::kPrefillContextMarks.size() == 6, "product ladder has six marks");
    expect(q36::detail::kPrefillContextMarks[0] == 24576, "mark 24576");
    expect(q36::detail::kPrefillContextMarks[1] == 36864, "mark 36864");
    expect(q36::detail::kPrefillContextMarks[2] == 53248, "mark 53248");
    expect(q36::detail::kPrefillContextMarks[3] == 77824, "mark 77824");
    expect(q36::detail::kPrefillContextMarks[4] == 102400, "mark 102400");
    expect(q36::detail::kPrefillContextMarks[5] == 151552, "mark 151552");
    for (const std::uint32_t mark : q36::detail::kPrefillContextMarks) {
        expect(mark % 4096u == 0, "context marks are 4096-aligned chunk ends");
        expect(mark != 24000 && mark != 36000 && mark != 52000 && mark != 76000 && mark != 100000 &&
                   mark != 10240 && mark != 25000 && mark != 50000 && mark != 80000 &&
                   mark != 120000 && mark != 150000,
               "advertised F is never the raw named size");
    }
    for (std::size_t i = 0; i < q36::detail::kPrefillContextMarks.size(); ++i) {
        const std::uint32_t mark = q36::detail::kPrefillContextMarks[i];
        expect(q36::detail::next_prefill_context_mark(mark - 1) == mark, "next mark just before");
        if (i + 1 < q36::detail::kPrefillContextMarks.size()) {
            const std::uint32_t following = q36::detail::kPrefillContextMarks[i + 1];
            expect(q36::detail::next_prefill_context_mark(mark) == following, "next mark at freeze");
            expect(q36::detail::next_prefill_context_mark(mark + 1) == following,
                   "next mark just after freeze");
        } else {
            expect(!q36::detail::next_prefill_context_mark(mark).has_value(), "no mark after last");
            expect(!q36::detail::next_prefill_context_mark(mark + 1).has_value(),
                   "no mark past last");
        }
    }
    expect(q36::detail::next_prefill_context_mark(0) == 24576, "next mark from 0");
    expect(q36::detail::next_prefill_context_mark(4096) == 24576,
           "chunk end 4096 still waits for first mark");
    expect(q36::detail::next_prefill_context_mark(12288) == 24576,
           "chunk end 12288 still waits for first mark");
    expect(q36::detail::next_prefill_context_mark(25000) == 36864,
           "off-grid F still uses next mark");

    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 4096, 12288, false,
                                                                 true),
           "chunk end 4096 does not freeze");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 8192, 12288, false,
                                                                 true),
           "chunk end 8192 does not freeze");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 10240, 12288, false,
                                                                 true),
           "raw named size 10240 is not a freeze coordinate");
    expect(q36::detail::should_freeze_prefill_context_checkpoint(true, true, 12288, 12288, false,
                                                                true),
           "chunk end 12288 freezes");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 12288, 12288, true,
                                                                 true),
           "already-captured F does not freeze again");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 200000, 0, false,
                                                                 true),
           "no freeze after the last mark");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(12288, 0) == 12288,
           "first freeze reports coverage from 0");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(24576, 12288) == 12288,
           "second freeze reports only the new rung");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(24576, 0) == 24576,
           "two marks in one request report coverage from 0");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(16384, 0) == 16384,
           "catch-up freeze reports the advertised chunk end");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(12288, 12288) == 0,
           "same frontier is not newly frozen");
    expect(q36::detail::advertised_context_checkpoint_frontier(16000) == 16000,
           "catch-up advertised F is the later chunk end");

    const std::uint32_t catch_up_first = 8000u + 4096u;
    expect(catch_up_first == 12096 &&
               !q36::detail::should_freeze_prefill_context_checkpoint(true, true, catch_up_first,
                                                                     12288, false, true),
           "resume 8000 first 4096 chunk ends at 12096 and does not freeze");
    expect(q36::detail::should_freeze_prefill_context_checkpoint(true, true, 16000, 12288, false,
                                                                true) &&
               q36::detail::advertised_context_checkpoint_frontier(16000) == 16000,
           "resume 8000 then prefill to 16000 freezes 16000 not 12288");
    expect(q36::detail::should_freeze_prefill_context_checkpoint(true, true, 8192u + 4096u, 12288,
                                                                false, true) &&
               q36::detail::advertised_context_checkpoint_frontier(12288) == 12288,
           "resume 8192 first chunk end 12288 freezes 12288");

    expect(!q36::detail::should_freeze_prefill_context_checkpoint(false, true, 15000, 12288, false,
                                                                 true),
           "decode-only past 12288 does not freeze");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, false, 12288, 12288, false,
                                                                 true),
           "capture-disabled prefill does not freeze");

    expect(q36::detail::capture_prefill_context_checkpoints(true, true),
           "MTP or DFlash + prefix reuse captures");
    expect(!q36::detail::capture_prefill_context_checkpoints(false, true),
           "--no-prefix-reuse skips capture");
    expect(!q36::detail::capture_prefill_context_checkpoints(true, false),
           "ordinary does not capture");
    expect(!q36::detail::capture_prefill_context_checkpoints(false, false),
           "reuse-off and ordinary does not capture");

    expect(q36::detail::retain_context_checkpoint_head(12288, 12288), "keep head at restore F");
    expect(q36::detail::retain_context_checkpoint_head(8192, 12288), "keep heads before F");
    expect(!q36::detail::retain_context_checkpoint_head(24576, 12288), "drop heads after F");

    q36::PreparedPromptData prompt;
    prompt.token_ids   = {1, 2, 3, 4};
    prompt.token_types = {0, 0, 0, 0};
    prompt.positions   = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    expect(q36::detail::prefix_items_complete_at(prompt.vision_items, 4),
           "text-only frontier is complete");
    q36::PreparedPromptData vision = identity_prompt();
    expect(!q36::detail::prefix_items_complete_at(vision.vision_items, 2),
           "frontier that splits a vision item is not capturable");
    expect(q36::detail::prefix_items_complete_at(vision.vision_items, 3),
           "frontier after a complete vision item is capturable");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(
               true, true, 12288, 12288, false,
               q36::detail::prefix_items_complete_at(vision.vision_items, 2)),
           "maybe_freeze must not capture when prefix_items_complete_at is false");
    expect(q36::detail::should_freeze_prefill_context_checkpoint(
               true, true, 12288, 12288, false,
               q36::detail::prefix_items_complete_at(prompt.vision_items, 4)),
           "maybe_freeze captures when prefix items are complete at F");

    const std::array<std::uint32_t, 1> ladders_13k{12288u};
    const auto rewrite_loses = q36::detail::select_resident_prefill_reuse(
        false, 13000, true, 8000, Path::RestoreTurnCheckpoint, ladders_13k);
    expect(rewrite_loses.path == Path::RestoreContextCheckpoint && rewrite_loses.frontier == 12288,
           "rewrite at 8000 loses to ladder 12288 on a 13k matching prompt");
    const auto current_wins = q36::detail::select_resident_prefill_reuse(
        true, 13000, true, 8000, Path::RestoreTurnCheckpoint, ladders_13k);
    expect(current_wins.path == Path::AppendAtFrontier && current_wins.frontier == 13000,
           "current frontier still wins if it matches");
    const auto e_eq_f = q36::detail::select_resident_prefill_reuse(
        true, 12288, true, 8000, Path::RestoreTurnCheckpoint, ladders_13k);
    expect(e_eq_f.path == Path::AppendAtFrontier && e_eq_f.frontier == 12288,
           "E==F still-resident match is AppendAtFrontier, not ladder restore");
    const std::array<std::uint32_t, 1> tie{8000u};
    const auto equal_f = q36::detail::select_resident_prefill_reuse(
        false, 13000, true, 8000, Path::RestoreTurnCheckpoint, tie);
    expect(equal_f.path == Path::RestoreTurnCheckpoint && equal_f.frontier == 8000,
           "rewrite wins a tie at the same F as a ladder head");
    const auto nothing = q36::detail::select_resident_prefill_reuse(
        false, 0, false, 0, Path::FullReset, std::span<const std::uint32_t>{});
    expect(nothing.path == Path::FullReset && nothing.frontier == 0,
           "no matching head is FullReset");

    expect(q36::detail::is_complete_checkpoint_restore(Path::RestoreContextCheckpoint),
           "RestoreContextCheckpoint is a complete checkpoint restore");
    expect(!q36::detail::is_rewrite_checkpoint_restore(Path::RestoreContextCheckpoint),
           "RestoreContextCheckpoint is not a rewrite restore");
    expect(q36::detail::mtp_complete_checkpoint_ready(Path::RestoreContextCheckpoint, 12288, 12287),
           "RestoreContextCheckpoint with mtp_kv_valid >= F-1 is not forced to FullReset");
    expect(!q36::detail::mtp_complete_checkpoint_ready(Path::RestoreContextCheckpoint, 12288, 12286),
           "short MTP KV is not checkpoint-ready");
    expect(!q36::detail::mtp_bridge_reads_rewrite_hidden(Path::RestoreContextCheckpoint),
           "context restore installs/reads tail_hidden not rewrite hidden");
    expect(q36::detail::mtp_bridge_reads_rewrite_hidden(Path::RestoreTurnCheckpoint),
           "rewrite restore still reads rewrite hidden");

    q36::detail::PrefixHash128 hash_a{.lo = 1, .hi = 2};
    q36::detail::PrefixHash128 hash_b{.lo = 3, .hi = 4};
    expect(q36::detail::staging_holds_restore_identity(true, 0, hash_a, 12288, 0, hash_a, 12288),
           "staging D2D requires matching lane+hash+F");
    expect(!q36::detail::staging_holds_restore_identity(true, 1, hash_a, 12288, 0, hash_a, 12288),
           "wrong staging lane must not D2D");
    expect(!q36::detail::staging_holds_restore_identity(true, 0, hash_b, 12288, 0, hash_a, 12288),
           "wrong staging hash must not D2D");
    expect(!q36::detail::staging_holds_restore_identity(true, 0, hash_a, 24576, 0, hash_a, 12288),
           "wrong staging F must not D2D");
    expect(!q36::detail::staging_holds_restore_identity(false, 0, hash_a, 12288, 0, hash_a, 12288),
           "empty staging must not D2D");

    expect(q36::detail::occupy_drops_rewrite_ahead_of_restore(Path::RestoreContextCheckpoint, true,
                                                             15000, 12288),
           "occupy drops rewrite when rewrite.frontier > restore F");
    expect(!q36::detail::occupy_drops_rewrite_ahead_of_restore(Path::RestoreContextCheckpoint, true,
                                                              8000, 12288),
           "occupy keeps rewrite when rewrite.frontier <= restore F");
    expect(!q36::detail::occupy_drops_rewrite_ahead_of_restore(Path::AppendAtFrontier, true, 15000,
                                                              12288),
           "append occupy does not drop rewrite via the restore helper");
    expect(q36::detail::occupy_clears_context_checkpoints(Path::FullReset),
           "FullReset clears ladder heads");
    expect(!q36::detail::occupy_clears_context_checkpoints(Path::AppendAtFrontier),
           "AppendAtFrontier does not clear the ladder");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::AppendAtFrontier, 12288, 13000),
           "AppendAtFrontier keeps heads with frontier <= new base");
    expect(!q36::detail::occupy_keeps_context_checkpoint_head(Path::AppendAtFrontier, 24576, 13000),
           "AppendAtFrontier drops heads with frontier > new base");
    expect(!q36::detail::occupy_keeps_context_checkpoint_head(Path::FullReset, 12288, 0),
           "FullReset keeps no ladder heads");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreContextCheckpoint, 12288,
                                                            12288),
           "restore keeps the head at F");
    expect(!q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreContextCheckpoint, 24576,
                                                             12288),
           "restore drops heads after F");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreTurnCheckpoint, 12288,
                                                            13000),
           "rewrite occupy keeps ladder heads with frontier <= new base");
    expect(!q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreTurnCheckpoint, 24576,
                                                             13000),
           "rewrite occupy drops ladder heads after the new base");
    expect(q36::detail::next_prefill_context_mark(24576) == 36864,
           "rollback to first mark recaptures the second");
    expect(q36::detail::next_prefill_context_mark(26000) == 36864,
           "catch-up head between marks still recaptures the second");

    expect(q36::detail::should_freeze_prefill_context_checkpoint(true, true, 24576, 24576, false,
                                                                true),
           "product first mark 24576 freezes at that chunk end");
    expect(!q36::detail::should_freeze_prefill_context_checkpoint(true, true, 24384, 24576, false,
                                                                 true),
           "chunk end 24384 still waits for 24576");
    expect(q36::detail::should_freeze_prefill_context_checkpoint(true, true, 25000, 24576, false,
                                                                true) &&
               q36::detail::advertised_context_checkpoint_frontier(25000) == 25000,
           "catch-up past 24576 freezes the later chunk end");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(36864, 24576) == 12288,
           "second product mark reports only the new rung");
    expect(q36::detail::newly_frozen_context_checkpoint_tokens(36864, 0) == 36864,
           "two product marks in one request report coverage from 0");

    const std::array<std::uint32_t, 3> occupy_heads{8192u, 24576u, 36864u};
    std::vector<std::uint32_t> kept;
    for (const std::uint32_t frontier : occupy_heads) {
        if (q36::detail::retain_context_checkpoint_head(frontier, 24576)) {
            kept.push_back(frontier);
        }
    }
    expect(kept.size() == 2 && kept[0] == 8192 && kept[1] == 24576,
           "occupy drop keeps heads at or before restore F");
    expect(q36::detail::next_prefill_context_mark(24576) == 36864,
           "occupy restore to 24576 recaptures 36864 next");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreContextCheckpoint, 8192,
                                                            24576) ==
               q36::detail::retain_context_checkpoint_head(8192, 24576),
           "occupy keep-helper matches retain for restore");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::FullReset, 24576, 0) == false,
           "FullReset occupy keeps no heads");

    expect(q36::detail::mtp_prefix_reuse_ready(Path::FullReset, 0, 0, false, false),
           "FullReset does not require MTP reuse readiness");
    expect(q36::detail::mtp_prefix_reuse_ready(Path::RestoreContextCheckpoint, 24576, 24575, false,
                                              true),
           "checkpoint restore with mtp_kv_valid == F-1 is ready");
    expect(!q36::detail::mtp_prefix_reuse_ready(Path::RestoreContextCheckpoint, 24576, 24574, true,
                                               true),
           "checkpoint restore with mtp_kv_valid == F-2 FullResets");
    expect(!q36::detail::mtp_prefix_reuse_ready(Path::RestoreTurnCheckpoint, 24576, 24574, true,
                                               true),
           "turn restore with mtp_kv_valid == F-2 FullResets");
    expect(q36::detail::mtp_prefix_reuse_ready(Path::RestoreResponseCheckpoint, 24576, 24575, true,
                                              true),
           "response restore with mtp_kv_valid == F-1 is ready");
    expect(!q36::detail::mtp_prefix_reuse_ready(Path::RestoreContextCheckpoint, 24576, 24575, true,
                                               false),
           "checkpoint restore without an MTP cache FullResets");
    expect(q36::detail::mtp_prefix_reuse_ready(Path::AppendAtFrontier, 24576, 24575, true, true),
           "append with tail_hidden and F-1 MTP KV is ready");
    expect(!q36::detail::mtp_prefix_reuse_ready(Path::AppendAtFrontier, 24576, 24575, false, true),
           "append without tail_hidden FullResets");

    expect(q36::detail::is_staged_checkpoint_restore(Path::RestoreTurnRollback),
           "RestoreTurnRollback is a staged complete-checkpoint restore");
    expect(q36::detail::is_complete_checkpoint_restore(Path::RestoreTurnRollback),
           "RestoreTurnRollback is a complete checkpoint restore");
    expect(!q36::detail::is_rewrite_checkpoint_restore(Path::RestoreTurnRollback),
           "RestoreTurnRollback is not a rewrite restore");
    expect(q36::detail::mtp_complete_checkpoint_ready(Path::RestoreTurnRollback, 1000, 999),
           "RestoreTurnRollback with mtp_kv_valid >= F-1 is ready");
    expect(!q36::detail::mtp_complete_checkpoint_ready(Path::RestoreTurnRollback, 1000, 998),
           "RestoreTurnRollback with mtp_kv_valid == F-2 FullResets");
    expect(q36::detail::mtp_prefix_reuse_ready(Path::RestoreTurnRollback, 1000, 999, false, true),
           "mtp_prefix_reuse_ready accepts RestoreTurnRollback at F-1");
    expect(!q36::detail::mtp_prefix_reuse_ready(Path::RestoreTurnRollback, 1000, 998, true, true),
           "mtp_prefix_reuse_ready FullResets RestoreTurnRollback at F-2");
    expect(!q36::detail::mtp_bridge_reads_rewrite_hidden(Path::RestoreTurnRollback),
           "turn-rollback restore reads tail_hidden not rewrite hidden");
    expect(q36::detail::occupy_drops_rewrite_ahead_of_restore(Path::RestoreTurnRollback, true, 1500,
                                                             1000),
           "rollback occupy drops rewrite ahead of E");
    expect(!q36::detail::occupy_drops_rewrite_ahead_of_restore(Path::RestoreTurnRollback, true, 800,
                                                              1000),
           "rollback occupy keeps rewrite at or before E");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreTurnRollback, 1000, 1000),
           "rollback restore keeps the head at E");
    expect(!q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreTurnRollback, 24576,
                                                             1000),
           "rollback restore drops heads after E");

    using Kind = q36::detail::ContextCheckpointKind;
    expect(q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1200, true, true,
                                                     false, true),
           "append with prompt_tokens > E pins rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1000, true, true,
                                                      false, true),
           "exact-hit append does not replace an older rollback pin");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 0, 100, true, true,
                                                      false, true),
           "first-visit E==0 does not pin rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::RestoreTurnRollback, 1000, 1200, true,
                                                      true, false, true),
           "restore occupy does not recapture rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1200, true, true,
                                                      true, true),
           "skip pin when any head already sits at E");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1200, false, true,
                                                      false, true),
           "MTP-off / reuse-off does not pin rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1200, true, false,
                                                      false, true),
           "invalid tail_hidden does not pin rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::AppendAtFrontier, 1000, 1200, true, true,
                                                      false, false),
           "incomplete Vision item at E skips the rollback pin");

    const std::array<q36::detail::PrefillReuseHead, 1> rollback_e1{
        q36::detail::PrefillReuseHead{1000u, Kind::TurnRollback}};
    const auto current_beats_rollback = q36::detail::select_resident_prefill_reuse(
        true, 2000, true, 800, Path::RestoreTurnCheckpoint, rollback_e1);
    expect(current_beats_rollback.path == Path::AppendAtFrontier &&
               current_beats_rollback.frontier == 2000,
           "current matching E still wins against rollback");
    const auto rewrite_longer_than_rollback = q36::detail::select_resident_prefill_reuse(
        false, 2000, true, 1500, Path::RestoreTurnCheckpoint, rollback_e1);
    expect(rewrite_longer_than_rollback.path == Path::RestoreTurnCheckpoint &&
               rewrite_longer_than_rollback.frontier == 1500,
           "regenerate: rewrite longer than rollback wins");
    const auto rollback_longer_than_rewrite = q36::detail::select_resident_prefill_reuse(
        false, 2000, true, 800, Path::RestoreTurnCheckpoint, rollback_e1);
    expect(rollback_longer_than_rewrite.path == Path::RestoreTurnRollback &&
               rollback_longer_than_rewrite.frontier == 1000,
           "edit last user: rollback longer than rewrite wins");
    const auto edit_misses_rewrite = q36::detail::select_resident_prefill_reuse(
        false, 2000, false, 1500, Path::RestoreTurnCheckpoint, rollback_e1);
    expect(edit_misses_rewrite.path == Path::RestoreTurnRollback &&
               edit_misses_rewrite.frontier == 1000,
           "edit last user: unmatched rewrite falls through to rollback E");
    const std::array<q36::detail::PrefillReuseHead, 1> rollback_tie{
        q36::detail::PrefillReuseHead{800u, Kind::TurnRollback}};
    const auto rewrite_tie_rollback = q36::detail::select_resident_prefill_reuse(
        false, 2000, true, 800, Path::RestoreTurnCheckpoint, rollback_tie);
    expect(rewrite_tie_rollback.path == Path::RestoreTurnCheckpoint &&
               rewrite_tie_rollback.frontier == 800,
           "same-F rewrite vs rollback keeps rewrite");
    expect(q36::detail::reuse_path_for_context_checkpoint_kind(Kind::TurnRollback) ==
               Path::RestoreTurnRollback,
           "turn_rollback kind reports restore_turn_rollback not restore_context_checkpoint");
    expect(q36::detail::reuse_path_for_context_checkpoint_kind(Kind::Ladder) ==
               Path::RestoreContextCheckpoint,
           "ladder kind still reports restore_context_checkpoint");
    expect(q36::detail::reuse_path_for_context_checkpoint_kind(Kind::OnDemand) ==
               Path::RestoreContextCheckpoint,
           "reserved OnDemand kind is not RestoreTurnRollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::FullReset, 1000, 1200, true, true, false,
                                                      true),
           "FullReset occupy does not pin rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::RestoreContextCheckpoint, 1000, 1200,
                                                      true, true, false, true),
           "ladder restore occupy does not pin rollback");
    expect(!q36::detail::should_capture_turn_rollback(Path::RestoreTurnCheckpoint, 1000, 1200, true,
                                                      true, false, true),
           "rewrite restore occupy does not pin rollback");
    expect(q36::detail::occupy_keeps_context_checkpoint_head(Path::RestoreTurnRollback, 800, 1000),
           "rollback restore keeps a shorter ladder under E");
    expect(q36::detail::occupy_clears_context_checkpoints(Path::FullReset),
           "FullReset drops every context-checkpoint head");
    expect(!q36::detail::occupy_clears_context_checkpoints(Path::RestoreTurnRollback),
           "rollback restore does not FullReset the head list");
    const std::array<q36::detail::PrefillReuseHead, 2> two_rollbacks{
        q36::detail::PrefillReuseHead{800u, Kind::TurnRollback},
        q36::detail::PrefillReuseHead{1000u, Kind::TurnRollback}};
    const auto longest_rollback = q36::detail::select_resident_prefill_reuse(
        false, 2000, false, 0, Path::RestoreTurnCheckpoint, two_rollbacks);
    expect(longest_rollback.path == Path::RestoreTurnRollback &&
               longest_rollback.frontier == 1000,
           "longest matching rollback head wins");
    const std::array<q36::detail::PrefillReuseHead, 2> rollback_and_ladder{
        q36::detail::PrefillReuseHead{1000u, Kind::TurnRollback},
        q36::detail::PrefillReuseHead{24576u, Kind::Ladder}};
    const auto ladder_over_rollback = q36::detail::select_resident_prefill_reuse(
        false, 30000, false, 0, Path::RestoreTurnCheckpoint, rollback_and_ladder);
    expect(ladder_over_rollback.path == Path::RestoreContextCheckpoint &&
               ladder_over_rollback.frontier == 24576,
           "longer ladder beats shorter rollback");
    const auto prefix_hits_rollback_not_ladder = q36::detail::select_resident_prefill_reuse(
        false, 30000, false, 0, Path::RestoreTurnCheckpoint, rollback_e1);
    expect(prefix_hits_rollback_not_ladder.path == Path::RestoreTurnRollback &&
               prefix_hits_rollback_not_ladder.frontier == 1000,
           "prefix that misses the ladder still hits rollback E");

    expect(q36::detail::restore_may_d2d_staging(true, true),
           "staging D2D requires a published host head");
    expect(!q36::detail::restore_may_d2d_staging(true, false),
           "occupied staging without a head must not D2D");
    expect(!q36::detail::restore_may_d2d_staging(false, true),
           "host unpack when staging identity does not match");

    const std::array<std::uint32_t, 2> two_ladders{24576u, 36864u};
    const auto longest = q36::detail::select_resident_prefill_reuse(
        false, 40000, true, 8000, Path::RestoreTurnCheckpoint, two_ladders);
    expect(longest.path == Path::RestoreContextCheckpoint && longest.frontier == 36864,
           "longest ladder head beats shorter rewrite");
    const std::array<std::uint32_t, 1> short_ladder{24576u};
    const auto rewrite_longer = q36::detail::select_resident_prefill_reuse(
        false, 40000, true, 36864, Path::RestoreTurnCheckpoint, short_ladder);
    expect(rewrite_longer.path == Path::RestoreTurnCheckpoint && rewrite_longer.frontier == 36864,
           "longer rewrite beats shorter ladder");
    const std::array<std::uint32_t, 1> same_f{24576u};
    const auto rewrite_tie = q36::detail::select_resident_prefill_reuse(
        false, 30000, true, 24576, Path::RestoreTurnCheckpoint, same_f);
    expect(rewrite_tie.path == Path::RestoreTurnCheckpoint && rewrite_tie.frontier == 24576,
           "rewrite wins a product-mark same-F tie");
    const auto response_tie = q36::detail::select_resident_prefill_reuse(
        false, 30000, true, 24576, Path::RestoreResponseCheckpoint, same_f);
    expect(response_tie.path == Path::RestoreResponseCheckpoint &&
               response_tie.frontier == 24576,
           "response rewrite also wins a same-F ladder tie");

    using Slots = q36::detail::mechanism_slots::LinearStateSlots;
    expect(Slots::state_slot_count(1, true) == 3, "C=1 MTP/DFlash GDN pool is 2C+1");
    expect(Slots::staging_state_slot(1) == 2, "C=1 staging is slot 2C");
    expect(Slots::current_state_slot(0, 1) == 0, "C=1 current is slot 0");
    expect(Slots::rewrite_checkpoint_state_slot(0, 1) == 1, "C=1 rewrite is slot C");
    expect(Slots::state_slot_count(1, false) == 2, "ordinary GDN pool is 2C, no staging");
    expect(Slots::state_slot_count(2, true) == 5, "C=2 MTP/DFlash GDN pool is 2C+1");
    expect(Slots::staging_state_slot(2) == 4, "C=2 staging is slot 2C");
    expect(Slots::current_state_slot(1, 2) == 1, "C=2 lane 1 current is slot 1");
    expect(Slots::rewrite_checkpoint_state_slot(1, 2) == 3, "C=2 lane 1 rewrite is slot C+1");
    expect(Slots::state_slot_count(8, true) == 17, "C=8 MTP/DFlash GDN pool is 2C+1");
    expect(Slots::staging_state_slot(8) == 16, "C=8 staging is slot 2C");
    expect(Slots::current_state_slot(7, 8) == 7, "C=8 last-lane current is slot 7");
    expect(Slots::rewrite_checkpoint_state_slot(7, 8) == 15, "C=8 last-lane rewrite is slot C+7");
    expect(Slots::staging_state_slot(3) == 6, "C=3 staging is slot 2C");
}

} // namespace

int main() {
    test_topology();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_prefix_hash_and_dflash_gate();
    test_prefill_context_marks();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}

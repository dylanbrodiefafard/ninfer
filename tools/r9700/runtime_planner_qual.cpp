#include "core/device.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/swa.h"
#include "ops/r9700/gdn_recurrence/gdn_verify_profile.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"
#include "targets/qwen3_8_27b/impl/variant.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"

#define NINFER_QWEN3_VARIANT    ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/layouts.h"
#include "targets/qwen3/impl/runtime/workspace_recipe.h"
#undef NINFER_QWEN3_RUNTIME_NS
#undef NINFER_QWEN3_VARIANT

#include <ninfer/targets/qwen3/runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Variant = ninfer::targets::qwen3_8_27b::detail::Variant;

// Each single-sequence (B=1) Device Graph definition is lowered to persistent decode kernels and
// charged this allowance (layouts_impl.h kR9700PersistentDecodeDefinitionBytes).
constexpr std::size_t kLoweredDefinitionBytes = 6ULL * 1024ULL * 1024ULL;

void require(bool condition, const char* message);

template <class Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

void qualify_host_request_lane_cap() {
    constexpr auto max_batch        = static_cast<std::int32_t>(ninfer::kMaximumConcurrency);
    constexpr auto too_many         = max_batch + 1;
    constexpr auto dflash_max_batch = static_cast<std::int32_t>(ninfer::kDFlashMaximumConcurrency);
    constexpr auto dflash_too_many  = dflash_max_batch + 1;
    constexpr auto q4        = ninfer::QType::Q4G64_F16S;
    constexpr auto profile   = Variant::WeightsProfile::R9700Q4G64DFlash2Q4Evaluation;

    (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16, dflash_max_batch);
    (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, dflash_max_batch);
    (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes({0, 4096}, 1, 16,
                                                                            dflash_max_batch);
    (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(48, max_batch, 16);
    (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 1, 16);
    (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
        profile, ninfer::targets::qwen3::TextPhase::Verify, max_batch, 2, 16);

    require_invalid_argument(
        [&] {
            (void)ninfer::ops::dflash2_path_select_workspace_capacity_bytes(q4, 1, 16,
                                                                            dflash_too_many);
        },
        "DFlash path selection accepted an oversized request batch");
    require_invalid_argument(
        [&] { (void)ninfer::ops::swa_workspace_capacity_bytes({0, 4096}, 1, 16, dflash_too_many); },
        "SWA accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                {0, 4096}, 1, 16, dflash_too_many);
        },
        "bidirectional attention accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)ninfer::ops::gated_delta_net_replay_record_workspace_capacity_bytes(48, too_many,
                                                                                      16);
        },
        "GDN replay accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 1, 16);
        },
        "target GDN snapshot accepted an oversized request batch");
    require_invalid_argument(
        [&] {
            (void)Variant::gdn_input_projection_record_workspace_capacity_bytes(
                profile, ninfer::targets::qwen3::TextPhase::Verify, too_many, 2, 16);
        },
        "target GDN record accepted an oversized request batch");
}

void qualify_host_split512_routing() {
    namespace kv  = ninfer::ops::r9700::kv;
    namespace q27 = ninfer::targets::qwen3_8_27b::detail;
    require(!kv::use_split512_attention(1U, 8192U, false) &&
                !kv::use_split512_attention(1U, 8192U, true) &&
                !kv::use_split512_attention(4U, 8191U, true) &&
                !kv::use_split512_attention(4U, 8192U, false) &&
                kv::use_split512_attention(4U, 8192U, true) &&
                !kv::use_split512_attention(2U, 32768U, true) &&
                !kv::use_split512_attention(5U, 32768U, true),
            "split-512 production selector differs from its admitted boundary");
    require(!kv::use_dense_prefill_attention(127U, 127U) &&
                kv::use_dense_prefill_attention(128U, 128U) &&
                kv::use_dense_prefill_attention(4096U, 4096U) &&
                !kv::use_dense_prefill_attention(8193U, 8193U) &&
                kv::use_dense_prefill_attention(4096U, 8192U) &&
                kv::use_dense_prefill_attention(8192U, 32768U) &&
                kv::use_dense_prefill_attention(128U, 262144U),
            "dense-prefill production selector differs from its admitted boundary");
    require(q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, false) ==
                    kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(8192U, 1U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(1U, 8192U, true) == 0U &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, false) ==
                    kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(8192U, 4U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8192U, true) ==
                    kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(4U, 8192U) &&
                q27::r9700_full_attention_workspace_capacity_bytes(4U, 8191U, true) == 0U,
            "packed/split-512 caller-owned workspace selection differs");
    // A plan sized for its maximum row count and visible frontier covers every narrower
    // host-fixed call (packed 1..8 and mid-row 9..1023 rows) at every smaller frontier.
    for (const std::uint32_t max_rows : {9U, 16U, 20U, 40U, 127U, 300U, 2048U}) {
        for (std::size_t envelope = 64U; envelope <= 40960U; envelope += 64U) {
            const std::size_t planned =
                q27::r9700_full_attention_workspace_capacity_bytes(max_rows, envelope, false);
            for (std::uint32_t rows = 1U; rows <= std::min(max_rows, kv::kMidRowsMaximumRows);
                 ++rows) {
                for (const std::size_t frontier : {envelope, envelope - 63U, envelope / 2U}) {
                    if (frontier < rows) continue;
                    const std::size_t call =
                        rows <= 8U
                            ? kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(frontier,
                                                                                      rows)
                            : kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(frontier, rows);
                    require(call <= planned,
                            "attention workspace plan does not cover a narrower call");
                }
            }
        }
    }
    require(q27::r9700_full_attention_workspace_capacity_bytes(2048U, 262144U, false) <=
                std::size_t{kv::kMidRowsPartialRows} * 24U * 258U * sizeof(float),
            "dense prefill plans more attention workspace than the mid-row bound");
    const auto has_classes = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                                std::initializer_list<std::uint32_t> wanted) {
        return std::all_of(wanted.begin(), wanted.end(), [&](std::uint32_t value) {
            return std::any_of(profiles.begin(), profiles.end(), [&](const auto& profile) {
                return profile.topology_class == value;
            });
        });
    };
    const auto class_at = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                             std::uint32_t frontier) {
        const auto found = std::find_if(profiles.begin(), profiles.end(), [&](const auto& profile) {
            return profile.min <= frontier && frontier <= profile.max;
        });
        if (found == profiles.end()) {
            throw std::runtime_error("split-512 graph profile coverage is incomplete");
        }
        return found->topology_class;
    };
    const auto ordinary     = Variant::ordinary_graph_profiles(32768U);
    const auto mtp          = Variant::mtp_graph_profiles(32768U, 3U);
    const auto dflash       = Variant::dflash_graph_profiles(32768U, 3U, 1U, 4U);
    const auto only_classes = [](const std::vector<Variant::GraphExecutionProfile>& profiles,
                                 std::initializer_list<std::uint32_t> allowed) {
        return std::all_of(profiles.begin(), profiles.end(), [&](const auto& profile) {
            return std::find(allowed.begin(), allowed.end(), profile.topology_class) !=
                   allowed.end();
        });
    };
    require(has_classes(ordinary, {3U}) && only_classes(ordinary, {0U, 3U}) &&
                has_classes(mtp, {2U, 3U}) && only_classes(mtp, {0U, 2U, 3U}) &&
                has_classes(dflash, {3U}) && only_classes(dflash, {0U, 3U}) &&
                class_at(ordinary, 8190U) == 3U && class_at(ordinary, 8191U) == 3U &&
                class_at(mtp, 8185U) == 2U && class_at(mtp, 8186U) == 3U &&
                class_at(dflash, 8188U) == 3U &&
                class_at(Variant::ordinary_graph_profiles(63U), 62U) == 0U,
            "split-512 Device Graph topology classes are incomplete");
    for (const std::uint32_t width : {5U, 6U}) {
        for (std::uint32_t batch = 1; batch <= ninfer::kDFlashMaximumConcurrency; ++batch) {
            const auto profiles = Variant::dflash_graph_profiles(16384U, width - 1U, batch, width);
            const bool g16      = ninfer::targets::qwen3::detail::kR9700TextKVValueGroup == 16;
            const std::uint32_t verify_class = g16 ? 3U : 0U;
            require(class_at(profiles, 8191U - width) == verify_class &&
                        class_at(profiles, 8192U - width) == verify_class &&
                        class_at(profiles, 16383U) == verify_class &&
                        class_at(profiles, 4095U - width) == verify_class &&
                        class_at(profiles, 4096U - width) == verify_class,
                    "DFlash W5/W6 graph key omitted the split verify topology");
            const auto tiny = Variant::dflash_graph_profiles(63U, width - 1U, batch, width);
            require(class_at(tiny, 62U) == 0U,
                    "DFlash graph key must use capacity-clipped visible context");
        }
    }
    std::printf("r9700_runtime_planner: PASS host split-512 production routing\n");
}

void qualify_host_attention_parity_routing() {
    namespace kv  = ninfer::ops::r9700::kv;
    namespace q27 = ninfer::targets::qwen3_8_27b::detail;
    // Split verify partials: min(64, ceil(context/256)) chunks x rows x 24 heads x (256 numerator
    // + origin + denominator) FP32.
    const auto verify_bytes = [](std::uint32_t rows, std::size_t context) {
        return std::min<std::size_t>(64U, (context + 255U) / 256U) * rows * 24U * 258U *
               sizeof(float);
    };
    const std::size_t kW5C134Bytes = verify_bytes(5U, 134U);
    const std::size_t kW5C140Bytes = verify_bytes(5U, 140U);
    const std::size_t kW6C134Bytes = verify_bytes(6U, 134U);
    const std::size_t kW6C135Bytes = verify_bytes(6U, 135U);
    const bool text_enabled        = kv::kTextP129WmmaTailCandidate;
    for (const std::size_t context : {64U, 133U, 4100U, 8191U, 8192U, 15200U, 32768U, 262144U}) {
        for (std::uint32_t rows = 1U; rows <= 8U; ++rows) {
            require(kv::use_packed_decode_attention(rows, context, false) &&
                        q27::r9700_full_attention_workspace_capacity_bytes(rows, context, false) >=
                            verify_bytes(rows, context),
                    "packed decode route or split workspace omitted an admitted context");
        }
    }
    require(!kv::use_packed_decode_attention(4U, 63U, false) &&
                !kv::use_packed_decode_attention(4U, 262145U, false) &&
                !kv::use_packed_decode_attention(4U, 134U, true) &&
                !kv::use_packed_decode_attention(0U, 134U, false) &&
                !kv::use_packed_decode_attention(9U, 135U, false) &&
                kv::use_text_p129_wmma_tail(129U, 129U) == text_enabled &&
                !kv::use_text_p129_wmma_tail(128U, 128U) &&
                !kv::use_text_p129_wmma_tail(129U, 130U),
            "attention parity selectors escaped their exact Text/packed cells");
    require(
        q27::r9700_full_attention_workspace_capacity_bytes(5U, 134U, false) == kW5C134Bytes &&
            q27::r9700_full_attention_workspace_capacity_bytes(5U, 140U, false) == kW5C140Bytes &&
            q27::r9700_full_attention_workspace_capacity_bytes(5U, 8191U, false) ==
                verify_bytes(5U, 8191U) &&
            q27::r9700_full_attention_workspace_capacity_bytes(5U, 134U, true) == 0U &&
            q27::r9700_full_attention_workspace_capacity_bytes(6U, 134U, false) == kW6C134Bytes &&
            q27::r9700_full_attention_workspace_capacity_bytes(6U, 135U, false) == kW6C135Bytes &&
            q27::r9700_full_attention_workspace_capacity_bytes(6U, 135U, true) == 0U,
        "packed decode production workspace escaped width/tree selection");
    // The 129-row plan also covers every narrower host-fixed call at that frontier.
    std::size_t narrower = kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(129U, 8U);
    for (std::uint32_t rows = 9U; rows <= 129U; ++rows)
        require(kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(129U, rows) <=
                    kv::fp8_int4_kv_attention_mid_rows_workspace_capacity_bytes(129U, 129U),
                "mid-row capacity bound does not cover a narrower call");
    narrower =
        std::max(narrower, kv::fp8_int4_kv_attention_mid_rows_workspace_capacity_bytes(129U, 129U));
    require(
        q27::r9700_full_attention_workspace_capacity_bytes(129U, 129U, false) ==
            std::max(narrower, text_enabled
                                   ? q27::r9700_full_attention_score_workspace_capacity_bytes(129U)
                                   : std::size_t{0U}),
        "Text P129 tail candidate changed the fused dense caller-owned workspace");
    std::printf("r9700_runtime_planner: PASS host attention parity routing/capacity "
                "text_p129_selector=%u\n",
                text_enabled ? 1U : 0U);
}

void qualify_host_ordinary_graph_allowance() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    runtime::SequencePlanningInputs inputs{};
    inputs.weights_profile = Variant::WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation;
    inputs.prefill_chunk   = 2048U;
    inputs.speculative_backend = ninfer::SpeculativeBackend::None;
    inputs.features            = {.vision = false, .speculative = ninfer::SpeculativeBackend::None};
    for (const std::uint32_t capacity : {256U, 1024U, 4096U, 4224U}) {
        const std::size_t definitions = capacity == 256U    ? 2U
                                        : capacity == 1024U ? 3U
                                        : capacity == 4096U ? 4U
                                                            : 5U;
        for (std::uint32_t concurrency = 1U; concurrency <= ninfer::kMaximumConcurrency;
             ++concurrency) {
            inputs.capacity         = capacity;
            inputs.max_concurrency  = concurrency;
            inputs.use_device_graph = true;
            const auto page_size    = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
            const std::uint32_t physical_pages =
                concurrency * ((capacity + page_size - 1U) / page_size);
            const auto plan =
                runtime::build_sequence_candidate_for_qualification(inputs, physical_pages);
            const std::size_t expected =
                (24ULL + 20ULL + 4ULL * definitions * concurrency) * 1024ULL * 1024ULL +
                definitions * kLoweredDefinitionBytes; // B=1 definitions
            require(plan->graph_definition_count == definitions * concurrency &&
                        plan->graph_executable_count == definitions * concurrency &&
                        plan->graph_allowance_bytes == expected,
                    "ordinary allowance omitted immutable profile executables");
            if (capacity == 256U && concurrency == 1U) {
                require(plan->graph_allowance_bytes >= 44ULL * 1024ULL * 1024ULL,
                        "ordinary allowance misses observed 44 MiB C1/capacity256 preparation");
            }
            if (capacity == 4224U && concurrency == 1U) {
                require(plan->graph_allowance_bytes >= 54525952ULL,
                        "ordinary allowance misses observed 52 MiB tiled preparation");
            }
            inputs.use_device_graph = false;
            const auto eager =
                runtime::build_sequence_candidate_for_qualification(inputs, physical_pages);
            require(eager->graph_allowance_bytes == 0U && eager->graph_definition_count == 0U &&
                        eager->graph_executable_count == 0U,
                    "eager ordinary plan reserves graph residency");
        }
    }
    std::printf(
        "r9700_runtime_planner: PASS host ordinary C1..16/context1024,4096,4224 allowance\n");
}

void qualify_host_hybrid_allocation_bound() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    for (const auto profile : {Variant::WeightsProfile::R9700Q4G64Evaluation,
                               Variant::WeightsProfile::R9700Q4W8Evaluation,
                               Variant::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation}) {
        runtime::SequencePlanningInputs inputs{};
        inputs.weights_profile     = profile;
        inputs.capacity            = 262144U;
        inputs.max_concurrency     = 4U;
        inputs.prefill_chunk       = 2048U;
        inputs.speculative_backend = ninfer::SpeculativeBackend::None;
        inputs.features = {.vision = false, .speculative = ninfer::SpeculativeBackend::None};
        inputs.use_device_graph = true;
        const auto minimum  = runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
        const auto adjacent = runtime::build_sequence_candidate_for_qualification(inputs, 4097U);
        const auto distant  = runtime::build_sequence_candidate_for_qualification(inputs, 4233U);
        const std::size_t logical = minimum->persistent.bytes + minimum->workspace.capacity +
                                    minimum->request_transient_capacity_bytes +
                                    minimum->graph_allowance_bytes;
        const auto expected_bound =
            profile == Variant::WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ? 4U << 20U
                                                                                      : 0U;
        require(minimum->device_reservation_bytes == logical + expected_bound,
                "hybrid physical arena bound missing or changed nonhybrid reservation");
        const auto increment =
            adjacent->device_reservation_bytes - minimum->device_reservation_bytes;
        require(distant->device_reservation_bytes ==
                    minimum->device_reservation_bytes + 137U * increment,
                "physical arena bound made the KV reservation curve non-affine");
    }
    for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency; ++concurrency) {
        for (const auto backend :
             {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::Mtp}) {
            if (backend == ninfer::SpeculativeBackend::DFlash &&
                concurrency > ninfer::kDFlashMaximumConcurrency)
                continue;
            runtime::SequencePlanningInputs inputs{};
            inputs.weights_profile     = Variant::WeightsProfile::R9700Lq4;
            inputs.capacity            = 262144U;
            inputs.max_concurrency     = concurrency;
            inputs.prefill_chunk       = 2048U;
            inputs.draft_window        = backend == ninfer::SpeculativeBackend::DFlash ? 7U : 5U;
            inputs.speculative_backend = backend;
            inputs.features            = {.vision = false, .speculative = backend};
            inputs.use_device_graph    = true;
            const auto without_disk =
                runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
            inputs.kv_disk_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
            const auto with_disk =
                runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
            // Independently derive packed page geometry from the represented codec:
            // 256 FP8 keys, 128 packed INT4 value bytes, FP16 scales per KV head.
            constexpr std::size_t head_bytes =
                256U + 128U + 2U * (256U / ninfer::targets::qwen3::detail::kR9700TextKVValueGroup);
            constexpr std::size_t page_bytes    = std::size_t{16U} * 4U * head_bytes * 64U;
            constexpr std::size_t physical_unit = 2ULL * 1024ULL * 1024ULL;
            const std::size_t expected_disk =
                (page_bytes * 32U + physical_unit - 1U) / physical_unit * physical_unit +
                (backend == ninfer::SpeculativeBackend::DFlash ? 1U : 2U) * physical_unit;
            require(with_disk->device_reservation_bytes - without_disk->device_reservation_bytes ==
                        expected_disk + physical_unit,
                    "KV capacity omits the SSD restore device allocations");
            const auto next = runtime::build_sequence_candidate_for_qualification(inputs, 4097U);
            require(next->device_reservation_bytes - with_disk->device_reservation_bytes ==
                        page_bytes * (backend == ninfer::SpeculativeBackend::Mtp ? 17U : 16U) / 16U,
                    "SSD restore staging changed the affine per-page KV capacity cost");
        }
    }
    std::printf("r9700_runtime_planner: PASS host arena bounds and C1..16 SSD affine capacity\n");
}

void qualify_host_dflash_graph_allowance() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    for (std::uint32_t concurrency = 1U; concurrency <= ninfer::kDFlashMaximumConcurrency;
         ++concurrency) {
        for (const std::uint32_t capacity : {64U, 1024U, 4240U, 262144U}) {
            for (const std::uint32_t k : {1U, 3U, 7U}) {
                runtime::SequencePlanningInputs inputs{};
                inputs.weights_profile     = Variant::WeightsProfile::R9700Q4G64Evaluation;
                inputs.capacity            = capacity;
                inputs.max_concurrency     = concurrency;
                inputs.prefill_chunk       = std::min(capacity, 2048U);
                inputs.draft_window        = k;
                inputs.adaptive_draft      = k == 7U;
                inputs.speculative_backend = ninfer::SpeculativeBackend::DFlash;
                inputs.proposal_head       = ninfer::ProposalHead::Optimized;
                inputs.features            = {.vision        = false,
                                              .speculative   = ninfer::SpeculativeBackend::DFlash,
                                              .proposal_head = ninfer::ProposalHead::Optimized};
                inputs.use_device_graph    = true;
                const auto plan            = runtime::build_sequence_candidate_for_qualification(
                    inputs, concurrency * ((capacity + 63U) / 64U));
                std::size_t definitions = 0U;
                std::size_t lowered     = 0U;
                for (const auto draft : ninfer::targets::qwen3::adaptive_draft_ks(
                         inputs.speculative_backend, k, inputs.adaptive_draft)) {
                    const auto width =
                        ninfer::targets::qwen3::dflash_verify_width<Variant::DFlashConfig>(draft);
                    for (std::uint32_t batch = 1U; batch <= concurrency; ++batch) {
                        const auto count =
                            Variant::dflash_graph_profiles(capacity, draft, batch, width).size();
                        definitions += count;
                        if (batch == 1U) lowered += count;
                    }
                }
                require(plan->graph_definition_count == definitions &&
                            plan->graph_executable_count == definitions,
                        "DFlash immutable executable inventory omits a reachable profile");
                require(plan->graph_allowance_bytes ==
                            (48ULL + 20ULL + 4ULL * definitions) * 1024ULL * 1024ULL +
                                lowered * kLoweredDefinitionBytes,
                        "DFlash reservation omits immutable executable residency");
                inputs.use_device_graph = false;
                const auto eager        = runtime::build_sequence_candidate_for_qualification(
                    inputs, concurrency * ((capacity + 63U) / 64U));
                require(eager->graph_allowance_bytes == 0 && eager->graph_executable_count == 0,
                        "eager DFlash reserves graph residency");
            }
        }
    }
    // These constants come from independent physical measurements, not a model
    // of executable counts. Keep the production capacity regression explicit.
    constexpr std::array<std::size_t, ninfer::kDFlashMaximumConcurrency> graph_mib{
        318, 578, 842, 1102, 1364, 1624, 1884, 2144};
    constexpr std::array<std::size_t, ninfer::kDFlashMaximumConcurrency> host_argument_graph_mib{
        72, 56, 60, 60, 62, 62, 62, 62};
    const auto setting = [](const char* name, const char* expected) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, expected) == 0;
    };
    const bool bulk_arguments =
        setting("DEBUG_HIP_GRAPH_CLASSIC_PATH", "0") && setting("ROC_ENABLE_LARGE_BAR", "0") &&
        setting("ROC_SKIP_KERNEL_ARG_COPY", "0") && setting("HSA_DISABLE_FRAGMENT_ALLOCATOR", "1");
    const auto& measured_graph_mib = bulk_arguments ? host_argument_graph_mib : graph_mib;
    constexpr std::array<std::size_t, ninfer::kMaximumConcurrency> main_driver_mib{
        6, 6, 4, 4, 2, 4, 4, 6, 32, 32, 32, 34, 34, 34, 34, 34};
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    constexpr bool production_attention = false;
#else
    constexpr bool production_attention = true;
#endif
    constexpr auto planes = ninfer::targets::qwen3::detail::kR9700TextKVPlaneLayouts;
    const bool measured_build =
        production_attention && ninfer::targets::qwen3::detail::kR9700TextKVValueGroup == 16 &&
        planes.key == ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor &&
        planes.value == ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
        planes.value_scale == ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
        ninfer::ops::r9700::linear::kQ4ActivationBits == 8U &&
        ninfer::ops::r9700::linear::kW8ActivationBits == 8U &&
        ninfer::ops::r9700::linear::kQ4PrefillA4Families == 0U &&
        !ninfer::ops::r9700::linear::kDFlashSmallTCandidateEnabled &&
        !ninfer::ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled &&
        !ninfer::ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate &&
        ninfer::ops::r9700::kv::kFp8QkWmmaDecode &&
        !ninfer::ops::r9700::kv::kTextP129WmmaTailCandidate;
    for (std::uint32_t c = 1; c <= ninfer::kDFlashMaximumConcurrency; ++c) {
        runtime::SequencePlanningInputs inputs{};
        inputs.weights_profile     = Variant::WeightsProfile::R9700Lq4;
        inputs.capacity            = 262144U;
        inputs.max_concurrency     = c;
        inputs.prefill_chunk       = 2048U;
        inputs.draft_window        = 7U;
        inputs.adaptive_draft      = true;
        inputs.speculative_backend = ninfer::SpeculativeBackend::DFlash;
        inputs.proposal_head       = ninfer::ProposalHead::Optimized;
        inputs.features            = {.vision        = false,
                                      .speculative   = ninfer::SpeculativeBackend::DFlash,
                                      .proposal_head = ninfer::ProposalHead::Optimized};
        inputs.use_device_graph    = true;
        const auto calibrated      = Variant::calibrated_dflash_graph_allowance_bytes(
            inputs.weights_profile, inputs.features, inputs.capacity, c, inputs.prefill_chunk,
            inputs.draft_window, inputs.adaptive_draft);
        require(calibrated.has_value() == measured_build,
                "production calibration applicability differs from measured build profile");
        for (const bool disk : {false, true}) {
            inputs.kv_disk_capacity_bytes = disk ? 1ULL << 30U : 0U;
            const auto plan = runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
            require(plan->graph_definition_count == std::size_t{65U} * c &&
                        plan->graph_executable_count == std::size_t{65U} * c,
                    "production immutable executable inventory changed; remeasure residency");
            if (measured_build)
                require(plan->graph_allowance_bytes ==
                            (measured_graph_mib[c - 1U] + 20U) * (1ULL << 20U),
                        "production C-specific graph reservation differs from measurement");
            const auto logical = plan->persistent.bytes + plan->workspace.capacity +
                                 plan->request_transient_capacity_bytes +
                                 plan->graph_allowance_bytes;
            constexpr std::size_t head_bytes =
                256U + 128U + 2U * (256U / ninfer::targets::qwen3::detail::kR9700TextKVValueGroup);
            constexpr std::size_t staging_bytes = std::size_t{16U} * 4U * head_bytes * 64U * 32U;
            const std::size_t disk_mib = (staging_bytes + (2U << 20U) - 1U) / (2U << 20U) * 2U + 2U;
            const std::size_t internal_mib =
                4U + 2U + main_driver_mib[c - 1U] + (disk ? disk_mib + 2U : 0U);
            require(plan->device_reservation_bytes == logical + internal_mib * (1ULL << 20U),
                    "NInfer-owned driver or SSD storage escaped the internal capacity budget");
        }
        for (const std::uint32_t chunk : {1024U, 4096U}) {
            const bool qualified = measured_build && c == 6U && chunk == 1024U && bulk_arguments;
            require(Variant::calibrated_dflash_graph_allowance_bytes(
                        inputs.weights_profile, inputs.features, inputs.capacity, c, chunk,
                        inputs.draft_window, inputs.adaptive_draft)
                            .has_value() == qualified,
                    "prefill chunk calibration escaped its measured concurrency/environment");
        }
    }
    for (std::uint32_t c = 1; c <= ninfer::kMaximumConcurrency; ++c) {
        for (const auto backend :
             {ninfer::SpeculativeBackend::Mtp, ninfer::SpeculativeBackend::None}) {
            const bool mtp = backend == ninfer::SpeculativeBackend::Mtp;
            runtime::SequencePlanningInputs inputs{};
            inputs.weights_profile     = Variant::WeightsProfile::R9700Lq4;
            inputs.capacity            = 262144U;
            inputs.max_concurrency     = c;
            inputs.prefill_chunk       = 2048U;
            inputs.draft_window        = mtp ? 5U : 0U;
            inputs.adaptive_draft      = mtp;
            inputs.speculative_backend = backend;
            inputs.features            = {.vision = false, .speculative = backend};
            inputs.use_device_graph    = true;
            const auto plan    = runtime::build_sequence_candidate_for_qualification(inputs, 4096U);
            const auto logical = plan->persistent.bytes + plan->workspace.capacity +
                                 plan->request_transient_capacity_bytes +
                                 plan->graph_allowance_bytes;
            require(plan->device_reservation_bytes ==
                        logical + (4U + 2U + main_driver_mib[c - 1U]) * (1ULL << 20U),
                    "MTP/ordinary driver reservation differs from the measured C-specific bound");
        }
    }

    // Exercise the measured chunk1024 admission independently of the caller's
    // shell environment, and restore every driver setting before returning.
    struct SavedSetting {
        const char* name;
        std::optional<std::string> value;

        explicit SavedSetting(const char* key) : name(key) {
            if (const char* current = std::getenv(key)) { value = current; }
        }

        ~SavedSetting() {
            if (value) {
                (void)::setenv(name, value->c_str(), 1);
            } else {
                (void)::unsetenv(name);
            }
        }
    } classic("DEBUG_HIP_GRAPH_CLASSIC_PATH"), large_bar("ROC_ENABLE_LARGE_BAR"),
        argument_copy("ROC_SKIP_KERNEL_ARG_COPY"), fragment("HSA_DISABLE_FRAGMENT_ALLOCATOR");

    const std::array<std::pair<const char*, const char*>, 4> bulk_settings{
        {{"DEBUG_HIP_GRAPH_CLASSIC_PATH", "0"},
         {"ROC_ENABLE_LARGE_BAR", "0"},
         {"ROC_SKIP_KERNEL_ARG_COPY", "0"},
         {"HSA_DISABLE_FRAGMENT_ALLOCATOR", "1"}}};
    for (const auto& [name, value] : bulk_settings)
        require(::setenv(name, value, 1) == 0, "cannot set graph qualification environment");
    const ninfer::targets::qwen3::StartupFeatures features{
        .vision        = false,
        .speculative   = ninfer::SpeculativeBackend::DFlash,
        .proposal_head = ninfer::ProposalHead::Optimized};
    const auto chunk_calibration = [&](Variant::WeightsProfile profile, auto selected_features,
                                       std::uint32_t capacity, std::uint32_t concurrency,
                                       std::uint32_t chunk, std::uint32_t drafts, bool adaptive) {
        return Variant::calibrated_dflash_graph_allowance_bytes(
            profile, selected_features, capacity, concurrency, chunk, drafts, adaptive);
    };
    for (std::uint32_t c = 1; c <= ninfer::kMaximumConcurrency; ++c) {
        const auto measured = chunk_calibration(Variant::WeightsProfile::R9700Lq4, features,
                                                262144U, c, 1024U, 7U, true);
        require(measured.has_value() == (measured_build && c == 6U),
                "chunk1024 calibration admitted an unmeasured concurrency");
        if (measured)
            require(*measured == 62ULL * 1024ULL * 1024ULL,
                    "C6/chunk1024 complete graph residency differs from measurement");
    }
    for (const auto& [name, value] : bulk_settings) {
        require(::setenv(name, value[0] == '0' ? "1" : "0", 1) == 0,
                "cannot vary graph qualification environment");
        require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, features, 262144U, 6U, 1024U,
                                   7U, true),
                "chunk1024 calibration admitted an unmeasured runtime environment");
        require(::setenv(name, value, 1) == 0, "cannot restore graph qualification environment");
    }
    for (const std::uint32_t capacity : {131072U, 262080U})
        require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, features, capacity, 6U, 1024U,
                                   7U, true),
                "chunk1024 calibration admitted an unmeasured context");
    for (const std::uint32_t drafts : {3U, 5U, 6U})
        require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, features, 262144U, 6U, 1024U,
                                   drafts, true),
                "chunk1024 calibration admitted an unmeasured draft inventory");
    require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, features, 262144U, 6U, 1024U, 7U,
                               false),
            "chunk1024 calibration admitted a fixed-draft inventory");
    auto incompatible   = features;
    incompatible.vision = true;
    require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, incompatible, 262144U, 6U, 1024U,
                               7U, true),
            "chunk1024 calibration admitted Vision allocations");
    incompatible               = features;
    incompatible.proposal_head = ninfer::ProposalHead::Full;
    require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, incompatible, 262144U, 6U, 1024U,
                               7U, true),
            "chunk1024 calibration admitted an unmeasured proposal head");
    incompatible             = features;
    incompatible.speculative = ninfer::SpeculativeBackend::Mtp;
    require(!chunk_calibration(Variant::WeightsProfile::R9700Lq4, incompatible, 262144U, 6U, 1024U,
                               7U, true),
            "chunk1024 calibration admitted a different speculative backend");
    require(!chunk_calibration(Variant::WeightsProfile::R9700Q4G64Evaluation, features, 262144U, 6U,
                               1024U, 7U, true),
            "chunk1024 calibration admitted an evaluation weight profile");
    std::printf("r9700_runtime_planner: PASS immutable DFlash C1..8/K/context inventory\n");
}

void qualify_host_mtp_graph_allowance() {
    namespace runtime = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;

    struct Case {
        std::uint32_t context, concurrency, drafts;
        bool adaptive;
        std::size_t definitions;
    };

    // Includes singleton, the reproduced short-context failure, full
    // context profiles/split topologies, and all three independently captured Ks.
    for (const auto c :
         {Case{64, 1, 3, false, 1}, Case{1024, 1, 3, false, 3}, Case{1024, 4, 5, false, 12},
          Case{262144, 1, 3, false, 11}, Case{262144, 4, 5, false, 32},
          Case{262144, 4, 5, true, 108}, Case{262144, 16, 5, true, 432}}) {
        runtime::SequencePlanningInputs inputs{
            .weights_profile     = Variant::WeightsProfile::R9700Q4G64Evaluation,
            .capacity            = c.context,
            .max_concurrency     = c.concurrency,
            .prefill_chunk       = std::min(c.context, 2048U),
            .draft_window        = c.drafts,
            .adaptive_draft      = c.adaptive,
            .speculative_backend = ninfer::SpeculativeBackend::Mtp,
            .proposal_head       = ninfer::ProposalHead::Full,
            .features            = {.vision        = false,
                                    .speculative   = ninfer::SpeculativeBackend::Mtp,
                                    .proposal_head = ninfer::ProposalHead::Full},
            .use_device_graph    = true,
        };
        const auto plan = runtime::build_sequence_candidate_for_qualification(
            inputs, c.concurrency * ((c.context + 63U) / 64U));
        require(plan->graph_definition_count == c.definitions &&
                    plan->graph_executable_count == c.definitions,
                "MTP fixed/adaptive graph inventory differs from its captured K/B profiles");
        const auto expected = (46ULL + 20ULL + 4ULL * c.definitions) * 1024ULL * 1024ULL +
                              c.definitions / c.concurrency * kLoweredDefinitionBytes;
        require(plan->graph_allowance_bytes == expected,
                "MTP allowance omitted immutable executable residency");
        inputs.use_device_graph = false;
        const auto eager        = runtime::build_sequence_candidate_for_qualification(
            inputs, c.concurrency * ((c.context + 63U) / 64U));
        require(eager->graph_allowance_bytes == 0 && eager->graph_definition_count == 0 &&
                    eager->graph_executable_count == 0,
                "eager MTP reserves graph residency");
    }
    std::printf("r9700_runtime_planner: PASS host MTP immutable graph allowance\n");
}

using WeightsProfile = ninfer::targets::qwen3_8_27b::detail::WeightsProfile;

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && NINFER_R9700_XATTENTION_STRIDE == 16 &&      \
    NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
constexpr std::uint32_t kCapacityEnvelopeContext       = 262144U;
constexpr std::uint32_t kCapacityEnvelopeChunk         = 4096U;
constexpr std::size_t kCapacityEnvelopeSparseLeafBytes = 601361408ULL;
constexpr std::size_t kCapacityEnvelopeWorkspaceBytes  = 1085967619ULL;
#endif

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::size_t qualify_plan(ninfer::DeviceContext& device, std::uint32_t concurrency,
                         ninfer::SpeculativeBackend backend, std::uint32_t drafts, bool vision,
                         std::uint32_t dflash_verify_width = 0,
                         WeightsProfile weights_profile    = WeightsProfile::R9700W8G32Candidate,
                         bool use_device_graph = false, std::uint32_t max_context = 2048,
                         std::uint32_t prefill_chunk = 128) {
    ninfer::EngineOptions options;
    options.device                   = device.device;
    options.max_context              = max_context;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency          = concurrency;
    options.prefill_chunk            = prefill_chunk;
    options.enable_vision            = vision;
    options.use_device_graph         = use_device_graph;
    options.speculative.backend      = backend;
    options.speculative.draft_tokens = drafts;
    options.speculative.dflash_verify_width = dflash_verify_width;

    auto planner =
        ninfer::targets::qwen3::make_sequence_planner<Variant>(device, options, weights_profile);
    const auto curve = planner.capacity_curve();
    require(curve.main_page_tokens == 64U, "planner page size is not the typed-cache page size");
    const std::uint32_t logical_pages = 1U + (max_context - 1U) / curve.main_page_tokens;
    require(curve.minimum_main_page_groups == std::max(logical_pages, concurrency),
            "planner minimum physical page count is wrong");
    require(curve.maximum_main_page_groups == concurrency * logical_pages,
            "planner maximum physical page count is wrong");
    require(curve.minimum_device_reservation_bytes != 0U,
            "planner produced an empty device reservation");

    auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);
    require(plan.capacity() == options.max_context, "plan logical capacity changed");
    require(plan.kv_capacity() >= options.max_context &&
                plan.kv_capacity() - options.max_context < curve.main_page_tokens,
            "plan physical KV capacity does not match page-rounded logical capacity");
    require(plan.max_concurrency() == concurrency, "plan concurrency changed");
    require(plan.device_reservation_bytes() == curve.minimum_device_reservation_bytes,
            "plan reservation disagrees with its affine curve");
    require(plan.workspace_capacity_bytes() != 0U, "plan workspace is empty");
    require((plan.request_transient_capacity_bytes() != 0U) == vision,
            "plan Vision transient ownership is wrong");
    // Rebuild the runtime allocation prefix through ordinary Text prefill attention. Comparing
    // only the leaf scratch size with the global arena would miss live query/gate/K/V/result
    // tensors that precede Variant::full_attention in the same scope.
    {
        using TextConfig         = Variant::TextConfig;
        const std::int32_t chunk = static_cast<std::int32_t>(
            std::min<std::uint32_t>(options.prefill_chunk, options.max_context));
        ninfer::WorkspaceLayoutBuilder runtime_prefix;
        (void)
            ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::text_prefill_roots<
                TextConfig>(runtime_prefix, chunk, vision ? 3 : 0, vision ? chunk : 0);
        {
            auto stage = runtime_prefix.scope();
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_projection<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch            = runtime_prefix.scope();
                const std::size_t bytes = Variant::attention_projection_workspace_capacity_bytes(
                    weights_profile, ninfer::targets::qwen3::TextPhase::Prefill, 1, chunk);
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
            (void)ninfer::targets::qwen3::detail::qwen3_8_27b_r9700::workspace_recipe::
                text_attention_results<TextConfig>(runtime_prefix, chunk);
            {
                auto scratch = runtime_prefix.scope();
                const std::size_t bytes =
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
                    Variant::text_prefill_attention_workspace_capacity_bytes(chunk,
                                                                             options.max_context);
#else
                    Variant::full_attention_workspace_capacity_bytes(chunk, options.max_context,
                                                                     false);
#endif
                if (bytes != 0U) (void)runtime_prefix.alloc_bytes(bytes);
            }
        }
        require(plan.impl_->workspace.text_prefill >= runtime_prefix.peak_bytes(1),
                "Text-prefill plan does not cover its runtime attention allocation prefix");
    }
    const std::uint32_t expected_dflash_width =
        backend == ninfer::SpeculativeBackend::DFlash
            ? ninfer::targets::qwen3::dflash_verify_width<Variant::DFlashConfig>(
                  drafts, dflash_verify_width)
            : 0U;
    require(plan.impl_->dflash_verify_width == expected_dflash_width,
            "plan did not preserve its fixed DFlash verify width");
    if (use_device_graph) {
        std::size_t definitions = 0U;
        std::size_t lowered     = 0U;
        for (std::uint32_t batch = 1U; batch <= concurrency; ++batch) {
            const auto count_profiles = [&](const auto& profiles) {
                definitions += profiles.size();
                if (batch == 1U) lowered += profiles.size();
            };
            if (backend == ninfer::SpeculativeBackend::None) {
                count_profiles(Variant::ordinary_graph_profiles(max_context));
            } else if (backend == ninfer::SpeculativeBackend::Mtp) {
                count_profiles(Variant::mtp_graph_profiles(max_context, drafts));
            } else {
                count_profiles(Variant::dflash_graph_profiles(max_context, drafts, batch,
                                                              expected_dflash_width));
            }
        }
        const std::size_t family_mib = backend == ninfer::SpeculativeBackend::None  ? 24U
                                       : backend == ninfer::SpeculativeBackend::Mtp ? 46U
                                                                                    : 48U;
        require(plan.impl_->graph_definition_count == definitions &&
                    plan.impl_->graph_executable_count == definitions,
                "immutable graph inventory omitted a reachable exact-B profile");
        require(plan.impl_->graph_allowance_bytes ==
                    (family_mib + 20U + 4U * definitions) * 1024U * 1024U +
                        lowered * kLoweredDefinitionBytes,
                "immutable graph allowance differs from its complete profile inventory");
    }
    return plan.workspace_capacity_bytes();
}

void qualify_host_xattention_capacity_envelope() {
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && NINFER_R9700_XATTENTION_STRIDE == 16 &&      \
    NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
    const std::size_t sparse_leaf = Variant::text_prefill_attention_workspace_capacity_bytes(
        static_cast<std::int32_t>(kCapacityEnvelopeChunk),
        static_cast<std::int32_t>(kCapacityEnvelopeContext));
    require(sparse_leaf == kCapacityEnvelopeSparseLeafBytes,
            "S16/tau900 Text-prefill sparse leaf workspace envelope changed");
    std::printf("r9700_runtime_planner: PASS host XAttention S16/tau900 T=%u context=%u "
                "sparse_leaf_bytes=%zu expected_planner_workspace_bytes=%zu\n",
                kCapacityEnvelopeChunk, kCapacityEnvelopeContext, sparse_leaf,
                kCapacityEnvelopeWorkspaceBytes);
#else
    throw std::runtime_error(
        "host XAttention capacity envelope requires the S16/tau900 qualification build");
#endif
}

void print_host_hybrid_capacity_authority() {
    namespace runtime                      = ninfer::targets::qwen3::detail::qwen3_8_27b_r9700;
    constexpr std::uint32_t capacity       = 262144U;
    constexpr std::uint32_t page_tokens    = 64U;
    constexpr std::uint32_t minimum_groups = capacity / page_tokens;
    std::printf("target,weights_profile,capacity_tokens,page_tokens,prefill_chunk,kv_value_group,"
                "speculative_backend,draft_tokens,proposal_head,device_graph,"
                "concurrency,minimum_groups,maximum_groups,minimum_sequence_bytes,workspace_bytes,"
                "graph_allowance_bytes,request_transient_bytes,minimum_reservation_bytes,"
                "kv_payload_bytes,kv_increment_bytes\n");
    for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency; ++concurrency) {
        const runtime::SequencePlanningInputs inputs{
            .weights_profile          = WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation,
            .capacity                 = capacity,
            .max_concurrency          = concurrency,
            .prefill_chunk            = 8192U,
            .draft_window             = 3U,
            .dflash_verify_width      = 0U,
            .speculative_backend      = ninfer::SpeculativeBackend::Mtp,
            .proposal_head            = ninfer::ProposalHead::Optimized,
            .features                 = {.vision        = false,
                                         .speculative   = ninfer::SpeculativeBackend::Mtp,
                                         .proposal_head = ninfer::ProposalHead::Optimized},
            .use_device_graph         = true,
            .device                   = 0,
            .kv_ram_capacity_bytes    = 0U,
            .context_checkpoint_marks = {},
        };
        const auto minimum =
            runtime::build_sequence_candidate_for_qualification(inputs, minimum_groups);
        const auto adjacent =
            runtime::build_sequence_candidate_for_qualification(inputs, minimum_groups + 1U);
        const std::size_t increment =
            adjacent->device_reservation_bytes - minimum->device_reservation_bytes;
        std::printf("qwen3_8_27b_r9700,R9700Q4G64Fp8FourRoleN16K16Evaluation,"
                    "262144,64,8192,16,mtp,3,optimized,1,"
                    "%u,%u,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
                    concurrency, minimum_groups, concurrency * minimum_groups,
                    minimum->persistent.bytes, minimum->workspace.capacity,
                    minimum->graph_allowance_bytes, minimum->request_transient_capacity_bytes,
                    minimum->device_reservation_bytes, minimum->persistent.kv_payload_bytes,
                    increment);
    }
}

std::uint32_t parse_host_width_argument(const char* text, const char* label) {
    std::uint32_t value = 0;
    const std::string_view input{text};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return value;
}

void print_host_hybrid_width_authority(int argc, char** argv) {
    if (argc != 6) {
        throw std::invalid_argument(
            "--host-hybrid-widths-csv requires PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH");
    }
    const std::uint32_t prefill      = parse_host_width_argument(argv[2], "prefill width");
    const std::uint32_t concurrency  = parse_host_width_argument(argv[3], "maximum concurrency");
    const std::uint32_t mtp_width    = parse_host_width_argument(argv[4], "MTP width");
    const std::uint32_t dflash_width = parse_host_width_argument(argv[5], "DFlash width");
    if (concurrency > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("hybrid width inventory exceeds product concurrency");
    }
    const std::uint32_t verify_widths[]{mtp_width, dflash_width};
    const auto widths = Variant::ExecutionState::eager_widths(prefill, concurrency, verify_widths);
    std::printf("prefill,max_concurrency,mtp_width,dflash_width,prepared_width\n");
    for (const std::uint32_t width : widths) {
        std::printf("%u,%u,%u,%u,%u\n", prefill, concurrency, mtp_width, dflash_width, width);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--host-xattention-capacity-envelope") {
            qualify_host_xattention_capacity_envelope();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-request-lane-cap") {
            qualify_host_request_lane_cap();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-split512-routing") {
            qualify_host_split512_routing();
            qualify_host_hybrid_allocation_bound();
            qualify_host_ordinary_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-dflash-graph-allowance") {
            qualify_host_dflash_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-mtp-graph-allowance") {
            qualify_host_mtp_graph_allowance();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-attention-parity-routing") {
            qualify_host_attention_parity_routing();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-hybrid-capacity-csv") {
            print_host_hybrid_capacity_authority();
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--host-hybrid-widths-csv") {
            print_host_hybrid_width_authority(argc, argv);
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: ninfer_r9700_runtime_planner_qual "
                "[--host-xattention-capacity-envelope|--host-request-lane-cap|"
                "--host-split512-routing|--host-dflash-graph-allowance|--host-mtp-graph-allowance|"
                "--host-attention-parity-routing|"
                "--host-hybrid-capacity-csv|"
                "--host-hybrid-widths-csv PREFILL MAX_CONCURRENCY MTP_WIDTH DFLASH_WIDTH]");
        }
        ninfer::DeviceContext device(0);
        for (std::uint32_t concurrency = 1; concurrency <= ninfer::kMaximumConcurrency;
             ++concurrency) {
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, true);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::None, 0, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32768);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, 32774);
            if (concurrency > ninfer::kDFlashMaximumConcurrency) continue;
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 4, false);
            qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 5, false, 6);
            if (concurrency == 1U) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, 1, false, 2,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true, 128);
            }
            for (std::uint32_t k = 1; k <= 5; ++k) {
                qualify_plan(device, concurrency, ninfer::SpeculativeBackend::DFlash, k, false, 0,
                             WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, true, 32768U + k + 1U);
            }
        }
        const std::size_t w8_workspace = qualify_plan(device, ninfer::kMaximumConcurrency,
                                                      ninfer::SpeculativeBackend::None, 0, true);
        const std::size_t mixed_workspace =
            qualify_plan(device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0,
                         true, 0, WeightsProfile::R9700Q4W8Evaluation);
        const std::size_t q4_workspace =
            qualify_plan(device, ninfer::kMaximumConcurrency, ninfer::SpeculativeBackend::None, 0,
                         true, 0, WeightsProfile::R9700Q4G64Evaluation);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION) && NINFER_R9700_XATTENTION_STRIDE == 16 &&      \
    NINFER_R9700_XATTENTION_TAU_PERMILLE == 900
        const std::size_t capacity_envelope_workspace =
            qualify_plan(device, 1, ninfer::SpeculativeBackend::Mtp, 3, false, 0,
                         WeightsProfile::R9700Q4G64Evaluation, true, kCapacityEnvelopeContext,
                         kCapacityEnvelopeChunk);
        require(capacity_envelope_workspace == kCapacityEnvelopeWorkspaceBytes,
                "S16/tau900 4K/262K global planner workspace envelope changed");
#endif
        (void)qualify_plan(device, ninfer::kDFlashMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 5, false, 6,
                           WeightsProfile::R9700Q4G64DFlash2Q4Evaluation);
        (void)qualify_plan(device, ninfer::kDFlashMaximumConcurrency,
                           ninfer::SpeculativeBackend::DFlash, 5, false, 6,
                           WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation);
        const std::size_t dflash_q4_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4G64DFlash2Q4Evaluation, 128);
        const std::size_t dflash_mixed_linear = Variant::linear_workspace_capacity_bytes(
            WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation, 128);
        const std::size_t base_q4_linear =
            Variant::linear_workspace_capacity_bytes(WeightsProfile::R9700Q4G64Evaluation, 128);
        require(Variant::dflash_matrix_qtype(WeightsProfile::R9700Q4G64DFlash2Q4Evaluation) ==
                        ninfer::QType::Q4G64_F16S &&
                    Variant::dflash_matrix_qtype(WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation) ==
                        ninfer::QType::Q4G64_F16S,
                "DFlash evaluation profile did not select persistent Q4 child matrices");
        require(dflash_q4_linear == dflash_mixed_linear,
                "DFlash Q4 companions disagree on their activation workspace");
        require(dflash_q4_linear > base_q4_linear,
                "DFlash feature projection did not enlarge Q4 activation workspace to K=25600");
        if constexpr (ninfer::ops::r9700::linear::kW8ActivationBits == 8) {
            require(w8_workspace == mixed_workspace,
                    "A8 W8 and mixed plans disagree on their largest activation image");
            require(w8_workspace > q4_workspace,
                    "A8G32 W8 plan did not reserve its larger scale image");
        } else {
            require(mixed_workspace > w8_workspace,
                    "mixed Q4/W8 plan did not reserve caller-owned Q4 activation workspace");
            require(q4_workspace > mixed_workspace,
                    "all-Q4 plan did not reserve its larger MLP/Vision activation workspace");
        }
        std::printf("r9700_runtime_planner: PASS q4_activation_bits=%d q4_prefill_cta_profile=%.*s "
                    "w8_activation_bits=%d C=1..16 "
                    "ordinary/vision/MTP/DFlash fixed widths; 32K MTP3 graph allowance; "
                    "exact 32K DFlash K=1..11 C=1..8 graph inventories; profile-consistent integer "
                    "workspace including DFlash Q4 companions\n",
                    ninfer::ops::r9700::linear::kQ4ActivationBits,
                    static_cast<int>(ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.size()),
                    ninfer::ops::r9700::linear::kQ4PrefillCtaProfile.data(),
                    ninfer::ops::r9700::linear::kW8ActivationBits);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "r9700_runtime_planner: FAIL: %s\n", error.what());
        return 1;
    }
}

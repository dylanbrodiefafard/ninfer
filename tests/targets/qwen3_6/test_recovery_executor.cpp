#include "runtime/engine/concurrent_executor.h"

#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "text/unicode.h"

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::PrefixReuseSource;
using ninfer::TokenId;
using ninfer::runtime::AdmissionResources;
using ninfer::runtime::BatchedGeneratedRound;
using ninfer::runtime::CacheRestoreFailure;
using ninfer::runtime::PrefillPace;
using ninfer::runtime::PrefillStepResult;
using ninfer::runtime::RequestPlanSummary;
using ninfer::runtime::RoundBudget;
using ninfer::runtime::TransientRegion;
using ninfer::targets::qwen3_6::Frontend;
using ninfer::targets::qwen3_6::FrontendResources;
using ninfer::targets::qwen3_6::FrontendTestAccess;
using ninfer::targets::qwen3_6::OutputSession;
using ninfer::targets::qwen3_6::PreparedPrompt;

// Toy tokenizer ids for "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n".
// The suffix is the recovery thinking prologue the splice requires.
constexpr TokenId kResidentPrefix[]     = {248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
constexpr std::uint32_t kResidentTokens = sizeof(kResidentPrefix) / sizeof(kResidentPrefix[0]);
constexpr std::uint32_t kOutputBudget   = 8;
constexpr std::uint64_t kRamEntryId     = 7;
constexpr TokenId kCallerStop           = 0;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

std::string read_template() {
    const std::string path = std::string(NINFER_SOURCE_DIR) +
                             "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja";
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open " + path); }
    std::string source{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

// The recovery notice is literal client-side text between template markup, so the toy
// vocabulary gives every byte an ordinary byte-level symbol and the notice round-trips as those.
std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

constexpr std::string_view kFragmentOpen  = "<|im_start|>system\n";
constexpr std::string_view kFragmentClose = "<|im_end|>\n";

std::string
recovery_notice(std::uint32_t attempt = 1,
                ninfer::targets::qwen3_6::GenerationRecoveryNotice reason =
                    ninfer::targets::qwen3_6::GenerationRecoveryNotice::RepeatedReasoning) {
    ninfer::PromptInput input;
    input.options.enable_thinking       = true;
    input.options.add_generation_prompt = true;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    input.messages.push_back(std::move(user));
    const auto recovery = ninfer::targets::qwen3_6::GenerationRecoveryContext::analyze(input);
    if (!recovery) { throw std::runtime_error("probe input is not eligible for recovery"); }
    const auto insert = recovery->recovery_insert({}, attempt, reason);
    if (insert.size() != 1 || insert[0].parts.size() != 1 ||
        insert[0].parts[0].kind != ninfer::MessagePartKind::Text) {
        throw std::runtime_error("probe recovery insert was not one system notice");
    }
    return insert[0].parts[0].text;
}

FrontendResources resources() {
    FrontendResources result;
    result.chat_template_jinja = read_template();
    nlohmann::json tokens      = nlohmann::json::array({added(1, "helloST"),
                                                        added(2, "OPtail"),
                                                        added(3, "thought</thi"),
                                                        added(4, "nk>\n\nanswer"),
                                                        added(6, "<eos>", true),
                                                        added(7, "<0.0 seconds>"),
                                                        added(14, "   \n"),
                                                        added(15, "answer"),
                                                        added(16, "<tool_"),
                                                        added(17, "call>"),
                                                        added(18, "<function=f>"),
                                                        added(19, "</function>"),
                                                        added(20, "</tool_call>"),
                                                        added(21, "<tool_call>"),
                                                        added(22, "preface"),
                                                        added(23, "call"),
                                                        added(24, "a <"),
                                                        added(30, "user\n"),
                                                        added(31, "assistant\n"),
                                                        added(32, "\n"),
                                                        added(33, "system\n"),
                                                        added(248045, "<|im_start|>", true),
                                                        added(248046, "<|im_end|>", true),
                                                        added(248053, "<|vision_start|>", true),
                                                        added(248054, "<|vision_end|>", true),
                                                        added(248056, "<|image_pad|>", true),
                                                        added(248057, "<|video_pad|>", true),
                                                        added(248068, "<think>"),
                                                        added(248069, "</think>")});
    nlohmann::json vocab       = {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}};
    for (int byte = 0; byte <= 255; ++byte) {
        const std::string symbol = byte_level_symbol(static_cast<std::uint8_t>(byte));
        if (!vocab.contains(symbol)) { vocab[symbol] = 1000 + byte; }
    }
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

constexpr std::uint32_t kPromptTokens = 64;
constexpr std::uint32_t kRamReuse     = 32;
constexpr std::uint32_t kDiskReuse    = 48;
constexpr std::uint64_t kDiskEntryId  = 9;

enum class CacheCase {
    RecoveryRamRestoreFails,
    RecoveryResidentHit,
    RecoveryRamHit,
    RecoveryDiskHit,
    RecoveryDiskClaimMiss,
    RecoveryDiskRestoreFails,
    RecoveryPlanFails,
    AdmitRamHit,
    AdmitDiskLongerThanRam,
    AdmitRamRestoreThenCold,
    AdmitDiskRestoreThenCold,
};

const char* cache_case_name(CacheCase script) {
    switch (script) {
    case CacheCase::RecoveryRamRestoreFails:
        return "recovery RAM restore failure";
    case CacheCase::RecoveryResidentHit:
        return "recovery resident hit";
    case CacheCase::RecoveryRamHit:
        return "recovery RAM hit";
    case CacheCase::RecoveryDiskHit:
        return "recovery disk hit";
    case CacheCase::RecoveryDiskClaimMiss:
        return "recovery disk claim miss";
    case CacheCase::RecoveryDiskRestoreFails:
        return "recovery disk restore failure";
    case CacheCase::RecoveryPlanFails:
        return "recovery plan failure";
    case CacheCase::AdmitRamHit:
        return "admission RAM hit";
    case CacheCase::AdmitDiskLongerThanRam:
        return "admission longer disk hit";
    case CacheCase::AdmitRamRestoreThenCold:
        return "admission restore then cold";
    case CacheCase::AdmitDiskRestoreThenCold:
        return "admission disk restore then cold";
    }
    return "unknown cache case";
}

struct ProbePlan {
    RequestPlanSummary fields;
    ninfer::runtime::ResolvedExecutionOptions execution;
    bool force_cold  = false;
    bool allow_reuse = true;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return fields; }
};

ProbePlan fitted_plan() {
    ProbePlan plan;
    plan.fields.prompt_tokens           = kPromptTokens;
    plan.fields.effective_output_tokens = kOutputBudget;
    plan.fields.effective_limit_reason  = FinishReason::OutputLimit;
    plan.fields.service_work_quanta     = 4;
    plan.fields.transient_alignment     = 1;
    plan.fields.admission               = AdmissionResources{1, 1, 0};
    return plan;
}

struct RamSnapshot {
    std::uint32_t system_prompt_limit     = 0;
    std::size_t system_prompt_entries     = 0;
    std::size_t system_prompt_used_bytes  = 0;
    std::uint64_t system_prompt_restores  = 0;
    std::uint64_t system_prompt_evictions = 0;
    std::size_t capacity_bytes            = 0;
    std::size_t used_bytes                = 0;
    std::size_t entry_count               = 0;
    std::uint64_t captures                = 0;
    std::uint64_t restores                = 0;
    std::uint64_t evictions               = 0;
    std::uint64_t drops                   = 0;
    double save_seconds                   = 0;
    double load_seconds                   = 0;
};

struct DiskSnapshot {
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes     = 0;
    std::size_t entry_count    = 0;
    std::uint64_t captures     = 0;
    std::uint64_t restores     = 0;
    std::uint64_t evictions    = 0;
    std::uint64_t drops        = 0;
    std::array<std::uint64_t, ninfer::kKvDiskDropReasonCount> drop_reasons{};
    double save_seconds          = 0;
    double load_seconds          = 0;
    std::uint64_t sequence       = 0;
    bool defrag_active           = false;
    bool spill_active            = false;
    std::uint64_t storage_bytes  = 0;
    bool restore_active          = false;
    std::size_t pending_jobs     = 0;
    double spill_elapsed_seconds = 0;
};

struct GpuPoolSnapshot {
    std::uint32_t page_group_count = 0;
    std::uint32_t entitled_pages   = 0;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t free_pages       = 0;
};

struct GpuSnapshot {
    GpuPoolSnapshot main;
    GpuPoolSnapshot spec;
};

struct RamCopySeconds {
    double save = 0;
    double load = 0;
};

struct DiskCopySeconds {
    double save = 0;
    double load = 0;
    double h2d  = 0;
};

// Records executor cache decisions. abort_lane succeeds on every call: recovery
// and admission both abort a lane that may already have been cleared.
class ProbeProgram {
public:
    ProbeProgram() { trace.reserve(32); }

    std::atomic<bool> report_copies_ready{false};
    std::atomic<bool> cache_idle_permission{false};
    std::atomic<bool> execution_while_idle{false};
    std::atomic<std::uint32_t> idle_true_count{0};

    virtual ~ProbeProgram() = default;

    [[nodiscard]] virtual bool chunked_execution_enabled() const noexcept { return false; }

    [[nodiscard]] virtual AdmissionResources completion_resources(const ProbePlan& base,
                                                                  std::uint32_t) const {
        return base.summary().lifetime_admission;
    }

    [[nodiscard]] virtual bool shrink_window(std::uint32_t,
                                             const ninfer::runtime::ExecutionWindow&) {
        return false;
    }

    [[nodiscard]] virtual ninfer::runtime::ExecutionWindow plan_window(std::uint32_t,
                                                                       std::uint32_t) const {
        throw std::logic_error("RAM-off probe planned a live window");
    }

    [[nodiscard]] virtual ninfer::runtime::ExecutionWindow
    plan_suspended_window(std::uint64_t, std::uint32_t) const {
        throw std::logic_error("RAM-off probe planned a suspended window");
    }

    [[nodiscard]] virtual bool grant_window(std::uint32_t,
                                            const ninfer::runtime::ExecutionWindow&) {
        throw std::logic_error("RAM-off probe granted a live window");
    }

    [[nodiscard]] virtual bool reserve_suspension_credit(std::uint64_t, std::size_t) {
        throw std::logic_error("RAM-off probe reserved suspension storage");
    }

    virtual void release_suspension_credit(std::uint64_t) {}

    [[nodiscard]] virtual bool start_suspend(std::uint32_t, std::uint64_t) {
        throw std::logic_error("RAM-off probe suspended a request");
    }

    [[nodiscard]] virtual bool suspension_ready(std::uint64_t) const { return true; }

    virtual void finish_suspend(std::uint32_t, std::uint64_t) {
        throw std::logic_error("RAM-off probe finished a suspension");
    }

    [[nodiscard]] virtual bool start_resume(std::uint32_t, std::uint64_t,
                                            const ninfer::runtime::ExecutionWindow&) {
        throw std::logic_error("RAM-off probe resumed a request");
    }

    [[nodiscard]] virtual bool resume_ready(std::uint64_t) const { return true; }

    virtual void finish_resume(std::uint32_t, std::uint64_t) {
        throw std::logic_error("RAM-off probe finished a resume");
    }

    virtual void discard_suspended(std::uint64_t) {}

    CacheCase script             = CacheCase::RecoveryRamRestoreFails;
    std::uint32_t stale_failures = 0;
    bool cache_write             = true;
    bool saw_disposable_base     = false;
    ninfer::runtime::CacheRestoreFailureKind restore_failure_kind =
        ninfer::runtime::CacheRestoreFailureKind::InvalidSource;
    std::vector<std::string> trace;
    std::uint32_t copied_tokens             = 0;
    std::uint32_t abort_count               = 0;
    std::uint32_t aborts_at_prefill         = 0;
    std::uint32_t prefill_count             = 0;
    std::uint32_t prefill_lane              = 99;
    std::uint32_t prefill_tokens            = 0;
    std::uint32_t prefill_reusable          = 99;
    PrefixReuseSource prefill_source        = PrefixReuseSource::HostRam;
    std::uint64_t prefill_ram_entry         = 1;
    std::uint64_t prefill_disk_entry        = 1;
    std::uint32_t offered_ram               = 0;
    std::uint32_t offered_disk              = 0;
    std::uint32_t restore_ram_count         = 0;
    std::uint32_t restore_disk_count        = 0;
    std::uint32_t claim_ram_count           = 0;
    std::uint32_t claim_disk_count          = 0;
    std::uint32_t consume_ram_count         = 0;
    std::uint32_t consume_disk_count        = 0;
    std::uint32_t discard_ram_count         = 0;
    std::uint32_t invalidate_disk_count     = 0;
    std::uint64_t claimed_ram_entry         = 0;
    std::uint64_t claimed_disk_entry        = 0;
    bool prefill_terminal                   = false;
    bool saw_force_cold                     = false;
    bool cancel_on_restore                  = false;
    std::atomic<bool>* request_cancellation = nullptr;
    bool lifecycle_retries                  = false;
    // Lane 1 decodes beside the retry; its one round must not wait for the host restore.
    bool peer_decoding = false;
    // With peer_decoding: the retry is cancelled during its host restore. The hold must stay
    // parked while the peer decodes, and drain only once its copies settle.
    bool cancel_held                   = false;
    std::uint32_t decodes_since_cancel = 0;
    bool cancel_begun                  = false;
    // Peer rounds before its stop token, and peer rounds after a hold cancel until the fenced
    // copies settle. A settle before the peer stops proves the hold finishes beside decoding.
    std::uint32_t peer_round_limit     = 1;
    std::uint32_t settle_after_decodes = 1;
    // With peer_decoding: the retry is cancelled while its failed restore is parked.
    bool cancel_failed_hold = false;
    // With peer_decoding: drive a new admission (AdmitRamRestoreThenCold) instead of a retry.
    bool admit_beside_peer = false;
    // The cold prefill after a failed recovery restore cannot be planned (request-local).
    bool cold_plan_fails = false;
    std::vector<std::vector<TokenId>> prefilled_prompts;
    // While held, simulates the disk tier holding its mutex (for example during index maintenance).
    std::atomic<bool> copies_ready_held{false};
    mutable std::atomic<bool> copies_ready_entered{false};

    void note(const char* event) {
        std::lock_guard lock(trace_mu_);
        trace.emplace_back(event);
    }

    [[nodiscard]] std::uint32_t ram_reuse() const noexcept {
        switch (script) {
        case CacheCase::RecoveryRamRestoreFails:
        case CacheCase::RecoveryRamHit:
        case CacheCase::AdmitRamHit:
        case CacheCase::AdmitDiskLongerThanRam:
        case CacheCase::AdmitRamRestoreThenCold:
            return kRamReuse;
        case CacheCase::RecoveryDiskHit:
            return 16;
        case CacheCase::RecoveryResidentHit:
        case CacheCase::RecoveryDiskClaimMiss:
        case CacheCase::RecoveryDiskRestoreFails:
        case CacheCase::RecoveryPlanFails:
        case CacheCase::AdmitDiskRestoreThenCold:
            return 0;
        }
        return 0;
    }

    [[nodiscard]] std::uint32_t disk_reuse() const noexcept {
        switch (script) {
        case CacheCase::RecoveryDiskHit:
        case CacheCase::RecoveryDiskClaimMiss:
        case CacheCase::RecoveryDiskRestoreFails:
        case CacheCase::AdmitDiskLongerThanRam:
        case CacheCase::AdmitDiskRestoreThenCold:
            return kDiskReuse;
        case CacheCase::RecoveryRamRestoreFails:
        case CacheCase::RecoveryResidentHit:
        case CacheCase::RecoveryRamHit:
        case CacheCase::RecoveryPlanFails:
        case CacheCase::AdmitRamHit:
        case CacheCase::AdmitRamRestoreThenCold:
            return 0;
        }
        return 0;
    }

    [[nodiscard]] virtual AdmissionResources admission_capacity() const noexcept {
        return AdmissionResources{peer_decoding ? 2U : 1U, 2, 0};
    }

    [[nodiscard]] bool retain_reusable_lane(std::uint32_t) {
        note("retain");
        return script == CacheCase::RecoveryResidentHit;
    }

    [[nodiscard]] bool copy_reusable_prompt(std::uint32_t, std::uint32_t prompt_tokens,
                                            std::vector<TokenId>& tokens, std::uint32_t&) {
        copied_tokens = prompt_tokens;
        if (lifecycle_retries) {
            if (prefilled_prompts.empty() || prefilled_prompts.back().size() != prompt_tokens) {
                return false;
            }
            tokens = prefilled_prompts.back();
            return true;
        }
        if (prompt_tokens != kResidentTokens) { return false; }
        note("copy");
        tokens.assign(std::begin(kResidentPrefix), std::end(kResidentPrefix));
        return true;
    }

    [[nodiscard]] virtual ProbePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const ninfer::runtime::ResolvedExecutionOptions& options) {
        note("plan_base");
        saw_disposable_base = saw_disposable_base || !options.cache_write;
        if (script == CacheCase::RecoveryPlanFails) {
            throw ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                                       "probe retry prompt does not fit");
        }
        ProbePlan plan = fitted_plan();
        plan.execution = options;
        if (lifecycle_retries) {
            plan.fields.prompt_tokens           = prompt.summary().prompt_tokens;
            plan.fields.effective_output_tokens = options.requested_output_tokens;
            plan.fields.service_work_quanta     = options.requested_output_tokens + 8;
        }
        plan.force_cold  = options.force_cold_prefill;
        plan.allow_reuse = options.allow_prefix_reuse;
        saw_force_cold   = saw_force_cold || options.force_cold_prefill;
        return plan;
    }

    [[nodiscard]] virtual ProbePlan plan_request_for_lane(std::uint32_t, const PreparedPrompt&,
                                                          const ProbePlan& base) {
        note("plan_cold");
        if (cold_plan_fails) {
            throw ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                                       "probe cold retry does not fit");
        }
        ProbePlan plan   = lifecycle_retries ? base : fitted_plan();
        plan.force_cold  = base.force_cold;
        plan.allow_reuse = base.allow_reuse;
        if (script == CacheCase::RecoveryResidentHit && !base.force_cold && base.allow_reuse) {
            plan.fields.reusable_prompt_tokens =
                lifecycle_retries
                    ? (prefilled_prompts.empty() ? 0 : prefilled_prompts.back().size())
                    : kResidentTokens;
            if (plan.fields.reusable_prompt_tokens != 0) {
                plan.fields.reuse_source = PrefixReuseSource::VramResident;
            }
        }
        return plan;
    }

    [[nodiscard]] virtual ProbePlan plan_ram_reuse(const PreparedPrompt&, const ProbePlan& base) {
        note("plan_ram");
        ProbePlan plan   = fitted_plan();
        plan.force_cold  = base.force_cold;
        plan.allow_reuse = base.allow_reuse;
        if (base.force_cold || !base.allow_reuse) { return plan; }
        const std::uint32_t reuse = ram_reuse();
        if (reuse > offered_ram) { offered_ram = reuse; }
        if (reuse == 0) { return plan; }
        plan.fields.reusable_prompt_tokens = reuse;
        plan.fields.reuse_source           = PrefixReuseSource::HostRam;
        plan.fields.ram_entry_id           = kRamEntryId;
        return plan;
    }

    [[nodiscard]] ProbePlan plan_disk_reuse(const PreparedPrompt&, const ProbePlan& base) {
        note("plan_disk");
        ProbePlan plan   = fitted_plan();
        plan.force_cold  = base.force_cold;
        plan.allow_reuse = base.allow_reuse;
        if (base.force_cold || !base.allow_reuse) { return plan; }
        const std::uint32_t reuse = disk_reuse();
        if (reuse > offered_disk) { offered_disk = reuse; }
        if (reuse == 0) { return plan; }
        plan.fields.reusable_prompt_tokens = reuse;
        plan.fields.reuse_source           = PrefixReuseSource::HostDisk;
        plan.fields.disk_entry_id          = kDiskEntryId;
        return plan;
    }

    [[nodiscard]] virtual bool can_admit_lane(std::uint32_t, const ProbePlan&) const noexcept {
        return true;
    }

    [[nodiscard]] virtual bool
    can_admit_lane_after_retained_eviction(std::uint32_t, const ProbePlan&) const noexcept {
        return false;
    }

    [[nodiscard]] virtual bool
    can_admit_lane_after_releasing(std::uint32_t lane, const ProbePlan& plan,
                                   std::span<const std::uint32_t>) const noexcept {
        return can_admit_lane(lane, plan);
    }

    [[nodiscard]] ninfer::GenerationTimings generation_timings_lane(std::uint32_t) const noexcept {
        return {};
    }

    [[nodiscard]] ninfer::SpeculativeStats speculative_stats_lane(std::uint32_t) const noexcept {
        return {};
    }

    [[nodiscard]] std::uint32_t
    captured_context_checkpoint_tokens_lane(std::uint32_t) const noexcept {
        return 0;
    }

    [[nodiscard]] std::uint32_t
    restored_context_checkpoint_tokens_lane(std::uint32_t) const noexcept {
        return 0;
    }

    virtual void abort_lane(std::uint32_t) noexcept {
        note("abort");
        ++abort_count;
    }

    virtual void claim_ram_entry(std::uint64_t entry_id) {
        note("claim_ram");
        ++claim_ram_count;
        claimed_ram_entry = entry_id;
    }

    virtual void restore_ram_entry(std::uint32_t, std::uint64_t, const ProbePlan&) {
        note("restore_ram");
        ++restore_ram_count;
        if (restore_ram_count <= stale_failures) {
            throw CacheRestoreFailure("probe stale RAM plan",
                                      ninfer::runtime::CacheRestoreFailureKind::StalePlan);
        }
        if (cancel_held && request_cancellation != nullptr) {
            request_cancellation->store(true, std::memory_order_release);
        }
        if (script == CacheCase::RecoveryRamRestoreFails ||
            (script == CacheCase::AdmitRamRestoreThenCold && restore_ram_count == 1)) {
            throw CacheRestoreFailure("probe host restore failed", restore_failure_kind);
        }
        ram_timings_pending_ = true;
    }

    virtual void release_ram_entry(std::uint64_t) { note("release_ram"); }

    virtual void discard_ram_capture(std::uint64_t) {
        note("discard_ram");
        ++discard_ram_count;
    }

    void cancel_disk_restore() { note("cancel_disk"); }

    void begin_copy_hold_cancel() {
        note("begin_cancel");
        cancel_begun = true;
        if (cancel_failed_hold && request_cancellation != nullptr) {
            request_cancellation->store(true, std::memory_order_release);
        }
    }

    [[nodiscard]] bool copy_hold_cancel_settled() const {
        return decodes_since_cancel >= settle_after_decodes;
    }

    virtual void synchronize_all() { note("sync"); }

    [[nodiscard]] virtual PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                               PreparedPrompt prompt,
                                                               ProbePlan plan, TransientRegion,
                                                               PrefillPace, const OutputSession*) {
        note("start_prefill");
        ++prefill_count;
        aborts_at_prefill  = abort_count;
        prefill_lane       = lane;
        prefill_tokens     = prompt.summary().prompt_tokens;
        prefill_reusable   = plan.summary().reusable_prompt_tokens;
        prefill_source     = plan.summary().reuse_source;
        prefill_ram_entry  = plan.summary().ram_entry_id;
        prefill_disk_entry = plan.summary().disk_entry_id;
        if (lifecycle_retries) {
            auto data = ninfer::targets::qwen3_6::PreparedPromptAccess::take(std::move(prompt));
            prefilled_prompts.push_back(std::move(data.token_ids));
            // Two failed reasoning attempts, then a caller stop on the second retry.
            licensed_ = prefill_count < 3 ? 0 : 15;
        }
        PrefillStepResult step;
        step.complete                     = true;
        step.processed_prompt_tokens      = 1;
        step.round.tokens                 = std::span<const TokenId>(&licensed_, 1);
        step.summary.prompt_tokens        = prefill_tokens;
        step.summary.reused_prompt_tokens = prefill_reusable;
        step.summary.prefix_reuse_source  = prefill_source;
        step.summary.prefix_reuse_path    = prefill_reusable == 0
                                                ? ninfer::PrefixReusePath::FullReset
                                                : ninfer::PrefixReusePath::AppendAtFrontier;
        return step;
    }

    void resolve_prefill_lane(std::uint32_t, bool terminal) {
        note("resolve_prefill");
        prefill_terminal = terminal;
    }

    [[nodiscard]] virtual bool kv_copies_ready() const {
        if (copies_ready_held.load()) {
            copies_ready_entered.store(true);
            while (copies_ready_held.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return report_copies_ready.load();
    }

    void set_cache_idle(bool idle) {
        if (idle) { idle_true_count.fetch_add(1); }
        cache_idle_permission.store(idle);
    }

    virtual void shutdown_kv_tiers(const ninfer::LoadProgress& = {}) {}

    [[nodiscard]] RamSnapshot kv_ram_snapshot() const noexcept { return {}; }

    [[nodiscard]] std::optional<DiskSnapshot> try_kv_disk_snapshot() const noexcept {
        return DiskSnapshot{};
    }

    [[nodiscard]] GpuSnapshot kv_gpu_snapshot() const noexcept { return {}; }

    [[nodiscard]] RamCopySeconds harvest_kv_ram_copy_seconds() {
        if (!std::exchange(ram_timings_pending_, false)) { return {}; }
        return {.save = 0.125, .load = 0.25};
    }

    [[nodiscard]] DiskCopySeconds harvest_kv_disk_copy_seconds() {
        if (!std::exchange(disk_timings_pending_, false)) { return {}; }
        return {.save = 0.5, .load = 1.0, .h2d = 2.0};
    }

    [[nodiscard]] bool kv_ram_copies_ready() const { return true; }

    [[nodiscard]] bool kv_disk_restore_failed() const { return false; }

    // A request starts at index version zero. Zero here would skip the host lookup.
    [[nodiscard]] virtual std::uint64_t kv_ram_index_version() const noexcept { return 1; }

    [[nodiscard]] std::uint64_t kv_disk_index_version() const noexcept { return 1; }

    [[nodiscard]] std::uint64_t pending_disk_restore_ticket() const noexcept { return 0; }

    [[nodiscard]] virtual bool has_retained_lane(std::uint32_t) const noexcept { return false; }

    [[nodiscard]] virtual std::uint64_t retained_use_tick(std::uint32_t) const noexcept {
        return 0;
    }

    [[nodiscard]] bool retained_lane_cache_writable(std::uint32_t lane) const noexcept {
        return has_retained_lane(lane);
    }

    [[nodiscard]] bool retained_lane_precedes(std::uint32_t left,
                                              std::uint32_t right) const noexcept {
        return retained_use_tick(left) < retained_use_tick(right) ||
               (retained_use_tick(left) == retained_use_tick(right) && left < right);
    }

    [[nodiscard]] bool preserve_retained_lane(std::uint32_t lane, std::uint64_t& id, bool may_block,
                                              bool& deferred) {
        return capture_retained_lane(lane, &id, may_block, &deferred, {});
    }

    void fork_retained_lane(std::uint32_t, std::uint32_t, const ProbePlan&) {
        throw std::logic_error("recovery probe does not provide resident fork state");
    }

    void retarget_plan_to_ram(ProbePlan& plan, std::uint64_t id) {
        plan.fields.reuse_source = PrefixReuseSource::HostRam;
        plan.fields.ram_entry_id = id;
    }

    virtual void mark_turn_closed(std::uint32_t) noexcept {}

    [[nodiscard]] virtual bool capture_retained_lane(std::uint32_t, std::uint64_t* = nullptr,
                                                     bool = true, bool* = nullptr,
                                                     std::span<const std::uint64_t> = {}) {
        return false;
    }

    [[nodiscard]] bool claim_disk_entry(std::uint64_t entry_id, std::uint32_t, std::uint64_t,
                                        std::uint64_t, std::uint32_t, ninfer::PrefixReusePath,
                                        std::uint64_t) {
        note("claim_disk");
        ++claim_disk_count;
        claimed_disk_entry = entry_id;
        return script != CacheCase::RecoveryDiskClaimMiss;
    }

    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t) { return true; }

    [[nodiscard]] virtual PrefillStepResult advance_prefill_lane(std::uint32_t, PrefillPace) {
        throw std::logic_error("cold prefill did not complete in one step");
    }

    [[nodiscard]] virtual std::uint32_t prefill_remaining_tokens(std::uint32_t) const {
        throw std::logic_error("probe prefill completes in one step");
    }

    [[nodiscard]] virtual bool prefill_suspendable(std::uint32_t) const { return false; }

    virtual void resume_prefill_lane(std::uint32_t) {
        throw std::logic_error("probe prefill cannot suspend");
    }

    [[nodiscard]] bool prefill_mixable(std::uint32_t) const { return false; }

    [[nodiscard]] ninfer::runtime::MixedGeneratedRound
    decode_batch_with_prefill(std::span<const std::uint32_t>, std::span<const RoundBudget>,
                              std::uint32_t) {
        throw std::logic_error("probe prefill is never mixable");
    }

    [[nodiscard]] virtual BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                                             std::span<const RoundBudget>) {
        if (peer_decoding) {
            note("decode");
            if (cancel_begun) { ++decodes_since_cancel; }
            const bool more = ++peer_rounds_ < peer_round_limit;
            return {.tokens     = std::span<const TokenId>(more ? &kPeerContinue : &kPeerStop, 1),
                    .row_counts = std::span<const std::int32_t>(&kPeerCount, 1),
                    .row_stride = 1};
        }
        if (lifecycle_retries && prefill_count < 3) {
            return {.tokens     = repeated_tokens_,
                    .row_counts = std::span<const std::int32_t>(&repeated_count_, 1),
                    .row_stride = 64};
        }
        throw std::logic_error("recovery entered decode after the cold prefill");
    }

    void release_disk_entry(std::uint64_t) { note("release_disk"); }

    void invalidate_disk_entry(std::uint64_t) {
        note("invalidate_disk");
        ++invalidate_disk_count;
        if (script == CacheCase::AdmitDiskRestoreThenCold) {
            // The disk tier refuses while another session still pins the entry.
            throw std::logic_error("disk invalidation requires drained and released entry");
        }
    }

    virtual void finish_ram_restore(std::uint64_t, ninfer::runtime::CacheReadIntent intent) {
        if (intent == ninfer::runtime::CacheReadIntent::Fork) {
            note("release_ram_restored");
        } else {
            note("consume_ram");
            ++consume_ram_count;
        }
    }

    void consume_disk_entry(std::uint64_t) {
        note("consume_disk");
        ++consume_disk_count;
    }

    void prefetch_disk_plan(std::uint64_t, const ProbePlan&) { note("prefetch_disk"); }

    void pump_disk_restore() {}

    [[nodiscard]] bool ram_restore_ready(std::uint64_t) const { return true; }

    [[nodiscard]] bool disk_restore_ready(std::uint64_t) const { return true; }

    [[nodiscard]] virtual bool kv_ram_reclaim_pending() const { return false; }

    void restore_disk_entry(std::uint32_t, std::uint64_t, const ProbePlan&) {
        note("restore_disk");
        ++restore_disk_count;
        if (restore_disk_count <= stale_failures) {
            throw CacheRestoreFailure("probe stale disk plan",
                                      ninfer::runtime::CacheRestoreFailureKind::StalePlan);
        }
        if (script == CacheCase::RecoveryDiskRestoreFails ||
            (script == CacheCase::AdmitDiskRestoreThenCold && restore_disk_count == 1)) {
            throw CacheRestoreFailure("probe disk restore failed", restore_failure_kind);
        }
        disk_timings_pending_ = true;
    }

    void wait_kv_ram_copies() {}

    void wait_kv_disk_copies() {}

    void wait_kv_ram_copies_on_compute() {
        if (cancel_on_restore && request_cancellation != nullptr) {
            request_cancellation->store(true, std::memory_order_release);
        }
    }

    virtual void evict_retained_lane(std::uint32_t) noexcept {}

    virtual void set_suppressed_tokens_lane(std::uint32_t, std::span<const TokenId>) {}

    virtual void clear_suppressed_tokens_lane(std::uint32_t) {}

    void set_typical_cycle_reasoning_lane(std::uint32_t, bool) {}

    virtual void resolve_pending_batch(std::span<const std::uint32_t>,
                                       std::span<const std::uint32_t>,
                                       std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                                       std::span<const std::uint8_t>) {}

private:
    static constexpr TokenId kPeerStop       = kCallerStop;
    static constexpr TokenId kPeerContinue   = 1000 + 'a';
    std::uint32_t peer_rounds_               = 0;
    static constexpr std::int32_t kPeerCount = 1;
    bool ram_timings_pending_                = false;
    bool disk_timings_pending_               = false;
    std::array<TokenId, 64> repeated_tokens_{};
    std::int32_t repeated_count_ = 64;
    TokenId licensed_            = kCallerStop;
    std::mutex trace_mu_;
};

struct ProbePackage {
    using Program         = ProbeProgram;
    using RequestBasePlan = ProbePlan;
    using RequestPlan     = ProbePlan;
};

struct ProbeLoaded {
    Frontend frontend;
};

struct ProbeMemory {
    void activate(std::size_t bytes, std::size_t alignment) {
        if (bytes != 0 || alignment != 1) {
            throw std::invalid_argument("recovery probe transient must stay empty");
        }
    }

    void deactivate() noexcept {}

    [[nodiscard]] TransientRegion region() const noexcept { return {}; }
};

struct RecoveryProbe {
    using Package         = ProbePackage;
    ProbeProgram* program = nullptr;
    ProbeLoaded* loaded   = nullptr;
    ProbeMemory request_memory;
};

const char* kExpectedTrace[] = {
    "retain",      "copy",      "plan_base",     "plan_ram",
    "plan_disk",   "abort",     "claim_ram",     "restore_ram",
    "cancel_disk", "sync",      "release_ram",   "discard_ram",
    "abort",       "plan_cold", "start_prefill", "resolve_prefill",
};

struct RecoveryOutcome {
    ninfer::GenerationResult result;
    double runtime_disk_h2d_seconds = 0;
    bool finished                   = false;
    std::string error;
    std::uint32_t attempts         = 0;
    std::uint32_t budget_remaining = 0;
    bool has_budget                = false;
    bool cache_fallback            = false;
    bool force_cold                = false;
    FinishReason finish            = FinishReason::None;
    FinishReason peer_finish       = FinishReason::None;
    bool pending_empty             = false;
    bool recovery_empty            = false;
    bool slot_empty                = false;
    bool executor_failed           = false;
};

int event_count(const std::vector<std::string>& trace, const char* name) {
    return static_cast<int>(std::count(trace.begin(), trace.end(), name));
}

bool in_order(const std::vector<std::string>& trace, std::initializer_list<const char*> steps) {
    auto cursor = trace.begin();
    for (const char* step : steps) {
        cursor = std::find(cursor, trace.end(), step);
        if (cursor == trace.end()) { return false; }
        ++cursor;
    }
    return true;
}

void dump_trace(const ProbeProgram& program) {
    std::cerr << cache_case_name(program.script) << " trace:";
    for (const std::string& event : program.trace) { std::cerr << ' ' << event; }
    std::cerr << '\n';
}

int fail_case(const ProbeProgram& program, bool ok, const char* message) {
    if (ok) { return 0; }
    std::cerr << cache_case_name(program.script) << ": " << message << '\n';
    return 1;
}

int check_scripted_recovery(const ProbeProgram& program, const RecoveryOutcome& outcome) {
    int failures = 0;
    if (!program.cache_write) {
        failures +=
            fail_case(program,
                      program.saw_disposable_base && program.consume_ram_count == 0 &&
                          program.discard_ram_count == 0 && program.invalidate_disk_count == 0,
                      "disposable recovery changed policy or consumed its healthy source");
    }
    if (program.cancel_held) {
        failures += fail_case(
            program,
            outcome.finished && outcome.error.empty() &&
                outcome.finish == FinishReason::Cancelled &&
                outcome.peer_finish == FinishReason::StopToken && program.prefill_count == 0 &&
                event_count(program.trace, "release_ram") == 1 &&
                in_order(program.trace, {"restore_ram", "begin_cancel", "decode", "cancel_disk"}) &&
                outcome.slot_empty && !outcome.executor_failed,
            "a cancelled host restore drained while the peer was decoding");
        if (failures != 0) { dump_trace(program); }
        return failures;
    }
    if (program.cancel_failed_hold) {
        // Cancelled while its failed restore is parked: the hold drains once its copies settle,
        // still drops the failed entry, and never prefills.
        failures += fail_case(
            program,
            outcome.finished && outcome.error.empty() &&
                outcome.finish == FinishReason::Cancelled &&
                outcome.peer_finish == FinishReason::StopToken && program.prefill_count == 0 &&
                program.discard_ram_count == 1 && event_count(program.trace, "release_ram") == 1 &&
                in_order(program.trace, {"restore_ram", "begin_cancel", "decode", "decode",
                                         "cancel_disk", "discard_ram", "decode"}) &&
                outcome.slot_empty && !outcome.executor_failed,
            "a cancelled failed hold did not drain and drop its entry");
        if (failures != 0) { dump_trace(program); }
        return failures;
    }
    if (program.script == CacheCase::RecoveryPlanFails || program.cold_plan_fails) {
        // A request-local error fails only the retry; the Engine and its peer continue.
        failures += fail_case(
            program,
            outcome.finished && outcome.error.find("probe") != std::string::npos &&
                outcome.pending_empty && outcome.recovery_empty && outcome.slot_empty &&
                !outcome.executor_failed && program.prefill_count == 0 &&
                (!program.peer_decoding || outcome.peer_finish == FinishReason::StopToken),
            "a request-local retry error escalated beyond its request");
        if (program.cold_plan_fails) {
            failures +=
                fail_case(program,
                          in_order(program.trace, {"restore_ram", "discard_ram", "plan_cold"}) &&
                              event_count(program.trace, "release_ram") == 1,
                          "the failed restore was not dropped before the cold retry");
        }
        if (failures != 0) { dump_trace(program); }
        return failures;
    }
    failures +=
        fail_case(program, outcome.finished && outcome.error.empty(),
                  outcome.error.empty() ? "the retry did not finish" : outcome.error.c_str());
    const bool cancelled = program.cancel_on_restore;
    failures += fail_case(program,
                          outcome.attempts == 1 && outcome.has_budget &&
                              outcome.budget_remaining == kOutputBudget - (cancelled ? 0 : 1) &&
                              outcome.finish ==
                                  (cancelled ? FinishReason::Cancelled : FinishReason::StopToken),
                          "the retry did not keep its output budget and finish");
    if (program.stale_failures != 0) {
        const bool ram      = program.script == CacheCase::RecoveryRamHit;
        const bool restored = program.stale_failures == 1;
        failures += fail_case(
            program,
            (ram ? program.restore_ram_count : program.restore_disk_count) == 2 &&
                program.discard_ram_count == 0 && program.invalidate_disk_count == 0 &&
                program.consume_ram_count == (ram && restored && program.cache_write ? 1U : 0U) &&
                program.prefill_count == 1 &&
                program.prefill_source ==
                    (restored ? (ram ? PrefixReuseSource::HostRam : PrefixReuseSource::HostDisk)
                              : PrefixReuseSource::None) &&
                program.prefill_reusable == (restored ? (ram ? kRamReuse : kDiskReuse) : 0) &&
                !outcome.cache_fallback && !outcome.force_cold,
            "stale recovery did not replan once and preserve its source and session");
        if (failures != 0) { dump_trace(program); }
        return failures;
    }
    const bool ram_hit  = program.script == CacheCase::RecoveryRamHit;
    const bool disk_hit = program.script == CacheCase::RecoveryDiskHit;
    failures += fail_case(program,
                          outcome.result.kv_ram_save_seconds == (ram_hit ? 0.125 : 0.0) &&
                              outcome.result.kv_ram_load_seconds == (ram_hit ? 0.25 : 0.0) &&
                              outcome.result.kv_disk_save_seconds == (disk_hit ? 0.5 : 0.0) &&
                              outcome.result.kv_disk_load_seconds == (disk_hit ? 1.0 : 0.0) &&
                              outcome.result.kv_disk_h2d_seconds == (disk_hit ? 2.0 : 0.0) &&
                              outcome.runtime_disk_h2d_seconds == (disk_hit ? 2.0 : 0.0),
                          "first-step completion lost restored-cache timings");
    failures += fail_case(program,
                          outcome.pending_empty && outcome.recovery_empty && outcome.slot_empty &&
                              !outcome.executor_failed,
                          "the retry left a queue, lane, or failed executor behind");
    failures += fail_case(program, program.copied_tokens == kResidentTokens,
                          "the retry copied a length other than the resident ledger");
    const bool cold_compute = program.prefill_count == 1 && program.prefill_reusable == 0 &&
                              program.prefill_source == PrefixReuseSource::None &&
                              program.prefill_ram_entry == 0 && program.prefill_disk_entry == 0 &&
                              program.prefill_terminal;
    switch (program.script) {
    case CacheCase::RecoveryRamRestoreFails: {
        // A decoding peer parks the failed hold: its copies are fenced, the peer decodes,
        // and only then is the entry dropped.
        std::vector<std::string> expected(std::begin(kExpectedTrace), std::end(kExpectedTrace));
        if (program.peer_decoding) {
            const auto restore = std::find(expected.begin(), expected.end(), "restore_ram");
            expected.insert(restore + 1, {"begin_cancel", "decode"});
        }
        const bool trace_ok = program.trace == expected;
        failures += fail_case(program, trace_ok,
                              "host restore failure did not abort, release, and cold-prefill");
        failures += fail_case(program,
                              program.claimed_ram_entry == kRamEntryId &&
                                  program.aborts_at_prefill == 2 && program.abort_count == 2 &&
                                  cold_compute && program.prefill_tokens > kResidentTokens,
                              "cold prefill did not follow the two idempotent lane aborts");
        failures += fail_case(program, !outcome.cache_fallback && !outcome.force_cold,
                              "the failed recovery restore stuck a cold fallback");
        break;
    }
    case CacheCase::RecoveryResidentHit:
        failures += fail_case(
            program,
            program.abort_count == 0 && event_count(program.trace, "plan_ram") == 0 &&
                event_count(program.trace, "plan_disk") == 0 && program.claim_ram_count == 0 &&
                program.prefill_count == 1 &&
                program.prefill_source == PrefixReuseSource::VramResident &&
                program.prefill_reusable == kResidentTokens &&
                program.prefill_tokens > kResidentTokens && program.prefill_ram_entry == 0 &&
                !outcome.cache_fallback && !outcome.force_cold,
            "a resident prefix was dropped instead of being prefilled in place");
        break;
    case CacheCase::RecoveryRamHit:
        failures += fail_case(
            program,
            program.abort_count == 1 &&
                program.consume_ram_count == (program.cache_write ? 1U : 0U) &&
                program.discard_ram_count == 0 && program.prefill_count == 1 &&
                program.prefill_source == PrefixReuseSource::HostRam &&
                program.prefill_reusable == kRamReuse && program.prefill_ram_entry == kRamEntryId &&
                event_count(program.trace, "plan_cold") == 0 &&
                in_order(program.trace,
                         {"abort", "restore_ram", "start_prefill",
                          program.cache_write ? "consume_ram" : "release_ram_restored"}) &&
                !outcome.cache_fallback && !outcome.force_cold && !program.saw_force_cold,
            "a RAM checkpoint was not restored onto the retry");
        break;
    case CacheCase::RecoveryDiskHit:
        failures += fail_case(program,
                              program.abort_count == 1 && program.consume_disk_count == 1 &&
                                  program.invalidate_disk_count == 0 &&
                                  program.restore_ram_count == 0 && program.prefill_count == 1 &&
                                  program.prefill_source == PrefixReuseSource::HostDisk &&
                                  program.prefill_reusable == kDiskReuse &&
                                  program.prefill_disk_entry == kDiskEntryId &&
                                  program.offered_disk > program.offered_ram &&
                                  program.prefill_reusable == program.offered_disk &&
                                  event_count(program.trace, "plan_cold") == 0 &&
                                  !outcome.cache_fallback && !outcome.force_cold,
                              "the longer disk checkpoint did not win the retry");
        break;
    case CacheCase::RecoveryDiskClaimMiss:
        failures +=
            fail_case(program,
                      program.abort_count == 2 && program.restore_disk_count == 0 &&
                          program.invalidate_disk_count == 0 &&
                          event_count(program.trace, "cancel_disk") == 0 && cold_compute &&
                          program.claimed_disk_entry == kDiskEntryId &&
                          in_order(program.trace, {"claim_disk", "plan_cold", "start_prefill"}) &&
                          !outcome.cache_fallback && !outcome.force_cold,
                      "a disk claim miss did not cold-prefill without invalidating an entry");
        break;
    case CacheCase::RecoveryDiskRestoreFails:
        failures +=
            fail_case(program,
                      program.abort_count == 2 && program.invalidate_disk_count == 1 &&
                          program.consume_disk_count == 0 && cold_compute &&
                          in_order(program.trace, {"restore_disk", "invalidate_disk", "plan_cold",
                                                   "start_prefill"}) &&
                          !outcome.cache_fallback && !outcome.force_cold,
                      "a failed disk restore did not invalidate that entry and cold-prefill");
        break;
    case CacheCase::RecoveryPlanFails:
        break;
    case CacheCase::AdmitRamHit:
    case CacheCase::AdmitDiskLongerThanRam:
    case CacheCase::AdmitRamRestoreThenCold:
    case CacheCase::AdmitDiskRestoreThenCold:
        failures += fail_case(program, false, "an admission script was driven as a retry");
        break;
    }
    if (program.peer_decoding) {
        const char* restore = disk_hit ? "restore_disk" : "restore_ram";
        failures += fail_case(program,
                              outcome.peer_finish == FinishReason::StopToken &&
                                  event_count(program.trace, "decode") == 1 &&
                                  in_order(program.trace, {restore, "decode", "start_prefill"}),
                              "the retry's host restore stalled the decoding lane");
    }
    if (failures != 0) { dump_trace(program); }
    return failures;
}

ninfer::PromptInput thinking_input() {
    ninfer::PromptInput input;
    input.options.enable_thinking       = true;
    input.options.add_generation_prompt = true;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::EngineOptions engine_options() {
    ninfer::EngineOptions options;
    options.max_concurrency      = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms   = 30000;
    options.max_context          = 2048;
    options.generation_recovery  = true;
    return options;
}

} // namespace

namespace ninfer::runtime {

template <class ProbeInstance>
int drive_scripted_recovery(ConcurrentExecutor<ProbeInstance>& executor) {
    using Request      = typename ConcurrentExecutor<ProbeInstance>::Request;
    Frontend& frontend = executor.instance_.loaded->frontend;
    auto input         = thinking_input();

    ninfer::StopPolicy stop;
    stop.token_ids        = {kCallerStop};
    const auto now        = ConcurrentExecutor<ProbeInstance>::Clock::now();
    ProbeProgram& program = *executor.instance_.program;

    // Lane 1 already decodes; its rounds are scripted by the probe's decode_batch.
    const auto make_peer = [&] {
        auto peer_prepared                       = frontend.prepare(input);
        const ninfer::PromptSummary peer_summary = peer_prepared.summary();
        auto peer_session = frontend.make_output_session(peer_prepared, stop, {});
        ResolvedRequestOptions peer_options;
        peer_options.execution.sampling.p_less         = true;
        peer_options.execution.requested_output_tokens = kOutputBudget;
        peer_options.stop                              = stop;
        auto made                                      = std::shared_ptr<Request>(
            new Request(2, std::move(peer_prepared), std::move(peer_session), peer_summary, 0.0,
                        std::move(peer_options), ninfer::OutputDelivery::TerminalOnly,
                        now + std::chrono::hours(1), now, ninfer::HostInputLease{}, false));
        made->budget.emplace(kOutputBudget, FinishReason::OutputLimit);
        made->lane                   = 1;
        made->decode_ready           = true;
        made->admission_resources    = AdmissionResources{1, 1, 0};
        made->remaining_service_work = kOutputBudget;
        return made;
    };
    const auto wait_peer = [](const std::shared_ptr<Request>& peer) -> std::optional<FinishReason> {
        std::unique_lock lock(peer->mutex);
        if (!peer->cv.wait_for(lock, std::chrono::seconds(5), [&] { return peer->done; })) {
            return std::nullopt;
        }
        return peer->result.finish_reason;
    };

    if (program.admit_beside_peer) {
        // A new admission's restore fails while the peer decodes: the hold stays parked
        // until its copies settle, then the request re-enters admission cold.
        auto peer          = make_peer();
        auto prepared      = frontend.prepare(input);
        const auto summary = prepared.summary();
        ResolvedRequestOptions options;
        options.execution.sampling.p_less    = true;
        options.execution.allow_prefix_reuse = true;
        options.execution.cache_write        = program.cache_write;
        options.stop                         = stop;
        typename ConcurrentExecutor<ProbeInstance>::Submission submission;
        {
            // Plant the peer and queue the request together, so the peer is still decoding.
            std::scoped_lock lock(executor.execution_mutex_);
            executor.slots_[1] = peer;
            submission = executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                                         ninfer::OutputDelivery::TerminalOnly);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancel(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        int failures = 0;
        try {
            const ninfer::GenerationResult result = submission.wait(nullptr, cancel);
            const auto peer_finish                = wait_peer(peer);
            bool executor_failed                  = false;
            {
                std::scoped_lock lock(executor.queue_mutex_);
                executor_failed = executor.failed_;
            }
            const auto& trace    = program.trace;
            const auto cancel_at = std::find(trace.begin(), trace.end(), "begin_cancel");
            const auto drop_at   = std::find(cancel_at, trace.end(), "cancel_disk");
            const auto settled_decodes =
                static_cast<std::uint32_t>(std::count(cancel_at, drop_at, "decode"));
            failures += fail_case(
                program,
                result.finish_reason == FinishReason::StopToken &&
                    result.reused_prompt_tokens == 0 && program.prefill_count == 1 &&
                    program.prefill_reusable == 0 && program.saw_force_cold &&
                    program.discard_ram_count == 1 && program.consume_ram_count == 0 &&
                    peer_finish == FinishReason::StopToken && !executor_failed,
                "a failed admission restore beside a decoding peer did not fall back cold");
            failures +=
                fail_case(program,
                          settled_decodes == program.settle_after_decodes &&
                              in_order(trace, {"restore_ram", "begin_cancel", "decode",
                                               "cancel_disk", "discard_ram", "decode"}) &&
                              in_order(trace, {"discard_ram", "start_prefill"}),
                          "the failed hold did not wait for its copies beside the decoding peer");
        } catch (const std::exception& error) {
            failures += fail_case(program, false, error.what());
        }
        if (failures != 0) { dump_trace(program); }
        return failures;
    }

    auto prepared                       = frontend.prepare(input);
    const ninfer::PromptSummary summary = prepared.summary();
    auto session                        = frontend.make_output_session(prepared, stop, {});

    ResolvedRequestOptions options;
    options.execution.sampling.p_less         = true;
    options.execution.allow_prefix_reuse      = true;
    options.execution.force_cold_prefill      = false;
    options.execution.requested_output_tokens = kOutputBudget;
    options.execution.cache_write             = program.cache_write;
    options.stop                              = stop;
    auto request                              = std::shared_ptr<Request>(
        new Request(1, std::move(prepared), std::move(session), summary, 0.0, std::move(options),
                    ninfer::OutputDelivery::TerminalOnly, now + std::chrono::hours(1), now,
                    ninfer::HostInputLease{}, true));
    int failures = check(request->recovery_context != nullptr, "the retry had no recovery context");
    request->budget.emplace(kOutputBudget, FinishReason::OutputLimit);
    request->lane                                    = 0;
    request->recovery_pending                        = true;
    request->resident_prompt_tokens                  = kResidentTokens;
    request->admission_resources                     = AdmissionResources{1, 1, 0};
    request->remaining_service_work                  = 4;
    request->recovery_cause                          = "repeated_reasoning";
    executor.instance_.program->request_cancellation = &request->cancelled;

    std::shared_ptr<Request> peer;
    if (program.peer_decoding) { peer = make_peer(); }

    {
        std::scoped_lock locks(executor.execution_mutex_, executor.queue_mutex_);
        if (peer) { executor.slots_[1] = peer; }
        executor.slots_[0] = request;
        executor.recovery_queue_.push_back(request);
        executor.control_dirty_.store(true, std::memory_order_release);
        executor.queue_cv_.notify_all();
    }

    {
        std::unique_lock lock(request->mutex);
        if (!request->cv.wait_for(lock, std::chrono::seconds(5), [&] { return request->done; })) {
            failures += check(false, "the host-restore failure did not finish");
            return failures;
        }
    }

    RecoveryOutcome outcome;
    if (peer) {
        const auto peer_finish = wait_peer(peer);
        if (!peer_finish) {
            failures += check(false, "the decoding lane did not finish");
            return failures;
        }
        outcome.peer_finish = *peer_finish;
    }
    outcome.finished = true;
    if (request->error != nullptr) {
        try {
            std::rethrow_exception(request->error);
        } catch (const std::exception& error) { outcome.error = error.what(); } catch (...) {
            outcome.error = "the retry completed with an unknown error";
        }
    }
    outcome.attempts         = request->recovery.attempts;
    outcome.has_budget       = request->budget.has_value();
    outcome.budget_remaining = request->budget ? request->budget->remaining() : 0;
    outcome.cache_fallback   = request->cache_fallback;
    outcome.force_cold       = request->options.execution.force_cold_prefill;
    outcome.finish           = request->result.finish_reason;
    outcome.result           = request->result;
    {
        std::scoped_lock locks(executor.execution_mutex_, executor.queue_mutex_);
        outcome.pending_empty            = executor.pending_.empty();
        outcome.recovery_empty           = executor.recovery_queue_.empty();
        outcome.slot_empty               = executor.slots_[0] == nullptr;
        outcome.executor_failed          = executor.failed_;
        outcome.runtime_disk_h2d_seconds = executor.runtime_stats().kv_disk_h2d_seconds;
        executor.instance_.program->request_cancellation = nullptr;
    }
    return failures + check_scripted_recovery(*executor.instance_.program, outcome);
}

} // namespace ninfer::runtime

struct ProbeSetup {
    bool cache_write                   = true;
    std::uint32_t stale_failures       = 0;
    bool cancel_on_restore             = false;
    bool peer_decoding                 = false;
    bool cancel_held                   = false;
    bool cold_plan_fails               = false;
    bool cancel_failed_hold            = false;
    bool admit_beside_peer             = false;
    std::uint32_t peer_round_limit     = 1;
    std::uint32_t settle_after_decodes = 1;
};

int run_recovery(Frontend& frontend, CacheCase script, ProbeSetup setup = {}) {
    ProbeProgram program;
    program.script               = script;
    program.cache_write          = setup.cache_write;
    program.stale_failures       = setup.stale_failures;
    program.cancel_on_restore    = setup.cancel_on_restore;
    program.peer_decoding        = setup.peer_decoding;
    program.cancel_held          = setup.cancel_held;
    program.cold_plan_fails      = setup.cold_plan_fails;
    program.cancel_failed_hold   = setup.cancel_failed_hold;
    program.admit_beside_peer    = setup.admit_beside_peer;
    program.peer_round_limit     = setup.peer_round_limit;
    program.settle_after_decodes = setup.settle_after_decodes;
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program = &program;
    instance.loaded  = &loaded;
    auto engine      = engine_options();
    if (setup.peer_decoding) { engine.max_concurrency = 2; }
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
    return ninfer::runtime::drive_scripted_recovery(executor);
}

int run_admission(Frontend& frontend, CacheCase script, bool cache_write = true,
                  ninfer::runtime::CacheRestoreFailureKind failure_kind =
                      ninfer::runtime::CacheRestoreFailureKind::InvalidSource) {
    ProbeProgram program;
    program.script               = script;
    program.restore_failure_kind = failure_kind;
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program = &program;
    instance.loaded  = &loaded;
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine_options());

    auto prepared      = frontend.prepare(thinking_input());
    const auto summary = prepared.summary();
    ninfer::runtime::ResolvedRequestOptions options;
    options.execution.sampling.p_less    = true;
    options.execution.allow_prefix_reuse = true;
    options.execution.cache_write        = cache_write;
    options.stop.token_ids               = {kCallerStop};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancel(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    try {
        auto submission = executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                                          ninfer::OutputDelivery::TerminalOnly);
        const ninfer::GenerationResult result = submission.wait(nullptr, cancel);
        int failures                          = 0;
        const bool cold = program.prefill_count == 1 && program.prefill_reusable == 0 &&
                          program.prefill_source == PrefixReuseSource::None &&
                          result.reused_prompt_tokens == 0 &&
                          result.prefix_reuse_source == PrefixReuseSource::None;
        if (failure_kind != ninfer::runtime::CacheRestoreFailureKind::InvalidSource) {
            const bool stale = failure_kind == ninfer::runtime::CacheRestoreFailureKind::StalePlan;
            failures += fail_case(
                program,
                result.finish_reason == FinishReason::StopToken && program.discard_ram_count == 0 &&
                    program.invalidate_disk_count == 0 &&
                    event_count(program.trace, "release_ram") +
                            event_count(program.trace, "release_disk") ==
                        1,
                "healthy source was discarded after a stale plan or destination failure");
            failures +=
                fail_case(program,
                          stale ? (!cold && !program.saw_force_cold &&
                                   program.restore_ram_count + program.restore_disk_count == 2)
                                : (cold && program.saw_force_cold),
                          "restore failure did not replan stale state or cold-prefill unavailable "
                          "destination");
            failures += fail_case(program, cache_write || program.consume_ram_count == 0,
                                  "disposable failure recovery consumed a healthy RAM source");
            if (failures != 0) { dump_trace(program); }
            return failures;
        }
        switch (script) {
        case CacheCase::AdmitRamHit:
            failures += fail_case(program,
                                  result.finish_reason == FinishReason::StopToken &&
                                      result.reused_prompt_tokens == kRamReuse &&
                                      result.prefix_reuse_source == PrefixReuseSource::HostRam &&
                                      program.prefill_ram_entry == kRamEntryId &&
                                      program.consume_ram_count == (cache_write ? 1U : 0U) &&
                                      program.abort_count == 0 && program.discard_ram_count == 0 &&
                                      program.restore_ram_count == 1 && !program.saw_force_cold,
                                  "admission did not restore the RAM checkpoint into the result");
            break;
        case CacheCase::AdmitDiskLongerThanRam:
            failures +=
                fail_case(program,
                          result.finish_reason == FinishReason::StopToken &&
                              result.reused_prompt_tokens == kDiskReuse &&
                              result.prefix_reuse_source == PrefixReuseSource::HostDisk &&
                              program.offered_disk > program.offered_ram &&
                              program.prefill_reusable == program.offered_disk &&
                              program.prefill_disk_entry == kDiskEntryId &&
                              program.consume_disk_count == 1 && program.restore_ram_count == 0 &&
                              program.consume_ram_count == 0 && program.abort_count == 0,
                          "admission did not prefer the longer disk checkpoint");
            break;
        case CacheCase::AdmitRamRestoreThenCold:
            failures +=
                fail_case(program,
                          result.finish_reason == FinishReason::StopToken && cold &&
                              program.saw_force_cold && program.discard_ram_count == 1 &&
                              program.consume_ram_count == 0 && program.claim_ram_count == 1 &&
                              program.restore_ram_count == 1 && program.offered_ram == kRamReuse &&
                              in_order(program.trace, {"claim_ram", "restore_ram", "discard_ram",
                                                       "start_prefill"}),
                          "a failed admission restore was reused instead of computed cold");
            break;
        case CacheCase::AdmitDiskRestoreThenCold:
            // The failed entry cannot be invalidated yet; cleanup is best-effort and the
            // request still falls back to a cold prefill.
            failures +=
                fail_case(program,
                          result.finish_reason == FinishReason::StopToken && cold &&
                              program.saw_force_cold && program.invalidate_disk_count == 1 &&
                              program.consume_disk_count == 0 && program.restore_disk_count == 1 &&
                              in_order(program.trace, {"claim_disk", "restore_disk", "release_disk",
                                                       "invalidate_disk", "start_prefill"}),
                          "a refused invalidation failed the cold fallback");
            break;
        case CacheCase::RecoveryRamRestoreFails:
        case CacheCase::RecoveryResidentHit:
        case CacheCase::RecoveryRamHit:
        case CacheCase::RecoveryDiskHit:
        case CacheCase::RecoveryDiskClaimMiss:
        case CacheCase::RecoveryDiskRestoreFails:
        case CacheCase::RecoveryPlanFails:
            failures += fail_case(program, false, "a retry script was submitted as a new request");
            break;
        }
        if (failures != 0) { dump_trace(program); }
        return failures;
    } catch (const std::exception& error) {
        std::cerr << cache_case_name(script) << ": " << error.what() << '\n';
        dump_trace(program);
        return 1;
    }
}

// The idle worker polls the cache tiers; a slow tier call must not hold the
// queue lock that submit() needs.
int run_idle_poll_does_not_block_submit(Frontend& frontend) {
    ProbeProgram program;
    program.script = CacheCase::AdmitRamHit;
    program.copies_ready_held.store(true);
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program = &program;
    instance.loaded  = &loaded;
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine_options());
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!program.copies_ready_entered.load() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!program.copies_ready_entered.load()) {
        program.copies_ready_held.store(false);
        return check(false, "idle worker never polled the cache tiers");
    }
    auto prepared      = frontend.prepare(thinking_input());
    const auto summary = prepared.summary();
    ninfer::runtime::ResolvedRequestOptions options;
    options.execution.sampling.p_less    = true;
    options.execution.allow_prefix_reuse = true;
    options.stop.token_ids               = {kCallerStop};
    // A submit that waits on the held poll is released by the watchdog and fails.
    std::atomic<bool> submitted{false};
    std::atomic<bool> fired{false};
    std::jthread watchdog([&] {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!submitted.load() && std::chrono::steady_clock::now() < limit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!submitted.load()) {
            fired.store(true);
            program.copies_ready_held.store(false);
        }
    });
    auto submission = executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                                      ninfer::OutputDelivery::TerminalOnly);
    submitted.store(true);
    watchdog.join();
    const bool prompt = !fired.load();
    program.copies_ready_held.store(false);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancel(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    const auto result = submission.wait(nullptr, cancel);
    int failures      = 0;
    failures += check(prompt, "submit waited on an idle-worker cache-tier poll");
    failures += check(result.finish_reason == FinishReason::StopToken,
                      "request submitted during an idle cache poll did not finish");
    return failures;
}

int run_retry_lifecycle(Frontend& frontend) {
    ProbeProgram program;
    program.script            = CacheCase::RecoveryResidentHit;
    program.lifecycle_retries = true;
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program   = &program;
    instance.loaded    = &loaded;
    auto engine        = engine_options();
    engine.max_context = 32768;
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);

    auto prepared               = frontend.prepare(thinking_input());
    const auto original_summary = prepared.summary();
    ninfer::runtime::ResolvedRequestOptions options;
    options.execution.sampling.p_less         = true;
    options.execution.sampling.temperature    = 1.0F;
    options.execution.allow_prefix_reuse      = true;
    options.execution.requested_output_tokens = 20000;
    options.stop.token_ids                    = {15};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancel(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    auto submission   = executor.submit(std::move(prepared), original_summary, 0.0,
                                        std::move(options), ninfer::OutputDelivery::TerminalOnly);
    const auto result = submission.wait(nullptr, cancel);
    int failures      = 0;
    failures += check(result.finish_reason == FinishReason::StopToken &&
                          result.recovery.attempts == 2 && result.recovery.prefill_samples == 2 &&
                          result.recovery.discarded_reasoning_tokens >= 8192,
                      "committed reasoning did not trigger and finish both bounded retries");
    failures += check(result.prompt.prompt_tokens == original_summary.prompt_tokens &&
                          result.generated_token_ids.size() > 8192 &&
                          result.generated_token_ids.size() < 20000 &&
                          result.generated_token_ids.size() ==
                              executor.runtime_stats().committed_decode_tokens + 3,
                      "retry lifecycle lost original prompt usage or generated-token accounting");
    failures += check(program.prefilled_prompts.size() == 3 && program.abort_count == 0,
                      "retry lifecycle did not preserve the resident lane twice");
    if (program.prefilled_prompts.size() == 3) {
        const auto& original = program.prefilled_prompts[0];
        const auto& first    = program.prefilled_prompts[1];
        const auto& second   = program.prefilled_prompts[2];
        // Each notice is markup around literal text; count its exact token run.
        const auto owned = resources();
        const ninfer::targets::qwen3_6::frontend_internal::Tokenizer tokenizer(
            {.tokenizer_json         = owned.tokenizer_json,
             .tokenizer_config_json  = owned.tokenizer_config_json,
             .generation_config_json = owned.generation_config_json});
        const auto notice_ids = [&](std::uint32_t attempt) {
            const std::string notice = recovery_notice(attempt);
            const std::string fragment =
                std::string(kFragmentOpen) + notice + std::string(kFragmentClose);
            const std::array<ninfer::targets::qwen3_6::frontend_internal::ByteSpan, 1> literal{
                {{.begin = kFragmentOpen.size(), .end = kFragmentOpen.size() + notice.size()}}};
            const std::vector<int> ids = tokenizer.encode(fragment, {}, literal);
            return std::vector<ninfer::TokenId>(ids.begin(), ids.end());
        };
        const auto occurrences = [](const std::vector<ninfer::TokenId>& haystack,
                                    const std::vector<ninfer::TokenId>& needle) {
            std::size_t count = 0;
            for (auto it = haystack.begin(); (it = std::search(it, haystack.end(), needle.begin(),
                                                               needle.end())) != haystack.end();
                 ++it) {
                ++count;
            }
            return count;
        };
        const auto first_notice  = notice_ids(1);
        const auto second_notice = notice_ids(2);
        failures += check(
            first.size() > original.size() && second.size() > first.size() &&
                std::equal(original.begin(), original.end(), first.begin()) &&
                std::equal(first.begin(), first.end(), second.begin()) &&
                occurrences(first, first_notice) == 1 && occurrences(second, first_notice) == 1 &&
                occurrences(second, second_notice) == 1 &&
                second.size() < original.size() + first_notice.size() + second_notice.size() + 128,
            "second retry lost the first notice or included failed generation");
    }
    return failures;
}

// Admission while another lane decodes and a full RAM tier is spilling on the disk worker
// (kv_ram_reclaim_pending). Lane 0 decodes; lanes 1 and 2 are retained. The admission picks
// lane 1, evicts lane 2 as its victim, and captures both. It must wait without claiming or
// capturing while the reclaim is pending, roll back the victim capture when the lane capture
// defers, pass its earlier capture as the reclaim keep list, and finish once the spill lands.
class DeferProgram : public ProbeProgram {
public:
    struct Capture {
        std::uint32_t lane = 0;
        bool may_block     = true;
        std::vector<std::uint64_t> keep;
        std::uint64_t id = 0;
        bool deferred    = false;
    };

    std::atomic<bool> reclaim_pending{false};
    std::atomic<bool> defer_lane_capture{false};
    std::atomic<bool> release_decoder{false};
    std::atomic<std::uint32_t> decode_calls{0};
    std::atomic<std::uint32_t> prefills{0};

    std::vector<Capture> captures() const {
        std::lock_guard lock(mu_);
        return captures_;
    }

    std::vector<std::uint64_t> discards() const {
        std::lock_guard lock(mu_);
        return discards_;
    }

    std::vector<std::uint32_t> closed() const {
        std::lock_guard lock(mu_);
        return closed_;
    }

    [[nodiscard]] AdmissionResources admission_capacity() const noexcept override {
        return AdmissionResources{3, 3, 0};
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const ProbePlan&) const noexcept override {
        return !has_retained_lane(lane);
    }

    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t,
                                           const ProbePlan&) const noexcept override {
        return true;
    }

    [[nodiscard]] bool
    can_admit_lane_after_releasing(std::uint32_t, const ProbePlan&,
                                   std::span<const std::uint32_t> victims) const noexcept override {
        return !victims.empty();
    }

    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept override {
        return lane < retained_.size() && retained_[lane].load();
    }

    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t lane) const noexcept override {
        return has_retained_lane(lane) ? 5 + 4 * static_cast<std::uint64_t>(lane) : 0;
    }

    void evict_retained_lane(std::uint32_t lane) noexcept override {
        if (lane < retained_.size()) { retained_[lane].store(false); }
    }

    [[nodiscard]] bool kv_ram_reclaim_pending() const override { return reclaim_pending.load(); }

    // The admission's captures complete at once, so its copy hold ends at the next boundary.
    [[nodiscard]] bool kv_copies_ready() const override { return true; }

    [[nodiscard]] bool capture_retained_lane(std::uint32_t lane, std::uint64_t* ram_entry_id,
                                             bool may_block, bool* deferred,
                                             std::span<const std::uint64_t> keep) override {
        std::lock_guard lock(mu_);
        Capture capture{.lane      = lane,
                        .may_block = may_block,
                        .keep      = std::vector<std::uint64_t>(keep.begin(), keep.end())};
        if (ram_entry_id != nullptr) { *ram_entry_id = 0; }
        if (deferred != nullptr) { *deferred = false; }
        if (lane == 1 && defer_lane_capture.exchange(false)) {
            capture.deferred = true;
            if (deferred != nullptr) { *deferred = true; }
            // The deferral means a spill is in flight; it lands when the test clears the flag.
            reclaim_pending.store(true);
            captures_.push_back(std::move(capture));
            return false;
        }
        capture.id = next_id_++;
        if (ram_entry_id != nullptr) { *ram_entry_id = capture.id; }
        captures_.push_back(std::move(capture));
        return true;
    }

    void discard_ram_capture(std::uint64_t entry_id) override {
        std::lock_guard lock(mu_);
        discards_.push_back(entry_id);
    }

    void mark_turn_closed(std::uint32_t lane) noexcept override {
        std::lock_guard lock(mu_);
        closed_.push_back(lane);
    }

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan, TransientRegion, PrefillPace,
                                                       const OutputSession*) override {
        if (cache_idle_permission.load()) { execution_while_idle.store(true); }
        prefills.fetch_add(1);
        if (lane < retained_.size()) { retained_[lane].store(false); }
        prefill_token_ = lane == 0 ? kDecodeToken : kCallerStop;
        PrefillStepResult step;
        step.complete                  = true;
        step.processed_prompt_tokens   = 1;
        step.round.tokens              = std::span<const TokenId>(&prefill_token_, 1);
        step.summary.prompt_tokens     = prompt.summary().prompt_tokens;
        step.summary.prefix_reuse_path = ninfer::PrefixReusePath::FullReset;
        return step;
    }

    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const RoundBudget>) override {
        if (cache_idle_permission.load()) { execution_while_idle.store(true); }
        decode_calls.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            decode_tokens_[row] = release_decoder.load() ? kCallerStop : kDecodeToken;
            decode_counts_[row] = 1;
        }
        return {.tokens     = std::span<const TokenId>(decode_tokens_.data(), lanes.size()),
                .row_counts = std::span<const std::int32_t>(decode_counts_.data(), lanes.size()),
                .row_stride = 1};
    }

private:
    static constexpr TokenId kDecodeToken = 23;
    mutable std::mutex mu_;
    std::vector<Capture> captures_;
    std::vector<std::uint64_t> discards_;
    std::vector<std::uint32_t> closed_;
    std::uint64_t next_id_ = 101;
    std::array<std::atomic<bool>, 3> retained_{false, true, true};
    TokenId prefill_token_ = kCallerStop;
    std::array<TokenId, 3> decode_tokens_{};
    std::array<std::int32_t, 3> decode_counts_{};
};

struct DeferPackage {
    using Program         = DeferProgram;
    using RequestBasePlan = ProbePlan;
    using RequestPlan     = ProbePlan;
};

struct DeferProbe {
    using Package         = DeferPackage;
    DeferProgram* program = nullptr;
    ProbeLoaded* loaded   = nullptr;
    ProbeMemory request_memory;
};

// A sampler that ranks the close token first after reasoning. Correct exclusion updates must
// select answer text instead; speculative proposals also include a second close before that update.
class ReasoningCloseProgram final : public ProbeProgram {
public:
    bool speculative                   = false;
    bool capture_logprobs              = false;
    bool close_in_prefill              = false;
    bool initial_close_excluded        = false;
    bool continued_with_close_excluded = false;
    std::vector<std::uint32_t> committed_prefixes;

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion, PrefillPace,
                                                       const OutputSession* output) override {
        output_ = output;
        excluded_.assign(plan.execution.suppressed_token_ids.begin(),
                         plan.execution.suppressed_token_ids.begin() +
                             plan.execution.suppressed_token_count);
        initial_close_excluded = close_excluded();
        thinking_              = output->in_reasoning();
        tokens_[0]             = !thinking_ ? 15 : close_in_prefill ? 248069 : 0;
        if (close_in_prefill) { emitted_close_ = true; }
        PrefillStepResult result;
        result.complete                  = true;
        result.processed_prompt_tokens   = 1;
        result.round.tokens              = std::span<const TokenId>(tokens_.data(), 1);
        result.round.logprobs            = logprobs();
        result.summary.prompt_tokens     = prompt.summary().prompt_tokens;
        result.summary.prefix_reuse_path = ninfer::PrefixReusePath::FullReset;
        return result;
    }

    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                                     std::span<const RoundBudget>) override {
        if (thinking_ && !emitted_close_) {
            if (close_excluded()) { throw std::logic_error("first close was excluded"); }
            tokens_        = {248069, 248069, 15};
            count_         = speculative ? 3 : 1;
            emitted_close_ = true;
        } else if (thinking_ && !emitted_answer_) {
            continued_with_close_excluded = close_excluded();
            tokens_[0]                    = close_excluded() ? 15 : 248069;
            count_                        = 1;
            emitted_answer_               = true;
        } else {
            continued_with_close_excluded = close_excluded();
            tokens_[0]                    = 6;
            count_                        = 1;
        }
        return {.tokens     = std::span<const TokenId>(tokens_.data(), count_),
                .logprobs   = logprobs(),
                .row_counts = std::span<const std::int32_t>(&count_, 1),
                .row_stride = static_cast<std::uint32_t>(count_)};
    }

    void set_suppressed_tokens_lane(std::uint32_t, std::span<const TokenId> tokens) override {
        excluded_.assign(tokens.begin(), tokens.end());
    }

    void clear_suppressed_tokens_lane(std::uint32_t) override { excluded_.clear(); }

    void resolve_pending_batch(std::span<const std::uint32_t>,
                               std::span<const std::uint32_t> accepted,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                               std::span<const std::uint8_t>) override {
        committed_prefixes.push_back(accepted.front());
    }

private:
    ninfer::runtime::RoundLogprobs logprobs() {
        if (!capture_logprobs) { return {}; }
        logprob_values_.fill(-0.25F);
        top_values_.fill(-0.5F);
        return {logprob_values_, top_ids_, top_values_};
    }

    bool close_excluded() const {
        if (output_ && output_->has_token_masks()) {
            const std::array<TokenId, 1> root{0};
            const std::array<std::int32_t, 1> parents{-1};
            std::vector<std::uint32_t> mask((248077 + 31) / 32);
            output_->fill_token_masks(root, parents, mask);
            if ((mask[248069 / 32] & (1U << (248069 % 32))) == 0) { return true; }
        }
        return std::find(excluded_.begin(), excluded_.end(), 248069) != excluded_.end();
    }

    const OutputSession* output_ = nullptr;
    std::array<float, 3> logprob_values_{};
    std::array<TokenId, std::size_t{3} * ninfer::kMaximumTopLogprobs> top_ids_{};
    std::array<float, std::size_t{3} * ninfer::kMaximumTopLogprobs> top_values_{};
    std::vector<TokenId> excluded_;
    std::array<TokenId, 3> tokens_{};
    std::int32_t count_  = 1;
    bool thinking_       = false;
    bool emitted_close_  = false;
    bool emitted_answer_ = false;
};

int run_reasoning_close_sampling(Frontend& frontend) {
    int failures = 0;
    for (const bool thinking : {false, true}) {
        for (const bool speculative : {false, true}) {
            for (const bool prefill_close : {false, true}) {
                if (!thinking && prefill_close) { continue; }
                ReasoningCloseProgram program;
                program.lifecycle_retries = true;
                program.script            = CacheCase::AdmitRamHit;
                program.speculative       = speculative;
                program.close_in_prefill  = prefill_close;
                ProbeLoaded loaded{frontend};
                RecoveryProbe instance{.program = &program, .loaded = &loaded};
                auto engine                = engine_options();
                engine.generation_recovery = false;
                ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
                auto input                    = thinking_input();
                input.options.enable_thinking = thinking;
                auto prompt                   = frontend.prepare(std::move(input));
                const auto summary            = prompt.summary();
                ninfer::runtime::ResolvedRequestOptions options;
                options.execution.allow_prefix_reuse      = false;
                options.execution.requested_output_tokens = 8;
                auto submission =
                    executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                    ninfer::OutputDelivery::TerminalOnly);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                const auto result =
                    submission.wait(nullptr, ninfer::CancellationView([deadline] {
                                        return std::chrono::steady_clock::now() >= deadline;
                                    }));
                const std::size_t expected_closes = thinking ? 1 : 0;
                failures += check(
                    result.finish_reason == FinishReason::StopToken && result.content == "answer" &&
                        std::count(result.generated_token_ids.begin(),
                                   result.generated_token_ids.end(),
                                   248069) == static_cast<std::ptrdiff_t>(expected_closes) &&
                        program.initial_close_excluded == !thinking &&
                        program.continued_with_close_excluded,
                    "executor failed to update close exclusion across prefill/decode");
                failures += check(!program.committed_prefixes.empty() &&
                                      program.committed_prefixes.front() == 1,
                                  "executor committed a repeated close from speculative columns");
            }
        }
    }
    return failures;
}

int run_reasoning_budget_close(Frontend& frontend) {
    int failures = 0;
    for (const bool speculative : {false, true}) {
        ReasoningCloseProgram program;
        program.lifecycle_retries = true;
        program.script            = CacheCase::AdmitRamHit;
        program.speculative       = speculative;
        program.capture_logprobs  = true;
        ProbeLoaded loaded{frontend};
        RecoveryProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
        auto prompt        = frontend.prepare(thinking_input());
        const auto summary = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.allow_prefix_reuse      = false;
        options.execution.requested_output_tokens = 8;
        options.output.thinking_token_budget      = 1;
        options.output.top_logprobs               = 1;
        auto submission     = executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                              ninfer::OutputDelivery::TerminalOnly);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        const auto result   = submission.wait(nullptr, ninfer::CancellationView([deadline] {
                                                return std::chrono::steady_clock::now() >= deadline;
                                              }));
        failures += check(result.content == "answer" && result.reasoning_limit.used_tokens == 1 &&
                              result.reasoning_limit.forced_closes == 1 &&
                              result.reasoning_limit.forced_tokens.size() == 1,
                          "forced reasoning close lost its independent intervention provenance");
        if (!result.reasoning_limit.forced_tokens.empty()) {
            const auto& forced = result.reasoning_limit.forced_tokens.front();
            failures += check(forced.generated_token_index == 1 && forced.token == 248069 &&
                                  forced.target_logprob == -0.25F,
                              "forced close reported masked confidence instead of target logprob");
        }
        failures += check(
            result.reasoning_logprobs.size() == 1 && result.reasoning_logprobs.front().token == 0 &&
                result.content_logprobs.size() == 1 && result.content_logprobs.front().token == 15,
            "forced close leaked policy framing into channel logprob records");
    }
    return failures;
}

// Real output policy and executor publication, with deterministic licensed tokens.
// The last row reaches the evidence threshold and natural EOS in the same round.
class RepeatedAnswerProgram final : public ProbeProgram {
public:
    std::size_t repeat_tokens = 4352;
    std::vector<TokenId> script_tokens;
    bool pause_for_cancel = false;

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t, PreparedPrompt prompt,
                                                       ProbePlan, TransientRegion, PrefillPace,
                                                       const OutputSession*) override {
        emitted_   = 1;
        tokens_[0] = script_tokens.empty() ? 15 : script_tokens.front();
        PrefillStepResult result;
        result.complete                = true;
        result.processed_prompt_tokens = 1;
        result.round.tokens            = std::span<const TokenId>(tokens_.data(), 1);
        result.summary.prompt_tokens   = prompt.summary().prompt_tokens;
        return result;
    }

    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                                     std::span<const RoundBudget>) override {
        // Cancellation is polled by the waiting client every10ms. Let it be
        // observed before this scripted program can finish in under10ms.
        if (pause_for_cancel) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); }
        if (!script_tokens.empty()) {
            before_ = emitted_;
            count_  = static_cast<std::int32_t>(
                std::min<std::size_t>(64, script_tokens.size() - emitted_));
            std::copy_n(script_tokens.begin() + emitted_, count_, tokens_.begin());
        } else {
            const auto count = std::min<std::size_t>(64, repeat_tokens - emitted_);
            std::fill_n(tokens_.begin(), count, 15);
            emitted_ += count;
            count_ = static_cast<std::int32_t>(count);
            if (emitted_ == repeat_tokens) { tokens_[count_++] = 6; }
        }
        return {.tokens     = std::span<const TokenId>(tokens_.data(), count_),
                .row_counts = std::span<const std::int32_t>(&count_, 1),
                .row_stride = static_cast<std::uint32_t>(count_)};
    }

    void resolve_pending_batch(std::span<const std::uint32_t>,
                               std::span<const std::uint32_t> accepted,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                               std::span<const std::uint8_t>) override {
        if (!script_tokens.empty()) { emitted_ = before_ + accepted.front(); }
    }
private:
    std::size_t emitted_ = 0;
    std::size_t before_  = 0;
    std::array<TokenId, 65> tokens_{};
    std::int32_t count_ = 0;
};

int run_repeated_answer(Frontend& frontend) {
    class Sink final : public ninfer::OutputSink {
    public:
        std::string text;

        void publish(ninfer::OutputDelta delta) override { text += delta.text; }
    };

    int failures = 0;
    enum class Mode {
        Guarded,
        Standard,
        Runaway,
        Disabled,
        Raw,
        Constrained,
        ToolArgument,
        CallerToken,
        CallerString,
        Cancelled
    };
    for (const auto mode : {Mode::Guarded, Mode::Standard, Mode::Runaway, Mode::Disabled, Mode::Raw,
                            Mode::Constrained, Mode::ToolArgument, Mode::CallerToken,
                            Mode::CallerString, Mode::Cancelled}) {
        for (const auto delivery :
             {ninfer::OutputDelivery::TerminalOnly, ninfer::OutputDelivery::Streaming}) {
            RepeatedAnswerProgram program;
            program.pause_for_cancel = mode == Mode::Cancelled;
            if (mode == Mode::Runaway) { program.repeat_tokens = 5000; }
            const bool should_fail =
                mode == Mode::Guarded || mode == Mode::Standard || mode == Mode::Runaway;
            program.lifecycle_retries = true;
            program.script            = CacheCase::AdmitRamHit;
            ProbeLoaded loaded{frontend};
            RecoveryProbe instance{.program = &program, .loaded = &loaded};
            auto engine                = engine_options();
            engine.max_context         = 16384;
            engine.generation_recovery = mode != Mode::Disabled;
            ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            if (mode == Mode::Constrained) {
                std::string literal;
                for (int i = 0; i < 4352; ++i) { literal += "answer"; }
                input.options.output_constraint = ninfer::OutputConstraint{
                    .kind   = ninfer::OutputConstraintKind::Grammar,
                    .source = "root ::= " + nlohmann::json(literal).dump()};
            }
            if (mode == Mode::ToolArgument) {
                std::string value = "</tool_call>";
                for (int i = 0; i < 4352; ++i) { value += "answer"; }
                input.options.tool_jsons.push_back(nlohmann::json{
                    {"type", "function"},
                    {"function",
                     {{"name", "f"},
                      {"parameters",
                       {{"type", "object"},
                        {"properties", {{"value", {{"const", value}}}}},
                        {"required", {"value"}},
                        {"additionalProperties",
                         false}}}}}}.dump());
                const auto add_text = [&](std::string_view text) {
                    for (const unsigned char byte : text) {
                        program.script_tokens.push_back(byte == '\n' ? 32 : 1000 + byte);
                    }
                };
                add_text("<tool_call>\n<function=f>\n<parameter=value>\n</tool_call>");
                program.script_tokens.insert(program.script_tokens.end(), 4352, 15);
                add_text("\n</parameter>\n</function>\n</tool_call>");
                program.script_tokens.push_back(6);
            }
            auto prompt        = frontend.prepare(std::move(input));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.sampling.p_less         = mode != Mode::Standard;
            options.execution.allow_prefix_reuse      = false;
            options.execution.requested_output_tokens = 6000;
            options.output.raw                        = mode == Mode::Raw;
            if (mode == Mode::CallerToken) { options.stop.token_ids = {6}; }
            if (mode == Mode::CallerString) {
                options.stop.strings.push_back({.text = "answeranswer"});
            }
            auto submission =
                executor.submit(std::move(prompt), summary, 0.0, std::move(options), delivery);
            Sink sink;
            bool rejected = false;
            try {
                const auto result = submission.wait(
                    &sink, ninfer::CancellationView([mode] { return mode == Mode::Cancelled; }));
                failures += check(!should_fail, "repetitive answer returned normal success");
                if (mode == Mode::ToolArgument) {
                    failures += check(result.tool_calls.size() == 1 && result.content.empty(),
                                      "repeated tool argument failed or leaked into prose");
                }
                if (mode == Mode::Cancelled) {
                    failures += check(result.finish_reason == FinishReason::Cancelled,
                                      "answer guard bypassed cancellation");
                } else if (mode == Mode::CallerString) {
                    failures += check(result.finish_reason == FinishReason::StopString,
                                      "answer guard bypassed caller string stop");
                } else {
                    failures += check(result.finish_reason == FinishReason::StopToken,
                                      "excluded answer route lost its natural/caller stop");
                }
            } catch (const ninfer::RequestError& error) {
                rejected = error.kind() == ninfer::RequestErrorKind::RecoveryExhausted &&
                           std::string_view(error.what()).find("repeated answer text") !=
                               std::string_view::npos;
            }
            if (rejected != should_fail) {
                std::cerr << "answer mode=" << static_cast<int>(mode)
                          << " delivery=" << static_cast<int>(delivery) << '\n';
            }
            failures +=
                check(rejected == should_fail, "repetitive-answer error scope is incorrect");
            if (should_fail && delivery == ninfer::OutputDelivery::Streaming) {
                std::string expected;
                const int published = mode == Mode::Runaway ? 4353 : 4352;
                for (int i = 0; i < published; ++i) { expected += "answer"; }
                failures += check(sink.text == expected,
                                  "answer failure lost/rewrote its published prefix");
            }
        }
    }
    return failures;
}

// The last licensed round contains the proof and a reasoning close, optionally
// followed by answer/tool output. The retry must not publish that abandoned tail.
class ReasoningBoundaryProgram final : public ProbeProgram {
public:
    std::vector<TokenId> first_attempt;
    bool pause_for_cancel = false;
    bool repeat_on_retry  = false;
    std::vector<std::vector<TokenId>> retry_attempts;
    std::vector<ninfer::ResolvedSamplingParameters> attempt_sampling;
    std::atomic<std::uint32_t> active_attempt{0};
    std::uint32_t retry_batch_limit = 0;
    std::atomic<std::size_t> committed_tokens{0};

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion, PrefillPace,
                                                       const OutputSession*) override {
        ++prefill_count;
        attempt_sampling.push_back(plan.execution.sampling);
        prefilled_prompts.push_back(FrontendTestAccess::inspect(prompt).token_ids);
        current_ = (prefill_count == 1 || repeat_on_retry) ? first_attempt
                   : FrontendTestAccess::inspect(prompt).starts_in_reasoning
                       ? std::vector<TokenId>{0, 248069, 15, 6}
                       : std::vector<TokenId>{15, 6};
        if (prefill_count > 1 && !retry_attempts.empty()) {
            current_ = retry_attempts.at(
                std::min<std::size_t>(prefill_count - 2, retry_attempts.size() - 1));
        }
        emitted_ = 1;
        committed_tokens.store(emitted_);
        active_attempt.store(prefill_count);
        tokens_[0] = current_.front();
        PrefillStepResult result;
        result.complete                = true;
        result.processed_prompt_tokens = 1;
        result.round.tokens            = std::span<const TokenId>(tokens_.data(), 1);
        result.summary.prompt_tokens   = prompt.summary().prompt_tokens;
        return result;
    }

    [[nodiscard]] BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t>, std::span<const RoundBudget> budgets) override {
        if (pause_for_cancel || (retry_batch_limit != 0 && prefill_count > 1)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        const auto remaining = current_.size() - emitted_;
        auto batch           = remaining <= 192 ? remaining : 64;
        if (retry_batch_limit != 0 && prefill_count > 1) {
            batch = std::min<std::size_t>(batch, retry_batch_limit);
        }
        count_ = static_cast<std::int32_t>(
            std::min<std::size_t>(batch, budgets.front().generated_tokens_remaining));
        std::copy_n(current_.begin() + emitted_, count_, tokens_.begin());
        return {.tokens     = std::span<const TokenId>(tokens_.data(), count_),
                .row_counts = std::span<const std::int32_t>(&count_, 1),
                .row_stride = static_cast<std::uint32_t>(count_)};
    }

    void resolve_pending_batch(std::span<const std::uint32_t>,
                               std::span<const std::uint32_t> accepted,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                               std::span<const std::uint8_t>) override {
        emitted_ += accepted.front();
        committed_tokens.store(emitted_);
    }
private:
    std::vector<TokenId> current_;
    std::size_t emitted_ = 0;
    std::array<TokenId, 192> tokens_{};
    std::int32_t count_ = 0;
};

int run_reasoning_budget_retry(Frontend& frontend) {
    enum class Mode { Useful, CappedNonempty, EmptyReplacement, NaturalClose, CallerStop };

    class Sink final : public ninfer::OutputSink {
    public:
        std::string content;

        void publish(ninfer::OutputDelta delta) override {
            if (delta.channel == ninfer::OutputChannel::Content) { content += delta.text; }
        }
    };

    int failures = 0;
    for (const auto mode : {Mode::Useful, Mode::CappedNonempty, Mode::EmptyReplacement,
                            Mode::NaturalClose, Mode::CallerStop}) {
        for (const auto delivery :
             {ninfer::OutputDelivery::TerminalOnly, ninfer::OutputDelivery::Streaming}) {
            ReasoningBoundaryProgram program;
            program.lifecycle_retries = true;
            program.script            = CacheCase::AdmitRamHit;
            program.first_attempt = mode == Mode::NaturalClose ? std::vector<TokenId>{248069, 15, 6}
                                                               : std::vector<TokenId>{0};
            program.retry_attempts = {mode == Mode::EmptyReplacement ? std::vector<TokenId>{6}
                                      : mode == Mode::CappedNonempty ? std::vector<TokenId>{15, 15}
                                                                     : std::vector<TokenId>{15, 6}};
            ProbeLoaded loaded{frontend};
            RecoveryProbe instance{.program = &program, .loaded = &loaded};
            auto engine                = engine_options();
            engine.max_context         = 16384;
            engine.generation_recovery = false;
            ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(
                instance, engine, ninfer::runtime::ReasoningLimitAction::Retry);
            auto prompt        = frontend.prepare(thinking_input());
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.requested_output_tokens = mode == Mode::CappedNonempty ? 3 : 8;
            options.output.thinking_token_budget      = 1;
            if (mode == Mode::CallerStop) { options.stop.token_ids = {0}; }
            auto submission =
                executor.submit(std::move(prompt), summary, 0.0, std::move(options), delivery);
            const bool should_fail = mode == Mode::CappedNonempty || mode == Mode::EmptyReplacement;
            Sink sink;
            bool failed         = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            try {
                const auto result = submission.wait(&sink, ninfer::CancellationView([deadline] {
                    return std::chrono::steady_clock::now() >= deadline;
                }));
                failures += check(!should_fail && result.finish_reason == FinishReason::StopToken,
                                  "reasoning budget retry accepted an invalid replacement");
                if (mode == Mode::Useful) {
                    failures += check(
                        result.content == "answer" && result.reasoning_limit.used_tokens == 1 &&
                            result.reasoning_limit.retries == 1 &&
                            result.generated_token_ids.size() == 3 && result.recovery.attempts == 1,
                        "independent budget retry lost output/accounting state");
                } else {
                    failures += check(result.reasoning_limit.retries == 0,
                                      "budget policy retried natural or caller-terminated output");
                }
            } catch (const ninfer::RequestError& error) {
                failed = error.kind() == ninfer::RequestErrorKind::RecoveryExhausted;
            }
            failures += check(failed == should_fail,
                              "budget replacement failure did not retain explicit error semantics");
            failures +=
                check(program.prefill_count ==
                          (mode == Mode::NaturalClose || mode == Mode::CallerStop ? 1U : 2U),
                      "disabled general recovery allowed an additional clean retry");
            if (should_fail) {
                failures += check(sink.content.empty(),
                                  "failed budget replacement published held answer text");
            }
        }
    }
    return failures;
}

int run_reasoning_repetition_boundaries(Frontend& frontend) {
    class Sink final : public ninfer::OutputSink {
    public:
        std::string content;

        void publish(ninfer::OutputDelta delta) override {
            if (delta.channel == ninfer::OutputChannel::Content) { content += delta.text; }
        }
    };
    enum class Mode {
        Close,
        NaturalStop,
        Tool,
        Budget,
        Media,
        Greedy,
        CallerToken,
        CallerString,
        Cancelled,
        Disabled,
        Legitimate
    };
    int failures        = 0;
    auto media_frontend = FrontendTestAccess::create_component(resources(), true);
    for (const auto mode : {Mode::Close, Mode::NaturalStop, Mode::Tool, Mode::Budget, Mode::Media,
                            Mode::Greedy, Mode::CallerToken, Mode::CallerString, Mode::Cancelled,
                            Mode::Disabled, Mode::Legitimate}) {
        for (const auto delivery :
             {ninfer::OutputDelivery::TerminalOnly, ninfer::OutputDelivery::Streaming}) {
            ReasoningBoundaryProgram program;
            program.lifecycle_retries = true;
            program.script            = CacheCase::AdmitRamHit;
            program.pause_for_cancel  = mode == Mode::Cancelled;
            program.first_attempt.assign(mode == Mode::Legitimate ? 512 : 4352, 0);
            program.first_attempt.push_back(248069);
            auto input = thinking_input();
            // Grammar-managed transitions may license answer/EOS or a complete
            // tool in the same round as the reasoning proof.
            if (mode != Mode::Close && mode != Mode::Budget && mode != Mode::Media) {
                input.options.tool_jsons.push_back(
                    R"({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{},"additionalProperties":false}}})");
            }
            if (mode == Mode::Tool) {
                for (const unsigned char byte :
                     std::string_view("<tool_call>\n<function=f>\n\n</function>\n</tool_call>")) {
                    program.first_attempt.push_back(byte == '\n' ? 32 : 1000 + byte);
                }
            } else if (mode != Mode::Budget) {
                program.first_attempt.push_back(15);
            }
            if (mode != Mode::Budget) { program.first_attempt.push_back(6); }
            if (mode == Mode::Media) {
                ninfer::MessagePart image;
                image.kind               = ninfer::MessagePartKind::Media;
                image.media.kind         = ninfer::MediaKind::Image;
                const std::string header = "P6\n64 64\n255\n";
                image.media.bytes.assign(header.begin(), header.end());
                image.media.bytes.resize(header.size() + 64 * 64 * 3, 128);
                image.media.media_type  = "image/x-portable-pixmap";
                image.media.source_name = "repetition.ppm";
                input.messages.front().parts.push_back(std::move(image));
            }
            auto& selected_frontend = mode == Mode::Media ? media_frontend : frontend;
            ProbeLoaded loaded{selected_frontend};
            RecoveryProbe instance{.program = &program, .loaded = &loaded};
            auto engine                = engine_options();
            engine.max_context         = 16384;
            engine.generation_recovery = mode != Mode::Disabled;
            ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
            auto prompt        = selected_frontend.prepare(std::move(input));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            // Standard and greedy sampling share the output-integrity guard.
            options.execution.sampling.p_less         = false;
            options.execution.sampling.temperature    = mode == Mode::Greedy ? 0.0F : 1.0F;
            options.execution.allow_prefix_reuse      = false;
            options.execution.requested_output_tokens = mode == Mode::Budget ? 4353 : 6000;
            if (mode == Mode::CallerToken) { options.stop.token_ids = {6}; }
            if (mode == Mode::CallerString) { options.stop.strings.push_back({.text = "answer"}); }
            const bool should_fail  = mode == Mode::Budget || mode == Mode::Media;
            const bool should_retry = mode == Mode::Close || mode == Mode::NaturalStop ||
                                      mode == Mode::Tool || mode == Mode::Greedy;
            auto submission =
                executor.submit(std::move(prompt), summary, 0.0, std::move(options), delivery);
            Sink sink;
            bool rejected       = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            try {
                const auto result =
                    submission.wait(&sink, ninfer::CancellationView([mode, deadline] {
                        return mode == Mode::Cancelled ||
                               std::chrono::steady_clock::now() >= deadline;
                    }));
                failures += check(!should_fail, "proved reasoning returned normal success");
                failures += check(result.recovery.attempts == (should_retry ? 1U : 0U),
                                  "reasoning boundary used the wrong retry count");
                failures += check(result.tool_calls.empty(), "abandoned reasoning leaked a tool");
                if (should_retry) {
                    failures +=
                        check(result.content == "answer" &&
                                  result.finish_reason == FinishReason::StopToken,
                              "reasoning retry leaked or concatenated its abandoned answer");
                    if (delivery == ninfer::OutputDelivery::Streaming) {
                        failures += check(sink.content == "answer",
                                          "reasoning retry streamed its abandoned tail");
                    }
                    std::string retry_prompt;
                    for (const auto token : program.prefilled_prompts.back()) {
                        retry_prompt += selected_frontend.token_bytes(token);
                    }
                    failures +=
                        check(retry_prompt.find(recovery_notice()) != std::string::npos &&
                                  retry_prompt.find("NInfer did not execute this proposed call") ==
                                      std::string::npos,
                              "reasoning proof fabricated duplicate-call feedback");
                }
                if (mode == Mode::CallerToken || mode == Mode::CallerString) {
                    failures += check(result.finish_reason == (mode == Mode::CallerToken
                                                                   ? FinishReason::StopToken
                                                                   : FinishReason::StopString),
                                      "reasoning proof bypassed an explicit caller stop");
                }
                if (mode == Mode::Cancelled) {
                    failures += check(result.finish_reason == FinishReason::Cancelled,
                                      "reasoning proof bypassed cancellation");
                }
            } catch (const ninfer::RequestError& error) {
                rejected = error.kind() == ninfer::RequestErrorKind::RecoveryExhausted;
                failures += check(rejected, "unexpected reasoning boundary request error");
            }
            if (rejected != should_fail) {
                std::cerr << "reasoning boundary mode=" << static_cast<int>(mode)
                          << " delivery=" << static_cast<int>(delivery) << '\n';
            }
            failures += check(rejected == should_fail, "reasoning failure scope is incorrect");
            if (should_fail) {
                failures += check(program.prefill_count == 1 && sink.content.empty(),
                                  "unsupported reasoning recovery published or retried its tail");
            }
        }
    }
    return failures;
}

// A public Close budget preserves reasoning instead of silently selecting clean retry.
int run_limited_recovery_policy(Frontend& frontend) {
    int failures = 0;
    for (const bool replay_failure : {false, true}) {
        ReasoningBoundaryProgram program;
        program.lifecycle_retries = true;
        program.script            = CacheCase::AdmitRamHit;
        if (replay_failure) {
            program.first_attempt.assign(256, 0);
            program.first_attempt.push_back(248069);
            program.first_attempt.insert(program.first_attempt.end(), 256, 0);
            program.first_attempt.push_back(248069);
            program.first_attempt.insert(program.first_attempt.end(), 256, 0);
            program.first_attempt.push_back(6);
        } else {
            program.first_attempt.assign(4352, 0);
            program.first_attempt.insert(program.first_attempt.end(), {248069, 15, 6});
        }
        ProbeLoaded loaded{frontend};
        RecoveryProbe instance{.program = &program, .loaded = &loaded};
        auto engine        = engine_options();
        engine.max_context = 32768;
        ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
        auto prompt        = frontend.prepare(thinking_input());
        const auto summary = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.allow_prefix_reuse      = false;
        options.execution.requested_output_tokens = 10000;
        options.output.thinking_token_budget      = 5000;
        auto submission = executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                          ninfer::OutputDelivery::TerminalOnly);
        try {
            const auto result = submission.wait(nullptr, {});
            failures += check(!replay_failure && result.content == "answer" &&
                                  result.reasoning_limit.used_tokens == 4352 &&
                                  result.recovery.attempts == 0 && program.prefill_count == 1,
                              "capped Close discarded repeated reasoning in an automatic retry");
        } catch (const ninfer::RequestError& error) {
            failures += check(
                replay_failure && error.kind() == ninfer::RequestErrorKind::RecoveryExhausted &&
                    error.recovery().attempts == 0 && program.prefill_count == 1 &&
                    error.reasoning_limit().budget_tokens == 5000,
                "capped Close failed replay validation without preserving no-retry policy");
        }
    }
    return failures;
}

int run_replay_replacement_publication(Frontend& frontend) {
    enum class Mode {
        Release,
        ContentRetry,
        Exhaust,
        Cancel,
        Standard,
        Greedy,
        InheritedCopy,
        Empty,
        Tool,
        BudgetEmpty,
        BudgetTool
    };
    int failures = 0;
    for (const auto mode : {Mode::Release, Mode::ContentRetry, Mode::Exhaust, Mode::Cancel,
                            Mode::Standard, Mode::Greedy, Mode::InheritedCopy, Mode::Empty,
                            Mode::Tool, Mode::BudgetEmpty, Mode::BudgetTool}) {
        for (const auto delivery :
             {ninfer::OutputDelivery::Streaming, ninfer::OutputDelivery::TerminalOnly}) {
            ReasoningBoundaryProgram program;
            program.lifecycle_retries = true;
            program.script            = CacheCase::AdmitRamHit;
            auto input                = thinking_input();
            if (mode == Mode::Tool || mode == Mode::BudgetTool) {
                input.options.tool_jsons.push_back(
                    R"({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{},"additionalProperties":false}}})");
            }
            std::vector<TokenId> reference(256, 0);
            program.first_attempt = reference;
            program.first_attempt.push_back(248069);
            program.first_attempt.insert(program.first_attempt.end(), reference.begin(),
                                         reference.end());
            program.first_attempt.push_back(248069);
            program.first_attempt.insert(program.first_attempt.end(), reference.begin(),
                                         reference.end());
            program.first_attempt.push_back(6);
            const std::vector<TokenId> accepted{15, 6};
            if (mode == Mode::ContentRetry || mode == Mode::Exhaust) {
                std::vector<TokenId> repeated;
                repeated.insert(repeated.end(), 4608, 15);
                repeated.push_back(6);
                program.retry_attempts = {repeated, mode == Mode::Exhaust ? repeated : accepted};
            } else if (mode == Mode::InheritedCopy) {
                auto copied = reference;
                copied.push_back(248069);
                copied.push_back(6);
                program.retry_attempts = {copied, accepted};
            } else if (mode == Mode::Empty) {
                program.retry_attempts = {{6}, accepted};
            } else if (mode == Mode::BudgetEmpty) {
                program.retry_attempts = {std::vector<TokenId>(128, 14)};
            } else if (mode == Mode::Tool || mode == Mode::BudgetTool) {
                std::vector<TokenId> tool;
                for (const unsigned char byte :
                     (mode == Mode::BudgetTool
                          ? std::string_view("<tool_call>\n<function=f>\n")
                          : std::string_view(
                                "<tool_call>\n<function=f>\n\n</function>\n</tool_call>"))) {
                    tool.push_back(byte == '\n' ? 32 : 1000 + byte);
                }
                if (mode == Mode::Tool) { tool.push_back(6); }
                program.retry_attempts = {tool};
            } else if (mode == Mode::Cancel) {
                std::vector<TokenId> partial;
                partial.insert(partial.end(), 32, 15);
                partial.push_back(6);
                program.retry_attempts    = {partial};
                program.retry_batch_limit = 1;
            } else {
                program.retry_attempts = {accepted};
            }

            class Sink final : public ninfer::OutputSink {
            public:
                std::string content;
                std::vector<ninfer::RecoveryEvent> recovery;

                void publish(ninfer::OutputDelta delta) override {
                    if (delta.channel == ninfer::OutputChannel::Content) { content += delta.text; }
                }

                void recovery_event(const ninfer::RecoveryEvent& event) override {
                    recovery.push_back(event);
                }
            } sink;

            ProbeLoaded loaded{frontend};
            RecoveryProbe instance{.program = &program, .loaded = &loaded};
            auto engine        = engine_options();
            engine.max_context = 32768;
            ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
            auto prompt        = frontend.prepare(std::move(input));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse = false;
            options.execution.requested_output_tokens =
                mode == Mode::BudgetEmpty ? 778
                : mode == Mode::BudgetTool
                    ? 770U + static_cast<std::uint32_t>(program.retry_attempts.front().size())
                    : 20000;
            auto& sampling               = options.execution.sampling;
            sampling                     = {.temperature       = mode == Mode::Greedy ? 0.0F : 1.5F,
                                            .top_k             = 19,
                                            .top_p             = 0.7F,
                                            .min_p             = 0.02F,
                                            .presence_penalty  = 0.4F,
                                            .frequency_penalty = 0.2F,
                                            .seed              = 20261020,
                                            .p_less            = mode != Mode::Standard};
            const auto original_sampling = sampling;
            auto submission =
                executor.submit(std::move(prompt), summary, 0.0, std::move(options), delivery);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            bool exhausted      = false;
            try {
                const auto result = submission.wait(&sink, ninfer::CancellationView([&] {
                    return (mode == Mode::Cancel && program.active_attempt.load() == 2 &&
                            program.committed_tokens.load() >= 6) ||
                           std::chrono::steady_clock::now() >= deadline;
                }));
                failures += check(mode != Mode::Exhaust && mode != Mode::BudgetEmpty &&
                                      mode != Mode::BudgetTool,
                                  "failed replacement returned success");
                if (mode == Mode::Cancel) {
                    failures += check(result.finish_reason == FinishReason::Cancelled &&
                                          !result.content.empty(),
                                      "cancelled replacement did not flush text");
                } else {
                    failures += check(result.content == (mode == Mode::Tool ? "" : "answer"),
                                      "rejected replacement leaked into accepted content");
                }
                failures += check(result.tool_calls.size() == (mode == Mode::Tool ? 1U : 0U),
                                  "replacement tool acceptance changed");
                failures += check(result.recovery.attempts ==
                                      (mode == Mode::ContentRetry || mode == Mode::InheritedCopy ||
                                               mode == Mode::Empty
                                           ? 2U
                                           : 1U),
                                  "replacement retry count changed");
                if (mode == Mode::ContentRetry) {
                    failures += check(result.recovery.discarded_content_tokens > 4500,
                                      "failed replacement content was not charged as discarded");
                    std::string notice;
                    for (const auto token : program.prefilled_prompts.back()) {
                        notice += frontend.token_bytes(token);
                    }
                    failures +=
                        check(notice.find("previous replacement repeated its answer text") !=
                                  std::string::npos,
                              "answer-repeat retry notice misstates the cause");
                }
                if (delivery == ninfer::OutputDelivery::Streaming) {
                    failures += check(sink.content == result.content,
                                      "replacement stream differs from accepted terminal output");
                }
            } catch (const ninfer::RequestError& error) {
                exhausted = error.kind() == ninfer::RequestErrorKind::RecoveryExhausted;
                failures += check(
                    (mode == Mode::Exhaust || mode == Mode::BudgetEmpty ||
                     mode == Mode::BudgetTool) &&
                        exhausted && error.recovery().attempts == (mode == Mode::Exhaust ? 2U : 1U),
                    "replacement exhaustion ignored its bounds");
                failures +=
                    check(sink.content.empty(), "failed replacement leaked into the stream");
            }
            failures += check(exhausted == (mode == Mode::Exhaust || mode == Mode::BudgetEmpty ||
                                            mode == Mode::BudgetTool),
                              "replacement error scope changed");
            for (std::size_t attempt = 1; attempt < program.attempt_sampling.size(); ++attempt) {
                const auto& actual  = program.attempt_sampling[attempt];
                const bool fallback = mode != Mode::Standard && mode != Mode::Greedy;
                failures += check(
                    actual.seed == original_sampling.seed &&
                        actual.temperature == (fallback ? 1.0F : original_sampling.temperature) &&
                        actual.p_less == (fallback ? false : original_sampling.p_less) &&
                        actual.top_k == (fallback ? 20 : original_sampling.top_k) &&
                        actual.top_p == (fallback ? 0.95F : original_sampling.top_p) &&
                        actual.min_p == (fallback ? 0.0F : original_sampling.min_p) &&
                        actual.presence_penalty ==
                            (fallback ? 0.0F : original_sampling.presence_penalty) &&
                        actual.frequency_penalty ==
                            (fallback ? 0.0F : original_sampling.frequency_penalty),
                    "replacement used an unintended sampling profile");
            }
            for (const auto& event : sink.recovery) {
                if (event.kind == ninfer::RecoveryEventKind::RetryStarted) {
                    failures +=
                        check(event.enable_thinking == false && event.sampling.has_value() &&
                                  event.sampling->temperature ==
                                      program.attempt_sampling.at(event.attempts).temperature,
                              "retry event omitted its actual sampling profile");
                }
            }
        }
    }
    return failures;
}

int run_withheld_reasoning_replay(Frontend& frontend) {
    class Sink final : public ninfer::OutputSink {
    public:
        std::string content;
        std::vector<ninfer::RecoveryEvent> recovery;

        void publish(ninfer::OutputDelta delta) override {
            if (delta.channel == ninfer::OutputChannel::Content) { content += delta.text; }
        }

        void recovery_event(const ninfer::RecoveryEvent& event) override {
            recovery.push_back(event);
        }
    };
    enum class Mode { Continue, Retry, Empty, Exhaust, Budget, Legitimate, Diverge, Tool, Cancel };
    int failures = 0;
    for (const auto mode : {Mode::Continue, Mode::Retry, Mode::Empty, Mode::Exhaust, Mode::Budget,
                            Mode::Legitimate, Mode::Diverge, Mode::Tool, Mode::Cancel}) {
        for (const auto delivery :
             {ninfer::OutputDelivery::TerminalOnly, ninfer::OutputDelivery::Streaming}) {
            ReasoningBoundaryProgram program;
            program.lifecycle_retries = true;
            program.script            = CacheCase::AdmitRamHit;
            program.repeat_on_retry   = mode == Mode::Exhaust;
            program.pause_for_cancel  = mode == Mode::Cancel;
            program.first_attempt.assign(256, 0);
            program.first_attempt.push_back(248069);
            const auto first_copy = mode == Mode::Diverge ? 128U : 256U;
            program.first_attempt.insert(program.first_attempt.end(), first_copy, 0);
            const bool repair =
                mode != Mode::Legitimate && mode != Mode::Diverge && mode != Mode::Cancel;
            if (repair) { program.first_attempt.push_back(248069); }
            const bool second =
                mode == Mode::Retry || mode == Mode::Exhaust || mode == Mode::Budget;
            if (second) {
                program.first_attempt.insert(program.first_attempt.end(), 256, 0);
            } else if (mode == Mode::Tool) {
                for (const unsigned char byte :
                     std::string_view("<tool_call>\n<function=f>\n\n</function>\n</tool_call>")) {
                    program.first_attempt.push_back(byte == '\n' ? 32 : 1000 + byte);
                }
            } else if (mode == Mode::Continue || mode == Mode::Diverge) {
                program.first_attempt.push_back(15);
            }
            program.first_attempt.push_back(6);
            auto input = thinking_input();
            if (mode == Mode::Tool) {
                input.options.tool_jsons.push_back(
                    R"({"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{},"additionalProperties":false}}})");
            }
            ProbeLoaded loaded{frontend};
            RecoveryProbe instance{.program = &program, .loaded = &loaded};
            auto engine        = engine_options();
            engine.max_context = 16384;
            ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine);
            auto prompt        = frontend.prepare(std::move(input));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.requested_output_tokens = mode == Mode::Budget ? 770 : 4096;
            // Exercise the same publication/recovery rule with standard and greedy sampling.
            options.execution.sampling.p_less      = false;
            options.execution.sampling.temperature = mode == Mode::Continue ? 0.0F : 1.0F;
            if (mode == Mode::Legitimate) { options.stop.token_ids = {6}; }
            // Exposed stop framing must not turn the empty first attempt into an answer.
            if (mode == Mode::Empty) {
                options.output.preserve_special_tokens = true;
                options.stop.publish_stop_token        = true;
            }
            auto submission =
                executor.submit(std::move(prompt), summary, 0.0, std::move(options), delivery);
            Sink sink;
            const auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            bool exhausted         = false;
            const bool should_fail = mode == Mode::Exhaust || mode == Mode::Budget;
            try {
                const auto result = submission.wait(&sink, ninfer::CancellationView([&] {
                    return (mode == Mode::Cancel && program.committed_tokens.load() >= 300) ||
                           std::chrono::steady_clock::now() >= deadline;
                }));
                failures += check(!should_fail, "unrecovered replay returned success");
                const bool retry = mode == Mode::Retry || mode == Mode::Empty;
                failures += check(result.recovery.attempts == (retry ? 1U : 0U),
                                  "replay used the wrong retry count");
                failures += check(result.recovery.framing_repairs == (repair ? 1U : 0U),
                                  "replay lost the actual framing-repair count");
                const auto discarded = mode == Mode::Retry ? 512U : repair ? 256U : 0U;
                failures += check(result.recovery.discarded_content_tokens == discarded,
                                  "withheld tokens were not accounted separately");
                std::string expected;
                if (mode == Mode::Legitimate) {
                    expected.assign(256, 'x');
                } else if (mode == Mode::Diverge) {
                    expected = std::string(128, 'x') + "answer";
                } else if (mode == Mode::Cancel) {
                    const auto close = std::find(result.generated_token_ids.begin(),
                                                 result.generated_token_ids.end(), 248069);
                    expected.assign(static_cast<std::size_t>(
                                        std::count(close + 1, result.generated_token_ids.end(), 0)),
                                    'x');
                    failures +=
                        check(!expected.empty() && result.finish_reason == FinishReason::Cancelled,
                              "cancellation did not reach a quarantined candidate");
                } else if (mode != Mode::Tool) {
                    expected = mode == Mode::Empty ? "answer<eos>" : "answer";
                }
                failures += check(result.content == expected,
                                  "terminal result leaked/erased replay content");
                if (delivery == ninfer::OutputDelivery::Streaming) {
                    failures += check(sink.content == expected,
                                      "stream and terminal replay publication differ");
                }
                failures += check(result.tool_calls.size() == (mode == Mode::Tool ? 1U : 0U),
                                  "framing repair lost or invented a tool call");
                if (retry) {
                    std::string notice;
                    for (const auto token : program.prefilled_prompts.back()) {
                        notice += frontend.token_bytes(token);
                    }
                    failures +=
                        check(notice.find("framing replay was confirmed earlier in this request") !=
                                      std::string::npos &&
                                  notice.find("NInfer did not execute this proposed call") ==
                                      std::string::npos,
                              "replay recovery did not use honest cause-specific feedback");
                }
            } catch (const ninfer::RequestError& error) {
                exhausted = error.kind() == ninfer::RequestErrorKind::RecoveryExhausted;
                failures += check(should_fail && exhausted, "unexpected replay request error");
                failures += check(error.recovery().attempts == (mode == Mode::Exhaust ? 2U : 0U),
                                  "replay exhaustion exceeded its attempt budget");
                failures +=
                    check(sink.content.empty(), "exhausted replay streamed abandoned output");
            }
            failures += check(exhausted == should_fail, "replay exhaustion scope mismatch");
        }
    }
    return failures;
}

int run_admission_defers_for_reclaim(Frontend& frontend) {
    DeferProgram program;
    // Plans sized from each request (service quanta cover the long decode); no resident reuse.
    program.lifecycle_retries = true;
    program.script            = CacheCase::AdmitRamHit;
    ProbeLoaded loaded{frontend};
    DeferProbe instance;
    instance.program           = &program;
    instance.loaded            = &loaded;
    auto engine                = engine_options();
    engine.max_concurrency     = 3;
    engine.max_context         = 32768;
    engine.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<DeferProbe> executor(instance, engine);

    auto submit = [&](std::uint32_t outputs) {
        auto prepared      = frontend.prepare(thinking_input());
        const auto summary = prepared.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.sampling.p_less         = true;
        options.execution.allow_prefix_reuse      = false;
        options.execution.requested_output_tokens = outputs;
        options.stop.token_ids                    = {kCallerStop};
        return executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                               ninfer::OutputDelivery::TerminalOnly);
    };
    auto wait_until = [](auto&& predicate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= deadline) { return false; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    ninfer::CancellationView cancel(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });

    int failures = 0;
    auto decoder = submit(30000);
    if (!wait_until([&] { return program.decode_calls.load() >= 2; })) {
        return check(false, "defer: the decoding request never entered decode");
    }
    program.reclaim_pending.store(true);
    program.defer_lane_capture.store(true);
    auto admitted                     = submit(8);
    const std::uint32_t decode_before = program.decode_calls.load();
    failures += check(wait_until([&] { return program.decode_calls.load() >= decode_before + 20; }),
                      "defer: decode stalled behind a pending reclaim");
    failures += check(program.captures().empty() && program.prefills.load() == 1,
                      "defer: admission captured or prefilled while a reclaim was pending");

    program.reclaim_pending.store(false);
    if (!wait_until([&] { return program.captures().size() >= 2; })) {
        return failures +
               check(false, "defer: admission did not capture after the reclaim cleared");
    }
    {
        const auto captures = program.captures();
        failures +=
            check(captures[0].lane == 2 && captures[0].id != 0 && !captures[0].may_block &&
                      captures[0].keep.empty(),
                  "defer: the victim capture was not first, non-blocking, with no keep list");
        failures += check(captures[1].lane == 1 && captures[1].deferred && !captures[1].may_block &&
                              captures[1].keep == std::vector<std::uint64_t>{captures[0].id},
                          "defer: the lane capture did not defer with the victim capture kept");
        failures += check(wait_until([&] { return !program.discards().empty(); }) &&
                              program.discards() == std::vector<std::uint64_t>{captures[0].id},
                          "defer: the deferred admission did not roll back the victim capture");
    }
    const std::uint32_t deferred_decode = program.decode_calls.load();
    failures +=
        check(wait_until([&] { return program.decode_calls.load() >= deferred_decode + 20; }),
              "defer: decode stalled behind a deferred admission");
    failures += check(program.captures().size() == 2 && program.prefills.load() == 1,
                      "defer: admission retried before the spill landed");

    program.reclaim_pending.store(false);
    const ninfer::GenerationResult admitted_result = admitted.wait(nullptr, cancel);
    {
        const auto captures = program.captures();
        failures += check(captures.size() == 4 && captures[2].lane == 2 && captures[3].lane == 1 &&
                              !captures[3].deferred &&
                              captures[3].keep == std::vector<std::uint64_t>{captures[2].id},
                          "defer: the retried admission did not capture victim then lane");
        failures += check(program.discards().size() == 1,
                          "defer: the completed admission rolled back its captures");
    }
    failures += check(admitted_result.finish_reason == FinishReason::StopToken,
                      "defer: the admitted request did not finish");
    program.release_decoder.store(true);
    const ninfer::GenerationResult decoder_result = decoder.wait(nullptr, cancel);
    failures += check(decoder_result.finish_reason == FinishReason::StopToken &&
                          decoder_result.generated_token_ids.size() > 40,
                      "defer: the decoding request did not keep decoding to its stop");
    auto closed = program.closed();
    std::sort(closed.begin(), closed.end());
    failures += check(closed == std::vector<std::uint32_t>{0, 1},
                      "defer: finished turns without tool calls were not marked closed");
    return failures;
}

int run_cache_maintenance_idle(Frontend& frontend) {
    int failures = 0;
    for (bool decoder_writes_cache : {false, true}) {
        DeferProgram program;
        program.lifecycle_retries = true;
        program.script            = CacheCase::AdmitRamHit;
        program.report_copies_ready.store(true);
        ProbeLoaded loaded{frontend};
        DeferProbe instance;
        instance.program           = &program;
        instance.loaded            = &loaded;
        auto engine                = engine_options();
        engine.max_concurrency     = 3;
        engine.max_context         = 32768;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<DeferProbe> executor(instance, engine);
        const auto wait_until = [](auto&& predicate) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!predicate()) {
                if (std::chrono::steady_clock::now() >= deadline) { return false; }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return true;
        };
        if (!wait_until([&] { return program.cache_idle_permission.load(); })) {
            return failures +
                   check(false, "maintenance: empty Engine did not enable idle maintenance");
        }
        const auto idle_count = program.idle_true_count.load();
        auto submit           = [&](std::uint32_t outputs, bool cache_write) {
            auto prepared      = frontend.prepare(thinking_input());
            const auto summary = prepared.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.sampling.p_less         = true;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = cache_write;
            options.execution.requested_output_tokens = outputs;
            options.stop.token_ids                    = {kCallerStop};
            return executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::TerminalOnly);
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        ninfer::CancellationView cancel(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        auto decoder = submit(30000, decoder_writes_cache);
        if (!wait_until([&] { return program.decode_calls.load() >= 2; })) {
            return failures + check(false, "maintenance: decoder did not start");
        }
        failures += check(!program.cache_idle_permission.load(),
                          "maintenance: decode retained idle permission");
        auto short_request      = submit(8, !decoder_writes_cache);
        const auto short_result = short_request.wait(nullptr, cancel);
        failures +=
            check(short_result.finish_reason == FinishReason::StopToken &&
                      !program.cache_idle_permission.load() &&
                      program.idle_true_count.load() == idle_count,
                  "maintenance: one completed request enabled maintenance beside an active peer");
        program.release_decoder.store(true);
        const auto decoder_result = decoder.wait(nullptr, cancel);
        failures += check(
            decoder_result.finish_reason == FinishReason::StopToken &&
                wait_until([&] { return program.cache_idle_permission.load(); }) &&
                program.idle_true_count.load() > idle_count && !program.execution_while_idle.load(),
            "maintenance: idle handoff failed for ordinary/disposable request activity");
    }
    return failures;
}

// Runs the production selector with scripted complete chunks. A gate after the first large
// chunk freezes ingress so queue timing cannot hide an interruption or its starvation bound.
class PrefillScheduleProgram : public ProbeProgram {
public:
    bool windowed = false;
    std::array<bool, ninfer::kMaximumConcurrency + 8> credits{};

    [[nodiscard]] bool chunked_execution_enabled() const noexcept override { return windowed; }

    [[nodiscard]] bool reserve_suspension_credit(std::uint64_t owner, std::size_t) override {
        credits.at(owner) = true;
        return true;
    }

    void release_suspension_credit(std::uint64_t owner) override { credits.at(owner) = false; }

    [[nodiscard]] bool kv_copies_ready() const override { return true; }

    std::atomic<bool> first_chunk_entered{false};
    std::atomic<bool> release_first_chunk{false};
    std::uint32_t lanes  = 2;
    bool suspendable     = true;
    bool short_fits      = true;
    bool fail_short_once = false;
    bool gate_small      = false;
    bool gate_decode     = false;
    std::atomic<bool> decode_entered{false};
    std::atomic<bool> release_decode{false};
    std::atomic<bool> small_entered{false};
    std::atomic<bool> release_small{false};
    std::vector<std::string> units;
    std::array<std::uint32_t, ninfer::kMaximumConcurrency> remaining{};
    std::array<std::uint32_t, ninfer::kMaximumConcurrency> lengths{};

    [[nodiscard]] AdmissionResources admission_capacity() const noexcept override {
        return {lanes, lanes, 0};
    }

    [[nodiscard]] ProbePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const ninfer::runtime::ResolvedExecutionOptions& options) override {
        ProbePlan plan;
        plan.fields.prompt_tokens           = prompt.summary().prompt_tokens;
        plan.fields.effective_output_tokens = options.requested_output_tokens;
        plan.fields.effective_limit_reason  = FinishReason::OutputLimit;
        plan.fields.service_work_quanta     = 8192;
        plan.fields.admission               = {1, 1, 0};
        plan.fields.lifetime_admission      = {1, lanes, 0};
        plan.fields.window_output_tokens    = options.requested_output_tokens;
        plan.fields.suspension_bytes        = 1;
        plan.fields.transient_alignment     = 1;
        plan.execution                      = options;
        return plan;
    }

    [[nodiscard]] ProbePlan plan_request_for_lane(std::uint32_t, const PreparedPrompt&,
                                                  const ProbePlan& base) override {
        return base;
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const ProbePlan& plan) const noexcept override {
        return short_fits || lane == 0 || plan.fields.prompt_tokens >= 4096;
    }

    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const ProbePlan& plan) const noexcept override {
        return can_admit_lane(lane, plan);
    }

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan, TransientRegion, PrefillPace pace,
                                                       const OutputSession*) override {
        if (windowed && prompt.summary().prompt_tokens < 4096 &&
            std::any_of(
                remaining.begin(), remaining.end(),
                [](std::uint32_t tokens) { return tokens >= 4096; }) &&
            !std::any_of(credits.begin(), credits.end(), [](bool credit) { return credit; })) {
            throw std::logic_error("RAM-on short interruption has no secured backing");
        }
        remaining[lane] = lengths[lane] = prompt.summary().prompt_tokens;
        units.push_back("start:" + std::to_string(lengths[lane]));
        if (lengths[lane] < 4096 && std::exchange(fail_short_once, false)) {
            throw ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                                       "scripted short prefill initialization failure");
        }
        return advance_prefill_lane(lane, pace);
    }

    [[nodiscard]] PrefillStepResult advance_prefill_lane(std::uint32_t lane,
                                                         PrefillPace pace) override {
        units.push_back("chunk:" + std::to_string(lengths[lane]) +
                        (pace == PrefillPace::Exclusive ? ":exclusive" : ":shared"));
        const auto processed = std::min(remaining[lane], 4096U);
        remaining[lane] -= processed;
        if (lengths[lane] >= 4096 && !first_chunk_entered.exchange(true)) {
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!release_first_chunk.load()) {
                if (std::chrono::steady_clock::now() >= limit) {
                    throw std::runtime_error("prefill schedule ingress gate timed out");
                }
                std::this_thread::yield();
            }
        }
        PrefillStepResult step;
        step.summary.prompt_tokens   = lengths[lane];
        step.processed_prompt_tokens = processed;
        step.complete                = remaining[lane] == 0;
        if (lengths[lane] < 4096 && gate_small && !small_entered.exchange(true)) {
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!release_small.load()) {
                if (std::chrono::steady_clock::now() >= limit) {
                    throw std::runtime_error("small prefill cancellation gate timed out");
                }
                std::this_thread::yield();
            }
        }
        if (step.complete) { step.round.tokens = {lengths[lane] == 16 ? &continue_ : &stop_, 1}; }
        return step;
    }

    [[nodiscard]] std::uint32_t prefill_remaining_tokens(std::uint32_t lane) const override {
        return remaining[lane];
    }

    [[nodiscard]] bool prefill_suspendable(std::uint32_t) const override { return suspendable; }

    void resume_prefill_lane(std::uint32_t lane) override {
        units.push_back("resume:" + std::to_string(lengths[lane]));
    }

    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> members,
                                                     std::span<const RoundBudget>) override {
        if (gate_decode && !decode_entered.exchange(true)) {
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!release_decode.load()) {
                if (std::chrono::steady_clock::now() >= limit) {
                    throw std::runtime_error("scripted decode ingress gate timed out");
                }
                std::this_thread::yield();
            }
        }
        units.push_back("decode");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        for (std::size_t row = 0; row < members.size(); ++row) {
            decode_tokens_[row] = continue_;
            decode_counts_[row] = 1;
        }
        return {.tokens     = std::span(decode_tokens_).first(members.size()),
                .row_counts = std::span(decode_counts_).first(members.size()),
                .row_stride = 1};
    }

private:
    std::array<TokenId, ninfer::kMaximumConcurrency> decode_tokens_{};
    std::array<std::int32_t, ninfer::kMaximumConcurrency> decode_counts_{};
    TokenId continue_ = 1000 + 'a';
    TokenId stop_     = kCallerStop;
};

struct PrefillSchedulePackage : ProbePackage {
    using Program = PrefillScheduleProgram;
};

struct PrefillScheduleProbe {
    using Package = PrefillSchedulePackage;
    PrefillScheduleProgram* program;
    ProbeLoaded* loaded;
    ProbeMemory request_memory;
};

// Gates hold complete round boundaries so all eight admissions and a middle-lane cancellation
// have deterministic membership, independent of worker/client thread timing.
class FullBatchProgram final : public PrefillScheduleProgram {
public:
    std::atomic<bool> full_entered{false};
    std::atomic<bool> release_full{false};
    bool compact_seen = false;
    std::vector<std::uint32_t> full_members;

    [[nodiscard]] BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> members,
                 std::span<const RoundBudget> budgets) override {
        if (members.size() == 8 && !full_entered.load()) {
            full_members.assign(members.begin(), members.end());
            full_entered.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!release_full.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("full batch cancellation gate timed out");
                }
                std::this_thread::yield();
            }
        }
        if (full_entered.load() && members.size() == 7) {
            compact_seen = std::find(members.begin(), members.end(), 3U) == members.end() &&
                           std::find(members.begin(), members.end(), 7U) != members.end();
        }
        (void)PrefillScheduleProgram::decode_batch(members, budgets);
        for (std::size_t row = 0; row < members.size(); ++row) {
            tokens_[row] = static_cast<TokenId>(1000 + 'a' + members[row]);
            counts_[row] = 1;
        }
        return {.tokens     = std::span(tokens_).first(members.size()),
                .row_counts = std::span(counts_).first(members.size()),
                .row_stride = 1};
    }

private:
    std::array<TokenId, ninfer::kMaximumConcurrency> tokens_{};
    std::array<std::int32_t, ninfer::kMaximumConcurrency> counts_{};
};

int run_full_batch_compaction(Frontend& frontend) {
    FullBatchProgram program;
    program.lanes       = 8;
    program.gate_decode = true;
    ProbeLoaded loaded{frontend};
    PrefillScheduleProbe instance{&program, &loaded, {}};
    auto options                = engine_options();
    options.max_concurrency     = 8;
    options.max_context         = 32768;
    options.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<PrefillScheduleProbe> executor(instance, options);
    auto submit = [&] {
        auto prompt        = frontend.prepare_tokens(std::vector<TokenId>(16, 1000 + 'a'));
        const auto summary = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions request;
        request.execution.allow_prefix_reuse      = false;
        request.execution.requested_output_tokens = 128;
        return executor.submit(std::move(prompt), summary, 0.0, std::move(request),
                               ninfer::OutputDelivery::TerminalOnly);
    };
    auto wait_gate = [](const std::atomic<bool>& gate) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!gate.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("full batch test did not reach round boundary");
            }
            std::this_thread::yield();
        }
    };
    using Submission = decltype(submit());
    std::array<Submission, 8> requests;
    requests[0] = submit();
    wait_gate(program.decode_entered);
    for (std::size_t lane = 1; lane < requests.size(); ++lane) { requests[lane] = submit(); }
    program.release_decode.store(true);
    wait_gate(program.full_entered);
    requests[3] = {};
    program.release_full.store(true);
    int failures = 0;
    const ninfer::CancellationView cancel;
    for (std::size_t lane = 0; lane < requests.size(); ++lane) {
        if (lane == 3) { continue; }
        const auto result = requests[lane].wait(nullptr, cancel);
        failures += check(result.finish_reason == FinishReason::OutputLimit &&
                              result.generated_token_ids.size() == 128,
                          "full batch request did not reach its output limit");
        failures +=
            check(result.generated_token_ids.size() == 128 &&
                      std::all_of(result.generated_token_ids.begin() + 1,
                                  result.generated_token_ids.end(),
                                  [&](TokenId token) {
                                      return token == static_cast<TokenId>(1000 + 'a' + lane);
                                  }),
                  "compact row output was attributed to the wrong lane");
    }
    failures +=
        check(program.full_members == std::vector<std::uint32_t>({0, 1, 2, 3, 4, 5, 6, 7}) &&
                  program.compact_seen,
              "full C8 round or lane7 after middle-lane cancellation was lost");
    return failures;
}

int run_prefill_interruptions(Frontend& frontend) {
    int failures = 0;
    for (const unsigned scenario : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
        PrefillScheduleProgram program;
        program.windowed        = scenario == 8;
        program.lanes           = scenario == 1 ? 1 : 2;
        program.suspendable     = scenario != 2;
        program.short_fits      = scenario != 3;
        program.gate_small      = scenario == 5 || scenario == 6;
        program.fail_short_once = scenario == 7;
        ProbeLoaded loaded{frontend};
        PrefillScheduleProbe instance{&program, &loaded, {}};
        auto engine                = engine_options();
        engine.max_concurrency     = program.lanes;
        engine.max_context         = 32768;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<PrefillScheduleProbe> executor(instance, engine);
        auto submit = [&](std::uint32_t tokens) {
            auto prompt        = frontend.prepare_tokens(std::vector<TokenId>(tokens, 1000 + 'a'));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.requested_output_tokens = 1;
            options.stop.token_ids                    = {kCallerStop};
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::TerminalOnly);
        };
        auto large       = submit(16384);
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!program.first_chunk_entered.load()) {
            if (std::chrono::steady_clock::now() >= limit) {
                throw std::runtime_error("large scripted prefill never started");
            }
            std::this_thread::yield();
        }
        auto earlier_large = scenario == 4 ? submit(8192) : decltype(large){};
        auto small1        = submit(128);
        auto small2        = submit(256);
        program.release_first_chunk.store(true);
        const ninfer::CancellationView cancel;
        if (scenario == 5 || scenario == 6) {
            const auto small_limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!program.small_entered.load()) {
                if (std::chrono::steady_clock::now() >= small_limit) {
                    throw std::runtime_error("small prefill never reached cancellation gate");
                }
                std::this_thread::yield();
            }
            if (scenario == 5) {
                large = {};
            } else {
                small1 = {};
            }
            program.release_small.store(true);
        }
        if (scenario != 5) { (void)large.wait(nullptr, cancel); }
        if (scenario == 4) { (void)earlier_large.wait(nullptr, cancel); }
        if (scenario == 7) {
            bool failed = false;
            try {
                (void)small1.wait(nullptr, cancel);
            } catch (const ninfer::RequestError& error) {
                failed = error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
            }
            failures += check(failed, "short initialization error was not request-local");
        } else if (scenario != 6) {
            (void)small1.wait(nullptr, cancel);
        }
        (void)small2.wait(nullptr, cancel);
        const auto first_small = std::find(program.units.begin(), program.units.end(), "start:128");
        const auto second_small =
            std::find(program.units.begin(), program.units.end(), "start:256");
        const auto resumed = std::find(program.units.begin(), program.units.end(), "resume:16384");
        if (scenario == 0 || scenario == 6 || scenario == 7 || scenario == 8) {
            failures += check(
                first_small < resumed && resumed < second_small && resumed + 1 < second_small &&
                    *(resumed + 1) == "chunk:16384:exclusive" &&
                    std::count(program.units.begin(), program.units.end(), "resume:16384") == 2,
                "short prefill interruption violated FIFO or chunk progress");
        } else if (scenario == 5) {
            failures +=
                check(resumed == program.units.end(), "cancelled suspended owner was resumed");
        } else {
            failures +=
                check(resumed == program.units.end() && first_small > program.units.begin() + 4,
                      "prefill interrupted without a usable spare lane/resource lease");
        }
    }
    return failures;
}

int run_prefill_thresholds(Frontend& frontend) {
    int failures = 0;
    for (const auto tokens : {8191U, 8192U, 8193U}) {
        PrefillScheduleProgram program;
        program.release_first_chunk.store(true);
        program.gate_decode = true;
        ProbeLoaded loaded{frontend};
        PrefillScheduleProbe instance{&program, &loaded, {}};
        auto options                = engine_options();
        options.max_concurrency     = 2;
        options.max_context         = 32768;
        options.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<PrefillScheduleProbe> executor(instance, options);
        auto submit = [&](std::uint32_t count, std::uint32_t outputs) {
            auto prompt        = frontend.prepare_tokens(std::vector<TokenId>(count, 1000 + 'a'));
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions request;
            request.execution.allow_prefix_reuse      = false;
            request.execution.requested_output_tokens = outputs;
            request.stop.token_ids                    = {kCallerStop};
            return executor.submit(std::move(prompt), summary, 0.0, std::move(request),
                                   ninfer::OutputDelivery::TerminalOnly);
        };
        auto background     = submit(16, 4096);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!program.decode_entered.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("scripted decode background never became ready");
            }
            std::this_thread::yield();
        }
        auto owner = submit(tokens, 1);
        program.release_decode.store(true);
        const ninfer::CancellationView cancel;
        (void)owner.wait(nullptr, cancel);
        (void)background.wait(nullptr, cancel);
        const auto start = std::find(program.units.begin(), program.units.end(),
                                     "start:" + std::to_string(tokens));
        if (start == program.units.end()) {
            throw std::runtime_error("scripted owner was not executed");
        }
        const std::string label = "chunk:" + std::to_string(tokens);
        std::vector<std::size_t> owner_chunks;
        for (auto it = start; it != program.units.end(); ++it) {
            if (it->starts_with(label + ":")) {
                owner_chunks.push_back(static_cast<std::size_t>(it - program.units.begin()));
            }
        }
        const auto expected_chunks = tokens == 8193 ? 3U : 2U;
        if (owner_chunks.size() != expected_chunks) {
            throw std::runtime_error("scripted owner had the wrong chunk count");
        }
        const auto shared =
            std::count(program.units.begin(), program.units.end(), label + ":shared");
        const auto final_chunk =
            program.units.begin() + static_cast<std::ptrdiff_t>(owner_chunks.back());
        failures += check(shared == (tokens == 8191 ? 1 : 2) &&
                              std::count(start, final_chunk, "decode") == (tokens == 8191 ? 0 : 12),
                          "4095/4096/4097 remaining-token pacing or tail yield suppression failed");
    }
    return failures;
}

// A tiny exact cache geometry exercises real executor pressure transitions without GPU state.
// Each request starts with three output tokens funded, has a six-token lifetime, and requires
// seven pages to finish. Two initial four-page reservations fit an eight-page shared pool;
// neither can grow beside its peer. The per-lane cursor represents committed continuation.
class ChunkedProgram : public ProbeProgram {
public:
    enum class HoldPhase { None, Capture, Restore, Parked };
    HoldPhase hold_phase = HoldPhase::None;
    enum class MetricHold { None, Extension, SingleBlocked };
    MetricHold metric_hold = MetricHold::None;

    void release_held_decode() {
        std::lock_guard lock(phase_mutex_);
        release_transfer.store(true);
        phase_cv_.notify_all();
    }

    bool fail_held_transfer     = false;
    std::uint32_t unsafe_aborts = 0;
    std::atomic<bool> release_transfer{false};
    std::array<std::size_t, 32> live_credits{};
    std::array<std::uint32_t, 32> credit_releases{};
    std::uint32_t discarded_images           = 0;
    std::uint32_t credit_releases_at_capture = 0;

    [[nodiscard]] bool wait_for_held_phase() {
        std::unique_lock lock(phase_mutex_);
        return phase_cv_.wait_until(lock,
                                    std::chrono::steady_clock::now() + std::chrono::seconds(5),
                                    [&] { return phase_entered_; });
    }

    [[nodiscard]] bool no_live_image() const { return !saved_; }

    bool thinking_budget_probe             = false;
    bool c8_admission_probe                = false;
    bool require_peer                      = false;
    bool wait_for_queue                    = false;
    std::uint32_t physical_concurrency     = 2;
    std::uint32_t physical_pages           = 8;
    std::uint32_t initial_window           = 3;
    bool allow_shrink                      = false;
    std::uint32_t shrinks                  = 0;
    std::uint32_t peak_residents           = 0;
    bool resumed_on_different_lane         = false;
    std::size_t ram_credit_ceiling         = 0;
    std::size_t total_ram_credit_ceiling   = 0;
    std::uint64_t single_token_window_seed = 0;
    bool replacement_started_beside_peer   = false;
    std::array<std::uint32_t, 32> credit_attempts{};
    std::atomic<bool> peer_submitted{false};
    std::uint32_t starts                   = 0;
    std::uint32_t suspensions              = 0;
    std::uint32_t resumes                  = 0;
    std::uint32_t extensions               = 0;
    std::uint32_t blocked_single_rounds    = 0;
    std::uint32_t paired_rounds            = 0;
    std::uint32_t credit_rejections        = 0;
    std::uint32_t smallest_admitted_window = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] bool chunked_execution_enabled() const noexcept override { return true; }

    [[nodiscard]] AdmissionResources admission_capacity() const noexcept override {
        return {physical_concurrency, physical_pages, 0};
    }

    [[nodiscard]] bool kv_copies_ready() const override { return true; }

    [[nodiscard]] ProbePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const ninfer::runtime::ResolvedExecutionOptions& options) override {
        auto plan                           = fitted_plan();
        plan.execution                      = options;
        plan.fields.prompt_tokens           = prompt.summary().prompt_tokens;
        plan.fields.effective_output_tokens = options.requested_output_tokens;
        plan.fields.window_output_tokens =
            std::min(options.requested_output_tokens,
                     options.reservation_window_tokens == 0
                         ? initial_window
                         : std::min(initial_window, options.reservation_window_tokens));
        if (single_token_window_seed != 0 && options.sampling.seed == single_token_window_seed)
            plan.fields.window_output_tokens = 1;
        plan.fields.admission           = {1, plan.fields.window_output_tokens + 1, 0};
        plan.fields.lifetime_admission  = {1, options.requested_output_tokens + 1, 0};
        plan.fields.suspension_bytes    = plan.fields.window_output_tokens + 1;
        plan.fields.service_work_quanta = options.requested_output_tokens + 1;
        return plan;
    }

    [[nodiscard]] ProbePlan plan_request_for_lane(std::uint32_t, const PreparedPrompt&,
                                                  const ProbePlan& base) override {
        return base;
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const ProbePlan& plan) const noexcept override {
        // Stagger the initial physical bind by one decode advance, so one blocked request
        // must coexist with a still-runnable peer instead of exhausting both windows together.
        if (require_peer && starts == 1 && cursor_[0] + cursor_[1] < 2) { return false; }
        return (pages_[lane] == 0 || retained_[lane]) &&
               used_pages() - pages_[lane] + plan.fields.admission.main_kv_pages <= physical_pages;
    }

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion, PrefillPace,
                                                       const OutputSession* output) override {
        ++starts;
        if (starts == 3) {
            for (std::size_t at = 0; at < pages_.size(); ++at)
                replacement_started_beside_peer = replacement_started_beside_peer ||
                                                  (at != lane && pages_[at] != 0 && !retained_[at]);
        }
        smallest_admitted_window =
            std::min(smallest_admitted_window, plan.fields.window_output_tokens);
        retained_[lane]              = false;
        pages_[lane]                 = plan.fields.admission.main_kv_pages;
        std::uint32_t resident_count = 0;
        for (std::size_t at = 0; at < pages_.size(); ++at) {
            if (pages_[at] != 0 && !retained_[at]) ++resident_count;
        }
        peak_residents  = std::max(peak_residents, resident_count);
        cursor_[lane]   = 1;
        lifetime_[lane] = plan.fields.effective_output_tokens;
        output_[lane]   = output;
        answered_[lane] = false;
        tokens_[0]      = thinking_budget_probe ? 0 : sequence_[0];
        PrefillStepResult result;
        result.complete                = true;
        result.processed_prompt_tokens = prompt.summary().prompt_tokens;
        result.round.tokens            = std::span<const TokenId>(tokens_.data(), 1);
        if (thinking_budget_probe) result.round.logprobs = logprobs();
        result.summary.prompt_tokens = prompt.summary().prompt_tokens;
        return result;
    }

    [[nodiscard]] ninfer::runtime::ExecutionWindow plan_window(std::uint32_t,
                                                               std::uint32_t end) const override {
        return {{1, end + 1, 0}, end, end + 1};
    }

    [[nodiscard]] ninfer::runtime::ExecutionWindow
    plan_suspended_window(std::uint64_t, std::uint32_t end) const override {
        return plan_window(0, end);
    }

    [[nodiscard]] bool grant_window(std::uint32_t lane,
                                    const ninfer::runtime::ExecutionWindow& window) override {
        // Keep the first completed window blocked long enough to admit its already queued
        // peer. This reproduces the pressure geometry independently of host scheduling speed.
        if (require_peer && starts < 2) { return false; }
        if (used_pages() - pages_[lane] + window.admission.main_kv_pages > physical_pages) {
            return false;
        }
        pages_[lane] = window.admission.main_kv_pages;
        ++extensions;
        return true;
    }

    [[nodiscard]] AdmissionResources completion_resources(const ProbePlan& base,
                                                          std::uint32_t) const override {
        auto resources = base.fields.lifetime_admission;
        return resources;
    }

    [[nodiscard]] bool shrink_window(std::uint32_t lane,
                                     const ninfer::runtime::ExecutionWindow& window) override {
        if (!allow_shrink) return false;
        if (window.output_tokens <= cursor_[lane] ||
            window.admission.main_kv_pages < cursor_[lane] + 1 ||
            window.admission.main_kv_pages >= pages_[lane]) {
            throw std::logic_error(
                "chunked probe shrink discarded current state or useful execution");
        }
        pages_[lane] = window.admission.main_kv_pages;
        ++shrinks;
        return true;
    }

    [[nodiscard]] bool reserve_suspension_credit(std::uint64_t owner, std::size_t bytes) override {
        ++credit_attempts.at(owner);
        std::size_t total = bytes;
        for (std::size_t at = 0; at < live_credits.size(); ++at)
            if (at != owner) total += live_credits[at];
        if ((ram_credit_ceiling != 0 && bytes > ram_credit_ceiling) ||
            (total_ram_credit_ceiling != 0 && total > total_ram_credit_ceiling)) {
            ++credit_rejections;
            return false;
        }
        live_credits.at(owner) = bytes;
        return true;
    }

    void release_suspension_credit(std::uint64_t owner) override {
        if (saved_ && saved_->id == owner) {
            throw std::logic_error(
                "chunked probe released RAM while its live image still owned it");
        }
        if (live_credits.at(owner) != 0) {
            live_credits[owner] = 0;
            ++credit_releases.at(owner);
        }
    }

    [[nodiscard]] bool suspension_ready(std::uint64_t) const override {
        if (fail_held_transfer && hold_phase == HoldPhase::Capture && release_transfer.load()) {
            throw std::runtime_error("probe shared transfer failure");
        }
        return hold_phase != HoldPhase::Capture || release_transfer.load();
    }

    [[nodiscard]] bool resume_ready(std::uint64_t) const override {
        if (fail_held_transfer && hold_phase == HoldPhase::Restore && release_transfer.load()) {
            throw std::runtime_error("probe shared transfer failure");
        }
        return hold_phase != HoldPhase::Restore || release_transfer.load();
    }

    [[nodiscard]] bool start_suspend(std::uint32_t lane, std::uint64_t id) override {
        if (saved_) { throw std::logic_error("chunked probe overwrote a live suspension"); }
        if (live_credits.at(id) == 0) {
            throw std::logic_error("chunked probe captured without secured RAM");
        }
        credit_releases_at_capture = credit_releases.at(id);
        saved_ = Saved{id, cursor_[lane], lane, lifetime_[lane], output_[lane], answered_[lane]};
        if (hold_phase == HoldPhase::Capture) { signal_held_phase(); }
        return true;
    }

    void finish_suspend(std::uint32_t lane, std::uint64_t id) override {
        if (!saved_ || saved_->id != id) {
            throw std::logic_error("chunked probe finished the wrong suspension");
        }
        pages_[lane]  = 0;
        cursor_[lane] = 0;
        saved_->phase = Saved::Phase::Parked;
        ++suspensions;
    }

    [[nodiscard]] bool start_resume(std::uint32_t lane, std::uint64_t id,
                                    const ninfer::runtime::ExecutionWindow& window) override {
        if (!saved_ || saved_->id != id || pages_[lane] != 0 ||
            used_pages() + window.admission.main_kv_pages > physical_pages) {
            return false;
        }
        if (hold_phase == HoldPhase::Parked && !release_transfer.load()) {
            signal_held_phase();
            return false;
        }
        if (fail_held_transfer && hold_phase == HoldPhase::Parked) {
            throw std::runtime_error("probe shared transfer failure");
        }
        cursor_[lane]             = saved_->cursor;
        lifetime_[lane]           = saved_->lifetime;
        output_[lane]             = saved_->output;
        answered_[lane]           = saved_->answered;
        resumed_on_different_lane = resumed_on_different_lane || lane != saved_->lane;
        pages_[lane]              = window.admission.main_kv_pages;
        saved_->phase             = Saved::Phase::Restoring;
        saved_->restore_lane      = lane;
        if (hold_phase == HoldPhase::Restore) { signal_held_phase(); }
        return true;
    }

    void finish_resume(std::uint32_t, std::uint64_t id) override {
        if (!saved_ || saved_->id != id) {
            throw std::logic_error("chunked probe resumed the wrong identity");
        }
        saved_.reset();
        ++resumes;
    }

    void discard_suspended(std::uint64_t id) override {
        if (saved_ && saved_->id == id) {
            if (fail_held_transfer) { throw std::runtime_error("probe discard fence failure"); }
            if (saved_->phase == Saved::Phase::Restoring) {
                pages_[saved_->restore_lane]  = 0;
                cursor_[saved_->restore_lane] = 0;
            }
            saved_.reset();
            ++discarded_images;
        }
    }

    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept override {
        return retained_[lane];
    }

    void evict_retained_lane(std::uint32_t lane) noexcept override {
        pages_[lane]    = 0;
        cursor_[lane]   = 0;
        retained_[lane] = false;
    }

    void resolve_pending_batch(std::span<const std::uint32_t> lanes, std::span<const std::uint32_t>,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled,
                               std::span<const std::uint8_t>) override {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (terminal[row] || cancelled[row]) { retained_[lanes[row]] = true; }
        }
    }

    void abort_lane(std::uint32_t lane) noexcept override {
        if (saved_ && ((saved_->phase == Saved::Phase::Restoring && saved_->restore_lane == lane) ||
                       (saved_->phase == Saved::Phase::Capturing && saved_->lane == lane))) {
            ++unsafe_aborts;
        }
        pages_[lane]    = 0;
        cursor_[lane]   = 0;
        retained_[lane] = false;
    }

    [[nodiscard]] BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const RoundBudget> budgets) override {
        if ((require_peer || wait_for_queue) && starts == 1) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!peer_submitted.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("chunked probe peer was not submitted");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (c8_admission_probe && starts == 8 && !metric_hold_entered_) {
            std::unique_lock lock(phase_mutex_);
            metric_hold_entered_ = true;
            phase_entered_       = true;
            phase_cv_.notify_all();
            if (!phase_cv_.wait_for(lock, std::chrono::seconds(5),
                                    [&] { return release_transfer.load(); })) {
                throw std::runtime_error("C8 window observation did not release decode");
            }
        }
        if (lanes.size() == 2) { ++paired_rounds; }
        if (require_peer && starts == 2 && lanes.size() == 1 && suspensions == 0) {
            ++blocked_single_rounds;
        }
        if (!metric_hold_entered_ &&
            ((metric_hold == MetricHold::Extension && extensions != 0) ||
             (metric_hold == MetricHold::SingleBlocked && blocked_single_rounds != 0))) {
            std::unique_lock lock(phase_mutex_);
            metric_hold_entered_ = true;
            phase_entered_       = true;
            phase_cv_.notify_all();
            if (!phase_cv_.wait_until(lock,
                                      std::chrono::steady_clock::now() + std::chrono::seconds(5),
                                      [&] { return release_transfer.load(); })) {
                throw std::runtime_error("chunked metric observation did not release decode");
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const auto lane = lanes[row];
            if (budgets[row].generated_tokens_remaining == 0 || cursor_[lane] >= lifetime_[lane] ||
                (!c8_admission_probe &&
                 (!thinking_budget_probe && cursor_[lane] >= sequence_.size()))) {
                throw std::logic_error("chunked probe decoded outside funded lifetime");
            }
            if (thinking_budget_probe) {
                if (output_[lane]->in_reasoning()) {
                    const std::array<TokenId, 1> root{0};
                    const std::array<std::int32_t, 1> parents{-1};
                    std::vector<std::uint32_t> mask((248077 + 31) / 32);
                    output_[lane]->fill_token_masks(root, parents, mask);
                    const bool reason_allowed = (mask[0] & 1U) != 0;
                    const bool close_allowed  = (mask[248069 / 32] & (1U << (248069 % 32))) != 0;
                    if (!reason_allowed && !close_allowed)
                        throw std::logic_error("joint probe lost reasoning and forced close");
                    tokens_[row] = reason_allowed ? 0 : 248069;
                } else if (!answered_[lane]) {
                    tokens_[row]    = 15;
                    answered_[lane] = true;
                } else {
                    tokens_[row] = 6;
                }
                ++cursor_[lane];
            } else {
                tokens_[row] = sequence_[cursor_[lane]++ % sequence_.size()];
            }
            counts_[row] = 1;
        }
        return {.tokens     = std::span<const TokenId>(tokens_.data(), lanes.size()),
                .logprobs   = thinking_budget_probe ? logprobs() : ninfer::runtime::RoundLogprobs{},
                .row_counts = std::span<const std::int32_t>(counts_.data(), lanes.size()),
                .row_stride = 1};
    }

    [[nodiscard]] static std::vector<TokenId> expected() {
        return {sequence_.begin(), sequence_.end()};
    }

private:
    struct Saved {
        std::uint64_t id;
        std::size_t cursor;
        std::uint32_t lane;
        std::uint32_t lifetime;
        const OutputSession* output;
        bool answered;
        enum class Phase { Capturing, Parked, Restoring } phase = Phase::Capturing;
        std::uint32_t restore_lane                              = 0;
    };

    void signal_held_phase() {
        std::lock_guard lock(phase_mutex_);
        phase_entered_ = true;
        phase_cv_.notify_one();
    }

    std::mutex phase_mutex_;
    std::condition_variable phase_cv_;
    bool phase_entered_       = false;
    bool metric_hold_entered_ = false;

    ninfer::runtime::RoundLogprobs logprobs() {
        logprob_values_.fill(-0.25F);
        top_values_.fill(-0.5F);
        return {logprob_values_, top_ids_, top_values_};
    }

    [[nodiscard]] std::uint32_t used_pages() const noexcept {
        return std::accumulate(pages_.begin(), pages_.end(), 0U);
    }

    static constexpr std::array<TokenId, 6> sequence_{15, 22, 23, 15, 22, 23};
    std::array<std::uint32_t, ninfer::kMaximumConcurrency> pages_{};
    std::array<bool, ninfer::kMaximumConcurrency> retained_{};
    std::array<std::size_t, ninfer::kMaximumConcurrency> cursor_{};
    std::array<std::uint32_t, ninfer::kMaximumConcurrency> lifetime_{};
    std::array<const OutputSession*, ninfer::kMaximumConcurrency> output_{};
    std::array<bool, ninfer::kMaximumConcurrency> answered_{};
    std::array<float, ninfer::kMaximumConcurrency> logprob_values_{};
    std::array<TokenId, std::size_t{ninfer::kMaximumConcurrency} * ninfer::kMaximumTopLogprobs>
        top_ids_{};
    std::array<float, std::size_t{ninfer::kMaximumConcurrency} * ninfer::kMaximumTopLogprobs>
        top_values_{};
    std::optional<Saved> saved_;
    std::array<TokenId, ninfer::kMaximumConcurrency> tokens_{};
    std::array<std::int32_t, ninfer::kMaximumConcurrency> counts_{};
};

struct ChunkedPackage {
    using Program         = ChunkedProgram;
    using RequestBasePlan = ProbePlan;
    using RequestPlan     = ProbePlan;
};

struct ChunkedProbe {
    using Package           = ChunkedPackage;
    ChunkedProgram* program = nullptr;
    ProbeLoaded* loaded     = nullptr;
    ProbeMemory request_memory;
};

// Models the shared RAM arena's FIFO eviction during live-credit allocation. The first
// request owns the completion cohort; its queued peer must fund credit after choosing reuse.
class AdmissionEvictionProgram final : public ChunkedProgram {
public:
    enum class Eviction { All, Longest, FailedReservation };

    explicit AdmissionEvictionProgram(Eviction eviction) : eviction_(eviction) {
        wait_for_queue = true;
    }

    std::uint32_t replans          = 0;
    std::uint32_t restores         = 0;
    std::uint32_t funding_attempts = 0;
    std::uint32_t admitted_window  = 0;
    bool evicted                   = false;

    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept override { return version_; }

    [[nodiscard]] ProbePlan plan_ram_reuse(const PreparedPrompt&, const ProbePlan& base) override {
        ++replans;
        auto plan = base;
        if (!available_) { return plan; }
        plan.fields.ram_entry_id           = available_;
        plan.fields.reuse_source           = PrefixReuseSource::HostRam;
        plan.fields.reusable_prompt_tokens = available_ == 1 ? 32 : 16;
        // A shorter prefix can retain more checkpoint heads and require larger backing.
        plan.fields.suspension_bytes =
            available_ == 1 && eviction_ != Eviction::FailedReservation ? 4 : 6;
        return plan;
    }

    [[nodiscard]] bool reserve_suspension_credit(std::uint64_t owner, std::size_t bytes) override {
        if (owner == 2) {
            ++funding_attempts;
            if (!evicted) {
                if (claimed_) { throw std::logic_error("eviction probe source is claimed"); }
                evicted    = true;
                available_ = eviction_ == Eviction::Longest ? 2 : 0;
                ++version_;
                if (eviction_ == Eviction::FailedReservation) { return false; }
            }
        }
        return ChunkedProgram::reserve_suspension_credit(owner, bytes);
    }

    void claim_ram_entry(std::uint64_t id) override {
        if (id != available_ || id == 0) {
            throw std::logic_error("RAM cache entry id is unknown");
        }
        if (claimed_) { throw std::logic_error("eviction probe source is already claimed"); }
        claimed_ = true;
    }

    void release_ram_entry(std::uint64_t id) override {
        if (id != available_ || !claimed_) {
            throw std::logic_error("eviction probe released an unowned source");
        }
        claimed_ = false;
    }

    void restore_ram_entry(std::uint32_t, std::uint64_t id, const ProbePlan&) override {
        if (id != available_ || !claimed_) {
            throw std::logic_error("eviction probe restored an unowned source");
        }
        ++restores;
    }

    void finish_ram_restore(std::uint64_t id, ninfer::runtime::CacheReadIntent intent) override {
        release_ram_entry(id);
        if (intent == ninfer::runtime::CacheReadIntent::Consume) {
            available_ = 0;
            ++version_;
        }
    }

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion region,
                                                       PrefillPace pace,
                                                       const OutputSession* output) override {
        if (starts == 1) {
            if (live_credits.at(2) < plan.fields.suspension_bytes) {
                throw std::logic_error("replanned admission has insufficient RAM backing");
            }
            admitted_window = plan.fields.window_output_tokens;
        }
        return ChunkedProgram::start_prefill_lane(lane, std::move(prompt), plan, region, pace,
                                                  output);
    }

private:
    Eviction eviction_;
    std::uint64_t version_   = 1;
    std::uint64_t available_ = 1;
    bool claimed_            = false;
};

struct AdmissionEvictionPackage : ProbePackage {
    using Program = AdmissionEvictionProgram;
};

struct AdmissionEvictionProbe {
    using Package                     = AdmissionEvictionPackage;
    AdmissionEvictionProgram* program = nullptr;
    ProbeLoaded* loaded               = nullptr;
    ProbeMemory request_memory;
};

int run_admission_credit_eviction(Frontend& frontend) {
    int failures = 0;
    for (const auto eviction :
         {AdmissionEvictionProgram::Eviction::All, AdmissionEvictionProgram::Eviction::Longest,
          AdmissionEvictionProgram::Eviction::FailedReservation}) {
        for (const bool cache_write : {false, true}) {
            AdmissionEvictionProgram program(eviction);
            ProbeLoaded loaded{frontend};
            AdmissionEvictionProbe instance{.program = &program, .loaded = &loaded};
            auto engine                = engine_options();
            engine.max_concurrency     = 2;
            engine.generation_recovery = false;
            ninfer::runtime::ConcurrentExecutor<AdmissionEvictionProbe> executor(instance, engine);
            const auto submit = [&](bool reuse) {
                auto input                    = thinking_input();
                input.options.enable_thinking = false;
                auto prompt                   = frontend.prepare(std::move(input));
                const auto summary            = prompt.summary();
                ninfer::runtime::ResolvedRequestOptions options;
                options.execution.allow_prefix_reuse      = reuse;
                options.execution.cache_write             = cache_write;
                options.execution.requested_output_tokens = 6;
                options.output.raw                        = true;
                return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                       ninfer::OutputDelivery::TerminalOnly);
            };
            auto first  = submit(false);
            auto second = submit(true);
            program.peer_submitted.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            ninfer::CancellationView cancel(
                [deadline] { return std::chrono::steady_clock::now() >= deadline; });
            const auto a = first.wait(nullptr, cancel);
            const auto b = second.wait(nullptr, cancel);
            if (!program.evicted || program.replans < 2 || program.funding_attempts < 2 ||
                program.admitted_window != 3) {
                std::cerr << "admission eviction case=" << static_cast<int>(eviction)
                          << " cache_write=" << cache_write << " evicted=" << program.evicted
                          << " replans=" << program.replans
                          << " funding=" << program.funding_attempts
                          << " window=" << program.admitted_window
                          << " restores=" << program.restores << '\n';
            }
            failures += check(a.generated_token_ids == ChunkedProgram::expected() &&
                                  b.generated_token_ids == ChunkedProgram::expected(),
                              "RAM credit eviction lost or replayed a continuation");
            failures +=
                check(program.evicted && program.replans >= 2 && program.funding_attempts >= 2 &&
                          program.admitted_window == 3 &&
                          program.restores ==
                              (eviction == AdmissionEvictionProgram::Eviction::Longest ? 1U : 0U),
                      "RAM credit eviction did not reselect and fund the surviving plan");
            auto subsequent = submit(false);
            failures += check(subsequent.wait(nullptr, cancel).generated_token_ids ==
                                  ChunkedProgram::expected(),
                              "RAM credit eviction left the Engine unavailable");
        }
    }
    return failures;
}

int run_eight_32k_windows(Frontend& frontend) {
    ChunkedProgram program;
    program.c8_admission_probe   = true;
    program.wait_for_queue       = true;
    program.physical_concurrency = 8;
    program.physical_pages       = 8 * (8192 + 1);
    program.initial_window       = 8192;
    ProbeLoaded loaded{frontend};
    ChunkedProbe instance{.program = &program, .loaded = &loaded};
    auto engine                = engine_options();
    engine.max_concurrency     = 8;
    engine.max_context         = 65536;
    engine.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
    const auto submit = [&] {
        auto prompt        = frontend.prepare_tokens({15});
        const auto summary = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.requested_output_tokens = 32768;
        options.execution.allow_prefix_reuse      = false;
        options.execution.cache_write             = false;
        options.output.raw                        = true;
        return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                               ninfer::OutputDelivery::TerminalOnly);
    };
    std::array<decltype(submit()), 8> requests;
    for (auto& request : requests) { request = submit(); }
    program.peer_submitted.store(true);
    const bool reached = program.wait_for_held_phase();
    int failures = check(reached && program.starts == 8 && program.peak_residents == 8 &&
                             program.smallest_admitted_window == 8192 && program.suspensions == 0,
                         "eight 32K limits charged lifetime output instead of 8K windows");
    for (auto& request : requests) { request = {}; }
    program.release_held_decode();
    return failures;
}

int run_chunked_continuation(Frontend& frontend, bool paired, bool initial_ram_pressure = false) {
    ChunkedProgram program;
    program.require_peer       = paired;
    program.ram_credit_ceiling = initial_ram_pressure ? 3 : 0;
    ProbeLoaded loaded{frontend};
    ChunkedProbe instance{.program = &program, .loaded = &loaded};
    auto engine                = engine_options();
    engine.max_concurrency     = 2;
    engine.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
    const auto submit = [&] {
        auto input                    = thinking_input();
        input.options.enable_thinking = false;
        auto prompt                   = frontend.prepare(std::move(input));
        const auto summary            = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.allow_prefix_reuse      = false;
        options.execution.cache_write             = false;
        options.execution.requested_output_tokens = 6;
        options.output.raw                        = true;
        return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                               ninfer::OutputDelivery::Streaming);
    };
    auto first = submit();
    std::optional<decltype(first)> second;
    if (paired) { second.emplace(submit()); }
    program.peer_submitted.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancellation(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    const auto a = first.wait(nullptr, cancellation);
    int failures = check(a.finish_reason == FinishReason::OutputLimit &&
                             a.generated_token_ids == ChunkedProgram::expected(),
                         "chunked: reservation window truncated or replayed first output");
    if (second) {
        const auto b = second->wait(nullptr, cancellation);
        if (b.finish_reason != FinishReason::OutputLimit ||
            b.generated_token_ids != ChunkedProgram::expected() || program.suspensions != 1 ||
            program.resumes != 1 || program.blocked_single_rounds == 0) {
            std::cerr << "chunked diagnostic ram_pressure=" << initial_ram_pressure
                      << " starts=" << program.starts << " suspended=" << program.suspensions
                      << " resumed=" << program.resumes << " extensions=" << program.extensions
                      << " paired=" << program.paired_rounds
                      << " blocked_peer=" << program.blocked_single_rounds
                      << " second_tokens=" << b.generated_token_ids.size()
                      << " second_finish=" << static_cast<int>(b.finish_reason) << '\n';
        }
        failures += check(b.finish_reason == FinishReason::OutputLimit &&
                              b.generated_token_ids == ChunkedProgram::expected(),
                          "chunked: resumed continuation lost or replayed output");
        failures += check(program.suspensions == 1 && program.resumes == 1 && program.starts == 2,
                          "chunked: pressure did not suspend/resume directly without prefill");
        failures += check(program.blocked_single_rounds != 0,
                          "chunked: a single blocked lane prevented its peer from advancing");
        if (initial_ram_pressure) {
            failures +=
                check(program.credit_rejections != 0 && program.smallest_admitted_window == 1,
                      "chunked: GPU-fit initial window never shrank to secured RAM fit");
        }
    } else {
        failures += check(program.extensions != 0 && program.starts == 1 &&
                              program.suspensions == 0 && program.resumes == 0,
                          "chunked: solitary growth unnecessarily suspended or repeated prefill");
    }
    return failures;
}

int run_chunked_transfer_failure(Frontend& frontend) {
    int failures = 0;
    for (const auto phase : {ChunkedProgram::HoldPhase::Capture, ChunkedProgram::HoldPhase::Restore,
                             ChunkedProgram::HoldPhase::Parked}) {
        ChunkedProgram program;
        program.require_peer       = true;
        program.hold_phase         = phase;
        program.fail_held_transfer = true;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 2;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        const auto submit = [&] {
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            auto prompt                   = frontend.prepare(std::move(input));
            const auto summary            = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = 6;
            options.output.raw                        = true;
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::Streaming);
        };
        auto peer   = submit();
        auto victim = submit();
        program.peer_submitted.store(true);
        const bool reached = program.wait_for_held_phase();
        program.release_transfer.store(true);
        std::uint32_t published_failures = 0;
        for (auto* request : {&peer, &victim}) {
            try {
                (void)request->wait(nullptr, {});
            } catch (const std::runtime_error& error) {
                if (std::string_view(error.what()) == "probe shared transfer failure") {
                    ++published_failures;
                } else {
                    throw;
                }
            }
        }
        // The peer may finish before a parked/restore fault; the suspended owner must fail.
        failures += check(reached && published_failures >= 1 && program.unsafe_aborts == 0 &&
                              !program.no_live_image(),
                          "failed live cleanup lost primary error or released an unfenced lane");
    }
    return failures;
}

int run_chunked_reasoning_boundary(Frontend& frontend) {
    int failures = 0;
    for (const std::uint32_t cap : {2U, 3U, 4U}) {
        ChunkedProgram program;
        program.require_peer          = true;
        program.thinking_budget_probe = true;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 2;
        engine.generation_recovery = true;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        const auto submit = [&] {
            auto prompt        = frontend.prepare(thinking_input());
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = 7;
            options.output.thinking_token_budget      = cap;
            options.output.top_logprobs               = 1;
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::Streaming);
        };
        auto first  = submit();
        auto second = submit();
        program.peer_submitted.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancellation(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        const auto a = first.wait(nullptr, cancellation);
        const auto b = second.wait(nullptr, cancellation);
        std::vector<TokenId> expected(cap, 0);
        expected.insert(expected.end(), {248069, 15, 6});
        for (const auto* result : {&a, &b}) {
            failures +=
                check(result->generated_token_ids == expected && result->content == "answer",
                      "joint chunk/cap: suspend or boundary replayed/dropped generation");
            failures +=
                check(result->reasoning_limit.used_tokens == cap &&
                          result->reasoning_limit.forced_closes == 1 &&
                          result->reasoning_limit.forced_tokens.size() == 1 &&
                          result->recovery.attempts == 0,
                      "joint chunk/cap: parser budget or intervention state was not preserved");
            if (!result->reasoning_limit.forced_tokens.empty()) {
                const auto& forced = result->reasoning_limit.forced_tokens.front();
                failures +=
                    check(forced.generated_token_index == cap && forced.token == 248069 &&
                              forced.target_logprob == -0.25F,
                          "joint chunk/cap: forced-close provenance changed across suspension");
            }
            failures += check(result->reasoning_logprobs.size() == cap &&
                                  result->content_logprobs.size() == 1 &&
                                  result->content_logprobs.front().token == 15,
                              "joint chunk/cap: framing leaked into channel logprob records");
        }
        failures +=
            check(program.starts == 2 && program.suspensions == 1 && program.resumes == 1 &&
                      program.resumed_on_different_lane && program.blocked_single_rounds != 0,
                  "joint chunk/cap: did not preserve live continuation through a different "
                  "physical lane");
    }
    return failures;
}

int run_chunked_lifecycle(Frontend& frontend) {
    int failures = 0;
    for (const auto phase : {ChunkedProgram::HoldPhase::Capture, ChunkedProgram::HoldPhase::Restore,
                             ChunkedProgram::HoldPhase::Parked}) {
        ChunkedProgram program;
        program.require_peer = true;
        program.hold_phase   = phase;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 2;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        const auto submit = [&](std::uint32_t output_tokens) {
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            auto prompt                   = frontend.prepare(std::move(input));
            const auto summary            = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = output_tokens;
            options.output.raw                        = true;
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::Streaming);
        };
        auto peer = submit(6);
        std::optional<decltype(peer)> victim(submit(6));
        program.peer_submitted.store(true);
        const bool reached = program.wait_for_held_phase();
        // Submission destruction synchronously publishes disconnect cancellation. The phase
        // gate remains closed until that store has happened; no timing sleep races a fence.
        victim.reset();
        // Parked work has no in-flight transfer to release. Keep restoration unavailable
        // until the scheduler observes cancellation, including retries already in progress.
        if (phase != ChunkedProgram::HoldPhase::Parked) { program.release_transfer.store(true); }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancellation(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        const auto peer_result  = peer.wait(nullptr, cancellation);
        auto after              = submit(2);
        const auto after_result = after.wait(nullptr, cancellation);
        failures +=
            check(reached && peer_result.generated_token_ids == ChunkedProgram::expected() &&
                      peer_result.finish_reason == FinishReason::OutputLimit &&
                      after_result.generated_token_ids == std::vector<TokenId>({15, 22}),
                  "chunked cancellation: transfer gate or surviving/new request lost progress");
        if (program.starts != 3 || !program.no_live_image() ||
            program.credit_releases[2] != program.credit_releases_at_capture + 1 ||
            std::any_of(program.live_credits.begin(), program.live_credits.end(),
                        [](auto n) { return n != 0; })) {
            std::cerr << "lifecycle diagnostic phase=" << static_cast<int>(phase)
                      << " starts=" << program.starts << " image=" << !program.no_live_image()
                      << " release2=" << program.credit_releases[2] << " credits=";
            for (auto credit : program.live_credits) { std::cerr << credit << ','; }
            std::cerr << '\n';
        }
        failures +=
            check(program.starts == 3 && program.no_live_image() &&
                      std::all_of(program.live_credits.begin(), program.live_credits.end(),
                                  [](auto bytes) { return bytes == 0; }) &&
                      program.credit_releases[2] == program.credit_releases_at_capture + 1,
                  "chunked cancellation: live image/credit ownership leaked or released twice");
        const auto stats = executor.runtime_stats();
        failures += check(stats.running_requests == 0 && stats.kv_suspended_requests == 0 &&
                              stats.kv_growth_blocked_requests == 0,
                          "chunked cancellation: cancelled continuation remained admitted");
        if (phase == ChunkedProgram::HoldPhase::Restore) {
            failures += check(
                program.suspensions == 1 && program.resumes == 1 && program.discarded_images == 0,
                "chunked cancellation: restore was not settled exactly once before lane cleanup");
        } else {
            failures += check(
                program.suspensions == 1 && program.resumes == 0 && program.discarded_images == 1,
                "chunked cancellation: captured/parked image was not discarded exactly once");
        }
    }
    return failures;
}

int run_chunked_finite_ram_and_replacement(Frontend& frontend) {
    int failures = 0;
    for (const bool no_credit_fits : {false, true}) {
        ChunkedProgram program;
        program.require_peer       = !no_credit_fits;
        program.ram_credit_ceiling = no_credit_fits ? 1 : 0;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 2;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        const auto submit = [&](std::uint32_t output_tokens) {
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            auto prompt                   = frontend.prepare(std::move(input));
            const auto summary            = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = output_tokens;
            options.output.raw                        = true;
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::Streaming);
        };
        auto first  = submit(6);
        auto second = submit(6);
        std::optional<decltype(first)> replacement;
        if (!no_credit_fits) { replacement.emplace(submit(2)); }
        program.peer_submitted.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancellation(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        const auto a = first.wait(nullptr, cancellation);
        const auto b = second.wait(nullptr, cancellation);
        failures += check(a.generated_token_ids == ChunkedProgram::expected() &&
                              b.generated_token_ids == ChunkedProgram::expected(),
                          "chunked finite RAM: escape drain lost a request or its continuation");
        if (replacement) {
            const auto c = replacement->wait(nullptr, cancellation);
            failures += check(
                c.generated_token_ids == std::vector<TokenId>({15, 22}) && program.starts == 3 &&
                    program.suspensions == 1 && program.resumes == 1,
                "chunked replacement: queued short request did not use pressure-released lane");
        } else {
            failures += check(
                program.starts == 2 && program.credit_rejections != 0 && program.suspensions == 0 &&
                    program.resumes == 0 && program.paired_rounds == 0,
                "chunked finite RAM: unfunded peer was admitted or escape did not drain serially");
        }
        failures += check(program.no_live_image() &&
                              std::all_of(program.live_credits.begin(), program.live_credits.end(),
                                          [](auto bytes) { return bytes == 0; }),
                          "chunked finite RAM: final live image/credit ownership leaked");
    }
    return failures;
}

int run_cohort_replacement_without_spare_ram(Frontend& frontend) {
    ChunkedProgram program;
    program.wait_for_queue           = true;
    program.physical_pages           = 10;
    program.total_ram_credit_ceiling = 2;
    program.single_token_window_seed = 2;
    ProbeLoaded loaded{frontend};
    ChunkedProbe instance{.program = &program, .loaded = &loaded};
    auto engine                = engine_options();
    engine.max_concurrency     = 2;
    engine.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
    const auto submit = [&](std::uint32_t output_tokens, std::uint64_t seed) {
        auto input                    = thinking_input();
        input.options.enable_thinking = false;
        auto prompt                   = frontend.prepare(std::move(input));
        const auto summary            = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.allow_prefix_reuse      = false;
        options.execution.cache_write             = false;
        options.execution.requested_output_tokens = output_tokens;
        options.execution.sampling.seed           = seed;
        options.output.raw                        = true;
        return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                               ninfer::OutputDelivery::Streaming);
    };
    auto first       = submit(6, 1);
    auto blocked     = submit(6, 2);
    auto replacement = submit(2, 3);
    program.peer_submitted.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancellation(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    const auto a = first.wait(nullptr, cancellation);
    const auto b = blocked.wait(nullptr, cancellation);
    const auto c = replacement.wait(nullptr, cancellation);
    return check(a.generated_token_ids == ChunkedProgram::expected() &&
                     b.generated_token_ids == ChunkedProgram::expected() &&
                     c.generated_token_ids == std::vector<TokenId>({15, 22}) &&
                     program.replacement_started_beside_peer && program.suspensions == 1 &&
                     program.resumes == 1 && program.credit_attempts[3] == 0 &&
                     program.no_live_image() &&
                     std::all_of(program.live_credits.begin(), program.live_credits.end(),
                                 [](auto bytes) { return bytes == 0; }),
                 "cohort replacement: spare RAM was required despite reserved completion capacity");
}

int run_chunked_active_window_metrics(Frontend& frontend) {
    int failures = 0;
    for (const bool paired : {false, true}) {
        ChunkedProgram program;
        program.require_peer = paired;
        program.metric_hold  = paired ? ChunkedProgram::MetricHold::SingleBlocked
                                      : ChunkedProgram::MetricHold::Extension;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 2;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        const auto submit = [&] {
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            auto prompt                   = frontend.prepare(std::move(input));
            const auto summary            = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = 6;
            options.output.raw                        = true;
            return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                   ninfer::OutputDelivery::Streaming);
        };
        auto first = submit();
        std::optional<decltype(first)> second;
        if (paired) { second.emplace(submit()); }
        program.peer_submitted.store(true);
        const bool reached            = program.wait_for_held_phase();
        const auto active             = executor.runtime_stats();
        const auto granted_extensions = program.extensions;
        program.release_held_decode();
        failures += check(reached && active.running_requests == (paired ? 2U : 1U),
                          "chunked metrics: observation did not hold an active continuation");
        if (paired) {
            failures += check(
                active.kv_growth_blocked_requests == 1 && active.decode_ready_requests == 1 &&
                    active.kv_live_suspensions == 0,
                "chunked metrics: growth-blocked transition was stale while a peer still decoded");
        } else {
            failures += check(
                active.kv_window_extensions == granted_extensions &&
                    active.kv_window_extensions != 0 && active.kv_growth_blocked_requests == 0,
                "chunked metrics: successful window grant was invisible until completion");
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancellation(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        const auto a = first.wait(nullptr, cancellation);
        failures += check(a.generated_token_ids == ChunkedProgram::expected(),
                          "chunked metrics: observing active counters prevented request progress");
        if (second) {
            const auto b = second->wait(nullptr, cancellation);
            failures += check(b.generated_token_ids == ChunkedProgram::expected(),
                              "chunked metrics: blocked peer did not resume after observation");
        }
    }
    return failures;
}

int run_completion_cohort(Frontend& frontend) {
    int failures = 0;
    // With no allocatable RAM credit, four requests can still run together when their
    // target-produced completion envelopes fit. A smaller pool permits three unbacked
    // members plus one backed peer, using only current windows for actual GPU ownership.
    for (const bool backed_peer : {false, true}) {
        ChunkedProgram program;
        program.physical_concurrency = 4;
        program.physical_pages       = backed_peer ? 25 : 28;
        program.ram_credit_ceiling   = backed_peer ? 4 : 1;
        program.wait_for_queue       = true;
        ProbeLoaded loaded{frontend};
        ChunkedProbe instance{.program = &program, .loaded = &loaded};
        auto engine                = engine_options();
        engine.max_concurrency     = 4;
        engine.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
        std::vector<decltype(executor.submit({}, {}, 0.0, {}, ninfer::OutputDelivery::Streaming))>
            requests;
        for (std::uint32_t i = 0; i < 4; ++i) {
            auto input                    = thinking_input();
            input.options.enable_thinking = false;
            auto prompt                   = frontend.prepare(std::move(input));
            const auto summary            = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions options;
            options.execution.allow_prefix_reuse      = false;
            options.execution.cache_write             = false;
            options.execution.requested_output_tokens = 6;
            options.output.raw                        = true;
            requests.push_back(executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                                               ninfer::OutputDelivery::Streaming));
        }
        program.peer_submitted.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        ninfer::CancellationView cancellation(
            [deadline] { return std::chrono::steady_clock::now() >= deadline; });
        for (auto& request : requests) {
            const auto result = request.wait(nullptr, cancellation);
            failures += check(
                result.generated_token_ids == ChunkedProgram::expected() &&
                    result.finish_reason == FinishReason::OutputLimit,
                "completion cohort: future proof lost progress or changed the output allowance");
        }
        failures +=
            check(program.peak_residents == 4 && program.starts == 4 && program.no_live_image() &&
                      std::all_of(program.live_credits.begin(), program.live_credits.end(),
                                  [](auto n) { return n == 0; }),
                  "completion cohort: unnecessary backing reduced residency or leaked ownership");
        if (!backed_peer) {
            failures += check(
                program.credit_rejections == 0 && program.suspensions == 0,
                "completion cohort: jointly feasible members requested unnecessary RAM backing");
        }
    }
    return failures;
}

int run_unused_window_reclamation(Frontend& frontend) {
    ChunkedProgram program;
    program.physical_pages = 10;
    program.initial_window = 5;
    program.allow_shrink   = true;
    program.wait_for_queue = true;
    ProbeLoaded loaded{frontend};
    ChunkedProbe instance{.program = &program, .loaded = &loaded};
    auto engine                = engine_options();
    engine.max_concurrency     = 2;
    engine.generation_recovery = false;
    ninfer::runtime::ConcurrentExecutor<ChunkedProbe> executor(instance, engine);
    const auto submit = [&] {
        auto input                    = thinking_input();
        input.options.enable_thinking = false;
        auto prompt                   = frontend.prepare(std::move(input));
        const auto summary            = prompt.summary();
        ninfer::runtime::ResolvedRequestOptions options;
        options.execution.allow_prefix_reuse      = false;
        options.execution.cache_write             = false;
        options.execution.requested_output_tokens = 6;
        options.output.raw                        = true;
        return executor.submit(std::move(prompt), summary, 0.0, std::move(options),
                               ninfer::OutputDelivery::Streaming);
    };
    auto first  = submit();
    auto second = submit();
    program.peer_submitted.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancellation(
        [deadline] { return std::chrono::steady_clock::now() >= deadline; });
    const auto a = first.wait(nullptr, cancellation);
    const auto b = second.wait(nullptr, cancellation);
    int failures =
        check(a.generated_token_ids == ChunkedProgram::expected() &&
                  b.generated_token_ids == ChunkedProgram::expected(),
              "window reclaim: reducing future entitlement lost current state or lifetime output");
    failures += check(
        program.shrinks != 0 && program.peak_residents == 2 &&
            program.smallest_admitted_window == 5 && program.starts == 2,
        "window reclaim: queued work reduced its own window instead of reclaiming unused slack");
    return failures;
}

class AvailabilityProgram final : public ProbeProgram {
public:
    enum class Fault {
        PrefillAllocation,
        DecodeAllocation,
        StableDecodeAllocation,
        PrefillInvariant,
        RequestLocal
    };

    explicit AvailabilityProgram(Fault fault) : fault_(fault) {}

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion region,
                                                       PrefillPace pace,
                                                       const OutputSession* output) override {
        if (!injected_) {
            {
                std::unique_lock lock(mutex_);
                prefill_entered_ = true;
                cv_.notify_all();
                if (!cv_.wait_for(lock, std::chrono::seconds(5),
                                  [&] { return release_prefill_; })) {
                    throw std::runtime_error("availability prefill gate timed out");
                }
            }
            injected_ = true;
            if (fault_ == Fault::PrefillAllocation) { throw std::bad_alloc(); }
            if (fault_ == Fault::PrefillInvariant) {
                throw std::logic_error("availability invariant failure");
            }
            if (fault_ == Fault::RequestLocal) {
                throw ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                                           "availability request-local failure");
            }
        }
        auto step =
            ProbeProgram::start_prefill_lane(lane, std::move(prompt), plan, region, pace, output);
        if (fault_ == Fault::DecodeAllocation || fault_ == Fault::StableDecodeAllocation) {
            step.round.tokens = std::span<const TokenId>(&continuation_, 1);
        }
        return step;
    }

    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                                     std::span<const RoundBudget>) override {
        if (fault_ == Fault::StableDecodeAllocation && ++decode_calls == 1) {
            return {.tokens     = std::span<const TokenId>(&continuation_, 1),
                    .row_counts = std::span<const std::int32_t>(&row_count_, 1),
                    .row_stride = 1};
        }
        throw std::bad_alloc();
    }

    std::uint32_t decode_calls = 0;

    void shutdown_kv_tiers(const ninfer::LoadProgress& = {}) override {
        if (fault_ == Fault::RequestLocal) { return; }
        std::unique_lock lock(mutex_);
        shutdown_entered_ = true;
        cv_.notify_all();
        if (!cv_.wait_for(lock, std::chrono::seconds(5), [&] { return release_shutdown_; })) {
            throw std::runtime_error("availability shutdown gate timed out");
        }
    }

    bool wait_for_prefill() { return wait_for(prefill_entered_); }

    bool wait_for_shutdown() { return wait_for(shutdown_entered_); }

    void release_prefill() { release(release_prefill_); }

    void release_shutdown() { release(release_shutdown_); }

private:
    bool wait_for(const bool& flag) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return flag; });
    }

    void release(bool& flag) {
        std::lock_guard lock(mutex_);
        flag = true;
        cv_.notify_all();
    }

    const Fault fault_;
    bool injected_                = false;
    const TokenId continuation_   = 1000 + 'a';
    const std::int32_t row_count_ = 1;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool prefill_entered_  = false;
    bool shutdown_entered_ = false;
    bool release_prefill_  = false;
    bool release_shutdown_ = false;
};

int run_executor_availability(Frontend& frontend) {
    int failures = 0;
    using Fault  = AvailabilityProgram::Fault;
    for (const auto fault :
         {Fault::PrefillAllocation, Fault::DecodeAllocation, Fault::StableDecodeAllocation,
          Fault::PrefillInvariant, Fault::RequestLocal}) {
        AvailabilityProgram program(fault);
        ProbeLoaded loaded{frontend};
        RecoveryProbe instance{&program, &loaded, {}};
        auto options                = engine_options();
        options.generation_recovery = false;
        ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, options);
        auto submit = [&] {
            auto prompt        = frontend.prepare(thinking_input());
            const auto summary = prompt.summary();
            ninfer::runtime::ResolvedRequestOptions request;
            request.execution.allow_prefix_reuse      = false;
            request.execution.requested_output_tokens = kOutputBudget;
            request.stop.token_ids                    = {kCallerStop};
            return executor.submit(std::move(prompt), summary, 0.0, std::move(request),
                                   ninfer::OutputDelivery::TerminalOnly);
        };
        failures += check(executor.available(), "idle executor was unavailable");
        auto first = submit();
        failures += check(program.wait_for_prefill(), "availability prefill did not start");
        std::optional<decltype(first)> queued;
        if (fault != Fault::StableDecodeAllocation) { queued.emplace(submit()); }
        failures += check(executor.available(), "busy executor was unavailable");
        program.release_prefill();
        if (fault != Fault::RequestLocal) {
            failures += check(program.wait_for_shutdown(), "fatal worker did not clean up tiers");
            failures +=
                check(!executor.available(), "fatal worker stayed available during cleanup");
            bool rejected = false;
            try {
                auto unexpected = submit();
            } catch (const ninfer::RequestError& error) {
                rejected = error.kind() == ninfer::RequestErrorKind::Unavailable;
            }
            failures += check(rejected, "failed worker accepted a request during tier cleanup");
            program.release_shutdown();
        }
        bool primary_preserved = false;
        try {
            (void)first.wait(nullptr, {});
        } catch (const std::bad_alloc&) {
            primary_preserved = fault == Fault::PrefillAllocation ||
                                fault == Fault::DecodeAllocation ||
                                fault == Fault::StableDecodeAllocation;
        } catch (const ninfer::RequestError& error) {
            primary_preserved = fault == Fault::RequestLocal &&
                                error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
        } catch (const std::logic_error& error) {
            primary_preserved = fault == Fault::PrefillInvariant &&
                                std::string(error.what()) == "availability invariant failure";
        }
        failures += check(primary_preserved, "worker failure lost its original error");
        if (queued) {
            bool queued_completed = false;
            try {
                const auto result = queued->wait(nullptr, {});
                queued_completed =
                    fault == Fault::RequestLocal && result.finish_reason == FinishReason::StopToken;
            } catch (const std::bad_alloc&) {
                queued_completed =
                    fault == Fault::PrefillAllocation || fault == Fault::DecodeAllocation;
            } catch (const std::logic_error& error) {
                queued_completed = fault == Fault::PrefillInvariant &&
                                   std::string(error.what()) == "availability invariant failure";
            }
            failures +=
                check(queued_completed, "queued request did not observe worker availability");
        } else {
            failures += check(program.decode_calls == 2,
                              "stable decode fault did not follow a successful round");
        }
        failures += check(executor.available() == (fault == Fault::RequestLocal),
                          "terminal worker availability did not match error ownership");
    }
    return failures;
}

int main(int argc, char** argv) {
    try {
        Frontend frontend = FrontendTestAccess::create_component(resources(), false);
        if (argc == 3 && std::string_view(argv[1]) == "--case" &&
            std::string_view(argv[2]) == "availability") {
            return run_executor_availability(frontend);
        }

        const CacheCase recovery_cases[] = {
            CacheCase::RecoveryRamRestoreFails, CacheCase::RecoveryResidentHit,
            CacheCase::RecoveryRamHit,          CacheCase::RecoveryDiskHit,
            CacheCase::RecoveryDiskClaimMiss,   CacheCase::RecoveryDiskRestoreFails,
        };
        const CacheCase admission_cases[] = {
            CacheCase::AdmitRamHit,
            CacheCase::AdmitDiskLongerThanRam,
            CacheCase::AdmitRamRestoreThenCold,
            CacheCase::AdmitDiskRestoreThenCold,
        };
        int failures = run_reasoning_close_sampling(frontend);
        failures += run_eight_32k_windows(frontend);
        failures += run_chunked_continuation(frontend, false);
        failures += run_chunked_continuation(frontend, true);
        failures += run_chunked_continuation(frontend, true, true);
        failures += run_admission_credit_eviction(frontend);
        failures += run_reasoning_budget_close(frontend);
        failures += run_limited_recovery_policy(frontend);
        failures += run_reasoning_budget_retry(frontend);
        failures += run_repeated_answer(frontend);
        failures += run_reasoning_repetition_boundaries(frontend);
        failures += run_withheld_reasoning_replay(frontend);
        failures += run_replay_replacement_publication(frontend);
        failures += run_chunked_reasoning_boundary(frontend);
        failures += run_chunked_lifecycle(frontend);
        failures += run_chunked_transfer_failure(frontend);
        failures += run_chunked_finite_ram_and_replacement(frontend);
        failures += run_cohort_replacement_without_spare_ram(frontend);
        failures += run_chunked_active_window_metrics(frontend);
        failures += run_completion_cohort(frontend);
        failures += run_unused_window_reclamation(frontend);
        failures += run_prefill_interruptions(frontend);
        failures += run_prefill_thresholds(frontend);
        failures += run_full_batch_compaction(frontend);
        for (const CacheCase script : recovery_cases) {
            failures += run_recovery(frontend, script);
        }
        failures += run_recovery(frontend, CacheCase::RecoveryRamHit, {.cancel_on_restore = true});
        failures += run_recovery(frontend, CacheCase::RecoveryDiskHit, {.cancel_on_restore = true});
        failures += run_recovery(frontend, CacheCase::RecoveryRamHit, {.peer_decoding = true});
        failures += run_recovery(frontend, CacheCase::RecoveryDiskHit, {.peer_decoding = true});
        failures +=
            run_recovery(frontend, CacheCase::RecoveryRamHit,
                         {.peer_decoding = true, .cancel_held = true, .peer_round_limit = 2});
        failures +=
            run_recovery(frontend, CacheCase::RecoveryRamRestoreFails, {.peer_decoding = true});
        failures += run_recovery(frontend, CacheCase::RecoveryRamRestoreFails,
                                 {.peer_decoding = true, .cold_plan_fails = true});
        failures += run_recovery(frontend, CacheCase::RecoveryPlanFails, {.peer_decoding = true});
        failures += run_recovery(frontend, CacheCase::RecoveryRamRestoreFails,
                                 {.peer_decoding        = true,
                                  .cancel_failed_hold   = true,
                                  .peer_round_limit     = 4,
                                  .settle_after_decodes = 2});
        failures += run_recovery(frontend, CacheCase::AdmitRamRestoreThenCold,
                                 {.peer_decoding        = true,
                                  .admit_beside_peer    = true,
                                  .peer_round_limit     = 6,
                                  .settle_after_decodes = 2});
        for (const CacheCase script : admission_cases) {
            failures += run_admission(frontend, script);
        }
        failures += run_admission(frontend, CacheCase::AdmitRamHit, false);
        for (const auto kind : {ninfer::runtime::CacheRestoreFailureKind::StalePlan,
                                ninfer::runtime::CacheRestoreFailureKind::DestinationUnavailable}) {
            for (const bool cache_write : {false, true}) {
                failures +=
                    run_admission(frontend, CacheCase::AdmitRamRestoreThenCold, cache_write, kind);
                failures +=
                    run_admission(frontend, CacheCase::AdmitDiskRestoreThenCold, cache_write, kind);
            }
        }
        failures += run_recovery(frontend, CacheCase::RecoveryRamHit, {.cache_write = false});
        failures += run_recovery(frontend, CacheCase::RecoveryDiskHit, {.cache_write = false});
        failures += run_recovery(frontend, CacheCase::RecoveryRamHit,
                                 {.cache_write      = false,
                                  .peer_decoding    = true,
                                  .cancel_held      = true,
                                  .peer_round_limit = 2});
        for (const auto script : {CacheCase::RecoveryRamHit, CacheCase::RecoveryDiskHit}) {
            for (const bool cache_write : {false, true}) {
                for (const std::uint32_t stale_failures : {1U, 2U}) {
                    failures += run_recovery(
                        frontend, script,
                        {.cache_write = cache_write, .stale_failures = stale_failures});
                }
            }
        }
        failures += run_retry_lifecycle(frontend);
        failures += run_idle_poll_does_not_block_submit(frontend);
        failures += run_admission_defers_for_reclaim(frontend);
        failures += run_cache_maintenance_idle(frontend);
        failures += run_executor_availability(frontend);
        std::cout << "recovery executor failures=" << failures << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "runtime/engine/concurrent_executor.h"

#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::PrefixReuseSource;
using ninfer::TokenId;
using ninfer::runtime::AdmissionResources;
using ninfer::runtime::BatchedGeneratedRound;
using ninfer::runtime::CacheRestoreFailure;
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
constexpr TokenId kResidentPrefix[] = {248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
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
    const std::string path =
        std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja";
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open " + path); }
    std::string source{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

// The toy vocabulary cannot piece the recovery notice back together. One added
// token holds the whole system fragment so the splice round-trips.
std::string recovery_fragment() {
    ninfer::PromptInput input;
    input.options.enable_thinking       = true;
    input.options.add_generation_prompt = true;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    input.messages.push_back(std::move(user));
    const auto recovery = ninfer::targets::qwen3_6::GenerationRecoveryContext::analyze(input);
    if (!recovery) { throw std::runtime_error("probe input is not eligible for recovery"); }
    const auto insert = recovery->recovery_insert({}, 1);
    if (insert.size() != 1 || insert[0].parts.size() != 1 ||
        insert[0].parts[0].kind != ninfer::MessagePartKind::Text) {
        throw std::runtime_error("probe recovery insert was not one system notice");
    }
    return "<|im_start|>system\n" + insert[0].parts[0].text + "<|im_end|>\n";
}

FrontendResources resources() {
    FrontendResources result;
    result.chat_template_jinja = read_template();
    nlohmann::json tokens      = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(14, "   \n"), added(15, "answer"), added(16, "<tool_"), added(17, "call>"),
         added(18, "<function=f>"), added(19, "</function>"), added(20, "</tool_call>"),
         added(21, "<tool_call>"), added(22, "preface"), added(23, "call"), added(24, "a <"),
         added(30, "user\n"), added(31, "assistant\n"), added(32, "\n"), added(33, "system\n"),
         added(248045, "<|im_start|>", true), added(248046, "<|im_end|>", true),
         added(248053, "<|vision_start|>", true), added(248054, "<|vision_end|>", true),
         added(248056, "<|image_pad|>", true), added(248057, "<|video_pad|>", true),
         added(248068, "<think>"), added(248069, "</think>"),
         added(500, recovery_fragment())});
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens", tokens}}
                                .dump();
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
        {"added_tokens_decoder", std::move(decoder)}}
                                       .dump();
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
    AdmitRamHit,
    AdmitDiskLongerThanRam,
    AdmitRamRestoreThenCold,
};

const char* cache_case_name(CacheCase script) {
    switch (script) {
    case CacheCase::RecoveryRamRestoreFails: return "recovery RAM restore failure";
    case CacheCase::RecoveryResidentHit: return "recovery resident hit";
    case CacheCase::RecoveryRamHit: return "recovery RAM hit";
    case CacheCase::RecoveryDiskHit: return "recovery disk hit";
    case CacheCase::RecoveryDiskClaimMiss: return "recovery disk claim miss";
    case CacheCase::RecoveryDiskRestoreFails: return "recovery disk restore failure";
    case CacheCase::AdmitRamHit: return "admission RAM hit";
    case CacheCase::AdmitDiskLongerThanRam: return "admission longer disk hit";
    case CacheCase::AdmitRamRestoreThenCold: return "admission restore then cold";
    }
    return "unknown cache case";
}

struct ProbePlan {
    RequestPlanSummary fields;
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
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes     = 0;
    std::size_t entry_count    = 0;
    std::uint64_t captures     = 0;
    std::uint64_t restores     = 0;
    std::uint64_t evictions    = 0;
    std::uint64_t drops        = 0;
    double save_seconds        = 0;
    double load_seconds        = 0;
};

struct DiskSnapshot {
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes     = 0;
    std::size_t entry_count    = 0;
    std::uint64_t captures     = 0;
    std::uint64_t restores     = 0;
    std::uint64_t evictions    = 0;
    std::uint64_t drops        = 0;
    double save_seconds        = 0;
    double load_seconds        = 0;
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

    CacheCase script = CacheCase::RecoveryRamRestoreFails;
    std::vector<std::string> trace;
    std::uint32_t copied_tokens      = 0;
    std::uint32_t abort_count        = 0;
    std::uint32_t aborts_at_prefill  = 0;
    std::uint32_t prefill_count      = 0;
    std::uint32_t prefill_lane       = 99;
    std::uint32_t prefill_tokens     = 0;
    std::uint32_t prefill_reusable   = 99;
    PrefixReuseSource prefill_source = PrefixReuseSource::HostRam;
    std::uint64_t prefill_ram_entry  = 1;
    std::uint64_t prefill_disk_entry = 1;
    std::uint32_t offered_ram        = 0;
    std::uint32_t offered_disk       = 0;
    std::uint32_t restore_ram_count  = 0;
    std::uint32_t restore_disk_count = 0;
    std::uint32_t claim_ram_count    = 0;
    std::uint32_t claim_disk_count   = 0;
    std::uint32_t consume_ram_count  = 0;
    std::uint32_t consume_disk_count = 0;
    std::uint32_t discard_ram_count  = 0;
    std::uint32_t invalidate_disk_count = 0;
    std::uint64_t claimed_ram_entry  = 0;
    std::uint64_t claimed_disk_entry = 0;
    bool prefill_terminal            = false;
    bool saw_force_cold              = false;

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
            return kDiskReuse;
        case CacheCase::RecoveryRamRestoreFails:
        case CacheCase::RecoveryResidentHit:
        case CacheCase::RecoveryRamHit:
        case CacheCase::AdmitRamHit:
        case CacheCase::AdmitRamRestoreThenCold:
            return 0;
        }
        return 0;
    }

    [[nodiscard]] AdmissionResources admission_capacity() const noexcept {
        return AdmissionResources{1, 1, 0};
    }

    [[nodiscard]] bool retain_reusable_lane(std::uint32_t) {
        note("retain");
        return script == CacheCase::RecoveryResidentHit;
    }

    [[nodiscard]] bool copy_reusable_prompt(std::uint32_t, std::uint32_t prompt_tokens,
                                            std::vector<TokenId>& tokens, std::uint32_t&) {
        copied_tokens = prompt_tokens;
        if (prompt_tokens != kResidentTokens) { return false; }
        note("copy");
        tokens.assign(std::begin(kResidentPrefix), std::end(kResidentPrefix));
        return true;
    }

    [[nodiscard]] ProbePlan plan_request_base(const PreparedPrompt&,
                                              const ninfer::runtime::ResolvedExecutionOptions& options) {
        note("plan_base");
        ProbePlan plan     = fitted_plan();
        plan.force_cold    = options.force_cold_prefill;
        plan.allow_reuse   = options.allow_prefix_reuse;
        saw_force_cold     = saw_force_cold || options.force_cold_prefill;
        return plan;
    }

    [[nodiscard]] ProbePlan plan_request_for_lane(std::uint32_t, const PreparedPrompt&,
                                                  const ProbePlan& base) {
        note("plan_cold");
        ProbePlan plan   = fitted_plan();
        plan.force_cold  = base.force_cold;
        plan.allow_reuse = base.allow_reuse;
        if (script == CacheCase::RecoveryResidentHit && !base.force_cold && base.allow_reuse) {
            plan.fields.reusable_prompt_tokens = kResidentTokens;
            plan.fields.reuse_source           = PrefixReuseSource::VramResident;
        }
        return plan;
    }

    [[nodiscard]] ProbePlan plan_ram_reuse(const PreparedPrompt&, const ProbePlan& base) {
        note("plan_ram");
        ProbePlan plan = fitted_plan();
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
        ProbePlan plan = fitted_plan();
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

    [[nodiscard]] bool can_admit_lane(std::uint32_t, const ProbePlan&) const noexcept {
        return true;
    }
    [[nodiscard]] bool can_admit_lane_after_retained_eviction(std::uint32_t,
                                                             const ProbePlan&) const noexcept {
        return false;
    }
    [[nodiscard]] bool can_admit_lane_after_releasing(std::uint32_t, const ProbePlan&,
                                                      std::span<const std::uint32_t>) const noexcept {
        return false;
    }

    [[nodiscard]] ninfer::GenerationTimings generation_timings_lane(std::uint32_t) const noexcept {
        return {};
    }
    [[nodiscard]] ninfer::SpeculativeStats speculative_stats_lane(std::uint32_t) const noexcept {
        return {};
    }
    [[nodiscard]] std::uint32_t captured_context_checkpoint_tokens_lane(std::uint32_t) const noexcept {
        return 0;
    }
    [[nodiscard]] std::uint32_t restored_context_checkpoint_tokens_lane(std::uint32_t) const noexcept {
        return 0;
    }

    void abort_lane(std::uint32_t) noexcept { note("abort"); ++abort_count; }

    void claim_ram_entry(std::uint64_t entry_id) {
        note("claim_ram");
        ++claim_ram_count;
        claimed_ram_entry = entry_id;
    }

    void restore_ram_entry(std::uint32_t, std::uint64_t, const ProbePlan&) {
        note("restore_ram");
        ++restore_ram_count;
        if (script == CacheCase::RecoveryRamRestoreFails ||
            (script == CacheCase::AdmitRamRestoreThenCold && restore_ram_count == 1)) {
            throw CacheRestoreFailure("probe host restore failed");
        }
    }

    void release_ram_entry(std::uint64_t) { note("release_ram"); }
    void discard_ram_capture(std::uint64_t) {
        note("discard_ram");
        ++discard_ram_count;
    }
    void cancel_disk_restore() { note("cancel_disk"); }
    void synchronize_all() { note("sync"); }

    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt prompt,
                                                       ProbePlan plan, TransientRegion,
                                                       const OutputSession*) {
        note("start_prefill");
        ++prefill_count;
        aborts_at_prefill = abort_count;
        prefill_lane      = lane;
        prefill_tokens    = prompt.summary().prompt_tokens;
        prefill_reusable  = plan.summary().reusable_prompt_tokens;
        prefill_source    = plan.summary().reuse_source;
        prefill_ram_entry = plan.summary().ram_entry_id;
        prefill_disk_entry = plan.summary().disk_entry_id;
        PrefillStepResult step;
        step.complete                = true;
        step.processed_prompt_tokens = 1;
        step.round.tokens            = std::span<const TokenId>(&licensed_, 1);
        step.summary.prompt_tokens   = prefill_tokens;
        step.summary.reused_prompt_tokens = prefill_reusable;
        step.summary.prefix_reuse_source = prefill_source;
        step.summary.prefix_reuse_path =
            prefill_reusable == 0 ? ninfer::PrefixReusePath::FullReset
                                  : ninfer::PrefixReusePath::AppendAtFrontier;
        return step;
    }

    void resolve_prefill_lane(std::uint32_t, bool terminal) {
        note("resolve_prefill");
        prefill_terminal = terminal;
    }

    [[nodiscard]] bool kv_copies_ready() const { return false; }
    void request_idle_spill() {}
    void shutdown_kv_tiers(ninfer::LoadProgress = {}) {}
    [[nodiscard]] RamSnapshot kv_ram_snapshot() const noexcept { return {}; }
    [[nodiscard]] DiskSnapshot kv_disk_snapshot() const noexcept { return {}; }
    [[nodiscard]] GpuSnapshot kv_gpu_snapshot() const noexcept { return {}; }
    [[nodiscard]] RamCopySeconds harvest_kv_ram_copy_seconds() { return {}; }
    [[nodiscard]] DiskCopySeconds harvest_kv_disk_copy_seconds() { return {}; }
    [[nodiscard]] bool kv_ram_copies_ready() const { return true; }
    [[nodiscard]] bool kv_disk_restore_failed() const { return false; }
    // A request starts at index version zero. Zero here would skip the host lookup.
    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept { return 1; }
    [[nodiscard]] std::uint64_t kv_disk_index_version() const noexcept { return 1; }
    [[nodiscard]] std::uint64_t pending_disk_restore_ticket() const noexcept { return 0; }
    [[nodiscard]] bool has_retained_lane(std::uint32_t) const noexcept { return false; }
    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t) const noexcept { return 0; }
    [[nodiscard]] bool capture_retained_lane(std::uint32_t, std::uint64_t* = nullptr) { return false; }
    [[nodiscard]] bool claim_disk_entry(std::uint64_t entry_id, std::uint32_t, std::uint64_t,
                                        std::uint64_t, std::uint32_t, ninfer::PrefixReusePath,
                                        std::uint64_t) {
        note("claim_disk");
        ++claim_disk_count;
        claimed_disk_entry = entry_id;
        return script != CacheCase::RecoveryDiskClaimMiss;
    }
    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t) { return true; }
    [[nodiscard]] PrefillStepResult advance_prefill_lane(std::uint32_t) {
        throw std::logic_error("cold prefill did not complete in one step");
    }
    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
                                                     std::span<const RoundBudget>) {
        throw std::logic_error("recovery entered decode after the cold prefill");
    }

    void release_disk_entry(std::uint64_t) { note("release_disk"); }
    void invalidate_disk_entry(std::uint64_t) {
        note("invalidate_disk");
        ++invalidate_disk_count;
    }
    void consume_ram_entry(std::uint64_t) {
        note("consume_ram");
        ++consume_ram_count;
    }
    void consume_disk_entry(std::uint64_t) {
        note("consume_disk");
        ++consume_disk_count;
    }
    void prefetch_disk_plan(std::uint64_t, const ProbePlan&) { note("prefetch_disk"); }
    void pump_disk_restore() {}
    void restore_disk_entry(std::uint32_t, std::uint64_t, const ProbePlan&) {
        note("restore_disk");
        ++restore_disk_count;
        if (script == CacheCase::RecoveryDiskRestoreFails) {
            throw CacheRestoreFailure("probe disk restore failed");
        }
    }
    void wait_kv_ram_copies() {}
    void wait_kv_disk_copies() {}
    void wait_kv_ram_copies_on_compute() {}
    void evict_retained_lane(std::uint32_t) noexcept {}
    void retain_lane(std::uint32_t) {}
    void set_suppressed_tokens_lane(std::uint32_t, std::span<const TokenId>) {}
    void clear_suppressed_tokens_lane(std::uint32_t) {}
    void set_typical_cycle_reasoning_lane(std::uint32_t, bool) {}
    void resolve_pending_batch(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                               std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                               std::span<const std::uint8_t>) {}

private:
    TokenId licensed_ = kCallerStop;
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
    using Package = ProbePackage;
    ProbeProgram* program = nullptr;
    ProbeLoaded* loaded   = nullptr;
    ProbeMemory request_memory;
};

const char* kExpectedTrace[] = {
    "retain",        "copy",       "plan_base",   "plan_ram",      "plan_disk",
    "abort",         "claim_ram",  "restore_ram", "cancel_disk",   "sync",
    "release_ram",   "discard_ram", "abort",      "abort",         "plan_cold",
    "start_prefill", "resolve_prefill",
};

struct RecoveryOutcome {
    bool finished                  = false;
    std::string error;
    std::uint32_t attempts         = 0;
    std::uint32_t budget_remaining = 0;
    bool has_budget                = false;
    bool cache_fallback            = false;
    bool force_cold                = false;
    FinishReason finish            = FinishReason::None;
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
    failures += fail_case(program, outcome.finished && outcome.error.empty(),
                          outcome.error.empty() ? "the retry did not finish" : outcome.error.c_str());
    failures += fail_case(program, outcome.attempts == 1 && outcome.has_budget &&
                                       outcome.budget_remaining == kOutputBudget - 1 &&
                                       outcome.finish == FinishReason::StopToken,
                          "the retry did not keep its output budget and finish");
    failures += fail_case(program, outcome.pending_empty && outcome.recovery_empty &&
                                       outcome.slot_empty && !outcome.executor_failed,
                          "the retry left a queue, lane, or failed executor behind");
    failures += fail_case(program, program.copied_tokens == kResidentTokens,
                          "the retry copied a length other than the resident ledger");
    const bool cold_compute = program.prefill_count == 1 && program.prefill_reusable == 0 &&
                              program.prefill_source == PrefixReuseSource::None &&
                              program.prefill_ram_entry == 0 && program.prefill_disk_entry == 0 &&
                              program.prefill_terminal;
    switch (program.script) {
    case CacheCase::RecoveryRamRestoreFails: {
        const bool trace_ok = program.trace.size() == std::size(kExpectedTrace) &&
                              std::equal(program.trace.begin(), program.trace.end(),
                                         std::begin(kExpectedTrace));
        failures += fail_case(program, trace_ok,
                              "host restore failure did not abort, release, and cold-prefill");
        failures += fail_case(program, program.claimed_ram_entry == kRamEntryId &&
                                           program.aborts_at_prefill == 3 && program.abort_count == 3 &&
                                           cold_compute && program.prefill_tokens > kResidentTokens,
                              "cold prefill did not follow the three idempotent lane aborts");
        failures += fail_case(program, !outcome.cache_fallback && !outcome.force_cold,
                              "the failed recovery restore stuck a cold fallback");
        break;
    }
    case CacheCase::RecoveryResidentHit:
        failures += fail_case(program,
                              program.abort_count == 0 && event_count(program.trace, "plan_ram") == 0 &&
                                  event_count(program.trace, "plan_disk") == 0 &&
                                  program.claim_ram_count == 0 && program.prefill_count == 1 &&
                                  program.prefill_source == PrefixReuseSource::VramResident &&
                                  program.prefill_reusable == kResidentTokens &&
                                  program.prefill_tokens > kResidentTokens &&
                                  program.prefill_ram_entry == 0 && !outcome.cache_fallback &&
                                  !outcome.force_cold,
                              "a resident prefix was dropped instead of being prefilled in place");
        break;
    case CacheCase::RecoveryRamHit:
        failures += fail_case(program,
                              program.abort_count == 1 && program.consume_ram_count == 1 &&
                                  program.discard_ram_count == 0 && program.prefill_count == 1 &&
                                  program.prefill_source == PrefixReuseSource::HostRam &&
                                  program.prefill_reusable == kRamReuse &&
                                  program.prefill_ram_entry == kRamEntryId &&
                                  event_count(program.trace, "plan_cold") == 0 &&
                                  in_order(program.trace, {"abort", "restore_ram", "start_prefill",
                                                           "consume_ram"}) &&
                                  !outcome.cache_fallback && !outcome.force_cold &&
                                  !program.saw_force_cold,
                              "a RAM checkpoint was not restored onto the retry");
        break;
    case CacheCase::RecoveryDiskHit:
        failures += fail_case(program,
                              program.abort_count == 1 && program.consume_disk_count == 1 &&
                                  program.invalidate_disk_count == 0 && program.restore_ram_count == 0 &&
                                  program.prefill_count == 1 &&
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
        failures += fail_case(program,
                              program.abort_count == 2 && program.restore_disk_count == 0 &&
                                  program.invalidate_disk_count == 0 &&
                                  event_count(program.trace, "cancel_disk") == 0 && cold_compute &&
                                  program.claimed_disk_entry == kDiskEntryId &&
                                  in_order(program.trace, {"claim_disk", "plan_cold", "start_prefill"}) &&
                                  !outcome.cache_fallback && !outcome.force_cold,
                              "a disk claim miss did not cold-prefill without invalidating an entry");
        break;
    case CacheCase::RecoveryDiskRestoreFails:
        failures += fail_case(program,
                              program.abort_count == 3 && program.invalidate_disk_count == 1 &&
                                  program.consume_disk_count == 0 && cold_compute &&
                                  in_order(program.trace, {"restore_disk", "invalidate_disk", "plan_cold",
                                                           "start_prefill"}) &&
                                  !outcome.cache_fallback && !outcome.force_cold,
                              "a failed disk restore did not invalidate that entry and cold-prefill");
        break;
    case CacheCase::AdmitRamHit:
    case CacheCase::AdmitDiskLongerThanRam:
    case CacheCase::AdmitRamRestoreThenCold:
        failures += fail_case(program, false, "an admission script was driven as a retry");
        break;
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
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
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
    using Request = typename ConcurrentExecutor<ProbeInstance>::Request;
    Frontend& frontend = executor.instance_.loaded->frontend;
    auto input = thinking_input();

    auto prepared = frontend.prepare(input);
    const ninfer::PromptSummary summary = prepared.summary();
    ninfer::StopPolicy stop;
    stop.token_ids = {kCallerStop};
    auto session = frontend.make_output_session(prepared, stop, {});

    ResolvedRequestOptions options;
    options.execution.sampling.p_less          = true;
    options.execution.allow_prefix_reuse       = true;
    options.execution.force_cold_prefill       = false;
    options.execution.requested_output_tokens  = kOutputBudget;
    options.stop                               = stop;
    const auto now = ConcurrentExecutor<ProbeInstance>::Clock::now();
    auto request = std::shared_ptr<Request>(new Request(
        1, std::move(prepared), std::move(session), summary, 0.0, std::move(options),
        ninfer::OutputDelivery::TerminalOnly, now + std::chrono::hours(1), now, ninfer::HostInputLease{},
        true));
    int failures = check(request->recovery_context != nullptr, "the retry had no recovery context");
    request->budget.emplace(kOutputBudget, FinishReason::OutputLimit);
    request->lane                    = 0;
    request->recovery_pending        = true;
    request->resident_prompt_tokens  = kResidentTokens;
    request->admission_resources     = AdmissionResources{1, 1, 0};
    request->remaining_service_work  = 4;
    request->recovery_cause          = "repeated_reasoning";

    {
        std::scoped_lock locks(executor.execution_mutex_, executor.queue_mutex_);
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
    outcome.finished = true;
    if (request->error != nullptr) {
        try {
            std::rethrow_exception(request->error);
        } catch (const std::exception& error) {
            outcome.error = error.what();
        } catch (...) {
            outcome.error = "the retry completed with an unknown error";
        }
    }
    outcome.attempts         = request->recovery.attempts;
    outcome.has_budget       = request->budget.has_value();
    outcome.budget_remaining = request->budget ? request->budget->remaining() : 0;
    outcome.cache_fallback   = request->cache_fallback;
    outcome.force_cold       = request->options.execution.force_cold_prefill;
    outcome.finish           = request->result.finish_reason;
    {
        std::scoped_lock locks(executor.execution_mutex_, executor.queue_mutex_);
        outcome.pending_empty   = executor.pending_.empty();
        outcome.recovery_empty  = executor.recovery_queue_.empty();
        outcome.slot_empty      = executor.slots_[0] == nullptr;
        outcome.executor_failed = executor.failed_;
    }
    return failures + check_scripted_recovery(*executor.instance_.program, outcome);
}

} // namespace ninfer::runtime

int run_recovery(Frontend& frontend, CacheCase script) {
    ProbeProgram program;
    program.script = script;
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program = &program;
    instance.loaded  = &loaded;
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine_options());
    return ninfer::runtime::drive_scripted_recovery(executor);
}

int run_admission(Frontend& frontend, CacheCase script) {
    ProbeProgram program;
    program.script = script;
    ProbeLoaded loaded{frontend};
    RecoveryProbe instance;
    instance.program = &program;
    instance.loaded  = &loaded;
    ninfer::runtime::ConcurrentExecutor<RecoveryProbe> executor(instance, engine_options());

    auto prepared = frontend.prepare(thinking_input());
    const auto summary = prepared.summary();
    ninfer::runtime::ResolvedRequestOptions options;
    options.execution.sampling.p_less    = true;
    options.execution.allow_prefix_reuse = true;
    options.stop.token_ids               = {kCallerStop};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    ninfer::CancellationView cancel([deadline] {
        return std::chrono::steady_clock::now() >= deadline;
    });
    try {
        auto submission = executor.submit(std::move(prepared), summary, 0.0, std::move(options),
                                          ninfer::OutputDelivery::TerminalOnly);
        const ninfer::GenerationResult result = submission.wait(nullptr, cancel);
        int failures = 0;
        const bool cold = program.prefill_count == 1 && program.prefill_reusable == 0 &&
                          program.prefill_source == PrefixReuseSource::None &&
                          result.reused_prompt_tokens == 0 &&
                          result.prefix_reuse_source == PrefixReuseSource::None;
        switch (script) {
        case CacheCase::AdmitRamHit:
            failures += fail_case(program,
                                  result.finish_reason == FinishReason::StopToken &&
                                      result.reused_prompt_tokens == kRamReuse &&
                                      result.prefix_reuse_source == PrefixReuseSource::HostRam &&
                                      program.prefill_ram_entry == kRamEntryId &&
                                      program.consume_ram_count == 1 && program.abort_count == 0 &&
                                      program.discard_ram_count == 0 && program.restore_ram_count == 1 &&
                                      !program.saw_force_cold,
                                  "admission did not restore the RAM checkpoint into the result");
            break;
        case CacheCase::AdmitDiskLongerThanRam:
            failures += fail_case(program,
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
            failures += fail_case(program,
                                  result.finish_reason == FinishReason::StopToken && cold &&
                                      program.saw_force_cold && program.discard_ram_count == 1 &&
                                      program.consume_ram_count == 0 && program.claim_ram_count == 1 &&
                                      program.restore_ram_count == 1 && program.offered_ram == kRamReuse &&
                                      in_order(program.trace, {"claim_ram", "restore_ram", "discard_ram",
                                                               "start_prefill"}),
                                  "a failed admission restore was reused instead of computed cold");
            break;
        case CacheCase::RecoveryRamRestoreFails:
        case CacheCase::RecoveryResidentHit:
        case CacheCase::RecoveryRamHit:
        case CacheCase::RecoveryDiskHit:
        case CacheCase::RecoveryDiskClaimMiss:
        case CacheCase::RecoveryDiskRestoreFails:
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

int main() {
    try {
        Frontend frontend = FrontendTestAccess::create_component(resources(), false);
        const CacheCase recovery_cases[] = {
            CacheCase::RecoveryRamRestoreFails, CacheCase::RecoveryResidentHit,
            CacheCase::RecoveryRamHit,          CacheCase::RecoveryDiskHit,
            CacheCase::RecoveryDiskClaimMiss,   CacheCase::RecoveryDiskRestoreFails,
        };
        const CacheCase admission_cases[] = {
            CacheCase::AdmitRamHit,
            CacheCase::AdmitDiskLongerThanRam,
            CacheCase::AdmitRamRestoreThenCold,
        };
        int failures = 0;
        for (const CacheCase script : recovery_cases) { failures += run_recovery(frontend, script); }
        for (const CacheCase script : admission_cases) { failures += run_admission(frontend, script); }
        std::cout << "recovery executor failures=" << failures << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

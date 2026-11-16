#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "config.h"
#include "load/binder.h"
#include "runtime/program.h"

#include "core/device.h"
#include "artifact/reader.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_cache.h"
#include <openssl/evp.h>
#include "runtime/contract/typical_cycle.h"
#include "targets/qwen3_6/impl/runtime/score_index.h"
#include <ninfer/targets/qwen3_6/generation_recovery.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

using Clock    = std::chrono::steady_clock;
namespace tier = qwen3_6::detail;

// The external table is immutable while the Engine runs. Disk reuse binds its complete contents,
// not a pathname or timestamps, because any changed row changes the represented model inputs.
[[nodiscard]] std::string table_content_identity(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot fingerprint n-gram table: " + path.string()); }
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(),
                                                                       EVP_MD_CTX_free);
    if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("cannot initialize n-gram SHA-256");
    }
    std::vector<char> buffer(std::size_t{4} << 20);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = file.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(hash.get(), buffer.data(), static_cast<std::size_t>(count)) != 1) {
            throw std::runtime_error("cannot update n-gram SHA-256");
        }
    }
    if (!file.eof()) { throw std::runtime_error("cannot read n-gram table fingerprint"); }
    std::array<unsigned char, 32> digest{};
    unsigned bytes = 0;
    if (EVP_DigestFinal_ex(hash.get(), digest.data(), &bytes) != 1 || bytes != digest.size()) {
        throw std::runtime_error("cannot finish n-gram SHA-256");
    }
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result             = "sha256:";
    for (const auto value : digest) {
        result += hex[value >> 4];
        result += hex[value & 15U];
    }
    return result;
}

// A prefill reports its throughput over its trailing steps covering at least this much time.
constexpr double kPrefillTailWindowSeconds = 1.0;

[[nodiscard]] double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    constexpr auto kPage = static_cast<std::uint32_t>(ops::kQsaPageTokens);
    return tokens / kPage + (tokens % kPage != 0 ? 1U : 0U);
}

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

[[nodiscard]] ops::QsaKvFormat qsa_kv_format(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::Nvfp4:
        return ops::QsaKvFormat::NVFP4G16;
    case KvCacheStorage::BFloat16:
        return ops::QsaKvFormat::BF16;
    case KvCacheStorage::Int8Group64:
        break;
    }
    throw std::invalid_argument("qwen3.8-flash-next stores QSA state only as nvfp4 or bf16");
}

[[nodiscard]] ops::QsaSelectBudget qsa_select_budget(std::optional<QsaSelectBudget> budget) {
    switch (budget.value_or(QsaSelectBudget::Blocks512)) {
    case QsaSelectBudget::Blocks512:
        return ops::QsaSelectBudget::Blocks512;
    case QsaSelectBudget::Blocks1024:
        return ops::QsaSelectBudget::Blocks1024;
    case QsaSelectBudget::Blocks2048:
        return ops::QsaSelectBudget::Blocks2048;
    }
    throw std::invalid_argument("invalid QSA select budget");
}

[[nodiscard]] ProgramOptions make_program_options(const EngineOptions& options) {
    ProgramOptions out;
    out.ngram_table    = options.ngram_table_path;
    out.context_tokens = static_cast<std::int32_t>(options.max_context);
    out.lanes          = static_cast<std::int32_t>(options.max_concurrency);
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        out.kv_capacity_tokens = options.kv_capacity.explicit_tokens;
    }
    out.speculative              = options.speculative;
    out.kv_format                = qsa_kv_format(options.kv_cache);
    out.forward.max_chunk_tokens = static_cast<std::int32_t>(options.prefill_chunk);
    out.forward.select_budget    = qsa_select_budget(options.qsa_select_budget);
    return out;
}

[[noreturn]] void unsupported(const char* operation) {
    throw std::logic_error(std::string("qwen3.8-flash-next has no ") + operation);
}

} // namespace

class EngineProgram::Impl {
public:
    struct LaneFence {
        CudaEvent event{cudaEventDisableTiming};
    };

    Impl(const LoadedModel& model, const EngineOptions& options, DeviceContext& context)
        : device(context), capacity(options.max_context), kv_cache(options.kv_cache),
          weights_bytes(model.materialization_stats().device_capacity_bytes),
          program_options(make_program_options(options)),
          image_slots(options.kv_ram_capacity_bytes == 0 ? 2U : 3U),
          program(model, program_options, context) {
        if (program.lanes() != static_cast<std::int32_t>(options.max_concurrency)) {
            throw std::logic_error("qwen3.8-flash-next Program lanes differ from the Engine's");
        }
        const std::size_t image_bytes = program.state().continuation_bytes();
        const std::size_t slab_bytes =
            std::size_t{image_slots} * options.max_concurrency * image_bytes;
        const std::size_t host_bytes = slab_bytes + options.kv_ram_capacity_bytes;
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo) { throw std::runtime_error("cannot read available host RAM"); }
        bool available_found = false;
        std::string key;
        std::size_t kib = 0;
        while (meminfo >> key >> kib) {
            if (key == "MemAvailable:") {
                available_found = true;
                if (host_bytes + (std::size_t{4} << 30) > kib * 1024U) {
                    throw std::runtime_error(
                        "Flash-Next checkpoint and RAM arenas must leave 4096 MiB available");
                }
                break;
            }
            meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        if (!available_found) {
            throw std::runtime_error("available host RAM is missing from /proc/meminfo");
        }
        checkpoint_slab = std::make_unique<HostPinnedArena>(slab_bytes);
        for (std::uint32_t lane = 0; lane < options.max_concurrency; ++lane) {
            bind_images(records.at(lane), lane);
        }
        suspended.reserve(kMaximumConcurrency);
        system_prompt_limit = options.kv_ram_system_prompts;
        if (options.kv_ram_capacity_bytes != 0) {
            ram = std::make_unique<tier::KVRamCache>(options.kv_ram_capacity_bytes,
                                                     system_prompt_limit);
        }
        if (options.kv_disk_capacity_bytes != 0) {
            const artifact::Reader reader(options.artifact_path);
            std::vector<artifact::TensorContentSelection> tensors;
            for (const auto& object : reader.objects()) {
                if (const auto* tensor = std::get_if<artifact::TensorDescriptor>(&object)) {
                    tensors.push_back({tensor->name, std::nullopt});
                }
            }
            auto fingerprint = tier::make_disk_fingerprint(
                reader.identity().model_id, reader.identity().weights_id,
                reader.tensor_content_identity(tensors), options.kv_cache,
                options.speculative.backend, program.state().pool(), nullptr, nullptr, nullptr);
            fingerprint.flash_next_state_bytes = image_bytes;
            fingerprint.flash_next_ngram_identity =
                table_content_identity(options.ngram_table_path);
            fingerprint.flash_next_select_budget =
                static_cast<std::uint32_t>(program_options.forward.select_budget);
            tier::DiskOpenConfig config;
            config.location           = options.kv_disk_location;
            config.capacity_bytes     = options.kv_disk_capacity_bytes;
            config.compress           = options.kv_disk_compress;
            config.max_context        = options.max_context;
            config.ram                = ram.get();
            config.fingerprint        = std::move(fingerprint);
            config.text_pool          = &program.state().pool();
            config.logical_page_bytes = paged_kv_logical_page_bytes(program.state().pool());
            disk                      = std::make_unique<tier::KVDiskCache>(std::move(config));
        }
    }

    ~Impl() {
        device.synchronize_all();
        disk.reset();
        ram.reset();
    }

    // Where an executor lane's request is in its lifecycle.
    enum class Phase : std::uint8_t {
        Idle,     // no Program lane
        Prefill,  // prompt tokens remain to be appended
        Sampled,  // a sampled token awaits the executor's resolution
        Ready,    // the last committed token is the next decode round's input
        Retained, // an idle lane with exact continuation state and no growth reservation
    };

    struct LaneRecord {
        Phase phase = Phase::Idle;
        std::optional<Lane> lane;
        std::uint32_t reserve_tokens = 0;
        std::vector<TokenId> prompt;
        std::size_t appended = 0;
        // Exactly the tokens whose effects are present in every model-state component.
        std::vector<TokenId> tokens;
        runtime::BeginSummary begin;
        bool allow_reuse   = false;
        bool cache_write   = true;
        bool turn_closed   = false;
        bool tier_restored = false;
        tier::ResidentPrefixIdentity identity;
        std::optional<std::uint32_t> system_capture;
        std::uint64_t disk_entry_id = 0;
        void* tier_image            = nullptr;
        // Covers copy-stream tier transfers and pinned-image callbacks in either direction.
        // A pending disk ticket separately owns scatter on the disk cache's streams.
        cudaEvent_t capture_done              = nullptr;
        std::uint32_t window_output_tokens    = 0;
        std::uint32_t effective_output_tokens = 0;
        std::uint64_t use_tick                = 0;
        std::optional<qwen3_6::RewriteCheckpointSpec> rewrite;
        std::optional<qwen3_6::RewriteCheckpointSpec> capture_rewrite;
        std::vector<std::uint32_t> closure_frontiers;
        std::uint32_t rollback_frontier = 0;
        void* rewrite_image             = nullptr;
        void* rollback_image            = nullptr;
        // The session that owns this request's output policy; it outlives the lane.
        const OutputSession* output = nullptr;
        LaneSampling sampling;
        bool token_logprobs          = false;
        bool typical_cycle_reasoning = false;
        // Committed output tokens; the last one is the next decode round's input while Ready.
        std::vector<TokenId> generated;
        std::array<TokenId, kDflashWidth> sampled_tokens{};
        std::uint32_t sampled_count = 0;
        TokenId pending_input       = -1;
        bool pending_deferred       = false;
        SpeculativeStats speculative;
        // The sampled token awaiting resolution and its records (RoundLogprobs storage).
        TokenId sampled = -1;
        std::array<float, 1> sampled_logprob{};
        std::array<TokenId, kMaximumTopLogprobs> top_ids{};
        std::array<float, kMaximumTopLogprobs> top_logprobs{};
        GenerationTimings timings;
        // (tokens, seconds) of each prefill step of the current attempt, for the tail rate.
        std::vector<std::pair<std::uint32_t, double>> prefill_steps;
    };

    struct Suspended {
        enum class Phase : std::uint8_t { Saving, Parked, Loading };
        Phase phase = Phase::Saving;
        LaneRecord request;
        std::uint32_t physical_lane = 0;
        std::uint64_t image_id      = 0;
        // Credit growth preserves existing byte offsets even when its arena block relocates.
        std::size_t sampling_counts_offset = 0;
    };

    static constexpr std::size_t kSamplingCountsBytes =
        std::size_t{kVocabRows} * sizeof(std::int32_t);

    void bind_images(LaneRecord& entry, std::uint32_t lane) {
        const auto count     = program.state().continuation_bytes();
        auto* base           = static_cast<std::byte*>(checkpoint_slab->base()) +
                               std::size_t{image_slots} * lane * count;
        entry.rewrite_image  = base;
        entry.rollback_image = base + count;
        entry.tier_image     = image_slots == 3 ? base + 2U * count : nullptr;
        entry.capture_done   = lane_fences.at(lane).event.get();
    }

    [[nodiscard]] runtime::ExecutionWindow execution_window(std::uint32_t prompt_tokens,
                                                            std::uint32_t output_tokens,
                                                            std::uint32_t minimum_frontier = 0) {
        const auto frontier = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, std::uint64_t{prompt_tokens} +
                                                  (output_tokens == 0 ? 0U : output_tokens - 1U)));
        const auto extent = std::max(frontier, minimum_frontier);
        runtime::ExecutionWindow result;
        result.admission     = {1, pages_for_tokens(extent), 0};
        result.output_tokens = output_tokens;
        if (ram) {
            result.suspension_bytes =
                tier::KVRamCache::live_image_budget(
                    program.state().pool(), nullptr, result.admission.main_kv_pages, 0,
                    std::size_t{extent} + 1U, 64U + 13U * (std::size_t{extent} + 1U), 0, 0, 0, 0, 1,
                    program.state().continuation_bytes()) +
                kSamplingCountsBytes;
        }
        return result;
    }

    [[nodiscard]] LaneRecord& record(std::uint32_t lane) {
        if (lane >= static_cast<std::uint32_t>(program.lanes())) {
            throw std::invalid_argument("qwen3.8-flash-next lane is out of range");
        }
        return records.at(lane);
    }

    [[nodiscard]] const LaneRecord& record(std::uint32_t lane) const {
        if (lane >= static_cast<std::uint32_t>(program.lanes())) {
            throw std::invalid_argument("qwen3.8-flash-next lane is out of range");
        }
        return records.at(lane);
    }

    [[nodiscard]] std::uint32_t super_chunk_tokens() const noexcept {
        return static_cast<std::uint32_t>(program_options.forward.super_chunk_tokens);
    }

    [[nodiscard]] std::uint64_t prefill_steps(std::uint32_t tokens) const noexcept {
        const std::uint64_t step = super_chunk_tokens();
        return (tokens + step - 1) / step;
    }

    // Restricts the lane's next sampled token to its output session's grammar, budget or replay
    // policy (OutputSession::fill_token_masks at the committed root), as the family Program does
    // for ordinary rounds. The Program consumes the mask with that token. Throws RequestError
    // (InvalidOutputConstraint) when the policy, after the suppressed tokens, admits no token; the
    // caller fails only that request.
    void arm_token_mask(LaneRecord& entry) {
        if (entry.output == nullptr || !entry.output->has_token_masks()) { return; }
        const std::array<TokenId, 1> root{0};
        const std::array<std::int32_t, 1> parent{-1};
        entry.output->fill_token_masks(root, parent, mask_words);
        bool eligible = false;
        for (std::size_t word = 0; word < mask_words.size() && !eligible; ++word) {
            std::uint32_t bits = mask_words.at(word);
            if (word + 1 == mask_words.size() && kTokenDomain % 32 != 0) {
                bits &= (std::uint32_t{1} << (kTokenDomain % 32)) - 1U;
            }
            for (const TokenId token : entry.sampling.suppressed_tokens) {
                if (static_cast<std::size_t>(token) / 32 == word) {
                    bits &= ~(std::uint32_t{1} << (static_cast<std::uint32_t>(token) % 32));
                }
            }
            eligible = bits != 0;
        }
        if (!eligible) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                               "the output constraint leaves no eligible token");
        }
        program.set_allowed_tokens(entry.lane.value(), mask_words);
    }

    // The cycle-exit continuation for the lane's next sampled token: thinking-phase p-less rows at
    // positive temperature exclude the continuation of a near-square suffix of their output.
    void arm_typical_exclude(LaneRecord& entry) {
        TokenId exclude = -1;
        if (entry.typical_cycle_reasoning) {
            exclude = runtime::typical_exclude_for_sample(
                true, entry.sampling.p_less, entry.sampling.temperature, entry.generated);
            if (entry.output != nullptr &&
                !entry.output->reasoning_cycle_exclusion_allowed(exclude)) {
                exclude = -1;
            }
        }
        if (exclude != entry.sampling.typical_exclude) {
            entry.sampling.typical_exclude = exclude;
            program.set_sampling(entry.lane.value(), entry.sampling);
        }
    }

    // Joins only this lane's tier uses before releasing pages. A pending disk destination is
    // cancelled here, so page safety does not depend on the executor draining scatter first.
    // The host-side reset precedes Program release and completes even if that release throws.
    void close(LaneRecord& entry) {
        const std::optional<Lane> lane = entry.lane;
        if (pending_disk_owner == &entry) {
            disk->cancel_restore();
            disk->release_restore_ticket(pending_disk_ticket);
            pending_disk_ticket = 0;
            pending_disk_owner  = nullptr;
        }
        program.extents().wait_tier_copy(entry.capture_done);
        entry.lane.reset();
        entry.phase  = Phase::Idle;
        entry.output = nullptr;
        std::vector<TokenId>().swap(entry.prompt);
        entry.tokens.clear();
        entry.rewrite.reset();
        entry.capture_rewrite.reset();
        entry.rollback_frontier = 0;
        entry.appended          = 0;
        entry.use_tick          = 0;
        entry.cache_write       = true;
        entry.allow_reuse       = false;
        entry.tier_restored     = false;
        entry.turn_closed       = false;
        entry.system_capture.reset();
        entry.disk_entry_id = 0;
        entry.identity.clear();
        entry.reserve_tokens = 0;
        if (lane) { program.close_lane(*lane); }
    }

    void snapshot(LaneRecord& entry, void* image) {
        CUDA_CHECK(cudaEventSynchronize(entry.capture_done));
        program.state().pack_continuation(entry.lane.value(), image, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }

    bool retain(LaneRecord& entry) {
        if (entry.phase == Phase::Retained) { return true; }
        if (!entry.lane || entry.tokens.empty() || !entry.allow_reuse) {
            close(entry);
            return false;
        }
        program.extents().wait_tier_copy(entry.capture_done);
        const std::array lanes{*entry.lane};
        program.flush_context(lanes);
        program.resize_lane(*entry.lane, static_cast<std::int32_t>(entry.tokens.size()),
                            static_cast<std::int64_t>(entry.tokens.size()));
        entry.phase          = Phase::Retained;
        entry.output         = nullptr;
        entry.reserve_tokens = static_cast<std::uint32_t>(entry.tokens.size());
        return true;
    }

    [[nodiscard]] std::span<std::byte> image(void* data) {
        return {static_cast<std::byte*>(data), program.state().continuation_bytes()};
    }

    bool capture(LaneRecord& entry, std::uint64_t* id, bool may_block, bool* deferred,
                 std::span<const std::uint64_t> keep, bool system = false,
                 bool complete_original = false, std::uint64_t live_owner = 0) {
        if (id != nullptr) { *id = 0; }
        if (deferred != nullptr) { *deferred = false; }
        if (!ram || !entry.lane ||
            (live_owner == 0 && (!entry.cache_write || !entry.allow_reuse))) {
            return false;
        }
        try {
            CUDA_CHECK(cudaEventSynchronize(entry.capture_done));
            const bool cut = live_owner == 0 && !system && !complete_original &&
                             entry.turn_closed && entry.rewrite &&
                             entry.rewrite->kind == qwen3_6::RewriteCheckpointKind::TurnClosure &&
                             entry.rewrite->generation_opener;
            const auto frontier =
                cut ? entry.rewrite->frontier : static_cast<std::uint32_t>(entry.tokens.size());
            std::vector<TokenId> ledger(entry.tokens.begin(), entry.tokens.begin() + frontier);
            ledger.push_back(frontier < entry.prompt.size() ? entry.prompt[frontier]
                                                            : std::max(entry.sampled, 0));
            auto identity = entry.identity;
            if (identity.size() < ledger.size()) {
                identity.append_generated(ledger.size() - identity.size(), 0);
            } else {
                identity.truncate(ledger.size());
            }
            if (!cut) { snapshot(entry, entry.tier_image); }
            tier::RamCaptureSource source;
            source.system_prompt             = system;
            source.live_owner                = live_owner;
            source.live_reserved_tail_bytes  = live_owner == 0 ? 0 : kSamplingCountsBytes;
            source.execution_frontier        = frontier;
            source.ledger_frontier           = frontier + 1;
            source.text_kv_valid             = frontier;
            source.tail_hidden_valid         = true;
            source.ledger                    = ledger;
            source.identity                  = &identity;
            source.hash_f                    = tier::prefix_hash_at(ledger, identity, frontier);
            source.text                      = &program.state().pages(*entry.lane);
            source.text_pool                 = &program.state().pool();
            source.text_pages                = pages_for_tokens(frontier);
            source.flash_next_state          = image(cut ? entry.rewrite_image : entry.tier_image);
            source.current_state.copies_done = entry.capture_done;
            if (!system && !cut && entry.rewrite) {
                source.rewrite_valid    = true;
                source.rewrite_kind     = entry.rewrite->kind;
                source.rewrite_frontier = entry.rewrite->frontier;
                source.hash_c_valid     = true;
                source.hash_c = tier::prefix_hash_at(ledger, identity, source.rewrite_frontier);
                source.flash_next_rewrite = image(entry.rewrite_image);
            }
            if (!system && entry.rollback_frontier != 0 && entry.rollback_frontier <= frontier) {
                tier::RamLadderHead head;
                head.frontier         = entry.rollback_frontier;
                head.hash             = tier::prefix_hash_at(ledger, identity, head.frontier);
                head.kind             = tier::ContextCheckpointKind::TurnRollback;
                head.flash_next_state = image(entry.rollback_image);
                head.copies_done      = entry.capture_done;
                source.ladder_heads.push_back(head);
            }
            source.disk_entry_id = entry.disk_entry_id;
            source.stream        = device.copy_stream;
            device.order_copy_after_compute();
            for (;;) {
                const auto result = ram->capture(source);
                if (result.status == tier::RamCaptureStatus::Captured) {
                    if (live_owner != 0) {
                        const auto [block, count] = ram->live_credit_view(live_owner);
                        auto* counts =
                            static_cast<std::byte*>(block) + count - kSamplingCountsBytes;
                        CUDA_CHECK(cudaMemcpyAsync(
                            counts, program.state().token_counts(*entry.lane), kSamplingCountsBytes,
                            cudaMemcpyDeviceToHost, device.copy_stream));
                    }
                    program.extents().record_tier_copy(entry.capture_done);
                    if (id != nullptr) { *id = result.entry_id; }
                    if (disk && !system && live_owner == 0) {
                        disk->note_ram_resident(result.entry_id, source.disk_entry_id);
                        if (!tiers_shutdown) { disk->request_idle_spill(); }
                    }
                    return true;
                }
                if (result.status == tier::RamCaptureStatus::Dropped) { return false; }
                if (disk) {
                    const auto reclaimed = disk->reclaim_ram_entry(may_block, keep);
                    if (reclaimed == tier::RamReclaim::Evicted ||
                        reclaimed == tier::RamReclaim::Retry) {
                        continue;
                    }
                    if (reclaimed == tier::RamReclaim::Pending) {
                        if (deferred != nullptr) { *deferred = true; }
                        return false;
                    }
                    ram->record_drop();
                    return false;
                }
                if (ram->retired_fence_pending()) {
                    if (!may_block && deferred != nullptr) {
                        *deferred = true;
                        return false;
                    }
                    if (may_block && ram->wait_retired_copies()) { continue; }
                }
                const auto victims = ram->unpinned_ids();
                const auto victim =
                    std::find_if(victims.begin(), victims.end(), [&](auto candidate) {
                        return std::find(keep.begin(), keep.end(), candidate) == keep.end();
                    });
                if (victim == victims.end()) {
                    if (system && ram->evict_system_prompt()) { continue; }
                    ram->record_drop();
                    return false;
                }
                if (!may_block && !ram->copies_ready(*victim)) {
                    if (deferred != nullptr) { *deferred = true; }
                    return false;
                }
                ram->wait_entry_copies(*victim);
                (void)ram->evict_one_unpinned(*victim);
            }
        } catch (const std::bad_alloc&) {
            ram->record_drop();
            return false;
        }
    }

    [[nodiscard]] runtime::PrefillStepResult run_prefill_step(LaneRecord& entry) {
        if (entry.phase != Phase::Prefill || !entry.lane || entry.appended > entry.prompt.size()) {
            throw std::logic_error("qwen3.8-flash-next lane has no prompt left to prefill");
        }
        const std::size_t begin = entry.appended;
        const std::size_t end   = std::min(entry.prompt.size(), begin + super_chunk_tokens());
        const auto started      = Clock::now();
        PrefillStep step;
        // Checkpoint boundaries are internal numerical boundaries, never scheduling boundaries.
        // Preserve the complete remaining suffix so ordinary super-chunks keep n-gram lookahead.
        do {
            const auto remaining = std::span<const TokenId>(entry.prompt).subspan(entry.appended);
            std::size_t stop     = end;
            auto split_at        = [&](std::uint32_t frontier) {
                if (frontier > entry.appended) { stop = std::min(stop, std::size_t{frontier}); }
            };
            for (const auto frontier : entry.closure_frontiers) { split_at(frontier); }
            if (entry.capture_rewrite) { split_at(entry.capture_rewrite->frontier); }
            if (entry.system_capture) { split_at(*entry.system_capture); }
            const bool capture = entry.capture_rewrite && stop == entry.capture_rewrite->frontier;
            const bool capture_tail =
                capture || (entry.system_capture && stop == *entry.system_capture);
            if (stop == entry.prompt.size()) { arm_token_mask(entry); }
            step =
                remaining.empty()
                    ? program.finish_prompt(*entry.lane)
                    : program.prefill_step(*entry.lane, remaining,
                                           static_cast<std::int32_t>(stop - entry.appended),
                                           capture_tail ? PrefillTail::Capture : PrefillTail::Skip);
            entry.tokens.insert(entry.tokens.end(), remaining.begin(),
                                remaining.begin() + step.consumed);
            entry.appended += static_cast<std::size_t>(step.consumed);
            if (entry.system_capture && entry.appended == *entry.system_capture) {
                (void)this->capture(entry, nullptr, true, nullptr, {}, true);
                // A partial final QSA page may be written by the next segment.
                CUDA_CHECK(cudaStreamWaitEvent(device.stream, entry.capture_done, 0));
                entry.system_capture.reset();
            }
            if (capture) {
                snapshot(entry, entry.rewrite_image);
                entry.rewrite = entry.capture_rewrite;
                entry.capture_rewrite.reset();
            }
        } while (entry.appended < end);
        const bool final     = entry.appended == entry.prompt.size();
        const auto processed = static_cast<std::uint32_t>(entry.appended - begin);
        const double seconds = seconds_since(started);
        entry.timings.prefill_seconds += seconds;
        entry.prefill_steps.emplace_back(processed, seconds);

        runtime::PrefillStepResult result;
        result.summary                 = entry.begin;
        result.processed_prompt_tokens = processed;
        // The prompt was copied when the lane opened; no host payload outlives it.
        result.host_input_consumed = true;
        result.complete            = final;
        if (!final) { return result; }

        if (step.next_token < 0 || step.next_token >= kTokenDomain) {
            throw std::logic_error("qwen3.8-flash-next prefill sampled outside the token domain");
        }
        entry.phase                 = Phase::Sampled;
        entry.sampled               = step.next_token;
        entry.sampled_logprob.at(0) = step.next_logprob;
        std::copy(step.top_ids.begin(), step.top_ids.end(), entry.top_ids.begin());
        std::copy(step.top_logprobs.begin(), step.top_logprobs.end(), entry.top_logprobs.begin());
        record_prefill_tail(entry);
        result.round.tokens = std::span<const TokenId>(&entry.sampled, 1);
        if (entry.token_logprobs) {
            result.round.logprobs = runtime::RoundLogprobs{
                .token_logprobs = entry.sampled_logprob,
                .top_ids        = entry.top_ids,
                .top_logprobs   = entry.top_logprobs,
            };
        }
        return result;
    }

    static void record_prefill_tail(LaneRecord& entry) {
        double seconds       = 0.0;
        std::uint64_t tokens = 0;
        for (auto step = entry.prefill_steps.rbegin(); step != entry.prefill_steps.rend(); ++step) {
            tokens += step->first;
            seconds += step->second;
            if (seconds >= kPrefillTailWindowSeconds) { break; }
        }
        entry.timings.prefill_tail_window_s = seconds;
        entry.timings.prefill_tail_tok_s =
            seconds > 0.0 ? static_cast<double>(tokens) / seconds : 0.0;
    }

    DeviceContext& device;
    const std::uint32_t capacity;
    const KvCacheStorage kv_cache;
    const std::uint64_t weights_bytes;
    const ProgramOptions program_options;
    const std::uint32_t image_slots;
    // Outlives Program, whose Forward destructor drains any copy queued before a request failed.
    std::unique_ptr<HostPinnedArena> checkpoint_slab;
    Program program;
    std::unique_ptr<tier::KVRamCache> ram;
    std::unique_ptr<tier::KVDiskCache> disk;
    std::uint32_t system_prompt_limit = 0;
    std::uint64_t pending_disk_ticket = 0;
    // Points into records while the restore ticket owns a destination's pages.
    LaneRecord* pending_disk_owner = nullptr;
    bool tiers_shutdown            = false;
    CudaEvent copy_cancel{cudaEventDisableTiming};
    std::uint64_t use_tick = 0;
    std::array<LaneFence, kMaximumConcurrency> lane_fences{};
    std::array<LaneRecord, kMaximumConcurrency> records{};
    std::unordered_map<std::uint64_t, Suspended> suspended;
    std::vector<std::uint32_t> mask_words =
        std::vector<std::uint32_t>(static_cast<std::size_t>(kAllowedTokenWords));
    // The Program call of a decode round over its compacted rows.
    static constexpr std::size_t kRoundTokens     = std::size_t{kMaximumConcurrency} * kDflashWidth;
    static constexpr std::size_t kRoundTopEntries = kRoundTokens * kMaximumTopLogprobs;
    std::array<Lane, kMaximumConcurrency> decode_lanes{};
    std::array<TokenId, kMaximumConcurrency> decode_inputs{};
    std::array<std::uint32_t, kMaximumConcurrency> decode_limits{}, pending_members{};
    std::size_t pending_size = 0;
    // Storage of the last decode round by membership row, which the executor reads until the
    // next round.
    std::array<TokenId, kRoundTokens> round_tokens{};
    std::array<float, kRoundTokens> round_logprobs{};
    std::array<std::int32_t, kMaximumConcurrency> round_counts{};
    std::array<TokenId, kRoundTopEntries> round_top_ids{};
    std::array<float, kRoundTopEntries> round_top_logprobs{};
};

static_assert(static_cast<std::uint32_t>(kTopLogprobs) == kMaximumTopLogprobs);
static_assert(static_cast<std::uint32_t>(kMaxLanes) == kMaximumConcurrency);
static_assert(static_cast<std::size_t>(kTokenDomain) == qwen3_6::kTokenDomain);

EngineProgram::EngineProgram(const LoadedModel& model, const EngineOptions& options,
                             DeviceContext& device)
    : impl_(std::make_unique<Impl>(model, options, device)) {}

EngineProgram::~EngineProgram() = default;

RequestBasePlan EngineProgram::plan_request_base(const PreparedPrompt& prompt,
                                                 const runtime::ResolvedExecutionOptions& options) {
    const qwen3_6::PreparedPromptData& data = qwen3_6::PreparedPromptAccess::view(prompt);
    if (data.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (data.token_ids.size() > impl_->capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (data.has_media()) { throw std::invalid_argument("qwen3.8-flash-next has no Vision"); }
    for (const TokenId id : data.token_ids) {
        if (id < 0 || id >= kTokenDomain) {
            throw std::invalid_argument("prompt contains a token outside the 248077-token domain");
        }
    }
    validate_sampling(options.sampling);
    if (options.capture_context_checkpoint) {
        throw std::invalid_argument("qwen3.8-flash-next has no context checkpoints");
    }
    if (options.suppressed_token_count > ops::SamplingConfig::kMaximumSuppressedTokens) {
        throw std::invalid_argument("too many suppressed sampling tokens");
    }

    RequestBasePlan base;
    runtime::RequestPlanSummary& summary = base.summary_;
    summary.prompt_tokens                = static_cast<std::uint32_t>(data.token_ids.size());
    summary.requested_output_tokens      = options.requested_output_tokens;
    const std::uint32_t capacity_output  = impl_->capacity - summary.prompt_tokens + 1U;
    summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
    summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                          ? FinishReason::OutputLimit
                                          : FinishReason::ContextCapacity;
    // Every output token but the last is appended to the lane.
    const auto lifetime_tokens =
        summary.prompt_tokens + std::max(summary.effective_output_tokens, 1U) - 1U;
    summary.lifetime_admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = pages_for_tokens(lifetime_tokens),
        .backend_kv_pages = 0,
    };
    if (impl_->ram && summary.effective_output_tokens != 0 &&
        options.reservation_window_tokens == 0) {
        throw std::invalid_argument("Flash-Next RAM-on admission needs a positive output window");
    }
    const auto window_tokens =
        impl_->ram ? std::min(summary.effective_output_tokens, options.reservation_window_tokens)
                   : summary.effective_output_tokens;
    const auto window            = impl_->execution_window(summary.prompt_tokens, window_tokens);
    base.reserve_tokens_         = summary.prompt_tokens + std::max(window_tokens, 1U) - 1U;
    summary.admission            = window.admission;
    summary.suspension_bytes     = window.suspension_bytes;
    summary.window_output_tokens = window_tokens;
    summary.transient_bytes      = 0;
    summary.transient_alignment  = 1;
    // One quantum per prefill step (super-chunk) and per decoded token. A generation-recovery
    // retry re-prefills a prompt bounded by the reserved context, at most maximum_attempts times.
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0U : summary.effective_output_tokens - 1U;
    summary.service_work_quanta = impl_->prefill_steps(summary.prompt_tokens) + decode_units;
    if (data.generation_recovery) {
        summary.service_work_quanta += qwen3_6::GenerationRecoveryContext::maximum_attempts *
                                       (impl_->prefill_steps(lifetime_tokens) + 2U);
    }

    base.sampling_ = options.sampling;
    base.suppressed_tokens_.reserve(options.suppressed_token_count);
    for (std::uint32_t i = 0; i < options.suppressed_token_count; ++i) {
        const TokenId token = options.suppressed_token_ids.at(i);
        if (token < 0 || token >= kTokenDomain) {
            throw std::invalid_argument("suppressed sampling token is outside the vocabulary");
        }
        base.suppressed_tokens_.push_back(token);
    }
    base.token_logprobs_ = options.token_logprobs;
    base.allow_reuse_    = options.allow_prefix_reuse && data.identity.reusable;
    base.force_cold_     = options.force_cold_prefill;
    base.cache_write_    = options.cache_write;
    if (options.cache_write && options.allow_prefix_reuse && data.identity.reusable) {
        base.rewrite_ = data.identity.rewrite_checkpoint;
    }
    return base;
}

RequestPlan EngineProgram::plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& prompt,
                                                 const RequestBasePlan& base) {
    const auto& entry = impl_->record(lane);
    RequestPlan plan;
    plan.base_ = base;
    plan.lane_ = lane;
    if (!base.allow_reuse_ || base.force_cold_ || entry.phase != Impl::Phase::Retained) {
        return plan;
    }
    const auto& incoming = qwen3_6::PreparedPromptAccess::view(prompt).token_ids;
    auto matches         = [&](std::size_t frontier) {
        return frontier > 0 && frontier <= incoming.size() && frontier <= entry.tokens.size() &&
               std::equal(entry.tokens.begin(),
                          entry.tokens.begin() + static_cast<std::ptrdiff_t>(frontier),
                          incoming.begin());
    };
    auto& summary = plan.base_.summary_;
    if (matches(entry.tokens.size())) {
        summary.reusable_prompt_tokens = static_cast<std::uint32_t>(entry.tokens.size());
        summary.disk_reuse_path        = PrefixReusePath::AppendAtFrontier;
    } else {
        if (entry.rewrite && matches(entry.rewrite->frontier)) {
            summary.reusable_prompt_tokens = entry.rewrite->frontier;
            summary.disk_reuse_path =
                entry.rewrite->kind == qwen3_6::RewriteCheckpointKind::TurnClosure
                    ? PrefixReusePath::RestoreTurnCheckpoint
                    : PrefixReusePath::RestoreResponseCheckpoint;
        }
        if (entry.rollback_frontier > summary.reusable_prompt_tokens &&
            matches(entry.rollback_frontier)) {
            summary.reusable_prompt_tokens = entry.rollback_frontier;
            summary.disk_reuse_path        = PrefixReusePath::RestoreTurnRollback;
        }
    }
    if (summary.reusable_prompt_tokens != 0) {
        summary.reuse_source = PrefixReuseSource::VramResident;
        // Prefill finalization is one scheduling quantum even when the suffix is empty.
        summary.service_work_quanta -= impl_->prefill_steps(summary.prompt_tokens);
        summary.service_work_quanta += std::max<std::uint64_t>(
            1, impl_->prefill_steps(summary.prompt_tokens - summary.reusable_prompt_tokens));
    }
    return plan;
}

RequestPlan EngineProgram::plan_ram_reuse(const PreparedPrompt& prompt,
                                          const RequestBasePlan& base) {
    RequestPlan plan;
    plan.base_ = base;
    if (!impl_->ram || !base.allow_reuse_ || base.force_cold_) { return plan; }
    const auto& data = qwen3_6::PreparedPromptAccess::view(prompt);
    const auto match = impl_->ram->plan_match(data, tier::prefix_hash_chain(data));
    if (!match) { return plan; }
    auto& summary                  = plan.base_.summary_;
    summary.reuse_source           = PrefixReuseSource::HostRam;
    summary.ram_entry_id           = match->entry_id;
    summary.reusable_prompt_tokens = match->reuse_base;
    summary.disk_reuse_path        = match->reuse;
    summary.service_work_quanta -= impl_->prefill_steps(summary.prompt_tokens);
    summary.service_work_quanta +=
        std::max<std::uint64_t>(1, impl_->prefill_steps(summary.prompt_tokens - match->reuse_base));
    return plan;
}

RequestPlan EngineProgram::plan_disk_reuse(const PreparedPrompt& prompt,
                                           const RequestBasePlan& base) {
    RequestPlan plan;
    plan.base_ = base;
    if (!impl_->disk || !base.allow_reuse_ || base.force_cold_) { return plan; }
    const auto& data = qwen3_6::PreparedPromptAccess::view(prompt);
    const auto match = impl_->disk->plan_match(data, tier::prefix_hash_chain(data));
    if (!match) { return plan; }
    auto& summary                     = plan.base_.summary_;
    summary.reuse_source              = PrefixReuseSource::HostDisk;
    summary.disk_entry_id             = match->entry_id;
    summary.reusable_prompt_tokens    = match->reuse_base;
    summary.disk_reuse_path           = match->reuse;
    summary.disk_execution_frontier   = match->execution_frontier;
    summary.disk_hash_f_lo            = match->hash_f.lo;
    summary.disk_hash_f_hi            = match->hash_f.hi;
    summary.disk_committed_generation = match->committed_generation;
    summary.service_work_quanta -= impl_->prefill_steps(summary.prompt_tokens);
    summary.service_work_quanta +=
        std::max<std::uint64_t>(1, impl_->prefill_steps(summary.prompt_tokens - match->reuse_base));
    return plan;
}

bool EngineProgram::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= static_cast<std::uint32_t>(impl_->program.lanes())) { return false; }
    const auto& entry    = impl_->records.at(lane);
    const bool consuming = plan.base_.cache_write_ && plan.lane_ == lane &&
                           plan.summary().reuse_source == PrefixReuseSource::VramResident;
    const bool empty     = entry.phase == Impl::Phase::Idle && !entry.lane;
    if (!empty && !(consuming && entry.phase == Impl::Phase::Retained)) { return false; }
    const std::int64_t credit = consuming ? pages_for_tokens(entry.reserve_tokens) : 0;
    return pages_for_tokens(plan.base_.reserve_tokens_) <=
           impl_->program.kv_free_tokens() / ops::kQsaPageTokens + credit;
}

bool EngineProgram::can_admit_lane_after_releasing(
    std::uint32_t lane, const RequestPlan& plan,
    std::span<const std::uint32_t> release_lanes) const noexcept {
    if (lane >= static_cast<std::uint32_t>(impl_->program.lanes())) { return false; }
    const auto& entry = impl_->records.at(lane);
    if (entry.phase != Impl::Phase::Idle && entry.phase != Impl::Phase::Retained) { return false; }
    const bool fork =
        !plan.base_.cache_write_ && plan.summary().reuse_source == PrefixReuseSource::VramResident;
    if (fork && plan.lane_ == lane && has_retained_lane(lane) &&
        (!impl_->ram || !retained_lane_cache_writable(lane))) {
        // The executor secures a complete-original RAM image before releasing this source.
        return false;
    }
    std::int64_t available = impl_->program.kv_free_tokens() / ops::kQsaPageTokens;
    if (has_retained_lane(lane)) { available += pages_for_tokens(entry.reserve_tokens); }
    std::array<bool, kMaximumConcurrency> counted{};
    counted.at(lane) = true;
    for (const auto victim : release_lanes) {
        if (victim >= static_cast<std::uint32_t>(impl_->program.lanes()) || counted.at(victim) ||
            (fork && victim == plan.lane_)) {
            continue;
        }
        counted.at(victim) = true;
        if (has_retained_lane(victim)) {
            available += pages_for_tokens(impl_->records.at(victim).reserve_tokens);
        }
    }
    return pages_for_tokens(plan.base_.reserve_tokens_) <= available;
}

bool EngineProgram::can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                           const RequestPlan& plan) const noexcept {
    std::array<std::uint32_t, kMaximumConcurrency> victims{};
    const auto count = static_cast<std::uint32_t>(impl_->program.lanes());
    for (std::uint32_t i = 0; i < count; ++i) { victims.at(i) = i; }
    return can_admit_lane_after_releasing(lane, plan, std::span(victims).first(count));
}

runtime::AdmissionResources EngineProgram::admission_capacity() const noexcept {
    return runtime::AdmissionResources{
        .active_lanes = static_cast<std::uint32_t>(impl_->program.lanes()),
        .main_kv_pages =
            static_cast<std::uint32_t>(impl_->program.kv_capacity_tokens() / ops::kQsaPageTokens),
        .backend_kv_pages = 0,
    };
}

#include "execution_windows_impl.h"

runtime::PrefillStepResult
EngineProgram::start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt, RequestPlan&& plan,
                                  runtime::TransientRegion transient, runtime::PrefillPace /*pace*/,
                                  const OutputSession* output) {
    Impl::LaneRecord& entry = impl_->record(lane);
    const RequestPlan owned = std::move(plan);
    const auto reuse        = owned.summary().reusable_prompt_tokens;
    if (entry.phase != Impl::Phase::Idle && entry.phase != Impl::Phase::Retained) {
        throw std::logic_error("qwen3.8-flash-next prefill lane is occupied");
    }
    const bool host_reuse = owned.summary().reuse_source == PrefixReuseSource::HostRam ||
                            owned.summary().reuse_source == PrefixReuseSource::HostDisk;
    if ((!host_reuse && owned.base_.cache_write_ && owned.lane_ != lane) || transient) {
        throw std::logic_error("qwen3.8-flash-next prefill plan does not match its lane");
    }
    qwen3_6::PreparedPromptData data = qwen3_6::PreparedPromptAccess::take(std::move(prompt));
    const RequestBasePlan& base      = owned.base_;
    if (data.token_ids.size() != base.summary_.prompt_tokens) {
        throw std::logic_error("qwen3.8-flash-next prefill prompt differs from its plan");
    }

    LaneSampling sampling;
    sampling.temperature       = base.sampling_.temperature;
    sampling.top_k             = base.sampling_.top_k;
    sampling.top_p             = base.sampling_.top_p;
    sampling.min_p             = base.sampling_.min_p;
    sampling.presence_penalty  = base.sampling_.presence_penalty;
    sampling.frequency_penalty = base.sampling_.frequency_penalty;
    sampling.p_less            = base.sampling_.p_less;
    sampling.seed              = base.sampling_.seed;
    sampling.suppressed_tokens = base.suppressed_tokens_;
    sampling.top_logprobs      = base.token_logprobs_;

    const void* image        = nullptr;
    const bool tier_restored = entry.tier_restored;
    if (tier_restored) {
        if (!host_reuse || reuse == 0 || !entry.lane) {
            throw std::logic_error("Flash-Next host restore lost its winning plan");
        }
        wait_kv_ram_copies_on_compute();
        image = entry.tier_image;
    } else if (reuse != 0 && base.cache_write_) {
        if (!entry.lane || entry.phase != Impl::Phase::Retained) {
            throw std::logic_error("Flash-Next retained plan lost its source");
        }
        if (owned.summary().disk_reuse_path == PrefixReusePath::RestoreTurnRollback) {
            image = entry.rollback_image;
        } else if (owned.summary().disk_reuse_path != PrefixReusePath::AppendAtFrontier) {
            image = entry.rewrite_image;
        }
    }
    if (reuse == 0 && entry.lane) { impl_->close(entry); }
    entry.generated.clear();
    entry.sampled_count = 0;
    entry.speculative   = {};
    entry.timings       = {};
    entry.prefill_steps.clear();
    entry.typical_cycle_reasoning = false;
    entry.token_logprobs          = base.token_logprobs_;
    entry.reserve_tokens          = base.reserve_tokens_;
    entry.window_output_tokens    = base.summary_.window_output_tokens;
    entry.effective_output_tokens = base.summary_.effective_output_tokens;
    entry.output                  = output;
    entry.closure_frontiers       = std::move(data.turn_closure_frontiers);
    entry.identity.assign(data);
    entry.system_capture.reset();
    if (impl_->ram && impl_->system_prompt_limit != 0 && base.cache_write_ && base.allow_reuse_ &&
        data.system_prompt_frontier && *data.system_prompt_frontier > reuse) {
        const auto hashes = tier::prefix_hash_chain(data);
        if (!impl_->ram->has_system_prompt(data, hashes.at(*data.system_prompt_frontier))) {
            entry.system_capture = data.system_prompt_frontier;
        }
    }
    entry.prompt        = std::move(data.token_ids);
    entry.tier_restored = false;
    entry.turn_closed   = false;
    entry.appended      = reuse;
    entry.allow_reuse   = base.allow_reuse_;
    entry.cache_write   = base.cache_write_;
    entry.begin         = runtime::BeginSummary{
        .prompt_tokens        = base.summary_.prompt_tokens,
        .reused_prompt_tokens = reuse,
        .prefix_reuse_path    = owned.summary().disk_reuse_path,
        .prefix_reuse_source  = owned.summary().reuse_source,
    };
    if (reuse != 0) {
        impl_->program.extents().wait_tier_copy(entry.capture_done);
        if (tier_restored) {
            impl_->program.restore_lane(entry.lane.value(), std::span(entry.prompt).first(reuse),
                                        sampling, image);
        } else {
            impl_->program.reuse_lane(entry.lane.value(), entry.reserve_tokens,
                                      std::span(entry.prompt).first(reuse), sampling, image);
        }
        entry.tokens.assign(entry.prompt.begin(), entry.prompt.begin() + reuse);
    } else {
        entry.tokens.clear();
        entry.rewrite.reset();
        entry.lane = impl_->program.open_lane(entry.reserve_tokens, sampling);
    }
    // Never snapshot a disposable branch. A selected checkpoint may be kept only when its exact
    // prefix is still present; otherwise the requested checkpoint is captured while prefilling.
    if (base.cache_write_ && base.rewrite_ && entry.rewrite && entry.rewrite->frontier <= reuse) {
        if (entry.rewrite->frontier == base.rewrite_->frontier) { entry.rewrite = base.rewrite_; }
    } else {
        entry.rewrite.reset();
    }
    entry.capture_rewrite =
        base.rewrite_ && base.rewrite_->frontier > reuse ? base.rewrite_ : std::nullopt;
    entry.rollback_frontier = 0;
    if (reuse != 0 && base.cache_write_ && entry.prompt.size() > reuse) {
        impl_->snapshot(entry, entry.rollback_image);
        entry.rollback_frontier = reuse;
    }
    if (!entry.lane) {
        impl_->close(entry);
        throw std::logic_error("qwen3.8-flash-next admitted a lane the Program cannot open");
    }
    entry.sampling = std::move(sampling);
    entry.phase    = Impl::Phase::Prefill;
    auto result    = impl_->run_prefill_step(entry);
    entry.use_tick = ++impl_->use_tick;
    return result;
}

runtime::PrefillStepResult EngineProgram::advance_prefill_lane(std::uint32_t lane,
                                                               runtime::PrefillPace /*pace*/) {
    return impl_->run_prefill_step(impl_->record(lane));
}

runtime::BatchedGeneratedRound
EngineProgram::decode_batch(std::span<const std::uint32_t> lanes,
                            std::span<const runtime::RoundBudget> budgets) {
    Impl& self = *impl_;
    if (lanes.empty() || lanes.size() > static_cast<std::size_t>(self.program.lanes()) ||
        budgets.size() != lanes.size()) {
        throw std::invalid_argument("qwen3.8-flash-next decode membership is invalid");
    }
    if (self.pending_size != 0) {
        throw std::logic_error("Flash-Next decode packet is unresolved");
    }
    runtime::BatchedGeneratedRound round;
    self.round_counts.fill(0);
    bool logprobs = false;
    // Rows that reach the Program, compacted; row_of maps them back to membership rows.
    std::array<std::size_t, kMaximumConcurrency> row_of{};
    std::size_t sampled_rows = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        Impl::LaneRecord& entry = self.record(lanes[row]);
        if (std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row),
                      lanes[row]) != lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("qwen3.8-flash-next decode repeats a lane");
        }
        if (entry.phase != Impl::Phase::Ready || !entry.lane || entry.generated.empty() ||
            budgets[row].generated_tokens_remaining == 0 ||
            self.program.committed(*entry.lane) >=
                static_cast<std::int32_t>(entry.reserve_tokens)) {
            throw std::logic_error("qwen3.8-flash-next decode row is not decode-ready");
        }
        self.arm_typical_exclude(entry);
        try {
            self.arm_token_mask(entry);
        } catch (const RequestError&) {
            // The request's output constraint admits no token: it fails alone and its lane
            // closes before the round (BatchedGeneratedRound::row_errors).
            round.row_errors.at(row) = std::current_exception();
            self.close(entry);
            continue;
        }
        round.cycle_exclusions.at(row)        = entry.sampling.typical_exclude >= 0;
        self.decode_lanes.at(sampled_rows)    = *entry.lane;
        self.decode_inputs.at(sampled_rows)   = entry.generated.back();
        self.decode_limits.at(sampled_rows)   = budgets[row].generated_tokens_remaining;
        self.pending_members.at(sampled_rows) = lanes[row];
        row_of.at(sampled_rows++)             = row;
        logprobs                              = logprobs || entry.token_logprobs;
    }

    const std::size_t batch = lanes.size();
    std::size_t width       = 1;
    if (sampled_rows != 0) {
        const auto started = Clock::now();
        const auto packet  = self.program.decode_round(
            std::span<const Lane>(self.decode_lanes).first(sampled_rows),
            std::span<const TokenId>(self.decode_inputs).first(sampled_rows),
            std::span<const std::uint32_t>(self.decode_limits).first(sampled_rows),
            logprobs ? RoundLogprobs::Report : RoundLogprobs::Skip);
        const double seconds = seconds_since(started);
        width                = static_cast<std::size_t>(packet.width);
        const bool deferred  = self.program.pending_speculative();
        for (std::size_t sampled = 0; sampled < sampled_rows; ++sampled) {
            const std::size_t row   = row_of.at(sampled);
            Impl::LaneRecord& entry = self.records.at(lanes[row]);
            const auto count        = packet.counts[sampled];
            if (count < 1 || count > packet.width) {
                throw std::logic_error("Flash-Next returned an invalid licensed packet extent");
            }
            entry.pending_input    = self.decode_inputs[sampled];
            entry.pending_deferred = deferred;
            entry.sampled_count    = static_cast<std::uint32_t>(count);
            if (!deferred) { entry.tokens.push_back(entry.pending_input); }
            entry.phase = Impl::Phase::Sampled;
            entry.timings.decode_seconds += seconds;
            self.round_counts[row] = count;
            for (std::size_t column = 0; column < static_cast<std::size_t>(count); ++column) {
                const auto from     = sampled * width + column;
                const auto to       = row * width + column;
                const TokenId token = packet.tokens[from];
                if (token < 0 || token >= kTokenDomain) {
                    throw std::logic_error("Flash-Next sampled outside the token domain");
                }
                entry.sampled_tokens[column] = token;
                self.round_tokens[to]        = token;
                if (logprobs) {
                    self.round_logprobs[to] = packet.logprobs[from];
                    std::copy_n(packet.top_ids.begin() +
                                    static_cast<std::ptrdiff_t>(from * kMaximumTopLogprobs),
                                kMaximumTopLogprobs,
                                self.round_top_ids.begin() +
                                    static_cast<std::ptrdiff_t>(to * kMaximumTopLogprobs));
                    std::copy_n(packet.top_logprobs.begin() +
                                    static_cast<std::ptrdiff_t>(from * kMaximumTopLogprobs),
                                kMaximumTopLogprobs,
                                self.round_top_logprobs.begin() +
                                    static_cast<std::ptrdiff_t>(to * kMaximumTopLogprobs));
                }
            }
            entry.sampled = entry.sampled_tokens[0];
        }
        self.pending_size = sampled_rows;
    }
    round.tokens     = std::span<const TokenId>(self.round_tokens).first(batch * width);
    round.row_counts = std::span<const std::int32_t>(self.round_counts).first(batch);
    round.row_stride = static_cast<std::uint32_t>(width);
    if (logprobs) {
        round.logprobs = runtime::RoundLogprobs{
            .token_logprobs = std::span<const float>(self.round_logprobs).first(batch * width),
            .top_ids        = std::span<const TokenId>(self.round_top_ids)
                                  .first(batch * width * kMaximumTopLogprobs),
            .top_logprobs   = std::span<const float>(self.round_top_logprobs)
                                  .first(batch * width * kMaximumTopLogprobs),
        };
    }
    return round;
}

std::uint32_t EngineProgram::prefill_remaining_tokens(std::uint32_t lane) const {
    const Impl::LaneRecord& entry = impl_->record(lane);
    if (entry.phase != Impl::Phase::Prefill) { return 0; }
    return static_cast<std::uint32_t>(entry.prompt.size() - entry.appended);
}

bool EngineProgram::prefill_suspendable(std::uint32_t lane) const {
    return impl_->record(lane).phase == Impl::Phase::Prefill;
}

void EngineProgram::resume_prefill_lane(std::uint32_t lane) const {
    if (impl_->record(lane).phase != Impl::Phase::Prefill) {
        throw std::logic_error("qwen3.8-flash-next resumed a lane that is not prefilling");
    }
}

runtime::MixedGeneratedRound
EngineProgram::decode_batch_with_prefill(std::span<const std::uint32_t> /*lanes*/,
                                         std::span<const runtime::RoundBudget> /*budgets*/,
                                         std::uint32_t /*owner_lane*/) {
    unsupported("mixed prefill and decode forward");
}

void EngineProgram::set_suppressed_tokens_lane(std::uint32_t lane,
                                               std::span<const TokenId> tokens) {
    Impl::LaneRecord& entry = impl_->record(lane);
    if (!entry.lane) { throw std::logic_error("qwen3.8-flash-next lane is not open"); }
    if (tokens.size() > static_cast<std::size_t>(ops::SamplingConfig::kMaximumSuppressedTokens)) {
        throw std::invalid_argument("too many suppressed sampling tokens");
    }
    for (const TokenId token : tokens) {
        if (token < 0 || token >= kTokenDomain) {
            throw std::invalid_argument("suppressed sampling token is outside the vocabulary");
        }
    }
    entry.sampling.suppressed_tokens.assign(tokens.begin(), tokens.end());
    impl_->program.set_sampling(*entry.lane, entry.sampling);
}

void EngineProgram::clear_suppressed_tokens_lane(std::uint32_t lane) {
    set_suppressed_tokens_lane(lane, {});
}

void EngineProgram::set_typical_cycle_reasoning_lane(std::uint32_t lane, bool enabled) {
    impl_->record(lane).typical_cycle_reasoning = enabled;
}

void EngineProgram::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    Impl::LaneRecord& entry = impl_->record(lane);
    if (entry.phase != Impl::Phase::Sampled) {
        throw std::logic_error("qwen3.8-flash-next prefill lane has no sampled token");
    }
    entry.generated.push_back(entry.sampled);
    entry.speculative = impl_->program.speculative_stats(entry.lane.value());
    if (terminal) {
        (void)impl_->retain(entry);
        return;
    }
    entry.phase = Impl::Phase::Ready;
}

void EngineProgram::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                          std::span<const std::uint32_t> accepted_tokens,
                                          std::span<const std::uint8_t> terminal,
                                          std::span<const std::uint8_t> cancelled,
                                          std::span<const std::uint8_t> rejected) {
    if (lanes.empty() || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        (!rejected.empty() && rejected.size() != lanes.size())) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }
    Impl& self = *impl_;
    if (lanes.size() != self.pending_size) {
        throw std::logic_error("Flash-Next resolution differs from its compact pending membership");
    }
    std::array<std::int32_t, kMaximumConcurrency> counts{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const Impl::LaneRecord& entry = self.record(lanes[row]);
        const bool discard            = !rejected.empty() && rejected[row] != 0;
        const auto count              = accepted_tokens[row];
        if (lanes[row] != self.pending_members[row] || entry.phase != Impl::Phase::Sampled ||
            entry.pending_deferred != self.program.pending_speculative() ||
            count > entry.sampled_count || ((cancelled[row] != 0 || discard) && count != 0) ||
            (cancelled[row] != 0 && terminal[row] == 0) ||
            (discard && (!entry.pending_deferred || terminal[row] != 0 || cancelled[row] != 0)) ||
            (count == 0 && cancelled[row] == 0 && !discard)) {
            throw std::logic_error("Flash-Next publication is not a valid licensed prefix");
        }
        counts[row] = static_cast<std::int32_t>(count);
    }
    const auto started = Clock::now();
    self.program.resolve_round(std::span<const std::int32_t>(counts).first(lanes.size()));
    const double seconds = seconds_since(started);
    self.pending_size    = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        Impl::LaneRecord& entry = self.record(lanes[row]);
        const auto count        = accepted_tokens[row];
        entry.timings.decode_seconds += seconds;
        entry.speculative = self.program.speculative_stats(entry.lane.value());
        if (entry.pending_deferred && count != 0) {
            // n licensed outputs execute the old anchor followed by their first n-1 IDs.
            entry.tokens.push_back(entry.pending_input);
            entry.tokens.insert(entry.tokens.end(), entry.sampled_tokens.begin(),
                                entry.sampled_tokens.begin() +
                                    static_cast<std::ptrdiff_t>(count - 1));
        }
        if (count != 0) {
            entry.generated.insert(entry.generated.end(), entry.sampled_tokens.begin(),
                                   entry.sampled_tokens.begin() +
                                       static_cast<std::ptrdiff_t>(count));
            entry.sampled = entry.generated.back();
        } else if (entry.pending_deferred) {
            entry.sampled = entry.generated.back();
        }
        entry.sampled_count = 0;
        if (terminal[row] != 0) {
            (void)self.retain(entry);
        } else {
            entry.phase = Impl::Phase::Ready;
        }
    }
}

void EngineProgram::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= kMaximumConcurrency) { return; }
    // The executor aborts only once an outcome is decided (cancellation, a failed request, or
    // fail_all after a device fault). Releasing the Program lane issues device work; if that
    // fails, the device error is sticky and the next Program call reports it, while the record
    // is already Idle.
    try {
        impl_->close(impl_->records.at(lane));
        // NOLINTNEXTLINE(bugprone-empty-catch): reported by the next device call; see above.
    } catch (...) {}
}

bool EngineProgram::retain_reusable_lane(std::uint32_t lane) {
    return impl_->retain(impl_->record(lane));
}

bool EngineProgram::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < static_cast<std::uint32_t>(impl_->program.lanes()) &&
           impl_->records.at(lane).phase == Impl::Phase::Retained;
}

bool EngineProgram::retained_lane_cache_writable(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) && impl_->records.at(lane).cache_write;
}

bool EngineProgram::retained_lane_precedes(std::uint32_t left, std::uint32_t right) const noexcept {
    const auto& a = impl_->records.at(left);
    const auto& b = impl_->records.at(right);
    if (a.cache_write != b.cache_write) { return !a.cache_write; }
    return a.use_tick < b.use_tick || (a.use_tick == b.use_tick && left < right);
}

void EngineProgram::evict_retained_lane(std::uint32_t lane) {
    if (has_retained_lane(lane)) { impl_->close(impl_->record(lane)); }
}

void EngineProgram::mark_turn_closed(std::uint32_t lane) noexcept {
    impl_->records.at(lane).turn_closed = true;
}

bool EngineProgram::copy_reusable_prompt(std::uint32_t lane, std::uint32_t prompt_tokens,
                                         std::vector<TokenId>& tokens,
                                         std::uint32_t& rewrite_frontier) const {
    tokens.clear();
    rewrite_frontier  = 0;
    const auto& entry = impl_->record(lane);
    if (!entry.lane || entry.tokens.size() < prompt_tokens) { return false; }
    tokens.assign(entry.tokens.begin(), entry.tokens.begin() + prompt_tokens);
    if (entry.rewrite) { rewrite_frontier = entry.rewrite->frontier; }
    return true;
}

bool EngineProgram::revert_cancelled_prefill_lane(std::uint32_t lane) {
    auto& entry = impl_->record(lane);
    if (entry.phase != Impl::Phase::Prefill || !entry.allow_reuse || !entry.cache_write) {
        return false;
    }
    const auto frontier = entry.rollback_frontier != 0
                              ? entry.rollback_frontier
                              : (entry.rewrite ? entry.rewrite->frontier : 0U);
    if (frontier == 0) { return false; }
    const void* image = entry.rollback_frontier != 0 ? entry.rollback_image : entry.rewrite_image;
    impl_->program.reuse_lane(entry.lane.value(), frontier, std::span(entry.tokens).first(frontier),
                              entry.sampling, image);
    entry.tokens.resize(frontier);
    if (entry.rewrite && entry.rewrite->frontier > frontier) { entry.rewrite.reset(); }
    return impl_->retain(entry);
}

void EngineProgram::fork_retained_lane(std::uint32_t destination, std::uint32_t source,
                                       const RequestPlan& plan) {
    auto& from          = impl_->record(source);
    auto& to            = impl_->record(destination);
    const auto frontier = plan.summary().reusable_prompt_tokens;
    if (destination == source || !has_retained_lane(source) || to.lane || plan.base_.cache_write_ ||
        frontier == 0 || plan.lane_ != source) {
        throw std::logic_error(
            "Flash-Next fork requires an independent destination and retained source");
    }
    to.lane = impl_->program.open_lane(plan.base_.reserve_tokens_, LaneSampling{});
    if (!to.lane) {
        throw std::logic_error("Flash-Next admitted fork cannot allocate its destination");
    }
    impl_->program.state().copy_prefix(to.lane.value(), from.lane.value(),
                                       static_cast<std::int32_t>(frontier), impl_->device.stream);
    const auto path = plan.summary().disk_reuse_path;
    if (path != PrefixReusePath::AppendAtFrontier) {
        impl_->program.state().unpack_continuation(
            *to.lane,
            path == PrefixReusePath::RestoreTurnRollback ? from.rollback_image : from.rewrite_image,
            impl_->device.stream);
    }
    // Completion protects the original pages and checkpoint images before the executor can
    // admit another reader or recycle their lane.
    CUDA_CHECK(cudaStreamSynchronize(impl_->device.stream));
}

#include "kv_tiers_impl.h"

qwen3_6::detail::KvGpuSnapshot EngineProgram::kv_gpu_snapshot() const noexcept {
    const std::int64_t capacity = impl_->program.kv_capacity_tokens();
    const std::int64_t free     = impl_->program.kv_free_tokens();
    const auto pages            = [](std::int64_t tokens) {
        return static_cast<std::uint32_t>(tokens / ops::kQsaPageTokens);
    };
    qwen3_6::detail::KvGpuSnapshot out;
    out.main.page_group_count = pages(capacity);
    out.main.entitled_pages   = pages(capacity - free);
    out.main.mapped_pages     = out.main.entitled_pages;
    out.main.free_pages       = pages(free);
    return out;
}

void EngineProgram::synchronize_all() { impl_->device.synchronize_all(); }

SpeculativeStats EngineProgram::speculative_stats_lane(std::uint32_t lane) const {
    return impl_->record(lane).speculative;
}

GenerationTimings EngineProgram::generation_timings_lane(std::uint32_t lane) const noexcept {
    if (lane >= kMaximumConcurrency) { return {}; }
    return impl_->records.at(lane).timings;
}

MemorySummary EngineProgram::memory_summary() const noexcept {
    const ProgramMemory memory = impl_->program.memory();
    MemorySummary out;
    out.device      = impl_->device.device;
    out.max_context = impl_->capacity;
    out.kv_capacity = static_cast<std::uint32_t>(impl_->program.kv_capacity_tokens());
    out.kv_cache    = impl_->kv_cache;
    out.weights =
        ArenaMemorySummary{impl_->weights_bytes, impl_->weights_bytes, impl_->weights_bytes};
    out.sequence = ArenaMemorySummary{memory.state_bytes, memory.state_bytes, memory.state_bytes};
    // The prefill streaming windows borrow expert-cache slots (ProgramMemory::window_bytes lies
    // inside expert_cache_bytes), so they are not workspace.
    const std::size_t workspace     = memory.activation_bytes + memory.workspace_bytes;
    out.workspace                   = ArenaMemorySummary{workspace, workspace, workspace};
    out.kv_payload_bytes            = memory.state_bytes;
    out.expert_cache_bytes          = memory.expert_cache_bytes;
    out.expert_cache_flex_bytes     = memory.flex_cache_bytes;
    out.checkpoint_image_host_bytes = impl_->checkpoint_slab->capacity();
    return out;
}

ScoreResult EngineProgram::score(PreparedPrompt&& prompt, RequestPlan&& plan,
                                 runtime::TransientRegion transient, ScoreOptions options) {
    const RequestPlan owned = std::move(plan);
    if (transient) { throw std::logic_error("qwen3.8-flash-next score has no transient region"); }
    if (options.schedule != ScoreSchedule::Prefill) {
        throw std::invalid_argument("qwen3.8-flash-next scores only with the prefill schedule");
    }
    const auto started                     = Clock::now();
    const qwen3_6::PreparedPromptData data = qwen3_6::PreparedPromptAccess::take(std::move(prompt));
    const auto prompt_tokens               = static_cast<std::uint32_t>(data.token_ids.size());
    if (prompt_tokens != owned.summary().prompt_tokens) {
        throw std::logic_error("qwen3.8-flash-next score prompt differs from its plan");
    }
    const std::uint32_t skip = resolve_score_skip(prompt_tokens, options.skip_tokens);

    ScoreResult result;
    result.schedule      = options.schedule;
    result.prompt_tokens = prompt_tokens;
    result.skip_tokens   = skip;
    // logprobs[i - 1] = log p(token i | tokens [0, i)); position p scores token p + 1.
    const std::vector<float> logprobs = impl_->program.score(data.token_ids);
    if (logprobs.size() + 1 != data.token_ids.size()) {
        throw std::logic_error("qwen3.8-flash-next score returned the wrong number of positions");
    }
    result.token_nlls.reserve(prompt_tokens - skip - 1U);
    for (std::uint32_t position = skip; position + 1 < prompt_tokens; ++position) {
        record_score_nll(result, -static_cast<double>(logprobs.at(position)));
    }
    result.score_seconds = seconds_since(started);
    if (result.tokens_scored > 0) {
        result.mean_nll   = result.sum_nll / static_cast<double>(result.tokens_scored);
        result.perplexity = std::exp(result.mean_nll);
    }
    return result;
}

} // namespace ninfer::targets::qwen3_8_flash_next

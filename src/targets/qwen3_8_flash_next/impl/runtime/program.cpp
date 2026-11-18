#include "runtime/program.h"

#include "core/device.h"
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
#    include "runtime/expert_helper_profile.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {

void validate_speculative_options(const SpeculativeOptions& options) {
    validate_speculative_backend(options.backend);
    if (options.backend == SpeculativeBackend::None) {
        if (options.draft_tokens != 0 || options.adaptive_draft ||
            options.proposal_head != ProposalHead::Full || options.dflash_verify_width != 0 ||
            options.dflash_p_less_draft_temperature) {
            throw std::invalid_argument("Flash-Next speculation options require --spec dflash");
        }
        return;
    }
    if (options.draft_tokens > kDflashMaxK) {
        throw std::invalid_argument("Flash-Next --draft-tokens must be in [0,3]");
    }
    if (options.proposal_head != ProposalHead::Full || options.dflash_verify_width != 0) {
        throw std::invalid_argument(
            "Flash-Next uses the full shared proposal head and automatic "
            "chain width; --lm-head-draft and --dflash-verify-width are unsupported");
    }
    if (options.dflash_p_less_draft_temperature &&
        (!std::isfinite(*options.dflash_p_less_draft_temperature) ||
         *options.dflash_p_less_draft_temperature < 0.0F ||
         *options.dflash_p_less_draft_temperature > 2.0F)) {
        throw std::invalid_argument("Flash-Next draft temperature must be finite and in [0,2]");
    }
}

static_assert(kTopLogprobs == kReportedAlternatives,
              "the Program reports every alternative the Forward ranks");

namespace {

bool speculative_enabled(const ProgramOptions& options) {
    return options.speculative.backend == SpeculativeBackend::DFlash;
}

// Returns the lane count after checking the opened table against the artifact and the lane
// options, so the checks run before any device state is allocated.
std::int32_t checked_lanes(const NgramTable& table, const LoadedModel& model,
                           const ProgramOptions& options) {
#ifndef NINFER_ENABLE_HIP_EXPERT_HELPER
    if (options.expert_helper_pci) {
        throw std::invalid_argument(
            "expert helper requires a build with NINFER_ENABLE_HIP_EXPERT_HELPER=ON");
    }
#endif
    const NgramHashConstants& stored          = table.constants();
    const NgramHashConstants& model_constants = model.ngram_constants();
    if (stored.multipliers != model_constants.multipliers ||
        stored.head_offsets != model_constants.head_offsets ||
        stored.head_sizes != model_constants.head_sizes) {
        throw std::runtime_error("the n-gram table's hash constants differ from the artifact's");
    }
    if (options.lanes < 1 || options.lanes > kMaxLanes) {
        throw std::invalid_argument("qwen3_8_flash_next Program: lanes must be in [1, " +
                                    std::to_string(kMaxLanes) + "]");
    }
    if (options.kv_capacity_tokens.has_value() && *options.kv_capacity_tokens < 1) {
        throw std::invalid_argument("qwen3_8_flash_next Program: kv_capacity_tokens must be "
                                    "positive");
    }
    validate_speculative_options(options.speculative);
    if ((speculative_enabled(options) && !model.view().dflash) ||
        (options.forced_adaptive_k &&
         (!options.speculative.adaptive_draft ||
          *options.forced_adaptive_k > options.speculative.draft_tokens))) {
        throw std::invalid_argument(
            "Flash-Next speculation requires companion weights and K in [0,3]");
    }
    return options.lanes;
}

// The shared QSA pool: the requested tokens, or every lane's whole (page-rounded) context.
std::int64_t pool_tokens(const ProgramOptions& options) {
    if (options.kv_capacity_tokens.has_value()) { return *options.kv_capacity_tokens; }
    const std::int64_t lane_pages =
        (static_cast<std::int64_t>(options.context_tokens) + ops::kQsaPageTokens - 1) /
        ops::kQsaPageTokens;
    return std::int64_t{options.lanes} * lane_pages * ops::kQsaPageTokens;
}

// The Forward options with the window size: expert_window_layers of the largest layer's record
// bytes, so a full window streams every non-resident record of any layer.
ForwardOptions forward_options(const LoadedModel& model, const ProgramOptions& options) {
    if (!(options.expert_window_layers >= 0.0 && options.expert_window_layers <= 1.0)) {
        throw std::invalid_argument("qwen3_8_flash_next Program: expert_window_layers must lie "
                                    "in [0, 1]");
    }
    const ExpertHostStore& store = model.experts();
    std::size_t largest          = 0;
    for (std::int32_t layer = 0; layer < store.layers(); ++layer) {
        std::size_t bytes = 0;
        for (std::int32_t expert = 0; expert < kRoutedExperts; ++expert) {
            bytes +=
                static_cast<std::size_t>(ops::moe_expert_record_bytes(store.codec(layer, expert)));
        }
        largest = std::max(largest, bytes);
    }
    constexpr auto kAlignment = static_cast<std::size_t>(ops::kMoeRecordAlignment);
    ForwardOptions forward    = options.forward;
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
    forward.expert_helper_device_bytes = 0;
    if (options.expert_helper_pci) {
        const auto max_t =
            static_cast<std::uint32_t>(options.lanes * (speculative_enabled(options) ? 4 : 1));
        forward.expert_helper_device_bytes = ExpertHelper::cuda_device_bytes(max_t);
    }
#endif
    if (speculative_enabled(options)) {
        forward.max_verify_width = kDflashWidth;
        forward.dflash_features  = DflashExecution::Enabled;
    }
    forward.expert_window_bytes =
        static_cast<std::size_t>(options.expert_window_layers * static_cast<double>(largest)) /
        kAlignment * kAlignment;
    return forward;
}

// The n-gram rows a super-chunk gathers, once the Forward options are known to be valid.
std::int32_t feeder_tokens(const LoadedModel& model, const ProgramOptions& options) {
    Forward::validate(forward_options(model, options));
    return options.forward.super_chunk_tokens;
}

// The slot plan: the requested bytes, or every free byte after the Forward still to be
// allocated and the reserve.
ExpertSlotPlan slot_plan(const LoadedModel& model, const ProgramOptions& options,
                         const State& state) {
    std::size_t budget = 0;
    if (options.expert_cache_bytes.has_value()) {
        budget = *options.expert_cache_bytes;
    } else {
        std::size_t free_bytes  = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t needed =
            Forward::device_bytes(model.view(), forward_options(model, options), state) +
            Program::kDeviceReserveBytes;
        if (free_bytes <= needed) {
            throw std::runtime_error("qwen3_8_flash_next Program: " + std::to_string(free_bytes) +
                                     " free device bytes leave no expert cache");
        }
        budget = free_bytes - needed;
    }
    return plan_expert_slots(budget, model.experts().codec_counts());
}

void validate_sampling(const LaneSampling& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("qwen3_8_flash_next Program: sampling parameters must be "
                                    "finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F || sampling.min_p < 0.0F ||
        sampling.min_p > 1.0F) {
        throw std::invalid_argument("qwen3_8_flash_next Program: top_p and min_p must be in "
                                    "[0, 1]");
    }
    if (sampling.suppressed_tokens.size() >
        static_cast<std::size_t>(ops::SamplingConfig::kMaximumSuppressedTokens)) {
        throw std::invalid_argument("qwen3_8_flash_next Program: too many suppressed tokens");
    }
    for (const std::int32_t token : sampling.suppressed_tokens) {
        if (token < 0 || token >= kTokenDomain) {
            throw std::invalid_argument("qwen3_8_flash_next Program: a suppressed token is "
                                        "outside the token domain");
        }
    }
    if (sampling.typical_exclude < -1 || sampling.typical_exclude >= kTokenDomain) {
        throw std::invalid_argument("qwen3_8_flash_next Program: typical_exclude is outside the "
                                    "token domain");
    }
}

} // namespace

Program::Program(const LoadedModel& model, const ProgramOptions& options, DeviceContext& device)
    : table_(NgramTable::Options{.path = options.ngram_table}),
      state_(checked_lanes(table_, model, options), options.context_tokens, pool_tokens(options),
             options.kv_format,
             speculative_enabled(options) ? DflashExecution::Enabled : DflashExecution::Disabled),
      feeder_(table_, feeder_tokens(model, options), options.lanes),
      dflash_(speculative_enabled(options)
                  ? std::make_unique<Dflash>(model.view(), state_, device.stream)
                  : nullptr),
      extents_(state_.pool(), options.kv_format, device.stream, device.copy_stream),
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
      helper_(options.expert_helper_pci
                  ? std::make_unique<ExpertHelper>(model, slot_plan(model, options, state_),
                                                   *options.expert_helper_pci, options.lanes,
                                                   speculative_enabled(options), device.stream,
                                                   extents_.flex_addresses().size())
                  : nullptr),
      residency_(
          model.experts(), helper_ ? helper_->primary_plan() : slot_plan(model, options, state_),
          extents_.flex_addresses(),
          helper_ ? std::span<const std::uint64_t>(kExpertHelperCalibCounts) : options.calib_counts,
          device.stream,
          helper_ ? std::span<const std::uint32_t>(helper_->placement().local_eligibility)
                  : std::span<const std::uint32_t>{},
          helper_ && helper_->placement().full_coverage ? ExpertWarmStart::AllEligible
                                                        : ExpertWarmStart::Calibrated),
#else
      residency_(model.experts(), slot_plan(model, options, state_), extents_.flex_addresses(),
                 options.calib_counts, device.stream),
#endif
      forward_(model.view(), state_, feeder_, residency_, forward_options(model, options),
               device.stream, dflash_.get()
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
                                  ,
               helper_.get()
#endif
                   ),
      stream_(device.stream), mask_staging_(static_cast<std::size_t>(options.lanes) *
                                            State::kAllowedTokenWordStride * sizeof(std::uint32_t)),
      speculative_options_(options.speculative), forced_adaptive_k_(options.forced_adaptive_k),
      expert_route_profile_path_(options.expert_route_profile_path) {
    write_route_profile(true);
}

Program::~Program() {
    try {
        write_route_profile(true);
    } catch (const std::exception& error) {
        // Destruction cannot propagate; a failed final diagnostic write is visible to the user.
        std::cerr << "expert route profile final write failed: " << error.what() << '\n';
    }
}

void Program::write_route_profile(bool force) {
    if (expert_route_profile_path_.empty()) { return; }
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - profile_written_ < std::chrono::minutes(5)) { return; }
    if (pending_batch_ || pending_speculative_) {
        throw std::logic_error("expert route profile requires a settled round");
    }
    write_expert_route_profile(expert_route_profile_path_, forward_.expert_route_profile(),
                               std::chrono::duration<double>(now - profile_started_).count());
    profile_written_ = now;
}

std::int32_t Program::lanes() const noexcept { return state_.lanes(); }

std::int64_t Program::kv_capacity_tokens() const noexcept { return state_.pool_tokens(); }

std::int64_t Program::kv_free_tokens() const noexcept { return state_.free_tokens(); }

bool Program::can_admit(std::int64_t reserve_tokens) const noexcept {
    return state_.can_open(reserve_tokens);
}

Program::LaneControl& Program::open_control(Lane lane) {
    if (!state_.is_open(lane)) {
        throw std::out_of_range("qwen3_8_flash_next Program: lane " + std::to_string(lane) +
                                " is not open");
    }
    return lanes_.at(static_cast<std::size_t>(lane));
}

std::optional<Lane> Program::open_lane(std::int64_t reserve_tokens, const LaneSampling& sampling) {
    validate_sampling(sampling);
    if (!state_.can_open(reserve_tokens)) { return std::nullopt; }
    // can_open bounds reserve_tokens by the lane capacity, which fits a page count.
    extents_.make_room(pages_for_tokens(static_cast<std::uint32_t>(reserve_tokens)), residency_);
    const std::optional<std::int32_t> lane = state_.open(reserve_tokens, stream_);
    if (!lane.has_value()) { return std::nullopt; }
    feeder_.reset(*lane);
    lanes_.at(static_cast<std::size_t>(*lane)) =
        LaneControl{.phase = LanePhase::Prompt, .sampling = sampling, .mask_armed = false};
    initialize_speculative(*lane);
    return lane;
}

void Program::close_lane(Lane lane) {
    LaneControl& control = open_control(lane);
    // Drops the lane's in-flight n-gram prefetch, if any.
    feeder_.reset(lane);
    state_.close(lane, stream_);
    control = LaneControl{};
    extents_.give_back(residency_);
}

void Program::resize_lane(Lane lane, std::int32_t frontier, std::int64_t reserve_tokens) {
    static_cast<void>(open_control(lane));
    // State::resize keeps at most the pages of the new reservation and maps up to it. Extents are
    // taken back only for an extent it accepts; it throws for the others before any page moves.
    if (reserve_tokens >= 1 && reserve_tokens <= state_.lane_capacity() && frontier >= 0 &&
        frontier <= reserve_tokens && frontier <= state_.committed(lane)) {
        const std::uint32_t pages  = pages_for_tokens(static_cast<std::uint32_t>(reserve_tokens));
        const std::uint32_t mapped = state_.pages(lane).mapped_page_count();
        if (pages > mapped) { extents_.make_room(pages - mapped, residency_); }
    }
    state_.resize(lane, frontier, reserve_tokens, stream_);
    extents_.give_back(residency_);
}

void Program::reuse_lane(Lane lane, std::int64_t reserve_tokens,
                         std::span<const std::int32_t> prefix, const LaneSampling& sampling,
                         const void* image) {
    auto& control = open_control(lane);
    validate_sampling(sampling);
    if (prefix.empty() || prefix.size() > static_cast<std::size_t>(state_.committed(lane)) ||
        (image == nullptr && prefix.size() != static_cast<std::size_t>(state_.committed(lane)))) {
        throw std::logic_error("Flash-Next reuse requires an exact continuation image");
    }
    feeder_.restore_history(lane, prefix);
    resize_lane(lane, static_cast<std::int32_t>(prefix.size()), reserve_tokens);
    if (image != nullptr) { state_.unpack_continuation(lane, image, stream_); }
    state_.reset_sampling(lane, stream_);
    control = LaneControl{.phase = LanePhase::Prompt, .sampling = sampling, .mask_armed = false};
    initialize_speculative(lane);
}

void Program::restore_lane(Lane lane, std::span<const std::int32_t> prefix,
                           const LaneSampling& sampling, const void* image, RestorePhase phase,
                           const SpeculativeStats* cumulative_stats) {
    auto& control = open_control(lane);
    validate_sampling(sampling);
    if ((cumulative_stats != nullptr && phase != RestorePhase::Decoding) ||
        control.phase != LanePhase::Prompt || state_.committed(lane) != 0 || image == nullptr ||
        prefix.empty() || prefix.size() > static_cast<std::size_t>(state_.reserved_tokens(lane))) {
        throw std::logic_error("Flash-Next restore requires an empty lane and complete image");
    }
    feeder_.restore_history(lane, prefix);
    state_.unpack_continuation(lane, image, stream_);
    state_.advance(lane, static_cast<std::int32_t>(prefix.size()));
    state_.reset_sampling(lane, stream_);
    control = LaneControl{.phase      = phase == RestorePhase::Prompt ? LanePhase::Prompt
                                                                      : LanePhase::Decoding,
                          .sampling   = sampling,
                          .mask_armed = false};
    initialize_speculative(lane, phase);
    if (cumulative_stats != nullptr) { control.speculative = *cumulative_stats; }
}

void Program::set_sampling(Lane lane, const LaneSampling& sampling) {
    LaneControl& control = open_control(lane);
    validate_sampling(sampling);
    control.sampling = sampling;
}

void Program::set_allowed_tokens(Lane lane, std::span<const std::uint32_t> words,
                                 const qwen3_6::OutputSession* output) {
    LaneControl& control = open_control(lane);
    if (words.size() != static_cast<std::size_t>(kAllowedTokenWords)) {
        throw std::invalid_argument("qwen3_8_flash_next Program: an eligibility mask holds " +
                                    std::to_string(kAllowedTokenWords) + " words");
    }
    // Bits past the token domain in the last word are ignored by the sampler.
    constexpr std::uint32_t kLastWordBits =
        kTokenDomain % 32 == 0 ? ~0U : (1U << (kTokenDomain % 32)) - 1U;
    const bool any =
        std::any_of(words.begin(), words.end() - 1, [](std::uint32_t word) { return word != 0; }) ||
        (words.back() & kLastWordBits) != 0;
    if (!any) {
        throw std::invalid_argument("qwen3_8_flash_next Program: an eligibility mask must allow "
                                    "a token");
    }
    // Staged on the host; upload_masks copies every lane's staging in one transfer before the
    // next sampling call. Sampling calls synchronize before returning, so no transfer out of the
    // staging is in flight here, except after a decode that threw before its replay: then the
    // staging is marked again and re-copied later in stream order, before any sampler reads it.
    std::ranges::copy(words, static_cast<std::uint32_t*>(mask_staging_.data()) +
                                 static_cast<std::size_t>(lane) * State::kAllowedTokenWordStride);
    control.mask_armed  = true;
    control.mask_output = output;
    masks_staged_       = true;
}

void Program::upload_masks() {
    if (!masks_staged_) { return; }
    CUDA_CHECK(cudaMemcpyAsync(state_.allowed_tokens(0), mask_staging_.data(), mask_staging_.size(),
                               cudaMemcpyHostToDevice, stream_));
    masks_staged_ = false;
}

void Program::clear_allowed_tokens(Lane lane) {
    auto& control       = open_control(lane);
    control.mask_armed  = false;
    control.mask_output = nullptr;
}

std::int32_t Program::committed(Lane lane) const { return state_.committed(lane); }

ops::SamplingConfig Program::sampling_config(Lane lane) {
    const LaneControl& control   = open_control(lane);
    const LaneSampling& sampling = control.sampling;
    ops::SamplingConfig config;
    config.temperature       = sampling.temperature;
    config.top_k             = sampling.top_k;
    config.top_p             = sampling.top_p;
    config.min_p             = sampling.min_p;
    config.presence_penalty  = sampling.presence_penalty;
    config.frequency_penalty = sampling.frequency_penalty;
    config.p_less            = sampling.p_less ? 1 : 0;
    config.seed              = sampling.seed;
    // Every positive-temperature token is counted, so a penalty enabled later by set_sampling
    // sees the lane's whole sampled history.
    config.token_counts           = state_.token_counts(lane);
    config.suppressed_token_count = static_cast<std::int32_t>(sampling.suppressed_tokens.size());
    std::ranges::copy(sampling.suppressed_tokens, config.suppressed_tokens);
    config.typical_exclude             = sampling.typical_exclude;
    config.allowed_token_words         = control.mask_armed ? state_.allowed_tokens(lane) : nullptr;
    config.allowed_token_column_stride = 0;
    return config;
}

PrefillStep Program::prefill_step(Lane lane, std::span<const std::int32_t> remaining,
                                  std::int32_t token_limit, PrefillTail tail) {
    LaneControl& control = open_control(lane);
    if (control.phase != LanePhase::Prompt) {
        throw std::logic_error("qwen3_8_flash_next Program: lane " + std::to_string(lane) +
                               " has completed its prompt");
    }
    if (remaining.empty() || static_cast<std::int64_t>(remaining.size()) >
                                 state_.reserved_tokens(lane) - state_.committed(lane)) {
        throw std::out_of_range("qwen3_8_flash_next Program: the remaining prompt is empty or "
                                "exceeds the lane's reservation");
    }
    if (dflash_) {
        const std::array lanes{lane};
        dflash_->flush(lanes);
    }
    PrefillStep step;
    step.consumed =
        forward_.prefill_step(lane, remaining, LogitsSink{}, nullptr, token_limit, tail);
    if (dflash_) { dflash_->append_prefill(lane, forward_.dflash_features()); }
    step.complete = static_cast<std::size_t>(step.consumed) == remaining.size();
    if (!step.complete) { return step; }
    PrefillStep result = finish_prompt(lane);
    result.consumed    = step.consumed;
    return result;
}

PrefillStep Program::finish_prompt(Lane lane) {
    auto& control = open_control(lane);
    if (control.phase != LanePhase::Prompt || state_.committed(lane) == 0) {
        throw std::logic_error("Flash-Next finalization requires a complete prompt frontier");
    }
    PrefillStep step;
    step.complete = true;
    upload_masks();
    const SampledRows rows = forward_.sample_retained(lane, sampling_config(lane));
    clear_allowed_tokens(lane);
    control.phase     = LanePhase::Decoding;
    step.next_token   = rows.tokens[0];
    step.next_logprob = rows.logprobs[0];
    const std::array fetch_lanes{lane};
    feeder_.prefetch_anchors(fetch_lanes, rows.tokens);
    if (control.sampling.top_logprobs) {
        std::copy_n(rows.top_ids.begin(), kTopLogprobs, step.top_ids.begin());
        std::copy_n(rows.top_logprobs.begin(), kTopLogprobs, step.top_logprobs.begin());
    }
    return step;
}

void Program::decode(std::span<const Lane> lanes, std::span<const std::int32_t> tokens,
                     std::span<std::int32_t> next, std::span<float> next_logprob,
                     std::span<std::int32_t> top_ids, std::span<float> top_logprobs) {
    if (pending_batch_) { throw std::logic_error("Flash-Next has an unresolved decode packet"); }
    const std::size_t rows   = lanes.size();
    const std::size_t ranked = rows * static_cast<std::size_t>(kTopLogprobs);
    if (next.size() != rows || (!next_logprob.empty() && next_logprob.size() != rows) ||
        top_ids.size() != top_logprobs.size() || (!top_ids.empty() && top_ids.size() != ranked)) {
        throw std::invalid_argument("qwen3_8_flash_next Program: decode output spans do not "
                                    "match the round's rows");
    }
    std::array<ops::SamplingConfig, kMaxLanes> configs{};
    if (rows > configs.size()) {
        throw std::invalid_argument("qwen3_8_flash_next Program: too many decode rows");
    }
    for (std::size_t i = 0; i < rows; ++i) {
        if (open_control(lanes[i]).phase != LanePhase::Decoding) {
            throw std::logic_error("qwen3_8_flash_next Program: lane " + std::to_string(lanes[i]) +
                                   " has not completed its prompt");
        }
        configs.at(i) = sampling_config(lanes[i]);
    }
    if (dflash_) { dflash_->flush(lanes); }
    upload_masks();
    const SampledRows sampled = forward_.decode(
        lanes, tokens, std::span<const ops::SamplingConfig>(configs.data(), rows),
        next_logprob.empty() && top_ids.empty() ? RoundLogprobs::Skip : RoundLogprobs::Report);
    for (std::size_t i = 0; i < rows; ++i) {
        clear_allowed_tokens(lanes[i]);
        next[i] = sampled.tokens[i];
        if (!next_logprob.empty()) { next_logprob[i] = sampled.logprobs[i]; }
    }
    if (dflash_) { dflash_->save_decode_features(lanes, forward_.dflash_features()); }
    if (!top_ids.empty()) {
        std::copy_n(sampled.top_ids.begin(), ranked, top_ids.begin());
        std::copy_n(sampled.top_logprobs.begin(), ranked, top_logprobs.begin());
    }
    last_round_ = RoundTiming{.gather_seconds = forward_.last_gather_seconds(),
                              .replay_ms      = forward_.last_replay_ms()};
    write_route_profile();
}

std::vector<float> Program::score(std::span<const std::int32_t> tokens) {
    if (tokens.size() < 2) { return {}; }
    const auto reserve             = static_cast<std::int64_t>(tokens.size() - 1);
    const std::optional<Lane> lane = open_lane(reserve, LaneSampling{});
    if (!lane.has_value()) {
        throw std::runtime_error("qwen3_8_flash_next Program: no lane or pages are free to score " +
                                 std::to_string(tokens.size()) +
                                 " tokens, or they exceed one lane's context");
    }
    try {
        std::vector<float> scores = forward_.score(*lane, tokens);
        close_lane(*lane);
        return scores;
    } catch (...) {
        close_lane(*lane);
        throw;
    }
}

ProgramMemory Program::memory() const noexcept {
    return ProgramMemory{
        .state_bytes      = state_.device_bytes(),
        .activation_bytes = forward_.activation_bytes()
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
                            + (helper_ ? helper_->memory().cuda_device_bytes : 0)
#endif
            ,
        .workspace_bytes = forward_.workspace_bytes() + (dflash_ ? dflash_->workspace_bytes() : 0),
        .window_bytes    = forward_.window_bytes(),
        .expert_cache_bytes = residency_.plan().bytes(),
        .flex_cache_bytes =
            static_cast<std::size_t>(residency_.flex_slots()) *
            static_cast<std::size_t>(ops::moe_expert_record_bytes(ops::MoeExpertCodec::F3G32)),
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
        .helper = helper_ ? std::optional<ExpertHelperMemory>(helper_->memory()) : std::nullopt,
#endif
    };
}

#include "runtime/speculative_program_impl.h"

} // namespace ninfer::targets::qwen3_8_flash_next

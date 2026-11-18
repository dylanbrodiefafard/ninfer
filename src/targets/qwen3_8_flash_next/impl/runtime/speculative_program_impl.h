// Included inside the target namespace by program.cpp. One pending compact packet is resolved
// before another forward; request publication, stops and cancellation stay in the Engine.

void Program::initialize_speculative(Lane lane, RestorePhase phase) {
    auto& stats = lanes_.at(static_cast<std::size_t>(lane)).speculative;
    if (!dflash_) {
        stats = {};
        return;
    }
    if (phase == RestorePhase::Prompt) {
        auto& calibration   = state_.dflash_lane(lane);
        const auto pending  = calibration.pending;
        calibration         = {};
        calibration.pending = pending;
    }
    stats = {.backend      = SpeculativeBackend::DFlash,
             .enabled      = true,
             .draft_window = speculative_options_.draft_tokens};
    stats.accepted_per_position.assign(kDflashMaxK, 0);
    stats.rounds_per_draft.assign(kDflashWidth, 0);
}

SpeculativeStats Program::speculative_stats(Lane lane) const {
    return lanes_.at(static_cast<std::size_t>(lane)).speculative;
}

void Program::flush_context(std::span<const Lane> lanes) {
    if (dflash_) { dflash_->flush(lanes); }
}

std::int32_t Program::choose_k(std::int32_t batch) {
    if (!dflash_) { return 0; }
    const auto ceiling = static_cast<std::int32_t>(speculative_options_.draft_tokens);
    if (!speculative_options_.adaptive_draft) { return ceiling; }
    if (forced_adaptive_k_) { return static_cast<std::int32_t>(*forced_adaptive_k_); }
    return std::min(ceiling, speculative_batch_policy(static_cast<std::size_t>(batch)).k);
}

SpeculativeRows Program::decode_round(std::span<const Lane> lanes,
                                      std::span<const std::int32_t> anchors,
                                      std::span<const std::uint32_t> output_limits,
                                      RoundLogprobs logprobs) {
    if (pending_batch_ || lanes.empty() ||
        lanes.size() > static_cast<std::size_t>(state_.lanes()) || anchors.size() != lanes.size() ||
        output_limits.size() != lanes.size()) {
        throw std::invalid_argument(
            "Flash-Next decode round has invalid membership or pending state");
    }
    const auto batch    = static_cast<std::int32_t>(lanes.size());
    auto k              = choose_k(batch);
    std::int32_t live_k = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (open_control(lanes[row]).phase != LanePhase::Decoding || output_limits[row] == 0) {
            throw std::logic_error("Flash-Next round row is not decode-ready");
        }
        auto config = sampling_config(lanes[row]);
        if (dflash_ && config.p_less != 0 && config.temperature > 0.0F) {
            config.draft_temperature =
                speculative_options_.dflash_p_less_draft_temperature.value_or(
                    state_.dflash_lane(lanes[row]).draft_temperature);
        }
        round_sampling_[row] = config;
        const auto room      = state_.reserved_tokens(lanes[row]) - state_.committed(lanes[row]);
        if (room < 1) { throw std::logic_error("Flash-Next round has no funded root slot"); }
        const auto request_k = dflash_ && speculative_options_.adaptive_draft && !forced_adaptive_k_
                                   ? std::min(k, state_.dflash_lane(lanes[row]).policy.ceiling)
                                   : k;
        const auto extent = std::min({static_cast<std::int64_t>(request_k),
                                      static_cast<std::int64_t>(output_limits[row]) - 1, room - 1});
        // A history-dependent eligibility mask describes only its root decision. Such a row
        // stays root-only instead of applying a stale mask to unobserved future histories.
        pending_extents_[row] =
            config.allowed_token_words == nullptr ? static_cast<std::int32_t>(extent) : 0;
        if (pending_extents_[row] != 0) { live_k = std::max(live_k, request_k); }
        pending_lanes_[row] = lanes[row];
    }
    // Fixed and forced arms retain their proposal width until every row is root-only. The
    // default table may shrink the shared width when every request has stepped down. Output
    // clipping alone must not change a symmetric proposal's width.
    if (speculative_options_.adaptive_draft && !forced_adaptive_k_) {
        k = live_k;
    } else if (live_k == 0) {
        k = 0;
    }
    pending_k_           = k;
    pending_speculative_ = dflash_ != nullptr;
    if (dflash_ == nullptr) {
        decode(lanes, anchors, std::span<std::int32_t>(ordinary_tokens_).first(lanes.size()),
               logprobs == RoundLogprobs::Report
                   ? std::span<float>(ordinary_logprobs_).first(lanes.size())
                   : std::span<float>{},
               logprobs == RoundLogprobs::Report
                   ? std::span<TokenId>(ordinary_top_ids_).first(lanes.size() * kTopLogprobs)
                   : std::span<TokenId>{},
               logprobs == RoundLogprobs::Report
                   ? std::span<float>(ordinary_top_logprobs_).first(lanes.size() * kTopLogprobs)
                   : std::span<float>{});
        std::fill_n(ordinary_counts_.begin(), lanes.size(), 1);
        std::fill_n(ordinary_accepted_.begin(), lanes.size(), 0);
        pending_round_ = {
            .width    = 1,
            .tokens   = std::span<const std::int32_t>(ordinary_tokens_).first(lanes.size()),
            .counts   = std::span<const std::int32_t>(ordinary_counts_).first(lanes.size()),
            .accepted = std::span<const std::int32_t>(ordinary_accepted_).first(lanes.size()),
            .logprobs = std::span<const float>(ordinary_logprobs_).first(lanes.size()),
            .top_ids =
                std::span<const TokenId>(ordinary_top_ids_).first(lanes.size() * kTopLogprobs),
            .top_logprobs =
                std::span<const float>(ordinary_top_logprobs_).first(lanes.size() * kTopLogprobs)};
    } else {
        upload_masks();
        pending_round_ = forward_.speculate(
            lanes, anchors,
            std::span<const ops::SamplingConfig>(round_sampling_).first(lanes.size()), k,
            std::span<const std::int32_t>(pending_extents_).first(lanes.size()), logprobs);
        last_round_ = {.gather_seconds = forward_.last_gather_seconds(),
                       .replay_ms      = forward_.last_replay_ms()};
        for (const auto lane : lanes) { open_control(lane).mask_armed = false; }
    }
    pending_batch_ = true;
    return pending_round_;
}

void Program::resolve_round(std::span<const std::int32_t> counts) {
    if (!pending_batch_ || counts.size() != pending_round_.counts.size()) {
        throw std::logic_error("Flash-Next resolution does not match its pending packet");
    }
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] < 0 || counts[row] > pending_round_.counts[row]) {
            throw std::invalid_argument("Flash-Next resolution exceeds its licensed packet");
        }
    }
    if (pending_speculative_) { forward_.resolve_speculative(counts); }
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (!dflash_ || counts[row] == 0) { continue; }
        auto& stats             = open_control(pending_lanes_[row]).speculative;
        const auto extent       = pending_speculative_ ? pending_extents_[row] : 0;
        stats.live_draft_tokens = static_cast<std::uint32_t>(pending_k_);
        ++stats.rounds_per_draft[static_cast<std::size_t>(pending_k_)];
        if (extent == 0) {
            ++stats.fallback_steps;
        } else {
            ++stats.rounds;
            stats.drafted_tokens += static_cast<std::uint32_t>(extent);
            const auto accepted = std::min(pending_round_.accepted[row], counts[row]);
            stats.accepted_tokens += static_cast<std::uint32_t>(accepted);
            for (std::int32_t hop = 0; hop < accepted; ++hop) {
                ++stats.accepted_per_position[static_cast<std::size_t>(hop)];
            }
        }
        if (speculative_options_.adaptive_draft && !forced_adaptive_k_) {
            auto& policy = state_.dflash_lane(pending_lanes_[row]).policy;
            // A smaller batch during drain must not raise an existing request's draft window.
            // Commit this deterministic cap only with publication, preserving rejection bytes.
            policy.ceiling =
                std::min(policy.ceiling, choose_k(static_cast<std::int32_t>(counts.size())));
            if (extent > 0 && counts[row] == pending_round_.counts[row]) {
                policy.observe(counts.size(), pending_k_, extent, pending_round_.accepted[row]);
            }
        }
        const auto& config = round_sampling_[row];
        if (pending_speculative_ && pending_k_ > 0 && extent > 0 && config.p_less != 0 &&
            config.temperature > 0.0F && config.draft_temperature > 0.0F) {
            stats.p_less_draft_temperature = config.draft_temperature;
            if (!speculative_options_.dflash_p_less_draft_temperature) {
                auto& calibration   = state_.dflash_lane(pending_lanes_[row]);
                constexpr auto grid = ops::kPLessProposalCalibrationTemperatures;
                for (std::int32_t hop = 0; hop < extent; ++hop) {
                    for (std::size_t g = 0; g < grid.size(); ++g) {
                        const auto index = (row * static_cast<std::size_t>(pending_k_) +
                                            static_cast<std::size_t>(hop)) *
                                               grid.size() +
                                           g;
                        const auto value = pending_round_.calibration[index];
                        if (value < 0.0F) { continue; }
                        if (!std::isfinite(value) || value > 1.000001F) {
                            throw std::logic_error("Flash-Next proposal calibration is invalid");
                        }
                        if (calibration.calibration_count[g] >= 256) {
                            calibration.calibration_count[g] /= 2;
                            calibration.calibration_sum[g] *= 0.5;
                        }
                        ++calibration.calibration_count[g];
                        calibration.calibration_sum[g] += value;
                    }
                }
                std::size_t best = 2; // The 0.4 prior until evidence separates the grid arms.
                const auto score = [&](std::size_t g) {
                    return (calibration.calibration_sum[g] + 16 * 0.65) /
                           (static_cast<double>(calibration.calibration_count[g]) + 16);
                };
                for (std::size_t g = 0; g < grid.size(); ++g) {
                    if (score(g) > score(best)) { best = g; }
                }
                calibration.draft_temperature = grid[best];
            }
        }
    }
    pending_batch_       = false;
    pending_speculative_ = false;
    write_route_profile();
}

// Included inside the target namespace by forward.cpp. The two compute graphs expose only
// licensed candidates. Engine publication resolves their authoritative prefix through the same
// record fold used by eager qualification; it never needs to undo provisional QSA slots.

void Forward::enqueue_spec_a(std::int32_t width, std::int32_t rows) {
    feature_offset_  = 0;
    feature_columns_ = width * rows;
    CUDA_CHECK(cudaMemcpyAsync(verify_.spec_ingress.data, spec_ingress_staging_.value().data(),
                               sizeof(SpecIngress), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(verify_.ingress.data, verify_ingress_staging_.value().data(),
                               sizeof(VerifyIngress), cudaMemcpyHostToDevice, stream_));
    const Tensor lanes = at(verify_.spec_ingress, offsetof(SpecIngress, lanes), DType::I32, {rows});
    const Tensor append_counts =
        at(verify_.spec_ingress, offsetof(SpecIngress, append_counts), DType::I32, {rows});
    const Tensor append_positions =
        at(verify_.spec_ingress, offsetof(SpecIngress, append_positions), DType::I32,
           {kDflashWidth, rows});
    const Tensor backup_slots = at(verify_.spec_ingress, offsetof(SpecIngress, backup_slots),
                                   DType::I32, {8 * kDflashWidth * rows});
    drafter_->backup_context(backup_slots, verify_.context_key_backup, verify_.context_value_backup,
                             rows);
    drafter_->append_pending(append_positions, append_counts, lanes);
    const std::array batch{verify_batch(width, rows)};
    const SubChunk& chunk = batch.front();
    if (width > 1) {
        const Tensor anchors =
            at(verify_.spec_ingress, offsetof(SpecIngress, anchors), DType::I32, {rows});
        const Tensor frontiers =
            at(verify_.spec_ingress, offsetof(SpecIngress, frontiers), DType::I32, {rows});
        const Tensor valid =
            at(verify_.spec_ingress, offsetof(SpecIngress, valid), DType::I32, {rows});
        const Tensor extents =
            at(verify_.spec_ingress, offsetof(SpecIngress, extents), DType::I32, {rows});
        auto* configs = reinterpret_cast<const ops::SamplingConfig*>(
            static_cast<const std::byte*>(verify_.spec_ingress.data) +
            offsetof(SpecIngress, configs));
        Tensor drafts =
            at(verify_.spec_egress, offsetof(SpecEgress, drafts), DType::I32, {width - 1, rows});
        Tensor selector_ids(verify_.selector_ids.data, DType::I32, {16, width - 1, rows});
        Tensor selector_q(verify_.selector_q.data, DType::FP32, {16, width - 1, rows});
        drafter_->propose(width, anchors, frontiers, valid, lanes, configs, drafts, selector_ids,
                          selector_q);
        Tensor ids = chunk.token_ids.view({width, rows});
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, ids, stream_);
        auto& host = *static_cast<SpecEgress*>(spec_egress_staging_.value().data());
        CUDA_CHECK(cudaMemcpyAsync(host.drafts.data(), drafts.data, drafts.bytes(),
                                   cudaMemcpyDeviceToHost, stream_));
    }
    // An external event node is observable by the host; an internal capture event is only a
    // graph dependency and cannot fence the pinned draft packet for CPU n-gram lookup.
    CUDA_CHECK(
        cudaEventRecordWithFlags(drafts_ready_.value().get(), stream_, cudaEventRecordExternal));
    const Tensor canonical = state_.ple_history();
    CUDA_CHECK(cudaMemcpyAsync(verify_.ple_states.data, canonical.data, canonical.bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
    Tensor embedded = columns(chunk_.block_input, kHidden, chunk.tokens());
    ops::embedding(chunk.token_ids, model_.embedding, embedded, stream_);
    Tensor residual = residual_columns(chunk);
    ops::hyper_connection_broadcast(embedded, residual, stream_);
    enqueue_layer_loop(batch, false, RowSource::Resident, {}, nullptr, 0, kPleLayer);
}

void Forward::enqueue_spec_b(std::int32_t width, std::int32_t rows, bool exchange_masks) {
    feature_offset_          = 0;
    feature_columns_         = width * rows;
    const auto columns_count = width * rows;
    CUDA_CHECK(cudaMemcpyAsync(chunk_.ple_rows.data, decode_rows_.data(),
                               static_cast<std::size_t>(columns_count) * kNgramTokenBytes,
                               cudaMemcpyHostToDevice, stream_));
    const std::array batch{verify_batch(width, rows)};
    // Graph A has constructed every chain input. The matcher reads fixed-pitch pinned
    // staging while the target runs; only target sampling consumes its configurations.
    std::optional<qwen3_6::TokenMaskExchange::Submission> masks;
    if (exchange_masks) {
        masks = token_masks_->enqueue(batch.front().token_ids.view({width, rows}), nullptr,
                                      batch.front().valid_columns, stream_, mask_stream_.get());
    }
    enqueue_layer_loop(batch, false, RowSource::Resident, {}, nullptr, kPleLayer, kLayers);
    head(0, columns_count);
    const auto* configs = reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const std::byte*>(verify_.spec_ingress.data) + offsetof(SpecIngress, configs));
    if (masks) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_, masks->ready, 0));
        configs = masks->sampling;
    }
    const Tensor logits_flat   = logits(columns_count);
    const Tensor packed_logits = logits_flat.view({kVocabRows, width, rows});
    const Tensor counts_pool   = state_.sampling_counts_pool();
    CUDA_CHECK(cudaMemcpyAsync(verify_.count_snapshot.data, counts_pool.data, counts_pool.bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
    Tensor licensed =
        at(verify_.spec_egress, offsetof(SpecEgress, tokens), DType::I32, {width, rows});
    Tensor counts   = at(verify_.spec_egress, offsetof(SpecEgress, counts), DType::I32, {rows});
    Tensor accepted = at(verify_.spec_egress, offsetof(SpecEgress, accepted), DType::I32, {rows});
    if (width == 1) {
        const Tensor positions =
            at(verify_.spec_ingress, offsetof(SpecIngress, frontiers), DType::I32, {rows});
        Tensor sampled = licensed.view({rows});
        ops::sample(logits_flat, sampled, kTokenDomain, configs, positions,
                    ops::kSamplePurposeDecode, work_, stream_);
        const Tensor valid =
            at(verify_.spec_ingress, offsetof(SpecIngress, valid), DType::I32, {rows});
        CUDA_CHECK(cudaMemcpyAsync(counts.data, valid.data, valid.bytes(), cudaMemcpyDeviceToDevice,
                                   stream_));
        CUDA_CHECK(cudaMemsetAsync(accepted.data, 0, accepted.bytes(), stream_));
    } else {
        Tensor target_flat(verify_.targets.data, DType::I32, {columns_count});
        ops::argmax(logits_flat, target_flat, kTokenDomain, configs, width, stream_);
        const Tensor target = target_flat.view({width, rows});
        const Tensor drafts =
            at(verify_.spec_egress, offsetof(SpecEgress, drafts), DType::I32, {width - 1, rows});
        const Tensor extents =
            at(verify_.spec_ingress, offsetof(SpecIngress, extents), DType::I32, {rows});
        Tensor lengths(verify_.lengths.data, DType::I32, {rows});
        Tensor anchors(verify_.anchors.data, DType::I32, {rows});
        const Tensor base =
            at(verify_.spec_ingress, offsetof(SpecIngress, frontiers), DType::I32, {rows});
        const Tensor roots =
            at(verify_.spec_ingress, offsetof(SpecIngress, anchors), DType::I32, {rows});
        CUDA_CHECK(cudaMemcpyAsync(lengths.data, base.data, base.bytes(), cudaMemcpyDeviceToDevice,
                                   stream_));
        CUDA_CHECK(cudaMemcpyAsync(anchors.data, roots.data, roots.bytes(),
                                   cudaMemcpyDeviceToDevice, stream_));
        const Tensor selector_ids(verify_.selector_ids.data, DType::I32, {16, width - 1, rows});
        const Tensor selector_q(verify_.selector_q.data, DType::FP32, {16, width - 1, rows});
        Tensor calibration = at(verify_.spec_egress, offsetof(SpecEgress, calibration), DType::FP32,
                                {ops::kPLessProposalCalibrationTemperatureCount, width - 1, rows});
        ops::speculative_accept_greedy_drafts(
            target, packed_logits, drafts, extents, lengths, anchors, licensed, counts, accepted,
            kTokenDomain, configs, work_, stream_, &selector_ids, &selector_q, &calibration);
    }
    const Tensor enabled =
        at(verify_.spec_ingress, offsetof(SpecIngress, logprobs), DType::I32, {rows});
    Tensor logprobs =
        at(verify_.spec_egress, offsetof(SpecEgress, logprobs), DType::FP32, {width, rows});
    Tensor top_ids      = at(verify_.spec_egress, offsetof(SpecEgress, top_ids), DType::I32,
                             {kReportedAlternatives, width, rows});
    Tensor top_logprobs = at(verify_.spec_egress, offsetof(SpecEgress, top_logprobs), DType::FP32,
                             {kReportedAlternatives, width, rows});
    ops::token_logprobs(packed_logits, licensed, enabled, &counts, nullptr, kTokenDomain, logprobs,
                        top_ids, top_logprobs, stream_);
    const Tensor verify_ids =
        at(verify_.spec_egress, offsetof(SpecEgress, verify_ids), DType::I32, {width, rows});
    CUDA_CHECK(cudaMemcpyAsync(verify_ids.data, batch.front().token_ids.data, verify_ids.bytes(),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(spec_egress_staging_.value().data(), verify_.spec_egress.data,
                               sizeof(SpecEgress), cudaMemcpyDeviceToHost, stream_));
}

void Forward::capture_spec_graphs() {
    const auto size = std::size_t{kDflashWidth} * static_cast<std::size_t>(state_.lanes());
    spec_a_definitions_.resize(size);
    spec_b_definitions_.resize(2 * size);
    spec_a_graphs_.resize(size);
    spec_b_graphs_.resize(2 * size);
    for (std::int32_t k = 0; k <= kDflashMaxK; ++k) {
        for (std::int32_t rows = 1; rows <= state_.lanes(); ++rows) {
            const auto index =
                static_cast<std::size_t>(k) * static_cast<std::size_t>(state_.lanes()) +
                static_cast<std::size_t>(rows - 1);
            spec_a_definitions_[index].capture(stream_,
                                               [this, k, rows] { enqueue_spec_a(k + 1, rows); });
            spec_a_graphs_[index].instantiate(spec_a_definitions_[index]);
            spec_a_graphs_[index].upload(stream_);
            for (const bool exchange : {false, true}) {
                const auto target_index = 2 * index + static_cast<std::size_t>(exchange);
                spec_b_definitions_[target_index].capture(
                    stream_, [this, k, rows, exchange] { enqueue_spec_b(k + 1, rows, exchange); });
                spec_b_graphs_[target_index].instantiate(spec_b_definitions_[target_index]);
                spec_b_graphs_[target_index].upload(stream_);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

SpeculativeRows Forward::speculate(std::span<const std::int32_t> lanes,
                                   std::span<const std::int32_t> anchors,
                                   std::span<const ops::SamplingConfig> configs, std::int32_t k,
                                   std::span<const std::int32_t> extents, RoundLogprobs logprobs,
                                   std::span<const qwen3_6::OutputSession* const> outputs) {
    if (drafter_ == nullptr || speculative_pending_ || k < 0 || k > kDflashMaxK || lanes.empty() ||
        lanes.size() > static_cast<std::size_t>(state_.lanes()) || anchors.size() != lanes.size() ||
        configs.size() != lanes.size() || extents.size() != lanes.size() ||
        outputs.size() != lanes.size()) {
        throw std::invalid_argument("Flash-Next speculative round has invalid membership or state");
    }
    const bool exchange_masks = std::any_of(outputs.begin(), outputs.end(), [](const auto* output) {
        return output != nullptr && output->has_token_masks();
    });
    if (exchange_masks) { token_masks_->bind(outputs, configs); }
    const auto width = k + 1;
    const auto rows  = static_cast<std::int32_t>(lanes.size());
    std::array<std::int32_t, kMaxLanes> valid{};
    std::array<std::int32_t, kVerifyMaxColumns> ids{};
    auto& ingress = *static_cast<SpecIngress*>(spec_ingress_staging_.value().data());
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (extents[row] < 0 || extents[row] > k || anchors[row] < 0 ||
            anchors[row] >= kTokenDomain) {
            throw std::invalid_argument("Flash-Next speculative extent or anchor is invalid");
        }
        if (configs[row].token_counts != nullptr &&
            configs[row].token_counts != state_.token_counts(lanes[row])) {
            throw std::invalid_argument(
                "Flash-Next speculative counts must use their lane-owned column");
        }
        valid[row]                 = extents[row] + 1;
        ingress.anchors[row]       = anchors[row];
        ingress.frontiers[row]     = state_.committed(lanes[row]);
        ingress.lanes[row]         = lanes[row];
        ingress.valid[row]         = valid[row];
        ingress.extents[row]       = extents[row];
        ingress.configs[row]       = configs[row];
        ingress.logprobs[row]      = logprobs == RoundLogprobs::Report ? 1 : 0;
        const auto pending         = state_.dflash_pending_count(lanes[row]);
        ingress.append_counts[row] = pending;
        if (pending > ingress.frontiers[row]) {
            throw std::logic_error("Flash-Next pending context exceeds its target frontier");
        }
        for (std::int32_t column = 0; column < kDflashWidth; ++column) {
            const auto index                = row * kDflashWidth + static_cast<std::size_t>(column);
            const auto position             = ingress.frontiers[row] - pending + column;
            ingress.append_positions[index] = position;
            // [128,capacity,8,lanes] is flattened into 128-element gather rows. The
            // fixed geometry bounds every index below 12*8*2048.
            for (std::int32_t head = 0; head < 8; ++head) {
                ingress.backup_slots[index * 8 + static_cast<std::size_t>(head)] =
                    static_cast<std::int32_t>((static_cast<std::int64_t>(lanes[row]) * 8 + head) *
                                                  state_.dflash_cache().padded_capacity() +
                                              position % kDflashWindow);
            }
        }
        for (std::int32_t column = 0; column < width; ++column) {
            ids[row * static_cast<std::size_t>(width) + static_cast<std::size_t>(column)] =
                anchors[row];
        }
    }
    const auto count = lanes.size() * static_cast<std::size_t>(width);
    prepare_verify(lanes, std::span<const std::int32_t>(ids).first(count), width,
                   std::span<const std::int32_t>(valid).first(lanes.size()));
    const auto index = static_cast<std::size_t>(k) * static_cast<std::size_t>(state_.lanes()) +
                       static_cast<std::size_t>(rows - 1);
    auto& output     = *static_cast<SpecEgress*>(spec_egress_staging_.value().data());
    std::span<const qwen3_6::TokenMaskResult> mask_results;
    try {
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
        if (helper_) {
            static_cast<void>(helper_->begin_round(static_cast<std::uint32_t>(width * rows)));
        }
#endif
        CUDA_CHECK(cudaEventRecord(replay_start_.get(), stream_));
        feature_offset_  = 0;
        feature_columns_ = width * rows;
        spec_a_graphs_.at(index).launch(stream_);
        CUDA_CHECK(cudaEventSynchronize(drafts_ready_.value().get()));
        auto& verify_ingress = *static_cast<VerifyIngress*>(verify_ingress_staging_.value().data());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            state_.set_dflash_pending_count(lanes[row], 0);
            for (std::int32_t column = 1; column < width; ++column) {
                verify_ingress.tokens[row * static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(column)] =
                    column <= extents[row] ? output.drafts[row * static_cast<std::size_t>(k) +
                                                           static_cast<std::size_t>(column - 1)]
                                           : anchors[row];
            }
        }
        const auto gather_start = std::chrono::steady_clock::now();
        const auto tokens       = std::span<const std::int32_t>(verify_ingress.tokens).first(count);
        feeder_.gather_proposals(lanes, tokens, width,
                                 {static_cast<std::uint8_t*>(decode_rows_.data()),
                                  count * static_cast<std::size_t>(kNgramTokenBytes)});
        last_gather_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gather_start).count();
        spec_b_graphs_.at(2 * index + static_cast<std::size_t>(exchange_masks)).launch(stream_);
        enqueue_round_cache_counts();
        CUDA_CHECK(cudaEventRecord(replay_stop_.get(), stream_));
        CUDA_CHECK(cudaEventSynchronize(replay_stop_.get()));
        if (exchange_masks) { mask_results = token_masks_->results(); }
        CUDA_CHECK(cudaEventElapsedTime(&last_replay_ms_, replay_start_.get(), replay_stop_.get()));
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
        if (helper_) { helper_->finish_round(); }
#endif
    } catch (...) {
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
        if (helper_) {
            drain_helper_round();
            throw;
        }
#endif
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaStreamSynchronize(fill_stream_.get()));
        CUDA_CHECK(cudaStreamSynchronize(mask_stream_.get()));
        throw;
    }
    const auto& packet = *static_cast<const VerifyIngress*>(verify_ingress_staging_.value().data());
    if (!std::equal(output.verify_ids.begin(),
                    output.verify_ids.begin() + static_cast<std::ptrdiff_t>(count),
                    packet.tokens.begin())) {
        throw std::logic_error(
            "Flash-Next CPU n-gram packet differs from its GPU verification IDs");
    }
    settle_round_cache_counts();
    ++decode_profile_rounds_;
    pending_verify_width_ = width;
    pending_verify_rows_  = rows;
    speculative_pending_  = true;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (output.counts[row] < 1 || output.counts[row] > valid[row]) {
            throw std::logic_error("Flash-Next accept produced an invalid licensed prefix");
        }
        licensed_counts_[row] = output.counts[row];
        mask_row_errors_[row] = {};
        if (exchange_masks && mask_results[row].error) {
            const auto safe       = static_cast<std::int32_t>(mask_results[row].safe_columns);
            licensed_counts_[row] = std::min(licensed_counts_[row], safe);
            if (safe == 0) { mask_row_errors_[row] = mask_results[row].error; }
            // Invalid columns carry no calibration evidence, even if the fallback
            // all-ones domain let device acceptance visit them before the CPU clamp.
            for (std::int32_t hop = safe; hop < k; ++hop) {
                const auto offset =
                    (row * static_cast<std::size_t>(k) + static_cast<std::size_t>(hop)) *
                    ops::kPLessProposalCalibrationTemperatureCount;
                std::fill_n(output.calibration.data() + offset,
                            ops::kPLessProposalCalibrationTemperatureCount, -1.0F);
            }
        }
        licensed_accepted_[row] = std::min(output.accepted[row], licensed_counts_[row]);
    }
    const auto span = [](const auto& array, std::size_t size) {
        return std::span(array).first(size);
    };
    return {.width      = width,
            .tokens     = span(output.tokens, count),
            .counts     = span(licensed_counts_, lanes.size()),
            .raw_counts = span(output.counts, lanes.size()),
            .accepted   = span(licensed_accepted_, lanes.size()),
            .row_errors = span(mask_row_errors_, lanes.size()),
            .calibration =
                span(output.calibration, lanes.size() * static_cast<std::size_t>(k) *
                                             ops::kPLessProposalCalibrationTemperatureCount),
            .logprobs     = span(output.logprobs, count),
            .top_ids      = span(output.top_ids, count * kReportedAlternatives),
            .top_logprobs = span(output.top_logprobs, count * kReportedAlternatives)};
}

void Forward::resolve_speculative(std::span<const std::int32_t> counts) {
    if (!speculative_pending_ || counts.size() != static_cast<std::size_t>(pending_verify_rows_)) {
        throw std::logic_error("Flash-Next has no matching speculative candidate to resolve");
    }
    const auto& ingress = *static_cast<const SpecIngress*>(spec_ingress_staging_.value().data());
    auto& output        = *static_cast<SpecEgress*>(spec_egress_staging_.value().data());
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] < 0 || counts[row] > output.counts[row]) {
            throw std::invalid_argument("Flash-Next licensed prefix resolution is invalid");
        }
    }
    // Full publication needs no count correction. A truncated/cancelled packet restores the
    // immutable pre-accept count column and adds only its published IDs; duplicate IDs coalesce.
    std::array<std::int32_t, kDflashWidth * kMaxLanes> ids{}, increments{}, destinations{};
    std::size_t patches = 0;
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] == output.counts[row] || ingress.configs[row].temperature <= 0.0F ||
            ingress.configs[row].token_counts == nullptr) {
            continue;
        }
        const auto lane    = ingress.lanes[row];
        const auto* source = static_cast<const std::int32_t*>(verify_.count_snapshot.data) +
                             static_cast<std::size_t>(lane) * kVocabRows;
        CUDA_CHECK(cudaMemcpyAsync(state_.token_counts(lane), source,
                                   static_cast<std::size_t>(kVocabRows) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, stream_));
        const auto first = patches;
        for (std::int32_t column = 0; column < counts[row]; ++column) {
            const auto token = output.tokens[row * static_cast<std::size_t>(pending_verify_width_) +
                                             static_cast<std::size_t>(column)];
            auto match       = first;
            while (match < patches && ids[match] != token) { ++match; }
            if (match == patches) {
                ids[patches]          = token;
                destinations[patches] = lane;
                CUDA_CHECK(cudaMemcpyAsync(&output.count_patch[patches], source + token,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_));
                ++patches;
            }
            ++increments[match];
        }
    }
    if (patches != 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (std::size_t patch = 0; patch < patches; ++patch) {
            output.count_patch[patch] += increments[patch];
            CUDA_CHECK(cudaMemcpyAsync(state_.token_counts(destinations[patch]) + ids[patch],
                                       &output.count_patch[patch], sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream_));
        }
    }
    const auto rows           = static_cast<std::int32_t>(counts.size());
    const Tensor backup_slots = at(verify_.spec_ingress, offsetof(SpecIngress, backup_slots),
                                   DType::I32, {8 * kDflashWidth * rows});
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] != 0) { continue; }
        drafter_->restore_context(backup_slots, verify_.context_key_backup,
                                  verify_.context_value_backup, static_cast<std::int32_t>(row),
                                  rows);
        state_.set_dflash_pending_count(ingress.lanes[row], ingress.append_counts[row]);
    }
    const auto width     = pending_verify_width_;
    speculative_pending_ = false;
    commit_verify(counts);
    // Only the publication prefix determines the next input. The last published output is not
    // among the inputs just committed, so fetch it after the authoritative raw history advances.
    // Snapshots keep this read independent of subsequent compaction, suspension or rejection.
    std::array<std::int32_t, kMaxLanes> next_lanes{}, next_tokens{};
    std::size_t next_rows = 0;
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] == 0) { continue; }
        next_lanes[next_rows]  = ingress.lanes[row];
        next_tokens[next_rows] = output.tokens[row * static_cast<std::size_t>(width) +
                                               static_cast<std::size_t>(counts[row] - 1)];
        ++next_rows;
    }
    if (next_rows != 0) {
        feeder_.prefetch_anchors(std::span<const std::int32_t>(next_lanes).first(next_rows),
                                 std::span<const std::int32_t>(next_tokens).first(next_rows));
    }
}

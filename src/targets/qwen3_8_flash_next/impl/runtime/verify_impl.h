// Included inside the target namespace by forward.cpp. Chain recording leaves the canonical
// recurrent states and logical frontiers untouched; QSA's causal consumers ignore provisional
// slots beyond the frontier, which subsequent execution overwrites.

template <class Allocator>
Forward::VerifyStorage Forward::allocate_verify(Allocator& allocator, const ForwardOptions& options,
                                                std::int32_t lanes) {
    if (options.max_verify_width == 0) { return {}; }
    const std::int32_t tokens = options.max_verify_width * lanes;
    VerifyStorage storage{
        .ingress = allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(VerifyIngress))}),
        .ple_states = allocator.alloc(DType::BF16, {9, kPleKeyRows, lanes + tokens}),
    };
    const std::int32_t outer = lanes * kGdnLayers;
    storage.convolution_records =
        allocator.alloc(DType::BF16, {kGdnConvChannels, options.max_verify_width, outer});
    storage.key_records =
        allocator.alloc(DType::BF16, {kGdnHeadDim, kGdnKeyHeads, options.max_verify_width, outer});
    storage.value_records = allocator.alloc(
        DType::BF16, {kGdnHeadDim, kGdnValueHeads, options.max_verify_width, outer});
    storage.gate_records =
        allocator.alloc(DType::FP32, {2, kGdnValueHeads, options.max_verify_width, outer});
    storage.commit_ingress =
        allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(VerifyCommitIngress))});
    storage.selected_ple    = allocator.alloc(DType::BF16, {9 * kPleKeyRows, lanes});
    storage.selected_logits = allocator.alloc(DType::BF16, {kVocabRows, lanes});
    if (options.trace_verify_routes) {
        storage.expert_routes = allocator.alloc(DType::I32, {ops::kMoeTopK, tokens, kLayers});
    }
    if (options.dflash_features == DflashExecution::Enabled) {
        storage.context_key_backup =
            allocator.alloc(DType::BF16, {1024, kDflashWidth * lanes, kDflashLayers});
        storage.context_value_backup =
            allocator.alloc(DType::BF16, {1024, kDflashWidth * lanes, kDflashLayers});
        storage.spec_ingress =
            allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(SpecIngress))});
        storage.spec_egress =
            allocator.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(SpecEgress))});
        storage.targets        = allocator.alloc(DType::I32, {kDflashWidth, lanes});
        storage.selector_ids   = allocator.alloc(DType::I32, {16, kDflashMaxK, lanes});
        storage.selector_q     = allocator.alloc(DType::FP32, {16, kDflashMaxK, lanes});
        storage.lengths        = allocator.alloc(DType::I32, {lanes});
        storage.anchors        = allocator.alloc(DType::I32, {lanes});
        storage.count_snapshot = allocator.alloc(DType::I32, {kVocabRows, lanes});
        storage.token_masks =
            allocator.alloc(DType::I32, {(kTokenDomain + 31) / 32, kDflashWidth, lanes});
        storage.mask_sampling = allocator.alloc(
            DType::U8, {static_cast<std::int32_t>(sizeof(ops::SamplingConfig)), lanes});
        storage.mask_nodes = allocator.alloc(DType::I32, {kDflashWidth, lanes, 2});
    }
    return storage;
}

std::size_t Forward::plan_verify_bytes(const ForwardOptions& options, std::int32_t lanes) {
    WorkspaceLayoutBuilder builder;
    static_cast<void>(allocate_verify(builder, options, lanes));
    return builder.peak_bytes(kAlignment);
}

Tensor Forward::verify_routes(std::int32_t layer) const {
    if (!options_.trace_verify_routes || pending_verify_width_ == 0 || layer < 0 ||
        layer >= kLayers) {
        throw std::logic_error("Flash-Next verification routes are unavailable");
    }
    auto* base = static_cast<std::byte*>(verify_.expert_routes.data) +
                 static_cast<std::size_t>(layer) * verify_.expert_routes.nb[2];
    return Tensor(base, DType::I32, {ops::kMoeTopK, pending_verify_width_ * pending_verify_rows_});
}

GdnReplayRecords Forward::verify_records(std::int32_t width) const {
    const std::int32_t outer = state_.lanes() * kGdnLayers;
    GdnReplayRecords records;
    records.spec = {.layers          = kGdnLayers,
                    .record_capacity = state_.lanes(),
                    .width           = width,
                    .conv_channels   = kGdnConvChannels,
                    .qk_heads        = kGdnKeyHeads,
                    .value_heads     = kGdnValueHeads,
                    .key_dim         = kGdnHeadDim,
                    .value_dim       = kGdnHeadDim};
    records.conv =
        Tensor(verify_.convolution_records.data, DType::BF16, {kGdnConvChannels, width, outer});
    records.key =
        Tensor(verify_.key_records.data, DType::BF16, {kGdnHeadDim, kGdnKeyHeads, width, outer});
    records.value = Tensor(verify_.value_records.data, DType::BF16,
                           {kGdnHeadDim, kGdnValueHeads, width, outer});
    records.gate =
        Tensor(verify_.gate_records.data, DType::FP32, {2, kGdnValueHeads, width, outer});
    return records;
}

Forward::SubChunk Forward::verify_batch(std::int32_t width, std::int32_t rows) const {
    const std::int32_t tokens = width * rows;
    return SubChunk{
        .width       = width,
        .rows        = rows,
        .max_visible = state_.lane_capacity(),
        .token_ids   = at(verify_.ingress, offsetof(VerifyIngress, tokens), DType::I32, {tokens}),
        .rope_positions = at(verify_.ingress, offsetof(VerifyIngress, rope_positions), DType::I32,
                             {tokens, kMropeAxes}),
        .slots = at(verify_.ingress, offsetof(VerifyIngress, slots), DType::I32, {width, rows}),
        .qsa_positions = at(verify_.ingress, offsetof(VerifyIngress, qsa_positions), DType::I32,
                            {width, kMropeAxes, rows}),
        .lanes         = at(verify_.ingress, offsetof(VerifyIngress, lanes), DType::I32, {rows}),
        .state_mode    = StateMode::Record,
        .valid_columns =
            at(verify_.ingress, offsetof(VerifyIngress, valid_columns), DType::I32, {rows}),
        .ple_snapshot_slots =
            at(verify_.ingress, offsetof(VerifyIngress, ple_snapshot_slots), DType::I32, {rows}),
    };
}

void Forward::prepare_verify(std::span<const std::int32_t> lanes,
                             std::span<const std::int32_t> tokens, std::int32_t width,
                             std::span<const std::int32_t> valid_columns) {
    require_resolved();
    if (width < 1 || width > options_.max_verify_width || lanes.empty() ||
        lanes.size() > static_cast<std::size_t>(state_.lanes()) ||
        valid_columns.size() != lanes.size() ||
        tokens.size() != lanes.size() * static_cast<std::size_t>(width)) {
        throw std::invalid_argument("Flash-Next verify: invalid chain dimensions or capacity");
    }
    pending_verify_width_ = 0;
    check_tokens(tokens);
    auto& ingress = *static_cast<VerifyIngress*>(verify_ingress_staging_.value().data());
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto earlier = lanes.first(row);
        if (std::find(earlier.begin(), earlier.end(), lanes[row]) != earlier.end()) {
            throw std::invalid_argument("Flash-Next verify: duplicate lane");
        }
        if (lanes[row] == feeder_.pending_lane()) {
            throw std::logic_error("Flash-Next verify: lane has a pending n-gram gather");
        }
        const std::int32_t frontier = state_.committed(lanes[row]);
        const std::int32_t valid    = valid_columns[row];
        if (valid < 1 || valid > width || valid > state_.reserved_tokens(lanes[row]) - frontier) {
            throw std::out_of_range(
                "Flash-Next verify: valid chain exceeds reserved lane capacity");
        }
        ingress.lanes[row]              = lanes[row];
        ingress.valid_columns[row]      = valid;
        ingress.ple_snapshot_slots[row] = state_.lanes() + static_cast<std::int32_t>(row) * width;
        for (std::int32_t column = 0; column < width; ++column) {
            const auto index =
                row * static_cast<std::size_t>(width) + static_cast<std::size_t>(column);
            const std::int32_t position = frontier + std::min(column, valid - 1);
            ingress.tokens[index]       = tokens[index];
            ingress.slots[index]        = column < valid ? position : -1;
            for (std::int32_t axis = 0; axis < kMropeAxes; ++axis) {
                ingress.rope_positions[static_cast<std::size_t>(axis) * tokens.size() + index] =
                    position;
                ingress.qsa_positions[row * static_cast<std::size_t>(kMropeAxes * width) +
                                      static_cast<std::size_t>(axis * width + column)] = position;
            }
        }
    }
}

Tensor Forward::verify(std::span<const std::int32_t> lanes, std::span<const std::int32_t> tokens,
                       std::int32_t width, std::span<const std::int32_t> valid_columns) {
    prepare_verify(lanes, tokens, width, valid_columns);
    auto& ingress = *static_cast<VerifyIngress*>(verify_ingress_staging_.value().data());
    try {
        CUDA_CHECK(cudaMemcpyAsync(verify_.ingress.data, &ingress, sizeof(ingress),
                                   cudaMemcpyHostToDevice, stream_));
        const Tensor canonical_ple = state_.ple_history();
        CUDA_CHECK(cudaMemcpyAsync(verify_.ple_states.data, canonical_ple.data,
                                   canonical_ple.bytes(), cudaMemcpyDeviceToDevice, stream_));
        feature_offset_  = 0;
        feature_columns_ = width * static_cast<std::int32_t>(lanes.size());
        const std::array batch{verify_batch(width, static_cast<std::int32_t>(lanes.size()))};
        const SubChunk& chunk = batch.front();
        Tensor embedded       = columns(chunk_.block_input, kHidden, chunk.tokens());
        ops::embedding(chunk.token_ids, model_.embedding, embedded, stream_);
        Tensor residual = residual_columns(chunk);
        ops::hyper_connection_broadcast(embedded, residual, stream_);
        enqueue_layer_loop(batch, false, RowSource::Resident, {}, nullptr, 0, kPleLayer);

        // Layer zero needs no n-gram rows. Its device work overlaps the host's gather; histories
        // are copied before drafting so rejected columns never advance the committed raw-token
        // history.
        const auto gather_start = std::chrono::steady_clock::now();
        auto rows =
            std::span<std::uint8_t>(static_cast<std::uint8_t*>(decode_rows_.data()),
                                    tokens.size() * static_cast<std::size_t>(kNgramTokenBytes));
        feeder_.gather_proposals(lanes, tokens, width, rows);
        last_gather_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gather_start).count();
        CUDA_CHECK(cudaMemcpyAsync(chunk_.ple_rows.data, rows.data(), rows.size(),
                                   cudaMemcpyHostToDevice, stream_));
        enqueue_layer_loop(batch, false, RowSource::Resident, {}, nullptr, kPleLayer, kLayers);
        head(0, chunk.tokens());
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        ++decode_profile_rounds_;
        pending_verify_width_ = width;
        pending_verify_rows_  = chunk.rows;
        return logits(chunk.tokens());
    } catch (...) {
        // An I/O or Op-validation failure may occur after layer zero started reading the pinned
        // ingress. Drain both streams before allowing this Forward's staging to be reused.
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaStreamSynchronize(fill_stream_.get()));
        throw;
    }
}

void Forward::commit_verify(std::span<const std::int32_t> commit_counts) {
    require_resolved();
    const std::int32_t width = pending_verify_width_;
    const std::int32_t rows  = pending_verify_rows_;
    if (width == 0 || commit_counts.size() != static_cast<std::size_t>(rows)) {
        throw std::logic_error("Flash-Next commit requires the latest unresolved verify frame");
    }
    const auto& ingress =
        *static_cast<const VerifyIngress*>(verify_ingress_staging_.value().data());
    auto& commit = *static_cast<VerifyCommitIngress*>(verify_commit_staging_.value().data());
    std::array<ops::GdnReplayFoldRow, kMaxLanes> fold{};
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto index = static_cast<std::size_t>(row);
        const auto count = commit_counts[index];
        const auto lane  = ingress.lanes[index];
        if (count < 0 || count > ingress.valid_columns[index] ||
            state_.committed(lane) != ingress.slots[index * static_cast<std::size_t>(width)]) {
            throw std::logic_error("Flash-Next commit prefix does not match its recorded source");
        }
        if (count > 0 && options_.dflash_features == DflashExecution::Enabled &&
            state_.dflash_pending_count(lane) != 0) {
            throw std::logic_error("Flash-Next commit would replace unappended DFlash context");
        }
        fold[index] = {.linear_state_slot = lane, .commit_columns = count, .path_length = -1};
        commit.selectors[index] = std::max(0, count - 1);
        commit.enabled[index]   = count == 0 ? 0 : 1;
        commit.counts[index]    = count;
    }
    pending_verify_width_ = 0;
    CUDA_CHECK(cudaMemcpyAsync(verify_.commit_ingress.data, &commit, sizeof(commit),
                               cudaMemcpyHostToDevice, stream_));
    const Tensor selectors =
        at(verify_.commit_ingress, offsetof(VerifyCommitIngress, selectors), DType::I32, {rows});
    const Tensor enabled =
        at(verify_.commit_ingress, offsetof(VerifyCommitIngress, enabled), DType::I32, {rows});
    const SubChunk batch = verify_batch(width, rows);
    ops::gdn_replay_fold(
        verify_records(width), state_.gdn_all_layers(),
        std::span<const ops::GdnReplayFoldRow>(fold.data(), static_cast<std::size_t>(rows)),
        stream_);
    constexpr std::int32_t ple_rows = 9 * kPleKeyRows;
    const std::size_t canonical_bytes =
        static_cast<std::size_t>(state_.lanes()) * ple_rows * sizeof(std::uint16_t);
    const Tensor snapshots(static_cast<std::byte*>(verify_.ple_states.data) + canonical_bytes,
                           DType::BF16, {ple_rows, width, rows});
    Tensor selected_ple = columns(verify_.selected_ple, ple_rows, rows);
    ops::speculative_select_accepted_hidden(snapshots, selectors, selected_ple, stream_);
    Tensor ple = state_.ple_history().view({ple_rows, 1, state_.lanes()});
    ops::scatter_bf16_batch(selected_ple.view({ple_rows, 1, rows}), batch.lanes, enabled, ple,
                            stream_);
    const Tensor packed_logits = logits(width * rows).view({kVocabRows, width, rows});
    Tensor selected_logits     = columns(verify_.selected_logits, kVocabRows, rows);
    ops::speculative_select_accepted_hidden(packed_logits, selectors, selected_logits, stream_);
    Tensor tails = state_.tail_logits_pool().view({kVocabRows, 1, state_.lanes()});
    ops::scatter_bf16_batch(selected_logits.view({kVocabRows, 1, rows}), batch.lanes, enabled,
                            tails, stream_);
    if (options_.dflash_features == DflashExecution::Enabled) {
        const Tensor counts =
            at(verify_.commit_ingress, offsetof(VerifyCommitIngress, counts), DType::I32, {rows});
        Tensor destination = state_.dflash_pending_features().slice(1, 0, width);
        ops::scatter_bf16_batch(dflash_features().view({kDflashFeatures, width, rows}), batch.lanes,
                                counts, destination, stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto lanes =
        std::span<const std::int32_t>(ingress.lanes).first(static_cast<std::size_t>(rows));
    const auto tokens =
        std::span<const std::int32_t>(ingress.tokens)
            .first(static_cast<std::size_t>(rows) * static_cast<std::size_t>(width));
    feeder_.commit_proposals(lanes, tokens, width, commit_counts);
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto index = static_cast<std::size_t>(row);
        state_.advance(lanes[index], commit_counts[index]);
        if (options_.dflash_features == DflashExecution::Enabled && commit_counts[index] != 0) {
            state_.set_dflash_pending_count(lanes[index], commit_counts[index]);
        }
    }
}

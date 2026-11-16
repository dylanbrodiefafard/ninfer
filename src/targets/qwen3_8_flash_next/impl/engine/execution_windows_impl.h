// Included inside the target namespace after EngineProgram::Impl. Live images borrow secured
// RAM credit and never enter prefix lookup or disk spill; request bookkeeping moves with them.

bool EngineProgram::chunked_execution_enabled() const noexcept { return impl_->ram != nullptr; }

runtime::AdmissionResources
EngineProgram::completion_resources(const RequestBasePlan& base,
                                    std::uint32_t prompt_token_bound) const {
    if (prompt_token_bound < base.summary().prompt_tokens) {
        throw std::invalid_argument("Flash-Next completion bound is shorter than its prompt");
    }
    return impl_->execution_window(prompt_token_bound, base.summary().effective_output_tokens)
        .admission;
}

runtime::ExecutionWindow EngineProgram::plan_window(std::uint32_t lane,
                                                    std::uint32_t output_tokens) const {
    const auto& entry = impl_->record(lane);
    if (!entry.lane || output_tokens > entry.effective_output_tokens) {
        throw std::logic_error("Flash-Next window exceeds its request lifetime");
    }
    return impl_->execution_window(static_cast<std::uint32_t>(entry.prompt.size()), output_tokens,
                                   static_cast<std::uint32_t>(entry.tokens.size()));
}

runtime::ExecutionWindow EngineProgram::plan_suspended_window(std::uint64_t owner,
                                                              std::uint32_t output_tokens) const {
    const auto& parked = impl_->suspended.at(owner);
    if (parked.phase != Impl::Suspended::Phase::Parked ||
        output_tokens > parked.request.effective_output_tokens) {
        throw std::logic_error("Flash-Next suspended window exceeds its request lifetime");
    }
    return impl_->execution_window(static_cast<std::uint32_t>(parked.request.prompt.size()),
                                   output_tokens,
                                   static_cast<std::uint32_t>(parked.request.tokens.size()));
}

bool EngineProgram::shrink_window(std::uint32_t lane, const runtime::ExecutionWindow& window) {
    auto& entry = impl_->record(lane);
    if (entry.phase != Impl::Phase::Ready || window.output_tokens >= entry.window_output_tokens ||
        window.output_tokens <= entry.generated.size()) {
        return false;
    }
    const auto expected = plan_window(lane, window.output_tokens);
    if (expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        window.admission.backend_kv_pages != 0) {
        throw std::logic_error("invalid Flash-Next window shrink");
    }
    const auto extent = static_cast<std::uint32_t>(entry.prompt.size()) + window.output_tokens - 1U;
    impl_->program.resize_lane(entry.lane.value(), static_cast<std::int32_t>(entry.tokens.size()),
                               extent);
    entry.reserve_tokens       = extent;
    entry.window_output_tokens = window.output_tokens;
    return true;
}

bool EngineProgram::grant_window(std::uint32_t lane, const runtime::ExecutionWindow& window) {
    auto& entry         = impl_->record(lane);
    const auto expected = plan_window(lane, window.output_tokens);
    if (window.output_tokens < entry.window_output_tokens ||
        expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        window.admission.backend_kv_pages != 0) {
        throw std::logic_error("invalid Flash-Next window grant");
    }
    const auto old_pages = pages_for_tokens(entry.reserve_tokens);
    if (window.admission.main_kv_pages >
        old_pages + impl_->program.kv_free_tokens() / ops::kQsaPageTokens) {
        return false;
    }
    const auto extent =
        static_cast<std::uint32_t>(entry.prompt.size()) + std::max(window.output_tokens, 1U) - 1U;
    impl_->program.resize_lane(entry.lane.value(), static_cast<std::int32_t>(entry.tokens.size()),
                               extent);
    entry.reserve_tokens       = extent;
    entry.window_output_tokens = window.output_tokens;
    return true;
}

bool EngineProgram::reserve_suspension_credit(std::uint64_t owner, std::size_t bytes) {
    return impl_->ram && impl_->ram->reserve_live_credit(owner, bytes);
}

void EngineProgram::release_suspension_credit(std::uint64_t owner) {
    if (impl_->ram) { impl_->ram->release_live_credit(owner); }
}

bool EngineProgram::start_suspend(std::uint32_t lane, std::uint64_t owner) {
    auto& entry = impl_->record(lane);
    if (!impl_->ram || entry.phase != Impl::Phase::Ready || !entry.lane ||
        impl_->suspended.contains(owner)) {
        throw std::logic_error("Flash-Next suspension requires a settled active continuation");
    }
    const auto [position, inserted] = impl_->suspended.try_emplace(owner);
    if (!inserted) { throw std::logic_error("duplicate Flash-Next suspension owner"); }
    auto& saved         = position->second;
    saved.physical_lane = lane;
    saved.sampling_counts_offset =
        impl_->ram->live_credit_view(owner).second - Impl::kSamplingCountsBytes;
    try {
        if (!impl_->capture(entry, &saved.image_id, false, nullptr, {}, false, true, owner)) {
            impl_->suspended.erase(position);
            return false;
        }
        return true;
    } catch (...) {
        impl_->suspended.erase(position);
        throw;
    }
}

bool EngineProgram::suspension_ready(std::uint64_t owner) const {
    const auto& saved = impl_->suspended.at(owner);
    if (saved.phase != Impl::Suspended::Phase::Saving) {
        return saved.phase == Impl::Suspended::Phase::Parked;
    }
    const auto status = cudaEventQuery(impl_->record(saved.physical_lane).capture_done);
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return impl_->ram->live_image_ready(saved.image_id);
}

void EngineProgram::finish_suspend(std::uint32_t lane, std::uint64_t owner) {
    auto& saved = impl_->suspended.at(owner);
    if (saved.phase != Impl::Suspended::Phase::Saving || saved.physical_lane != lane ||
        !suspension_ready(owner)) {
        throw std::logic_error("Flash-Next suspension has not settled");
    }
    auto& entry = impl_->record(lane);
    impl_->program.close_lane(entry.lane.value());
    entry.lane.reset();
    saved.request = std::move(entry);
    entry         = Impl::LaneRecord{};
    impl_->bind_images(entry, lane);
    saved.phase = Impl::Suspended::Phase::Parked;
}

bool EngineProgram::start_resume(std::uint32_t lane, std::uint64_t owner,
                                 const runtime::ExecutionWindow& window) {
    auto& saved         = impl_->suspended.at(owner);
    auto& entry         = impl_->record(lane);
    const auto expected = plan_suspended_window(owner, window.output_tokens);
    if (entry.lane || expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        window.admission.backend_kv_pages != 0) {
        throw std::logic_error("invalid Flash-Next resume destination");
    }
    const auto extent       = static_cast<std::uint32_t>(saved.request.prompt.size()) +
                              std::max(window.output_tokens, 1U) - 1U;
    const auto program_lane = impl_->program.open_lane(extent, saved.request.sampling);
    if (!program_lane) { return false; }
    entry = std::move(saved.request);
    impl_->bind_images(entry, lane);
    entry.lane                 = program_lane;
    entry.reserve_tokens       = extent;
    entry.window_output_tokens = window.output_tokens;
    saved.phase                = Impl::Suspended::Phase::Loading;
    saved.physical_lane        = lane;
    try {
        auto& state = impl_->program.state();
        tier::RamRestoreTarget target;
        target.text             = &state.pages(*entry.lane);
        target.text_pool        = &state.pool();
        target.text_dst_pages   = pages_for_tokens(static_cast<std::uint32_t>(entry.tokens.size()));
        target.flash_next_state = impl_->image(entry.tier_image);
        target.flash_next_rewrite  = impl_->image(entry.rewrite_image);
        target.flash_next_writable = entry.capture_done;
        target.reuse               = PrefixReusePath::AppendAtFrontier;
        target.reuse_base          = static_cast<std::uint32_t>(entry.tokens.size());
        target.stream              = impl_->device.copy_stream;
        impl_->device.order_copy_after_compute();
        const auto host = impl_->ram->unpack_device(saved.image_id, target);
        for (const auto& head : host.ladder_images) {
            if (head.kind == tier::ContextCheckpointKind::TurnRollback &&
                head.frontier == entry.rollback_frontier) {
                const HostCopy copy{entry.rollback_image, head.flash_next_state.data(),
                                    head.flash_next_state.size()};
                impl_->ram->copy_from_entry(saved.image_id, std::span(&copy, 1),
                                            std::span(&entry.capture_done, 1));
            }
        }
        return true;
    } catch (...) {
        impl_->device.synchronize_all();
        impl_->close(entry);
        throw;
    }
}

bool EngineProgram::resume_ready(std::uint64_t owner) const {
    const auto& saved = impl_->suspended.at(owner);
    if (saved.phase != Impl::Suspended::Phase::Loading) { return false; }
    const auto status = cudaEventQuery(impl_->record(saved.physical_lane).capture_done);
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return impl_->ram->copies_ready(saved.image_id);
}

void EngineProgram::finish_resume(std::uint32_t lane, std::uint64_t owner) {
    auto& saved = impl_->suspended.at(owner);
    if (saved.physical_lane != lane || !resume_ready(owner)) {
        throw std::logic_error("Flash-Next resume has not settled");
    }
    auto& entry = impl_->record(lane);
    impl_->ram->wait_entry_copies_on_stream(saved.image_id, impl_->device.stream);
    impl_->program.restore_lane(entry.lane.value(), entry.tokens, entry.sampling, entry.tier_image,
                                RestorePhase::Decoding, &entry.speculative);
    const auto [block, count] = impl_->ram->live_credit_view(owner);
    if (saved.sampling_counts_offset > count ||
        Impl::kSamplingCountsBytes > count - saved.sampling_counts_offset) {
        throw std::logic_error("Flash-Next saved sampling counts exceed the secured credit");
    }
    const auto* counts = static_cast<const std::byte*>(block) + saved.sampling_counts_offset;
    CUDA_CHECK(cudaMemcpyAsync(impl_->program.state().token_counts(entry.lane.value()), counts,
                               Impl::kSamplingCountsBytes, cudaMemcpyHostToDevice,
                               impl_->device.stream));
    CUDA_CHECK(cudaStreamSynchronize(impl_->device.stream));
    entry.phase = Impl::Phase::Ready;
    impl_->ram->retire_live_image(saved.image_id);
    impl_->suspended.erase(owner);
}

void EngineProgram::discard_suspended(std::uint64_t owner) {
    const auto position = impl_->suspended.find(owner);
    if (position == impl_->suspended.end()) { return; }
    auto& saved = position->second;
    if (saved.phase != Impl::Suspended::Phase::Parked) {
        auto& entry = impl_->record(saved.physical_lane);
        CUDA_CHECK(cudaEventSynchronize(entry.capture_done));
        impl_->close(entry);
    }
    impl_->ram->wait_live_image(saved.image_id);
    impl_->ram->retire_live_image(saved.image_id);
    impl_->suspended.erase(position);
}

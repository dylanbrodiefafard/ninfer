// Included inside the target namespace after EngineProgram::Impl is complete. The shared tiers
// own byte transport and indexing; this adapter owns the complete Flash-Next continuation layout.

void EngineProgram::retarget_plan_to_ram(RequestPlan& plan, std::uint64_t entry_id) {
    if (!impl_->ram || entry_id == 0) {
        throw std::logic_error("Flash-Next RAM retarget has no source");
    }
    plan.base_.summary_.reuse_source = PrefixReuseSource::HostRam;
    plan.base_.summary_.ram_entry_id = entry_id;
}

bool EngineProgram::preserve_retained_lane(std::uint32_t lane, std::uint64_t& entry_id,
                                           bool may_block, bool& deferred) {
    return impl_->capture(impl_->record(lane), &entry_id, may_block, &deferred, {}, false, true);
}

bool EngineProgram::capture_retained_lane(std::uint32_t lane, std::uint64_t* ram_entry_id,
                                          bool may_block, bool* deferred,
                                          std::span<const std::uint64_t> attempt_ram_ids) {
    if (!has_retained_lane(lane)) { return false; }
    return impl_->capture(impl_->record(lane), ram_entry_id, may_block, deferred, attempt_ram_ids);
}

void EngineProgram::restore_ram_entry(std::uint32_t lane, std::uint64_t entry_id,
                                      const RequestPlan& plan) {
    if (!impl_->ram || plan.summary().ram_entry_id != entry_id ||
        plan.summary().reuse_source != PrefixReuseSource::HostRam ||
        plan.summary().reusable_prompt_tokens == 0) {
        throw std::logic_error("Flash-Next RAM restore requires a winning complete-state plan");
    }
    auto& entry = impl_->record(lane);
    if (entry.lane) { throw std::logic_error("Flash-Next RAM restore needs an empty lane"); }
    const auto frontier = plan.summary().reusable_prompt_tokens;
    try {
        entry.lane = impl_->program.open_lane(plan.base_.reserve_tokens_, LaneSampling{});
        if (!entry.lane) { throw std::logic_error("Flash-Next RAM restore lost its reservation"); }
        auto& state = impl_->program.state();
        tier::RamRestoreTarget target;
        target.text_dst_pages      = pages_for_tokens(frontier);
        target.text                = &state.pages(*entry.lane);
        target.text_pool           = &state.pool();
        target.flash_next_state    = impl_->image(entry.tier_image);
        target.flash_next_rewrite  = impl_->image(entry.rewrite_image);
        target.flash_next_writable = entry.capture_done;
        target.reuse               = plan.summary().disk_reuse_path;
        target.reuse_base          = frontier;
        target.stream              = impl_->device.copy_stream;
        impl_->device.order_copy_after_compute();
        auto host = impl_->ram->unpack_device(entry_id, target);
        entry.tokens.assign(host.ledger.begin(), host.ledger.begin() + frontier);
        entry.identity      = std::move(host.identity);
        entry.disk_entry_id = host.disk_entry_id;
        entry.rewrite.reset();
        if (host.rewrite_valid && host.rewrite_frontier <= frontier) {
            entry.rewrite = qwen3_6::RewriteCheckpointSpec{.kind     = host.rewrite_kind,
                                                           .frontier = host.rewrite_frontier};
        }
        entry.rollback_frontier = 0;
        for (const auto& head : host.ladder_images) {
            if (head.kind == tier::ContextCheckpointKind::TurnRollback &&
                head.frontier <= frontier) {
                if (head.flash_next_state.size() != state.continuation_bytes()) {
                    throw std::logic_error("Flash-Next RAM rollback image is incomplete");
                }
                const HostCopy copy{entry.rollback_image, head.flash_next_state.data(),
                                    head.flash_next_state.size()};
                const auto fence = entry.capture_done;
                impl_->ram->copy_from_entry(entry_id, std::span(&copy, 1), std::span(&fence, 1));
                entry.rollback_frontier = head.frontier;
            }
        }
        impl_->program.extents().record_tier_copy(entry.capture_done);
        entry.tier_restored  = true;
        entry.phase          = Impl::Phase::Retained;
        entry.reserve_tokens = plan.base_.reserve_tokens_;
    } catch (const std::bad_alloc&) {
        impl_->device.synchronize_all();
        impl_->close(entry);
        throw runtime::CacheRestoreFailure(
            "Flash-Next RAM restore metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    } catch (...) {
        impl_->device.synchronize_all();
        impl_->close(entry);
        throw;
    }
}

void EngineProgram::restore_disk_entry(std::uint32_t lane, std::uint64_t entry_id,
                                       const RequestPlan& plan) {
    if (!impl_->disk || plan.summary().disk_entry_id != entry_id ||
        plan.summary().reuse_source != PrefixReuseSource::HostDisk ||
        plan.summary().reusable_prompt_tokens == 0) {
        throw std::logic_error("Flash-Next disk restore requires a winning complete-state plan");
    }
    auto& entry = impl_->record(lane);
    if (entry.lane || impl_->pending_disk_ticket != 0) {
        throw std::logic_error(
            "Flash-Next disk restore needs an empty lane and no pending restore");
    }
    const auto frontier = plan.summary().reusable_prompt_tokens;
    try {
        auto host = impl_->disk->load_host(entry_id, plan.summary().disk_committed_generation);
        if (!host) { throw runtime::CacheRestoreFailure("Flash-Next disk source is unavailable"); }
        entry.lane = impl_->program.open_lane(plan.base_.reserve_tokens_, LaneSampling{});
        if (!entry.lane) { throw std::logic_error("Flash-Next disk restore lost its reservation"); }
        auto& state = impl_->program.state();
        tier::DiskRestoreTarget target;
        target.text_dst_pages       = pages_for_tokens(frontier);
        target.text                 = &state.pages(*entry.lane);
        target.text_pool            = &state.pool();
        target.flash_next_state     = impl_->image(entry.tier_image);
        target.flash_next_rewrite   = impl_->image(entry.rewrite_image);
        target.flash_next_writable  = entry.capture_done;
        target.reuse                = plan.summary().disk_reuse_path;
        target.reuse_base           = frontier;
        target.committed_generation = plan.summary().disk_committed_generation;
        target.stream               = impl_->device.copy_stream;
        entry.tokens.assign(host->ledger.begin(), host->ledger.begin() + frontier);
        entry.identity      = std::move(host->identity);
        entry.disk_entry_id = entry_id;
        entry.rewrite.reset();
        if (host->rewrite_valid && host->rewrite_frontier <= frontier) {
            entry.rewrite = qwen3_6::RewriteCheckpointSpec{.kind     = host->rewrite_kind,
                                                           .frontier = host->rewrite_frontier};
        }
        entry.rollback_frontier = 0;
        for (const auto& head : host->ladders) {
            if (head.kind == tier::ContextCheckpointKind::TurnRollback &&
                head.frontier <= frontier) {
                tier::DiskCheckpointImageTarget image;
                image.frontier         = head.frontier;
                image.hash             = head.hash;
                image.kind             = head.kind;
                image.flash_next_state = impl_->image(entry.rollback_image);
                image.writable         = entry.capture_done;
                target.checkpoint_images.push_back(image);
                entry.rollback_frontier = head.frontier;
            }
        }
        impl_->device.order_copy_after_compute();
        impl_->pending_disk_ticket = impl_->disk->restore_device(entry_id, target);
        impl_->pending_disk_owner  = &entry;
        entry.tier_restored        = true;
        entry.phase                = Impl::Phase::Retained;
        entry.reserve_tokens       = plan.base_.reserve_tokens_;
    } catch (const std::bad_alloc&) {
        cancel_disk_restore();
        impl_->device.synchronize_all();
        impl_->close(entry);
        throw runtime::CacheRestoreFailure(
            "Flash-Next disk restore metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    } catch (...) {
        cancel_disk_restore();
        impl_->device.synchronize_all();
        impl_->close(entry);
        throw;
    }
}

bool EngineProgram::ram_restore_ready(std::uint64_t entry_id) const {
    return !impl_->ram || impl_->ram->copies_ready(entry_id);
}

bool EngineProgram::disk_restore_ready(std::uint64_t entry_id) const {
    return !impl_->disk || impl_->disk->restore_setup_ready(entry_id);
}

bool EngineProgram::kv_ram_reclaim_pending() const {
    return impl_->disk && impl_->disk->ram_reclaim_pending();
}

void EngineProgram::claim_ram_entry(std::uint64_t entry_id) { impl_->ram->claim(entry_id); }

void EngineProgram::release_ram_entry(std::uint64_t entry_id) { impl_->ram->release(entry_id); }

void EngineProgram::finish_ram_restore(std::uint64_t entry_id, runtime::CacheReadIntent intent) {
    if (intent == runtime::CacheReadIntent::Consume && !impl_->ram->is_system_prompt(entry_id)) {
        impl_->ram->consume(entry_id);
        if (impl_->disk) { impl_->disk->forget_ram_resident(entry_id); }
    } else {
        impl_->ram->release_restored(entry_id);
    }
}

bool EngineProgram::claim_disk_entry(std::uint64_t entry_id, std::uint32_t expected_frontier,
                                     std::uint64_t hash_lo, std::uint64_t hash_hi,
                                     std::uint32_t expected_reuse_base,
                                     PrefixReusePath expected_reuse,
                                     std::uint64_t expected_committed_generation) {
    return impl_->disk &&
           impl_->disk->claim(entry_id, {hash_lo, hash_hi}, expected_frontier, expected_reuse_base,
                              expected_reuse, expected_committed_generation);
}

void EngineProgram::release_disk_entry(std::uint64_t entry_id) { impl_->disk->release(entry_id); }

void EngineProgram::invalidate_disk_entry(std::uint64_t entry_id) {
    impl_->disk->invalidate_entry(entry_id);
}

void EngineProgram::consume_disk_entry(std::uint64_t entry_id) { impl_->disk->consume(entry_id); }

void EngineProgram::prefetch_disk_plan(std::uint64_t entry_id, const RequestPlan& plan) {
    if (impl_->disk) {
        impl_->disk->prefetch_window(
            entry_id, std::min(2U, pages_for_tokens(plan.summary().reusable_prompt_tokens)), 0);
    }
}

void EngineProgram::discard_ram_capture(std::uint64_t ram_id) {
    if (ram_id == 0 || !impl_->ram) { return; }
    if (impl_->disk) { impl_->disk->abandon_ram_spill(ram_id); }
    impl_->ram->discard(ram_id);
}

void EngineProgram::pump_disk_restore() {
    try {
        if (impl_->disk) { impl_->disk->pump_restore(impl_->device.copy_stream); }
    } catch (const std::bad_alloc&) {
        cancel_disk_restore();
        impl_->device.synchronize_all();
        throw runtime::CacheRestoreFailure(
            "Flash-Next disk copy metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    }
}

void EngineProgram::cancel_disk_restore() {
    if (!impl_->disk) { return; }
    impl_->disk->cancel_restore();
    if (impl_->pending_disk_ticket != 0) {
        impl_->disk->release_restore_ticket(impl_->pending_disk_ticket);
    }
    impl_->pending_disk_ticket = 0;
    impl_->pending_disk_owner  = nullptr;
}

void EngineProgram::begin_copy_hold_cancel() {
    if (impl_->disk) { impl_->disk->begin_cancel_restore(); }
    CUDA_CHECK(cudaEventRecord(impl_->copy_cancel.get(), impl_->device.copy_stream));
}

bool EngineProgram::copy_hold_cancel_settled() const {
    const auto status = cudaEventQuery(impl_->copy_cancel.get());
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return !impl_->disk || impl_->disk->restore_cancel_settled();
}

void EngineProgram::shutdown_kv_tiers(const LoadProgress& progress) {
    if (impl_->tiers_shutdown) { return; }
    impl_->tiers_shutdown = true;
    for (std::uint32_t lane = 0; lane < static_cast<std::uint32_t>(impl_->program.lanes());
         ++lane) {
        impl_->lane_activity(lane);
    }
    while (!impl_->suspended.empty()) { discard_suspended(impl_->suspended.begin()->first); }
    if (!impl_->disk) { return; }
    cancel_disk_restore();
    for (std::uint32_t lane = 0; lane < static_cast<std::uint32_t>(impl_->program.lanes());
         ++lane) {
        if (has_retained_lane(lane)) { (void)capture_retained_lane(lane); }
    }
    if (progress.callback) { progress.callback("kv-disk save cache entries", 0, 0); }
    impl_->disk->flush_not_durable_ram();
    impl_->disk->wait_idle_and_fsync();
}

void EngineProgram::set_cache_idle(bool idle) {
    if (impl_->disk) { impl_->disk->set_cache_idle(idle); }
}

tier::KvRamSnapshot EngineProgram::kv_ram_snapshot() const noexcept {
    return impl_->ram ? impl_->ram->snapshot() : tier::KvRamSnapshot{};
}

tier::KvRamCopySeconds EngineProgram::harvest_kv_ram_copy_seconds() {
    return impl_->ram ? impl_->ram->harvest_copy_seconds() : tier::KvRamCopySeconds{};
}

std::optional<tier::KvDiskSnapshot> EngineProgram::try_kv_disk_snapshot() const {
    return impl_->disk ? impl_->disk->try_snapshot() : std::optional{tier::KvDiskSnapshot{}};
}

tier::KvDiskCopySeconds EngineProgram::harvest_kv_disk_copy_seconds() {
    return impl_->disk ? impl_->disk->harvest_copy_seconds() : tier::KvDiskCopySeconds{};
}

bool EngineProgram::kv_ram_copies_ready() const {
    // A failed expiry capture can still have an asynchronous continuation snapshot in flight,
    // without a RAM record. Admission must preserve decode overlap for that fence as well.
    for (const auto& entry : impl_->records) {
        if (!entry.lane) { continue; }
        const auto status = cudaEventQuery(entry.capture_done);
        if (status == cudaErrorNotReady) { return false; }
        CUDA_CHECK(status);
    }
    return !impl_->ram || impl_->ram->pending_copies_ready();
}

bool EngineProgram::kv_disk_copies_ready() const {
    return !impl_->disk || impl_->disk->copies_ready();
}

bool EngineProgram::kv_disk_restore_failed() const {
    return impl_->disk && impl_->disk->restore_failed();
}

bool EngineProgram::kv_copies_ready() const {
    return kv_ram_copies_ready() && kv_disk_copies_ready();
}

void EngineProgram::wait_kv_ram_copies_on_compute() {
    if (impl_->ram) { impl_->ram->wait_pending_copies_on_stream(impl_->device.stream); }
    if (impl_->disk) {
        impl_->disk->wait_copies_on_stream(impl_->device.stream, impl_->pending_disk_ticket);
        if (impl_->pending_disk_ticket != 0) {
            impl_->disk->release_restore_ticket(impl_->pending_disk_ticket);
        }
        impl_->pending_disk_ticket = 0;
        impl_->pending_disk_owner  = nullptr;
    }
}

void EngineProgram::wait_kv_ram_copies() {
    if (impl_->ram) { impl_->ram->wait_pending_copies(); }
}

void EngineProgram::wait_kv_disk_copies() {
    if (impl_->disk) { impl_->disk->wait_copies(impl_->pending_disk_ticket); }
}

std::uint64_t EngineProgram::pending_disk_restore_ticket() const noexcept {
    return impl_->pending_disk_ticket;
}

std::uint64_t EngineProgram::kv_ram_index_version() const noexcept {
    return impl_->ram ? impl_->ram->index_version() : 0;
}

std::uint64_t EngineProgram::kv_disk_index_version() const {
    return impl_->disk ? impl_->disk->index_version() : 0;
}

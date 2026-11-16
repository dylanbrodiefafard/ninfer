// Exact complete-state transport at the real Flash-Next GDN/PLE/QSA geometry, without weights.
#include "runtime/state.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace fn   = ninfer::targets::qwen3_8_flash_next;
namespace tier = ninfer::targets::qwen3_6::detail;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

std::span<std::byte> bytes(ninfer::PinnedHostBuffer& image) {
    return {static_cast<std::byte*>(image.data()), image.size()};
}

void fill(fn::State& state, std::int32_t lane, ninfer::PinnedHostBuffer& image, unsigned seed,
          cudaStream_t stream) {
    const auto pattern = [&](const ninfer::Tensor& tensor) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, static_cast<int>(seed), tensor.bytes(), stream));
    };
    for (std::int32_t layer = 0; layer < fn::kGdnLayers; ++layer) {
        pattern(state.gdn_convolution_pool(layer));
        pattern(state.gdn_recurrent_pool(layer));
    }
    pattern(state.ple_history());
    pattern(state.tail_logits_pool());
    if (state.has_dflash()) {
        for (std::int32_t layer = 0; layer < fn::kDflashLayers; ++layer) {
            const auto context = state.dflash_cache().layer_view(static_cast<std::uint32_t>(layer));
            pattern(context.k);
            pattern(context.v);
        }
        pattern(state.dflash_pending_features());
        auto& calibration             = state.dflash_lane(lane);
        calibration.pending           = static_cast<std::int32_t>(seed % fn::kDflashWidth);
        calibration.draft_temperature = 0.6F;
        calibration.calibration_sum.fill(static_cast<double>(seed) / 16.0);
        calibration.calibration_count.fill(16);
    }
    state.pack_continuation(lane, image.data(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

struct Directory {
    Directory() {
        std::array<char, 64> name{};
        const std::string pattern =
            (std::filesystem::temp_directory_path() / "ninfer-fn-tier.XXXXXX").string();
        require(pattern.size() < name.size(), "tier scratch path is too long");
        std::memcpy(name.data(), pattern.c_str(), pattern.size() + 1);
        const auto* result = ::mkdtemp(name.data());
        if (result == nullptr) { throw std::runtime_error("cannot create tier scratch directory"); }
        path = result;
    }

    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

void run(ninfer::DeviceContext& device, ninfer::KvDiskCompress codec,
         fn::DflashExecution execution) {
    fn::State state(2, 256, 256, ninfer::ops::QsaKvFormat::NVFP4G16, execution);
    const auto source_lane = state.open(128, device.stream).value();
    state.advance(source_lane, 65);
    const auto image_bytes = state.continuation_bytes();
    ninfer::PinnedHostBuffer current(image_bytes), rewrite(image_bytes), rollback(image_bytes);
    ninfer::PinnedHostBuffer restored(image_bytes), restored_rewrite(image_bytes),
        restored_rollback(image_bytes);
    fill(state, source_lane, current, 11, device.stream);
    fill(state, source_lane, rewrite, 89, device.stream);
    fill(state, source_lane, rollback, 173, device.stream);
    state.unpack_continuation(source_lane, current.data(), device.stream);
    for (std::size_t p = 0; p < state.pool().plane_count(); ++p) {
        const auto& plane = state.pool().plane(p);
        CUDA_CHECK(
            cudaMemsetAsync(plane.data, static_cast<int>(17U + p), plane.bytes(), device.stream));
    }
    const auto page_bytes = ninfer::paged_kv_host_image_bytes(state.pool(), 2);
    ninfer::PinnedHostBuffer original_pages(page_bytes), restored_pages(page_bytes);
    ninfer::pack_paged_kv_allocation_to_host(state.pages(source_lane), state.pool(),
                                             original_pages.data(), 2, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));

    ninfer::targets::qwen3_6::PreparedPromptData prompt;
    prompt.token_ids.resize(70);
    prompt.token_types.resize(70, 0);
    prompt.positions.resize(210);
    prompt.identity.reusable = true;
    for (std::int32_t p = 0; p < 70; ++p) {
        prompt.token_ids[static_cast<std::size_t>(p)] = 1000 + p;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            prompt.positions[axis * 70 + static_cast<std::size_t>(p)] = p;
        }
    }
    tier::ResidentPrefixIdentity identity;
    identity.assign(prompt);
    identity.truncate(66);
    tier::RamCaptureSource source;
    source.execution_frontier = 65;
    source.ledger_frontier    = 66;
    source.text_kv_valid      = 65;
    source.tail_hidden_valid  = true;
    source.ledger             = std::span(prompt.token_ids).first(66);
    source.identity           = &identity;
    source.hash_f             = tier::prefix_hash_at(source.ledger, identity, 65);
    source.rewrite_valid      = true;
    source.rewrite_frontier   = 63;
    source.hash_c_valid       = true;
    source.hash_c             = tier::prefix_hash_at(source.ledger, identity, 63);
    source.text               = &state.pages(source_lane);
    source.text_pool          = &state.pool();
    source.flash_next_state   = bytes(current);
    source.flash_next_rewrite = bytes(rewrite);
    source.stream             = device.copy_stream;
    tier::RamLadderHead head;
    head.frontier         = 64;
    head.hash             = tier::prefix_hash_at(source.ledger, identity, 64);
    head.kind             = tier::ContextCheckpointKind::TurnRollback;
    head.flash_next_state = bytes(rollback);
    source.ladder_heads.push_back(head);
    tier::KVRamCache ram(std::size_t{768} << 20);
    device.order_copy_after_compute();
    const auto captured = ram.capture(source);
    require(captured.status == tier::RamCaptureStatus::Captured,
            "complete-state RAM capture failed");
    ram.wait_entry_copies(captured.entry_id);
    const auto destination = state.open(128, device.stream).value();
    require(state.pages(destination).page_ids()[0] != state.pages(source_lane).page_ids()[0],
            "restore fixture aliases source pages");
    auto check = [&](const ninfer::PinnedHostBuffer& expected, std::uint32_t pages) {
        require(std::memcmp(restored.data(), expected.data(), image_bytes) == 0,
                "transport changed GDN/PLE/logit image bytes");
        state.unpack_continuation(destination, restored.data(), device.stream);
        state.pack_continuation(destination, restored_rewrite.data(), device.stream);
        ninfer::pack_paged_kv_allocation_to_host(state.pages(destination), state.pool(),
                                                 restored_pages.data(), pages, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        require(std::memcmp(restored_rewrite.data(), expected.data(), image_bytes) == 0,
                "restored device GDN/PLE/logit state differs");
        // Packed images are plane-major, so truncating two pages to one changes every plane's
        // offset rather than selecting one contiguous prefix of the two-page image.
        std::size_t source_offset      = 0;
        std::size_t destination_offset = 0;
        for (std::size_t p = 0; p < state.pool().plane_count(); ++p) {
            const auto per_page = static_cast<std::size_t>(state.pool().plane(p).nb[3]);
            const auto count    = per_page * pages;
            require(std::memcmp(
                        static_cast<const std::byte*>(restored_pages.data()) + destination_offset,
                        static_cast<const std::byte*>(original_pages.data()) + source_offset,
                        count) == 0,
                    "restored QSA pages differ");
            source_offset += per_page * 2U;
            destination_offset += count;
        }
    };
    ram.claim(captured.entry_id);
    for (const auto path :
         {ninfer::PrefixReusePath::AppendAtFrontier, ninfer::PrefixReusePath::RestoreTurnCheckpoint,
          ninfer::PrefixReusePath::RestoreTurnRollback}) {
        tier::RamRestoreTarget target;
        target.text               = &state.pages(destination);
        target.text_pool          = &state.pool();
        target.text_dst_pages     = path == ninfer::PrefixReusePath::AppendAtFrontier ? 2U : 1U;
        target.flash_next_state   = bytes(restored);
        target.flash_next_rewrite = bytes(restored_rewrite);
        target.reuse              = path;
        target.reuse_base         = path == ninfer::PrefixReusePath::AppendAtFrontier        ? 65U
                                    : path == ninfer::PrefixReusePath::RestoreTurnCheckpoint ? 63U
                                                                                             : 64U;
        target.stream             = device.copy_stream;
        (void)ram.unpack_device(captured.entry_id, target);
        ram.wait_entry_copies(captured.entry_id);
        require(std::memcmp(restored_rewrite.data(), rewrite.data(), image_bytes) == 0,
                "RAM rewrite image differs");
        check(path == ninfer::PrefixReusePath::AppendAtFrontier        ? current
              : path == ninfer::PrefixReusePath::RestoreTurnCheckpoint ? rewrite
                                                                       : rollback,
              target.text_dst_pages);
    }
    ram.release(captured.entry_id);
    Directory directory;
    tier::DiskOpenConfig config;
    config.location           = directory.path;
    config.capacity_bytes     = std::size_t{512} << 20;
    config.compress           = codec;
    config.max_context        = 256;
    config.ram                = &ram;
    config.text_pool          = &state.pool();
    config.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(state.pool());
    config.fingerprint        = tier::make_disk_fingerprint(
        "qwen3.8-flash-next", "fixture", "exact-fixture", ninfer::KvCacheStorage::Nvfp4,
        execution == fn::DflashExecution::Enabled ? ninfer::SpeculativeBackend::DFlash
                                                  : ninfer::SpeculativeBackend::None,
        state.pool(), nullptr, nullptr, nullptr);
    config.fingerprint.flash_next_state_bytes    = image_bytes;
    config.fingerprint.flash_next_ngram_identity = "fixture-table";
    config.fingerprint.flash_next_select_budget  = 512;
    {
        tier::KVDiskCache disk(config);
        disk.note_ram_resident(captured.entry_id, 0);
        require(disk.emergency_spill_ram(captured.entry_id), "complete-state disk spill failed");
        disk.wait_idle_and_fsync();
    }
    tier::KVDiskCache reopened(config);
    const auto match = reopened.plan_match(prompt, tier::prefix_hash_chain(prompt));
    require(match.has_value() && match->reuse_base == 65, "reopened disk lost complete state");
    const auto entry_id = match.value().entry_id;
    require(reopened.claim(entry_id), "disk claim failed");
    for (const auto path :
         {ninfer::PrefixReusePath::AppendAtFrontier, ninfer::PrefixReusePath::RestoreTurnCheckpoint,
          ninfer::PrefixReusePath::RestoreTurnRollback}) {
        tier::DiskRestoreTarget target;
        target.text               = &state.pages(destination);
        target.text_pool          = &state.pool();
        target.text_dst_pages     = path == ninfer::PrefixReusePath::AppendAtFrontier ? 2U : 1U;
        target.flash_next_state   = bytes(restored);
        target.flash_next_rewrite = bytes(restored_rewrite);
        target.reuse              = path;
        target.reuse_base         = path == ninfer::PrefixReusePath::AppendAtFrontier        ? 65U
                                    : path == ninfer::PrefixReusePath::RestoreTurnCheckpoint ? 63U
                                                                                             : 64U;
        target.stream             = device.copy_stream;
        const auto ticket         = reopened.restore_device(entry_id, target);
        reopened.wait_copies(ticket);
        require(!reopened.restore_failed(), "disk restore failed");
        require(std::memcmp(restored_rewrite.data(), rewrite.data(), image_bytes) == 0,
                "disk rewrite image differs");
        check(path == ninfer::PrefixReusePath::AppendAtFrontier        ? current
              : path == ninfer::PrefixReusePath::RestoreTurnCheckpoint ? rewrite
                                                                       : rollback,
              target.text_dst_pages);
        reopened.release_restore_ticket(ticket);
        reopened.cancel_restore();
    }
    reopened.release(entry_id);
    state.pack_continuation(source_lane, restored.data(), device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    require(std::memcmp(restored.data(), current.data(), image_bytes) == 0,
            "restore overwrote its source lane");
    state.close(destination, device.stream);
    state.close(source_lane, device.stream);
    std::cout << "Flash-Next RAM and reopened "
              << (codec == ninfer::KvDiskCompress::Off ? "raw" : "Zstd")
              << " disk: current/rewrite/rollback, " << image_bytes
              << " continuation bytes and exact QSA pages pass\n";
}
} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        for (const auto execution : {fn::DflashExecution::Disabled, fn::DflashExecution::Enabled}) {
            run(device, ninfer::KvDiskCompress::Off, execution);
            run(device, ninfer::KvDiskCompress::Zstd, execution);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

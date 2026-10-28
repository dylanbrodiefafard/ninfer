#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "runtime/contract/types.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
namespace q3    = ninfer::targets::qwen3;
namespace cache = q3::detail;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

struct TemporaryDirectory {
    std::filesystem::path path;

    TemporaryDirectory() {
        // temp_directory_path honors TMPDIR, which the unit-test runner points at the build
        // tree so fsync-heavy disk-tier scratch stays off a container overlay.
        std::string pattern =
            (std::filesystem::temp_directory_path() / "ninfer-fixed-kv-XXXXXX").string();
        const char* created = ::mkdtemp(pattern.data());
        if (!created) { throw std::runtime_error("mkdtemp failed"); }
        path = created;
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

struct Pool {
    ninfer::Fp8KInt4VPagedKVSpec spec;
    ninfer::Fp8KInt4VPagedKVPoolLayout layout;
    std::unique_ptr<ninfer::DeviceArena> arena;
    std::unique_ptr<ninfer::PagedKVPool> storage;

    explicit Pool(std::uint32_t pages) {
        spec = {
            .page_group_count      = pages,
            .logical_page_capacity = 4,
            .table_rows            = 2,
            .layer_count           = 2,
            .head_dim              = 128,
            .num_kv_heads          = 2,
            .value_group           = 32,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor}};
        ninfer::LayoutBuilder builder;
        layout  = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        arena   = std::make_unique<ninfer::DeviceArena>(builder.finish(256));
        storage = std::make_unique<ninfer::PagedKVPool>(
            ninfer::DeviceSpan{arena->base(), arena->capacity()}, layout.storage);
    }

    auto semantics() const { return ninfer::fp8_k_int4_v_semantic_fingerprint(spec); }
};

q3::PreparedPromptData prompt(std::size_t tokens = 130) {
    q3::PreparedPromptData result;
    result.token_ids.resize(tokens);
    result.token_types.assign(tokens, 0);
    result.positions.resize(tokens * 3);
    for (std::size_t i = 0; i < tokens; ++i) {
        result.token_ids[i] = static_cast<ninfer::TokenId>(100 + i);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.positions[axis * tokens + i] = static_cast<std::int32_t>(i);
        }
    }
    q3::VisionItem image;
    image.content_digest[0] = 42;
    image.token_spans.push_back({8, 8});
    result.vision_items.push_back(image);
    std::fill(result.token_types.begin() + 8, result.token_types.begin() + 16, 1);
    return result;
}

cache::DiskOpenConfig config(const std::filesystem::path& path, Pool& pool, cache::KVRamCache& ram,
                             ninfer::KvDiskCompress compression,
                             std::size_t capacity_bytes = 32ULL << 20) {
    cache::DiskOpenConfig result;
    result.location           = path;
    result.capacity_bytes     = capacity_bytes;
    result.compress           = compression;
    result.max_context        = 256;
    result.ram                = &ram;
    result.text_pool          = pool.storage.get();
    result.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(*pool.storage);
    result.fingerprint        = cache::make_disk_fingerprint(
        "qwen3.8-27b", "r9700-int-candidate", "fixed-cache-test-artifact",
        ninfer::SpeculativeBackend::None, *pool.storage, nullptr, pool.semantics(), std::nullopt,
        nullptr, nullptr);
    return result;
}

void verify_crc() {
    std::vector<std::uint8_t> bytes(8193);
    for (std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = static_cast<std::uint8_t>(i * 73); }
    for (std::size_t length : {0U, 9U, 257U, 8193U}) {
        std::uint32_t oracle = ~0U;
        for (std::size_t i = 0; i < length; ++i) {
            oracle ^= bytes[i];
            for (int bit = 0; bit < 8; ++bit) {
                oracle = (oracle >> 1) ^ ((oracle & 1U) ? 0x82f63b78U : 0U);
            }
        }
        require(cache::KVDiskCache::test_crc32c(std::span(bytes).first(length)) == ~oracle,
                "CRC32C differs from independent bitwise oracle");
    }
}

std::uint64_t restore_with_submitted_join_check(cache::KVDiskCache& disk, std::uint64_t entry_id,
                                                const cache::DiskRestoreTarget& target);

void run_roundtrip(ninfer::DeviceContext& device, ninfer::KvDiskCompress compression) {
    TemporaryDirectory directory;
    Pool source_pool(8);
    cache::KVRamCache ram(8ULL << 20);
    auto allocation = source_pool.storage->reserve(3);
    allocation.materialize_pages(3, device.stream);
    const auto page_bytes = ninfer::paged_kv_logical_page_bytes(*source_pool.storage);
    ninfer::PinnedHostBuffer page(page_bytes);
    std::vector<std::vector<std::uint8_t>> expected(3, std::vector<std::uint8_t>(page_bytes));
    for (std::size_t index = 0; index < expected.size(); ++index) {
        for (std::size_t offset = 0; offset < page_bytes; ++offset) {
            expected[index][offset] = static_cast<std::uint8_t>((offset / 64 + index * 29) % 251);
        }
        std::memcpy(page.data(), expected[index].data(), page_bytes);
        ninfer::unpack_paged_kv_logical_page_from_host(
            allocation, *source_pool.storage, page.data(), static_cast<std::uint32_t>(index),
            device.copy_stream);
        device.synchronize_all();
    }

    auto retained = prompt();
    cache::ResidentPrefixIdentity identity;
    identity.assign(retained);
    cache::RamCaptureSource capture;
    capture.execution_frontier = 129;
    capture.ledger_frontier    = 130;
    capture.text_kv_valid      = 129;
    capture.tail_hidden_valid  = false;
    capture.ledger             = retained.token_ids;
    capture.identity           = &identity;
    capture.hash_f             = cache::prefix_hash_at(retained.token_ids, identity, 129);
    capture.text               = &allocation;
    capture.text_pool          = source_pool.storage.get();
    capture.text_semantics     = source_pool.semantics();
    capture.stream             = device.copy_stream;

    ram.test_fail_next_capture_metadata_allocation();
    require(ram.capture(capture).status == cache::RamCaptureStatus::Dropped,
            "optional RAM metadata allocation failure did not drop capture");
    const auto saved = ram.capture(capture);
    require(saved.status == cache::RamCaptureStatus::Captured, "RAM capture failed");
    ram.wait_pending_copies();
    const auto image = ram.host_kv(saved.entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::gather_logical_page_from_host_image(image.text, *source_pool.storage, 3, index,
                                                    page.data());
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "RAM logical page gather changed fixed-codec bytes");
    }
    {
        auto ram_destination = source_pool.storage->reserve(3);
        ram_destination.materialize_pages(3, device.stream);
        device.synchronize_all();
        cache::RamRestoreTarget target;
        target.text           = &ram_destination;
        target.text_pool      = source_pool.storage.get();
        target.text_semantics = source_pool.semantics();
        target.text_dst_pages = 3;
        target.stream         = device.copy_stream;
        ram.claim(saved.entry_id);
        const auto restored = ram.unpack_device(saved.entry_id, target);
        ram.wait_pending_copies();
        require(restored.ledger == retained.token_ids && restored.identity.matches(retained, 129),
                "RAM restore changed represented token/vision identity");
        for (std::uint32_t index = 0; index < 3; ++index) {
            ninfer::pack_paged_kv_logical_page_to_host(ram_destination, *source_pool.storage, index,
                                                       page.data(), device.copy_stream);
            device.synchronize_all();
            require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                    "RAM restore changed fixed-codec bytes");
        }
        ram.release(saved.entry_id);
    }
    {
        cache::KVDiskCache disk(config(directory.path, source_pool, ram, compression));
        disk.note_ram_resident(saved.entry_id, 0);
        require(disk.emergency_spill_ram(saved.entry_id), "SSD spill failed");
        disk.wait_idle_and_fsync();
        require(disk.ram_is_durable(saved.entry_id), "spill did not publish a durable RAM ticket");
    }
    require(ram.evict_one_unpinned(saved.entry_id), "durable RAM entry could not be evicted");

    // Startup uses a different physical page capacity, as enabling Vision can change VRAM.
    // Logical codec identity and packed disk bytes must remain reusable.
    Pool destination_pool(12);
    cache::KVDiskCache reopened(config(directory.path, destination_pool, ram, compression));
    const auto chain = cache::prefix_hash_chain(retained);
    const auto match = reopened.plan_match(retained, chain);
    require(match && match.value().reuse_base == 129, "durable vision prefix missing after reopen");
    auto different_media = retained;
    different_media.vision_items[0].content_digest[0] ^= 1;
    require(!reopened.plan_match(different_media, cache::prefix_hash_chain(different_media)),
            "changed Vision digest incorrectly reused disk state");
    require(reopened.claim(match.value().entry_id, match.value().hash_f,
                           match.value().execution_frontier, match.value().reuse_base,
                           match.value().reuse, match.value().committed_generation),
            "durable generation could not be claimed");
    auto destination = destination_pool.storage->reserve(3);
    destination.materialize_pages(3, device.stream);
    device.synchronize_all();
    cache::DiskRestoreTarget target;
    target.text           = &destination;
    target.text_pool      = destination_pool.storage.get();
    target.text_semantics = destination_pool.semantics();
    target.text_dst_pages = 3;
    target.reuse          = match.value().reuse;
    target.reuse_base     = match.value().reuse_base;
    target.stream         = device.copy_stream;
    const auto ticket = restore_with_submitted_join_check(reopened, match.value().entry_id, target);
    reopened.wait_copies(ticket);
    require(!reopened.restore_failed(), "disk restore failed");
    device.synchronize_all();
    reopened.release_restore_ticket(ticket);
    reopened.release(match.value().entry_id);
    for (std::uint32_t index = 0; index < 3; ++index) {
        ninfer::pack_paged_kv_logical_page_to_host(destination, *destination_pool.storage, index,
                                                   page.data(), device.copy_stream);
        device.synchronize_all();
        require(std::memcmp(page.data(), expected[index].data(), page_bytes) == 0,
                "SSD scatter/restore changed fixed-codec bytes");
    }

    const auto objects = reopened.test_main_page_ids(match.value().entry_id);
    require(!objects.empty(), "restored entry has no persistent pages");
    reopened.test_break_object(objects.front(), cache::DiskObjectKind::Main);
    require(reopened.claim(match.value().entry_id), "corruption probe could not claim entry");
    bool rejected                = false;
    std::uint64_t corrupt_ticket = 0;
    try {
        corrupt_ticket = reopened.restore_device(match.value().entry_id, target);
        reopened.wait_copies(corrupt_ticket);
    } catch (const ninfer::runtime::CacheRestoreFailure&) { rejected = true; }
    reopened.cancel_restore();
    device.synchronize_all();
    if (corrupt_ticket != 0) { reopened.release_restore_ticket(corrupt_ticket); }
    reopened.release(match.value().entry_id);
    require(rejected, "corrupted SSD page was not rejected by restore CRC validation");
    reopened.invalidate_entry(match.value().entry_id);
    require(!reopened.plan_match(retained, chain), "invalidated corrupt prefix remained reusable");
}

// Disk I/O stays off the decode and admission paths.

template <class Predicate>
bool wait_pred(Predicate predicate, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

// The compute-round join covers submitted private scatter work, but must not wait for SSD
// reads that have not produced GPU work yet. An independent stream avoids the full-restore
// join already present on target.stream masking a missing submitted-work dependency.
std::uint64_t restore_with_submitted_join_check(cache::KVDiskCache& disk, std::uint64_t entry_id,
                                                const cache::DiskRestoreTarget& target) {
    struct Probe {
        cache::KVDiskCache& disk;
        hipStream_t stream = nullptr;
        hipEvent_t done    = nullptr;

        explicit Probe(cache::KVDiskCache& owner) : disk(owner) {
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            try {
                HIP_CHECK(hipEventCreateWithFlags(&done, hipEventDisableTiming));
            } catch (...) {
                (void)hipStreamDestroy(stream);
                throw;
            }
        }

        void release_gates() {
            disk.test_release_page_read_barrier();
            disk.test_release_scatter_record_barrier();
        }

        ~Probe() {
            release_gates();
            (void)hipStreamSynchronize(stream);
            (void)hipEventDestroy(done);
            (void)hipStreamDestroy(stream);
        }

        bool ready() const {
            const hipError_t status = hipEventQuery(done);
            if (status == hipErrorNotReady) { return false; }
            HIP_CHECK(status);
            return true;
        }
    } probe(disk);

    std::uint64_t ticket = 0;
    try {
        disk.test_arm_page_read_barrier();
        ticket = disk.restore_device(entry_id, target);
        require(wait_pred([&] { return disk.test_page_read_entered(); }, std::chrono::seconds(2)),
                "submitted join fixture did not enter its held SSD read");
        disk.join_submitted_restore(probe.stream);
        HIP_CHECK(hipEventRecord(probe.done, probe.stream));
        require(wait_pred([&] { return probe.ready(); }, std::chrono::seconds(2)),
                "submitted restore join waited for a future SSD read");

        disk.test_arm_scatter_record_barrier(0);
        disk.test_release_page_read_barrier();
        require(wait_pred(
                    [&] {
                        disk.pump_restore(target.stream);
                        return disk.test_scatter_record_entered();
                    },
                    std::chrono::seconds(2)),
                "submitted join fixture did not enqueue its held scatter completion");
        HIP_CHECK(hipEventRecord(probe.done, probe.stream));
        require(wait_pred([&] { return probe.ready(); }, std::chrono::seconds(2)),
                "independent stream could not progress beside the held scatter");
        disk.join_submitted_restore(probe.stream);
        HIP_CHECK(hipEventRecord(probe.done, probe.stream));
        require(!wait_pred([&] { return probe.ready(); }, std::chrono::milliseconds(100)),
                "submitted restore join omitted the pending private scatter");
        disk.test_release_scatter_record_barrier();
        require(wait_pred([&] { return probe.ready(); }, std::chrono::seconds(2)),
                "submitted restore join did not complete after scatter release");
        return ticket;
    } catch (...) {
        probe.release_gates();
        if (ticket != 0) {
            disk.cancel_restore();
            disk.release_restore_ticket(ticket);
        }
        throw;
    }
}

// Blocks a stream behind a host callback until released, so copies enqueued after it stay pending.
struct StreamGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    bool finished = false;
    bool launched = false;

    static void callback(void* pointer) {
        auto& gate = *static_cast<StreamGate*>(pointer);
        std::unique_lock lock(gate.mutex);
        gate.cv.wait(lock, [&] { return gate.released; });
        gate.finished = true;
        gate.cv.notify_all();
    }

    void launch(hipStream_t stream) {
        HIP_CHECK(hipLaunchHostFunc(stream, callback, this));
        launched = true;
    }

    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }

    ~StreamGate() {
        release();
        if (launched) {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return finished; });
        }
    }
};

q3::PreparedPromptData text_prompt(const std::vector<ninfer::TokenId>& tokens) {
    q3::PreparedPromptData result;
    result.token_ids = tokens;
    result.token_types.assign(tokens.size(), 0);
    result.positions.resize(tokens.size() * 3);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.positions[axis * tokens.size() + i] = static_cast<std::int32_t>(i);
        }
    }
    return result;
}

// A Text-only retained lane whose ledger is `tokens` plus one uncommitted token, so a lookup of
// `retained` reuses the whole `tokens` prefix without requiring tail hidden state.
struct TextCapture {
    q3::PreparedPromptData retained;
    cache::ResidentPrefixIdentity identity;
    cache::RamCaptureSource source;

    TextCapture(const std::vector<ninfer::TokenId>& tokens, ninfer::PagedKVAllocation& allocation,
                Pool& pool, hipStream_t stream) {
        auto ledger = tokens;
        ledger.push_back(0);
        retained = text_prompt(ledger);
        identity.assign(retained);
        const auto frontier       = static_cast<std::uint32_t>(tokens.size());
        source.execution_frontier = frontier;
        source.ledger_frontier    = frontier + 1;
        source.text_kv_valid      = frontier;
        source.ledger             = retained.token_ids;
        source.identity           = &identity;
        source.hash_f             = cache::prefix_hash_at(retained.token_ids, identity, frontier);
        source.text               = &allocation;
        source.text_pool          = pool.storage.get();
        source.text_semantics     = pool.semantics();
        source.stream             = stream;
    }

    TextCapture(const TextCapture&)            = delete;
    TextCapture& operator=(const TextCapture&) = delete;
};

std::uint64_t capture_tokens(cache::KVRamCache& ram, Pool& pool,
                             ninfer::PagedKVAllocation& allocation, ninfer::DeviceContext& device,
                             const std::vector<ninfer::TokenId>& tokens) {
    TextCapture capture(tokens, allocation, pool, device.copy_stream);
    const auto result = ram.capture(capture.source);
    require(result.status == cache::RamCaptureStatus::Captured, "stall fixture RAM capture failed");
    device.synchronize_all();
    ram.wait_pending_copies();
    return result.entry_id;
}

struct PackedObjectLocation {
    std::filesystem::path path;
    std::uint64_t offset = 0;
    std::uint64_t extent = 0;
    std::uint64_t stored = 0;
};

std::optional<PackedObjectLocation> packed_object_location(const std::filesystem::path& root,
                                                           cache::DiskObjectKind kind,
                                                           std::uint64_t id) {
    const auto bytes = [&]() -> std::vector<std::uint8_t> {
        std::ifstream in(root / "PACKSET", std::ios::binary);
        std::vector<std::uint8_t> out((std::istreambuf_iterator<char>(in)), {});
        return out;
    }();
    if (bytes.size() != 32) { return std::nullopt; }
    std::uint64_t generation = 0;
    std::memcpy(&generation, bytes.data() + 12, sizeof(generation));
    auto location_from_payload = [&](const std::array<std::uint8_t, 52>& payload) {
        std::uint32_t segment = 0;
        std::uint64_t offset = 0, extent = 0, stored = 0;
        std::memcpy(&segment, payload.data() + 12, sizeof(segment));
        std::memcpy(&offset, payload.data() + 16, sizeof(offset));
        std::memcpy(&extent, payload.data() + 24, sizeof(extent));
        std::memcpy(&stored, payload.data() + 32, sizeof(stored));
        char name[32]{};
        std::snprintf(name, sizeof(name), "%08u.pack", segment);
        const char* dir = kind == cache::DiskObjectKind::Main      ? "main"
                          : kind == cache::DiskObjectKind::Backend ? "backend"
                          : kind == cache::DiskObjectKind::State   ? "state"
                          : kind == cache::DiskObjectKind::Ledger  ? "ledger"
                                                                   : "identity";
        return PackedObjectLocation{root / "packs" / "1" / dir / name, offset, extent, stored};
    };
    std::optional<PackedObjectLocation> found;
    std::ifstream base(root / "maps" / ("objects-" + std::to_string(generation) + ".base"),
                       std::ios::binary);
    std::array<std::uint8_t, 48> base_header{};
    if (base.read(reinterpret_cast<char*>(base_header.data()), base_header.size())) {
        std::uint64_t count = 0;
        std::memcpy(&count, base_header.data() + 32, sizeof(count));
        for (std::uint64_t n = 0; n < count; ++n) {
            std::array<std::uint8_t, 52> payload{};
            base.read(reinterpret_cast<char*>(payload.data()), payload.size());
            if (!base) { return std::nullopt; }
            std::uint64_t seen = 0;
            std::memcpy(&seen, payload.data(), sizeof(seen));
            if (seen == id && payload[8] == static_cast<std::uint8_t>(kind)) {
                found = location_from_payload(payload);
            }
        }
    }
    std::ifstream in(root / "maps" / ("objects-" + std::to_string(generation) + ".log"),
                     std::ios::binary);
    for (;;) {
        std::uint32_t frame = 0;
        in.read(reinterpret_cast<char*>(&frame), sizeof(frame));
        if (!in) { return found; }
        if (frame != 60) { return std::nullopt; }
        std::array<std::uint8_t, 56> frame_payload{};
        std::uint32_t crc = 0;
        in.read(reinterpret_cast<char*>(frame_payload.data()), frame_payload.size());
        in.read(reinterpret_cast<char*>(&crc), sizeof(crc));
        if (!in) { return found; }
        if ((frame_payload[0] != 1 && frame_payload[0] != 2 && frame_payload[0] != 3) ||
            frame_payload[1] != 0 || frame_payload[2] != 0 || frame_payload[3] != 0) {
            return std::nullopt;
        }
        std::array<std::uint8_t, 52> payload{};
        std::copy(frame_payload.begin() + 4, frame_payload.end(), payload.begin());
        std::uint64_t seen = 0;
        std::memcpy(&seen, payload.data(), sizeof(seen));
        if (seen != id || payload[8] != static_cast<std::uint8_t>(kind)) { continue; }
        if (frame_payload[0] == 2) {
            found.reset();
        } else {
            found = location_from_payload(payload);
        }
    }
}

std::uint64_t pack_file_bytes(const std::filesystem::path& root) {
    std::uint64_t bytes = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root / "packs"))
        if (entry.is_regular_file() && entry.path().extension() == ".pack")
            bytes += entry.file_size();
    return bytes;
}

void fill_pages(Pool& pool, ninfer::PagedKVAllocation& allocation, ninfer::DeviceContext& device,
                std::uint8_t seed) {
    const auto bytes = ninfer::paged_kv_logical_page_bytes(*pool.storage);
    ninfer::PinnedHostBuffer page(bytes);
    auto* data = static_cast<std::uint8_t*>(page.data());
    for (std::size_t i = 0; i < bytes; ++i) data[i] = static_cast<std::uint8_t>(i * 31 + seed);
    for (std::uint32_t i = 0; i < 2; ++i)
        ninfer::unpack_paged_kv_logical_page_from_host(allocation, *pool.storage, page.data(), i,
                                                       device.copy_stream);
    device.synchronize_all();
}

// A lookup that reuses the whole `tokens` prefix of a TextCapture without tail hidden state.
q3::PreparedPromptData lookup_prompt(std::vector<ninfer::TokenId> tokens) {
    tokens.push_back(0);
    return text_prompt(tokens);
}

std::optional<cache::DiskMatch> plan_tokens(cache::KVDiskCache& disk,
                                            const std::vector<ninfer::TokenId>& tokens) {
    const auto prompt = lookup_prompt(tokens);
    return disk.plan_match(prompt, cache::prefix_hash_chain(prompt));
}

std::uint64_t planned_entry(cache::KVDiskCache& disk, const std::vector<ninfer::TokenId>& tokens) {
    const auto match = plan_tokens(disk, tokens);
    return match ? match->entry_id : 0;
}

// One scheduler admission round against a committed disk entry: plan, claim and release.
bool disk_admission_round(cache::KVDiskCache& disk, const std::vector<ninfer::TokenId>& tokens,
                          std::uint64_t expected_entry) {
    const auto match = plan_tokens(disk, tokens);
    if (!match || match->entry_id != expected_entry || !disk.claim(match->entry_id)) {
        return false;
    }
    disk.release(match->entry_id);
    return true;
}

bool ram_resident(const cache::KVRamCache& ram, std::uint64_t id) {
    const auto ids = ram.fifo_ids();
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// Holds one unlocked disk-I/O window open, and releases it from a watchdog if a call under test
// blocks on the held window instead of returning, so a regression fails instead of hanging.
struct UnlockedIoHold {
    cache::KVDiskCache& disk;
    cache::DiskUnlockedIo window;
    std::atomic<bool> done{false};
    std::atomic<bool> fired{false};
    std::jthread watchdog;

    UnlockedIoHold(cache::KVDiskCache& target, cache::DiskUnlockedIo held)
        : disk(target), window(held) {
        disk.test_hold_unlocked_io(window, true);
    }

    UnlockedIoHold(const UnlockedIoHold&)            = delete;
    UnlockedIoHold& operator=(const UnlockedIoHold&) = delete;

    [[nodiscard]] bool entered() const {
        return wait_pred([&] { return disk.test_unlocked_io_entered(window); },
                         std::chrono::seconds(10));
    }

    void arm() {
        if (watchdog.joinable()) { watchdog.join(); }
        done.store(false);
        watchdog = std::jthread([this] {
            if (!wait_pred([&] { return done.load(); }, std::chrono::seconds(10))) {
                fired.store(true);
                disk.test_hold_unlocked_io(window, false);
            }
        });
    }

    // True when every armed call returned while the window was still held.
    bool disarm() {
        done.store(true);
        if (watchdog.joinable()) { watchdog.join(); }
        return !fired.load();
    }

    void release() { disk.test_hold_unlocked_io(window, false); }

    ~UnlockedIoHold() {
        done.store(true);
        release();
    }
};

// Releases a held spill encode on scope exit, and from a watchdog if a call under test blocks on
// the held spill instead of returning, so a regression fails instead of hanging.
struct EncodeHold {
    cache::KVDiskCache& disk;
    std::atomic<bool> done{false};
    std::atomic<bool> fired{false};
    std::jthread watchdog;

    explicit EncodeHold(cache::KVDiskCache& target) : disk(target) {
        disk.test_hold_spill_encode(true);
    }

    EncodeHold(const EncodeHold&)            = delete;
    EncodeHold& operator=(const EncodeHold&) = delete;

    [[nodiscard]] bool entered() const {
        return wait_pred([&] { return disk.test_spill_encode_entered(); },
                         std::chrono::seconds(10));
    }

    void arm() {
        watchdog = std::jthread([this] {
            if (!wait_pred([&] { return done.load(); }, std::chrono::seconds(10))) {
                fired.store(true);
                disk.test_hold_spill_encode(false);
            }
        });
    }

    // True when every armed call returned while the encode was still held.
    bool disarm() {
        done.store(true);
        if (watchdog.joinable()) { watchdog.join(); }
        return !fired.load();
    }

    ~EncodeHold() {
        done.store(true);
        disk.test_hold_spill_encode(false);
    }
};

// An idle prepare that makes capacity releases the mutex for the eviction's MANIFEST write. A
// cancellation in that window must stop the spill before it installs; otherwise
// cancel_idle_spill waits for the whole idle image write.
void idle_cancel_during_capacity_eviction(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    auto options = config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
    const auto first =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 21));
    std::size_t used = 0;
    {
        cache::KVDiskCache disk(options);
        disk.note_ram_resident(first, 0);
        require(disk.emergency_spill_ram(first), "idle-cancel-evict first spill failed");
        disk.wait_idle_and_fsync();
        used = disk.snapshot().used_bytes;
    }
    options.capacity_bytes = used;
    cache::KVDiskCache disk(options);
    const auto second =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 22));
    disk.note_ram_resident(second, 0);
    // The eviction's MANIFEST write is held until the cancellation has registered, so the
    // cancellation always lands inside the window regardless of host scheduling.
    disk.test_hold_manifest_io(true);
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_manifest_io_entered(); }, std::chrono::seconds(10))) {
        disk.test_hold_manifest_io(false);
        disk.cancel_idle_spill();
        require(false, "idle-cancel-evict idle prepare did not reach capacity eviction");
    }
    bool registered = false;
    bool prompt     = false;
    bool durable    = false;
    {
        // Any payload write from here on stays held: a cancel that waits for a spill installed
        // after the cancellation never returns while it is held.
        UnlockedIoHold write(disk, cache::DiskUnlockedIo::PayloadWrite);
        const std::uint64_t epoch = disk.test_idle_cancel_epoch();
        std::thread canceller([&] { disk.cancel_idle_spill(); });
        registered = wait_pred([&] { return disk.test_idle_cancel_epoch() != epoch; },
                               std::chrono::seconds(10));
        write.arm();
        disk.test_hold_manifest_io(false);
        canceller.join();
        prompt  = write.disarm();
        durable = disk.ram_is_durable(second);
    }
    disk.wait_idle_and_fsync();
    require(registered,
            "idle-cancel-evict cancellation did not register during the MANIFEST write");
    require(prompt, "idle cancel waited for a spill installed after cancellation");
    require(!durable, "idle spill committed after cancellation");
}

// Pinning an entry for a disk spill waits for that entry's own device copies, not every pending
// RAM copy: an unrelated gated capture must not stall it.
void spill_pin_waits_only_its_entry(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    hipStream_t side = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
    constexpr std::size_t kDelayBytes = 256ULL << 20;
    ninfer::DeviceBuffer delay_source(kDelayBytes);
    ninfer::PinnedHostBuffer delay_destination(kDelayBytes);
    bool captured_both = false;
    bool prompt        = false;
    bool spilled       = false;
    {
        StreamGate gate;
        gate.launch(side);
        TextCapture unrelated(std::vector<ninfer::TokenId>(64, 23), allocation, pool, side);
        const bool gated =
            ram.capture(unrelated.source).status == cache::RamCaptureStatus::Captured;
        // Keep the spilled entry's own D2H in flight briefly behind a large copy.
        HIP_CHECK(hipMemcpyAsync(delay_destination.data(), delay_source.data(), kDelayBytes,
                                 hipMemcpyDeviceToHost, device.copy_stream));
        TextCapture own(std::vector<ninfer::TokenId>(64, 24), allocation, pool, device.copy_stream);
        const auto captured = ram.capture(own.source);
        captured_both       = gated && captured.status == cache::RamCaptureStatus::Captured;
        if (captured_both) {
            disk.note_ram_resident(captured.entry_id, 0);
            std::atomic<bool> result{false};
            // Reaching the payload write proves the pin passed its own copy; the unrelated
            // capture stays gated until after the check, so a pin that waits on it never does.
            UnlockedIoHold write(disk, cache::DiskUnlockedIo::PayloadWrite);
            std::thread spiller([&] {
                try {
                    result.store(disk.emergency_spill_ram(captured.entry_id));
                    // NOLINTNEXTLINE(bugprone-empty-catch): a throw leaves result false
                } catch (...) {}
            });
            prompt = write.entered();
            write.release();
            gate.release();
            spiller.join();
            spilled = result.load();
        }
        gate.release();
    }
    device.synchronize_all();
    HIP_CHECK(hipStreamSynchronize(side));
    HIP_CHECK(hipStreamDestroy(side));
    disk.wait_idle_and_fsync();
    require(captured_both, "spill-own-copy fixture capture failed");
    require(prompt, "disk spill pin waited for an unrelated gated RAM copy");
    require(spilled, "spill-own-copy spill failed");
}

// Relocation yields to requests, preserves in-flight readers, and journals each move.
void bounded_idle_defrag(ninfer::DeviceContext& device) {
    for (int scenario = 0; scenario < 7; ++scenario) {
        TemporaryDirectory directory;
        Pool pool(8);
        cache::KVRamCache ram(16ULL << 20);
        auto allocation = pool.storage->reserve(3);
        allocation.materialize_pages(2, device.stream);
        const auto options =
            config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
        const std::vector<ninfer::TokenId> retained(128, 103);
        std::uint64_t page = 0;
        std::vector<std::uint8_t> expected;
        {
            cache::KVDiskCache disk(options);
            for (int round = 0; round < 3; ++round) {
                fill_pages(pool, allocation, device, static_cast<std::uint8_t>(51 + round));
                const auto ram_id = capture_tokens(ram, pool, allocation, device,
                                                   std::vector<ninfer::TokenId>(128, 101 + round));
                disk.note_ram_resident(ram_id, 0);
                require(disk.emergency_spill_ram(ram_id), "defrag fixture spill failed");
                require(ram.evict_one_unpinned(ram_id), "defrag RAM retirement failed");
                disk.forget_ram_resident(ram_id);
            }
            disk.wait_idle_and_fsync();
            const auto match = plan_tokens(disk, retained);
            require(match.has_value(), "defrag fixture missing");
            page = disk.test_main_page_ids(match.value().entry_id).at(0);
            const auto before =
                packed_object_location(directory.path, cache::DiskObjectKind::Main, page).value();
            require(before.offset != 0, "defrag fixture lacks movable page");
            expected.resize(before.stored);
            require(disk.test_read_object(cache::DiskObjectKind::Main, page, expected.data(),
                                          expected.size()),
                    "defrag oracle read failed");
            require(disk.test_fifo_evict_one() && disk.test_fifo_evict_one(),
                    "defrag eviction failed");
            const auto storage = pack_file_bytes(directory.path);
            disk.request_idle_spill();
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            require(packed_object_location(directory.path, cache::DiskObjectKind::Main, page)
                            .value()
                            .offset == before.offset,
                    "write-behind relocated without Engine idle permission");
            if (scenario == 0 || scenario == 1) {
                UnlockedIoHold hold(disk, scenario == 0 ? cache::DiskUnlockedIo::DefragCopy
                                                        : cache::DiskUnlockedIo::DefragPublish);
                disk.set_cache_idle(true);
                const bool entered = hold.entered();
                bool admitted      = false;
                if (entered) {
                    hold.arm();
                    admitted = plan_tokens(disk, retained).has_value();
                    admitted = hold.disarm() && admitted;
                    disk.set_cache_idle(false);
                }
                hold.release();
                disk.wait_idle_and_fsync();
                require(entered && admitted, "admission waited for idle relocation");
                require(packed_object_location(directory.path, cache::DiskObjectKind::Main, page)
                                .value()
                                .offset == before.offset,
                        "cancelled move published");
            } else if (scenario == 2) {
                UnlockedIoHold hold(disk, cache::DiskUnlockedIo::PayloadRead);
                std::vector<std::uint8_t> actual(expected.size());
                std::atomic<bool> ok{false};
                std::jthread reader([&] {
                    ok.store(disk.test_read_object(cache::DiskObjectKind::Main, page, actual.data(),
                                                   actual.size()));
                });
                const bool entered = hold.entered();
                disk.set_cache_idle(true);
                const bool moved = wait_pred(
                    [&] {
                        const auto loc = packed_object_location(directory.path,
                                                                cache::DiskObjectKind::Main, page);
                        return loc && loc->offset < before.offset;
                    },
                    std::chrono::seconds(5));
                const bool preserved =
                    std::filesystem::file_size(before.path) >= before.offset + before.extent;
                hold.release();
                reader.join();
                require(entered && moved && preserved && ok.load() && actual == expected,
                        "move invalidated a reader lease");
            } else if (scenario == 3 || scenario == 4) {
                UnlockedIoHold hold(disk, scenario == 3 ? cache::DiskUnlockedIo::DefragPublish
                                                        : cache::DiskUnlockedIo::DefragMapped);
                disk.set_cache_idle(true);
                const bool entered = hold.entered();
                TemporaryDirectory crash;
                if (entered)
                    std::filesystem::copy(directory.path, crash.path,
                                          std::filesystem::copy_options::recursive |
                                              std::filesystem::copy_options::overwrite_existing);
                hold.release();
                require(entered, "durability window not reached");
                auto recovered_options     = options;
                recovered_options.location = crash.path;
                cache::KVDiskCache recovered(recovered_options);
                std::vector<std::uint8_t> actual(expected.size());
                const auto loc =
                    packed_object_location(crash.path, cache::DiskObjectKind::Main, page);
                require(loc && (loc->offset < before.offset) == (scenario == 4) &&
                            recovered.test_read_object(cache::DiskObjectKind::Main, page,
                                                       actual.data(), actual.size()) &&
                            actual == expected,
                        "crash recovery selected invalid move");
            } else {
                const auto drops = disk.snapshot().drops;
                disk.test_arm_fault(scenario == 5 ? cache::DiskFaultPoint::AfterMoveAppend
                                                  : cache::DiskFaultPoint::AfterMoveSync);
                disk.set_cache_idle(true);
                require(wait_pred([&] { return disk.snapshot().drops > drops; },
                                  std::chrono::seconds(5)),
                        "move durability fault not reached");
                disk.set_cache_idle(false);
                disk.wait_idle_and_fsync();
                require(packed_object_location(directory.path, cache::DiskObjectKind::Main, page)
                                .value()
                                .offset == before.offset,
                        "failed move changed durable location");
            }
            disk.set_cache_idle(true);
            require(wait_pred(
                        [&] {
                            const auto loc = packed_object_location(
                                directory.path, cache::DiskObjectKind::Main, page);
                            return loc && loc->offset < before.offset &&
                                   std::filesystem::file_size(loc->path) <
                                       before.offset + before.extent;
                        },
                        std::chrono::seconds(8)),
                    "idle defrag did not move and trim");
            disk.set_cache_idle(false);
            disk.wait_idle_and_fsync();
            require(pack_file_bytes(directory.path) <= storage &&
                        disk.snapshot().storage_bytes == pack_file_bytes(directory.path),
                    "defrag grew or miscounted pack storage");
        }
        cache::KVDiskCache reopened(options);
        std::vector<std::uint8_t> actual(expected.size());
        require(plan_tokens(reopened, retained).has_value() &&
                    reopened.test_read_object(cache::DiskObjectKind::Main, page, actual.data(),
                                              actual.size()) &&
                    actual == expected,
                "defrag changed persisted payload");
    }
}

// Retries a non-blocking reclaim until it evicts. Reclaim has side effects, so it must not be a
// wait_pred predicate, which evaluates once more after succeeding.
bool reclaim_until_evicted(cache::KVDiskCache& disk, cache::RamReclaim& result) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (result != cache::RamReclaim::Evicted && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        result = disk.reclaim_ram_entry(false);
    }
    return result == cache::RamReclaim::Evicted;
}

// Reclaiming RAM for a capture while other lanes decode must not wait on a disk write: a
// disk-durable entry is evicted first, and without one the oldest entry is spilled on the disk
// worker and evicted once durable, never dropped unsaved and never spilled synchronously.
void ram_reclaim_never_blocks_on_disk(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const std::vector<ninfer::TokenId> older_tokens(64, 27);
    const auto older = capture_tokens(ram, pool, allocation, device, older_tokens);
    disk.note_ram_resident(older, 0);
    const auto newer =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 28));
    disk.note_ram_resident(newer, 0);
    require(disk.emergency_spill_ram(newer), "ram-reclaim fixture spill failed");
    disk.wait_idle_and_fsync();
    auto result         = cache::RamReclaim::NoVictim;
    bool durable_first  = false;
    bool unsaved_kept   = false;
    bool prompt         = false;
    std::uint64_t drops = 0;
    {
        // Every disk write from here on stays held: a reclaim that waits on one never returns
        // while it is held.
        UnlockedIoHold write(disk, cache::DiskUnlockedIo::PayloadWrite);
        write.arm();
        result        = disk.reclaim_ram_entry(false);
        durable_first = result == cache::RamReclaim::Evicted && !ram_resident(ram, newer) &&
                        ram_resident(ram, older);
        drops         = disk.snapshot().drops;
        result        = disk.reclaim_ram_entry(false);
        unsaved_kept  = result != cache::RamReclaim::Evicted && ram_resident(ram, older);
        prompt        = write.disarm();
    }
    const bool evicted_once_durable = reclaim_until_evicted(disk, result);
    const bool on_disk              = plan_tokens(disk, older_tokens).has_value();
    require(durable_first, "non-blocking reclaim did not evict the disk-durable entry first");
    require(unsaved_kept, "non-blocking reclaim dropped the unsaved oldest entry");
    require(prompt, "non-blocking reclaim waited on disk I/O or spilled synchronously");
    require(evicted_once_durable && !ram_resident(ram, older),
            "non-blocking reclaim never evicted the spilled entry");
    require(disk.snapshot().drops == drops, "non-blocking reclaim dropped an unsaved RAM entry");
    require(on_disk, "reclaimed RAM entry is not on disk");
}

void ram_reclaim_allocation_failure(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const auto entry =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 47));
    disk.note_ram_resident(entry, 0);
    require(disk.emergency_spill_ram(entry), "allocation fixture spill failed");
    disk.wait_idle_and_fsync();
    ram.wait_pending_copies();
    const auto before = disk.snapshot();
    for (const bool may_block : {false, true}) {
        cache::KVRamCache::test_fail_next_eviction_snapshot_allocation();
        require(disk.reclaim_ram_entry(may_block) == cache::RamReclaim::Failed &&
                    !cache::KVRamCache::test_eviction_snapshot_allocation_pending() &&
                    ram.fifo_ids() == std::vector<std::uint64_t>{entry} &&
                    !disk.ram_reclaim_pending(),
                "reclaim allocation failure escaped or changed its source");
    }
    const auto reason = static_cast<std::size_t>(ninfer::KvDiskDropReason::OptionalAlloc);
    require(disk.snapshot().drop_reasons[reason] == before.drop_reasons[reason] + 2 &&
                disk.reclaim_ram_entry(false) == cache::RamReclaim::Evicted &&
                ram.fifo_ids().empty(),
            "reclaim did not recover after allocation pressure");
}

// Under sustained load write-behind is still spilling the oldest entry when RAM fills. That entry
// is I/O-pinned, so it is not an eviction candidate; reclaim must finish its spill and evict it
// rather than drop a newer entry.
void ram_reclaim_waits_for_inflight_spill(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const auto older =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 37));
    disk.note_ram_resident(older, 0);
    const auto newer =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 38));
    disk.note_ram_resident(newer, 0);
    const auto drops = disk.snapshot().drops;
    auto result      = cache::RamReclaim::NoVictim;
    bool newer_kept  = false;
    bool prompt      = false;
    {
        // The write-behind of the oldest entry stays in flight until the reclaim has returned.
        UnlockedIoHold write(disk, cache::DiskUnlockedIo::PayloadWrite);
        disk.request_idle_spill();
        require(write.entered() && ram.test_io_pins(older) != 0,
                "ram-reclaim-inflight write-behind never started");
        write.arm();
        result     = disk.reclaim_ram_entry(false);
        newer_kept = result != cache::RamReclaim::Evicted && ram_resident(ram, newer);
        prompt     = write.disarm();
    }
    const bool evicted = reclaim_until_evicted(disk, result);
    require(newer_kept, "reclaim dropped a newer entry while the oldest was spilling");
    require(prompt, "non-blocking reclaim waited for the in-flight spill");
    require(evicted && !ram_resident(ram, older) && ram_resident(ram, newer),
            "reclaim did not evict the spilled oldest entry");
    require(disk.snapshot().drops == drops, "reclaim dropped an unsaved RAM entry");
}

// A non-blocking reclaim for an admission's second capture must not target the entry its first
// capture just made: the deferral would roll that entry back, the retry would recapture and
// target it again, and admission would defer until no other lane decodes. With every older entry
// unsavable this generation, the oldest one is dropped unsaved instead; with nothing else left
// the capture is refused rather than deferred.
void ram_reclaim_skips_attempt_captures(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const auto older =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 47));
    disk.note_ram_resident(older, 0);
    disk.test_arm_fail_prepare_spill();
    require(!disk.emergency_spill_ram(older), "ram-reclaim-attempt fixture spill did not fail");
    const auto attempt =
        capture_tokens(ram, pool, allocation, device, std::vector<ninfer::TokenId>(64, 48));
    disk.note_ram_resident(attempt, 0);
    const std::array<std::uint64_t, 1> keep{attempt};
    const auto drops = disk.snapshot().drops;
    const auto unsaved =
        disk.snapshot()
            .drop_reasons[static_cast<std::size_t>(ninfer::KvDiskDropReason::ReclaimUnsaved)];
    bool dropped_older = false;
    bool drop_counted  = false;
    bool refused       = false;
    bool prompt        = false;
    {
        // Every disk write from here on stays held: a reclaim that waits on one never returns
        // while it is held.
        UnlockedIoHold write(disk, cache::DiskUnlockedIo::PayloadWrite);
        write.arm();
        const auto first  = disk.reclaim_ram_entry(false, keep);
        dropped_older     = first == cache::RamReclaim::Evicted && !ram_resident(ram, older) &&
                            ram_resident(ram, attempt) && !disk.ram_reclaim_pending();
        drop_counted      = disk.snapshot().drops == drops + 1 &&
                            disk.snapshot().drop_reasons[static_cast<std::size_t>(
                                ninfer::KvDiskDropReason::ReclaimUnsaved)] == unsaved + 1;
        const auto second = disk.reclaim_ram_entry(false, keep);
        refused           = second == cache::RamReclaim::NoVictim && ram_resident(ram, attempt) &&
                            !disk.ram_reclaim_pending();
        prompt            = write.disarm();
    }
    require(
        dropped_older,
        "reclaim targeted this attempt's capture instead of dropping the unsavable older entry");
    require(drop_counted, "dropping the unsavable older entry was not counted");
    require(refused, "reclaim with only this attempt's capture left did not refuse the capture");
    require(prompt, "reclaim excluding this attempt's capture waited on disk I/O");
}

// restore_device waits for another entry's in-flight window reads. The readiness probe lets
// admission keep decoding instead of taking that wait.
void restore_setup_ready_tracks_window_reads(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(32ULL << 20);
    auto allocation = pool.storage->reserve(2);
    allocation.materialize_pages(1, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 32ULL << 20));

    struct BarrierGuard {
        cache::KVDiskCache& disk;

        ~BarrierGuard() { disk.test_release_page_read_barrier(); }
    } barrier{disk};

    const std::vector<ninfer::TokenId> tokens_a{12, 13, 14, 15};
    const std::vector<ninfer::TokenId> tokens_b{16, 17, 18, 19};
    for (const auto* tokens : {&tokens_a, &tokens_b}) {
        const auto ram_id = capture_tokens(ram, pool, allocation, device, *tokens);
        disk.note_ram_resident(ram_id, 0);
        require(disk.emergency_spill_ram(ram_id), "restore-ready fixture spill failed");
    }
    auto ledger_a = tokens_a;
    ledger_a.push_back(0);
    auto ledger_b = tokens_b;
    ledger_b.push_back(0);
    const auto prompt_a = text_prompt(ledger_a);
    const auto prompt_b = text_prompt(ledger_b);
    const auto match_a  = disk.plan_match(prompt_a, cache::prefix_hash_chain(prompt_a));
    const auto match_b  = disk.plan_match(prompt_b, cache::prefix_hash_chain(prompt_b));
    require(match_a && match_b && disk.claim(match_b.value().entry_id),
            "restore-ready fixture could not claim the prefetched entry");
    const bool idle_ready = disk.restore_setup_ready(match_a.value().entry_id);
    disk.test_arm_page_read_barrier();
    disk.prefetch_window(match_b.value().entry_id, 1, 0);
    if (!wait_pred([&] { return disk.test_page_read_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_page_read_barrier();
        disk.release(match_b.value().entry_id);
        require(false, "restore-ready prefetch did not enter its page read");
    }
    const bool busy_ready = disk.restore_setup_ready(match_a.value().entry_id);
    disk.test_release_page_read_barrier();
    const bool drained_ready =
        wait_pred([&] { return disk.restore_setup_ready(match_a.value().entry_id); },
                  std::chrono::seconds(2));
    disk.release(match_b.value().entry_id);
    require(idle_ready, "restore setup reported busy with no window reads");
    require(!busy_ready, "restore setup reported ready while another entry's read was in flight");
    require(drained_ready, "restore setup stayed busy after the window read finished");
}

// A stale claim (planned and claimed against one committed generation) must never restore a
// newer one. A claim that lands while an idle refresh is outside the mutex after its meta rename
// returns at once with the old generation; the refresh abandons, its meta rollback fails, and the
// new generation is published. The stale claim's host load and device restore are then a
// CacheRestoreFailure cache miss at setup.
void disk_claim_rejects_refreshed_generation(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(32ULL << 20);
    auto pages = pool.storage->reserve(1);
    pages.materialize_pages(1, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const std::vector<ninfer::TokenId> tokens(64, 71);
    // An exact-token lookup reuses the whole prefix only while the tail hidden state is valid.
    const auto prompt    = text_prompt(tokens);
    const auto chain     = cache::prefix_hash_chain(prompt);
    const auto ram_entry = [&](bool valid) {
        TextCapture source(tokens, pages, pool, device.copy_stream);
        source.source.tail_hidden_valid = valid;
        const auto result               = ram.capture(source.source);
        require(result.status == cache::RamCaptureStatus::Captured,
                "generation fixture capture failed");
        device.synchronize_all();
        ram.wait_pending_copies();
        return result.entry_id;
    };
    const auto capture = [&](bool valid, std::uint64_t ticket) {
        const auto id = ram_entry(valid);
        disk.note_ram_resident(id, ticket);
        require(disk.emergency_spill_ram(id), "generation fixture spill failed");
        return ram.load_host(id).disk_entry_id;
    };
    const auto id    = capture(true, 0);
    const auto match = disk.plan_match(prompt, chain);
    require(match && match.value().committed_generation != 0, "generation fixture match missing");
    require(capture(false, id) == id, "generation fixture failed to refresh same entry");
    const bool stale_claimed = disk.claim(
        id, match.value().hash_f, match.value().execution_frontier, match.value().reuse_base,
        match.value().reuse, match.value().committed_generation);
    if (stale_claimed) { disk.release(id); }
    require(!stale_claimed, "claim accepted changed same-hash frontier generation");
    require(!disk.plan_match(prompt, chain), "refreshed invalid tail remained reusable");
    require(capture(true, id) == id, "generation retry refresh failed");
    const auto fresh = disk.plan_match(prompt, chain);
    require(fresh && fresh.value().committed_generation > match.value().committed_generation &&
                disk.claim(id, fresh.value().hash_f, fresh.value().execution_frontier,
                           fresh.value().reuse_base, fresh.value().reuse,
                           fresh.value().committed_generation),
            "fresh generation could not be claimed after stale-plan rejection");
    disk.release(id);

    disk.note_ram_resident(ram_entry(true), id);
    bool reached      = false;
    bool claimed      = false;
    bool prompt_claim = false;
    {
        UnlockedIoHold renamed(disk, cache::DiskUnlockedIo::CommitMetaRenamed);
        disk.test_arm_fail_rollback_meta();
        disk.request_idle_spill();
        reached = renamed.entered();
        if (reached) {
            renamed.arm();
            claimed      = disk.claim(id, fresh.value().hash_f, fresh.value().execution_frontier,
                                      fresh.value().reuse_base, fresh.value().reuse,
                                      fresh.value().committed_generation);
            prompt_claim = renamed.disarm();
        } else {
            renamed.release();
            disk.cancel_idle_spill();
        }
    }
    if (claimed && !prompt_claim) { disk.release(id); }
    require(reached, "claim-during-refresh never reached the meta rename");
    require(prompt_claim, "claim waited for the in-flight refresh commit");
    require(claimed, "claim of the current generation failed during a refresh");
    disk.wait_idle_and_fsync();
    const auto stale_restore_misses = [&](auto&& restore) {
        try {
            restore();
        } catch (const ninfer::runtime::CacheRestoreFailure&) {
            return true;
        } catch (const std::exception& error) {
            std::cerr << "stale-generation restore threw " << error.what() << '\n';
        }
        return false;
    };
    const bool host_missed =
        stale_restore_misses([&] { (void)disk.load_host(id, fresh.value().committed_generation); });
    bool device_missed = false;
    {
        auto destination = pool.storage->reserve(1);
        destination.materialize_pages(1, device.stream);
        device.synchronize_all();
        cache::DiskRestoreTarget target;
        target.text                 = &destination;
        target.text_pool            = pool.storage.get();
        target.text_semantics       = pool.semantics();
        target.text_dst_pages       = 1;
        target.stream               = device.copy_stream;
        target.committed_generation = fresh.value().committed_generation;
        device_missed = stale_restore_misses([&] { (void)disk.restore_device(id, target); });
        if (!device_missed) {
            disk.cancel_restore();
            device.synchronize_all();
        }
    }
    // Planning skips a claimed entry, so the published generation is visible once released.
    disk.release(id);
    require(host_missed, "stale claim's host load did not miss on the published generation");
    require(device_missed, "stale claim's restore did not miss on the published generation");
    const auto after_refresh = disk.plan_match(prompt, chain);
    require(after_refresh &&
                after_refresh.value().committed_generation > fresh.value().committed_generation,
            "failed refresh rollback did not publish the new generation");
    require(disk.claim(id, after_refresh.value().hash_f, after_refresh.value().execution_frontier,
                       after_refresh.value().reuse_base, after_refresh.value().reuse,
                       after_refresh.value().committed_generation),
            "published refresh generation could not be claimed");
    disk.release(id);
}

// A disk-hit claim of an entry whose idle Extend is in flight returns at once, while the Extend's
// page write, its commit's object sync, or its commit after the meta.bin rename is still held:
// the scheduler never waits for spill I/O. The claim makes the Extend abandon (before the rename,
// or by rolling it back); the claimed generation stays in memory and on disk.
void claim_returns_during_in_flight_idle_extend(ninfer::DeviceContext& device) {
    using Window = cache::DiskUnlockedIo;
    for (const Window window :
         {Window::PayloadWrite, Window::CommitSync, Window::CommitMetaRenamed}) {
        TemporaryDirectory directory;
        Pool pool(8);
        cache::KVRamCache ram(64ULL << 20);
        auto allocation = pool.storage->reserve(4);
        allocation.materialize_pages(3, device.stream);
        const auto options =
            config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
        const std::vector<ninfer::TokenId> aligned(64, 4);
        std::uint64_t entry_b    = 0;
        std::uint32_t frontier_b = 0;
        {
            cache::KVDiskCache disk(options);
            const auto ram_b = capture_tokens(ram, pool, allocation, device, aligned);
            disk.note_ram_resident(ram_b, 0);
            require(disk.emergency_spill_ram(ram_b), "claim-idle spill of B failed");
            const auto match_b = plan_tokens(disk, aligned);
            require(match_b.has_value(), "claim-idle match of B failed");
            entry_b                               = match_b.value().entry_id;
            frontier_b                            = disk.test_load_meta(entry_b).execution_frontier;
            std::vector<ninfer::TokenId> extended = aligned;
            extended.push_back(0);
            extended.resize(128, 5);
            const auto ram_d = capture_tokens(ram, pool, allocation, device, extended);
            ram.set_disk_entry_id(ram_d, entry_b);
            disk.note_ram_resident(ram_d, entry_b);
            bool reached             = false;
            bool claimed             = false;
            bool returned_while_held = false;
            {
                UnlockedIoHold hold(disk, window);
                disk.request_idle_spill();
                reached = hold.entered();
                if (reached) {
                    std::atomic<bool> returned{false};
                    std::thread claimer([&] {
                        claimed = disk.claim(entry_b);
                        returned.store(true);
                    });
                    returned_while_held =
                        wait_pred([&] { return returned.load(); }, std::chrono::seconds(10));
                    hold.release();
                    claimer.join();
                }
                hold.release();
            }
            if (claimed && !returned_while_held) { disk.release(entry_b); }
            require(reached, "claim-idle never entered the held Extend I/O");
            require(returned_while_held, window == Window::PayloadWrite
                                             ? "claim waited for the idle Extend's payload write"
                                             : "claim waited for the idle Extend's commit fsync");
            require(claimed, "claim-idle claim of B failed");
            // The abandoned Extend drains (and rolls a renamed meta.bin back) while B is claimed.
            disk.wait_idle_and_fsync();
            disk.prefetch_window(entry_b, 1, 0);
            const bool unchanged = disk.test_load_meta(entry_b).execution_frontier == frontier_b;
            disk.release(entry_b);
            require(unchanged, "in-flight idle extend mutated a claimed generation");
        }
        cache::KVDiskCache reopened(options);
        const auto match = plan_tokens(reopened, aligned);
        require(match && match.value().entry_id == entry_b &&
                    match.value().reuse_base == frontier_b,
                "reopen after a claim-abandoned extend lost the claimed generation");
    }
}

// The measured stall: a RAM hit on an entry whose idle spill is in flight froze decoding for the
// whole spill. A restore only reads the entry: claim and consume return while the spill is held
// mid-encode, the spill keeps reading the I/O-pinned bytes and commits them, and the consumed
// block is freed by the spill's last unpin.
void ram_restore_does_not_wait_for_spill(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Zstd, 64ULL << 20));
    const std::vector<ninfer::TokenId> tokens(64, 57);
    const auto ram_id = capture_tokens(ram, pool, allocation, device, tokens);
    disk.note_ram_resident(ram_id, 0);
    const auto entry_bytes = ram.snapshot().used_bytes;
    bool prompt            = false;
    bool retained_block    = false;
    {
        EncodeHold hold(disk);
        disk.request_idle_spill();
        require(hold.entered() && ram.test_io_pins(ram_id) == 1,
                "restore-during-spill: idle spill never pinned and reached its encode");
        hold.arm();
        ram.claim(ram_id);
        ram.consume(ram_id);
        disk.forget_ram_resident(ram_id);
        prompt              = hold.disarm();
        const auto consumed = ram.snapshot();
        retained_block      = consumed.entry_count == 0 && consumed.used_bytes == entry_bytes &&
                              ram.test_io_pins(ram_id) == 1;
    }
    const bool committed = wait_pred(
        [&] { return ram.snapshot().used_bytes == 0 && plan_tokens(disk, tokens).has_value(); },
        std::chrono::seconds(10));
    require(prompt, "RAM claim/consume waited for the entry's in-flight spill");
    require(retained_block,
            "consumed entry left the index too late or lost its spill-pinned block");
    require(committed, "spill of the consumed entry did not commit and free its block");
    require(!ram.retired_pending(), "consumed block remained retired");
}

// Rolling a capture back never waits for its spill: abandon + discard return while the worker is
// held mid-encode, the encode stops at its cancellation point, nothing commits, and the block is
// freed when the abandoned prepare unpins it.
void abandoned_capture_cancels_spill_without_waiting(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Zstd, 64ULL << 20));
    const std::vector<ninfer::TokenId> tokens(64, 61);
    const auto ram_id = capture_tokens(ram, pool, allocation, device, tokens);
    disk.note_ram_resident(ram_id, 0);
    bool prompt    = false;
    bool unindexed = false;
    bool freed     = false;
    {
        EncodeHold hold(disk);
        disk.request_idle_spill();
        require(hold.entered(), "abandon-spill: idle spill never reached its encode");
        hold.arm();
        disk.abandon_ram_spill(ram_id);
        ram.discard(ram_id);
        prompt    = hold.disarm();
        unindexed = ram.snapshot().entry_count == 0 && !disk.test_has_ram_note(ram_id);
        // Still held: only the cancellation lets the prepare finish and unpin.
        freed = wait_pred([&] { return ram.snapshot().used_bytes == 0; }, std::chrono::seconds(10));
    }
    disk.wait_idle_and_fsync();
    const bool committed = plan_tokens(disk, tokens).has_value();
    require(prompt, "capture rollback waited for its in-flight spill");
    require(unindexed, "rolled-back capture stayed indexed or kept its disk note");
    require(freed, "abandoned prepare did not stop at its encode cancellation point");
    require(!committed, "abandoned capture was committed to disk");
}

// A disk claim during the unlocked encode of an idle Extend sees no installed spill to cancel.
// The prepare re-validates after relocking and abandons the Extend without marking the RAM entry
// failed, so the retry branches from the claimed parent instead of rewriting it.
void claim_during_encode_aborts_idle_extend(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Zstd, 64ULL << 20));
    const std::vector<ninfer::TokenId> aligned(64, 4);
    const auto ram_b = capture_tokens(ram, pool, allocation, device, aligned);
    disk.note_ram_resident(ram_b, 0);
    require(disk.emergency_spill_ram(ram_b), "claim-encode fixture spill of B failed");
    const auto match_b = plan_tokens(disk, aligned);
    require(match_b.has_value(), "claim-encode fixture match of B failed");
    const auto entry_b                    = match_b.value().entry_id;
    const auto generation_b               = disk.test_committed_generation(entry_b);
    const auto frontier_b                 = disk.test_load_meta(entry_b).execution_frontier;
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    const auto ram_d = capture_tokens(ram, pool, allocation, device, extended);
    ram.set_disk_entry_id(ram_d, entry_b);
    disk.note_ram_resident(ram_d, entry_b);
    const auto drops = disk.snapshot().drops;
    bool prompt      = false;
    bool claimed     = false;
    {
        EncodeHold hold(disk);
        disk.request_idle_spill();
        require(hold.entered() && disk.test_disk_io_pins(entry_b) != 0,
                "claim-encode: idle Extend never pinned B and reached its encode");
        hold.arm();
        claimed = disk.claim(entry_b);
        prompt  = hold.disarm();
    }
    require(prompt, "disk claim waited for a preparing spill");
    require(claimed, "claim-encode claim of B failed");
    const bool branched =
        wait_pred([&] { return disk.ram_is_durable(ram_d); }, std::chrono::seconds(10));
    const bool parent_kept = disk.test_committed_generation(entry_b) == generation_b &&
                             disk.test_load_meta(entry_b).execution_frontier == frontier_b;
    const bool not_failed  = disk.snapshot().drops == drops;
    disk.release(entry_b);
    require(branched, "abandoned Extend was not retried as a branch of the claimed parent");
    require(parent_kept, "idle spill rewrote a parent claimed during its encode");
    require(not_failed, "re-validated Extend was counted as a failed spill");
}

// The spill's record CRC32C work runs with the index mutex released: the raw state encode that
// seals each state record, and the page-batch header/CRC build. A scheduler admission round
// completes while either window is held, and the held spill still commits.
void spill_crc_runs_without_index_mutex(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20));
    const std::vector<ninfer::TokenId> resident(64, 81);
    const auto resident_ram = capture_tokens(ram, pool, allocation, device, resident);
    disk.note_ram_resident(resident_ram, 0);
    require(disk.emergency_spill_ram(resident_ram), "spill-crc-unlocked resident spill failed");
    const std::uint64_t resident_entry = planned_entry(disk, resident);
    bool encode_admitted               = false;
    bool encode_committed              = false;
    bool prepare_visible               = false;
    {
        const std::vector<ninfer::TokenId> tokens(64, 82);
        const auto ram_id = capture_tokens(ram, pool, allocation, device, tokens);
        disk.note_ram_resident(ram_id, 0);
        EncodeHold hold(disk);
        disk.request_idle_spill();
        require(hold.entered(), "spill-crc-unlocked raw spill never reached its encode");
        hold.arm();
        std::optional<cache::KvDiskSnapshot> active;
        const bool sampled = wait_pred(
            [&] {
                if (!active) { active = disk.try_snapshot(); }
                return active.has_value();
            },
            std::chrono::seconds(2));
        // Encoding still owns a private session. Preparation is active, but an installed
        // session's age is zero until prepare_spill publishes it.
        prepare_visible     = sampled && active && active->spill_active &&
                              active->spill_elapsed_seconds == 0.0 && active->captures == 1;
        const bool admitted = disk_admission_round(disk, resident, resident_entry);
        encode_admitted     = hold.disarm() && admitted;
        disk.test_hold_spill_encode(false);
        encode_committed =
            wait_pred([&] { return disk.ram_is_durable(ram_id); }, std::chrono::seconds(10));
    }
    bool crc_admitted    = false;
    bool crc_committed   = false;
    bool session_visible = false;
    {
        const std::vector<ninfer::TokenId> tokens(64, 83);
        const auto ram_id = capture_tokens(ram, pool, allocation, device, tokens);
        disk.note_ram_resident(ram_id, 0);
        UnlockedIoHold hold(disk, cache::DiskUnlockedIo::PageBatchCrc);
        disk.request_idle_spill();
        require(hold.entered(), "spill-crc-unlocked spill never reached its page-batch CRC build");
        hold.arm();
        std::optional<cache::KvDiskSnapshot> active;
        const bool sampled = wait_pred(
            [&] {
                if (!active) { active = disk.try_snapshot(); }
                return active.has_value();
            },
            std::chrono::seconds(2));
        session_visible     = sampled && active && active->spill_active &&
                              active->spill_elapsed_seconds > 0.0 && active->captures == 2;
        const bool admitted = disk_admission_round(disk, resident, resident_entry);
        crc_admitted        = hold.disarm() && admitted;
        hold.release();
        crc_committed = wait_pred(
            [&] { return disk.ram_is_durable(ram_id) && planned_entry(disk, tokens) != 0; },
            std::chrono::seconds(10));
    }
    require(prepare_visible, "held spill preparation hid its active snapshot");
    require(session_visible, "held spill session hid its active snapshot or elapsed age");
    require(encode_admitted, "admission waited for a raw spill's state encode and CRC");
    require(encode_committed, "raw spill held at its encode did not commit");
    require(crc_admitted, "admission waited for a spill's page-batch CRC build");
    require(crc_committed, "spill held at its page-batch CRC build did not commit");
    disk.wait_idle_and_fsync();
    const auto idle = disk.snapshot();
    require(!idle.spill_active && idle.pending_jobs == 0 && idle.spill_elapsed_seconds == 0.0 &&
                idle.captures == 3 && idle.save_seconds > 0.0,
            "completed spills retained active work or lost completed lifetime counters");
}

// Capacity eviction is two-phase: the FIFO victim leaves planning under the mutex, its tombstone
// is written and fsynced with the mutex released, and only then does it drop its references.
// Admission of an unrelated entry completes while the tombstone write is held, and the eviction
// is durable across a reopen.
void capacity_eviction_tombstone_runs_without_index_mutex(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    auto options = config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
    const std::vector<ninfer::TokenId> victim(64, 91);
    const std::vector<ninfer::TokenId> survivor(64, 92);
    const std::vector<ninfer::TokenId> incoming(64, 93);
    std::size_t used = 0;
    {
        cache::KVDiskCache disk(options);
        for (const auto* tokens : {&victim, &survivor}) {
            const auto ram_id = capture_tokens(ram, pool, allocation, device, *tokens);
            disk.note_ram_resident(ram_id, 0);
            require(disk.emergency_spill_ram(ram_id),
                    "evict-tombstone-unlocked fixture spill failed");
        }
        disk.wait_idle_and_fsync();
        used = disk.snapshot().used_bytes;
    }
    options.capacity_bytes = used;
    bool admitted          = false;
    bool victim_hidden     = false;
    bool victim_referenced = false;
    bool committed         = false;
    bool victim_dropped    = false;
    {
        cache::KVDiskCache disk(options);
        const std::uint64_t victim_entry   = planned_entry(disk, victim);
        const std::uint64_t survivor_entry = planned_entry(disk, survivor);
        const auto ram_id = capture_tokens(ram, pool, allocation, device, incoming);
        disk.note_ram_resident(ram_id, 0);
        {
            UnlockedIoHold hold(disk, cache::DiskUnlockedIo::EvictionTombstones);
            disk.request_idle_spill();
            require(hold.entered(),
                    "evict-tombstone-unlocked spill never reached capacity eviction");
            hold.arm();
            const bool round  = disk_admission_round(disk, survivor, survivor_entry);
            victim_hidden     = planned_entry(disk, victim) == 0 && !disk.claim(victim_entry);
            admitted          = hold.disarm() && round;
            victim_referenced = disk.test_entry_in_index(victim_entry);
        }
        committed =
            wait_pred([&] { return disk.ram_is_durable(ram_id); }, std::chrono::seconds(10));
        disk.wait_idle_and_fsync();
        victim_dropped = !disk.test_entry_in_index(victim_entry);
    }
    cache::KVDiskCache reopened(options);
    const bool restart_ok = planned_entry(reopened, victim) == 0 &&
                            planned_entry(reopened, survivor) != 0 &&
                            planned_entry(reopened, incoming) != 0;
    require(admitted, "admission waited for a capacity-eviction tombstone write");
    require(victim_hidden, "eviction victim stayed plannable while its tombstone was written");
    require(victim_referenced, "eviction victim dropped its references before a durable tombstone");
    require(committed, "spill did not commit after its capacity eviction");
    require(victim_dropped, "tombstoned eviction victim stayed indexed");
    require(restart_ok, "unlocked capacity eviction exposed the wrong restart generation");
}

// A capacity round whose victims did not all become durably tombstoned: the failed victim keeps
// its references, returns to service without a tombstone, and the next FIFO candidate is evicted
// instead. Optionally the failed victim is quarantined while its tombstone write is in flight; it
// then stays out of planning and the worker evicts it afterwards.
void capacity_eviction_tombstone_failure(ninfer::DeviceContext& device, bool quarantine) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    auto options = config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 64ULL << 20);
    const std::vector<ninfer::TokenId> failing(64, 111);
    const std::vector<ninfer::TokenId> evicted(64, 112);
    const std::vector<ninfer::TokenId> fallback(64, 113);
    // Larger than one resident and at most two: the first round takes two victims.
    const std::vector<ninfer::TokenId> incoming(128, 114);
    std::size_t used = 0;
    {
        cache::KVDiskCache disk(options);
        for (const auto* tokens : {&failing, &evicted, &fallback}) {
            const auto ram_id = capture_tokens(ram, pool, allocation, device, *tokens);
            disk.note_ram_resident(ram_id, 0);
            require(disk.emergency_spill_ram(ram_id), "evict-fail fixture spill failed");
        }
        disk.wait_idle_and_fsync();
        used = disk.snapshot().used_bytes;
    }
    options.capacity_bytes = used;
    bool committed         = false;
    bool failed_victim_ok  = false;
    bool moved_past        = false;
    {
        cache::KVDiskCache disk(options);
        const std::uint64_t failing_entry = planned_entry(disk, failing);
        const auto ram_id                 = capture_tokens(ram, pool, allocation, device, incoming);
        disk.note_ram_resident(ram_id, 0);
        disk.test_arm_fail_tombstone();
        {
            UnlockedIoHold hold(disk, cache::DiskUnlockedIo::EvictionTombstones);
            disk.request_idle_spill();
            require(hold.entered(), "evict-fail spill never reached capacity eviction");
            if (quarantine) { disk.invalidate_entry(failing_entry); }
        }
        committed =
            wait_pred([&] { return disk.ram_is_durable(ram_id); }, std::chrono::seconds(10));
        disk.wait_idle_and_fsync();
        if (quarantine) {
            failed_victim_ok = planned_entry(disk, failing) == 0 && !disk.claim(failing_entry) &&
                               wait_pred([&] { return !disk.test_entry_in_index(failing_entry); },
                                         std::chrono::seconds(10));
        } else {
            failed_victim_ok = planned_entry(disk, failing) == failing_entry;
        }
        moved_past = planned_entry(disk, evicted) == 0 && planned_entry(disk, fallback) == 0;
    }
    cache::KVDiskCache reopened(options);
    const bool restart_ok = (planned_entry(reopened, failing) != 0) == !quarantine &&
                            planned_entry(reopened, evicted) == 0 &&
                            planned_entry(reopened, fallback) == 0 &&
                            planned_entry(reopened, incoming) != 0;
    require(committed, "evict-fail spill did not commit after skipping the failed victim");
    require(failed_victim_ok, quarantine
                                  ? "quarantine during a failed eviction did not keep the entry out"
                                  : "victim without a durable tombstone did not return to service");
    require(moved_past, "capacity eviction did not move past the failed victim");
    require(restart_ok, "failed eviction round exposed the wrong restart generation");
}

// Repeated FIFO churn must reuse holes without exceeding the actual pack budget.
void actual_disk_capacity(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(16ULL << 20);
    auto allocation = pool.storage->reserve(3);
    allocation.materialize_pages(2, device.stream);
    const auto budget  = 4 * ninfer::paged_kv_logical_page_bytes(*pool.storage) + (128ULL << 10);
    const auto options = config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, budget);
    {
        cache::KVDiskCache disk(options);
        for (int round = 0; round < 12; ++round) {
            fill_pages(pool, allocation, device, static_cast<std::uint8_t>(round));
            const auto ram_id = capture_tokens(ram, pool, allocation, device,
                                               std::vector<ninfer::TokenId>(128, 200 + round));
            disk.note_ram_resident(ram_id, 0);
            require(disk.emergency_spill_ram(ram_id), "capacity churn spill failed");
            require(ram.evict_one_unpinned(ram_id), "capacity RAM retirement failed");
            disk.forget_ram_resident(ram_id);
            disk.wait_idle_and_fsync();
            require(pack_file_bytes(directory.path) <= budget &&
                        disk.snapshot().storage_bytes == pack_file_bytes(directory.path),
                    "physical pack storage exceeded capacity");
            require(plan_tokens(disk, std::vector<ninfer::TokenId>(128, 200 + round)).has_value(),
                    "newest capacity entry missing");
        }
        require(disk.snapshot().evictions > 0, "capacity fixture did not evict");
    }
    cache::KVDiskCache reopened(options);
    require(plan_tokens(reopened, std::vector<ninfer::TokenId>(128, 211)).has_value(),
            "capacity churn lost newest entry on restart");
    require(reopened.snapshot().storage_bytes == pack_file_bytes(directory.path) &&
                pack_file_bytes(directory.path) <= budget,
            "reopen miscounted physical storage");
}

// invalidate_entry never throws or waits for a spill that pins the entry: the quarantine is
// immediate, the spill built on it abandons, and the disk worker evicts the entry once its pins
// drop. The quarantine survives a reopen.
void invalidate_during_spill_defers_eviction(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(64ULL << 20);
    auto allocation = pool.storage->reserve(4);
    allocation.materialize_pages(3, device.stream);
    const auto options =
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Zstd, 64ULL << 20);
    const std::vector<ninfer::TokenId> aligned(64, 6);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 7);
    std::uint64_t entry_b  = 0;
    bool prompt            = false;
    bool quarantined       = false;
    bool evicted_after_pin = false;
    bool retried           = false;
    bool not_republished   = false;
    {
        cache::KVDiskCache disk(options);
        const auto ram_b = capture_tokens(ram, pool, allocation, device, aligned);
        disk.note_ram_resident(ram_b, 0);
        require(disk.emergency_spill_ram(ram_b), "invalidate-spill fixture spill of B failed");
        const auto match_b = plan_tokens(disk, aligned);
        require(match_b.has_value(), "invalidate-spill fixture match of B failed");
        entry_b          = match_b.value().entry_id;
        const auto ram_d = capture_tokens(ram, pool, allocation, device, extended);
        ram.set_disk_entry_id(ram_d, entry_b);
        disk.note_ram_resident(ram_d, entry_b);
        {
            EncodeHold hold(disk);
            disk.request_idle_spill();
            require(hold.entered() && disk.test_disk_io_pins(entry_b) != 0,
                    "invalidate-spill: idle Extend never pinned B and reached its encode");
            hold.arm();
            bool threw = false;
            try {
                disk.invalidate_entry(entry_b);
            } catch (const std::exception&) { threw = true; }
            prompt               = hold.disarm() && !threw;
            const auto replanned = plan_tokens(disk, aligned);
            quarantined = !disk.claim(entry_b) && !(replanned && replanned->entry_id == entry_b) &&
                          disk.test_entry_in_index(entry_b);
        }
        evicted_after_pin = wait_pred(
            [&] {
                return !disk.test_entry_in_index(entry_b) && disk.test_pending_invalidations() == 0;
            },
            std::chrono::seconds(10));
        retried = wait_pred([&] { return disk.ram_is_durable(ram_d); }, std::chrono::seconds(10));
        const auto match_d = plan_tokens(disk, extended);
        not_republished    = match_d && match_d->entry_id != entry_b;
        disk.wait_idle_and_fsync();
    }
    cache::KVDiskCache reopened(options);
    const bool stayed_invalid = !reopened.test_entry_in_index(entry_b) && !reopened.claim(entry_b);
    require(prompt, "invalidation of a spill-pinned entry threw or waited");
    require(quarantined, "pinned invalid entry stayed selectable or was evicted under its pin");
    require(evicted_after_pin, "disk worker did not evict the invalid entry after its pin dropped");
    require(retried, "spill of the extension was not retried without its invalid parent");
    require(not_republished, "abandoned Extend republished the invalid entry");
    require(stayed_invalid, "invalidated entry resurrected on reopen");
}

// Rolling back a capture that a spill already pinned retires it at once; its block stays
// allocated until the spill's unpin.
void discard_while_spill_pinned_defers_free(ninfer::DeviceContext& device) {
    TemporaryDirectory directory;
    Pool pool(8);
    cache::KVRamCache ram(16ULL << 20);
    auto allocation = pool.storage->reserve(2);
    allocation.materialize_pages(1, device.stream);
    cache::KVDiskCache disk(
        config(directory.path, pool, ram, ninfer::KvDiskCompress::Off, 32ULL << 20));
    const auto ram_id = capture_tokens(ram, pool, allocation, device, {4, 4, 4, 4});
    disk.note_ram_resident(ram_id, 0);
    const auto entry_bytes = ram.snapshot().used_bytes;
    ram.pin_for_io(ram_id);
    if (ram.evict_one_unpinned(ram_id)) {
        ram.unpin_for_io(ram_id);
        require(false, "reclaim eviction succeeded while an I/O pin was held");
    }
    disk.abandon_ram_spill(ram_id);
    ram.discard(ram_id);
    const bool retired = !disk.test_has_ram_note(ram_id) && ram.snapshot().entry_count == 0 &&
                         ram.snapshot().used_bytes == entry_bytes && ram.test_io_pins(ram_id) == 1;
    ram.unpin_for_io(ram_id);
    const bool freed = ram.snapshot().used_bytes == 0 && !ram.retired_pending();
    require(retired, "discard of a pinned capture did not retire it with its block intact");
    require(freed, "the spill's unpin did not free the discarded block");
}

void run_stall_cases(ninfer::DeviceContext& device) {
    idle_cancel_during_capacity_eviction(device);
    spill_pin_waits_only_its_entry(device);
    bounded_idle_defrag(device);
    ram_reclaim_never_blocks_on_disk(device);
    ram_reclaim_allocation_failure(device);
    ram_reclaim_waits_for_inflight_spill(device);
    ram_reclaim_skips_attempt_captures(device);
    restore_setup_ready_tracks_window_reads(device);
    disk_claim_rejects_refreshed_generation(device);
    claim_returns_during_in_flight_idle_extend(device);
    ram_restore_does_not_wait_for_spill(device);
    abandoned_capture_cancels_spill_without_waiting(device);
    claim_during_encode_aborts_idle_extend(device);
    spill_crc_runs_without_index_mutex(device);
    capacity_eviction_tombstone_runs_without_index_mutex(device);
    capacity_eviction_tombstone_failure(device, false);
    capacity_eviction_tombstone_failure(device, true);
    actual_disk_capacity(device);
    invalidate_during_spill_defers_eviction(device);
    discard_while_spill_pinned_defers_free(device);
}

} // namespace

int main(int argc, char** argv) {
    try {
        verify_crc();
        if (argc == 2 && std::string_view(argv[1]) == "--crc-only") {
            std::cout << "fixed cache CRC32C oracle: PASS\n";
            return 0;
        }
        ninfer::DeviceContext device;
        if (argc == 3 && std::string_view(argv[1]) == "--case" &&
            std::string_view(argv[2]) == "stalls") {
            run_stall_cases(device);
            std::cout << "disk tier stays off decode and admission paths: PASS\n";
            return 0;
        }
        run_roundtrip(device, ninfer::KvDiskCompress::Off);
        run_roundtrip(device, ninfer::KvDiskCompress::Zstd);
        run_stall_cases(device);
        std::cout << "fixed RAM/SSD cache exact bytes, restart, Vision identity, CRC and "
                     "invalidation: PASS\n";
        std::cout << "disk tier stays off decode and admission paths: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}

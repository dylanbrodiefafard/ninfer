#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/fp8_int4_paged_kv_cache.h"
#include "core/linear_attention_state.h"
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"
#include "targets/qwen3/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <new>
#include <cstdlib>
#include <utility>
#include <span>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

thread_local int credit_allocations_before_failure = -1;
thread_local bool credit_allocation_failed         = false;

void* operator new(std::size_t bytes) {
    if (credit_allocations_before_failure >= 0) {
        const bool fail_allocation = credit_allocations_before_failure == 0;
        --credit_allocations_before_failure;
        if (fail_allocation) {
            credit_allocation_failed = true;
            throw std::bad_alloc();
        }
    }
    if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) { return memory; }
    throw std::bad_alloc();
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }

void operator delete(void* memory) noexcept { std::free(memory); }

void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

void operator delete[](void* memory) noexcept { std::free(memory); }

void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

namespace q3 = ninfer::targets::qwen3;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

q3::PreparedPromptData prompt_data(std::int32_t first_token, std::int32_t count = 7) {
    q3::PreparedPromptData prompt;
    for (std::int32_t i = 0; i < count; ++i) { prompt.token_ids.push_back(first_token + i); }
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(prompt.token_ids.size() * 3U);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t token = 0; token < prompt.token_ids.size(); ++token) {
            prompt.positions[axis * prompt.token_ids.size() + token] =
                static_cast<std::int32_t>(token);
        }
    }
    return prompt;
}

void paint_allocation(const ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                      std::uint8_t seed, hipStream_t stream) {
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        for (std::size_t logical = 0; logical < allocation.page_ids().size(); ++logical) {
            const std::int32_t physical = allocation.page_ids()[logical];
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::uint8_t value = static_cast<std::uint8_t>(
                    seed + 13U * plane_index + 3U * logical + static_cast<std::size_t>(head));
                if (pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor) {
                    const std::size_t begin = static_cast<std::size_t>(physical) * plane.nb[3];
                    HIP_CHECK(hipMemsetAsync(static_cast<std::uint8_t*>(plane.data) + begin, value,
                                             static_cast<std::size_t>(plane.nb[3]), stream));
                    break;
                }
                const std::size_t begin = static_cast<std::size_t>(head) * plane.nb[3] +
                                          static_cast<std::size_t>(physical) * plane.nb[2];
                HIP_CHECK(hipMemsetAsync(static_cast<std::uint8_t*>(plane.data) + begin, value,
                                         static_cast<std::size_t>(plane.nb[2]), stream));
            }
        }
    }
}

void verify_allocation(const ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                       std::uint8_t seed, const char* label, std::size_t page_count = 0) {
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<std::uint8_t> host(plane.bytes());
        HIP_CHECK(hipMemcpy(host.data(), plane.data, host.size(), hipMemcpyDeviceToHost));
        for (std::size_t logical = 0;
             logical < (page_count ? page_count : allocation.page_ids().size()); ++logical) {
            const std::int32_t physical = allocation.page_ids()[logical];
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::uint8_t expected = static_cast<std::uint8_t>(
                    seed + 13U * plane_index + 3U * logical + static_cast<std::size_t>(head));
                const std::size_t begin =
                    pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor
                        ? static_cast<std::size_t>(physical) * plane.nb[3]
                        : static_cast<std::size_t>(head) * plane.nb[3] +
                              static_cast<std::size_t>(physical) * plane.nb[2];
                const std::size_t bytes =
                    pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor
                        ? static_cast<std::size_t>(plane.nb[3])
                        : static_cast<std::size_t>(plane.nb[2]);
                for (std::size_t byte = 0; byte < bytes; ++byte) {
                    if (host[begin + byte] != expected) {
                        throw std::runtime_error(std::string(label) +
                                                 " restored logical-page byte mismatch");
                    }
                }
                if (pool.plane_order(plane_index) == ninfer::PagedKVPlaneOrder::PageMajor) {
                    break;
                }
            }
        }
    }
}

bool all_bytes_equal(const ninfer::DeviceBuffer& buffer, std::uint8_t value) {
    std::vector<std::uint8_t> host(buffer.size());
    buffer.copy_to_host(host.data(), host.size());
    return std::all_of(host.begin(), host.end(),
                       [value](std::uint8_t byte) { return byte == value; });
}

// Holds the completion fence after transfer submission, leaving DMA APIs usable during checks.
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

    void arm(q3::detail::KVRamCache& cache) {
        cache.test_before_copy_completion(
            [](void* state, hipStream_t stream) {
                static_cast<StreamGate*>(state)->launch(stream);
            },
            this);
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

// One-page Text pool plus a retained-lane capture source with a tail-hidden image.
struct CopyFixture {
    ninfer::Fp8KInt4VPagedKVSpec spec;
    ninfer::Fp8KInt4VSemanticFingerprint semantics;
    ninfer::Fp8KInt4VPagedKVPoolLayout layout;
    std::unique_ptr<ninfer::DeviceBuffer> backing;
    std::unique_ptr<ninfer::PagedKVPool> pool;

    explicit CopyFixture(const ninfer::Fp8KInt4VPagedKVSpec& text_spec) : spec(text_spec) {
        semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(spec);
        ninfer::LayoutBuilder builder;
        layout  = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        backing = std::make_unique<ninfer::DeviceBuffer>(builder.finish(256));
        pool    = std::make_unique<ninfer::PagedKVPool>(
            ninfer::DeviceSpan{backing->data(), backing->size()}, layout.storage);
    }
};

struct HiddenCapture {
    q3::PreparedPromptData prompt;
    q3::detail::ResidentPrefixIdentity identity;
    q3::detail::RamCaptureSource source;

    HiddenCapture(CopyFixture& fixture, const ninfer::PagedKVAllocation& allocation,
                  std::int32_t first_token, const ninfer::Tensor& hidden, hipStream_t stream,
                  std::uint32_t frontier = 6)
        : prompt(prompt_data(first_token, static_cast<std::int32_t>(frontier + 1U))) {
        identity.assign(prompt);
        source.execution_frontier = frontier;
        source.ledger_frontier    = frontier + 1U;
        source.text_kv_valid      = frontier;
        source.tail_hidden_valid  = true;
        source.ledger             = prompt.token_ids;
        source.identity           = &identity;
        source.hash_f         = q3::detail::prefix_hash_at(prompt.token_ids, identity, frontier);
        source.text           = &allocation;
        source.text_pool      = fixture.pool.get();
        source.text_semantics = fixture.semantics;
        source.tail_hidden    = &hidden;
        source.stream         = stream;
    }

    HiddenCapture(const HiddenCapture&)            = delete;
    HiddenCapture& operator=(const HiddenCapture&) = delete;
};

void system_prompt_lru(ninfer::DeviceContext& device, const ninfer::Fp8KInt4VPagedKVSpec& spec) {
    CopyFixture fixture(spec);
    auto allocation = fixture.pool->reserve(1);
    allocation.materialize_pages(1, device.stream);
    paint_allocation(*fixture.pool, allocation, 0x31, device.stream);
    ninfer::DeviceBuffer hidden(256);
    hidden.fill(0x52);
    ninfer::Tensor tensor(hidden.data(), ninfer::DType::U8, {256});
    device.synchronize_all();
    HiddenCapture a(fixture, allocation, 1001, tensor, device.copy_stream);
    HiddenCapture b(fixture, allocation, 2001, tensor, device.copy_stream);
    HiddenCapture c(fixture, allocation, 3001, tensor, device.copy_stream);
    a.source.system_prompt = b.source.system_prompt = c.source.system_prompt = true;
    // A true system boundary has no sampling hidden; its user suffix will produce one.
    a.source.tail_hidden       = nullptr;
    a.source.tail_hidden_valid = false;
    q3::detail::KVRamCache disabled(8ULL << 20);
    require(disabled.capture(a.source).status == q3::detail::RamCaptureStatus::Dropped,
            "system cache is not opt-in");
    q3::detail::KVRamCache cache(8ULL << 20, 2);
    const auto first  = cache.capture(a.source);
    const auto second = cache.capture(b.source);
    require(first.status == q3::detail::RamCaptureStatus::Captured &&
                second.status == q3::detail::RamCaptureStatus::Captured,
            "system prefix capture failed");
    cache.wait_entry_copies_on_stream(first.entry_id, device.stream);
    cache.wait_entry_copies_on_stream(second.entry_id, device.stream);
    paint_allocation(*fixture.pool, allocation, 0x71, device.stream);
    HIP_CHECK(hipMemsetAsync(hidden.data(), 0x72, hidden.size(), device.stream));
    device.synchronize_all();
    auto different_user                   = prompt_data(1001, 9);
    different_user.token_ids[6]           = 900001;
    different_user.token_ids[7]           = 900002;
    different_user.system_prompt_frontier = 6;
    auto match = cache.plan_match(different_user, q3::detail::prefix_hash_chain(different_user));
    require(match && match->entry_id == first.entry_id && match->reuse_base == 6,
            "system match required the first user token to agree");
    auto destination = fixture.pool->reserve(1);
    destination.materialize_pages(1, device.stream);
    q3::detail::RamRestoreTarget target;
    target.text           = &destination;
    target.text_pool      = fixture.pool.get();
    target.text_semantics = fixture.semantics;
    target.text_dst_pages = 1;
    target.tail_hidden    = &tensor;
    target.stream         = device.copy_stream;
    cache.claim(first.entry_id);
    const auto restored = cache.unpack_device(first.entry_id, target);
    require(!restored.tail_hidden_valid, "system prefix invented a sampling hidden");
    device.synchronize_all();
    cache.consume(first.entry_id);
    verify_allocation(*fixture.pool, destination, 0x31, "system prefix restore", 1);
    require(all_bytes_equal(hidden, 0x72), "system restore wrote an absent sampling hidden");
    require(cache.snapshot().system_prompt_entries == 2 &&
                cache.snapshot().system_prompt_restores == 1,
            "system hit consumed the retained source");
    const auto third = cache.capture(c.source);
    device.synchronize_all();
    require(third.status == q3::detail::RamCaptureStatus::Captured &&
                cache.plan_match(a.prompt, q3::detail::prefix_hash_chain(a.prompt)) &&
                !cache.plan_match(b.prompt, q3::detail::prefix_hash_chain(b.prompt)) &&
                cache.snapshot().system_prompt_entries == 2 &&
                cache.snapshot().system_prompt_evictions == 1 &&
                cache.snapshot().used_bytes <= cache.snapshot().capacity_bytes,
            "system quota did not evict the least recently restored prefix");
    require(cache.fifo_ids().empty() && cache.unpinned_ids().empty(),
            "system anchors entered ordinary FIFO/disk eviction");
    q3::detail::KVRamCache pinned(8ULL << 20, 1);
    const auto held = pinned.capture(a.source);
    pinned.claim(held.entry_id);
    require(pinned.capture(b.source).status == q3::detail::RamCaptureStatus::Dropped &&
                pinned.is_claimed(held.entry_id),
            "LRU evicted an in-flight restore source");
    pinned.release(held.entry_id);
}

// All three cache planes may use different physical page orders. Forked pages must be
// independent even when the logical page list spans several physical runs.
void source_preserving_reads(ninfer::DeviceContext& device, ninfer::Fp8KInt4VPagedKVSpec spec) {
    spec.page_group_count = 8;
    CopyFixture fixture(spec);
    auto source    = fixture.pool->reserve(2);
    auto separator = fixture.pool->reserve(1);
    source.materialize_pages(1, device.stream);
    separator.materialize_pages(1, device.stream);
    source.materialize_pages(2, device.stream);
    auto destination = fixture.pool->reserve(2);
    destination.materialize_pages(2, device.stream);
    paint_allocation(*fixture.pool, source, 0x21, device.stream);
    ninfer::copy_paged_kv_allocation(source, destination, *fixture.pool, 2, device.stream);
    device.synchronize_all();
    verify_allocation(*fixture.pool, destination, 0x21, "fork");
    paint_allocation(*fixture.pool, destination, 0x71, device.stream);
    device.synchronize_all();
    verify_allocation(*fixture.pool, source, 0x21, "unchanged fork source");

    q3::detail::KVRamCache cache(8ULL << 20);
    ninfer::DeviceBuffer hidden(256);
    hidden.fill(0x4c);
    ninfer::Tensor hidden_tensor(hidden.data(), ninfer::DType::U8, {256});
    HiddenCapture capture(fixture, source, 971, hidden_tensor, device.copy_stream);
    const auto captured = cache.capture(capture.source);
    require(captured.status == q3::detail::RamCaptureStatus::Captured,
            "source-preserving read capture failed");
    device.synchronize_all();
    q3::detail::RamRestoreTarget target;
    target.text           = &destination;
    target.text_pool      = fixture.pool.get();
    target.text_semantics = fixture.semantics;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_tensor;
    target.stream         = device.copy_stream;
    cache.claim(captured.entry_id);
    (void)cache.unpack_device(captured.entry_id, target);
    device.synchronize_all();
    cache.release_restored(captured.entry_id);
    require(cache.snapshot().entry_count == 1 && cache.snapshot().restores == 1 &&
                cache.plan_match(capture.prompt, q3::detail::prefix_hash_chain(capture.prompt)),
            "fork restore consumed its source or missed restore accounting");
    paint_allocation(*fixture.pool, destination, 0x91, device.stream);
    device.synchronize_all();
    cache.claim(captured.entry_id);
    (void)cache.unpack_device(captured.entry_id, target);
    device.synchronize_all();
    verify_allocation(*fixture.pool, destination, 0x21, "RAM source preserved", 1);
    require(all_bytes_equal(hidden, 0x4c), "RAM source hidden image changed");
    require(cache.snapshot().entry_count == 1, "second read lost source before consume");
    cache.consume(captured.entry_id);
    require(cache.snapshot().entry_count == 0 && cache.snapshot().restores == 2,
            "ordinary read did not consume after the fork");
}

void live_credit_allocation_failure() {
    constexpr std::size_t capacity = 1ULL << 20;
    for (const bool relocate : {false, true}) {
        bool completed         = false;
        bool exercised_failure = false;
        for (int stage = 0; stage < 16 && !completed; ++stage) {
            q3::detail::KVRamCache cache(capacity);
            std::pair<void*, std::size_t> original;
            std::pair<void*, std::size_t> blocker;
            if (relocate) {
                if (!cache.reserve_live_credit(1, 256) || !cache.reserve_live_credit(2, 256)) {
                    throw std::runtime_error(
                        "credit allocation fixture could not reserve its credits");
                }
                original = cache.live_credit_view(1);
                blocker  = cache.live_credit_view(2);
                std::memset(original.first, 0x5a, original.second);
                std::memset(blocker.first, 0xa5, blocker.second);
            }
            const auto before                 = cache.snapshot().used_bytes;
            credit_allocation_failed          = false;
            credit_allocations_before_failure = stage;
            bool escaped                      = false;
            bool reserved                     = false;
            try {
                reserved = cache.reserve_live_credit(1, 512);
            } catch (const std::bad_alloc&) { escaped = true; }
            credit_allocations_before_failure = -1;
            if (credit_allocation_failed) {
                exercised_failure = true;
                if (escaped || reserved || cache.snapshot().used_bytes != before) {
                    throw std::runtime_error(
                        "credit metadata pressure escaped or changed secured capacity");
                }
                if (relocate) {
                    const auto current   = cache.live_credit_view(1);
                    const auto peer      = cache.live_credit_view(2);
                    const auto unchanged = [](const auto& view, unsigned char value) {
                        const auto bytes =
                            std::span(static_cast<const unsigned char*>(view.first), view.second);
                        return std::all_of(bytes.begin(), bytes.end(),
                                           [value](auto byte) { return byte == value; });
                    };
                    if (current != original || peer != blocker || !unchanged(current, 0x5a) ||
                        !unchanged(peer, 0xa5)) {
                        throw std::runtime_error(
                            "failed credit relocation changed existing ownership");
                    }
                }
                if (!cache.reserve_live_credit(1, 512)) {
                    throw std::runtime_error(
                        "credit funding did not recover after metadata pressure");
                }
            } else {
                completed = reserved && !escaped;
            }
            cache.release_live_credit(1);
            if (relocate) { cache.release_live_credit(2); }
            // Snapshot counters cannot expose a provisional arena block leaked before credit
            // publication. Reusing the complete arena checks the allocation ownership itself.
            if (!cache.reserve_live_credit(3, capacity)) {
                throw std::runtime_error("failed credit publication leaked pinned arena capacity");
            }
            cache.release_live_credit(3);
        }
        if (!completed || !exercised_failure) {
            throw std::runtime_error(
                "credit allocation regression did not exercise failure and success");
        }
    }
}

void live_credit_image_ownership(ninfer::DeviceContext& device,
                                 const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    auto source = fixture.pool->reserve(1);
    source.materialize_pages(1, device.stream);
    paint_allocation(*fixture.pool, source, 0x38, device.stream);
    ninfer::DeviceBuffer hidden_storage(256);
    hidden_storage.fill(0x61);
    ninfer::Tensor hidden(hidden_storage.data(), ninfer::DType::U8, {256});
    q3::detail::KVRamCache cache(8ULL << 20);
    constexpr std::uint64_t owner       = 42;
    constexpr std::size_t initial_bytes = 1ULL << 20;
    require(cache.reserve_live_credit(owner, initial_bytes), "live credit could not be secured");
    const auto [original_block, original_bytes] = cache.live_credit_view(owner);
    constexpr std::size_t tail_bytes            = 32;
    const auto tail_offset                      = original_bytes - tail_bytes;
    std::memset(static_cast<std::uint8_t*>(original_block) + tail_offset, 0xab, tail_bytes);
    device.order_copy_after_compute();
    HiddenCapture capture(fixture, source, 1231, hidden, device.copy_stream);
    capture.source.live_owner               = owner;
    capture.source.live_reserved_tail_bytes = tail_bytes;
    const auto image                        = cache.capture(capture.source);
    require(image.status == q3::detail::RamCaptureStatus::Captured, "secured live capture dropped");
    cache.wait_live_image(image.entry_id);
    require(cache.live_image_ready(image.entry_id), "completed live fence was not ready");
    require(cache.fifo_ids().empty() && cache.unpinned_ids().empty() &&
                !cache.plan_match(capture.prompt, q3::detail::prefix_hash_chain(capture.prompt)),
            "live suspension entered optional prefix lookup/eviction");
    require(cache.reserve_live_credit(owner, 2ULL << 20), "live image credit could not grow");
    const auto [grown_block, grown_bytes] = cache.live_credit_view(owner);
    require(
        grown_bytes >= (2ULL << 20) &&
            std::all_of(static_cast<const std::uint8_t*>(grown_block) + tail_offset,
                        static_cast<const std::uint8_t*>(grown_block) + tail_offset + tail_bytes,
                        [](auto value) { return value == 0xab; }),
        "credit growth changed parked sampler-tail bytes");
    auto destination = fixture.pool->reserve(1);
    destination.materialize_pages(1, device.copy_stream);
    q3::detail::RamRestoreTarget target;
    target.text           = &destination;
    target.text_pool      = fixture.pool.get();
    target.text_semantics = fixture.semantics;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden;
    target.stream         = device.copy_stream;
    (void)cache.unpack_device(image.entry_id, target);
    cache.wait_live_image(image.entry_id);
    verify_allocation(*fixture.pool, destination, 0x38, "live image retained across credit growth",
                      1);
    require(all_bytes_equal(hidden_storage, 0x61), "live hidden image changed");
    bool refused = false;
    try {
        cache.release_live_credit(owner);
    } catch (const std::logic_error&) { refused = true; }
    require(refused, "live image lost its backing credit");
    cache.retire_live_image(image.entry_id);
    require(cache.reserve_live_credit(owner, 512ULL << 10), "resident credit shrink failed");
    require(cache.snapshot().used_bytes == (512ULL << 10), "resident credit retained excess RAM");
    cache.release_live_credit(owner);
    require(cache.snapshot().used_bytes == 0, "released live credit leaked RAM");
}

// Two restores in one round each own an in-flight H2D. Consuming the first must account its copy
// as a load, not orphan it as save time.
void two_pending_loads_fold_as_loads(ninfer::DeviceContext& device,
                                     const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    constexpr std::size_t kBulkBytes = 8ULL << 20;
    ninfer::DeviceBuffer bulk(kBulkBytes);
    bulk.fill(0x3c);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {static_cast<std::int32_t>(kBulkBytes)});
    q3::detail::KVRamCache cache(64ULL << 20);
    std::vector<std::uint64_t> ids;
    for (const std::int32_t first : {701, 801}) {
        auto source = fixture.pool->reserve(1);
        source.materialize_pages(1, device.stream);
        device.order_copy_after_compute();
        HiddenCapture capture(fixture, source, first, hidden, device.copy_stream);
        const auto result = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured, "two-load capture failed");
        ids.push_back(result.entry_id);
    }
    const double saved = cache.harvest_copy_seconds().save;
    auto dest_a        = fixture.pool->reserve(1);
    auto dest_b        = fixture.pool->reserve(1);
    dest_a.materialize_pages(1, device.stream);
    dest_b.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    ninfer::DeviceBuffer out_a(kBulkBytes);
    ninfer::DeviceBuffer out_b(kBulkBytes);
    ninfer::Tensor out_a_t(out_a.data(), ninfer::DType::U8,
                           {static_cast<std::int32_t>(kBulkBytes)});
    ninfer::Tensor out_b_t(out_b.data(), ninfer::DType::U8,
                           {static_cast<std::int32_t>(kBulkBytes)});
    const auto restore = [&](std::uint64_t id, ninfer::PagedKVAllocation& dest,
                             ninfer::Tensor& out) {
        q3::detail::RamRestoreTarget target;
        target.text           = &dest;
        target.text_pool      = fixture.pool.get();
        target.text_semantics = fixture.semantics;
        target.text_dst_pages = 1;
        target.tail_hidden    = &out;
        target.stream         = device.copy_stream;
        cache.claim(id);
        (void)cache.unpack_device(id, target);
    };
    restore(ids[0], dest_a, out_a_t);
    restore(ids[1], dest_b, out_b_t);
    cache.consume(ids[0]);
    cache.consume(ids[1]);
    // Consume retires both entries; their unharvested H2D counts toward the lifetime load once
    // it lands, and the lifetime save stays the captures' alone.
    device.synchronize_all();
    (void)cache.harvest_copy_seconds();
    const auto totals = cache.snapshot();
    dest_a.release();
    dest_b.release();
    if (totals.save_seconds != saved || totals.load_seconds <= 0.0) {
        std::cerr << "two pending loads lifetime save=" << totals.save_seconds << " (captures "
                  << saved << ") load=" << totals.load_seconds << '\n';
    }
    require(totals.save_seconds == saved && totals.load_seconds > 0.0,
            "a second restore orphaned the first restore's load as save time");
}

// A stream-ordered wait must order the consumer stream without blocking the host.
void stream_wait_does_not_block_host(ninfer::DeviceContext& device,
                                     const ninfer::Fp8KInt4VPagedKVSpec& text_spec) {
    CopyFixture fixture(text_spec);
    q3::detail::KVRamCache cache(64ULL << 20);
    auto source = fixture.pool->reserve(1);
    source.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    ninfer::DeviceBuffer bulk(1ULL << 20);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {1 << 20});
    bool captured = false;
    bool prompt   = false;
    std::atomic<bool> wait_threw{false};
    {
        StreamGate gate;
        gate.arm(cache);
        HiddenCapture capture(fixture, source, 901, hidden, device.copy_stream);
        captured = cache.capture(capture.source).status == q3::detail::RamCaptureStatus::Captured;
        if (captured) {
            std::atomic<bool> returned{false};
            std::thread waiter([&] {
                try {
                    cache.wait_pending_copies_on_stream(device.stream);
                } catch (...) { wait_threw.store(true); }
                returned.store(true);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (!returned.load() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            prompt = returned.load();
            gate.release();
            waiter.join();
        }
    }
    device.synchronize_all();
    source.release();
    require(captured, "stream-wait capture failed");
    require(prompt, "stream-ordered RAM wait blocked the host on a gated copy");
    require(!wait_threw.load(), "stream-ordered RAM wait threw");
}

// A restore that fails validation after enqueueing H2D reads must still fence the record;
// otherwise its host block looks idle while a copy reads it.
void failed_unpack_fences_enqueued_reads(ninfer::DeviceContext& device,
                                         const ninfer::Fp8KInt4VPagedKVSpec& text_spec,
                                         std::uint32_t pages) {
    CopyFixture fixture(text_spec);
    q3::detail::KVRamCache cache(64ULL << 20);
    ninfer::DeviceBuffer bulk(1ULL << 20);
    ninfer::Tensor hidden(bulk.data(), ninfer::DType::U8, {1 << 20});
    std::uint64_t entry_id = 0;
    {
        auto source = fixture.pool->reserve(pages);
        source.materialize_pages(pages, device.stream);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        HiddenCapture capture(fixture, source, 1001, hidden, device.copy_stream, pages * 64U - 1U);
        const auto result = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured,
                "failed-unpack capture failed");
        entry_id = result.entry_id;
    }
    auto dest = fixture.pool->reserve(pages);
    dest.materialize_pages(pages, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    q3::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = fixture.pool.get();
    target.text_semantics = fixture.semantics;
    target.text_dst_pages = pages;
    target.stream         = device.copy_stream;
    // A context-checkpoint restore naming a frontier with no stored head fails after the text
    // pages are already enqueued.
    target.reuse      = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    target.reuse_base = 3;
    cache.claim(entry_id);
    bool threw  = false;
    bool fenced = false;
    {
        StreamGate gate;
        gate.arm(cache);
        try {
            (void)cache.unpack_device(entry_id, target);
        } catch (const std::logic_error&) { threw = true; }
        fenced = threw && !cache.copies_ready(entry_id);
        gate.release();
    }
    device.synchronize_all();
    cache.release(entry_id);
    dest.release();
    require(threw, "failed-unpack fixture did not reject the missing head");
    require(fenced, "a failed restore left its enqueued H2D reads unfenced");
}

// Rewrite checkpoints live in lane-owned pinned host images. The RAM tier copies those images
// host-to-host on capture and restore, beside the device GDN/DFlash current state. Exact bytes.
enum class StateRestore : std::uint8_t { Frontier, Turn, Response, Drop };

void rewrite_images_round_trip(ninfer::DeviceContext& device,
                               const ninfer::Fp8KInt4VPagedKVSpec& text_spec, StateRestore mode,
                               bool from_disk) {
    CopyFixture fixture(text_spec);
    ninfer::LayoutBuilder state_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        state_builder, ninfer::LinearAttentionStatePoolSpec{.layers         = 3,
                                                            .conv_channels  = 40,
                                                            .conv_width     = 3,
                                                            .value_heads    = 2,
                                                            .value_head_dim = 8,
                                                            .key_head_dim   = 8,
                                                            .slot_count     = 2});
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(state_builder, 2, 128, 2, 16, 2);
    ninfer::DeviceBuffer state_backing(state_builder.finish(256));
    const ninfer::DeviceSpan state_span{state_backing.data(), state_backing.size()};
    ninfer::LinearAttentionStatePool gdn(state_span, gdn_layout);
    ninfer::CyclicKVCache cyclic(state_span, cyclic_layout);
    constexpr std::int32_t lane = 1;

    const auto pattern = [](std::size_t bytes, std::uint32_t seed) {
        std::vector<std::uint8_t> out(bytes);
        for (std::uint8_t& byte : out) {
            seed = seed * 1664525U + 1013904223U;
            byte = static_cast<std::uint8_t>(seed >> 24);
        }
        return out;
    };
    const std::size_t conv_bytes   = gdn.conv_host_image_bytes();
    const std::size_t rec_bytes    = gdn.recurrent_host_image_bytes();
    const std::size_t cyclic_bytes = cyclic.lane_host_bytes();
    const auto current_conv        = pattern(conv_bytes, 11);
    const auto current_rec         = pattern(rec_bytes, 12);
    const auto current_cyclic      = pattern(cyclic_bytes, 13);
    const auto rewrite_conv        = pattern(conv_bytes, 21);
    const auto rewrite_rec         = pattern(rec_bytes, 22);
    const auto rewrite_cyclic      = pattern(cyclic_bytes, 23);
    gdn.unpack_slot_from_host(lane, current_conv.data(), current_rec.data(), device.stream);
    cyclic.copy_lane_from_host(current_cyclic.data(), lane, device.stream);
    ninfer::DeviceBuffer hidden_bytes(2 * 1024);
    hidden_bytes.fill(0x5a);
    ninfer::Tensor tail_hidden(hidden_bytes.data(), ninfer::DType::U8, {1024});
    ninfer::Tensor rewrite_hidden(static_cast<std::uint8_t*>(hidden_bytes.data()) + 1024,
                                  ninfer::DType::U8, {1024});

    q3::detail::KVRamCache cache(64ULL << 20);
    std::uint64_t entry_id = 0;
    {
        auto source = fixture.pool->reserve(1);
        source.materialize_pages(1, device.stream);
        device.order_copy_after_compute();
        HiddenCapture capture(fixture, source, 1101, tail_hidden, device.copy_stream);
        capture.source.gdn                       = &gdn;
        capture.source.gdn_current_slot          = lane;
        capture.source.dflash_local              = &cyclic;
        capture.source.dflash_lane               = lane;
        capture.source.rewrite_valid             = true;
        capture.source.rewrite_frontier          = 4;
        capture.source.rewrite_kind              = mode == StateRestore::Response
                                                       ? q3::RewriteCheckpointKind::ResponseReplay
                                                       : q3::RewriteCheckpointKind::TurnClosure;
        capture.source.rewrite_checkpoint_hidden = &rewrite_hidden;
        capture.source.rewrite_state             = {rewrite_conv.data(), rewrite_rec.data(),
                                                    rewrite_cyclic.data()};
        const auto result                        = cache.capture(capture.source);
        device.synchronize_all();
        source.release();
        require(result.status == q3::detail::RamCaptureStatus::Captured,
                "rewrite-image capture failed");
        entry_id = result.entry_id;
    }

    const bool read_frontier = mode == StateRestore::Frontier || mode == StateRestore::Drop;
    const bool read_rewrite  = mode != StateRestore::Drop;
    gdn.zero_slot(lane, device.stream);
    hidden_bytes.fill(0xcc);
    std::vector<std::uint8_t> zero_cyclic(cyclic_bytes, 0);
    cyclic.copy_lane_from_host(zero_cyclic.data(), lane, device.stream);
    auto dest = fixture.pool->reserve(1);
    dest.materialize_pages(1, device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    std::vector<std::uint8_t> out_conv(conv_bytes, 0);
    std::vector<std::uint8_t> out_rec(rec_bytes, 0);
    std::vector<std::uint8_t> out_cyclic(cyclic_bytes, 0);
    q3::detail::RamRestoreTarget target;
    target.text                      = &dest;
    target.text_pool                 = fixture.pool.get();
    target.text_semantics            = fixture.semantics;
    target.text_dst_pages            = 1;
    target.gdn                       = &gdn;
    target.gdn_current_slot          = lane;
    target.rewrite_state             = {out_conv.data(), out_rec.data(), out_cyclic.data()};
    target.tail_hidden               = &tail_hidden;
    target.rewrite_checkpoint_hidden = &rewrite_hidden;
    target.dflash_local              = &cyclic;
    target.dflash_lane               = lane;
    target.stream                    = device.copy_stream;
    target.keep_rewrite_checkpoint   = read_rewrite;
    target.reuse      = mode == StateRestore::Turn ? ninfer::PrefixReusePath::RestoreTurnCheckpoint
                        : mode == StateRestore::Response
                            ? ninfer::PrefixReusePath::RestoreResponseCheckpoint
                            : ninfer::PrefixReusePath::AppendAtFrontier;
    target.reuse_base = read_frontier ? 6 : 4;
    if (from_disk) {
        std::string temporary =
            (std::filesystem::temp_directory_path() / "ninfer-state-restore-XXXXXX").string();
        require(::mkdtemp(temporary.data()) != nullptr, "disk state scratch creation failed");

        struct Scratch {
            std::filesystem::path path;

            ~Scratch() {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        } scratch{temporary};

        q3::detail::DiskOpenConfig config;
        config.location           = scratch.path;
        config.capacity_bytes     = 64ULL << 20;
        config.max_context        = 64;
        config.ram                = &cache;
        config.text_pool          = fixture.pool.get();
        config.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(*fixture.pool);
        config.hidden_bytes       = 1024;
        config.fingerprint        = q3::detail::make_disk_fingerprint(
            "qwen3.8-27b", "r9700-lq4", "state-restore-fixture", ninfer::SpeculativeBackend::DFlash,
            *fixture.pool, nullptr, fixture.semantics, std::nullopt, &gdn, &cyclic);
        q3::detail::KVDiskCache disk(config);
        disk.note_ram_resident(entry_id, 0);
        require(disk.emergency_spill_ram(entry_id), "state-image disk spill failed");
        disk.wait_idle_and_fsync();
        const auto id = cache.load_host(entry_id).disk_entry_id;
        require(id != 0 && disk.claim(id), "state-image disk claim failed");
        q3::detail::DiskRestoreTarget disk_target;
        disk_target.text                      = target.text;
        disk_target.text_pool                 = target.text_pool;
        disk_target.text_semantics            = target.text_semantics;
        disk_target.text_dst_pages            = target.text_dst_pages;
        disk_target.gdn                       = target.gdn;
        disk_target.gdn_current_slot          = target.gdn_current_slot;
        disk_target.rewrite_state             = target.rewrite_state;
        disk_target.tail_hidden               = target.tail_hidden;
        disk_target.rewrite_checkpoint_hidden = target.rewrite_checkpoint_hidden;
        disk_target.dflash_local              = target.dflash_local;
        disk_target.dflash_lane               = target.dflash_lane;
        disk_target.keep_rewrite_checkpoint   = target.keep_rewrite_checkpoint;
        disk_target.reuse                     = target.reuse;
        disk_target.reuse_base                = target.reuse_base;
        disk_target.stream                    = target.stream;
        const auto ticket                     = disk.restore_device(id, disk_target);
        disk.wait_copies(ticket);
        require(!disk.restore_failed(), "state-image disk restore failed");
        require(disk.test_state_decode_count() == (mode == StateRestore::Frontier ? 6 : 3),
                "disk decoded a state set the planned path discards");
        disk.release_restore_ticket(ticket);
        disk.release(id);
    } else {
        cache.claim(entry_id);
        const auto restored = cache.unpack_device(entry_id, target);
        require(restored.rewrite_valid && restored.rewrite_frontier == 4,
                "RAM restore lost the rewrite checkpoint metadata");
        cache.consume(entry_id);
    }
    device.synchronize_all();
    std::vector<std::uint8_t> back_conv(conv_bytes);
    std::vector<std::uint8_t> back_rec(rec_bytes);
    std::vector<std::uint8_t> back_cyclic(cyclic_bytes);
    gdn.pack_slot_to_host(lane, back_conv.data(), back_rec.data(), device.stream);
    cyclic.copy_lane_to_host(lane, back_cyclic.data(), device.stream);
    HIP_CHECK(hipStreamSynchronize(device.stream));
    dest.release();
    require(out_conv == (read_rewrite ? rewrite_conv : std::vector<std::uint8_t>(conv_bytes, 0)) &&
                out_rec == (read_rewrite ? rewrite_rec : std::vector<std::uint8_t>(rec_bytes, 0)) &&
                out_cyclic == (read_rewrite ? rewrite_cyclic : zero_cyclic),
            "restore changed or omitted the wrong rewrite host image");
    require(
        back_conv == (read_frontier ? current_conv : std::vector<std::uint8_t>(conv_bytes, 0)) &&
            back_rec == (read_frontier ? current_rec : std::vector<std::uint8_t>(rec_bytes, 0)) &&
            back_cyclic == (read_frontier ? current_cyclic : zero_cyclic),
        "restore changed or omitted the wrong frontier GDN/DFlash lane");
    std::vector<std::uint8_t> hidden_got(hidden_bytes.size());
    hidden_bytes.copy_to_host(hidden_got.data(), hidden_got.size());
    require(
        std::all_of(hidden_got.begin(), hidden_got.begin() + 1024,
                    [read_frontier](auto byte) { return byte == (read_frontier ? 0x5a : 0xcc); }) &&
            std::all_of(hidden_got.begin() + 1024, hidden_got.end(),
                        [read_rewrite](auto byte) { return byte == (read_rewrite ? 0x5a : 0xcc); }),
        "restore changed or omitted the wrong hidden panel");
}

// Independent byte oracle for the production three-plane codec. In particular, the
// expected host image is built from documented [plane][page][head] geometry, not from
// either transfer implementation or its tensor strides.
void head_major_host_transfer_oracle(ninfer::DeviceContext& device) {
    struct Case {
        std::uint32_t pages;
        std::uint32_t restored;
        int fragmentation; // 0: contiguous, 1: every page, 2: mixed short/long runs
    };

    constexpr std::uint32_t heads = 4;
    constexpr std::array<std::size_t, 3> head_bytes{64U * 256U, 64U * 128U, 64U * 16U * 2U};
    for (const auto test :
         std::array{Case{4, 4, 0}, Case{5, 5, 0}, Case{5, 4, 0}, Case{32, 32, 0}, Case{32, 17, 0},
                    Case{32, 32, 1}, Case{32, 17, 1}, Case{32, 32, 2}, Case{32, 17, 2}}) {
        ninfer::Fp8KInt4VPagedKVSpec spec{
            .page_group_count      = test.pages * (test.fragmentation ? 4U : 2U) + 2U,
            .logical_page_capacity = test.pages,
            .table_rows            = 2,
            .layer_count           = 16,
            .head_dim              = 256,
            .num_kv_heads          = static_cast<std::int32_t>(heads),
            .value_group           = 16,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor},
        };
        ninfer::LayoutBuilder builder;
        const auto layout = ninfer::plan_fp8_k_int4_v_paged_kv_pool(builder, spec);
        ninfer::DeviceBuffer backing(builder.finish(256));
        ninfer::PagedKVPool pool({backing.data(), backing.size()}, layout.storage);
        std::vector<ninfer::PagedKVAllocation> blockers;
        blockers.reserve(test.pages * 2U);
        auto materialize = [&](ninfer::PagedKVAllocation& allocation) {
            if (!test.fragmentation) {
                allocation.materialize_pages(test.pages, device.stream);
                return;
            }
            for (std::uint32_t page = 0; page < test.pages; ++page) {
                allocation.materialize_pages(page + 1U, device.stream);
                if (test.fragmentation == 1 || page % 7U == 0U) {
                    auto blocker = pool.reserve(1);
                    blocker.materialize_pages(1, device.stream);
                    blockers.push_back(std::move(blocker));
                }
            }
        };
        auto source = pool.reserve(test.pages);
        materialize(source);
        auto destination = pool.reserve(test.pages);
        materialize(destination);

        std::size_t expected_size = 0;
        for (std::size_t plane = 0; plane < 48; ++plane)
            expected_size += test.pages * heads * head_bytes[plane % 3];
        require(expected_size == ninfer::paged_kv_host_image_bytes(pool, test.pages),
                "production transfer image size disagrees with exact codec oracle");
        std::vector<std::uint8_t> expected(expected_size);
        for (std::size_t byte = 0; byte < expected.size(); byte += sizeof(std::uint64_t)) {
            const std::uint64_t value =
                0x9e3779b97f4a7c15ULL * (byte / 8U + 1U) ^ 0xd1b54a32d192ed03ULL;
            std::memcpy(expected.data() + byte, &value, sizeof(value));
        }
        // This formula comes directly from each plane's physical storage contract.
        const auto physical_offset = [&](std::size_t plane, std::int32_t physical,
                                         std::size_t head) {
            const auto page = static_cast<std::size_t>(physical);
            const auto slot =
                plane % 3 == 0 ? head * spec.page_group_count + page : page * heads + head;
            return slot * head_bytes[plane % 3];
        };
        std::size_t image_offset = 0;
        for (std::size_t plane = 0; plane < 48; ++plane) {
            const auto& tensor = pool.plane(plane);
            const auto row     = head_bytes[plane % 3];
            require(tensor.bytes() == spec.page_group_count * heads * row,
                    "production plane size disagrees with exact codec oracle");
            std::vector<std::uint8_t> physical(tensor.bytes(), 0xa5);
            for (std::size_t page = 0; page < test.pages; ++page)
                for (std::size_t head = 0; head < heads; ++head)
                    std::memcpy(physical.data() +
                                    physical_offset(plane, source.page_ids()[page], head),
                                expected.data() + image_offset + (page * heads + head) * row, row);
            HIP_CHECK(hipMemcpyAsync(tensor.data, physical.data(), physical.size(),
                                     hipMemcpyHostToDevice, device.stream));
            device.synchronize(); // pageable initialization stays alive until consumed
            image_offset += test.pages * heads * row;
        }
        ninfer::PinnedHostBuffer image(expected_size + 256U);
        std::memset(image.data(), 0xcc, expected_size + 256U);
        ninfer::pack_paged_kv_allocation_to_host(source, pool, image.data(), test.pages,
                                                 device.stream);
        // No intervening synchronization: stream ordering must make the freshly captured
        // image visible to H2D. Partial restores still stride planes by captured pages.
        ninfer::unpack_paged_kv_allocation_from_host(destination, pool, image.data(), test.pages,
                                                     test.restored, device.stream);
        device.synchronize();
        require(std::memcmp(image.data(), expected.data(), expected_size) == 0,
                "production D2H transfer differs from independent byte-image oracle");
        const auto* guard = static_cast<const std::uint8_t*>(image.data()) + expected_size;
        require(std::all_of(guard, guard + 256U, [](auto byte) { return byte == 0xcc; }),
                "production D2H transfer overwrote its image guard");
        image_offset = 0;
        for (std::size_t plane = 0; plane < 48; ++plane) {
            const auto& tensor = pool.plane(plane);
            const auto row     = head_bytes[plane % 3];
            std::vector<std::uint8_t> physical(tensor.bytes(), 0xa5), actual(tensor.bytes());
            for (std::size_t page = 0; page < test.pages; ++page)
                for (std::size_t head = 0; head < heads; ++head) {
                    const auto* value =
                        expected.data() + image_offset + (page * heads + head) * row;
                    std::memcpy(physical.data() +
                                    physical_offset(plane, source.page_ids()[page], head),
                                value, row);
                    if (page < test.restored)
                        std::memcpy(physical.data() +
                                        physical_offset(plane, destination.page_ids()[page], head),
                                    value, row);
                }
            HIP_CHECK(hipMemcpy(actual.data(), tensor.data, actual.size(), hipMemcpyDeviceToHost));
            require(actual == physical,
                    "production H2D transfer changed expected bytes or untouched physical storage");
            image_offset += test.pages * heads * row;
        }
    }
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        require(device.is_gfx1201() && device.props.warpSize == 32,
                "KV RAM qualifier requires wave32 gfx1201");

        const ninfer::Fp8KInt4VPagedKVSpec text_spec{
            .page_group_count      = 10,
            .logical_page_capacity = 8,
            .table_rows            = 4,
            .layer_count           = 2,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor},
        };
        const ninfer::Fp8KInt4VPagedKVSpec mtp_spec{
            .page_group_count      = 8,
            .logical_page_capacity = 8,
            .table_rows            = 4,
            .layer_count           = 1,
            .head_dim              = 256,
            .num_kv_heads          = 4,
            .value_group           = 16,
            .plane_layouts = {.key         = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor,
                              .value       = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor,
                              .value_scale = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor},
        };
        const auto text_semantics = ninfer::fp8_k_int4_v_semantic_fingerprint(text_spec);
        const auto mtp_semantics  = ninfer::fp8_k_int4_v_semantic_fingerprint(mtp_spec);

        ninfer::LayoutBuilder text_source_builder;
        const auto text_source_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(text_source_builder, text_spec);
        ninfer::DeviceBuffer text_source_backing(text_source_builder.finish(256));
        ninfer::PagedKVPool text_source_pool(
            {text_source_backing.data(), text_source_backing.size()}, text_source_layout.storage);
        auto text_source = text_source_pool.reserve(3);
        text_source.materialize_pages(1, device.stream);
        auto text_hole = text_source_pool.reserve(1);
        text_hole.materialize_pages(1, device.stream);
        text_source.materialize_pages(3, device.stream);

        ninfer::LayoutBuilder mtp_source_builder;
        const auto mtp_source_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(mtp_source_builder, mtp_spec);
        ninfer::DeviceBuffer mtp_source_backing(mtp_source_builder.finish(256));
        ninfer::PagedKVPool mtp_source_pool({mtp_source_backing.data(), mtp_source_backing.size()},
                                            mtp_source_layout.storage);
        auto mtp_source = mtp_source_pool.reserve(3);
        mtp_source.materialize_pages(1, device.stream);
        auto mtp_hole = mtp_source_pool.reserve(1);
        mtp_hole.materialize_pages(1, device.stream);
        mtp_source.materialize_pages(3, device.stream);

        require(text_source_pool.plane_count() == 6 && mtp_source_pool.plane_count() == 3,
                "typed Text/MTP pools do not own exactly three planes per layer");
        require(text_source.page_ids()[1] != text_source.page_ids()[0] + 1 &&
                    mtp_source.page_ids()[1] != mtp_source.page_ids()[0] + 1,
                "source page maps were not fragmented");
        paint_allocation(text_source_pool, text_source, 0x21U, device.stream);
        paint_allocation(mtp_source_pool, mtp_source, 0x71U, device.stream);
        device.order_copy_after_compute();

        q3::PreparedPromptData prompt = prompt_data(101);
        q3::detail::ResidentPrefixIdentity identity;
        identity.assign(prompt);
        q3::detail::RamCaptureSource source;
        source.execution_frontier = 6;
        source.ledger_frontier    = 7;
        source.text_kv_valid      = 6;
        source.mtp_kv_valid       = 5;
        source.ledger             = prompt.token_ids;
        source.identity           = &identity;
        source.hash_f             = q3::detail::prefix_hash_at(prompt.token_ids, identity, 6);
        source.text               = &text_source;
        source.text_pool          = &text_source_pool;
        source.text_semantics     = text_semantics;
        source.backend            = &mtp_source;
        source.backend_pool       = &mtp_source_pool;
        source.backend_semantics  = mtp_semantics;
        source.stream             = device.copy_stream;

        q3::detail::KVRamCache cache(8U << 20);
        require(cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "Text+MTP KV RAM capture failed");
        const auto match = cache.plan_match(prompt, q3::detail::prefix_hash_chain(prompt));
        require(match.has_value(), "captured KV RAM entry did not match its exact identity");

        ninfer::LayoutBuilder text_dest_builder;
        const auto text_dest_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(text_dest_builder, text_spec);
        ninfer::DeviceBuffer text_dest_backing(text_dest_builder.finish(256));
        ninfer::PagedKVPool text_dest_pool({text_dest_backing.data(), text_dest_backing.size()},
                                           text_dest_layout.storage);
        auto text_dest = text_dest_pool.reserve(3);
        text_dest.materialize_pages(1, device.stream);
        auto text_dest_hole = text_dest_pool.reserve(1);
        text_dest_hole.materialize_pages(1, device.stream);
        text_dest.materialize_pages(3, device.stream);

        ninfer::LayoutBuilder mtp_dest_builder;
        const auto mtp_dest_layout =
            ninfer::plan_fp8_k_int4_v_paged_kv_pool(mtp_dest_builder, mtp_spec);
        ninfer::DeviceBuffer mtp_dest_backing(mtp_dest_builder.finish(256));
        ninfer::PagedKVPool mtp_dest_pool({mtp_dest_backing.data(), mtp_dest_backing.size()},
                                          mtp_dest_layout.storage);
        auto mtp_dest = mtp_dest_pool.reserve(3);
        mtp_dest.materialize_pages(1, device.stream);
        auto mtp_dest_hole = mtp_dest_pool.reserve(1);
        mtp_dest_hole.materialize_pages(1, device.stream);
        mtp_dest.materialize_pages(3, device.stream);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        text_dest_backing.fill(0xA5);
        mtp_dest_backing.fill(0xA5);

        q3::detail::RamRestoreTarget target;
        target.text_dst_pages    = 3;
        target.backend_dst_pages = 3;
        target.text              = &text_dest;
        target.text_pool         = &text_dest_pool;
        target.text_semantics    = text_semantics;
        target.backend           = &mtp_dest;
        target.backend_pool      = &mtp_dest_pool;
        target.backend_semantics = mtp_semantics;
        target.stream            = device.stream;

        auto wrong                       = target;
        wrong.text_semantics.value_group = 32;
        bool rejected_mismatch           = false;
        try {
            (void)cache.unpack_device(match.value().entry_id, wrong);
        } catch (const std::invalid_argument&) { rejected_mismatch = true; }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) && all_bytes_equal(mtp_dest_backing, 0xA5),
                "semantic mismatch modified a destination byte");

        auto wrong_backend                           = target;
        wrong_backend.backend_semantics->value_group = 32;
        rejected_mismatch                            = false;
        try {
            (void)cache.unpack_device(match.value().entry_id, wrong_backend);
        } catch (const std::invalid_argument&) { rejected_mismatch = true; }
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(rejected_mismatch, "backend semantic fingerprint mismatch was accepted");
        require(all_bytes_equal(text_dest_backing, 0xA5) && all_bytes_equal(mtp_dest_backing, 0xA5),
                "backend semantic mismatch modified a destination byte");

        const q3::detail::RamRestoredHost restored =
            cache.unpack_device(match.value().entry_id, target);
        HIP_CHECK(hipStreamSynchronize(device.stream));
        require(restored.text_kv_valid == 6 && restored.mtp_kv_valid == 5,
                "restored Text/MTP frontiers changed");
        verify_allocation(text_dest_pool, text_dest, 0x21U, "Text");
        verify_allocation(mtp_dest_pool, mtp_dest, 0x71U, "MTP");
        const auto copy_seconds = cache.harvest_copy_seconds();
        require(copy_seconds.save >= 0.0 && copy_seconds.load >= 0.0,
                "copy timing accounting is invalid");

        const std::size_t entry_bytes = cache.snapshot().used_bytes;
        require(entry_bytes != 0, "captured KV RAM entry has no storage");
        q3::detail::KVRamCache eviction_cache(entry_bytes + 255U);
        require(eviction_cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "eviction probe first capture failed");
        require(eviction_cache.capture(source).status ==
                    q3::detail::RamCaptureStatus::NeedsEviction,
                "full RAM cache did not request eviction");
        const std::vector<std::uint64_t> unpinned = eviction_cache.unpinned_ids();
        require(!unpinned.empty(), "eviction probe found no unpinned entry");
        // An evicted block is freed at return only once its copies completed.
        eviction_cache.wait_entry_copies(unpinned.front());
        require(eviction_cache.evict_one_unpinned(unpinned.front()),
                "eviction probe could not release the oldest entry");
        require(eviction_cache.capture(source).status == q3::detail::RamCaptureStatus::Captured,
                "eviction probe replacement capture failed");
        const auto eviction = eviction_cache.snapshot();
        require(eviction.entry_count == 1 && eviction.captures == 2 && eviction.evictions == 1,
                "unpinned FIFO entry was not evicted for the replacement image");

        head_major_host_transfer_oracle(device);
        system_prompt_lru(device, text_spec);
        source_preserving_reads(device, text_spec);
        auto mixed_spec                = text_spec;
        mixed_spec.plane_layouts.key   = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor;
        mixed_spec.plane_layouts.value = ninfer::Fp8KInt4VPlaneLayout::FeatureFastestPageMajor;
        mixed_spec.plane_layouts.value_scale = ninfer::Fp8KInt4VPlaneLayout::TokenFastestHeadMajor;
        source_preserving_reads(device, mixed_spec);
        live_credit_allocation_failure();
        live_credit_image_ownership(device, text_spec);
        two_pending_loads_fold_as_loads(device, text_spec);
        stream_wait_does_not_block_host(device, text_spec);
        for (const std::uint32_t pages : {1U, 5U})
            failed_unpack_fences_enqueued_reads(device, text_spec, pages);
        for (bool disk : {false, true}) {
            for (StateRestore mode : {StateRestore::Frontier, StateRestore::Turn,
                                      StateRestore::Response, StateRestore::Drop}) {
                rewrite_images_round_trip(device, text_spec, mode, disk);
            }
        }

        std::cout << "kv_ram_cache: PASS text_planes=" << text_source_pool.plane_count()
                  << " mtp_planes=" << mtp_source_pool.plane_count()
                  << " entry_bytes=" << entry_bytes << " evictions=" << eviction.evictions << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_ram_cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}

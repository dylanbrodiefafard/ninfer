#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/cache_cuda_event.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {
namespace {

constexpr std::uint32_t kRamMagic      = 0x4D41524E;
constexpr std::uint32_t kRamVersion    = 5;
constexpr std::size_t kSectionCount    = 12;
constexpr std::size_t kHostAlign       = 8;
constexpr std::size_t kDeviceAlign     = 256;
constexpr std::size_t kFingerprint     = 56;
constexpr std::size_t kLadderMetaBytes = 88;

std::size_t align_up(std::size_t value, std::size_t align) {
    return (value + align - 1) & ~(align - 1);
}

struct Cursor {
    std::uint8_t* p   = nullptr;
    std::uint8_t* end = nullptr;

    void u8(std::uint8_t v) {
        if (p >= end) { throw std::logic_error("RAM entry write overflow"); }
        *p++ = v;
    }

    void u32(std::uint32_t v) {
        u8(static_cast<std::uint8_t>(v));
        u8(static_cast<std::uint8_t>(v >> 8));
        u8(static_cast<std::uint8_t>(v >> 16));
        u8(static_cast<std::uint8_t>(v >> 24));
    }

    void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }

    void u64(std::uint64_t v) {
        for (int s = 0; s < 64; s += 8) { u8(static_cast<std::uint8_t>(v >> s)); }
    }

    void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }

    void bytes(const void* data, std::size_t n) {
        const auto* raw = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { u8(raw[i]); }
    }
};

struct InCursor {
    const std::uint8_t* p   = nullptr;
    const std::uint8_t* end = nullptr;

    [[nodiscard]] std::uint8_t u8() {
        if (p >= end) { throw std::logic_error("RAM entry read overflow"); }
        return *p++;
    }

    [[nodiscard]] std::uint32_t u32() {
        const std::uint32_t a = u8(), b = u8(), c = u8(), d = u8();
        return a | (b << 8) | (c << 16) | (d << 24);
    }

    [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t v = 0;
        for (int s = 0; s < 64; s += 8) { v |= static_cast<std::uint64_t>(u8()) << s; }
        return v;
    }

    [[nodiscard]] std::int64_t i64() { return static_cast<std::int64_t>(u64()); }

    void bytes(void* data, std::size_t n) {
        auto* raw = static_cast<std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { raw[i] = u8(); }
    }

    void skip(std::size_t n) {
        if (static_cast<std::size_t>(end - p) < n) {
            throw std::logic_error("RAM entry skip overflow");
        }
        p += n;
    }
};

void write_fingerprint(Cursor& w, const Tensor& plane, PagedKVPlaneOrder order) {
    w.u8(static_cast<std::uint8_t>(plane.dtype));
    w.u8(static_cast<std::uint8_t>(order));
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    for (int i = 0; i < 4; ++i) { w.i32(plane.ne[i]); }
    for (int i = 0; i < 4; ++i) { w.i64(plane.nb[i]); }
}

void check_fingerprint(InCursor& r, const Tensor& plane, PagedKVPlaneOrder order,
                       const char* label) {
    const auto dtype        = static_cast<DType>(r.u8());
    const auto stored_order = static_cast<PagedKVPlaneOrder>(r.u8());
    r.skip(6);
    std::int32_t ne[4];
    std::int64_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = r.i32(); }
    for (int i = 0; i < 4; ++i) { nb[i] = r.i64(); }
    if (dtype != plane.dtype || stored_order != order) {
        throw std::logic_error(std::string(label) + " plane dtype/order mismatch");
    }
    for (int i = 0; i < 4; ++i) {
        if (ne[i] != plane.ne[i] || nb[i] != plane.nb[i]) {
            throw std::logic_error(std::string(label) + " plane geometry mismatch");
        }
    }
}

struct HeaderView {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    bool hash_c_valid                     = false;
    std::uint32_t rewrite_frontier        = 0;
    std::uint32_t text_mapped_pages       = 0;
    std::uint32_t backend_mapped_pages    = 0;
    std::uint32_t text_plane_count        = 0;
    std::uint32_t backend_plane_count     = 0;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool has_gdn                      = false;
    bool has_dflash                   = false;
    std::uint32_t cyclic_layers       = 0;
    std::uint32_t cyclic_capacity     = 0;
    std::uint32_t cyclic_padded       = 0;
    std::int32_t cyclic_kv_heads      = 0;
    std::int32_t cyclic_head_dim      = 0;
    std::int32_t cyclic_lane_capacity = 0;
    std::uint64_t tail_hidden_bytes   = 0;
    std::uint64_t gdn_conv_bytes      = 0;
    std::uint64_t gdn_recurrent_bytes = 0;
    std::uint64_t cyclic_lane_bytes   = 0;
    std::uint32_t ladder_count        = 0;
    std::uint64_t disk_entry_id       = 0;
    std::array<std::uint64_t, kSectionCount> offset{};
    std::array<std::uint64_t, kSectionCount> length{};
    std::uint64_t entry_bytes = 0;
    std::size_t header_bytes  = 0;
    std::vector<RamLadderIndex> ladders;
    std::vector<RamLadderImage> ladder_images;
};

constexpr std::size_t kFixedHeader      = 364;
constexpr std::size_t kDiskTicketOffset = 156;

std::size_t header_bytes_for(std::uint32_t text_planes, std::uint32_t backend_planes,
                             std::uint32_t ladder_count) {
    return align_up(kFixedHeader + kFingerprint * (text_planes + backend_planes) +
                        static_cast<std::size_t>(ladder_count) * kLadderMetaBytes,
                    kHostAlign);
}

void write_fixed_header(Cursor& w, const HeaderView& h) {
    w.u32(kRamMagic);
    w.u32(kRamVersion);
    w.u32(h.execution_frontier);
    w.u32(h.ledger_frontier);
    w.i32(h.rope_delta);
    w.u32(h.text_kv_valid);
    w.u32(h.mtp_kv_valid);
    w.u32(h.dflash_context_frontier);
    w.u8(h.tail_hidden_valid ? 1 : 0);
    w.u8(h.rewrite_valid ? 1 : 0);
    w.u8(static_cast<std::uint8_t>(h.rewrite_kind));
    w.u8(h.hash_c_valid ? 1 : 0);
    w.u32(h.rewrite_frontier);
    w.u32(h.text_mapped_pages);
    w.u32(h.backend_mapped_pages);
    w.u32(h.text_plane_count);
    w.u32(h.backend_plane_count);
    w.u64(h.hash_f.lo);
    w.u64(h.hash_f.hi);
    w.u64(h.hash_c.lo);
    w.u64(h.hash_c.hi);
    w.u8(h.has_gdn ? 1 : 0);
    w.u8(h.has_dflash ? 1 : 0);
    w.u8(0);
    w.u8(0);
    w.u32(h.cyclic_layers);
    w.u32(h.cyclic_capacity);
    w.u32(h.cyclic_padded);
    w.i32(h.cyclic_kv_heads);
    w.i32(h.cyclic_head_dim);
    w.i32(h.cyclic_lane_capacity);
    w.u64(h.tail_hidden_bytes);
    w.u64(h.gdn_conv_bytes);
    w.u64(h.gdn_recurrent_bytes);
    w.u64(h.cyclic_lane_bytes);
    w.u32(h.ladder_count);
    w.u32(0);
    w.u64(h.disk_entry_id);
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        w.u64(h.offset[i]);
        w.u64(h.length[i]);
    }
    w.u64(h.entry_bytes);
}

HeaderView read_header(const void* block, std::size_t bytes) {
    if (block == nullptr || bytes < kFixedHeader) {
        throw std::logic_error("RAM entry header is truncated");
    }
    const auto* raw = static_cast<const std::uint8_t*>(block);
    InCursor r{raw, raw + bytes};
    HeaderView h;
    if (r.u32() != kRamMagic || r.u32() != kRamVersion) {
        throw std::logic_error("RAM entry magic/version mismatch");
    }
    h.execution_frontier      = r.u32();
    h.ledger_frontier         = r.u32();
    h.rope_delta              = r.i32();
    h.text_kv_valid           = r.u32();
    h.mtp_kv_valid            = r.u32();
    h.dflash_context_frontier = r.u32();
    h.tail_hidden_valid       = r.u8() != 0;
    h.rewrite_valid           = r.u8() != 0;
    h.rewrite_kind            = static_cast<RewriteCheckpointKind>(r.u8());
    h.hash_c_valid            = r.u8() != 0;
    h.rewrite_frontier        = r.u32();
    h.text_mapped_pages       = r.u32();
    h.backend_mapped_pages    = r.u32();
    h.text_plane_count        = r.u32();
    h.backend_plane_count     = r.u32();
    h.hash_f.lo               = r.u64();
    h.hash_f.hi               = r.u64();
    h.hash_c.lo               = r.u64();
    h.hash_c.hi               = r.u64();
    h.has_gdn                 = r.u8() != 0;
    h.has_dflash              = r.u8() != 0;
    r.skip(2);
    h.cyclic_layers        = r.u32();
    h.cyclic_capacity      = r.u32();
    h.cyclic_padded        = r.u32();
    h.cyclic_kv_heads      = r.i32();
    h.cyclic_head_dim      = r.i32();
    h.cyclic_lane_capacity = r.i32();
    h.tail_hidden_bytes    = r.u64();
    h.gdn_conv_bytes       = r.u64();
    h.gdn_recurrent_bytes  = r.u64();
    h.cyclic_lane_bytes    = r.u64();
    h.ladder_count         = r.u32();
    r.skip(4);
    h.disk_entry_id = r.u64();
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        h.offset[i] = r.u64();
        h.length[i] = r.u64();
    }
    h.entry_bytes  = r.u64();
    h.header_bytes = header_bytes_for(h.text_plane_count, h.backend_plane_count, h.ladder_count);
    if (h.entry_bytes > bytes || h.header_bytes > bytes) {
        throw std::logic_error("RAM entry header size is inconsistent");
    }
    const std::size_t meta_begin =
        kFixedHeader + kFingerprint * (h.text_plane_count + h.backend_plane_count);
    InCursor meta{raw + meta_begin, raw + h.header_bytes};
    h.ladders.reserve(h.ladder_count);
    h.ladder_images.reserve(h.ladder_count);
    for (std::uint32_t i = 0; i < h.ladder_count; ++i) {
        RamLadderIndex index;
        index.frontier = meta.u32();
        index.kind     = static_cast<ContextCheckpointKind>(meta.u32());
        index.hash.lo  = meta.u64();
        index.hash.hi  = meta.u64();
        RamLadderImage image;
        image.frontier        = index.frontier;
        image.hash            = index.hash;
        image.kind            = index.kind;
        const auto conv_off   = meta.u64();
        const auto rec_off    = meta.u64();
        const auto hidden_off = meta.u64();
        const auto dflash_off = meta.u64();
        image.conv_bytes      = static_cast<std::size_t>(meta.u64());
        image.recurrent_bytes = static_cast<std::size_t>(meta.u64());
        image.hidden_bytes    = static_cast<std::size_t>(meta.u64());
        image.dflash_bytes    = static_cast<std::size_t>(meta.u64());
        image.conv            = image.conv_bytes != 0 ? raw + conv_off : nullptr;
        image.recurrent       = image.recurrent_bytes != 0 ? raw + rec_off : nullptr;
        image.hidden          = image.hidden_bytes != 0 ? raw + hidden_off : nullptr;
        image.dflash          = image.dflash_bytes != 0 ? raw + dflash_off : nullptr;
        h.ladders.push_back(index);
        h.ladder_images.push_back(image);
    }
    return h;
}

const std::uint8_t* section_ptr(const void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<const std::uint8_t*>(block) + header.offset[index];
}

std::uint8_t* section_ptr(void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<std::uint8_t*>(block) + header.offset[index];
}

void verify_pool(InCursor& r, const PagedKVPool& pool, std::uint32_t stored_planes,
                 const char* label) {
    if (stored_planes != pool.plane_count()) {
        throw std::logic_error(std::string(label) + " plane count mismatch");
    }
    for (std::uint32_t i = 0; i < stored_planes; ++i) {
        check_fingerprint(r, pool.plane(i), pool.plane_order(), label);
    }
}

void verify_cyclic(const HeaderView& header, const CyclicKVCache& cache) {
    if (header.cyclic_layers != cache.layer_count() || header.cyclic_capacity != cache.capacity() ||
        header.cyclic_padded != cache.padded_capacity() ||
        header.cyclic_kv_heads != cache.num_kv_heads() ||
        header.cyclic_head_dim != cache.head_dim() ||
        header.cyclic_lane_capacity != cache.lane_capacity() ||
        header.cyclic_lane_bytes != cache.lane_host_bytes()) {
        throw std::logic_error("RAM entry cyclic geometry mismatch");
    }
}

RamRestoredHost host_from_header(const void* block, const HeaderView& header) {
    RamRestoredHost out;
    out.execution_frontier      = header.execution_frontier;
    out.ledger_frontier         = header.ledger_frontier;
    out.rope_delta              = header.rope_delta;
    out.text_kv_valid           = header.text_kv_valid;
    out.mtp_kv_valid            = header.mtp_kv_valid;
    out.dflash_context_frontier = header.dflash_context_frontier;
    out.tail_hidden_valid       = header.tail_hidden_valid;
    out.rewrite_valid           = header.rewrite_valid;
    out.rewrite_kind            = header.rewrite_kind;
    out.rewrite_frontier        = header.rewrite_frontier;
    out.backend_image_present   = header.backend_mapped_pages > 0;
    const auto* ledger          = section_ptr(block, header, 0);
    if (header.length[0] != header.ledger_frontier * sizeof(TokenId)) {
        throw std::logic_error("RAM entry ledger size mismatch");
    }
    out.ledger.resize(header.ledger_frontier);
    if (!out.ledger.empty()) { std::memcpy(out.ledger.data(), ledger, header.length[0]); }
    const auto* identity = section_ptr(block, header, 1);
    out.identity.unpack(identity, static_cast<std::size_t>(header.length[1]));
    out.ladders       = header.ladders;
    out.ladder_images = header.ladder_images;
    out.disk_entry_id = header.disk_entry_id;
    return out;
}

} // namespace

KVRamCache::KVRamCache(std::size_t capacity_bytes) : arena_(capacity_bytes) {}

KVRamCache::KVRamCache(HostPinnedArena&& arena) : arena_(std::move(arena)) {}

KVRamCache::~KVRamCache() {
    // Teardown must remain allocation-free, including after host-memory pressure.
    // The disk owner has already joined its workers; no snapshot is needed.
    for (auto& [id, record] : records_) {
        if (record.copies_start != nullptr) {
            (void)cudaEventDestroy(record.copies_start);
            record.copies_start = nullptr;
        }
        if (record.copies_done != nullptr) {
            maybe_copy_sync_hold();
            (void)cudaEventSynchronize(record.copies_done);
            (void)cudaEventDestroy(record.copies_done);
            record.copies_done = nullptr;
        }
        if (record.block_done != nullptr) {
            (void)cudaEventSynchronize(record.block_done);
            (void)cudaEventDestroy(record.block_done);
            record.block_done = nullptr;
        }
    }
    records_.clear();
    fifo_.clear();
}

KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) {
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.retired) {
        throw std::logic_error("RAM cache entry id is unknown");
    }
    return it->second;
}

const KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) const {
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.retired) {
        throw std::logic_error("RAM cache entry id is unknown");
    }
    return it->second;
}

const KVRamCache::Record& KVRamCache::require_block(std::uint64_t entry_id) const {
    const auto it = records_.find(entry_id);
    if (it == records_.end()) { throw std::logic_error("RAM cache entry id is unknown"); }
    return it->second;
}

void KVRamCache::retire_locked(std::uint64_t entry_id) {
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.retired) { return; }
    it->second.retired = true;
    it->second.pinned  = false;
    fifo_.erase(std::remove(fifo_.begin(), fifo_.end(), entry_id), fifo_.end());
    // A retired block's unharvested copy time joins the lifetime totals when the block is freed.
    drop_pending_id(entry_id);
    bump_version();
    if (retired_free_locked(it->second)) {
        free_block_locked(it->second);
        records_.erase(it);
    }
}

bool KVRamCache::retired_free_locked(const Record& record) const {
    if (!record.retired || record.io_pins != 0) { return false; }
    for (cudaEvent_t fence : {record.copies_done, record.block_done}) {
        if (fence == nullptr) { continue; }
        const cudaError_t ready = cudaEventQuery(fence);
        if (ready == cudaErrorNotReady) { return false; }
        CUDA_CHECK(ready);
    }
    return true;
}

void KVRamCache::free_block_locked(Record& record) {
    // The copy fence has completed, so the timing pair is final and the block has no reader.
    // Arena free uses metadata reserved at allocation time and cannot allocate.
    add_unbilled_seconds(record, harvest_record(record));
    if (record.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.copies_start));
        record.copies_start = nullptr;
    }
    if (record.copies_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.copies_done));
        record.copies_done = nullptr;
    }
    if (record.block_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.block_done));
        record.block_done = nullptr;
    }
    record.copies_timed = false;
    if (record.block != nullptr) {
        arena_.free(record.block);
        record.block = nullptr;
    }
}

void KVRamCache::reap_retired_locked() {
    for (auto it = records_.begin(); it != records_.end();) {
        if (!retired_free_locked(it->second)) {
            ++it;
            continue;
        }
        free_block_locked(it->second);
        it = records_.erase(it);
    }
}

bool KVRamCache::retired_pending() const {
    std::lock_guard lock(io_mutex_);
    // A block whose fence completed and whose pins dropped is free at the next reap.
    for (const auto& [id, record] : records_) {
        if (record.retired && !retired_free_locked(record)) { return true; }
    }
    return false;
}

bool KVRamCache::retired_fence_pending() const {
    std::lock_guard lock(io_mutex_);
    for (const auto& [id, record] : records_) {
        if (record.retired && record.io_pins == 0 && !retired_free_locked(record)) { return true; }
    }
    return false;
}

bool KVRamCache::wait_retired_copies() {
    std::vector<std::uint64_t> ids;
    std::vector<cudaEvent_t> events;
    std::size_t before = 0;
    {
        std::lock_guard lock(io_mutex_);
        before = records_.size();
        reap_retired_locked();
        for (const auto& [id, record] : records_) {
            if (record.retired && record.io_pins == 0 &&
                (record.copies_done != nullptr || record.block_done != nullptr)) {
                ids.push_back(id);
            }
        }
        events.reserve(2 * ids.size());
        // The pins keep each fence alive while it is waited with the mutex released.
        for (const std::uint64_t id : ids) {
            Record& record = records_.at(id);
            ++record.io_pins;
            for (cudaEvent_t fence : {record.copies_done, record.block_done}) {
                if (fence != nullptr) { events.push_back(fence); }
            }
        }
    }
    maybe_copy_sync_hold();
    try {
        for (cudaEvent_t event : events) { CUDA_CHECK(cudaEventSynchronize(event)); }
    } catch (...) {
        unpin_copy_events(ids);
        throw;
    }
    unpin_copy_events(ids);
    std::lock_guard lock(io_mutex_);
    return records_.size() < before;
}

void KVRamCache::create_copy_event(cudaEvent_t* event, unsigned int flags) {
    if (fail_copy_event_allocation_after_ >= 0) {
        const bool fail_now = fail_copy_event_allocation_after_ == 0;
        --fail_copy_event_allocation_after_;
        if (fail_now) { check_cache_cuda_event_allocation(cudaErrorMemoryAllocation); }
    }
    create_cache_cuda_event(event, flags);
}

void KVRamCache::begin_copies(Record& record, cudaStream_t stream) {
    if (record.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.copies_start));
        record.copies_start = nullptr;
    }
    create_copy_event(&record.copies_start, cudaEventDefault);
    CUDA_CHECK(cudaEventRecord(record.copies_start, stream));
    record.copies_timed = false;
}

void KVRamCache::record_copies(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) {
        create_copy_event(&record.copies_done, cudaEventBlockingSync);
    }
    CUDA_CHECK(cudaEventRecord(record.copies_done, stream));
    record.copies_timed = record.copies_start != nullptr;
}

double KVRamCache::copy_elapsed_seconds(const Record& record) const {
    if (!record.copies_timed || record.copies_start == nullptr || record.copies_done == nullptr) {
        return 0;
    }
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, record.copies_start, record.copies_done));
    return static_cast<double>(milliseconds) / 1000.0;
}

void KVRamCache::add_unbilled_seconds(const Record& record, double seconds) noexcept {
    // No request harvested this copy before its record was retired or re-timed (a rolled-back
    // capture, a failed admission): it counts toward the lifetime totals only, never toward
    // whichever unrelated request harvests next.
    if (record.copies_are_load) {
        load_seconds_ += seconds;
    } else {
        save_seconds_ += seconds;
    }
}

double KVRamCache::harvest_record(Record& record) {
    if (!record.copies_timed || record.copies_start == nullptr || record.copies_done == nullptr) {
        return 0;
    }
    const double seconds = copy_elapsed_seconds(record);
    record.copies_timed  = false;
    CUDA_CHECK(cudaEventDestroy(record.copies_start));
    record.copies_start = nullptr;
    return seconds;
}

KvRamCopySeconds KVRamCache::harvest_copy_seconds() {
    std::lock_guard lock(io_mutex_);
    reap_retired_locked();
    KvRamCopySeconds out;
    // Copies still in flight stay pending; removing ids only shrinks the vectors.
    const auto harvest_ready = [&](std::vector<std::uint64_t>& ids, double& billed, double& total) {
        std::size_t kept = 0;
        for (const std::uint64_t id : ids) {
            const auto it = records_.find(id);
            if (it == records_.end()) { continue; }
            if (!copies_ready_locked(id)) {
                ids[kept++] = id;
                continue;
            }
            const double seconds = harvest_record(it->second);
            billed += seconds;
            total += seconds;
        }
        ids.resize(kept);
    };
    harvest_ready(pending_save_ids_, out.save, save_seconds_);
    harvest_ready(pending_load_ids_, out.load, load_seconds_);
    return out;
}

bool KVRamCache::copies_ready_locked(std::uint64_t entry_id) const {
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.copies_done == nullptr) { return true; }
    const cudaError_t ready = cudaEventQuery(it->second.copies_done);
    if (ready == cudaErrorNotReady) { return false; }
    CUDA_CHECK(ready);
    return true;
}

bool KVRamCache::copies_ready(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    return copies_ready_locked(entry_id);
}

bool KVRamCache::pending_copies_ready() const {
    std::lock_guard lock(io_mutex_);
    for (std::uint64_t id : pending_save_ids_) {
        if (!copies_ready_locked(id)) { return false; }
    }
    for (std::uint64_t id : pending_load_ids_) {
        if (!copies_ready_locked(id)) { return false; }
    }
    return true;
}

void KVRamCache::wait_pending_copies_on_stream(cudaStream_t stream) {
    std::vector<cudaEvent_t> events;
    std::vector<std::uint64_t> pinned;
    {
        std::lock_guard lock(io_mutex_);
        pin_pending_copy_events(events, pinned);
    }
    try {
        // The pins only keep each event alive while it is enqueued; the stream
        // wait captures the event's current work, so the host does not block.
        for (cudaEvent_t event : events) {
            if (stream != nullptr) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
            } else {
                CUDA_CHECK(cudaEventSynchronize(event));
            }
        }
    } catch (...) {
        unpin_copy_events(pinned);
        throw;
    }
    unpin_copy_events(pinned);
}

void KVRamCache::wait_pending_copies() {
    std::vector<cudaEvent_t> events;
    std::vector<std::uint64_t> pinned;
    {
        std::lock_guard lock(io_mutex_);
        pin_pending_copy_events(events, pinned);
    }
    maybe_copy_sync_hold();
    try {
        for (cudaEvent_t event : events) { CUDA_CHECK(cudaEventSynchronize(event)); }
    } catch (...) {
        unpin_copy_events(pinned);
        throw;
    }
    unpin_copy_events(pinned);
}

void KVRamCache::wait_entry_copies(std::uint64_t entry_id) {
    cudaEvent_t event = nullptr;
    {
        std::lock_guard lock(io_mutex_);
        const auto it = records_.find(entry_id);
        if (it == records_.end() || it->second.copies_done == nullptr) { return; }
        ++it->second.io_pins;
        event = it->second.copies_done;
    }
    try {
        CUDA_CHECK(cudaEventSynchronize(event));
    } catch (...) {
        unpin_copy_events({entry_id});
        throw;
    }
    unpin_copy_events({entry_id});
}

void KVRamCache::wait_copies(Record& record) {
    if (record.copies_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(record.copies_done)); }
}

void KVRamCache::wait_copies_on_stream(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) { return; }
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, record.copies_done, 0));
        return;
    }
    wait_copies(record);
}

void KVRamCache::maybe_copy_sync_hold() const {
    copy_sync_entered_.store(true, std::memory_order_release);
    while (copy_sync_held_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVRamCache::pin_pending_copy_events(std::vector<cudaEvent_t>& events,
                                         std::vector<std::uint64_t>& ids) {
    // Publish pins only after both snapshot buffers can hold every event.
    // A failed optional snapshot must not strand pins or fail the request.
    try {
        if (fail_copy_snapshot_allocation_stage_ == 0) {
            fail_copy_snapshot_allocation_stage_ = -1;
            throw std::bad_alloc();
        }
        const std::size_t count = pending_save_ids_.size() + pending_load_ids_.size();
        ids.reserve(count);
        if (fail_copy_snapshot_allocation_stage_ == 1) {
            fail_copy_snapshot_allocation_stage_ = -1;
            throw std::bad_alloc();
        }
        events.reserve(count);
    } catch (const std::bad_alloc&) {
        // Every event is already recorded by the single executor. Holding the
        // mutex retains its owner while synchronously completing that snapshot;
        // CUDA completion does not require another cache operation or worker.
        maybe_copy_sync_hold();
        const auto wait = [&](std::uint64_t id) {
            const auto it = records_.find(id);
            if (it != records_.end() && it->second.copies_done != nullptr) {
                CUDA_CHECK(cudaEventSynchronize(it->second.copies_done));
            }
        };
        for (const auto id : pending_save_ids_) { wait(id); }
        for (const auto id : pending_load_ids_) { wait(id); }
        return;
    }
    auto pin = [&](std::uint64_t id) {
        const auto it = records_.find(id);
        if (it == records_.end() || it->second.copies_done == nullptr) { return; }
        ++it->second.io_pins;
        ids.push_back(id);
        events.push_back(it->second.copies_done);
    };
    for (std::uint64_t id : pending_save_ids_) { pin(id); }
    for (std::uint64_t id : pending_load_ids_) { pin(id); }
}

void KVRamCache::unpin_copy_events(const std::vector<std::uint64_t>& ids) noexcept {
    std::lock_guard lock(io_mutex_);
    for (std::uint64_t id : ids) {
        const auto it = records_.find(id);
        if (it == records_.end() || it->second.io_pins == 0) { continue; }
        --it->second.io_pins;
    }
    // The waited fences have completed: a retired block whose last pin was one of these frees
    // now. CUDA failures abort inside CUDA_CHECK; nothing else here throws.
    try {
        reap_retired_locked();
    } catch (...) { // NOLINT(bugprone-empty-catch): noexcept; the next mutating call reaps again.
    }
}

void KVRamCache::drop_pending_save(std::uint64_t entry_id) noexcept {
    pending_save_ids_.erase(
        std::remove(pending_save_ids_.begin(), pending_save_ids_.end(), entry_id),
        pending_save_ids_.end());
}

void KVRamCache::drop_pending_id(std::uint64_t entry_id) noexcept {
    drop_pending_save(entry_id);
    pending_load_ids_.erase(
        std::remove(pending_load_ids_.begin(), pending_load_ids_.end(), entry_id),
        pending_load_ids_.end());
}

void KVRamCache::claim(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    Record& record = require(entry_id);
    if (record.pinned) { throw std::logic_error("RAM cache entry is already claimed"); }
    record.pinned = true;
    bump_version();
}

bool KVRamCache::is_claimed(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    const auto it = records_.find(entry_id);
    return it != records_.end() && it->second.pinned;
}

void KVRamCache::release(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    Record& record = require(entry_id);
    if (!record.pinned) { throw std::logic_error("RAM cache entry is not claimed"); }
    record.pinned = false;
    bump_version();
}

void KVRamCache::consume(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    reap_retired_locked();
    const Record& record = require(entry_id);
    if (!record.pinned) { throw std::logic_error("RAM cache consume requires a claimed entry"); }
    ++restores_;
    retire_locked(entry_id);
}

void KVRamCache::discard(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    reap_retired_locked();
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.retired) { return; }
    if (it->second.pinned) { throw std::logic_error("RAM cache cannot discard a claimed entry"); }
    ++evictions_;
    retire_locked(entry_id);
}

std::vector<std::uint64_t> KVRamCache::unpinned_ids() const {
    std::lock_guard lock(io_mutex_);
    std::vector<std::uint64_t> ids;
    ids.reserve(fifo_.size());
    for (std::uint64_t id : fifo_) {
        const Record& record = require(id);
        if (!record.pinned && record.io_pins == 0) { ids.push_back(id); }
    }
    return ids;
}

std::vector<std::uint64_t> KVRamCache::fifo_ids() const {
    std::lock_guard lock(io_mutex_);
    return {fifo_.begin(), fifo_.end()};
}

void KVRamCache::pin_for_io(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    Record& record = require(entry_id);
    ++record.io_pins;
}

void KVRamCache::unpin_for_io(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.io_pins == 0) {
        throw std::logic_error("RAM cache I/O pin is not held");
    }
    --it->second.io_pins;
    // The last reader of a retired block frees it once its copy fence has completed.
    if (retired_free_locked(it->second)) {
        free_block_locked(it->second);
        records_.erase(it);
    }
}

bool KVRamCache::evict_one_unpinned(std::uint64_t entry_id) {
    std::lock_guard lock(io_mutex_);
    reap_retired_locked();
    const auto it = records_.find(entry_id);
    if (it == records_.end() || it->second.retired) { return false; }
    if (it->second.pinned || it->second.io_pins != 0) { return false; }
    ++evictions_;
    retire_locked(entry_id);
    return true;
}

void KVRamCache::set_disk_entry_id(std::uint64_t entry_id, std::uint64_t disk_id) {
    std::lock_guard lock(io_mutex_);
    const auto live = records_.find(entry_id);
    // A spill that outlived its retired source has nobody left to read the ticket.
    if (live != records_.end() && live->second.retired) { return; }
    Record& record = require(entry_id);
    if (fail_next_ticket_write_) {
        fail_next_ticket_write_ = false;
        throw std::logic_error("RAM disk ticket write failed");
    }
    if (record.block == nullptr || record.bytes < kDiskTicketOffset + 8) {
        throw std::logic_error("RAM entry header is truncated");
    }
    auto* raw = static_cast<std::uint8_t*>(record.block);
    Cursor w{raw + kDiskTicketOffset, raw + kDiskTicketOffset + 8};
    w.u64(disk_id);
    record.disk_entry_id = disk_id;
}

std::uint64_t KVRamCache::disk_entry_id(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    return require(entry_id).disk_entry_id;
}

const void* KVRamCache::host_block(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    return require_block(entry_id).block;
}

std::size_t KVRamCache::host_bytes(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    return require_block(entry_id).bytes;
}

KVRamCache::HostKvView KVRamCache::host_kv(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    const Record& record    = require_block(entry_id);
    const HeaderView header = read_header(record.block, record.bytes);
    const auto* raw         = static_cast<const std::uint8_t*>(record.block);
    HostKvView view;
    view.text_pages               = header.text_mapped_pages;
    view.backend_pages            = header.backend_mapped_pages;
    view.text                     = section_ptr(record.block, header, 2);
    view.backend                  = section_ptr(record.block, header, 3);
    view.ledger                   = section_ptr(record.block, header, 0);
    view.ledger_bytes             = static_cast<std::size_t>(header.length[0]);
    view.identity                 = section_ptr(record.block, header, 1);
    view.identity_bytes           = static_cast<std::size_t>(header.length[1]);
    view.gdn_conv_current         = section_ptr(record.block, header, 4);
    view.gdn_recurrent_current    = section_ptr(record.block, header, 6);
    view.gdn_conv_checkpoint      = section_ptr(record.block, header, 5);
    view.gdn_recurrent_checkpoint = section_ptr(record.block, header, 7);
    view.tail_hidden              = section_ptr(record.block, header, 8);
    view.rewrite_hidden           = section_ptr(record.block, header, 9);
    view.dflash_local             = section_ptr(record.block, header, 10);
    view.dflash_rewrite           = section_ptr(record.block, header, 11);
    view.gdn_conv_bytes           = static_cast<std::size_t>(header.gdn_conv_bytes);
    view.gdn_recurrent_bytes      = static_cast<std::size_t>(header.gdn_recurrent_bytes);
    view.hidden_bytes             = static_cast<std::size_t>(header.tail_hidden_bytes);
    view.rewrite_hidden_bytes     = static_cast<std::size_t>(header.length[9]);
    view.cyclic_bytes             = static_cast<std::size_t>(header.cyclic_lane_bytes);
    view.ladder_images            = header.ladder_images;
    (void)raw;
    return view;
}

std::optional<RamMatch> KVRamCache::plan_match(const PreparedPromptData& prompt,
                                               std::span<const PrefixHash128> hash_chain,
                                               const ReuseBackendPolicy& policy) {
    std::lock_guard lock(io_mutex_);
    if (fail_next_plan_metadata_allocation_.exchange(false, std::memory_order_acq_rel)) {
        throw std::bad_alloc();
    }
    std::optional<RamMatch> best;
    for (std::uint64_t id : fifo_) {
        const Record& record = require(id);
        if (record.pinned) { continue; }
        const auto hash_hits = [&](std::uint32_t frontier, PrefixHash128 hash) {
            return frontier > 0 && frontier < hash_chain.size() && hash_chain[frontier] == hash;
        };
        const bool frontier_hash = hash_hits(record.execution_frontier, record.hash_f);
        const bool checkpoint_hash =
            record.hash_c_valid && hash_hits(record.checkpoint_frontier, record.hash_c);
        bool ladder_hash = false;
        for (const RamLadderIndex& ladder : record.ladders) {
            if (hash_hits(ladder.frontier, ladder.hash)) {
                ladder_hash = true;
                break;
            }
        }
        if (!frontier_hash && !checkpoint_hash && !ladder_hash) { continue; }

        const HeaderView header    = read_header(record.block, record.bytes);
        const RamRestoredHost host = host_from_header(record.block, header);
        const ResidentReuseState state{
            .mtp_kv_valid            = host.mtp_kv_valid,
            .dflash_context_frontier = host.dflash_context_frontier,
            .tail_hidden_valid       = host.tail_hidden_valid,
            .backend_image_present   = host.backend_image_present,
        };
        RamMatch candidate;
        candidate.entry_id  = id;
        const auto consider = [&](PrefixReusePath path, std::uint32_t base) {
            if (base == 0 ||
                !reuse_candidate_ready(state, path, base, prompt.token_ids.size(), policy)) {
                return;
            }
            ++exact_comparisons_;
            if (!prefix_matches(prompt, host.ledger, host.identity, base)) { return; }
            if (base > candidate.reuse_base) {
                candidate.reuse      = path;
                candidate.reuse_base = base;
            }
        };
        if (frontier_hash) {
            consider(PrefixReusePath::AppendAtFrontier, record.execution_frontier);
        }
        if (checkpoint_hash) { consider(record.checkpoint_path, record.checkpoint_frontier); }
        for (const RamLadderIndex& ladder : record.ladders) {
            if (hash_hits(ladder.frontier, ladder.hash)) {
                consider(reuse_path_for_context_checkpoint_kind(ladder.kind), ladder.frontier);
            }
        }
        if (candidate.reuse_base == 0) { continue; }
        if (!best || candidate.reuse_base > best->reuse_base) { best = candidate; }
    }
    return best;
}

RamRestoredHost KVRamCache::load_host(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    const Record& record = require_block(entry_id);
    return host_from_header(record.block, read_header(record.block, record.bytes));
}

KvRamSnapshot KVRamCache::snapshot() const noexcept {
    std::lock_guard lock(io_mutex_);
    std::size_t used = 0;
    for (const auto& entry : records_) { used += entry.second.bytes; }
    return KvRamSnapshot{
        .capacity_bytes = arena_.capacity(),
        .used_bytes     = used,
        .entry_count    = fifo_.size(),
        .captures       = captures_,
        .restores       = restores_,
        .evictions      = evictions_,
        .drops          = drops_,
        .save_seconds   = save_seconds_,
        .load_seconds   = load_seconds_,
    };
}

void KVRamCache::record_drop() {
    std::lock_guard lock(io_mutex_);
    ++drops_;
    bump_version();
}

RamCaptureResult KVRamCache::capture(const RamCaptureSource& source) try {
    if (source.identity == nullptr || source.text == nullptr || source.text_pool == nullptr) {
        throw std::invalid_argument("RAM capture source is incomplete");
    }
    if (source.ledger.size() != source.ledger_frontier ||
        source.ledger_frontier != source.execution_frontier + 1) {
        throw std::logic_error("RAM capture ledger frontier is inconsistent");
    }
    if (fail_next_capture_) {
        fail_next_capture_ = false;
        ++drops_;
        bump_version();
        return {RamCaptureStatus::Dropped};
    }

    HeaderView header;
    header.execution_frontier      = source.execution_frontier;
    header.ledger_frontier         = source.ledger_frontier;
    header.rope_delta              = source.rope_delta;
    header.text_kv_valid           = source.text_kv_valid;
    header.mtp_kv_valid            = source.mtp_kv_valid;
    header.dflash_context_frontier = source.dflash_context_frontier;
    header.tail_hidden_valid       = source.tail_hidden_valid;
    header.rewrite_valid           = source.rewrite_valid;
    header.rewrite_kind            = source.rewrite_kind;
    header.hash_c_valid            = source.hash_c_valid;
    header.rewrite_frontier        = source.rewrite_frontier;
    header.text_mapped_pages       = source.text_pages.value_or(source.text->mapped_page_count());
    header.backend_mapped_pages =
        source.backend ? source.backend_pages.value_or(source.backend->mapped_page_count()) : 0;
    if (header.text_mapped_pages > source.text->mapped_page_count() ||
        (source.backend && header.backend_mapped_pages > source.backend->mapped_page_count())) {
        throw std::invalid_argument("RAM capture page extent exceeds mapped pages");
    }
    const bool current_from_host = source.current_state.conv != nullptr;
    if (current_from_host &&
        (source.current_state.recurrent == nullptr ||
         (source.dflash_local != nullptr && source.current_state.dflash == nullptr))) {
        throw std::invalid_argument("RAM capture current host image is incomplete");
    }
    header.text_plane_count = static_cast<std::uint32_t>(source.text_pool->plane_count());
    header.backend_plane_count =
        source.backend_pool ? static_cast<std::uint32_t>(source.backend_pool->plane_count()) : 0;
    header.hash_f     = source.hash_f;
    header.hash_c     = source.hash_c;
    header.has_gdn    = source.gdn != nullptr;
    header.has_dflash = source.dflash_local != nullptr;
    if (source.dflash_local != nullptr) {
        header.cyclic_layers        = source.dflash_local->layer_count();
        header.cyclic_capacity      = source.dflash_local->capacity();
        header.cyclic_padded        = source.dflash_local->padded_capacity();
        header.cyclic_kv_heads      = source.dflash_local->num_kv_heads();
        header.cyclic_head_dim      = source.dflash_local->head_dim();
        header.cyclic_lane_capacity = source.dflash_local->lane_capacity();
        header.cyclic_lane_bytes    = source.dflash_local->lane_host_bytes();
    }
    if (source.tail_hidden != nullptr) { header.tail_hidden_bytes = source.tail_hidden->bytes(); }
    if (source.gdn != nullptr) {
        header.gdn_conv_bytes      = source.gdn->conv_host_image_bytes();
        header.gdn_recurrent_bytes = source.gdn->recurrent_host_image_bytes();
    }
    header.ladder_count  = static_cast<std::uint32_t>(source.ladder_heads.size());
    header.disk_entry_id = source.disk_entry_id;

    std::array<std::size_t, kSectionCount> lengths{};
    std::array<std::size_t, kSectionCount> aligns{};
    lengths[0]  = source.ledger.size() * sizeof(TokenId);
    lengths[1]  = source.identity->packed_bytes();
    lengths[2]  = paged_kv_host_image_bytes(*source.text_pool, header.text_mapped_pages);
    lengths[3]  = source.backend_pool
                      ? paged_kv_host_image_bytes(*source.backend_pool, header.backend_mapped_pages)
                      : 0;
    lengths[4]  = source.gdn ? source.gdn->conv_host_image_bytes() : 0;
    lengths[5]  = source.gdn && source.rewrite_valid ? source.gdn->conv_host_image_bytes() : 0;
    lengths[6]  = source.gdn ? source.gdn->recurrent_host_image_bytes() : 0;
    lengths[7]  = source.gdn && source.rewrite_valid ? source.gdn->recurrent_host_image_bytes() : 0;
    lengths[8]  = source.tail_hidden ? source.tail_hidden->bytes() : 0;
    lengths[9]  = source.rewrite_valid && source.rewrite_checkpoint_hidden
                      ? source.rewrite_checkpoint_hidden->bytes()
                      : 0;
    lengths[10] = source.dflash_local ? source.dflash_local->lane_host_bytes() : 0;
    lengths[11] =
        source.dflash_local && source.rewrite_valid ? source.dflash_local->lane_host_bytes() : 0;
    if ((lengths[5] != 0 || lengths[7] != 0) &&
        (source.rewrite_state.conv == nullptr || source.rewrite_state.recurrent == nullptr)) {
        throw std::invalid_argument("RAM capture rewrite checkpoint has no GDN host image");
    }
    if (lengths[11] != 0 && source.rewrite_state.dflash == nullptr) {
        throw std::invalid_argument("RAM capture rewrite checkpoint has no DFlash host image");
    }
    aligns[0] = kHostAlign;
    aligns[1] = kHostAlign;
    for (std::size_t i = 2; i < kSectionCount; ++i) { aligns[i] = kDeviceAlign; }

    const std::size_t header_bytes =
        header_bytes_for(header.text_plane_count, header.backend_plane_count, header.ladder_count);
    std::size_t cursor = header_bytes;
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        if (lengths[i] == 0) { continue; }
        cursor           = align_up(cursor, aligns[i]);
        header.offset[i] = cursor;
        header.length[i] = lengths[i];
        cursor += lengths[i];
    }

    struct LadderLayout {
        std::uint64_t conv_off   = 0;
        std::uint64_t rec_off    = 0;
        std::uint64_t hidden_off = 0;
        std::uint64_t dflash_off = 0;
        std::uint64_t conv_len   = 0;
        std::uint64_t rec_len    = 0;
        std::uint64_t hidden_len = 0;
        std::uint64_t dflash_len = 0;
    };

    std::vector<LadderLayout> ladder_layout(source.ladder_heads.size());
    for (std::size_t i = 0; i < source.ladder_heads.size(); ++i) {
        const RamLadderHead& head = source.ladder_heads[i];
        auto place                = [&](std::size_t bytes, std::uint64_t& off, std::uint64_t& len) {
            if (bytes == 0) { return; }
            cursor = align_up(cursor, kDeviceAlign);
            off    = cursor;
            len    = bytes;
            cursor += bytes;
        };
        place(head.conv_bytes, ladder_layout[i].conv_off, ladder_layout[i].conv_len);
        place(head.recurrent_bytes, ladder_layout[i].rec_off, ladder_layout[i].rec_len);
        place(head.hidden_bytes, ladder_layout[i].hidden_off, ladder_layout[i].hidden_len);
        place(head.dflash_bytes, ladder_layout[i].dflash_off, ladder_layout[i].dflash_len);
    }
    header.entry_bytes  = align_up(cursor, kDeviceAlign);
    header.header_bytes = header_bytes;

    if (header.entry_bytes > arena_.capacity()) {
        ++drops_;
        bump_version();
        return {RamCaptureStatus::Dropped};
    }

    void* block = nullptr;
    {
        // The arena is shared with the disk worker, whose last I/O unpin frees retired blocks.
        std::lock_guard lock(io_mutex_);
        reap_retired_locked();
        block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    }
    if (block == nullptr) {
        // A claimed restore source cannot be evicted to capture its GPU victim.
        // Check the largest interval that eviction could actually create before
        // asking the caller to discard any other reusable entries. I/O pins and
        // retired blocks are temporary: the reclaim path waits for their spill or fence.
        std::lock_guard lock(io_mutex_);
        std::vector<std::pair<std::size_t, std::size_t>> claimed;
        const auto* base = static_cast<const std::uint8_t*>(arena_.base());
        for (const auto& [id, record] : records_) {
            if (!record.pinned) { continue; }
            const auto offset = static_cast<const std::uint8_t*>(record.block) - base;
            claimed.emplace_back(static_cast<std::size_t>(offset), record.bytes);
        }
        std::sort(claimed.begin(), claimed.end());
        std::size_t begin   = 0;
        std::size_t largest = 0;
        for (const auto& [offset, bytes] : claimed) {
            largest = std::max(largest, offset - begin);
            begin   = offset + bytes;
        }
        largest = std::max(largest, arena_.capacity() - begin);
        if (header.entry_bytes > largest) {
            ++drops_;
            bump_version();
            return {RamCaptureStatus::Dropped};
        }
        return {RamCaptureStatus::NeedsEviction};
    }

    bool copies_launched     = false;
    std::uint64_t live_id    = 0;
    cudaEvent_t copies_start = nullptr;
    try {
        auto* raw = static_cast<std::uint8_t*>(block);
        std::memset(raw, 0, header_bytes);
        Cursor w{raw, raw + header_bytes};
        write_fixed_header(w, header);
        for (std::uint32_t i = 0; i < header.text_plane_count; ++i) {
            write_fingerprint(w, source.text_pool->plane(i), source.text_pool->plane_order());
        }
        if (source.backend_pool != nullptr) {
            for (std::uint32_t i = 0; i < header.backend_plane_count; ++i) {
                write_fingerprint(w, source.backend_pool->plane(i),
                                  source.backend_pool->plane_order());
            }
        }
        for (std::size_t i = 0; i < source.ladder_heads.size(); ++i) {
            const RamLadderHead& head  = source.ladder_heads[i];
            const LadderLayout& layout = ladder_layout[i];
            w.u32(head.frontier);
            w.u32(static_cast<std::uint32_t>(head.kind));
            w.u64(head.hash.lo);
            w.u64(head.hash.hi);
            w.u64(layout.conv_off);
            w.u64(layout.rec_off);
            w.u64(layout.hidden_off);
            w.u64(layout.dflash_off);
            w.u64(layout.conv_len);
            w.u64(layout.rec_len);
            w.u64(layout.hidden_len);
            w.u64(layout.dflash_len);
        }

        if (lengths[0] != 0) {
            std::memcpy(raw + header.offset[0], source.ledger.data(), lengths[0]);
        }
        // Rewrite and ladder images are pinned host buffers; they copy by host callbacks on the
        // cache's host-copy stream so a capture during decode does not run ~150 MB memcpys here.
        std::vector<HostCopy> host_copies;
        host_copies.reserve(6 + 4 * source.ladder_heads.size());
        if (lengths[1] != 0) { source.identity->pack(raw + header.offset[1]); }
        const auto start_device_copies = [&] {
            if (copies_start != nullptr) { return; }
            create_copy_event(&copies_start, cudaEventDefault);
            CUDA_CHECK(cudaEventRecord(copies_start, source.stream));
        };
        if (lengths[2] != 0) {
            start_device_copies();
            pack_paged_kv_allocation_to_host(*source.text, *source.text_pool,
                                             raw + header.offset[2], header.text_mapped_pages,
                                             source.stream);
            copies_launched = true;
        }
        if (source.backend != nullptr && lengths[3] != 0) {
            start_device_copies();
            pack_paged_kv_allocation_to_host(*source.backend, *source.backend_pool,
                                             raw + header.offset[3], header.backend_mapped_pages,
                                             source.stream);
            copies_launched = true;
        }
        if (source.gdn != nullptr) {
            if (current_from_host) {
                if (lengths[4] != 0) {
                    host_copies.push_back(
                        {raw + header.offset[4], source.current_state.conv, lengths[4]});
                }
                if (lengths[6] != 0) {
                    host_copies.push_back(
                        {raw + header.offset[6], source.current_state.recurrent, lengths[6]});
                }
            } else if (lengths[4] != 0 || lengths[6] != 0) {
                start_device_copies();
                source.gdn->pack_slot_to_host(source.gdn_current_slot, raw + header.offset[4],
                                              raw + header.offset[6], source.stream);
                copies_launched = true;
            }
            if (lengths[5] != 0) {
                host_copies.push_back(
                    {raw + header.offset[5], source.rewrite_state.conv, lengths[5]});
            }
            if (lengths[7] != 0) {
                host_copies.push_back(
                    {raw + header.offset[7], source.rewrite_state.recurrent, lengths[7]});
            }
        }
        if (source.tail_hidden != nullptr && lengths[8] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[8], source.tail_hidden->data, lengths[8],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }
        if (source.rewrite_checkpoint_hidden != nullptr && lengths[9] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[9],
                                       source.rewrite_checkpoint_hidden->data, lengths[9],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }
        if (source.dflash_local != nullptr && lengths[10] != 0) {
            if (current_from_host) {
                host_copies.push_back(
                    {raw + header.offset[10], source.current_state.dflash, lengths[10]});
            } else {
                start_device_copies();
                source.dflash_local->copy_lane_to_host(source.dflash_lane, raw + header.offset[10],
                                                       source.stream);
                copies_launched = true;
            }
        }
        if (lengths[11] != 0) {
            host_copies.push_back(
                {raw + header.offset[11], source.rewrite_state.dflash, lengths[11]});
        }
        for (std::size_t i = 0; i < source.ladder_heads.size(); ++i) {
            const RamLadderHead& head  = source.ladder_heads[i];
            const LadderLayout& layout = ladder_layout[i];
            if (layout.conv_len != 0) {
                if (head.conv == nullptr) {
                    throw std::invalid_argument("RAM ladder conv image is null");
                }
                host_copies.push_back(
                    {raw + layout.conv_off, head.conv, static_cast<std::size_t>(layout.conv_len)});
            }
            if (layout.rec_len != 0) {
                if (head.recurrent == nullptr) {
                    throw std::invalid_argument("RAM ladder recurrent image is null");
                }
                host_copies.push_back({raw + layout.rec_off, head.recurrent,
                                       static_cast<std::size_t>(layout.rec_len)});
            }
            if (layout.hidden_len != 0) {
                if (head.hidden == nullptr) {
                    throw std::invalid_argument("RAM ladder hidden image is null");
                }
                host_copies.push_back({raw + layout.hidden_off, head.hidden,
                                       static_cast<std::size_t>(layout.hidden_len)});
            }
            if (layout.dflash_len != 0) {
                if (head.dflash == nullptr) {
                    throw std::invalid_argument("RAM ladder DFlash cyclic image is null");
                }
                host_copies.push_back({raw + layout.dflash_off, head.dflash,
                                       static_cast<std::size_t>(layout.dflash_len)});
            }
        }
        if (!host_copies.empty()) {
            start_device_copies();
            // The image copies wait only for each image's own fence, so they overlap the KV and
            // state D2H above; the record fence recorded below on the copy stream covers both.
            std::vector<cudaEvent_t> image_fences;
            image_fences.reserve(2 + source.ladder_heads.size());
            image_fences.push_back(source.rewrite_state.copies_done);
            image_fences.push_back(source.current_state.copies_done);
            for (const RamLadderHead& head : source.ladder_heads) {
                image_fences.push_back(head.copies_done);
            }
            copies_launched = true;
            host_copies_.run(host_copies, image_fences, source.stream);
        }

        if (next_id_ == 0) { throw std::logic_error("RAM cache entry id overflow"); }
        Record record;
        record.id                  = next_id_++;
        record.hash_f              = source.hash_f;
        record.hash_c              = source.hash_c;
        record.hash_c_valid        = source.hash_c_valid;
        record.execution_frontier  = source.execution_frontier;
        record.checkpoint_frontier = source.rewrite_frontier;
        record.checkpoint_valid    = source.rewrite_valid;
        record.checkpoint_path     = source.rewrite_kind == RewriteCheckpointKind::TurnClosure
                                         ? PrefixReusePath::RestoreTurnCheckpoint
                                         : PrefixReusePath::RestoreResponseCheckpoint;
        if (fail_next_capture_metadata_allocation_) {
            fail_next_capture_metadata_allocation_ = false;
            throw std::bad_alloc();
        }
        record.ladders.reserve(source.ladder_heads.size());
        for (const RamLadderHead& head : source.ladder_heads) {
            record.ladders.push_back(
                RamLadderIndex{.frontier = head.frontier, .hash = head.hash, .kind = head.kind});
        }
        record.block         = block;
        record.bytes         = header.entry_bytes;
        record.disk_entry_id = source.disk_entry_id;
        record.copies_start  = copies_start;
        std::lock_guard lock(io_mutex_);
        const auto [it, inserted] = records_.emplace(record.id, record);
        if (!inserted) { throw std::logic_error("RAM cache entry id already exists"); }
        copies_start = nullptr;
        live_id      = record.id;
        fifo_.push_back(record.id);
        record_copies(it->second, source.stream);
        pending_save_ids_.push_back(record.id);
        ++captures_;
        bump_version();
        return {RamCaptureStatus::Captured, record.id};
    } catch (...) {
        if (copies_launched) {
            maybe_copy_sync_hold();
            (void)cudaStreamSynchronize(host_copies_.stream());
            if (source.stream != nullptr) {
                (void)cudaStreamSynchronize(source.stream);
            } else {
                (void)cudaDeviceSynchronize();
            }
        }
        if (copies_start != nullptr) { (void)cudaEventDestroy(copies_start); }
        if (live_id != 0) {
            // The stream is drained, so the block frees now unless a reader already pinned it.
            std::lock_guard lock(io_mutex_);
            retire_locked(live_id);
        } else {
            std::lock_guard lock(io_mutex_);
            arena_.free(block);
        }
        throw;
    }
} catch (const std::bad_alloc&) {

    // Capturing an optional cache image must not fail an admitted request.
    // The inner transaction has already fenced DMA and reclaimed any partial image.
    record_drop();
    return {RamCaptureStatus::Dropped};
}

RamRestoredHost KVRamCache::unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target) {
    HeaderView header;
    std::uint8_t* raw      = nullptr;
    cudaEvent_t prior_done = nullptr;
    {
        std::lock_guard lock(io_mutex_);
        Record& record          = require(entry_id);
        header                  = read_header(record.block, record.bytes);
        raw                     = static_cast<std::uint8_t*>(record.block);
        prior_done              = record.copies_done;
        const auto* fingerprint = raw + kFixedHeader;
        InCursor fp{fingerprint, raw + header.header_bytes};
        if (target.text == nullptr || target.text_pool == nullptr) {
            throw std::invalid_argument("RAM restore target is incomplete");
        }
        verify_pool(fp, *target.text_pool, header.text_plane_count, "text KV");
        if (header.backend_plane_count != 0) {
            if (target.backend == nullptr || target.backend_pool == nullptr) {
                throw std::logic_error("RAM restore is missing the backend pool");
            }
            verify_pool(fp, *target.backend_pool, header.backend_plane_count, "backend KV");
        }
        if (header.has_dflash) {
            if (target.dflash_local == nullptr) {
                throw std::logic_error("RAM restore is missing DFlash cyclic state");
            }
            verify_cyclic(header, *target.dflash_local);
            if (header.length[11] != 0 &&
                header.length[11] != target.dflash_local->lane_host_bytes()) {
                throw std::logic_error("RAM entry DFlash rewrite-checkpoint geometry mismatch");
            }
        }
        if (header.has_gdn) {
            if (target.gdn == nullptr) {
                throw std::logic_error("RAM restore is missing GDN state");
            }
            if (header.gdn_conv_bytes != target.gdn->conv_host_image_bytes() ||
                header.gdn_recurrent_bytes != target.gdn->recurrent_host_image_bytes()) {
                throw std::logic_error("RAM entry GDN geometry mismatch");
            }
        }
        if (target.tail_hidden != nullptr &&
            header.tail_hidden_bytes != target.tail_hidden->bytes()) {
            throw std::logic_error("RAM entry hidden geometry mismatch");
        }
        if (target.rewrite_checkpoint_hidden != nullptr && header.length[9] != 0 &&
            header.length[9] != target.rewrite_checkpoint_hidden->bytes()) {
            throw std::logic_error("RAM entry rewrite-checkpoint hidden geometry mismatch");
        }
        drop_pending_save(entry_id);
    }
    // The block's capture must land before it is read. The scheduler polls copies_ready() while
    // other lanes decode, so this waits only when nothing else can run.
    if (prior_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(prior_done)); }
    double harvested          = 0;
    cudaEvent_t harvest_start = nullptr;
    {
        std::lock_guard lock(io_mutex_);
        Record& record = require(entry_id);
        if (record.copies_timed && record.copies_start != nullptr &&
            record.copies_done != nullptr) {
            harvest_start = record.copies_start;
        }
    }
    if (harvest_start != nullptr && prior_done != nullptr) {
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, harvest_start, prior_done));
        harvested = static_cast<double>(milliseconds) / 1000.0;
    }
    {
        std::lock_guard lock(io_mutex_);
        Record& record = require(entry_id);
        add_unbilled_seconds(record, harvested);
        if (record.copies_start != nullptr) {
            CUDA_CHECK(cudaEventDestroy(record.copies_start));
            record.copies_start = nullptr;
        }
        record.copies_timed = false;
        // Reserve before any H2D is enqueued so publishing the load cannot fail.
        pending_load_ids_.reserve(pending_load_ids_.size() + 1);
        begin_copies(record, target.stream);
        record.copies_are_load = true;
    }
    // Every exit after the first enqueued read re-records copies_done, so the
    // host block stays fenced until the H2D reads that were issued complete.
    try {
        unpack_paged_kv_allocation_from_host(*target.text, *target.text_pool,
                                             raw + header.offset[2], header.text_mapped_pages,
                                             target.text_dst_pages, target.stream);
        if (target.backend != nullptr && header.length[3] != 0) {
            unpack_paged_kv_allocation_from_host(
                *target.backend, *target.backend_pool, raw + header.offset[3],
                header.backend_mapped_pages, target.backend_dst_pages, target.stream);
        }
        const bool context_head =
            is_staged_checkpoint_restore(target.reuse) && target.reuse_base != 0;
        const RamLadderImage* matched_head = nullptr;
        if (context_head) {
            for (const RamLadderImage& image : header.ladder_images) {
                if (image.frontier == target.reuse_base) {
                    matched_head = &image;
                    break;
                }
            }
            if (matched_head == nullptr) {
                throw std::logic_error(
                    "RAM restore is missing the matched context-checkpoint head");
            }
            if (reuse_path_for_context_checkpoint_kind(matched_head->kind) != target.reuse) {
                throw std::logic_error(
                    "RAM restore context-checkpoint kind does not match the reuse path");
            }
            if (target.gdn != nullptr) {
                if (matched_head->conv_bytes != target.gdn->conv_host_image_bytes() ||
                    matched_head->recurrent_bytes != target.gdn->recurrent_host_image_bytes()) {
                    throw std::logic_error("RAM context-checkpoint GDN geometry mismatch");
                }
                target.gdn->unpack_slot_from_host(target.gdn_current_slot, matched_head->conv,
                                                  matched_head->recurrent, target.stream);
            }
            // The restored lane claims a valid tail hidden from the head, so a head without
            // one cannot serve a staged restore.
            if (target.tail_hidden != nullptr && target.tail_hidden->bytes() != 0 &&
                (matched_head->hidden == nullptr || matched_head->hidden_bytes == 0)) {
                throw std::logic_error("RAM context-checkpoint head has no hidden image");
            }
            if (target.tail_hidden != nullptr && matched_head->hidden != nullptr &&
                matched_head->hidden_bytes != 0) {
                if (matched_head->hidden_bytes != target.tail_hidden->bytes()) {
                    throw std::logic_error("RAM context-checkpoint hidden geometry mismatch");
                }
                CUDA_CHECK(cudaMemcpyAsync(target.tail_hidden->data, matched_head->hidden,
                                           matched_head->hidden_bytes, cudaMemcpyHostToDevice,
                                           target.stream));
            }
            if (target.dflash_local != nullptr) {
                if (matched_head->dflash == nullptr || matched_head->dflash_bytes == 0 ||
                    matched_head->dflash_bytes != target.dflash_local->lane_host_bytes()) {
                    throw std::logic_error(
                        "RAM context-checkpoint DFlash cyclic geometry mismatch");
                }
                target.dflash_local->copy_lane_from_host(matched_head->dflash, target.dflash_lane,
                                                         target.stream);
            }
        } else if (target.gdn != nullptr && (header.length[4] != 0 || header.length[6] != 0)) {
            target.gdn->unpack_slot_from_host(target.gdn_current_slot, raw + header.offset[4],
                                              raw + header.offset[6], target.stream);
        }
        const bool unpack_rewrite = !context_head || header.rewrite_frontier <= target.reuse_base;
        std::vector<HostCopy> rewrite_copies;
        rewrite_copies.reserve(3);
        if (unpack_rewrite && target.gdn != nullptr &&
            (header.length[5] != 0 || header.length[7] != 0)) {
            if (target.rewrite_state.conv == nullptr || target.rewrite_state.recurrent == nullptr) {
                throw std::logic_error("RAM restore is missing the rewrite-checkpoint GDN image");
            }
            rewrite_copies.push_back({target.rewrite_state.conv, raw + header.offset[5],
                                      static_cast<std::size_t>(header.length[5])});
            rewrite_copies.push_back({target.rewrite_state.recurrent, raw + header.offset[7],
                                      static_cast<std::size_t>(header.length[7])});
        }
        if (!context_head && target.tail_hidden != nullptr && header.length[8] != 0) {
            CUDA_CHECK(cudaMemcpyAsync(target.tail_hidden->data, raw + header.offset[8],
                                       static_cast<std::size_t>(header.length[8]),
                                       cudaMemcpyHostToDevice, target.stream));
        }
        if (unpack_rewrite && target.rewrite_checkpoint_hidden != nullptr &&
            header.length[9] != 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                target.rewrite_checkpoint_hidden->data, raw + header.offset[9],
                static_cast<std::size_t>(header.length[9]), cudaMemcpyHostToDevice, target.stream));
        }
        if (!context_head && target.dflash_local != nullptr && header.length[10] != 0) {
            target.dflash_local->copy_lane_from_host(raw + header.offset[10], target.dflash_lane,
                                                     target.stream);
        }
        if (unpack_rewrite && target.dflash_local != nullptr && header.length[11] != 0) {
            if (target.rewrite_state.dflash == nullptr) {
                throw std::logic_error(
                    "RAM restore is missing the rewrite-checkpoint DFlash image");
            }
            rewrite_copies.push_back({target.rewrite_state.dflash, raw + header.offset[11],
                                      static_cast<std::size_t>(header.length[11])});
        }
        // After every earlier reader or writer of the lane's image, but not behind the KV and
        // state H2D above: the block's contents completed before prior_done was waited.
        cudaEvent_t image_fence = target.rewrite_state.copies_done;
        host_copies_.run(rewrite_copies, std::span(&image_fence, 1), target.stream);
    } catch (...) {
        std::lock_guard lock(io_mutex_);
        const auto it = records_.find(entry_id);
        if (it != records_.end()) { record_copies(it->second, target.stream); }
        throw;
    }
    {
        std::lock_guard lock(io_mutex_);
        Record& record = require(entry_id);
        record_copies(record, target.stream);
        pending_load_ids_.push_back(entry_id);
    }
    if (fail_next_restore_metadata_allocation_.exchange(false, std::memory_order_acq_rel)) {
        throw std::bad_alloc();
    }
    return host_from_header(raw, header);
}

void KVRamCache::copy_from_entry(std::uint64_t entry_id, std::span<const HostCopy> copies,
                                 std::span<const cudaEvent_t> image_fences) {
    if (copies.empty()) { return; }
    std::lock_guard lock(io_mutex_);
    Record& record = require(entry_id);
    if (record.block_done == nullptr) {
        create_copy_event(&record.block_done, cudaEventDisableTiming | cudaEventBlockingSync);
    }
    // The block's contents completed before unpack_device returned, so the copies need only
    // their destinations' fences.
    host_copies_.run_detached(copies, image_fences, record.block_done);
}

void KVRamCache::test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte) {
    Record& record          = require(entry_id);
    const HeaderView header = read_header(record.block, record.bytes);
    auto* identity_bytes    = section_ptr(record.block, header, 1);
    ResidentPrefixIdentity identity;
    identity.unpack(identity_bytes, static_cast<std::size_t>(header.length[1]));
    identity.test_tamper_content_digest(0, byte);
    identity.pack(identity_bytes);
}

std::size_t KVRamCache::test_pending_copy_count() const noexcept {
    std::lock_guard lock(io_mutex_);
    return pending_save_ids_.size() + pending_load_ids_.size();
}

std::uint32_t KVRamCache::test_io_pins(std::uint64_t entry_id) const {
    std::lock_guard lock(io_mutex_);
    const auto it = records_.find(entry_id);
    return it == records_.end() ? 0 : it->second.io_pins;
}

void KVRamCache::test_hold_copy_sync(bool held) {
    copy_sync_entered_.store(false, std::memory_order_release);
    copy_sync_held_.store(held, std::memory_order_release);
}

bool KVRamCache::test_copy_sync_entered() const {
    return copy_sync_entered_.load(std::memory_order_acquire);
}

} // namespace ninfer::targets::qwen3_6::detail

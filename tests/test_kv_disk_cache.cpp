#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;
namespace fs  = std::filesystem;

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

struct PackedObjectLocation {
    fs::path path;
    std::uint64_t offset = 0;
    std::uint64_t extent = 0;
    std::uint64_t stored = 0;
};

std::optional<PackedObjectLocation> packed_object_location(const fs::path& root,
                                                           q36::detail::DiskObjectKind kind,
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
        const char* dir = kind == q36::detail::DiskObjectKind::Main ? "main" :
                          kind == q36::detail::DiskObjectKind::Backend ? "backend" :
                          kind == q36::detail::DiskObjectKind::State ? "state" :
                          kind == q36::detail::DiskObjectKind::Ledger ? "ledger" : "identity";
        return PackedObjectLocation{root / "packs" / std::to_string(generation) / dir / name,
                                    offset, extent, stored};
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
        if (frame_payload[0] != 1 || frame_payload[1] != 0 || frame_payload[2] != 0 ||
            frame_payload[3] != 0) {
            return std::nullopt;
        }
        std::array<std::uint8_t, 52> payload{};
        std::copy(frame_payload.begin() + 4, frame_payload.end(), payload.begin());
        std::uint64_t seen = 0;
        std::memcpy(&seen, payload.data(), sizeof(seen));
        if (seen != id || payload[8] != static_cast<std::uint8_t>(kind)) { continue; }
        found = location_from_payload(payload);
    }
}

bool packed_read_prefix(const fs::path& root, q36::detail::DiskObjectKind kind,
                        std::uint64_t id, void* dst, std::size_t bytes) {
    const auto loc = packed_object_location(root, kind, id);
    if (!loc || bytes > loc->stored) { return false; }
    std::ifstream in(loc->path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(loc->offset));
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
}

bool packed_write_prefix(const fs::path& root, q36::detail::DiskObjectKind kind,
                         std::uint64_t id, const void* src, std::size_t bytes) {
    const auto loc = packed_object_location(root, kind, id);
    if (!loc || bytes > loc->extent) { return false; }
    const int fd = ::open(loc->path.c_str(), O_WRONLY);
    if (fd < 0) { return false; }
    const ssize_t written = ::pwrite(fd, src, bytes, static_cast<off_t>(loc->offset));
    const bool ok = written == static_cast<ssize_t>(bytes) && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

bool packed_write_at(const fs::path& root, q36::detail::DiskObjectKind kind, std::uint64_t id,
                     std::uint64_t relative, const void* src, std::size_t bytes) {
    const auto loc = packed_object_location(root, kind, id);
    if (!loc || relative > loc->extent || bytes > loc->extent - relative) { return false; }
    const int fd = ::open(loc->path.c_str(), O_WRONLY);
    if (fd < 0) { return false; }
    const ssize_t written = ::pwrite(fd, src, bytes,
                                     static_cast<off_t>(loc->offset + relative));
    const bool ok = written == static_cast<ssize_t>(bytes) && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("test could not write binary fixture"); }
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    if (!out) { throw std::runtime_error("test could not write binary fixture"); }
}

void append_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) { throw std::runtime_error("test could not append binary fixture"); }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) { throw std::runtime_error("test could not append binary fixture"); }
}

using FileSizeSnapshot = std::vector<std::pair<std::string, std::uintmax_t>>;

FileSizeSnapshot file_size_snapshot(const fs::path& root) {
    FileSizeSnapshot out;
    std::error_code ec;
    if (!fs::exists(root, ec)) {
        if (ec) { throw std::runtime_error("test could not inspect fixture tree"); }
        return out;
    }
    for (fs::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file(ec) || ec) { continue; }
        const auto bytes = it->file_size(ec);
        if (ec) { break; }
        out.emplace_back(fs::relative(it->path(), root, ec).string(), bytes);
        if (ec) { break; }
    }
    if (ec) { throw std::runtime_error("test could not snapshot fixture tree"); }
    std::sort(out.begin(), out.end());
    return out;
}

std::uint32_t test_crc32c(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = ~0U;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82F63B78U & static_cast<std::uint32_t>(-(crc & 1U)));
        }
    }
    return ~crc;
}

fs::path active_map_path(const fs::path& root, const char* suffix) {
    const auto packset = read_bytes(root / "PACKSET");
    if (packset.size() != 32) { return {}; }
    std::uint64_t generation = 0;
    std::memcpy(&generation, packset.data() + 12, sizeof(generation));
    return root / "maps" / ("objects-" + std::to_string(generation) + suffix);
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

q36::PreparedPromptData text_prompt(std::vector<ninfer::TokenId> tokens) {
    q36::PreparedPromptData prompt;
    prompt.token_ids = std::move(tokens);
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(3 * prompt.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i) {
            prompt.positions[static_cast<std::size_t>(axis) * prompt.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    return prompt;
}

std::optional<std::uint64_t> capture_or_evict(q36::detail::KVRamCache& cache,
                                              const q36::detail::RamCaptureSource& source) {
    // Test fixtures often seed source tensors through streams other than the capture stream.
    // Establish the production caller's compute->copy ordering contract before capture.
    CUDA_CHECK(cudaDeviceSynchronize());
    for (;;) {
        if (auto id = cache.capture(source)) { return id; }
        const auto victim = cache.peek_oldest_unpinned();
        if (!victim) {
            cache.record_drop();
            return std::nullopt;
        }
        cache.evict_one_unpinned(*victim);
    }
}

void fill_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        unsigned char seed) {
    const auto pages                      = allocation.page_ids();
    const ninfer::PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
                std::memset(host.data() + begin, value, static_cast<std::size_t>(tensor.nb[3]));
            } else {
                for (std::int64_t head = 0; head < tensor.ne[3]; ++head) {
                    const std::size_t begin =
                        static_cast<std::size_t>(pages[i]) * static_cast<std::size_t>(tensor.nb[2]) +
                        static_cast<std::size_t>(head) * static_cast<std::size_t>(tensor.nb[3]);
                    std::memset(host.data() + begin, value, static_cast<std::size_t>(tensor.nb[2]));
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
}

struct TmpDir {
    fs::path path;
    explicit TmpDir(const char* tag) {
        static std::uint32_t seq = 0;
        path = fs::temp_directory_path() / ("ninfer-kv-disk-" + std::to_string(::getpid()) + "-" +
                                            tag + "-" + std::to_string(++seq));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TmpDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TmpDir(const TmpDir&)            = delete;
    TmpDir& operator=(const TmpDir&) = delete;
};

q36::detail::RamCaptureSource make_source(q36::PreparedPromptData& retained,
                                          q36::detail::ResidentPrefixIdentity& identity,
                                          ninfer::PagedKVAllocation& alloc, ninfer::PagedKVPool& pool,
                                          cudaStream_t stream, std::uint32_t execution_frontier) {
    retained.token_types.assign(retained.token_ids.size(), 0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    identity.assign(retained);
    q36::detail::RamCaptureSource source;
    source.execution_frontier = execution_frontier;
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f =
        q36::detail::prefix_hash_at(retained.token_ids, identity, execution_frontier);
    source.text      = &alloc;
    source.text_pool = &pool;
    source.stream    = stream;
    return source;
}

std::uint64_t capture_tokens(q36::detail::KVRamCache& ram, ninfer::PagedKVPool& pool,
                             ninfer::PagedKVAllocation& alloc, ninfer::DeviceContext& ctx,
                             const std::vector<ninfer::TokenId>& prompt_tokens) {
    auto prompt   = text_prompt(prompt_tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream,
                              static_cast<std::uint32_t>(prompt_tokens.size()));
    auto id     = capture_or_evict(ram, source);
    if (!id) { throw std::runtime_error("RAM capture failed"); }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    return *id;
}

std::uint64_t capture_tokens_hidden(q36::detail::KVRamCache& ram, ninfer::PagedKVPool& pool,
                                      ninfer::PagedKVAllocation& alloc, ninfer::DeviceContext& ctx,
                                      const std::vector<ninfer::TokenId>& prompt_tokens,
                                      ninfer::Tensor& hidden) {
    auto prompt   = text_prompt(prompt_tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream,
                               static_cast<std::uint32_t>(prompt_tokens.size()));
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) { throw std::runtime_error("RAM capture failed"); }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    return *id;
}

template <typename Pred>
bool wait_pred(Pred pred, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!pred() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

q36::detail::DiskOpenConfig
disk_config(const fs::path& location, q36::detail::KVRamCache& ram, ninfer::PagedKVPool& text,
            const ninfer::PagedKVPool* backend, ninfer::SpeculativeBackend speculative,
            std::size_t capacity, std::uint32_t max_context,
            ninfer::KvDiskCompress compress = ninfer::KvDiskCompress::Off,
            const ninfer::LinearAttentionStatePool* gdn = nullptr,
            const ninfer::CyclicKVCache* cyclic = nullptr) {
    q36::detail::DiskOpenConfig cfg;
    cfg.location       = location;
    cfg.capacity_bytes = capacity;
    cfg.compress       = compress;
    cfg.max_context    = max_context;
    cfg.ram            = &ram;
    cfg.fingerprint    = q36::detail::make_disk_fingerprint(
        "qwen3.6-27b", "groupwise-int", ninfer::KvCacheStorage::Int8Group64, speculative, text,
        backend, gdn, cyclic);
    cfg.text_pool          = &text;
    cfg.backend_pool       = backend;
    cfg.logical_page_bytes = ninfer::paged_kv_logical_page_bytes(text);
    if (backend != nullptr) {
        cfg.logical_page_bytes =
            std::max(cfg.logical_page_bytes, ninfer::paged_kv_logical_page_bytes(*backend));
    }
    cfg.gdn_staging_bytes = 256;
    if (gdn != nullptr) {
        cfg.gdn_staging_bytes =
            std::max(cfg.gdn_staging_bytes,
                     gdn->conv_host_image_bytes() + gdn->recurrent_host_image_bytes());
    }
    if (cyclic != nullptr) {
        cfg.gdn_staging_bytes = std::max(cfg.gdn_staging_bytes, cyclic->lane_host_bytes());
    }
    return cfg;
}

int wait_restore_bounded(q36::detail::KVDiskCache& disk, ninfer::DeviceContext& ctx,
                         const char* label) {
    std::atomic<bool> done{false};
    std::exception_ptr error;
    std::thread waiter([&] {
        try {
            disk.wait_copies();
        } catch (...) { error = std::current_exception(); }
        done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!done.load()) {
        disk.cancel_restore();
        waiter.join();
        return fail(label);
    }
    waiter.join();
    if (error) { std::rethrow_exception(error); }
    return 0;
}

int claim_and_restore_match(q36::detail::KVDiskCache& disk, ninfer::DeviceContext& ctx,
                             ninfer::PagedKVPool& pool, const q36::detail::DiskMatch& match,
                             std::uint32_t text_dst_pages, const char* claim_fail,
                             const char* hung) {
    if (!disk.claim(match.entry_id, match.hash_f, match.execution_frontier)) {
        return fail(claim_fail);
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(text_dst_pages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = text_dst_pages;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match.entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, hung); rc != 0) {
            disk.release(match.entry_id);
            dest.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match.entry_id);
        dest.release();
        std::cerr << hung << " threw: " << e.what() << '\n';
        return 1;
    }
    disk.release(match.entry_id);
    dest.release();
    return 0;
}

int test_gather_both_orders(ninfer::DeviceContext& ctx) {
    int failures = 0;
    for (auto order : {ninfer::PagedKVPlaneOrder::PageMajor, ninfer::PagedKVPlaneOrder::HeadMajor}) {
        auto plan = plan_paged_cache(6, 6, 2,
                                     {{ninfer::DType::I8, 64, 2}, {ninfer::DType::FP16, 1, 2}},
                                     order);
        ninfer::DeviceArena arena(plan.bytes);
        ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
        auto alloc = pool.reserve(4);
        alloc.materialize_pages(3, ctx.stream);
        fill_logical_pages(pool, alloc, 9);
        ctx.synchronize_all();
        const std::uint32_t pages     = alloc.mapped_page_count();
        const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, pages);
        const std::size_t page_bytes  = ninfer::paged_kv_logical_page_bytes(pool);
        ninfer::PinnedHostBuffer image(image_bytes);
        ninfer::pack_paged_kv_allocation_to_host(alloc, pool, image.data(), ctx.stream);
        ctx.synchronize_all();
        for (std::uint32_t i = 0; i < pages; ++i) {
            ninfer::PinnedHostBuffer packed(page_bytes);
            std::vector<unsigned char> gathered(page_bytes);
            ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, i, packed.data(), ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            ninfer::gather_logical_page_from_host_image(image.data(), pool, pages, i,
                                                        gathered.data());
            if (std::memcmp(packed.data(), gathered.data(), page_bytes) != 0) {
                std::size_t first = 0;
                while (first < page_bytes &&
                       static_cast<const unsigned char*>(packed.data())[first] == gathered[first]) {
                    ++first;
                }
                std::cerr << "gather != pack for order "
                          << (order == ninfer::PagedKVPlaneOrder::PageMajor ? "PageMajor"
                                                                            : "HeadMajor")
                          << " page " << i << " at " << first << " packed="
                          << static_cast<unsigned>(
                                 static_cast<const unsigned char*>(packed.data())[first])
                          << " gathered=" << static_cast<unsigned>(gathered[first]) << '\n';
                ++failures;
            }
        }
        alloc.release();
    }
    return failures;
}

int test_batch_page_unpack(ninfer::DeviceContext& ctx) {
    auto plan = plan_paged_cache(8, 6, 2,
                                 {{ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::FP16, 1, 2},
                                  {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    auto src = pool.reserve(3);
    src.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, src, 19);
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    auto dest = pool.reserve(3);
    dest.materialize_pages(3, ctx.stream);
    for (std::uint32_t i = 0; i < 3; ++i) {
        std::vector<unsigned char> packed(page_bytes);
        ninfer::pack_paged_kv_logical_page_to_host(src, pool, i, packed.data(), ctx.stream);
        ctx.synchronize_all();
        ninfer::unpack_paged_kv_logical_page_from_host(dest, pool, packed.data(), i, ctx.stream);
        ctx.synchronize_all();
        std::vector<unsigned char> got(page_bytes);
        ninfer::pack_paged_kv_logical_page_to_host(dest, pool, i, got.data(), ctx.stream);
        ctx.synchronize_all();
        if (got != packed) {
            src.release();
            dest.release();
            return fail("batch page unpack did not match packed source");
        }
    }
    src.release();
    dest.release();
    return 0;
}

int test_device_scatter_page_unpack(ninfer::DeviceContext& ctx) {
    for (auto order : {ninfer::PagedKVPlaneOrder::PageMajor,
                       ninfer::PagedKVPlaneOrder::HeadMajor}) {
        auto layout = plan_paged_cache(12, 6, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}},
                                       order);
        ninfer::DeviceArena arena(layout.bytes);
        ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout.layout);
        auto src = pool.reserve(3);
        auto dst = pool.reserve(3);
        src.materialize_pages(3, ctx.stream);
        dst.materialize_pages(3, ctx.stream);
        fill_logical_pages(pool, src, 37);
        const auto scatter = ninfer::make_paged_kv_scatter_plan(pool);
        const std::size_t descriptor_bytes =
            scatter.planes.size() * sizeof(ninfer::PagedKVScatterPlane);
        ninfer::DeviceBuffer device_planes(descriptor_bytes);
        ninfer::DeviceBuffer device_page(scatter.page_bytes);
        CUDA_CHECK(cudaMemcpyAsync(device_planes.p, scatter.planes.data(), descriptor_bytes,
                                   cudaMemcpyHostToDevice, ctx.copy_stream));
        for (std::uint32_t logical = 0; logical < 3; ++logical) {
            std::vector<unsigned char> packed(scatter.page_bytes);
            std::vector<unsigned char> got(scatter.page_bytes);
            ninfer::pack_paged_kv_logical_page_to_host(src, pool, logical, packed.data(),
                                                       ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(device_page.p, packed.data(), packed.size(),
                                       cudaMemcpyHostToDevice, ctx.copy_stream));
            ninfer::scatter_paged_kv_logical_page_from_device(
                dst, pool, device_page.p,
                static_cast<const ninfer::PagedKVScatterPlane*>(device_planes.p),
                scatter.planes.size(), scatter.max_plane_bytes, logical, ctx.copy_stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.copy_stream));
            ninfer::pack_paged_kv_logical_page_to_host(dst, pool, logical, got.data(), ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (got != packed) {
                src.release();
                dst.release();
                return fail(order == ninfer::PagedKVPlaneOrder::PageMajor
                                ? "PageMajor device scatter did not match packed bytes"
                                : "HeadMajor device scatter did not match packed bytes");
            }
        }
        src.release();
        dst.release();
    }
    return 0;
}

int test_lock_and_fingerprint(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("lock");
    q36::detail::KVRamCache ram(8ULL << 20);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    {
        q36::detail::KVDiskCache first(cfg);
        bool threw = false;
        try {
            q36::detail::KVDiskCache second(cfg);
        } catch (const std::runtime_error&) { threw = true; }
        if (!threw) { return fail("same-process second KVDiskCache did not throw"); }
    }
    cfg.fingerprint.model_id = "other";
    bool threw               = false;
    try {
        q36::detail::KVDiskCache mismatch(cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("model_id") != std::string::npos;
    }
    if (!threw) { return fail("fingerprint mismatch did not name model_id"); }
    (void)ctx;
    return 0;
}

int test_ram_io_pin_and_capture_id(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache ram(8ULL << 20);
    const auto id = capture_tokens(ram, pool, alloc, ctx, {10, 11, 12, 13});
    if (!ram.peek_oldest_unpinned() || *ram.peek_oldest_unpinned() != id) {
        alloc.release();
        return fail("peek_oldest_unpinned did not return the captured id");
    }
    ram.pin_for_io(id);
    const auto prompt = text_prompt({10, 11, 12, 13});
    if (!ram.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        ram.unpin_for_io(id);
        alloc.release();
        return fail("plan_match hid an I/O-pinned RAM entry");
    }
    ram.claim(id);
    std::atomic<bool> consumed{false};
    std::thread waiter([&] {
        ram.consume(id);
        consumed.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (consumed.load()) {
        waiter.join();
        alloc.release();
        return fail("consume did not wait for I/O pin");
    }
    ram.unpin_for_io(id);
    waiter.join();
    if (!consumed.load()) {
        alloc.release();
        return fail("consume did not finish after unpin");
    }
    alloc.release();
    return 0;
}

int test_flush_reports_progress(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("flush-prog");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 3);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(64, 1);
    std::vector<ninfer::TokenId> tokens_b(64, 2);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> events;
    disk.flush_not_durable_ram(
        [&](std::uint64_t done, std::uint64_t total) { events.push_back({done, total}); });
    if (events.empty() || events.front() != std::pair<std::uint64_t, std::uint64_t>{0, 2} ||
        events.back() != std::pair<std::uint64_t, std::uint64_t>{2, 2}) {
        alloc.release();
        return fail("flush_not_durable_ram did not report 0/2 then 2/2");
    }
    bool saw_one = false;
    for (const auto& event : events) {
        if (event == std::pair<std::uint64_t, std::uint64_t>{1, 2}) { saw_one = true; }
    }
    if (!saw_one) {
        alloc.release();
        return fail("flush_not_durable_ram skipped the mid-spill progress event");
    }
    if (!disk.ram_is_durable(ram_a) || !disk.ram_is_durable(ram_b)) {
        alloc.release();
        return fail("flush_not_durable_ram left a not-durable RAM note");
    }
    alloc.release();
    return 0;
}

int test_spill_match_extend_branch(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("spill");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 3);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));

    std::vector<ninfer::TokenId> aligned(64, 7);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("emergency spill of aligned chat failed");
    }
    const auto prompt_a = text_prompt(aligned);
    const auto match_a  = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!match_a || match_a->reuse_base != 64) {
        alloc.release();
        return fail("disk did not match spilled aligned chat");
    }
    const auto pages_a = disk.test_main_page_ids(match_a->entry_id);
    if (pages_a.size() != 1) {
        alloc.release();
        return fail("aligned F=64 did not store one main page");
    }

    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_b, match_a->entry_id);
    disk.note_ram_resident(ram_b, match_a->entry_id);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("aligned extend spill failed");
    }
    const auto prompt_b = text_prompt(extended);
    const auto match_b  = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b || match_b->entry_id != match_a->entry_id) {
        alloc.release();
        return fail("aligned extend did not refresh the same entry");
    }
    const auto pages_b = disk.test_main_page_ids(match_b->entry_id);
    if (pages_b.size() != 2 || pages_b[0] != pages_a[0]) {
        alloc.release();
        return fail("aligned extend cloned the last full page");
    }

    std::vector<ninfer::TokenId> unaligned(65, 9);
    const auto ram_u = capture_tokens(ram, pool, alloc, ctx, unaligned);
    disk.note_ram_resident(ram_u, 0);
    if (!disk.emergency_spill_ram(ram_u)) {
        alloc.release();
        return fail("unaligned create spill failed");
    }
    const auto prompt_u = text_prompt(unaligned);
    const auto match_u  = disk.plan_match(prompt_u, q36::detail::prefix_hash_chain(prompt_u));
    if (!match_u) {
        alloc.release();
        return fail("unaligned chat did not match");
    }
    const auto pages_u = disk.test_main_page_ids(match_u->entry_id);
    std::vector<ninfer::TokenId> unaligned_ext = unaligned;
    unaligned_ext.push_back(0);
    unaligned_ext.resize(80, 10);
    const auto ram_ue = capture_tokens(ram, pool, alloc, ctx, unaligned_ext);
    ram.set_disk_entry_id(ram_ue, match_u->entry_id);
    disk.note_ram_resident(ram_ue, match_u->entry_id);
    if (!disk.emergency_spill_ram(ram_ue)) {
        alloc.release();
        return fail("unaligned extend spill failed");
    }
    const auto pages_ue = disk.test_main_page_ids(match_u->entry_id);
    if (pages_ue.size() < 2 || pages_ue[0] != pages_u[0] || pages_ue[1] == pages_u[1]) {
        alloc.release();
        return fail("unaligned extend did not clone the partial last page");
    }

    std::vector<ninfer::TokenId> child = aligned;
    child.push_back(99);
    child.resize(70, 11);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
    ram.set_disk_entry_id(ram_c, match_a->entry_id);
    disk.note_ram_resident(ram_c, match_a->entry_id);
    const auto parent_ref = disk.test_object_refcount(pages_b[0]);
    if (!disk.emergency_spill_ram(ram_c)) {
        alloc.release();
        return fail("branch spill failed");
    }
    const auto prompt_c = text_prompt(child);
    const auto match_c  = disk.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c));
    if (!match_c || match_c->entry_id == match_a->entry_id) {
        alloc.release();
        return fail("branch did not create a child entry");
    }
    if (!disk.test_entry_in_index(match_a->entry_id)) {
        alloc.release();
        return fail("branch evicted the parent");
    }
    if (disk.test_object_refcount(pages_b[0]) <= parent_ref) {
        alloc.release();
        return fail("branch did not increment shared page refcount");
    }
    alloc.release();
    return 0;
}

int test_mtp_f1_backend_pages(ninfer::DeviceContext& ctx) {
    auto text_plan = plan_paged_cache(4, 4, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto backend_plan = plan_paged_cache(4, 4, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool backend({backend_arena.base(), backend_arena.capacity()},
                                backend_plan.layout);
    auto text_alloc    = text.reserve(2);
    auto backend_alloc = backend.reserve(2);
    text_alloc.materialize_pages(1, ctx.stream);
    TmpDir dir("mtp");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto prompt   = text_prompt({42});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, text_alloc, text, ctx.copy_stream, 1);
    source.mtp_kv_valid = 0;
    source.backend      = &backend_alloc;
    source.backend_pool = &backend;
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=1 capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, text, &backend, ninfer::SpeculativeBackend::Mtp,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=1 spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=1 disk match failed");
    }
    if (!disk.test_backend_page_ids(match->entry_id).empty()) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=1 stored backend pages");
    }
    text_alloc.release();
    backend_alloc.release();
    return 0;
}

int test_ladders_and_rollback(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("ladders");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier = 3;
    rollback.hash     = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind     = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden   = hidden_host.data();
    rollback.hidden_bytes = 64;
    q36::detail::RamLadderHead l1 = rollback;
    l1.frontier                   = 4;
    l1.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    l1.kind = q36::detail::ContextCheckpointKind::Ladder;
    q36::detail::RamLadderHead l2 = l1;
    l2.frontier                   = 6;
    l2.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    q36::detail::RamLadderHead l3 = l1;
    l3.frontier                   = 5;
    l3.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    source.ladder_heads = {rollback, l1, l2, l3};
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("ladder capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("ladder spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("ladder spill did not match");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.rollback.kind != q36::detail::ContextCheckpointKind::TurnRollback ||
        meta.rollback.frontier != 3) {
        alloc.release();
        return fail("turn-rollback was not stored in the rollback slot");
    }
    if (meta.ladders[0].frontier != 6 || meta.ladders[1].frontier != 5 ||
        meta.ladders[0].kind != q36::detail::ContextCheckpointKind::Ladder ||
        meta.ladders[1].kind != q36::detail::ContextCheckpointKind::Ladder) {
        alloc.release();
        return fail("disk did not keep the two highest-F Ladder heads");
    }
    if (meta.ladders[0].frontier == meta.rollback.frontier ||
        meta.ladders[1].kind == q36::detail::ContextCheckpointKind::TurnRollback) {
        alloc.release();
        return fail("turn-rollback was selected as a ladder");
    }
    alloc.release();
    return 0;
}

int test_dflash_cyclic(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("dflash");
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache cyclic({cyclic_arena.base(), cyclic_arena.capacity()}, cyclic_layout);
    ninfer::CyclicKVCacheLayerView layer = cyclic.layer_view(0);
    std::vector<unsigned char> k_pattern(layer.k.slice(3, 0, 1).bytes(), 0x3c);
    CUDA_CHECK(cudaMemcpy(layer.k.slice(3, 0, 1).data, k_pattern.data(), k_pattern.size(),
                          cudaMemcpyHostToDevice));
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto prompt   = text_prompt({1, 2, 3, 4});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source            = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.dflash_local    = &cyclic;
    source.dflash_lane     = 0;
    q36::detail::RamLadderHead ladder;
    ladder.frontier     = 2;
    ladder.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    ladder.kind         = q36::detail::ContextCheckpointKind::Ladder;
    ladder.dflash       = nullptr;
    ladder.dflash_bytes = cyclic.lane_host_bytes();
    std::vector<unsigned char> cyclic_host(cyclic.lane_host_bytes());
    cyclic.copy_lane_to_host(0, cyclic_host.data(), ctx.stream);
    ctx.synchronize_all();
    ladder.dflash       = cyclic_host.data();
    source.ladder_heads = {ladder};
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("DFlash cyclic capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::DFlash,
                           32ULL << 20, 4096);
    cfg.fingerprint =
        q36::detail::make_disk_fingerprint("qwen3.6-27b", "groupwise-int",
                                           ninfer::KvCacheStorage::Int8Group64,
                                           ninfer::SpeculativeBackend::DFlash, pool, nullptr,
                                           nullptr, &cyclic);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("DFlash cyclic spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("DFlash cyclic match failed");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.ladders[0].dflash_id == 0 || meta.current_cyclic_id == 0) {
        alloc.release();
        return fail("DFlash cyclic missing on kept heads");
    }
    alloc.release();
    return 0;
}

int test_crash_before_meta(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("crash");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t parent = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {1, 2, 3, 4});
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("crash-sim parent spill failed");
        }
        const auto prompt = text_prompt({1, 2, 3, 4});
        const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("crash-sim parent did not match");
        }
        parent = match->entry_id;
        std::vector<ninfer::TokenId> ext{1, 2, 3, 4, 5, 6, 7, 8};
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, ext);
        ram.set_disk_entry_id(ram_b, parent);
        disk.note_ram_resident(ram_b, parent);
        disk.test_arm_crash_before_meta();
        (void)disk.emergency_spill_ram(ram_b);
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto prompt = text_prompt({1, 2, 3, 4});
    const auto match  = reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || match->entry_id != parent || match->reuse_base != 4) {
        alloc.release();
        return fail("crash after objects before meta lost the old generation");
    }
    const auto ext_prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    const auto ext_match =
        reopened.plan_match(ext_prompt, q36::detail::prefix_hash_chain(ext_prompt));
    if (ext_match && ext_match->reuse_base == 8) {
        alloc.release();
        return fail("crash-sim published the new generation");
    }
    alloc.release();
    return 0;
}

int test_skip_too_long_gc(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("skip");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           8ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk(cfg);
        std::vector<ninfer::TokenId> long_chat(80, 1);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, long_chat);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("skip-too-long parent spill failed");
        }
    }
    cfg.max_context = 16;
    q36::detail::KVDiskCache small(cfg);
    if (small.test_skipped_count() == 0) {
        alloc.release();
        return fail("too-long entry was not skipped");
    }
    if (small.snapshot().used_bytes != 0) {
        alloc.release();
        return fail("skipped too-long tree was billed against unique_bytes");
    }
    if (small.plan_match(text_prompt(std::vector<ninfer::TokenId>(80, 1)),
                         q36::detail::prefix_hash_chain(
                             text_prompt(std::vector<ninfer::TokenId>(80, 1))))) {
        alloc.release();
        return fail("skipped too-long entry remained hittable");
    }
    const auto ram_short = capture_tokens(ram, pool, alloc, ctx, {9, 8, 7, 6});
    small.note_ram_resident(ram_short, 0);
    if (!small.emergency_spill_ram(ram_short)) {
        alloc.release();
        return fail("spill after skip-too-long failed");
    }
    alloc.release();
    return 0;
}

int test_zstd_fail_writes_raw(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("zstd");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(256);
    hid.fill(0xab);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt({4, 5, 6, 7});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("zstd-fail capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Zstd);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.test_force_zstd_fail();
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("zstd-fail spill dropped instead of writing raw");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("zstd-fail raw spill did not match");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.current_hidden_id == 0) {
        alloc.release();
        return fail("zstd-fail did not persist hidden");
    }
    unsigned char codec = 0xff;
    if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State,
                            meta.current_hidden_id, &codec, 1) || codec != 0) {
        alloc.release();
        return fail("zstd-fail did not write a raw codec header");
    }
    alloc.release();
    return 0;
}

int test_zstd_reopen_with_compress_off(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("zstd-reopen");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, alloc, 44);
    ninfer::DeviceBuffer hid(256);
    hid.fill(0xcd);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt({4, 5, 6, 7});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("zstd-reopen capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Zstd);
    cfg.hidden_bytes = 256;
    std::uint64_t entry = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("zstd spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("zstd spill did not match");
        }
        entry = match->entry_id;
        const auto meta = disk.test_load_meta(entry);
        unsigned char codec = 0xff;
        if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State,
                                meta.current_hidden_id, &codec, 1) || codec != 1) {
            alloc.release();
            return fail("zstd spill did not write a zstd codec header");
        }
    }
    cfg.compress = ninfer::KvDiskCompress::Off;
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        alloc.release();
        return fail("compress-off reopen lost the zstd generation");
    }
    ninfer::DeviceBuffer hid_out(256);
    hid_out.fill(0);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {256});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    reopened.claim(entry);
    reopened.restore_device(entry, target);
    try {
        if (const int rc = wait_restore_bounded(reopened, ctx, "zstd-reopen restore hung"); rc != 0) {
            reopened.release(entry);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        reopened.release(entry);
        dest.release();
        alloc.release();
        std::cerr << "zstd-reopen restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> got(256);
    CUDA_CHECK(cudaMemcpy(got.data(), hid_out.p, got.size(), cudaMemcpyDeviceToHost));
    const auto bad = std::find_if(got.begin(), got.end(), [](unsigned char c) { return c != 0xcd; });
    if (bad != got.end()) {
        std::cerr << "compress-off hidden mismatch at " << std::distance(got.begin(), bad)
                  << ": got=" << static_cast<unsigned>(*bad) << " expected=205\n";
        reopened.cancel_restore();
        reopened.release(entry);
        dest.release();
        alloc.release();
        return fail("compress-off reopen did not decode zstd hidden");
    }
    reopened.cancel_restore();
    reopened.release(entry);
    dest.release();
    alloc.release();
    return 0;
}

int test_zstd_capacity_uses_physical_bytes(ninfer::DeviceContext& ctx,
                                           ninfer::PagedKVPool& pool) {
    TmpDir dir("zstd-capacity");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    constexpr std::size_t hidden_bytes = 1U << 20;
    ninfer::DeviceBuffer hid(hidden_bytes);
    hid.fill(0x5a);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8,
                          {static_cast<std::int64_t>(hidden_bytes)});
    auto prompt = text_prompt({4, 6, 8, 10});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    const auto ram_id = capture_or_evict(ram, source);
    if (!ram_id) {
        alloc.release();
        return fail("zstd-capacity capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    const std::size_t logical_page = ninfer::paged_kv_logical_page_bytes(pool);
    const std::size_t page_file =
        (logical_page + q36::detail::kDiskPageHeaderBytes +
         q36::detail::kDiskPageIoAlignment - 1) &
        ~(static_cast<std::size_t>(q36::detail::kDiskPageIoAlignment) - 1);
    const std::size_t capacity = page_file + (128U << 10);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           capacity, 4096, ninfer::KvDiskCompress::Zstd);
    cfg.hidden_bytes = hidden_bytes;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*ram_id, 0);
    if (!disk.emergency_spill_ram(*ram_id)) {
        alloc.release();
        return fail("compressible state fitting physical capacity was rejected");
    }
    const auto snapshot = disk.snapshot();
    if (snapshot.used_bytes > capacity || snapshot.used_bytes >= page_file + hidden_bytes) {
        alloc.release();
        return fail("zstd capacity was not charged at physical encoded bytes");
    }
    if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        alloc.release();
        return fail("physically admitted zstd generation was not restorable");
    }
    alloc.release();
    return 0;
}

int test_packset_bootstrap_recovery(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir fresh("pack-bootstrap");
    q36::detail::KVRamCache fresh_ram(8ULL << 20);
    auto fresh_cfg = disk_config(fresh.path, fresh_ram, pool, nullptr,
                                 ninfer::SpeculativeBackend::None, 32ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk(fresh_cfg);
    }
    std::error_code ec;
    fs::remove(fresh.path / "PACKSET", ec);
    if (ec) { return fail("bootstrap could not create a partial PACKSET tree"); }
    try {
        q36::detail::KVDiskCache recovered(fresh_cfg);
        if (!fs::exists(fresh.path / "PACKSET")) {
            return fail("bootstrap recovery did not republish PACKSET");
        }
    } catch (const std::exception& e) {
        std::cerr << "bootstrap recovery threw: " << e.what() << '\n';
        return 1;
    }

    TmpDir partial_map("pack-bootstrap-partial-map");
    q36::detail::KVRamCache partial_map_ram(8ULL << 20);
    auto partial_map_cfg = disk_config(partial_map.path, partial_map_ram, pool, nullptr,
                                       ninfer::SpeculativeBackend::None, 32ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk(partial_map_cfg);
    }
    fs::remove(partial_map.path / "PACKSET", ec);
    if (ec) { return fail("partial map fixture could not remove PACKSET"); }
    const auto partial_base = partial_map.path / "maps" / "objects-1.base";
    append_bytes(partial_base, {0x01});
    const auto partial_before = read_bytes(partial_base);
    bool partial_rejected = false;
    try {
        q36::detail::KVDiskCache should_reject(partial_map_cfg);
    } catch (const std::exception&) {
        partial_rejected = true;
    }
    if (!partial_rejected) { return fail("bootstrap accepted a partial map-only tree"); }
    if (read_bytes(partial_base) != partial_before || fs::exists(partial_map.path / "PACKSET")) {
        return fail("bootstrap mutated a partial map-only tree");
    }

    TmpDir nonempty("pack-bootstrap-nonempty");
    q36::detail::KVRamCache nonempty_ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto nonempty_cfg = disk_config(nonempty.path, nonempty_ram, pool, nullptr,
                                    ninfer::SpeculativeBackend::None, 32ULL << 20, 4096);
    std::uint64_t main_id = 0;
    {
        q36::detail::KVDiskCache disk(nonempty_cfg);
        const auto ram_id = capture_tokens(nonempty_ram, pool, alloc, ctx, {7, 8, 9, 10});
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("bootstrap nonempty fixture spill failed");
        }
        const auto prompt = text_prompt({7, 8, 9, 10});
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match || disk.test_main_page_ids(match->entry_id).empty()) {
            alloc.release();
            return fail("bootstrap nonempty fixture has no indexed main page");
        }
        main_id = disk.test_main_page_ids(match->entry_id).front();
    }
    const auto pack = packed_object_location(
        nonempty.path, q36::detail::DiskObjectKind::Main, main_id);
    if (!pack) {
        alloc.release();
        return fail("bootstrap nonempty fixture has no main pack");
    }
    const auto before = read_bytes(pack->path);
    fs::remove(nonempty.path / "PACKSET", ec);
    if (ec) {
        alloc.release();
        return fail("bootstrap nonempty fixture could not remove PACKSET");
    }
    bool rejected = false;
    try {
        q36::detail::KVDiskCache should_reject(nonempty_cfg);
    } catch (const std::exception&) {
        rejected = true;
    }
    const auto after = read_bytes(pack->path);
    alloc.release();
    if (!rejected) { return fail("bootstrap overwrote a nonempty tree without PACKSET"); }
    if (before != after) { return fail("bootstrap mutated a nonempty pack before rejecting it"); }
    return 0;
}

int test_pack_map_recovery_boundaries(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    const std::vector<ninfer::TokenId> tokens(128, 13);
    auto create_store = [&](const fs::path& path, std::uint32_t batch) {
        q36::detail::KVRamCache ram(32ULL << 20);
        auto alloc = pool.reserve(4);
        alloc.materialize_pages(3, ctx.stream);
        auto cfg = disk_config(path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        cfg.pack_page_batch = batch;
        {
            q36::detail::KVDiskCache disk(cfg);
            const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
            disk.note_ram_resident(ram_id, 0);
            if (!disk.emergency_spill_ram(ram_id)) {
                alloc.release();
                return false;
            }
            disk.wait_idle_and_fsync();
        }
        alloc.release();
        return true;
    };
    auto matches = [&](const fs::path& path) {
        q36::detail::KVRamCache ram(8ULL << 20);
        auto cfg = disk_config(path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        q36::detail::KVDiskCache disk(cfg);
        const auto prompt = text_prompt(tokens);
        return disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt)).has_value();
    };
    auto construction_rejected = [&](const fs::path& path) {
        q36::detail::KVRamCache ram(8ULL << 20);
        auto cfg = disk_config(path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        try {
            q36::detail::KVDiskCache disk(cfg);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    auto first_main_frame = [](const std::vector<std::uint8_t>& bytes) {
        for (std::size_t off = 0; off + 64 <= bytes.size();) {
            std::uint32_t frame = 0;
            std::memcpy(&frame, bytes.data() + off, sizeof(frame));
            if (frame != 60) { break; }
            if (bytes[off + 16] == static_cast<std::uint8_t>(q36::detail::DiskObjectKind::Main)) {
                return std::vector<std::uint8_t>(bytes.begin() + static_cast<std::ptrdiff_t>(off),
                                                 bytes.begin() + static_cast<std::ptrdiff_t>(off + 64));
            }
            off += 64;
        }
        return std::vector<std::uint8_t>{};
    };

    {
        TmpDir dir("map-tail");
        if (!create_store(dir.path, 8)) { return fail("map-tail fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        const auto before = read_bytes(log);
        append_bytes(log, {0xa5, 0, 0, 0});
        if (!matches(dir.path)) { return fail("torn map-log tail hid a valid entry"); }
        if (read_bytes(log).size() != before.size()) {
            return fail("torn map-log tail was not truncated");
        }
    }
    {
        TmpDir dir("map-over-64m");
        if (!create_store(dir.path, 8)) { return fail("large-map fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        auto bytes = read_bytes(log);
        const auto frame = first_main_frame(bytes);
        if (frame.empty()) { return fail("large-map fixture has no main Put"); }
        constexpr std::size_t kFormerHostReadLimit = 64ULL << 20;
        bytes.reserve(kFormerHostReadLimit + frame.size());
        while (bytes.size() <= kFormerHostReadLimit) {
            bytes.insert(bytes.end(), frame.begin(), frame.end());
        }
        write_bytes(log, bytes);
        if (!matches(dir.path)) {
            return fail("valid object map above 64 MiB failed restart audit");
        }
    }
    {
        TmpDir dir("orphan-map-missing-roll");
        if (!create_store(dir.path, 8)) { return fail("orphan-roll fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        auto orphan = first_main_frame(read_bytes(log));
        if (orphan.empty()) { return fail("orphan-roll fixture has no main Put"); }
        std::uint64_t id = 0;
        std::memcpy(&id, orphan.data() + 8, sizeof(id));
        id += 1ULL << 40;
        std::memcpy(orphan.data() + 8, &id, sizeof(id));
        const std::uint32_t missing_segment = 99;
        std::memcpy(orphan.data() + 20, &missing_segment, sizeof(missing_segment));
        const std::uint64_t offset = 0;
        std::memcpy(orphan.data() + 24, &offset, sizeof(offset));
        const std::uint32_t crc = test_crc32c(orphan.data(), 60);
        std::memcpy(orphan.data() + 60, &crc, sizeof(crc));
        append_bytes(log, orphan);
        if (!matches(dir.path)) {
            return fail("unpublished Put to missing rolled segment poisoned startup");
        }
        if (!create_store(dir.path, 8) || !matches(dir.path)) {
            return fail("cache was unusable after missing rolled-segment orphan recovery");
        }
    }
    {
        TmpDir dir("stale-pack-generation");
        if (!create_store(dir.path, 8)) { return fail("stale-generation fixture spill failed"); }
        fs::create_directories(dir.path / "packs" / "999" / "main");
        write_bytes(dir.path / "packs" / "999" / "main" / "00000000.pack", {1, 2, 3});
        write_bytes(dir.path / "maps" / "objects-999.base", {4, 5, 6});
        write_bytes(dir.path / "maps" / "objects-999.log", {7, 8, 9});
        if (!matches(dir.path)) { return fail("stale-generation fixture was not restorable"); }
        if (fs::exists(dir.path / "packs" / "999") ||
            fs::exists(dir.path / "maps" / "objects-999.base") ||
            fs::exists(dir.path / "maps" / "objects-999.log")) {
            return fail("startup left an unpublished KV pack generation");
        }
    }
    {
        TmpDir dir("map-base-crc");
        if (!create_store(dir.path, 8)) { return fail("base-crc fixture spill failed"); }
        const auto path = active_map_path(dir.path, ".base");
        auto bytes = read_bytes(path);
        if (bytes.size() < 4) { return fail("base-crc fixture is too short"); }
        bytes.back() ^= 1U;
        write_bytes(path, bytes);
        if (!construction_rejected(dir.path)) { return fail("bad base CRC was accepted"); }
    }
    {
        TmpDir dir("map-packset-crc");
        if (!create_store(dir.path, 8)) { return fail("PACKSET-crc fixture spill failed"); }
        auto bytes = read_bytes(dir.path / "PACKSET");
        if (bytes.size() < 4) { return fail("PACKSET-crc fixture is too short"); }
        bytes.back() ^= 1U;
        write_bytes(dir.path / "PACKSET", bytes);
        if (!construction_rejected(dir.path)) { return fail("bad PACKSET CRC was accepted"); }
    }
    {
        TmpDir dir("map-log-crc");
        if (!create_store(dir.path, 8)) { return fail("log-crc fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        auto bytes = read_bytes(log);
        if (bytes.size() < 64) { return fail("log-crc fixture has no complete frame"); }
        bytes[60] ^= 1U;
        write_bytes(log, bytes);
        if (!construction_rejected(dir.path)) { return fail("bad map-log CRC was accepted"); }
    }
    {
        TmpDir dir("map-log-type");
        if (!create_store(dir.path, 8)) { return fail("log-type fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        auto bytes = read_bytes(log);
        if (bytes.size() < 64) { return fail("log-type fixture has no complete frame"); }
        bytes[4] = 2;
        const std::uint32_t crc = test_crc32c(bytes.data(), 60);
        std::memcpy(bytes.data() + 60, &crc, sizeof(crc));
        write_bytes(log, bytes);
        if (!construction_rejected(dir.path)) { return fail("unknown map-log type was accepted"); }
    }
    {
        TmpDir dir("map-duplicate");
        if (!create_store(dir.path, 8)) { return fail("duplicate-map fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        const auto frame = first_main_frame(read_bytes(log));
        if (frame.empty()) { return fail("duplicate-map fixture has no main Put"); }
        append_bytes(log, frame);
        if (!matches(dir.path)) { return fail("identical duplicate Put was rejected"); }
        auto conflict = frame;
        std::uint64_t offset = 0;
        std::memcpy(&offset, conflict.data() + 24, sizeof(offset));
        offset += q36::detail::kDiskPageIoAlignment;
        std::memcpy(conflict.data() + 24, &offset, sizeof(offset));
        const std::uint32_t crc = test_crc32c(conflict.data(), 60);
        std::memcpy(conflict.data() + 60, &crc, sizeof(crc));
        append_bytes(log, conflict);
        if (!construction_rejected(dir.path)) { return fail("conflicting duplicate Put was accepted"); }
    }
    {
        TmpDir dir("map-overlap");
        if (!create_store(dir.path, 8)) { return fail("overlap-map fixture spill failed"); }
        const auto log = active_map_path(dir.path, ".log");
        const auto frame = first_main_frame(read_bytes(log));
        if (frame.empty()) { return fail("overlap-map fixture has no main Put"); }
        auto overlap = frame;
        std::uint64_t id = 0;
        std::memcpy(&id, overlap.data() + 8, sizeof(id));
        id += 1ULL << 40;
        std::memcpy(overlap.data() + 8, &id, sizeof(id));
        const std::uint32_t crc = test_crc32c(overlap.data(), 60);
        std::memcpy(overlap.data() + 60, &crc, sizeof(crc));
        append_bytes(log, overlap);
        if (!construction_rejected(dir.path)) { return fail("overlapping distinct map extents were accepted"); }
    }
    return 0;
}

int test_pack_page_batch_variants(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    for (const std::uint32_t batch : {1U, 4U, 8U}) {
        TmpDir dir((std::string("pack-batch-") + std::to_string(batch)).c_str());
        q36::detail::KVRamCache ram(32ULL << 20);
        auto alloc = pool.reserve(4);
        alloc.materialize_pages(3, ctx.stream);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        cfg.pack_page_batch = batch;
        q36::detail::KVDiskCache disk(cfg);
        const std::vector<ninfer::TokenId> tokens(128, static_cast<ninfer::TokenId>(batch + 20));
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("pack page batch variant spill failed");
        }
        const auto prompt = text_prompt(tokens);
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("pack page batch variant was not hittable");
        }
        const auto pages = disk.test_main_page_ids(match->entry_id);
        if (pages.size() < 2) {
            alloc.release();
            return fail("pack page batch variant stored too few pages");
        }
        const auto first = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main,
                                                  pages[0]);
        const auto second = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main,
                                                   pages[1]);
        if (!first || !second || first->path != second->path ||
            second->offset != first->offset + first->extent) {
            alloc.release();
            return fail("pack page batch variant did not preserve adjacent page extents");
        }
        alloc.release();
    }
    return 0;
}

int test_pack_rollover_and_partial_pwritev(ninfer::DeviceContext& ctx,
                                           ninfer::PagedKVPool& pool) {
    constexpr std::uint64_t kSegmentBytes = 1ULL << 30;
    const std::uint64_t payload = ninfer::paged_kv_logical_page_bytes(pool);
    const std::uint64_t extent =
        (q36::detail::kDiskPageHeaderBytes + payload + q36::detail::kDiskPageIoAlignment - 1) &
        ~(static_cast<std::uint64_t>(q36::detail::kDiskPageIoAlignment) - 1);
    auto exercise = [&](const char* name, std::uint64_t tail, std::uint32_t expected_segment,
                        std::uint64_t expected_offset, bool partial) -> int {
        TmpDir dir(name);
        q36::detail::KVRamCache ram(32ULL << 20);
        auto alloc = pool.reserve(2);
        alloc.materialize_pages(1, ctx.stream);
        fill_logical_pages(pool, alloc, partial ? 41 : 57);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        const std::vector<ninfer::TokenId> tokens(64, partial ? 41 : 57);
        {
            q36::detail::KVDiskCache disk(cfg);
            disk.test_set_pack_position(q36::detail::DiskObjectKind::Main, 0, tail);
            if (partial) { disk.test_arm_partial_pwritev(7); }
            const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
            disk.note_ram_resident(ram_id, 0);
            if (!disk.emergency_spill_ram(ram_id)) {
                alloc.release();
                return fail("pack rollover/partial spill failed");
            }
            const auto prompt = text_prompt(tokens);
            const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
            if (!match) {
                alloc.release();
                return fail("pack rollover/partial entry was not hittable");
            }
            const auto pages = disk.test_main_page_ids(match->entry_id);
            const auto loc = pages.empty()
                                 ? std::optional<PackedObjectLocation>{}
                                 : packed_object_location(dir.path,
                                                          q36::detail::DiskObjectKind::Main,
                                                          pages.front());
            if (!loc || loc->offset != expected_offset ||
                loc->path.filename() !=
                    (expected_segment == 0 ? "00000000.pack" : "00000001.pack") ||
                loc->extent != extent) {
                alloc.release();
                return fail("pack rollover chose the wrong segment boundary");
            }
        }
        q36::detail::KVDiskCache reopened(cfg);
        const auto prompt = text_prompt(tokens);
        if (!reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("partial pwritev/rollover record failed restart audit");
        }
        alloc.release();
        return 0;
    };

    if (const int rc = exercise("pack-roll-exact", kSegmentBytes - extent, 0,
                                kSegmentBytes - extent, true);
        rc != 0) {
        return rc;
    }
    return exercise("pack-roll-next", kSegmentBytes - extent +
                                          q36::detail::kDiskPageIoAlignment,
                    1, 0, false);
}

int test_compaction_reclaims_retired_pack_generation(ninfer::DeviceContext& ctx,
                                                     ninfer::PagedKVPool& pool) {
    TmpDir dir("pack-compaction");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    const std::vector<ninfer::TokenId> tokens_a(64, 31);
    const std::vector<ninfer::TokenId> tokens_b(128, 47);
    std::uint64_t entry_b = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("pack-compaction spill A failed");
        }
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("pack-compaction spill B failed");
        }
        const auto match_b =
            disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
        if (!match_b) {
            alloc.release();
            return fail("pack-compaction could not find B before eviction");
        }
        entry_b = match_b->entry_id;
        if (!disk.test_fifo_evict_one()) {
            alloc.release();
            return fail("pack-compaction could not evict A");
        }
    }
    const auto old_packset = read_bytes(dir.path / "PACKSET");
    if (old_packset.size() != 32) {
        alloc.release();
        return fail("pack-compaction PACKSET is missing before reopen");
    }
        std::uint64_t old_generation = 0;
        std::memcpy(&old_generation, old_packset.data() + 12, sizeof(old_generation));
        const auto old_base =
            dir.path / "maps" / ("objects-" + std::to_string(old_generation) + ".base");
        const auto old_log =
            dir.path / "maps" / ("objects-" + std::to_string(old_generation) + ".log");
        {
            q36::detail::KVDiskCache reopened(cfg);
            auto old_generation_lease = reopened.test_hold_active_generation();
        const auto match_b =
            reopened.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
        if (!match_b || match_b->entry_id != entry_b) {
            alloc.release();
            return fail("pack-compaction lost the retained entry on reopen");
        }
        // Reopen leaves the evicted map entry as a zero-ref map record.  The
        // idle maintenance path must compact it away and release the retired
        // generation without losing B.
        reopened.wait_idle_and_fsync();
        const auto new_packset = read_bytes(dir.path / "PACKSET");
        if (new_packset.size() != 32) {
            alloc.release();
            return fail("pack-compaction PACKSET is missing after compaction");
        }
        std::uint64_t new_generation = 0;
        std::memcpy(&new_generation, new_packset.data() + 12, sizeof(new_generation));
        if (new_generation <= old_generation ||
            !fs::exists(dir.path / "packs" / std::to_string(old_generation)) ||
            !fs::exists(old_base) || !fs::exists(old_log)) {
            alloc.release();
            return fail("pack-compaction ignored a held retired-generation pack/map lease");
        }
        const auto retained =
            reopened.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
        if (!retained || retained->entry_id != entry_b) {
            alloc.release();
            return fail("pack-compaction lost the retained entry after rewrite");
        }
        old_generation_lease.reset();
        reopened.wait_idle_and_fsync();
        if (fs::exists(dir.path / "packs" / std::to_string(old_generation)) ||
            fs::exists(old_base) || fs::exists(old_log)) {
            alloc.release();
            return fail("pack-compaction did not reap released generation pack/maps");
        }
    }
    alloc.release();
    return 0;
}

int test_spill_durability_phase_matrix(ninfer::DeviceContext& ctx,
                                       ninfer::PagedKVPool& pool) {
    using Fault = q36::detail::DiskFaultPoint;
    const std::array pre_meta = {
        Fault::AfterRecordWrite, Fault::AfterPackSync, Fault::AfterMapAppend,
        Fault::AfterMapSync, Fault::AfterMetaTmpWrite, Fault::AfterMetaTmpSync,
    };
    const std::array post_rename = {Fault::AfterMetaRename, Fault::AfterEntryDirSync,
                                    Fault::AfterEntryParentSync};
    auto exercise = [&](Fault point, bool published, int ordinal) -> int {
        TmpDir dir((std::string("spill-phase-") + std::to_string(ordinal)).c_str());
        q36::detail::KVRamCache ram(32ULL << 20);
        auto alloc = pool.reserve(2);
        alloc.materialize_pages(1, ctx.stream);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        const std::vector<ninfer::TokenId> tokens(64, 101 + ordinal);
        {
            q36::detail::KVDiskCache disk(cfg);
            const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
            disk.note_ram_resident(ram_id, 0);
            disk.test_arm_fault(point);
            (void)disk.emergency_spill_ram(ram_id);
        }
        q36::detail::KVDiskCache reopened(cfg);
        const auto prompt = text_prompt(tokens);
        const bool matched = static_cast<bool>(
            reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt)));
        if (matched != published) {
            alloc.release();
            return fail("spill durability phase exposed the wrong restart generation");
        }
        alloc.release();
        return 0;
    };
    int ordinal = 0;
    for (Fault point : pre_meta) {
        if (const int rc = exercise(point, false, ordinal++); rc != 0) { return rc; }
    }
    for (Fault point : post_rename) {
        if (const int rc = exercise(point, true, ordinal++); rc != 0) { return rc; }
    }
    return 0;
}

int test_packset_bootstrap_phase_matrix(ninfer::PagedKVPool& pool) {
    using Fault = q36::detail::DiskFaultPoint;
    const std::array points = {Fault::AfterPacksetTmpWrite, Fault::AfterPacksetTmpSync,
                               Fault::AfterPacksetRename, Fault::AfterPacksetRootSync};
    int ordinal = 0;
    for (Fault point : points) {
        TmpDir dir((std::string("packset-phase-") + std::to_string(ordinal++)).c_str());
        q36::detail::KVRamCache ram(8ULL << 20);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               16ULL << 20, 4096);
        cfg.test_fault_point = point;
        bool threw = false;
        try {
            q36::detail::KVDiskCache failed(cfg);
        } catch (...) {
            threw = true;
        }
        if (!threw) { return fail("PACKSET bootstrap phase fault did not interrupt publication"); }
        cfg.test_fault_point = Fault::None;
        try {
            q36::detail::KVDiskCache reopened(cfg);
        } catch (const std::exception& e) {
            std::cerr << "PACKSET bootstrap recovery threw: " << e.what() << '\n';
            return fail("PACKSET bootstrap phase was not restart recoverable");
        }
    }
    return 0;
}

int test_compaction_publication_phase_matrix(ninfer::DeviceContext& ctx,
                                             ninfer::PagedKVPool& pool) {
    using Fault = q36::detail::DiskFaultPoint;
    const std::array points = {
        Fault::CompactionAfterPackCopy, Fault::CompactionAfterPackSync,
        Fault::CompactionAfterBaseWrite, Fault::CompactionAfterBaseSync,
        Fault::CompactionAfterLogWrite, Fault::CompactionAfterLogSync,
        Fault::CompactionAfterPacksetTmpWrite, Fault::CompactionAfterPacksetTmpSync,
        Fault::CompactionAfterPacksetRename, Fault::CompactionAfterRootSync,
    };
    int ordinal = 0;
    for (Fault point : points) {
        TmpDir dir((std::string("compact-phase-") + std::to_string(ordinal)).c_str());
        q36::detail::KVRamCache ram(48ULL << 20);
        auto alloc = pool.reserve(3);
        alloc.materialize_pages(2, ctx.stream);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               64ULL << 20, 4096);
        const std::vector<ninfer::TokenId> dropped(64, 121 + ordinal);
        const std::vector<ninfer::TokenId> retained(128, 141 + ordinal);
        {
            q36::detail::KVDiskCache disk(cfg);
            for (const auto* tokens : {&dropped, &retained}) {
                const auto ram_id = capture_tokens(ram, pool, alloc, ctx, *tokens);
                disk.note_ram_resident(ram_id, 0);
                if (!disk.emergency_spill_ram(ram_id)) {
                    alloc.release();
                    return fail("compaction phase fixture spill failed");
                }
            }
            if (!disk.test_fifo_evict_one()) {
                alloc.release();
                return fail("compaction phase fixture eviction failed");
            }
        }
        {
            q36::detail::KVDiskCache disk(cfg);
            const auto packset = read_bytes(dir.path / "PACKSET");
            if (packset.size() != 32) {
                alloc.release();
                return fail("compaction phase PACKSET fixture is invalid");
            }
            std::uint64_t old_generation = 0;
            std::memcpy(&old_generation, packset.data() + 12, sizeof(old_generation));
            const auto old_base =
                dir.path / "maps" / ("objects-" + std::to_string(old_generation) + ".base");
            const auto old_log =
                dir.path / "maps" / ("objects-" + std::to_string(old_generation) + ".log");
            if (point == Fault::CompactionAfterPacksetRename) {
                disk.test_arm_fault_sequence(point,
                    Fault::CompactionRecoveryBeforeRootSync);
            } else {
                disk.test_arm_fault(point);
            }
            disk.wait_idle_and_fsync();
            if (point == Fault::CompactionAfterPacksetRename) {
                const auto old_root = dir.path / "packs" / std::to_string(old_generation);
                if (!fs::exists(old_root) || !fs::exists(old_base) || !fs::exists(old_log)) {
                    alloc.release();
                    return fail("uncertain PACKSET publication reaped old generation pack/maps");
                }
                disk.wait_idle_and_fsync();
                if (fs::exists(old_root) || fs::exists(old_base) || fs::exists(old_log)) {
                    alloc.release();
                    return fail("durable PACKSET retry did not reap old generation pack/maps");
                }
            }
        }
        try {
            q36::detail::KVDiskCache reopened(cfg);
            const auto prompt = text_prompt(retained);
            if (!reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
                alloc.release();
                return fail("compaction phase lost the retained restart generation");
            }
            reopened.wait_idle_and_fsync();
        } catch (const std::exception& e) {
            std::cerr << "compaction phase " << ordinal << " reopen threw: " << e.what() << '\n';
            alloc.release();
            return fail("compaction phase left an unreadable published generation");
        }
        ++ordinal;
        alloc.release();
    }
    return 0;
}

int test_tombstone_durability_phase_matrix(ninfer::DeviceContext& ctx,
                                           ninfer::PagedKVPool& pool) {
    using Fault = q36::detail::DiskFaultPoint;
    const std::array points = {Fault::AfterTombstoneTmpWrite, Fault::AfterTombstoneTmpSync,
                               Fault::AfterTombstoneRename, Fault::AfterTombstoneDirSync};
    int ordinal = 0;
    for (Fault point : points) {
        TmpDir dir((std::string("tombstone-phase-") + std::to_string(ordinal)).c_str());
        q36::detail::KVRamCache ram(24ULL << 20);
        auto alloc = pool.reserve(2);
        alloc.materialize_pages(1, ctx.stream);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        const std::vector<ninfer::TokenId> tokens(64, 171 + ordinal);
        {
            q36::detail::KVDiskCache disk(cfg);
            const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
            disk.note_ram_resident(ram_id, 0);
            if (!disk.emergency_spill_ram(ram_id)) {
                alloc.release();
                return fail("tombstone phase fixture spill failed");
            }
            disk.test_arm_fault(point);
            (void)disk.test_fifo_evict_one();
        }
        q36::detail::KVDiskCache reopened(cfg);
        const auto prompt = text_prompt(tokens);
        const bool matched = static_cast<bool>(
            reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt)));
        const bool tombstone_published =
            point == Fault::AfterTombstoneRename || point == Fault::AfterTombstoneDirSync;
        if (matched == tombstone_published) {
            alloc.release();
            return fail("tombstone durability phase exposed the wrong restart generation");
        }
        ++ordinal;
        alloc.release();
    }
    return 0;
}

int test_physical_preflight_rejects_before_pack_write(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("pack-preflight");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    const std::vector<ninfer::TokenId> tokens{71, 72, 73, 74};
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    const auto before = read_bytes(active_map_path(dir.path, ".log"));
    disk.test_set_free_bytes_override(0);
    if (disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("pack-preflight admitted a spill with no physical room");
    }
    const auto after = read_bytes(active_map_path(dir.path, ".log"));
    if (before != after || disk.snapshot().used_bytes != 0 ||
        disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("pack-preflight wrote or published data before physical admission");
    }
    alloc.release();
    return 0;
}

int test_low_free_subthreshold_garbage_rejects_without_write(
    ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("pack-low-free-garbage");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    std::vector<std::vector<ninfer::TokenId>> retained;
    const auto total_bytes = [](const auto& snapshot) {
        std::uint64_t total = 0;
        for (const auto& [path, bytes] : snapshot) {
            (void)path;
            total += bytes;
        }
        return total;
    };
    std::uint64_t before_last_spill = 0;
    for (ninfer::TokenId value = 81; value < 86; ++value) {
        if (value == 85) {
            before_last_spill = total_bytes(file_size_snapshot(dir.path / "packs"));
        }
        retained.emplace_back(64, value);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, retained.back());
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("low-free fixture spill failed");
        }
    }
    const std::uint64_t append_bytes =
        total_bytes(file_size_snapshot(dir.path / "packs")) - before_last_spill;
    if (append_bytes == 0) {
        alloc.release();
        return fail("low-free fixture could not measure append bytes");
    }
    if (!disk.test_fifo_evict_one()) {
        alloc.release();
        return fail("low-free fixture could not create sub-threshold garbage");
    }
    const auto before_packs = file_size_snapshot(dir.path / "packs");
    const auto before_maps = file_size_snapshot(dir.path / "maps");
    const auto before_packset = read_bytes(dir.path / "PACKSET");
    const std::vector<ninfer::TokenId> rejected(64, 99);
    const auto rejected_ram = capture_tokens(ram, pool, alloc, ctx, rejected);
    disk.note_ram_resident(rejected_ram, 0);
    constexpr std::uint64_t kSegmentSlack = 1ULL << 30;
    disk.test_set_free_bytes_override(append_bytes + kSegmentSlack - 1);
    if (disk.emergency_spill_ram(rejected_ram)) {
        alloc.release();
        return fail("low-free sub-threshold-garbage spill was admitted");
    }
    if (file_size_snapshot(dir.path / "packs") != before_packs ||
        file_size_snapshot(dir.path / "maps") != before_maps ||
        read_bytes(dir.path / "PACKSET") != before_packset) {
        alloc.release();
        return fail("low-free refusal wrote or published packed state");
    }
    for (std::size_t i = 1; i < retained.size(); ++i) {
        const auto prompt = text_prompt(retained[i]);
        if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("low-free refusal lost an existing retained entry");
        }
    }
    const auto rejected_prompt = text_prompt(rejected);
    if (disk.plan_match(rejected_prompt, q36::detail::prefix_hash_chain(rejected_prompt))) {
        alloc.release();
        return fail("low-free refusal published the rejected entry");
    }
    const std::vector<ninfer::TokenId> accepted(64, 100);
    const auto accepted_ram = capture_tokens(ram, pool, alloc, ctx, accepted);
    disk.note_ram_resident(accepted_ram, 0);
    disk.test_set_free_bytes_override(append_bytes + kSegmentSlack);
    if (!disk.emergency_spill_ram(accepted_ram)) {
        alloc.release();
        return fail("low-free exact append-plus-slack boundary was rejected");
    }
    const auto accepted_prompt = text_prompt(accepted);
    if (!disk.plan_match(accepted_prompt, q36::detail::prefix_hash_chain(accepted_prompt))) {
        alloc.release();
        return fail("low-free exact boundary spill was not published");
    }
    alloc.release();
    return 0;
}

int test_restore_pread_fail(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("pread");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("pread-fail spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("pread-fail match failed");
    }
    const auto pages = disk.test_main_page_ids(match->entry_id);
    disk.claim(match->entry_id);
    disk.test_break_object(pages.front(), q36::detail::DiskObjectKind::Main);
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    bool threw = false;
    try {
        disk.wait_copies();
    } catch (const std::runtime_error&) { threw = true; }
    if (!threw) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("broken object restore did not fail");
    }
    if (disk.copies_ready()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("failed restore reported copies_ready");
    }
    disk.cancel_restore();
    if (!disk.copies_ready()) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("cancel_restore left copies_ready false");
    }
    disk.release(match->entry_id);
    if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        dest.release();
        alloc.release();
        return fail("restore pread fail consumed the pin");
    }
    dest.release();
    alloc.release();
    return 0;
}

int test_cancelled_restore_job_does_not_poison_next(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-job");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("stale-job spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("stale-job spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("stale-job did not persist two distinct entries");
    }

    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;

    auto wait_dequeued = [&](const char* label) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!disk.test_restore_job_dequeued() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_restore_job_dequeued()) { return fail(label); }
        return 0;
    };

    if (!disk.claim(match_a->entry_id)) {
        dest.release();
        alloc.release();
        return fail("stale-job claim A failed");
    }
    disk.test_arm_restore_job_barrier();
    disk.restore_device(match_a->entry_id, target);
    if (const int rc = wait_dequeued("stale-job A RestoreRead was not dequeued"); rc != 0) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return rc;
    }
    const auto pages_a = disk.test_main_page_ids(match_a->entry_id);
    if (pages_a.empty()) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return fail("stale-job A stored no main pages");
    }
    disk.test_break_object(pages_a.front(), q36::detail::DiskObjectKind::Main);
    disk.cancel_restore();
    disk.release(match_a->entry_id);

    if (!disk.claim(match_b->entry_id)) {
        dest.release();
        alloc.release();
        return fail("stale-job claim B failed");
    }
    const auto drops_before = disk.snapshot().drops;
    disk.restore_device(match_b->entry_id, target);
    disk.test_release_restore_job_barrier();
    try {
        if (const int rc =
                wait_restore_bounded(disk, ctx, "stale-job different-entry restore hung");
            rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-job different-entry restore threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.snapshot().drops != drops_before) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("stale RestoreRead failure poisoned a different-entry restore");
    }
    disk.cancel_restore();

    disk.test_arm_restore_job_barrier();
    disk.restore_device(match_b->entry_id, target);
    if (const int rc = wait_dequeued("stale-job same-entry RestoreRead was not dequeued");
        rc != 0) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return rc;
    }
    disk.cancel_restore();
    const auto drops_same = disk.snapshot().drops;
    disk.restore_device(match_b->entry_id, target);
    disk.test_release_restore_job_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "stale-job same-entry restore hung");
            rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-job same-entry restore threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.snapshot().drops != drops_same) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("cancelled RestoreRead poisoned a same-entry restore");
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    if (!disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b))) {
        dest.release();
        alloc.release();
        return fail("stale-job restore consumed the inclusive B entry");
    }
    dest.release();
    alloc.release();
    return 0;
}

int test_stale_state_load_does_not_publish_wrong_head(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-state");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid_cur(64);
    hid_cur.fill(0xaa);
    ninfer::Tensor hidden_cur(hid_cur.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden_cur;
    std::vector<unsigned char> rollback_host(64, 0xbb);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = rollback_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("stale-state capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_state_barrier(); }
    } barrier{disk};
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("stale-state spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("stale-state exact match failed");
    }
    auto prompt3 = text_prompt({3, 3, 3});
    const auto rb =
        disk.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
    if (!rb || rb->reuse_base != 3 ||
        rb->reuse != ninfer::PrefixReusePath::RestoreTurnRollback) {
        alloc.release();
        return fail("stale-state did not advertise the rollback head");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("stale-state claim failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    hid_out.fill(0);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget current;
    current.text           = &dest;
    current.text_pool      = &pool;
    current.text_dst_pages = 1;
    current.tail_hidden    = &hidden_out;
    current.stream         = ctx.copy_stream;
    disk.test_arm_restore_state_barrier();
    disk.restore_device(match->entry_id, current);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_state_entered() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_state_entered()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("stale-state load did not enter the I/O window");
    }
    disk.cancel_restore();
    hid_out.fill(0);
    q36::detail::DiskRestoreTarget rollback_tgt;
    rollback_tgt.text           = &dest;
    rollback_tgt.text_pool      = &pool;
    rollback_tgt.text_dst_pages = 1;
    rollback_tgt.tail_hidden    = &hidden_out;
    rollback_tgt.reuse          = rb->reuse;
    rollback_tgt.reuse_base     = rb->reuse_base;
    rollback_tgt.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, rollback_tgt);
    disk.test_release_restore_state_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "stale-state rollback restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-state rollback restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> got(64);
    CUDA_CHECK(cudaMemcpy(got.data(), hid_out.p, got.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(got.begin(), got.end(), [](unsigned char c) { return c != 0xbb; })) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("stale current-state load published into a rollback restore");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_stale_state_failure_does_not_poison_next(ninfer::DeviceContext& ctx,
                                                  ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-fail");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_state_barrier(); }
    } barrier{disk};

    ninfer::DeviceBuffer hid_a(64);
    hid_a.fill(0x11);
    ninfer::Tensor hidden_a(hid_a.p, ninfer::DType::U8, {64});
    auto prompt_a = text_prompt({2, 3, 4, 5});
    auto retained_a = prompt_a;
    retained_a.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_a;
    auto source_a = make_source(retained_a, identity_a, alloc, pool, ctx.copy_stream, 4);
    source_a.tail_hidden = &hidden_a;
    auto ram_a = capture_or_evict(ram, source_a);
    if (!ram_a) {
        alloc.release();
        return fail("stale-fail capture A failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*ram_a, 0);
    if (!disk.emergency_spill_ram(*ram_a)) {
        alloc.release();
        return fail("stale-fail spill A failed");
    }

    ninfer::DeviceBuffer hid_b(64);
    hid_b.fill(0x22);
    ninfer::Tensor hidden_b(hid_b.p, ninfer::DType::U8, {64});
    auto prompt_b = text_prompt({6, 7, 8, 9});
    auto retained_b = prompt_b;
    retained_b.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_b;
    auto source_b = make_source(retained_b, identity_b, alloc, pool, ctx.copy_stream, 4);
    source_b.tail_hidden = &hidden_b;
    auto ram_b = capture_or_evict(ram, source_b);
    if (!ram_b) {
        alloc.release();
        return fail("stale-fail capture B failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*ram_b, 0);
    if (!disk.emergency_spill_ram(*ram_b)) {
        alloc.release();
        return fail("stale-fail spill B failed");
    }

    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("stale-fail did not persist two entries");
    }
    const auto hidden_id = disk.test_load_meta(match_a->entry_id).current_hidden_id;
    if (hidden_id == 0) {
        alloc.release();
        return fail("stale-fail stored no current hidden");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("stale-fail claim A failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    disk.test_reset_restore_loop_idle_spins();
    disk.test_arm_restore_state_barrier();
    disk.restore_device(match_a->entry_id, target);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_state_entered() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_state_entered()) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return fail("stale-fail state load did not enter the I/O window");
    }
    disk.test_break_object(hidden_id, q36::detail::DiskObjectKind::State);
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    if (!disk.claim(match_b->entry_id)) {
        dest.release();
        alloc.release();
        return fail("stale-fail claim B failed");
    }
    const auto drops_before = disk.snapshot().drops;
    disk.restore_device(match_b->entry_id, target);
    disk.test_release_restore_state_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "stale-fail B restore hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-fail B restore threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.snapshot().drops != drops_before) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("stale state-load failure poisoned a different-entry restore");
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_copy_event_lease_survives_next_restore(ninfer::DeviceContext& ctx,
                                               ninfer::PagedKVPool& pool) {
    TmpDir dir("copy-lease");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_copy_lease_barrier(); }
    } barrier{disk};

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("copy-lease spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("copy-lease spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a   = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b   = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("copy-lease did not persist two entries");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("copy-lease claim A failed");
    }
    auto dest_a = pool.reserve(2);
    dest_a.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target_a;
    target_a.text           = &dest_a;
    target_a.text_pool      = &pool;
    target_a.text_dst_pages = 1;
    target_a.stream         = ctx.copy_stream;
    disk.test_arm_copy_lease_barrier();
    disk.restore_device(match_a->entry_id, target_a);
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        try {
            disk.wait_copies();
        } catch (...) { wait_error = std::current_exception(); }
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.test_copy_event_leased() && std::chrono::steady_clock::now() < deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.test_copy_event_leased()) {
        disk.test_release_copy_lease_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("copy-lease waiter did not lease a completion event");
    }
    if (disk.test_retired_copy_events() < 1) {
        disk.test_release_copy_lease_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("copy-lease close did not retire the generation event");
    }
    if (!disk.claim(match_b->entry_id)) {
        disk.test_release_copy_lease_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("copy-lease claim B failed");
    }
    auto dest_b = pool.reserve(2);
    dest_b.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target_b;
    target_b.text           = &dest_b;
    target_b.text_pool      = &pool;
    target_b.text_dst_pages = 1;
    target_b.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target_b);
    const auto live_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    cudaEvent_t live = nullptr;
    std::size_t retired = disk.test_retired_copy_events();
    while (std::chrono::steady_clock::now() < live_deadline) {
        disk.pump_restore(ctx.copy_stream);
        live    = disk.test_copies_done();
        retired = disk.test_retired_copy_events();
        if (live != nullptr || retired >= 2) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const bool rotated = (live != nullptr && retired >= 1) || retired >= 2;
    disk.test_release_copy_lease_barrier();
    waiter.join();
    if (wait_error) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        disk.release(match_b->entry_id);
        dest_a.release();
        dest_b.release();
        alloc.release();
        std::rethrow_exception(wait_error);
    }
    if (!rotated) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        disk.release(match_b->entry_id);
        dest_a.release();
        dest_b.release();
        alloc.release();
        return fail("next restore reused the leased completion event");
    }
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "copy-lease B restore hung"); rc != 0) {
            disk.release(match_a->entry_id);
            disk.release(match_b->entry_id);
            dest_a.release();
            dest_b.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        disk.release(match_b->entry_id);
        dest_a.release();
        dest_b.release();
        alloc.release();
        std::cerr << "copy-lease B restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    disk.release(match_b->entry_id);
    dest_a.release();
    dest_b.release();
    alloc.release();
    return 0;
}

int test_wait_copies_does_not_follow_replacement_generation(ninfer::DeviceContext& ctx,
                                                       ninfer::PagedKVPool& pool) {
    TmpDir dir("wait-epoch");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() {
            disk.test_release_restore_job_barrier();
            disk.test_release_wait_epoch_barrier();
        }
    } barrier{disk};

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("wait-epoch spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("wait-epoch spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a   = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b   = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("wait-epoch did not persist two entries");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("wait-epoch claim A failed");
    }
    auto dest_a = pool.reserve(2);
    dest_a.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target_a;
    target_a.text           = &dest_a;
    target_a.text_pool      = &pool;
    target_a.text_dst_pages = 1;
    target_a.stream         = ctx.copy_stream;
    disk.test_arm_restore_job_barrier();
    disk.test_arm_wait_epoch_barrier();
    disk.restore_device(match_a->entry_id, target_a);
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        try {
            disk.wait_copies();
        } catch (...) { wait_error = std::current_exception(); }
    });
    const auto latch_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.test_wait_epoch_latched() && std::chrono::steady_clock::now() < latch_deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.test_wait_epoch_latched()) {
        disk.test_release_restore_job_barrier();
        disk.test_release_wait_epoch_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("wait-epoch waiter did not latch restore A");
    }
    disk.test_release_restore_job_barrier();
    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.copies_ready() && !disk.restore_failed() &&
           std::chrono::steady_clock::now() < ready_deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.copies_ready()) {
        disk.test_release_wait_epoch_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("wait-epoch restore A did not complete while waiter was latched");
    }
    if (!disk.claim(match_b->entry_id)) {
        disk.test_release_wait_epoch_barrier();
        disk.cancel_restore();
        waiter.join();
        disk.release(match_a->entry_id);
        dest_a.release();
        alloc.release();
        return fail("wait-epoch claim B failed");
    }
    auto dest_b = pool.reserve(2);
    dest_b.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target_b;
    target_b.text           = &dest_b;
    target_b.text_pool      = &pool;
    target_b.text_dst_pages = 1;
    target_b.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target_b);
    disk.cancel_restore();
    disk.test_release_wait_epoch_barrier();
    waiter.join();
    if (wait_error) {
        disk.release(match_a->entry_id);
        disk.release(match_b->entry_id);
        dest_a.release();
        dest_b.release();
        alloc.release();
        std::rethrow_exception(wait_error);
    }
    if (disk.test_last_wait_copy_event() == nullptr) {
        disk.release(match_a->entry_id);
        disk.release(match_b->entry_id);
        dest_a.release();
        dest_b.release();
        alloc.release();
        return fail("wait-epoch waiter leased no event after B was canceled");
    }
    disk.release(match_a->entry_id);
    disk.release(match_b->entry_id);
    dest_a.release();
    dest_b.release();
    alloc.release();
    return 0;
}

int test_disk_load_timing_survives_session_close(ninfer::DeviceContext& ctx,
                                                ninfer::PagedKVPool& pool) {
    TmpDir dir("load-timing");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("load-timing spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("load-timing match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("load-timing claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "load-timing restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "load-timing restore threw: " << e.what() << '\n';
        return 1;
    }
    const auto copies = disk.harvest_copy_seconds();
    if (copies.load <= 0.0 && disk.test_timing_harvests() == 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("disk load timing was lost at session close");
    }
    if (copies.h2d <= 0.0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("disk H2D timing was not harvested after wait_copies");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_load_includes_pread(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("load-pread");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("pread-load spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("pread-load match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("pread-load claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.test_set_payload_io_stall_ms(80);
    const auto restore_started = std::chrono::steady_clock::now();
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "pread-load restore hung"); rc != 0) {
            disk.test_set_payload_io_stall_ms(0);
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.test_set_payload_io_stall_ms(0);
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "pread-load restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.test_set_payload_io_stall_ms(0);
    const double restore_wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - restore_started).count();
    const auto copies = disk.harvest_copy_seconds();
    if (copies.load < 0.06) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "disk load omitted pread wait, load=" << copies.load << "s\n";
        return 1;
    }
    if (copies.load > restore_wall + 0.05) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "disk load exceeded restore wall, load=" << copies.load
                  << "s wall=" << restore_wall << "s\n";
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_load_is_ssd_to_host_wall(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("load-host-wall");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(6);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(192, 13);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("host-wall spill failed");
    }
    (void)disk.harvest_copy_seconds();
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || disk.test_main_page_ids(match->entry_id).size() < 3) {
        alloc.release();
        return fail("host-wall spill did not store three main pages");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("host-wall claim failed");
    }
    auto dest = pool.reserve(6);
    dest.materialize_pages(3, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 3;
    target.stream         = ctx.copy_stream;
    disk.test_set_payload_io_stall_ms(40);
    const auto restore_started = std::chrono::steady_clock::now();
    disk.restore_device(match->entry_id, target);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.copies_ready()) {
        if (disk.restore_failed() || std::chrono::steady_clock::now() > deadline) {
            disk.test_set_payload_io_stall_ms(0);
            disk.cancel_restore();
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return fail("host-wall restore did not become copy-ready");
        }
        disk.pump_restore(ctx.copy_stream);
    }
    const double restore_wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - restore_started).count();
    try {
        disk.wait_copies();
    } catch (const std::exception& e) {
        disk.test_set_payload_io_stall_ms(0);
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "host-wall wait_copies threw: " << e.what() << '\n';
        return 1;
    }
    disk.test_set_payload_io_stall_ms(0);
    const auto copies = disk.harvest_copy_seconds();
    if (copies.load < 0.10) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "host-wall load omitted overlapped page reads, load=" << copies.load << "s\n";
        return 1;
    }
    if (copies.load > restore_wall + 0.02) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "disk load is not an SSD-to-host wall, load=" << copies.load
                  << "s ready_wall=" << restore_wall << "s\n";
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_h2d_not_billed_until_wait_copies(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("h2d-wait");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("h2d-wait spill failed");
    }
    (void)disk.harvest_copy_seconds();
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("h2d-wait match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("h2d-wait claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.copies_ready()) {
        if (disk.restore_failed() || std::chrono::steady_clock::now() > deadline) {
            disk.cancel_restore();
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return fail("h2d-wait restore did not become copy-ready");
        }
        disk.pump_restore(ctx.copy_stream);
    }
    const auto before = disk.harvest_copy_seconds();
    if (before.h2d != 0.0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("disk H2D timing was harvested before wait_copies");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_abandoned_prefetch_does_not_bill_load(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("prefetch-bill");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("prefetch-bill spill failed");
    }
    (void)disk.harvest_copy_seconds();
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("prefetch-bill match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("prefetch-bill claim failed");
    }
    disk.prefetch_window(match->entry_id, 1, 0);
    const auto live_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    double billed            = 0.0;
    while (std::chrono::steady_clock::now() < live_deadline) {
        billed += disk.harvest_copy_seconds().load;
        if (billed > 0.0) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (billed <= 0.0) {
        disk.release(match->entry_id);
        alloc.release();
        return fail("claimed prefetch did not bill SSD read");
    }
    disk.release(match->entry_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    const auto extra = disk.harvest_copy_seconds();
    if (extra.load > 0.0) {
        alloc.release();
        return fail("released prefetch billed SSD read onto pending load");
    }
    alloc.release();
    return 0;
}

int test_cancelled_state_read_does_not_bill_load(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("state-bill");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid_cur(64);
    hid_cur.fill(0xaa);
    ninfer::Tensor hidden_cur(hid_cur.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden_cur;
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("state-bill capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() {
            disk.test_set_state_decode_stall_ms(0);
            disk.test_release_restore_state_barrier();
        }
    } barrier{disk};
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("state-bill spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("state-bill match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("state-bill claim failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    (void)disk.harvest_copy_seconds();
    disk.test_arm_restore_state_barrier();
    disk.restore_device(match->entry_id, target);
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_state_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_state_entered()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("state-bill state load did not enter the I/O window");
    }
    (void)disk.harvest_copy_seconds();
    disk.cancel_restore();
    (void)disk.harvest_copy_seconds();
    disk.test_set_state_decode_stall_ms(80);
    disk.test_release_restore_state_barrier();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const auto extra = disk.harvest_copy_seconds();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (extra.load > 0.0) {
        return fail("cancelled state read billed SSD time onto pending load");
    }
    return 0;
}

int test_wait_copies_injected_failure_is_not_pread(ninfer::PagedKVPool& pool) {
    TmpDir dir("wait-fail");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.test_arm_fail_wait_copies();
    try {
        disk.wait_copies();
        return fail("injected wait_copies did not throw");
    } catch (const std::logic_error&) {
    } catch (const std::exception& e) {
        std::cerr << "injected wait_copies threw " << e.what() << '\n';
        return 1;
    }
    if (disk.restore_failed()) {
        return fail("injected wait_copies set restore_failed_");
    }
    return 0;
}

int test_wait_copies_after_close_leases_ticket(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("wait-ticket");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("wait-ticket spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("wait-ticket match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("wait-ticket claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.copies_ready() && !disk.restore_failed() &&
           std::chrono::steady_clock::now() < deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.copies_ready()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("wait-ticket restore did not close before wait_copies");
    }
    try {
        disk.wait_copies();
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "wait-ticket wait_copies threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.test_last_wait_copy_event() == nullptr) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("wait_copies after session close leased no completion event");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_released_prefetch_does_not_poison_restore(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("prefetch-release");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("prefetch-release spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("prefetch-release spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a   = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b   = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("prefetch-release did not persist two entries");
    }
    if (!disk.claim(match_b->entry_id)) {
        alloc.release();
        return fail("prefetch-release claim B failed");
    }
    disk.test_arm_restore_job_barrier();
    disk.prefetch_window(match_b->entry_id, 1, 0);
    const auto deq_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_job_dequeued() && std::chrono::steady_clock::now() < deq_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_job_dequeued()) {
        disk.test_release_restore_job_barrier();
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("prefetch-release job did not dequeue");
    }
    if (disk.test_disk_io_pins(match_b->entry_id) == 0) {
        disk.test_release_restore_job_barrier();
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("prefetch-release did not hold an I/O pin");
    }
    disk.release(match_b->entry_id);
    disk.test_release_restore_job_barrier();
    const auto pin_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (disk.test_disk_io_pins(match_b->entry_id) != 0 &&
           std::chrono::steady_clock::now() < pin_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (disk.test_disk_io_pins(match_b->entry_id) != 0) {
        alloc.release();
        return fail("prefetch-release left a stuck I/O pin");
    }
    if (disk.restore_failed()) {
        alloc.release();
        return fail("abandoned prefetch published restore_failed_");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("prefetch-release claim A failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_a->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "prefetch-release A restore hung");
            rc != 0) {
            disk.release(match_a->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "prefetch-release A restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_stale_prefetch_does_not_survive_inplace_extend(ninfer::DeviceContext& ctx,
                                                     ninfer::PagedKVPool& pool) {
    TmpDir dir("prefetch-extend");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 11);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> parent_tokens(63, 4);
    const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent_tokens);
    disk.note_ram_resident(ram_p, 0);
    if (!disk.emergency_spill_ram(ram_p)) {
        alloc.release();
        return fail("prefetch-extend parent spill failed");
    }
    const auto match_p = disk.plan_match(text_prompt(parent_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(parent_tokens)));
    if (!match_p || match_p->reuse_base != 63) {
        alloc.release();
        return fail("prefetch-extend parent did not match at frontier 63");
    }
    const std::uint64_t entry_id = match_p->entry_id;
    if (disk.test_committed_generation(entry_id) != 1) {
        alloc.release();
        return fail("prefetch-extend parent committed generation was not 1");
    }
    if (!disk.claim(entry_id)) {
        alloc.release();
        return fail("prefetch-extend parent claim failed");
    }
    disk.prefetch_window(entry_id, 1, 0);
    const auto fill_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.test_window_filled_for(entry_id) &&
           std::chrono::steady_clock::now() < fill_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.test_window_filled_for(entry_id)) {
        disk.release(entry_id);
        alloc.release();
        return fail("prefetch-extend did not fill a window slot");
    }
    disk.release(entry_id);
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> child_tokens = parent_tokens;
    child_tokens.push_back(0);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child_tokens);
    ram.set_disk_entry_id(ram_c, entry_id);
    disk.note_ram_resident(ram_c, entry_id);
    if (!disk.emergency_spill_ram(ram_c)) {
        alloc.release();
        return fail("prefetch-extend in-place spill failed");
    }
    const auto match_c = disk.plan_match(text_prompt(child_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(child_tokens)));
    if (!match_c || match_c->entry_id != entry_id) {
        alloc.release();
        return fail("prefetch-extend did not refresh the same disk entry");
    }
    if (disk.test_committed_generation(entry_id) != 2) {
        alloc.release();
        return fail("prefetch-extend did not advance committed generation");
    }
    if (disk.test_window_filled_for(entry_id)) {
        alloc.release();
        return fail("prefetch-extend left a filled slot from the prior generation");
    }
    if (!disk.claim(entry_id)) {
        alloc.release();
        return fail("prefetch-extend reclaim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "prefetch-extend restore hung");
            rc != 0) {
            disk.release(entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(entry_id);
        dest.release();
        alloc.release();
        std::cerr << "prefetch-extend restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<unsigned char> src(page_bytes);
    std::vector<unsigned char> dst(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, 0, src.data(), ctx.stream);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, dst.data(), ctx.stream);
    ctx.synchronize_all();
    if (src != dst) {
        disk.cancel_restore();
        disk.release(entry_id);
        dest.release();
        alloc.release();
        return fail("prefetch-extend restore published stale pre-extend KV");
    }
    disk.cancel_restore();
    disk.release(entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_abandoned_prefetch_failure_does_not_poison_other_restore(
    ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("prefetch-fail-other");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_page_read_barrier(); }
    } barrier{disk};

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("prefetch-fail-other spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("prefetch-fail-other spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a   = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b   = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("prefetch-fail-other did not persist two entries");
    }
    if (!disk.claim(match_b->entry_id)) {
        alloc.release();
        return fail("prefetch-fail-other claim B failed");
    }
    disk.test_arm_page_read_barrier();
    disk.prefetch_window(match_b->entry_id, 1, 0);
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_page_read_entered() && std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_page_read_entered()) {
        disk.test_release_page_read_barrier();
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("prefetch-fail-other did not enter page read");
    }
    disk.release(match_b->entry_id);
    disk.test_arm_fail_page_read();
    disk.test_release_page_read_barrier();
    const auto pin_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (disk.test_disk_io_pins(match_b->entry_id) != 0 &&
           std::chrono::steady_clock::now() < pin_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (disk.test_disk_io_pins(match_b->entry_id) != 0) {
        alloc.release();
        return fail("prefetch-fail-other left a stuck I/O pin");
    }
    if (disk.restore_failed()) {
        alloc.release();
        return fail("abandoned prefetch failure published restore_failed_");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("prefetch-fail-other claim A failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_a->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "prefetch-fail-other A restore hung");
            rc != 0) {
            disk.release(match_a->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "prefetch-fail-other A restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_ticket_survives_host_wait_until_stream_wait(ninfer::DeviceContext& ctx,
                                                             ninfer::PagedKVPool& pool) {
    TmpDir dir("ticket-stream");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("ticket-stream spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("ticket-stream match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("ticket-stream claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    const std::uint64_t ticket = disk.restore_device(match->entry_id, target);
    if (ticket == 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("ticket-stream restore returned no epoch");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!disk.copies_ready() && !disk.restore_failed() &&
           std::chrono::steady_clock::now() < deadline) {
        disk.pump_restore(ctx.copy_stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!disk.copies_ready()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("ticket-stream restore did not close before wait_copies");
    }
    try {
        disk.wait_copies(ticket);
        disk.wait_copies_on_stream(ctx.stream, ticket);
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "ticket-stream wait threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.test_last_wait_copy_event() == nullptr) {
        disk.release_restore_ticket(ticket);
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("wait_copies_on_stream after wait_copies leased no generation event");
    }
    if (disk.test_retired_copy_events() == 0 && disk.test_copies_done() == nullptr) {
        disk.release_restore_ticket(ticket);
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("generation event was destroyed before the restore ticket was released");
    }
    disk.release_restore_ticket(ticket);
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_state_setup_failure_does_not_spin(ninfer::DeviceContext& ctx,
                                                     ninfer::PagedKVPool& pool) {
    TmpDir dir("state-setup-fail");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("state-setup-fail spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("state-setup-fail match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("state-setup-fail claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    const auto drops_before = disk.snapshot().drops;
    disk.test_arm_fail_restore_state_setup();
    disk.restore_device(match->entry_id, target);
    bool threw = false;
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "state-setup-fail restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw || !disk.restore_failed()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("state-setup-fail did not publish a restore failure");
    }
    if (disk.snapshot().drops <= drops_before) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("state-setup-fail did not account a drop");
    }
    disk.cancel_restore();

    const auto post_spawn_drops = disk.snapshot().drops;
    disk.test_arm_fail_after_checkpoint_prepare_start();
    disk.restore_device(match->entry_id, target);
    threw = false;
    try {
        if (const int rc = wait_restore_bounded(
                disk, ctx, "post-checkpoint-prepare failure deadlocked restore ownership");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw || !disk.restore_failed() || disk.snapshot().drops <= post_spawn_drops) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("post-checkpoint-prepare failure was not published without deadlock");
    }
    disk.cancel_restore();

    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(
                disk, ctx, "restore after checkpoint-prepare failure hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "restore after checkpoint-prepare failure threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_cancel_before_h2d_releases_ticket_event(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("cancel-no-h2d");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};

    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("cancel-no-h2d spill failed");
    }
    const auto prompt = text_prompt({2, 3, 4, 5});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("cancel-no-h2d match failed");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("cancel-no-h2d claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.test_arm_restore_job_barrier();
    const std::uint64_t ticket = disk.restore_device(match->entry_id, target);
    if (ticket == 0) {
        disk.test_release_restore_job_barrier();
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("cancel-no-h2d restore returned no epoch");
    }
    const auto deq_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_job_dequeued() && std::chrono::steady_clock::now() < deq_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    disk.cancel_restore();
    disk.test_release_restore_job_barrier();
    disk.release_restore_ticket(ticket);
    if (disk.test_copies_done() != nullptr || disk.test_retired_copy_events() != 0) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("cancel before H2D leaked a completion event after ticket release");
    }
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_stale_state_io_aborts_between_objects(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-decode");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() {
            disk.test_set_state_decode_stall_ms(0);
            disk.test_release_restore_state_barrier();
        }
    } barrier{disk};

    ninfer::DeviceBuffer hid_a(64);
    hid_a.fill(0x11);
    ninfer::Tensor hidden_a(hid_a.p, ninfer::DType::U8, {64});
    auto prompt_a   = text_prompt({2, 3, 4, 5, 6, 7, 8, 1});
    auto retained_a = prompt_a;
    retained_a.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_a;
    auto source_a = make_source(retained_a, identity_a, alloc, pool, ctx.copy_stream, 8);
    source_a.tail_hidden = &hidden_a;
    std::vector<unsigned char> hid_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained_a.token_ids, identity_a, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hid_host.data();
    rollback.hidden_bytes = 64;
    q36::detail::RamLadderHead l1 = rollback;
    l1.frontier                   = 6;
    l1.hash = q36::detail::prefix_hash_at(retained_a.token_ids, identity_a, 6);
    l1.kind = q36::detail::ContextCheckpointKind::Ladder;
    q36::detail::RamLadderHead l2 = rollback;
    l2.frontier                   = 5;
    l2.hash = q36::detail::prefix_hash_at(retained_a.token_ids, identity_a, 5);
    l2.kind = q36::detail::ContextCheckpointKind::Ladder;
    source_a.ladder_heads = {rollback, l1, l2};
    auto ram_a = capture_or_evict(ram, source_a);
    if (!ram_a) {
        alloc.release();
        return fail("stale-decode capture A failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*ram_a, 0);
    if (!disk.emergency_spill_ram(*ram_a)) {
        alloc.release();
        return fail("stale-decode spill A failed");
    }

    ninfer::DeviceBuffer hid_b(64);
    hid_b.fill(0x22);
    ninfer::Tensor hidden_b(hid_b.p, ninfer::DType::U8, {64});
    auto prompt_b   = text_prompt({9, 8, 7, 6});
    auto retained_b = prompt_b;
    retained_b.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_b;
    auto source_b = make_source(retained_b, identity_b, alloc, pool, ctx.copy_stream, 4);
    source_b.tail_hidden = &hidden_b;
    auto ram_b = capture_or_evict(ram, source_b);
    if (!ram_b) {
        alloc.release();
        return fail("stale-decode capture B failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*ram_b, 0);
    if (!disk.emergency_spill_ram(*ram_b)) {
        alloc.release();
        return fail("stale-decode spill B failed");
    }

    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("stale-decode did not persist two entries");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("stale-decode claim A failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    disk.test_set_state_decode_stall_ms(120);
    disk.test_arm_restore_state_barrier();
    disk.restore_device(match_a->entry_id, target);
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_state_entered() && std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_state_entered()) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return fail("stale-decode state load did not enter the I/O window");
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    if (!disk.claim(match_b->entry_id)) {
        dest.release();
        alloc.release();
        return fail("stale-decode claim B failed");
    }
    disk.restore_device(match_b->entry_id, target);
    disk.test_release_restore_state_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "stale-decode B restore hung");
            rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-decode B restore threw: " << e.what() << '\n';
        return 1;
    }
    const int decodes = disk.test_state_decode_count();
    disk.test_set_state_decode_stall_ms(0);
    if (decodes > 3) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-decode kept the I/O thread on canceled objects: " << decodes << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_page_read_throw_does_not_stick_inflight(ninfer::DeviceContext& ctx,
                                                ninfer::PagedKVPool& pool) {
    TmpDir dir("page-throw");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));

    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {2, 3, 4, 5});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("page-throw spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {6, 7, 8, 9});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("page-throw spill B failed");
    }
    const auto prompt_a = text_prompt({2, 3, 4, 5});
    const auto prompt_b = text_prompt({6, 7, 8, 9});
    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("page-throw did not persist two entries");
    }
    if (!disk.claim(match_a->entry_id)) {
        alloc.release();
        return fail("page-throw claim A failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.test_arm_fail_page_read();
    disk.restore_device(match_a->entry_id, target);
    bool threw = false;
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "page-throw A restore hung"); rc != 0) {
            disk.release(match_a->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception&) { threw = true; }
    if (!threw) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return fail("injected page-read throw did not fail restore");
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    if (!disk.claim(match_b->entry_id)) {
        dest.release();
        alloc.release();
        return fail("page-throw claim B failed");
    }
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "page-throw B restore hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "page-throw B restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_capture_returns_id(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache ram(8ULL << 20);
    const auto first = capture_tokens(ram, pool, alloc, ctx, {1, 1, 1, 1});
    if (!ram.peek_oldest_unpinned() || *ram.peek_oldest_unpinned() != first) {
        alloc.release();
        return fail("capture did not return a peekable id");
    }
    alloc.release();
    return 0;
}

int test_wait_copies_idle_does_not_hang(ninfer::PagedKVPool& pool) {
    TmpDir dir("idle-wait");
    q36::detail::KVRamCache ram(8ULL << 20);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           16ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    if (!disk.copies_ready()) { return fail("idle disk copies_ready was false"); }
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        disk.wait_copies();
        done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!done.load()) {
        waiter.detach();
        return fail("idle wait_copies hung");
    }
    waiter.join();
    return 0;
}

int test_restore_three_pages_and_prefetch(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("restore3");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(6);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(192, 13);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("three-page spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || disk.test_main_page_ids(match->entry_id).size() < 3) {
        alloc.release();
        return fail("three-page spill did not store three main pages");
    }
    disk.claim(match->entry_id);
    auto dest = pool.reserve(6);
    dest.materialize_pages(3, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 3;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "restore of three pages hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "three-page restore threw: " << e.what() << '\n';
        return 1;
    }
    {
        const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
        for (std::uint32_t i = 0; i < 3; ++i) {
            std::vector<unsigned char> src(page_bytes);
            std::vector<unsigned char> dst(page_bytes);
            ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, i, src.data(), ctx.stream);
            ninfer::pack_paged_kv_logical_page_to_host(dest, pool, i, dst.data(), ctx.stream);
            ctx.synchronize_all();
            if (src != dst) {
                disk.cancel_restore();
                disk.release(match->entry_id);
                dest.release();
                alloc.release();
                return fail("three-page restore did not round-trip KV payload");
            }
        }
    }
    disk.cancel_restore();
    disk.prefetch_window(match->entry_id, 1, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    target.text_dst_pages = 1;
    dest.release();
    dest = pool.reserve(6);
    dest.materialize_pages(1, ctx.stream);
    target.text = &dest;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc =
                wait_restore_bounded(disk, ctx, "prefetch-covered restore hung without RestoreRead");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "prefetch restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    dest.release();
    dest = pool.reserve(6);
    dest.materialize_pages(1, ctx.stream);
    target.text           = &dest;
    target.text_dst_pages = 1;
    disk.restore_device(match->entry_id, target);
    disk.cancel_restore();
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "restore after cancel_restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "cancel-then-restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_completion_waits_for_final_scatter(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("scatter-join");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto source = pool.reserve(2);
    source.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, source, 73);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, source, ctx, {7, 3, 7, 3});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        source.release();
        return fail("scatter-join spill failed");
    }
    const auto prompt = text_prompt({7, 3, 7, 3});
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || !disk.claim(match->entry_id)) {
        source.release();
        return fail("scatter-join claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.test_arm_scatter_record_barrier(0);
    disk.restore_device(match->entry_id, target);
    std::atomic<bool> waiter_done{false};
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        try {
            disk.wait_copies();
        } catch (...) {
            wait_error = std::current_exception();
        }
        waiter_done.store(true, std::memory_order_release);
    });
    if (!wait_pred([&] { return disk.test_scatter_record_entered(); },
                   std::chrono::seconds(2))) {
        disk.test_release_scatter_record_barrier();
        waiter.join();
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        source.release();
        return fail("final scatter never reached its completion-record barrier");
    }
    cudaEvent_t copies_done = nullptr;
    if (!wait_pred(
            [&] {
                copies_done = disk.test_copies_done();
                return copies_done != nullptr;
            },
            std::chrono::seconds(2))) {
        disk.test_release_scatter_record_barrier();
        waiter.join();
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        source.release();
        return fail("final scatter did not form the aggregate completion event");
    }
    if (cudaEventQuery(copies_done) != cudaErrorNotReady) {
        disk.test_release_scatter_record_barrier();
        waiter.join();
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        source.release();
        return fail("aggregate completion event ignored the gated scatter stream");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (waiter_done.load(std::memory_order_acquire) || disk.copies_ready()) {
        disk.test_release_scatter_record_barrier();
        waiter.join();
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        source.release();
        return fail("restore completed before the final scatter event was recorded");
    }
    disk.test_release_scatter_record_barrier();
    waiter.join();
    if (wait_error) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        source.release();
        std::rethrow_exception(wait_error);
    }
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    ninfer::PinnedHostBuffer expected(page_bytes);
    ninfer::PinnedHostBuffer actual(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(source, pool, 0, expected.data(), ctx.stream);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, actual.data(), ctx.stream);
    ctx.synchronize_all();
    const bool equal = std::memcmp(expected.data(), actual.data(), page_bytes) == 0;
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    source.release();
    return equal ? 0 : fail("final-scatter completion published a corrupt KV page");
}

int test_extend_preserves_disk_claim(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-extend");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> aligned(64, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("claim-extend parent spill failed");
    }
    const auto match_a = disk.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match_a) {
        alloc.release();
        return fail("claim-extend parent match failed");
    }
    disk.claim(match_a->entry_id);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.resize(128, 5);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_b, match_a->entry_id);
    disk.note_ram_resident(ram_b, match_a->entry_id);
    if (!disk.emergency_spill_ram(ram_b)) {
        disk.release(match_a->entry_id);
        alloc.release();
        return fail("claim-extend child spill failed");
    }
    try {
        disk.release(match_a->entry_id);
    } catch (const std::logic_error&) {
        alloc.release();
        return fail("extend wiped a live disk claim");
    }
    alloc.release();
    return 0;
}

int test_claimed_generation_not_extended(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-gen");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 11);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> aligned(64, 4);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("claimed-gen spill of B failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match_b || match_b->reuse_base != 64) {
        alloc.release();
        return fail("claimed-gen did not match B");
    }
    const auto pages_b    = disk.test_main_page_ids(match_b->entry_id);
    const auto frontier_b = disk.test_load_meta(match_b->entry_id).execution_frontier;
    disk.claim(match_b->entry_id);
    disk.prefetch_window(match_b->entry_id, 1, 0);
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_c, match_b->entry_id);
    disk.note_ram_resident(ram_c, match_b->entry_id);
    if (!disk.emergency_spill_ram(ram_c)) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("claimed-gen emergency spill of C failed");
    }
    if (disk.test_load_meta(match_b->entry_id).execution_frontier != frontier_b) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("claimed disk generation was extended under restore");
    }
    const auto pages_after = disk.test_main_page_ids(match_b->entry_id);
    if (pages_after != pages_b) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("claimed disk page list changed under restore");
    }
    const auto match_c =
        disk.plan_match(text_prompt(extended), q36::detail::prefix_hash_chain(text_prompt(extended)));
    if (!match_c || match_c->entry_id == match_b->entry_id) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("spill of claimed parent did not branch a child");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "claimed-gen restore of B hung");
            rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (...) {
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("claimed-gen restore of B threw");
    }
    if (disk.test_load_meta(match_b->entry_id).execution_frontier != frontier_b) {
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("restore saw a mutated claimed generation");
    }
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_claimed_parent_branch_clones_partial_page(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-e63");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 11);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> parent_tokens(63, 4);
    const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent_tokens);
    disk.note_ram_resident(ram_p, 0);
    if (!disk.emergency_spill_ram(ram_p)) {
        alloc.release();
        return fail("claimed-E63 parent spill failed");
    }
    const auto match_p = disk.plan_match(text_prompt(parent_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(parent_tokens)));
    if (!match_p || match_p->reuse_base != 63) {
        alloc.release();
        return fail("claimed-E63 parent did not match at frontier 63");
    }
    if (disk.test_load_meta(match_p->entry_id).execution_frontier != 63) {
        alloc.release();
        return fail("claimed-E63 parent execution frontier was not 63");
    }
    const auto pages_p = disk.test_main_page_ids(match_p->entry_id);
    if (pages_p.size() != 1) {
        alloc.release();
        return fail("claimed-E63 parent did not store one main page");
    }
    if (!disk.claim(match_p->entry_id)) {
        alloc.release();
        return fail("claimed-E63 parent claim failed");
    }
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> child_tokens = parent_tokens;
    child_tokens.push_back(0);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child_tokens);
    ram.set_disk_entry_id(ram_c, match_p->entry_id);
    disk.note_ram_resident(ram_c, match_p->entry_id);
    if (!disk.emergency_spill_ram(ram_c)) {
        disk.release(match_p->entry_id);
        alloc.release();
        return fail("claimed-E63 child spill failed");
    }
    const auto match_c = disk.plan_match(text_prompt(child_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(child_tokens)));
    if (!match_c || match_c->entry_id == match_p->entry_id) {
        disk.release(match_p->entry_id);
        alloc.release();
        return fail("claimed-E63 spill did not branch a child");
    }
    const auto pages_c = disk.test_main_page_ids(match_c->entry_id);
    if (pages_c.size() != 1 || pages_c[0] == pages_p[0]) {
        disk.release(match_p->entry_id);
        alloc.release();
        return fail("claimed-E63 branch reused the parent's stale last-page KV");
    }
    if (disk.test_main_page_ids(match_p->entry_id) != pages_p) {
        disk.release(match_p->entry_id);
        alloc.release();
        return fail("claimed-E63 branch mutated the parent page list");
    }
    if (!disk.claim(match_c->entry_id)) {
        disk.release(match_p->entry_id);
        alloc.release();
        return fail("claimed-E63 child claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_c->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "claimed-E63 child restore hung");
            rc != 0) {
            disk.release(match_c->entry_id);
            disk.release(match_p->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_c->entry_id);
        disk.release(match_p->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "claimed-E63 child restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<unsigned char> src(page_bytes);
    std::vector<unsigned char> dst(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, 0, src.data(), ctx.stream);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, dst.data(), ctx.stream);
    ctx.synchronize_all();
    if (src != dst) {
        disk.cancel_restore();
        disk.release(match_c->entry_id);
        disk.release(match_p->entry_id);
        dest.release();
        alloc.release();
        return fail("claimed-E63 restore published parent KV at the executed child token");
    }
    disk.cancel_restore();
    disk.release(match_c->entry_id);
    disk.release(match_p->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_consume_failure_keeps_ram_note(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("consume-note");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {5, 5, 5, 5});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("consume-note spill failed");
    }
    ram.claim(ram_id);
    const auto restores_before = ram.snapshot().restores;
    ram.test_fail_next_copy_sync();
    bool threw = false;
    try {
        ram.consume(ram_id);
    } catch (const std::runtime_error&) { threw = true; }
    if (!threw) {
        alloc.release();
        return fail("consume-note consume did not throw");
    }
    if (!ram.is_claimed(ram_id)) {
        alloc.release();
        return fail("consume-note dropped the RAM record before commit");
    }
    if (!disk.test_has_ram_note(ram_id)) {
        alloc.release();
        return fail("consume-note forgot the disk note before RAM consume committed");
    }
    if (ram.snapshot().restores != restores_before) {
        ram.release(ram_id);
        alloc.release();
        return fail("consume-note counted a restore before commit");
    }
    ram.release(ram_id);
    alloc.release();
    return 0;
}

int test_claim_cancels_in_flight_idle_extend(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-idle");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> aligned(64, 4);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("claim-idle spill of B failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match_b) {
        alloc.release();
        return fail("claim-idle match of B failed");
    }
    const auto frontier_b = disk.test_load_meta(match_b->entry_id).execution_frontier;
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_d, match_b->entry_id);
    disk.note_ram_resident(ram_d, match_b->entry_id);
    disk.test_set_payload_io_stall_ms(250);
    disk.request_idle_spill();
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_payload_io_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_payload_io_entered()) {
        disk.test_set_payload_io_stall_ms(0);
        alloc.release();
        return fail("claim-idle never entered payload I/O");
    }
    if (!disk.claim(match_b->entry_id)) {
        disk.test_set_payload_io_stall_ms(0);
        alloc.release();
        return fail("claim-idle claim of B failed");
    }
    disk.test_set_payload_io_stall_ms(0);
    disk.prefetch_window(match_b->entry_id, 1, 0);
    if (disk.test_load_meta(match_b->entry_id).execution_frontier != frontier_b) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("in-flight idle extend mutated a claimed generation");
    }
    disk.release(match_b->entry_id);
    alloc.release();
    return 0;
}

int test_cancel_after_meta_rename_keeps_parent(ninfer::DeviceContext& ctx,
                                               ninfer::PagedKVPool& pool) {
    TmpDir dir("rename-cancel");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    const auto frontier_b = static_cast<std::uint32_t>(aligned.size());
    std::uint64_t entry_b = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("rename-cancel spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("rename-cancel match of B failed");
        }
        entry_b = match_b->entry_id;
        std::vector<ninfer::TokenId> extended = aligned;
        extended.push_back(0);
        extended.resize(128, 5);
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, entry_b);
        disk.note_ram_resident(ram_d, entry_b);
        disk.test_arm_stall_after_meta_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("rename-cancel never reached meta rename");
        }
        disk.cancel_idle_spill();
        if (!disk.test_entry_in_index(entry_b) ||
            disk.test_load_meta(entry_b).execution_frontier != frontier_b) {
            alloc.release();
            return fail("cancel after meta rename dropped the parent generation");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto match = reopened.plan_match(text_prompt(aligned),
                                           q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match || match->entry_id != entry_b || match->reuse_base != frontier_b) {
        alloc.release();
        return fail("reopen after rename-cancel did not hit the original generation");
    }
    alloc.release();
    return 0;
}

int test_claim_missing_after_fifo_evict(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-miss");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           192 * 1024, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(192, 1);
    std::vector<ninfer::TokenId> tokens_b(192, 2);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("claim-miss spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("claim-miss spill B failed");
    }
    const auto match_b = disk.plan_match(text_prompt(tokens_b),
                                         q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    if (!match_b) {
        alloc.release();
        return fail("claim-miss match B failed");
    }
    const auto id_b = match_b->entry_id;
    for (int i = 0; i < 24 && disk.test_entry_in_index(id_b); ++i) {
        std::vector<ninfer::TokenId> extra(192, static_cast<ninfer::TokenId>(10 + i));
        const auto ram_x = capture_tokens(ram, pool, alloc, ctx, extra);
        disk.note_ram_resident(ram_x, 0);
        (void)disk.emergency_spill_ram(ram_x);
    }
    if (disk.test_entry_in_index(id_b)) {
        (void)disk.claim(id_b);
        disk.release(id_b);
        alloc.release();
        return fail("claim-miss did not FIFO-evict B");
    }
    if (disk.claim(id_b)) {
        disk.release(id_b);
        alloc.release();
        return fail("claim of FIFO-evicted entry succeeded");
    }
    alloc.release();
    return 0;
}

int test_stale_plan_claim_rejects_extended_generation(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-plan");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    ninfer::DeviceBuffer hid_a(256);
    std::vector<unsigned char> seed_a(256, 0xaa);
    hid_a.copy_from_host(seed_a.data(), seed_a.size());
    ninfer::Tensor hidden_a(hid_a.p, ninfer::DType::U8, {256});
    ninfer::DeviceBuffer hid_b(256);
    std::vector<unsigned char> seed_b(256, 0xbb);
    hid_b.copy_from_host(seed_b.data(), seed_b.size());
    ninfer::Tensor hidden_b(hid_b.p, ninfer::DType::U8, {256});
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    cfg.hidden_bytes = 256;
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> aligned(64, 4);
    auto prompt_a   = text_prompt(aligned);
    auto retained_a = prompt_a;
    retained_a.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_a;
    auto source_a        = make_source(retained_a, identity_a, alloc, pool, ctx.copy_stream, 64);
    source_a.tail_hidden = &hidden_a;
    auto ram_a           = capture_or_evict(ram, source_a);
    if (!ram_a) {
        alloc.release();
        return fail("stale-plan capture of F=64 failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*ram_a, 0);
    if (!disk.emergency_spill_ram(*ram_a)) {
        alloc.release();
        return fail("stale-plan spill of F=64 failed");
    }
    const auto planned = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!planned || planned->execution_frontier != 64) {
        alloc.release();
        return fail("stale-plan match of F=64 failed");
    }
    const auto planned_id   = planned->entry_id;
    const auto planned_hash = planned->hash_f;
    const auto planned_f    = planned->execution_frontier;
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    auto prompt_b   = text_prompt(extended);
    auto retained_b = prompt_b;
    retained_b.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_b;
    auto source_b        = make_source(retained_b, identity_b, alloc, pool, ctx.copy_stream, 128);
    source_b.tail_hidden = &hidden_b;
    auto ram_b           = capture_or_evict(ram, source_b);
    if (!ram_b) {
        alloc.release();
        return fail("stale-plan capture of F=128 failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    ram.set_disk_entry_id(*ram_b, planned_id);
    disk.note_ram_resident(*ram_b, planned_id);
    if (!disk.emergency_spill_ram(*ram_b)) {
        alloc.release();
        return fail("stale-plan extend spill failed");
    }
    if (disk.test_load_meta(planned_id).execution_frontier != 128) {
        alloc.release();
        return fail("stale-plan idle extend did not publish F=128");
    }
    if (disk.claim(planned_id, planned_hash, planned_f)) {
        disk.release(planned_id);
        alloc.release();
        return fail("stale-plan claim accepted the extended generation");
    }
    const auto live = disk.test_load_meta(planned_id);
    if (live.execution_frontier != 128) {
        alloc.release();
        return fail("stale-plan live meta is not F=128");
    }
    auto wrong_hash = live.hash_f;
    wrong_hash.lo ^= 1ULL;
    if (disk.claim(planned_id, wrong_hash, live.execution_frontier)) {
        disk.release(planned_id);
        alloc.release();
        return fail("stale-plan claim accepted a mismatched live hash");
    }
    if (disk.claim(planned_id, live.hash_f, planned_f)) {
        disk.release(planned_id);
        alloc.release();
        return fail("stale-plan claim accepted the live hash at the stale frontier");
    }
    if (!disk.claim(planned_id, live.hash_f, live.execution_frontier)) {
        alloc.release();
        return fail("stale-plan claim of the live generation failed");
    }
    ninfer::DeviceBuffer hid_out(256);
    hid_out.fill(0);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {256});
    auto dest = pool.reserve(4);
    dest.materialize_pages(3, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    disk.restore_device(planned_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "stale-plan restore hung"); rc != 0) {
            disk.release(planned_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(planned_id);
        dest.release();
        alloc.release();
        std::cerr << "stale-plan restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> got(256);
    CUDA_CHECK(cudaMemcpy(got.data(), hid_out.p, got.size(), cudaMemcpyDeviceToHost));
    disk.release(planned_id);
    dest.release();
    if (std::any_of(got.begin(), got.end(), [](unsigned char c) { return c != 0xbb; })) {
        alloc.release();
        return fail("live-generation restore did not return F=128 hidden");
    }
    alloc.release();
    return 0;
}

int test_post_rename_failure_keeps_a_valid_generation(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("post-rename");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("post-rename spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("post-rename match of B failed");
        }
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, match_b->entry_id);
        disk.note_ram_resident(ram_d, match_b->entry_id);
        const auto pages_b = disk.test_main_page_ids(match_b->entry_id);
        disk.test_arm_fail_after_meta_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("post-rename never reached meta rename");
        }
        disk.wait_idle_and_fsync();
        if (!disk.test_objects_fsynced_before_meta()) {
            alloc.release();
            return fail("post-rename commit fsynced meta without durable object names");
        }
        if (disk.ram_is_durable(ram_d)) {
            alloc.release();
            return fail("post-rename fsync failure marked RAM durable");
        }
        const auto live128 =
            disk.plan_match(text_prompt(extended),
                            q36::detail::prefix_hash_chain(text_prompt(extended)));
        if (!live128 || live128->reuse_base != 128) {
            alloc.release();
            return fail("post-rename fsync failure did not publish the new generation");
        }
        for (std::uint64_t id : pages_b) {
            if (id != 0 && !packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
                alloc.release();
                return fail("post-rename failure deleted the previous generation's objects");
            }
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit64 =
        reopened.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    const auto hit128 =
        reopened.plan_match(text_prompt(extended),
                            q36::detail::prefix_hash_chain(text_prompt(extended)));
    if (!hit128 || hit128->reuse_base != 128) {
        alloc.release();
        return fail("post-rename failure did not keep the new generation hittable");
    }
    const auto meta = reopened.test_load_meta(hit128->entry_id);
    for (std::uint64_t id : meta.main_page_ids) {
        if (id != 0 && !packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
            alloc.release();
            return fail("post-rename new meta references a missing page object");
        }
    }
    if (const int rc = claim_and_restore_match(reopened, ctx, pool, *hit128, 2,
                                                "post-rename claim of the kept generation failed",
                                                "post-rename restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    alloc.release();
    return 0;
}

int test_rollback_meta_failure_keeps_a_valid_generation(ninfer::DeviceContext& ctx,
                                                        ninfer::PagedKVPool& pool) {
    TmpDir dir("rollback-fail");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("rollback-fail spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("rollback-fail match of B failed");
        }
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, match_b->entry_id);
        disk.note_ram_resident(ram_d, match_b->entry_id);
        disk.test_arm_stall_after_meta_rename();
        disk.test_arm_fail_rollback_meta();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("rollback-fail never reached meta rename");
        }
        disk.cancel_idle_spill();
        disk.wait_idle_and_fsync();
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit64 =
        reopened.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    const auto hit128 =
        reopened.plan_match(text_prompt(extended),
                            q36::detail::prefix_hash_chain(text_prompt(extended)));
    if (!hit128 || hit128->reuse_base != 128) {
        alloc.release();
        return fail("rollback meta failure did not keep a consistent generation hittable");
    }
    const auto meta = reopened.test_load_meta(hit128->entry_id);
    for (std::uint64_t id : meta.main_page_ids) {
        if (id != 0 && !packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
            alloc.release();
            return fail("rollback-fail new meta references a missing page object");
        }
    }
    if (const int rc = claim_and_restore_match(reopened, ctx, pool, *hit128, 2,
                                                "rollback-fail claim of the kept generation failed",
                                                "rollback-fail restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    alloc.release();
    return 0;
}

int test_stale_manifest_discovers_branch(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-man");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 7);
    std::vector<ninfer::TokenId> child = aligned;
    child.push_back(99);
    child.resize(70, 11);
    const auto man = dir.path / "MANIFEST";
    const auto bak = dir.path / "MANIFEST.bak";
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("stale-manifest parent spill failed");
        }
        disk.wait_idle_and_fsync();
        fs::copy_file(man, bak);
        const auto match_a = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_a) {
            alloc.release();
            return fail("stale-manifest parent match failed");
        }
        const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
        ram.set_disk_entry_id(ram_c, match_a->entry_id);
        disk.note_ram_resident(ram_c, match_a->entry_id);
        if (!disk.emergency_spill_ram(ram_c)) {
            alloc.release();
            return fail("stale-manifest branch spill failed");
        }
        disk.wait_idle_and_fsync();
    }
    fs::copy_file(bak, man, fs::copy_options::overwrite_existing);
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit = reopened.plan_match(text_prompt(child),
                                         q36::detail::prefix_hash_chain(text_prompt(child)));
    if (!hit || hit->reuse_base < 64) {
        alloc.release();
        return fail("stale MANIFEST hid the durable branch generation");
    }
    const auto parent = reopened.plan_match(text_prompt(aligned),
                                            q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!parent) {
        alloc.release();
        return fail("stale MANIFEST hid the durable parent generation");
    }
    if (const int rc = claim_and_restore_match(reopened, ctx, pool, *hit, 2,
                                                "stale-manifest claim of the branch failed",
                                                "stale-manifest branch restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    alloc.release();
    return 0;
}

int test_gc_skipped_shared_pages(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("skip-share");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 7);
    std::vector<ninfer::TokenId> child = aligned;
    child.push_back(99);
    child.resize(70, 11);
    std::uint64_t shared_page = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("skip-share parent spill failed");
        }
        const auto match_a = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_a) {
            alloc.release();
            return fail("skip-share parent match failed");
        }
        const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
        ram.set_disk_entry_id(ram_c, match_a->entry_id);
        disk.note_ram_resident(ram_c, match_a->entry_id);
        if (!disk.emergency_spill_ram(ram_c)) {
            alloc.release();
            return fail("skip-share branch spill failed");
        }
        const auto match_c = disk.plan_match(text_prompt(child),
                                             q36::detail::prefix_hash_chain(text_prompt(child)));
        if (!match_c) {
            alloc.release();
            return fail("skip-share branch match failed");
        }
        const auto pages_a = disk.test_main_page_ids(match_a->entry_id);
        const auto pages_c = disk.test_main_page_ids(match_c->entry_id);
        if (pages_a.empty() || pages_c.empty() || pages_a[0] != pages_c[0]) {
            alloc.release();
            return fail("skip-share parent and branch do not share a page");
        }
        shared_page = pages_a[0];
    }
    cfg.max_context = 16;
    {
        q36::detail::KVDiskCache skipped(cfg);
        if (skipped.test_skipped_count() < 2) {
            alloc.release();
            return fail("skip-share did not skip parent and branch");
        }
        if (!skipped.test_gc_skipped_one()) {
            alloc.release();
            return fail("skip-share failed to GC one skipped tree");
        }
        if (skipped.test_skipped_count() != 1) {
            alloc.release();
            return fail("skip-share GC did not leave one skipped tree");
        }
        if (shared_page == 0 || !packed_object_location(
                                    dir.path, q36::detail::DiskObjectKind::Main, shared_page)) {
            alloc.release();
            return fail("skip-share GC deleted a page still owned by the surviving tree");
        }
        skipped.wait_idle_and_fsync();
        if (!packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, shared_page)) {
            alloc.release();
            return fail("compaction dropped a shared object retained only by a skipped tree");
        }
    }
    cfg.max_context = 4096;
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit = reopened.plan_match(text_prompt(child),
                                         q36::detail::prefix_hash_chain(text_prompt(child)));
    if (!hit) {
        alloc.release();
        return fail("GC of one skipped tree corrupted the shared-page survivor");
    }
    if (const int rc = claim_and_restore_match(reopened, ctx, pool, *hit, 2,
                                                "skip-share claim of the survivor failed",
                                                "skip-share survivor restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    alloc.release();
    return 0;
}

int test_truncated_page_payload_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("trunc-page");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t page_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("trunc-page spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("trunc-page match failed");
        }
        const auto pages = disk.test_main_page_ids(match->entry_id);
        if (pages.empty()) {
            alloc.release();
            return fail("trunc-page stored no main page");
        }
        page_id = pages.front();
    }
    const auto page = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, page_id);
    if (!page) {
        alloc.release();
        return fail("trunc-page could not locate packed page");
    }
    fs::resize_file(page->path, page->offset + q36::detail::kDiskPageHeaderBytes);
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("header-only truncated page remained hittable");
    }
    if (reopened.snapshot().drops == 0) {
        alloc.release();
        return fail("header-only truncated page did not increment drops");
    }
    alloc.release();
    return 0;
}

int test_truncated_state_payload_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("trunc-state");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(4096);
    std::vector<unsigned char> seed(4096, 0xcd);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {4096});
    auto prompt   = text_prompt({2, 3, 4, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("trunc-state capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 4096;
    std::uint64_t hidden_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("trunc-state spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("trunc-state match failed");
        }
        hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
        if (hidden_id == 0) {
            alloc.release();
            return fail("trunc-state stored no hidden object");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto match = reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || !reopened.claim(match->entry_id)) {
        alloc.release();
        return fail("direct-state truncation setup was not hittable");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid_out(4096);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {4096});
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    reopened.test_arm_direct_state_read_barrier();
    reopened.restore_device(match->entry_id, target);
    if (!wait_pred([&] { return reopened.test_direct_state_read_entered(); },
                   std::chrono::seconds(2))) {
        reopened.test_release_direct_state_read_barrier();
        reopened.cancel_restore();
        reopened.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("aligned state restore did not enter O_DIRECT route");
    }
    struct DirectStateBarrierGuard {
        q36::detail::KVDiskCache& disk;
        bool active = true;
        ~DirectStateBarrierGuard() {
            if (active) { disk.test_release_direct_state_read_barrier(); }
        }
    } barrier_guard{reopened};
    const auto hidden_loc = packed_object_location(dir.path, q36::detail::DiskObjectKind::State,
                                                   hidden_id);
    if (!hidden_loc) {
        reopened.test_release_direct_state_read_barrier();
        reopened.cancel_restore();
        reopened.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("trunc-state could not locate packed state");
    }
    fs::resize_file(hidden_loc->path,
                    hidden_loc->offset + q36::detail::kDiskStatePayloadOffset);
    reopened.test_release_direct_state_read_barrier();
    barrier_guard.active = false;
    bool threw = false;
    try {
        reopened.wait_copies();
    } catch (const std::runtime_error&) { threw = true; }
    reopened.cancel_restore();
    reopened.release(match->entry_id);
    dest.release();
    if (!threw || reopened.snapshot().drops == 0) {
        alloc.release();
        return fail("truncated aligned direct-state payload did not fail restore");
    }
    alloc.release();
    return 0;
}

int test_live_state_gap_crc_is_checked(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("state-gap-crc");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(4096);
    std::vector<unsigned char> seed(4096, 0x5a);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {4096});
    const auto prompt = text_prompt({12, 13, 14, 15});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto ram_id = capture_or_evict(ram, source);
    if (!ram_id) {
        alloc.release();
        return fail("state-gap-crc capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 4096;
    q36::detail::KVDiskCache disk(cfg);
    disk.note_ram_resident(*ram_id, 0);
    if (!disk.emergency_spill_ram(*ram_id)) {
        alloc.release();
        return fail("state-gap-crc spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("state-gap-crc match failed");
    }
    const auto hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
    const auto hidden_loc =
        packed_object_location(dir.path, q36::detail::DiskObjectKind::State, hidden_id);
    if (!hidden_loc) {
        alloc.release();
        return fail("state-gap-crc could not locate hidden state");
    }
    const unsigned char corrupt = 0xa7;
    if (!packed_write_at(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                         q36::detail::kDiskCodecHeaderBytes, &corrupt, 1)) {
        alloc.release();
        return fail("state-gap-crc could not corrupt reserved gap");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("state-gap-crc claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid_out(4096);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {4096});
    q36::detail::DiskRestoreTarget target;
    target.text = &dest;
    target.text_pool = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden = &hidden_out;
    target.stream = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    bool threw = false;
    try {
        (void)wait_restore_bounded(disk, ctx, "state-gap-crc restore hung");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (!threw || disk.snapshot().drops == 0) {
        return fail("live state restore accepted corruption in reserved gap");
    }
    return 0;
}

int test_empty_codec_state_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("empty-codec");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(256);
    std::vector<unsigned char> seed(256, 0xef);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt({2, 3, 4, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("empty-codec capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t hidden_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("empty-codec spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("empty-codec match failed");
        }
        hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
        if (hidden_id == 0) {
            alloc.release();
            return fail("empty-codec stored no hidden object");
        }
    }
    std::vector<std::uint8_t> hdr(q36::detail::kDiskCodecHeaderBytes, 0);
    hdr[0] = 0;
    hdr[1] = static_cast<std::uint8_t>(q36::detail::DiskStateKind::TailHidden);
    const std::uint64_t unc = 256;
    const std::uint64_t cmp = 0;
    std::memcpy(hdr.data() + 4, &unc, 8);
    std::memcpy(hdr.data() + 12, &cmp, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                             hdr.data(), hdr.size())) {
        alloc.release();
        return fail("empty-codec could not patch packed state");
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("empty-codec state remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("empty-codec state did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "empty-codec reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_wrapped_compressed_bytes_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("wrap-cmp");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(256);
    std::vector<unsigned char> seed(256, 0xab);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt({2, 3, 4, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("wrap-cmp capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t hidden_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("wrap-cmp spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("wrap-cmp match failed");
        }
        hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
        if (hidden_id == 0) {
            alloc.release();
            return fail("wrap-cmp stored no hidden object");
        }
    }
    std::vector<std::uint8_t> hdr(q36::detail::kDiskCodecHeaderBytes, 0);
    hdr[0] = 1;
    hdr[1] = static_cast<std::uint8_t>(q36::detail::DiskStateKind::TailHidden);
    const std::uint64_t unc = 256;
    const std::uint64_t cmp = std::numeric_limits<std::uint64_t>::max();
    std::memcpy(hdr.data() + 4, &unc, 8);
    std::memcpy(hdr.data() + 12, &cmp, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                             hdr.data(), hdr.size())) {
        alloc.release();
        return fail("wrap-cmp could not patch packed state");
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("wrapped compressed_bytes remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("wrapped compressed_bytes did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "wrap-cmp reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_short_page_id_vector_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("short-pages");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("short-pages spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("short-pages match failed");
        }
        entry_id = match->entry_id;
        if (disk.test_main_page_ids(entry_id).empty()) {
            alloc.release();
            return fail("short-pages stored no main pages");
        }
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    in.close();
    if (bytes.size() < 16) {
        alloc.release();
        return fail("short-pages meta.bin is truncated");
    }
    const std::size_t counts = bytes.size() - 16;
    std::uint32_t zero        = 0;
    std::memcpy(bytes.data() + counts, &zero, 4);
    std::memcpy(bytes.data() + counts + 4, &zero, 4);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(text_prompt(tokens),
                                q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("short page-id vector remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("short page-id vector did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "short-pages reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_corrupt_unselected_checkpoint_is_omitted(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("omit-head");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    q36::detail::RamLadderHead l1 = rollback;
    l1.frontier                   = 6;
    l1.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    l1.kind = q36::detail::ContextCheckpointKind::Ladder;
    q36::detail::RamLadderHead l2 = l1;
    l2.frontier                   = 5;
    l2.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    source.ladder_heads = {rollback, l1, l2};
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("omit-head capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("omit-head spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("omit-head match failed");
    }
    const auto meta      = disk.test_load_meta(match->entry_id);
    const auto hidden_id = meta.ladders[0].hidden_id;
    if (hidden_id == 0) {
        alloc.release();
        return fail("omit-head stored no ladder hidden");
    }
    {
        std::vector<std::uint8_t> hdr(q36::detail::kDiskCodecHeaderBytes, 0);
        hdr[0] = 0;
        hdr[1] = static_cast<std::uint8_t>(q36::detail::DiskStateKind::LadderHidden);
        const std::uint64_t unc = 64;
        const std::uint64_t cmp = 0;
        std::memcpy(hdr.data() + 4, &unc, 8);
        std::memcpy(hdr.data() + 12, &cmp, 8);
        if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                                 hdr.data(), hdr.size())) {
            alloc.release();
            return fail("omit-head could not patch packed checkpoint state");
        }
    }
    auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("omit-head load_host missed the live entry");
    }
    if (disk.populate_checkpoint_images(*host)) {
        alloc.release();
        return fail("corrupt checkpoint populate succeeded");
    }
    if (!host->ladder_images.empty()) {
        alloc.release();
        return fail("failed populate still published checkpoint images");
    }
    alloc.release();
    return 0;
}

int test_corrupt_checkpoint_fails_restore(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("ckpt-pread");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    q36::detail::RamLadderHead ladder = rollback;
    ladder.frontier                   = 5;
    ladder.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    ladder.kind = q36::detail::ContextCheckpointKind::Ladder;
    source.ladder_heads = {rollback, ladder};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("ckpt-pread capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("ckpt-pread spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("ckpt-pread match failed");
    }
    const auto meta      = disk.test_load_meta(match->entry_id);
    const auto hidden_id = meta.ladders[0].hidden_id != 0 ? meta.ladders[0].hidden_id
                                                           : meta.rollback.hidden_id;
    if (hidden_id == 0) {
        alloc.release();
        return fail("ckpt-pread stored no checkpoint hidden");
    }
    if (!disk.claim(match->entry_id)) {
        alloc.release();
        return fail("ckpt-pread claim failed");
    }
    disk.test_break_object(hidden_id, q36::detail::DiskObjectKind::State);
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    const auto drops_before = disk.snapshot().drops;
    disk.restore_device(match->entry_id, target);
    bool threw = false;
    std::string what;
    try {
        disk.wait_copies();
    } catch (const std::runtime_error& e) {
        threw = true;
        what  = e.what();
    }
    if (!threw || what != "KV disk restore pread failed") {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("corrupt checkpoint restore did not fail the request");
    }
    if (disk.snapshot().drops <= drops_before) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("corrupt checkpoint restore did not increment drops");
    }
    if (disk.copies_ready()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("failed checkpoint restore reported copies_ready");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        dest.release();
        alloc.release();
        return fail("corrupt checkpoint restore consumed the inclusive entry");
    }
    dest.release();
    alloc.release();
    return 0;
}

int test_missing_ledger_skips_entry(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("miss-ledger");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t ledger_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("miss-ledger spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("miss-ledger match failed");
        }
        ledger_id = disk.test_load_meta(match->entry_id).ledger_id;
        if (ledger_id == 0) {
            alloc.release();
            return fail("miss-ledger stored no ledger object");
        }
    }
    const auto ledger = packed_object_location(dir.path, q36::detail::DiskObjectKind::Ledger,
                                               ledger_id);
    if (!ledger) {
        alloc.release();
        return fail("miss-ledger could not locate packed ledger");
    }
    fs::resize_file(ledger->path, ledger->offset);
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(text_prompt(tokens),
                                q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("missing ledger remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("missing ledger did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "missing ledger reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_manifest_duplicate_ids_rebuild(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("dup-man");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("dup-manifest spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("dup-manifest match failed");
        }
        entry_id = match->entry_id;
    }
    {
        std::vector<std::uint8_t> man;
        auto u32 = [&](std::uint32_t v) {
            man.push_back(static_cast<std::uint8_t>(v));
            man.push_back(static_cast<std::uint8_t>(v >> 8));
            man.push_back(static_cast<std::uint8_t>(v >> 16));
            man.push_back(static_cast<std::uint8_t>(v >> 24));
        };
        auto u64 = [&](std::uint64_t v) {
            for (int s = 0; s < 64; s += 8) {
                man.push_back(static_cast<std::uint8_t>(v >> s));
            }
        };
        man.insert(man.end(), q36::detail::kDiskManifestMagic,
                   q36::detail::kDiskManifestMagic + 8);
        u32(q36::detail::kDiskFormatVersion);
        u32(2);
        u64(entry_id);
        u64(entry_id);
        std::ofstream out(dir.path / "MANIFEST", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(man.data()),
                  static_cast<std::streamsize>(man.size()));
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        const auto match =
            reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match || match->entry_id != entry_id) {
            alloc.release();
            return fail("duplicate MANIFEST hid the durable generation");
        }
        if (reopened.snapshot().entry_count != 1) {
            alloc.release();
            return fail("duplicate MANIFEST produced a duplicate index entry");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "duplicate MANIFEST reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_meta_entry_id_mismatch_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("id-mismatch");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("id-mismatch spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("id-mismatch match failed");
        }
        entry_id = match->entry_id;
    }
    const auto src = dir.path / "entries" / std::to_string(entry_id);
    const auto dst = dir.path / "entries" / std::to_string(entry_id + 1000);
    fs::copy(src, dst, fs::copy_options::recursive);
    try {
        q36::detail::KVDiskCache reopened(cfg);
        const auto match =
            reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match || match->entry_id != entry_id) {
            alloc.release();
            return fail("directory/meta id mismatch hid the matching generation");
        }
        if (reopened.test_entry_in_index(entry_id + 1000)) {
            alloc.release();
            return fail("directory/meta id mismatch was indexed");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("directory/meta id mismatch did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "id-mismatch reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_refresh_clears_absent_rollback(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("refresh-rb");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("refresh-rollback capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("refresh-rollback first spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || disk.test_load_meta(match->entry_id).rollback.frontier != 3) {
        alloc.release();
        return fail("refresh-rollback did not persist the rollback slot");
    }
    const auto old_hidden = disk.test_load_meta(match->entry_id).rollback.hidden_id;
    ram.claim(*id);
    ram.consume(*id);
    const auto ram2 = capture_tokens(ram, pool, alloc, ctx, tokens);
    ram.set_disk_entry_id(ram2, match->entry_id);
    disk.note_ram_resident(ram2, match->entry_id);
    if (!disk.emergency_spill_ram(ram2)) {
        alloc.release();
        return fail("refresh-rollback second spill failed");
    }
    const auto live = disk.test_load_meta(match->entry_id);
    if (live.rollback.frontier != 0 || live.rollback.hidden_id != 0) {
        alloc.release();
        return fail("refresh kept a rollback slot the captured image no longer had");
    }
    if (old_hidden != 0 &&
        fs::exists(dir.path / "objects" / "state" / std::to_string(old_hidden))) {
        alloc.release();
        return fail("refresh left an unreferenced rollback object");
    }
    alloc.release();
    return 0;
}

int test_plan_match_does_not_wait_on_payload_io(ninfer::DeviceContext& ctx,
                                                ninfer::PagedKVPool& pool) {
    TmpDir dir("plan-ssd");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(64, 6);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    disk.test_set_payload_io_stall_ms(250);
    std::atomic<bool> spilled{false};
    std::thread spiller([&] { spilled.store(disk.emergency_spill_ram(ram_id)); });
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_payload_io_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_payload_io_entered()) {
        disk.test_set_payload_io_stall_ms(0);
        spiller.join();
        alloc.release();
        return fail("payload I/O stall was not reached");
    }
    const auto t0      = std::chrono::steady_clock::now();
    const auto unrelated = text_prompt({1, 2, 3, 4});
    (void)disk.plan_match(unrelated, q36::detail::prefix_hash_chain(unrelated));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    disk.test_set_payload_io_stall_ms(0);
    spiller.join();
    if (elapsed > std::chrono::milliseconds(100)) {
        alloc.release();
        return fail("plan_match waited on SSD payload I/O");
    }
    if (!spilled.load()) {
        alloc.release();
        return fail("stalled emergency spill failed");
    }
    alloc.release();
    return 0;
}

int test_populate_does_not_hold_mutex_across_pread(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("pop-ssd");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(4096);
    hid.fill(0x22);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {4096});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(4096, 0x22);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id               = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("populate-mutex capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 4096;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("populate-mutex spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("populate-mutex match failed");
    }
    auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("populate-mutex load_host missed the live entry");
    }
    disk.test_set_payload_io_stall_ms(250);
    std::thread populator([&] { (void)disk.populate_checkpoint_images(*host); });
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_payload_io_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_payload_io_entered()) {
        disk.test_set_payload_io_stall_ms(0);
        populator.join();
        alloc.release();
        return fail("populate I/O stall was not reached");
    }
    const auto t0        = std::chrono::steady_clock::now();
    const auto unrelated = text_prompt({9, 9, 9, 9});
    (void)disk.plan_match(unrelated, q36::detail::prefix_hash_chain(unrelated));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    disk.test_set_payload_io_stall_ms(0);
    populator.join();
    if (elapsed > std::chrono::milliseconds(100)) {
        alloc.release();
        return fail("plan_match waited on populate_checkpoint_images pread");
    }
    alloc.release();
    return 0;
}

int test_idle_pin_aborts_for_ram_claim(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("idle-pin");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {8, 8, 8, 8});
    disk.note_ram_resident(ram_id, 0);
    for (int i = 0; i < 40; ++i) {
        disk.request_idle_spill();
        std::this_thread::sleep_for(std::chrono::microseconds(50 * ((i % 5) + 1)));
        disk.begin_ram_idle_exclusion(ram_id);
        if (!disk.ram_is_durable(ram_id)) {
            ram.claim(ram_id);
            ram.release(ram_id);
            disk.end_ram_idle_exclusion(ram_id);
            if (!ram.peek_oldest_unpinned() || *ram.peek_oldest_unpinned() != ram_id) {
                alloc.release();
                return fail("idle pin-before-spill left a leftover I/O pin");
            }
        } else {
            disk.end_ram_idle_exclusion(ram_id);
        }
    }
    alloc.release();
    return 0;
}

int test_ram_idle_exclusion_covers_claim(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("idle-excl");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {8, 8, 8, 8});
    disk.note_ram_resident(ram_id, 0);
    disk.begin_ram_idle_exclusion(ram_id);
    disk.request_idle_spill();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    if (ram.test_io_pins(ram_id) != 0) {
        disk.end_ram_idle_exclusion(ram_id);
        alloc.release();
        return fail("idle spill pinned RAM while exclusion was held");
    }
    ram.claim(ram_id);
    disk.end_ram_idle_exclusion(ram_id);
    if (disk.ram_is_durable(ram_id)) {
        ram.release(ram_id);
        alloc.release();
        return fail("idle spill committed during claim exclusion");
    }
    ram.release(ram_id);
    alloc.release();
    return 0;
}

int test_discard_keeps_note_until_evict(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("discard-note");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {4, 4, 4, 4});
    disk.note_ram_resident(ram_id, 0);
    ram.pin_for_io(ram_id);
    disk.begin_ram_idle_exclusion(ram_id);
    if (ram.evict_one_unpinned(ram_id)) {
        disk.end_ram_idle_exclusion(ram_id);
        alloc.release();
        return fail("evict succeeded while an I/O pin was held");
    }
    if (!disk.test_has_ram_note(ram_id)) {
        ram.unpin_for_io(ram_id);
        disk.end_ram_idle_exclusion(ram_id);
        alloc.release();
        return fail("RAM note was erased before a failed evict");
    }
    disk.end_ram_idle_exclusion(ram_id);
    ram.unpin_for_io(ram_id);
    disk.begin_ram_idle_exclusion(ram_id);
    if (!ram.evict_one_unpinned(ram_id)) {
        disk.end_ram_idle_exclusion(ram_id);
        alloc.release();
        return fail("evict failed after the I/O pin was dropped");
    }
    disk.forget_ram_resident(ram_id);
    disk.end_ram_idle_exclusion(ram_id);
    if (disk.test_has_ram_note(ram_id)) {
        alloc.release();
        return fail("RAM note survived a successful evict");
    }
    alloc.release();
    return 0;
}

int test_checkpoint_images_survive_host_load(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("images");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(4096);
    hid.fill(0x22);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {4096});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(4096, 0x22);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 4096;
    q36::detail::RamLadderHead ladder = rollback;
    ladder.frontier                   = 6;
    ladder.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    ladder.kind = q36::detail::ContextCheckpointKind::Ladder;
    source.ladder_heads = {rollback, ladder};
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("checkpoint-image capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("checkpoint-image spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("checkpoint-image match failed");
    }
    auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("checkpoint-image load_host missed the live entry");
    }
    disk.test_arm_direct_state_read_barrier();
    std::atomic<bool> populate_ok{false};
    std::thread populate([&] {
        populate_ok.store(disk.populate_checkpoint_images(*host), std::memory_order_release);
    });
    if (!wait_pred([&] { return disk.test_direct_state_read_entered(); },
                   std::chrono::seconds(2))) {
        disk.test_release_direct_state_read_barrier();
        populate.join();
        alloc.release();
        return fail("retained checkpoint did not enter aligned direct-state route");
    }
    disk.test_release_direct_state_read_barrier();
    populate.join();
    if (!populate_ok.load(std::memory_order_acquire) || host->ladders.size() != 2 ||
        host->ladder_images.size() != 2) {
        alloc.release();
        return fail("populate_checkpoint_images did not restore rollback and ladder payloads");
    }
    if (host->ladder_images[0].hidden == nullptr || host->ladder_images[1].hidden == nullptr) {
        alloc.release();
        return fail("checkpoint images lost hidden payloads");
    }
    if (host->ladder_images[0].hidden->size() != hidden_host.size() ||
        std::memcmp(host->ladder_images[0].hidden->data(), hidden_host.data(),
                    hidden_host.size()) != 0) {
        alloc.release();
        return fail("checkpoint final pinned owner contains the wrong hidden payload");
    }
    if (reinterpret_cast<std::uintptr_t>(host->ladder_images[0].hidden->data()) %
            q36::detail::kDiskPageIoAlignment !=
        0) {
        alloc.release();
        return fail("checkpoint final pinned owner is not direct-I/O aligned");
    }
    alloc.release();
    return 0;
}

int test_ram_header_ticket_roundtrip(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache ram(8ULL << 20);
    auto prompt   = text_prompt({3, 3, 3, 3});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source          = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.disk_entry_id = 77;
    auto id              = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("ticket capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    if (ram.disk_entry_id(*id) != 77) {
        alloc.release();
        return fail("RAM record lost the capture ticket");
    }
    ram.set_disk_entry_id(*id, 88);
    const auto host = ram.load_host(*id);
    if (host.disk_entry_id != 88 || ram.disk_entry_id(*id) != 88) {
        alloc.release();
        return fail("set_disk_entry_id did not persist into the header");
    }
    ram.claim(*id);
    ram.consume(*id);
    source.disk_entry_id = 88;
    auto recaptured      = capture_or_evict(ram, source);
    if (!recaptured || ram.disk_entry_id(*recaptured) != 88) {
        alloc.release();
        return fail("recapture after consume lost the disk ticket");
    }
    alloc.release();
    return 0;
}

int test_inclusive_disk_after_ram_consume(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("inclusive");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {9, 9, 9, 9});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("inclusive spill failed");
    }
    const auto prompt = text_prompt({9, 9, 9, 9});
    ram.claim(ram_id);
    ram.consume(ram_id);
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("disk hit disappeared after RAM consume");
    }
    alloc.release();
    return 0;
}

int test_wait_idle_after_prefetch_and_claimed_evict(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("idle-fsync");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           4ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {1, 2, 3, 4});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("idle-fsync first spill failed");
    }
    const auto prompt_a = text_prompt({1, 2, 3, 4});
    const auto match_a  = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!match_a) {
        alloc.release();
        return fail("idle-fsync match failed");
    }
    disk.prefetch_window(match_a->entry_id, 1, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    disk.cancel_restore();
    std::atomic<bool> idle_done{false};
    std::thread idle_waiter([&] {
        disk.wait_idle_and_fsync();
        idle_done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!idle_done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!idle_done.load()) {
        idle_waiter.detach();
        alloc.release();
        return fail("wait_idle_and_fsync hung after prefetch cancel");
    }
    idle_waiter.join();

    disk.claim(match_a->entry_id);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {9, 8, 7, 6});
    disk.note_ram_resident(ram_b, 0);
    (void)disk.emergency_spill_ram(ram_b);
    if (!disk.test_entry_in_index(match_a->entry_id)) {
        alloc.release();
        return fail("capacity pressure evicted a claimed disk entry");
    }
    disk.release(match_a->entry_id);
    alloc.release();
    return 0;
}

int test_wait_idle_with_live_prefetch(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("prefetch-idle");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {2, 2, 2, 2});
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("prefetch-idle spill failed");
    }
    const auto match =
        disk.plan_match(text_prompt({2, 2, 2, 2}),
                        q36::detail::prefix_hash_chain(text_prompt({2, 2, 2, 2})));
    if (!match) {
        alloc.release();
        return fail("prefetch-idle match failed");
    }
    disk.claim(match->entry_id);
    disk.prefetch_window(match->entry_id, 1, 0);
    std::atomic<bool> idle_done{false};
    std::thread idle_waiter([&] {
        disk.wait_idle_and_fsync();
        idle_done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!idle_done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!idle_done.load()) {
        disk.cancel_restore();
        idle_waiter.detach();
        disk.release(match->entry_id);
        alloc.release();
        return fail("wait_idle_and_fsync hung with leftover prefetch_q");
    }
    idle_waiter.join();
    disk.release(match->entry_id);
    alloc.release();
    return 0;
}

int test_cancel_idle_spill_unpins_peek_victim(ninfer::DeviceContext& ctx,
                                              ninfer::PagedKVPool& pool) {
    TmpDir dir("idle-d");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto d_id = capture_tokens(ram, pool, alloc, ctx, {3, 3, 3, 3});
    disk.note_ram_resident(d_id, 0);
    disk.request_idle_spill();
    std::optional<std::uint64_t> peek;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (std::chrono::steady_clock::now() < deadline) {
        disk.cancel_idle_spill();
        peek = ram.peek_oldest_unpinned();
        if (peek && *peek == d_id) { break; }
        disk.request_idle_spill();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!peek || *peek != d_id) {
        alloc.release();
        return fail("cancel_idle_spill left D un-peekable");
    }
    if (!disk.ram_is_durable(d_id)) { (void)disk.emergency_spill_ram(d_id); }
    disk.forget_ram_resident(d_id);
    if (!ram.evict_one_unpinned(d_id)) {
        alloc.release();
        return fail("evict of D after cancel_idle_spill failed");
    }
    alloc.release();
    return 0;
}

int test_ticket_write_fail_unpins(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("ticket-fail");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens{4, 4, 4, 4};
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    ram.test_fail_next_ticket_write();
    const bool spilled = disk.emergency_spill_ram(ram_id);
    if (!spilled) {
        alloc.release();
        return fail("ticket-write failure dropped a published SSD generation");
    }
    if (!ram.peek_oldest_unpinned() || *ram.peek_oldest_unpinned() != ram_id) {
        alloc.release();
        return fail("ticket-write failure left a leftover I/O pin");
    }
    const auto match =
        disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
    if (!match) {
        alloc.release();
        return fail("published SSD generation was not hittable after ticket-write failure");
    }
    if (ram.disk_entry_id(ram_id) != match->entry_id) {
        alloc.release();
        return fail("RAM ticket and header did not converge on the published child");
    }
    alloc.release();
    return 0;
}

int test_zero_kv_gdn_restore_and_cancel(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("gdn0");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 16,
                      .conv_channels  = 64,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 8,
                      .key_head_dim   = 8,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x5a);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x5b);
    if (conv.size() % q36::detail::kDiskPageIoAlignment != 0 ||
        rec.size() % q36::detail::kDiskPageIoAlignment != 0) {
        return fail("direct-state GDN test geometry is not aligned");
    }
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x5c);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, alloc, 0xaa);
    auto prompt   = text_prompt({5, 5, 5, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source            = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.gdn             = &gdn;
    source.gdn_current_slot = 0;
    source.tail_hidden     = &hidden;
    auto id                = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("gdn capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn, nullptr);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("gdn spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("gdn match failed");
    }
    disk.claim(match->entry_id);
    gdn.zero_slot(1, ctx.stream);
    ninfer::DeviceBuffer hid_out(64);
    hid_out.fill(0);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    std::vector<unsigned char> poison(ninfer::paged_kv_logical_page_bytes(pool), 0xff);
    ninfer::unpack_paged_kv_logical_page_from_host(dest, pool, poison.data(), 0, ctx.stream);
    ctx.synchronize_all();
    q36::detail::DiskRestoreTarget target;
    target.text               = &dest;
    target.text_pool          = &pool;
    target.text_dst_pages     = 0;
    target.gdn                = &gdn;
    target.gdn_current_slot   = 1;
    target.tail_hidden        = &hidden_out;
    target.stream             = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "zero-KV GDN restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "zero-KV GDN restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> conv_got(conv.size());
    std::vector<unsigned char> rec_got(rec.size());
    gdn.pack_slot_to_host(1, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != conv || rec_got != rec) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("zero-KV restore did not unpack GDN");
    }
    std::vector<unsigned char> hid_got(64);
    CUDA_CHECK(cudaMemcpy(hid_got.data(), hid_out.p, hid_got.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(hid_got.begin(), hid_got.end(), [](unsigned char c) { return c != 0x5c; })) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("zero-KV restore did not unpack hidden");
    }
    std::vector<unsigned char> kv_got(poison.size());
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, kv_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (kv_got != poison) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("zero-KV restore wrote dest KV pages");
    }
    disk.cancel_restore();
    gdn.zero_slot(1, ctx.stream);
    hid_out.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    disk.test_arm_fail_after_state_h2d_enqueue();
    disk.restore_device(match->entry_id, target);
    bool injected_threw = false;
    try {
        disk.wait_copies();
    } catch (const std::runtime_error&) { injected_threw = true; }
    disk.cancel_restore();
    if (!injected_threw) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("post-enqueue state H2D failure did not fail restore");
    }
    gdn.zero_slot(1, ctx.stream);
    hid_out.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "second GDN restore after cancel hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "second GDN restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    gdn.pack_slot_to_host(1, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != conv) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("cancel_restore broke GDN host staging for the second restore");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_rewrite_restore_skips_frontier_gdn(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("skip-cur");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> frontier_conv(gdn.conv_host_image_bytes(), 0xa1);
    std::vector<unsigned char> frontier_rec(gdn.recurrent_host_image_bytes(), 0xa2);
    std::vector<unsigned char> rewrite_conv(gdn.conv_host_image_bytes(), 0xb1);
    std::vector<unsigned char> rewrite_rec(gdn.recurrent_host_image_bytes(), 0xb2);
    gdn.unpack_slot_from_host(0, frontier_conv.data(), frontier_rec.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, rewrite_conv.data(), rewrite_rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto prompt   = text_prompt({5, 5, 5, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source                   = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.gdn                    = &gdn;
    source.gdn_current_slot      = 0;
    source.gdn_checkpoint_slot  = 1;
    source.rewrite_valid       = true;
    source.rewrite_kind         = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier     = 2;
    source.hash_c_valid        = true;
    source.hash_c               = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("skip-current capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("skip-current spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("skip-current match failed");
    }
    auto restore_one = [&](const q36::detail::DiskMatch& restore_match,
                            ninfer::PrefixReusePath reuse, std::uint32_t reuse_base,
                            std::int32_t current_slot, std::int32_t checkpoint_slot,
                            const char* hung) -> int {
        if (!disk.claim(restore_match.entry_id, restore_match.hash_f,
                         restore_match.execution_frontier, reuse_base,
                         reuse)) {
            return fail("skip-current claim failed");
        }
        auto dest = pool.reserve(2);
        dest.materialize_pages(1, ctx.stream);
        q36::detail::DiskRestoreTarget target;
        target.text                = &dest;
        target.text_pool           = &pool;
        target.text_dst_pages     = 1;
        target.gdn                 = &gdn;
        target.gdn_current_slot   = current_slot;
        target.gdn_checkpoint_slot = checkpoint_slot;
        target.reuse               = reuse;
        target.reuse_base         = reuse_base;
        target.stream              = ctx.copy_stream;
        disk.restore_device(restore_match.entry_id, target);
        try {
            if (const int rc = wait_restore_bounded(disk, ctx, hung); rc != 0) {
                disk.release(restore_match.entry_id);
                dest.release();
                return rc;
            }
        } catch (const std::exception& e) {
            disk.release(restore_match.entry_id);
            dest.release();
            std::cerr << hung << " threw: " << e.what() << '\n';
            return 1;
        }
        disk.cancel_restore();
        disk.release(restore_match.entry_id);
        dest.release();
        return 0;
    };

    gdn.zero_slot(2, ctx.stream);
    gdn.zero_slot(3, ctx.stream);
    ctx.synchronize_all();
    if (const int rc = restore_one(*match, ninfer::PrefixReusePath::AppendAtFrontier, 4, 2, 3,
                                      "append restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> conv_got(frontier_conv.size());
    std::vector<unsigned char> rec_got(frontier_rec.size());
    gdn.pack_slot_to_host(2, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != frontier_conv || rec_got != frontier_rec) {
        alloc.release();
        return fail("append restore skipped required frontier GDN");
    }
    gdn.pack_slot_to_host(3, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != rewrite_conv || rec_got != rewrite_rec) {
        alloc.release();
        return fail("append restore did not unpack rewrite GDN");
    }

    std::vector<unsigned char> poison_conv(frontier_conv.size(), 0xcc);
    std::vector<unsigned char> poison_rec(frontier_rec.size(), 0xcd);
    gdn.unpack_slot_from_host(2, poison_conv.data(), poison_rec.data(), ctx.stream);
    gdn.zero_slot(3, ctx.stream);
    ctx.synchronize_all();
    if (const int rc = restore_one(*match, ninfer::PrefixReusePath::RestoreTurnCheckpoint, 2, 2, 3,
                                      "rewrite restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    ctx.synchronize_all();
    gdn.pack_slot_to_host(2, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != poison_conv || rec_got != poison_rec) {
        alloc.release();
        return fail("rewrite restore unpacked wasted frontier GDN into current");
    }
    gdn.pack_slot_to_host(3, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != rewrite_conv || rec_got != rewrite_rec) {
        alloc.release();
        return fail("rewrite restore did not unpack rewrite GDN into checkpoint slot");
    }

    auto response_capture_prompt = text_prompt({6, 6, 6, 6});
    auto response_retained = response_capture_prompt;
    response_retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity response_identity;
    auto response_source = make_source(response_retained, response_identity, alloc, pool,
                                       ctx.copy_stream, 4);
    response_source.gdn = &gdn;
    response_source.gdn_current_slot = 0;
    response_source.gdn_checkpoint_slot = 1;
    response_source.rewrite_valid = true;
    response_source.rewrite_kind = q36::RewriteCheckpointKind::ResponseReplay;
    response_source.rewrite_frontier = 2;
    response_source.hash_c_valid = true;
    response_source.hash_c =
        q36::detail::prefix_hash_at(response_retained.token_ids, response_identity, 2);
    q36::detail::RamLadderHead response_ladder;
    response_ladder.frontier = 1;
    response_ladder.hash =
        q36::detail::prefix_hash_at(response_retained.token_ids, response_identity, 1);
    response_ladder.kind = q36::detail::ContextCheckpointKind::Ladder;
    response_ladder.conv = frontier_conv.data();
    response_ladder.conv_bytes = frontier_conv.size();
    response_ladder.recurrent = frontier_rec.data();
    response_ladder.recurrent_bytes = frontier_rec.size();
    response_source.ladder_heads = {response_ladder};
    const auto response_ram = capture_or_evict(ram, response_source);
    if (!response_ram) {
        alloc.release();
        return fail("response-checkpoint capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.note_ram_resident(*response_ram, 0);
    if (!disk.emergency_spill_ram(*response_ram)) {
        alloc.release();
        return fail("response-checkpoint spill failed");
    }
    const auto response_prompt = text_prompt({6, 6, 7, 7});
    const auto response_match =
        disk.plan_match(response_prompt, q36::detail::prefix_hash_chain(response_prompt));
    if (!response_match ||
        response_match->reuse != ninfer::PrefixReusePath::RestoreResponseCheckpoint) {
        alloc.release();
        return fail("response checkpoint did not plan the exact reuse path");
    }
    gdn.unpack_slot_from_host(2, poison_conv.data(), poison_rec.data(), ctx.stream);
    gdn.zero_slot(3, ctx.stream);
    ctx.synchronize_all();
    if (const int rc = restore_one(*response_match,
                                   ninfer::PrefixReusePath::RestoreResponseCheckpoint, 2, 2, 3,
                                   "response-checkpoint restore hung");
        rc != 0) {
        alloc.release();
        return rc;
    }
    ctx.synchronize_all();
    gdn.pack_slot_to_host(2, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != poison_conv || rec_got != poison_rec) {
        alloc.release();
        return fail("response restore unpacked wasted frontier GDN into current");
    }
    gdn.pack_slot_to_host(3, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != rewrite_conv || rec_got != rewrite_rec) {
        alloc.release();
        return fail("response restore did not unpack rewrite GDN into checkpoint slot");
    }
    alloc.release();
    return 0;
}

int test_pinned_state_h2d_matches_heap(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("pin-state");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache cyclic({cyclic_arena.base(), cyclic_arena.capacity()}, cyclic_layout);
    ninfer::LayoutBuilder rewrite_cyclic_builder;
    const auto rewrite_cyclic_layout =
        ninfer::plan_cyclic_kv_cache(rewrite_cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena rewrite_cyclic_arena(rewrite_cyclic_builder.finish(256));
    ninfer::CyclicKVCache rewrite_cyclic({rewrite_cyclic_arena.base(), rewrite_cyclic_arena.capacity()},
                                        rewrite_cyclic_layout);
    std::vector<unsigned char> frontier_conv(gdn.conv_host_image_bytes(), 0xa1);
    std::vector<unsigned char> frontier_rec(gdn.recurrent_host_image_bytes(), 0xa2);
    std::vector<unsigned char> rewrite_conv(gdn.conv_host_image_bytes(), 0xb1);
    std::vector<unsigned char> rewrite_rec(gdn.recurrent_host_image_bytes(), 0xb2);
    gdn.unpack_slot_from_host(0, frontier_conv.data(), frontier_rec.data(), ctx.stream);
    gdn.unpack_slot_from_host(1, rewrite_conv.data(), rewrite_rec.data(), ctx.stream);
    std::vector<unsigned char> cyclic_cur(cyclic.lane_host_bytes(), 0xa4);
    std::vector<unsigned char> cyclic_rw(cyclic.lane_host_bytes(), 0xb4);
    cyclic.copy_lane_from_host(cyclic_cur.data(), 0, ctx.stream);
    rewrite_cyclic.copy_lane_from_host(cyclic_rw.data(), 0, ctx.stream);
    ninfer::DeviceBuffer hid(4096);
    hid.fill(0xa3);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {4096});
    ninfer::DeviceBuffer hid_rw(4096);
    hid_rw.fill(0xb3);
    ninfer::Tensor hidden_rw(hid_rw.p, ninfer::DType::U8, {4096});
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto prompt   = text_prompt({5, 5, 5, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source                       = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.gdn                        = &gdn;
    source.gdn_current_slot         = 0;
    source.gdn_checkpoint_slot       = 1;
    source.tail_hidden                = &hidden;
    source.rewrite_checkpoint_hidden = &hidden_rw;
    source.rewrite_valid             = true;
    source.rewrite_kind               = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier            = 2;
    source.hash_c_valid              = true;
    source.hash_c                     = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.dflash_local               = &cyclic;
    source.dflash_lane                = 0;
    source.dflash_checkpoint            = &rewrite_cyclic;
    ctx.synchronize_all();
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("pin-state capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::DFlash,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn, &cyclic);
    cfg.hidden_bytes = 4096;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("pin-state spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("pin-state match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier, 4,
                     ninfer::PrefixReusePath::AppendAtFrontier)) {
        alloc.release();
        return fail("pin-state claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    gdn.zero_slot(2, ctx.stream);
    gdn.zero_slot(3, ctx.stream);
    ninfer::DeviceBuffer hid_out(4096);
    hid_out.fill(0x00);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {4096});
    ninfer::DeviceBuffer hid_out_rw(4096);
    hid_out_rw.fill(0x00);
    ninfer::Tensor hidden_out_rw(hid_out_rw.p, ninfer::DType::U8, {4096});
    std::vector<unsigned char> cyclic_zero(cyclic.lane_host_bytes(), 0);
    cyclic.copy_lane_from_host(cyclic_zero.data(), 0, ctx.stream);
    rewrite_cyclic.copy_lane_from_host(cyclic_zero.data(), 0, ctx.stream);
    ctx.synchronize_all();
    q36::detail::DiskRestoreTarget target;
    target.text                       = &dest;
    target.text_pool                  = &pool;
    target.text_dst_pages            = 1;
    target.gdn                        = &gdn;
    target.gdn_current_slot          = 2;
    target.gdn_checkpoint_slot      = 3;
    target.tail_hidden                = &hidden_out;
    target.rewrite_checkpoint_hidden = &hidden_out_rw;
    target.dflash_local               = &cyclic;
    target.dflash_checkpoint           = &rewrite_cyclic;
    target.dflash_lane                = 0;
    target.reuse                      = ninfer::PrefixReusePath::AppendAtFrontier;
    target.reuse_base                = 4;
    target.stream                     = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "pin-state restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "pin-state restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    std::vector<unsigned char> conv_got(frontier_conv.size());
    std::vector<unsigned char> rec_got(frontier_rec.size());
    gdn.pack_slot_to_host(2, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != frontier_conv || rec_got != frontier_rec) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state current GDN did not match heap decode (conv/rec overlay?)");
    }
    gdn.pack_slot_to_host(3, conv_got.data(), rec_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (conv_got != rewrite_conv || rec_got != rewrite_rec) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state rewrite GDN did not match heap decode");
    }
    std::vector<unsigned char> hid_got(4096);
    CUDA_CHECK(cudaMemcpy(hid_got.data(), hid_out.p, hid_got.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(hid_got.begin(), hid_got.end(), [](unsigned char c) { return c != 0xa3; })) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state hidden did not match heap decode");
    }
    CUDA_CHECK(cudaMemcpy(hid_got.data(), hid_out_rw.p, hid_got.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(hid_got.begin(), hid_got.end(), [](unsigned char c) { return c != 0xb3; })) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state rewrite hidden did not match heap decode");
    }
    std::vector<unsigned char> cyc_got(cyclic.lane_host_bytes());
    cyclic.copy_lane_to_host(0, cyc_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (cyc_got != cyclic_cur) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state cyclic did not match heap decode");
    }
    rewrite_cyclic.copy_lane_to_host(0, cyc_got.data(), ctx.stream);
    ctx.synchronize_all();
    if (cyc_got != cyclic_rw) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("pin-state rewrite cyclic did not match heap decode");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_c1_order_disk_b_ram_c(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("c1-api");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, alloc, 0xb0);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {7, 7, 7, 7});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("C=1 API spill of B failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt({7, 7, 7, 7}),
                        q36::detail::prefix_hash_chain(text_prompt({7, 7, 7, 7})));
    if (!match_b) {
        alloc.release();
        return fail("C=1 API disk B match failed");
    }
    ram.claim(ram_b);
    ram.consume(ram_b);
    fill_logical_pages(pool, alloc, 0xc0);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, {8, 8, 8, 8});
    disk.note_ram_resident(ram_c, 0);
    fill_logical_pages(pool, alloc, 0xd0);
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, {9, 9, 9, 9});
    disk.note_ram_resident(ram_d, 0);
    disk.request_idle_spill();
    disk.claim(match_b->entry_id);
    disk.prefetch_window(match_b->entry_id, 1, 0);
    if (!disk.emergency_spill_ram(ram_c)) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("C=1 API emergency spill of C failed");
    }
    disk.cancel_idle_spill();
    const auto peek = ram.peek_oldest_unpinned();
    if (!peek) {
        disk.release(match_b->entry_id);
        alloc.release();
        return fail("C=1 API peek was empty after cancel_idle_spill");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "C=1 API restore of B hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "C=1 API restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<unsigned char> got(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, got.data(), ctx.stream);
    ctx.synchronize_all();
    const bool looks_c =
        std::all_of(got.begin(), got.end(), [](unsigned char c) { return c == 0xc0; });
    if (looks_c) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("C=1 restore of B used C's KV payload");
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_recapture_keeps_ladders(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("recap-ladders");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    q36::detail::RamLadderHead l1 = rollback;
    l1.frontier                   = 6;
    l1.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 6);
    l1.kind = q36::detail::ContextCheckpointKind::Ladder;
    q36::detail::RamLadderHead l2 = l1;
    l2.frontier                   = 5;
    l2.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    source.ladder_heads = {rollback, l1, l2};
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("recapture-ladder capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("recapture-ladder spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("recapture-ladder match failed");
    }
    auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("recapture-ladder load_host missed the live entry");
    }
    if (!disk.populate_checkpoint_images(*host)) {
        alloc.release();
        return fail("recapture-ladder populate failed");
    }
    ram.claim(*id);
    ram.consume(*id);
    if (host->ladder_images.size() != 3) {
        alloc.release();
        return fail("recapture-ladder populate did not return rollback + two ladders");
    }
    std::vector<q36::detail::RamLadderHead> recap_heads;
    recap_heads.reserve(host->ladder_images.size());
    for (const auto& image : host->ladder_images) {
        q36::detail::RamLadderHead head;
        head.frontier        = image.frontier;
        head.hash            = image.hash;
        head.kind            = image.kind;
        head.conv            = image.conv ? image.conv->data() : nullptr;
        head.recurrent       = image.recurrent ? image.recurrent->data() : nullptr;
        head.hidden          = image.hidden ? image.hidden->data() : nullptr;
        head.dflash          = image.dflash ? image.dflash->data() : nullptr;
        head.conv_bytes      = image.conv ? image.conv->size() : 0;
        head.recurrent_bytes = image.recurrent ? image.recurrent->size() : 0;
        head.hidden_bytes    = image.hidden ? image.hidden->size() : 0;
        head.dflash_bytes    = image.dflash ? image.dflash->size() : 0;
        recap_heads.push_back(head);
    }
    source.ladder_heads = recap_heads;
    source.disk_entry_id           = match->entry_id;
    auto recaptured                = capture_or_evict(ram, source);
    if (!recaptured) {
        alloc.release();
        return fail("recapture with ladder images failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    ram.set_disk_entry_id(*recaptured, match->entry_id);
    disk.note_ram_resident(*recaptured, match->entry_id);
    if (!disk.emergency_spill_ram(*recaptured)) {
        alloc.release();
        return fail("recapture-ladder second spill failed");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.rollback.frontier != 3 || meta.ladders[0].frontier != 6 ||
        meta.ladders[1].frontier != 5 || meta.rollback.hidden_id == 0 ||
        meta.ladders[0].hidden_id == 0) {
        alloc.release();
        return fail("recapture spill lost rollback or ladder slots");
    }
    alloc.release();
    return 0;
}

int test_mixed_zstd_and_raw_hidden(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("mixed-codec");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid_a(256);
    std::vector<unsigned char> seed_a(256, 0xaa);
    hid_a.copy_from_host(seed_a.data(), seed_a.size());
    ninfer::Tensor hidden_a(hid_a.p, ninfer::DType::U8, {256});
    ninfer::DeviceBuffer hid_b(256);
    std::vector<unsigned char> seed_b(256, 0xbb);
    hid_b.copy_from_host(seed_b.data(), seed_b.size());
    ninfer::Tensor hidden_b(hid_b.p, ninfer::DType::U8, {256});
    auto prompt_a   = text_prompt({1, 1, 1, 1});
    auto retained_a = prompt_a;
    retained_a.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_a;
    auto source_a        = make_source(retained_a, identity_a, alloc, pool, ctx.copy_stream, 4);
    source_a.tail_hidden = &hidden_a;
    auto id_a            = capture_or_evict(ram, source_a);
    if (!id_a) {
        alloc.release();
        return fail("mixed-codec A capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    {
        const auto view = ram.host_kv(*id_a);
        const auto* hid = static_cast<const unsigned char*>(view.tail_hidden);
        if (view.hidden_bytes != 256 || hid == nullptr || hid[0] != 0xaa) {
            alloc.release();
            return fail("mixed-codec RAM capture of A did not store 0xaa hidden");
        }
    }
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Zstd);
    cfg.hidden_bytes = 256;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id_a, 0);
    if (!disk.emergency_spill_ram(*id_a)) {
        alloc.release();
        return fail("mixed-codec A spill failed");
    }
    auto prompt_b   = text_prompt({2, 2, 2, 2});
    auto retained_b = prompt_b;
    retained_b.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity_b;
    auto source_b        = make_source(retained_b, identity_b, alloc, pool, ctx.copy_stream, 4);
    source_b.tail_hidden = &hidden_b;
    auto id_b            = capture_or_evict(ram, source_b);
    if (!id_b) {
        alloc.release();
        return fail("mixed-codec B capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    disk.test_force_zstd_fail();
    disk.note_ram_resident(*id_b, 0);
    if (!disk.emergency_spill_ram(*id_b)) {
        alloc.release();
        return fail("mixed-codec B raw spill failed");
    }
    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    const auto match_b = disk.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_a || !match_b) {
        alloc.release();
        return fail("mixed-codec match failed");
    }
    auto restore_hidden = [&](std::uint64_t entry, unsigned char expect) -> int {
        ninfer::DeviceBuffer out(256);
        out.fill(0);
        ctx.synchronize_all();
        ninfer::Tensor tout(out.p, ninfer::DType::U8, {256});
        auto dest = pool.reserve(2);
        dest.materialize_pages(1, ctx.stream);
        q36::detail::DiskRestoreTarget target;
        target.text           = &dest;
        target.text_pool      = &pool;
        target.text_dst_pages = 1;
        target.tail_hidden    = &tout;
        target.stream         = ctx.copy_stream;
        disk.claim(entry);
        disk.restore_device(entry, target);
        try {
            if (const int rc = wait_restore_bounded(disk, ctx, "mixed-codec restore hung");
                rc != 0) {
                disk.release(entry);
                dest.release();
                return rc;
            }
        } catch (const std::exception& e) {
            disk.release(entry);
            dest.release();
            std::cerr << "mixed-codec restore threw: " << e.what() << '\n';
            return 1;
        }
        ctx.synchronize_all();
        std::vector<unsigned char> got(256);
        CUDA_CHECK(cudaMemcpy(got.data(), out.p, got.size(), cudaMemcpyDeviceToHost));
        disk.cancel_restore();
        disk.release(entry);
        dest.release();
        if (std::any_of(got.begin(), got.end(),
                        [expect](unsigned char c) { return c != expect; })) {
            std::cerr << "mixed-codec hidden mismatch expect=" << static_cast<int>(expect)
                      << " got=" << static_cast<int>(got[0]) << '\n';
            return fail("mixed-codec hidden mismatch");
        }
        return 0;
    };
    if (const int rc = restore_hidden(match_a->entry_id, 0xaa); rc != 0) {
        alloc.release();
        return rc;
    }
    if (const int rc = restore_hidden(match_b->entry_id, 0xbb); rc != 0) {
        alloc.release();
        return rc;
    }
    alloc.release();
    return 0;
}

int test_dflash_open_skips_missing_cyclic(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("dflash-skip");
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache cyclic({cyclic_arena.base(), cyclic_arena.capacity()}, cyclic_layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto prompt   = text_prompt({1, 2, 3, 4});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source         = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.dflash_local = &cyclic;
    source.dflash_lane  = 0;
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("dflash-skip capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::DFlash,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, nullptr, &cyclic);
    std::uint64_t entry = 0;
    std::uint64_t cyclic_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("dflash-skip spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("dflash-skip match failed");
        }
        entry     = match->entry_id;
        cyclic_id = disk.test_load_meta(entry).current_cyclic_id;
        if (cyclic_id == 0) {
            alloc.release();
            return fail("dflash-skip spill stored no cyclic object");
        }
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry) / "meta.bin";
    auto bytes           = [&] {
        std::ifstream in(meta_path, std::ios::binary);
        return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    }();
    constexpr std::size_t kGdnOffset    = 100;
    constexpr std::size_t kCyclicOffset = 116;
    if (bytes.size() < kCyclicOffset + 8) {
        alloc.release();
        return fail("dflash-skip meta.bin too small to patch");
    }
    std::uint64_t stored_cyclic = 0;
    std::memcpy(&stored_cyclic, bytes.data() + kCyclicOffset, 8);
    if (stored_cyclic != cyclic_id) {
        alloc.release();
        return fail("dflash-skip current_cyclic_id is not at meta offset 116");
    }
    std::memset(bytes.data() + kCyclicOffset, 0, 8);
    std::uint64_t gdn_id = 0;
    std::memcpy(&gdn_id, bytes.data() + kGdnOffset, 8);
    if (gdn_id == 0) {
        gdn_id = 1;
        std::memcpy(bytes.data() + kGdnOffset, &gdn_id, 8);
    }
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        alloc.release();
        return fail("missing cyclic_id remained hittable");
    }
    if (reopened.snapshot().drops == 0) {
        alloc.release();
        return fail("missing cyclic_id did not increment drops");
    }
    alloc.release();
    return 0;
}

int test_mtp_backend_cow_and_dflash2(ninfer::DeviceContext& ctx) {
    auto text_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto backend_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool backend({backend_arena.base(), backend_arena.capacity()},
                                backend_plan.layout);
    auto text_alloc    = text.reserve(4);
    auto backend_alloc = backend.reserve(4);
    text_alloc.materialize_pages(3, ctx.stream);
    backend_alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(text, text_alloc, 1);
    fill_logical_pages(backend, backend_alloc, 2);
    TmpDir dir("mtp-cow");
    q36::detail::KVRamCache ram(64ULL << 20);
    std::vector<ninfer::TokenId> t65(65, 42);
    auto prompt   = text_prompt(t65);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source         = make_source(retained, identity, text_alloc, text, ctx.copy_stream, 65);
    source.mtp_kv_valid = 64;
    source.backend      = &backend_alloc;
    source.backend_pool = &backend;
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=65 capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, text, &backend, ninfer::SpeculativeBackend::Mtp,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=65 spill failed");
    }
    const auto match65 = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match65) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=65 match failed");
    }
    const auto back_a = disk.test_backend_page_ids(match65->entry_id);
    if (back_a.size() != 1) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=65 did not store one backend page");
    }
    std::vector<ninfer::TokenId> t129 = t65;
    t129.push_back(0);
    t129.resize(129, 42);
    auto prompt129   = text_prompt(t129);
    auto retained129 = prompt129;
    retained129.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity129;
    auto source129         = make_source(retained129, identity129, text_alloc, text, ctx.copy_stream,
                                         129);
    source129.mtp_kv_valid = 128;
    source129.backend      = &backend_alloc;
    source129.backend_pool = &backend;
    auto id129             = capture_or_evict(ram, source129);
    if (!id129) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=129 capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    ram.set_disk_entry_id(*id129, match65->entry_id);
    disk.note_ram_resident(*id129, match65->entry_id);
    if (!disk.emergency_spill_ram(*id129)) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP F=129 extend failed");
    }
    const auto back_b = disk.test_backend_page_ids(match65->entry_id);
    if (back_b.size() != 2 || back_b[0] != back_a[0] || back_b[1] == back_a[0]) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP aligned extend cloned or dropped backend page 0");
    }
    text_alloc.release();
    backend_alloc.release();

    TmpDir dir2("dflash2");
    q36::detail::KVRamCache ram2(32ULL << 20);
    auto text_alloc2 = text.reserve(4);
    text_alloc2.materialize_pages(3, ctx.stream);
    auto cfg2 = disk_config(dir2.path, ram2, text, nullptr, ninfer::SpeculativeBackend::DFlash,
                            32ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk2(cfg2);
        std::vector<ninfer::TokenId> aligned(64, 3);
        const auto ram_a = capture_tokens(ram2, text, text_alloc2, ctx, aligned);
        disk2.note_ram_resident(ram_a, 0);
        if (!disk2.emergency_spill_ram(ram_a)) {
            text_alloc2.release();
            return fail("DFlash2 create spill failed");
        }
        const auto match =
            disk2.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match || !disk2.test_backend_page_ids(match->entry_id).empty()) {
            text_alloc2.release();
            return fail("DFlash2 stored backend pages");
        }
        std::vector<ninfer::TokenId> ext = aligned;
        ext.resize(128, 4);
        const auto ram_e = capture_tokens(ram2, text, text_alloc2, ctx, ext);
        ram2.set_disk_entry_id(ram_e, match->entry_id);
        disk2.note_ram_resident(ram_e, match->entry_id);
        if (!disk2.emergency_spill_ram(ram_e)) {
            text_alloc2.release();
            return fail("DFlash2 extend spill failed");
        }
        if (!disk2.test_backend_page_ids(match->entry_id).empty()) {
            text_alloc2.release();
            return fail("DFlash2 extend grew a backend pool");
        }
        const auto live_match =
            disk2.plan_match(text_prompt(ext), q36::detail::prefix_hash_chain(text_prompt(ext)));
        if (!live_match) {
            text_alloc2.release();
            return fail("DFlash2 extend did not stay hittable");
        }
        disk2.claim(live_match->entry_id, live_match->hash_f, live_match->execution_frontier);
        {
            auto dest = text.reserve(4);
            dest.materialize_pages(2, ctx.stream);
            q36::detail::DiskRestoreTarget target;
            target.text              = &dest;
            target.text_pool         = &text;
            target.text_dst_pages    = 2;
            target.backend_dst_pages = 0;
            target.stream            = ctx.copy_stream;
            disk2.restore_device(live_match->entry_id, target);
            try {
                if (const int rc = wait_restore_bounded(disk2, ctx, "DFlash2 restore hung"); rc != 0) {
                    disk2.release(live_match->entry_id);
                    dest.release();
                    text_alloc2.release();
                    return rc;
                }
            } catch (const std::exception& e) {
                disk2.release(live_match->entry_id);
                dest.release();
                text_alloc2.release();
                std::cerr << "DFlash2 restore threw: " << e.what() << '\n';
                return 1;
            }
            disk2.release(live_match->entry_id);
            dest.release();
        }
        disk2.wait_idle_and_fsync();
    }
    {
        q36::detail::KVDiskCache reopened(cfg2);
        std::vector<ninfer::TokenId> ext(64, 3);
        ext.resize(128, 4);
        const auto hit =
            reopened.plan_match(text_prompt(ext), q36::detail::prefix_hash_chain(text_prompt(ext)));
        if (!hit) {
            text_alloc2.release();
            return fail("DFlash2 reopen dropped the entry");
        }
        if (!reopened.test_backend_page_ids(hit->entry_id).empty()) {
            text_alloc2.release();
            return fail("DFlash2 reopen invented backend pages");
        }
    }
    text_alloc2.release();
    return 0;
}

int test_branch_share_point_l(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("branch-l");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> parent(128, 7);
    const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent);
    disk.note_ram_resident(ram_p, 0);
    if (!disk.emergency_spill_ram(ram_p)) {
        alloc.release();
        return fail("branch-L parent spill failed");
    }
    const auto match_p =
        disk.plan_match(text_prompt(parent), q36::detail::prefix_hash_chain(text_prompt(parent)));
    if (!match_p) {
        alloc.release();
        return fail("branch-L parent match failed");
    }
    const auto pages_p = disk.test_main_page_ids(match_p->entry_id);
    std::vector<ninfer::TokenId> child(parent.begin(), parent.begin() + 64);
    child.push_back(99);
    child.resize(70, 11);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
    ram.set_disk_entry_id(ram_c, match_p->entry_id);
    disk.note_ram_resident(ram_c, match_p->entry_id);
    if (!disk.emergency_spill_ram(ram_c)) {
        alloc.release();
        return fail("branch-L child spill failed");
    }
    const auto match_c =
        disk.plan_match(text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child)));
    if (!match_c || match_c->entry_id == match_p->entry_id) {
        alloc.release();
        return fail("branch-L did not create a child");
    }
    const auto pages_c = disk.test_main_page_ids(match_c->entry_id);
    if (pages_c.size() != 2 || pages_c[0] != pages_p[0] || pages_c[1] == pages_p[1]) {
        alloc.release();
        return fail("branch-L did not share page 0 at L=64");
    }
    if (disk.test_main_page_ids(match_p->entry_id) != pages_p) {
        alloc.release();
        return fail("branch-L mutated the parent page list");
    }
    alloc.release();
    return 0;
}

int test_fingerprint_and_location_file(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("fp-file");
    q36::detail::KVRamCache ram(8ULL << 20);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           16ULL << 20, 4096);
    {
        q36::detail::KVDiskCache first(cfg);
    }
    cfg.fingerprint.weights_id = "other-weights";
    bool threw                 = false;
    try {
        q36::detail::KVDiskCache mismatch(cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("weights_id") != std::string::npos;
    }
    if (!threw) { return fail("fingerprint mismatch did not name weights_id"); }
    const auto file_path = dir.path / "not-a-dir";
    {
        std::ofstream out(file_path);
        out << "x";
    }
    q36::detail::DiskOpenConfig file_cfg = cfg;
    file_cfg.fingerprint.weights_id      = "groupwise-int";
    file_cfg.location                    = file_path;
    threw                                = false;
    try {
        q36::detail::KVDiskCache file_disk(file_cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("directory") != std::string::npos;
    }
    if (!threw) { return fail("file location did not fail construction"); }
    try {
        ninfer::validate_kv_disk_options(0, 1, "p");
        return fail("disk without RAM did not throw");
    } catch (const std::invalid_argument&) {}
    try {
        ninfer::validate_kv_disk_options(1, 0, "p");
        return fail("location without capacity did not throw");
    } catch (const std::invalid_argument&) {}
    (void)ctx;
    return 0;
}

int test_restore_second_entry_without_cancel(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("restore-b2");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_b(192, 13);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("B2 spill of B failed");
    }
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> tokens_d(192, 17);
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, tokens_d);
    disk.note_ram_resident(ram_d, 0);
    if (!disk.emergency_spill_ram(ram_d)) {
        alloc.release();
        return fail("B2 spill of D failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    const auto match_d =
        disk.plan_match(text_prompt(tokens_d), q36::detail::prefix_hash_chain(text_prompt(tokens_d)));
    if (!match_b || !match_d || match_b->entry_id == match_d->entry_id) {
        alloc.release();
        return fail("B2 did not produce two disk entries");
    }
    disk.claim(match_b->entry_id);
    disk.prefetch_window(match_b->entry_id, 3, 0);
    auto dest = pool.reserve(4);
    dest.materialize_pages(3, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 3;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "B2 restore of B hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "B2 restore of B threw: " << e.what() << '\n';
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    try {
        disk.pump_restore(ctx.copy_stream);
    } catch (const std::exception& e) {
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "pump after successful restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.release(match_b->entry_id);
    disk.claim(match_d->entry_id);
    disk.restore_device(match_d->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "B2 restore of D hung"); rc != 0) {
            disk.release(match_d->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match_d->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "B2 restore of D threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<unsigned char> src(page_bytes);
    std::vector<unsigned char> got(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, 0, src.data(), ctx.stream);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, got.data(), ctx.stream);
    ctx.synchronize_all();
    if (src != got) {
        disk.release(match_d->entry_id);
        dest.release();
        alloc.release();
        return fail("second restore reused the previous entry's window pages");
    }
    disk.release(match_d->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_cancel_restore_then_restore_other(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("cancel-h2d");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_b(192, 13);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("cancel-h2d spill of B failed");
    }
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> tokens_d(192, 17);
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, tokens_d);
    disk.note_ram_resident(ram_d, 0);
    if (!disk.emergency_spill_ram(ram_d)) {
        alloc.release();
        return fail("cancel-h2d spill of D failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    const auto match_d =
        disk.plan_match(text_prompt(tokens_d), q36::detail::prefix_hash_chain(text_prompt(tokens_d)));
    if (!match_b || !match_d || match_b->entry_id == match_d->entry_id) {
        alloc.release();
        return fail("cancel-h2d did not produce two disk entries");
    }
    disk.claim(match_b->entry_id);
    auto dest = pool.reserve(4);
    dest.materialize_pages(3, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 3;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_b->entry_id, target);
    disk.pump_restore(ctx.copy_stream);
    disk.pump_restore(ctx.copy_stream);
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    disk.claim(match_d->entry_id);
    disk.restore_device(match_d->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "cancel-h2d restore of D hung"); rc != 0) {
            disk.release(match_d->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.release(match_d->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "cancel-h2d restore of D threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<unsigned char> src(page_bytes);
    std::vector<unsigned char> got(page_bytes);
    ninfer::pack_paged_kv_logical_page_to_host(alloc, pool, 0, src.data(), ctx.stream);
    ninfer::pack_paged_kv_logical_page_to_host(dest, pool, 0, got.data(), ctx.stream);
    ctx.synchronize_all();
    if (src != got) {
        disk.release(match_d->entry_id);
        dest.release();
        alloc.release();
        return fail("cancel-h2d restore mixed the cancelled window into D");
    }
    disk.release(match_d->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_refresh_does_not_exceed_capacity(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("refresh-cap");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(64, 4);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 64);
    ninfer::DeviceBuffer hid(256);
    std::vector<unsigned char> seed(256, 0x11);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    source.tail_hidden = &hidden;
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("refresh-cap first capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::size_t used = 0;
    std::uint64_t entry = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("refresh-cap first spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("refresh-cap first match failed");
        }
        entry = match->entry_id;
        used = disk.snapshot().used_bytes;
        if (used == 0) {
            alloc.release();
            return fail("refresh-cap first spill billed no unique bytes");
        }
        disk.wait_idle_and_fsync();
    }
    cfg.capacity_bytes = used;
    ram.claim(*id);
    ram.consume(*id);
    std::vector<unsigned char> hidden_host(256, 0x22);
    auto source2 = make_source(retained, identity, alloc, pool, ctx.copy_stream, 64);
    source2.tail_hidden = &hidden;
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 256;
    q36::detail::RamLadderHead l1 = rollback;
    l1.frontier                   = 4;
    l1.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    l1.kind = q36::detail::ContextCheckpointKind::Ladder;
    q36::detail::RamLadderHead l2 = l1;
    l2.frontier                   = 5;
    l2.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    source2.ladder_heads = {rollback, l1, l2};
    auto id2 = capture_or_evict(ram, source2);
    if (!id2) {
        alloc.release();
        return fail("refresh-cap ladder capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    q36::detail::KVDiskCache disk(cfg);
    ram.set_disk_entry_id(*id2, entry);
    disk.note_ram_resident(*id2, entry);
    (void)disk.emergency_spill_ram(*id2);
    if (disk.snapshot().used_bytes > cfg.capacity_bytes) {
        alloc.release();
        return fail("refresh credited retained KV pages and exceeded capacity");
    }
    alloc.release();
    return 0;
}

int test_claim_rejects_checkpoint_only_refresh(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-ckpt");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("claim-ckpt capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("claim-ckpt first spill failed");
    }
    auto prompt3 = text_prompt({3, 3, 3});
    const auto planned = disk.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
    if (!planned || planned->reuse_base != 3) {
        alloc.release();
        return fail("claim-ckpt did not plan the rollback head");
    }
    const auto planned_id   = planned->entry_id;
    const auto planned_hash = planned->hash_f;
    const auto planned_f    = planned->execution_frontier;
    ram.claim(*id);
    ram.consume(*id);
    const auto ram2 = capture_tokens(ram, pool, alloc, ctx, tokens);
    ram.set_disk_entry_id(ram2, planned_id);
    disk.note_ram_resident(ram2, planned_id);
    if (!disk.emergency_spill_ram(ram2)) {
        alloc.release();
        return fail("claim-ckpt refresh spill failed");
    }
    if (disk.test_load_meta(planned_id).rollback.frontier != 0) {
        alloc.release();
        return fail("claim-ckpt refresh kept the rollback slot");
    }
    if (disk.claim(planned_id, planned_hash, planned_f, planned->reuse_base)) {
        disk.release(planned_id);
        alloc.release();
        return fail("claim accepted a checkpoint head Refresh had removed");
    }
    const auto live = disk.test_load_meta(planned_id);
    if (!disk.claim(planned_id, live.hash_f, live.execution_frontier, live.execution_frontier)) {
        alloc.release();
        return fail("claim of the live frontier head failed");
    }
    disk.release(planned_id);
    alloc.release();
    return 0;
}

int test_claim_rejects_replaced_head_at_same_frontier(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-kind");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead ladder;
    ladder.frontier     = 4;
    ladder.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    ladder.kind         = q36::detail::ContextCheckpointKind::Ladder;
    ladder.hidden       = hidden_host.data();
    ladder.hidden_bytes = 64;
    source.ladder_heads  = {ladder};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("claim-kind capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("claim-kind first spill failed");
    }
    auto prompt4 = text_prompt({3, 3, 3, 3});
    const auto planned = disk.plan_match(prompt4, q36::detail::prefix_hash_chain(prompt4));
    if (!planned || planned->reuse_base != 4 ||
        planned->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        alloc.release();
        return fail("claim-kind did not plan the ladder head");
    }
    const auto planned_id   = planned->entry_id;
    const auto planned_hash = planned->hash_f;
    const auto planned_f    = planned->execution_frontier;
    ram.claim(*id);
    ram.consume(*id);
    q36::detail::RamLadderHead rollback = ladder;
    rollback.kind                      = q36::detail::ContextCheckpointKind::TurnRollback;
    auto source2                        = make_source(retained, identity, alloc, pool,
                                 ctx.copy_stream, 8);
    source2.tail_hidden                 = &hidden;
    source2.ladder_heads              = {rollback};
    auto id2 = capture_or_evict(ram, source2);
    if (!id2) {
        alloc.release();
        return fail("claim-kind refresh capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    ram.set_disk_entry_id(*id2, planned_id);
    disk.note_ram_resident(*id2, planned_id);
    if (!disk.emergency_spill_ram(*id2)) {
        alloc.release();
        return fail("claim-kind refresh spill failed");
    }
    if (disk.claim(planned_id, planned_hash, planned_f, planned->reuse_base, planned->reuse)) {
        disk.release(planned_id);
        alloc.release();
        return fail("claim accepted a different head kind at the planned frontier");
    }
    const auto live = disk.test_load_meta(planned_id);
    if (!disk.claim(planned_id, live.hash_f, live.execution_frontier, 4,
                    ninfer::PrefixReusePath::RestoreTurnRollback)) {
        alloc.release();
        return fail("claim of the replacement rollback head failed");
    }
    disk.release(planned_id);
    alloc.release();
    return 0;
}

int test_invalid_checkpoint_kind_is_skipped(ninfer::DeviceContext& ctx,
                                              ninfer::PagedKVPool& pool) {
    TmpDir dir("bad-kind");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("bad-kind capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t entry = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("bad-kind spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("bad-kind match failed");
        }
        entry = match->entry_id;
        disk.wait_idle_and_fsync();
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry) / "meta.bin";
    auto bytes            = [&] {
        std::ifstream in(meta_path, std::ios::binary);
        return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    }();
    bool patched = false;
    for (std::size_t i = 0; i + 8 <= bytes.size(); ++i) {
        std::uint32_t frontier = 0;
        std::uint32_t kind    = 0;
        std::memcpy(&frontier, bytes.data() + i, 4);
        std::memcpy(&kind, bytes.data() + i + 4, 4);
        if (frontier == 3 && kind == 1) {
            kind = 99;
            std::memcpy(bytes.data() + i + 4, &kind, 4);
            patched = true;
            break;
        }
    }
    if (!patched) {
        alloc.release();
        return fail("bad-kind did not find the rollback kind field");
    }
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("invalid checkpoint kind remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("invalid checkpoint kind did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "bad-kind reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_refresh_does_not_evict_sibling(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("refresh-sib");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens_a(64, 4);
    std::vector<ninfer::TokenId> tokens_b(64, 7);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::size_t used = 0;
    std::uint64_t entry_a = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("refresh-sib spill of A failed");
        }
        const auto match_a =
            disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
        if (!match_a) {
            alloc.release();
            return fail("refresh-sib match of A failed");
        }
        entry_a = match_a->entry_id;
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("refresh-sib spill of B failed");
        }
        used = disk.snapshot().used_bytes;
        disk.wait_idle_and_fsync();
    }
    cfg.capacity_bytes = used;
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_a2 = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    ram.set_disk_entry_id(ram_a2, entry_a);
    disk.note_ram_resident(ram_a2, entry_a);
    if (!disk.emergency_spill_ram(ram_a2)) {
        alloc.release();
        return fail("refresh-sib refresh of A failed");
    }
    const auto hit_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    if (!hit_b) {
        alloc.release();
        return fail("refresh charged retained pages and evicted sibling B");
    }
    if (disk.snapshot().used_bytes > cfg.capacity_bytes) {
        alloc.release();
        return fail("refresh unique_bytes exceeded capacity");
    }
    alloc.release();
    return 0;
}

int test_raw_state_unequal_codec_bytes_is_skipped(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("raw-uneq");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hid(256);
    std::vector<unsigned char> seed(256, 0xcd);
    hid.copy_from_host(seed.data(), seed.size());
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt({2, 3, 4, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("raw-uneq capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t hidden_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("raw-uneq spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("raw-uneq match failed");
        }
        hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
        if (hidden_id == 0) {
            alloc.release();
            return fail("raw-uneq stored no hidden object");
        }
        disk.wait_idle_and_fsync();
    }
    std::vector<std::uint8_t> blob(q36::detail::kDiskCodecHeaderBytes);
    if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                            blob.data(), blob.size())) {
        alloc.release();
        return fail("raw-uneq state blob too small to patch");
    }
    blob[0] = static_cast<std::uint8_t>(q36::detail::DiskCodec::Raw);
    const std::uint64_t unc = 256;
    const std::uint64_t cmp = 128;
    std::memcpy(blob.data() + 4, &unc, 8);
    std::memcpy(blob.data() + 12, &cmp, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                             blob.data(), blob.size())) {
        alloc.release();
        return fail("raw-uneq could not patch packed state");
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("raw unequal codec state remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("raw unequal codec state did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "raw-uneq reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_gdn_checkpoint_without_hidden_is_not_selected(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("gdn-nohid");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x42);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source             = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    source.gdn              = &gdn;
    source.gdn_current_slot = 0;
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 3;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv.data();
    rollback.recurrent       = rec.data();
    rollback.conv_bytes      = conv.size();
    rollback.recurrent_bytes = rec.size();
    source.ladder_heads      = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("gdn-nohid capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("gdn-nohid spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("gdn-nohid frontier match failed");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.rollback.gdn_id == 0 || meta.rollback.hidden_id != 0) {
        alloc.release();
        return fail("gdn-nohid did not persist GDN-only rollback");
    }
    auto prompt3 = text_prompt({3, 3, 3});
    const auto ckpt = disk.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
    if (ckpt && ckpt->reuse_base == 3) {
        alloc.release();
        return fail("gdn-nohid advertised a hidden-less rollback head");
    }
    auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("gdn-nohid load_host missed the live entry");
    }
    for (const auto& slot : host->ladders) {
        if (slot.frontier == 3) {
            alloc.release();
            return fail("gdn-nohid load_host advertised the incomplete rollback");
        }
    }
    if (!disk.populate_checkpoint_images(*host)) {
        alloc.release();
        return fail("gdn-nohid populate failed on a skipped incomplete rollback");
    }
    for (const auto& image : host->ladder_images) {
        if (image.frontier == 3) {
            alloc.release();
            return fail("gdn-nohid populate installed a hidden-less rollback");
        }
    }
    alloc.release();
    return 0;
}

int test_load_host_misses_evicted_entry(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("load-evict");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, {1, 1, 1, 1});
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("load-evict spill of A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, {2, 2, 2, 2});
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("load-evict spill of B failed");
    }
    const auto prompt_a = text_prompt({1, 1, 1, 1});
    const auto match_a = disk.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!match_a) {
        alloc.release();
        return fail("load-evict match of A failed");
    }
    const auto id_a = match_a->entry_id;
    if (!disk.test_fifo_evict_one()) {
        alloc.release();
        return fail("load-evict fifo evict failed");
    }
    std::optional<q36::detail::DiskRestoredHost> host;
    try {
        host = disk.load_host(id_a);
    } catch (...) {
        alloc.release();
        return fail("load_host threw after capacity eviction");
    }
    if (host) {
        alloc.release();
        return fail("load_host returned a host for an evicted entry");
    }
    alloc.release();
    return 0;
}

int test_wrong_gdn_decoded_size_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("gdn-size");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x61);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x62);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto prompt   = text_prompt({5, 5, 5, 5});
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source             = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.gdn              = &gdn;
    source.gdn_current_slot = 0;
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("gdn-size capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    std::uint64_t gdn_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("gdn-size spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("gdn-size match failed");
        }
        gdn_id = disk.test_load_meta(match->entry_id).current_gdn_id;
        if (gdn_id == 0) {
            alloc.release();
            return fail("gdn-size stored no current GDN object");
        }
        disk.wait_idle_and_fsync();
    }
    std::vector<std::uint8_t> blob(q36::detail::kDiskCodecHeaderBytes);
    if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State, gdn_id,
                            blob.data(), blob.size())) {
        alloc.release();
        return fail("gdn-size blob too small to patch");
    }
    const std::uint64_t unc = 1;
    std::memcpy(blob.data() + 4, &unc, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, gdn_id,
                             blob.data(), blob.size())) {
        alloc.release();
        return fail("gdn-size could not patch packed state");
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("wrong GDN decoded size remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("wrong GDN decoded size was not dropped");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "gdn-size reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_plan_match_does_not_wait_on_manifest(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("plan-manifest");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(64, 6);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    disk.test_set_manifest_io_stall_ms(250);
    std::atomic<bool> spilled{false};
    std::thread spiller([&] { spilled.store(disk.emergency_spill_ram(ram_id)); });
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_manifest_io_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_manifest_io_entered()) {
        disk.test_set_manifest_io_stall_ms(0);
        spiller.join();
        alloc.release();
        return fail("manifest I/O stall was not reached");
    }
    const auto t0        = std::chrono::steady_clock::now();
    const auto unrelated = text_prompt({1, 2, 3, 4});
    (void)disk.plan_match(unrelated, q36::detail::prefix_hash_chain(unrelated));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    disk.test_set_manifest_io_stall_ms(0);
    spiller.join();
    if (elapsed > std::chrono::milliseconds(100)) {
        alloc.release();
        return fail("plan_match waited on manifest fsync");
    }
    if (!spilled.load()) {
        alloc.release();
        return fail("stalled manifest spill failed");
    }
    alloc.release();
    return 0;
}

int test_disk_fifo_evict_clears_ram_durable(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("ram-durable");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(64, 9);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("durable-clear spill failed");
    }
    if (!disk.ram_is_durable(ram_id)) {
        alloc.release();
        return fail("spill did not mark RAM durable");
    }
    const auto prompt = text_prompt(tokens);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("durable-clear match failed");
    }
    if (ram.disk_entry_id(ram_id) != match->entry_id) {
        alloc.release();
        return fail("RAM ticket did not name the spilled entry");
    }
    if (!disk.test_fifo_evict_one()) {
        alloc.release();
        return fail("FIFO evict of the spilled entry failed");
    }
    if (disk.ram_is_durable(ram_id)) {
        alloc.release();
        return fail("disk FIFO-evict left RAM durable");
    }
    if (ram.disk_entry_id(ram_id) != 0) {
        alloc.release();
        return fail("disk FIFO-evict left a RAM ticket");
    }
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("emergency spill after disk FIFO-evict failed");
    }
    if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        alloc.release();
        return fail("re-spill after disk FIFO-evict was not hittable");
    }
    alloc.release();
    return 0;
}

int test_lock_does_not_wipe_tmp_before_flock(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("lock-tmp");
    fs::create_directories(dir.path / "tmp");
    const auto keep = dir.path / "tmp" / "keep-me";
    {
        std::ofstream out(keep);
        out << "keep";
    }
    const auto lock_path = dir.path / "LOCK";
    const int fd         = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) { return fail("could not create LOCK for exclusive holder"); }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        return fail("could not take exclusive LOCK");
    }
    q36::detail::KVRamCache ram(8ULL << 20);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    bool threw = false;
    try {
        q36::detail::KVDiskCache second(cfg);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("locked") != std::string::npos;
    }
    const bool kept = fs::exists(keep);
    (void)::flock(fd, LOCK_UN);
    ::close(fd);
    (void)ctx;
    if (!threw) { return fail("locked location did not report the exclusive lock"); }
    if (!kept) { return fail("tmp was wiped before exclusive lock"); }
    return 0;
}

std::vector<std::uint64_t> read_manifest_ids(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
    if (bytes.size() < 16) { return {}; }
    if (std::memcmp(bytes.data(), q36::detail::kDiskManifestMagic, 8) != 0) { return {}; }
    std::uint32_t count = 0;
    std::memcpy(&count, bytes.data() + 12, 4);
    if (bytes.size() < 16 + static_cast<std::size_t>(count) * 8) { return {}; }
    std::vector<std::uint64_t> ids(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::memcpy(&ids[i], bytes.data() + 16 + static_cast<std::size_t>(i) * 8, 8);
    }
    return ids;
}

int test_claim_does_not_hang_when_idle_spill_has_no_inflight(ninfer::DeviceContext& ctx,
                                                             ninfer::PagedKVPool& pool) {
    TmpDir dir("claim-idle-0");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> aligned(64, 4);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("claim-idle-0 spill of B failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match_b) {
        alloc.release();
        return fail("claim-idle-0 match of B failed");
    }
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_d, match_b->entry_id);
    disk.note_ram_resident(ram_d, match_b->entry_id);
    for (int i = 0; i < 20; ++i) {
        disk.request_idle_spill();
        std::atomic<bool> done{false};
        std::thread claimer([&] {
            try {
                if (disk.claim(match_b->entry_id)) { disk.release(match_b->entry_id); }
            } catch (const std::logic_error&) {}
            done.store(true);
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!done.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!done.load()) {
            claimer.detach();
            alloc.release();
            return fail("claim hung waiting for an idle spill with no inflight payload");
        }
        claimer.join();
        std::atomic<bool> cancelled{false};
        std::thread canceler([&] {
            disk.cancel_idle_of_ram(ram_d);
            cancelled.store(true);
        });
        const auto cancel_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!cancelled.load() && std::chrono::steady_clock::now() < cancel_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!cancelled.load()) {
            canceler.detach();
            alloc.release();
            return fail("cancel_idle_of_ram hung with no inflight payload");
        }
        canceler.join();
    }
    alloc.release();
    return 0;
}

int test_fifo_evict_rewrites_manifest(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("evict-man");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    std::vector<ninfer::TokenId> tokens_a(64, 1);
    std::vector<ninfer::TokenId> tokens_b(64, 2);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("evict-man spill A failed");
    }
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("evict-man spill B failed");
    }
    const auto match_a =
        disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
    if (!match_a) {
        alloc.release();
        return fail("evict-man match A failed");
    }
    const auto id_a = match_a->entry_id;
    if (!disk.test_fifo_evict_one()) {
        alloc.release();
        return fail("evict-man fifo evict failed");
    }
    const auto ids = read_manifest_ids(dir.path / "MANIFEST");
    if (std::find(ids.begin(), ids.end(), id_a) != ids.end()) {
        alloc.release();
        return fail("fifo evict left the victim in MANIFEST");
    }
    if (fs::exists(dir.path / "entries" / std::to_string(id_a))) {
        alloc.release();
        return fail("fifo evict left the victim entry directory");
    }
    alloc.release();
    return 0;
}

int test_smaller_capacity_open_persists_eviction(ninfer::DeviceContext& ctx,
                                                  ninfer::PagedKVPool& pool) {
    TmpDir dir("small-cap");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens_a(192, 1);
    std::vector<ninfer::TokenId> tokens_b(192, 2);
    std::uint64_t id_a = 0;
    std::uint64_t id_b = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("small-cap spill A failed");
        }
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("small-cap spill B failed");
        }
        const auto match_a =
            disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
        const auto match_b =
            disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
        if (!match_a || !match_b) {
            alloc.release();
            return fail("small-cap match failed");
        }
        id_a = match_a->entry_id;
        id_b = match_b->entry_id;
        disk.wait_idle_and_fsync();
    }
    auto used = 0ULL;
    {
        q36::detail::KVDiskCache measure(cfg);
        used = measure.snapshot().used_bytes;
    }
    if (used == 0) {
        alloc.release();
        return fail("small-cap could not measure used bytes");
    }
    cfg.capacity_bytes = used > 1 ? used - 1 : 1;
    {
        q36::detail::KVDiskCache small(cfg);
        const auto ids = read_manifest_ids(dir.path / "MANIFEST");
        const bool a_listed = std::find(ids.begin(), ids.end(), id_a) != ids.end();
        const bool b_listed = std::find(ids.begin(), ids.end(), id_b) != ids.end();
        if (a_listed && b_listed) {
            alloc.release();
            return fail("smaller-capacity open did not persist FIFO eviction to MANIFEST");
        }
        if (small.snapshot().used_bytes > cfg.capacity_bytes) {
            alloc.release();
            return fail("smaller-capacity open exceeded unique-byte capacity");
        }
    }
    cfg.capacity_bytes = 64ULL << 20;
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit_a =
        reopened.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
    const auto hit_b =
        reopened.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    const int hits = (hit_a ? 1 : 0) + (hit_b ? 1 : 0);
    if (hits != 1) {
        alloc.release();
        return fail("smaller-capacity eviction did not remain after reopen at full capacity");
    }
    alloc.release();
    return 0;
}

int test_misplaced_checkpoint_kind_is_skipped(ninfer::DeviceContext& ctx,
                                               ninfer::PagedKVPool& pool) {
    TmpDir dir("slot-kind");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier     = 3;
    rollback.hash         = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind         = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.hidden       = hidden_host.data();
    rollback.hidden_bytes = 64;
    source.ladder_heads   = {rollback};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("slot-kind capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::uint64_t entry = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("slot-kind spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("slot-kind match failed");
        }
        entry = match->entry_id;
        disk.wait_idle_and_fsync();
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry) / "meta.bin";
    auto bytes            = [&] {
        std::ifstream in(meta_path, std::ios::binary);
        return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    }();
    constexpr std::size_t kRollbackFrontierOffset = 148;
    bool patched = bytes.size() >= kRollbackFrontierOffset + 8;
    std::uint32_t frontier = 0;
    std::uint32_t kind     = 0;
    if (patched) {
        std::memcpy(&frontier, bytes.data() + kRollbackFrontierOffset, 4);
        std::memcpy(&kind, bytes.data() + kRollbackFrontierOffset + 4, 4);
        patched = frontier == 3 && kind == 1;
    }
    if (patched) {
        kind = 0;
        std::memcpy(bytes.data() + kRollbackFrontierOffset + 4, &kind, 4);
    }
    if (!patched) {
        alloc.release();
        return fail("slot-kind did not find the rollback kind field");
    }
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("misplaced Ladder-in-rollback slot remained hittable");
        }
        if (reopened.snapshot().drops == 0) {
            alloc.release();
            return fail("misplaced checkpoint kind did not increment drops");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "slot-kind reopen threw: " << e.what() << '\n';
        return 1;
    }
    alloc.release();
    return 0;
}

int test_missing_current_hidden_is_not_advertised(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("hidden-adv");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(64, 3);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("hidden-advert spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("hidden-advert match failed");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (!meta.tail_hidden_valid || meta.current_hidden_id != 0) {
        alloc.release();
        return fail("spill did not store tail_hidden_valid without current_hidden_id");
    }
    const auto host = disk.load_host(match->entry_id);
    if (!host) {
        alloc.release();
        return fail("hidden-advert load_host missed");
    }
    if (!host->tail_hidden_valid) {
        alloc.release();
        return fail("load_host dropped tail_hidden_valid when hidden_bytes is 0");
    }
    ninfer::DeviceBuffer hid(256);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        dest.release();
        alloc.release();
        return fail("hidden-advert claim failed");
    }
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    bool threw = false;
    try {
        disk.wait_copies();
    } catch (const std::runtime_error&) { threw = true; }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    if (!threw) {
        alloc.release();
        return fail("restore of missing current hidden succeeded");
    }
    alloc.release();
    return 0;
}

bool disk_tree_has_regular_files(const fs::path& root) {
    std::error_code ec;
    if (!fs::exists(root, ec)) { return false; }
    for (auto it = fs::recursive_directory_iterator(root, ec);
         it != fs::recursive_directory_iterator(); ++it) {
        if (it->is_regular_file()) { return true; }
    }
    return false;
}

int test_cancel_create_after_rename_does_not_leak(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("create-cancel");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(64, 4);
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        disk.test_arm_stall_after_meta_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("create-cancel never reached meta rename");
        }
        disk.cancel_idle_spill();
        disk.wait_idle_and_fsync();
        if (disk.snapshot().used_bytes != 0) {
            alloc.release();
            return fail("cancelled Create left billed unique bytes");
        }
        if (disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("cancelled Create remained hittable");
        }
        if (disk_tree_has_regular_files(dir.path / "objects")) {
            alloc.release();
            return fail("cancelled Create left object files");
        }
        if (disk_tree_has_regular_files(dir.path / "entries")) {
            alloc.release();
            return fail("cancelled Create left an entry directory");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("reopen resurrected a cancelled Create generation");
    }
    if (reopened.snapshot().used_bytes != 0) {
        alloc.release();
        return fail("reopen billed unique bytes for a cancelled Create");
    }
    alloc.release();
    return 0;
}

int test_cancel_branch_after_rename_drops_shared_refs(ninfer::DeviceContext& ctx,
                                                       ninfer::PagedKVPool& pool) {
    TmpDir dir("branch-cancel");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> parent(128, 7);
    std::vector<ninfer::TokenId> child(parent.begin(), parent.begin() + 64);
    child.push_back(99);
    child.resize(70, 11);
    std::uint64_t parent_id = 0;
    std::uint64_t parent_page = 0;
    std::uint64_t parent_refs = 0;
    std::size_t used_after_parent = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent);
        disk.note_ram_resident(ram_p, 0);
        if (!disk.emergency_spill_ram(ram_p)) {
            alloc.release();
            return fail("branch-cancel parent spill failed");
        }
        const auto match_p =
            disk.plan_match(text_prompt(parent), q36::detail::prefix_hash_chain(text_prompt(parent)));
        if (!match_p) {
            alloc.release();
            return fail("branch-cancel parent match failed");
        }
        parent_id         = match_p->entry_id;
        const auto pages  = disk.test_main_page_ids(parent_id);
        if (pages.empty()) {
            alloc.release();
            return fail("branch-cancel parent has no pages");
        }
        parent_page        = pages.front();
        parent_refs         = disk.test_object_refcount(parent_page);
        used_after_parent  = disk.snapshot().used_bytes;
        const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
        ram.set_disk_entry_id(ram_c, parent_id);
        disk.note_ram_resident(ram_c, parent_id);
        disk.test_arm_stall_after_meta_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("branch-cancel never reached meta rename");
        }
        disk.cancel_idle_spill();
        disk.wait_idle_and_fsync();
        if (!disk.test_entry_in_index(parent_id)) {
            alloc.release();
            return fail("cancelled Branch dropped the parent");
        }
        if (disk.test_object_refcount(parent_page) != parent_refs) {
            alloc.release();
            return fail("cancelled Branch leaked shared page refs");
        }
        if (disk.snapshot().used_bytes != used_after_parent) {
            alloc.release();
            return fail("cancelled Branch changed unique-byte billing");
        }
        if (disk.plan_match(text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child))) &&
            disk.plan_match(text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child)))
                    ->reuse_base >= 70) {
            alloc.release();
            return fail("cancelled Branch remained hittable at the child frontier");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit_p =
        reopened.plan_match(text_prompt(parent), q36::detail::prefix_hash_chain(text_prompt(parent)));
    if (!hit_p || hit_p->entry_id != parent_id) {
        alloc.release();
        return fail("reopen lost the parent after cancelled Branch");
    }
    const auto hit_c =
        reopened.plan_match(text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child)));
    if (hit_c && hit_c->entry_id != parent_id && hit_c->reuse_base >= 70) {
        alloc.release();
        return fail("reopen resurrected a cancelled Branch generation");
    }
    alloc.release();
    return 0;
}

int test_incomplete_rollback_does_not_poison_ladder_restore(ninfer::DeviceContext& ctx,
                                                          ninfer::PagedKVPool& pool) {
    TmpDir dir("poison-rb");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x42);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source             = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    source.gdn              = &gdn;
    source.gdn_current_slot = 0;
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    std::vector<unsigned char> hidden_host(64, 0x11);
    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 3;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv.data();
    rollback.recurrent       = rec.data();
    rollback.conv_bytes      = conv.size();
    rollback.recurrent_bytes = rec.size();
    q36::detail::RamLadderHead ladder = rollback;
    ladder.kind         = q36::detail::ContextCheckpointKind::Ladder;
    ladder.hidden       = hidden_host.data();
    ladder.hidden_bytes = 64;
    source.ladder_heads = {rollback, ladder};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("poison-rb capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("poison-rb spill failed");
    }
    auto prompt3 = text_prompt({3, 3, 3});
    const auto match = disk.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
    if (!match || match->reuse_base != 3 ||
        match->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        alloc.release();
        return fail("poison-rb did not plan the complete Ladder");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.rollback.gdn_id == 0 || meta.rollback.hidden_id != 0 ||
        meta.ladders[0].frontier != 3 || meta.ladders[0].hidden_id == 0) {
        alloc.release();
        return fail("poison-rb did not persist incomplete rollback plus complete Ladder");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier, match->reuse_base,
                     match->reuse)) {
        alloc.release();
        return fail("poison-rb claim of the Ladder failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.reuse          = match->reuse;
    target.reuse_base     = match->reuse_base;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        disk.wait_copies();
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "poison-rb restore threw: " << e.what() << '\n';
        return fail("incomplete rollback poisoned a valid Ladder restore");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_shutdown_retries_failed_ram_spill(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("shut-retry");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, {4, 5, 6, 7});
    disk.note_ram_resident(ram_id, 0);
    disk.test_arm_crash_before_meta();
    if (disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("injected pre-meta crash still marked the RAM generation durable");
    }
    if (disk.ram_is_durable(ram_id)) {
        alloc.release();
        return fail("failed spill left ram_is_durable true");
    }
    disk.flush_not_durable_ram();
    if (!disk.ram_is_durable(ram_id)) {
        alloc.release();
        return fail("shutdown flush skipped a RAM generation whose prior spill failed");
    }
    const auto prompt = text_prompt({4, 5, 6, 7});
    if (!disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        alloc.release();
        return fail("shutdown retry did not publish the recovered generation");
    }
    alloc.release();
    return 0;
}

int test_failed_entry_unlink_does_not_resurrect(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("unlink-res");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> parent(128, 7);
    std::vector<ninfer::TokenId> child(parent.begin(), parent.begin() + 64);
    child.push_back(99);
    child.resize(70, 11);
    std::uint64_t parent_id = 0;
    std::uint64_t child_id  = 0;
    fs::path child_dir;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent);
        disk.note_ram_resident(ram_p, 0);
        if (!disk.emergency_spill_ram(ram_p)) {
            alloc.release();
            return fail("unlink-res parent spill failed");
        }
        const auto parent_match =
            disk.plan_match(text_prompt(parent), q36::detail::prefix_hash_chain(text_prompt(parent)));
        if (!parent_match) {
            alloc.release();
            return fail("unlink-res parent match failed");
        }
        parent_id = parent_match->entry_id;
        ram.set_disk_entry_id(ram_p, parent_id);
        const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child);
        disk.note_ram_resident(ram_c, parent_id);
        if (!disk.emergency_spill_ram(ram_c)) {
            alloc.release();
            return fail("unlink-res child spill failed");
        }
        const auto child_match =
            disk.plan_match(text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child)));
        if (!child_match || child_match->entry_id == parent_id) {
            alloc.release();
            return fail("unlink-res child match failed");
        }
        child_id  = child_match->entry_id;
        child_dir = dir.path / "entries" / std::to_string(child_id);
        if (!disk.claim(parent_id)) {
            alloc.release();
            return fail("unlink-res claim of parent failed");
        }
        std::error_code ec;
        fs::permissions(child_dir, fs::perms::none, ec);
        if (!disk.test_fifo_evict_one()) {
            fs::permissions(child_dir, fs::perms::owner_all, ec);
            disk.release(parent_id);
            alloc.release();
            return fail("unlink-res did not evict the chmod'd child");
        }
        disk.release(parent_id);
        fs::permissions(child_dir, fs::perms::owner_all, ec);
        disk.wait_idle_and_fsync();
    }
    std::error_code rec;
    fs::permissions(child_dir, fs::perms::owner_all, rec);
    q36::detail::KVDiskCache reopened(cfg);
    const auto parent_hit = reopened.plan_match(
        text_prompt(parent), q36::detail::prefix_hash_chain(text_prompt(parent)));
    if (!parent_hit || parent_hit->entry_id != parent_id) {
        alloc.release();
        return fail("unlink-res lost the parent generation");
    }
    const auto child_hit = reopened.plan_match(
        text_prompt(child), q36::detail::prefix_hash_chain(text_prompt(child)));
    if (child_hit && child_hit->entry_id == child_id && child_hit->reuse_base >= 70) {
        alloc.release();
        return fail("failed entry-dir unlink resurrected the evicted child");
    }
    alloc.release();
    return 0;
}

int test_oversized_state_object_does_not_abort_restore(ninfer::DeviceContext& ctx,
                                                      ninfer::PagedKVPool& pool) {
    TmpDir dir("huge-state");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x42);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source             = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    source.gdn              = &gdn;
    source.gdn_current_slot = 0;
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("huge-state capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    q36::detail::KVDiskCache disk(std::move(cfg));
    disk.note_ram_resident(*id, 0);
    if (!disk.emergency_spill_ram(*id)) {
        alloc.release();
        return fail("huge-state spill failed");
    }
    const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("huge-state match failed");
    }
    const auto meta = disk.test_load_meta(match->entry_id);
    if (meta.current_gdn_id == 0) {
        alloc.release();
        return fail("huge-state did not persist current GDN");
    }
    const auto gdn_loc = packed_object_location(dir.path, q36::detail::DiskObjectKind::State,
                                                meta.current_gdn_id);
    const int fd = gdn_loc ? ::open(gdn_loc->path.c_str(), O_RDWR) : -1;
    if (fd < 0) {
        alloc.release();
        return fail("huge-state could not open the GDN object");
    }
    const int truncated = ::ftruncate(fd, static_cast<off_t>(1ULL << 40));
    ::close(fd);
    if (truncated != 0) {
        alloc.release();
        return fail("huge-state could not extend the GDN object");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("huge-state claim failed");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text             = &dest;
    target.text_pool        = &pool;
    target.text_dst_pages   = 1;
    target.gdn              = &gdn;
    target.gdn_current_slot = 0;
    target.stream           = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "huge-state restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "huge-state restore threw: " << e.what() << '\n';
        return fail("oversized state object aborted restore");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    alloc.release();
    return 0;
}

int test_cancel_create_fsync_does_not_hold_mutex(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("cancel-fsync");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(cfg);
    std::vector<ninfer::TokenId> tokens(64, 4);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    disk.test_arm_stall_after_meta_rename();
    disk.test_set_fsync_stall_ms(250);
    disk.request_idle_spill();
    const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (!disk.test_meta_renamed() && std::chrono::steady_clock::now() < renamed_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_meta_renamed()) {
        disk.test_set_fsync_stall_ms(0);
        disk.cancel_idle_spill();
        alloc.release();
        return fail("cancel-fsync never reached meta rename");
    }
    disk.cancel_idle_spill();
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (!disk.test_fsync_entered() && std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_fsync_entered()) {
        disk.test_set_fsync_stall_ms(0);
        disk.wait_idle_and_fsync();
        alloc.release();
        return fail("cancel-fsync never reached directory fsync");
    }
    const auto t0        = std::chrono::steady_clock::now();
    const auto unrelated = text_prompt({1, 2, 3, 4});
    (void)disk.plan_match(unrelated, q36::detail::prefix_hash_chain(unrelated));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    disk.test_set_fsync_stall_ms(0);
    disk.wait_idle_and_fsync();
    if (elapsed > std::chrono::milliseconds(100)) {
        alloc.release();
        return fail("plan_match waited on cancelled-Create directory fsync");
    }
    alloc.release();
    return 0;
}

int test_refresh_does_not_evict_self_before_commit(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("refresh-self");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(1, ctx.stream);
    std::vector<ninfer::TokenId> tokens{4, 5, 6, 7};
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::size_t used = 0;
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("refresh-self first spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("refresh-self first match failed");
        }
        entry_id = match->entry_id;
        used     = disk.snapshot().used_bytes;
        disk.wait_idle_and_fsync();
    }
    cfg.capacity_bytes = used;
    ninfer::DeviceBuffer hid(256);
    hid.fill(0xab);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {256});
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source        = make_source(retained, identity, alloc, pool, ctx.copy_stream, 4);
    source.tail_hidden = &hidden;
    auto ram_id       = capture_or_evict(ram, source);
    if (!ram_id) {
        alloc.release();
        return fail("refresh-self hidden capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    q36::detail::KVDiskCache disk(cfg);
    ram.set_disk_entry_id(*ram_id, entry_id);
    disk.note_ram_resident(*ram_id, entry_id);
    disk.test_arm_crash_before_meta();
    (void)disk.emergency_spill_ram(*ram_id);
    if (disk.snapshot().evictions != 0) {
        alloc.release();
        return fail("refresh FIFO-evicted its own committed generation");
    }
    const auto hit =
        disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
    if (!hit || hit->entry_id != entry_id) {
        alloc.release();
        return fail("refresh evicted its committed generation before the replacement was durable");
    }
    alloc.release();
    return 0;
}

int test_oversized_host_files_are_skipped_on_open(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    auto spill_one = [&](const fs::path& location, std::uint64_t& ledger_id, std::uint64_t& identity_id,
                          std::uint64_t& entry_id) -> int {
        q36::detail::KVRamCache ram(16ULL << 20);
        auto alloc = pool.reserve(2);
        alloc.materialize_pages(1, ctx.stream);
        auto cfg = disk_config(location, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
        {
            q36::detail::KVDiskCache disk(cfg);
            const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
            disk.note_ram_resident(ram_id, 0);
            if (!disk.emergency_spill_ram(ram_id)) {
                alloc.release();
                return fail("huge-host spill failed");
            }
            const auto match =
                disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
            if (!match) {
                alloc.release();
                return fail("huge-host match failed");
            }
            entry_id    = match->entry_id;
            const auto meta = disk.test_load_meta(entry_id);
            ledger_id    = meta.ledger_id;
            identity_id  = meta.identity_id;
        }
        alloc.release();
        return 0;
    };

    {
        TmpDir dir("huge-ledger");
        std::uint64_t ledger_id = 0, identity_id = 0, entry_id = 0;
        if (const int rc = spill_one(dir.path, ledger_id, identity_id, entry_id); rc != 0) {
            return rc;
        }
        const auto ledger = packed_object_location(dir.path, q36::detail::DiskObjectKind::Ledger,
                                                   ledger_id);
        const int fd = ledger ? ::open(ledger->path.c_str(), O_RDWR) : -1;
        if (fd < 0) { return fail("huge-host could not open the ledger"); }
        const int truncated = ::ftruncate(fd, static_cast<off_t>(1ULL << 40));
        ::close(fd);
        if (truncated != 0) { return fail("huge-host could not extend the ledger"); }
        q36::detail::KVRamCache ram(16ULL << 20);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        try {
            q36::detail::KVDiskCache reopened(cfg);
        if (!reopened.plan_match(text_prompt({2, 3, 4, 5}),
                                 q36::detail::prefix_hash_chain(text_prompt({2, 3, 4, 5})))) {
                return fail("oversized pack tail invalidated its bounded ledger extent");
            }
        } catch (const std::exception& e) {
            std::cerr << "huge-ledger reopen threw: " << e.what() << '\n';
            return fail("oversized ledger aborted Engine construction");
        }
    }
    {
        TmpDir dir("huge-ident");
        std::uint64_t ledger_id = 0, identity_id = 0, entry_id = 0;
        if (const int rc = spill_one(dir.path, ledger_id, identity_id, entry_id); rc != 0) {
            return rc;
        }
        if (identity_id == 0) { return fail("huge-ident stored no identity"); }
        const auto identity = packed_object_location(dir.path,
                                                     q36::detail::DiskObjectKind::Identity,
                                                     identity_id);
        const int fd = identity ? ::open(identity->path.c_str(), O_RDWR) : -1;
        if (fd < 0) { return fail("huge-host could not open identity"); }
        const int truncated = ::ftruncate(fd, static_cast<off_t>(1ULL << 40));
        ::close(fd);
        if (truncated != 0) { return fail("huge-host could not extend identity"); }
        q36::detail::KVRamCache ram(16ULL << 20);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        try {
            q36::detail::KVDiskCache reopened(cfg);
            if (!reopened.plan_match(text_prompt({2, 3, 4, 5}),
                                     q36::detail::prefix_hash_chain(text_prompt({2, 3, 4, 5})))) {
                return fail("oversized pack tail invalidated its bounded identity extent");
            }
        } catch (const std::exception& e) {
            std::cerr << "huge-ident reopen threw: " << e.what() << '\n';
            return fail("oversized identity aborted Engine construction");
        }
    }
    {
        TmpDir dir("huge-man");
        std::uint64_t ledger_id = 0, identity_id = 0, entry_id = 0;
        if (const int rc = spill_one(dir.path, ledger_id, identity_id, entry_id); rc != 0) {
            return rc;
        }
        std::vector<std::uint8_t> man(16 + 4, 0);
        std::memcpy(man.data(), q36::detail::kDiskManifestMagic, 8);
        const std::uint32_t version = q36::detail::kDiskFormatVersion;
        const std::uint32_t n      = 0xffffffffu;
        std::memcpy(man.data() + 8, &version, 4);
        std::memcpy(man.data() + 12, &n, 4);
        {
            std::ofstream out(dir.path / "MANIFEST", std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(man.data()),
                      static_cast<std::streamsize>(man.size()));
        }
        q36::detail::KVRamCache ram(16ULL << 20);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        try {
            q36::detail::KVDiskCache reopened(cfg);
            if (!reopened.plan_match(text_prompt({2, 3, 4, 5}),
                                     q36::detail::prefix_hash_chain(text_prompt({2, 3, 4, 5})))) {
                return fail("torn MANIFEST count did not rebuild from entries/");
            }
        } catch (const std::exception& e) {
            std::cerr << "huge-man reopen threw: " << e.what() << '\n';
            return fail("huge MANIFEST count aborted Engine construction");
        }
    }
    {
        TmpDir dir("huge-meta-n");
        std::uint64_t ledger_id = 0, identity_id = 0, entry_id = 0;
        if (const int rc = spill_one(dir.path, ledger_id, identity_id, entry_id); rc != 0) {
            return rc;
        }
        const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
        std::ifstream in(meta_path, std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
        in.close();
        if (bytes.size() < 24) { return fail("huge-meta-n meta.bin is truncated"); }
        const std::uint32_t huge = 0xffffffffu;
        std::memcpy(bytes.data() + bytes.size() - 16, &huge, 4);
        {
            std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
        q36::detail::KVRamCache ram(16ULL << 20);
        auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                               32ULL << 20, 4096);
        try {
            q36::detail::KVDiskCache reopened(cfg);
            if (reopened.plan_match(text_prompt({2, 3, 4, 5}),
                                    q36::detail::prefix_hash_chain(text_prompt({2, 3, 4, 5})))) {
                return fail("huge meta page-vector count remained hittable");
            }
        } catch (const std::exception& e) {
            std::cerr << "huge-meta-n reopen threw: " << e.what() << '\n';
            return fail("huge meta page-vector count aborted Engine construction");
        }
    }
    return 0;
}

int test_tombstone_survives_failed_entries_unlink(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("tombstone");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(64, 9);
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("tombstone spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("tombstone match failed");
        }
        entry_id = match->entry_id;
        std::error_code ec;
        fs::permissions(dir.path / "entries", fs::perms::none, ec);
        if (!disk.test_fifo_evict_one()) {
            fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
            alloc.release();
            return fail("tombstone evict failed");
        }
        if (!fs::exists(dir.path / "tombstones" / std::to_string(entry_id))) {
            fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
            alloc.release();
            return fail("evict did not write a tombstone outside entries/");
        }
        disk.wait_idle_and_fsync();
        fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
    }
    std::error_code rec;
    fs::permissions(dir.path / "entries", fs::perms::owner_all, rec);
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("failed entries/ unlink resurrected a tombstoned generation");
    }
    alloc.release();
    return 0;
}

int test_text_kv_valid_cannot_hide_short_pages(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("kv-valid");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(128, 3);
    std::uint64_t entry_id = 0;
    std::size_t page_bytes = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("kv-valid spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match || match->reuse_base != 128) {
            alloc.release();
            return fail("kv-valid match failed");
        }
        entry_id = match->entry_id;
        page_bytes = disk.test_main_page_ids(entry_id).size() * 8;
        if (page_bytes < 16) {
            alloc.release();
            return fail("kv-valid stored fewer than two main pages");
        }
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    in.close();
    if (bytes.size() < 36 + page_bytes) {
        alloc.release();
        return fail("kv-valid meta.bin is truncated");
    }
    const std::uint32_t one = 1;
    const std::uint32_t zero = 0;
    std::memcpy(bytes.data() + 32, &one, 4);
    const std::size_t counts = bytes.size() - page_bytes - 8;
    std::memcpy(bytes.data() + counts, &one, 4);
    std::memcpy(bytes.data() + counts + 4, &zero, 4);
    bytes.resize(counts + 8 + 8);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("short page vector hidden by text_kv_valid remained hittable");
    }
    alloc.release();
    return 0;
}

int test_tombstone_does_not_poison_reused_ids(ninfer::DeviceContext& ctx,
                                                ninfer::PagedKVPool& pool) {
    TmpDir dir("tomb-reuse");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> first(64, 9);
    std::vector<ninfer::TokenId> second(64, 11);
    std::uint64_t first_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, first);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("tomb-reuse first spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)));
        if (!match) {
            alloc.release();
            return fail("tomb-reuse first match failed");
        }
        first_id = match->entry_id;
        std::error_code ec;
        fs::permissions(dir.path / "entries", fs::perms::none, ec);
        if (!disk.test_fifo_evict_one()) {
            fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
            alloc.release();
            return fail("tomb-reuse evict failed");
        }
        fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
    }
    std::error_code rec;
    fs::permissions(dir.path / "entries", fs::perms::owner_all, rec);
    std::uint64_t second_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, second);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("tomb-reuse second spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(second), q36::detail::prefix_hash_chain(text_prompt(second)));
        if (!match) {
            alloc.release();
            return fail("tomb-reuse second match failed");
        }
        second_id = match->entry_id;
        if (second_id == first_id) {
            alloc.release();
            return fail("new spill reused a tombstoned entry id");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)))) {
        alloc.release();
        return fail("tombstoned first generation was resurrected");
    }
    if (!reopened.plan_match(text_prompt(second),
                              q36::detail::prefix_hash_chain(text_prompt(second)))) {
        alloc.release();
        return fail("new generation after a leftover tombstone was not hittable");
    }
    alloc.release();
    return 0;
}

int test_listed_tombstone_is_not_loaded(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("tomb-listed");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(64, 9);
    const auto man = dir.path / "MANIFEST";
    const auto bak = dir.path / "MANIFEST.bak";
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("tomb-listed spill failed");
        }
        disk.wait_idle_and_fsync();
        fs::copy_file(man, bak);
        std::error_code ec;
        fs::permissions(dir.path / "entries", fs::perms::none, ec);
        if (!disk.test_fifo_evict_one()) {
            fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
            alloc.release();
            return fail("tomb-listed evict failed");
        }
        fs::permissions(dir.path / "entries", fs::perms::owner_all, ec);
    }
    std::error_code rec;
    fs::permissions(dir.path / "entries", fs::perms::owner_all, rec);
    fs::copy_file(bak, man, fs::copy_options::overwrite_existing);
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("stale MANIFEST resurrected a tombstoned generation");
    }
    alloc.release();
    return 0;
}

int test_tombstone_write_failure_does_not_unlink(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("tomb-fail");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(64, 9);
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("tomb-fail spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("tomb-fail match failed");
        }
        entry_id = match->entry_id;
        disk.test_arm_fail_tombstone();
        if (disk.test_fifo_evict_one()) {
            alloc.release();
            return fail("evict succeeded without a durable tombstone");
        }
        if (!fs::exists(dir.path / "entries" / std::to_string(entry_id) / "meta.bin")) {
            alloc.release();
            return fail("tombstone write failure still destroyed the entry directory");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("generation unlinked without a durable tombstone was lost");
    }
    alloc.release();
    return 0;
}

int test_rollback_after_rename_keeps_previous(ninfer::DeviceContext& ctx,
                                               ninfer::PagedKVPool& pool) {
    TmpDir dir("rollback-rename");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("rollback-rename spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("rollback-rename match of B failed");
        }
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, match_b->entry_id);
        disk.note_ram_resident(ram_d, match_b->entry_id);
        disk.test_arm_stall_after_meta_rename();
        disk.test_arm_fail_after_rollback_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("rollback-rename never reached meta rename");
        }
        disk.cancel_idle_spill();
        disk.wait_idle_and_fsync();
        const auto live128 =
            disk.plan_match(text_prompt(extended),
                            q36::detail::prefix_hash_chain(text_prompt(extended)));
        if (live128 && live128->reuse_base == 128) {
            alloc.release();
            return fail("post-rollback-rename failure published the new generation in memory");
        }
        if (!packed_object_location(dir.path, q36::detail::DiskObjectKind::Main,
                                    disk.test_main_page_ids(match_b->entry_id).front())) {
            alloc.release();
            return fail("uncertain rollback durability deleted objects of the new generation");
        }
        if (disk.test_disk_io_pins(match_b->entry_id) != 0) {
            alloc.release();
            return fail("uncertain rollback leaked a disk I/O pin");
        }
        if (!ram.evict_one_unpinned(ram_d)) {
            alloc.release();
            return fail("uncertain rollback leaked a RAM I/O pin");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto hit64 =
        reopened.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)));
    const auto hit128 =
        reopened.plan_match(text_prompt(extended),
                            q36::detail::prefix_hash_chain(text_prompt(extended)));
    if (!hit64 || hit64->reuse_base != 64) {
        alloc.release();
        return fail("rollback after rename did not keep the previous generation");
    }
    if (hit128 && hit128->reuse_base == 128) {
        alloc.release();
        return fail("rollback after rename left the new generation hittable");
    }
    alloc.release();
    return 0;
}

int test_wrong_hidden_size_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("hidden-sz");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0x11);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    source.tail_hidden = &hidden;
    auto id            = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("hidden-sz capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    std::uint64_t hidden_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("hidden-sz spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("hidden-sz match failed");
        }
        hidden_id = disk.test_load_meta(match->entry_id).current_hidden_id;
        if (hidden_id == 0) {
            alloc.release();
            return fail("hidden-sz did not persist tail hidden");
        }
    }
    std::vector<std::uint8_t> hdr(q36::detail::kDiskCodecHeaderBytes);
    if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                            hdr.data(), hdr.size())) {
        alloc.release();
        return fail("hidden-sz state object is truncated");
    }
    const std::uint64_t wrong = 8;
    std::memcpy(hdr.data() + 4, &wrong, 8);
    std::memcpy(hdr.data() + 12, &wrong, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, hidden_id,
                             hdr.data(), hdr.size())) { return fail("hidden-sz could not patch pack"); }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("wrong hidden decoded size remained hittable");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "hidden-sz reopen threw: " << e.what() << '\n';
        return fail("wrong hidden decoded size aborted Engine construction");
    }
    alloc.release();
    return 0;
}

int test_huge_identity_token_count_is_skipped(ninfer::DeviceContext& ctx,
                                              ninfer::PagedKVPool& pool) {
    TmpDir dir("ident-cnt");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t identity_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("ident-cnt spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("ident-cnt match failed");
        }
        identity_id = disk.test_load_meta(match->entry_id).identity_id;
        if (identity_id == 0) {
            alloc.release();
            return fail("ident-cnt stored no identity");
        }
    }
    const std::uint32_t huge = 0xffffffffu;
    std::uint32_t item_count = 0;
    std::array<std::uint8_t, 8> identity_hdr{};
    std::memcpy(identity_hdr.data(), &huge, 4);
    std::memcpy(identity_hdr.data() + 4, &item_count, 4);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::Identity, identity_id,
                             identity_hdr.data(), identity_hdr.size())) {
        alloc.release(); return fail("ident-cnt could not patch pack");
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(text_prompt(tokens),
                                 q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("huge identity token count remained hittable");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "ident-cnt reopen threw: " << e.what() << '\n';
        return fail("huge identity token count aborted Engine construction");
    }
    alloc.release();
    return 0;
}

int test_huge_compressed_size_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("huge-cmp");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x42);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source             = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    source.gdn              = &gdn;
    source.gdn_current_slot = 0;
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("huge-cmp capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    std::uint64_t gdn_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("huge-cmp spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            alloc.release();
            return fail("huge-cmp match failed");
        }
        gdn_id = disk.test_load_meta(match->entry_id).current_gdn_id;
        if (gdn_id == 0) {
            alloc.release();
            return fail("huge-cmp did not persist current GDN");
        }
    }
    std::vector<std::uint8_t> hdr(q36::detail::kDiskCodecHeaderBytes);
    if (!packed_read_prefix(dir.path, q36::detail::DiskObjectKind::State, gdn_id,
                            hdr.data(), hdr.size())) {
        alloc.release();
        return fail("huge-cmp state object is truncated");
    }
    const std::uint64_t huge_cmp = (64ULL << 20) + 1;
    std::memcpy(hdr.data() + 12, &huge_cmp, 8);
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::State, gdn_id,
                             hdr.data(), hdr.size())) { return fail("huge-cmp could not patch pack"); }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
            alloc.release();
            return fail("huge compressed size remained hittable");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "huge-cmp reopen threw: " << e.what() << '\n';
        return fail("huge compressed size aborted Engine construction");
    }
    alloc.release();
    return 0;
}

int test_index_scan_is_capped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("scan-cap");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("scan-cap spill failed");
        }
    }
    for (int i = 10; i < 30; ++i) {
        fs::create_directories(dir.path / "entries" / std::to_string(i));
    }
    cfg.max_index_entries = 2;
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (!reopened.plan_match(text_prompt(tokens),
                                 q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("capped index scan dropped the committed generation");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "scan-cap reopen threw: " << e.what() << '\n';
        return fail("capped index scan aborted Engine construction");
    }
    alloc.release();
    return 0;
}

int test_tombstone_survives_until_manifest_persist(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("tomb-persist");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(64, 9);
    std::uint64_t entry_id = 0;
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("tomb-persist spill failed");
    }
    const auto match =
        disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
    if (!match) {
        alloc.release();
        return fail("tomb-persist match failed");
    }
    entry_id = match->entry_id;
    if (!disk.test_fifo_evict_one_unpersisted()) {
        alloc.release();
        return fail("tomb-persist evict failed");
    }
    if (!fs::exists(dir.path / "tombstones" / std::to_string(entry_id))) {
        alloc.release();
        return fail("tombstone was cleared before MANIFEST exclusion");
    }
    if (fs::exists(dir.path / "entries" / std::to_string(entry_id) / "meta.bin")) {
        alloc.release();
        return fail("tomb-persist left the entry directory");
    }
    alloc.release();
    return 0;
}

int test_capped_extra_object_ids_are_not_reused(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("obj-hw");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> first{2, 3, 4, 5};
    std::uint64_t high_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, first);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("obj-hw first spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)));
        if (!match) {
            alloc.release();
            return fail("obj-hw first match failed");
        }
        const auto pages = disk.test_main_page_ids(match->entry_id);
        if (pages.empty()) {
            alloc.release();
            return fail("obj-hw stored no main pages");
        }
        high_id = 0;
        for (std::uint64_t id : pages) { high_id = std::max(high_id, id); }
    }
    fs::create_directories(dir.path / "entries" / "99");
    cfg.max_index_entries = 1;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, std::vector<ninfer::TokenId>{9, 8, 7, 6});
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("obj-hw second spill failed");
        }
    }
    cfg.max_index_entries = 4096;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto match = disk.plan_match(text_prompt({9, 8, 7, 6}),
                                           q36::detail::prefix_hash_chain(text_prompt({9, 8, 7, 6})));
        if (!match) {
            alloc.release();
            return fail("capped high-water spill did not reopen");
        }
        const auto ids = disk.test_main_page_ids(match->entry_id);
        if (ids.empty() || *std::min_element(ids.begin(), ids.end()) <= high_id) {
            alloc.release();
            return fail("object-map high-water reused an id after capped reopen");
        }
    }
    if (packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, high_id + 1)) {
        alloc.release();
        return fail("object-map fabricated an uncommitted high-water object");
    }
    alloc.release();
    return 0;
}

int test_failed_object_write_does_not_leak(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("obj-fail");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    auto count_main = [&] {
        std::size_t n = 0;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir.path / "objects" / "main", ec)) {
            if (entry.is_regular_file()) { ++n; }
        }
        return n;
    };
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, std::vector<ninfer::TokenId>{2, 3, 4, 5});
    disk.note_ram_resident(ram_id, 0);
    disk.test_arm_fail_object_write();
    const bool spilled = disk.emergency_spill_ram(ram_id);
    if (spilled) {
        alloc.release();
        return fail("injected object write failure still committed");
    }
    if (count_main() != 0) {
        alloc.release();
        return fail("failed object write left an unpublished main object");
    }
    alloc.release();
    return 0;
}

int test_kind_conflict_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("kind-col");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    std::uint64_t page_id   = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("kind-col spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("kind-col match failed");
        }
        entry_id = match->entry_id;
        const auto pages = disk.test_main_page_ids(entry_id);
        if (pages.empty()) {
            alloc.release();
            return fail("kind-col stored no main pages");
        }
        page_id = pages.front();
    }
    std::vector<std::uint8_t> blob(q36::detail::kDiskCodecHeaderBytes + 8, 0);
    blob[0] = 0;
    blob[1] = static_cast<std::uint8_t>(q36::detail::DiskStateKind::TailHidden);
    const std::uint64_t eight = 8;
    std::memcpy(blob.data() + 4, &eight, 8);
    std::memcpy(blob.data() + 12, &eight, 8);
    {
        std::ofstream out(dir.path / "objects" / "state" / std::to_string(page_id),
                          std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(blob.size()));
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> meta((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    in.close();
    // encode_meta: magic 8, version 4, then packed scalars through identity_id at
    // byte 100, current_gdn_id at 108, current_hidden_id at 116.
    constexpr std::size_t kCurrentHiddenIdOff = 116;
    if (meta.size() < kCurrentHiddenIdOff + 8) {
        alloc.release();
        return fail("kind-col meta.bin is truncated");
    }
    std::memcpy(meta.data() + kCurrentHiddenIdOff, &page_id, 8);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(meta.data()),
                  static_cast<std::streamsize>(meta.size()));
    }
    try {
        q36::detail::KVDiskCache reopened(cfg);
        if (reopened.plan_match(text_prompt(tokens),
                                 q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
            alloc.release();
            return fail("conflicting object-kind namespaces remained hittable");
        }
    } catch (const std::exception& e) {
        alloc.release();
        std::cerr << "kind-col reopen threw: " << e.what() << '\n';
        return fail("kind conflict aborted Engine construction");
    }
    alloc.release();
    return 0;
}

int test_short_text_kv_valid_is_skipped(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("kv-short");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(128, 3);
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("kv-short spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match || match->reuse_base != 128) {
            alloc.release();
            return fail("kv-short match failed");
        }
        entry_id = match->entry_id;
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    in.close();
    if (bytes.size() < 36) {
        alloc.release();
        return fail("kv-short meta.bin is truncated");
    }
    const std::uint32_t one = 1;
    std::memcpy(bytes.data() + 32, &one, 4);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("text_kv_valid shorter than execution_frontier remained hittable");
    }
    alloc.release();
    return 0;
}

int test_skipped_corrupt_ids_do_not_poison_valid_entry(ninfer::DeviceContext& ctx,
                                                       ninfer::PagedKVPool& pool) {
    TmpDir dir("skip-poison");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> long_chat(80, 1);
    std::vector<ninfer::TokenId> short_chat{9, 8, 7, 6};
    std::uint64_t long_id = 0;
    std::uint64_t short_page = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_long = capture_tokens(ram, pool, alloc, ctx, long_chat);
        disk.note_ram_resident(ram_long, 0);
        if (!disk.emergency_spill_ram(ram_long)) {
            alloc.release();
            return fail("skip-poison long spill failed");
        }
        const auto long_match =
            disk.plan_match(text_prompt(long_chat), q36::detail::prefix_hash_chain(text_prompt(long_chat)));
        if (!long_match) {
            alloc.release();
            return fail("skip-poison long match failed");
        }
        long_id = long_match->entry_id;
        const auto ram_short = capture_tokens(ram, pool, alloc, ctx, short_chat);
        disk.note_ram_resident(ram_short, 0);
        if (!disk.emergency_spill_ram(ram_short)) {
            alloc.release();
            return fail("skip-poison short spill failed");
        }
        const auto short_match =
            disk.plan_match(text_prompt(short_chat),
                            q36::detail::prefix_hash_chain(text_prompt(short_chat)));
        if (!short_match) {
            alloc.release();
            return fail("skip-poison short match failed");
        }
        const auto pages = disk.test_main_page_ids(short_match->entry_id);
        if (pages.empty()) {
            alloc.release();
            return fail("skip-poison short stored no main pages");
        }
        short_page = pages.front();
    }
    const auto meta_path = dir.path / "entries" / std::to_string(long_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> meta((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    in.close();
    constexpr std::size_t kCurrentHiddenIdOff = 116;
    if (meta.size() < kCurrentHiddenIdOff + 8) {
        alloc.release();
        return fail("skip-poison meta.bin is truncated");
    }
    std::memcpy(meta.data() + kCurrentHiddenIdOff, &short_page, 8);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(meta.data()),
                  static_cast<std::streamsize>(meta.size()));
    }
    cfg.max_context = 16;
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.plan_match(text_prompt(short_chat),
                              q36::detail::prefix_hash_chain(text_prompt(short_chat)))) {
        alloc.release();
        return fail("valid entry was poisoned by a skipped too-long tree");
    }
    alloc.release();
    return 0;
}

int test_failed_object_unlink_keeps_tombstone(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("unlink-obj");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    std::uint64_t page_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("unlink-obj spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("unlink-obj match failed");
        }
        entry_id = match->entry_id;
        const auto pages = disk.test_main_page_ids(entry_id);
        if (pages.empty()) {
            alloc.release();
            return fail("unlink-obj stored no main pages");
        }
        page_id = pages.front();
        if (!disk.test_fifo_evict_one()) {
            alloc.release();
            return fail("unlink-obj evict failed");
        }
        const auto location = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main,
                                                     page_id);
        if (!location || !fs::exists(location->path)) {
            alloc.release();
            return fail("packed eviction deleted a retained pack extent");
        }
    }
    if (fs::exists(dir.path / "tombstones" / std::to_string(entry_id))) {
        alloc.release();
        return fail("packed eviction left a completed tombstone");
    }
    const auto location = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, page_id);
    if (!location || !fs::exists(location->path)) {
        alloc.release();
        return fail("packed eviction reclaimed its extent before compaction");
    }
    alloc.release();
    return 0;
}

int test_ledger_frontier_must_be_execution_plus_one(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("ledger-f");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("ledger-f spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("ledger-f match failed");
        }
        entry_id = match->entry_id;
        if (disk.test_load_meta(entry_id).ledger_frontier != 5) {
            alloc.release();
            return fail("ledger-f spill did not store execution_frontier+1");
        }
    }
    const auto meta_path = dir.path / "entries" / std::to_string(entry_id) / "meta.bin";
    std::ifstream in(meta_path, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    in.close();
    if (bytes.size() < 28) {
        alloc.release();
        return fail("ledger-f meta.bin is truncated");
    }
    const std::uint32_t four = 4;
    std::memcpy(bytes.data() + 24, &four, 4);
    {
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("ledger_frontier == execution_frontier remained hittable");
    }
    alloc.release();
    return 0;
}

int test_speculative_head_without_backend_pages_is_skipped(ninfer::DeviceContext& ctx) {
    auto text_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto backend_plan = plan_paged_cache(8, 8, 2, {{ninfer::DType::I8, 64, 2}});
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool backend({backend_arena.base(), backend_arena.capacity()},
                                backend_plan.layout);
    auto text_alloc    = text.reserve(4);
    auto backend_alloc = backend.reserve(4);
    text_alloc.materialize_pages(3, ctx.stream);
    backend_alloc.materialize_pages(1, ctx.stream);
    TmpDir dir("mtp-short");
    q36::detail::KVRamCache ram(64ULL << 20);
    std::vector<ninfer::TokenId> t129(129, 42);
    auto prompt   = text_prompt(t129);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source         = make_source(retained, identity, text_alloc, text, ctx.copy_stream, 129);
    source.mtp_kv_valid = 64;
    source.backend      = &backend_alloc;
    source.backend_pool = &backend;
    auto id             = capture_or_evict(ram, source);
    if (!id) {
        text_alloc.release();
        backend_alloc.release();
        return fail("mtp-short capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    auto cfg = disk_config(dir.path, ram, text, &backend, ninfer::SpeculativeBackend::Mtp,
                           64ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk(std::move(cfg));
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            text_alloc.release();
            backend_alloc.release();
            return fail("mtp-short spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            text_alloc.release();
            backend_alloc.release();
            return fail("mtp-short in-process match failed");
        }
        if (disk.test_backend_page_ids(match->entry_id).size() != 1) {
            text_alloc.release();
            backend_alloc.release();
            return fail("mtp-short did not store one backend page");
        }
    }
    auto cfg2 = disk_config(dir.path, ram, text, &backend, ninfer::SpeculativeBackend::Mtp,
                            64ULL << 20, 4096);
    q36::detail::KVDiskCache reopened(cfg2);
    if (reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        text_alloc.release();
        backend_alloc.release();
        return fail("MTP head without enough backend pages remained hittable");
    }
    text_alloc.release();
    backend_alloc.release();
    return 0;
}

int test_fingerprint_state_over_64mib_is_accepted(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("big-state");
    constexpr std::uint64_t hidden_bytes = 65ULL << 20;
    q36::detail::KVRamCache ram(192ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, alloc, 0x91);
    ninfer::DeviceBuffer hidden_storage(hidden_bytes);
    hidden_storage.fill(0x6d);
    ninfer::Tensor hidden(hidden_storage.p, ninfer::DType::U8,
                          {static_cast<std::int64_t>(hidden_bytes)});
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           512ULL << 20, 4096);
    cfg.hidden_bytes = hidden_bytes;
    const std::vector<ninfer::TokenId> tokens_a(64, 73);
    const std::vector<ninfer::TokenId> tokens_b(64, 89);
    std::uint64_t entry_b = 0;
    std::uint64_t old_generation = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens_hidden(ram, pool, alloc, ctx, tokens_a, hidden);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("big-state spill A failed");
        }
        const auto ram_b = capture_tokens_hidden(ram, pool, alloc, ctx, tokens_b, hidden);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("big-state spill B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(tokens_b),
                                              q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
        if (!match_b) {
            alloc.release();
            return fail("big-state match B failed");
        }
        entry_b = match_b->entry_id;
        if (!disk.test_fifo_evict_one()) {
            alloc.release();
            return fail("big-state could not evict A before compaction");
        }
    }
    {
        const auto packset = read_bytes(dir.path / "PACKSET");
        if (packset.size() != 32) {
            alloc.release();
            return fail("big-state PACKSET is missing before compaction");
        }
        std::memcpy(&old_generation, packset.data() + 12, sizeof(old_generation));
        q36::detail::KVDiskCache compacting(cfg);
        compacting.wait_idle_and_fsync();
        const auto compacted = read_bytes(dir.path / "PACKSET");
        std::uint64_t new_generation = 0;
        if (compacted.size() != 32) {
            alloc.release();
            return fail("big-state PACKSET is missing after compaction");
        }
        std::memcpy(&new_generation, compacted.data() + 12, sizeof(new_generation));
        if (new_generation <= old_generation) {
            alloc.release();
            return fail("big-state retained object prevented compaction");
        }
    }
    q36::detail::KVDiskCache reopened(cfg);
    const auto prompt_b = text_prompt(tokens_b);
    const auto match_b = reopened.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b || match_b->entry_id != entry_b) {
        alloc.release();
        return fail("big-state compaction/reopen lost B");
    }
    ninfer::DeviceBuffer hidden_out_storage(hidden_bytes);
    hidden_out_storage.fill(0);
    ninfer::Tensor hidden_out(hidden_out_storage.p, ninfer::DType::U8,
                              {static_cast<std::int64_t>(hidden_bytes)});
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    if (!reopened.claim(match_b->entry_id, match_b->hash_f, match_b->execution_frontier)) {
        dest.release();
        alloc.release();
        return fail("big-state compacted entry claim failed");
    }
    reopened.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(reopened, ctx, "big-state restore hung"); rc != 0) {
            reopened.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        reopened.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "big-state restore threw: " << e.what() << '\n';
        return 1;
    }
    ctx.synchronize_all();
    unsigned char first = 0;
    unsigned char last = 0;
    CUDA_CHECK(cudaMemcpy(&first, hidden_out_storage.p, 1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last, static_cast<unsigned char*>(hidden_out_storage.p) +
                                     hidden_bytes - 1,
                          1, cudaMemcpyDeviceToHost));
    reopened.cancel_restore();
    reopened.release(match_b->entry_id);
    dest.release();
    if (first != 0x6d || last != 0x6d) {
        alloc.release();
        return fail("big-state compact/reopen restore corrupted hidden state");
    }
    alloc.release();
    return 0;
}

int test_torn_tombstone_does_not_drop_entry(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("torn-tomb");
    q36::detail::KVRamCache ram(16ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens{2, 3, 4, 5};
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
        disk.note_ram_resident(ram_id, 0);
        if (!disk.emergency_spill_ram(ram_id)) {
            alloc.release();
            return fail("torn-tomb spill failed");
        }
        const auto match =
            disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
        if (!match) {
            alloc.release();
            return fail("torn-tomb match failed");
        }
        entry_id = match->entry_id;
    }
    fs::create_directories(dir.path / "tombstones");
    {
        std::ofstream out(dir.path / "tombstones" / std::to_string(entry_id),
                          std::ios::binary | std::ios::trunc);
        out.write("xx", 2);
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("torn tombstone discarded a committed generation");
    }
    alloc.release();
    return 0;
}

int test_too_long_corrupt_identity_is_not_skipped(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("skip-ident");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> long_chat(80, 1);
    std::vector<ninfer::TokenId> short_chat{9, 8, 7, 6};
    std::uint64_t long_id = 0;
    std::uint64_t identity_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_long = capture_tokens(ram, pool, alloc, ctx, long_chat);
        disk.note_ram_resident(ram_long, 0);
        if (!disk.emergency_spill_ram(ram_long)) {
            alloc.release();
            return fail("skip-ident long spill failed");
        }
        const auto long_match =
            disk.plan_match(text_prompt(long_chat), q36::detail::prefix_hash_chain(text_prompt(long_chat)));
        if (!long_match) {
            alloc.release();
            return fail("skip-ident long match failed");
        }
        long_id = long_match->entry_id;
        identity_id = disk.test_load_meta(long_id).identity_id;
        const auto ram_short = capture_tokens(ram, pool, alloc, ctx, short_chat);
        disk.note_ram_resident(ram_short, 0);
        if (!disk.emergency_spill_ram(ram_short)) {
            alloc.release();
            return fail("skip-ident short spill failed");
        }
    }
    const char junk[16] = {static_cast<char>(0xff)};
    if (!packed_write_prefix(dir.path, q36::detail::DiskObjectKind::Identity, identity_id,
                             junk, sizeof(junk))) {
        alloc.release();
        return fail("skip-ident could not corrupt packed identity");
    }
    cfg.max_context = 16;
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.test_skipped_count() != 0) {
        alloc.release();
        return fail("corrupt too-long identity was installed as a skipped tree");
    }
    if (!reopened.plan_match(text_prompt(short_chat),
                              q36::detail::prefix_hash_chain(text_prompt(short_chat)))) {
        alloc.release();
        return fail("valid sibling was lost after a corrupt too-long identity");
    }
    alloc.release();
    return 0;
}

int test_stale_manifest_extras_keep_entry_id_order(ninfer::DeviceContext& ctx,
                                                    ninfer::PagedKVPool& pool) {
    TmpDir dir("extra-fifo");
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> first(64, 9);
    std::vector<ninfer::TokenId> second(64, 11);
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, first);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("extra-fifo first spill failed");
        }
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, second);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("extra-fifo second spill failed");
        }
    }
    std::vector<std::uint8_t> empty_manifest(16, 0);
    std::memcpy(empty_manifest.data(), q36::detail::kDiskManifestMagic, 8);
    const std::uint32_t version = q36::detail::kDiskFormatVersion;
    std::memcpy(empty_manifest.data() + 8, &version, 4);
    {
        std::ofstream out(dir.path / "MANIFEST", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(empty_manifest.data()),
                  static_cast<std::streamsize>(empty_manifest.size()));
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.test_fifo_evict_one()) {
        alloc.release();
        return fail("extra-fifo evict failed");
    }
    if (reopened.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)))) {
        alloc.release();
        return fail("stale-MANIFEST extras evicted the newer extra first");
    }
    if (!reopened.plan_match(text_prompt(second),
                              q36::detail::prefix_hash_chain(text_prompt(second)))) {
        alloc.release();
        return fail("stale-MANIFEST extras did not keep the newer extra");
    }
    alloc.release();
    return 0;
}

int test_prepare_spill_exception_releases_pins(ninfer::DeviceContext& ctx,
                                              ninfer::PagedKVPool& pool) {
    TmpDir dir("prep-throw");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("prep-throw spill of B failed");
    }
    const auto match_b = disk.plan_match(text_prompt(aligned),
                                         q36::detail::prefix_hash_chain(text_prompt(aligned)));
    if (!match_b) {
        alloc.release();
        return fail("prep-throw match of B failed");
    }
    const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
    ram.set_disk_entry_id(ram_d, match_b->entry_id);
    disk.note_ram_resident(ram_d, match_b->entry_id);
    disk.test_arm_fail_prepare_spill();
    if (disk.emergency_spill_ram(ram_d)) {
        alloc.release();
        return fail("prepare_spill exception still marked the spill durable");
    }
    if (disk.test_disk_io_pins(match_b->entry_id) != 0) {
        alloc.release();
        return fail("prepare_spill exception leaked a disk I/O pin");
    }
    if (!ram.evict_one_unpinned(ram_d)) {
        alloc.release();
        return fail("prepare_spill exception leaked a RAM I/O pin");
    }
    if (!disk.plan_match(text_prompt(aligned), q36::detail::prefix_hash_chain(text_prompt(aligned)))) {
        alloc.release();
        return fail("prepare_spill exception dropped the committed parent");
    }
    alloc.release();
    return 0;
}

int test_failed_spill_jobs_do_not_commit_retry(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    TmpDir dir("stale-job");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens(128, 7);
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    disk.test_arm_fail_object_write();
    if (disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("stale-job first spill succeeded after injected page write failure");
    }
    if (disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)))) {
        alloc.release();
        return fail("stale leftover commit published an incomplete generation");
    }
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("retry after purged spill jobs failed");
    }
    const auto match =
        disk.plan_match(text_prompt(tokens), q36::detail::prefix_hash_chain(text_prompt(tokens)));
    if (!match || match->reuse_base != 128) {
        alloc.release();
        return fail("retry after purged spill jobs was not hittable");
    }
    const auto pages = disk.test_main_page_ids(match->entry_id);
    if (pages.size() < 2 || pages[0] == 0 || pages[1] == 0) {
        alloc.release();
        return fail("retry published a generation with missing page IDs");
    }
    alloc.release();
    return 0;
}

int test_uncertainty_holds_reclaim_on_evict_and_reopen(ninfer::DeviceContext& ctx,
                                                        ninfer::PagedKVPool& pool) {
    TmpDir dir("hold-reclaim");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> aligned(64, 4);
    std::vector<ninfer::TokenId> extended = aligned;
    extended.push_back(0);
    extended.resize(128, 5);
    std::vector<std::uint64_t> pages_b;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("hold-reclaim spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("hold-reclaim match of B failed");
        }
        pages_b = disk.test_main_page_ids(match_b->entry_id);
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, match_b->entry_id);
        disk.note_ram_resident(ram_d, match_b->entry_id);
        disk.test_arm_fail_after_meta_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("hold-reclaim never reached meta rename");
        }
        disk.wait_idle_and_fsync();
        for (std::uint64_t id : pages_b) {
            if (id != 0 && !packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
                alloc.release();
                return fail("uncertain rename deleted predecessor objects before reclamation");
            }
        }
        if (!disk.test_fifo_evict_one()) {
            alloc.release();
            return fail("hold-reclaim evict failed");
        }
        for (std::uint64_t id : pages_b) {
            if (id != 0 && !packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
                alloc.release();
                return fail("fifo eviction left uncertainty-protected predecessor objects");
            }
        }
    }
    TmpDir dir2("hold-reopen");
    auto cfg2 = disk_config(dir2.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                            64ULL << 20, 4096);
    {
        q36::detail::KVDiskCache disk(cfg2);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, aligned);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("hold-reopen spill of B failed");
        }
        const auto match_b = disk.plan_match(text_prompt(aligned),
                                             q36::detail::prefix_hash_chain(text_prompt(aligned)));
        if (!match_b) {
            alloc.release();
            return fail("hold-reopen match of B failed");
        }
        const auto ram_d = capture_tokens(ram, pool, alloc, ctx, extended);
        ram.set_disk_entry_id(ram_d, match_b->entry_id);
        disk.note_ram_resident(ram_d, match_b->entry_id);
        disk.test_arm_stall_after_meta_rename();
        disk.test_arm_fail_after_rollback_rename();
        disk.request_idle_spill();
        const auto renamed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!disk.test_meta_renamed() &&
               std::chrono::steady_clock::now() < renamed_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!disk.test_meta_renamed()) {
            disk.cancel_idle_spill();
            alloc.release();
            return fail("hold-reopen never reached meta rename");
        }
        disk.cancel_idle_spill();
        disk.wait_idle_and_fsync();
    }
    q36::detail::KVDiskCache reopened(cfg2);
    if (!reopened.plan_match(text_prompt(aligned),
                              q36::detail::prefix_hash_chain(text_prompt(aligned)))) {
        alloc.release();
        return fail("reopen after uncertain rollback lost the previous generation");
    }
    const auto hit128 = reopened.plan_match(text_prompt(extended),
                                             q36::detail::prefix_hash_chain(text_prompt(extended)));
    if (hit128 && hit128->reuse_base == 128) {
        alloc.release();
        return fail("reopen after uncertain rollback kept the unpublished new generation");
    }
    // A durable location append can precede a failed meta publication.  In v4
    // that extent is intentionally retained as unreachable pack garbage until
    // generation compaction; the observable requirement is that it is not
    // resurrected as a matching entry.
    alloc.release();
    return 0;
}

int test_emergency_promote_survives_restore(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("em-promote");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> tokens_a(64, 3);
    std::vector<ninfer::TokenId> tokens_b(128, 7);
    q36::detail::KVDiskCache disk(cfg);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("em-promote spill of A failed");
    }
    const auto match_a =
        disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
    if (!match_a) {
        alloc.release();
        return fail("em-promote match of A failed");
    }
    ram.claim(ram_a);
    ram.consume(ram_a);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    disk.test_set_payload_io_stall_ms(300);
    disk.request_idle_spill();
    const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (!disk.test_payload_io_entered() &&
           std::chrono::steady_clock::now() < entered_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_payload_io_entered()) {
        disk.test_set_payload_io_stall_ms(0);
        disk.cancel_idle_spill();
        alloc.release();
        return fail("em-promote idle spill never entered payload I/O");
    }
    std::atomic<bool> spilled{false};
    std::atomic<bool> done{false};
    std::thread spiller([&] {
        spilled.store(disk.emergency_spill_ram(ram_b));
        done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto dest = pool.reserve(4);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    if (!disk.claim(match_a->entry_id, match_a->hash_f, match_a->execution_frontier)) {
        disk.test_set_payload_io_stall_ms(0);
        spiller.join();
        dest.release();
        alloc.release();
        return fail("em-promote claim of A failed");
    }
    disk.restore_device(match_a->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "em-promote restore of A hung");
            rc != 0) {
            disk.test_set_payload_io_stall_ms(0);
            disk.release(match_a->entry_id);
            dest.release();
            spiller.join();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.test_set_payload_io_stall_ms(0);
        disk.release(match_a->entry_id);
        dest.release();
        spiller.join();
        alloc.release();
        std::cerr << "em-promote restore threw: " << e.what() << '\n';
        return 1;
    }
    disk.test_set_payload_io_stall_ms(0);
    const auto spill_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!done.load() && std::chrono::steady_clock::now() < spill_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!done.load()) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        spiller.join();
        alloc.release();
        return fail("em-promote emergency spill hung under restore");
    }
    spiller.join();
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    dest.release();
    if (!spilled.load() || !disk.ram_is_durable(ram_b)) {
        alloc.release();
        return fail("promoted emergency spill was cancelled or not durable");
    }
    if (ram.test_io_pins(ram_b) != 0) {
        alloc.release();
        return fail("promoted emergency spill leaked a RAM I/O pin");
    }
    if (!ram.evict_one_unpinned(ram_b)) {
        alloc.release();
        return fail("promoted emergency spill left RAM pinned");
    }
    const auto match_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    if (!match_b || match_b->reuse_base != 128) {
        alloc.release();
        return fail("promoted emergency spill was not hittable");
    }
    alloc.release();
    return 0;
}

int test_corrupt_meta_does_not_delete_objects(ninfer::DeviceContext& ctx,
                                               ninfer::PagedKVPool& pool) {
    TmpDir dir("corrupt-meta");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> first(64, 9);
    std::vector<ninfer::TokenId> second(64, 11);
    std::uint64_t entry_a = 0;
    std::vector<std::uint64_t> pages_a;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, first);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("corrupt-meta spill of A failed");
        }
        const auto match_a =
            disk.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)));
        if (!match_a) {
            alloc.release();
            return fail("corrupt-meta match of A failed");
        }
        entry_a = match_a->entry_id;
        pages_a = disk.test_main_page_ids(match_a->entry_id);
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, second);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("corrupt-meta spill of B failed");
        }
    }
    {
        std::ofstream out(dir.path / "entries" / std::to_string(entry_a) / "meta.bin",
                          std::ios::binary | std::ios::trunc);
        const char junk[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        out.write(junk, sizeof(junk));
    }
    std::vector<std::uint64_t> page_sizes;
    for (std::uint64_t id : pages_a) {
        const auto location = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id);
        if (!location || !fs::exists(location->path)) {
            alloc.release();
            return fail("corrupt-meta page missing before reopen");
        }
        page_sizes.push_back(location->extent);
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (reopened.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)))) {
        alloc.release();
        return fail("corrupt meta remained hittable");
    }
    if (!reopened.plan_match(text_prompt(second),
                             q36::detail::prefix_hash_chain(text_prompt(second)))) {
        alloc.release();
        return fail("sibling was lost after a corrupt neighbor meta");
    }
    for (std::size_t i = 0; i < pages_a.size(); ++i) {
        const auto location = packed_object_location(dir.path, q36::detail::DiskObjectKind::Main,
                                                     pages_a[i]);
        if (!location || !fs::exists(location->path) || location->extent != page_sizes[i]) {
            alloc.release();
            return fail("orphan reclaim deleted objects of a corrupt entry meta");
        }
    }
    alloc.release();
    return 0;
}

int test_capped_extras_keep_smallest_and_objects(ninfer::DeviceContext& ctx,
                                                   ninfer::PagedKVPool& pool) {
    TmpDir dir("cap-extra");
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(3, ctx.stream);
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           64ULL << 20, 4096);
    std::vector<ninfer::TokenId> first(64, 9);
    std::vector<ninfer::TokenId> second(64, 11);
    std::vector<ninfer::TokenId> third(64, 13);
    std::uint64_t id_a = 0;
    std::uint64_t id_b = 0;
    std::uint64_t id_c = 0;
    std::vector<std::uint64_t> pages_b;
    {
        q36::detail::KVDiskCache disk(cfg);
        const auto ram_a = capture_tokens(ram, pool, alloc, ctx, first);
        disk.note_ram_resident(ram_a, 0);
        if (!disk.emergency_spill_ram(ram_a)) {
            alloc.release();
            return fail("cap-extra spill of A failed");
        }
        const auto match_a =
            disk.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)));
        if (!match_a) {
            alloc.release();
            return fail("cap-extra match of A failed");
        }
        id_a = match_a->entry_id;
        const auto ram_b = capture_tokens(ram, pool, alloc, ctx, second);
        disk.note_ram_resident(ram_b, 0);
        if (!disk.emergency_spill_ram(ram_b)) {
            alloc.release();
            return fail("cap-extra spill of B failed");
        }
        const auto match_b =
            disk.plan_match(text_prompt(second), q36::detail::prefix_hash_chain(text_prompt(second)));
        if (!match_b) {
            alloc.release();
            return fail("cap-extra match of B failed");
        }
        id_b = match_b->entry_id;
        pages_b = disk.test_main_page_ids(match_b->entry_id);
        const auto ram_c = capture_tokens(ram, pool, alloc, ctx, third);
        disk.note_ram_resident(ram_c, 0);
        if (!disk.emergency_spill_ram(ram_c)) {
            alloc.release();
            return fail("cap-extra spill of C failed");
        }
        const auto match_c =
            disk.plan_match(text_prompt(third), q36::detail::prefix_hash_chain(text_prompt(third)));
        if (!match_c) {
            alloc.release();
            return fail("cap-extra match of C failed");
        }
        id_c = match_c->entry_id;
    }
    std::vector<std::uint8_t> empty_manifest(16, 0);
    std::memcpy(empty_manifest.data(), q36::detail::kDiskManifestMagic, 8);
    const std::uint32_t version = q36::detail::kDiskFormatVersion;
    std::memcpy(empty_manifest.data() + 8, &version, 4);
    {
        std::ofstream out(dir.path / "MANIFEST", std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(empty_manifest.data()),
                  static_cast<std::streamsize>(empty_manifest.size()));
    }
    cfg.max_index_entries = 1;
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.test_entry_in_index(id_a) || reopened.test_entry_in_index(id_b) ||
        reopened.test_entry_in_index(id_c)) {
        alloc.release();
        return fail("capped extras did not keep the smallest entry id");
    }
    if (!reopened.plan_match(text_prompt(first), q36::detail::prefix_hash_chain(text_prompt(first)))) {
        alloc.release();
        return fail("capped extras dropped the resident generation");
    }
    if (reopened.plan_match(text_prompt(second),
                            q36::detail::prefix_hash_chain(text_prompt(second))) ||
        reopened.plan_match(text_prompt(third),
                            q36::detail::prefix_hash_chain(text_prompt(third)))) {
        alloc.release();
        return fail("capped extras billed an unindexed extra as live");
    }
    if (!fs::exists(dir.path / "entries" / std::to_string(id_b)) ||
        !fs::exists(dir.path / "entries" / std::to_string(id_c))) {
        alloc.release();
        return fail("capped extras deleted unindexed entry trees");
    }
    for (std::uint64_t id : pages_b) {
        if (!packed_object_location(dir.path, q36::detail::DiskObjectKind::Main, id)) {
            alloc.release();
            return fail("capped extras deleted unindexed extra objects");
        }
    }
    alloc.release();
    return 0;
}

int test_zero_hidden_bytes_preserves_heads(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    TmpDir dir("zero-hid");
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 4,
                      .conv_width     = 2,
                      .value_heads    = 1,
                      .value_head_dim = 4,
                      .key_head_dim   = 2,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    std::vector<unsigned char> conv(gdn.conv_host_image_bytes(), 0x41);
    std::vector<unsigned char> rec(gdn.recurrent_host_image_bytes(), 0x42);
    gdn.unpack_slot_from_host(0, conv.data(), rec.data(), ctx.stream);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(2);
    alloc.materialize_pages(2, ctx.stream);
    std::vector<ninfer::TokenId> tokens(8, 3);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    auto source                  = make_source(retained, identity, alloc, pool, ctx.copy_stream, 8);
    source.gdn                   = &gdn;
    source.gdn_current_slot     = 0;
    source.gdn_checkpoint_slot  = 0;
    source.rewrite_valid        = true;
    source.rewrite_kind         = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier      = 4;
    source.hash_c_valid         = true;
    source.hash_c               = q36::detail::prefix_hash_at(retained.token_ids, identity, 4);
    q36::detail::RamLadderHead rollback;
    rollback.frontier        = 3;
    rollback.hash            = q36::detail::prefix_hash_at(retained.token_ids, identity, 3);
    rollback.kind            = q36::detail::ContextCheckpointKind::TurnRollback;
    rollback.conv            = conv.data();
    rollback.recurrent       = rec.data();
    rollback.conv_bytes      = conv.size();
    rollback.recurrent_bytes = rec.size();
    q36::detail::RamLadderHead ladder = rollback;
    ladder.frontier                  = 5;
    ladder.hash = q36::detail::prefix_hash_at(retained.token_ids, identity, 5);
    ladder.kind = q36::detail::ContextCheckpointKind::Ladder;
    source.ladder_heads = {rollback, ladder};
    auto id = capture_or_evict(ram, source);
    if (!id) {
        alloc.release();
        return fail("zero-hid capture failed");
    }
    ctx.synchronize_all();
    ram.wait_pending_copies();
    const auto ram_host = ram.load_host(*id);
    if (!ram_host.tail_hidden_valid || !ram_host.rewrite_valid) {
        alloc.release();
        return fail("zero-hid RAM dropped validity flags");
    }
    auto cfg = disk_config(dir.path, ram, pool, nullptr, ninfer::SpeculativeBackend::None,
                           32ULL << 20, 4096, ninfer::KvDiskCompress::Off, &gdn);
    std::uint64_t entry_id = 0;
    {
        q36::detail::KVDiskCache disk(cfg);
        disk.note_ram_resident(*id, 0);
        if (!disk.emergency_spill_ram(*id)) {
            alloc.release();
            return fail("zero-hid spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match || match->reuse_base != 8) {
            alloc.release();
            return fail("zero-hid exact frontier was not selected");
        }
        entry_id            = match->entry_id;
        const auto meta = disk.test_load_meta(match->entry_id);
        if (!meta.tail_hidden_valid || meta.current_hidden_id != 0 || meta.rewrite_hidden_id != 0 ||
            meta.rollback.hidden_id != 0 || meta.ladders[0].hidden_id != 0) {
            alloc.release();
            return fail("zero-hid stored unexpected hidden objects");
        }
        auto host = disk.load_host(match->entry_id);
        if (!host || !host->tail_hidden_valid || !host->rewrite_valid) {
            alloc.release();
            return fail("zero-hid load_host dropped current or rewrite validity");
        }
        bool saw_rollback = false;
        bool saw_ladder    = false;
        for (const auto& slot : host->ladders) {
            if (slot.frontier == 3 && slot.kind == q36::detail::ContextCheckpointKind::TurnRollback) {
                saw_rollback = true;
            }
            if (slot.frontier == 5 && slot.kind == q36::detail::ContextCheckpointKind::Ladder) {
                saw_ladder = true;
            }
        }
        if (!saw_rollback || !saw_ladder) {
            alloc.release();
            return fail("zero-hid load_host omitted GDN-only checkpoint heads");
        }
        if (!disk.populate_checkpoint_images(*host)) {
            alloc.release();
            return fail("zero-hid populate failed on GDN-only heads");
        }
        bool img_rollback = false;
        bool saw_ladder_img = false;
        for (const auto& image : host->ladder_images) {
            if (image.frontier == 3) { img_rollback = true; }
            if (image.frontier == 5) { saw_ladder_img = true; }
        }
        if (!img_rollback || !saw_ladder_img) {
            alloc.release();
            return fail("zero-hid populate omitted GDN-only checkpoint images");
        }
        auto prompt3 = text_prompt({3, 3, 3});
        const auto rb =
            disk.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
        if (!rb || rb->reuse_base != 3 ||
            rb->reuse != ninfer::PrefixReusePath::RestoreTurnRollback) {
            alloc.release();
            return fail("zero-hid did not select the GDN-only rollback head");
        }
        auto prompt4 = text_prompt({3, 3, 3, 3});
        const auto rw =
            disk.plan_match(prompt4, q36::detail::prefix_hash_chain(prompt4));
        if (!rw || rw->reuse_base != 4 ||
            rw->reuse != ninfer::PrefixReusePath::RestoreTurnCheckpoint) {
            alloc.release();
            return fail("zero-hid did not select the GDN-only rewrite head");
        }
        auto prompt5 = text_prompt({3, 3, 3, 3, 3});
        const auto ld =
            disk.plan_match(prompt5, q36::detail::prefix_hash_chain(prompt5));
        if (!ld || ld->reuse_base != 5 ||
            ld->reuse != ninfer::PrefixReusePath::RestoreContextCheckpoint) {
            alloc.release();
            return fail("zero-hid did not select the GDN-only ladder head");
        }
        if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier, 8,
                         ninfer::PrefixReusePath::AppendAtFrontier)) {
            alloc.release();
            return fail("zero-hid claim of exact frontier failed");
        }
        auto dest = pool.reserve(4);
        dest.materialize_pages(1, ctx.stream);
        q36::detail::DiskRestoreTarget target;
        target.text              = &dest;
        target.text_pool         = &pool;
        target.text_dst_pages    = 1;
        target.gdn               = &gdn;
        target.gdn_current_slot = 1;
        target.gdn_checkpoint_slot = 2;
        target.stream            = ctx.copy_stream;
        disk.restore_device(match->entry_id, target);
        try {
            if (const int rc = wait_restore_bounded(disk, ctx, "zero-hid exact restore hung");
                rc != 0) {
                disk.release(match->entry_id);
                dest.release();
                alloc.release();
                return rc;
            }
        } catch (const std::exception& e) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            std::cerr << "zero-hid exact restore threw: " << e.what() << '\n';
            return 1;
        }
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier, 3,
                         ninfer::PrefixReusePath::RestoreTurnRollback)) {
            alloc.release();
            return fail("zero-hid claim of rollback head failed");
        }
        auto dest_rb = pool.reserve(4);
        dest_rb.materialize_pages(1, ctx.stream);
        q36::detail::DiskRestoreTarget rb_target;
        rb_target.text              = &dest_rb;
        rb_target.text_pool         = &pool;
        rb_target.text_dst_pages    = 1;
        rb_target.gdn               = &gdn;
        rb_target.gdn_current_slot = 1;
        rb_target.gdn_checkpoint_slot = 2;
        rb_target.reuse            = ninfer::PrefixReusePath::RestoreTurnRollback;
        rb_target.reuse_base       = 3;
        rb_target.stream            = ctx.copy_stream;
        disk.restore_device(match->entry_id, rb_target);
        try {
            if (const int rc = wait_restore_bounded(disk, ctx, "zero-hid rollback restore hung");
                rc != 0) {
                disk.release(match->entry_id);
                dest_rb.release();
                alloc.release();
                return rc;
            }
        } catch (const std::exception& e) {
            disk.release(match->entry_id);
            dest_rb.release();
            alloc.release();
            std::cerr << "zero-hid rollback restore threw: " << e.what() << '\n';
            return 1;
        }
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest_rb.release();
    }
    q36::detail::KVDiskCache reopened(cfg);
    if (!reopened.test_entry_in_index(entry_id)) {
        alloc.release();
        return fail("zero-hid reopen dropped the entry");
    }
    const auto again = reopened.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!again || again->reuse_base != 8) {
        alloc.release();
        return fail("zero-hid reopen lost exact reuse");
    }
    auto prompt3 = text_prompt({3, 3, 3});
    const auto rb =
        reopened.plan_match(prompt3, q36::detail::prefix_hash_chain(prompt3));
    if (!rb || rb->reuse_base != 3) {
        alloc.release();
        return fail("zero-hid reopen lost the rollback head");
    }
    const auto host = reopened.load_host(entry_id);
    if (!host || !host->tail_hidden_valid || !host->rewrite_valid) {
        alloc.release();
        return fail("zero-hid reopen dropped validity flags");
    }
    alloc.release();
    return 0;
}

constexpr std::uint32_t kRestoreReaders = 8;

q36::detail::DiskOpenConfig reader_disk_config(const fs::path& location, q36::detail::KVRamCache& ram,
                                                ninfer::PagedKVPool& pool, std::size_t capacity,
                                                std::uint32_t max_context) {
    auto cfg = disk_config(location, ram, pool, nullptr, ninfer::SpeculativeBackend::None, capacity,
                            max_context);
    cfg.restore_io_threads = kRestoreReaders;
    return cfg;
}

int expect_mapped_pages_equal(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& src,
                              const ninfer::PagedKVAllocation& dst, std::uint32_t pages,
                              ninfer::DeviceContext& ctx, const char* label) {
    const std::size_t page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    for (std::uint32_t i = 0; i < pages; ++i) {
        std::vector<unsigned char> a(page_bytes);
        std::vector<unsigned char> b(page_bytes);
        ninfer::pack_paged_kv_logical_page_to_host(src, pool, i, a.data(), ctx.stream);
        ninfer::pack_paged_kv_logical_page_to_host(dst, pool, i, b.data(), ctx.stream);
        ctx.synchronize_all();
        if (a != b) {
            std::cerr << label << " page " << i << " mismatch\n";
            return 1;
        }
    }
    return 0;
}

int test_restore_io_threads_roundtrip(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-roundtrip");
    constexpr std::uint32_t kPages = 8;
    auto plan = plan_paged_cache(kPages * 2, kPages, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(kPages);
    alloc.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(kPages * 64, 13);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-roundtrip spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match || disk.test_main_page_ids(match->entry_id).size() < kPages) {
        alloc.release();
        return fail("readers-roundtrip did not store all main pages");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-roundtrip claim failed");
    }
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-roundtrip restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-roundtrip threw: " << e.what() << '\n';
        return 1;
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, kPages, ctx, "readers-roundtrip") != 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-roundtrip restored wrong KV");
    }
    disk.cancel_restore();
    disk.prefetch_window(match->entry_id, kPages, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    dest.release();
    dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    target.text = &dest;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-roundtrip second restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-roundtrip second restore threw: " << e.what() << '\n';
        return 1;
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, kPages, ctx, "readers-roundtrip-2") != 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-roundtrip second restore restored wrong KV");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    fill_logical_pages(pool, alloc, 77);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> other(kPages * 64, 9);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, other);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        dest.release();
        return fail("readers-roundtrip writer spill after restore failed");
    }
    if (!disk.plan_match(text_prompt(other), q36::detail::prefix_hash_chain(text_prompt(other)))) {
        alloc.release();
        dest.release();
        return fail("readers-roundtrip writer spill was not hittable");
    }
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_idle_extra_readers(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-idle");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(128, 4);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-idle spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-idle match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-idle claim failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-idle restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-idle threw: " << e.what() << '\n';
        return 1;
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, 2, ctx, "readers-idle") != 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-idle restored wrong KV");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    fill_logical_pages(pool, alloc, 41);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("readers-idle spill B failed");
    }
    const auto match_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    if (!match_b) {
        alloc.release();
        return fail("readers-idle match B failed");
    }
    if (!disk.claim(match_b->entry_id, match_b->hash_f, match_b->execution_frontier)) {
        alloc.release();
        return fail("readers-idle claim B failed");
    }
    dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    target.text = &dest;
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-idle restore B hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-idle B threw: " << e.what() << '\n';
        return 1;
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, 2, ctx, "readers-idle-B") != 0) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-idle B restored wrong KV");
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_pread_fail(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-pread");
    constexpr std::uint32_t kPages = 8;
    auto plan = plan_paged_cache(kPages * 2, kPages, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(kPages);
    alloc.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(kPages * 64, 13);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-pread spill A failed");
    }
    fill_logical_pages(pool, alloc, 33);
    std::vector<ninfer::TokenId> tokens_b(kPages * 64, 7);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    if (!disk.emergency_spill_ram(ram_b)) {
        alloc.release();
        return fail("readers-pread spill B failed");
    }
    const auto match_a =
        disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)));
    const auto match_b =
        disk.plan_match(text_prompt(tokens_b), q36::detail::prefix_hash_chain(text_prompt(tokens_b)));
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id) {
        alloc.release();
        return fail("readers-pread did not persist two entries");
    }
    const auto pages_a = disk.test_main_page_ids(match_a->entry_id);
    if (pages_a.size() < 4) {
        alloc.release();
        return fail("readers-pread A stored too few pages");
    }
    if (!disk.claim(match_a->entry_id, match_a->hash_f, match_a->execution_frontier)) {
        alloc.release();
        return fail("readers-pread claim A failed");
    }
    disk.test_break_object(pages_a[3], q36::detail::DiskObjectKind::Main);
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match_a->entry_id, target);
    bool threw = false;
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-pread A restore hung"); rc != 0) {
            disk.release(match_a->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::runtime_error&) { threw = true; }
    if (!threw) {
        disk.cancel_restore();
        disk.release(match_a->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-pread broken page did not fail restore");
    }
    disk.cancel_restore();
    disk.release(match_a->entry_id);
    if (!disk.plan_match(text_prompt(tokens_a), q36::detail::prefix_hash_chain(text_prompt(tokens_a)))) {
        dest.release();
        alloc.release();
        return fail("readers-pread consumed A after pread fail");
    }
    if (!disk.claim(match_b->entry_id, match_b->hash_f, match_b->execution_frontier)) {
        dest.release();
        alloc.release();
        return fail("readers-pread claim B failed");
    }
    dest.release();
    dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    target.text = &dest;
    disk.restore_device(match_b->entry_id, target);
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-pread B restore hung"); rc != 0) {
            disk.release(match_b->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-pread B threw: " << e.what() << '\n';
        return 1;
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, kPages, ctx, "readers-pread-B") != 0) {
        disk.cancel_restore();
        disk.release(match_b->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-pread B restored wrong KV after A failed");
    }
    disk.cancel_restore();
    disk.release(match_b->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_cancel_does_not_poison_next(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-cancel");
    constexpr std::uint32_t kPages = 8;
    auto plan = plan_paged_cache(kPages * 2, kPages, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(kPages);
    alloc.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens(kPages * 64, 13);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-cancel spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-cancel match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-cancel claim failed");
    }
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.stream         = ctx.copy_stream;
    disk.test_arm_restore_job_barrier();
    disk.restore_device(match->entry_id, target);
    const auto deq_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!disk.test_restore_job_dequeued() && std::chrono::steady_clock::now() < deq_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!disk.test_restore_job_dequeued()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-cancel RestoreRead was not dequeued");
    }
    disk.cancel_restore();
    const auto drops_before = disk.snapshot().drops;
    disk.restore_device(match->entry_id, target);
    disk.test_release_restore_job_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-cancel restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-cancel restore threw: " << e.what() << '\n';
        return 1;
    }
    if (disk.snapshot().drops != drops_before) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-cancel poisoned the following restore");
    }
    if (expect_mapped_pages_equal(pool, alloc, dest, kPages, ctx, "readers-cancel") != 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-cancel restored wrong KV");
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_single_state_owner(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-state-owner");
    constexpr std::uint32_t kPages = 8;
    auto plan = plan_paged_cache(kPages * 2, kPages, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(kPages);
    alloc.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0xaa);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_state_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens(kPages * 64, 13);
    const auto ram_id = capture_tokens_hidden(ram, pool, alloc, ctx, tokens, hidden);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-state-owner spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-state-owner match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-state-owner claim failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    disk.test_arm_restore_state_barrier();
    disk.restore_device(match->entry_id, target);
    if (!wait_pred([&] { return disk.test_restore_state_entered(); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-state-owner state load did not enter");
    }
    if (!wait_pred([&] { return disk.test_window_inflight() == 0; }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-state-owner page fills did not drain");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto inflight = disk.test_restore_state_inflight();
    if (inflight != 1) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-state-owner concurrent state loads: inflight=" << inflight << '\n';
        return 1;
    }
    const auto spins = disk.test_restore_loop_idle_spins();
    if (spins != 0) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-state-owner non-owner idle spins=" << spins << '\n';
        return fail("readers-state-owner non-owner workers spun during state load");
    }
    disk.test_release_restore_state_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-state-owner restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-state-owner threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_idle_waits_for_restore(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-idle-wait");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_page_read_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-idle-wait spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-idle-wait match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-idle-wait claim A failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.test_set_payload_io_stall_ms(0);
    disk.test_arm_page_read_barrier();
    disk.restore_device(match->entry_id, target);
    if (!wait_pred([&] { return disk.test_page_read_entered(); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-idle-wait restore pread did not enter");
    }
    disk.request_idle_spill();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    if (disk.test_payload_io_entered()) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-idle-wait idle pwrite overlapped restore pread");
    }
    disk.test_release_page_read_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-idle-wait restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-idle-wait threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_claim_does_not_wait_other_ram_idle(ninfer::DeviceContext& ctx) {
    TmpDir dir("claim-duplex-idle");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("claim-duplex-idle spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("claim-duplex-idle match A failed");
    }
    const auto drops_before = disk.snapshot().drops;
    disk.test_set_payload_io_stall_ms(400);
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_payload_io_entered(); }, std::chrono::seconds(2))) {
        disk.test_set_payload_io_stall_ms(0);
        alloc.release();
        return fail("claim-duplex-idle idle never entered payload I/O");
    }
    const auto claim_start = std::chrono::steady_clock::now();
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        disk.test_set_payload_io_stall_ms(0);
        alloc.release();
        return fail("claim-duplex-idle claim A failed");
    }
    const auto claim_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - claim_start)
                              .count();
    if (claim_ms > 200) {
        disk.test_set_payload_io_stall_ms(0);
        disk.release(match->entry_id);
        alloc.release();
        std::cerr << "claim-duplex-idle claim waited " << claim_ms
                  << "ms for other-ram idle persist\n";
        return fail("claim-duplex-idle waited for other-ram idle persist");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.restore_device(match->entry_id, target);
    if (!wait_pred([&] { return disk.test_page_read_count() != 0; }, std::chrono::seconds(2))) {
        disk.test_set_payload_io_stall_ms(0);
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("claim-duplex-idle RestoreRead did not overlap in-hand idle pwrite");
    }
    disk.test_set_payload_io_stall_ms(0);
    if (disk.snapshot().drops != drops_before) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("claim-duplex-idle counted in-hand idle as a drop");
    }
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "claim-duplex-idle restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "claim-duplex-idle threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    return 0;
}

int test_restore_io_threads_emergency_waits_for_prefetch(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-emerg-wait");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_page_read_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-emerg-wait spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-emerg-wait match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-emerg-wait claim A failed");
    }
    disk.test_set_payload_io_stall_ms(0);
    disk.test_arm_page_read_barrier();
    disk.prefetch_window(match->entry_id, 2, 0);
    if (!wait_pred([&] { return disk.test_page_read_entered(); }, std::chrono::seconds(2))) {
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-emerg-wait prefetch pread did not enter");
    }
    std::atomic<bool> spilled{false};
    std::thread emergency([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const bool overlapped = disk.test_payload_io_entered();
    disk.test_release_page_read_barrier();
    if (emergency.joinable()) { emergency.join(); }
    disk.release(match->entry_id);
    alloc.release();
    if (overlapped) {
        return fail("readers-emerg-wait emergency pwrite overlapped prefetch pread");
    }
    if (!spilled.load()) { return fail("readers-emerg-wait emergency spill failed after prefetch"); }
    return 0;
}

int test_restore_io_threads_prefetch_at_most_two_pages(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-prefetch-2");
    constexpr std::uint32_t kPages = 8;
    auto plan = plan_paged_cache(kPages * 2, kPages, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(kPages);
    alloc.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, alloc, 21);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(kPages * 64, 13);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-prefetch-2 spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-prefetch-2 match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-prefetch-2 claim failed");
    }
    disk.prefetch_window(match->entry_id, kPages, 0);
    if (!wait_pred([&] { return disk.test_page_read_count() > 0; }, std::chrono::seconds(2))) {
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-prefetch-2 did not read any page");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const int reads = disk.test_page_read_count();
    disk.release(match->entry_id);
    alloc.release();
    if (reads > 2) {
        std::cerr << "readers-prefetch-2 queued " << reads << " page reads\n";
        return fail("readers-prefetch-2 prefetched more than 2 pages");
    }
    return 0;
}

int test_restore_io_threads_restore_does_not_double_fill_prefetch(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-no-dup");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens(128, 4);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-no-dup spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-no-dup match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-no-dup claim failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.test_arm_restore_job_barrier();
    disk.prefetch_window(match->entry_id, 2, 0);
    if (!wait_pred([&] { return disk.test_restore_job_dequeued(); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-no-dup prefetch was not dequeued");
    }
    disk.restore_device(match->entry_id, target);
    disk.test_release_restore_job_barrier();
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-no-dup restore hung"); rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-no-dup threw: " << e.what() << '\n';
        return 1;
    }
    const int reads = disk.test_page_read_count();
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (reads != 2) {
        std::cerr << "readers-no-dup page reads=" << reads << '\n';
        return fail("readers-no-dup RestoreRead duplicated PrefetchWindow pages");
    }
    return 0;
}

int test_restore_io_threads_cancel_drains_state(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-cancel-state");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    ninfer::DeviceBuffer hid(64);
    hid.fill(0xaa);
    ninfer::Tensor hidden(hid.p, ninfer::DType::U8, {64});
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    cfg.hidden_bytes = 64;
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens(128, 4);
    const auto ram_id = capture_tokens_hidden(ram, pool, alloc, ctx, tokens, hidden);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-cancel-state spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-cancel-state match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-cancel-state claim failed");
    }
    ninfer::DeviceBuffer hid_out(64);
    ninfer::Tensor hidden_out(hid_out.p, ninfer::DType::U8, {64});
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.tail_hidden    = &hidden_out;
    target.stream         = ctx.copy_stream;
    disk.test_set_state_decode_stall_ms(150);
    disk.restore_device(match->entry_id, target);
    if (!wait_pred([&] { return disk.test_restore_state_inflight() >= 1; },
                    std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-cancel-state never entered state load");
    }
    disk.cancel_restore();
    const auto inflight = disk.test_restore_state_inflight();
    disk.test_set_state_decode_stall_ms(0);
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (inflight != 0) {
        std::cerr << "readers-cancel-state inflight after cancel=" << inflight << '\n';
        return fail("readers-cancel-state returned with state I/O still running");
    }
    return 0;
}

int test_restore_io_threads_emergency_waits_for_in_hand_reader(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-emerg-claim");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-emerg-claim spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-emerg-claim match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-emerg-claim claim A failed");
    }
    disk.test_set_payload_io_stall_ms(0);
    disk.test_arm_restore_job_barrier();
    disk.prefetch_window(match->entry_id, 2, 0);
    if (!wait_pred([&] { return disk.test_restore_job_dequeued(); }, std::chrono::seconds(2))) {
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-emerg-claim prefetch was not dequeued");
    }
    std::atomic<bool> spilled{false};
    std::thread emergency([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const bool overlapped = disk.test_payload_io_entered();
    disk.test_release_restore_job_barrier();
    if (emergency.joinable()) { emergency.join(); }
    disk.release(match->entry_id);
    alloc.release();
    if (overlapped) {
        return fail("readers-emerg-claim emergency pwrite overlapped in-hand prefetch");
    }
    if (!spilled.load()) {
        return fail("readers-emerg-claim emergency spill failed after in-hand prefetch");
    }
    return 0;
}

int test_restore_io_threads_readers_sleep_while_emergency(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-emerg-sleep");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-emerg-sleep spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-emerg-sleep match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-emerg-sleep claim A failed");
    }
    disk.test_set_payload_io_stall_ms(300);
    std::atomic<bool> spilled{false};
    std::thread emergency([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    if (!wait_pred([&] { return disk.test_payload_io_entered(); }, std::chrono::seconds(2))) {
        disk.test_set_payload_io_stall_ms(0);
        if (emergency.joinable()) { emergency.join(); }
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-emerg-sleep emergency never entered payload I/O");
    }
    disk.test_reset_restore_loop_idle_spins();
    disk.prefetch_window(match->entry_id, 2, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto spins = disk.test_restore_loop_idle_spins();
    const int reads   = disk.test_page_read_count();
    disk.test_set_payload_io_stall_ms(0);
    if (emergency.joinable()) { emergency.join(); }
    disk.release(match->entry_id);
    alloc.release();
    if (reads != 0) {
        return fail("readers-emerg-sleep prefetch pread ran during emergency");
    }
    if (spins != 0) {
        std::cerr << "readers-emerg-sleep restore_loop idle spins=" << spins << '\n';
        return fail("readers-emerg-sleep restore workers spun on parked prefetch");
    }
    if (!spilled.load()) { return fail("readers-emerg-sleep emergency spill failed"); }
    return 0;
}

int test_restore_io_threads_restore_waits_for_in_hand_emergency(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-restore-emerg");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-restore-emerg spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-restore-emerg match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-restore-emerg claim A failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.test_set_payload_io_stall_ms(400);
    std::atomic<bool> spilled{false};
    std::thread emergency([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    if (!wait_pred([&] { return disk.test_payload_io_entered(); }, std::chrono::seconds(2))) {
        disk.test_set_payload_io_stall_ms(0);
        if (emergency.joinable()) { emergency.join(); }
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-restore-emerg emergency never entered payload I/O");
    }
    disk.restore_device(match->entry_id, target);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool overlapped = disk.test_page_read_count() != 0;
    disk.test_set_payload_io_stall_ms(0);
    if (emergency.joinable()) { emergency.join(); }
    if (overlapped) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-restore-emerg RestoreRead overlapped in-hand emergency pwrite");
    }
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-restore-emerg restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-restore-emerg threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (!spilled.load()) { return fail("readers-restore-emerg emergency spill failed"); }
    return 0;
}

int test_restore_io_threads_prefetch_waits_for_in_hand_idle(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-prefetch-idle");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-prefetch-idle spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-prefetch-idle match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-prefetch-idle claim A failed");
    }
    disk.test_set_payload_io_stall_ms(400);
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_payload_io_entered(); }, std::chrono::seconds(2))) {
        disk.test_set_payload_io_stall_ms(0);
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-prefetch-idle idle never entered payload I/O");
    }
    disk.prefetch_window(match->entry_id, 2, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool overlapped = disk.test_page_read_count() != 0;
    disk.test_set_payload_io_stall_ms(0);
    disk.release(match->entry_id);
    alloc.release();
    if (overlapped) {
        return fail("readers-prefetch-idle prefetch pread overlapped in-hand idle pwrite");
    }
    return 0;
}

int test_restore_io_threads_wait_idle_waits_for_in_hand_reader(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-wait-idle-claim");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_restore_job_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-wait-idle-claim spill A failed");
    }
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-wait-idle-claim match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-wait-idle-claim claim A failed");
    }
    disk.test_arm_restore_job_barrier();
    disk.prefetch_window(match->entry_id, 2, 0);
    if (!wait_pred([&] { return disk.test_restore_job_dequeued(); }, std::chrono::seconds(2))) {
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-wait-idle-claim prefetch was not dequeued");
    }
    std::atomic<bool> idle_done{false};
    std::thread idle_waiter([&] {
        disk.wait_idle_and_fsync();
        idle_done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const bool returned_early = idle_done.load();
    disk.test_release_restore_job_barrier();
    if (idle_waiter.joinable()) { idle_waiter.join(); }
    disk.release(match->entry_id);
    alloc.release();
    if (returned_early) {
        return fail("readers-wait-idle-claim wait_idle returned during in-hand reader job");
    }
    return 0;
}

int test_restore_io_threads_restore_waits_for_taken_emergency(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-restore-taken");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_payload_take_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-restore-taken spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-restore-taken match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-restore-taken claim A failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.test_arm_payload_take_barrier();
    std::atomic<bool> spilled{false};
    std::thread emergency([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    if (!wait_pred([&] { return disk.test_payload_take_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_payload_take_barrier();
        if (emergency.joinable()) { emergency.join(); }
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-restore-taken emergency was not taken");
    }
    disk.restore_device(match->entry_id, target);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool overlapped = disk.test_page_read_count() != 0;
    disk.test_release_payload_take_barrier();
    if (emergency.joinable()) { emergency.join(); }
    if (overlapped) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-restore-taken RestoreRead overlapped taken emergency before payload");
    }
    try {
        if (const int rc = wait_restore_bounded(disk, ctx, "readers-restore-taken restore hung");
            rc != 0) {
            disk.release(match->entry_id);
            dest.release();
            alloc.release();
            return rc;
        }
    } catch (const std::exception& e) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        std::cerr << "readers-restore-taken threw: " << e.what() << '\n';
        return 1;
    }
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (!spilled.load()) { return fail("readers-restore-taken emergency spill failed"); }
    return 0;
}

int test_restore_io_threads_prefetch_does_not_cancel_in_hand_idle(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-prefetch-nocancel");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() {
            disk.test_release_payload_take_barrier();
            disk.test_release_restore_job_barrier();
        }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-prefetch-nocancel spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-prefetch-nocancel match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-prefetch-nocancel claim A failed");
    }
    disk.test_arm_payload_take_barrier();
    disk.test_arm_restore_job_barrier();
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_payload_take_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_payload_take_barrier();
        disk.test_release_restore_job_barrier();
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-prefetch-nocancel idle was not taken");
    }
    disk.prefetch_window(match->entry_id, 2, 0);
    (void)wait_pred([&] { return disk.test_restore_job_dequeued(); }, std::chrono::milliseconds(200));
    disk.test_release_payload_take_barrier();
    disk.test_release_restore_job_barrier();
    std::atomic<bool> idle_done{false};
    std::thread idle_waiter([&] {
        disk.wait_idle_and_fsync();
        idle_done.store(true);
    });
    if (!wait_pred([&] { return idle_done.load(); }, std::chrono::seconds(8))) {
        idle_waiter.detach();
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-prefetch-nocancel wait_idle hung");
    }
    if (idle_waiter.joinable()) { idle_waiter.join(); }
    const bool durable = disk.ram_is_durable(ram_b);
    disk.release(match->entry_id);
    alloc.release();
    if (!durable) {
        return fail("readers-prefetch-nocancel prefetch cancelled in-hand idle");
    }
    return 0;
}

int test_restore_io_threads_already_does_not_treat_unfilled_as_page0(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-already-00");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_slot_assign_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens(128, 4);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    if (!disk.emergency_spill_ram(ram_id)) {
        alloc.release();
        return fail("readers-already-00 spill failed");
    }
    const auto prompt = text_prompt(tokens);
    const auto match  = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-already-00 match failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-already-00 claim failed");
    }
    auto dest = pool.reserve(4);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.copy_stream;
    disk.test_arm_fail_page_read();
    disk.test_arm_slot_assign_barrier();
    disk.restore_device(match->entry_id, target);
    if (!wait_pred(
            [&] {
                return disk.test_slot_assign_entered() && disk.test_window_assigned(0, 1) &&
                       !disk.test_restore_read_queued(0, 0) && !disk.test_reader_claim(0, 0);
            },
            std::chrono::seconds(2))) {
        disk.cancel_restore();
        disk.release(match->entry_id);
        dest.release();
        alloc.release();
        return fail("readers-already-00 page 1 assigned without page 0 still covering");
    }
    disk.restore_device(match->entry_id, target);
    const bool covering = disk.test_restore_read_queued(0, 0) || disk.test_reader_claim(0, 0) ||
                          disk.test_window_assigned(0, 0);
    disk.test_release_slot_assign_barrier();
    disk.cancel_restore();
    disk.release(match->entry_id);
    dest.release();
    alloc.release();
    if (!covering) {
        return fail("readers-already-00 skipped RestoreRead for main page 0");
    }
    return 0;
}

int test_restore_io_threads_payload_throw_does_not_leak_inflight(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-payload-throw");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_payload_take_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-payload-throw spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_c(128, 8);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, tokens_c);
    disk.note_ram_resident(ram_c, 0);
    disk.test_arm_payload_take_barrier();
    disk.test_arm_fail_after_payload_take();
    std::thread emergency([&] { (void)disk.emergency_spill_ram(ram_c); });
    if (!wait_pred([&] { return disk.test_payload_take_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_payload_take_barrier();
        if (emergency.joinable()) { emergency.join(); }
        alloc.release();
        return fail("readers-payload-throw emergency was not taken");
    }
    disk.test_release_payload_take_barrier();
    std::atomic<bool> idle_done{false};
    std::thread idle_waiter([&] {
        disk.wait_idle_and_fsync();
        idle_done.store(true);
    });
    if (!wait_pred([&] { return idle_done.load(); }, std::chrono::seconds(5))) {
        idle_waiter.detach();
        if (emergency.joinable()) { emergency.detach(); }
        alloc.release();
        return fail("readers-payload-throw wait_idle hung after payload-take throw");
    }
    if (idle_waiter.joinable()) { idle_waiter.join(); }
    if (emergency.joinable()) { emergency.join(); }
    const auto inflight = disk.test_payload_io_inflight();
    alloc.release();
    if (inflight != 0) {
        std::cerr << "readers-payload-throw inflight=" << inflight << '\n';
        return fail("readers-payload-throw leaked payload_io_inflight_");
    }
    return 0;
}

int test_restore_io_threads_idle_beats_queued_prefetch(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-idle-gt-prefetch");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_payload_take_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens_a(128, 4);
    const auto ram_a = capture_tokens(ram, pool, alloc, ctx, tokens_a);
    disk.note_ram_resident(ram_a, 0);
    if (!disk.emergency_spill_ram(ram_a)) {
        alloc.release();
        return fail("readers-idle-gt-prefetch spill A failed");
    }
    fill_logical_pages(pool, alloc, 9);
    std::vector<ninfer::TokenId> tokens_b(128, 8);
    const auto ram_b = capture_tokens(ram, pool, alloc, ctx, tokens_b);
    disk.note_ram_resident(ram_b, 0);
    const auto prompt = text_prompt(tokens_a);
    const auto match   = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        alloc.release();
        return fail("readers-idle-gt-prefetch match A failed");
    }
    if (!disk.claim(match->entry_id, match->hash_f, match->execution_frontier)) {
        alloc.release();
        return fail("readers-idle-gt-prefetch claim A failed");
    }
    disk.test_arm_payload_take_barrier();
    disk.request_idle_spill();
    if (!wait_pred([&] { return disk.test_payload_take_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_payload_take_barrier();
        disk.release(match->entry_id);
        alloc.release();
        return fail("readers-idle-gt-prefetch idle was not taken");
    }
    disk.prefetch_window(match->entry_id, 2, 0);
    disk.test_release_payload_take_barrier();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto preempted = disk.test_prefetch_preempted_idle();
    disk.release(match->entry_id);
    alloc.release();
    if (preempted != 0) {
        std::cerr << "readers-idle-gt-prefetch prefetch-over-idle=" << preempted << '\n';
        return fail("readers-idle-gt-prefetch prefetch took while idle remained queued");
    }
    return 0;
}

int test_restore_io_threads_emergency_waits_for_filled_window(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-filled-window");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 11);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> parent_tokens(63, 4);
    const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent_tokens);
    disk.note_ram_resident(ram_p, 0);
    if (!disk.emergency_spill_ram(ram_p)) {
        alloc.release();
        return fail("readers-filled-window parent spill failed");
    }
    const auto match_p = disk.plan_match(text_prompt(parent_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(parent_tokens)));
    if (!match_p) {
        alloc.release();
        return fail("readers-filled-window parent match failed");
    }
    const std::uint64_t entry_id = match_p->entry_id;
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(entry_id, target);
    if (!wait_pred([&] { return disk.test_window_filled_for(entry_id); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        dest.release();
        alloc.release();
        return fail("readers-filled-window restore did not fill a window slot");
    }
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> child_tokens = parent_tokens;
    child_tokens.push_back(0);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child_tokens);
    ram.set_disk_entry_id(ram_c, entry_id);
    disk.note_ram_resident(ram_c, entry_id);
    std::atomic<bool> spilled{false};
    std::thread extend([&] { spilled.store(disk.emergency_spill_ram(ram_c)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const bool overlapped = spilled.load();
    disk.cancel_restore();
    if (extend.joinable()) { extend.join(); }
    dest.release();
    alloc.release();
    if (overlapped) {
        return fail("readers-filled-window emergency committed over filled restore window");
    }
    return 0;
}

int test_restore_io_threads_destructor_joins_queued_emergency(ninfer::DeviceContext& ctx) {
    TmpDir dir("readers-dtor-join");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(64ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 11);
    auto cfg = reader_disk_config(dir.path, ram, pool, 64ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    std::vector<ninfer::TokenId> parent_tokens(63, 4);
    const auto ram_p = capture_tokens(ram, pool, alloc, ctx, parent_tokens);
    disk.note_ram_resident(ram_p, 0);
    if (!disk.emergency_spill_ram(ram_p)) {
        alloc.release();
        return fail("readers-dtor-join parent spill failed");
    }
    const auto match_p = disk.plan_match(text_prompt(parent_tokens),
                                           q36::detail::prefix_hash_chain(text_prompt(parent_tokens)));
    if (!match_p) {
        alloc.release();
        return fail("readers-dtor-join parent match failed");
    }
    const std::uint64_t entry_id = match_p->entry_id;
    auto dest = pool.reserve(2);
    dest.materialize_pages(1, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.copy_stream;
    disk.restore_device(entry_id, target);
    if (!wait_pred([&] { return disk.test_window_filled_for(entry_id); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        dest.release();
        alloc.release();
        return fail("readers-dtor-join restore did not fill a window slot");
    }
    fill_logical_pages(pool, alloc, 99);
    ctx.synchronize_all();
    std::vector<ninfer::TokenId> child_tokens = parent_tokens;
    child_tokens.push_back(0);
    const auto ram_c = capture_tokens(ram, pool, alloc, ctx, child_tokens);
    ram.set_disk_entry_id(ram_c, entry_id);
    disk.note_ram_resident(ram_c, entry_id);
    std::thread extend([&] { (void)disk.emergency_spill_ram(ram_c); });
    if (!wait_pred([&] { return disk.test_emergency_queued(); }, std::chrono::seconds(2))) {
        disk.cancel_restore();
        if (extend.joinable()) { extend.join(); }
        dest.release();
        alloc.release();
        return fail("readers-dtor-join emergency was not queued");
    }
    std::atomic<bool> joined{false};
    std::thread stopper([&] {
        disk.test_stop_io_threads();
        joined.store(true);
    });
    if (!wait_pred([&] { return joined.load(); }, std::chrono::seconds(8))) {
        stopper.detach();
        extend.detach();
        dest.release();
        alloc.release();
        return fail("readers-dtor-join destructor hung with queued same-entry emergency");
    }
    if (stopper.joinable()) { stopper.join(); }
    if (extend.joinable()) { extend.join(); }
    dest.release();
    alloc.release();
    return 0;
}

int test_emergency_spill_stopping_waits_for_in_hand_commit(ninfer::DeviceContext& ctx) {
    TmpDir dir("spill-stop-commit");
    auto plan = plan_paged_cache(8, 4, 2,
                                  {{ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::I8, 64, 2},
                                   {ninfer::DType::FP16, 1, 2},
                                   {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    q36::detail::KVRamCache ram(32ULL << 20);
    auto alloc = pool.reserve(4);
    alloc.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, alloc, 5);
    auto cfg = reader_disk_config(dir.path, ram, pool, 32ULL << 20, 4096);
    q36::detail::KVDiskCache disk(std::move(cfg));
    struct BarrierGuard {
        q36::detail::KVDiskCache& disk;
        ~BarrierGuard() { disk.test_release_payload_take_barrier(); }
    } barrier{disk};
    std::vector<ninfer::TokenId> tokens(128, 4);
    const auto ram_id = capture_tokens(ram, pool, alloc, ctx, tokens);
    disk.note_ram_resident(ram_id, 0);
    disk.test_arm_payload_take_barrier();
    std::atomic<bool> returned{false};
    std::atomic<bool> ok{false};
    std::thread spill([&] {
        ok.store(disk.emergency_spill_ram(ram_id));
        returned.store(true);
    });
    if (!wait_pred([&] { return disk.test_payload_take_entered(); }, std::chrono::seconds(2))) {
        disk.test_release_payload_take_barrier();
        if (spill.joinable()) { spill.join(); }
        alloc.release();
        return fail("spill-stop-commit emergency was not taken");
    }
    disk.test_set_stopping();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const bool returned_early = returned.load();
    const bool spilled          = ok.load();
    disk.test_release_payload_take_barrier();
    disk.test_stop_io_threads();
    if (spill.joinable()) { spill.join(); }
    const bool durable = disk.ram_is_durable(ram_id);
    alloc.release();
    if (returned_early && !spilled) {
        return fail("spill-stop-commit emergency_spill_ram returned false while commit was in-hand");
    }
    if (!ok.load()) {
        return fail("spill-stop-commit emergency spill failed after drain");
    }
    if (!durable) {
        return fail("spill-stop-commit drain did not mark RAM durable");
    }
    return 0;
}

int test_disk_crc32c_matches_naive_oracle() {
    constexpr char kCheck[] = "123456789";
    if (q36::detail::KVDiskCache::test_crc32c(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(kCheck), sizeof(kCheck) - 1)) != 0xe3069283U) {
        return fail("disk CRC32C failed the canonical check vector");
    }
    std::vector<std::uint8_t> data((1U << 20) + 32);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (std::uint8_t& byte : data) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        byte = static_cast<std::uint8_t>((state * 0x2545f4914f6cdd1dULL) >> 56);
    }
    constexpr std::size_t kSizes[] = {0,
                                      1,
                                      7,
                                      8,
                                      15,
                                      4095,
                                      4096,
                                      4097,
                                      10'000,
                                      (512U << 10) - 1,
                                      512U << 10,
                                      (512U << 10) + 1,
                                      1U << 20};
    for (std::size_t offset = 0; offset < 8; ++offset) {
        for (std::size_t size : kSizes) {
            const auto bytes = std::span<const std::uint8_t>(data.data() + offset, size);
            if (q36::detail::KVDiskCache::test_crc32c(bytes) !=
                test_crc32c(bytes.data(), bytes.size())) {
                return fail("disk CRC32C diverged from the naive oracle");
            }
        }
    }
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    failures += test_disk_crc32c_matches_naive_oracle();
    ninfer::DeviceContext ctx(0);
    auto paged_plan = plan_paged_cache(12, 12, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);

    failures += test_gather_both_orders(ctx);
    failures += test_batch_page_unpack(ctx);
    failures += test_device_scatter_page_unpack(ctx);
    failures += test_lock_and_fingerprint(ctx, paged_pool);
    failures += test_ram_io_pin_and_capture_id(ctx, paged_pool);
    failures += test_flush_reports_progress(ctx, paged_pool);
    failures += test_spill_match_extend_branch(ctx, paged_pool);
    failures += test_mtp_f1_backend_pages(ctx);
    failures += test_ladders_and_rollback(ctx, paged_pool);
    failures += test_dflash_cyclic(ctx, paged_pool);
    failures += test_crash_before_meta(ctx, paged_pool);
    failures += test_skip_too_long_gc(ctx, paged_pool);
    failures += test_zstd_fail_writes_raw(ctx, paged_pool);
    failures += test_zstd_reopen_with_compress_off(ctx, paged_pool);
    failures += test_zstd_capacity_uses_physical_bytes(ctx, paged_pool);
    failures += test_packset_bootstrap_recovery(ctx, paged_pool);
    failures += test_packset_bootstrap_phase_matrix(paged_pool);
    failures += test_pack_map_recovery_boundaries(ctx, paged_pool);
    failures += test_pack_page_batch_variants(ctx, paged_pool);
    failures += test_pack_rollover_and_partial_pwritev(ctx, paged_pool);
    failures += test_compaction_reclaims_retired_pack_generation(ctx, paged_pool);
    failures += test_spill_durability_phase_matrix(ctx, paged_pool);
    failures += test_compaction_publication_phase_matrix(ctx, paged_pool);
    failures += test_tombstone_durability_phase_matrix(ctx, paged_pool);
    failures += test_physical_preflight_rejects_before_pack_write(ctx, paged_pool);
    failures += test_low_free_subthreshold_garbage_rejects_without_write(ctx, paged_pool);
    failures += test_restore_pread_fail(ctx, paged_pool);
    failures += test_cancelled_restore_job_does_not_poison_next(ctx, paged_pool);
    failures += test_stale_state_load_does_not_publish_wrong_head(ctx, paged_pool);
    failures += test_stale_state_failure_does_not_poison_next(ctx, paged_pool);
    failures += test_copy_event_lease_survives_next_restore(ctx, paged_pool);
    failures += test_wait_copies_does_not_follow_replacement_generation(ctx, paged_pool);
    failures += test_disk_load_timing_survives_session_close(ctx, paged_pool);
    failures += test_restore_load_includes_pread(ctx, paged_pool);
    failures += test_restore_load_is_ssd_to_host_wall(ctx, paged_pool);
    failures += test_h2d_not_billed_until_wait_copies(ctx, paged_pool);
    failures += test_abandoned_prefetch_does_not_bill_load(ctx, paged_pool);
    failures += test_cancelled_state_read_does_not_bill_load(ctx, paged_pool);
    failures += test_wait_copies_injected_failure_is_not_pread(paged_pool);
    failures += test_wait_copies_after_close_leases_ticket(ctx, paged_pool);
    failures += test_released_prefetch_does_not_poison_restore(ctx, paged_pool);
    failures += test_stale_prefetch_does_not_survive_inplace_extend(ctx, paged_pool);
    failures += test_abandoned_prefetch_failure_does_not_poison_other_restore(ctx, paged_pool);
    failures += test_restore_ticket_survives_host_wait_until_stream_wait(ctx, paged_pool);
    failures += test_restore_state_setup_failure_does_not_spin(ctx, paged_pool);
    failures += test_cancel_before_h2d_releases_ticket_event(ctx, paged_pool);
    failures += test_stale_state_io_aborts_between_objects(ctx, paged_pool);
    failures += test_page_read_throw_does_not_stick_inflight(ctx, paged_pool);
    failures += test_capture_returns_id(ctx, paged_pool);
    failures += test_wait_copies_idle_does_not_hang(paged_pool);
    failures += test_restore_three_pages_and_prefetch(ctx, paged_pool);
    failures += test_restore_completion_waits_for_final_scatter(ctx, paged_pool);
    failures += test_extend_preserves_disk_claim(ctx, paged_pool);
    failures += test_claimed_generation_not_extended(ctx, paged_pool);
    failures += test_claimed_parent_branch_clones_partial_page(ctx, paged_pool);
    failures += test_consume_failure_keeps_ram_note(ctx, paged_pool);
    failures += test_claim_cancels_in_flight_idle_extend(ctx, paged_pool);
    failures += test_cancel_after_meta_rename_keeps_parent(ctx, paged_pool);
    failures += test_claim_missing_after_fifo_evict(ctx, paged_pool);
    failures += test_stale_plan_claim_rejects_extended_generation(ctx, paged_pool);
    failures += test_post_rename_failure_keeps_a_valid_generation(ctx, paged_pool);
    failures += test_rollback_meta_failure_keeps_a_valid_generation(ctx, paged_pool);
    failures += test_stale_manifest_discovers_branch(ctx, paged_pool);
    failures += test_gc_skipped_shared_pages(ctx, paged_pool);
    failures += test_truncated_page_payload_is_skipped(ctx, paged_pool);
    failures += test_truncated_state_payload_is_skipped(ctx, paged_pool);
    failures += test_live_state_gap_crc_is_checked(ctx, paged_pool);
    failures += test_empty_codec_state_is_skipped(ctx, paged_pool);
    failures += test_wrapped_compressed_bytes_is_skipped(ctx, paged_pool);
    failures += test_short_page_id_vector_is_skipped(ctx, paged_pool);
    failures += test_corrupt_unselected_checkpoint_is_omitted(ctx, paged_pool);
    failures += test_corrupt_checkpoint_fails_restore(ctx, paged_pool);
    failures += test_missing_ledger_skips_entry(ctx, paged_pool);
    failures += test_manifest_duplicate_ids_rebuild(ctx, paged_pool);
    failures += test_meta_entry_id_mismatch_is_skipped(ctx, paged_pool);
    failures += test_refresh_clears_absent_rollback(ctx, paged_pool);
    failures += test_plan_match_does_not_wait_on_payload_io(ctx, paged_pool);
    failures += test_populate_does_not_hold_mutex_across_pread(ctx, paged_pool);
    failures += test_idle_pin_aborts_for_ram_claim(ctx, paged_pool);
    failures += test_ram_idle_exclusion_covers_claim(ctx, paged_pool);
    failures += test_discard_keeps_note_until_evict(ctx, paged_pool);
    failures += test_checkpoint_images_survive_host_load(ctx, paged_pool);
    failures += test_ram_header_ticket_roundtrip(ctx, paged_pool);
    failures += test_inclusive_disk_after_ram_consume(ctx, paged_pool);
    failures += test_wait_idle_after_prefetch_and_claimed_evict(ctx, paged_pool);
    failures += test_wait_idle_with_live_prefetch(ctx, paged_pool);
    failures += test_cancel_idle_spill_unpins_peek_victim(ctx, paged_pool);
    failures += test_ticket_write_fail_unpins(ctx, paged_pool);
    failures += test_zero_kv_gdn_restore_and_cancel(ctx, paged_pool);
    failures += test_rewrite_restore_skips_frontier_gdn(ctx, paged_pool);
    failures += test_pinned_state_h2d_matches_heap(ctx, paged_pool);
    failures += test_c1_order_disk_b_ram_c(ctx, paged_pool);
    failures += test_recapture_keeps_ladders(ctx, paged_pool);
    failures += test_mixed_zstd_and_raw_hidden(ctx, paged_pool);
    failures += test_dflash_open_skips_missing_cyclic(ctx, paged_pool);
    failures += test_mtp_backend_cow_and_dflash2(ctx);
    failures += test_branch_share_point_l(ctx, paged_pool);
    failures += test_fingerprint_and_location_file(ctx, paged_pool);
    failures += test_restore_second_entry_without_cancel(ctx, paged_pool);
    failures += test_cancel_restore_then_restore_other(ctx, paged_pool);
    failures += test_refresh_does_not_exceed_capacity(ctx, paged_pool);
    failures += test_claim_rejects_checkpoint_only_refresh(ctx, paged_pool);
    failures += test_claim_rejects_replaced_head_at_same_frontier(ctx, paged_pool);
    failures += test_invalid_checkpoint_kind_is_skipped(ctx, paged_pool);
    failures += test_refresh_does_not_evict_sibling(ctx, paged_pool);
    failures += test_raw_state_unequal_codec_bytes_is_skipped(ctx, paged_pool);
    failures += test_gdn_checkpoint_without_hidden_is_not_selected(ctx, paged_pool);
    failures += test_load_host_misses_evicted_entry(ctx, paged_pool);
    failures += test_wrong_gdn_decoded_size_is_skipped(ctx, paged_pool);
    failures += test_plan_match_does_not_wait_on_manifest(ctx, paged_pool);
    failures += test_disk_fifo_evict_clears_ram_durable(ctx, paged_pool);
    failures += test_lock_does_not_wipe_tmp_before_flock(ctx, paged_pool);
    failures += test_missing_current_hidden_is_not_advertised(ctx, paged_pool);
    failures += test_claim_does_not_hang_when_idle_spill_has_no_inflight(ctx, paged_pool);
    failures += test_fifo_evict_rewrites_manifest(ctx, paged_pool);
    failures += test_smaller_capacity_open_persists_eviction(ctx, paged_pool);
    failures += test_misplaced_checkpoint_kind_is_skipped(ctx, paged_pool);
    failures += test_cancel_create_after_rename_does_not_leak(ctx, paged_pool);
    failures += test_cancel_branch_after_rename_drops_shared_refs(ctx, paged_pool);
    failures += test_incomplete_rollback_does_not_poison_ladder_restore(ctx, paged_pool);
    failures += test_shutdown_retries_failed_ram_spill(ctx, paged_pool);
    failures += test_failed_entry_unlink_does_not_resurrect(ctx, paged_pool);
    failures += test_oversized_state_object_does_not_abort_restore(ctx, paged_pool);
    failures += test_cancel_create_fsync_does_not_hold_mutex(ctx, paged_pool);
    failures += test_refresh_does_not_evict_self_before_commit(ctx, paged_pool);
    failures += test_oversized_host_files_are_skipped_on_open(ctx, paged_pool);
    failures += test_tombstone_survives_failed_entries_unlink(ctx, paged_pool);
    failures += test_text_kv_valid_cannot_hide_short_pages(ctx, paged_pool);
    failures += test_tombstone_does_not_poison_reused_ids(ctx, paged_pool);
    failures += test_listed_tombstone_is_not_loaded(ctx, paged_pool);
    failures += test_tombstone_write_failure_does_not_unlink(ctx, paged_pool);
    failures += test_rollback_after_rename_keeps_previous(ctx, paged_pool);
    failures += test_wrong_hidden_size_is_skipped(ctx, paged_pool);
    failures += test_huge_identity_token_count_is_skipped(ctx, paged_pool);
    failures += test_huge_compressed_size_is_skipped(ctx, paged_pool);
    failures += test_index_scan_is_capped(ctx, paged_pool);
    failures += test_tombstone_survives_until_manifest_persist(ctx, paged_pool);
    failures += test_capped_extra_object_ids_are_not_reused(ctx, paged_pool);
    failures += test_failed_object_write_does_not_leak(ctx, paged_pool);
    failures += test_kind_conflict_is_skipped(ctx, paged_pool);
    failures += test_short_text_kv_valid_is_skipped(ctx, paged_pool);
    failures += test_skipped_corrupt_ids_do_not_poison_valid_entry(ctx, paged_pool);
    failures += test_failed_object_unlink_keeps_tombstone(ctx, paged_pool);
    failures += test_ledger_frontier_must_be_execution_plus_one(ctx, paged_pool);
    failures += test_speculative_head_without_backend_pages_is_skipped(ctx);
    failures += test_fingerprint_state_over_64mib_is_accepted(ctx, paged_pool);
    failures += test_torn_tombstone_does_not_drop_entry(ctx, paged_pool);
    failures += test_too_long_corrupt_identity_is_not_skipped(ctx, paged_pool);
    failures += test_stale_manifest_extras_keep_entry_id_order(ctx, paged_pool);
    failures += test_prepare_spill_exception_releases_pins(ctx, paged_pool);
    failures += test_failed_spill_jobs_do_not_commit_retry(ctx, paged_pool);
    failures += test_uncertainty_holds_reclaim_on_evict_and_reopen(ctx, paged_pool);
    failures += test_emergency_promote_survives_restore(ctx, paged_pool);
    failures += test_corrupt_meta_does_not_delete_objects(ctx, paged_pool);
    failures += test_capped_extras_keep_smallest_and_objects(ctx, paged_pool);
    failures += test_zero_hidden_bytes_preserves_heads(ctx, paged_pool);
    failures += test_restore_io_threads_roundtrip(ctx);
    failures += test_restore_io_threads_idle_extra_readers(ctx);
    failures += test_restore_io_threads_pread_fail(ctx);
    failures += test_restore_io_threads_cancel_does_not_poison_next(ctx);
    failures += test_restore_io_threads_single_state_owner(ctx);
    failures += test_restore_io_threads_idle_waits_for_restore(ctx);
    failures += test_claim_does_not_wait_other_ram_idle(ctx);
    failures += test_restore_io_threads_emergency_waits_for_prefetch(ctx);
    failures += test_restore_io_threads_prefetch_at_most_two_pages(ctx);
    failures += test_restore_io_threads_restore_does_not_double_fill_prefetch(ctx);
    failures += test_restore_io_threads_cancel_drains_state(ctx);
    failures += test_restore_io_threads_emergency_waits_for_in_hand_reader(ctx);
    failures += test_restore_io_threads_readers_sleep_while_emergency(ctx);
    failures += test_restore_io_threads_restore_waits_for_in_hand_emergency(ctx);
    failures += test_restore_io_threads_prefetch_waits_for_in_hand_idle(ctx);
    failures += test_restore_io_threads_wait_idle_waits_for_in_hand_reader(ctx);
    failures += test_restore_io_threads_restore_waits_for_taken_emergency(ctx);
    failures += test_restore_io_threads_prefetch_does_not_cancel_in_hand_idle(ctx);
    failures += test_restore_io_threads_already_does_not_treat_unfilled_as_page0(ctx);
    failures += test_restore_io_threads_payload_throw_does_not_leak_inflight(ctx);
    failures += test_restore_io_threads_idle_beats_queued_prefetch(ctx);
    failures += test_restore_io_threads_emergency_waits_for_filled_window(ctx);
    failures += test_restore_io_threads_destructor_joins_queued_emergency(ctx);
    failures += test_emergency_spill_stopping_waits_for_in_hand_commit(ctx);

    return failures == 0 ? 0 : fail("kv disk cache unit test failed");
}

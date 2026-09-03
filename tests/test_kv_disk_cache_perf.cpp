#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/statfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;
namespace fs  = std::filesystem;

constexpr std::uint32_t kBenchmarkTokens = 33'981;
constexpr std::uint32_t kPages =
    (kBenchmarkTokens + ninfer::kPagedKVPageSize - 1) / ninfer::kPagedKVPageSize;
static_assert(kPages == 531);
constexpr std::size_t kMinImageBytes   = 1ULL << 30;
constexpr double kMinSaveRestoreMBps  = 40.0;
constexpr int kReps                   = 3;

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

std::vector<ninfer::PagedKVPlaneSpec> int8_text_planes() {
    std::vector<ninfer::PagedKVPlaneSpec> planes;
    planes.reserve(64);
    for (int layer = 0; layer < 16; ++layer) {
        planes.push_back({ninfer::DType::I8, 256, 4, 256});
        planes.push_back({ninfer::DType::I8, 256, 4, 256});
        planes.push_back({ninfer::DType::FP16, 4, 4, 256});
        planes.push_back({ninfer::DType::FP16, 4, 4, 256});
    }
    return planes;
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

void fill_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        unsigned char seed) {
    const auto pages = allocation.page_ids();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
            std::uint64_t state = 0x9e3779b97f4a7c15ULL ^
                                  (static_cast<std::uint64_t>(seed) << 48) ^
                                  (static_cast<std::uint64_t>(plane) << 24) ^ i;
            for (std::size_t byte = 0; byte < static_cast<std::size_t>(tensor.nb[3]); ++byte) {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                host[begin + byte] = static_cast<unsigned char>(
                    (state * 0x2545f4914f6cdd1dULL) >> 56);
            }
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
}

int expect_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                         unsigned char seed) {
    const auto pages = allocation.page_ids();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
            std::uint64_t state = 0x9e3779b97f4a7c15ULL ^
                                  (static_cast<std::uint64_t>(seed) << 48) ^
                                  (static_cast<std::uint64_t>(plane) << 24) ^ i;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            const auto expected = static_cast<unsigned char>(
                (state * 0x2545f4914f6cdd1dULL) >> 56);
            if (host[begin] != expected) {
                std::cerr << "disk_perf restore page " << i << " plane " << plane << " mismatch\n";
                return 1;
            }
        }
    }
    return 0;
}

std::optional<std::uint64_t> capture_or_evict(q36::detail::KVRamCache& cache,
                                              const q36::detail::RamCaptureSource& source) {
    for (;;) {
        if (auto id = cache.capture(source)) { return id; }
        const auto victim = cache.peek_oldest_unpinned();
        if (!victim) { return std::nullopt; }
        cache.evict_one_unpinned(*victim);
    }
}

double timed_mbps(std::size_t bytes, double seconds) {
    if (seconds <= 0.0) { return 0.0; }
    return (static_cast<double>(bytes) / 1.0e6) / seconds;
}

std::optional<std::uint64_t> disk_write_bytes() {
    const char* value = std::getenv("NINFER_KV_DISK_PERF_DEVICES");
    if (value == nullptr || *value == '\0') { return std::nullopt; }
    std::vector<std::string> devices;
    std::istringstream names(value);
    for (std::string name; std::getline(names, name, ',');) {
        if (!name.empty()) { devices.push_back(std::move(name)); }
    }
    std::ifstream stats("/proc/diskstats");
    std::uint64_t sectors = 0;
    for (std::string line; std::getline(stats, line);) {
        std::istringstream fields(line);
        unsigned major = 0;
        unsigned minor = 0;
        std::string name;
        std::uint64_t reads = 0, read_merges = 0, read_sectors = 0, read_ms = 0;
        std::uint64_t writes = 0, write_merges = 0, write_sectors = 0;
        if (!(fields >> major >> minor >> name >> reads >> read_merges >> read_sectors >> read_ms >>
              writes >> write_merges >> write_sectors)) {
            continue;
        }
        if (std::find(devices.begin(), devices.end(), name) != devices.end()) {
            sectors += write_sectors;
        }
    }
    return sectors * 512;
}

std::uint32_t physical_replication() {
    const char* value = std::getenv("NINFER_KV_DISK_PERF_REPLICAS");
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(
            "NINFER_KV_DISK_PERF_REPLICAS is required with NINFER_KV_DISK_PERF_DEVICES");
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 32) {
        throw std::invalid_argument("NINFER_KV_DISK_PERF_REPLICAS must be an integer from 1 to 32");
    }
    return static_cast<std::uint32_t>(parsed);
}

bool sync_benchmark_pool() {
    const char* value = std::getenv("NINFER_KV_DISK_PERF_ZPOOL");
    if (value == nullptr || *value == '\0') { return true; }
    const std::string pool(value);
    if (!std::all_of(pool.begin(), pool.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-';
        })) {
        return false;
    }
    return std::system(("zpool sync " + pool).c_str()) == 0;
}

bool sync_benchmark_storage(const fs::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) { return false; }
    const bool ok = ::syncfs(fd) == 0;
    ::close(fd);
    return ok && sync_benchmark_pool();
}

bool path_is_tmpfs(const fs::path& path) {
    struct statfs st {};
    if (::statfs(path.c_str(), &st) != 0) { return false; }
    return st.f_type == TMPFS_MAGIC;
}

std::string fs_type_name(const fs::path& path) {
    struct statfs st {};
    if (::statfs(path.c_str(), &st) != 0) { return "unknown"; }
    if (st.f_type == TMPFS_MAGIC) { return "tmpfs"; }
#ifdef XFS_SUPER_MAGIC
    if (st.f_type == XFS_SUPER_MAGIC) { return "xfs"; }
#endif
#ifdef EXT4_SUPER_MAGIC
    if (st.f_type == EXT4_SUPER_MAGIC) { return "ext4"; }
#endif
    if (static_cast<unsigned long>(st.f_type) == 0x2fc12fc1UL) { return "zfs"; }
    std::ostringstream out;
    out << "0x" << std::hex << st.f_type;
    return out.str();
}

fs::path perf_store_dir() {
    if (const char* env = std::getenv("NINFER_KV_DISK_PERF_DIR")) {
        return fs::path(env) / ("ninfer-kv-disk-perf-" + std::to_string(::getpid()));
    }
#ifdef NINFER_SOURCE_DIR
    return fs::path(NINFER_SOURCE_DIR) / "out" /
           ("kv-disk-perf-" + std::to_string(::getpid()));
#else
    return fs::current_path() / ("ninfer-kv-disk-perf-" + std::to_string(::getpid()));
#endif
}

bool write_all(int fd, const char* data, std::size_t bytes) {
    std::size_t off = 0;
    while (off < bytes) {
        const ssize_t wrote = ::write(fd, data + off, bytes - off);
        if (wrote <= 0) { return false; }
        off += static_cast<std::size_t>(wrote);
    }
    return true;
}

void drop_file_cache(const fs::path& dir) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file(ec) || ec) { continue; }
        const int fd = ::open(it->path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) { continue; }
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
}

double posix_write_fsync_mbps(const fs::path& file, std::size_t bytes) {
    std::vector<char> buf(1 << 20, 0x5a);
    const int fd = ::open(file.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { return 0.0; }
    const auto start = std::chrono::steady_clock::now();
    std::size_t left = bytes;
    while (left > 0) {
        const std::size_t n = std::min(left, buf.size());
        if (!write_all(fd, buf.data(), n)) {
            ::close(fd);
            return 0.0;
        }
        left -= n;
    }
    if (::fsync(fd) != 0) {
        ::close(fd);
        return 0.0;
    }
    ::close(fd);
    return timed_mbps(
        bytes, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

double posix_direct_mbps(const fs::path& file, std::size_t bytes, bool write) {
    constexpr std::size_t kAlign = 4096;
    const std::size_t aligned = (bytes + kAlign - 1) / kAlign * kAlign;
    void* buf                = nullptr;
    if (::posix_memalign(&buf, kAlign, 1 << 20) != 0) { return 0.0; }
    if (write) { std::memset(buf, 0x5a, 1 << 20); }
    const int flags = O_CLOEXEC | O_DIRECT | (write ? (O_CREAT | O_WRONLY | O_TRUNC) : O_RDONLY);
    const int fd    = ::open(file.c_str(), flags, 0644);
    if (fd < 0) {
        std::free(buf);
        return 0.0;
    }
    const auto start = std::chrono::steady_clock::now();
    std::size_t off = 0;
    while (off < aligned) {
        const std::size_t n = std::min(aligned - off, static_cast<std::size_t>(1 << 20));
        if (write) {
            if (!write_all(fd, static_cast<const char*>(buf), n)) {
                ::close(fd);
                std::free(buf);
                return 0.0;
            }
            off += n;
        } else {
            const ssize_t got = ::read(fd, buf, n);
            if (got <= 0) { break; }
            off += static_cast<std::size_t>(got);
        }
    }
    if (write && ::fsync(fd) != 0) {
        ::close(fd);
        std::free(buf);
        return 0.0;
    }
    ::close(fd);
    std::free(buf);
    return timed_mbps(
        bytes, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

double posix_read_mbps(const fs::path& file, std::size_t bytes) {
    const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return 0.0; }
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    std::vector<char> buf(1 << 20);
    const auto start = std::chrono::steady_clock::now();
    std::size_t got  = 0;
    while (got < bytes) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n <= 0) { break; }
        got += static_cast<std::size_t>(n);
    }
    ::close(fd);
    return timed_mbps(
        got, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}


q36::detail::DiskOpenConfig make_cfg(const fs::path& location, q36::detail::KVRamCache& ram,
                                       ninfer::PagedKVPool& pool, std::uint32_t window,
                                       std::uint32_t page_batch,
                                       std::uint32_t io_threads = 1) {
    q36::detail::DiskOpenConfig cfg;
    cfg.location             = location;
    cfg.capacity_bytes       = 4ULL << 30;
    cfg.compress             = ninfer::KvDiskCompress::Off;
    cfg.max_context          = kPages * 64;
    cfg.ram                  = &ram;
    cfg.fingerprint          = q36::detail::make_disk_fingerprint(
        "qwen3.6-27b", "groupwise-int", ninfer::KvCacheStorage::Int8Group64,
        ninfer::SpeculativeBackend::None, pool, nullptr, nullptr, nullptr);
    cfg.text_pool            = &pool;
    cfg.logical_page_bytes  = ninfer::paged_kv_logical_page_bytes(pool);
    cfg.gdn_staging_bytes    = 256;
    cfg.restore_window_slots = window;
    cfg.restore_io_threads  = io_threads;
    cfg.pack_page_batch      = page_batch;
    return cfg;
}

std::uint32_t benchmark_page_batch() {
    const char* value = std::getenv("NINFER_KV_DISK_PERF_BATCH");
    if (value == nullptr || *value == '\0') { return 4; }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 8) {
        throw std::invalid_argument("NINFER_KV_DISK_PERF_BATCH must be an integer from 1 to 8");
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint32_t benchmark_state_readers() {
    const char* value = std::getenv("NINFER_KV_DISK_PERF_STATE_READERS");
    if (value == nullptr || *value == '\0') { return 16; }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 16) {
        throw std::invalid_argument(
            "NINFER_KV_DISK_PERF_STATE_READERS must be an integer from 1 to 16");
    }
    return static_cast<std::uint32_t>(parsed);
}

int restore_once(q36::detail::KVDiskCache& disk, ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool,
                 const q36::detail::DiskMatch& match, const fs::path& disk_dir, bool verify,
                 double* restore_s, double* harvest_s) {
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    q36::detail::DiskRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.stream         = ctx.copy_stream;
    target.reuse          = match.reuse;
    target.reuse_base     = match.reuse_base;
    drop_file_cache(disk_dir);
    const auto restore_start = std::chrono::steady_clock::now();
    disk.restore_device(match.entry_id, target);
    try {
        disk.wait_copies();
    } catch (...) {
        dest.release();
        return fail("disk_perf restore failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.copy_stream));
    if (disk.restore_failed()) {
        dest.release();
        return fail("disk_perf restore failed");
    }
    *restore_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - restore_start).count();
    *harvest_s = disk.harvest_copy_seconds().load;
    if (verify && expect_logical_pages(pool, dest, 21) != 0) {
        dest.release();
        return fail("disk_perf restore corrupted logical pages");
    }
    dest.release();
    return 0;
}

void run_crc_benchmark_if_requested() {
    const char* crc_perf = std::getenv("NINFER_KV_DISK_PERF_CRC");
    if (crc_perf == nullptr || std::string_view(crc_perf) != "1") { return; }
    constexpr std::size_t kMaxCrcBytes = 256ULL << 20;
    constexpr std::size_t kSweepBytes = 1ULL << 30;
    std::vector<std::uint8_t> input(kMaxCrcBytes);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (std::uint8_t& byte : input) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        byte = static_cast<std::uint8_t>((state * 0x2545f4914f6cdd1dULL) >> 56);
    }
    for (std::size_t bytes : {std::size_t{4096}, std::size_t{64ULL << 10},
                              std::size_t{128ULL << 10}, std::size_t{256ULL << 10},
                              std::size_t{512ULL << 10}, std::size_t{1ULL << 20},
                              std::size_t{4ULL << 20}, std::size_t{256ULL << 20}}) {
        const std::size_t reps = std::max<std::size_t>(4, kSweepBytes / bytes);
        std::uint32_t checksum = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < reps; ++rep) {
            input[0] ^= static_cast<std::uint8_t>(rep + 1);
            checksum ^= q36::detail::KVDiskCache::test_crc32c(
                std::span<const std::uint8_t>(input.data(), bytes));
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::cerr << "kv_disk_perf crc32c bytes=" << bytes << " reps=" << reps << " "
                  << timed_mbps(bytes * reps, elapsed) << " MB/s checksum=" << checksum << "\n";
    }
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

    const std::uint32_t page_batch = benchmark_page_batch();
    run_crc_benchmark_if_requested();
    if (const char* crc_only = std::getenv("NINFER_KV_DISK_PERF_CRC_ONLY");
        crc_only != nullptr && std::string_view(crc_only) == "1") {
        return 0;
    }
    ninfer::DeviceContext ctx(0);
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = kPages * 2,
                                                       .logical_page_capacity = kPages,
                                                       .table_rows            = 2,
                                                       .plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                                                       .planes                = int8_text_planes()});
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout);
    auto source = pool.reserve(kPages);
    source.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, source, 21);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, kPages);
    if (image_bytes < kMinImageBytes) {
        std::cerr << "disk_perf image is " << image_bytes << " bytes, expected >= "
                  << kMinImageBytes << '\n';
        source.release();
        return 1;
    }

    auto root = perf_store_dir();
    fs::remove_all(root);
    fs::create_directories(root);
    if (path_is_tmpfs(root)) {
        fs::remove_all(root);
        return fail("disk_perf store is on tmpfs; set NINFER_KV_DISK_PERF_DIR to an SSD path");
    }
    if (disk_write_bytes().has_value() && fs_type_name(root) == "zfs") {
        const char* pool = std::getenv("NINFER_KV_DISK_PERF_ZPOOL");
        if (pool == nullptr || *pool == '\0') {
            fs::remove_all(root);
            return fail("disk_perf physical ZFS accounting requires NINFER_KV_DISK_PERF_ZPOOL");
        }
    }

    std::vector<ninfer::TokenId> tokens(kBenchmarkTokens, 7);
    auto prompt   = text_prompt(tokens);
    auto retained = prompt;
    retained.token_ids.push_back(0);
    q36::detail::ResidentPrefixIdentity identity;
    retained.token_types.assign(retained.token_ids.size(), 0);
    retained.positions.resize(3 * retained.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < retained.token_ids.size(); ++i) {
            retained.positions[static_cast<std::size_t>(axis) * retained.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    identity.assign(retained);
    q36::detail::RamCaptureSource capture;
    capture.execution_frontier = static_cast<std::uint32_t>(tokens.size());
    capture.ledger_frontier    = static_cast<std::uint32_t>(retained.token_ids.size());
    capture.text_kv_valid      = capture.execution_frontier;
    capture.ledger             = retained.token_ids;
    capture.identity           = &identity;
    capture.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                              capture.execution_frontier);
    capture.text               = &source;
    capture.text_pool          = &pool;
    capture.stream             = ctx.copy_stream;

    const auto posix_file = root / "posix-seq.bin";
    const double posix_write = posix_write_fsync_mbps(posix_file, image_bytes);
    const double posix_read = posix_read_mbps(posix_file, image_bytes);
    fs::remove(posix_file);
    const auto direct_file = root / "posix-direct.bin";
    const double direct_write = posix_direct_mbps(direct_file, image_bytes, true);
    const double direct_read = posix_direct_mbps(direct_file, image_bytes, false);
    fs::remove(direct_file);
    std::cerr << "kv_disk_perf dir=" << root << " fstype=" << fs_type_name(root)
              << " tokens=" << kBenchmarkTokens << " pages=" << kPages
              << " image_bytes=" << image_bytes
              << " posix_write_fsync=" << posix_write << " MB/s posix_direct_write=" << direct_write
              << " MB/s posix_read=" << posix_read << " MB/s posix_direct_read=" << direct_read
              << " MB/s\n";

    auto spill_rep = [&](std::uint32_t window, int rep, double* spill_s,
                           double* harvest_s, std::size_t* used,
                           std::optional<std::uint64_t>* physical_written,
                           double* device_sync_s) -> int {
        const auto dir = root / ("spill-" + std::to_string(window) + "-" +
                                  std::to_string(rep));
        fs::remove_all(dir);
        fs::create_directories(dir);
        q36::detail::KVRamCache ram(2ULL << 30);
        auto cfg = make_cfg(dir, ram, pool, window, page_batch);
        q36::detail::KVDiskCache disk(std::move(cfg));
        const auto ram_id = capture_or_evict(ram, capture);
        if (!ram_id) { return fail("disk_perf RAM capture failed"); }
        ctx.synchronize_all();
        ram.wait_pending_copies();
        disk.note_ram_resident(*ram_id, 0);
        const bool measure_physical = disk_write_bytes().has_value();
        if (measure_physical && !sync_benchmark_storage(dir)) {
            return fail("disk_perf pre-sample storage sync failed");
        }
        const auto physical_before = disk_write_bytes();
        const auto spill_start = std::chrono::steady_clock::now();
        if (!disk.emergency_spill_ram(*ram_id)) { return fail("disk_perf emergency spill failed"); }
        const auto committed = std::chrono::steady_clock::now();
        *spill_s = std::chrono::duration<double>(committed - spill_start).count();
        *harvest_s = disk.harvest_copy_seconds().save;
        *used      = disk.snapshot().used_bytes;
        if (physical_before && !sync_benchmark_storage(dir)) {
            return fail("disk_perf post-sample storage sync failed");
        }
        *device_sync_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - spill_start).count();
        const auto physical_after = disk_write_bytes();
        if (physical_before && physical_after && *physical_after >= *physical_before) {
            *physical_written = *physical_after - *physical_before;
        }
        if (*used < kMinImageBytes) { return fail("disk_perf spill unique bytes too small"); }
        return 0;
    };

    double sum_s = 0.0;
    std::uint64_t physical_sum = 0;
    std::uint64_t payload_sum = 0;
    double physical_seconds = 0.0;
    int physical_samples = 0;
    const bool physical_enabled = disk_write_bytes().has_value();
    const std::uint32_t replicas = physical_enabled ? physical_replication() : 1;
    for (int rep = 0; rep < kReps; ++rep) {
        double spill_s = 0;
        double harvest_s = 0;
        std::size_t used = 0;
        std::optional<std::uint64_t> physical_written;
        double device_sync_s = 0;
        if (const int rc = spill_rep(2, rep, &spill_s, &harvest_s, &used, &physical_written,
                                     &device_sync_s);
            rc != 0) {
            source.release();
            fs::remove_all(root);
            return rc;
        }
        const double mbps =
            spill_s > 0.0 ? (static_cast<double>(used) / 1.0e6) / spill_s : 0.0;
        bool clean_physical = false;
        std::uint64_t normalized_write = 0;
        if (physical_written) {
            normalized_write = *physical_written / replicas;
            const double ratio = static_cast<double>(normalized_write) /
                                 static_cast<double>(used);
            // With compression disabled, the cache's durable extent accounting
            // should closely track mirror-normalized leaf writes.  A wider
            // interval admits unrelated writes from other datasets on the pool
            // and turns a global disk counter into a misleading cache result.
            clean_physical = ratio >= 0.98 && ratio <= 1.03;
        }
        std::cerr << "kv_disk_perf save window=2 rep=" << rep << " " << mbps << " MB/s ("
                  << spill_s * 1e3 << " ms, harvest=" << harvest_s * 1e3 << " ms) used=" << used
                  << (physical_written ? " physical_write=" + std::to_string(*physical_written)
                                       : std::string{})
                  << (physical_written ? " normalized_write=" + std::to_string(normalized_write)
                                       : std::string{})
                  << (physical_written ? " device_sync_ms=" +
                                             std::to_string(device_sync_s * 1e3)
                                       : std::string{})
                  << (physical_written ? clean_physical ? " physical_sample=accepted"
                                                        : " physical_sample=contaminated"
                                       : std::string{})
                  << "\n";
        if (mbps < kMinSaveRestoreMBps) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf spill is slower than 40 MB/s");
        }
        sum_s += spill_s;
        if (clean_physical) {
            physical_sum += *physical_written;
            payload_sum += used;
            physical_seconds += device_sync_s;
            ++physical_samples;
        }
    }
    std::cerr << "kv_disk_perf save mean=" << (image_bytes / 1.0e6) / (sum_s / kReps)
              << " MB/s";
    if (physical_samples != 0) {
        std::cerr << " aggregate_device_write=" << physical_sum
                  << " bytes replicas=" << replicas
                  << " accepted_samples=" << physical_samples
                  << " normalized_device_rate="
                  << (static_cast<double>(physical_sum) / replicas / 1.0e6) / physical_seconds
                  << " MB/s payload_device_rate="
                  << (static_cast<double>(payload_sum) / 1.0e6) / physical_seconds << " MB/s";
    } else if (physical_enabled) {
        source.release();
        fs::remove_all(root);
        return fail("disk_perf has no uncontaminated physical write samples");
    }
    std::cerr << "\n";

    const auto restore_dir = root / "restore-store";
    fs::remove_all(restore_dir);
    fs::create_directories(restore_dir);
    std::uint64_t spilled_id = 0;
    std::uint32_t frontier = 0;
    {
        q36::detail::KVRamCache ram(2ULL << 30);
        auto cfg = make_cfg(restore_dir, ram, pool, 2, page_batch);
        q36::detail::KVDiskCache disk(std::move(cfg));
        const auto ram_id = capture_or_evict(ram, capture);
        if (!ram_id) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf restore-store RAM capture failed");
        }
        ctx.synchronize_all();
        ram.wait_pending_copies();
        disk.note_ram_resident(*ram_id, 0);
        if (!disk.emergency_spill_ram(*ram_id)) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf restore-store spill failed");
        }
        const auto match = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match || match->reuse_base != tokens.size()) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf spilled chat is not hittable");
        }
        spilled_id = match->entry_id;
        frontier    = match->execution_frontier;
    }

    const std::uint32_t windows[] = {2, 4, 8, 16, 32};
    for (const std::uint32_t window : windows) {
        q36::detail::KVRamCache ram(32ULL << 20);
        auto cfg = make_cfg(restore_dir, ram, pool, window, page_batch, 8);
        q36::detail::KVDiskCache disk(std::move(cfg));
        const auto planned = disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!planned || planned->entry_id != spilled_id || planned->reuse_base != frontier) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf restore reopen lost the spilled chat");
        }
        const q36::detail::DiskMatch match = *planned;
        if (!disk.claim(match.entry_id, match.hash_f, match.execution_frontier)) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf restore claim failed");
        }
        double sum_s = 0.0;
        for (int rep = 0; rep < kReps; ++rep) {
            double restore_s = 0;
            double harvest_s = 0;
            if (const int rc = restore_once(disk, ctx, pool, match, restore_dir, rep == 0, &restore_s,
                                           &harvest_s);
                rc != 0) {
                disk.release(match.entry_id);
                source.release();
                fs::remove_all(root);
                return rc;
            }
            const double mbps =
                restore_s > 0.0 ? (static_cast<double>(image_bytes) / 1.0e6) / restore_s : 0.0;
            std::cerr << "kv_disk_perf restore window=" << window << " readers=8 rep=" << rep << " " << mbps
                      << " MB/s (" << restore_s * 1e3 << " ms, harvest=" << harvest_s * 1e3
                      << " ms)\n";
            if (mbps < kMinSaveRestoreMBps) {
                disk.release(match.entry_id);
                source.release();
                fs::remove_all(root);
                return fail("disk_perf restore is slower than 40 MB/s");
            }
            sum_s += restore_s;
        }
        std::cerr << "kv_disk_perf restore window=" << window
                  << " mean=" << (image_bytes / 1.0e6) / (sum_s / kReps) << " MB/s\n";
        disk.release(match.entry_id);
    }

    if (const char* startup_perf = std::getenv("NINFER_KV_DISK_PERF_STARTUP");
        startup_perf != nullptr && std::string_view(startup_perf) == "1") {
        for (int rep = 0; rep < kReps; ++rep) {
            drop_file_cache(restore_dir);
            q36::detail::KVRamCache startup_ram(32ULL << 20);
            auto startup_cfg = make_cfg(restore_dir, startup_ram, pool, 32, page_batch, 16);
            const auto begin = std::chrono::steady_clock::now();
            q36::detail::KVDiskCache startup_disk(std::move(startup_cfg));
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (!startup_disk.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
                source.release();
                fs::remove_all(root);
                return fail("disk_perf startup lost the 531-page entry");
            }
            std::cerr << "kv_disk_perf startup objects=" << kPages << " rep=" << rep << " "
                      << elapsed * 1e3 << " ms\n";
        }
    }

    if (const char* state_perf = std::getenv("NINFER_KV_DISK_PERF_STATE");
        state_perf != nullptr && std::string_view(state_perf) == "1") {
        ninfer::LayoutBuilder gdn_builder;
        const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
            gdn_builder, {.layers         = 48,
                          .conv_channels  = 10'240,
                          .conv_width     = 3,
                          .value_heads    = 48,
                          .value_head_dim = 128,
                          .key_head_dim   = 128,
                          .slot_count     = 2,
                          .conv_dtype     = ninfer::DType::BF16});
        ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
        ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()},
                                             gdn_layout);
        gdn.zero_slot(0, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const std::size_t state_bytes =
            gdn.conv_host_image_bytes() + gdn.recurrent_host_image_bytes();
        if (state_bytes < (128ULL << 20) || state_bytes % 4096 != 0) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf state geometry is not representative and aligned");
        }

        const auto state_dir = root / "state-store";
        fs::remove_all(state_dir);
        fs::create_directories(state_dir);
        q36::detail::KVRamCache state_ram(3ULL << 30);
        const std::uint32_t state_readers = benchmark_state_readers();
        auto state_cfg = make_cfg(state_dir, state_ram, pool, 32, page_batch, state_readers);
        state_cfg.fingerprint = q36::detail::make_disk_fingerprint(
            "qwen3.6-27b", "groupwise-int", ninfer::KvCacheStorage::Int8Group64,
            ninfer::SpeculativeBackend::None, pool, nullptr, &gdn, nullptr);
        state_cfg.gdn_staging_bytes = state_bytes;
        q36::detail::KVDiskCache state_disk(std::move(state_cfg));
        auto state_prompt = text_prompt({7, 7, 7, 7});
        auto state_retained = state_prompt;
        state_retained.token_ids.push_back(0);
        state_retained.token_types.assign(state_retained.token_ids.size(), 0);
        state_retained.positions.resize(3 * state_retained.token_ids.size());
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < state_retained.token_ids.size(); ++i) {
                state_retained.positions[static_cast<std::size_t>(axis) *
                                             state_retained.token_ids.size() +
                                         i] = static_cast<std::int32_t>(i);
            }
        }
        q36::detail::ResidentPrefixIdentity state_identity;
        state_identity.assign(state_retained);
        auto state_capture = capture;
        state_capture.execution_frontier = 4;
        state_capture.ledger_frontier = 5;
        state_capture.text_kv_valid = 4;
        state_capture.ledger = state_retained.token_ids;
        state_capture.identity = &state_identity;
        state_capture.hash_f = q36::detail::prefix_hash_at(state_retained.token_ids,
                                                           state_identity, 4);
        state_capture.gdn = &gdn;
        state_capture.gdn_current_slot = 0;
        const auto ram_id = capture_or_evict(state_ram, state_capture);
        if (!ram_id) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf state RAM capture failed");
        }
        ctx.synchronize_all();
        state_ram.wait_pending_copies();
        state_disk.note_ram_resident(*ram_id, 0);
        if (!state_disk.emergency_spill_ram(*ram_id)) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf state spill failed");
        }
        const auto match = state_disk.plan_match(state_prompt,
                                                 q36::detail::prefix_hash_chain(state_prompt));
        if (!match || !state_disk.claim(match->entry_id, match->hash_f,
                                        match->execution_frontier)) {
            source.release();
            fs::remove_all(root);
            return fail("disk_perf state claim failed");
        }
        double state_sum = 0.0;
        for (int rep = 0; rep < kReps; ++rep) {
            auto dest = pool.reserve(1);
            dest.materialize_pages(1, ctx.stream);
            q36::detail::DiskRestoreTarget target;
            target.text = &dest;
            target.text_pool = &pool;
            target.text_dst_pages = 1;
            target.gdn = &gdn;
            target.gdn_current_slot = 1;
            target.stream = ctx.copy_stream;
            target.reuse = match->reuse;
            target.reuse_base = match->reuse_base;
            drop_file_cache(state_dir);
            const auto begin = std::chrono::steady_clock::now();
            state_disk.restore_device(match->entry_id, target);
            state_disk.wait_copies();
            CUDA_CHECK(cudaStreamSynchronize(ctx.copy_stream));
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            const double host_load = state_disk.harvest_copy_seconds().load;
            std::cerr << "kv_disk_perf state readers=" << state_readers << " rep=" << rep << " "
                      << timed_mbps(state_bytes, host_load) << " MB/s (wall="
                      << elapsed * 1e3 << " ms, load=" << host_load * 1e3
                      << " ms, bytes=" << state_bytes << ")\n";
            state_sum += host_load;
            dest.release();
        }
        std::cerr << "kv_disk_perf state readers=" << state_readers << " mean="
                  << timed_mbps(state_bytes * kReps, state_sum) << " MB/s\n";
        state_disk.release(match->entry_id);

        if (const char* compact_perf = std::getenv("NINFER_KV_DISK_PERF_COMPACTION");
            compact_perf != nullptr && std::string_view(compact_perf) == "1") {
            for (ninfer::TokenId token : {8, 9}) {
                auto extra_prompt = text_prompt({token, token, token, token});
                auto extra_retained = extra_prompt;
                extra_retained.token_ids.push_back(0);
                extra_retained.token_types.assign(extra_retained.token_ids.size(), 0);
                extra_retained.positions.resize(3 * extra_retained.token_ids.size());
                for (int axis = 0; axis < 3; ++axis) {
                    for (std::size_t i = 0; i < extra_retained.token_ids.size(); ++i) {
                        extra_retained.positions[static_cast<std::size_t>(axis) *
                                                     extra_retained.token_ids.size() +
                                                 i] = static_cast<std::int32_t>(i);
                    }
                }
                q36::detail::ResidentPrefixIdentity extra_identity;
                extra_identity.assign(extra_retained);
                auto extra_capture = state_capture;
                extra_capture.ledger = extra_retained.token_ids;
                extra_capture.identity = &extra_identity;
                extra_capture.hash_f = q36::detail::prefix_hash_at(
                    extra_retained.token_ids, extra_identity, 4);
                const auto extra_ram_id = capture_or_evict(state_ram, extra_capture);
                if (!extra_ram_id) {
                    source.release();
                    fs::remove_all(root);
                    return fail("disk_perf compaction RAM capture failed");
                }
                ctx.synchronize_all();
                state_ram.wait_pending_copies();
                state_disk.note_ram_resident(*extra_ram_id, 0);
                if (!state_disk.emergency_spill_ram(*extra_ram_id)) {
                    source.release();
                    fs::remove_all(root);
                    return fail("disk_perf compaction spill failed");
                }
            }
            if (!state_disk.test_fifo_evict_one()) {
                source.release();
                fs::remove_all(root);
                return fail("disk_perf compaction eviction failed");
            }
            const std::size_t retained_bytes = state_disk.snapshot().used_bytes;
            const auto begin = std::chrono::steady_clock::now();
            state_disk.wait_idle_and_fsync();
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            std::cerr << "kv_disk_perf compaction retained=" << retained_bytes << " "
                      << timed_mbps(retained_bytes, elapsed) << " MB/s (" << elapsed * 1e3
                      << " ms)\n";
        }
    }

    source.release();
    fs::remove_all(root);
    return 0;
}

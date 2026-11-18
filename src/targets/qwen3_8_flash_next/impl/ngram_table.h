#pragma once

// Qwen3.8-Flash-Next n-gram (PLE) table on SSD: exact row addressing and direct-I/O row reads.

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

inline constexpr std::int32_t kNgramEosToken    = 248044;
inline constexpr std::int32_t kNgramHeads       = 16; // eight bigram heads, then eight trigram
inline constexpr std::int32_t kNgramRowWidth    = 160;
inline constexpr std::int32_t kNgramRecordBytes = 2 + kNgramRowWidth; // FP16 scale, int8 codes
inline constexpr std::int32_t kNgramPageBytes   = 4096;
inline constexpr std::int32_t kNgramRowsPerPage = kNgramPageBytes / kNgramRecordBytes;
inline constexpr std::int32_t kNgramTokenBytes  = kNgramHeads * kNgramRecordBytes;
inline constexpr std::size_t kNgramCacheWays    = 8;

// Hash constants copied exactly from the checkpoint's int64 tensors.
struct NgramHashConstants {
    std::array<std::uint64_t, 3> multipliers{};
    std::array<std::int64_t, kNgramHeads> head_offsets{};
    std::array<std::int64_t, kNgramHeads> head_sizes{};
};

// Raw-token history of one request: the two previous tokens, oldest first. A new request starts
// from two missing predecessors, which read as EOS.
struct NgramHistory {
    std::array<std::int32_t, 2> tokens{kNgramEosToken, kNgramEosToken};
};

/**
 * Table rows of `token` after `history`, in head order, and the history advanced past it.
 *
 * With p1 the previous and p2 the one before: t1 = EOS if p1 is EOS, else p1; t2 = EOS if p1 or p2
 * is EOS, else p2 (an EOS predecessor resets everything at or before it). bigram = t0*m0 ^ t1*m1
 * and trigram = bigram ^ t2*m2 in wrapping unsigned 64-bit arithmetic; head h takes the
 * nonnegative remainder of its hash (bigram for h < 8) by head_sizes[h] plus head_offsets[h].
 * Token ids are nonnegative. Exact integer arithmetic; no failure modes.
 */
[[nodiscard]] std::array<std::int64_t, kNgramHeads>
ngram_row_ids(std::int32_t token, NgramHistory& history, const NgramHashConstants& constants);

/**
 * The page-aligned int8 n-gram table produced by the study converter: a 4 KiB JSON header (format
 * "int8_row", rows, dim 160, row_bytes 162, rows_per_page 25, page_bytes 4096, body_offset 4096,
 * and the hash constants), then pages of 25 records that never cross a page.
 *
 * gather() reads each token's 16 records through a bounded 8-way set-associative LRU row cache
 * (set = row mod sets). A row present when the gather starts is copied from the cache; every
 * distinct page holding an absent row is read once, as a whole 4 KiB page with O_DIRECT, by a
 * fixed pool of I/O threads. After the reads, recency is replayed in logical access order (token,
 * then head), so the cache then holds exactly what a sequential LRU over those accesses would.
 * The object is used from one thread at a time; its I/O threads are private. Construction
 * validates the header against the file size and throws std::runtime_error on any mismatch,
 * malformed header or I/O failure, and std::invalid_argument on invalid options.
 */
class NgramTable {
public:
    struct Options {
        std::filesystem::path path;
        std::size_t cache_rows  = std::size_t{1} << 20; // rounded down to whole 8-row sets
        std::int32_t io_threads = 48;
    };

    explicit NgramTable(const Options& options);
    ~NgramTable();

    NgramTable(const NgramTable&)            = delete;
    NgramTable& operator=(const NgramTable&) = delete;

    [[nodiscard]] const NgramHashConstants& constants() const noexcept { return constants_; }

    [[nodiscard]] std::int64_t rows() const noexcept { return rows_; }

    // Pages read from the file by gather() since construction (one per issued page read).
    [[nodiscard]] std::uint64_t pages_read() const noexcept { return pages_read_; }

    /**
     * Writes the 16 records of each token, in token then head order, to out ([T][16][162] bytes,
     * out.size() == T * 2592) and advances history past the tokens. Throws std::runtime_error on
     * an I/O failure (history and out are then unspecified; the cache is unchanged and the table
     * remains usable) and std::invalid_argument on a size mismatch.
     */
    void gather(std::span<const std::int32_t> tokens, NgramHistory& history,
                std::span<std::uint8_t> out);

    /**
     * The lane-batched form of one decode round: token t follows the history *histories[t] and
     * advances it, so tokens naming the same history must appear in sequence order. Records,
     * caching, page reads and failures are as for the single-history form over the same tokens.
     * histories.size() must equal tokens.size() with no null entry (std::invalid_argument).
     */
    void gather(std::span<const std::int32_t> tokens, std::span<NgramHistory* const> histories,
                std::span<std::uint8_t> out);

private:
    class FileDescriptor {
    public:
        explicit FileDescriptor(const std::filesystem::path& path); // O_RDONLY | O_DIRECT
        ~FileDescriptor();
        FileDescriptor(const FileDescriptor&)            = delete;
        FileDescriptor& operator=(const FileDescriptor&) = delete;

        [[nodiscard]] int get() const noexcept { return fd_; }

    private:
        int fd_;
    };

    struct AlignedFree {
        void operator()(std::uint8_t* pointer) const noexcept;
    };

    // One table row access of a gather: the row and its record's byte offset in the output.
    struct Access {
        std::int64_t row;
        std::size_t offset;
    };

    // One row to fetch: its page, its slot within that page, and its output byte offset.
    struct Miss {
        std::int64_t page;
        std::int32_t slot;
        std::size_t offset;
    };

    struct CacheWay {
        std::int64_t row       = -1;
        std::uint64_t last_use = 0;
    };

    // Fetches the records of the rows in accesses_ (built by a gather) into out.
    void fetch(std::span<std::uint8_t> out);
    // Appends token t's 16 row accesses to accesses_.
    void add_accesses(std::size_t t, const std::array<std::int64_t, kNgramHeads>& rows);
    [[nodiscard]] bool cache_find(std::int64_t row, std::uint8_t* destination) const;
    void cache_touch(std::int64_t row, const std::uint8_t* record);
    void io_loop(std::uint8_t* page);
    void read_pages(std::uint8_t* out);
    void stop_workers() noexcept;

    FileDescriptor file_;
    std::int64_t rows_ = 0;
    NgramHashConstants constants_;
    std::uint64_t pages_read_ = 0;

    std::size_t cache_sets_ = 0;
    std::vector<CacheWay> cache_ways_;
    std::vector<std::uint8_t> cache_records_;
    std::uint64_t use_clock_ = 0;

    // Per-gather scratch reused across calls. Misses are sorted by page; request i covers
    // misses_[page_starts_[i], page_starts_[i + 1]).
    std::vector<Access> accesses_;
    std::vector<Miss> misses_;
    std::vector<std::size_t> page_starts_;

    // One 4 KiB direct-I/O page buffer per I/O thread, allocated before the threads start.
    std::unique_ptr<std::uint8_t, AlignedFree> worker_pages_;

    // Guarded by mutex_: the published request count, the next unclaimed and the finished
    // request counts, the output pointer, the first failure, and the stop flag.
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    std::size_t pending_requests_  = 0;
    std::size_t next_request_      = 0;
    std::size_t finished_requests_ = 0;
    std::uint8_t* pending_out_     = nullptr;
    std::exception_ptr failure_;
    bool stopping_ = false;
    std::vector<std::thread> io_threads_;
};

} // namespace ninfer::targets::qwen3_8_flash_next

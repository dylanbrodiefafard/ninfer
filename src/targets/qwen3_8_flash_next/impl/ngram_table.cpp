#include "ngram_table.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

std::runtime_error io_error(const std::string& what) {
    return std::runtime_error("ngram table: " + what + ": " + std::strerror(errno));
}

// Reads exactly one aligned page at byte offset `offset` into `buffer` (page-aligned).
void read_page(int fd, std::uint8_t* buffer, std::int64_t offset) {
    std::size_t done = 0;
    while (done < kNgramPageBytes) {
        const ssize_t n = ::pread(fd, buffer + done, kNgramPageBytes - done,
                                  static_cast<off_t>(offset + static_cast<std::int64_t>(done)));
        if (n < 0 && errno == EINTR) { continue; }
        if (n < 0) { throw io_error("pread at " + std::to_string(offset)); }
        if (n == 0) {
            throw std::runtime_error("ngram table: unexpected end of file reading page at " +
                                     std::to_string(offset));
        }
        done += static_cast<std::size_t>(n);
    }
}

// A JSON integer representable as int64 (the checkpoint's integer type); anything else, including
// a floating-point value that would truncate, is rejected.
std::int64_t header_integer(const nlohmann::json& value, const std::string& name) {
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<std::uint64_t>() >
             static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) {
        throw std::runtime_error("ngram table: header " + name + " must be an int64 integer");
    }
    return value.get<std::int64_t>();
}

template <std::size_t N>
std::array<std::int64_t, N> header_integers(const nlohmann::json& value, const char* name) {
    if (!value.is_array() || value.size() != N) {
        throw std::runtime_error(std::string("ngram table: header ") + name + " must have " +
                                 std::to_string(N) + " entries");
    }
    std::array<std::int64_t, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = header_integer(value[i], std::string(name) + "[" + std::to_string(i) + "]");
    }
    return out;
}

struct Header {
    std::int64_t rows = 0;
    NgramHashConstants constants;
};

// Parses and validates the header page against a body of `body_pages` pages. Every bound is checked
// by subtraction or division against values derived from the file size, so no check can overflow.
Header parse_header(const std::uint8_t* page, std::int64_t body_pages) {
    try {
        const auto* text = reinterpret_cast<const char*>(page);
        const auto json  = nlohmann::json::parse(text, text + ::strnlen(text, kNgramPageBytes));
        if (json.at("format") != "int8_row" ||
            header_integer(json.at("dim"), "dim") != kNgramRowWidth ||
            header_integer(json.at("row_bytes"), "row_bytes") != kNgramRecordBytes ||
            header_integer(json.at("rows_per_page"), "rows_per_page") != kNgramRowsPerPage ||
            header_integer(json.at("page_bytes"), "page_bytes") != kNgramPageBytes ||
            header_integer(json.at("body_offset"), "body_offset") != kNgramPageBytes) {
            throw std::runtime_error("ngram table: unsupported header layout");
        }
        Header header;
        header.rows = header_integer(json.at("rows"), "rows");
        // The body holds exactly ceil(rows / 25) pages.
        if (header.rows < 1 || (header.rows - 1) / kNgramRowsPerPage + 1 != body_pages) {
            throw std::runtime_error("ngram table: file size does not match its header");
        }
        const auto multipliers =
            header_integers<3>(json.at("layer_multipliers"), "layer_multipliers");
        for (std::size_t i = 0; i < multipliers.size(); ++i) {
            // Two's-complement reinterpretation: the hash multiplies in wrapping 64-bit arithmetic.
            header.constants.multipliers[i] = static_cast<std::uint64_t>(multipliers[i]);
        }
        header.constants.head_sizes = header_integers<kNgramHeads>(
            json.at("ngram_heads_vocab_sizes"), "ngram_heads_vocab_sizes");
        header.constants.head_offsets =
            header_integers<kNgramHeads>(json.at("ngram_heads_offsets"), "ngram_heads_offsets");
        for (std::size_t head = 0; head < kNgramHeads; ++head) {
            const std::int64_t size   = header.constants.head_sizes[head];
            const std::int64_t offset = header.constants.head_offsets[head];
            if (size < 1 || size > header.rows || offset < 0 || offset > header.rows - size) {
                throw std::runtime_error("ngram table: head " + std::to_string(head) +
                                         " lies outside the table");
            }
        }
        return header;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("ngram table: malformed header: ") + error.what());
    }
}

} // namespace

std::array<std::int64_t, kNgramHeads> ngram_row_ids(std::int32_t token, NgramHistory& history,
                                                    const NgramHashConstants& constants) {
    const std::int32_t previous = history.tokens[1];
    const std::int32_t before   = history.tokens[0];
    const std::int32_t t1       = previous == kNgramEosToken ? kNgramEosToken : previous;
    const std::int32_t t2 =
        previous == kNgramEosToken || before == kNgramEosToken ? kNgramEosToken : before;
    const auto product = [&](std::int32_t value, std::size_t which) {
        return static_cast<std::uint64_t>(value) * constants.multipliers[which];
    };
    const std::uint64_t bigram  = product(token, 0) ^ product(t1, 1);
    const std::uint64_t trigram = bigram ^ product(t2, 2);
    std::array<std::int64_t, kNgramHeads> rows{};
    for (std::int32_t head = 0; head < kNgramHeads; ++head) {
        const auto hash  = static_cast<std::int64_t>(head < kNgramHeads / 2 ? bigram : trigram);
        const auto size  = constants.head_sizes[static_cast<std::size_t>(head)];
        std::int64_t rem = hash % size;
        if (rem < 0) { rem += size; }
        rows[static_cast<std::size_t>(head)] =
            rem + constants.head_offsets[static_cast<std::size_t>(head)];
    }
    history.tokens = {previous, token};
    return rows;
}

NgramTable::FileDescriptor::FileDescriptor(const std::filesystem::path& path)
    : fd_(::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)) {
    if (fd_ < 0) { throw io_error("open " + path.string()); }
}

NgramTable::FileDescriptor::~FileDescriptor() { ::close(fd_); }

void NgramTable::AlignedFree::operator()(std::uint8_t* pointer) const noexcept {
    std::free(pointer);
}

NgramTable::NgramTable(const Options& options) : file_(options.path) {
    if (options.io_threads < 1 || options.cache_rows < kNgramCacheWays) {
        throw std::invalid_argument("ngram table: io_threads >= 1 and cache_rows >= 8 required");
    }
    const auto pages = static_cast<std::size_t>(options.io_threads);
    void* buffers    = std::aligned_alloc(kNgramPageBytes, pages * kNgramPageBytes);
    if (buffers == nullptr) { throw std::bad_alloc(); }
    worker_pages_.reset(static_cast<std::uint8_t*>(buffers));

    struct stat status{};
    if (::fstat(file_.get(), &status) != 0) { throw io_error("fstat"); }
    if (status.st_size < 2 * kNgramPageBytes || status.st_size % kNgramPageBytes != 0) {
        throw std::runtime_error(
            "ngram table: file is not a whole number of pages past the header");
    }
    read_page(file_.get(), worker_pages_.get(), 0);
    Header header = parse_header(worker_pages_.get(), status.st_size / kNgramPageBytes - 1);
    rows_         = header.rows;
    constants_    = header.constants;

    cache_sets_ = options.cache_rows / kNgramCacheWays;
    cache_ways_.resize(cache_sets_ * kNgramCacheWays);
    cache_records_.resize(cache_sets_ * kNgramCacheWays * kNgramRecordBytes);

    io_threads_.reserve(pages);
    try {
        for (std::size_t i = 0; i < pages; ++i) {
            std::uint8_t* page = worker_pages_.get() + i * kNgramPageBytes;
            io_threads_.emplace_back([this, page] { io_loop(page); });
        }
    } catch (...) {
        stop_workers();
        throw;
    }
}

NgramTable::~NgramTable() { stop_workers(); }

void NgramTable::stop_workers() noexcept {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    work_ready_.notify_all();
    for (auto& thread : io_threads_) { thread.join(); }
}

bool NgramTable::cache_find(std::int64_t row, std::uint8_t* destination) const {
    const std::size_t first = static_cast<std::size_t>(row) % cache_sets_ * kNgramCacheWays;
    for (std::size_t way = first; way < first + kNgramCacheWays; ++way) {
        if (cache_ways_[way].row == row) {
            std::memcpy(destination, cache_records_.data() + way * kNgramRecordBytes,
                        kNgramRecordBytes);
            return true;
        }
    }
    return false;
}

void NgramTable::cache_touch(std::int64_t row, const std::uint8_t* record) {
    const std::size_t first = static_cast<std::size_t>(row) % cache_sets_ * kNgramCacheWays;
    std::size_t victim      = first;
    for (std::size_t way = first; way < first + kNgramCacheWays; ++way) {
        if (cache_ways_[way].row == row) {
            cache_ways_[way].last_use = ++use_clock_;
            return;
        }
        // Empty ways have last_use 0 and are filled first.
        if (cache_ways_[way].last_use < cache_ways_[victim].last_use) { victim = way; }
    }
    cache_ways_[victim] = {row, ++use_clock_};
    std::memcpy(cache_records_.data() + victim * kNgramRecordBytes, record, kNgramRecordBytes);
}

void NgramTable::io_loop(std::uint8_t* page) {
    std::unique_lock lock(mutex_);
    for (;;) {
        work_ready_.wait(lock, [&] { return stopping_ || next_request_ < pending_requests_; });
        if (stopping_) { return; }
        const std::size_t request = next_request_++;
        std::uint8_t* const out   = pending_out_;
        lock.unlock();
        // misses_ and page_starts_ are not modified while requests are pending.
        std::exception_ptr error;
        try {
            const std::size_t begin = page_starts_[request];
            read_page(file_.get(), page, kNgramPageBytes + misses_[begin].page * kNgramPageBytes);
            for (std::size_t i = begin; i < page_starts_[request + 1]; ++i) {
                std::memcpy(out + misses_[i].offset,
                            page + static_cast<std::size_t>(misses_[i].slot) * kNgramRecordBytes,
                            kNgramRecordBytes);
            }
        } catch (...) { error = std::current_exception(); }
        lock.lock();
        if (error && !failure_) { failure_ = error; }
        if (++finished_requests_ == pending_requests_) { work_done_.notify_all(); }
    }
}

void NgramTable::read_pages(std::uint8_t* out) {
    const std::size_t requests = page_starts_.size() - 1;
    std::unique_lock lock(mutex_);
    pending_requests_  = requests;
    pending_out_       = out;
    next_request_      = 0;
    finished_requests_ = 0;
    failure_           = nullptr;
    work_ready_.notify_all();
    work_done_.wait(lock, [&] { return finished_requests_ == requests; });
    pending_requests_ = 0;
    pages_read_ += requests;
    if (failure_) { std::rethrow_exception(failure_); }
}

void NgramTable::add_accesses(std::size_t t, const std::array<std::int64_t, kNgramHeads>& rows) {
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::size_t offset =
            (t * kNgramHeads + head) * static_cast<std::size_t>(kNgramRecordBytes);
        accesses_.push_back({rows[head], offset});
    }
}

void NgramTable::gather(std::span<const std::int32_t> tokens, NgramHistory& history,
                        std::span<std::uint8_t> out) {
    if (out.size() != tokens.size() * static_cast<std::size_t>(kNgramTokenBytes)) {
        throw std::invalid_argument("ngram table: out must hold 2592 bytes per token");
    }
    accesses_.clear();
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        add_accesses(t, ngram_row_ids(tokens[t], history, constants_));
    }
    fetch(out);
}

void NgramTable::gather(std::span<const std::int32_t> tokens,
                        std::span<NgramHistory* const> histories, std::span<std::uint8_t> out) {
    if (out.size() != tokens.size() * static_cast<std::size_t>(kNgramTokenBytes) ||
        histories.size() != tokens.size() ||
        std::ranges::find(histories, nullptr) != histories.end()) {
        throw std::invalid_argument("ngram table: out must hold 2592 bytes and histories one "
                                    "non-null history per token");
    }
    accesses_.clear();
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        add_accesses(t, ngram_row_ids(tokens[t], *histories[t], constants_));
    }
    fetch(out);
}

void NgramTable::fetch(std::span<std::uint8_t> out) {
    // Hits are copied without touching recency, so the replay below sees accesses in order.
    misses_.clear();
    for (const Access& access : accesses_) {
        if (!cache_find(access.row, out.data() + access.offset)) {
            misses_.push_back({access.row / kNgramRowsPerPage,
                               static_cast<std::int32_t>(access.row % kNgramRowsPerPage),
                               access.offset});
        }
    }
    if (!misses_.empty()) {
        std::sort(misses_.begin(), misses_.end(),
                  [](const Miss& a, const Miss& b) { return a.page < b.page; });
        page_starts_.clear();
        for (std::size_t i = 0; i < misses_.size(); ++i) {
            if (i == 0 || misses_[i].page != misses_[i - 1].page) { page_starts_.push_back(i); }
        }
        page_starts_.push_back(misses_.size());
        read_pages(out.data());
    }
    for (const Access& access : accesses_) { cache_touch(access.row, out.data() + access.offset); }
}

} // namespace ninfer::targets::qwen3_8_flash_next

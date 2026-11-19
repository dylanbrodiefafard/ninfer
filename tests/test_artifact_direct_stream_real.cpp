// Streams the two routed-record tensors of the real Flash-Next artifact with Reader::stream and
// compares every byte with an independent read path (buffered pread in 1 MiB windows, dropping
// the page cache behind itself). A first pass with a no-op consumer measures the direct-read
// rate. No GPU is used, and peak resident memory must stay far below the 50 GB streamed.
#include "artifact/reader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using ninfer::artifact::Reader;

constexpr std::size_t kWindowBytes       = std::size_t{1} << 20;
constexpr std::uint64_t kMaxPeakRssBytes = std::uint64_t{1} << 30;
constexpr const char* kRecordTensors[2]  = {"text/routed_experts/e2m1_records",
                                            "text/routed_experts/f3_records"};

struct FreeDeleter {
    void operator()(std::byte* pointer) const noexcept { std::free(pointer); }
};

std::uint64_t peak_rss_bytes() {
    std::ifstream status("/proc/self/status");
    for (std::string line; std::getline(status, line);) {
        if (line.starts_with("VmHWM:")) { return std::stoull(line.substr(6)) * 1024; }
    }
    throw std::runtime_error("VmHWM is missing from /proc/self/status");
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
    if (path == nullptr || *path == '\0') {
        std::cout << "skip: set NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS\n";
        return 77;
    }
    try {
        const Reader reader(path);
        std::array<std::unique_ptr<std::byte, FreeDeleter>, Reader::stream_buffer_count> owners;
        Reader::StreamBuffers buffers;
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            owners[i].reset(static_cast<std::byte*>(
                std::aligned_alloc(Reader::direct_io_alignment, Reader::stream_buffer_bytes)));
            if (!owners[i]) { throw std::bad_alloc(); }
            buffers[i] = std::span(owners[i].get(), Reader::stream_buffer_bytes);
        }

        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) { throw std::runtime_error("independent open failed"); }
        std::vector<std::byte> window(kWindowBytes);
        std::uint64_t total = 0;
        double seconds      = 0.0;
        for (const char* name : kRecordTensors) {
            const auto payload = reader.payload(name);
            const ninfer::artifact::FileRange range{payload.absolute_offset, payload.data.size()};
            const auto begin = std::chrono::steady_clock::now();
            const auto physical =
                reader.stream(range, buffers, [](ninfer::artifact::PayloadSpan, std::size_t) {
                    return Reader::Completion{};
                });
            seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            total += physical;
            (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

            std::uint64_t compared = 0;
            (void)reader.stream(
                range, buffers, [&](ninfer::artifact::PayloadSpan chunk, std::size_t) {
                    for (std::size_t done = 0; done < chunk.data.size();) {
                        const std::size_t bytes = std::min(window.size(), chunk.data.size() - done);
                        const auto offset       = chunk.absolute_offset + done;
                        for (std::size_t got = 0; got < bytes;) {
                            const auto n = ::pread(fd, window.data() + got, bytes - got,
                                                   static_cast<off_t>(offset + got));
                            if (n <= 0) { throw std::runtime_error("independent pread failed"); }
                            got += static_cast<std::size_t>(n);
                        }
                        if (std::memcmp(window.data(), chunk.data.data() + done, bytes) != 0) {
                            throw std::runtime_error(std::string(name) +
                                                     " differs at file offset " +
                                                     std::to_string(offset));
                        }
                        (void)::posix_fadvise(fd, static_cast<off_t>(offset),
                                              static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
                        done += bytes;
                    }
                    compared += chunk.data.size();
                    return Reader::Completion{};
                });
            if (compared != range.bytes) { throw std::runtime_error("comparison was incomplete"); }
        }
        ::close(fd);
        const auto peak = peak_rss_bytes();
        std::cout << "streamed " << total << " physical bytes at "
                  << static_cast<double>(total) / seconds / 1e9 << " GB/s; peak RSS " << peak
                  << " bytes\n";
        if (peak > kMaxPeakRssBytes) { throw std::runtime_error("peak RSS grew with the stream"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

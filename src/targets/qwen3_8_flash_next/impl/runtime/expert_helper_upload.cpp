#include "runtime/expert_helper_upload.h"

#include "core/expert_mailbox.h"
#include "runtime/expert_helper_placement.h"

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
ExpertHelperUploadStats upload_expert_helper_records(std::span<const ExpertHelperUpload> records,
                                                     HipBuffer& destination,
                                                     const HipStream& stream) {
    ExpertHelperUploadStats stats;
    std::size_t end = 0;
    for (const auto& record : records) {
        const auto bytes = record.source.size();
        if ((bytes != static_cast<std::size_t>(ops::kMoeF3RecordBytes) &&
             bytes != static_cast<std::size_t>(ops::kMoeNvfp4RecordBytes)) ||
            record.destination_offset % 256 != 0 || record.destination_offset < end ||
            record.destination_offset > destination.size() ||
            bytes > destination.size() - record.destination_offset) {
            throw std::invalid_argument("expert helper upload: record geometry/range/overlap");
        }
        end = record.destination_offset + bytes;
        stats.bytes += bytes;
    }
    stats.records = records.size();
    if (records.empty()) { return stats; }
    const auto start = std::chrono::steady_clock::now();
    std::array<HipPinnedBuffer, 2> staging{HipPinnedBuffer(kHelperUploadStagingBytes),
                                           HipPinnedBuffer(kHelperUploadStagingBytes)};
    std::array<HipEvent, 2> events;
    HipBuffer device_checksums(records.size() * sizeof(std::uint64_t));
    HipPinnedBuffer checksums(records.size() * sizeof(std::uint64_t));
    std::vector<std::uint64_t> expected(records.size());
    try {
        for (std::size_t i = 0; i < records.size(); ++i) {
            const auto lane = i % 2;
            if (i >= 2) { events[lane].synchronize(); }
            const auto& record = records[i];
            std::memcpy(staging[lane].data(), record.source.data(), record.source.size());
            expected[i] = expert_record_checksum(record.source);
            auto* dst   = static_cast<std::byte*>(destination.data()) + record.destination_offset;
            hip_copy_to_device(dst, staging[lane].data(), record.source.size(), stream);
            events[lane].record(stream);
            expert_mailbox_hip_checksum(dst, record.source.size(),
                                        static_cast<std::uint64_t*>(device_checksums.data()) + i,
                                        stream);
        }
        hip_copy_to_host(checksums.data(), device_checksums.data(),
                         records.size() * sizeof(std::uint64_t), stream);
        stream.synchronize();
    } catch (...) {
        // A failed synchronization is not permission to free a still-referenced staging buffer.
        try {
            stream.synchronize();
        } catch (...) { std::terminate(); }
        throw;
    }
    const auto* actual = static_cast<const std::uint64_t*>(checksums.data());
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw HipError("expert helper upload checksum mismatch at record " + std::to_string(i));
        }
    }
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
}
} // namespace ninfer::targets::qwen3_8_flash_next

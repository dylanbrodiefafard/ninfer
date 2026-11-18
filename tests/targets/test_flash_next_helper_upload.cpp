#include "runtime/expert_helper_upload.h"
#include "runtime/expert_helper_placement.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;

int main() {
    try {
        HipDevice device("0000:13:00.0");
        HipStream stream;
        std::vector<std::vector<std::byte>> source;
        std::vector<ExpertHelperUpload> uploads;
        std::size_t bytes = 0;
        for (std::size_t record = 0; record < 7; ++record) {
            const auto size = static_cast<std::size_t>(record % 2 == 0 ? ops::kMoeF3RecordBytes
                                                                       : ops::kMoeNvfp4RecordBytes);
            auto& data      = source.emplace_back(size);
            for (std::size_t i = 0; i < size; ++i) {
                data[i] = static_cast<std::byte>((i * 71 + record * 113) & 255);
            }
            uploads.push_back({data, bytes});
            bytes += size + 256; // distinct initialized alignment gaps are not record bytes
        }
        HipBuffer destination(bytes);
        hip_zero(destination.data(), destination.size(), stream);
        stream.synchronize();
        const auto result = upload_expert_helper_records(uploads, destination, stream);
        std::vector<std::byte> actual(bytes);
        hip_copy_to_host(actual.data(), destination.data(), bytes, stream);
        stream.synchronize();
        std::size_t expected_bytes = 0;
        for (const auto& record : uploads) {
            if (std::memcmp(actual.data() + record.destination_offset, record.source.data(),
                            record.source.size()) != 0) {
                throw std::runtime_error("helper upload byte mismatch");
            }
            for (std::size_t i = 0; i < 256; ++i) {
                if (actual[record.destination_offset + record.source.size() + i] != std::byte{0}) {
                    throw std::runtime_error("helper upload changed gap bytes");
                }
            }
            expected_bytes += record.source.size();
        }
        if (result.bytes != expected_bytes || result.records != 7) {
            throw std::runtime_error("helper upload byte accounting mismatch");
        }
        bool invalid = false;
        try {
            const ExpertHelperUpload bad{uploads.front().source, 1};
            static_cast<void>(
                upload_expert_helper_records(std::span(&bad, 1), destination, stream));
        } catch (const std::invalid_argument&) { invalid = true; }
        if (!invalid) { throw std::runtime_error("helper upload accepted misaligned record"); }
        std::printf("upload records=%zu bytes=%zu staging=2x%zu seconds=%.6f checksum=ok "
                    "exact-readback=ok untouched-gaps=ok\n",
                    result.records, result.bytes, kHelperUploadStagingBytes, result.seconds);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

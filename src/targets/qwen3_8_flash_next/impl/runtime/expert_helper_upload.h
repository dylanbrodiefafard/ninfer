#pragma once

#include "core/hip_device.h"

#include <cstddef>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next {
struct ExpertHelperUpload {
    std::span<const std::byte> source;
    std::size_t destination_offset;
};

struct ExpertHelperUploadStats {
    std::size_t bytes = 0, records = 0;
    double seconds = 0;
};

// Exact record spans only (either supported codec), sorted disjoint 256-byte-aligned destinations.
// Uses exactly two 16 MiB HIP-pinned staging buffers and an event before each reuse. Independently
// computed CPU and GPU per-record U64 checksums must agree before return. No host-store HIP
// registration, second record arena, repacking or codec conversion. Throws on mismatch/API errors;
// all queued transfers are joined before staging or source lifetimes can end, including unwinds.
[[nodiscard]] ExpertHelperUploadStats
upload_expert_helper_records(std::span<const ExpertHelperUpload> records, HipBuffer& destination,
                             const HipStream& stream);
} // namespace ninfer::targets::qwen3_8_flash_next

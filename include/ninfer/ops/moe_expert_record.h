#pragma once

// Routed-expert record geometry shared by the CUDA and HIP sides: pure host constants with no
// device-runtime dependency. The byte layout itself is specified once, in "Routed-expert record"
// of ninfer/ops/moe_experts.h; these are the constants that layout names.

#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next expert geometry: hidden width H and intermediate width I.
inline constexpr std::int32_t kMoeHidden       = 2560;
inline constexpr std::int32_t kMoeIntermediate = 640;

/** Storage codec of a routed-expert record; bit 0 of the record's expert-table entry. */
enum class MoeExpertCodec : std::uint8_t {
    Nvfp4G16 = 0,
    F3G32    = 1,
};

// A record is a sequence of fixed-size chunk records, each covering 32 matrix rows and 128
// consecutive K.
inline constexpr std::int64_t kMoeGateUpChunks    = 800; // 40 row tiles x 20 K chunks
inline constexpr std::int64_t kMoeDownChunks      = 400; // 80 row tiles x 5 K chunks
inline constexpr std::int64_t kMoeF3ChunkBytes    = 1664;
inline constexpr std::int64_t kMoeNvfp4ChunkBytes = 2304;
inline constexpr std::int64_t kMoeRecordAlignment = 256;
inline constexpr std::int64_t kMoeF3RecordBytes =
    (kMoeGateUpChunks + kMoeDownChunks) * kMoeF3ChunkBytes;
inline constexpr std::int64_t kMoeNvfp4RecordBytes =
    (kMoeGateUpChunks + kMoeDownChunks) * kMoeNvfp4ChunkBytes;
static_assert(kMoeF3RecordBytes == 1996800 && kMoeNvfp4RecordBytes == 2764800);

[[nodiscard]] constexpr std::int64_t moe_expert_chunk_bytes(MoeExpertCodec codec) {
    return codec == MoeExpertCodec::F3G32 ? kMoeF3ChunkBytes : kMoeNvfp4ChunkBytes;
}

[[nodiscard]] constexpr std::int64_t moe_expert_record_bytes(MoeExpertCodec codec) {
    return codec == MoeExpertCodec::F3G32 ? kMoeF3RecordBytes : kMoeNvfp4RecordBytes;
}

} // namespace ninfer::ops

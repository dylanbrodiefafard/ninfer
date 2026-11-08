#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q5ScalarDecodeAtom {
    static constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        decode_pair(codes, high, scales, group_index, lane, w0, w1);
    }

    __device__ static __forceinline__ void
    decode_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
                std::int64_t group_index, int lane, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup)));
        const std::uint8_t packed =
            codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        w0 = static_cast<float>(q0) * scale;
        w1 = static_cast<float>(q1) * scale;
    }
};

struct Q5MmaDecodeAtom {
    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* staged_codes,
                                                                 const std::uint8_t* staged_high,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t staged_group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            staged_codes[staged_group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            staged_high[staged_group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail

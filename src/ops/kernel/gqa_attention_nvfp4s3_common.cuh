#pragma once

// Shared Sage3 NVFP4 constants and tiny device helpers. Kept out of the
// prefill kernel header so decode / TMA translation units do not parse the
// occupancy-2 prefill body (that include was a multi-minute single-core cicc).

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

// S3 constant folding: P is amplified by 448*6 inside the exp2 (the constant is
// added to the -max term in log2 domain); the per-16-key block scale then maps the
// amplified block max into e4m3, and P/scale lands in [0, 6] (e2m1 range).
inline constexpr float kGqaS3Fp8ScaleLog2 = -11.392317422778762f; // log2f(1.f / (448 * 6))
inline constexpr float kGqaS3Fp4ScaleLog2 = -2.584962500721156f;  // log2f(1.f / 6.f)
inline constexpr float kGqaS3PvAmpLog2    = -kGqaS3Fp8ScaleLog2;  // log2f(448 * 6)
inline constexpr float kGqaS3SfLog2 = -kGqaS3Fp8ScaleLog2 + kGqaS3Fp4ScaleLog2; // log2f(448)

inline constexpr int kGqaPrefillNvfp4s3Warps     = 16;
inline constexpr int kGqaPrefillNvfp4s3Threads    = kGqaPrefillNvfp4s3Warps * 32;
inline constexpr int kGqaPrefillNvfp4s3Br         = 128;
inline constexpr int kGqaPrefillNvfp4s3Bc         = 64;
inline constexpr int kGqaPrefillNvfp4s3Groups     = kGqaNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4s3CodeW      = kGqaNvfp4CodeWidth;
inline constexpr int kGqaPrefillNvfp4s3RowTiles   = kGqaPrefillNvfp4s3Br / 16;
inline constexpr int kGqaPrefillNvfp4s3DConsumers =
    kGqaPrefillNvfp4s3Warps / kGqaPrefillNvfp4s3RowTiles;
// P codes: [128 rows][48 B rows, 32 B used]. 48 B stride keeps 16 B ldmatrix
// windows aligned with no swizzle.
inline constexpr int kGqaPrefillNvfp4s3P4RowBytes = 48;
inline constexpr int kGqaPrefillNvfp4s3P4Bytes    = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3P4RowBytes;
// P per-16-key block scales: [128 rows][4 e4m3].
inline constexpr int kGqaPrefillNvfp4s3PsfBytes   = kGqaPrefillNvfp4s3Br * 4;
// V codes transposed to d-major: [256 d rows][48 B rows, 32 B used].
inline constexpr int kGqaPrefillNvfp4s3VtBytes     = kGqaPrefillHeadDim * kGqaPrefillNvfp4s3P4RowBytes;
// V per-(d, 16-key) block scales (d-major PV SFB operand): [256 d rows][4 e4m3].
inline constexpr int kGqaPrefillNvfp4s3VsfBytes    = kGqaPrefillHeadDim * 4;

inline constexpr int kGqaPrefillNvfp4s3QBytes      = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3QScaleBytes = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3Groups;
inline constexpr int kGqaPrefillNvfp4s3KBytes      = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3VBytes      = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3KScaleBytes = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3Groups;
// V-block scale plane is double-buffered (tile parity): the next tile's cp.async
// is issued while the current tile's PV mma is still reading its SFB plane, so a
// single buffer races (the async load can land mid-mma). One 1024 B stage each.
inline constexpr int kGqaPrefillNvfp4s3VsfStages = 2;
inline constexpr int kGqaPrefillNvfp4s3StatsBytes  =
    2 * kGqaPrefillNvfp4s3Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillNvfp4s3SmemBytes =
    kGqaPrefillNvfp4s3QBytes + kGqaPrefillNvfp4s3QScaleBytes + kGqaPrefillNvfp4s3KBytes +
    kGqaPrefillNvfp4s3VBytes + kGqaPrefillNvfp4s3P4Bytes + kGqaPrefillNvfp4s3PsfBytes +
    kGqaPrefillNvfp4s3VtBytes +
    kGqaPrefillNvfp4s3VsfBytes * kGqaPrefillNvfp4s3VsfStages + kGqaPrefillNvfp4s3KScaleBytes +
    kGqaPrefillNvfp4s3VsfBytes + kGqaPrefillNvfp4s3StatsBytes;

static_assert(kGqaPrefillNvfp4s3Groups == 16);
static_assert(kGqaPrefillNvfp4s3DConsumers == 2);
static_assert(kGqaPrefillNvfp4s3SmemBytes == 58880);

// Tile geometry for the exact (keep_frac=1) occupancy-2 path. The 32 KiB
// skip-list/proxy arrays are compile-excluded, Br is 64 so 8 warps keep the
// same PV accumulator footprint (DConsumers=2, PVNt=16), and 256 threads ×
// 128 regs × 2 CTAs fit the 64K register file. Dynamic smem is 44800 B;
// 2 × 43.8 KB = 87.5 KB < 100 KB/SM.
template <int BrT, int WarpsT>
struct GqaPrefillNvfp4s3Tile {
    static constexpr int Br          = BrT;
    static constexpr int Warps       = WarpsT;
    static constexpr int Threads     = WarpsT * 32;
    static constexpr int RowTiles    = BrT / 16;
    static constexpr int DConsumers  = WarpsT / RowTiles;
    static constexpr int QBytes      = BrT * kGqaPrefillNvfp4s3CodeW;
    static constexpr int QScaleBytes = BrT * kGqaPrefillNvfp4s3Groups;
    static constexpr int P4Bytes     = BrT * kGqaPrefillNvfp4s3P4RowBytes;
    static constexpr int PsfBytes    = BrT * 4;
    static constexpr int StatsBytes  = 2 * BrT * static_cast<int>(sizeof(float));
    static constexpr int SmemBytes =
        QBytes + QScaleBytes + kGqaPrefillNvfp4s3KBytes + kGqaPrefillNvfp4s3VBytes + P4Bytes +
        PsfBytes + kGqaPrefillNvfp4s3VtBytes +
        kGqaPrefillNvfp4s3VsfBytes * kGqaPrefillNvfp4s3VsfStages + kGqaPrefillNvfp4s3KScaleBytes +
        StatsBytes;
    static_assert(BrT % 16 == 0);
    static_assert(WarpsT % RowTiles == 0);
    static_assert(DConsumers == 2);
};

using GqaPrefillNvfp4s3Occ2 = GqaPrefillNvfp4s3Tile<64, 8>;
static_assert(GqaPrefillNvfp4s3Occ2::SmemBytes == 44800);
static_assert(GqaPrefillNvfp4s3Occ2::SmemBytes * 2 <= 102400);

// Sage V scale plane: d-major [page][kv_head][d 256][key_block 4] (1024 B per
// page-head, same size as the prod per-key plane).
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_s3_v_scale_index(int physical_page, int kv_head,
                                                             int d, int key_block) {
    return paged_kv_element_offset<kGqaNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head, 0, 0) +
           static_cast<std::int64_t>(d) * 4 + key_block;
}

// Pack two floats into one e2m1x2 byte (low nibble = first operand).
__device__ __forceinline__ std::uint8_t gqa_s3_cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n"
        " .reg .b8 b;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        " mov.b32 %0, {b, b, b, b};\n"
        "}"
        : "=r"(tmp)
        : "f"(lo), "f"(hi));
    return static_cast<std::uint8_t>(tmp);
}

// Per-nibble e2m1 decode (software). Avoids the hardware cvt.rn.f16x2.e2m1x2 path
// (__nv_fp4x2_e2m1 -> float2), which on sm_120a swaps the two halves when both
// nibbles are negative. Matches the host decode_e2m1_word table exactly.
__device__ __forceinline__ float gqa_s3_e2m1_value(std::uint8_t code) {
    static const float kGqaS3E2m1Mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = kGqaS3E2m1Mags[code & 0x07u];
    return (code & 0x08u) ? -mag : mag;
}

__device__ __forceinline__ float gqa_s3_k_smem_dequant(const std::uint8_t* k_codes,
                                                       const std::uint8_t* k_scale, int key,
                                                       int d) {
    const int grp          = d >> 4;
    const int logical_byte = d >> 1;
    const int phys         = gqa_nvfp4_swizzle_byte(key, logical_byte);
    const std::uint8_t cb  = k_codes[key * kGqaPrefillNvfp4s3CodeW + phys];
    const std::uint8_t nib = (d & 1) ? static_cast<std::uint8_t>((cb >> 4) & 0x0fu)
                                    : static_cast<std::uint8_t>(cb & 0x0fu);
    return gqa_s3_e2m1_value(nib) *
           detail::decode_nvfp4_e4m3(k_scale[key * kGqaPrefillNvfp4s3Groups + grp]);
}

} // namespace ninfer::ops

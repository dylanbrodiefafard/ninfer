#pragma once

// Host-visible XAttention ranker scratch sizing. The packed-K / logit / mass /
// keep buffers are transient GQA workspace (Prompt + xattn_tau < 1), not a
// hidden process-lifetime cudaMalloc.

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaXattnPrefillBr  = 128;
inline constexpr int kGqaXattnPrefillBc  = 64;
inline constexpr int kGqaXattnStride     = 16;
inline constexpr int kGqaXattnIRows      = kGqaXattnPrefillBr / kGqaXattnStride;
inline constexpr int kGqaXattnHeadDim    = 256;
inline constexpr int kGqaXattnRankTiles  = 4096; // 262144 / Bc
inline constexpr int kGqaXattnBf16Bytes  = 2;

inline int gqa_xattn_n_kb(int table_pages, std::uint32_t max_visible_keys) {
    const int env_pages =
        static_cast<int>((max_visible_keys + static_cast<std::uint32_t>(kGqaXattnPrefillBc) - 1u) /
                         static_cast<std::uint32_t>(kGqaXattnPrefillBc));
    int n = table_pages < env_pages ? table_pages : env_pages;
    if (n > kGqaXattnRankTiles) { n = kGqaXattnRankTiles; }
    if (n < 1) { n = 1; }
    return n;
}

inline std::size_t gqa_xattn_scratch_bytes(int q_heads, int kv_heads, int n_br, int n_kb) {
    const int n_j = n_kb * (kGqaXattnPrefillBc / kGqaXattnStride);
    auto align256 = [](std::size_t x) { return (x + 255u) & ~std::size_t{255}; };
    std::size_t b = 0;
    b = align256(b) + static_cast<std::size_t>(kGqaXattnBf16Bytes) *
                          static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(n_j) *
                          static_cast<std::size_t>(kGqaXattnStride) *
                          static_cast<std::size_t>(kGqaXattnHeadDim);
    b = align256(b) + sizeof(float) * static_cast<std::size_t>(q_heads) *
                          static_cast<std::size_t>(n_br) * static_cast<std::size_t>(kGqaXattnIRows) *
                          static_cast<std::size_t>(n_j);
    b = align256(b) + sizeof(float) * static_cast<std::size_t>(q_heads) *
                          static_cast<std::size_t>(n_br) * static_cast<std::size_t>(n_kb);
    b = align256(b) + sizeof(std::uint16_t) * static_cast<std::size_t>(q_heads) *
                          static_cast<std::size_t>(n_br) * static_cast<std::size_t>(n_kb);
    b = align256(b) + sizeof(int) * static_cast<std::size_t>(q_heads) *
                          static_cast<std::size_t>(n_br);
    return b;
}

} // namespace ninfer::ops

#pragma once

#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr std::int64_t kChunkSize    = kChunkTokens;
inline constexpr std::int64_t kSubChunkSize = 16;

static_assert(kChunkSize % kSubChunkSize == 0, "kChunkSize must be a multiple of kSubChunkSize");

struct bh_decode_t {
    int b;
    int h_v;

    static __device__ __forceinline__ bh_decode_t of(int bh, int H_v) {
        bh_decode_t r{};
        r.b   = bh / H_v;
        r.h_v = bh - r.b * H_v;
        return r;
    }

    static __device__ __forceinline__ bh_decode_t of(int bh, head_map qk_map) {
        bh_decode_t r{};
        const int H_v   = qk_map.H_v;
        r.b             = bh / H_v;
        const int cta_h = bh - r.b * H_v;
        r.h_v           = qk_map.cta_h_v(cta_h);
        return r;
    }
};

template <int TILES, int N>
__device__ __forceinline__ void zero_frag(float (&frag)[TILES][N]) {
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
        for (int e = 0; e < N; ++e) { frag[t][e] = 0.0f; }
    }
}

inline constexpr int BT    = static_cast<int>(kChunkSize);
inline constexpr int BC    = static_cast<int>(kSubChunkSize);
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

} // namespace ninfer::ops::detail::gated_delta_net::chunked

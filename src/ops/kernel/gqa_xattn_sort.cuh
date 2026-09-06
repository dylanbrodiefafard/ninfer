#pragma once

// Deterministic shared-memory rank ordering for XAttention. Scores sort descending;
// equal finite or infinite scores use the non-negative key-block id ascending, with
// negative padding ids ordered last.

#include <cuda_runtime.h>

namespace ninfer::ops {

template <int N, int Threads>
__device__ void gqa_xattn_bitonic_sort_desc(float* keys, int* ids, int tid) {
    static_assert((N & (N - 1)) == 0, "bitonic length must be a power of two");
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < N; i += Threads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool want_i_better = (i & k) == 0;
                    const bool i_better =
                        keys[i] > keys[ixj] ||
                        (keys[i] == keys[ixj] && ids[i] >= 0 &&
                         (ids[ixj] < 0 || ids[i] < ids[ixj]));
                    if (i_better != want_i_better) {
                        const float tk = keys[i];
                        keys[i]        = keys[ixj];
                        keys[ixj]      = tk;
                        const int ti   = ids[i];
                        ids[i]         = ids[ixj];
                        ids[ixj]       = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// nsort is the next supported power of two; callers pad keys/ids to that length.
// noinline keeps the rank network out of the attention MMA kernel's register file.
template <int Threads>
__device__ __noinline__ void gqa_xattn_sort_desc(float* keys, int* ids, int nsort, int tid) {
    if (nsort <= 64) {
        gqa_xattn_bitonic_sort_desc<64, Threads>(keys, ids, tid);
    } else if (nsort <= 128) {
        gqa_xattn_bitonic_sort_desc<128, Threads>(keys, ids, tid);
    } else if (nsort <= 256) {
        gqa_xattn_bitonic_sort_desc<256, Threads>(keys, ids, tid);
    } else if (nsort <= 512) {
        gqa_xattn_bitonic_sort_desc<512, Threads>(keys, ids, tid);
    } else if (nsort <= 1024) {
        gqa_xattn_bitonic_sort_desc<1024, Threads>(keys, ids, tid);
    } else if (nsort <= 2048) {
        gqa_xattn_bitonic_sort_desc<2048, Threads>(keys, ids, tid);
    } else {
        gqa_xattn_bitonic_sort_desc<4096, Threads>(keys, ids, tid);
    }
}

} // namespace ninfer::ops

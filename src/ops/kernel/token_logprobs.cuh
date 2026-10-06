#pragma once

// Implements: include/ninfer/ops/token_logprobs.h

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kTokenLogprobsBlock = 1024;

// A [.., W, B] panel addressed by byte strides, so strided views bind without a copy.
template <class T>
struct TokenLogprobsPanel {
    T* data                       = nullptr;
    std::int64_t slot_pitch_bytes = 0;
    std::int64_t row_pitch_bytes  = 0;
};

template <class T>
__device__ __forceinline__ T* token_logprobs_at(const TokenLogprobsPanel<T>& panel, int slot,
                                                int row) {
    using Byte = std::conditional_t<std::is_const_v<T>, const unsigned char, unsigned char>;
    return reinterpret_cast<T*>(reinterpret_cast<Byte*>(panel.data) +
                                static_cast<std::int64_t>(slot) * panel.slot_pitch_bytes +
                                static_cast<std::int64_t>(row) * panel.row_pitch_bytes);
}

struct TokenLogprobsArgs {
    TokenLogprobsPanel<const __nv_bfloat16> logits; // slot pitch is the column pitch
    TokenLogprobsPanel<const std::int32_t> tokens;
    TokenLogprobsPanel<const std::int32_t> columns; // null data selects column == slot
    const std::int32_t* row_enabled = nullptr;
    const std::int32_t* counts      = nullptr; // null activates every slot of an enabled row
    TokenLogprobsPanel<float> token_logprob;
    TokenLogprobsPanel<std::int32_t> top_ids;
    TokenLogprobsPanel<float> top_logprobs;
    std::int32_t token_domain = 0;
    std::int32_t top_count    = 0;
};

// Grid: blockIdx.x is the slot i in [0,W) and blockIdx.y the row b in [0,B); one block owns one
// slot and an inactive slot's block returns before touching its outputs. Thread t owns the token
// strip {t, t + kTokenLogprobsBlock, ...} of the slot's logit column, so one warp reads 32
// adjacent logits per step.
//
// Shared memory: warp_keys and warp_sums hold one entry per warp for the block reductions.
//
// Passes: (1) each thread finds the best ordering key of its strip and a block reduction yields
// the column maximum; (2) each thread sums exp(z - max) over its strip and a block reduction
// yields the log-sum-exp; (3) rank k is the block maximum of the per-thread keys. After rank k
// is taken, only the thread owning that token rescans its strip for its best key below it; every
// other thread's key already ranks below the winner. Each block reduction ends with a barrier, so
// the shared arrays are never rewritten while a thread still reads them.
//
// Requires 1 <= top_count <= token_domain, finite logits, and that the launcher validated every
// pitch as a multiple of its element size. Token indices fit 32 bits; byte offsets are 64-bit.
__launch_bounds__(kTokenLogprobsBlock) __global__
    void token_logprobs_kernel(TokenLogprobsArgs args) {
    const int slot = static_cast<int>(blockIdx.x);
    const int row  = static_cast<int>(blockIdx.y);
    if (args.row_enabled[row] == 0) { return; }
    if (args.counts != nullptr && slot >= args.counts[row]) { return; }

    const int column =
        args.columns.data != nullptr ? *token_logprobs_at(args.columns, slot, row) : slot;
    const __nv_bfloat16* logits = token_logprobs_at(args.logits, column, row);
    const int tid               = static_cast<int>(threadIdx.x);

    __shared__ unsigned long long warp_keys[kTokenLogprobsBlock / 32];
    __shared__ float warp_sums[kTokenLogprobsBlock / 32];

    unsigned long long local_key = 0ull;
    for (std::int32_t v = tid; v < args.token_domain; v += kTokenLogprobsBlock) {
        const unsigned long long key = sampling_sort_key(__bfloat162float(logits[v]), v);
        if (key > local_key) { local_key = key; }
    }
    unsigned long long winner = sampling_block_max_key_broadcast(local_key, warp_keys);
    const float max_logit     = sampling_key_float(winner);

    float local_sum = 0.0f;
    for (std::int32_t v = tid; v < args.token_domain; v += kTokenLogprobsBlock) {
        local_sum += expf(__bfloat162float(logits[v]) - max_logit);
    }
    const float log_sum_exp = max_logit + logf(sampling_block_sum_fast(local_sum, warp_sums));

    if (tid == 0) {
        const std::int32_t token = *token_logprobs_at(args.tokens, slot, row);
        *token_logprobs_at(args.token_logprob, slot, row) =
            __bfloat162float(logits[token]) - log_sum_exp;
    }

    std::int32_t* top_ids = token_logprobs_at(args.top_ids, slot, row);
    float* top_logprobs   = token_logprobs_at(args.top_logprobs, slot, row);
    for (std::int32_t rank = 0; rank < args.top_count; ++rank) {
        const int winner_id = sampling_key_index(winner);
        if (tid == 0) {
            top_ids[rank]      = winner_id;
            top_logprobs[rank] = sampling_key_float(winner) - log_sum_exp;
        }
        if (rank + 1 == args.top_count) { break; }
        if (tid == winner_id % kTokenLogprobsBlock) {
            local_key = 0ull;
            for (std::int32_t v = tid; v < args.token_domain; v += kTokenLogprobsBlock) {
                const unsigned long long key = sampling_sort_key(__bfloat162float(logits[v]), v);
                if (key < winner && key > local_key) { local_key = key; }
            }
        }
        winner = sampling_block_max_key_broadcast(local_key, warp_keys);
    }
}

} // namespace ninfer::ops

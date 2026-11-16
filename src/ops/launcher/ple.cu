#include "ops/launcher/ple.h"

#include "core/device.h"
#include "ninfer/ops/ple.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden    = kPleHidden;
constexpr int kStreams   = kPleStreams;
constexpr int kChannels  = kPleChannels;
constexpr int kHistory   = kPleConvHistory;
constexpr int kThreads   = 256;
constexpr int kWarps     = kThreads / kWarpSize;
constexpr int kPerThread = kHidden / kThreads;
static_assert(kHidden % kThreads == 0 && kChannels % kThreads == 0);

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_bits(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

// Grid (16 rows, T), block 160: thread j decodes value j of row r of token t from its int8_row
// record (FP16 scale at byte 0, int8 codes from byte 2).
__global__ void __launch_bounds__(kPleRowWidth)
    decode_rows_kernel(const std::uint8_t* __restrict__ rows,
                       std::uint16_t* __restrict__ embedding) {
    const int row          = static_cast<int>(blockIdx.x);
    const std::int64_t tok = blockIdx.y;
    const int j            = static_cast<int>(threadIdx.x);
    const std::uint8_t* record =
        rows + (tok * kPleRowsPerToken + row) * static_cast<std::int64_t>(kPleInt8RowBytes);
    const float scale = __half2float(*reinterpret_cast<const __half*>(record));
    const auto code   = static_cast<std::int8_t>(record[2 + j]);
    embedding[tok * kPleEmbedding + row * kPleRowWidth + j] =
        float_to_bf16_bits(static_cast<float>(code) * scale);
}

// Position of launch token t in the packed call and whether it lies in its row's valid prefix.
struct ChunkToken {
    std::int64_t flat; // packed column b * width + j
    int row;           // b
    int local;         // column within the chunk, j - chunk_begin
    int valid;         // valid columns of the row within the chunk
};

__device__ __forceinline__ int chunk_valid(const PleInjectChunk& chunk, int row) {
    const int valid = chunk.valid_columns == nullptr ? chunk.width : chunk.valid_columns[row];
    return min(max(valid - chunk.chunk_begin, 0), chunk.chunk_columns);
}

__device__ __forceinline__ ChunkToken chunk_token(const PleInjectChunk& chunk, int token) {
    const int row   = token / chunk.chunk_columns;
    const int local = token - row * chunk.chunk_columns;
    return {static_cast<std::int64_t>(row) * chunk.width + chunk.chunk_begin + local, row, local,
            chunk_valid(chunk, row)};
}

__device__ __forceinline__ std::int64_t slot_base(int slot) {
    return static_cast<std::int64_t>(slot) * kChannels * kHistory;
}

// Grid (4 streams, B * chunk columns), block 256. One pass gathers four block sums for stream i of
// launch token t: sum key^2, sum R^2, sum (key * gamma_key) * (R * gamma_query), and sum value^2.
// Thread 0 derives the gate g and sigmoid(g), and the RMS of G = sigmoid(g) * value from
// sigmoid(g)^2 * sum value^2, and publishes them through shared memory (one barrier). Every thread
// then writes its values of C = BF16(G * inv_rms(G) * gamma_conv) to the workspace
// [B * chunk columns][10240]; thread 0 stores sigmoid(g). Tokens past their row's valid prefix
// return before any access (the exit is uniform per CTA).
__global__ void __launch_bounds__(kThreads)
    gate_kernel(const std::uint16_t* __restrict__ residual, const float* __restrict__ key,
                const float* __restrict__ value, const std::uint16_t* __restrict__ norm_key,
                const std::uint16_t* __restrict__ norm_query,
                const std::uint16_t* __restrict__ norm_conv, std::uint16_t* __restrict__ conv_input,
                float* __restrict__ gate, PleInjectChunk chunk) {
    __shared__ float warp_sums[kWarps];
    __shared__ float broadcast[2];
    const int stream       = static_cast<int>(blockIdx.x);
    const std::int64_t tok = blockIdx.y;
    const ChunkToken at    = chunk_token(chunk, static_cast<int>(tok));
    if (at.local >= at.valid) { return; }
    const std::int64_t flat = at.flat * kChannels + stream * kHidden;
    const std::int64_t work = tok * kChannels + stream * kHidden;
    float key_squares = 0.0F, residual_squares = 0.0F, dot = 0.0F, value_squares = 0.0F;
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        const int d      = static_cast<int>(threadIdx.x) + i * kThreads;
        const float k    = key[flat + d];
        const float r    = bf16_bits_to_float(residual[flat + d]);
        const float v    = value[at.flat * kHidden + d];
        key_squares      = fmaf(k, k, key_squares);
        residual_squares = fmaf(r, r, residual_squares);
        value_squares    = fmaf(v, v, value_squares);
        dot = fmaf(k * (1.0F + bf16_bits_to_float(norm_key[stream * kHidden + d])),
                   r * (1.0F + bf16_bits_to_float(norm_query[stream * kHidden + d])), dot);
    }
    key_squares      = block_reduce_sum<kThreads>(key_squares, warp_sums);
    residual_squares = block_reduce_sum<kThreads>(residual_squares, warp_sums);
    dot              = block_reduce_sum<kThreads>(dot, warp_sums);
    value_squares    = block_reduce_sum<kThreads>(value_squares, warp_sums);
    if (threadIdx.x == 0) {
        constexpr float kInverseWidth = 1.0F / static_cast<float>(kHidden);
        const float inverse_key       = rsqrtf(key_squares * kInverseWidth + 1.0e-6F);
        const float inverse_residual  = rsqrtf(residual_squares * kInverseWidth + 1.0e-6F);
        const float s = inverse_key * inverse_residual * dot * rsqrtf(static_cast<float>(kHidden));
        const float magnitude = sqrtf(fmaxf(fabsf(s), 1.0e-6F));
        const float g         = s > 0.0F ? magnitude : (s < 0.0F ? -magnitude : 0.0F);
        const float scale     = sigmoid(g);
        broadcast[0]          = scale;
        broadcast[1]          = rsqrtf(scale * scale * value_squares * kInverseWidth + 1.0e-6F);
        gate[tok * kStreams + stream] = scale;
    }
    __syncthreads();
    const float scale         = broadcast[0];
    const float inverse_gated = broadcast[1];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        const int d = static_cast<int>(threadIdx.x) + i * kThreads;
        conv_input[work + d] =
            float_to_bf16_bits(scale * value[at.flat * kHidden + d] * inverse_gated *
                               (1.0F + bf16_bits_to_float(norm_conv[stream * kHidden + d])));
    }
}

// Grid (10240/256 channel groups, B * chunk columns), block 256: thread c of launch token t
// evaluates the dilated causal convolution of channel c from the chunk's C rows of the same row
// and, for taps before the chunk, the row's history slot (entry 9 + tau for tau in [-9,-1]), and
// adds G + SiLU(conv) to R in FP32. Tokens past their row's valid prefix are not touched.
__global__ void __launch_bounds__(kThreads)
    inject_kernel(std::uint16_t* __restrict__ residual, const float* __restrict__ value,
                  const std::uint16_t* __restrict__ conv_weight,
                  const std::uint16_t* __restrict__ conv_input, const float* __restrict__ gate,
                  const std::uint16_t* __restrict__ conv_states, PleInjectChunk chunk) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const std::int64_t tok = blockIdx.y;
    const ChunkToken at    = chunk_token(chunk, static_cast<int>(tok));
    if (at.local >= at.valid) { return; }
    const std::uint16_t* history = conv_states + slot_base(chunk.initial_slots[at.row]) +
                                   static_cast<std::int64_t>(c) * kHistory;
    float conv                   = 0.0F;
#pragma unroll
    for (int tap = 0; tap < kPleConvWidth; ++tap) {
        const int source         = at.local - kPleConvDilation * (kPleConvWidth - 1 - tap);
        const std::uint16_t bits = source >= 0
                                       ? conv_input[(tok - at.local + source) * kChannels + c]
                                       : history[kHistory + source];
        conv                     = fmaf(bf16_bits_to_float(conv_weight[c * kPleConvWidth + tap]),
                                        bf16_bits_to_float(bits), conv);
    }
    const int stream         = c / kHidden;
    const int d              = c - stream * kHidden;
    const std::int64_t index = at.flat * kChannels + c;
    const float gated        = gate[tok * kStreams + stream] * value[at.flat * kHidden + d];
    residual[index] = float_to_bf16_bits(bf16_bits_to_float(residual[index]) + gated + silu(conv));
}

// Grid (10240/256, B), block 256: thread c of row b loads its channel's nine entries of the row's
// initial history into registers, then publishes. Final: the last nine C of (history, the row's
// valid columns of the chunk) to the publish slot, which for ple_inject is the initial slot.
// EveryColumn: after each valid column j, the last nine C through j to publish slot + j. Because
// a thread reads its channel of every source before writing that channel of any destination, a
// row's initial slot may be one of its destinations. Rows touch disjoint slots by contract.
template <PleStatePublication Publication>
__global__ void __launch_bounds__(kThreads)
    state_kernel(const std::uint16_t* __restrict__ conv_input, std::uint16_t* conv_states,
                 PleInjectChunk chunk) {
    const int c     = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int row   = static_cast<int>(blockIdx.y);
    const int valid = chunk_valid(chunk, row);
    if (valid == 0) { return; }
    const std::int64_t first_token = static_cast<std::int64_t>(row) * chunk.chunk_columns;
    const std::uint16_t* source =
        conv_states + slot_base(chunk.initial_slots[row]) + static_cast<std::int64_t>(c) * kHistory;
    std::uint16_t window[kHistory];
#pragma unroll
    for (int j = 0; j < kHistory; ++j) { window[j] = source[j]; }
    const auto entry = [&](int position) {
        return position >= 0 ? conv_input[(first_token + position) * kChannels + c]
                             : window[kHistory + position];
    };
    const int publish = chunk.publish_slots[row];
    if constexpr (Publication == PleStatePublication::Final) {
        std::uint16_t* destination =
            conv_states + slot_base(publish) + static_cast<std::int64_t>(c) * kHistory;
        std::uint16_t next[kHistory];
#pragma unroll
        for (int j = 0; j < kHistory; ++j) { next[j] = entry(valid - kHistory + j); }
#pragma unroll
        for (int j = 0; j < kHistory; ++j) { destination[j] = next[j]; }
    } else {
        for (int column = 0; column < valid; ++column) {
            std::uint16_t* destination =
                conv_states + slot_base(publish + column) + static_cast<std::int64_t>(c) * kHistory;
#pragma unroll
            for (int j = 0; j < kHistory; ++j) {
                destination[j] = entry(column + 1 - kHistory + j);
            }
        }
    }
}

} // namespace

void ple_decode_rows_launch(const std::uint8_t* rows, std::uint16_t* embedding, std::int32_t tokens,
                            cudaStream_t stream) {
    decode_rows_kernel<<<dim3(kPleRowsPerToken, tokens), kPleRowWidth, 0, stream>>>(rows,
                                                                                    embedding);
    CUDA_CHECK(cudaGetLastError());
}

void ple_inject_launch(std::uint16_t* residual, const float* key, const float* value,
                       const std::uint16_t* norm_key, const std::uint16_t* norm_query,
                       const std::uint16_t* norm_conv, const std::uint16_t* conv_weight,
                       std::uint16_t* conv_states, std::uint16_t* conv_input, float* gate,
                       const PleInjectChunk& chunk, PleStatePublication publication,
                       cudaStream_t stream) {
    const unsigned tokens = static_cast<unsigned>(chunk.batch * chunk.chunk_columns);
    gate_kernel<<<dim3(kStreams, tokens), kThreads, 0, stream>>>(
        residual, key, value, norm_key, norm_query, norm_conv, conv_input, gate, chunk);
    CUDA_CHECK(cudaGetLastError());
    inject_kernel<<<dim3(kChannels / kThreads, tokens), kThreads, 0, stream>>>(
        residual, value, conv_weight, conv_input, gate, conv_states, chunk);
    CUDA_CHECK(cudaGetLastError());
    const dim3 state_grid(kChannels / kThreads, static_cast<unsigned>(chunk.batch));
    if (publication == PleStatePublication::Final) {
        state_kernel<PleStatePublication::Final>
            <<<state_grid, kThreads, 0, stream>>>(conv_input, conv_states, chunk);
    } else {
        state_kernel<PleStatePublication::EveryColumn>
            <<<state_grid, kThreads, 0, stream>>>(conv_input, conv_states, chunk);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

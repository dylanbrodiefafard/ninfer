#pragma once

// Implements: include/ninfer/ops/sigmoid_mul.h
// Match: contiguous BF16 inputs. Registered aligned/eight-element domains use
// one 16-byte pack per thread; BF16x2 and scalar routes preserve correctness for
// smaller alignments and odd tails. Sigmoid remains FP32 expf, not a fit.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/a4_activation.cuh"
#include "ops/kernel/a8_activation.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSigmoidGateMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 sigmoid_gate_mul_pair(const __nv_bfloat162& gate,
                                                                const __nv_bfloat162& x) {
    const float r0 = __low2float(x) * sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void sigmoid_gate_mul_scalar_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                               std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

__launch_bounds__(256) __global__
    void sigmoid_gate_mul_bf16x8_kernel(const Bf16x8Pack* gate, Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const Bf16x8Pack gv = load_vec<Bf16x8Pack>(gate + i);
        Bf16x8Pack xv       = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = sigmoid_gate_mul_pair(gv.pair[pair], xv.pair[pair]);
        }
        store_vec(x + i, xv);
    }
}

__launch_bounds__(256) __global__
    void sigmoid_gate_mul_bf16x2_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                        std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSigmoidGateMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    auto* x2          = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kSigmoidGateMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1  = gate2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 g2  = gate2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 g3  = gate2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = sigmoid_gate_mul_pair(g0, x0);
            x2[j + 1]                = sigmoid_gate_mul_pair(g1, x1);
            x2[j + 2]                = sigmoid_gate_mul_pair(g2, x2v);
            x2[j + 3]                = sigmoid_gate_mul_pair(g3, x3);
        } else {
            x2[j] = sigmoid_gate_mul_pair(g0, x0);
            if (j + 1 < n2) { x2[j + 1] = sigmoid_gate_mul_pair(gate2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = sigmoid_gate_mul_pair(gate2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * sigmoid(__bfloat162float(gate[i])));
    }
}

// Implements: include/ninfer/ops/sigmoid_mul.h (sigmoid_mul_a8)
// Match: contiguous BF16 gate/x [Rows,T]; BF16x2 loads need only 4-byte alignment. Grid: one CTA
// per token column; thread i holds BF16 pairs i + Block*k, k < Rows/(2*Block), computed with
// sigmoid_gate_mul_pair so the values equal sigmoid_mul's. Shared memory: Block/32 + 1 floats for
// the column maximum; the two barriers are inside a8_block_column_scale, which every thread
// reaches.
template <int Rows, int Block>
__launch_bounds__(Block) __global__
    void sigmoid_gate_mul_a8_kernel(const __nv_bfloat162* gate, const __nv_bfloat162* x,
                                    std::uint8_t* codes, float* scales) {
    static_assert(Rows % (2 * Block) == 0);
    constexpr int kPairsPerToken  = Rows / 2;
    constexpr int kPairsPerThread = kPairsPerToken / Block;
    const std::int64_t token      = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t col_base   = token * kPairsPerToken;

    __nv_bfloat162 values[kPairsPerThread];
    float maximum = 0.0F;
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const std::int64_t pair = col_base + threadIdx.x + k * Block;
        values[k]               = sigmoid_gate_mul_pair(gate[pair], x[pair]);
        maximum                 = fmaxf(maximum, a8_pair_maximum(values[k]));
    }

    __shared__ float warp_maxima[Block / kWarpSize];
    __shared__ float column_scale;
    const float scale   = a8_block_column_scale<Block>(maximum, warp_maxima, &column_scale);
    const float inverse = a8_inverse_scale(scale);
    auto* code_pairs    = reinterpret_cast<std::uint16_t*>(codes);
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        code_pairs[col_base + threadIdx.x + k * Block] = a8_encode_pair(values[k], inverse);
    }
    if (threadIdx.x == 0) { scales[token] = scale; }
}

// Implements: include/ninfer/ops/sigmoid_mul.h (sigmoid_mul_a4)
// Match: contiguous 16-byte aligned BF16 gate/x [Rows,T]. Thread i of the grid owns 16-value group
// i of the first round_up(T, 16) columns (Rows/16 groups per column): for a real column it computes
// the group's eight pairs with sigmoid_gate_mul_pair, so the values equal sigmoid_mul's, and
// publishes their NVFP4 codes and scale; for a padding column it writes the group's zero scale.
// No shared memory or barriers.
template <int Rows>
__launch_bounds__(256) __global__
    void sigmoid_gate_mul_a4_kernel(const Bf16x8Pack* gate, const Bf16x8Pack* x,
                                    std::int32_t tokens, std::uint8_t* codes, std::uint8_t* scales,
                                    float input_scale_divisor) {
    constexpr int kGroupsPerToken = Rows / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int token = task / kGroupsPerToken;
    const int group = task - token * kGroupsPerToken;
    if (token >= tokens) {
        if (token < (tokens + 15) / 16 * 16) {
            scales[a4_scale_offset<kGroupsPerToken>(token, group)] = 0;
        }
        return;
    }
    const std::int64_t pack = static_cast<std::int64_t>(task) * 2;
    std::uint32_t bits[8];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const Bf16x8Pack gv = load_vec<Bf16x8Pack>(gate + pack + half);
        const Bf16x8Pack xv = load_vec<Bf16x8Pack>(x + pack + half);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const __nv_bfloat162_raw product = sigmoid_gate_mul_pair(gv.pair[pair], xv.pair[pair]);
            bits[half * 4 + pair]            = static_cast<std::uint32_t>(product.x) |
                                               (static_cast<std::uint32_t>(product.y) << 16);
        }
    }
    const uint4 lo = make_uint4(bits[0], bits[1], bits[2], bits[3]);
    const uint4 hi = make_uint4(bits[4], bits[5], bits[6], bits[7]);
    a4_publish_group16<Rows>(codes, scales, input_scale_divisor, token, group, lo, hi);
}

} // namespace ninfer::ops

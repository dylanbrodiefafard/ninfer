#pragma once

// ninfer::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "ops/kernel/a8_activation.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,
};

// Which representations of the normalized row a kernel publishes: the BF16 tensor, the
// row-scaled E4M3 activation (include/ninfer/ops/a8_activation.h) of those BF16 values, or both.
enum class RmsOutput {
    Bf16,
    A8,
    Bf16AndA8,
};

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    return value;
}

inline constexpr int kRmsWarpMaxPairsPerLane = 4;

// One warp normalizes one row of D in {64, 128, 192, 256}: lane l holds BF16 pairs l + 32k,
// reduces its sum of squares with warp shuffles, and returns the BF16-rounded normalized pairs in
// `normalized` (entries for pairs >= D/2 are left unset). Every lane of the warp must call it.
// rmsnorm_warp_bf16x2_kernel and the per-token A8 producer share this function so that both
// publish identical BF16 values for a row.
template <RmsEpilogue Epilogue>
__device__ __forceinline__ void
rmsnorm_warp_row(const __nv_bfloat162* x_row, const __nv_bfloat162* weight,
                 const __nv_bfloat162* z_row, std::int32_t d, float eps, int lane,
                 __nv_bfloat162 (&normalized)[kRmsWarpMaxPairsPerLane]) {
    const int pairs = d / 2;
    __nv_bfloat162 values[kRmsWarpMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kRmsWarpMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k]       = x_row[pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kRmsWarpMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) { zf = __bfloat1622float2(z_row[pair]); }
            normalized[k] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row (rmsnorm_warp_row). Block is
// a scheduling choice rather than part of the row geometry.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                    std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 normalized[kRmsWarpMaxPairsPerLane];
    rmsnorm_warp_row<Epilogue>(x + row_base, weight,
                               Epilogue == RmsEpilogue::Gated ? z + row_base : nullptr, d, eps,
                               lane, normalized);
#pragma unroll
    for (int k = 0; k < kRmsWarpMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) { out[row_base + pair] = normalized[k]; }
    }
}

// Implements: include/ninfer/ops/gated_rmsnorm.h (gated_rmsnorm_a8)
// Match: aligned contiguous BF16 x/z [HeadDim, Heads, T], gated epilogue, HeadDim 128.
// Grid: one CTA per token column. Each of the Block/32 warps normalizes heads warp, warp + W, ...
// with rmsnorm_warp_row, so every head's BF16 values equal gated_rmsnorm's; the CTA then encodes
// the whole Heads*HeadDim column with one A8 scale. Lane l of a warp holds pairs l and l + 32 of
// each of its heads. Shared memory: Block/32 + 1 floats for the column maximum. Two
// __syncthreads() inside a8_block_column_scale; no early return before them.
template <int HeadDim, int Heads, int Block>
__launch_bounds__(Block) __global__
    void gated_rmsnorm_a8_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                 const __nv_bfloat162* z, std::uint8_t* codes, float* scales,
                                 float eps) {
    static_assert(HeadDim == 128, "two BF16 pairs per lane per head");
    static_assert(Block % kWarpSize == 0 && Heads % (Block / kWarpSize) == 0);
    constexpr int kWarps         = Block / kWarpSize;
    constexpr int kHeadsPerWarp  = Heads / kWarps;
    constexpr int kPairsPerHead  = HeadDim / 2;
    constexpr int kPairsPerLane  = kPairsPerHead / kWarpSize;
    constexpr int kPairsPerToken = Heads * kPairsPerHead;
    static_assert(kPairsPerLane <= kRmsWarpMaxPairsPerLane);

    const int lane              = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp              = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t token    = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t col_base = token * kPairsPerToken;

    __nv_bfloat162 normalized[kHeadsPerWarp][kRmsWarpMaxPairsPerLane];
    float maximum = 0.0F;
#pragma unroll
    for (int slot = 0; slot < kHeadsPerWarp; ++slot) {
        const std::int64_t head_base = col_base + (warp + slot * kWarps) * kPairsPerHead;
        rmsnorm_warp_row<RmsEpilogue::Gated>(x + head_base, weight, z + head_base, HeadDim, eps,
                                             lane, normalized[slot]);
#pragma unroll
        for (int k = 0; k < kPairsPerLane; ++k) {
            maximum = fmaxf(maximum, a8_pair_maximum(normalized[slot][k]));
        }
    }

    __shared__ float warp_maxima[kWarps];
    __shared__ float column_scale;
    const float scale   = a8_block_column_scale<Block>(maximum, warp_maxima, &column_scale);
    const float inverse = a8_inverse_scale(scale);
    auto* code_pairs    = reinterpret_cast<std::uint16_t*>(codes);
#pragma unroll
    for (int slot = 0; slot < kHeadsPerWarp; ++slot) {
        const std::int64_t head_base = col_base + (warp + slot * kWarps) * kPairsPerHead;
#pragma unroll
        for (int k = 0; k < kPairsPerLane; ++k) {
            code_pairs[head_base + lane + k * kWarpSize] =
                a8_encode_pair(normalized[slot][k], inverse);
        }
    }
    if (threadIdx.x == 0) { scales[token] = scale; }
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(kFullWarpMask, inv, 0);

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (Epilogue == RmsEpilogue::Gated) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
// Output selects the published representations: the A8 codes/scales encode exactly the BF16
// values written to `out` (A8 alone keeps them in registers instead). The A8 epilogue's
// a8_block_column_scale adds two __syncthreads(); every thread of the CTA reaches it.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread,
          RmsOutput Output = RmsOutput::Bf16>
__device__ __forceinline__ void
rmsnorm_cta_bf16x2_body(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                        const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                        std::int64_t rows, float eps, std::uint8_t* codes = nullptr,
                        float* scales = nullptr) {
    static_assert(Block % kWarpSize == 0);
    constexpr bool kWritesBf16 = Output != RmsOutput::A8;
    constexpr bool kWritesA8   = Output != RmsOutput::Bf16;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    float maximum = 0.0F;
#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) {
                zf = __bfloat1622float2(z[row_base + pair]);
            }
            const auto normalized =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
            if constexpr (kWritesBf16) { out[row_base + pair] = normalized; }
            if constexpr (kWritesA8) {
                values[k] = normalized;
                maximum   = fmaxf(maximum, a8_pair_maximum(normalized));
            }
        }
    }
    if constexpr (kWritesA8) {
        // warp_sums is free again: block_reduce_sum ends with a barrier and inv_shared was read
        // after the following one.
        const float scale   = a8_block_column_scale<Block>(maximum, warp_sums, &inv_shared);
        const float inverse = a8_inverse_scale(scale);
        auto* code_pairs    = reinterpret_cast<std::uint16_t*>(codes);
#pragma unroll
        for (int k = 0; k < MaxPairsPerThread; ++k) {
            if (k < pairs_per_thread) {
                code_pairs[row_base + threadIdx.x + k * Block] = a8_encode_pair(values[k], inverse);
            }
        }
        if (threadIdx.x == 0) { scales[row] = scale; }
    }
}

template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread,
          RmsOutput Output = RmsOutput::Bf16>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                   std::int64_t rows, float eps, std::uint8_t* codes = nullptr,
                                   float* scales = nullptr) {
    rmsnorm_cta_bf16x2_body<Epilogue, Block, MaxPairsPerThread, Output>(x, weight, z, out, d, rows,
                                                                        eps, codes, scales);
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (Epilogue == RmsEpilogue::Gated) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths.
template <RmsEpilogue Epilogue>
__device__ __forceinline__ void
rmsnorm_generic_body(const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* z,
                     __nv_bfloat16* out, std::int32_t d, std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const std::int64_t index = base + i;
        const float xv           = __bfloat162float(x[index]);
        const float wv           = __bfloat162float(weight[i]);
        float zv                 = 0.0f;
        if constexpr (Epilogue == RmsEpilogue::Gated) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
    }
}

template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    rmsnorm_generic_body<Epilogue>(x, weight, z, out, d, rows, eps);
}

} // namespace ninfer::ops

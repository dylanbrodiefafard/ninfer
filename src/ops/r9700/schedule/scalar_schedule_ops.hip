#include "ops/r9700/schedule/scalar_schedule_ops.h"

#include "ninfer/types.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::schedule {
namespace {

constexpr std::uint32_t kBlock            = 256;
constexpr std::uint32_t kWave             = 32;
constexpr std::uint32_t kNllWaves         = kBlock / kWave;
constexpr std::uint32_t kMaxPortableGridY = 65535;

struct alignas(16) Bf16x8 {
    hip_bfloat16 value[8];
};

static_assert(sizeof(Bf16x8) == 16);

__global__ void add_bias_x8_kernel(const Bf16x8* __restrict__ bias, Bf16x8* __restrict__ values,
                                   std::uint32_t packs, std::size_t rows) {
    const std::uint32_t pack = blockIdx.x * blockDim.x + threadIdx.x;
    if (pack >= packs) { return; }
    const Bf16x8 addend = bias[pack];
    for (std::size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        Bf16x8 result = values[row * packs + pack];
#pragma unroll
        for (std::uint32_t item = 0; item < 8; ++item) {
            result.value[item] = hip_bfloat16(static_cast<float>(result.value[item]) +
                                              static_cast<float>(addend.value[item]));
        }
        values[row * packs + pack] = result;
    }
}

__global__ void add_bias_scalar_kernel(const hip_bfloat16* __restrict__ bias,
                                       hip_bfloat16* __restrict__ values, std::uint32_t features,
                                       std::size_t elements) {
    const std::size_t first  = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = first; index < elements; index += stride) {
        values[index] = hip_bfloat16(static_cast<float>(values[index]) +
                                     static_cast<float>(bias[index % features]));
    }
}

__device__ float wave_max(float value) {
#pragma unroll
    for (std::uint32_t delta = kWave / 2; delta != 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down(value, delta, kWave));
    }
    return value;
}

__device__ float wave_sum(float value) {
#pragma unroll
    for (std::uint32_t delta = kWave / 2; delta != 0; delta >>= 1) {
        value += __shfl_down(value, delta, kWave);
    }
    return value;
}

__launch_bounds__(kBlock) __global__
    void nll_from_logits_kernel(const hip_bfloat16* __restrict__ logits,
                                const std::int32_t* __restrict__ targets,
                                float* __restrict__ output, std::uint32_t physical_rows,
                                std::uint32_t valid_rows) {
    const std::uint32_t column = blockIdx.x;
    const std::size_t base     = static_cast<std::size_t>(column) * physical_rows;
    const std::uint32_t lane   = threadIdx.x & (kWave - 1);
    const std::uint32_t wave   = threadIdx.x / kWave;

    float maximum = -INFINITY;
    for (std::uint32_t row = threadIdx.x; row < valid_rows; row += blockDim.x) {
        maximum = fmaxf(maximum, static_cast<float>(logits[base + row]));
    }
    maximum = wave_max(maximum);

    __shared__ float partial[kNllWaves];
    if (lane == 0) { partial[wave] = maximum; }
    __syncthreads();
    maximum = threadIdx.x < kNllWaves ? partial[lane] : -INFINITY;
    if (wave == 0) { maximum = wave_max(maximum); }
    if (threadIdx.x == 0) { partial[0] = maximum; }
    __syncthreads();
    maximum = partial[0];
    // The sum reduction below rewrites partial[0]; every wave must have read the maximum first.
    __syncthreads();

    float sum = 0.0F;
    for (std::uint32_t row = threadIdx.x; row < valid_rows; row += blockDim.x) {
        sum += expf(static_cast<float>(logits[base + row]) - maximum);
    }
    sum = wave_sum(sum);
    if (lane == 0) { partial[wave] = sum; }
    __syncthreads();
    sum = threadIdx.x < kNllWaves ? partial[lane] : 0.0F;
    if (wave == 0) { sum = wave_sum(sum); }
    if (threadIdx.x == 0) {
        const std::int32_t target = targets[column];
        output[column]            = maximum + logf(sum) - static_cast<float>(logits[base + target]);
    }
}

__global__ void prepare_masked_block_kernel(const std::int32_t* __restrict__ anchors,
                                            const std::int32_t* __restrict__ lengths,
                                            const std::int32_t* __restrict__ valid_columns,
                                            std::int32_t mask_id, std::int32_t* __restrict__ ids,
                                            std::int32_t* __restrict__ positions,
                                            std::uint32_t width, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const std::uint32_t column = index % width;
    const std::uint32_t batch  = index / width;
    const std::int32_t valid   = valid_columns[batch];
    ids[index]                 = column == 0 ? anchors[batch] : mask_id;
    positions[index]           = lengths[batch] + min(static_cast<std::int32_t>(column), valid - 1);
}

__global__ void prepare_ragged_prefix_kernel(
    const uint4* __restrict__ source, const std::int32_t* __restrict__ lanes,
    const std::int32_t* __restrict__ starts, const std::int32_t* __restrict__ ends,
    uint4* __restrict__ destination, std::int32_t* __restrict__ positions,
    std::int32_t* __restrict__ counts, std::uint32_t vectors_per_column, std::uint32_t width,
    std::size_t source_column_stride, std::size_t source_lane_stride) {
    const std::uint32_t column    = blockIdx.x;
    const std::uint32_t batch     = blockIdx.y;
    const std::int32_t count      = ends[batch] - starts[batch];
    const bool live               = static_cast<std::int32_t>(column) < count;
    const std::size_t source_base = static_cast<std::size_t>(lanes[batch]) * source_lane_stride +
                                    static_cast<std::size_t>(column) * source_column_stride;
    const std::size_t destination_base =
        (static_cast<std::size_t>(batch) * width + column) * vectors_per_column;
    const uint4 zero{};
    for (std::uint32_t vector = threadIdx.x; vector < vectors_per_column; vector += blockDim.x) {
        destination[destination_base + vector] = live ? source[source_base + vector] : zero;
    }
    if (threadIdx.x == 0) {
        positions[static_cast<std::size_t>(batch) * width + column] =
            starts[batch] + (live ? static_cast<std::int32_t>(column) : max(count - 1, 0));
        if (column == 0) { counts[batch] = count; }
    }
}

} // namespace

hipError_t add_bias_bf16(const hip_bfloat16* bias, hip_bfloat16* values, std::uint32_t features,
                         std::size_t rows, hipStream_t stream) noexcept {
    if (bias == nullptr || values == nullptr || features == 0 || rows == 0) {
        return hipErrorInvalidValue;
    }
    const auto addresses =
        reinterpret_cast<std::uintptr_t>(bias) | reinterpret_cast<std::uintptr_t>(values);
    if ((features % 8U) == 0 && (addresses & 15U) == 0) {
        const std::uint32_t packs = features / 8U;
        const dim3 grid((packs + kBlock - 1U) / kBlock,
                        static_cast<unsigned int>(std::min<std::size_t>(rows, kMaxPortableGridY)));
        add_bias_x8_kernel<<<grid, kBlock, 0, stream>>>(
            reinterpret_cast<const Bf16x8*>(bias), reinterpret_cast<Bf16x8*>(values), packs, rows);
    } else {
        if (rows > SIZE_MAX / features) { return hipErrorInvalidValue; }
        const std::size_t elements = rows * features;
        const std::size_t blocks =
            std::min<std::size_t>((elements + kBlock - 1U) / kBlock, kMaxPortableGridY);
        add_bias_scalar_kernel<<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(
            bias, values, features, elements);
    }
    return hipGetLastError();
}

hipError_t nll_from_logits_bf16(const hip_bfloat16* logits, const std::int32_t* targets,
                                float* output, std::uint32_t physical_rows,
                                std::uint32_t valid_rows, std::uint32_t columns,
                                hipStream_t stream) noexcept {
    if (logits == nullptr || targets == nullptr || output == nullptr || physical_rows == 0 ||
        valid_rows == 0 || valid_rows > physical_rows || columns == 0) {
        return hipErrorInvalidValue;
    }
    nll_from_logits_kernel<<<columns, kBlock, 0, stream>>>(logits, targets, output, physical_rows,
                                                           valid_rows);
    return hipGetLastError();
}

hipError_t prepare_masked_block_i32(const std::int32_t* anchors, const std::int32_t* lengths,
                                    const std::int32_t* valid_columns, std::int32_t mask_id,
                                    std::int32_t* ids, std::int32_t* positions, std::uint32_t width,
                                    std::uint32_t batch, hipStream_t stream) noexcept {
    if (anchors == nullptr || lengths == nullptr || valid_columns == nullptr || ids == nullptr ||
        positions == nullptr || width == 0 || width > 16 || batch == 0 ||
        batch > kDFlashMaximumConcurrency || mask_id < 0) {
        return hipErrorInvalidValue;
    }
    const std::uint32_t elements  = width * batch;
    constexpr std::uint32_t block = 128;
    prepare_masked_block_kernel<<<(elements + block - 1U) / block, block, 0, stream>>>(
        anchors, lengths, valid_columns, mask_id, ids, positions, width, elements);
    return hipGetLastError();
}

hipError_t prepare_ragged_prefix_bf16(const void* source, const std::int32_t* lanes,
                                      const std::int32_t* starts, const std::int32_t* ends,
                                      void* destination, std::int32_t* positions,
                                      std::int32_t* counts, std::uint32_t vectors_per_column,
                                      std::uint32_t width, std::uint32_t batch,
                                      std::size_t source_column_stride,
                                      std::size_t source_lane_stride, hipStream_t stream) noexcept {
    if (source == nullptr || lanes == nullptr || starts == nullptr || ends == nullptr ||
        destination == nullptr || positions == nullptr || counts == nullptr ||
        vectors_per_column == 0 || width == 0 || width > 16 || batch == 0 ||
        batch > kDFlashMaximumConcurrency || source_column_stride < vectors_per_column ||
        source_lane_stride < static_cast<std::size_t>(width) * source_column_stride) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(width, batch);
    prepare_ragged_prefix_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const uint4*>(source), lanes, starts, ends, static_cast<uint4*>(destination),
        positions, counts, vectors_per_column, width, source_column_stride, source_lane_stride);
    return hipGetLastError();
}

} // namespace ninfer::ops::r9700::schedule

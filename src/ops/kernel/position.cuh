#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

__global__ void offset_i32_position_rows_kernel(const std::int32_t* source,
                                                const std::int32_t* deltas,
                                                std::int32_t* destination, std::int32_t width) {
    const std::int32_t column =
        static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.y);
    if (column < width) {
        const std::int32_t index = row * width + column;
        destination[index]       = source[index] + deltas[row];
    }
}

} // namespace ninfer::ops

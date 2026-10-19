#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

// Copies disjoint device rows in stream order. Buffers must cover the pitched row spans and
// remain valid until the stream completes. Source and destination spans must not overlap.
// Addresses, pitches and row_bytes must be four-byte aligned; each pitched span must fit in
// INT32_MAX bytes, and rows must not exceed 65535. Empty copies enqueue no work. Invalid
// inputs throw std::invalid_argument; unrepresentable spans throw std::overflow_error.
// A nonempty transfer is one kernel node, whose parameters can change during CUDA Graph
// updates. Captured multidimensional memcpy nodes cannot change their workspace addresses.
void copy_device_rows_async(void* destination, std::size_t destination_pitch_bytes,
                            const void* source, std::size_t source_pitch_bytes,
                            std::size_t row_bytes, std::size_t rows, cudaStream_t stream);

} // namespace ninfer

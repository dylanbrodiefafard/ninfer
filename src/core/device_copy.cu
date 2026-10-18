#include "core/device_copy.h"

#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr int kCopyThreads = 256;

// Grid y selects one pitched row; grid x partitions that row into 256-vector tiles. Every
// thread copies one disjoint vector, with a tail guard. There is no shared memory, asynchronous
// staging or synchronization. Host validation bounds every byte span below 2^31 and guarantees
// alignment for Vector. uint4 moves aligned payloads 16 bytes per lane; uint32_t handles the
// four-byte token-panel pitches. Values are copied bit-for-bit without floating-point casts.
template <class Vector>
__global__ void copy_device_rows_kernel(Vector* __restrict__ destination,
                                        const Vector* __restrict__ source, int destination_pitch,
                                        int source_pitch, int row_units) {
    const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row    = static_cast<int>(blockIdx.y);
    if (column < row_units) {
        destination[row * destination_pitch + column] = source[row * source_pitch + column];
    }
}

} // namespace

void copy_device_rows_async(void* destination, std::size_t destination_pitch_bytes,
                            const void* source, std::size_t source_pitch_bytes,
                            std::size_t row_bytes, std::size_t rows, cudaStream_t stream) {
    if (rows == 0 || row_bytes == 0) { return; }
    const auto alignment = reinterpret_cast<std::uintptr_t>(destination) |
                           reinterpret_cast<std::uintptr_t>(source) | destination_pitch_bytes |
                           source_pitch_bytes | row_bytes;
    constexpr std::size_t max_index = std::numeric_limits<int>::max();
    if (destination == nullptr || source == nullptr || (alignment & 3U) != 0 ||
        row_bytes > destination_pitch_bytes || row_bytes > source_pitch_bytes ||
        destination_pitch_bytes > max_index || source_pitch_bytes > max_index || rows > 65535) {
        throw std::invalid_argument("device row copy has an invalid pointer, alignment or shape");
    }
    if (rows - 1 > (max_index - row_bytes) / destination_pitch_bytes ||
        rows - 1 > (max_index - row_bytes) / source_pitch_bytes) {
        throw std::overflow_error("device row copy span exceeds INT32_MAX bytes");
    }
    const auto launch = [&]<class Vector>() {
        const auto units = static_cast<int>(row_bytes / sizeof(Vector));
        const dim3 grid(static_cast<unsigned int>((units + kCopyThreads - 1) / kCopyThreads),
                        static_cast<unsigned int>(rows));
        copy_device_rows_kernel<<<grid, kCopyThreads, 0, stream>>>(
            static_cast<Vector*>(destination), static_cast<const Vector*>(source),
            static_cast<int>(destination_pitch_bytes / sizeof(Vector)),
            static_cast<int>(source_pitch_bytes / sizeof(Vector)), units);
    };
    if ((alignment & 15U) == 0) {
        launch.template operator()<uint4>();
    } else {
        launch.template operator()<std::uint32_t>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

// Each thread expands one packed word. The 256-by-256 grid covers all four-code inputs;
// threads have disjoint outputs, no shared storage, and no synchronization requirement.
__global__ void expand_codes(std::uint32_t* output) {
    const unsigned packed = blockIdx.x * blockDim.x + threadIdx.x;
    output[packed]        = ninfer::ops::detail::nvfp4_codes_to_fp8(packed);
}

} // namespace

int main() {
    try {
        const ninfer::DeviceContext device;
        constexpr unsigned kWords = 65536;
        ninfer::DeviceBuffer output(kWords * sizeof(std::uint32_t));
        expand_codes<<<256, 256, 0, device.stream>>>(static_cast<std::uint32_t*>(output.p));
        CUDA_CHECK(cudaGetLastError());
        device.synchronize();
        std::vector<std::uint32_t> actual(kWords);
        output.copy_to_host(actual.data(), actual.size() * sizeof(std::uint32_t));
        // Exact E4M3 encodings of the sixteen signed E2M1 values, including both zeros.
        constexpr std::array<std::uint32_t, 16> kOracle{
            0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
            0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc,
        };
        for (unsigned packed = 0; packed < kWords; ++packed) {
            std::uint32_t expected = 0;
            for (unsigned lane = 0; lane < 4; ++lane) {
                expected |= kOracle[(packed >> (4 * lane)) & 15U] << (8 * lane);
            }
            if (actual[packed] != expected) {
                std::cerr << "NVFP4 expansion mismatch at packed word " << packed << '\n';
                return 1;
            }
        }
        std::cout << "OK NVFP4 expansion: all 65536 packed words\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 expansion: " << error.what() << '\n';
        return 1;
    }
}

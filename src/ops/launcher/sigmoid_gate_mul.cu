// Implements: include/ninfer/ops/sigmoid_mul.h
// Finite dispatch: aligned BF16x8 production route, BF16x2 fallback, then
// scalar fallback for two-byte-aligned sliced storage.
#include "ops/launcher/sigmoid_gate_mul.h"

#include "ops/common/math.h"
#include "ops/kernel/sigmoid_gate_mul.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

void sigmoid_gate_mul_bf16x8_launch(const Tensor& gate, Tensor& x, int block, cudaStream_t stream) {
    const std::int64_t packs = x.numel() / 8;
    constexpr int kMaxGrid   = 4096;
    const int grid           = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
    sigmoid_gate_mul_bf16x8_kernel<<<grid, block, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(gate.data), static_cast<Bf16x8Pack*>(x.data), packs);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    const std::int64_t n   = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const auto gate_addr   = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (((gate_addr | x_addr) & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        sigmoid_gate_mul_bf16x8_launch(gate, x, kBlock, stream);
        return;
    }
    if (((gate_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(
            std::min<std::int64_t>(kMaxGrid, div_up(n, static_cast<std::int64_t>(kBlock))));
        sigmoid_gate_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSigmoidGateMulPairsPerThread;
    const int grid                        = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock))));

    sigmoid_gate_mul_bf16x2_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_gate_mul_a8_launch(const Tensor& gate, const Tensor& x, A8Activation& activation,
                                cudaStream_t stream) {
    // One CTA per token column; at verify widths the grid is only T CTAs, so 1024 threads each
    // evaluate three pairs of the 3072-pair column to keep the dependent expf chain short.
    constexpr int kRows  = 6144;
    constexpr int kBlock = 1024;
    sigmoid_gate_mul_a8_kernel<kRows, kBlock>
        <<<static_cast<unsigned int>(x.ne[1]), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(gate.data),
            static_cast<const __nv_bfloat162*>(x.data),
            static_cast<std::uint8_t*>(activation.codes.data),
            static_cast<float*>(activation.scales.data));
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_gate_mul_a4_launch(const Tensor& gate, const Tensor& x, A4Activation& activation,
                                cudaStream_t stream) {
    constexpr int kRows       = 6144;
    constexpr int kBlock      = 256;
    const std::int32_t tokens = x.ne[1];
    const std::int64_t tasks  = static_cast<std::int64_t>((tokens + 15) / 16 * 16) * (kRows / 16);
    // The kernel indexes groups in 32 bits.
    if (tasks > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("sigmoid_mul_a4: T exceeds the 32-bit group index");
    }
    sigmoid_gate_mul_a4_kernel<kRows>
        <<<static_cast<unsigned int>(div_up(tasks, static_cast<std::int64_t>(kBlock))), kBlock, 0,
           stream>>>(
            static_cast<const Bf16x8Pack*>(gate.data), static_cast<const Bf16x8Pack*>(x.data),
            tokens, static_cast<std::uint8_t*>(activation.codes.data),
            static_cast<std::uint8_t*>(activation.scales.data), activation.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

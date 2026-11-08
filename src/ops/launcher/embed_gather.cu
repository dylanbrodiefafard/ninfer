// ninfer::ops - embedding launcher: variant grid/block/stream setup.
#include "ops/launcher/embed_gather.h"

#include "ops/common/math.h"
#include "ops/kernel/embed_gather.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 128;

template <int BlocksPerToken, int Threads>
void launch_fp8(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int grid = ids.ne[0] * BlocksPerToken;
    embed_gather_fp8_kernel<BlocksPerToken, Threads><<<grid, Threads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales), static_cast<__nv_bfloat16*>(out.data));
}

int grid_for(std::int64_t n) {
    return static_cast<int>(
        std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
}

} // namespace

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_dense_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(table.data),
        static_cast<__nv_bfloat16*>(out.data), d, T);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_w8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    const auto aligned16 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
    };
    if (d == kEmbedGatherW8TextD && table.padded_shape[1] == kEmbedGatherW8TextD &&
        aligned16(codes) && aligned16(out.data) &&
        (reinterpret_cast<std::uintptr_t>(scales) & 1U) == 0) {
        embed_gather_w8_row_5120_kernel<<<T, kEmbedGatherW8TextThreads, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_w8_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream) {
    const std::int32_t T = ids.ne[0];
    if (T <= 48) {
        launch_fp8<10, 32>(ids, table, out, stream);
    } else {
        launch_fp8<1, 256>(ids, table, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

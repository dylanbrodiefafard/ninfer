// Implements: include/ninfer/ops/stream_mean.h
#include "ops/launcher/stream_mean.h"

#include "core/device.h"
#include "ops/kernel/stream_mean.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void stream_mean_launch(const Tensor& residual, Tensor& out, cudaStream_t stream) {
    const auto tokens = static_cast<unsigned int>(residual.ne[2]);
    stream_mean_kernel<<<tokens, kStreamMeanThreads, 0, stream>>>(
        static_cast<const uint4*>(residual.data), static_cast<__nv_bfloat16*>(out.data),
        out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

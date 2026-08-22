#include "ops/launcher/nll_from_logits.h"

#include "ops/kernel/nll_from_logits.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {

void nll_from_logits_launch(const Tensor& logits, const Tensor& targets, Tensor& out,
                            std::int32_t valid_rows, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }
    nll_from_logits_kernel<<<static_cast<unsigned int>(t_count), kNllFromLogitsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(targets.data), static_cast<float*>(out.data), valid_rows,
        physical_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

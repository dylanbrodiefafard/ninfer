// Implements: include/ninfer/ops/dflash2_path_select.h
#include "ops/launcher/dflash2_path_select.h"

#include "ops/kernel/dflash2_path_select.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_path_select_bf16_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) {
    const std::int32_t n_rows = out.ne[0];
    const std::int32_t k_rows = x.ne[0];
    const std::int32_t cols   = x.ne[1];
    const dim3 grid(static_cast<unsigned int>(n_rows), static_cast<unsigned int>(cols));
    dflash2_path_select_bf16_gemv_kernel<<<grid, kDflash2PathSelectGemmBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata != nullptr ? weight.qdata : weight.payload),
        static_cast<__nv_bfloat16*>(out.data), n_rows, k_rows);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_path_select_launch(const Tensor& logits, const Tensor& hidden_proj,
                                const Tensor& pred_code, const Tensor& succ_code,
                                const Tensor& anchors, Tensor& path, float temperature,
                                unsigned long long seed, cudaStream_t stream,
                                const Tensor* logit_token_ids) {
    const std::int32_t vocab          = logits.ne[0];
    const std::int32_t tokens         = logits.ne[1];
    const std::int32_t batch          = logits.ne[2];
    const std::int32_t codebook_rows  = pred_code.ne[1];
    dflash2_path_select_kernel<<<static_cast<unsigned int>(batch), kDflash2PathSelectBlock, 0,
                                 stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const __nv_bfloat16*>(hidden_proj.data),
        static_cast<const __nv_bfloat16*>(pred_code.data),
        static_cast<const __nv_bfloat16*>(succ_code.data),
        static_cast<const std::int32_t*>(anchors.data),
        logit_token_ids != nullptr ? static_cast<const std::int32_t*>(logit_token_ids->data)
                                   : nullptr,
        static_cast<std::int32_t*>(path.data), vocab, tokens, batch, codebook_rows, temperature,
        seed);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

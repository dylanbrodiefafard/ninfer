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

void dflash2_column_topk_launch(const Tensor& logits, float* split_val, int* split_idx,
                                float* cand_val, int* cand_idx, const Tensor* logit_token_ids,
                                cudaStream_t stream) {
    const std::int32_t vocab  = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    const std::int32_t batch  = logits.ne[2];
    const dim3 split_grid(static_cast<unsigned int>(kDflash2PathSelectTopkSplits),
                          static_cast<unsigned int>(tokens), static_cast<unsigned int>(batch));
    dflash2_column_topk_split_kernel<<<split_grid, kDflash2PathSelectBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), split_val, split_idx, vocab, tokens, batch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 merge_grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(batch));
    dflash2_column_topk_merge_kernel<<<merge_grid, kDflash2PathSelectBlock, 0, stream>>>(
        split_val, split_idx, cand_val, cand_idx,
        logit_token_ids != nullptr ? static_cast<const std::int32_t*>(logit_token_ids->data)
                                   : nullptr,
        vocab, tokens, batch);
    CUDA_CHECK(cudaGetLastError());
}

Dflash2CodebookDevice codebook_device(const Tensor* bf16, const Weight* nvfp4) {
    Dflash2CodebookDevice out{};
    if (nvfp4 != nullptr) {
        out.nvfp4_codes  = static_cast<const std::uint8_t*>(nvfp4->qdata);
        out.nvfp4_scales = static_cast<const std::uint8_t*>(nvfp4->scales);
        out.inv_dw       = 1.0F / nvfp4->weight_scale_divisor;
        return out;
    }
    out.bf16 = static_cast<const __nv_bfloat16*>(bf16->data);
    return out;
}

void dflash2_path_select_launch(const float* cand_val, const int* cand_idx,
                                const Tensor& hidden_proj, const Tensor* pred_bf16,
                                const Tensor* succ_bf16, const Weight* pred_nvfp4,
                                const Weight* succ_nvfp4, const Tensor& anchors,
                                const Tensor& logical_positions, Tensor& path,
                                std::int32_t tokens, std::int32_t batch,
                                const SamplingConfig* configs, cudaStream_t stream,
                                Tensor* selector_ids, Tensor* selector_q,
                                unsigned long long seed_xor, std::int32_t position_offset,
                                bool force_greedy) {
    dflash2_path_select_kernel<<<static_cast<unsigned int>(batch), kDflash2PathSelectBlock, 0,
                                 stream>>>(
        cand_val, cand_idx, static_cast<const __nv_bfloat16*>(hidden_proj.data),
        codebook_device(pred_bf16, pred_nvfp4), codebook_device(succ_bf16, succ_nvfp4),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(logical_positions.data),
        static_cast<std::int32_t*>(path.data),
        selector_ids != nullptr ? static_cast<std::int32_t*>(selector_ids->data) : nullptr,
        selector_q != nullptr ? static_cast<float*>(selector_q->data) : nullptr, tokens, batch,
        configs, seed_xor, position_offset, force_greedy);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_tree_select_launch(const float* cand_val, const int* cand_idx,
                                const Tensor& hidden_proj, const Tensor* pred_bf16,
                                const Tensor* succ_bf16, const Weight* pred_nvfp4,
                                const Weight* succ_nvfp4, const Tensor& anchors,
                                const Tensor& frontiers, Tensor& verify_ids, Tensor& parent_index,
                                Tensor& cache_positions, Tensor& rope_positions,
                                Tensor& ancestor_mask, Tensor& valid_columns, std::int32_t tokens,
                                std::int32_t batch, std::int32_t out_width, cudaStream_t stream) {
    dflash2_tree_select_kernel<<<static_cast<unsigned int>(batch), kDflash2PathSelectBlock, 0,
                                 stream>>>(
        cand_val, cand_idx, static_cast<const __nv_bfloat16*>(hidden_proj.data),
        codebook_device(pred_bf16, pred_nvfp4), codebook_device(succ_bf16, succ_nvfp4),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(frontiers.data),
        static_cast<std::int32_t*>(verify_ids.data),
        static_cast<std::int32_t*>(parent_index.data),
        static_cast<std::int32_t*>(cache_positions.data),
        static_cast<std::int32_t*>(rope_positions.data),
        static_cast<std::int32_t*>(ancestor_mask.data),
        static_cast<std::int32_t*>(valid_columns.data), tokens, batch, out_width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

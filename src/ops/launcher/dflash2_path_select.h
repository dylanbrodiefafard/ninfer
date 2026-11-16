#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_path_select_bf16_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
// Bytes of split-key scratch dflash2_column_topk_launch needs for `columns` logit columns of at
// most kDflash2PathSelectMaxLogitRows rows each.
[[nodiscard]] std::size_t dflash2_column_topk_split_bytes(std::int64_t columns);
// Writes each column's top-16 values and (mapped) ids in rank order to cand_val/cand_idx
// [16, T*B]. split_keys holds dflash2_column_topk_split_bytes(T*B) bytes, 16-byte aligned.
void dflash2_column_topk_launch(const Tensor& logits, std::uint64_t* split_keys, float* cand_val,
                                int* cand_idx, const Tensor* logit_token_ids, cudaStream_t stream);
void dflash2_path_select_launch(const float* cand_val, const int* cand_idx,
                                const Tensor& hidden_proj, const Tensor* pred_bf16,
                                const Tensor* succ_bf16, const Weight* pred_nvfp4,
                                const Weight* succ_nvfp4, const Tensor& anchors,
                                const Tensor& logical_positions, Tensor& path, std::int32_t tokens,
                                std::int32_t batch, const SamplingConfig* configs,
                                cudaStream_t stream, Tensor* selector_ids = nullptr,
                                Tensor* selector_q = nullptr, unsigned long long seed_xor = 0,
                                std::int32_t position_offset = 0, bool force_greedy = false);
void dflash2_tree_select_launch(
    const float* cand_val, const int* cand_idx, const Tensor& hidden_proj, const Tensor* pred_bf16,
    const Tensor* succ_bf16, const Weight* pred_nvfp4, const Weight* succ_nvfp4,
    const Tensor& anchors, const Tensor& frontiers, Tensor& verify_ids, Tensor& parent_index,
    Tensor& cache_positions, Tensor& rope_positions, Tensor& ancestor_mask, Tensor& valid_columns,
    std::int32_t tokens, std::int32_t batch, std::int32_t out_width, const SamplingConfig* configs,
    float p_less_tree_temperature, cudaStream_t stream);

} // namespace ninfer::ops::detail

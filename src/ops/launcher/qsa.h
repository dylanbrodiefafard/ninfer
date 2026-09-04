#pragma once

#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail {

void qsa_state_append_launch(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                             const Tensor& position_ids, const Tensor& append_ids,
                             QsaStateView state, cudaStream_t stream);
void qsa_index_select_launch(const Tensor& raw_query, const QsaStateView& state,
                             const Tensor& query_ids, const Tensor& visible_ids,
                             const Tensor& visible_offsets, const Tensor& query_norm_weight,
                             const Tensor& key_norm_weight, Tensor& selected_ids,
                             Tensor& selected_count, Tensor& workspace, cudaStream_t stream);
void qsa_selected_attention_launch(const Tensor& q, const Tensor& selected_ids,
                                   const Tensor& selected_count, const QsaStateView& state,
                                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail

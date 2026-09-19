#pragma once
#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail {
inline constexpr int kQsaPagedSelectorWords = 512 + 1024 + 512;
void qsa_paged_append_launch(const Tensor&, const Tensor&, const Tensor&,
    const QsaBatchControls&, QsaPagedStateView, cudaStream_t);
void qsa_paged_select_launch(const Tensor&, const QsaPagedStateView&,
    const QsaBatchControls&, int, const Tensor&, const Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);
void qsa_paged_attention_launch(const Tensor&, const Tensor&, const Tensor&,
    const QsaPagedStateView&, const QsaBatchControls&, Tensor&, cudaStream_t);
void qsa_paged_mask_launch(Tensor&, const QsaBatchControls&, cudaStream_t);
// Shared by the composite and primitive wrappers; no device data is inspected.
void qsa_validate_paged(const QsaPagedStateView&, const QsaBatchControls&,
    int width, int batch, int max_visible, const char* op);
void qsa_validate_batch(int width, int batch);
}

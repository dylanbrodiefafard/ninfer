#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Width-four causal convolution, SiLU, and q/k/value split of an already projected BF16
// [10240, W, B] plane (the conv record of a record route) from the read-only history selected by
// initial_state_slots. parent_index, when non-null and non-empty, is the I32 [W, B] tree-parent
// operand of gdn_input_proj_conv_record.
void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream,
                                      const Tensor* parent_index = nullptr);

// The same convolution over a projected BF16 [10240, W, B] plane, publishing the BF16 history
// after valid column j of row b to conv_states slot snapshot_base_slots[b] + j. A row's initial
// slot may lie in its own destination interval: each thread loads its channel of the initial
// history before it writes that channel of any snapshot.
void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream);

} // namespace ninfer::ops::detail

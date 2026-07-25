#pragma once

#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail

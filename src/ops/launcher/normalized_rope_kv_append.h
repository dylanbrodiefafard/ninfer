#pragma once

#include "ninfer/ops/normalized_rope_kv_append.h"

namespace ninfer::ops::detail {

void normalized_rope_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& gamma,
                                      const Tensor& positions, const Tensor& counts,
                                      const Tensor& lanes, float epsilon, float theta,
                                      KVCacheAppendPrefixExecutionEnvelope envelope,
                                      CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops::detail

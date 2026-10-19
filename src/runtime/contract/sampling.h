#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. P-less defaults to temperature 1.5; other
// omitted model-owned fields use the registered preset. An omitted seed remains deterministic.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace ninfer::runtime

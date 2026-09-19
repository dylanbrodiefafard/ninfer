#pragma once

#include "ninfer/types.h"
#include <span>

namespace ninfer::ops { struct SamplingConfig; }

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

// Boundary-only rollback of a licensed-but-unpublished suffix. All sampler consumers must
// already be drained. Preserves the shared positive-temperature occurrence-count policy;
// greedy/null-count configurations are no-ops. Throws on count underflow before publication.
void rollback_sampling_counts(const ops::SamplingConfig&, std::span<const TokenId> tokens);

} // namespace ninfer::runtime

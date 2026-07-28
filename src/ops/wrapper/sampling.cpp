// ninfer::ops - sample wrapper: public api validation and dispatch.
#include "ninfer/ops/sampling.h"

#include "ops/common/sampling_workspace.h"
#include "ops/launcher/sampling.h" // detail::sample_column_launch

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_columns,
                                              std::int32_t max_columns) {
    if (token_domain <= 0 || min_columns <= 0 || max_columns < min_columns) {
        throw std::invalid_argument("sampling workspace: invalid profile or column interval");
    }
    if (token_domain <= kSamplerTileItems || min_columns > kSamplerMaxColumns) { return 0; }
    return detail::sampling_workspace_exact_bytes(token_domain,
                                                  std::min(max_columns, kSamplerMaxColumns));
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* config, const std::int32_t* pos_base, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("sample: out shape must be [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) {
        throw std::invalid_argument("sample: logits columns must be positive");
    }
    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("sample: logits/out data must be non-null");
    }
    if (config == nullptr) { throw std::invalid_argument("sample: config must be non-null"); }
    if (pos_base == nullptr) { throw std::invalid_argument("sample: pos_base must be non-null"); }

    auto scratch_scope = workspace.scope();
    const std::size_t bytes =
        sampling_workspace_capacity_bytes(token_domain, logits.ne[1], logits.ne[1]);
    const DeviceSpan scratch = bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
    detail::sample_column_launch(logits, out, token_domain, config, pos_base, purpose, scratch,
                                 stream);
}

} // namespace ninfer::ops

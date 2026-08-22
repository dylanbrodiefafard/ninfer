#include "ninfer/ops/nll_from_logits.h"

#include "ops/launcher/nll_from_logits.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

void nll_from_logits(const Tensor& logits, const Tensor& targets, Tensor& out,
                     std::int32_t valid_rows, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("nll_from_logits: logits must be BF16");
    }
    if (targets.dtype != DType::I32) {
        throw std::invalid_argument("nll_from_logits: targets must be I32");
    }
    if (out.dtype != DType::FP32) { throw std::invalid_argument("nll_from_logits: out must be FP32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("nll_from_logits: logits must be rank-2 [vocab,T]");
    }
    if (targets.ne[1] != 1 || targets.ne[2] != 1 || targets.ne[3] != 1) {
        throw std::invalid_argument("nll_from_logits: targets must be rank-1 [T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("nll_from_logits: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0) {
        throw std::invalid_argument("nll_from_logits: logits extents must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("nll_from_logits: valid_rows must be in [1, logits.ne[0]]");
    }
    if (targets.ne[0] != logits.ne[1] || out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("nll_from_logits: targets/out shape must be [logits.ne[1]]");
    }
    if (!logits.is_contiguous() || !targets.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("nll_from_logits: tensors must be contiguous");
    }
    if (logits.data == nullptr || targets.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("nll_from_logits: tensors must be non-null");
    }
    if (out.data == logits.data || out.data == targets.data || logits.data == targets.data) {
        throw std::invalid_argument("nll_from_logits: logits, targets, and out must not alias");
    }

    detail::nll_from_logits_launch(logits, targets, out, valid_rows, stream);
}

} // namespace ninfer::ops

#include "ninfer/ops/token_logprobs.h"

#include "ops/launcher/token_logprobs.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void reject(const char* message) {
    throw std::invalid_argument(std::string("token_logprobs: ") + message);
}

// Rank-2 [W,B] panel whose two pitches are element multiples.
void require_slot_row_panel(const Tensor& tensor, DType dtype, std::int32_t width,
                            std::int32_t batch, const char* message) {
    const auto element = static_cast<std::int64_t>(dtype_size(dtype));
    if (tensor.data == nullptr || tensor.dtype != dtype || tensor.ne[0] != width ||
        tensor.ne[1] != batch || tensor.ne[2] != 1 || tensor.ne[3] != 1 || tensor.nb[0] < element ||
        tensor.nb[0] % element != 0 || tensor.nb[1] < element || tensor.nb[1] % element != 0) {
        reject(message);
    }
}

// Rank-3 [dense,W,B] panel: dense first axis, element-multiple pitches on the outer two.
void require_dense_slot_row_panel(const Tensor& tensor, DType dtype, std::int32_t dense,
                                  std::int32_t width, std::int32_t batch, const char* message) {
    const auto element = static_cast<std::int64_t>(dtype_size(dtype));
    if (tensor.data == nullptr || tensor.dtype != dtype || tensor.ne[0] != dense ||
        tensor.ne[1] != width || tensor.ne[2] != batch || tensor.ne[3] != 1 ||
        tensor.nb[0] != element || tensor.nb[1] < element * dense || tensor.nb[1] % element != 0 ||
        tensor.nb[2] < element || tensor.nb[2] % element != 0) {
        reject(message);
    }
}

void require_row_vector(const Tensor& tensor, std::int32_t batch, const char* message) {
    if (tensor.data == nullptr || tensor.dtype != DType::I32 || tensor.ne[0] != batch ||
        tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous()) {
        reject(message);
    }
}

} // namespace

void token_logprobs(const Tensor& logits, const Tensor& tokens, const Tensor& row_enabled,
                    const Tensor* counts, const Tensor* columns, std::int32_t token_domain,
                    Tensor& token_logprob, Tensor& top_ids, Tensor& top_logprobs,
                    cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t column_count  = logits.ne[1];
    const std::int32_t width         = tokens.ne[0];
    const std::int32_t batch         = tokens.ne[1];
    const std::int32_t top_count     = top_ids.ne[0];

    require_dense_slot_row_panel(logits, DType::BF16, physical_rows, column_count, batch,
                                 "logits must be BF16 [physical_rows,C,B] with a dense first axis");
    require_slot_row_panel(tokens, DType::I32, width, batch, "tokens must be I32 [W,B]");
    require_row_vector(row_enabled, batch, "row_enabled must be contiguous I32 [B]");
    if (counts != nullptr) {
        require_row_vector(*counts, batch, "counts must be contiguous I32 [B]");
    }
    if (columns != nullptr) {
        require_slot_row_panel(*columns, DType::I32, width, batch, "columns must be I32 [W,B]");
    } else if (width > column_count) {
        reject("W must not exceed the logit column count without columns");
    }
    if (top_count < 1 || top_count > kMaximumTopLogprobs) {
        reject("K must be in [1,kMaximumTopLogprobs]");
    }
    if (token_domain < top_count || token_domain > physical_rows) {
        reject("token_domain must be in [K,physical_rows]");
    }
    require_slot_row_panel(token_logprob, DType::FP32, width, batch,
                           "token_logprob must be FP32 [W,B]");
    require_dense_slot_row_panel(top_ids, DType::I32, top_count, width, batch,
                                 "top_ids must be I32 [K,W,B] with a dense first axis");
    require_dense_slot_row_panel(top_logprobs, DType::FP32, top_count, width, batch,
                                 "top_logprobs must be FP32 [K,W,B] with a dense first axis");
    if (token_logprob.data == top_logprobs.data || token_logprob.data == logits.data ||
        top_logprobs.data == logits.data || top_ids.data == tokens.data ||
        top_ids.data == row_enabled.data || (counts != nullptr && top_ids.data == counts->data) ||
        (columns != nullptr && top_ids.data == columns->data)) {
        reject("outputs must not alias inputs or each other");
    }

    detail::token_logprobs_launch(logits, tokens, row_enabled, counts, columns, token_domain,
                                  token_logprob, top_ids, top_logprobs, stream);
}

} // namespace ninfer::ops

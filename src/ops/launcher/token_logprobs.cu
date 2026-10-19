// Implements: include/ninfer/ops/token_logprobs.h
// Match: wrapper-validated operands whose pitches are multiples of their element size.
#include "ops/launcher/token_logprobs.h"

#include "ops/kernel/token_logprobs.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {

// Rank-2 [W,B] panel: axis 0 is the slot, axis 1 the row.
template <class T>
TokenLogprobsPanel<T> slot_row_panel(const Tensor& tensor) {
    return {static_cast<T*>(tensor.data), tensor.nb[0], tensor.nb[1]};
}

// Rank-3 [dense,W,B] panel: axis 1 is the slot (or logit column), axis 2 the row.
template <class T>
TokenLogprobsPanel<T> dense_slot_row_panel(const Tensor& tensor) {
    return {static_cast<T*>(tensor.data), tensor.nb[1], tensor.nb[2]};
}

} // namespace

void token_logprobs_launch(const Tensor& logits, const Tensor& tokens, const Tensor& row_enabled,
                           const Tensor* counts, const Tensor* columns, std::int32_t token_domain,
                           Tensor& token_logprob, Tensor& top_ids, Tensor& top_logprobs,
                           cudaStream_t stream) {
    TokenLogprobsArgs args;
    args.logits      = dense_slot_row_panel<const __nv_bfloat16>(logits);
    args.tokens      = slot_row_panel<const std::int32_t>(tokens);
    args.row_enabled = static_cast<const std::int32_t*>(row_enabled.data);
    if (counts != nullptr) { args.counts = static_cast<const std::int32_t*>(counts->data); }
    if (columns != nullptr) { args.columns = slot_row_panel<const std::int32_t>(*columns); }
    args.token_logprob = slot_row_panel<float>(token_logprob);
    args.top_ids       = dense_slot_row_panel<std::int32_t>(top_ids);
    args.top_logprobs  = dense_slot_row_panel<float>(top_logprobs);
    args.token_domain  = token_domain;
    args.top_count     = top_ids.ne[0];

    const dim3 grid(static_cast<unsigned int>(tokens.ne[0]),
                    static_cast<unsigned int>(tokens.ne[1]));
    token_logprobs_kernel<<<grid, kTokenLogprobsBlock, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

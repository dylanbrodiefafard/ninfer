#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] != 1) { throw std::invalid_argument("bf16 attn_input_proj: unsupported T"); }
    bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail

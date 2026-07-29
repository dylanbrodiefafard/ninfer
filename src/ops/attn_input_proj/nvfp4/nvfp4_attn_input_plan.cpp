#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] >= kNvfp4FirstSmallT && x.ne[1] <= kNvfp4LastSmallT) {
        nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    throw std::invalid_argument("nvfp4 attn_input_proj: unsupported T");
}

} // namespace ninfer::ops::detail

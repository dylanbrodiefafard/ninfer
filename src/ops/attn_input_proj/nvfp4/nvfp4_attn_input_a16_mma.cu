#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

// The W4A8 MMA family with BF16 activations: exact E2M1 x E4M3 weight expansion feeds BF16 MMAs
// that accumulate each output in ascending K order in FP32. Its wide path (N > 6144) stores whole
// eight-row vectors from 64- or 128-row tiles, so every vector lies inside one of the four ranges.
void nvfp4_attn_input_a16_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Nvfp4FlashNextAttnInputGeometry;
    using Output   = Nvfp4AttentionInput2560Output;
    static_assert(Geometry::kOutputRows == Output::kParentRows);
    if (weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows || x.ne[1] < 2) {
        throw std::invalid_argument("nvfp4 attn_input_proj A16 MMA: invalid exact problem");
    }
    launch_nvfp4_w4a8_mma<Geometry>(
        weight, x.ne[1], Nvfp4Bf16Activation{static_cast<const __nv_bfloat16*>(x.data)},
        Nvfp4IdentityEpilogue{},
        Output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
               static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        stream);
}

} // namespace ninfer::ops::detail

#include "ops/launcher/gated_residual_stem.h"
#include "core/device.h"
#include <cuda_bf16.h>
namespace ninfer::ops::detail {
namespace {
__global__ void add(const __nv_bfloat16* e, const __nv_bfloat16* h,
                    __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int ei = (i / 10240) * 2560 + i % 2560;
        out[i] = __float2bfloat16_rn(__bfloat162float(h[i]) + __bfloat162float(e[ei]));
    }
}
}
void gated_residual_stem_add_launch(const Tensor& e, const Tensor& h, Tensor& out,
                                    cudaStream_t stream) {
    const int count = 10240 * out.ne[2];
    add<<<(count+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(e.data),
        static_cast<const __nv_bfloat16*>(h.data), static_cast<__nv_bfloat16*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}
}

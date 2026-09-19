#include "ops/launcher/gated_residual_stem.h"
#include "core/device.h"
#include "ops/common/warp.cuh"
#include <cuda_bf16.h>
namespace ninfer::ops::detail {
namespace {
template<int Width>
__global__ void normalize(const __nv_bfloat16* input,const __nv_bfloat16* weight,float* output) {
    const int base=blockIdx.x*Width;
    float sum=0;
    for(int k=threadIdx.x;k<Width;k+=256) {const float x=__bfloat162float(input[base+k]);sum=fmaf(x,x,sum);}
    __shared__ float partial[8];
    __shared__ float inverse;
    sum=block_reduce_sum<256>(sum,partial);
    if(threadIdx.x==0) inverse=rsqrtf(sum/float(Width)+1e-6f);
    __syncthreads();
    for(int k=threadIdx.x;k<Width;k+=256)
        output[base+k]=__bfloat162float(input[base+k])*inverse*(1.f+__bfloat162float(weight[k]));
}
template<int Tokens>
__global__ void project(const float* input,const __nv_bfloat16* weight,__nv_bfloat16* output,int width) {
    const int row=blockIdx.x*4+threadIdx.x/32,token=blockIdx.y*Tokens,lane=threadIdx.x%32;
    float sum[Tokens]{};
    for(int k=lane;k<2560;k+=32) {
        const float w=__bfloat162float(weight[row*2560+k]);
#pragma unroll
        for(int t=0;t<Tokens;++t) if(token+t<width) sum[t]=fmaf(input[(token+t)*2560+k],w,sum[t]);
    }
#pragma unroll
    for(int t=0;t<Tokens;++t) {
        sum[t]=warp_reduce_sum(sum[t]);
        if(lane==0 && token+t<width) output[(token+t)*2560+row]=__float2bfloat16_rn(sum[t]);
    }
}
__global__ void add(const __nv_bfloat16* e, const __nv_bfloat16* h,
                    __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int ei = (i / 10240) * 2560 + i % 2560;
        out[i] = __float2bfloat16_rn(__bfloat162float(h[i]) + __bfloat162float(e[ei]));
    }
}
}
void gated_residual_stem_normalize_launch(const Tensor& input,const Tensor& weight,
                                          Tensor& out,cudaStream_t stream) {
    if(input.ne[0]==2560) normalize<2560><<<input.ne[1],256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),static_cast<const __nv_bfloat16*>(weight.data),static_cast<float*>(out.data));
    else normalize<10240><<<input.ne[1],256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),static_cast<const __nv_bfloat16*>(weight.data),static_cast<float*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}
void gated_residual_stem_project_launch(const Tensor& input,const Weight& weight,Tensor& out,cudaStream_t stream) {
    project<4><<<dim3(640,(input.ne[1]+3)/4),128,0,stream>>>(static_cast<const float*>(input.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),static_cast<__nv_bfloat16*>(out.data),input.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
void gated_residual_stem_add_launch(const Tensor& e, const Tensor& h, Tensor& out,
                                    cudaStream_t stream) {
    const int count = 10240 * out.ne[2];
    add<<<(count+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(e.data),
        static_cast<const __nv_bfloat16*>(h.data), static_cast<__nv_bfloat16*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}
}

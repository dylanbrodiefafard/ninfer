#include "ops/launcher/qsa_paged.h"
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include <cuda_bf16.h>
#include <cfloat>
#include <climits>

namespace ninfer::ops::detail {
namespace {
struct Pages {
    unsigned char *k,*v,*ks,*vs;
    __nv_bfloat16* raw;
    int *positions;
    const int* tables;
    int logical;
};
struct Controls { const int *rows,*valid,*frontier,*positions; int width; };
Pages pages(const QsaPagedStateView& s) {
    return {static_cast<unsigned char*>(s.k.data),static_cast<unsigned char*>(s.v.data),
        static_cast<unsigned char*>(s.k_scales.data),static_cast<unsigned char*>(s.v_scales.data),
        static_cast<__nv_bfloat16*>(s.raw_index_keys.data),static_cast<int*>(s.positions.data),
        static_cast<const int*>(s.block_tables.data),s.block_tables.ne[0]};
}
Controls controls(const QsaBatchControls& c) {
    return {static_cast<const int*>(c.table_rows.data),static_cast<const int*>(c.valid_columns.data),
        static_cast<const int*>(c.frontiers.data),static_cast<const int*>(c.positions.data),
        c.positions.ne[1]};
}
__device__ int physical(Pages p,Controls c,int batch,int id) {
    return p.tables[(id>>6)+p.logical*c.rows[batch]];
}
__device__ float rotate(float lo,float hi,int pair,bool upper,const int* position) {
    // Ordinals extend to 262144: forming the angle in FP32 can move top-k rank
    // boundaries even while vector error remains small. Keep phase/trig in FP64.
    double sn,cs;
    sincos(static_cast<double>(position[pair%3])*pow(1.e7,-double(pair)/32.),&sn,&cs);
    return static_cast<float>(upper?hi*cs+lo*sn:lo*cs-hi*sn);
}
template<bool Bf16>
__device__ float read(const unsigned char* data,const unsigned char* scale,int d,
                      int page,int offset,int head) {
    const long long row=offset+64LL*(head+2LL*page);
    if constexpr(Bf16) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(data)[d+256LL*row]);
    const float2 pair=decode_nvfp4_e2m1x2(data[(d>>1)+128LL*row]);
    return ((d&1)?pair.y:pair.x)*decode_nvfp4_e4m3(scale[(d>>4)+16LL*row]);
}
template<bool Bf16>
__global__ void append(const __nv_bfloat16* k,const __nv_bfloat16* v,
    const __nv_bfloat16* raw,Pages p,Controls c) {
    const int token=blockIdx.x,b=token/c.width,j=token%c.width,d=threadIdx.x;
    if(j>=c.valid[b]) return;
    __shared__ int page;
    const int id=c.frontier[b]+j;
    if(d==0) page=physical(p,c,b,id);
    __syncthreads();
    for(int head=0;head<2;++head) {
        const long long src=256LL*(head+2LL*token);
        const long long row=(id&63)+64LL*(head+2LL*page);
        if constexpr(Bf16) {
            reinterpret_cast<__nv_bfloat16*>(p.k)[d+256LL*row]=k[d+src];
            reinterpret_cast<__nv_bfloat16*>(p.v)[d+256LL*row]=v[d+src];
        } else if(d<16) {
            const auto kq=quantize_nvfp4_k16(k+src+16*d,1.F);
            const auto vq=quantize_nvfp4_k16(v+src+16*d,1.F);
            auto* kd=reinterpret_cast<unsigned int*>(p.k+128LL*row+8*d);
            auto* vd=reinterpret_cast<unsigned int*>(p.v+128LL*row+8*d);
            kd[0]=kq.codes_lo;kd[1]=kq.codes_hi;
            vd[0]=vq.codes_lo;vd[1]=vq.codes_hi;
            p.ks[d+16LL*row]=kq.scale;p.vs[d+16LL*row]=vq.scale;
        }
    }
    const long long meta=(id&63)+64LL*page;
    if(d<128) p.raw[d+128LL*meta]=raw[d+128LL*token];
    if(d<3) p.positions[d+3LL*meta]=c.positions[d+3LL*token];
}

// Per query: rotated query[512], old/new scores[1024], old ranks[512].
__global__ void prepare(const __nv_bfloat16* q,const float* gamma,Controls c,
    int* selected,int* counts,float* scratch) {
    const int t=blockIdx.x,d=threadIdx.x,b=t/c.width,j=t%c.width;
    float* row=scratch+static_cast<long long>(t)*kQsaPagedSelectorWords;
    for(int i=d;i<2051;i+=128) selected[i+2051LL*t]=-1;
    for(int i=d;i<512;i+=128) { row[512+i]=-FLT_MAX;reinterpret_cast<int*>(row+1536)[i]=INT_MAX; }
    if(d==0) counts[t]=0;
    if(j>=c.valid[b]) return;
    __shared__ float sq[128], norm[128];
    for(int head=0;head<4;++head) {
        const float value=__bfloat162float(q[d+128LL*(head+4LL*t)]);
        sq[d]=value*value;
        __syncthreads();
        for(int stride=64;stride;stride>>=1) {
            if(d<stride) sq[d]+=sq[d+stride];
            __syncthreads();
        }
        norm[d]=value*rsqrtf(sq[0]/128.F+1.e-6F)*gamma[d];
        __syncthreads();
        float result=norm[d];
        if(d<64) result=rotate(norm[d&31],norm[(d&31)+32],d&31,d>=32,c.positions+3LL*t);
        row[d+128*head]=result;
        __syncthreads();
    }
}
__global__ void score(Pages p,Controls c,const float* gamma,float* scratch,int first) {
    const int rank=first+blockIdx.x,t=blockIdx.y,d=threadIdx.x,b=t/c.width,j=t%c.width;
    float* row=scratch+static_cast<long long>(t)*kQsaPagedSelectorWords;
    if(j>=c.valid[b] || rank>=(c.frontier[b]+j+1)/4) {
        if(d==0) row[1024+blockIdx.x]=-FLT_MAX;
        return;
    }
    __shared__ float pooled[128],reduction[128],key[128];
    __shared__ int page;
    if(d==0) page=physical(p,c,b,4*rank);
    __syncthreads();
    // A rank-four block cannot cross a P=64 page boundary.
    const long long base=(4*rank&63)+64LL*page;
    float value=0;
    for(int r=0;r<4;++r) value+=__bfloat162float(p.raw[d+128LL*(base+r)]);
    value=__bfloat162float(__float2bfloat16_rn(value*.25F));
    pooled[d]=value;reduction[d]=value*value;
    __syncthreads();
    for(int stride=64;stride;stride>>=1) {
        if(d<stride) reduction[d]+=reduction[d+stride];
        __syncthreads();
    }
    pooled[d]=value*rsqrtf(reduction[0]/128.F+1.e-6F)*gamma[d];
    __syncthreads();
    value=pooled[d];
    if(d<64) value=rotate(pooled[d&31],pooled[(d&31)+32],d&31,d>=32,p.positions+3LL*base);
    key[d]=value;
    __syncthreads();
    float result=0;
    for(int h=0;h<4;++h) {
        reduction[d]=key[d]*row[d+128*h];
        __syncthreads();
        for(int stride=64;stride;stride>>=1) {
            if(d<stride) reduction[d]+=reduction[d+stride];
            __syncthreads();
        }
        result+=fmaxf(0.F,reduction[0]);
        __syncthreads();
    }
    if(d==0) row[1024+blockIdx.x]=result*rsqrtf(128.F);
}
__device__ bool better(float a,int ai,float b,int bi) {
    return a>b || (a==b && ai<bi);
}
__global__ void merge(Controls c,float* scratch,int first) {
    const int t=blockIdx.x,d=threadIdx.x,b=t/c.width,j=t%c.width;
    if(j>=c.valid[b]) return;
    float* row=scratch+static_cast<long long>(t)*kQsaPagedSelectorWords;
    __shared__ float scores[1024];
    __shared__ int ranks[1024];
    const int complete=(c.frontier[b]+j+1)/4;
    scores[d]=row[512+d];ranks[d]=reinterpret_cast<int*>(row+1536)[d];
    scores[d+512]=row[1024+d];ranks[d+512]=first+d<complete?first+d:INT_MAX;
    __syncthreads();
    for(int size=2;size<=1024;size<<=1) for(int stride=size/2;stride;stride>>=1) {
        const int left=(d/stride)*(2*stride)+d%stride,right=left+stride;
        const bool desc=(left&size)==0;
        if(desc?better(scores[right],ranks[right],scores[left],ranks[left]):
                better(scores[left],ranks[left],scores[right],ranks[right])) {
            const float f=scores[left];scores[left]=scores[right];scores[right]=f;
            const int i=ranks[left];ranks[left]=ranks[right];ranks[right]=i;
        }
        __syncthreads();
    }
    row[512+d]=scores[d];reinterpret_cast<int*>(row+1536)[d]=ranks[d];
}
__global__ void finish(Controls c,const float* scratch,int* selected,int* counts) {
    const int t=blockIdx.x,d=threadIdx.x,b=t/c.width,j=t%c.width;
    if(j>=c.valid[b]) return;
    const int visible=c.frontier[b]+j+1,complete=visible/4,kept=min(complete,512);
    const int* ranks=reinterpret_cast<const int*>(scratch+static_cast<long long>(t)*kQsaPagedSelectorWords+1536);
    if(d<kept) for(int r=0;r<4;++r) selected[4*d+r+2051LL*t]=4*ranks[d]+r;
    if(d==0) {
        int n=4*kept;
        for(int id=4*complete;id<visible;++id) { selected[n+2051LL*t]=id;++n; }
        counts[t]=n;
    }
}

template<bool Bf16>
__global__ void attention(const __nv_bfloat16* q,const int* ids,const int* counts,
    Pages p,Controls c,__nv_bfloat16* out,int bound) {
    const int d=threadIdx.x,lane=d&31,warp=d>>5,head=blockIdx.x,t=blockIdx.y;
    const int b=t/c.width,j=t%c.width,count=counts[t];
    const long long output=d+256LL*(head+24LL*t);
    if(j>=c.valid[b] || count<=0) { out[output]=__float2bfloat16_rn(0);return; }
    __shared__ float scores[2051],reduce[256];
    __shared__ int locations[2051];
    // Resolve each selected logical position once per CTA; all feature lanes reuse it.
    for(int r=d;r<count;r+=256) {
        const int id=ids[r+static_cast<long long>(bound)*t];
        locations[r]=(id&63)+64*physical(p,c,b,id);
    }
    __syncthreads();
    const int kv=head/12;
    for(int r=warp;r<count;r+=8) {
        const int location=locations[r];
        float dot=0;
        for(int f=lane;f<256;f+=32)
            dot+=__bfloat162float(q[f+256LL*(head+24LL*t)])*
                read<Bf16>(p.k,p.ks,f,location>>6,location&63,kv);
        for(int delta=16;delta;delta>>=1) dot+=__shfl_down_sync(0xffffffffU,dot,delta);
        if(lane==0) scores[r]=dot*.0625F;
    }
    __syncthreads();
    float maximum=-FLT_MAX;
    for(int r=d;r<count;r+=256) maximum=fmaxf(maximum,scores[r]);
    reduce[d]=maximum;__syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(d<stride) reduce[d]=fmaxf(reduce[d],reduce[d+stride]);
        __syncthreads();
    }
    maximum=reduce[0];
    __syncthreads();
    float sum=0;
    for(int r=d;r<count;r+=256) { scores[r]=expf(scores[r]-maximum);sum+=scores[r]; }
    reduce[d]=sum;__syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(d<stride) reduce[d]+=reduce[d+stride];
        __syncthreads();
    }
    const float inverse=1.F/reduce[0];float result=0;
    for(int r=0;r<count;++r) {
        const int location=locations[r];
        result=fmaf(scores[r]*inverse,read<Bf16>(p.v,p.vs,d,location>>6,location&63,kv),result);
    }
    out[output]=__float2bfloat16_rn(result);
}
__global__ void mask(__nv_bfloat16* out,Controls c,int rows,int tokens) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*tokens) return;
    const int t=i/rows;
    if(t%c.width>=c.valid[t/c.width]) out[i]=__float2bfloat16_rn(0.F);
}
}
void qsa_paged_append_launch(const Tensor& k,const Tensor& v,const Tensor& raw,
    const QsaBatchControls& c,QsaPagedStateView s,cudaStream_t stream) {
    const auto* kp=static_cast<const __nv_bfloat16*>(k.data);
    const auto* vp=static_cast<const __nv_bfloat16*>(v.data);
    const auto* rp=static_cast<const __nv_bfloat16*>(raw.data);
    if(s.format==QsaKvFormat::BF16) append<true><<<k.ne[2]*k.ne[3],256,0,stream>>>(kp,vp,rp,pages(s),controls(c));
    else append<false><<<k.ne[2]*k.ne[3],256,0,stream>>>(kp,vp,rp,pages(s),controls(c));
    CUDA_CHECK(cudaGetLastError());
}
void qsa_paged_select_launch(const Tensor& q,const QsaPagedStateView& s,
    const QsaBatchControls& c,int maximum,const Tensor& qw,const Tensor& kw,
    Tensor& ids,Tensor& counts,Tensor& scratch,cudaStream_t stream) {
    const int tokens=q.ne[2]*q.ne[3];auto* work=static_cast<float*>(scratch.data);
    auto* id=static_cast<int*>(ids.data);auto* count=static_cast<int*>(counts.data);
    prepare<<<tokens,128,0,stream>>>(static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const float*>(qw.data),controls(c),id,count,work);
    CUDA_CHECK(cudaGetLastError());
    for(int first=0;first<maximum/4;first+=512) {
        score<<<dim3(512,tokens),128,0,stream>>>(pages(s),controls(c),static_cast<const float*>(kw.data),work,first);
        CUDA_CHECK(cudaGetLastError());
        merge<<<tokens,512,0,stream>>>(controls(c),work,first);
        CUDA_CHECK(cudaGetLastError());
    }
    finish<<<tokens,512,0,stream>>>(controls(c),work,id,count);
    CUDA_CHECK(cudaGetLastError());
}
void qsa_paged_attention_launch(const Tensor& q,const Tensor& ids,const Tensor& counts,
    const QsaPagedStateView& s,const QsaBatchControls& c,Tensor& out,cudaStream_t stream) {
    const dim3 grid(24,q.ne[2]*q.ne[3]);
    const auto* qp=static_cast<const __nv_bfloat16*>(q.data);
    const auto* ip=static_cast<const int*>(ids.data);const auto* cp=static_cast<const int*>(counts.data);
    auto* op=static_cast<__nv_bfloat16*>(out.data);
    if(s.format==QsaKvFormat::BF16) attention<true><<<grid,256,0,stream>>>(qp,ip,cp,pages(s),controls(c),op,ids.ne[0]);
    else attention<false><<<grid,256,0,stream>>>(qp,ip,cp,pages(s),controls(c),op,ids.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
void qsa_paged_mask_launch(Tensor& out,const QsaBatchControls& c,cudaStream_t stream) {
    const int tokens=out.ne[1]*out.ne[2];
    mask<<<(out.ne[0]*tokens+255)/256,256,0,stream>>>(static_cast<__nv_bfloat16*>(out.data),
        controls(c),out.ne[0],tokens);
    CUDA_CHECK(cudaGetLastError());
}
}

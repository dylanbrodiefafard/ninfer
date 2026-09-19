#include "targets/qwen4/vision.h"

#include "artifact/binder.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ninfer::targets::qwen4 {
namespace {
std::size_t storage(int width,int patches) {
    if(patches<=0 || patches%4 || patches>std::numeric_limits<int>::max()/4608)
        throw std::invalid_argument("Qwen4 Vision requires a positive multiple-of-four patch capacity");
    return std::size_t(width)*patches*2;
}
std::size_t scratch_bytes(int patches,int segments) {
    if(segments<=0 || segments>patches/4)
        throw std::invalid_argument("Qwen4 Vision invalid segment capacity");
    const std::size_t bytes=ops::vision_attention_workspace_capacity_bytes(4,patches,1,segments);
    return std::max<std::size_t>(bytes,256);
}
int checked_segments(int patches,int segments) {
    if(segments<=0 || patches<=0 || segments>patches/4)
        throw std::invalid_argument("Qwen4 Vision invalid startup segment capacity");
    return segments;
}
Weight matrix(void* data,int n,int k) {
    Weight w{};
    w.payload=w.qdata=data; w.payload_bytes=std::size_t(n)*k*2;
    w.qtype=QType::BF16_CTRL; w.layout=QuantLayout::Contiguous;
    w.n=w.shape[0]=w.padded_shape[0]=n; w.k=w.shape[1]=w.padded_shape[1]=k;
    w.ndim=2; return w;
}
Tensor bf16(DeviceBuffer& b,int width,int columns) { return Tensor(b.p,DType::BF16,{width,columns}); }
}

VisionControl prepare_vision_control(std::span<const VisionGrid> grids) {
    if(grids.empty()) throw std::invalid_argument("Qwen4 Vision empty grid list");
    std::int64_t count=0;
    for(auto grid:grids) {
        if(grid.temporal<=0 || grid.height<=0 || grid.width<=0 || grid.height%2 || grid.width%2)
            throw std::invalid_argument("Qwen4 Vision grids require positive T and even H/W");
        const auto plane=std::int64_t(grid.height)*grid.width;
        if(plane>std::numeric_limits<int>::max()/4608 || grid.temporal>
           (std::numeric_limits<int>::max()/4608-count)/plane)
            throw std::overflow_error("Qwen4 Vision patch count exceeds supported tensor extent");
        count+=plane*grid.temporal;
    }
    VisionControl c; c.patches=static_cast<int>(count);
    c.positions.resize(2*count); c.position_indices.resize(4*count); c.position_weights.resize(4*count);
    c.segments.push_back(0); int p=0;
    for(auto grid:grids) for(int t=0;t<grid.temporal;++t) {
        for(int by=0;by<grid.height;by+=2) for(int bx=0;bx<grid.width;bx+=2)
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx,++p) {
                const int y=by+dy,x=bx+dx;
                c.positions[p]=y; c.positions[count+p]=x;
                const float fy=float(y)*47.0F/float(grid.height-1);
                const float fx=float(x)*47.0F/float(grid.width-1);
                const int iy=int(std::floor(fy)),ix=int(std::floor(fx));
                const float wy=fy-iy,wx=fx-ix;
                for(int cy=0;cy<2;++cy) for(int cx=0;cx<2;++cx) {
                    const int i=4*p+2*cy+cx;
                    c.position_indices[i]=std::min(47,iy+cy)*48+std::min(47,ix+cx);
                    c.position_weights[i]=(cy?wy:1.0F-wy)*(cx?wx:1.0F-wx);
                }
            }
        c.segments.push_back(p);
    }
    return c;
}

std::unique_ptr<LoadedVision> LoadedVision::load(const std::filesystem::path& path,DeviceContext& device) {
    artifact::Reader reader(path);
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-vision-qualification","nvidia-bf16-source"})
        throw std::invalid_argument("Qwen4 Vision requires the exact native qualification artifact");
    artifact::Binder binder(reader);
    std::map<std::string,artifact::ObjectHandle> handles;
    auto bind=[&](const std::string& name,std::initializer_list<std::uint64_t> shape) {
        auto h=binder.require_tensor("model.visual."+name,artifact::NumericFormat::BF16,
            artifact::StorageLayout::ContiguousLeV1,std::span(shape));
        binder.materialize_on_device(h); handles.emplace(name,h);
    };
    bind("patch_embed.proj.weight",{1152,3,2,16,16}); bind("patch_embed.proj.bias",{1152});
    bind("pos_embed.weight",{2304,1152});
    for(int layer=0;layer<27;++layer) {
        const auto p="blocks."+std::to_string(layer)+".";
        for(const char* name:{"norm1.weight","norm1.bias","norm2.weight","norm2.bias"}) bind(p+name,{1152});
        for(auto [name,n,k]:std::array<std::tuple<const char*,int,int>,4>{{{"attn.qkv",3456,1152},
                {"attn.proj",1152,1152},{"mlp.linear_fc1",4304,1152},{"mlp.linear_fc2",1152,4304}}}) {
            bind(p+name+".weight",{std::uint64_t(n),std::uint64_t(k)});
            bind(p+name+".bias",{std::uint64_t(n)});
        }
    }
    bind("merger.norm.weight",{1152}); bind("merger.norm.bias",{1152});
    bind("merger.linear_fc1.weight",{4608,4608}); bind("merger.linear_fc1.bias",{4608});
    bind("merger.linear_fc2.weight",{2560,4608}); bind("merger.linear_fc2.bias",{2560});
    auto result=std::unique_ptr<LoadedVision>(new LoadedVision);
    result->backing_=artifact::materialize(reader,binder.finish(),device);
    auto ptr=[&](const std::string& name) { return result->backing_.device_data(handles.at(name)); };
    auto tensor=[&](const std::string& name,int width) { return Tensor(ptr(name),DType::BF16,{width}); };
    auto& w=result->weights_;
    w.patch=matrix(ptr("patch_embed.proj.weight"),1152,1536);
    w.patch_bias=tensor("patch_embed.proj.bias",1152);
    w.positions=Tensor(ptr("pos_embed.weight"),DType::BF16,{1152,2304});
    for(int layer=0;layer<27;++layer) {
        const auto p="blocks."+std::to_string(layer)+"."; auto& b=w.blocks[layer];
        b.norm1_weight=tensor(p+"norm1.weight",1152); b.norm1_bias=tensor(p+"norm1.bias",1152);
        b.norm2_weight=tensor(p+"norm2.weight",1152); b.norm2_bias=tensor(p+"norm2.bias",1152);
        b.qkv=matrix(ptr(p+"attn.qkv.weight"),3456,1152); b.qkv_bias=tensor(p+"attn.qkv.bias",3456);
        b.output=matrix(ptr(p+"attn.proj.weight"),1152,1152); b.output_bias=tensor(p+"attn.proj.bias",1152);
        b.up=matrix(ptr(p+"mlp.linear_fc1.weight"),4304,1152); b.up_bias=tensor(p+"mlp.linear_fc1.bias",4304);
        b.down=matrix(ptr(p+"mlp.linear_fc2.weight"),1152,4304); b.down_bias=tensor(p+"mlp.linear_fc2.bias",1152);
    }
    w.merger={tensor("merger.norm.weight",1152),tensor("merger.norm.bias",1152),
        matrix(ptr("merger.linear_fc1.weight"),4608,4608),tensor("merger.linear_fc1.bias",4608),
        matrix(ptr("merger.linear_fc2.weight"),2560,4608),tensor("merger.linear_fc2.bias",2560)};
    return result;
}

struct VisionProgram::Storage {
    std::size_t hidden,qkv,up,output,positions,segments,indices,weights,merger,pinned,scratch;
    Storage(int p,int s):hidden(storage(1152,p)),qkv(storage(3456,p)),up(storage(4304,p)),
        output(storage(640,p)),positions(std::size_t(p)*8),
        segments((std::size_t(checked_segments(p,s))+1)*4),indices(std::size_t(p)*16),weights(indices),
        merger(ops::vision_patch_merger_workspace_bytes(p/4)),
        pinned(positions+segments+indices+weights),scratch(scratch_bytes(p,s)) {}
    std::size_t device() const {return 4*hidden+qkv+up+output+positions+segments+indices+weights+merger+scratch;}
};

std::size_t VisionProgram::device_bytes(int patches,int segments) {return Storage(patches,segments).device();}
std::size_t VisionProgram::pinned_bytes(int patches,int segments) {return Storage(patches,segments).pinned;}
VisionProgram::VisionProgram(const VisionWeights& weights,int patches,int segments)
    :VisionProgram(weights,patches,segments,Storage(patches,segments)) {}
VisionProgram::VisionProgram(const VisionWeights& weights,int patches,int segments,const Storage& s)
    :weights_(weights),max_patches_(patches),max_segments_(segments),
     residual_(s.hidden),norm_(s.hidden),qkv_(s.qkv),attended_(s.hidden),projection_(s.hidden),up_(s.up),
     output_(s.output),positions_(s.positions),segment_offsets_(s.segments),position_indices_(s.indices),
     position_weights_(s.weights),merger_workspace_(s.merger),control_staging_(s.pinned),workspace_(s.scratch) {}

void VisionProgram::configure(const VisionControl& c,cudaStream_t stream) {
    if(c.patches<=0 || c.patches%4 || c.patches>max_patches_ || c.segments.size()<2 ||
       c.segments.size()>std::size_t(max_segments_)+1 || c.positions.size()!=std::size_t(c.patches)*2 ||
       c.position_indices.size()!=std::size_t(c.patches)*4 || c.position_weights.size()!=std::size_t(c.patches)*4 ||
       c.segments.front()!=0 || c.segments.back()!=c.patches)
        throw std::invalid_argument("Qwen4 Vision control does not fit startup capacity");
    for(std::size_t i=1;i<c.segments.size();++i)
        if(c.segments[i]<=c.segments[i-1] || c.segments[i]%4)
            throw std::invalid_argument("Qwen4 Vision invalid merge-aligned segment");
    for(int p:c.positions) if(p<0) throw std::invalid_argument("Qwen4 Vision negative position");
    for(int i:c.position_indices) if(i<0 || i>=2304) throw std::invalid_argument("Qwen4 Vision invalid position index");
    for(float w:c.position_weights) if(!std::isfinite(w) || w<0 || w>1) throw std::invalid_argument("Qwen4 Vision invalid position weight");
    auto* staging=static_cast<std::byte*>(control_staging_.data());
    auto upload=[&](DeviceBuffer& buffer,const void* source,std::size_t bytes) {
        std::memcpy(staging,source,bytes);
        CUDA_CHECK(cudaMemcpyAsync(buffer.p,staging,bytes,cudaMemcpyHostToDevice,stream));
        staging+=bytes;
    };
    upload(positions_,c.positions.data(),c.positions.size()*4);
    upload(segment_offsets_,c.segments.data(),c.segments.size()*4);
    upload(position_indices_,c.position_indices.data(),c.position_indices.size()*4);
    upload(position_weights_,c.position_weights.data(),c.position_weights.size()*4);
    patches_=c.patches; segments_=static_cast<int>(c.segments.size())-1;
}

Tensor VisionProgram::encoder_output() const { return Tensor(residual_.p,DType::BF16,{1152,patches_}); }

Tensor VisionProgram::execute(const Tensor& patches,cudaStream_t stream,const VisionTrace* trace) {
    if(patches_<=0 || patches.dtype!=DType::BF16 || !patches.data || !patches.is_contiguous() ||
       patches.ne[0]!=1536 || patches.ne[1]!=patches_ || patches.ne[2]!=1 || patches.ne[3]!=1)
        throw std::invalid_argument("Qwen4 Vision requires configured contiguous BF16 packed patches");
    Tensor x=bf16(residual_,1152,patches_),n=bf16(norm_,1152,patches_),qkv=bf16(qkv_,3456,patches_),
        a=bf16(attended_,1152,patches_),y=bf16(projection_,1152,patches_),up=bf16(up_,4304,patches_);
    Tensor pos(positions_.p,DType::I32,{patches_,2}),segments(segment_offsets_.p,DType::I32,{segments_+1}),
        indices(position_indices_.p,DType::I32,{4,patches_}),weights(position_weights_.p,DType::FP32,{4,patches_});
    auto linear=[&](const Tensor& input,const Weight& w,const Tensor& bias,Tensor& out) {
        ops::linear_bias(input,w,bias,out,stream);
    };
    linear(patches,weights_.patch,weights_.patch_bias,x);
    ops::vision_pos_embed_add(weights_.positions,indices,weights,x,stream);
    if(trace && trace->capture) trace->capture(trace->context,-1,x,stream);
    int layer=0;
    for(const auto& b:weights_.blocks) {
        ops::layer_norm(x,b.norm1_weight,b.norm1_bias,1e-6F,n,stream);
        linear(n,b.qkv,b.qkv_bias,qkv);
        Tensor q(qkv.data,DType::BF16,{72,16,patches_}),
            k(static_cast<std::byte*>(qkv.data)+1152*2,DType::BF16,{72,16,patches_}),
            v(static_cast<std::byte*>(qkv.data)+2304*2,DType::BF16,{72,16,patches_});
        q.nb[2]=k.nb[2]=v.nb[2]=qkv.nb[1];
        ops::rope(pos,72,10000.F,q,k,stream);
        Tensor heads=a.reshape({72,16,patches_});
        ops::vision_attention(q,k,v,segments,workspace_,heads,stream);
        linear(a,b.output,b.output_bias,y); ops::residual_add(y,x,stream);
        ops::layer_norm(x,b.norm2_weight,b.norm2_bias,1e-6F,n,stream);
        linear(n,b.up,b.up_bias,up); ops::gelu(up,ops::GeluMode::Tanh,stream);
        linear(up,b.down,b.down_bias,y); ops::residual_add(y,x,stream);
        if(trace && trace->capture) trace->capture(trace->context,layer,x,stream);
        ++layer;
    }
    Tensor groups=x.reshape({1152,4,patches_/4}),out=bf16(output_,2560,patches_/4),
        merger_workspace(merger_workspace_.p,DType::U8,{static_cast<int>(merger_workspace_.bytes)});
    ops::vision_patch_merger(groups,weights_.merger,out,merger_workspace,stream);
    return out;
}
} // namespace ninfer::targets::qwen4

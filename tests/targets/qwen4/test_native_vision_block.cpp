#include "artifact/reader.h"
#include "ops/direct_bf16_weight.h"
#include "targets/qwen4/vision_oracles.h"
#include "targets/qwen4/native_sequence_components.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/vision_attention.h"
#include "ninfer/ops/vision_pos_embed.h"
#include <cstdlib>
#include <map>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using direct_bf16_weight::HostWeight;
namespace {
constexpr int D=1152;
// Declared before experiments. Linear retains the existing BF16 A16 criterion.
constexpr ReductionCriterion linear_gate{1.0/256,1.0/256,2.0/256};
constexpr ReductionCriterion chain_gate{.02,.005,.02};
int failures=0;
std::vector<double> wide(const std::vector<float>& x) { return {x.begin(),x.end()}; }
std::vector<float> narrow(const std::vector<double>& x) { return {x.begin(),x.end()}; }
std::vector<double> matrix(const HostWeight& w,const std::vector<double>& x,int t) {
    std::vector<double> y(w.n*t);
    for(int p=0;p<t;++p) for(int n=0;n<w.n;++n)
        y[p*w.n+n]=direct_bf16_weight::dot_fp64_values<double>(w,n,
            std::span<const double>(x.data()+p*w.k,w.k));
    return y;
}
std::vector<float> gpu_matrix(const HostWeight& w,const std::vector<float>& x,int t) {
    direct_bf16_weight::DeviceWeight weight(w);
    auto dx=to_device_bf16(x);
    GuardedDeviceBuffer dy(w.n*t*2);
    WorkspaceArena arena(std::max<std::size_t>(256,ops::linear_workspace_capacity_bytes(QType::BF16_CTRL,w.n,w.k,
        ops::LinearPolicy::A16Only,1,t)));
    Tensor tx(dx.p,DType::BF16,{w.k,t}), ty(dy.data(),DType::BF16,{w.n,t});
    ops::linear(tx,weight.view(),ty,ops::LinearPolicy::A16Only,arena,nullptr);
    auto actual=from_device_bf16(dy.data(),w.n*t);
    failures+=verify_reduction("Vision BF16 Linear "+std::to_string(w.n)+"x"+std::to_string(w.k),
        actual,matrix(w,wide(x),t),linear_gate);
    failures+=dy.verify_guards("Vision Linear");
    return narrow(actual);
}
void bias(std::vector<double>& x,const std::vector<float>& b) {
    for(std::size_t i=0;i<x.size();++i) x[i]+=b[i%b.size()];
}
void gpu_bias(std::vector<float>& x,const std::vector<float>& b) {
    auto dx=to_device_bf16(x), db=to_device_bf16(b);
    Tensor tx(dx.p,DType::BF16,{static_cast<int>(b.size()),static_cast<int>(x.size()/b.size())});
    Tensor tb(db.p,DType::BF16,{static_cast<int>(b.size())});
    ops::add_bias(tb,tx,nullptr); x=narrow(from_device_bf16(dx,x.size()));
}
std::vector<float> gpu_norm(const std::vector<float>& x,const std::vector<float>& w,
                             const std::vector<float>& b,int t) {
    auto dx=to_device_bf16(x), dw=to_device_bf16(w), db=to_device_bf16(b);
    GuardedDeviceBuffer dy(x.size()*2);
    Tensor tx(dx.p,DType::BF16,{D,t}), tw(dw.p,DType::BF16,{D}), tb(db.p,DType::BF16,{D}),
        ty(dy.data(),DType::BF16,{D,t});
    ops::layer_norm(tx,tw,tb,1e-6f,ty,nullptr);
    failures+=dy.verify_guards("Vision norm");
    return narrow(from_device_bf16(dy.data(),x.size()));
}
void gpu_residual(std::vector<float>& x,const std::vector<float>& y,int t) {
    auto dx=to_device_bf16(x),dy=to_device_bf16(y);
    Tensor tx(dx.p,DType::BF16,{D,t}),ty(dy.p,DType::BF16,{D,t});
    ops::residual_add(ty,tx,nullptr); x=narrow(from_device_bf16(dx,x.size()));
}
struct Source {
    artifact::Reader reader;
    std::map<std::string,HostWeight> matrices;
    std::map<std::string,std::vector<float>> vectors;
    explicit Source(const std::string& path):reader(path) {
        if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-vision-block-qualification","nvidia-bf16-source"})
            throw std::runtime_error("invalid native Vision block identity");
        const auto load=[&](const std::string& name,std::vector<std::uint64_t> shape) {
            const auto* o=reader.find("model.visual."+name);
            const auto* d=o?std::get_if<artifact::TensorDescriptor>(o):nullptr;
            if(!d || d->format!=artifact::NumericFormat::BF16 ||
               d->layout!=artifact::StorageLayout::ContiguousLeV1 || d->shape!=shape)
                throw std::runtime_error("invalid native Vision tensor "+name);
            auto bytes=reader.payload(*o).data;
            std::vector<std::uint16_t> bits(bytes.size()/2);
            for(std::size_t i=0;i<bits.size();++i)
                bits[i]=std::to_integer<unsigned>(bytes[i*2]) | (std::to_integer<unsigned>(bytes[i*2+1])<<8);
            return bits;
        };
        matrices["patch"]={D,1536,load("patch_embed.proj.weight",{D,3,2,16,16})};
        for(const auto& [name,n,k]:std::vector<std::tuple<std::string,int,int>>{
            {"attn.qkv",3456,D},{"attn.proj",D,D},{"mlp.linear_fc1",4304,D},{"mlp.linear_fc2",D,4304}})
            matrices[name]={n,k,load("blocks.0."+name+".weight",{static_cast<unsigned>(n),static_cast<unsigned>(k)})};
        for(const auto& [name,width]:std::vector<std::pair<std::string,int>>{
            {"patch_embed.proj.bias",D},{"pos_embed.weight",2304*D},
            {"blocks.0.norm1.weight",D},{"blocks.0.norm1.bias",D},
            {"blocks.0.norm2.weight",D},{"blocks.0.norm2.bias",D},
            {"blocks.0.attn.qkv.bias",3456},{"blocks.0.attn.proj.bias",D},
            {"blocks.0.mlp.linear_fc1.bias",4304},{"blocks.0.mlp.linear_fc2.bias",D}}) {
            const auto bits=load(name,name=="pos_embed.weight"?std::vector<std::uint64_t>{2304,D}:
                std::vector<std::uint64_t>{static_cast<unsigned>(width)});
            auto& values=vectors[name]; values.reserve(bits.size());
            for(auto b:bits) values.push_back(bf16_to_f32(b));
        }
    }
};
struct Positions {
    std::vector<int> rope,indices,segments;
    std::vector<float> weights;
    explicit Positions(int t):rope(2*t),indices(4*t),weights(4*t) {
        segments=t==4?std::vector<int>{0,t}:std::vector<int>{0,(t/8)*4,t};
        for(std::size_t s=0;s+1<segments.size();++s) {
            const int begin=segments[s],width=(segments[s+1]-begin)/2;
            for(int group=0;group<width/2;++group) for(int row=0;row<2;++row) for(int col=0;col<2;++col) {
                const int p=begin+group*4+row*2+col,x=group*2+col;
                rope[p]=row; rope[t+p]=x;
                // Explicit 2x2 merge-major patches, align-corners interpolation on source48x48.
                const double fy=row*47.0,fx=x*47.0/(width-1);
                const int y0=static_cast<int>(fy),x0=static_cast<int>(fx),y1=std::min(47,y0+1),x1=std::min(47,x0+1);
                indices[p*4]=y0*48+x0; indices[p*4+1]=y0*48+x1;
                indices[p*4+2]=y1*48+x0; indices[p*4+3]=y1*48+x1;
                weights[p*4]=(1-(fy-y0))*(1-(fx-x0)); weights[p*4+1]=(1-(fy-y0))*(fx-x0);
                weights[p*4+2]=(fy-y0)*(1-(fx-x0)); weights[p*4+3]=(fy-y0)*(fx-x0);
            }
        }
    }
};
template<class Scalar> std::vector<Scalar> split(const std::vector<Scalar>& qkv,int t,int part) {
    std::vector<Scalar> out(D*t);
    for(int p=0;p<t;++p) std::copy_n(qkv.data()+p*3*D+part*D,D,out.data()+p*D);
    return out;
}
std::vector<float> gpu_attention(std::vector<float> q,std::vector<float> k,
                                  const std::vector<float>& v,const Positions& pos,int t) {
    auto dq=to_device_bf16(q),dk=to_device_bf16(k),dv=to_device_bf16(v);
    auto dp=to_device(pos.rope),ds=to_device(pos.segments);
    Tensor tq(dq.p,DType::BF16,{72,16,t}),tk(dk.p,DType::BF16,{72,16,t}),tv(dv.p,DType::BF16,{72,16,t}),
        tp(dp.p,DType::I32,{t,2}),ts(ds.p,DType::I32,{static_cast<int>(pos.segments.size())});
    ops::rope(tp,72,10000.f,tq,tk,nullptr);
    GuardedDeviceBuffer out(D*t*2);
    Tensor ty(out.data(),DType::BF16,{72,16,t});
    const int count=pos.segments.size()-1;
    WorkspaceArena arena(std::max<std::size_t>(256,ops::vision_attention_workspace_capacity_bytes(t,t,count,count)));
    ops::vision_attention(tq,tk,tv,ts,arena,ty,nullptr);
    failures+=out.verify_guards("Vision attention");
    return narrow(from_device_bf16(out.data(),D*t));
}
// Complete source patch+position+encoder formula. Shared norm/RoPE/attention oracles preserve
// FP64 ideal intermediates, not production's norm/linear/bias/activation workspaces.
std::vector<double> oracle(Source& s,const std::vector<float>& patches,const Positions& pos,int t) {
    auto x=matrix(s.matrices.at("patch"),wide(patches),t);
    bias(x,s.vectors.at("patch_embed.proj.bias"));
    const auto& table=s.vectors.at("pos_embed.weight");
    for(int p=0;p<t;++p) for(int d=0;d<D;++d) {
        double v=0;
        for(int c=0;c<4;++c) v+=double(table[pos.indices[p*4+c]*D+d])*pos.weights[p*4+c];
        // Position interpolation publishes BF16, and the source position addition does too.
        x[p*D+d]=qwen4_sequence::represented(x[p*D+d]+qwen4_sequence::represented(v));
    }
    auto n=vision_source::norm(x,s.vectors.at("blocks.0.norm1.weight"),s.vectors.at("blocks.0.norm1.bias"),t);
    auto qkv=matrix(s.matrices.at("attn.qkv"),n,t); bias(qkv,s.vectors.at("blocks.0.attn.qkv.bias"));
    auto q=vision_source::rotary(split(qkv,t,0),pos.rope,t),k=vision_source::rotary(split(qkv,t,1),pos.rope,t);
    auto a=vision_source::attention(q,k,split(qkv,t,2),pos.segments);
    auto y=matrix(s.matrices.at("attn.proj"),a,t); bias(y,s.vectors.at("blocks.0.attn.proj.bias"));
    for(std::size_t i=0;i<x.size();++i) x[i]+=y[i];
    n=vision_source::norm(x,s.vectors.at("blocks.0.norm2.weight"),s.vectors.at("blocks.0.norm2.bias"),t);
    auto up=matrix(s.matrices.at("mlp.linear_fc1"),n,t); bias(up,s.vectors.at("blocks.0.mlp.linear_fc1.bias"));
    for(auto& z:up) z=.5*z*(1+std::tanh(std::sqrt(2/std::acos(-1.0))*(z+.044715*z*z*z)));
    y=matrix(s.matrices.at("mlp.linear_fc2"),up,t); bias(y,s.vectors.at("blocks.0.mlp.linear_fc2.bias"));
    for(std::size_t i=0;i<x.size();++i) x[i]+=y[i];
    return x;
}
std::vector<float> execute(Source& s,const std::vector<float>& patches,const Positions& pos,int t) {
    auto x=gpu_matrix(s.matrices.at("patch"),patches,t); gpu_bias(x,s.vectors.at("patch_embed.proj.bias"));
    auto dx=to_device_bf16(x),dt=to_device_bf16(s.vectors.at("pos_embed.weight"));
    auto di=to_device(pos.indices),dw=to_device(pos.weights);
    Tensor tx(dx.p,DType::BF16,{D,t}),tt(dt.p,DType::BF16,{D,2304}),ti(di.p,DType::I32,{4,t}),tw(dw.p,DType::FP32,{4,t});
    ops::vision_pos_embed_add(tt,ti,tw,tx,nullptr); x=narrow(from_device_bf16(dx,D*t));
    auto n=gpu_norm(x,s.vectors.at("blocks.0.norm1.weight"),s.vectors.at("blocks.0.norm1.bias"),t);
    auto qkv=gpu_matrix(s.matrices.at("attn.qkv"),n,t); gpu_bias(qkv,s.vectors.at("blocks.0.attn.qkv.bias"));
    auto a=gpu_attention(split(qkv,t,0),split(qkv,t,1),split(qkv,t,2),pos,t);
    auto y=gpu_matrix(s.matrices.at("attn.proj"),a,t); gpu_bias(y,s.vectors.at("blocks.0.attn.proj.bias"));
    gpu_residual(x,y,t);
    n=gpu_norm(x,s.vectors.at("blocks.0.norm2.weight"),s.vectors.at("blocks.0.norm2.bias"),t);
    auto up=gpu_matrix(s.matrices.at("mlp.linear_fc1"),n,t); gpu_bias(up,s.vectors.at("blocks.0.mlp.linear_fc1.bias"));
    auto du=to_device_bf16(up); Tensor tu(du.p,DType::BF16,{4304,t}); ops::gelu(tu,ops::GeluMode::Tanh,nullptr);
    up=narrow(from_device_bf16(du,up.size()));
    y=gpu_matrix(s.matrices.at("mlp.linear_fc2"),up,t); gpu_bias(y,s.vectors.at("blocks.0.mlp.linear_fc2.bias"));
    gpu_residual(x,y,t); return x;
}
void tails() {
    for(auto [n,k]:std::vector<std::pair<int,int>>{{1152,1536},{3456,1152},{1152,1152},{4304,1152},{1152,4304}}) {
        HostWeight w{n,k,std::vector<std::uint16_t>(n*k)};
        // Nonzero final16 K elements and final16 output rows make dropped tails observable.
        for(int row=0;row<n;++row) for(int col=k-16;col<k;++col)
            w.bits[row*k+col]=f32_to_bf16((row%3-1)*.5f+(col%2?.25f:-.25f));
        for(int t:{1,4,128,132}) {
            std::vector<float> x(k*t);
            for(int p=0;p<t;++p) for(int col=k-16;col<k;++col) x[p*k+col]=((p+col)%7-3)*.125f;
            (void)gpu_matrix(w,x,t);
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--native-real") {
            const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
            if(!root) { std::cout<<"SKIP: native Vision source absent\n"; return 77; }
            Source s(std::string(root)+"/qwen4-vision-block.ninfer");
            for(int t:{4,12,132}) {
                std::vector<float> patches(1536*t); fill_uniform(patches,971+t,-1.f,1.f); round_to_bf16(patches);
                Positions positions(t);
                auto expected=oracle(s,patches,positions,t),got=wide(execute(s,patches,positions,t));
                failures+=verify_reduction("native Vision complete P="+std::to_string(t),got,expected,chain_gate);
                if(t==12) {
                    auto changed=patches;
                    for(std::size_t i=positions.segments[1]*1536;i<changed.size();++i) changed[i]=-changed[i];
                    auto altered=execute(s,changed,positions,t);
                    std::vector<float> first(altered.begin(),altered.begin()+positions.segments[1]*D);
                    auto original=narrow(got); original.resize(first.size());
                    failures+=verify_exact("Vision segment isolation",first,original);
                }
            }
        } else tails();
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

#include "targets/qwen4/native_compute.h"

#include "artifact/typed_binding.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ple.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int D=2560,F=10240,S=2051;
constexpr std::size_t ConvBytes=F*3*2, RecurrentBytes=128*128*48*4, PleBytes=F*9*2;
int checked(int value,int maximum,const char* name) {
    if(value<1 || value>maximum) throw std::invalid_argument(name);
    return value;
}
std::size_t workspace_bytes(const LoadedNativeFirstBlock& model,int width) {
    std::size_t bytes=std::max(ops::gated_residual_workspace_capacity_bytes(width,QType::BF16_CTRL,QType::BF16_CTRL),
        ops::ple_workspace_capacity_bytes(width,QType::BF16_CTRL,QType::BF16_CTRL));
    for(int layer=0;layer<4;++layer) {
        bytes=std::max(bytes,ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(model.layers()[layer].moe,width,ops::LinearPolicy::A16Only));
        if(layer<3) bytes=std::max(bytes,ops::gated_delta_net_layer_workspace_capacity_bytes(width,
            QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL));
    }
    return bytes;
}
void input(const Tensor& value,int rows,int width) {
    if(!value.data || value.dtype!=DType::BF16 || !value.is_contiguous() ||
       value.ne[0]!=rows || value.ne[1]!=width || value.ne[2]!=1 || value.ne[3]!=1 ||
       (reinterpret_cast<std::uintptr_t>(value.data)&15))
        throw std::invalid_argument("native first-block represented input");
}
}

std::unique_ptr<LoadedNativeFirstBlock> LoadedNativeFirstBlock::load(
    const std::filesystem::path& root,const std::filesystem::path& ple_path,DeviceContext& device) {
    auto model=std::unique_ptr<LoadedNativeFirstBlock>(new LoadedNativeFirstBlock);
    for(int layer=0;layer<4;++layer) {
        artifact::Reader reader(root/("qwen4-native-layer-"+std::to_string(layer)+".ninfer"));
        if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-first-block-qualification","nvidia-nvfp4-a16-tiled"})
            throw std::invalid_argument("native first-block prepared artifact identity");
        artifact::Binder binder(reader);
        std::map<std::string,artifact::ObjectHandle> handles;
        const std::string prefix="model.language_model.layers."+std::to_string(layer)+".";
        auto bind=[&](const std::string& name,std::initializer_list<std::uint64_t> shape,
                      artifact::NumericFormat format=artifact::NumericFormat::BF16) {
            handles.emplace(name,artifact::bind_device_tensor(binder,prefix+name,format,shape));
        };
        for(const std::string& p:{"attn_hyper_connection.","mlp_hyper_connection."}) {
            bind(p+"hc_norm.weight",{F},artifact::NumericFormat::FP32);
            bind(p+"block_inject_weight.weight",{4,F},artifact::NumericFormat::FP32);
            bind(p+"input_mix_weight_down.weight",{320,F}); bind(p+"input_mix_weight_up.weight",{F,320});
        }
        const std::string m="mlp.";
        bind(m+"gate.weight",{512,D}); bind(m+"shared_expert_gate.weight",{1,D});
        bind(m+"shared_expert.gate_proj.weight",{640,D}); bind(m+"shared_expert.up_proj.weight",{640,D});
        bind(m+"shared_expert.down_proj.weight",{D,640});
        for(const std::string& role:{"gate_proj","up_proj","down_proj"}) {
            const bool down=role=="down_proj";
            bind(m+"experts."+role+".weight",{512,std::uint64_t(down?D:640),std::uint64_t(down?640:D)},
                 artifact::NumericFormat::NVFP4_EXPERT_F32M);
        }
        if(layer<3) {
            const std::string g="linear_attn.";
            bind(g+"in_proj_qkv.weight",{10240,D}); bind(g+"in_proj_z.weight",{6144,D});
            bind(g+"out_proj.weight",{D,6144});
            bind(g+"in_proj_a.weight",{48,D},artifact::NumericFormat::FP32);
            bind(g+"in_proj_b.weight",{48,D},artifact::NumericFormat::FP32);
            bind(g+"conv1d.weight",{10240,1,4},artifact::NumericFormat::FP32);
            bind(g+"ssm_a",{48},artifact::NumericFormat::FP32);
            bind(g+"dt_bias",{48},artifact::NumericFormat::FP32);
            bind(g+"norm.weight",{128},artifact::NumericFormat::FP32);
        } else {
            const std::string a="self_attn.";
            bind(a+"indexer.index_qk_proj.weight",{640,D});
            bind(a+"indexer.q_layernorm.weight",{128},artifact::NumericFormat::FP32);
            bind(a+"indexer.k_layernorm.weight",{128},artifact::NumericFormat::FP32);
            bind(a+"q_norm.weight",{256},artifact::NumericFormat::FP32);
            bind(a+"k_norm.weight",{256},artifact::NumericFormat::FP32);
            bind(a+"q_proj.weight",{12288,D}); bind(a+"k_proj.weight",{512,D});
            bind(a+"v_proj.weight",{512,D}); bind(a+"o_proj.weight",{D,6144});
        }
        auto& backing=model->backing_[layer];
        backing=artifact::materialize(reader,binder.finish(),device);
        auto weight=[&](const std::string& name,int n,int k) {
            return artifact::materialized_weight(backing,handles.at(name),artifact::NumericFormat::BF16,n,k);
        };
        auto tensor=[&](const std::string& name,std::initializer_list<int> shape,DType type=DType::FP32) {
            return Tensor(backing.device_data(handles.at(name)),type,shape);
        };
        auto gr=[&](const std::string& p) {
            return NativeGrWeights{tensor(p+"hc_norm.weight",{F}),tensor(p+"block_inject_weight.weight",{F,4}),
                weight(p+"input_mix_weight_down.weight",320,F),weight(p+"input_mix_weight_up.weight",F,320)};
        };
        auto& w=model->layers_[layer];
        w.attention_gr=gr("attn_hyper_connection."); w.moe_gr=gr("mlp_hyper_connection.");
        auto bank=[&](const char* role,int n,int k) {
            return artifact::materialized_nvfp4_expert_weight(backing,handles.at(m+"experts."+role+".weight"),512,n,k);
        };
        w.moe={weight(m+"gate.weight",512,D),bank("gate_proj",640,D),bank("up_proj",640,D),bank("down_proj",D,640),
            tensor(m+"shared_expert_gate.weight",{D},DType::BF16),weight(m+"shared_expert.gate_proj.weight",640,D),
            weight(m+"shared_expert.up_proj.weight",640,D),weight(m+"shared_expert.down_proj.weight",D,640)};
        if(layer<3) {
            const std::string g="linear_attn.";
            w.gdn={weight(g+"in_proj_qkv.weight",10240,D),weight(g+"in_proj_z.weight",6144,D),
                tensor(g+"in_proj_a.weight",{D,48}),tensor(g+"in_proj_b.weight",{D,48}),
                tensor(g+"conv1d.weight",{4,10240}),tensor(g+"ssm_a",{48}),tensor(g+"dt_bias",{48}),
                tensor(g+"norm.weight",{128}),weight(g+"out_proj.weight",D,6144)};
        } else {
            const std::string a="self_attn.";
            Weight iq=weight(a+"indexer.index_qk_proj.weight",640,D),ik=iq;
            iq.n=iq.shape[0]=iq.padded_shape[0]=512; iq.payload_bytes=512ULL*D*2;
            ik.n=ik.shape[0]=ik.padded_shape[0]=128; ik.payload_bytes=128ULL*D*2;
            ik.payload=ik.qdata=static_cast<const std::byte*>(ik.payload)+512ULL*D*2;
            w.qsa={iq,ik,weight(a+"q_proj.weight",12288,D),weight(a+"k_proj.weight",512,D),
                weight(a+"v_proj.weight",512,D),weight(a+"o_proj.weight",D,6144),
                tensor(a+"indexer.q_layernorm.weight",{128}),tensor(a+"indexer.k_layernorm.weight",{128}),
                tensor(a+"q_norm.weight",{256}),tensor(a+"k_norm.weight",{256})};
        }
    }
    artifact::Reader reader(ple_path);
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-ple-component-qualification","nvidia-bf16-source"})
        throw std::invalid_argument("native first-block PLE component identity");
    artifact::Binder binder(reader);
    const std::string p="model.language_model.layers.1.ple.";
    auto bind=[&](const char* role,std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_device_tensor(binder,p+role,artifact::NumericFormat::BF16,shape);
    };
    const auto key=bind("key_proj.weight",{F,D}), value=bind("value_proj.weight",{D,D}),
        kn=bind("norm_key.weight",{F}),qn=bind("norm_query.weight",{F}),cn=bind("norm_conv.weight",{F}),
        conv=bind("conv1d.weight",{F,1,4});
    auto& b=model->backing_[4]; b=artifact::materialize(reader,binder.finish(),device);
    model->ple_={artifact::materialized_weight(b,key,artifact::NumericFormat::BF16,F,D),
        artifact::materialized_weight(b,value,artifact::NumericFormat::BF16,D,D),
        Tensor(b.device_data(kn),DType::BF16,{F}),Tensor(b.device_data(qn),DType::BF16,{F}),
        Tensor(b.device_data(cn),DType::BF16,{F}),Tensor(b.device_data(conv),DType::BF16,{4,F})};
    return model;
}

struct NativeFirstBlock::Slot {
    int frontier=0,pending=0;
    std::vector<int> host_ids,host_positions,host_visible,host_offsets;
    DeviceBuffer hidden,ple_embedding,features,ids,positions,visible,offsets,selected,counts;
    DeviceBuffer committed_conv,provisional_conv,committed_ssm,provisional_ssm,committed_ple,provisional_ple;
    DeviceBuffer k,v,raw_keys,key_positions,record_conv,record_key,record_value,record_gate,record_ple;
    Slot(int capacity,int width)
        :hidden(std::size_t(F)*width*2),ple_embedding(std::size_t(D)*width*2),features(std::size_t(F)*width*4*2),
         ids(width*4),positions(width*12),visible(std::size_t(capacity)*width*4),offsets((width+1)*4),
         selected(std::size_t(S)*width*4),counts(width*4),committed_conv(3*ConvBytes),provisional_conv(3*ConvBytes),
         committed_ssm(3*RecurrentBytes),provisional_ssm(3*RecurrentBytes),committed_ple(PleBytes),provisional_ple(PleBytes),
         k(std::size_t(512)*capacity*2),v(std::size_t(512)*capacity*2),raw_keys(std::size_t(128)*capacity*2),
         key_positions(std::size_t(3)*capacity*4),record_conv(3ULL*F*16*2),record_key(3ULL*128*48*16*2),
         record_value(3ULL*128*48*16*2),record_gate(3ULL*2*48*16*4),record_ple(F*16*2) {
        host_ids.reserve(width); host_positions.reserve(width*3);
        host_visible.reserve(std::size_t(capacity)*width); host_offsets.reserve(width+1);
    }
};

NativeFirstBlock::NativeFirstBlock(const LoadedNativeFirstBlock& model,int requests,int capacity,int width)
    :model_(model),requests_(checked(requests,4,"native first-block requests")),
     capacity_(checked(capacity,4096,"native first-block capacity")),max_width_(checked(width,capacity_,"native first-block width")),
     mixed_(std::size_t(D)*width*2),block_(std::size_t(D)*width*2),scale_(width*4*2),routes_(width*10*4),
     route_weights_(width*10*4),qsa_workspace_(ops::qsa_verifier_workspace_bytes(width,
         QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL)),workspace_(workspace_bytes(model,width)) {
    for(int i=0;i<requests_;++i) { slots_[i]=std::make_unique<Slot>(capacity,width); reset(i,nullptr); }
    CUDA_CHECK(cudaStreamSynchronize(nullptr)); // Startup initialization before nonblocking streams consume state.
}
NativeFirstBlock::~NativeFirstBlock()=default;
NativeFirstBlock::Slot& NativeFirstBlock::slot(int index) const {
    if(index<0 || index>=requests_) throw std::invalid_argument("native first-block slot");
    return *slots_[index];
}
int NativeFirstBlock::frontier(int index) const { return slot(index).frontier; }
void NativeFirstBlock::reset(int index,cudaStream_t stream) {
    auto& s=slot(index);
    for(auto* b:{&s.committed_conv,&s.provisional_conv,&s.committed_ssm,&s.provisional_ssm,
                 &s.committed_ple,&s.provisional_ple,&s.k,&s.v,&s.raw_keys,&s.key_positions})
        CUDA_CHECK(cudaMemsetAsync(b->p,0,b->bytes,stream));
    s.frontier=0; s.pending=0;
}
NativeFirstBlockState NativeFirstBlock::state(int index,bool provisional) const {
    auto& s=slot(index);
    NativeFirstBlockState result;
    auto* conv=static_cast<std::byte*>((provisional?s.provisional_conv:s.committed_conv).p);
    auto* rec=static_cast<std::byte*>((provisional?s.provisional_ssm:s.committed_ssm).p);
    for(int i=0;i<3;++i) result.gdn[i]={Tensor(conv+i*ConvBytes,DType::BF16,{F,3}),
                                                   Tensor(rec+i*RecurrentBytes,DType::FP32,{128,128,48})};
    result.ple_conv=Tensor((provisional?s.provisional_ple:s.committed_ple).p,DType::BF16,{F,9});
    result.qsa={ops::QsaKvFormat::BF16,Tensor(s.k.p,DType::BF16,{256,capacity_,2}),
        Tensor(s.v.p,DType::BF16,{256,capacity_,2}),{},{},Tensor(s.raw_keys.p,DType::BF16,{128,capacity_}),
        Tensor(s.key_positions.p,DType::I32,{3,capacity_})};
    return result;
}
NativeFirstBlockRecords NativeFirstBlock::records(int index,int width) const {
    if(width<2 || width>16 || width>max_width_) throw std::invalid_argument("native first-block replay width");
    auto& s=slot(index); NativeFirstBlockRecords r;
    for(int i=0;i<3;++i) r.gdn[i]={
        Tensor(static_cast<std::byte*>(s.record_conv.p)+std::size_t(i)*F*16*2,DType::BF16,{F,width,1}),
        Tensor(static_cast<std::byte*>(s.record_key.p)+std::size_t(i)*128*48*16*2,DType::BF16,{128,48,width,1}),
        Tensor(static_cast<std::byte*>(s.record_value.p)+std::size_t(i)*128*48*16*2,DType::BF16,{128,48,width,1}),
        Tensor(static_cast<std::byte*>(s.record_gate.p)+std::size_t(i)*2*48*16*4,DType::FP32,{2,48,width,1})};
    r.ple_conv=Tensor(s.record_ple.p,DType::BF16,{F,width}); return r;
}

void NativeFirstBlock::prepare(int index,std::span<const std::array<int,3>> positions,
                              const Tensor& hidden,const Tensor& ple_embedding,cudaStream_t stream) {
    auto& s=slot(index); const int width=static_cast<int>(positions.size());
    if(width<1 || width>max_width_ || s.frontier+width>capacity_ || s.pending)
        throw std::invalid_argument("native first-block pending/capacity violation");
    input(hidden,F,width); input(ple_embedding,D,width);
    for(auto p:positions) for(int coordinate:p) if(coordinate<0)
        throw std::invalid_argument("native first-block negative RoPE position");
    s.host_ids.clear(); s.host_positions.clear(); s.host_visible.clear(); s.host_offsets.assign(1,0);
    for(int t=0;t<width;++t) {
        s.host_ids.push_back(s.frontier+t);
        s.host_positions.insert(s.host_positions.end(),positions[t].begin(),positions[t].end());
        for(int i=0;i<=s.frontier+t;++i) s.host_visible.push_back(i);
        s.host_offsets.push_back(static_cast<int>(s.host_visible.size()));
    }
    for(auto [buffer,host]:std::array<std::pair<DeviceBuffer*,std::vector<int>*>,4>{{
            {&s.ids,&s.host_ids},{&s.positions,&s.host_positions},{&s.visible,&s.host_visible},{&s.offsets,&s.host_offsets}}})
        CUDA_CHECK(cudaMemcpyAsync(buffer->p,host->data(),host->size()*4,cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(s.hidden.p,hidden.data,hidden.bytes(),cudaMemcpyDeviceToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(s.ple_embedding.p,ple_embedding.data,ple_embedding.bytes(),cudaMemcpyDeviceToDevice,stream));
    s.pending=width;
}

NativeFirstBlockOutput NativeFirstBlock::enqueue(int index,int width,bool record,cudaStream_t stream,
                                               const NativeBlockTrace* trace) {
    auto& s=slot(index);
    if(width!=s.pending || width<1 || width>max_width_ || (record && (width<2 || width>16)))
        throw std::invalid_argument("native first-block unprepared width");
    auto before=state(index,false),after=state(index,true);
    auto replay=record?records(index,width):NativeFirstBlockRecords{};
    Tensor residual(s.hidden.p,DType::BF16,{D,4,width}),mixed(mixed_.p,DType::BF16,{D,width}),
        block(block_.p,DType::BF16,{D,width}),scale(scale_.p,DType::BF16,{4,width}),
        routes(routes_.p,DType::I32,{10,width}),probabilities(route_weights_.p,DType::FP32,{10,width});
    NativeFirstBlockOutput result{residual,{}};
    auto capture=[&](int layer,NativeBlockTraceStage stage,const Tensor& value) {
        if(trace && trace->capture) trace->capture(trace->context,layer,stage,value,stream);
    };
    for(int layer=0;layer<4;++layer) {
        if(layer==1) {
            const auto& p=model_.ple();
            Tensor embedding(s.ple_embedding.p,DType::BF16,{D,width});
            ops::ple_inject(residual,embedding,p.key,p.value,p.key_norm,p.query_norm,p.conv_norm,p.conv,
                before.ple_conv,after.ple_conv,residual,workspace_,ops::PleNormFormat::ZeroCenteredBf16,stream,
                record?&replay.ple_conv:nullptr);
            capture(layer,NativeBlockTraceStage::Ple,residual);
        }
        const auto& w=model_.layers()[layer]; const auto& a=w.attention_gr;
        ops::gated_residual_read_write(residual,a.norm,a.down,a.up,a.inject,mixed,scale,workspace_,stream);
        capture(layer,NativeBlockTraceStage::AttentionRead,mixed); capture(layer,NativeBlockTraceStage::AttentionWrite,scale);
        if(layer<3) ops::gated_delta_net_layer(mixed,w.gdn,before.gdn[layer].conv,after.gdn[layer].conv,
            before.gdn[layer].recurrence,after.gdn[layer].recurrence,block,workspace_,stream,{},record?&replay.gdn[layer]:nullptr);
        else {
            Tensor ids(s.ids.p,DType::I32,{width}),pos(s.positions.p,DType::I32,{3,width}),
                visible(s.visible.p,DType::I32,{capacity_*width}),offsets(s.offsets.p,DType::I32,{width+1}),
                selected(s.selected.p,DType::I32,{S,width}),count(s.counts.p,DType::I32,{width}),
                workspace(qsa_workspace_.p,DType::U8,{static_cast<int>(qsa_workspace_.bytes)});
            ops::qsa_verifier(mixed,ids,pos,visible,offsets,w.qsa,after.qsa,selected,count,block,workspace,stream);
        }
        capture(layer,NativeBlockTraceStage::Mixer,block);
        ops::gated_residual_inject(residual,block,scale,residual,stream); capture(layer,NativeBlockTraceStage::AttentionInjected,residual);
        const auto& m=w.moe_gr;
        ops::gated_residual_read_write(residual,m.norm,m.down,m.up,m.inject,mixed,scale,workspace_,stream);
        capture(layer,NativeBlockTraceStage::MoeRead,mixed); capture(layer,NativeBlockTraceStage::MoeWrite,scale);
        ops::qwen4_sparse_moe_resident(mixed,w.moe,routes,probabilities,block,workspace_,stream,ops::LinearPolicy::A16Only);
        capture(layer,NativeBlockTraceStage::MoeIds,routes); capture(layer,NativeBlockTraceStage::MoeWeights,probabilities);
        capture(layer,NativeBlockTraceStage::Moe,block);
        ops::gated_residual_inject(residual,block,scale,residual,stream); capture(layer,NativeBlockTraceStage::MoeInjected,residual);
        result.layer_hidden[layer]=Tensor(static_cast<std::byte*>(s.features.p)+std::size_t(layer)*F*max_width_*2,DType::BF16,{D,4,width});
        CUDA_CHECK(cudaMemcpyAsync(result.layer_hidden[layer].data,residual.data,residual.bytes(),cudaMemcpyDeviceToDevice,stream));
    }
    return result;
}

void NativeFirstBlock::commit_all(int index,cudaStream_t stream) {
    auto& s=slot(index);
    if(!s.pending) throw std::logic_error("native first-block commit without prepared execution");
    for(auto [to,from]:std::array<std::pair<DeviceBuffer*,DeviceBuffer*>,3>{{
            {&s.committed_conv,&s.provisional_conv},{&s.committed_ssm,&s.provisional_ssm},{&s.committed_ple,&s.provisional_ple}}})
        CUDA_CHECK(cudaMemcpyAsync(to->p,from->p,to->bytes,cudaMemcpyDeviceToDevice,stream));
    publish_frontier(index,s.pending);
}
void NativeFirstBlock::publish_frontier(int index,int accepted) {
    auto& s=slot(index);
    if(!s.pending || accepted<0 || accepted>s.pending) throw std::invalid_argument("native first-block accepted frontier");
    s.frontier+=accepted; s.pending=0;
}
void NativeFirstBlock::discard(int index) { auto& s=slot(index); s.pending=0; }
void NativeFirstBlock::restore_frontier(int index,int frontier) {
    auto& s=slot(index);
    if(s.pending || frontier<0 || frontier>capacity_) throw std::invalid_argument("native first-block restore frontier");
    s.frontier=frontier;
}
} // namespace ninfer::targets::qwen4

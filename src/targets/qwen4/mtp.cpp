#include "targets/qwen4/mtp.h"

#include "artifact/typed_binding.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/gated_residual_stem.h"

#include <algorithm>
#include <bit>
#include <map>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int D=2560,F=10240,S=ops::kQsaSelectedCapacity;
Tensor bf16(DeviceBuffer& b,int width,int count) { return Tensor(b.p,DType::BF16,{width,count}); }
void require_input(const Tensor& x,int width,int columns) {
    if(!x.data || x.dtype!=DType::BF16 || !x.is_contiguous() || x.ne[0]!=width ||
       x.ne[1]!=columns || x.ne[2]!=1 || x.ne[3]!=1 ||
       (reinterpret_cast<std::uintptr_t>(x.data)&15U))
        throw std::invalid_argument("Qwen4 MTP invalid represented input shape");
}
bool overlaps(const Tensor& a,const void* data,std::size_t size) {
    const auto x=reinterpret_cast<std::uintptr_t>(a.data), y=reinterpret_cast<std::uintptr_t>(data);
    return x<y+size && y<x+a.bytes();
}
}

std::unique_ptr<LoadedMtp> LoadedMtp::load(const std::filesystem::path& path,DeviceContext& device) {
    artifact::Reader reader(path);
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-mtp-qualification","limpincat-nvfp4-w4a16-source"})
        throw std::invalid_argument("Qwen4 MTP requires the exact native W4A16 fixture");
    artifact::Binder binder(reader);
    std::map<std::string,artifact::ObjectHandle> handles;
    auto bind=[&](const std::string& name,std::initializer_list<std::uint64_t> shape,
                  artifact::NumericFormat format=artifact::NumericFormat::BF16) {
        handles.emplace(name,artifact::bind_device_tensor(binder,"mtp."+name,format,shape));
    };
    bind("fc_embedding.weight",{D,D}); bind("fc_hidden.weight",{D,D});
    bind("pre_fc_norm_embedding.weight",{D}); bind("pre_fc_norm_hidden.weight",{F});
    for(const auto& p:std::array<std::string,3>{"layers.0.attn_hyper_connection.",
               "layers.0.mlp_hyper_connection.","hyper_connection_mixer."}) {
        bind(p+"hc_norm.weight",{F}); bind(p+"input_mix_weight_down.weight",{320,F});
        bind(p+"input_mix_weight_up.weight",{F,320});
        if(p!="hyper_connection_mixer.") bind(p+"block_inject_weight.weight",{4,F});
    }
    const std::string a="layers.0.self_attn.",m="layers.0.mlp.";
    bind(a+"indexer.index_qk_proj.weight",{640,D});
    bind(a+"indexer.q_layernorm.weight",{128}); bind(a+"indexer.k_layernorm.weight",{128});
    bind(a+"q_norm.weight",{256}); bind(a+"k_norm.weight",{256});
    bind(a+"q_proj.weight",{12288,D}); bind(a+"k_proj.weight",{512,D});
    bind(a+"v_proj.weight",{512,D}); bind(a+"o_proj.weight",{D,6144});
    bind(m+"gate.weight",{512,D}); bind(m+"shared_expert_gate.weight",{1,D});
    bind(m+"shared_expert.gate_proj.weight",{640,D}); bind(m+"shared_expert.up_proj.weight",{640,D});
    bind(m+"shared_expert.down_proj.weight",{D,640});
    for(const char* role:{"gate_proj","up_proj","down_proj"}) {
        const bool down=std::string(role)=="down_proj";
        bind(m+"experts."+role+".weight",{512,std::uint64_t(down?D:640),std::uint64_t(down?640:D)},
             artifact::NumericFormat::NVFP4_EXPERT_F32M);
    }
    auto result=std::unique_ptr<LoadedMtp>(new LoadedMtp);
    result->backing_=artifact::materialize(reader,binder.finish(),device);
    auto weight=[&](const std::string& name,int n,int k) {
        return artifact::materialized_weight(result->backing_,handles.at(name),artifact::NumericFormat::BF16,n,k);
    };
    auto tensor=[&](const std::string& name,int n) {
        return Tensor(result->backing_.device_data(handles.at(name)),DType::BF16,{n});
    };
    // Lossless source-control widening plus the architecture's explicit unit offset.
    auto control=[&](const std::string& name,bool offset,std::initializer_list<int> shape) {
        const auto* object=reader.find("mtp."+name);
        const auto raw=reader.payload(*object).data;
        std::vector<float> values(raw.size()/2);
        for(std::size_t i=0;i<values.size();++i) {
            const std::uint32_t bits=std::to_integer<unsigned>(raw[i*2]) |
                                      (std::to_integer<unsigned>(raw[i*2+1])<<8);
            values[i]=std::bit_cast<float>(bits<<16)+(offset?1.0F:0.0F);
        }
        result->controls_.emplace_back(values.size()*4);
        result->controls_.back().copy_from_host(values.data(),values.size()*4);
        return Tensor(result->controls_.back().p,DType::FP32,shape);
    };
    auto gr=[&](const std::string& p,bool inject) {
        MtpGrWeights g{control(p+"hc_norm.weight",true,{F}),{},
            weight(p+"input_mix_weight_down.weight",320,F),weight(p+"input_mix_weight_up.weight",F,320)};
        if(inject) g.inject=control(p+"block_inject_weight.weight",false,{F,4});
        return g;
    };
    auto& w=result->weights_;
    w.embedding_norm=tensor("pre_fc_norm_embedding.weight",D); w.hidden_norm=tensor("pre_fc_norm_hidden.weight",F);
    w.embedding=weight("fc_embedding.weight",D,D); w.hidden=weight("fc_hidden.weight",D,D);
    w.attention_gr=gr("layers.0.attn_hyper_connection.",true);
    w.moe_gr=gr("layers.0.mlp_hyper_connection.",true); w.final_gr=gr("hyper_connection_mixer.",false);
    Weight iq=weight(a+"indexer.index_qk_proj.weight",640,D),ik=iq;
    iq.n=iq.shape[0]=iq.padded_shape[0]=512; iq.payload_bytes=512ULL*D*2;
    ik.n=ik.shape[0]=ik.padded_shape[0]=128; ik.payload_bytes=128ULL*D*2;
    ik.payload=ik.qdata=static_cast<const std::byte*>(ik.payload)+512ULL*D*2;
    w.attention={iq,ik,weight(a+"q_proj.weight",12288,D),weight(a+"k_proj.weight",512,D),
        weight(a+"v_proj.weight",512,D),weight(a+"o_proj.weight",D,6144),
        control(a+"indexer.q_layernorm.weight",true,{128}),control(a+"indexer.k_layernorm.weight",true,{128}),
        control(a+"q_norm.weight",true,{256}),control(a+"k_norm.weight",true,{256})};
    auto bank=[&](const char* role,int n,int k) {
        return artifact::materialized_nvfp4_expert_weight(result->backing_,handles.at(m+"experts."+role+".weight"),512,n,k);
    };
    w.moe={weight(m+"gate.weight",512,D),bank("gate_proj",640,D),bank("up_proj",640,D),
        bank("down_proj",D,640),tensor(m+"shared_expert_gate.weight",D),
        weight(m+"shared_expert.gate_proj.weight",640,D),weight(m+"shared_expert.up_proj.weight",640,D),
        weight(m+"shared_expert.down_proj.weight",D,640)};
    return result;
}


struct MtpProgram::Storage {
    int columns;
    std::size_t residual,mixed,scale,selected,counts,routes,qsa,scratch;
    Storage(int width,int batch) {
        if(width<1 || width>4096 || batch<1 || batch>4)
            throw std::invalid_argument("Qwen4 MTP startup geometry");
        columns=std::max(width,std::min(width,16)*batch);
        residual=std::size_t(F)*columns*2;mixed=std::size_t(D)*columns*2;
        scale=std::size_t(4)*columns*2;selected=std::size_t(S)*columns*4;
        counts=std::size_t(columns)*4;routes=counts*10;
        const auto qbytes=[](int w,int b) {return ops::qsa_verifier_workspace_bytes(w,b,
            QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL);};
        qsa=std::max(qbytes(width,1),qbytes(std::min(width,16),batch));
        const ops::Qwen4ResidentSparseMoeStorageProfile profile{
            QType::NVFP4_EXPERT_F32M,QType::NVFP4_EXPERT_F32M,QType::NVFP4_EXPERT_F32M,
            QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL};
        scratch=std::max({ops::gated_residual_stem_workspace_capacity_bytes(columns),
            ops::gated_residual_workspace_capacity_bytes(columns,QType::BF16_CTRL,QType::BF16_CTRL),
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(profile,columns)});
    }
    std::size_t bytes() const {return residual+3*mixed+scale+selected+counts+2*routes+qsa+scratch;}
};
std::size_t MtpProgram::device_bytes(int width,int batch) {return Storage(width,batch).bytes();}
MtpProgram::MtpProgram(const MtpWeights& w,int width,int batch):MtpProgram(w,width,batch,Storage(width,batch)) {}
MtpProgram::MtpProgram(const MtpWeights& w,int width,int batch,const Storage& s)
    :weights_(w),max_width_(width),max_batch_(batch),residual_(s.residual),mixed_(s.mixed),
     block_(s.mixed),scale_(s.scale),logit_(s.mixed),selected_(s.selected),counts_(s.counts),
     routes_(s.routes),route_weights_(s.routes),qsa_workspace_(s.qsa),workspace_(s.scratch) {}
Tensor MtpProgram::selected_ids(int width,int batch) const {
    return Tensor(selected_.p,DType::I32,{S,width,batch});
}
Tensor MtpProgram::selected_count(int width,int batch) const {
    return Tensor(counts_.p,DType::I32,{width,batch});
}
MtpOutput MtpProgram::output(int width,int batch) const {
    return {Tensor(residual_.p,DType::BF16,{F,width,batch}),Tensor(logit_.p,DType::BF16,{D,width,batch})};
}
MtpOutput MtpProgram::execute(const Tensor& embedding,const Tensor& hidden,
    const ops::QsaBatchControls& controls,ops::QsaPagedStateView state,int maximum,
    const Tensor* frozen_ids,const Tensor* frozen_count,cudaStream_t stream,const MtpTrace* trace) {
    const int width=embedding.ne[1],batch=embedding.ne[2],columns=width*batch;
    if(width<1 || width>max_width_ || batch<1 || batch>max_batch_ || (batch>1 && width>16) ||
       embedding.ne[0]!=D || hidden.ne[0]!=F || embedding.ne[3]!=1 || hidden.ne[1]!=width || hidden.ne[2]!=batch || hidden.ne[3]!=1 ||
       bool(frozen_ids)!=bool(frozen_count) || state.format!=ops::QsaKvFormat::BF16)
        throw std::invalid_argument("Qwen4 MTP compact paged input geometry");
    require_input(embedding.view({D,columns}),D,columns);require_input(hidden.view({F,columns}),F,columns);
    if(overlaps(embedding,hidden.data,hidden.bytes()))
        throw std::invalid_argument("Qwen4 MTP input buffers overlap");
    for(const DeviceBuffer* buffer:{&residual_,&mixed_,&block_,&scale_,&logit_})
        if(overlaps(embedding,buffer->p,buffer->bytes) || overlaps(hidden,buffer->p,buffer->bytes))
            throw std::invalid_argument("Qwen4 MTP input aliases execution scratch");
    Tensor residual(residual_.p,DType::BF16,{D,4,columns}),mixed=bf16(mixed_,D,columns),
        block=bf16(block_,D,columns),scale=bf16(scale_,4,columns),logit=bf16(logit_,D,columns);
    auto capture=[&](MtpTraceStage stage,const Tensor& value) {
        if(trace && trace->capture) trace->capture(trace->context,stage,value,stream);
    };
    auto e=embedding.view({D,columns}),h=hidden.view({D,4,columns});
    ops::gated_residual_stem(e,h,weights_.embedding_norm,weights_.hidden_norm,
        weights_.embedding,weights_.hidden,residual,workspace_,stream);capture(MtpTraceStage::Stem,residual);
    const auto& a=weights_.attention_gr;
    ops::gated_residual_read_write(residual,a.norm,a.down,a.up,a.inject,mixed,scale,workspace_,stream);
    capture(MtpTraceStage::AttentionRead,mixed);capture(MtpTraceStage::AttentionWrite,scale);
    auto selection=selected_ids(width,batch),counts=selected_count(width,batch);
    Tensor qws(qsa_workspace_.p,DType::U8,{static_cast<int>(qsa_workspace_.bytes)});
    auto qin=mixed.view({D,width,batch}),qout=block.view({D,width,batch});
    if(frozen_ids) ops::qsa_verifier_selected(qin,controls,maximum,weights_.attention,state,
        *frozen_ids,*frozen_count,qout,qws,stream);
    else ops::qsa_verifier(qin,controls,maximum,weights_.attention,state,selection,counts,qout,qws,stream);
    capture(MtpTraceStage::Attention,block);
    ops::gated_residual_inject(residual,block,scale,residual,stream);capture(MtpTraceStage::AttentionInjected,residual);
    const auto& m=weights_.moe_gr;
    ops::gated_residual_read_write(residual,m.norm,m.down,m.up,m.inject,mixed,scale,workspace_,stream);
    capture(MtpTraceStage::MoeRead,mixed);capture(MtpTraceStage::MoeWrite,scale);
    Tensor routes(routes_.p,DType::I32,{10,columns}),probabilities(route_weights_.p,DType::FP32,{10,columns});
    ops::qwen4_sparse_moe_resident(mixed,weights_.moe,routes,probabilities,block,workspace_,stream,ops::LinearPolicy::A16Only);
    capture(MtpTraceStage::MoeIds,routes);capture(MtpTraceStage::MoeWeights,probabilities);capture(MtpTraceStage::Moe,block);
    ops::gated_residual_inject(residual,block,scale,residual,stream);capture(MtpTraceStage::MoeInjected,residual);
    const auto& f=weights_.final_gr;
    ops::gated_residual_read(residual,f.norm,f.down,f.up,logit,workspace_,stream);capture(MtpTraceStage::FinalRead,logit);
    return {residual.view({F,width,batch}),logit.view({D,width,batch})};
}
} // namespace ninfer::targets::qwen4

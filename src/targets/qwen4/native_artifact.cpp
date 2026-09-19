#include "targets/qwen4/native_artifact.h"
#include "artifact/typed_binding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ninfer::targets::qwen4 {
namespace {
using Format=artifact::NumericFormat;
constexpr int D=2560,F=10240;
const std::string Main="model.language_model.";

bool fp8_role(const std::string& name) {
    for(int layer:{0,3}) for(const char* role:{"gate","up","down"})
        if(name==Main+"layers."+std::to_string(layer)+".mlp.shared_expert."+role+"_proj.weight") return true;
    for(const char* role:{"in_proj_qkv","in_proj_z","out_proj"})
        if(name==Main+"layers.0.linear_attn."+role+".weight") return true;
    for(const char* role:{"q_proj","k_proj","v_proj","o_proj"})
        if(name==Main+"layers.3.self_attn."+role+".weight") return true;
    return false;
}

class Bind {
public:
    explicit Bind(const artifact::Reader& reader):reader(reader),binder(reader) {}
    void tensor(const std::string& name,std::initializer_list<std::uint64_t> shape,
                Format format=Format::BF16,artifact::TensorPlacement placement=artifact::TensorPlacement::Device) {
        const auto handle=artifact::bind_tensor(binder,name,format,shape,placement);
        result.tensors.emplace(name,NativeArtifactTensor{handle,format,std::vector<std::uint64_t>(shape)});
    }
    void matrix(const std::string& name,int n,int k) {
        Format format=Format::BF16;
        if(fp8_role(name)) {
            const auto* descriptor=reader.find(name);
            const auto* t=descriptor?std::get_if<artifact::TensorDescriptor>(descriptor):nullptr;
            if(t && t->format==Format::FP8_E4M3FN_TENSOR_F32M) format=t->format;
        }
        tensor(name,{std::uint64_t(n),std::uint64_t(k)},format);
    }
    void gr(const std::string& p,bool inject) {
        tensor(p+"hc_norm.weight",{F},Format::FP32);
        matrix(p+"input_mix_weight_down.weight",320,F);matrix(p+"input_mix_weight_up.weight",F,320);
        if(inject) tensor(p+"block_inject_weight.weight",{4,F},Format::FP32);
    }
    void moe(const std::string& p) {
        matrix(p+"gate.weight",512,D);tensor(p+"shared_expert_gate.weight",{1,D});
        for(const std::string role:{"gate","up","down"}) {
            const int n=role=="down"?D:640,k=role=="down"?640:D;
            matrix(p+"shared_expert."+role+"_proj.weight",n,k);
            tensor(p+"experts."+role+"_proj.weight",{512,std::uint64_t(n),std::uint64_t(k)},Format::NVFP4_EXPERT_F32M);
        }
    }
    void qsa(const std::string& p) {
        matrix(p+"indexer.index_qk_proj.weight",640,D);
        tensor(p+"indexer.q_layernorm.weight",{128},Format::FP32);
        tensor(p+"indexer.k_layernorm.weight",{128},Format::FP32);
        tensor(p+"q_norm.weight",{256},Format::FP32);tensor(p+"k_norm.weight",{256},Format::FP32);
        matrix(p+"q_proj.weight",12288,D);matrix(p+"k_proj.weight",512,D);
        matrix(p+"v_proj.weight",512,D);matrix(p+"o_proj.weight",D,6144);
    }
    void gdn(const std::string& p) {
        matrix(p+"in_proj_qkv.weight",F,D);matrix(p+"in_proj_z.weight",6144,D);matrix(p+"out_proj.weight",D,6144);
        tensor(p+"in_proj_a.weight",{48,D},Format::FP32);tensor(p+"in_proj_b.weight",{48,D},Format::FP32);
        tensor(p+"conv1d.weight",{F,1,4},Format::FP32);tensor(p+"ssm_a",{48},Format::FP32);
        tensor(p+"dt_bias",{48},Format::FP32);tensor(p+"norm.weight",{128},Format::FP32);
    }
    const artifact::Reader& reader;
    artifact::Binder binder;
    NativeArtifactPlan result;
};

constexpr std::array<std::tuple<const char*,int,int>,7> DraftMatrices{{
    {"self_attn.q_proj.weight",6144,D},{"self_attn.k_proj.weight",512,D},
    {"self_attn.v_proj.weight",512,D},{"self_attn.o_proj.weight",D,6144},
    {"mlp.gate_proj.weight",7680,D},{"mlp.up_proj.weight",7680,D},{"mlp.down_proj.weight",D,7680}}};
constexpr std::array<std::tuple<const char*,int,int>,4> VisionMatrices{{
    {"attn.qkv",3456,1152},{"attn.proj",1152,1152},
    {"mlp.linear_fc1",4304,1152},{"mlp.linear_fc2",1152,4304}}};

class Views {
public:
    Views(const artifact::Reader& reader,const NativeArtifactPlan& plan,const artifact::MaterializedArtifact& backing)
        :reader(reader),plan(plan),backing(backing) {}
    Tensor tensor(const std::string& name,std::initializer_list<int> shape) const {
        const auto& t=plan.tensors.at(name);
        return artifact::materialized_tensor(backing,t.handle,t.format,shape);
    }
    Weight matrix(const std::string& name,int n,int k) const {
        const auto& t=plan.tensors.at(name);
        if(t.format!=Format::NVFP4) return artifact::materialized_weight(backing,t.handle,t.format,n,k);
        const std::array<std::uint64_t,2> shape{std::uint64_t(n),std::uint64_t(k)};
        const auto geometry=artifact::block_scale_geometry(t.format,shape);
        float divisor;
        std::memcpy(&divisor,reader.payload(*reader.find(name)).data.data()+geometry.weight_divisor_offset,4);
        if(!(divisor>0) || !std::isfinite(divisor)) throw std::invalid_argument("invalid native DFlash NVFP4 divisor");
        const auto* data=static_cast<const std::byte*>(backing.device_data(t.handle));
        Weight w{};w.payload=w.qdata=data;w.payload_bytes=geometry.encoded_bytes;
        w.scales=data+geometry.scale_plane_offset;w.qtype=QType::NVFP4;
        w.layout=QuantLayout::BlockScaleK16M128x4;w.scale_dtype=DType::FP8_E4M3FN;
        w.group=w.group_size=16;w.ndim=2;w.n=w.shape[0]=w.padded_shape[0]=n;w.k=w.shape[1]=w.padded_shape[1]=k;
        w.weight_scale_divisor=divisor;w.input_scale_divisor=1;return w;
    }
    NativeGrWeights gr(const std::string& p,bool inject) const {
        return {tensor(p+"hc_norm.weight",{F}),inject?tensor(p+"block_inject_weight.weight",{F,4}):Tensor{},
            matrix(p+"input_mix_weight_down.weight",320,F),matrix(p+"input_mix_weight_up.weight",F,320)};
    }
    ops::Qwen4ResidentSparseMoeWeights moe(const std::string& p) const {
        auto bank=[&](const char* role,int n,int k) {
            return artifact::materialized_nvfp4_expert_weight(backing,plan.tensors.at(p+"experts."+role+"_proj.weight").handle,512,n,k);
        };
        return {matrix(p+"gate.weight",512,D),bank("gate",640,D),bank("up",640,D),bank("down",D,640),
            tensor(p+"shared_expert_gate.weight",{D}),matrix(p+"shared_expert.gate_proj.weight",640,D),
            matrix(p+"shared_expert.up_proj.weight",640,D),matrix(p+"shared_expert.down_proj.weight",D,640)};
    }
    ops::QsaVerifierWeights qsa(const std::string& p) const {
        Weight iq=matrix(p+"indexer.index_qk_proj.weight",640,D),ik=iq;
        iq.n=iq.shape[0]=iq.padded_shape[0]=512;iq.payload_bytes=512ULL*D*2;
        ik.n=ik.shape[0]=ik.padded_shape[0]=128;ik.payload_bytes=128ULL*D*2;
        ik.payload=ik.qdata=static_cast<const std::byte*>(ik.payload)+512ULL*D*2;
        return {iq,ik,matrix(p+"q_proj.weight",12288,D),matrix(p+"k_proj.weight",512,D),
            matrix(p+"v_proj.weight",512,D),matrix(p+"o_proj.weight",D,6144),
            tensor(p+"indexer.q_layernorm.weight",{128}),tensor(p+"indexer.k_layernorm.weight",{128}),
            tensor(p+"q_norm.weight",{256}),tensor(p+"k_norm.weight",{256})};
    }
    ops::GatedDeltaNetLayerWeights gdn(const std::string& p) const {
        return {matrix(p+"in_proj_qkv.weight",F,D),matrix(p+"in_proj_z.weight",6144,D),
            tensor(p+"in_proj_a.weight",{D,48}),tensor(p+"in_proj_b.weight",{D,48}),
            tensor(p+"conv1d.weight",{4,F}),tensor(p+"ssm_a",{48}),tensor(p+"dt_bias",{48}),
            tensor(p+"norm.weight",{128}),matrix(p+"out_proj.weight",D,6144)};
    }
    const artifact::Reader& reader;
    const NativeArtifactPlan& plan;
    const artifact::MaterializedArtifact& backing;
};
} // namespace

NativeArtifactPlan bind_native_artifact(const artifact::Reader& reader) {
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-preview","nvfp4-a16"})
        throw std::invalid_argument("requires the complete exact native preview artifact");
    Bind b(reader);
    for(int layer=0;layer<48;++layer) {
        const auto p=Main+"layers."+std::to_string(layer)+".";
        b.gr(p+"attn_hyper_connection.",true);b.gr(p+"mlp_hyper_connection.",true);b.moe(p+"mlp.");
        if(layer%4==3) b.qsa(p+"self_attn.");else b.gdn(p+"linear_attn.");
    }
    b.matrix(Main+"embed_tokens.weight",248320,D);b.matrix("lm_head.weight",248320,D);
    b.gr(Main+"hyper_connection_mixer.",false);
    const auto ple=Main+"layers.1.ple.";
    b.matrix(ple+"key_proj.weight",F,D);b.matrix(ple+"value_proj.weight",D,D);
    for(const char* role:{"norm_key","norm_query","norm_conv"}) b.tensor(ple+role+".weight",{F});
    b.tensor(ple+"conv1d.weight",{F,1,4});
    const auto* p=reader.find("ple.table");
    const auto* table=p?std::get_if<artifact::TensorDescriptor>(p):nullptr;
    if(table && table->format==Format::FP8_E4M3FN_TENSOR_BF16S) {
        b.result.ple_format=NativePleFormat::Fp8;
        b.tensor("ple.table",{320001536,160},table->format,artifact::TensorPlacement::ResidentHost);
    } else b.tensor("ple.table",{128,2500012,160},Format::NVFP4_PARTITION_F32M,artifact::TensorPlacement::ResidentHost);
    const std::string v="model.visual.";
    b.tensor(v+"patch_embed.proj.weight",{1152,3,2,16,16});b.tensor(v+"patch_embed.proj.bias",{1152});
    b.tensor(v+"pos_embed.weight",{2304,1152});
    for(int layer=0;layer<27;++layer) {
        const auto p=v+"blocks."+std::to_string(layer)+".";
        for(const char* role:{"norm1.weight","norm1.bias","norm2.weight","norm2.bias"}) b.tensor(p+role,{1152});
        for(auto [role,n,k]:VisionMatrices) {b.matrix(p+role+".weight",n,k);b.tensor(p+role+".bias",{std::uint64_t(n)});}
    }
    b.tensor(v+"merger.norm.weight",{1152});b.tensor(v+"merger.norm.bias",{1152});
    b.matrix(v+"merger.linear_fc1.weight",4608,4608);b.tensor(v+"merger.linear_fc1.bias",{4608});
    b.matrix(v+"merger.linear_fc2.weight",D,4608);b.tensor(v+"merger.linear_fc2.bias",{D});
    b.matrix("mtp.fc_embedding.weight",D,D);b.matrix("mtp.fc_hidden.weight",D,D);
    b.tensor("mtp.pre_fc_norm_embedding.weight",{D});b.tensor("mtp.pre_fc_norm_hidden.weight",{F});
    b.gr("mtp.layers.0.attn_hyper_connection.",true);b.gr("mtp.layers.0.mlp_hyper_connection.",true);
    b.gr("mtp.hyper_connection_mixer.",false);b.qsa("mtp.layers.0.self_attn.");b.moe("mtp.layers.0.mlp.");
    if(reader.find("dflash.fc.weight")) {
        b.result.dflash=true;
        const auto* descriptor=std::get_if<artifact::TensorDescriptor>(reader.find("dflash.fc.weight"));
        if(!descriptor || (descriptor->format!=Format::BF16 && descriptor->format!=Format::NVFP4))
            throw std::invalid_argument("native DFlash requires BF16 or NVFP4 matrices");
        b.result.dflash_format=descriptor->format;
        b.tensor("dflash.fc.weight",{D,12800},descriptor->format);
        b.tensor("dflash.hidden_norm.weight",{D});b.tensor("dflash.norm.weight",{D});
        for(int layer=0;layer<5;++layer) {
            const auto p="dflash.layers."+std::to_string(layer)+".";
            b.tensor(p+"input_layernorm.weight",{D});b.tensor(p+"post_attention_layernorm.weight",{D});
            b.tensor(p+"self_attn.q_norm.weight",{256});b.tensor(p+"self_attn.k_norm.weight",{256});
            for(auto [role,n,k]:DraftMatrices) b.tensor(p+role,{std::uint64_t(n),std::uint64_t(k)},descriptor->format);
        }
    }
    b.result.frontend=text::qwen::bind_frontend_resources(b.binder);
    b.result.resources.emplace("native-profile.json",artifact::bind_raw_resource(b.binder,"native-profile.json"));
    b.result.materialization=b.binder.finish();
    return std::move(b.result);
}

void admit_native_device(const NativeArtifactPlan& plan,std::uint64_t available,std::uint64_t reserve) {
    if(reserve>available || plan.materialization.device_capacity_bytes>available-reserve)
        throw std::length_error("native Qwen4 GPU-resident compute and startup reserve exceed available VRAM; native compute streaming is not supported");
}

NativePleTable native_ple_table_view(std::span<const std::byte> bytes,NativePleFormat format) {
    NativePleTable result;result.format=format;
    if(format==NativePleFormat::Nvfp4) {
        if(bytes.size()!=28800138752ULL) throw std::invalid_argument("native NVFP4 PLE encoded extent");
        result.nvfp4={reinterpret_cast<const std::uint8_t*>(bytes.data()),128,2500012,bytes.size()};
    } else {
        constexpr std::uint64_t codes=320001536ULL*160;
        if(bytes.size()!=codes+2) throw std::invalid_argument("native FP8 PLE encoded extent");
        const auto low=std::to_integer<unsigned>(bytes[codes]),high=std::to_integer<unsigned>(bytes[codes+1]);
        result.fp8_scale_bits=std::uint16_t(low|(high<<8));
        if((result.fp8_scale_bits&0x8000U) || (result.fp8_scale_bits&0x7fffU)==0 ||
           (result.fp8_scale_bits&0x7f80U)==0x7f80U)
            throw std::invalid_argument("native FP8 PLE scalar must be positive finite BF16");
        result.fp8={reinterpret_cast<const std::uint8_t*>(bytes.data()),320001536,codes};
    }
    return result;
}

std::unique_ptr<LoadedNativeModel> LoadedNativeModel::load(const std::filesystem::path& path,
    DeviceContext& device,std::uint64_t reserve) {
    artifact::Reader reader(path);
    const auto plan=bind_native_artifact(reader);
    std::size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));
    admit_native_device(plan,free,reserve);
    auto result=std::unique_ptr<LoadedNativeModel>(new LoadedNativeModel);
    result->backing_=artifact::materialize(reader,plan.materialization,device);
    result->resources_=plan.resources;
    result->frontend_=text::qwen::take_frontend_resources(result->backing_,plan.frontend);
    const Views r(reader,plan,result->backing_);auto& w=result->view_;
    for(int layer=0;layer<48;++layer) {
        const auto p=Main+"layers."+std::to_string(layer)+".";auto& l=w.layers[layer];
        l.attention_gr=r.gr(p+"attn_hyper_connection.",true);l.moe_gr=r.gr(p+"mlp_hyper_connection.",true);
        l.moe=r.moe(p+"mlp.");if(layer%4==3) l.qsa=r.qsa(p+"self_attn.");else l.gdn=r.gdn(p+"linear_attn.");
    }
    w.token_embedding=r.matrix(Main+"embed_tokens.weight",248320,D);w.output_head=r.matrix("lm_head.weight",248320,D);
    w.final_gr=r.gr(Main+"hyper_connection_mixer.",false);
    const auto p=Main+"layers.1.ple.";
    w.ple={r.matrix(p+"key_proj.weight",F,D),r.matrix(p+"value_proj.weight",D,D),
        r.tensor(p+"norm_key.weight",{F}),r.tensor(p+"norm_query.weight",{F}),r.tensor(p+"norm_conv.weight",{F}),
        r.tensor(p+"conv1d.weight",{4,F})};
    const auto table=result->backing_.mapped_tensor_bytes(plan.tensors.at("ple.table").handle);
    w.ple_table=native_ple_table_view(table,plan.ple_format);
    const std::string v="model.visual.";
    w.vision.patch=r.matrix(v+"patch_embed.proj.weight",1152,1536);
    w.vision.patch_bias=r.tensor(v+"patch_embed.proj.bias",{1152});w.vision.positions=r.tensor(v+"pos_embed.weight",{1152,2304});
    for(int layer=0;layer<27;++layer) {
        const auto p=v+"blocks."+std::to_string(layer)+".";auto& b=w.vision.blocks[layer];
        b.norm1_weight=r.tensor(p+"norm1.weight",{1152});b.norm1_bias=r.tensor(p+"norm1.bias",{1152});
        b.norm2_weight=r.tensor(p+"norm2.weight",{1152});b.norm2_bias=r.tensor(p+"norm2.bias",{1152});
        std::array<Weight*,4> matrices{&b.qkv,&b.output,&b.up,&b.down};
        std::array<Tensor*,4> biases{&b.qkv_bias,&b.output_bias,&b.up_bias,&b.down_bias};
        for(std::size_t i=0;i<VisionMatrices.size();++i) {
            auto [role,n,k]=VisionMatrices[i];*matrices[i]=r.matrix(p+role+".weight",n,k);*biases[i]=r.tensor(p+role+".bias",{n});
        }
    }
    w.vision.merger={r.tensor(v+"merger.norm.weight",{1152}),r.tensor(v+"merger.norm.bias",{1152}),
        r.matrix(v+"merger.linear_fc1.weight",4608,4608),r.tensor(v+"merger.linear_fc1.bias",{4608}),
        r.matrix(v+"merger.linear_fc2.weight",D,4608),r.tensor(v+"merger.linear_fc2.bias",{D})};
    auto mg=[&](const std::string& p,bool inject) {const auto g=r.gr(p,inject);return MtpGrWeights{g.norm,g.inject,g.down,g.up};};
    w.mtp.embedding_norm=r.tensor("mtp.pre_fc_norm_embedding.weight",{D});w.mtp.hidden_norm=r.tensor("mtp.pre_fc_norm_hidden.weight",{F});
    w.mtp.embedding=r.matrix("mtp.fc_embedding.weight",D,D);w.mtp.hidden=r.matrix("mtp.fc_hidden.weight",D,D);
    w.mtp.attention_gr=mg("mtp.layers.0.attn_hyper_connection.",true);w.mtp.moe_gr=mg("mtp.layers.0.mlp_hyper_connection.",true);
    w.mtp.final_gr=mg("mtp.hyper_connection_mixer.",false);w.mtp.attention=r.qsa("mtp.layers.0.self_attn.");w.mtp.moe=r.moe("mtp.layers.0.mlp.");
    if(plan.dflash) {
        auto& d=w.dflash.emplace();d.feature=r.matrix("dflash.fc.weight",D,12800);
        d.hidden_norm=r.tensor("dflash.hidden_norm.weight",{D});d.final_norm=r.tensor("dflash.norm.weight",{D});
        for(int layer=0;layer<5;++layer) {
            const auto p="dflash.layers."+std::to_string(layer)+".";auto& b=d.layers[layer];
            b.input_norm=r.tensor(p+"input_layernorm.weight",{D});b.post_norm=r.tensor(p+"post_attention_layernorm.weight",{D});
            b.query_norm=r.tensor(p+"self_attn.q_norm.weight",{256});b.key_norm=r.tensor(p+"self_attn.k_norm.weight",{256});
            std::array<Weight*,7> matrices{&b.query,&b.key,&b.value,&b.output,&b.gate,&b.up,&b.down};
            for(std::size_t i=0;i<DraftMatrices.size();++i) {auto [role,n,k]=DraftMatrices[i];*matrices[i]=r.matrix(p+role,n,k);}
        }
    }
    return result;
}

std::span<const std::byte> LoadedNativeModel::resource(std::string_view name) const {
    return backing_.resource_bytes(resources_.at(std::string(name)));
}
} // namespace ninfer::targets::qwen4

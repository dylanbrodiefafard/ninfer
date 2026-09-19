#include "targets/qwen4/native_artifact.h"
#include "artifact_fixture.h"
#include "ninfer/engine.h"

#include <iostream>
#include <unistd.h>

using namespace ninfer;
namespace q4=ninfer::targets::qwen4;
namespace {
using Json=nlohmann::json;
using F=artifact::NumericFormat;
using L=artifact::StorageLayout;

// Independent exact source-shape directory witness, never an executable checkpoint. Sparse
// payload holes permit full-inventory binding/admission checks without downloading 100+ GB.
Json directory(bool fp8_ple,bool draft,bool fp8_projection) {
    Json objects=Json::array();std::uint64_t offset=0;
    auto tensor=[&](const std::string& name,std::initializer_list<std::uint64_t> shape,F format=F::BF16) {
        L layout=L::ContiguousLeV1;
        if(format==F::NVFP4_EXPERT_F32M) layout=L::ExpertBlockScaleK16M128x4V1;
        if(format==F::NVFP4_PARTITION_F32M) layout=L::PartitionedRowBlockScaleK16V1;
        if(format==F::FP8_E4M3FN_TENSOR_BF16S) layout=L::TensorScaleV1;
        if(format==F::FP8_E4M3FN_TENSOR_F32M) layout=L::TensorCalibratedV1;
        if(format==F::NVFP4) layout=L::BlockScaleK16M128x4V1;
        offset=test::artifact_fixture::align_up(offset,artifact::tensor_alignment(layout));
        const auto bytes=artifact::tensor_encoded_size(layout,format,std::span(shape));
        objects.push_back({{"name",name},{"kind","tensor"},{"shape",std::vector(shape)},
            {"format",artifact::format_name(format)},{"layout",artifact::layout_name(layout)},
            {"offset",offset},{"bytes",bytes}});offset+=bytes;
    };
    auto gr=[&](const std::string& p,bool inject) {
        tensor(p+"hc_norm.weight",{10240},F::FP32);
        tensor(p+"input_mix_weight_down.weight",{320,10240});tensor(p+"input_mix_weight_up.weight",{10240,320});
        if(inject) tensor(p+"block_inject_weight.weight",{4,10240},F::FP32);
    };
    auto moe=[&](const std::string& p) {
        tensor(p+"gate.weight",{512,2560});tensor(p+"shared_expert_gate.weight",{1,2560});
        tensor(p+"shared_expert.gate_proj.weight",{640,2560});tensor(p+"shared_expert.up_proj.weight",{640,2560});
        tensor(p+"shared_expert.down_proj.weight",{2560,640});
        tensor(p+"experts.gate_proj.weight",{512,640,2560},F::NVFP4_EXPERT_F32M);
        tensor(p+"experts.up_proj.weight",{512,640,2560},F::NVFP4_EXPERT_F32M);
        tensor(p+"experts.down_proj.weight",{512,2560,640},F::NVFP4_EXPERT_F32M);
    };
    auto qsa=[&](const std::string& p) {
        tensor(p+"q_proj.weight",{12288,2560});tensor(p+"k_proj.weight",{512,2560});
        tensor(p+"v_proj.weight",{512,2560});tensor(p+"o_proj.weight",{2560,6144});
        tensor(p+"q_norm.weight",{256},F::FP32);tensor(p+"k_norm.weight",{256},F::FP32);
        tensor(p+"indexer.index_qk_proj.weight",{640,2560});
        tensor(p+"indexer.q_layernorm.weight",{128},F::FP32);tensor(p+"indexer.k_layernorm.weight",{128},F::FP32);
    };
    const std::string main="model.language_model.";
    for(int i=0;i<48;++i) {
        const auto p=main+"layers."+std::to_string(i)+".";
        gr(p+"attn_hyper_connection.",true);gr(p+"mlp_hyper_connection.",true);moe(p+"mlp.");
        if(i%4==3) qsa(p+"self_attn.");else {
            const auto g=p+"linear_attn.";
            tensor(g+"in_proj_qkv.weight",{10240,2560},i==0 && fp8_projection?F::FP8_E4M3FN_TENSOR_F32M:F::BF16);
            tensor(g+"in_proj_z.weight",{6144,2560});tensor(g+"out_proj.weight",{2560,6144});
            tensor(g+"in_proj_a.weight",{48,2560},F::FP32);tensor(g+"in_proj_b.weight",{48,2560},F::FP32);
            tensor(g+"conv1d.weight",{10240,1,4},F::FP32);tensor(g+"ssm_a",{48},F::FP32);
            tensor(g+"dt_bias",{48},F::FP32);tensor(g+"norm.weight",{128},F::FP32);
        }
    }
    tensor(main+"embed_tokens.weight",{248320,2560});tensor("lm_head.weight",{248320,2560});
    gr(main+"hyper_connection_mixer.",false);
    const auto ple=main+"layers.1.ple.";
    tensor(ple+"key_proj.weight",{10240,2560});tensor(ple+"value_proj.weight",{2560,2560});
    for(const char* n:{"norm_key","norm_query","norm_conv"}) tensor(ple+n+".weight",{10240});
    tensor(ple+"conv1d.weight",{10240,1,4});
    if(fp8_ple) tensor("ple.table",{320001536,160},F::FP8_E4M3FN_TENSOR_BF16S);
    else tensor("ple.table",{128,2500012,160},F::NVFP4_PARTITION_F32M);
    const std::string v="model.visual.";
    tensor(v+"patch_embed.proj.weight",{1152,3,2,16,16});tensor(v+"patch_embed.proj.bias",{1152});tensor(v+"pos_embed.weight",{2304,1152});
    for(int i=0;i<27;++i) {
        const auto p=v+"blocks."+std::to_string(i)+".";
        for(const char* role:{"norm1.weight","norm1.bias","norm2.weight","norm2.bias"}) tensor(p+role,{1152});
        tensor(p+"attn.qkv.weight",{3456,1152});tensor(p+"attn.qkv.bias",{3456});
        tensor(p+"attn.proj.weight",{1152,1152});tensor(p+"attn.proj.bias",{1152});
        tensor(p+"mlp.linear_fc1.weight",{4304,1152});tensor(p+"mlp.linear_fc1.bias",{4304});
        tensor(p+"mlp.linear_fc2.weight",{1152,4304});tensor(p+"mlp.linear_fc2.bias",{1152});
    }
    tensor(v+"merger.norm.weight",{1152});tensor(v+"merger.norm.bias",{1152});
    tensor(v+"merger.linear_fc1.weight",{4608,4608});tensor(v+"merger.linear_fc1.bias",{4608});
    tensor(v+"merger.linear_fc2.weight",{2560,4608});tensor(v+"merger.linear_fc2.bias",{2560});
    tensor("mtp.fc_embedding.weight",{2560,2560});tensor("mtp.fc_hidden.weight",{2560,2560});
    tensor("mtp.pre_fc_norm_embedding.weight",{2560});tensor("mtp.pre_fc_norm_hidden.weight",{10240});
    gr("mtp.layers.0.attn_hyper_connection.",true);gr("mtp.layers.0.mlp_hyper_connection.",true);
    gr("mtp.hyper_connection_mixer.",false);qsa("mtp.layers.0.self_attn.");moe("mtp.layers.0.mlp.");
    if(draft) {
        tensor("dflash.fc.weight",{2560,12800},F::NVFP4);tensor("dflash.hidden_norm.weight",{2560});tensor("dflash.norm.weight",{2560});
        for(int i=0;i<5;++i) {
            const auto p="dflash.layers."+std::to_string(i)+".";
            tensor(p+"input_layernorm.weight",{2560});tensor(p+"post_attention_layernorm.weight",{2560});
            tensor(p+"self_attn.q_norm.weight",{256});tensor(p+"self_attn.k_norm.weight",{256});
            tensor(p+"self_attn.q_proj.weight",{6144,2560},F::NVFP4);tensor(p+"self_attn.k_proj.weight",{512,2560},F::NVFP4);
            tensor(p+"self_attn.v_proj.weight",{512,2560},F::NVFP4);tensor(p+"self_attn.o_proj.weight",{2560,6144},F::NVFP4);
            tensor(p+"mlp.gate_proj.weight",{7680,2560},F::NVFP4);tensor(p+"mlp.up_proj.weight",{7680,2560},F::NVFP4);
            tensor(p+"mlp.down_proj.weight",{2560,7680},F::NVFP4);
        }
    }
    for(const char* name:{"frontend/tokenizer.json","frontend/tokenizer_config.json","frontend/chat_template.jinja",
                         "frontend/generation_config.json","frontend/preprocessor_config.json","frontend/video_preprocessor_config.json","native-profile.json"}) {
        objects.push_back({{"name",name},{"kind","resource"},{"encoding","raw-bytes-v1"},{"offset",offset},{"bytes",2}});offset+=2;
    }
    return {{"identity",{{"model_id","qwen4/native-preview"},{"weights_id","nvfp4-a16"}}},{"objects",objects}};
}

test::artifact_fixture::TemporaryArtifact sparse(const Json& doc,std::uint16_t fp8_scale=0x3e80) {
    const auto path=std::filesystem::temp_directory_path()/("ninfer_qwen4_native_metadata_"+std::to_string(getpid())+
        "_"+std::to_string(fp8_scale)+".ninfer");
    if(std::filesystem::exists(path)) throw std::runtime_error("metadata witness path already exists");
    const auto text=doc.dump();const auto base=test::artifact_fixture::align_up(text.size()+16,4096);
    std::uint64_t end=0;for(const auto& o:doc["objects"]) end=std::max(end,o["offset"].get<std::uint64_t>()+o["bytes"].get<std::uint64_t>());
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(test::artifact_fixture::kMagic.data()),8);
    std::array<std::byte,8> length{};test::artifact_fixture::write_u64_le(length.data(),text.size());
    file.write(reinterpret_cast<const char*>(length.data()),8);file<<text;
    file.seekp(base+end-1);file.put(0);
    for(const auto& object:doc["objects"]) if(object["name"]=="ple.table"&&
        object["format"]==artifact::format_name(F::FP8_E4M3FN_TENSOR_BF16S)) {
        file.seekp(base+object["offset"].get<std::uint64_t>()+object["bytes"].get<std::uint64_t>()-2);
        file.put(char(fp8_scale&255));file.put(char(fp8_scale>>8));
    }
    if(!file) throw std::runtime_error("failed to create sparse metadata-only witness");
    return {path};
}

bool rejected(Json doc) {
    auto fixture=sparse(doc);
    try {artifact::Reader reader(fixture.path);(void)q4::bind_native_artifact(reader);return false;}
    catch(const std::invalid_argument&) {return true;}
    catch(const artifact::ArtifactError&) {return true;}
}
}

int main() {
    try {
        int failures=0;
        for(bool fp8:{false,true}) {
            auto fixture=sparse(directory(fp8,fp8,fp8));artifact::Reader reader(fixture.path);
            const auto plan=q4::bind_native_artifact(reader);
            failures+=plan.tensors.size()!=std::size_t(fp8?1635:1577);
            failures+=plan.dflash!=fp8;
            failures+=plan.materialization.mapped_tensor_objects.size()!=1 || !plan.materialization.mapped_tensor_objects[0].resident;
            failures+=plan.materialization.mapped_tensor_objects[0].bytes!=(fp8?51200245762ULL:28800138752ULL);
            failures+=plan.materialization.device_objects.size()!=plan.tensors.size()-1;
            if(fp8) {
                const auto table=q4::native_ple_table_view(reader.payload("ple.table").data,
                    q4::NativePleFormat::Fp8);
                failures+=table.fp8_scale_bits!=0x3e80||table.fp8.bytes!=51200245760ULL;
                auto bad=sparse(directory(true,true,true),0x7f80);artifact::Reader bad_reader(bad.path);
                const auto bad_plan=q4::bind_native_artifact(bad_reader);
                try {
                    (void)q4::native_ple_table_view(bad_reader.payload("ple.table").data,
                        q4::NativePleFormat::Fp8);++failures;
                } catch(const std::invalid_argument&) {}
            }
            try {q4::admit_native_device(plan,32ULL<<30,256ULL<<20);++failures;}catch(const std::length_error&) {}
            q4::admit_native_device(plan,plan.materialization.device_capacity_bytes+(256ULL<<20),256ULL<<20);
            EngineOptions options;options.artifact_path=fixture.path;options.max_context=64;
            options.kv_capacity=KvCapacityPolicy::explicit_capacity(64);options.prefill_chunk=8;
            // The real registered Engine must reject this complete source-sized metadata
            // before touching sparse payload holes, allocating model VRAM or locking PLE.
            try {Engine engine(options);++failures;}
            catch(const std::invalid_argument& error) {
                failures+=std::string(error.what()).find("device memory")==std::string::npos;
            }
            std::cout<<"Complete native metadata: "<<plan.tensors.size()<<" tensors, "
                <<plan.materialization.device_capacity_bytes<<" GPU bytes; 32GiB rejected before allocation\n";
        }
        auto missing=directory(false,false,false);
        for(auto it=missing["objects"].begin();it!=missing["objects"].end();++it)
            if((*it)["name"]=="model.language_model.layers.47.self_attn.o_proj.weight") {missing["objects"].erase(it);break;}
        failures+=!rejected(missing);
        auto swapped=directory(false,false,false);
        for(auto& obj:swapped["objects"]) if(obj["name"]=="model.language_model.layers.46.linear_attn.norm.weight")
            obj["name"]="model.language_model.layers.46.self_attn.q_norm.weight";
        failures+=!rejected(swapped);
        auto protected_weight=directory(false,false,false);
        for(auto& obj:protected_weight["objects"]) if(obj["name"]=="model.language_model.layers.47.self_attn.indexer.index_qk_proj.weight") {
            obj["shape"]={1280,1280}; // Same byte extent, wrong logical geometry.
        }
        failures+=!rejected(protected_weight);
        std::cout<<"Native complete artifact metadata/admission failures="<<failures<<'\n';return failures?1:0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

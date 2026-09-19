// Opt-in diagnostic: authentic full-target prompt features, followed by unload and
// same-input BF16/NVFP4 drafter comparison. No speculative verifier execution or PPL claim.
#include "targets/qwen4/dflash.h"
#include "dflash_test_cache.h"
#include "targets/qwen4/verifier.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_text_panel.h"
#include "ninfer/ops/linear.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace q4 = ninfer::targets::qwen4;
namespace {
constexpr int D=2560, F=12800, V=248320, T=24, K=7;

void write_json(const std::filesystem::path& path,const nlohmann::json& value) {
    if(std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite diagnostic report");
    std::ofstream output(path); output<<value.dump(2)<<'\n';
    if(!output) throw std::runtime_error("failed to write diagnostic report");
}

void capture(const std::filesystem::path& target,const std::filesystem::path& native,
             const std::filesystem::path& output,DeviceContext& device) {
    const auto binary=output/"qwen4-dflash-target-features.bin";
    if(std::filesystem::exists(binary) || std::filesystem::exists(output/"qwen4-dflash-target-features.json"))
        throw std::runtime_error("refusing to overwrite accepted target feature panel");
    qwen4_sequence::TextPanel panel(native.string());
    std::vector<std::uint16_t> features(F*T);
    auto slot=to_device_i32({0}),valid=to_device_i32({1});
    DeviceBuffer feature(F*2);
    q4::DFlashFeatureSink sink{Tensor(feature.p,DType::BF16,{F,1,1}),
        Tensor(slot.p,DType::I32,{1}),Tensor(valid.p,DType::I32,{1})};
    {
        auto model=q4::verifier::LoadedModel::load(target,device);
        q4::verifier::Program program(*model,device,q4::verifier::DiagnosticSnapshots::Disabled);
        program.reset();
        for(int t=0;t<T;++t) {
            const auto result=program.execute_token(panel.tokens[t],panel.tokens[t+1],&sink);
            (void)result;
            device.synchronize();
            feature.copy_to_host(features.data()+t*F,F*2);
            if(program.frontier()!=t+1) throw std::runtime_error("accepted prompt frontier mismatch");
            std::cout<<"Captured accepted prompt token "<<t+1<<'/'<<T<<std::endl;
        }
        device.synchronize();
    } // All target weights, host locks, expert staging and continuation drain before drafter load.
    for(auto value:features) if(!std::isfinite(bf16_to_f32(value)))
        throw std::runtime_error("nonfinite target feature");
    std::ofstream file(binary,std::ios::binary);
    file.write(reinterpret_cast<const char*>(features.data()),features.size()*2);
    if(!file) throw std::runtime_error("failed to write target features");
    write_json(output/"qwen4-dflash-target-features.json",{
        {"profile","qwen4-ud-iq1-s-diagnostic-accepted-prompt"},{"target",target.string()},
        {"features","BF16 little-endian [24,12800], column-major logical [12800,24]"},
        {"taps",{4,16,24,36,44}},{"token_ids",panel.tokens},
        {"accepted_prompt_tokens",T},{"qsa_kv","NVFP4-G16"},
        {"boundary","Sequential teacher-forced prompt tokens, not generated-token acceptance. The existing full48 diagnostic target produces the attention-GR input taps. PLE is fully populated and locked before load succeeds; ordinary diagnostic expert streaming is unchanged. Not native NVFP4 target features."}});
}

double relative_l2(const std::vector<double>& reference,const std::vector<double>& value) {
    double error=0,norm=0;
    for(std::size_t i=0;i<reference.size();++i) {
        if(!std::isfinite(reference[i]) || !std::isfinite(value[i])) throw std::runtime_error("nonfinite drafter output");
        const double delta=double(value[i])-reference[i];error+=delta*delta;norm+=double(reference[i])*reference[i];
    }
    return std::sqrt(error/std::max(norm,1e-300));
}

struct Result { std::vector<double> hidden,logits; };
Result run(const std::filesystem::path& root,const std::string& profile,const Weight& head,
           const std::vector<std::uint16_t>& features,const std::vector<std::uint16_t>& embeddings,
           int context,DeviceContext& device) {
    auto model=q4::LoadedDFlash::load(root/("qwen4-dflash-"+profile+".ninfer"),device);
    ninfer::test::qwen4::DFlashTestCache cache(32,1,device.stream);
    q4::DFlashProgram program(model->weights(),32,T,1,cache.views());
    auto df=to_device(features),de=to_device(embeddings),dn=to_device_i32({context}),
        dv=to_device_i32({K}),ds=to_device_i32({0});
    std::vector<int> positions(T),queries(K);
    std::iota(positions.begin(),positions.end(),0);std::iota(queries.begin(),queries.end(),context);
    auto dp=to_device(positions),dq=to_device(queries);
    Tensor counts(dn.p,DType::I32,{1}),valid(dv.p,DType::I32,{1}),slots(ds.p,DType::I32,{1});
    program.append_accepted_context(Tensor(df.p,DType::BF16,{F,T,1}),Tensor(dp.p,DType::I32,{T,1}),
        Tensor(dp.p,DType::I32,{T,1}),counts,slots,{std::uint32_t(context),std::uint32_t(context)},device.stream);
    auto hidden=program.draft_embeddings(Tensor(de.p,DType::BF16,{D,K,1}),Tensor(dq.p,DType::I32,{K,1}),
        counts,valid,slots,device.stream);
    DeviceBuffer storage(std::size_t(V)*K*2);
    Tensor logits(storage.p,DType::BF16,{V,K});
    ops::linear(hidden.view({D,K}),head,logits,device.stream);
    device.synchronize();
    return {from_device_bf16(hidden.data,hidden.numel()),from_device_bf16(logits.data,logits.numel())};
}

void compare(const std::filesystem::path& native,const std::filesystem::path& root,
             const std::filesystem::path& output,DeviceContext& device) {
    qwen4_sequence::TextPanel panel(native.string());
    nlohmann::json provenance;
    std::ifstream metadata(output/"qwen4-dflash-target-features.json"); metadata>>provenance;
    if(provenance.at("profile")!="qwen4-ud-iq1-s-diagnostic-accepted-prompt" ||
       provenance.at("accepted_prompt_tokens")!=T || provenance.at("token_ids")!=nlohmann::json(panel.tokens) ||
       provenance.at("taps")!=nlohmann::json::array({4,16,24,36,44}))
        throw std::runtime_error("target feature provenance mismatch");
    const auto binary=output/"qwen4-dflash-target-features.bin";
    if(std::filesystem::file_size(binary)!=F*T*2) throw std::runtime_error("target feature panel extent mismatch");
    std::vector<std::uint16_t> features(F*T);
    std::ifstream file(binary,std::ios::binary); file.read(reinterpret_cast<char*>(features.data()),features.size()*2);
    if(!file) throw std::runtime_error("truncated target features");
    qwen4_native::Bf16Source source((native/"qwen4-endpoint.ninfer").string());
    direct_bf16_weight::DeviceWeight head({V,D,source.bits("lm_head.weight",{V,D})});
    artifact::Reader inputs(root/"qwen4-dflash-inputs.ninfer");
    if(inputs.identity()!=artifact::ArtifactIdentity{"qwen4/native-dflash-inputs-qualification","nvidia-bf16-source"})
        throw std::runtime_error("wrong native DFlash shared input identity");
    const auto* mask_object=inputs.find("mask.embedding");
    const auto* mask_desc=mask_object?std::get_if<artifact::TensorDescriptor>(mask_object):nullptr;
    if(!mask_desc || mask_desc->format!=artifact::NumericFormat::BF16 || mask_desc->shape!=std::vector<std::uint64_t>{D})
        throw std::runtime_error("wrong DFlash mask row");
    const auto mask=inputs.payload(*mask_object).data;
    const auto anchors=panel.payload("token.embeddings",artifact::NumericFormat::BF16,{33,D});
    nlohmann::json cases=nlohmann::json::array();
    for(int context:{8,16,24}) {
        std::vector<std::uint16_t> embeddings(D*K);
        std::memcpy(embeddings.data(),anchors.data()+context*D*2,D*2);
        for(int q=1;q<K;++q) std::memcpy(embeddings.data()+q*D,mask.data(),D*2);
        const auto bf16=run(root,"bf16",head.view(),features,embeddings,context,device);
        const auto nvfp4=run(root,"nvfp4",head.view(),features,embeddings,context,device);
        nlohmann::json queries=nlohmann::json::array();
        int agreements=0;
        for(int q=0;q<K;++q) {
            const double* a=bf16.logits.data()+q*V; const double* b=nvfp4.logits.data()+q*V;
            const int ia=std::max_element(a,a+V)-a,ib=std::max_element(b,b+V)-b;
            const double ma=a[ia],mb=b[ib];double za=0,zb=0;
            for(int v=0;v<V;++v) {za+=std::exp(double(a[v])-ma);zb+=std::exp(double(b[v])-mb);}
            const double la=ma+std::log(za),lb=mb+std::log(zb);double kl=0;
            for(int v=0;v<V;++v) kl+=std::exp(double(a[v])-la)*(double(a[v])-la-double(b[v])+lb);
            const int label=panel.tokens[context+q+1];
            agreements+=ia==ib;
            queries.push_back({{"query",q},{"bf16_top1",ia},{"nvfp4_top1",ib},{"label",label},
                {"kl_bf16_to_nvfp4_nats",kl},{"bf16_label_nll",la-a[label]},{"nvfp4_label_nll",lb-b[label]}});
        }
        nlohmann::json result={{"context",context},{"hidden_relative_l2",relative_l2(bf16.hidden,nvfp4.hidden)},
            {"logits_relative_l2",relative_l2(bf16.logits,nvfp4.logits)},{"top1_agreement",agreements},
            {"queries",queries}};
        std::cout<<result.dump()<<std::endl;cases.push_back(result);
    }
    write_json(output/"qwen4-dflash-target-comparison.json",{{"feature_provenance",provenance},
        {"cases",cases},{"shared_embeddings_head","Pinned NVIDIA BF16 source, identical in both drafter profiles"},
        {"activation_profile","A16 only"},{"vocabulary",V},
        {"distribution","Full 248320-head domain, temperature 1, no additional mask or conditioning; diagnostic only, not the p-less/epsilon proposal law"},
        {"boundary","Pairwise quantization-quality diagnostic, not a kernel oracle or numerical pass gate. Three teacher-forced prompt contexts of one 33-token panel; no model PPL, speculative acceptance rate, or production quality claim. DFlash kernels have separate represented-input mathematical oracle qualification."}});
}
} // namespace

int main(int argc,char** argv) {
    const char* target=std::getenv("NINFER_QWEN4_WEIGHTS");
    const char* native=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    const char* draft=std::getenv("NINFER_QWEN4_NATIVE_DFLASH");
    const char* output=std::getenv("NINFER_QWEN4_DFLASH_TARGET_OUTPUT");
    if(argc!=2 || !native || !draft || !output) return 77;
    try {
        DeviceContext device;
        if(std::string(argv[1])=="--capture" && target) capture(target,native,output,device);
        else if(std::string(argv[1])=="--compare") compare(native,draft,output,device);
        else throw std::runtime_error("requires --capture with target or --compare");
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}

#include "artifact/typed_binding.h"
#include "artifact/materializer.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>

using namespace ninfer;
using namespace ninfer::test;

int components(const std::string& path,DeviceContext& device) {
    artifact::Reader reader(path);
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-weight-candidate-components","source-bf16-row-fp8-a16"})
        throw std::invalid_argument("wrong candidate component identity");
    artifact::Binder binder(reader);std::map<std::string,artifact::ObjectHandle> handles;
    for(const auto& obj:reader.objects()) {
        const auto& t=std::get<artifact::TensorDescriptor>(obj);
        const auto h=binder.require_tensor(t.name,t.format,t.layout,t.shape);
        binder.materialize_on_device(h);handles.emplace(t.name,h);
    }
    auto owner=artifact::materialize(reader,binder.finish(),device);
    auto tensor=[&](const std::string& name,std::initializer_list<int> dims) {
        const auto& t=std::get<artifact::TensorDescriptor>(*reader.find(name));
        return artifact::materialized_tensor(owner,handles.at(name),t.format,dims);
    };
    auto weight=[&](const std::string& name,int n,int k) {
        const auto& t=std::get<artifact::TensorDescriptor>(*reader.find(name));
        return artifact::materialized_weight(owner,handles.at(name),t.format,n,k);
    };
    auto expected=[&](const std::string& name) {
        const auto raw=reader.payload(name).data;std::vector<float> result(raw.size()/4);
        std::memcpy(result.data(),raw.data(),raw.size());return std::vector<double>(result.begin(),result.end());
    };
    auto hidden=tensor("hidden",{2560,4,3}),norm=tensor("gr.norm",{10240});
    const auto down=weight("gr_down.weight",320,10240),up=weight("gr_up.weight",10240,320);
    const auto key=weight("ple_key.weight",10240,2560),value=weight("ple_value.weight",2560,2560);
    auto embedding=tensor("ple.embedding",{2560,3}),kn=tensor("ple.key_norm",{10240}),
        qn=tensor("ple.query_norm",{10240}),cn=tensor("ple.conv_norm",{10240}),
        conv=tensor("ple.conv",{4,10240}),state=tensor("ple.state",{10240,9});
    WorkspaceArena workspace(std::max({ops::gated_residual_workspace_capacity_bytes(3,down.qtype,up.qtype),
        ops::ple_workspace_capacity_bytes(3,key.qtype,value.qtype),
        ops::ple_workspace_capacity_bytes(3,key.qtype,QType::BF16_CTRL)}));
    GuardedDeviceBuffer out(10240*3*2),next(10240*9*2);
    Tensor read(out.data(),DType::BF16,{2560,3}),output(out.data(),DType::BF16,{2560,4,3}),
        next_state(next.data(),DType::BF16,{10240,9});
    int failures=0;
    ops::gated_residual_read(hidden,norm,down,up,read,workspace,device.stream);device.synchronize();
    failures+=verify_reduction("native rowFP8 final GR represented oracle",from_device_bf16(out.data(),2560*3),
        expected("gr.oracle"),ReductionCriterion{.006,.004,.01});
    for(const std::string mode:{"key","both"}) {
        const auto v=mode=="key"?weight("ple_value.source",2560,2560):value;
        ops::ple_inject(hidden,embedding,key,v,kn,qn,cn,conv,state,next_state,output,workspace,
            ops::PleNormFormat::ZeroCenteredBf16,device.stream);device.synchronize();
        failures+=verify_reduction("native rowFP8 PLE "+mode+" represented oracle",
            from_device_bf16(out.data(),10240*3),expected("ple."+mode+".oracle"),ReductionCriterion{.02,1e-4,.02});
        failures+=verify_reduction("native rowFP8 PLE "+mode+" history oracle",
            from_device_bf16(next.data(),10240*9),expected("ple."+mode+".state_oracle"),ReductionCriterion{1./256,1e-5,1./128});
    }
    return failures+out.verify_guards("candidate component output")+next.verify_guards("candidate PLE state");
}

int main() {
    const char* path=std::getenv("NINFER_QWEN4_WEIGHT_CANDIDATES");
    if(!path) {std::cout<<"SKIP original-BF16 weight candidate fixture unset\n";return 77;}
    try {
        if(require_cuda()!=0) return 1;
        DeviceContext device;
        artifact::Reader reader(path);
        if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-weight-candidates","source-bf16-row-fp8-a16"})
            throw std::invalid_argument("wrong original-BF16 candidate identity");
        artifact::Binder binder(reader);
        struct Case {const char* name;int n,k;};
        const Case cases[]={{"head",248320,2560},{"gr_down",320,10240},{"gr_up",10240,320},
                            {"ple_key",10240,2560},{"ple_value",2560,2560}};
        std::map<std::string,artifact::ObjectHandle> weights,inputs;
        for(const auto& c:cases) {
            weights.emplace(c.name,artifact::bind_device_tensor(binder,std::string(c.name)+".weight",
                artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,{std::uint64_t(c.n),std::uint64_t(c.k)}));
            inputs.emplace(c.name,artifact::bind_device_tensor(binder,std::string(c.name)+".input",
                artifact::NumericFormat::BF16,{3,std::uint64_t(c.k)}));
            const auto h=binder.require_tensor(std::string(c.name)+".oracle",artifact::NumericFormat::FP32,
                artifact::StorageLayout::ContiguousLeV1,std::array<std::uint64_t,2>{3,std::uint64_t(c.n)});
            binder.validate_only(h);
        }
        auto owner=artifact::materialize(reader,binder.finish(),device);
        int failures=0;
        for(const auto& c:cases) {
            const auto weight=artifact::materialized_weight(owner,weights.at(c.name),artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,c.n,c.k);
            auto input=artifact::materialized_tensor(owner,inputs.at(c.name),artifact::NumericFormat::BF16,{c.k,3});
            const auto raw=reader.payload(std::string(c.name)+".oracle").data;
            std::vector<float> expected(3*c.n);std::memcpy(expected.data(),raw.data(),raw.size());
            GuardedDeviceBuffer output(3*c.n*2);
            Tensor y(output.data(),DType::BF16,{c.n,3});
            WorkspaceArena workspace(std::max<std::size_t>(256,ops::linear_workspace_capacity_bytes(
                weight.qtype,c.n,c.k,ops::LinearPolicy::A16Only,1,3)));
            auto call=[&](bool split) {
                for(int first=0;first<3;) {
                    const int width=split?1:3;
                    auto x=input.slice(1,first,width),out=y.slice(1,first,width);
                    ops::linear(x,weight,out,ops::LinearPolicy::A16Only,workspace,device.stream);
                    first+=width;
                }
            };
            for(int mode=0;mode<3;++mode) {
                if(mode==2) {
                    DecodeGraphDefinition definition;DecodeGraphExecutable graph;
                    definition.capture(device.stream,[&]{call(false);});graph.instantiate(definition);
                    graph.launch(device.stream);device.synchronize();
                } else {call(mode==1);device.synchronize();}
                const auto actual=from_device_bf16(output.data(),3*c.n);
                failures+=verify_reduction(std::string("native row-FP8 A16 ")+c.name+" mode="+std::to_string(mode),
                    actual,std::vector<double>(expected.begin(),expected.end()),ReductionCriterion{1./256,1./256,2./256});
                failures+=output.verify_guards(c.name);
            }
        }
        auto component_path=std::filesystem::path(path);component_path.replace_extension(".components.ninfer");
        failures+=components(component_path.string(),device);
        std::cout<<(failures?"FAIL":"PASS")<<" native represented-weight candidates; source loss is separate\n";
        return failures?1:0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

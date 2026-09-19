#include "targets/qwen4/mtp.h"
#include "mtp_test_program.h"
#include "ops/op_tester.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::targets::qwen4;
using Json=nlohmann::json;
using ninfer::test::qwen4::MtpControl;
using ninfer::test::qwen4::prepare_mtp_control;
namespace {
constexpr int D=2560,F=10240,S=ops::kQsaSelectedCapacity;
constexpr ReductionCriterion composed{.02,.005,.02};
const char* names[]{"stem","attention_read","attention_write","attention","attention_injected",
                    "moe_read","moe_write","moe_ids","moe_weights","moe","moe_injected","final_read"};
Json represented(const Tensor& t) {
    Json result;
    result["shape"]={t.ne[0],t.ne[1],t.ne[2],t.ne[3]};
    if(t.dtype==DType::BF16) {
        result["dtype"]="BF16"; result["words"]=from_device<std::uint16_t>(t.data,t.bytes()/2);
    } else if(t.dtype==DType::I32) {
        result["dtype"]="I32"; result["values"]=from_device<int>(t.data,t.bytes()/4);
    } else {
        result["dtype"]="F32"; result["values"]=from_device<float>(t.data,t.bytes()/4);
    }
    return result;
}
void capture(void* context,MtpTraceStage stage,const Tensor& t,cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    (*static_cast<Json*>(context))[names[static_cast<int>(stage)]]=represented(t);
}
Json live_state(const ninfer::test::qwen4::MtpTestProgram& p) {
    const auto s=p.state();
    const int live=p.frontier(),capacity=s.k.ne[1];
    Json result{{"frontier",live},{"selected",represented(p.selected_ids())},
                {"count",represented(p.selected_count())}};
    for(auto [name,t]:std::array<std::pair<const char*,Tensor>,2>{{{"k",s.k},{"v",s.v}}}) {
        const auto raw=from_device<std::uint16_t>(t.data,t.bytes()/2);
        std::vector<std::uint16_t> visible;
        for(int head=0;head<2;++head)
            visible.insert(visible.end(),raw.begin()+head*capacity*256,
                           raw.begin()+(head*capacity+live)*256);
        result[name]=visible;
    }
    result["raw_keys"]=from_device<std::uint16_t>(s.raw_index_keys.data,live*128);
    result["positions"]=from_device<int>(s.positions.data,live*3);
    return result;
}
int control_test() {
    const std::array<std::array<int,3>,2> p{{{107,31,19},{108,35,20}}};
    auto c=prepare_mtp_control(3,p);
    int errors=verify_exact("MTP independent logical positions",c.ids,std::vector<int>{3,4});
    errors+=verify_exact("MTP distinct RoPE coordinates",c.positions,std::vector<int>{107,31,19,108,35,20});
    errors+=verify_exact("MTP causal CSR offsets",c.offsets,std::vector<int>{0,4,9});
    errors+=verify_exact("MTP causal CSR visibility",c.visible,std::vector<int>{0,1,2,3,0,1,2,3,4});
    try { (void)prepare_mtp_control(4095,p); ++errors; } catch(const std::invalid_argument&) {}
    return errors;
}

int four_requests(const LoadedMtp& model,const Tensor& embeddings,const Tensor& hidden,
                   DeviceContext& device) {
    std::array<std::unique_ptr<ninfer::test::qwen4::MtpTestProgram>,4> programs;
    std::array<Json,4> states;
    std::array<std::array<std::array<int,3>,3>,4> positions;
    for(int b=0;b<4;++b) {
        programs[b]=std::make_unique<ninfer::test::qwen4::MtpTestProgram>(model.weights(),32,1,device.stream);
        for(int t=0;t<3;++t) positions[b][t]={101+b*11+t,37+b*5+t*2,19+b*7+t*3};
        const auto control=prepare_mtp_control(0,std::span(positions[b]).first(1));
        (void)programs[b]->extend(embeddings.slice(1,b,1),hidden.slice(1,b,1),control,device.stream);
        device.synchronize();states[b]=live_state(*programs[b]);
    }
    std::array<std::array<std::vector<double>,2>,4> outputs;
    int failures=0;
    for(int t=1;t<3;++t) for(int b:{3,1,0,2}) {
        const auto control=prepare_mtp_control(t,std::span(positions[b]).subspan(t,1));
        const auto result=programs[b]->draft(embeddings.slice(1,b+t,1),control,device.stream);
        device.synchronize();
        outputs[b][t-1]=from_device_bf16(result.carried_hidden.data,F);
        states[b]=live_state(*programs[b]);
        for(int other=0;other<4;++other) if(other!=b && live_state(*programs[other])!=states[other]) {
            std::cerr<<"MTP interleaved request changed another request's cache/frontier\n";++failures;
        }
    }
    ninfer::test::qwen4::MtpTestProgram independent(model.weights(),32,1,device.stream);
    MtpTrace eager; // Same qualified schedule, observer-free but explicitly eager for graph comparison.
    for(int b=0;b<4;++b) {
        independent.reset(device.stream);
        const auto seed=prepare_mtp_control(0,std::span(positions[b]).first(1));
        (void)independent.extend(embeddings.slice(1,b,1),hidden.slice(1,b,1),seed,device.stream);
        device.synchronize();
        for(int t=1;t<3;++t) {
            const auto control=prepare_mtp_control(t,std::span(positions[b]).subspan(t,1));
            const auto result=independent.draft(embeddings.slice(1,b+t,1),control,device.stream,&eager);
            device.synchronize();
            failures+=verify_exact("MTP C4 interleaved graph versus independent eager request",
                from_device_bf16(result.carried_hidden.data,F),outputs[b][t-1]);
        }
        if(live_state(independent)!=states[b]) {
            std::cerr<<"MTP graph and independent request final cache/domain differ\n";++failures;
        }
    }
    return failures;
}
int real(const std::string& path,const std::string& trace_path) {
    DeviceContext device;
    auto model=LoadedMtp::load(path,device);
    ninfer::test::qwen4::MtpTestProgram p(model->weights(),32,5,device.stream);
    std::vector<float> embeddings(9*D),hidden(9*F);
    fill_uniform(embeddings,41001,-.15F,.15F); fill_uniform(hidden,41003,-.7F,.7F);
    for(int t=0;t<9;++t) for(int b=0;b<4;++b) for(int d=0;d<D;++d)
        hidden[t*F+b*D+d]*=std::array<float,4>{.125F,.5F,1.F,2.F}[b];
    round_to_bf16(embeddings); round_to_bf16(hidden);
    auto de=to_device_bf16(embeddings),dh=to_device_bf16(hidden);
    Tensor all_e(de.p,DType::BF16,{D,9}),all_h(dh.p,DType::BF16,{F,9});
    std::array<std::array<int,3>,9> positions;
    for(int i=0;i<9;++i) positions[i]={97+i,43+2*i,11+3*i};
    const auto initial=prepare_mtp_control(0,std::span(positions).first(5));
    Json report{{"profile","qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16"},{"records",Json::array()}};
    auto run=[&](const MtpControl& control,bool draft,bool trace) {
        auto e=all_e.slice(1,control.frontier,control.ids.size());
        auto h=all_h.slice(1,control.frontier,control.ids.size());
        Json record{{"draft",draft},{"frontier",control.frontier},{"positions",control.positions},
                    {"embedding",represented(e)}};
        if(!draft) record["target_hidden"]=represented(h);
        Json stages;
        MtpTrace observer{&stages,capture};
        auto output=draft?p.draft(e,control,device.stream,trace?&observer:nullptr):
            p.extend(e,h,control,device.stream,trace?&observer:nullptr);
        device.synchronize();
        if(trace) {
            record["stages"]=std::move(stages);
            record["selected_ids"]=represented(p.selected_ids());
            record["selected_count"]=represented(p.selected_count());
            auto state=p.state();
            // Preserve physical KV-head-major storage; the record frontier identifies live rows.
            Json cache;
            for(auto [name,t]:std::array<std::pair<const char*,Tensor>,4>{{{"k",state.k},{"v",state.v},
                         {"raw_keys",state.raw_index_keys},{"positions",state.positions}}})
                cache[name]=represented(t);
            record["cache"]=std::move(cache);
            report["records"].push_back(std::move(record));
        }
        return from_device_bf16(output.carried_hidden.data,output.carried_hidden.bytes()/2);
    };
    const auto seed=run(initial,false,true);
    const auto seed_ids=from_device<int>(p.selected_ids().data,S),seed_count=from_device<int>(p.selected_count().data,1);
    int failures=0;
    // A rejected first draft must not start a provisional transaction or alter its seed.
    const auto next_control=prepare_mtp_control(5,std::span(positions).subspan(5,1));
    Tensor unaligned(static_cast<std::byte*>(de.p)+2,DType::BF16,{D,1});
    try { (void)p.draft(unaligned,next_control,device.stream); ++failures; }
    catch(const std::invalid_argument&) {}
    device.synchronize();
    if(p.frontier()!=5) ++failures;
    failures+=verify_exact("MTP rejected first draft selection",from_device<int>(p.selected_ids().data,S),seed_ids);
    failures+=verify_exact("MTP rejected first draft count",from_device<int>(p.selected_count().data,1),seed_count);
    std::vector<std::vector<double>> draft_outputs;
    for(int i=0;i<4;++i) {
        const auto c=prepare_mtp_control(5+i,std::span(positions).subspan(5+i,1));
        draft_outputs.push_back(run(c,true,true));
        failures+=verify_exact("MTP frozen selected IDs",from_device<int>(p.selected_ids().data,S),seed_ids);
        failures+=verify_exact("MTP frozen selected count",from_device<int>(p.selected_count().data,1),seed_count);
    }
    p.discard_drafts(device.stream); device.synchronize();
    if(p.frontier()!=5) ++failures;
    for(int i=0;i<4;++i) {
        const auto c=prepare_mtp_control(5+i,std::span(positions).subspan(5+i,1));
        failures+=verify_exact("MTP rejected-chain replay",run(c,true,false),draft_outputs[i]);
    }
    p.discard_drafts(device.stream); device.synchronize();
    // Every partial/full accepted count is replayed from TARGET hidden and compared with
    // a fresh target-aligned schedule. No discarded draft hidden is folded as committed state.
    for(int accepted=0;accepted<=4;++accepted) {
        p.reset(device.stream); (void)run(initial,false,false);
        for(int i=0;i<4;++i) {
            auto c=prepare_mtp_control(5+i,std::span(positions).subspan(5+i,1));
            (void)run(c,true,false);
        }
        p.discard_drafts(device.stream); device.synchronize();
        std::vector<double> retained;
        if(accepted) {
            const auto replay=prepare_mtp_control(5,std::span(positions).subspan(5,accepted));
            retained=run(replay,false,false);
        }
        const auto retained_state=live_state(p);
        p.reset(device.stream); (void)run(initial,false,false);
        if(accepted) {
            const auto replay=prepare_mtp_control(5,std::span(positions).subspan(5,accepted));
            failures+=verify_exact("MTP target-aligned accepted replay",run(replay,false,false),retained);
        }
        if(live_state(p)!=retained_state) {
            std::cerr<<"MTP retained cache/domain differs from fresh target replay at accepted="<<accepted<<'\n';
            ++failures;
        }
        if(p.frontier()!=5+accepted) ++failures;
    }
    p.reset(device.stream);
    const auto first=prepare_mtp_control(0,std::span(positions).first(4)); (void)run(first,false,false);
    const auto last=prepare_mtp_control(4,std::span(positions).subspan(4,1));
    const auto chunked=run(last,false,false);
    failures+=verify_reduction("MTP whole5 versus4+1 carried state",chunked,
                               std::span(seed).subspan(4*F,F),composed);
    failures+=four_requests(*model,all_e,all_h,device);
    if(!trace_path.empty()) {
        if(std::filesystem::exists(trace_path)) throw std::runtime_error("refusing to replace MTP trace");
        std::ofstream file(trace_path); if(!file) throw std::runtime_error("cannot create MTP trace");
        file<<report.dump()<<'\n';
    }
    std::cout<<"MTP native state/replay failures="<<failures<<'\n';
    return failures;
}
}
int main(int argc,char** argv) {
    try {
        int failures=control_test();
        if(argc==1) return failures?1:0;
        if(std::string(argv[1])!="--native-real" || (argc!=2 && argc!=4) ||
            (argc==4 && std::string(argv[2])!="--trace")) throw std::invalid_argument("MTP test arguments");
        const char* root=std::getenv("NINFER_QWEN4_MTP");
        if(!root) { std::cout<<"SKIP NINFER_QWEN4_MTP unset\n"; return 77; }
        if(require_cuda()!=0) return 1;
        failures+=real(std::string(root)+"/qwen4-mtp-nvfp4.ninfer",argc==4?argv[3]:"");
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

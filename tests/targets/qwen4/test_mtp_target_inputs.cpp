// Staged diagnostic target carry -> native NVFP4/A16 private MTP -> the exact
// diagnostic shared endpoints. No full-target co-residency or product registration.
#include "artifact/typed_binding.h"
#include "targets/qwen4/mtp.h"
#include "mtp_test_program.h"
#include "targets/qwen4/verifier.h"
#include "ninfer/ops/ggml_embedding.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace q4=ninfer::targets::qwen4;
using Json=nlohmann::json;
namespace {
constexpr int D=2560,F=10240,V=248320,C=24,K=3,S=ops::kQsaSelectedCapacity;
const char* names[]{"stem","attention_read","attention_write","attention","attention_injected",
    "moe_read","moe_write","moe_ids","moe_weights","moe","moe_injected","final_read"};
Json represented(const Tensor& value) {
    Json out{{"shape",{value.ne[0],value.ne[1],value.ne[2],value.ne[3]}}};
    if(value.dtype==DType::BF16) {out["dtype"]="BF16";out["words"]=from_device<std::uint16_t>(value.data,value.bytes()/2);}
    else if(value.dtype==DType::I32) {out["dtype"]="I32";out["values"]=from_device<int>(value.data,value.bytes()/4);}
    else {out["dtype"]="F32";out["values"]=from_device<float>(value.data,value.bytes()/4);}
    return out;
}
void capture(void* context,q4::MtpTraceStage stage,const Tensor& value,cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    (*static_cast<Json*>(context))[names[static_cast<int>(stage)]]=represented(value);
}
Json live_state(const ninfer::test::qwen4::MtpTestProgram& program) {
    auto s=program.state();const int live=program.frontier(),capacity=s.k.ne[1];
    Json out{{"frontier",live},{"selected",represented(program.selected_ids())},
             {"count",represented(program.selected_count())}};
    for(auto [name,t]:std::array<std::pair<const char*,Tensor>,2>{{{"k",s.k},{"v",s.v}}}) {
        auto words=from_device<std::uint16_t>(t.data,t.bytes()/2);
        std::vector<std::uint16_t> visible;
        for(int h=0;h<2;++h) visible.insert(visible.end(),words.begin()+h*capacity*256,
            words.begin()+(h*capacity+live)*256);
        out[name]=visible;
    }
    out["keys"]=from_device<std::uint16_t>(s.raw_index_keys.data,live*128);
    out["positions"]=from_device<int>(s.positions.data,live*3);
    return out;
}
std::vector<std::uint16_t> plane(const Json& metadata,const std::filesystem::path& path,
                                const char* name,int rows,int columns) {
    const auto& p=metadata.at("planes").at(name);
    const auto bytes=std::size_t(rows)*columns*2,offset=p.at("offset").get<std::size_t>();
    if(p.at("dtype")!="BF16" || p.at("shape")!=Json::array({rows,columns}) ||
       p.at("bytes")!=bytes || offset>std::filesystem::file_size(path) ||
       bytes>std::filesystem::file_size(path)-offset) throw std::runtime_error("invalid captured MTP plane");
    std::vector<std::uint16_t> values(bytes/2);
    std::ifstream input(path,std::ios::binary);input.seekg(offset);
    input.read(reinterpret_cast<char*>(values.data()),bytes);
    if(!input) throw std::runtime_error("truncated captured MTP plane");
    for(auto x:values) if(!std::isfinite(bf16_to_f32(x))) throw std::runtime_error("nonfinite captured MTP input");
    return values;
}
ninfer::test::qwen4::MtpControl control(int start,int count) {
    std::vector<std::array<int,3>> positions(count);
    for(int i=0;i<count;++i) positions[i]={start+i,start+i,start+i};
    return ninfer::test::qwen4::prepare_mtp_control(start,positions);
}
int run(const std::filesystem::path& target,const std::filesystem::path& mtp,
        const std::filesystem::path& output,DeviceContext& device) {
    const auto trace_path=output/"qwen4-mtp-target-trace.json",report_path=output/"qwen4-mtp-target-integration.json";
    if(std::filesystem::exists(trace_path) || std::filesystem::exists(report_path))
        throw std::runtime_error("refusing to overwrite MTP integration evidence");
    Json input;std::ifstream metadata(output/"qwen4-mtp-target-inputs.json");metadata>>input;
    if(input.at("target")!=target.string())
        throw std::runtime_error("captured MTP target provenance mismatch");
    const auto prompt_ids=input.at("prompt_token_ids").get<std::vector<int>>();
    const auto next_ids=input.at("aligned_next_token_ids").get<std::vector<int>>();
    const auto verified_ids=input.at("verified_ids").get<std::vector<int>>();
    const auto licensed=input.at("licensed_ids").get<std::vector<int>>();
    const int accepted=input.at("committed_inputs");
    std::vector<int> positions(C);std::iota(positions.begin(),positions.end(),0);
    if(prompt_ids.size()<C || next_ids.size()!=C || input.at("prompt_positions")!=Json(positions) ||
       next_ids.back()!=input.at("initial_anchor") || verified_ids.size()!=8 ||
       verified_ids.front()!=next_ids.back() || accepted<1 || accepted>8 || licensed.size()!=std::size_t(accepted))
        throw std::runtime_error("captured MTP input alignment mismatch");
    for(int t=0;t<C-1;++t) if(next_ids[t]!=prompt_ids[t+1]) throw std::runtime_error("MTP prompt shift mismatch");
    std::vector<int> verified_positions(8);std::iota(verified_positions.begin(),verified_positions.end(),C);
    if(input.at("verified_positions")!=Json(verified_positions)) throw std::runtime_error("MTP verified source positions mismatch");
    for(int t=0;t<accepted-1;++t) if(licensed[t]!=verified_ids[t+1]) throw std::runtime_error("MTP licensed prefix mismatch");
    for(const auto* ids:{&next_ids,&licensed}) for(int id:*ids) if(id<0 || id>=V) throw std::runtime_error("invalid captured token");
    const auto binary=output/"qwen4-mtp-target-inputs.bin";
    auto hidden=to_device(plane(input,binary,"prompt_hidden",C,F));
    auto captured_e=plane(input,binary,"prompt_next_embedding",C,D);
    auto verified_h=to_device(plane(input,binary,"verified_hidden",8,F));

    // Only these two original packed objects are materialized; the full diagnostic
    // binder is identity authority, not authorization to load/stream other weights.
    artifact::Reader reader(target);const auto bound=q4::verifier::bind_artifact(reader);
    artifact::MaterializationPlan plan;plan.object_count=bound.materialization.object_count;
    for(auto object:bound.materialization.device_objects) {
        if(object.object.index!=bound.bindings.token_embedding.index && object.object.index!=bound.bindings.output_head.index) continue;
        object.offset=(plan.device_capacity_bytes+object.alignment-1)/object.alignment*object.alignment;
        plan.device_capacity_bytes=object.offset+object.bytes;plan.device_objects.push_back(object);
    }
    if(plan.device_objects.size()!=2) throw std::runtime_error("MTP shared endpoint inventory mismatch");
    auto weights=artifact::materialize(reader,plan,device);
    const auto embedding=artifact::materialized_weight(weights,bound.bindings.token_embedding,artifact::NumericFormat::Q4_K,V,D);
    const auto head=artifact::materialized_weight(weights,bound.bindings.output_head,artifact::NumericFormat::Q4_K,V,D);
    DeviceBuffer seed_e(D*C*2),replay_e(D*8*2),one_e(D*2),logit_storage(V*2),proposal(4),draw(4),counts(V*4);
    Tensor seed_embeddings(seed_e.p,DType::BF16,{D,C}),target_hidden(hidden.p,DType::BF16,{F,C});
    auto gather=[&](const std::vector<int>& ids,DeviceBuffer& destination) {
        for(std::size_t i=0;i<ids.size();++i) {
            Tensor row(static_cast<std::byte*>(destination.p)+i*D*2,DType::BF16,{D});
            ops::ggml_q4_k_embedding_row(embedding,ids[i],row,device.stream);
        }
    };
    gather(next_ids,seed_e);gather(licensed,replay_e);device.synchronize();
    int failures=verify_exact("MTP source-shifted diagnostic embeddings",from_device<std::uint16_t>(seed_e,D*C),captured_e);
    auto model=q4::LoadedMtp::load(mtp/"qwen4-mtp-nvfp4.ninfer",device);
    ninfer::test::qwen4::MtpTestProgram program(model->weights(),64,C,device.stream);
    Json trace{{"profile","qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16"},{"records",Json::array()},
        {"input_provenance",input},{"boundary","Actual diagnostic target carried hidden and shared Q4_K endpoints; native private W4A16 block. Not native full-target quality."}};
    auto execute=[&](const ninfer::test::qwen4::MtpControl& c,const Tensor& e,const Tensor& h,bool draft,bool traced) {
        Json stages; q4::MtpTrace observer{&stages,capture};
        Json record{{"draft",draft},{"frontier",c.frontier},{"positions",c.positions},{"embedding",represented(e)}};
        if(!draft) record["target_hidden"]=represented(h);
        auto out=draft?program.draft(e,c,device.stream,traced?&observer:nullptr):
            program.extend(e,h,c,device.stream,traced?&observer:nullptr);
        device.synchronize();
        if(traced) {
            record["stages"]=std::move(stages);record["selected_ids"]=represented(program.selected_ids());
            record["selected_count"]=represented(program.selected_count());auto s=program.state();
            record["cache"]={{"k",represented(s.k)},{"v",represented(s.v)},
                {"raw_keys",represented(s.raw_index_keys)},{"positions",represented(s.positions)}};
            trace["records"].push_back(std::move(record));
        }
        return out;
    };
    auto seed=[&](bool traced) {return execute(control(0,C),seed_embeddings,target_hidden,false,traced);};
    auto out=seed(true);const auto seed_state=live_state(program);
    auto selected=from_device<int>(program.selected_ids().data,S),selected_count=from_device<int>(program.selected_count().data,1);
    ops::SamplingConfig greedy,stochastic;stochastic.temperature=.6F;stochastic.p_less=1;stochastic.seed=1939;
    stochastic.token_counts=static_cast<int*>(counts.p);counts.fill();
    auto greedy_config=to_device(std::vector{greedy}),sample_config=to_device(std::vector{stochastic}),logical=to_device_i32({C});
    Tensor logits(logit_storage.p,DType::BF16,{V,1}),greedy_id(proposal.p,DType::I32,{1}),sample_id(draw.p,DType::I32,{1}),position(logical.p,DType::I32,{1});
    WorkspaceArena sample_workspace(ops::sampling_workspace_capacity_bytes(V,1,1));
    std::vector<int> draft_ids,draws;std::vector<std::vector<std::uint16_t>> carried;
    for(int step=0;step<K;++step) {
        // The logit-input from private position C-1+step predicts position C+1+step.
        // RNG keys name its preceding token position C+step, independent of private RoPE.
        auto last=out.logit_input.slice(1,out.logit_input.ne[1]-1,1).view({D});
        ops::ggml_block_linear(last,head,logits,device.stream);
        const int key=C+step;logical.copy_from_host(&key,4);
        ops::sample(logits,greedy_id,V,static_cast<ops::SamplingConfig*>(greedy_config.p),position,
            ops::kSamplePurposeDecode,sample_workspace,device.stream);
        ops::sample(logits,sample_id,V,static_cast<ops::SamplingConfig*>(sample_config.p),position,
            ops::kSamplePurposeDecode,sample_workspace,device.stream);
        device.synchronize();draft_ids.push_back(from_device<int>(proposal,1)[0]);draws.push_back(from_device<int>(draw,1)[0]);
        const auto values=from_device_bf16(logits.data,V);
        failures+=draft_ids.back()!=std::distance(values.begin(),std::max_element(values.begin(),values.end()));
        if(step<K-1) {
            gather({draft_ids.back()},one_e);
            out=execute(control(C+step,1),Tensor(one_e.p,DType::BF16,{D,1}),{},true,true);
            carried.push_back(from_device<std::uint16_t>(out.carried_hidden.data,F));
            failures+=verify_exact("actual-input MTP frozen domain",from_device<int>(program.selected_ids().data,S),selected);
            failures+=verify_exact("actual-input MTP frozen count",from_device<int>(program.selected_count().data,1),selected_count);
        }
    }
    std::vector<int> expected_counts(V);for(int id:draws) {if(id<0 || id>=V) throw std::runtime_error("sample outside vocabulary");++expected_counts[id];}
    failures+=verify_exact("actual MTP head common p-less occurrence updates",from_device<int>(counts,V),expected_counts);
    program.discard_drafts(device.stream);device.synchronize();
    if(live_state(program)!=seed_state) {std::cerr<<"actual MTP seed restoration mismatch\n";++failures;}
    for(int step=0;step<K-1;++step) {
        gather({draft_ids[step]},one_e);
        auto replay=execute(control(C+step,1),Tensor(one_e.p,DType::BF16,{D,1}),{},true,false);
        failures+=verify_exact("actual MTP frozen-domain graph versus traced eager",from_device<std::uint16_t>(replay.carried_hidden.data,F),carried[step]);
    }
    // Real target verification rows came from the separately executed DFlash round.
    // They prove MTP caller replay alignment, not MTP proposal acceptance quality.
    std::vector<int> keeps{0};if(accepted>1) keeps.push_back(1);keeps.push_back(accepted);
    for(int keep:keeps) {
        program.discard_drafts(device.stream);
        Tensor e(replay_e.p,DType::BF16,{D,std::max(keep,1)}),h(verified_h.p,DType::BF16,{F,std::max(keep,1)});
        std::vector<std::uint16_t> accepted_output;
        if(keep) {auto r=execute(control(C,keep),e,h,false,false);accepted_output=from_device<std::uint16_t>(r.carried_hidden.data,F*keep);}
        const auto retained=live_state(program);
        program.reset(device.stream);(void)seed(false);
        if(keep) {
            auto r=execute(control(C,keep),e,h,false,false);
            failures+=verify_exact("actual target-hidden accepted replay",from_device<std::uint16_t>(r.carried_hidden.data,F*keep),accepted_output);
        }
        if(live_state(program)!=retained || program.frontier()!=C+keep) {std::cerr<<"actual target replay state mismatch\n";++failures;}
        program.reset(device.stream);(void)seed(false);
        for(int step=0;step<K-1;++step) {gather({draft_ids[step]},one_e);(void)execute(control(C+step,1),Tensor(one_e.p,DType::BF16,{D,1}),{},true,false);}
    }
    device.synchronize();
    std::ofstream trace_file(trace_path);trace_file<<trace.dump()<<'\n';
    Json report{{"provenance",input},{"drafts",draft_ids},{"p_less_draws",draws},{"temperature",.6},
        {"epsilon",ops::kPLessLogitPerturbation},{"support_cap",ops::kPLessMaxEffectiveSupport},
        {"replayed_input_prefixes",keeps},{"failures",failures},{"oracle_trace",trace_path.string()},
        {"boundary","24 real target rows, teacher-provided initial anchor, source-aligned native W4A16 MTP seed and greedy feedback; common p-less sampling is an isolated head probe, not a replacement draft policy. Accepted target rows come from the natural DFlash verification round. No MTP acceptance-rate, PPL, native full-target or performance claim."}};
    std::ofstream report_file(report_path);report_file<<report.dump(2)<<'\n';
    if(!trace_file || !report_file) throw std::runtime_error("cannot write MTP integration evidence");
    std::cout<<report.dump(2)<<std::endl;return failures;
}
}
int main() {
    const char* target=std::getenv("NINFER_QWEN4_WEIGHTS"),*mtp=std::getenv("NINFER_QWEN4_MTP"),
        *output=std::getenv("NINFER_QWEN4_MTP_TARGET_OUTPUT");
    if(!target || !mtp || !output) return 77;
    try {DeviceContext device;return run(target,mtp,output,device)?1:0;}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

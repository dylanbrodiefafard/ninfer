// One exact diagnostic composition: load only shared endpoints + NVFP4 drafter,
// unload them, then run the existing full target and its shared acceptance law.
#include "artifact/typed_binding.h"
#include "targets/qwen4/dflash.h"
#include "dflash_test_cache.h"
#include "targets/qwen4/verifier.h"
#include "targets/qwen4/native_text_panel.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/ggml_embedding.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ninfer/ops/speculative_round.h"
#include "runtime/contract/sampling.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace q4 = ninfer::targets::qwen4;
namespace {
constexpr int D=2560,F=12800,V=248320,C=24,K=7,W=K+1;

std::vector<int> proposals(const std::filesystem::path& target,const std::filesystem::path& draft,
    const std::filesystem::path& output,const qwen4_sequence::TextPanel& panel,DeviceContext& device) {
    nlohmann::json metadata;
    std::ifstream input(output/"qwen4-dflash-target-features.json");input>>metadata;
    if(metadata.at("profile")!="qwen4-ud-iq1-s-diagnostic-accepted-prompt" ||
       metadata.at("accepted_prompt_tokens")!=C || metadata.at("token_ids")!=nlohmann::json(panel.tokens) ||
       metadata.at("target")!=target.string() || metadata.at("taps")!=nlohmann::json::array({4,16,24,36,44}))
        throw std::runtime_error("staged diagnostic feature provenance mismatch");
    const auto feature_path=output/"qwen4-dflash-target-features.bin";
    if(std::filesystem::file_size(feature_path)!=F*C*2) throw std::runtime_error("wrong feature extent");
    std::vector<std::uint16_t> features(F*C);
    std::ifstream file(feature_path,std::ios::binary);file.read(reinterpret_cast<char*>(features.data()),features.size()*2);
    if(!file) throw std::runtime_error("truncated feature panel");
    artifact::Reader reader(target);
    const auto full=q4::verifier::bind_artifact(reader);
    artifact::MaterializationPlan plan;
    plan.object_count=full.materialization.object_count;
    for(auto object:full.materialization.device_objects) {
        if(object.object.index!=full.bindings.token_embedding.index && object.object.index!=full.bindings.output_head.index) continue;
        object.offset=(plan.device_capacity_bytes+object.alignment-1)/object.alignment*object.alignment;
        plan.device_capacity_bytes=object.offset+object.bytes;
        plan.device_objects.push_back(object);
    }
    if(plan.device_objects.size()!=2) throw std::runtime_error("diagnostic shared endpoint inventory mismatch");
    auto weights=artifact::materialize(reader,plan,device);
    const auto embedding=artifact::materialized_weight(weights,full.bindings.token_embedding,artifact::NumericFormat::Q4_K,V,D);
    const auto head=artifact::materialized_weight(weights,full.bindings.output_head,artifact::NumericFormat::Q4_K,V,D);
    auto model=q4::LoadedDFlash::load(draft/"qwen4-dflash-nvfp4.ninfer",device);
    ninfer::test::qwen4::DFlashTestCache cache(32,1,device.stream);
    q4::DFlashProgram program(model->weights(),32,C,1,cache.views());
    auto df=to_device(features),dn=to_device_i32({C}),ds=to_device_i32({0}),dv=to_device_i32({K});
    std::vector<int> pos(C),query(K);std::iota(pos.begin(),pos.end(),0);std::iota(query.begin(),query.end(),C);
    auto dp=to_device(pos),dq=to_device(query);
    Tensor length(dn.p,DType::I32,{1}),slots(ds.p,DType::I32,{1}),valid(dv.p,DType::I32,{1});
    program.append_accepted_context(Tensor(df.p,DType::BF16,{F,C,1}),Tensor(dp.p,DType::I32,{C,1}),
        Tensor(dp.p,DType::I32,{C,1}),length,slots,{C,C},device.stream);
    DeviceBuffer rows(D*K*2),logit_storage(std::size_t(V)*K*2),ids(K*4);
    Tensor embeddings(rows.p,DType::BF16,{D,K,1}),logits(logit_storage.p,DType::BF16,{V,K}),drafts(ids.p,DType::I32,{K});
    for(int q=0;q<K;++q) {
        auto row=embeddings.slice(1,q,1).view({D});
        ops::ggml_q4_k_embedding_row(embedding,q?248077:panel.tokens[C],row,device.stream);
    }
    const auto hidden=program.draft_embeddings(embeddings,Tensor(dq.p,DType::I32,{K,1}),length,valid,slots,device.stream);
    ops::ggml_block_linear(hidden.view({D,K}),head,logits,device.stream);
    ops::argmax(logits,drafts,V,device.stream);
    device.synchronize();
    auto result=from_device<int>(ids,K);
    std::cout<<"Actual diagnostic shared endpoints, NVFP4 draft proposals: "<<nlohmann::json(result).dump()<<std::endl;
    return result;
}

std::vector<std::uint8_t> snapshot(const q4::verifier::State& state) {
    std::vector<std::uint8_t> result;
    auto add=[&](const Tensor& value) {
        const auto size=result.size();result.resize(size+value.bytes());
        CUDA_CHECK(cudaMemcpy(result.data()+size,value.data,value.bytes(),cudaMemcpyDeviceToHost));
    };
    for(int layer=0;layer<48;++layer) {
        if(state.gdn()[layer]) {add(state.gdn()[layer]->conv);add(state.gdn()[layer]->recurrence);}
        if(state.qsa()[layer]) {
            const auto& q=*state.qsa()[layer];
            add(q.k);add(q.v);add(q.k_scales);add(q.v_scales);add(q.raw_index_keys);add(q.positions);
        }
    }
    add(state.ple_conv_state());add(state.ple_token_history());add(state.residual());
    return result;
}

int verify(const std::filesystem::path& target,const std::filesystem::path& output,
    const qwen4_sequence::TextPanel& panel,const std::vector<int>& draft,DeviceContext& device) {
    auto model=q4::verifier::LoadedModel::load(target,device);
    q4::verifier::Program program(*model,device,q4::verifier::DiagnosticSnapshots::Disabled);
    DeviceBuffer feature_row(F*2);
    auto feature_slot=to_device_i32({0}),feature_valid=to_device_i32({1});
    q4::DFlashFeatureSink sink{Tensor(feature_row.p,DType::BF16,{F,1,1}),
        Tensor(feature_slot.p,DType::I32,{1}),Tensor(feature_valid.p,DType::I32,{1})};
    std::vector<std::uint16_t> prior_features(F*C);
    std::ifstream feature_file(output/"qwen4-dflash-target-features.bin",std::ios::binary);
    feature_file.read(reinterpret_cast<char*>(prior_features.data()),prior_features.size()*2);
    if(!feature_file) throw std::runtime_error("truncated staged context features");
    auto step=[&](int id,bool capture=false) {
        auto out=program.execute_token(id,0,capture?&sink:nullptr);device.synchronize();
        return from_device<std::uint16_t>(out.logits.data,V);
    };
    auto prompt=[&] {program.reset();for(int t=0;t<C;++t) step(panel.tokens[t]);};
    std::vector<std::uint16_t> prompt_hidden,verified_hidden,prompt_next_embedding;
    auto append_hidden=[&](std::vector<std::uint16_t>& destination) {
        const auto h=from_device<std::uint16_t>(program.state().residual().data,10240);
        destination.insert(destination.end(),h.begin(),h.end());
    };
    program.reset();
    DeviceBuffer embedding_row(D*2);
    Tensor row(embedding_row.p,DType::BF16,{D});
    for(int t=0;t<C;++t) {
        step(panel.tokens[t],true);append_hidden(prompt_hidden);
        const auto actual_features=from_device<std::uint16_t>(feature_row,F);
        if(!std::equal(actual_features.begin(),actual_features.end(),prior_features.begin()+t*F))
            throw std::runtime_error("staged DFlash context differs from fresh actual target taps");
        ops::ggml_q4_k_embedding_row(model->view().token_embedding,panel.tokens[t+1],row,device.stream);
        device.synchronize();
        const auto bits=from_device<std::uint16_t>(embedding_row,D);
        prompt_next_embedding.insert(prompt_next_embedding.end(),bits.begin(),bits.end());
    }
    std::vector<std::vector<std::uint8_t>> states;states.push_back(snapshot(program.state()));
    auto da=to_device_i32({panel.tokens[C]}),dd=to_device(draft),de=to_device_i32({K}),db=to_device_i32({C});
    DeviceBuffer verify_ids(W*4),positions(W*4),logit_storage(std::size_t(V)*W*2),target_ids(W*4);
    Tensor anchors(da.p,DType::I32,{1}),drafts(dd.p,DType::I32,{K,1}),extents(de.p,DType::I32,{1}),
        lengths(db.p,DType::I32,{1}),inputs(verify_ids.p,DType::I32,{W,1}),pos(positions.p,DType::I32,{W,1}),
        logits(logit_storage.p,DType::BF16,{V,W,1}),target_tokens(target_ids.p,DType::I32,{W,1});
    ops::speculative_prepare_verify_inputs(anchors,drafts,lengths,extents,inputs,pos,device.stream);
    device.synchronize();
    const auto ids=from_device<int>(verify_ids,W),rope=from_device<int>(positions,W);
    std::vector<std::vector<std::uint16_t>> target_logits;
    for(int t=0;t<W;++t) {
        if(ids[t]!=(t?draft[t-1]:panel.tokens[C]) || rope[t]!=C+t)
            throw std::runtime_error("common verify input mismatch");
        target_logits.push_back(step(ids[t]));
        append_hidden(verified_hidden);
        logit_storage.copy_from_host(target_logits.back().data(),V*2,std::size_t(t)*V*2);
        states.push_back(snapshot(program.state()));
    }
    auto flat_logits=logits.view({V,W}),flat_ids=target_tokens.view({W});
    ops::argmax(flat_logits,flat_ids,V,device.stream);
    DeviceBuffer license_storage(W*4),license_count(4),accept_count(4),occurrences(V*4);
    occurrences.fill();
    Tensor licensed(license_storage.p,DType::I32,{W,1}),count(license_count.p,DType::I32,{1}),accepted(accept_count.p,DType::I32,{1});
    ops::SamplingConfig config;config.temperature=.6F;config.p_less=1;config.seed=1939;
    config.token_counts=static_cast<int*>(occurrences.p);
    auto configs=to_device(std::vector{config});
    WorkspaceArena workspace(ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(V,K,K,1,1));
    ops::speculative_accept_greedy_drafts(target_tokens,logits,drafts,extents,lengths,anchors,licensed,count,accepted,V,
        static_cast<ops::SamplingConfig*>(configs.p),workspace,device.stream);
    device.synchronize();
    const int n=from_device<int>(license_count,1)[0],a=from_device<int>(accept_count,1)[0];
    const auto produced=from_device<int>(license_storage,W),all_counts=from_device<int>(occurrences,V);
    if(n<1 || n>W || a!=n-1 || from_device<int>(db,1)[0]!=C+n || from_device<int>(da,1)[0]!=produced[n-1])
        throw std::runtime_error("common acceptance extent/frontier/anchor mismatch");
    for(int t=0;t<a;++t) if(produced[t]!=draft[t]) throw std::runtime_error("licensed draft mismatch");
    for(int t=n;t<W;++t) if(produced[t]!=0) throw std::runtime_error("licensed invalid tail not zero");
    std::vector<int> expected_counts(V);
    for(int t=0;t<n;++t) ++expected_counts[produced[t]];
    int failures=verify_exact("actual round common sampling counts",all_counts,expected_counts);
    std::vector<int> keeps{0};if(n>1) keeps.push_back(1);keeps.push_back(n);
    nlohmann::json checks=nlohmann::json::array();
    for(int keep:keeps) {
        occurrences.copy_from_host(all_counts.data(),V*4);
        runtime::rollback_sampling_counts(config,std::span(produced).subspan(keep,n-keep));
        std::fill(expected_counts.begin(),expected_counts.end(),0);
        for(int t=0;t<keep;++t) ++expected_counts[produced[t]];
        failures+=verify_exact("actual round published occurrence prefix",from_device<int>(occurrences,V),expected_counts);
        prompt();for(int t=0;t<keep;++t) step(ids[t]);
        failures+=program.frontier()!=C+keep;
        failures+=verify_exact("actual round complete accepted/cancelled state replay",snapshot(program.state()),states[keep]);
        const int next_anchor=keep?produced[keep-1]:panel.tokens[C];
        const auto next_logits=step(next_anchor);
        const auto next_state=snapshot(program.state());
        if(keep<n) {
            failures+=verify_exact("actual round unpublished suffix continuation",next_logits,target_logits[keep]);
            failures+=verify_exact("actual round unpublished suffix state",next_state,states[keep+1]);
        } else {
            // Independent fresh causal replay, including the real correction/bonus, has no rejected suffix.
            prompt();step(panel.tokens[C]);for(int t=0;t<keep-1;++t) step(produced[t]);
            const auto baseline=step(next_anchor);
            failures+=verify_exact("actual correction fresh sequential logits",next_logits,baseline);
            failures+=verify_exact("actual correction fresh sequential full state",next_state,snapshot(program.state()));
        }
        checks.push_back({{"published",keep},{"next_anchor",next_anchor},{"state_bytes",states[keep].size()}});
    }
    nlohmann::json report={{"target",target.string()},{"draft_profile","NVFP4 A16 with actual diagnostic Q4_K embedding/head"},
        {"context",C},{"drafts",draft},{"licensed",std::vector(produced.begin(),produced.begin()+n)},
        {"accepted_drafts",a},{"temperature",.6},{"p_less",true},{"epsilon",ops::kPLessLogitPerturbation},
        {"support_cap",ops::kPLessMaxEffectiveSupport},{"seed",1939},{"vocabulary",V},
        {"checks",checks},{"failures",failures},{"boundary","One natural diagnostic round, not acceptance quality or production speculative scheduling. Exact complete-state reset/replay is deliberately tool-only; no new verifier API or ordinary native-weight streaming."}};
    std::ofstream file(output/"qwen4-dflash-staged-round.json");file<<report.dump(2)<<'\n';
    if(!file) throw std::runtime_error("cannot write staged round report");
    std::ofstream mtp_binary(output/"qwen4-mtp-target-inputs.bin",std::ios::binary);
    nlohmann::json planes=nlohmann::json::object();
    auto plane=[&](const char* name,const std::vector<std::uint16_t>& values,int rows,int columns) {
        planes[name]={{"offset",std::uint64_t(mtp_binary.tellp())},{"bytes",values.size()*2},
            {"shape",{rows,columns}},{"dtype","BF16"}};
        mtp_binary.write(reinterpret_cast<const char*>(values.data()),values.size()*2);
    };
    plane("prompt_hidden",prompt_hidden,C,10240);
    plane("prompt_next_embedding",prompt_next_embedding,C,D);
    plane("verified_hidden",verified_hidden,W,10240);
    if(!mtp_binary) throw std::runtime_error("cannot write actual MTP input capture");
    std::vector<int> prompt_positions(C),verified_positions(W);
    std::iota(prompt_positions.begin(),prompt_positions.end(),0);std::iota(verified_positions.begin(),verified_positions.end(),C);
    nlohmann::json mtp={{"planes",planes},{"target",target.string()},
        {"prompt_token_ids",std::vector(panel.tokens.begin(),panel.tokens.begin()+C)},
        {"aligned_next_token_ids",std::vector(panel.tokens.begin()+1,panel.tokens.begin()+C+1)},
        {"prompt_positions",prompt_positions},{"initial_anchor",panel.tokens[C]},
        {"verified_ids",ids},{"verified_positions",verified_positions},
        {"licensed_ids",std::vector(produced.begin(),produced.begin()+n)},
        {"accepted_drafts",a},{"committed_inputs",n},
        {"boundary","Actual diagnostic four-stream residual R_t after execute_token(x_t), never final_hidden. Prompt next embeddings are exact GPU-decoded diagnostic Q4_K E[x_(t+1)]. Initial anchor is teacher-forced; licensed last token is the sampled correction/bonus. Pair verified_hidden[j] with licensed_ids[j] only for j<committed_inputs."}};
    std::ofstream mtp_json(output/"qwen4-mtp-target-inputs.json");mtp_json<<mtp.dump(2)<<'\n';
    if(!mtp_json) throw std::runtime_error("cannot write actual MTP capture metadata");
    std::cout<<report.dump(2)<<std::endl;
    return failures;
}
} // namespace

int main() {
    const char* target=std::getenv("NINFER_QWEN4_WEIGHTS"),*native=std::getenv("NINFER_QWEN4_NATIVE_LAYERS"),
        *draft=std::getenv("NINFER_QWEN4_NATIVE_DFLASH"),*output=std::getenv("NINFER_QWEN4_DFLASH_TARGET_OUTPUT");
    if(!target || !native || !draft || !output) return 77;
    try {
        if(std::filesystem::exists(std::filesystem::path(output)/"qwen4-dflash-staged-round.json") ||
           std::filesystem::exists(std::filesystem::path(output)/"qwen4-mtp-target-inputs.json") ||
           std::filesystem::exists(std::filesystem::path(output)/"qwen4-mtp-target-inputs.bin"))
            throw std::runtime_error("refusing to replace staged-round report");
        DeviceContext device;qwen4_sequence::TextPanel panel(native);
        const auto draft_ids=proposals(target,draft,output,panel,device);
        // All draft/endpoint owners have drained and destructed before the full target is loaded.
        return verify(target,output,panel,draft_ids,device)?1:0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

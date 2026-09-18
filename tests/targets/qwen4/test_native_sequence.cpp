#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_text_panel.h"
#include "ops/op_tester.h"
#include "ninfer/ops/ngram_embedding.h"
#include <cstdlib>
#include <iostream>

using namespace ninfer::test;
using namespace ninfer::test::qwen4_sequence;
using namespace ninfer;

namespace {
int verify_text_hashes(const TextPanel& panel) {
    const ops::NgramRowConfig config{248320,248044,0,1234,20000000};
    const auto prepared=ops::prepare_ngram_row_config(config);
    std::array<std::int32_t,2> history{248044,248044};
    std::vector<std::int32_t> host_rows;
    for(auto token:panel.tokens) {
        const auto next=ops::ngram_row_ids_host_step(token,history,prepared);
        host_rows.insert(host_rows.end(),next.row_ids.begin(),next.row_ids.end());
        history=next.new_history;
    }
    int failures=verify_exact("native text source-oracle host hashes",host_rows,panel.global_rows);
    for(bool chunked:{false,true}) {
        GuardedDeviceBuffer state(8);
        const std::int32_t initial[]{248044,248044};
        state.copy_from_host(initial,sizeof(initial));
        Tensor state_view(state.data(),DType::I32,{2,1});
        std::vector<std::int32_t> actual;
        for(int begin=0;begin<33;) {
            const int count=chunked?std::min(begin==0?32:1,33-begin):33;
            GuardedDeviceBuffer input(count*4),valid(4),rows(count*16*4);
            input.copy_from_host(panel.tokens.data()+begin,count*4);
            valid.copy_from_host(&count,4);
            Tensor input_view(input.data(),DType::I32,{count,1});
            Tensor valid_view(valid.data(),DType::I32,{1});
            Tensor row_view(rows.data(),DType::I32,{16,count,1});
            ops::ngram_row_ids(input_view,valid_view,state_view,config,row_view,state_view,nullptr);
            cuda_synchronize();
            const auto result=from_device<std::int32_t>(rows.data(),count*16);
            actual.insert(actual.end(),result.begin(),result.end());
            failures+=input.verify_guards("native tokens")+valid.verify_guards("native valid")+
                rows.verify_guards("native hashes");
            begin+=count;
        }
        failures+=verify_exact("native text source-oracle GPU hashes",actual,panel.global_rows);
        failures+=verify_exact("native text hash continuation",from_device<std::int32_t>(state.data(),2),
            std::vector<std::int32_t>{panel.tokens[31],panel.tokens[32]});
        failures+=state.verify_guards("native hash history");
    }
    return failures;
}
}

int main(int argc,char** argv) {
    const bool storage_experiment=argc==2 && std::string(argv[1])=="--fp8-residual-experiment";
    const bool nvfp4_diagnostics=argc==2 && std::string(argv[1])=="--nvfp4-diagnostics";
    const bool selective_a8=argc==2 && std::string(argv[1])=="--native-text-a8";
    const bool calibrated=selective_a8 || (argc==2 && std::string(argv[1])=="--native-text-fp8");
    const bool text_panel=calibrated || (argc==2 && std::string(argv[1])=="--native-text");
    if(argc!=1 && !storage_experiment && !nvfp4_diagnostics && !text_panel) { std::cerr<<"Unknown native-sequence argument\n"; return 1; }
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { std::cout<<"SKIP native sequence: NINFER_QWEN4_NATIVE_LAYERS unset\n"; return 77; }
    if(require_cuda()!=0) { return 1; }
    // Predeclared accumulated gate: BF16 public boundaries, two different stateful
    // mixers, four native MoEs and independently represented BF16/diagnostic NVFP4 KV;
    // not an LM quality gate.
    constexpr ReductionCriterion accumulated{0.02,0.005,0.02};
    int failures=0;
    const auto panel=text_panel?std::make_unique<TextPanel>(root):nullptr;
    if(panel) { failures+=verify_text_hashes(*panel); }
    const std::vector<int> widths=text_panel?std::vector<int>{33}:std::vector<int>{5,17};
    for(bool diagnostic_nvfp4:{false,true}) for(int tokens:widths) {
    // Native admission uses the BF16 reference. The explicitly selected compressed
    // assessment retains its unchanged failed accumulated gate and nonzero exit status.
    if(diagnostic_nvfp4!=nvfp4_diagnostics) { continue; }
    if(storage_experiment && (diagnostic_nvfp4 || tokens!=17)) { continue; }
    if(nvfp4_diagnostics && !diagnostic_nvfp4) { continue; }
    std::cout<<"native contiguous layers0..3 source-BF16 KV="
        <<(diagnostic_nvfp4?"diagnostic-nvfp4":"bf16")<<" T="<<tokens
        <<" projection_profile="<<(selective_a8?"calibrated-selective-a8":calibrated?"calibrated-a16":"bf16-a16")<<'\n';
    Result seed;
    seed.actual.resize(tokens*10240);
    fill_uniform(seed.actual,7239U,-.7F,.7F); round_to_bf16(seed.actual);
    if(panel) { seed.actual=panel->residual; }
    seed.reference=seed.actual;
    std::vector<float> baseline_whole,baseline_reference;
    std::vector<int> baseline_ids;
    for(bool fp8_residual:{false,true}) {
    if(fp8_residual && !storage_experiment) { continue; }
    std::vector<float> whole,whole_reference;
    std::vector<int> whole_ids;
    for(bool partitioned:{false,true}) {
        Result residual=seed;
        std::vector<int> discrete_ids;
        if(fp8_residual) {
            residual=residual_fp8_store(residual,std::string(root)+"/qwen4-layer-0.ninfer",0,"attn");
            failures+=residual.failures;
        }
        for(int layer=0;layer<4;++layer) {
            const auto path=std::string(root)+"/qwen4-layer-"+std::to_string(layer)+".ninfer";
            // Full native first block. Optional text panel uses source token embeddings and
            // exact hash-selected rows; neither panel is a full-model quality measurement.
            if(layer==1) {
                residual=ple(root,residual,partitioned,diagnostic_nvfp4,text_panel);
                failures+=residual.failures;
                failures+=verify_reduction("native source PLE1 accumulated",wide(residual.actual),
                    wide(residual.reference),accumulated);
            }
            if(fp8_residual && layer==1) {
                residual=residual_fp8_store(residual,path,layer,"attn");
                failures+=residual.failures;
            }
            const auto gr=read(path,layer,residual,"attn",partitioned);
            const auto mixer=calibrated && layer==3
                ? qsa_calibrated(root,path,gr.mixed,partitioned)
                : calibrated && layer==0 ? gdn_calibrated(root,path,layer,gr.mixed,partitioned,selective_a8?2:0)
                : layer==3 ? qsa(path,gr.mixed,partitioned,diagnostic_nvfp4)
                           : gdn(path,layer,gr.mixed,partitioned);
            auto attention=inject(residual,mixer,gr.scale,partitioned);
            failures+=attention.failures;
            if(fp8_residual) {
                attention=residual_fp8_store(attention,path,layer,"mlp");
                failures+=attention.failures;
            }
            const auto mlp=read(path,layer,attention,"mlp",partitioned);
            const auto expert=calibrated && (layer==0 || layer==3)
                ? moe_calibrated(root,path,layer,mlp.mixed,partitioned,selective_a8?7:0)
                : moe(path,layer,mlp.mixed,partitioned);
            discrete_ids.insert(discrete_ids.end(),mixer.discrete_ids.begin(),mixer.discrete_ids.end());
            discrete_ids.insert(discrete_ids.end(),expert.discrete_ids.begin(),expert.discrete_ids.end());
            residual=inject(attention,expert,mlp.scale,partitioned);
            failures+=gr.mixed.failures+gr.scale.failures+mixer.failures;
            failures+=mlp.mixed.failures+mlp.scale.failures+expert.failures+residual.failures;
            if(fp8_residual) {
                // Store every branch write, including the value consumed by the next PLE.
                // GR probe uses the next available reader; the final store has no such probe.
                const int next=std::min(layer+1,3);
                residual=residual_fp8_store(residual,
                    std::string(root)+"/qwen4-layer-"+std::to_string(next)+".ninfer",
                    next,layer==3?"":"attn");
                failures+=residual.failures;
            }
            failures+=verify_reduction("native layer"+std::to_string(layer)+" accumulated",
                wide(residual.actual),wide(residual.reference),accumulated);
        }
        if(!partitioned) { whole=residual.actual; whole_reference=residual.reference; whole_ids=discrete_ids; }
        else { failures+=verify_reduction("native sequence whole versus prefill(T-1) decode1",wide(residual.actual),
            std::vector<double>(whole.begin(),whole.end()),accumulated); }
    }
    if(!fp8_residual) { baseline_whole=whole; baseline_reference=whole_reference; baseline_ids=whole_ids; }
    else {
        const auto expected=wide(baseline_whole),candidate=wide(whole);
        const auto loss=compute_reduction_stats(candidate.data(),expected.data(),expected.size());
        const auto original_formula=wide(baseline_reference),stored_formula=wide(whole_reference);
        const auto ideal_loss=compute_reduction_stats(stored_formula.data(),original_formula.data(),original_formula.size());
        if(whole_ids.size()!=baseline_ids.size()) { throw std::runtime_error("storage trace extent changed"); }
        int changed_ids=0;
        for(std::size_t i=0;i<whole_ids.size();++i) { changed_ids+=whole_ids[i]!=baseline_ids[i]; }
        std::cout<<"FP8_RESIDUAL_NATIVE_CHAIN T="<<tokens<<" gpu_relative_l2="<<loss.relative_l2
            <<" gpu_max_abs="<<loss.maximum_absolute_error<<" formula_storage_relative_l2="<<ideal_loss.relative_l2
            <<" selected_id_slot_differences="<<changed_ids
            <<" boundary=decoded-consumer-no-gpu-fp8-codec-no-ppl\n";
        if(loss.first_non_finite>=0 || ideal_loss.first_non_finite>=0) { ++failures; }
    }
    }
    }
    if(storage_experiment || nvfp4_diagnostics || text_panel) { return failures?1:0; }
    // Narrow additional admission witness: MLP GR -> actually-active routed W4A4 MoE ->
    // injection. Repetition guarantees hot expert counts >=32, unlike a random T65 panel.
    // This is not an A4 qualification of the stateful attention/GDN blocks above.
    Result repeated;
    std::vector<float> token(10240);
    fill_uniform(token,7239U,-.7F,.7F); round_to_bf16(token);
    for(int t=0;t<65;++t) { repeated.actual.insert(repeated.actual.end(),token.begin(),token.end()); }
    repeated.reference=repeated.actual;
    for(int layer:{0,3}) {
        std::vector<float> whole;
        const auto path=std::string(root)+"/qwen4-layer-"+std::to_string(layer)+".ninfer";
        for(bool partitioned:{false,true}) {
            const auto gr=read(path,layer,repeated,"mlp",partitioned);
            const auto expert=moe(path,layer,gr.mixed,partitioned,true);
            const auto result=inject(repeated,expert,gr.scale,partitioned);
            failures+=gr.mixed.failures+gr.scale.failures+expert.failures+result.failures;
            failures+=verify_reduction("native active A4 MLP residual accumulated",wide(result.actual),
                wide(result.reference),accumulated);
            if(!partitioned) { whole=result.actual; }
            else { failures+=verify_reduction("native active A4 MLP residual T65 versus64+1",
                wide(result.actual),wide(whole),accumulated); }
        }
    }
    return failures?1:0;
}

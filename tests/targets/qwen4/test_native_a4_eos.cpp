#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_text_panel.h"
#include "ops/op_tester.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace ninfer::test;
using namespace ninfer::test::qwen4_sequence;

int main(int argc,char** argv) {
    const bool attribution_only=argc==2 && std::string_view(argv[1])=="--attribution-only";
    if(argc!=1 && !attribution_only) { throw std::invalid_argument("expected --attribution-only"); }
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { return 77; }
    if(require_cuda()!=0) { return 1; }
    const TextPanel panel(root);
    if(panel.tokens.size()!=33 || panel.tokens[15]!=248044) {
        throw std::runtime_error("real EOS witness changed");
    }
    const auto path=std::string(root)+"/qwen4-layer-0.ninfer";
    Result residual;residual.actual=panel.residual;residual.reference=residual.actual;
    const auto gr=read(path,0,residual,"attn",false);
    const auto mixer=gdn(path,0,gr.mixed,false);
    const auto attention=inject(residual,mixer,gr.scale,false);
    const auto mlp=read(path,0,attention,"mlp",false);
    int failures=gr.mixed.failures+gr.scale.failures+mixer.failures+attention.failures+
        mlp.mixed.failures+mlp.scale.failures;
    const std::vector<float> token(mlp.mixed.actual.begin()+15*2560,mlp.mixed.actual.begin()+16*2560);
    Result repeated;
    for(int t=0;t<65;++t) { repeated.actual.insert(repeated.actual.end(),token.begin(),token.end()); }
    repeated.reference=repeated.actual; // Local represented-input qualification, not accumulated drift.
    std::vector<int> baseline_ids;
    std::vector<float> baseline_output;
    for(bool a4:{false,true}) for(bool partitioned:{false,true}) {
        const auto output=moe(path,0,repeated,partitioned,a4);
        failures+=output.failures;
        if(!a4 && !partitioned) {
            baseline_ids=output.discrete_ids;baseline_output=output.actual;
            moe_a4_calibration_coverage(path,0,token,std::span(baseline_ids).first(10));
            if(attribution_only) { return failures?1:0; }
        }
        failures+=verify_exact("real EOS A4 protected routes",output.discrete_ids,baseline_ids);
        double error2=0,norm2=0,maximum=0,delta2=0;
        for(std::size_t i=0;i<output.actual.size();++i) {
            const double e=double(output.actual[i])-output.reference[i];error2+=e*e;
            norm2+=double(output.reference[i])*output.reference[i];maximum=std::max(maximum,std::abs(e));
            const double delta=double(output.actual[i])-baseline_output[i];delta2+=delta*delta;
        }
        // Supplementary BF16 represented-reference summaries; the adapter already tests
        // each route directly against its complete FP64 oracle using unchanged criteria.
        std::cout<<"A4_REAL_EOS token=15 repeats=65 policy="<<(a4?"A4":"A16")
            <<" chunk="<<partitioned<<" local_failures="<<output.failures
            <<" represented_reference_relative_l2="<<std::sqrt(error2/norm2)
            <<" represented_reference_max_abs="<<maximum
            <<" versus_A16_relative_l2="<<std::sqrt(delta2/norm2)<<std::endl;
    }
    return failures?1:0;
}

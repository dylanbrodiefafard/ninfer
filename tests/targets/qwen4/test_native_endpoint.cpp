#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_sequence_components.h"

#include <cstdlib>
#include <iostream>
#include <thread>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qwen4_sequence;
namespace {
constexpr int hidden=2560, vocabulary=248320;
// Existing BF16 Linear criterion, unchanged. Composition adds one observed BF16 GR
// boundary and uses the existing GR read 0.6% budget, not a model quality tolerance.
constexpr ReductionCriterion linear_gate{1./256,1./256,2./256};
constexpr ReductionCriterion composition_gate{.006,.004,.01};

std::vector<double> project(const direct_bf16_weight::HostWeight& weight,
                            const std::vector<float>& x) {
    const int tokens=x.size()/hidden;
    std::vector<double> out(vocabulary*tokens);
    const unsigned threads=std::min(32U,std::max(1U,std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for(unsigned thread=0;thread<threads;++thread) {
        workers.emplace_back([&,thread] {
            for(int row=thread;row<vocabulary;row+=threads) {
                for(int t=0;t<tokens;++t) {
                    out[t*vocabulary+row]=direct_bf16_weight::dot_fp64(weight,row,
                        std::span<const float>(x.data()+t*hidden,hidden));
                }
            }
        });
    }
    for(auto& worker:workers) { worker.join(); }
    return out;
}

std::vector<float> execute(const direct_bf16_weight::DeviceWeight& weight,
                            const std::vector<float>& x,int& failures,bool partitioned=false) {
    const int tokens=x.size()/hidden;
    auto dx=to_device_bf16(x);
    GuardedDeviceBuffer dy(std::size_t(vocabulary)*tokens*2);
    Tensor input(dx.p,DType::BF16,{hidden,tokens}),output(dy.data(),DType::BF16,{vocabulary,tokens});
    WorkspaceArena workspace(256);
    for(int begin=0;begin<tokens;) {
        const int count=partitioned && begin==0?std::max(1,tokens-1):tokens-begin;
        auto cx=input.slice(1,begin,count),cy=output.slice(1,begin,count);
        ops::linear(cx,weight.view(),cy,ops::LinearPolicy::A16Only,workspace,nullptr);
        begin+=count;
    }
    cuda_synchronize();
    failures+=dy.verify_guards("native BF16 head output");
    const auto result=from_device_bf16(dy.data(),std::size_t(vocabulary)*tokens);
    return {result.begin(),result.end()};
}
}

int main() {
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { std::cout<<"SKIP native endpoint: source fixture unset\n"; return 77; }
    if(require_cuda()!=0) { return 1; }
    const auto path=std::string(root)+"/qwen4-endpoint.ninfer";
    qwen4_native::Bf16Source source(path);
    direct_bf16_weight::DeviceWeight head({vocabulary,hidden,source.bits("lm_head.weight",{vocabulary,hidden})});
    int failures=0;
    // Three distinct represented columns reused to cover every output row at decode,
    // small-T cutoff, MMA tail and full-tile+tail without duplicating expensive CPU dots.
    std::vector<float> panel(hidden*3);
    fill_uniform(panel,81091U,-.75F,.75F); round_to_bf16(panel);
    const auto oracle=project(head.host,panel);
    for(int tokens:{1,2,27,28,33,129}) {
        std::vector<float> input(hidden*tokens);
        for(int t=0;t<tokens;++t) {
            std::copy_n(panel.data()+(t%3)*hidden,hidden,input.data()+t*hidden);
        }
        const auto result=execute(head,input,failures);
        double worst=0;
        for(int t=0;t<tokens;++t) {
            const std::vector<double> actual(result.begin()+std::size_t(t)*vocabulary,
                                             result.begin()+std::size_t(t+1)*vocabulary);
            const std::vector<double> expected(oracle.begin()+(t%3)*vocabulary,oracle.begin()+(t%3+1)*vocabulary);
            failures+=verify_reduction("native head T="+std::to_string(tokens),actual,expected,linear_gate);
            worst=std::max(worst,compute_reduction_stats(actual.data(),expected.data(),actual.size()).relative_l2);
        }
        std::cout<<"native BF16 head T="<<tokens<<" all_vocab_worst_relL2="<<worst<<'\n';
    }
    Result residual;
    residual.actual.resize(10240*3);
    fill_uniform(residual.actual,82911U,-.7F,.7F); round_to_bf16(residual.actual);
    residual.reference=residual.actual;
    std::vector<double> expected;
    for(bool partitioned:{false,true}) {
        const auto read=final_read(path,residual,partitioned);
        failures+=read.failures;
        const auto local=project(head.host,read.actual);
        if(expected.empty()) { expected=project(head.host,read.reference); }
        const auto actual=wide(execute(head,read.actual,failures,partitioned));
        for(int t=0;t<3;++t) {
            const auto first=std::size_t(t)*vocabulary,last=first+vocabulary;
            const std::vector<double> actual_token(actual.begin()+first,actual.begin()+last);
            const std::vector<double> local_token(local.begin()+first,local.begin()+last);
            const std::vector<double> expected_token(expected.begin()+first,expected.begin()+last);
            failures+=verify_reduction("native final head local token="+std::to_string(t),
                actual_token,local_token,linear_gate);
            failures+=verify_reduction("native final GR-read to head token="+std::to_string(t),
                actual_token,expected_token,composition_gate);
        }
        std::cout<<"native endpoint T=3 partitioned="<<partitioned<<" local_relL2="
            <<compute_reduction_stats(actual.data(),local.data(),actual.size()).relative_l2
            <<" composed_relL2="<<compute_reduction_stats(actual.data(),expected.data(),actual.size()).relative_l2<<'\n';
    }
    std::cout<<(failures?"FAIL":"PASS")<<" native endpoint\n";
    return failures?1:0;
}

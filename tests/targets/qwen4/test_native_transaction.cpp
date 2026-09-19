#include "targets/qwen4/transaction.h"
#include "artifact/reader.h"
#include "core/decode_graph.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::targets::qwen4;
namespace {
constexpr int F=10240,D=2560,V=248320,W=4;
constexpr ReductionCriterion fold_criterion{2.7e-3,1e-5,3.9e-3};
constexpr ReductionCriterion composition{.02,.005,.02};
template<class Function> int rejects(const char* label,Function&& f) {
    try {f();} catch(const std::logic_error&) {return 0;}
    std::cerr<<"FAIL missing transaction boundary rejection: "<<label<<'\n';return 1;
}
std::vector<std::uint8_t> qsa_prefix(const NativeFirstBlockState& state,int frontier) {
    std::vector<std::uint8_t> result;
    auto append=[&](const void* data,std::size_t bytes) {
        auto part=from_device<std::uint8_t>(data,bytes);result.insert(result.end(),part.begin(),part.end());
    };
    for(auto plane:{state.qsa.k,state.qsa.v}) for(int head=0;head<2;++head)
        append(static_cast<const std::byte*>(plane.data)+head*plane.ne[1]*256*2,frontier*256*2);
    append(state.qsa.raw_index_keys.data,frontier*128*2);append(state.qsa.positions.data,frontier*3*4);
    return result;
}

std::vector<std::uint8_t> live(const NativeTransaction& owner,int slot) {
    std::vector<std::uint8_t> bytes;
    auto append=[&](const Tensor& tensor) {
        auto part=from_device<std::uint8_t>(tensor.data,tensor.bytes());
        bytes.insert(bytes.end(),part.begin(),part.end());
    };
    const auto state=owner.state(slot);
    for(auto layer:state.gdn) {append(layer.conv);append(layer.recurrence);}
    append(state.ple_conv);append(owner.raw_token_history(slot));append(owner.continuation(slot));
    append(owner.sampling_counts(slot));
    const int frontier=owner.frontier(slot);
    if(frontier) {
        for(auto plane:{state.qsa.k,state.qsa.v}) for(int head=0;head<2;++head)
            append(Tensor(static_cast<std::byte*>(plane.data)+head*plane.ne[1]*256*2,DType::BF16,{256,frontier}));
        append(Tensor(state.qsa.raw_index_keys.data,DType::BF16,{128,frontier}));
        append(Tensor(state.qsa.positions.data,DType::I32,{3,frontier}));
    }
    const auto position=owner.anchor_position(slot);
    const std::array host{frontier,owner.anchor(slot),position[0],position[1],position[2]};
    auto add_host=[&](const void* p,std::size_t n) {
        const auto* data=static_cast<const std::uint8_t*>(p);bytes.insert(bytes.end(),data,data+n);
    };
    add_host(host.data(),sizeof(host));
    add_host(owner.published(slot).data(),owner.published(slot).size_bytes());
    return bytes;
}
struct Panel {
    DeviceBuffer hidden,embedding;
    explicit Panel(const std::string& root,cudaStream_t stream):embedding(D*16*2) {
        std::vector<float> values(F*16);fill_uniform(values,7239U,-.7F,.7F);round_to_bf16(values);
        hidden=to_device_bf16(values);
        artifact::Reader rows(root+"/qwen4-ple-rows.ninfer");
        const auto payload=rows.payload("ple.rows").data;
        std::vector<std::uint8_t> codes(16*D);
        for(int i=0;i<16*16;++i) for(int d=0;d<160;++d)
            codes[i*160+d]=std::to_integer<unsigned>(payload[((i*7+i/16)%16)*160+d]);
        const unsigned scale=std::to_integer<unsigned>(payload[16*160])|
                             (std::to_integer<unsigned>(payload[16*160+1])<<8);
        auto packed=to_device(codes);
        Tensor in(packed.p,DType::U8,{160,16,16}),out(embedding.p,DType::BF16,{160,16,16});
        ops::ple_fp8_decode_rows(in,scale,out,stream);CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    void prepare(NativeTransaction& owner,int slot,std::span<const int> tokens,int offset=0) {
        std::array<std::array<int,3>,16> positions;
        const auto base=owner.anchor_position(slot);
        for(std::size_t t=0;t<tokens.size();++t) for(int axis=0;axis<3;++axis) positions[t][axis]=base[axis]+t;
        Tensor h(static_cast<std::byte*>(hidden.p)+offset*F*2,DType::BF16,{F,static_cast<int>(tokens.size())});
        Tensor e(static_cast<std::byte*>(embedding.p)+offset*D*2,DType::BF16,{D,static_cast<int>(tokens.size())});
        owner.prepare(slot,tokens,std::span(positions).first(tokens.size()),h,e);
    }
};
NativeResolution resolution(int slot,std::span<const int> licensed,int count,ops::SamplingConfig config={}) {
    return {slot,{licensed},{static_cast<unsigned>(count),FinishReason::None,count==0},config};
}
void seed(NativeTransaction& owner,int slot,Panel& panel) {
    owner.reset(slot,11,{41,9,3});
    const std::array input{11,13},licensed{13,17};
    panel.prepare(owner,slot,input);owner.enqueue(slot);
    const std::array rows{resolution(slot,licensed,2)};owner.resolve(rows);owner.retain(slot);
}
std::vector<float> bf16_values(const Tensor& tensor,int count) {
    const auto values=from_device_bf16(tensor.data,count);
    return {values.begin(),values.end()};
}
struct FoldOracle {
    std::array<std::vector<double>,3> recurrent;
    std::array<std::vector<std::uint16_t>,3> conv;
    std::vector<std::uint16_t> ple;
    std::vector<int> history;
};
FoldOracle oracle(const NativeTransaction& owner,int slot,std::span<const int> tokens,int accepted) {
    FoldOracle result;
    const auto before=owner.state(slot);const auto records=owner.records(slot);
    auto exact_tail=[&](const Tensor& state,const Tensor& record,int history) {
        auto words=from_device<std::uint16_t>(state.data,state.numel());
        const auto appended=from_device<std::uint16_t>(record.data,accepted*F);
        words.insert(words.end(),appended.begin(),appended.end());
        return std::vector<std::uint16_t>(words.end()-F*history,words.end());
    };
    for(int layer=0;layer<3;++layer) {
        const auto initial=from_device<float>(before.gdn[layer].recurrence.data,128*128*48);
        result.conv[layer]=exact_tail(before.gdn[layer].conv,records.gdn[layer].conv,3);
        if(!accepted) {result.recurrent[layer]={initial.begin(),initial.end()};continue;}
        gdn_ref::Inputs in;
        in.head_dim=128;in.qk_heads=48;in.value_heads=48;in.tokens=accepted;in.state=initial;
        in.q.assign(128*48*accepted,0);in.k=bf16_values(records.gdn[layer].key,128*48*accepted);
        in.v=bf16_values(records.gdn[layer].value,128*48*accepted);
        const auto controls=from_device<float>(records.gdn[layer].gate.data,2*48*accepted);
        in.g.resize(48*accepted);in.beta.resize(48*accepted);
        for(int i=0;i<48*accepted;++i) {in.g[i]=controls[2*i];in.beta[i]=controls[2*i+1];}
        result.recurrent[layer]=gdn_ref::evaluate(in,1/std::sqrt(128.),true).final_state;
    }
    result.ple=exact_tail(before.ple_conv,records.ple_conv,9);
    result.history=from_device<int>(owner.raw_token_history(slot).data,2);
    result.history.insert(result.history.end(),tokens.begin(),tokens.begin()+accepted);
    result.history.erase(result.history.begin(),result.history.end()-2);
    return result;
}
int verify_oracle(const NativeTransaction& owner,int slot,const FoldOracle& reference) {
    const auto state=owner.state(slot);int failures=0;
    for(int layer=0;layer<3;++layer) {
        const auto values=from_device<float>(state.gdn[layer].recurrence.data,128*128*48);
        failures+=verify_reduction("native transaction independent GDN recurrence",
            std::vector<double>(values.begin(),values.end()),reference.recurrent[layer],fold_criterion);
        failures+=verify_exact("native transaction exact GDN convolution prefix",
            from_device<std::uint16_t>(state.gdn[layer].conv.data,F*3),reference.conv[layer]);
    }
    failures+=verify_exact("native transaction exact PLE convolution prefix",
        from_device<std::uint16_t>(state.ple_conv.data,F*9),reference.ple);
    failures+=verify_exact("native transaction raw history including EOS",
        from_device<int>(owner.raw_token_history(slot).data,2),reference.history);
    return failures;
}
int prefixes(const LoadedNativeFirstBlock& model,Panel& panel,DeviceContext& device) {
    NativeTransaction owner(model,4,64,16,device.stream),fresh(model,1,64,16,device.stream);
    seed(owner,0,panel);seed(fresh,0,panel);
    const auto retained=live(owner,0),inactive=live(owner,3);
    const auto historical_qsa=qsa_prefix(owner.state(0),2);
    const std::array verified{17,248044,19,23},licensed{248044,19,23,29};
    int failures=0;
    panel.prepare(owner,0,verified,2);
    const std::array premature{resolution(0,licensed,1)};
    failures+=rejects("stale previous records cannot publish prepared-only input",[&]{owner.resolve(premature);});
    const std::array cancel{resolution(0,{},0)};owner.resolve(cancel);
    failures+=verify_exact("prepared-only cancellation unchanged",live(owner,0),retained);
    for(int accepted=0;accepted<=W;++accepted) {
        owner.restore(0);fresh.restore(0);
        panel.prepare(owner,0,verified,2);owner.enqueue(0);
        failures+=rejects("duplicate eager enqueue",[&]{owner.enqueue(0);});device.synchronize();
        const auto expected=oracle(owner,0,verified,accepted);
        const std::array rows{resolution(0,licensed,accepted)};owner.resolve(rows);
        failures+=verify_oracle(owner,0,expected);
        failures+=verify_exact("historical QSA KV/index/positions unchanged",qsa_prefix(owner.state(0),2),historical_qsa);
        failures+=owner.frontier(0)!=2+accepted;
        failures+=owner.anchor(0)!=(accepted?licensed[accepted-1]:17);
        failures+=owner.anchor_position(0)!=std::array{43+accepted,11+accepted,5+accepted};
        failures+=verify_exact("native inactive request unchanged",live(owner,3),inactive);
        if(!accepted) failures+=verify_exact("native cancellation complete state unchanged",live(owner,0),retained);
        else {
            panel.prepare(fresh,0,std::span(verified).first(accepted),2);fresh.enqueue(0);
            const std::array direct{resolution(0,std::span(licensed).first(accepted),accepted)};fresh.resolve(direct);
            for(int layer=0;layer<3;++layer) {
                const auto a=from_device<float>(owner.state(0).gdn[layer].recurrence.data,128*128*48);
                const auto b=from_device<float>(fresh.state(0).gdn[layer].recurrence.data,128*128*48);
                failures+=verify_reduction("native prefix fresh replay supplementary recurrence",
                    std::vector<double>(a.begin(),a.end()),std::vector<double>(b.begin(),b.end()),composition);
            }
            failures+=verify_reduction("native accepted continuation fresh replay",
                from_device_bf16(owner.continuation(0).data,F),from_device_bf16(fresh.continuation(0).data,F),composition);
        }
        owner.restore(0);
        failures+=verify_exact("native retained live state and publication restored",live(owner,0),retained);
    }
    // Width16 exercises a convolution suffix longer than either history and the record maximum.
    std::array<int,16> long_input,long_licensed;
    for(int i=0;i<16;++i) {long_input[i]=i?100+i:17;long_licensed[i]=i<15?101+i:200;}
    panel.prepare(owner,0,long_input);owner.enqueue(0);device.synchronize();
    const auto expected=oracle(owner,0,long_input,16);
    const std::array rows{resolution(0,long_licensed,16)};owner.resolve(rows);
    failures+=verify_oracle(owner,0,expected);
    return failures;
}

int graphs(const LoadedNativeFirstBlock& model,Panel& panel,DeviceContext& device) {
    int failures=0;
    for(int requests=1;requests<=4;++requests) {
        NativeTransaction graph_owner(model,requests,32,2,device.stream),eager(model,requests,32,2,device.stream);
        std::array<std::array<int,2>,4> input,licensed;
        auto prepare=[&](NativeTransaction& owner,int round) {
            for(int slot=0;slot<requests;++slot) {
                input[slot]={owner.anchor(slot),100+slot+round*10};licensed[slot]={input[slot][1],200+slot+round*10};
                panel.prepare(owner,slot,input[slot],slot*2);
            }
        };
        for(int slot=0;slot<requests;++slot) {
            graph_owner.reset(slot,7+slot,{11+slot*9,3+slot,19+slot*2});
            eager.reset(slot,7+slot,{11+slot*9,3+slot,19+slot*2});
        }
        prepare(graph_owner,0);device.synchronize();
        const std::array slots{0,1,2,3};
        auto graph=graph_owner.capture_graph(std::span(slots).first(requests));
        const std::array premature{resolution(0,licensed[0],1)};
        failures+=rejects("capture alone cannot publish",[&]{graph_owner.resolve(premature);});
        for(int round=0;round<2;++round) {
            if(round) prepare(graph_owner,round);
            graph->launch();
            failures+=rejects("duplicate graph launch",[&]{graph->launch();});
            std::array<NativeResolution,4> result;
            for(int row=0;row<requests;++row) {
                const int slot=requests-1-row,count=(slot+round)%3;
                result[row]=resolution(slot,licensed[slot],count);
            }
            graph_owner.resolve(std::span(result).first(requests));
            failures+=rejects("duplicate publication",[&]{graph_owner.resolve(std::span(result).first(requests));});
            prepare(eager,round);
            for(int slot=0;slot<requests;++slot) eager.enqueue(slot);
            eager.resolve(std::span(result).first(requests));
            for(int slot=0;slot<requests;++slot)
                failures+=verify_exact("native C1..4 graph/eager complete transaction",live(graph_owner,slot),live(eager,slot));
        }
        const auto before=live(graph_owner,0);
        const std::array only_anchor{graph_owner.anchor(0)};panel.prepare(graph_owner,0,only_anchor);
        failures+=rejects("graph fixed slot/width provenance",[&]{graph->launch();});
        const std::array cancel{resolution(0,{},0)};graph_owner.resolve(cancel);
        failures+=verify_exact("wrong-width graph rejection unchanged",live(graph_owner,0),before);
    }
    return failures;
}

int common_sampling(const LoadedNativeFirstBlock& model,Panel& panel,DeviceContext& device) {
    NativeTransaction owner(model,1,32,W,device.stream);
    owner.reset(0,17,{8,11,3});owner.retain(0);
    const auto retained=live(owner,0);
    // Controlled target logits isolate the publication transaction, not model acceptance
    // quality. The real shared p-less accept law (including epsilon/floor) is invoked.
    const std::array verified{17,19,19,19},expected_licensed{19,19,19,31};
    std::vector<float> logits(V*W,-80.F);
    for(int t=0;t<W;++t) {logits[t*V+expected_licensed[t]]=4.F;logits[t*V+37]=0.F;}
    auto dl=to_device_bf16(logits),dt=to_device(std::vector<int>(expected_licensed.begin(),expected_licensed.end())),
        dd=to_device(std::vector<int>{19,19,19}),de=to_device(std::vector<int>{3});
    DeviceBuffer lengths(4),anchors(4),licensed(16),counts(4),accepted(4);
    Tensor lt(dl.p,DType::BF16,{V,W,1}),tt(dt.p,DType::I32,{W,1}),drafts(dd.p,DType::I32,{3,1}),
        extents(de.p,DType::I32,{1}),length_tensor(lengths.p,DType::I32,{1}),
        anchor_tensor(anchors.p,DType::I32,{1}),license_tensor(licensed.p,DType::I32,{W,1}),
        count_tensor(counts.p,DType::I32,{1}),accept_tensor(accepted.p,DType::I32,{1});
    WorkspaceArena workspace(ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(V,3,3,1,1));
    ops::SamplingConfig config;
    config.temperature=.6F;config.p_less=1;config.seed=1939;
    config.token_counts=static_cast<int*>(owner.sampling_counts(0).data);
    auto dc=to_device(std::vector{config});
    int failures=0;
    for(int keep:{0,1,2,4}) {
        owner.restore(0);
        const int initial_length=owner.frontier(0),initial_anchor=owner.anchor(0);
        lengths.copy_from_host(&initial_length,4);anchors.copy_from_host(&initial_anchor,4);
        panel.prepare(owner,0,verified);owner.enqueue(0);
        ops::speculative_accept_greedy_drafts(tt,lt,drafts,extents,length_tensor,anchor_tensor,
            license_tensor,count_tensor,accept_tensor,V,static_cast<ops::SamplingConfig*>(dc.p),workspace,device.stream);
        device.synchronize();
        const auto generated=from_device<int>(licensed,W);
        failures+=verify_exact("shared p-less licensed tokens",generated,
            std::vector<int>(expected_licensed.begin(),expected_licensed.end()));
        failures+=from_device<int>(counts,1)[0]!=4 || from_device<int>(accepted,1)[0]!=3;
        // Shared accept mutates scratch cursor/counters, never authoritative host publication.
        failures+=from_device<int>(lengths,1)[0]!=4 || from_device<int>(anchors,1)[0]!=31;
        failures+=owner.frontier(0)!=0 || owner.anchor(0)!=17 || !owner.published(0).empty();
        const auto expected=oracle(owner,0,verified,keep);
        const std::array result{resolution(0,generated,keep,config)};owner.resolve(result);
        failures+=verify_oracle(owner,0,expected);
        const auto actual_counts=from_device<int>(owner.sampling_counts(0).data,V);
        std::vector<int> expected_counts(V);
        for(int i=0;i<keep;++i) ++expected_counts[generated[i]];
        failures+=verify_exact("shared speculative counts rollback exact published prefix",actual_counts,expected_counts);
        failures+=owner.frontier(0)!=keep || owner.published(0).size()!=static_cast<std::size_t>(keep);
        failures+=owner.anchor(0)!=(keep?generated[keep-1]:17);
        failures+=owner.anchor_position(0)!=std::array{8+keep,11+keep,3+keep};
        if(!keep) failures+=verify_exact("common accept cancellation restores all owners",live(owner,0),retained);
        owner.restore(0);
        failures+=verify_exact("retained sampling counts and logical RNG position",live(owner,0),retained);
    }
    return failures;
}

int run(const std::string& source,const std::string& prepared) {
    DeviceContext device;
    auto model=LoadedNativeFirstBlock::load(prepared,source+"/qwen4-ple-component.ninfer",device);
    Panel panel(source,device.stream);
    int failures=prefixes(*model,panel,device)+graphs(*model,panel,device)+common_sampling(*model,panel,device);
    std::cout<<"Native speculative prefix/retention/C1..4 graph failures="<<failures<<'\n';
    return failures?1:0;
}
}
int main() {
    const char* source=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    const char* prepared=std::getenv("NINFER_QWEN4_NATIVE_COMPUTE");
    if(!source || !prepared) {std::cout<<"SKIP native transaction artifacts unset\n";return 77;}
    try {return run(source,prepared);} catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}

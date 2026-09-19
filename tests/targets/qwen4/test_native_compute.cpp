#include "targets/qwen4/native_compute.h"
#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_decoder.h"
#include "targets/qwen4/native_runtime_layout.h"
#include "artifact/reader.h"
#include "core/decode_graph.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <cstdlib>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::targets::qwen4;
namespace seq=ninfer::test::qwen4_sequence;
namespace {
constexpr int D=2560,F=10240,T=5;
constexpr ReductionCriterion accumulated{.02,.005,.02};
// Exercise the exact native runtime's layer-major body with the authentic four-layer prefix.
// No layer is repeated to stand in for the absent 44 layers. The complete NativeState allocation
// is used, but only the three executed GDN layers and one executed QSA layer are inspected.
int shared_decoder(const LoadedNativeFirstBlock& model,const Tensor& hidden,const Tensor& ple,
                   std::span<const double> reference,DeviceContext& device) {
    NativeRuntimeConfig config;config.requests=4;config.context_tokens=128;config.kv_tokens=512;
    config.prefill_width=65;config.verify_width=3;config.vision=false;
    NativeState state({4,128,512,3,ops::QsaKvFormat::BF16},device.stream);
    WorkspaceArena shared(native_layout::shared_bytes(config));auto s=native_layout::allocate(shared,config);
    const int n=native_layout::columns(config);
    std::size_t scratch=ops::ple_workspace_capacity_bytes(n,model.ple().key.qtype,model.ple().value.qtype);
    for(int l=0;l<4;++l) {
        const auto& w=model.layers()[l];
        scratch=std::max({scratch,ops::gated_residual_workspace_capacity_bytes(n,w.attention_gr.down.qtype,w.attention_gr.up.qtype),
            ops::gated_residual_workspace_capacity_bytes(n,w.moe_gr.down.qtype,w.moe_gr.up.qtype),
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(w.moe,n)});
        if(l<3)scratch=std::max(scratch,ops::gated_delta_net_layer_batch_workspace_capacity_bytes(n,1,
            w.gdn.qkv.qtype,w.gdn.z.qtype,w.gdn.output.qtype));
        if(l<3)scratch=std::max(scratch,ops::gated_delta_net_layer_workspace_capacity_bytes(n,
            w.gdn.qkv.qtype,w.gdn.z.qtype,w.gdn.output.qtype));
    }
    WorkspaceArena workspace(scratch);
    const auto qsa_size=std::max(ops::qsa_verifier_workspace_bytes(n,1,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL),
        ops::qsa_verifier_workspace_bytes(3,4,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL));
    DeviceBuffer qsa_buffer(qsa_size);Tensor qsa(qsa_buffer.p,DType::U8,{int(qsa_size)});
    auto v=[](const Tensor& t,std::initializer_list<int> shape){return Tensor(t.data,t.dtype,shape);};
    auto upload=[&](const Tensor& t,const auto& values) {
        CUDA_CHECK(cudaMemcpy(t.data,values.data(),values.size()*sizeof(values[0]),cudaMemcpyHostToDevice));
    };
    auto body=[&](int width,int batch,int full_prefill_slot=-1) {
        workspace.reset();
        const ops::QsaBatchControls controls{v(s.slots,{batch}),v(s.valid,{batch}),v(s.frontiers,{batch}),v(s.positions,{3,width,batch})};
        NativeDecoderViews views{v(s.residual,{D,4,width,batch}),v(s.ple_embedding,{D,width,batch}),
            v(s.mixed,{D,width,batch}),v(s.block,{D,width,batch}),v(s.scale,{4,width,batch}),
            v(s.routes,{10,width,batch}),v(s.probabilities,{10,width,batch}),v(s.selected,{2051,width,batch}),
            v(s.selected_count,{width,batch}),{},v(s.compact_rows,{batch})};
        enqueue_native_decoder(model.layers(),model.ple(),state,controls,128,views,false,workspace,qsa,device.stream,full_prefill_slot);
    };
    std::array<int,4> slots{2,0,3,1},lengths{1,1,1,1},frontiers{};
    auto reset=[&] {for(int i=0;i<4;++i){state.reset(i);state.reserve(i,128);}device.synchronize();};
    auto commit=[&](int width,int batch) {
        auto counts=v(s.counts,{batch});
        state.commit(std::span(slots).first(batch),std::span(lengths).first(batch),std::span(lengths).first(batch),
            width,false,v(s.ids,{width,batch}),v(s.residual,{F,width,batch}),v(s.slots,{batch}),counts);
    };
    reset();slots[0]=0;lengths[0]=T;
    state.materialize(0,T);std::array<int,3*T> positions{};std::array<int,T> ids{};
    for(int t=0;t<T;++t){positions[3*t]=t;positions[3*t+1]=t+1;positions[3*t+2]=t+2;ids[t]=100+t;}
    upload(s.positions,positions);upload(s.ids,ids);upload(s.slots,slots);upload(s.valid,lengths);upload(s.frontiers,frontiers);
    CUDA_CHECK(cudaMemcpyAsync(s.residual.data,hidden.data,hidden.bytes(),cudaMemcpyDeviceToDevice,device.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.ple_embedding.data,ple.data,ple.bytes(),cudaMemcpyDeviceToDevice,device.stream));
    body(T,1);commit(T,1);device.synchronize();
    int failures=verify_reduction("shared native paged decoder actual0..3 independent accumulated oracle",
        from_device_bf16(s.residual.data,F*T),reference,accumulated);
    slots={2,0,3,1};lengths.fill(1);
    auto prepare=[&](int round) {
        device.synchronize();std::array<int,12> pos{};std::array<int,4> tokens{};
        for(int b=0;b<4;++b) {
            const int col=(b+round)%T;frontiers[b]=state.frontier(slots[b]);state.materialize(slots[b],frontiers[b]+1);
            tokens[b]=100+col;pos[3*b]=37+b*11+round;pos[3*b+1]=13+b+round;pos[3*b+2]=5+b*3+round;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.residual.data)+b*F*2,hidden.slice(1,col,1).data,F*2,
                                       cudaMemcpyDeviceToDevice,device.stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.ple_embedding.data)+b*D*2,ple.slice(1,col,1).data,D*2,
                                       cudaMemcpyDeviceToDevice,device.stream));
        }
        upload(s.positions,pos);upload(s.ids,tokens);upload(s.slots,slots);upload(s.valid,lengths);upload(s.frontiers,frontiers);
    };
    auto snapshot=[&] {
        std::vector<std::uint8_t> bytes;
        auto append=[&](const Tensor& t){auto part=from_device<std::uint8_t>(t.data,t.bytes());bytes.insert(bytes.end(),part.begin(),part.end());};
        for(int l=0;l<3;++l){append(state.gdn(false).conv[l]);append(state.gdn(false).recurrent[l]);}
        append(state.ple(false));append(state.token_history());append(state.continuation());
        // Only defined cache rows are observable; never compare uninitialized page tails.
        const auto cache=state.qsa(0);const auto tables=from_device<int>(cache.block_tables.data,cache.block_tables.numel());
        for(int slot=0;slot<4;++slot)for(int t=0;t<state.frontier(slot);++t) {
            const int page=tables[slot*cache.block_tables.ne[0]+t/64],offset=t%64;
            for(const auto* plane:{&cache.k,&cache.v})for(int h=0;h<2;++h)
                append(Tensor(static_cast<std::byte*>(plane->data)+((page*2+h)*64+offset)*256*2,DType::BF16,{256}));
            append(Tensor(static_cast<std::byte*>(cache.raw_index_keys.data)+(page*64+offset)*128*2,DType::BF16,{128}));
            append(Tensor(static_cast<std::byte*>(cache.positions.data)+(page*64+offset)*3*4,DType::I32,{3}));
        }
        return bytes;
    };
    std::array<std::vector<double>,2> eager;
    std::array<std::vector<std::uint8_t>,2> eager_state;
    reset();
    for(int round=0;round<2;++round){prepare(round);body(1,4);commit(1,4);device.synchronize();
        eager[round]=from_device_bf16(s.residual.data,F*4);eager_state[round]=snapshot();}
    reset();prepare(0);device.synchronize();DecodeGraphDefinition definition;DecodeGraphExecutable graph;
    definition.capture(device.stream,[&]{body(1,4);});graph.instantiate(definition);graph.upload(device.stream);
    for(int round=0;round<2;++round){if(round)prepare(round);graph.launch(device.stream);commit(1,4);device.synchronize();
        failures+=verify_exact("shared native compactB4 graph/eager represented output",from_device_bf16(s.residual.data,F*4),eager[round]);
        failures+=verify_exact("shared native compactB4 graph/eager real-prefix owned state",snapshot(),eager_state[round]);}
    // A real65-row prefix crosses the chunked GDN route boundary. This supplementary route
    // comparison checks the shared schedule; independent GDN/GR/PLE/QSA/MoE oracles remain
    // the mathematical admission evidence, and the above five-row whole formula is unchanged.
    std::array<std::vector<double>,2> prefill;
    for(int route=0;route<2;++route) {
        reset();slots[0]=2;lengths[0]=65;frontiers.fill(0);state.materialize(2,65);
        std::array<int,65> tokens{};std::array<int,195> pos{};
        for(int t=0;t<65;++t) {
            tokens[t]=100+t%T;pos[3*t]=t;pos[3*t+1]=t+1;pos[3*t+2]=t+2;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.residual.data)+std::size_t(t)*F*2,
                hidden.slice(1,t%T,1).data,F*2,cudaMemcpyDeviceToDevice,device.stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.ple_embedding.data)+std::size_t(t)*D*2,
                ple.slice(1,t%T,1).data,D*2,cudaMemcpyDeviceToDevice,device.stream));
        }
        upload(s.positions,pos);upload(s.ids,tokens);upload(s.slots,slots);upload(s.valid,lengths);upload(s.frontiers,frontiers);
        body(65,1,route?2:-1);commit(65,1);device.synchronize();prefill[route]=from_device_bf16(s.residual.data,F*65);
    }
    failures+=verify_reduction("shared native real-prefix chunked GDN route",prefill[1],prefill[0],accumulated);
    return failures;
}
std::vector<std::uint8_t> snapshot(const NativeFirstBlockState& state) {
    std::vector<std::uint8_t> result;
    auto append=[&](const Tensor& t) {
        auto bytes=from_device<std::uint8_t>(t.data,t.bytes());
        result.insert(result.end(),bytes.begin(),bytes.end());
    };
    for(auto g:state.gdn) { append(g.conv); append(g.recurrence); }
    append(state.ple_conv); append(state.qsa.k); append(state.qsa.v);
    append(state.qsa.raw_index_keys); append(state.qsa.positions);
    return result;
}
int run(const std::string& source,const std::string& prepared) {
    DeviceContext device;
    auto model=LoadedNativeFirstBlock::load(prepared,source+"/qwen4-ple-component.ninfer",device);
    NativeFirstBlock runtime(*model,4,32,T);
    seq::Result seed;
    seed.actual.resize(F*T); fill_uniform(seed.actual,7239U,-.7F,.7F); round_to_bf16(seed.actual);
    seed.reference=seed.actual;
    auto dh=to_device_bf16(seed.actual);
    Tensor hidden(dh.p,DType::BF16,{F,T});
    artifact::Reader rows(source+"/qwen4-ple-rows.ninfer");
    if(rows.identity()!=artifact::ArtifactIdentity{"qwen4/native-ple-qualification","nvidia-fp8-source-rows"})
        throw std::runtime_error("native first-block PLE source rows");
    const auto* obj=rows.find("ple.rows");
    const auto payload=rows.payload(*obj).data;
    std::vector<std::uint8_t> packed(T*D);
    for(int i=0;i<T*16;++i) {
        const int row=(i*7+i/16)%16;
        for(int d=0;d<160;++d) packed[i*160+d]=std::to_integer<unsigned>(payload[row*160+d]);
    }
    const auto scale=std::to_integer<unsigned>(payload[16*160])|
                      (std::to_integer<unsigned>(payload[16*160+1])<<8);
    auto dpacked=to_device(packed); DeviceBuffer de(T*D*2);
    Tensor encoded(dpacked.p,DType::U8,{160,16,T}),decoded(de.p,DType::BF16,{160,16,T});
    ops::ple_fp8_decode_rows(encoded,scale,decoded,device.stream);
    Tensor embedding(de.p,DType::BF16,{D,T});
    std::array<std::array<int,3>,T> positions;
    for(int t=0;t<T;++t) positions[t]={t,t+1,t+2};
    runtime.prepare(0,positions,hidden,embedding,device.stream);
    const auto output=runtime.enqueue(0,T,true,device.stream);
    runtime.commit_all(0,device.stream); device.synchronize();
    std::array<std::vector<double>,4> whole;
    for(int layer=0;layer<4;++layer) whole[layer]=from_device_bf16(output.layer_hidden[layer].data,F*T);
    const auto whole_final=from_device_bf16(output.hidden.data,F*T);
    int failures=0;
    // Existing single mathematical oracles consume unchanged source artifacts. Their test-only
    // source-order adapters do not bind the prepared artifact and are not production execution.
    auto reference=seed;
    for(int layer=0;layer<4;++layer) {
        const auto path=source+"/qwen4-layer-"+std::to_string(layer)+".ninfer";
        if(layer==1) { reference=seq::ple(source,reference,false); failures+=reference.failures; }
        const auto a=seq::read(path,layer,reference);
        const auto mix=layer==3?seq::qsa(path,a.mixed,false):seq::gdn(path,layer,a.mixed,false);
        reference=seq::inject(reference,mix,a.scale);
        const auto m=seq::read(path,layer,reference,"mlp");
        const auto expert=seq::moe(path,layer,m.mixed);
        reference=seq::inject(reference,expert,m.scale);
        failures+=a.mixed.failures+a.scale.failures+mix.failures+m.mixed.failures+m.scale.failures+expert.failures+reference.failures;
        failures+=verify_reduction("native owned layer"+std::to_string(layer)+" independent accumulated oracle",
                                   whole[layer],seq::wide(reference.reference),accumulated);
    }
    failures+=shared_decoder(*model,hidden,embedding,seq::wide(reference.reference),device);
    runtime.reset(0,device.stream);
    runtime.prepare(0,std::span(positions).first(4),hidden.slice(1,0,4),embedding.slice(1,0,4),device.stream);
    (void)runtime.enqueue(0,4,true,device.stream); runtime.commit_all(0,device.stream);
    runtime.prepare(0,std::span(positions).last(1),hidden.slice(1,4,1),embedding.slice(1,4,1),device.stream);
    const auto last=runtime.enqueue(0,1,false,device.stream); runtime.commit_all(0,device.stream); device.synchronize();
    failures+=verify_reduction("native owned whole5 versus4+1 continuation",from_device_bf16(last.hidden.data,F),
        std::span(whole_final).subspan(4*F,F),accumulated);
    if(runtime.frontier(0)!=5) ++failures;

    // Four requests share immutable model/scratch but own all continuation buffers. Capture one
    // fixed-width C4 round, then replay it at a different frontier/visibility without recapture.
    std::array<std::array<std::vector<double>,4>,2> eager_output;
    std::array<std::array<std::vector<std::uint8_t>,4>,2> eager_state;
    auto prepare_round=[&](int round) {
        for(int slot=0;slot<4;++slot) {
            const int column=(slot+round)%T;
            const std::array<std::array<int,3>,1> pos{{{37+slot*11+round,13+slot+round,5+slot*3+round}}};
            runtime.prepare(slot,pos,hidden.slice(1,column,1),embedding.slice(1,column,1),device.stream);
        }
    };
    for(int slot=0;slot<4;++slot) runtime.reset(slot,device.stream);
    for(int round=0;round<2;++round) {
        prepare_round(round);
        for(int slot=0;slot<4;++slot) {
            const auto value=runtime.enqueue(slot,1,false,device.stream); runtime.commit_all(slot,device.stream);
            device.synchronize(); eager_output[round][slot]=from_device_bf16(value.hidden.data,F);
            eager_state[round][slot]=snapshot(runtime.state(slot,false));
        }
    }
    for(int slot=0;slot<4;++slot) runtime.reset(slot,device.stream);
    prepare_round(0); device.synchronize();
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    std::array<NativeFirstBlockOutput,4> graph_output;
    definition.capture(device.stream,[&] {
        for(int slot=0;slot<4;++slot) graph_output[slot]=runtime.enqueue(slot,1,false,device.stream);
    });
    graph.instantiate(definition); graph.upload(device.stream);
    for(int round=0;round<2;++round) {
        if(round) prepare_round(round);
        graph.launch(device.stream);
        for(int slot=0;slot<4;++slot) runtime.commit_all(slot,device.stream);
        device.synchronize();
        for(int slot=0;slot<4;++slot) {
            failures+=verify_exact("native C4 graph/eager represented output",
                from_device_bf16(graph_output[slot].hidden.data,F),eager_output[round][slot]);
            failures+=verify_exact("native C4 graph/eager complete owned state",snapshot(runtime.state(slot,false)),eager_state[round][slot]);
            if(runtime.frontier(slot)!=round+1) ++failures;
        }
    }
    NativeFirstBlock isolated(*model,1,32,T);
    for(int slot=0;slot<4;++slot) {
        isolated.reset(0,device.stream);
        for(int round=0;round<2;++round) {
            const int column=(slot+round)%T;
            const std::array<std::array<int,3>,1> pos{{{37+slot*11+round,13+slot+round,5+slot*3+round}}};
            isolated.prepare(0,pos,hidden.slice(1,column,1),embedding.slice(1,column,1),device.stream);
            const auto value=isolated.enqueue(0,1,false,device.stream); isolated.commit_all(0,device.stream);
            device.synchronize();
            failures+=verify_exact("native interleaved slot versus independent C1 output",
                from_device_bf16(value.hidden.data,F),eager_output[round][slot]);
            failures+=verify_exact("native interleaved slot versus independent C1 state",
                snapshot(isolated.state(0,false)),eager_state[round][slot]);
        }
    }
    std::cout<<"Native owned first-block numerical/continuation/C4-graph failures="<<failures<<'\n';
    return failures;
}
}
int main() {
    const char* source=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    const char* prepared=std::getenv("NINFER_QWEN4_NATIVE_COMPUTE");
    if(!source || !prepared) { std::cout<<"SKIP native first-block artifacts unset\n"; return 77; }
    if(require_cuda()!=0) return 1;
    try { return run(source,prepared)?1:0; }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

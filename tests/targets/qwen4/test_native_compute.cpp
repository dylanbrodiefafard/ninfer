#include "targets/qwen4/native_compute.h"
#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_decoder.h"
#include "targets/qwen4/native_ple_fetch.h"
#include "targets/qwen4/native_runtime_layout.h"
#include "targets/qwen4/native_text_panel.h"
#include "artifact/typed_binding.h"
#include "artifact/reader.h"
#include "core/decode_graph.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::targets::qwen4;
namespace seq=ninfer::test::qwen4_sequence;
namespace {
constexpr int D=2560,F=10240,T=5;
constexpr ReductionCriterion accumulated{.02,.005,.02};

int prefill_policies(const LoadedNativeFirstBlock& model,const std::string& source,
                    const std::string& prepared,DeviceContext& device) {
    artifact::Reader reader(prepared+"/qwen4-native-fp8-projections.ninfer");
    if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-prefill-qualification","senfu-fp8-tiled"})
        throw std::runtime_error("native prefill requires audited prepared FP8 overrides");
    artifact::Binder binder(reader);std::map<std::string,artifact::ObjectHandle> handles;
    for(const auto& obj:reader.objects()) {
        const auto& t=std::get<artifact::TensorDescriptor>(obj);
        handles.emplace(t.name,artifact::bind_device_tensor(binder,t.name,
            artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M,{t.shape.at(0),t.shape.at(1)}));
    }
    auto owner=artifact::materialize(reader,binder.finish(),device);
    auto layers=model.layers();
    auto matrix=[&](const std::string& name,int n,int k) {
        return artifact::materialized_weight(owner,handles.at(name),artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M,n,k);
    };
    const std::string prefix="model.language_model.layers.";
    auto& g=layers[0].gdn;
    g.qkv=matrix(prefix+"0.linear_attn.in_proj_qkv.weight",10240,D);
    g.z=matrix(prefix+"0.linear_attn.in_proj_z.weight",6144,D);
    g.output=matrix(prefix+"0.linear_attn.out_proj.weight",D,6144);
    auto& q=layers[3].qsa;
    q.core_query_gate=matrix(prefix+"3.self_attn.q_proj.weight",12288,D);
    q.core_key=matrix(prefix+"3.self_attn.k_proj.weight",512,D);
    q.core_value=matrix(prefix+"3.self_attn.v_proj.weight",512,D);
    q.output=matrix(prefix+"3.self_attn.o_proj.weight",D,6144);
    for(int l:{0,3}) {
        auto& m=layers[l].moe;const auto p=prefix+std::to_string(l)+".mlp.shared_expert.";
        m.shared_gate_proj=matrix(p+"gate_proj.weight",640,D);
        m.shared_up=matrix(p+"up_proj.weight",640,D);m.shared_down=matrix(p+"down_proj.weight",D,640);
    }
    seq::TextPanel panel(source);seq::Result reference;reference.actual=panel.residual;reference.reference=reference.actual;
    int failures=0;
    // Same complete independent component formulas as the native text qualification, not
    // a second native implementation used as the mathematical oracle.
    for(int l=0;l<4;++l) {
        const auto path=source+"/qwen4-layer-"+std::to_string(l)+".ninfer";
        if(l==1) {reference=seq::ple(source,reference,false,false,true);failures+=reference.failures;}
        const auto a=seq::read(path,l,reference);
        const auto mixer=l==0?seq::gdn_calibrated(source,path,l,a.mixed,false,0):
            l==3?seq::qsa_calibrated(source,path,a.mixed,false):seq::gdn(path,l,a.mixed,false);
        const auto attention=seq::inject(reference,mixer,a.scale);
        const auto m=seq::read(path,l,attention,"mlp");
        const auto expert=(l==0||l==3)?seq::moe_calibrated(source,path,l,m.mixed,false,0):seq::moe(path,l,m.mixed);
        reference=seq::inject(attention,expert,m.scale);
        failures+=a.mixed.failures+a.scale.failures+mixer.failures+attention.failures+
            m.mixed.failures+m.scale.failures+expert.failures+reference.failures;
    }
    NativeRuntimeConfig config;config.requests=4;config.context_tokens=128;config.kv_tokens=512;
    config.prefill_width=65;config.verify_width=3;config.vision=false;
    WorkspaceArena shared(native_layout::shared_bytes(config));auto s=native_layout::allocate(shared,config);
    NativeState state({4,128,512,3,ops::QsaKvFormat::BF16},device.stream);
    std::size_t bytes=ops::ple_workspace_capacity_bytes(65,model.ple().key.qtype,model.ple().value.qtype);
    for(int l=0;l<4;++l) for(int p=0;p<4;++p) {
        const auto policy=native_prefill_policy(static_cast<NativePrefillPolicy>(p),l);const auto& w=layers[l];
        bytes=std::max({bytes,ops::gated_residual_workspace_capacity_bytes(65,w.attention_gr.down.qtype,w.attention_gr.up.qtype),
            ops::gated_residual_workspace_capacity_bytes(65,w.moe_gr.down.qtype,w.moe_gr.up.qtype),
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(w.moe,65,policy.routed,policy.shared)});
        if(l<3) bytes=std::max({bytes,ops::gated_delta_net_layer_workspace_capacity_bytes(65,w.gdn.qkv.qtype,w.gdn.z.qtype,w.gdn.output.qtype,policy.gdn),
            ops::gated_delta_net_layer_batch_workspace_capacity_bytes(65,1,w.gdn.qkv.qtype,w.gdn.z.qtype,w.gdn.output.qtype)});
    }
    WorkspaceArena scratch(bytes);
    DeviceBuffer qbuffer(ops::qsa_verifier_workspace_bytes(65,1,q.core_query_gate.qtype,q.core_key.qtype,q.core_value.qtype,q.output.qtype));
    Tensor qs(qbuffer.p,DType::U8,{int(qbuffer.bytes)});
    auto v=[](const Tensor& t,std::initializer_list<int> shape){return Tensor(t.data,t.dtype,shape);};
    auto copy=[&](const Tensor& t,const auto& values){CUDA_CHECK(cudaMemcpy(t.data,values.data(),values.size()*sizeof(values[0]),cudaMemcpyHostToDevice));};
    // The table component is a represented real row panel, not a residency claim.
    artifact::Binder table_binder(panel.reader);
    const auto table_handle=artifact::bind_tensor(table_binder,"ple.rows",artifact::NumericFormat::FP8_E4M3FN_TENSOR_BF16S,
        {panel.row_count,160},artifact::TensorPlacement::ResidentHost);
    for(const auto& obj:panel.reader.objects()) {
        const auto& t=std::get<artifact::TensorDescriptor>(obj);
        if(t.name!="ple.rows") {const auto h=table_binder.require_tensor(t.name,t.format,t.layout,t.shape);table_binder.validate_only(h);}
    }
    auto table_owner=artifact::materialize(panel.reader,table_binder.finish(),device);
    const auto ple=table_owner.mapped_tensor_bytes(table_handle);
    const auto* codes=reinterpret_cast<const std::uint8_t*>(ple.data());
    const ops::PleResidentFp8Table table{codes,panel.row_count,panel.row_count*160};
    const std::uint16_t scale=codes[panel.row_count*160]|(codes[panel.row_count*160+1]<<8);
    PinnedHostBuffer pinned(65*D);DeviceBuffer packed(65*D);
    auto prepare=[&](int width,int batch,bool repeated=false) {
        device.synchronize();
        std::vector<int> slots(batch),valid(batch,width),frontier(batch),positions(3*width*batch);
        for(int b=0;b<batch;++b) {slots[b]=b;state.reset(b);state.reserve(b,128);state.materialize(b,width);}
        for(int i=0;i<width*batch;++i) for(int axis=0;axis<3;++axis) positions[3*i+axis]=i%width+axis;
        copy(s.slots,slots);copy(s.valid,valid);copy(s.frontiers,frontier);copy(s.positions,positions);
        std::vector<float> values;std::vector<int> rows;
        for(int t=0;t<width*batch;++t) {
            const int source_column=repeated?15:t;
            values.insert(values.end(),panel.residual.begin()+F*source_column,panel.residual.begin()+F*(source_column+1));
            rows.insert(rows.end(),panel.local_rows.begin()+16*source_column,panel.local_rows.begin()+16*(source_column+1));
        }
        auto input=to_device_bf16(values);
        CUDA_CHECK(cudaMemcpy(s.residual.data,input.p,F*width*batch*2,cudaMemcpyDeviceToDevice));
        Tensor encoded(packed.p,DType::U8,{160,16,width*batch}),decoded(s.ple_embedding.data,DType::BF16,{160,16,width*batch});
        ops::ple_fp8_stage_rows_batch(table,rows,width*batch,pinned.data(),pinned.size(),encoded,device.stream);
        ops::ple_fp8_decode_rows(encoded,scale,decoded,device.stream);
    };
    auto body=[&](int width,int batch,bool recorded,NativePrefillPolicy policy) {
        scratch.reset();const ops::QsaBatchControls controls{v(s.slots,{batch}),v(s.valid,{batch}),v(s.frontiers,{batch}),v(s.positions,{3,width,batch})};
        NativeDecoderViews views{v(s.residual,{D,4,width,batch}),v(s.ple_embedding,{D,width,batch}),
            v(s.mixed,{D,width,batch}),v(s.block,{D,width,batch}),v(s.scale,{4,width,batch}),v(s.routes,{10,width,batch}),
            v(s.probabilities,{10,width,batch}),v(s.selected,{2051,width,batch}),v(s.selected_count,{width,batch})};
        enqueue_native_decoder(layers,model.ple(),state,controls,128,views,recorded,scratch,qs,device.stream,
            width>16&&batch==1&&!recorded?0:-1,policy);
    };
    for(auto policy:{NativePrefillPolicy::A16,NativePrefillPolicy::SelectiveA8}) {
        prepare(33,1);body(33,1,false,policy);device.synchronize();
        failures+=verify_reduction("native prefill policy "+std::to_string(int(policy))+" independent accumulated oracle",
            from_device_bf16(s.residual.data,F*33),seq::wide(reference.reference),accumulated);
    }
    for(auto policy:{NativePrefillPolicy::RoutedA4,NativePrefillPolicy::RoutedA4SelectiveA8}) {
        prepare(65,1,true);body(65,1,false,policy);device.synchronize();
        // Candidate execution witness, not a relaxed whole-chain screen: independently
        // qualify the final actual decoder MoE from its represented public input.
        seq::Result input;const auto mixed=from_device_bf16(s.mixed.data,D*65);
        input.actual.assign(mixed.begin(),mixed.end());input.reference=input.actual;
        const auto actual=from_device_bf16(s.block.data,D*65);
        const auto expected=seq::moe_calibrated(source,source+"/qwen4-layer-3.ninfer",3,input,false,
            selective_a8(policy)?7:0,true);
        failures+=expected.failures;
        failures+=verify_reduction("native routed-A4 decoder final MoE complete oracle",actual,
            seq::wide(expected.reference),ReductionCriterion{.16,1./32768,.16});
        std::cout<<"Native routed-A4 prefill policy="<<int(policy)<<" executed; no whole-chain quality admission\n";
    }
    // All prefill candidate choices must leave compact decode and recorded verification
    // identical, including externally observable recurrent records, with real FP8 matrices.
    for(int width:{1,3}) {
        std::vector<std::uint8_t> baseline;
        for(int pass=0;pass<8;++pass) {
            const int p=pass%4;
            prepare(width,4);
            if(pass>=4) {
                DecodeGraphDefinition definition;DecodeGraphExecutable graph;
                definition.capture(device.stream,[&]{body(width,4,width>1,static_cast<NativePrefillPolicy>(p));});
                graph.instantiate(definition);graph.launch(device.stream);device.synchronize();
            } else {body(width,4,width>1,static_cast<NativePrefillPolicy>(p));device.synchronize();}
            auto output=from_device<std::uint8_t>(s.residual.data,F*width*4*2);
            for(int l=0;l<3;++l) for(const auto& t:{state.gdn(true).conv[l],state.gdn(true).recurrent[l]}) {
                auto data=from_device<std::uint8_t>(t.data,t.bytes());output.insert(output.end(),data.begin(),data.end());
            }
            if(width>1) {
                const auto records=state.records(width,4);
                for(int l=0;l<3;++l) {
                    const auto r=records.layer(l,4);
                    for(const auto& t:{r.conv,r.key,r.value,r.gate}) {
                        auto data=from_device<std::uint8_t>(t.data,t.bytes());output.insert(output.end(),data.begin(),data.end());
                    }
                }
            }
            if(pass==0) baseline=std::move(output);
            else failures+=verify_exact("native prefill policy preserves compact A16 output/state",output,baseline);
        }
    }
    return failures;
}
// Exercise the exact native runtime's layer-major body with the authentic four-layer prefix.
// No layer is repeated to stand in for the absent 44 layers. The complete NativeState allocation
// is used, but only the three executed GDN layers and one executed QSA layer are inspected.
int shared_decoder(const LoadedNativeFirstBlock& model,const Tensor& hidden,const NativePleTable& ple_table,
                   std::span<const double> reference,DeviceContext& device) {
    const bool overlap=std::getenv("NINFER_QWEN4_PLE_OVERLAP")!=nullptr;
    NativeRuntimeConfig config;config.requests=4;config.context_tokens=overlap?512:128;
    config.kv_tokens=4*config.context_tokens;
    config.prefill_width=overlap?257:65;config.verify_width=3;config.vision=false;
    NativeState state({4,config.context_tokens,config.kv_tokens,3,ops::QsaKvFormat::BF16},device.stream);
    NativeState* execution_state=&state;
    WorkspaceArena shared(native_layout::shared_bytes(config));auto s=native_layout::allocate(shared,config);
    const int n=native_layout::columns(config);
    PinnedHostBuffer pinned(2560*n);
    NativePleFetch fetch;
    std::vector<int> ple_ids(16*n);
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
            v(s.selected_count,{width,batch}),{},v(s.compact_rows,{batch}),fetch.ready()};
        enqueue_native_decoder(model.layers(),model.ple(),*execution_state,controls,config.context_tokens,views,false,workspace,qsa,device.stream,full_prefill_slot);
    };
    auto fetch_rows=[&](int width,int batch,int round) {
        fetch.synchronize();
        for(int b=0;b<batch;++b)for(int t=0;t<width;++t)for(int h=0;h<16;++h) {
            const int source=(batch>1?b+round:t)%T;
            ple_ids[(b*width+t)*16+h]=((source*16+h)*7+source)%16;
        }
        fetch.enqueue(ple_table,{ple_ids.data(),std::size_t(16)*width*batch},width*batch,
            pinned.data(),pinned.size(),v(s.packed_rows,{160,16,width*batch}),
            v(s.ple_embedding,{160,16,width*batch}));
    };
    std::array<int,4> slots{2,0,3,1},lengths{1,1,1,1},frontiers{};
    auto reset=[&] {for(int i=0;i<4;++i){execution_state->reset(i);execution_state->reserve(i,config.context_tokens);}device.synchronize();};
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
    fetch_rows(T,1,0);
    body(T,1);commit(T,1);device.synchronize();
    int failures=verify_reduction("shared native paged decoder actual0..3 independent accumulated oracle",
        from_device_bf16(s.residual.data,F*T),reference,accumulated);
    slots={2,0,3,1};lengths.fill(1);
    auto prepare=[&](int round) {
        device.synchronize();std::array<int,12> pos{};std::array<int,4> tokens{};
        fetch_rows(1,4,round);
        for(int b=0;b<4;++b) {
            const int col=(b+round)%T;frontiers[b]=state.frontier(slots[b]);state.materialize(slots[b],frontiers[b]+1);
            tokens[b]=100+col;pos[3*b]=37+b*11+round;pos[3*b+1]=13+b+round;pos[3*b+2]=5+b*3+round;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.residual.data)+b*F*2,hidden.slice(1,col,1).data,F*2,
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
        fetch_rows(65,1,0);
        std::array<int,65> tokens{};std::array<int,195> pos{};
        for(int t=0;t<65;++t) {
            tokens[t]=100+t%T;pos[3*t]=t;pos[3*t+1]=t+1;pos[3*t+2]=t+2;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.residual.data)+std::size_t(t)*F*2,
                hidden.slice(1,t%T,1).data,F*2,cudaMemcpyDeviceToDevice,device.stream));
        }
        upload(s.positions,pos);upload(s.ids,tokens);upload(s.slots,slots);upload(s.valid,lengths);upload(s.frontiers,frontiers);
        body(65,1,route?2:-1);commit(65,1);device.synchronize();prefill[route]=from_device_bf16(s.residual.data,F*65);
    }
    failures+=verify_reduction("shared native real-prefix chunked GDN route",prefill[1],prefill[0],accumulated);
    // Optional critical-path measurement: authentic resident layers0..3 plus the COMPLETE
    // native NVFP4 PLE mapping. This is not a model throughput benchmark. No streamed experts,
    // table cache fallback, partial table, or invented layers participate.
    if(const char* root=std::getenv("NINFER_QWEN4_PLE_OVERLAP")) {
        // Numerical identity above uses BF16 KV. Performance uses the engine's default
        // NVFP4-G16 storage; these distinct claims must not silently share a dtype.
        NativeState perf_state({4,config.context_tokens,config.kv_tokens,3,ops::QsaKvFormat::NVFP4G16},device.stream);
        execution_state=&perf_state;
        artifact::Reader reader(std::string(root)+"/qwen4-ple-nvfp4.ninfer");
        if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-ple-qualification","primitive-nvfp4-complete-table"})
            throw std::runtime_error("PLE overlap requires the exact complete native table");
        artifact::Binder binder(reader);
        const std::array<std::uint64_t,3> shape{128,2500012,160};
        const auto handle=binder.require_tensor("ple.table",artifact::NumericFormat::NVFP4_PARTITION_F32M,
            artifact::StorageLayout::PartitionedRowBlockScaleK16V1,shape);
        binder.map_tensor_on_host(handle,true);
        auto owner=artifact::materialize(reader,binder.finish(),device);
        const auto table=native_ple_table_view(owner.mapped_tensor_bytes(handle),NativePleFormat::Nvfp4);
        std::cout<<"PLE_OVERLAP resident_locked_bytes="<<owner.stats().resident_locked_bytes<<'\n';
        cudaEvent_t begin,end,ready;
        CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));CUDA_CHECK(cudaEventCreate(&ready));
        CUDA_CHECK(cudaProfilerStart());
        for(const auto [width,batch]:std::array<std::pair<int,int>,7>{{{1,1},{1,2},{1,3},{1,4},{3,4},{65,1},{257,1}}}) {
            std::vector<std::uint16_t> baseline;
            for(int replay=0;replay<2;++replay) for(int async=0;async<2;++async) {
                DecodeGraphDefinition measured_definition;DecodeGraphExecutable measured_graph;
                if(replay) {
                    measured_definition.capture(device.stream,[&]{body(width,batch,width>16?2:-1);});
                    measured_graph.instantiate(measured_definition);measured_graph.upload(device.stream);
                    device.synchronize();
                }
                std::vector<double> total,gather,available;
                for(int repetition=-2;repetition<9;++repetition) {
                    const auto label=std::string("ple-overlap.")+(async?"async":"serialized")+
                        ".w"+std::to_string(width)+".b"+std::to_string(batch)+
                        (replay?".graph":".eager");
                    nvtxRangePushA(label.c_str());
                    reset();slots={2,0,3,1};lengths.fill(width);frontiers.fill(0);
                    const int columns=width*batch;
                    std::vector<int> tokens(columns),pos(3*columns);
                    for(int b=0;b<batch;++b) {
                        perf_state.materialize(slots[b],width);
                        for(int t=0;t<width;++t) {
                            const int col=b*width+t;tokens[col]=100+col%T;
                            pos[3*col]=t;pos[3*col+1]=t+1;pos[3*col+2]=t+2;
                            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(s.residual.data)+std::size_t(col)*F*2,
                                hidden.slice(1,col%T,1).data,F*2,cudaMemcpyDeviceToDevice,device.stream));
                        }
                    }
                    upload(s.positions,pos);upload(s.ids,tokens);upload(s.slots,slots);upload(s.valid,lengths);upload(s.frontiers,frontiers);
                    device.synchronize();fetch.synchronize();
                    // Identical, changing random-access rows for both routes. Integer-only
                    // addressing is outside the measured gather interval in this component AB.
                    for(int i=0;i<16*columns;++i)
                        ple_ids[i]=std::uint64_t((i+1)*104729ULL+(repetition+3)*7919ULL)*15485863ULL%(128ULL*2500012);
                    auto packed_rows=v(s.packed_rows,{94,16,columns});
                    auto decoded_rows=v(s.ple_embedding,{160,16,columns});
                    CUDA_CHECK(cudaEventRecord(begin,device.stream));
                    const auto start=std::chrono::steady_clock::now();
                    if(async) {
                        CUDA_CHECK(cudaStreamWaitEvent(fetch.stream(),begin,0));
                        fetch.enqueue(table,{ple_ids.data(),std::size_t(16)*columns},columns,
                            pinned.data(),pinned.size(),packed_rows,decoded_rows);
                        CUDA_CHECK(cudaEventRecord(ready,fetch.stream()));
                    } else {
                        ops::ple_nvfp4_stage_rows_batch(table.nvfp4,{ple_ids.data(),std::size_t(16)*columns},columns,
                            pinned.data(),pinned.size(),packed_rows,device.stream);
                        ops::ple_nvfp4_decode_rows(packed_rows,decoded_rows,device.stream);
                        CUDA_CHECK(cudaEventRecord(ready,device.stream));
                    }
                    const auto submitted=std::chrono::steady_clock::now();
                    if(replay) measured_graph.launch(device.stream);
                    else body(width,batch,width>16?slots[0]:-1);
                    CUDA_CHECK(cudaEventRecord(end,device.stream));CUDA_CHECK(cudaEventSynchronize(end));
                    fetch.synchronize();
                    if(repetition>=0) {
                        float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));total.push_back(ms*1000);
                        CUDA_CHECK(cudaEventElapsedTime(&ms,begin,ready));available.push_back(ms*1000);
                        gather.push_back(std::chrono::duration<double,std::micro>(submitted-start).count());
                    }
                    if(repetition==8) {
                        auto actual=from_device<std::uint16_t>(s.residual.data,std::size_t(F)*columns);
                        if(!async && !replay)baseline=std::move(actual);
                        else failures+=verify_exact("complete native PLE serialized/overlapped prefix4",actual,baseline);
                    }
                    nvtxRangePop();
                }
                const auto median=[](std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
                std::cout<<"PLE_OVERLAP W="<<width<<" B="<<batch<<" async="<<async
                    <<" graph="<<replay
                    <<" prefix4_us="<<median(total)<<" gather_submit_us="<<median(gather)
                    <<" rows_ready_us="<<median(available)<<'\n';
            }
        }
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaEventDestroy(ready));CUDA_CHECK(cudaEventDestroy(end));CUDA_CHECK(cudaEventDestroy(begin));
        execution_state=&state;
    }
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
    const NativePleTable ple_table{NativePleFormat::Fp8,{},
        {reinterpret_cast<const std::uint8_t*>(payload.data()),16,16*160},std::uint16_t(scale)};
    failures+=shared_decoder(*model,hidden,ple_table,seq::wide(reference.reference),device);
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
int main(int argc,char** argv) {
    const char* source=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    const char* prepared=std::getenv("NINFER_QWEN4_NATIVE_COMPUTE");
    if(!source || !prepared) { std::cout<<"SKIP native first-block artifacts unset\n"; return 77; }
    if(require_cuda()!=0) return 1;
    try {
        if(argc==2 && std::string_view(argv[1])=="--prefill-policies") {
            DeviceContext device;auto model=LoadedNativeFirstBlock::load(prepared,std::string(source)+"/qwen4-ple-component.ninfer",device);
            return prefill_policies(*model,source,prepared,device)?1:0;
        }
        if(argc!=1) throw std::invalid_argument("unknown native compute test mode");
        return run(source,prepared)?1:0;
    }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

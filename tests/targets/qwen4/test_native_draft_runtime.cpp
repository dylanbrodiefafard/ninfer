// Real native draft components and authentic shared endpoints. The carried target inputs
// come from the separately identified full48 diagnostic target, not a native-target PPL run.
#include "targets/qwen4/native_draft_runtime.h"
#include "targets/qwen4/native_runtime.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_text_panel.h"
#include "artifact/typed_binding.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace q4=ninfer::targets::qwen4;
namespace {
constexpr int D=2560,F=10240,V=248320;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
std::vector<std::uint16_t> read_words(const std::filesystem::path& path,std::size_t words,std::size_t offset=0) {
    std::ifstream input(path,std::ios::binary);input.seekg(offset);
    std::vector<std::uint16_t> result(words);input.read(reinterpret_cast<char*>(result.data()),words*2);
    require(bool(input),"missing/truncated actual target capture");return result;
}
std::vector<std::uint16_t> download(const Tensor& tensor,DeviceContext& device) {
    device.synchronize();return from_device<std::uint16_t>(tensor.data,tensor.numel());
}
int compare(const char* name,const std::vector<std::uint16_t>& got,const std::vector<std::uint16_t>& expected) {
    for(auto x:got) require(std::isfinite(bf16_to_f32(x)),"nonfinite native draft logits");
    return verify_exact(name,got,expected);
}
int compare_replay(const std::vector<std::uint16_t>& got,const std::vector<std::uint16_t>& expected) {
    std::vector<double> actual(got.size()),reference(expected.size());
    double error=0,norm=0;
    for(std::size_t i=0;i<got.size();++i) {
        actual[i]=bf16_to_f32(got[i]);reference[i]=bf16_to_f32(expected[i]);
        error+=std::pow(actual[i]-reference[i],2);norm+=reference[i]*reference[i];
    }
    std::cout<<"MTP whole-prefix versus one-row reseed supplementary rel_l2="<<std::sqrt(error/norm)<<'\n';
    // Whole-prefix and one-row execution select different qualified A16 projection routes.
    // Their BF16 boundaries need not be bit-identical. This uses the existing component
    // composition profile; local independent FP64 Op oracles remain the math authority.
    return verify_reduction("native MTP reseed versus fresh actual target-prefix replay",actual,reference,{.02,.005,.02});
}
q4::NativeRuntimeConfig config(bool mtp,bool graph,int requests=4) {
    q4::NativeRuntimeConfig c;c.requests=requests;c.context_tokens=128;c.kv_tokens=256;
    c.prefill_width=4;c.verify_width=4;c.vision=false;c.mtp=mtp;c.dflash=!mtp;c.use_cuda_graph=graph;return c;
}
std::vector<std::uint8_t> mtp_snapshot(q4::NativeDraftRuntime& runtime,int slot,DeviceContext& device) {
    device.synchronize();const auto state=runtime.mtp_state();const auto seed=runtime.mtp_seed();
    const auto pages=from_device<int>(state.block_tables.data,state.block_tables.numel());
    std::vector<std::uint8_t> result;
    for(const auto& plane:{state.k,state.v,state.raw_index_keys,state.positions}) {
        const auto bytes=from_device<std::uint8_t>(plane.data,plane.bytes());
        for(int t=0;t<runtime.frontier(slot);++t) {
            const int page=pages[slot*state.block_tables.ne[0]+t/64];require(page>=0,"MTP live logical page missing");
            for(int h=0;h<plane.ne[2];++h) {
                const auto offset=(t%64)*plane.nb[1]+h*plane.nb[2]+page*plane.nb[3];
                result.insert(result.end(),bytes.begin()+offset,bytes.begin()+offset+plane.ne[0]*plane.nb[0]);
            }
        }
    }
    for(const auto& plane:{seed.carry.slice(1,slot,1),seed.target.slice(1,slot,1),
        seed.selected_ids.slice(1,slot,1),seed.selected_count.slice(0,slot,1)}) {
        const auto bytes=from_device<std::uint8_t>(plane.data,plane.bytes());result.insert(result.end(),bytes.begin(),bytes.end());
    }
    return result;
}
std::vector<std::uint8_t> dflash_snapshot(q4::NativeDraftRuntime& runtime,int slot,DeviceContext& device) {
    device.synchronize();std::vector<std::uint8_t> result;
    for(int layer=0;layer<5;++layer) {
        const auto cache=runtime.dflash_cache(layer);
        const auto pages=from_device<int>(cache.block_tables.data,cache.block_tables.numel());
        for(const auto& plane:{cache.k_pages,cache.v_pages}) {
            const auto bytes=from_device<std::uint8_t>(plane.data,plane.bytes());
            for(int t=0;t<runtime.frontier(slot);++t) {
                const int page=pages[slot*cache.block_tables.ne[0]+t/64];require(page>=0,"DFlash live logical page missing");
                for(int h=0;h<plane.ne[3];++h) {
                    const auto offset=(t%64)*plane.nb[1]+page*plane.nb[2]+h*plane.nb[3];
                    result.insert(result.end(),bytes.begin()+offset,bytes.begin()+offset+plane.ne[0]*plane.nb[0]);
                }
            }
        }
    }
    return result;
}
struct Endpoints {
    qwen4_sequence::TextPanel panel;
    direct_bf16_weight::DeviceWeight head;
    DeviceBuffer embedding;
    std::unique_ptr<artifact::MaterializedArtifact> candidate_owner;
    Weight candidate_head,candidate_embedding;
    Endpoints(const std::filesystem::path& root,const std::filesystem::path& draft,DeviceContext& device)
        :panel(root.string()),head({V,D,qwen4_native::Bf16Source((root/"qwen4-endpoint.ninfer").string())
             .bits("lm_head.weight",{V,D})}),embedding(std::size_t(V)*D*2) {
        // Unaddressed rows are NaNs: accidentally sampling/feeding an unavailable fixture
        // token fails visibly. This allocation is not a fabricated full model artifact.
        CUDA_CHECK(cudaMemsetAsync(embedding.p,0xff,embedding.bytes,device.stream));
        const auto rows=panel.payload("token.embeddings",artifact::NumericFormat::BF16,{33,D});
        for(int t=0;t<33;++t) CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+std::size_t(panel.tokens[t])*D*2,
            rows.data()+std::size_t(t)*D*2,D*2,cudaMemcpyHostToDevice,device.stream));
        artifact::Reader inputs(draft/"qwen4-dflash-inputs.ninfer");
        const auto mask=inputs.payload(*inputs.find("mask.embedding")).data;
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+std::size_t(248077)*D*2,
            mask.data(),D*2,cudaMemcpyHostToDevice,device.stream));
        device.synchronize();
    }
    void use_fp8(const std::filesystem::path& head_path,const std::filesystem::path& rows_path,DeviceContext& device) {
        artifact::Reader reader(head_path);artifact::Binder binder(reader);
        require(reader.identity()==artifact::ArtifactIdentity{"qwen4/native-weight-candidates","source-bf16-row-fp8-a16"},
            "FP8 head candidate identity");
        artifact::ObjectHandle handle{};
        for(const auto& object:reader.objects()) {
            const auto& t=std::get<artifact::TensorDescriptor>(object);
            const auto h=binder.require_tensor(t.name,t.format,t.layout,t.shape);
            if(t.name=="head.weight") {binder.materialize_on_device(h);handle=h;}
            else binder.validate_only(h);
        }
        candidate_owner=std::make_unique<artifact::MaterializedArtifact>(artifact::materialize(reader,binder.finish(),device));
        candidate_head=artifact::materialized_weight(*candidate_owner,handle,artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,V,D);
        // Same exact shape as head, but distinct backing and scale plane. Unavailable
        // source rows stay NaN so this cannot masquerade as a complete embedding table.
        candidate_embedding=candidate_head;candidate_embedding.payload=candidate_embedding.qdata=embedding.p;
        const auto shape=std::array<std::uint64_t,2>{V,D};
        const auto full=artifact::row_scale_geometry(artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,shape);
        candidate_embedding.scales=static_cast<std::byte*>(embedding.p)+full.scale_plane_offset;
        CUDA_CHECK(cudaMemsetAsync(embedding.p,0x7f,full.encoded_bytes,device.stream));
        artifact::Reader rows(rows_path);
        require(rows.identity()==artifact::ArtifactIdentity{"qwen4/native-embedding-row-candidate","original-bf16-row-fp8"},
            "FP8 addressed embedding candidate identity");
        const auto raw_ids=rows.payload("token.ids").data,raw=rows.payload("embedding.rows").data;
        require(raw_ids.size()==34*4,"FP8 embedding row IDs");
        std::array<int,34> ids{};std::memcpy(ids.data(),raw_ids.data(),raw_ids.size());
        const auto subset=artifact::row_scale_geometry(artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            std::array<std::uint64_t,2>{34,D});
        require(raw.size()==subset.encoded_bytes,"FP8 embedding candidate extent");
        for(int i=0;i<34;++i) {
            require(ids[i]>=0 && ids[i]<V,"FP8 candidate token range");
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+std::size_t(ids[i])*D,
                raw.data()+std::size_t(i)*D,D,cudaMemcpyHostToDevice,device.stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+full.scale_plane_offset+std::size_t(ids[i])*2,
                raw.data()+subset.scale_plane_offset+i*2,2,cudaMemcpyHostToDevice,device.stream));
        }
        device.synchronize();
    }
    void bind(q4::NativeModelView& view) const {
        if(candidate_owner) {view.output_head=candidate_head;view.token_embedding=candidate_embedding;return;}
        view.output_head=head.view();view.token_embedding=head.view();
        view.token_embedding.payload=view.token_embedding.qdata=embedding.p;
    }
};
std::vector<std::array<int,3>> positions(int start,int count) {
    std::vector<std::array<int,3>> result(count);
    for(int t=0;t<count;++t) result[t]={start+t,start+t,start+t};return result;
}
int mtp_case(Endpoints& endpoint,const std::filesystem::path& root,const std::filesystem::path& captured,DeviceContext& device) {
    auto loaded=q4::LoadedMtp::load(root/"qwen4-mtp-nvfp4.ninfer",device);
    q4::NativeModelView model;endpoint.bind(model);model.mtp=loaded->weights();
    nlohmann::json metadata;std::ifstream input(captured/"qwen4-mtp-target-inputs.json");input>>metadata;
    const auto captured_ids=metadata.at("prompt_token_ids").get<std::vector<int>>();
    require(captured_ids.size()==24 && std::equal(captured_ids.begin(),captured_ids.end(),endpoint.panel.tokens.begin()),
        "actual target/native embedding token alignment");
    const auto& p=metadata.at("planes").at("prompt_hidden");
    require(p.at("shape")==nlohmann::json::array({24,F}) && p.at("dtype")=="BF16","actual target carry geometry");
    auto hidden=to_device(read_words(captured/"qwen4-mtp-target-inputs.bin",24*F,p.at("offset")));
    q4::NativeDraftRuntime graph(model,config(true,true),device),eager(model,config(true,false),device);
    const std::array<int,4> order{3,1,0,2};
    std::array<std::vector<int>,4> tokens;std::array<std::vector<std::array<int,3>>,4> pos;
    std::vector<q4::NativeMtpInputRow> rows;
    for(int b=0;b<4;++b) {
        const int n=b+1,slot=order[b];tokens[b].assign(endpoint.panel.tokens.begin()+1,endpoint.panel.tokens.begin()+1+n);
        pos[b]=positions(0,n);rows.push_back({slot,Tensor(hidden.p,DType::BF16,{F,n}),tokens[b],pos[b]});
        graph.reserve(slot,32);eager.reserve(slot,32);
    }
    auto actual=download(graph.extend_mtp(rows,4),device),expected=download(eager.extend_mtp(rows,4),device);
    int failures=0;
    std::array<std::vector<std::uint8_t>,4> snapshots;
    for(int b=0;b<4;++b) {
        const int n=b+1;
        failures+=compare("native MTP compact C4 graph/eager live logits",
            {actual.begin()+b*4*V,actual.begin()+(b*4+n)*V},
            {expected.begin()+b*4*V,expected.begin()+(b*4+n)*V});
        require(graph.frontier(order[b])==n,"MTP ragged frontier");
        graph.retain(order[b]);eager.retain(order[b]);
        snapshots[b]=mtp_snapshot(graph,order[b],device);
        failures+=verify_exact("native MTP graph/eager complete logical state",snapshots[b],mtp_snapshot(eager,order[b],device));
    }
    std::array<int,4> proposals{};std::array<std::array<int,3>,4> draft_pos{};
    for(int b=0;b<4;++b) {proposals[b]=endpoint.panel.tokens[b+2];draft_pos[b]={b+1,b+1,b+1};}
    auto baseline=download(graph.draft_mtp(order,proposals,draft_pos),device);
    failures+=compare("native MTP frozen-domain C4 graph/eager",baseline,download(eager.draft_mtp(order,proposals,draft_pos),device));
    for(int slot:order) {graph.discard_mtp(slot);eager.discard_mtp(slot);}
    failures+=compare("native MTP discard and replay",download(graph.draft_mtp(order,proposals,draft_pos),device),baseline);
    for(int slot:order) graph.discard_mtp(slot);
    // Replacing a private seed overwrites its last live cache row. Restoring a retained
    // checkpoint must restore that row, not only the frontier and carried hidden state.
    std::array<int,4> replacement{};std::array<std::array<int,3>,4> seed_pos{};
    for(int b=0;b<4;++b) {replacement[b]=endpoint.panel.tokens[12+b];seed_pos[b]={b,b,b};}
    for(int slot:order) graph.retain(slot,q4::NativeCheckpoint::Prompt);
    (void)graph.reseed_mtp(order,replacement,seed_pos);device.synchronize();
    std::array<std::vector<std::uint8_t>,4> replacement_snapshots;
    for(int b=0;b<4;++b) {
        const int slot=order[b];replacement_snapshots[b]=mtp_snapshot(graph,slot,device);
        graph.retain(slot);graph.restore(slot,q4::NativeCheckpoint::Prompt);
        failures+=verify_exact("native MTP restore exact live pages and all seed planes",mtp_snapshot(graph,order[b],device),snapshots[b]);
        graph.restore(slot);
        failures+=verify_exact("native MTP distinct same-frontier retained anchor image",mtp_snapshot(graph,slot,device),replacement_snapshots[b]);
        graph.restore(slot,q4::NativeCheckpoint::Prompt);graph.retain(slot);
    }
    failures+=compare("native MTP retained seed tail restore",download(graph.draft_mtp(order,proposals,draft_pos),device),baseline);
    for(int slot:order) graph.discard_mtp(slot);
    // Independent single-request fresh replay checks the caller's actual R_t/E[x_(t+1)]
    // pairing, compact row placement, and reseed from target carry rather than private carry.
    for(int b=0;b<4;++b) {
        q4::NativeDraftRuntime single(model,config(true,false,1),device);single.reserve(0,32);
        auto shifted=tokens[b];shifted.back()=replacement[b];
        const q4::NativeMtpInputRow row{0,rows[b].target_hidden,shifted,pos[b]};
        (void)single.extend_mtp(std::span(&row,1),4);
        const int slot=order[b],zero=0;
        auto reseeded=download(graph.reseed_mtp(std::span(&slot,1),std::span(&replacement[b],1),std::span(&seed_pos[b],1)),device);
        (void)reseeded;
        auto fresh=download(single.draft_mtp(std::span(&zero,1),std::span(&proposals[b],1),std::span(&draft_pos[b],1)),device);
        auto restored=download(graph.draft_mtp(std::span(&slot,1),std::span(&proposals[b],1),std::span(&draft_pos[b],1)),device);
        failures+=compare_replay(restored,fresh);
        graph.discard_mtp(slot);graph.restore(slot);graph.release_reservation(slot);graph.reserve(slot,32);
    }
    failures+=compare("native MTP retained unbind/rebind C4",download(graph.draft_mtp(order,proposals,draft_pos),device),baseline);
    for(int slot:order) {graph.discard_mtp(slot);graph.reset(slot);}
    require(!graph.has_retained(0) && graph.frontier(0)==0,"MTP reset invalidates retained ownership");
    return failures;
}
int dflash_case(Endpoints& endpoint,const std::filesystem::path& root,const std::filesystem::path& captured,DeviceContext& device) {
    auto loaded=q4::LoadedDFlash::load(root/"qwen4-dflash-nvfp4.ninfer",device);
    q4::NativeModelView model;endpoint.bind(model);model.dflash=loaded->weights();
    nlohmann::json metadata;std::ifstream input(captured/"qwen4-dflash-target-features.json");input>>metadata;
    require(metadata.at("token_ids").get<std::vector<int>>()==endpoint.panel.tokens &&
        metadata.at("accepted_prompt_tokens")==24,"actual DFlash feature alignment");
    auto features=to_device(read_words(captured/"qwen4-dflash-target-features.bin",24*12800));
    q4::NativeDraftRuntime graph(model,config(false,true),device),eager(model,config(false,false),device);
    const std::array<int,4> order{3,1,0,2};std::array<int,4> anchors{},at{};
    for(int b=0;b<4;++b) {
        const int slot=order[b],n=b+1;anchors[b]=endpoint.panel.tokens[n];at[b]=n;
        std::vector<int> pos(n);std::iota(pos.begin(),pos.end(),0);
        for(auto* runtime:{&graph,&eager}) {
            runtime->reserve(slot,32);runtime->append_dflash(slot,Tensor(features.p,DType::BF16,{12800,n}),pos);
            runtime->retain(slot,q4::NativeCheckpoint::Prompt);runtime->retain(slot);
        }
    }
    int failures=0;
    auto baseline=download(graph.draft_dflash(order,anchors,at,3),device);
    failures+=compare("native DFlash C4 graph/eager exact shared endpoint",baseline,download(eager.draft_dflash(order,anchors,at,3),device));
    std::array<std::vector<std::uint8_t>,4> snapshots;
    for(int b=0;b<4;++b) {
        snapshots[b]=dflash_snapshot(graph,order[b],device);
        failures+=verify_exact("native DFlash graph/eager all live cache layers",snapshots[b],dflash_snapshot(eager,order[b],device));
    }
    failures+=compare("native DFlash noise is not accepted context",download(graph.draft_dflash(order,anchors,at,3),device),baseline);
    for(int b=0;b<4;++b) {
        const int slot=order[b],n=b+1;const std::array<int,1> pos{n};
        graph.append_dflash(slot,Tensor(static_cast<std::uint16_t*>(features.p)+n*12800,DType::BF16,{12800,1}),pos);
        graph.retain(slot);graph.restore(slot,q4::NativeCheckpoint::Prompt);
        require(graph.frontier(slot)==n && !graph.has_retained(slot),"DFlash shorter Prompt restore invalidates later Retained");
        failures+=verify_exact("native DFlash Prompt restore exact live cache",dflash_snapshot(graph,slot,device),snapshots[b]);
        graph.retain(slot);graph.release_reservation(slot);graph.reserve(slot,32);
        const int zero=0;q4::NativeDraftRuntime single(model,config(false,false,1),device);single.reserve(0,32);
        std::vector<int> prefix(n);std::iota(prefix.begin(),prefix.end(),0);
        single.append_dflash(0,Tensor(features.p,DType::BF16,{12800,n}),prefix);
        failures+=compare("native DFlash C4 versus independent C1 request",
            download(graph.draft_dflash(std::span(&slot,1),std::span(&anchors[b],1),std::span(&at[b],1),3),device),
            download(single.draft_dflash(std::span(&zero,1),std::span(&anchors[b],1),std::span(&at[b],1),3),device));
    }
    failures+=compare("native DFlash restored complete C4 context",download(graph.draft_dflash(order,anchors,at,3),device),baseline);
    for(int slot:order) graph.reset(slot);
    // Four physical P64 pages, not four private context-sized caches. Trial admission is
    // nonmutating, and reset releases the exact exclusive growth entitlement.
    graph.reserve(0,128);graph.reserve(1,64);graph.reserve(2,64);
    require(!graph.can_reserve(3,64),"DFlash shared page budget over-admitted");
    graph.reset(0);require(graph.can_reserve(3,128),"DFlash reset did not release shared pages");
    return failures;
}
}
int main(int argc,char** argv) {
    const bool fp8=argc==2 && std::string_view(argv[1])=="--fp8-endpoints";
    if(argc!=1 && !fp8) return 1;
    const std::filesystem::path root="/models/qwen4-native-layers",mtp="/models/qwen4-mtp",draft="/models/qwen4-dflash";
    const std::filesystem::path captured="/src/out/qwen4-dflash-target";
    for(const auto& path:{root/"qwen4-endpoint.ninfer",root/"qwen4-text-panel.ninfer",mtp/"qwen4-mtp-nvfp4.ninfer",
        draft/"qwen4-dflash-nvfp4.ninfer",draft/"qwen4-dflash-inputs.ninfer",captured/"qwen4-mtp-target-inputs.bin",
        captured/"qwen4-dflash-target-features.bin"}) if(!std::filesystem::exists(path)) {
        std::cout<<"SKIP missing exact native draft qualification prerequisite: "<<path<<'\n';return 77;
    }
    try {
        DeviceContext device;Endpoints endpoint(root,draft,device);
        if(fp8) endpoint.use_fp8("/src/out/qwen4-native-weight-assessment.ninfer",
            "/src/out/qwen4-native-embedding-rows-fp8.ninfer",device);
        const int failures=mtp_case(endpoint,mtp,captured,device)+dflash_case(endpoint,draft,captured,device);
        std::cout<<(failures?"FAIL":"PASS")<<" native draft runtime actual components: "<<failures<<'\n';return failures?1:0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

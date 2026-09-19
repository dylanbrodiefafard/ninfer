#include "targets/qwen4/dflash.h"
#include "dflash_test_cache.h"
#include "targets/qwen4/dflash_features.h"
#include "ops/op_tester.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "ninfer/ops/argmax.h"

#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <thread>

using namespace ninfer;
using namespace ninfer::test;
namespace q4 = ninfer::targets::qwen4;
namespace {
class Golden {
public:
    explicit Golden(const std::filesystem::path& path): file_(path,std::ios::binary) {
        std::ifstream metadata(path.parent_path()/(path.stem().string()+".json"));
        if(!file_ || !metadata) throw std::runtime_error("missing native DFlash oracle planes");
        metadata >> json;
        for(const auto& plane:json.at("planes")) planes_.emplace(plane.at("name"),plane);
    }
    template<class T> std::vector<T> read(const std::string& name) {
        const auto& p=planes_.at(name);
        const std::size_t bytes=p.at("bytes");
        const std::string dtype=p.at("dtype");
        if(bytes%sizeof(T) || (sizeof(T)==2 && dtype!="BF16") ||
           (std::is_same_v<T,double> && dtype!="F64") ||
           (std::is_same_v<T,std::int64_t> && dtype!="I64"))
            throw std::runtime_error("wrong DFlash oracle plane dtype");
        std::vector<T> values(bytes/sizeof(T));
        file_.seekg(std::streamoff(p.at("offset").get<std::uint64_t>()));
        file_.read(reinterpret_cast<char*>(values.data()),std::streamsize(bytes));
        if(!file_) throw std::runtime_error("truncated DFlash oracle plane");
        return values;
    }
    nlohmann::json json;
private:
    std::ifstream file_;
    std::map<std::string,nlohmann::json> planes_;
};
struct Trace {
    struct Plane { q4::DFlashBoundary stage; int layer; std::array<int,4> shape; DeviceBuffer storage; };
    std::vector<Plane> planes;
    q4::DFlashTrace sink() {
        return {this,[](void* self,q4::DFlashBoundary stage,int layer,const Tensor& value,cudaStream_t stream) {
            auto& trace=*static_cast<Trace*>(self);
            trace.planes.push_back({stage,layer,{value.ne[0],value.ne[1],value.ne[2],value.ne[3]},DeviceBuffer(value.bytes())});
            CUDA_CHECK(cudaMemcpyAsync(trace.planes.back().storage.p,value.data,value.bytes(),
                                       cudaMemcpyDeviceToDevice,stream));
        }};
    }
    void write(const std::filesystem::path& path) {
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        nlohmann::json metadata=nlohmann::json::array();
        for(const auto& p:planes) {
            std::vector<std::uint16_t> bits(p.storage.bytes/2);
            p.storage.copy_to_host(bits.data(),p.storage.bytes);
            metadata.push_back({{"stage",int(p.stage)},{"layer",p.layer},{"shape",p.shape},
                {"offset",std::uint64_t(file.tellp())},{"bytes",p.storage.bytes}});
            file.write(reinterpret_cast<const char*>(bits.data()),p.storage.bytes);
        }
        if(!file) throw std::runtime_error("cannot write DFlash diagnostic trace");
        std::ofstream index(path.parent_path()/(path.stem().string()+".json")); index<<metadata.dump(2)<<'\n';
    }
};
std::vector<std::uint16_t> snapshot(q4::DFlashProgram& program) {
    std::vector<std::uint16_t> result;
    for(int layer=0;layer<5;++layer) {
        auto c=program.cache(layer);
        for(const auto& plane:{c.k_pages,c.v_pages}) {
            auto words=from_device<std::uint16_t>(plane.data,plane.numel());
            result.insert(result.end(),words.begin(),words.end());
        }
    }
    return result;
}
void clear(q4::DFlashProgram& program,cudaStream_t stream) {
    for(int layer=0;layer<5;++layer) {
        auto c=program.cache(layer);
        CUDA_CHECK(cudaMemsetAsync(c.k_pages.data,0x3f,c.k_pages.bytes(),stream));
        CUDA_CHECK(cudaMemsetAsync(c.v_pages.data,0xbf,c.v_pages.bytes(),stream));
    }
}
int capture() {
    DeviceContext device;
    constexpr int width=3,batch=2,slots=4;
    std::vector<std::uint16_t> expected(12800*width*slots,0x3f80);
    auto packed=to_device(expected),lanes=to_device_i32({3,1}),valid=to_device_i32({2,3});
    q4::DFlashFeatureSink sink{Tensor(packed.p,DType::BF16,{12800,width,slots}),
        Tensor(lanes.p,DType::I32,{batch}),Tensor(valid.p,DType::I32,{batch})};
    const std::array<int,5> taps{4,16,24,36,44};
    for(int layer=0;layer<48;++layer) {
        std::vector<std::uint16_t> input(2560*width*batch);
        for(std::size_t i=0;i<input.size();++i) input[i]=std::uint16_t(i*13+layer*137);
        auto source=to_device(input);
        sink.capture(layer,Tensor(source.p,DType::BF16,{2560,width,batch}),device.stream);
        device.synchronize();
        for(int tap=0;tap<5;++tap) if(layer==taps[tap])
            for(int b=0;b<batch;++b) for(int t=0;t<(b==0?2:3);++t) for(int d=0;d<2560;++d)
                expected[(b==0?3:1)*width*12800+t*12800+tap*2560+d]=input[(b*width+t)*2560+d];
    }
    return verify_exact("exact learned-GR DFlash feature taps and request placement",
                         from_device<std::uint16_t>(packed,expected.size()),expected);
}
int batch_case(q4::LoadedDFlash& model,const std::vector<std::uint16_t>& original_features,
               const std::vector<std::uint16_t>& noise,const std::vector<int>& positions,
               const std::vector<int>& query_positions,DeviceContext& device) {
    constexpr int count=65,width=7,batch=4;
    ninfer::test::qwen4::DFlashTestCache cache(128,batch,device.stream);
    q4::DFlashProgram program(model.weights(),128,count,batch,cache.views());
    clear(program,device.stream);
    std::vector<std::uint16_t> features(12800*count*batch),embeddings(2560*width*batch);
    std::vector<int> pos(count*batch),cache_pos(count*batch),queries(width*batch);
    for(int b=0;b<batch;++b) {
        for(std::size_t i=0;i<original_features.size();++i)
            features[b*original_features.size()+i]=original_features[i] ^ ((i%4)==std::size_t(b)?0x8000:0);
        std::copy(noise.begin(),noise.end(),embeddings.begin()+b*noise.size());
        std::copy(positions.begin(),positions.end(),pos.begin()+b*count);
        std::iota(cache_pos.begin()+b*count,cache_pos.begin()+(b+1)*count,0);
        std::copy(query_positions.begin(),query_positions.end(),queries.begin()+b*width);
    }
    auto df=to_device(features),de=to_device(embeddings),dp=to_device(pos),dc=to_device(cache_pos),dq=to_device(queries);
    auto dn=to_device_i32({0,1,3,65}),dv=to_device_i32({7,7,7,7}),ds=to_device_i32({3,0,2,1});
    Tensor counts(dn.p,DType::I32,{batch}),valid(dv.p,DType::I32,{batch}),slots(ds.p,DType::I32,{batch});
    Tensor input(de.p,DType::BF16,{2560,width,batch}),qpos(dq.p,DType::I32,{width,batch});
    program.append_accepted_context(Tensor(df.p,DType::BF16,{12800,count,batch}),
        Tensor(dp.p,DType::I32,{count,batch}),Tensor(dc.p,DType::I32,{count,batch}),
        counts,slots,{0,count},device.stream);
    auto output=program.draft_embeddings(input,qpos,counts,valid,slots,device.stream);
    device.synchronize();
    const auto packed=from_device<std::uint16_t>(output.data,output.numel());
    int failures=0;
    for(int b=0;b<batch;++b) {
        const auto single=program.draft_embeddings(input.slice(2,b,1),qpos.slice(1,b,1),
            counts.slice(0,b,1),valid.slice(0,b,1),slots.slice(0,b,1),device.stream);
        device.synchronize();
        const std::vector<std::uint16_t> expected(packed.begin()+b*2560*width,packed.begin()+(b+1)*2560*width);
        failures+=verify_exact("DFlash four-request cache/slot isolation",
            from_device<std::uint16_t>(single.data,2560*width),expected);
    }
    return failures;
}
int run_profile(const std::filesystem::path& root,const std::string& profile,DeviceContext& device) {
    Golden golden(root/("qwen4-dflash-"+profile+"-goldens.bin"));
    auto model=q4::LoadedDFlash::load(root/("qwen4-dflash-"+profile+".ninfer"),device);
    ninfer::test::qwen4::DFlashTestCache cache(128,4,device.stream);
    q4::DFlashProgram program(model->weights(),128,65,4,cache.views());
    int failures=0;
    for(const auto& spec:golden.json.at("cases")) {
        const std::string name=spec.at("name");
        const int count=spec.at("context"),width=spec.at("queries");
        auto features=golden.read<std::uint16_t>(name+"/input/features");
        auto noise=golden.read<std::uint16_t>(name+"/input/noise_embeddings");
        auto pos64=golden.read<std::int64_t>(name+"/input/context_positions");
        auto query64=golden.read<std::int64_t>(name+"/input/query_positions");
        std::vector<int> positions(pos64.begin(),pos64.end()),query_positions(query64.begin(),query64.end());
        std::vector<int> cache_positions(count);std::iota(cache_positions.begin(),cache_positions.end(),0);
        auto dnoise=to_device(noise),dquery=to_device(query_positions);
        auto dlength=to_device_i32({count}),dvalid=to_device_i32({width}),dslot=to_device_i32({2});
        Tensor input(dnoise.p,DType::BF16,{2560,width,1});
        Tensor queries(dquery.p,DType::I32,{width,1});
        Tensor length(dlength.p,DType::I32,{1}),valid(dvalid.p,DType::I32,{1}),slot(dslot.p,DType::I32,{1});
        clear(program,device.stream);
        Trace trace;
        if(count) {
            auto df=to_device(features),dp=to_device(positions),dc=to_device(cache_positions);
            program.append_accepted_context(Tensor(df.p,DType::BF16,{12800,count,1}),
                Tensor(dp.p,DType::I32,{count,1}),Tensor(dc.p,DType::I32,{count,1}),length,slot,
                {std::uint32_t(count),std::uint32_t(count)},device.stream,trace.sink());
            device.synchronize();
        }
        device.synchronize();
        const auto committed=snapshot(program);
        auto output=program.draft_embeddings(input,queries,length,valid,slot,device.stream,trace.sink());
        device.synchronize();
        auto actual=from_device_bf16(output.data,output.numel());
        // Each explicit public-Op BF16 materialization is part of this composed route; the trace
        // checker separately checks every Op against FP64 from its ACTUAL represented inputs.
        constexpr ReductionCriterion criterion{.02,.005,.02};
        failures+=verify_reduction((profile+" five-layer public-boundary composition "+name).c_str(),
            actual,golden.read<double>(name+"/materialized/hidden"),criterion);
        const auto ideal=golden.read<double>(name+"/hidden");
        double error=0,norm=0;
        for(std::size_t i=0;i<actual.size();++i) { error+=std::pow(actual[i]-ideal[i],2); norm+=ideal[i]*ideal[i]; }
        std::cout<<profile<<' '<<name<<" whole-unmaterialized-ideal rel_l2="<<std::sqrt(error/norm)
                 <<" (separate 0.02 quality screen; not an Op correctness criterion)\n";
        failures+=verify_exact("DFlash noise never mutates accepted context cache",snapshot(program),committed);
        if(const char* directory=std::getenv("NINFER_QWEN4_DFLASH_TRACE"))
            trace.write(std::filesystem::path(directory)/(profile+"-"+name+".bin"));

        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
        (void)program.draft_embeddings(input,queries,length,valid,slot,device.stream);
        CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        CUDA_CHECK(cudaGraphLaunch(executable,device.stream));device.synchronize();
        failures+=verify_exact("DFlash graph replay",from_device_bf16(output.data,output.numel()),actual);
        CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));

        if(count) {
            // Replaying accepted features in chunks must preserve exact persistent K/V and final
            // draft values. Distinct logical and RoPE positions above expose accidental aliasing.
            clear(program,device.stream);
            for(int begin=0;begin<count;begin+=7) {
                const int n=std::min(7,count-begin);
                std::vector<std::uint16_t> f(features.begin()+begin*12800,features.begin()+(begin+n)*12800);
                std::vector<int> p(positions.begin()+begin,positions.begin()+begin+n);
                std::vector<int> c(cache_positions.begin()+begin,cache_positions.begin()+begin+n);
                auto df=to_device(f),dp=to_device(p),dc=to_device(c),dn=to_device_i32({n});
                program.append_accepted_context(Tensor(df.p,DType::BF16,{12800,n,1}),
                    Tensor(dp.p,DType::I32,{n,1}),Tensor(dc.p,DType::I32,{n,1}),
                    Tensor(dn.p,DType::I32,{1}),slot,{std::uint32_t(n),std::uint32_t(n)},device.stream);
                device.synchronize();
            }
            // Different qualified Linear T routes need not be bit-identical. Compare the
            // recomputed draft against the same mathematical composed oracle, not a copied route.
            output=program.draft_embeddings(input,queries,length,valid,slot,device.stream);
            device.synchronize();
            failures+=verify_reduction((profile+" chunked accepted context "+name).c_str(),
                from_device_bf16(output.data,output.numel()),golden.read<double>(name+"/materialized/hidden"),criterion);

            // Physical suffix poison must not cross the accepted-count boundary.
            const auto before=snapshot(program);
            auto df=to_device(features),dp=to_device(positions),dc=to_device(cache_positions),dz=to_device_i32({0});
            CUDA_CHECK(cudaMemsetAsync(df.p,0xff,features.size()*2,device.stream));
            program.append_accepted_context(Tensor(df.p,DType::BF16,{12800,count,1}),
                Tensor(dp.p,DType::I32,{count,1}),Tensor(dc.p,DType::I32,{count,1}),
                Tensor(dz.p,DType::I32,{1}),slot,{0,std::uint32_t(count)},device.stream);
            device.synchronize();
            failures+=verify_exact("DFlash rejected context cannot mutate cache",snapshot(program),before);
            if(count==65) failures+=batch_case(*model,features,noise,positions,query_positions,device);
        }
    }
    return failures;
}

// The full shared head is authentic. Only the addressed embedding rows are materialized in a
// vocabulary-sized test allocation; this is not a fabricated complete target checkpoint.
int endpoint(const std::filesystem::path& root,const std::filesystem::path& target,DeviceContext& device) {
    constexpr int hidden=2560,vocabulary=248320,width=7;
    qwen4_native::Bf16Source source((target/"qwen4-endpoint.ninfer").string());
    direct_bf16_weight::DeviceWeight head({vocabulary,hidden,source.bits("lm_head.weight",{vocabulary,hidden})});
    artifact::Reader inputs(root/"qwen4-dflash-inputs.ninfer");
    const auto payload=[&](const char* name) { return inputs.payload(*inputs.find(name)).data; };
    const auto ids=payload("anchor.ids"),rows=payload("anchor.embeddings"),mask=payload("mask.embedding");
    std::int32_t anchor;
    std::memcpy(&anchor,ids.data(),4);
    DeviceBuffer embedding(std::size_t(vocabulary)*hidden*2);
    CUDA_CHECK(cudaMemsetAsync(embedding.p,0xff,embedding.bytes,device.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+std::size_t(anchor)*hidden*2,
        rows.data(),hidden*2,cudaMemcpyHostToDevice,device.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.p)+std::size_t(248077)*hidden*2,
        mask.data(),hidden*2,cudaMemcpyHostToDevice,device.stream));
    Weight shared=head.view(); shared.payload=shared.qdata=embedding.p;
    std::vector<std::uint16_t> expected_embeddings(hidden*width);
    std::memcpy(expected_embeddings.data(),rows.data(),hidden*2);
    for(int t=1;t<width;++t) std::memcpy(expected_embeddings.data()+t*hidden,mask.data(),hidden*2);
    auto de=to_device(expected_embeddings),da=to_device_i32({anchor}),dap=to_device_i32({20}),
        dl=to_device_i32({3}),dv=to_device_i32({width}),ds=to_device_i32({0});
    auto dq=to_device_i32({20,21,22,23,24,25,26});
    Tensor anchors(da.p,DType::I32,{1}),anchor_positions(dap.p,DType::I32,{1}),
        lengths(dl.p,DType::I32,{1}),valid(dv.p,DType::I32,{1}),slots(ds.p,DType::I32,{1});
    GuardedDeviceBuffer logits_buffer(std::size_t(vocabulary)*width*2);
    Tensor logits(logits_buffer.data(),DType::BF16,{vocabulary,width,1});
    DeviceBuffer proposed(width*4),final(hidden*width*2);
    Tensor proposals(proposed.p,DType::I32,{width});
    q4::DFlashTrace trace{&final,[](void* self,q4::DFlashBoundary stage,int,const Tensor& value,cudaStream_t stream) {
        if(stage==q4::DFlashBoundary::FinalOutput)
            CUDA_CHECK(cudaMemcpyAsync(static_cast<DeviceBuffer*>(self)->p,value.data,value.bytes(),
                                       cudaMemcpyDeviceToDevice,stream));
    }};
    int failures=0;
    for(const std::string profile:{"bf16","nvfp4"}) {
        auto model=q4::LoadedDFlash::load(root/("qwen4-dflash-"+profile+".ninfer"),device);
        ninfer::test::qwen4::DFlashTestCache cache(128,1,device.stream);
        q4::DFlashProgram program(model->weights(),128,3,1,cache.views());
        clear(program,device.stream);
        Golden golden(root/("qwen4-dflash-"+profile+"-goldens.bin"));
        auto features=to_device(golden.read<std::uint16_t>("c3_k4/input/features"));
        auto rp=to_device_i32({17,18,19}),cp=to_device_i32({0,1,2});
        program.append_accepted_context(Tensor(features.p,DType::BF16,{12800,3,1}),
            Tensor(rp.p,DType::I32,{3,1}),Tensor(cp.p,DType::I32,{3,1}),lengths,slots,{3,3},device.stream);
        program.draft(anchors,anchor_positions,lengths,valid,slots,shared,head.view(),logits,device.stream,trace);
        auto flat_logits=logits.view({vocabulary,width});
        ops::argmax(flat_logits,proposals,vocabulary,device.stream);
        device.synchronize();
        const auto actual=from_device_bf16(logits.data,logits.numel());
        const auto final_bits=from_device<std::uint16_t>(final,hidden*width);
        std::vector<float> activation(final_bits.size());
        std::transform(final_bits.begin(),final_bits.end(),activation.begin(),bf16_to_f32);
        std::vector<double> oracle(actual.size());
        const unsigned threads=std::min(32U,std::max(1U,std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        for(unsigned worker=0;worker<threads;++worker) workers.emplace_back([&,worker] {
            for(int row=worker;row<vocabulary;row+=threads) for(int t=0;t<width;++t)
                oracle[std::size_t(t)*vocabulary+row]=direct_bf16_weight::dot_fp64(head.host,row,
                    std::span<const float>(activation.data()+t*hidden,hidden));
        });
        for(auto& worker:workers) worker.join();
        std::vector<std::int32_t> expected_proposals(width);
        for(int t=0;t<width;++t) {
            const auto begin=std::size_t(t)*vocabulary,end=begin+vocabulary;
            failures+=verify_reduction(profile+" DFlash all-vocabulary head query="+std::to_string(t),
                std::vector<double>(actual.begin()+begin,actual.begin()+end),
                std::vector<double>(oracle.begin()+begin,oracle.begin()+end),{1./256,1./256,2./256});
            expected_proposals[t]=std::max_element(actual.begin()+begin,actual.begin()+end)-(actual.begin()+begin);
        }
        failures+=verify_exact("DFlash every query including query zero produces a proposal",
            from_device<std::int32_t>(proposed,width),expected_proposals);
        failures+=logits_buffer.verify_guards("DFlash full head");
        const auto manual=program.draft_embeddings(Tensor(de.p,DType::BF16,{hidden,width,1}),
            Tensor(dq.p,DType::I32,{width,1}),lengths,valid,slots,device.stream);
        device.synchronize();
        failures+=verify_exact("DFlash authentic anchor/mask gather and shifted positions",
            from_device<std::uint16_t>(manual.data,manual.numel()),final_bits);
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
        program.draft(anchors,anchor_positions,lengths,valid,slots,shared,head.view(),logits,device.stream);
        ops::argmax(flat_logits,proposals,vocabulary,device.stream);
        CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        CUDA_CHECK(cudaGraphLaunch(executable,device.stream));device.synchronize();
        failures+=verify_exact("DFlash full endpoint graph logits",from_device_bf16(logits.data,logits.numel()),actual);
        failures+=verify_exact("DFlash full endpoint graph proposals",from_device<std::int32_t>(proposed,width),expected_proposals);
        CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));
    }
    return failures;
}
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--capture") return capture()?1:0;
    const char* root=std::getenv("NINFER_QWEN4_DFLASH");
    if(!root) { std::cout<<"SKIP: native DFlash qualification artifacts absent\n";return 77; }
    try {
        DeviceContext device;
        if(argc==2 && std::string(argv[1])=="--endpoint") {
            const char* target=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
            if(!target) { std::cout<<"SKIP: authentic shared target head absent\n";return 77; }
            return endpoint(root,target,device)?1:0;
        }
        const int failures=run_profile(root,"bf16",device)+run_profile(root,"nvfp4",device);
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}

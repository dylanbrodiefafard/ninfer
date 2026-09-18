#include "ninfer/ops/gated_residual_stem.h"
#include "ops/direct_bf16_weight.h"
#include "artifact/reader.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr int D=2560, F=10240;
// Declared before measuring synthetic or authentic source profiles.
constexpr ReductionCriterion criterion{.01,.005,.02};

std::vector<std::uint16_t> source(const artifact::Reader& r,const char* name,
                                 std::vector<std::uint64_t> shape) {
    const auto* o=r.find(name);
    const auto* d=o?std::get_if<artifact::TensorDescriptor>(o):nullptr;
    if(!d || d->format!=artifact::NumericFormat::BF16 ||
       d->layout!=artifact::StorageLayout::ContiguousLeV1 || d->shape!=shape)
        throw std::runtime_error("invalid native MTP stem source role");
    auto bytes=r.payload(*o).data;
    std::vector<std::uint16_t> result(bytes.size()/2);
    for(std::size_t i=0;i<result.size();++i)
        result[i]=std::to_integer<unsigned>(bytes[2*i]) | (std::to_integer<unsigned>(bytes[2*i+1])<<8);
    return result;
}
std::vector<float> values(const std::vector<std::uint16_t>& bits) {
    std::vector<float> out(bits.size());
    std::transform(bits.begin(),bits.end(),out.begin(),bf16_to_f32);
    return out;
}
struct Fixture {
    direct_bf16_weight::HostWeight ew,hw;
    std::vector<float> en,hn;
    Fixture():ew(direct_bf16_weight::make_patterned(D,D,173)),
              hw(direct_bf16_weight::make_patterned(D,D,197)),en(D),hn(F) {
        fill_uniform(en,199,-.2f,.2f); fill_uniform(hn,211,-.2f,.2f);
        round_to_bf16(en); round_to_bf16(hn);
    }
    explicit Fixture(const std::string& path) {
        artifact::Reader r(path);
        if(r.identity()!=artifact::ArtifactIdentity{"qwen4/native-mtp-stem-qualification","nvidia-bf16-source"})
            throw std::runtime_error("wrong MTP stem fixture identity");
        ew={D,D,source(r,"mtp.fc_embedding.weight",{D,D})};
        hw={D,D,source(r,"mtp.fc_hidden.weight",{D,D})};
        en=values(source(r,"mtp.pre_fc_norm_embedding.weight",{D}));
        hn=values(source(r,"mtp.pre_fc_norm_hidden.weight",{F}));
    }
};

// Complete mathematical oracle, no private normalization/projection storage casts.
std::vector<double> oracle(const Fixture& f,const std::vector<float>& e,
                           const std::vector<float>& h,int t) {
    auto we=values(f.ew.bits), wh=values(f.hw.bits);
    std::vector<double> result(F*t), ne(D), nh(F);
    for(int token=0;token<t;++token) {
        double es=0,hs=0;
        for(int k=0;k<D;++k) es+=double(e[token*D+k])*e[token*D+k];
        for(int k=0;k<F;++k) hs+=double(h[token*F+k])*h[token*F+k];
        const double ei=1/std::sqrt(es/D+1e-6), hi=1/std::sqrt(hs/F+1e-6);
        for(int k=0;k<D;++k) ne[k]=e[token*D+k]*ei*(1.0+f.en[k]);
        for(int k=0;k<F;++k) nh[k]=h[token*F+k]*hi*(1.0+f.hn[k]);
        for(int n=0;n<D;++n) {
            double ep=0;
            for(int k=0;k<D;++k) ep+=we[n*D+k]*ne[k];
            for(int branch=0;branch<4;++branch) {
                double hp=0;
                for(int k=0;k<D;++k) hp+=wh[n*D+k]*nh[branch*D+k];
                result[token*F+branch*D+n]=ep+hp;
            }
        }
    }
    return result;
}
int run(const Fixture& f,int t,bool graph) {
    std::vector<float> e(D*t), h(F*t);
    fill_uniform(e,223+t,-.8f,.8f); fill_uniform(h,227+t,-1.f,1.f);
    // Deliberately unequal branch variance detects accidental four independent norms.
    constexpr float scale[4]={0.00390625f,.125f,2.f,16.f};
    for(int token=0;token<t;++token) for(int b=0;b<4;++b) for(int k=0;k<D;++k)
        h[token*F+b*D+k]*=scale[b];
    round_to_bf16(e); round_to_bf16(h);
    auto expected=oracle(f,e,h,t);
    auto de=to_device_bf16(e), dh=to_device_bf16(h), den=to_device_bf16(f.en), dhn=to_device_bf16(f.hn);
    direct_bf16_weight::DeviceWeight dew(f.ew), dhw(f.hw);
    GuardedDeviceBuffer dy(F*t*2), ws(ops::gated_residual_stem_workspace_capacity_bytes(t));
    WorkspaceArena arena(DeviceSpan{ws.data(),ws.bytes()});
    Tensor en(den.p,DType::BF16,{D}), hn(dhn.p,DType::BF16,{F});
    auto call=[&](int start,int count,cudaStream_t stream) {
        Tensor et(static_cast<std::uint16_t*>(de.p)+D*start,DType::BF16,{D,count});
        Tensor ht(static_cast<std::uint16_t*>(dh.p)+F*start,DType::BF16,{D,4,count});
        Tensor out(static_cast<std::uint16_t*>(dy.data())+F*start,DType::BF16,{D,4,count});
        ops::gated_residual_stem(et,ht,en,hn,dew.view(),dhw.view(),out,arena,stream);
        if(arena.used()!=0) throw std::runtime_error("stem failed workspace scope restoration");
    };
    call(0,t,nullptr);
    auto got=from_device_bf16(dy.data(),F*t);
    int failures=verify_reduction("stem whole T="+std::to_string(t),got,expected,criterion);
    if(t>1) {
        call(0,t-1,nullptr); call(t-1,1,nullptr);
        failures+=verify_reduction("stem prefill/decode",from_device_bf16(dy.data(),F*t),expected,criterion);
    }
    if(graph) {
        cudaStream_t stream; cudaGraph_t g; cudaGraphExec_t exec;
        cuda_check(cudaStreamCreate(&stream),"stem stream");
        cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"stem capture");
        call(0,t,stream);
        cuda_check(cudaStreamEndCapture(stream,&g),"stem end capture");
        cuda_check(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0),"stem instantiate");
        for(int replay=0;replay<2;++replay) {
            cuda_check(cudaGraphLaunch(exec,stream),"stem replay");
            cuda_check(cudaStreamSynchronize(stream),"stem sync");
            failures+=verify_exact("stem stable graph",from_device_bf16(dy.data(),F*t),got);
        }
        cuda_check(cudaGraphExecDestroy(exec),"stem destroy exec");
        cuda_check(cudaGraphDestroy(g),"stem destroy graph");
        cuda_check(cudaStreamDestroy(stream),"stem destroy stream");
    }
    // Invalid alias and undersized workspace must fail before launches.
    Tensor et(de.p,DType::BF16,{D,t}), ht(dh.p,DType::BF16,{D,4,t}), out(dy.data(),DType::BF16,{D,4,t});
    bool rejected=false;
    try { ops::gated_residual_stem(et,ht,en,hn,dew.view(),dhw.view(),ht,arena,nullptr); }
    catch(const std::invalid_argument&) { rejected=true; }
    failures+=!rejected;
    WorkspaceArena short_arena(DeviceSpan{ws.data(),ws.bytes()-1});
    rejected=false;
    try { ops::gated_residual_stem(et,ht,en,hn,dew.view(),dhw.view(),out,short_arena,nullptr); }
    catch(const std::invalid_argument&) { rejected=true; }
    failures+=!rejected;
    failures+=dy.verify_guards("stem output")+ws.verify_guards("stem workspace");
    failures+=dew.verify_preserved("stem embedding matrix")+dhw.verify_preserved("stem hidden matrix");
    return failures;
}
}
int main(int argc,char** argv) {
    try {
        int failures=0;
        if(argc==2 && std::string(argv[1])=="--native-real") {
            const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
            if(!root) { std::cout<<"SKIP: native MTP stem fixture not configured\n"; return 77; }
            Fixture f(std::string(root)+"/qwen4-mtp-stem.ninfer");
            for(int t:{1,17,28}) failures+=run(f,t,t==17);
        } else {
            Fixture f;
            for(int t:{1,7,27,28,33}) failures+=run(f,t,t==7);
        }
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

#include "ops/linear/linear_test_common.h"
#include "ops/linear/fp8/fp8_tensor.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/op_tester.h"
#include "artifact/reader.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qualification=ninfer::test::linear;
namespace qw=ninfer::test::quantized_weight;

struct Shape { int n,k; };
constexpr std::array shapes{Shape{10240,2560},Shape{6144,2560},Shape{12288,2560},
    Shape{512,2560},Shape{2560,6144},Shape{640,2560},Shape{2560,640}};
const artifact::Reader* real_source=nullptr;
const artifact::TensorDescriptor* real_tensor=nullptr;

qw::PackedWeight make_weight(int n,int k,unsigned seed) {
    auto result=qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S,n,k,seed);
    result.scale_plane_offset=static_cast<std::uint64_t>(n)*k;
    result.scale_plane_bytes=8;
    result.payload.resize(result.scale_plane_offset+8);
    qw::detail::store_u32_le(result.payload,result.scale_plane_offset,0x3ae2b719U);
    qw::detail::store_u32_le(result.payload,result.scale_plane_offset+4,0x3ae58f63U);
    result.weight.qtype=QType::FP8_E4M3FN_TENSOR_F32M;
    result.weight.layout=QuantLayout::TensorCalibrated;
    result.weight.scale_dtype=DType::FP32;
    result.weight.payload_bytes=result.payload.size();
    result.weight.group=0; result.weight.group_size=0;
    result.weight.scale_ne[0]=2;
    result.weight.scale_nb[0]=4;
    for(int i=1;i<4;++i) { result.weight.scale_nb[i]=8; }
    if(real_source) {
        const auto raw=real_source->payload(real_tensor->name).data;
        if(raw.size()!=result.payload.size()) { throw std::runtime_error("native calibrated payload size"); }
        std::memcpy(result.payload.data(),raw.data(),raw.size());
    }
    return result;
}

// Independent enumerated exact codec oracle, not CUDA's conversion intrinsic.
std::uint8_t encode_e4m3(float value) {
    return qw::detail::encode_e4m3fn(value);
}

int codec() {
    int failures=0;
    std::vector<float> witnesses{0.F,-0.F,512.F,-512.F,1000.F,-1000.F};
    for(int code=0;code<126;++code) {
        const float mid=static_cast<float>((qw::detail::decode_e4m3fn(code)+qw::detail::decode_e4m3fn(code+1))*.5);
        const auto bits=f32_to_bf16(mid);
        for(int delta:{-1,0,1}) {
            if(static_cast<int>(bits)+delta<0) { continue; }
            const float v=bf16_to_f32(static_cast<std::uint16_t>(bits+delta));
            witnesses.push_back(v); witnesses.push_back(-v);
        }
    }
    constexpr int tokens=5;
    for(int k:{640,2560,6144}) {
    auto host=make_weight(k==2560?640:2560,k,831);
    DeviceBuffer weights(host.payload.size());
    std::vector<float> input(tokens*k);
    for(std::size_t i=0;i<input.size();++i) { input[i]=witnesses[i%witnesses.size()]; }
    std::fill(input.begin(),input.begin()+k,0.F); // zero keeps the positive source floor
    for(int d=0;d<k;++d) { input[k+d]=std::ldexp(input[2*k+d],-10); }
    round_to_bf16(input); auto device_input=to_device_bf16(input);
    Tensor x(device_input.p,DType::BF16,{k,tokens});
    GuardedDeviceBuffer codebuf(input.size()),scalebuf(tokens*sizeof(float));
    for(float multiplier:{1.F,0.0376674123108387F}) {
        qw::detail::store_u32_le(host.payload,host.scale_plane_offset+4,qw::detail::float_bits(multiplier));
        weights.copy_from_host(host.payload.data(),host.payload.size());
        const Weight w=host.device_weight(weights.p);
        ops::detail::launch_fp8_tensor_quantize(x,w,
            {static_cast<std::uint8_t*>(codebuf.data()),static_cast<float*>(scalebuf.data())},nullptr);
        cuda_synchronize();
        std::vector<std::uint8_t> expected(input.size());
        std::vector<float> expected_scales(tokens,multiplier);
        for(int token=0;token<tokens;++token) {
            float maximum=0.F;
            for(int d=0;d<k;++d) { maximum=std::max(maximum,std::abs(input[token*k+d])); }
            expected_scales[token]=std::max(multiplier,maximum/448.F);
            for(int d=0;d<k;++d) { expected[token*k+d]=encode_e4m3(input[token*k+d]/expected_scales[token]); }
        }
        failures+=verify_exact("calibrated E4M3 ties/subnormals/saturation",from_device<std::uint8_t>(codebuf.data(),input.size()),expected);
        failures+=verify_exact("guarded calibrated input scale words",from_device<float>(scalebuf.data(),tokens),expected_scales);
        failures+=codebuf.verify_guards("FP8 codes"); failures+=scalebuf.verify_guards("FP8 scales");
    }
    }
    return failures;
}

int matrix(const char* label,int n,int k,bool full) {
    const auto problem=ops::detail::resolve_fp8_problem(n,k);
    const int a8=ops::detail::fp8_tensor_a8_first_t(n,k);
    const int mma=ops::detail::fp8_exact_a16_gemm_first_t(problem);
    // Same predeclared Linear gates as existing FP8: A16 relL2=1/256, abs=1/256,
    // gross=2/256; A8 relL2=.04, abs=1/256, gross=.06. These are implementation
    // profiles, not quality budgets. Oracle is the existing independent naive FP64 GEMM.
    const std::array a16{
        qualification::Invocation{1},qualification::Invocation{2},qualification::Invocation{4},
        qualification::Invocation{mma-1},qualification::Invocation{mma},
        qualification::Invocation{128},qualification::Invocation{129,qualification::CallForm::Policy,ops::LinearPolicy::A16Only,true}};
    const std::array a8_calls{
        qualification::Invocation{a8-1,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8},
        qualification::Invocation{a8,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8},
        qualification::Invocation{64,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8},
        qualification::Invocation{65,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8},
        qualification::Invocation{129,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8,true},
        qualification::Invocation{512,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8}};
    int failures=qualification::run_shape(label,qualification::ActivationCompute::A16,make_weight,
        {n,k,907,qualification::Comparison::Sampled,true,a16});
    failures+=qualification::run_shape(label,qualification::ActivationCompute::A8,make_weight,
        {n,k,913,qualification::Comparison::Sampled,true,a8_calls});
    if(full) {
        const std::array calls{qualification::Invocation{a8,qualification::CallForm::Policy,ops::LinearPolicy::AllowA8}};
        failures+=qualification::run_shape(label,qualification::ActivationCompute::A8,make_weight,
            {n,k,917,qualification::Comparison::Full,true,calls});
    }
    return failures;
}

void benchmark(int n,int k) {
    auto host=make_weight(n,k,913);
    DeviceBuffer device_weight(host.payload.size());
    device_weight.copy_from_host(host.payload.data(),host.payload.size());
    const Weight weight=host.device_weight(device_weight.p);
    const int threshold=ops::detail::fp8_tensor_a8_first_t(n,k);
    const std::set<int> widths{1,5,7,8,9,12,13,16,17,20,21,24,25,28,29,
                              threshold-1,threshold,threshold+1,32,64,129,512};
    for(int t:widths) {
        std::vector<float> values(static_cast<std::size_t>(k)*t); fill_uniform(values,831,-.5F,.5F);
        auto input=to_device_bf16(values); DeviceBuffer output(static_cast<std::size_t>(n)*t*2);
        Tensor x(input.p,DType::BF16,{k,t}),y(output.p,DType::BF16,{n,t});
        for(auto policy:{ops::LinearPolicy::A16Only,ops::LinearPolicy::AllowA8}) {
            DeviceArena scratch(std::max<std::size_t>(256,ops::linear_workspace_capacity_bytes(weight.qtype,n,k,policy,t,t)));
            for(int warm=0;warm<10;++warm) { ops::linear(x,weight,y,policy,scratch,nullptr); }
            cudaEvent_t begin,end; cuda_check(cudaEventCreate(&begin),"event"); cuda_check(cudaEventCreate(&end),"event");
            cuda_check(cudaEventRecord(begin),"event record");
            for(int repeat=0;repeat<100;++repeat) { ops::linear(x,weight,y,policy,scratch,nullptr); }
            cuda_check(cudaEventRecord(end),"event record"); cuda_check(cudaEventSynchronize(end),"event sync");
            float ms; cuda_check(cudaEventElapsedTime(&ms,begin,end),"event elapsed");
            cuda_check(cudaEventDestroy(begin),"event destroy"); cuda_check(cudaEventDestroy(end),"event destroy");
            std::cout<<"BENCH "<<real_tensor->name<<" T="<<t<<" policy="
                <<(policy==ops::LinearPolicy::A16Only?"A16":"AllowA8")<<" us="<<ms*10<<'\n';
        }
    }
}
} // namespace

int main(int argc,char**argv) {
    try {
        if(require_cuda()) { return 1; }
        if(argc==2 && (std::string(argv[1])=="--native-real" || std::string(argv[1])=="--bench-native")) {
            const bool bench=std::string(argv[1])=="--bench-native";
            const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
            if(!root) { std::cout<<"SKIP native FP8 projections\n"; return 77; }
            artifact::Reader reader(std::string(root)+"/qwen4-fp8-projections.ninfer");
            if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-fp8-projection-qualification","senfu-fp8-source"}
                || reader.objects().size()!=13) { throw std::runtime_error("wrong native FP8 fixture"); }
            artifact::Binder binder(reader);
            std::vector<artifact::ObjectHandle> handles;
            for(const auto& object:reader.objects()) {
                const auto* tensor=std::get_if<artifact::TensorDescriptor>(&object);
                if(!tensor || tensor->shape.size()!=2) { throw std::runtime_error("invalid projection object"); }
                handles.push_back(artifact::bind_device_tensor(binder,tensor->name,
                    artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M,{tensor->shape[0],tensor->shape[1]}));
            }
            DeviceContext device(0);
            auto resident=artifact::materialize(reader,binder.finish(),device);
            real_source=&reader; int failures=0;
            std::size_t object_index=0;
            for(const auto& object:reader.objects()) {
                real_tensor=std::get_if<artifact::TensorDescriptor>(&object);
                if(!real_tensor || real_tensor->format!=artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M || real_tensor->shape.size()!=2)
                    { throw std::runtime_error("invalid calibrated projection"); }
                const auto bound=artifact::materialized_weight(resident,handles[object_index++],
                    real_tensor->format,real_tensor->shape[0],real_tensor->shape[1]);
                ops::detail::validate_fp8_tensor_weight(bound,"native artifact qualification");
                std::array<std::uint32_t,2> actual_words{},expected_words{};
                cuda_check(cudaMemcpy(actual_words.data(),bound.scales,8,cudaMemcpyDeviceToHost),"native source multipliers");
                const auto bytes=reader.payload(*real_tensor).data;
                std::memcpy(expected_words.data(),bytes.data()+bytes.size()-8,8);
                if(actual_words!=expected_words) { throw std::runtime_error("source scalar words changed in materialization"); }
                if(bench) { benchmark(real_tensor->shape[0],real_tensor->shape[1]); }
                else { failures+=matrix(real_tensor->name.c_str(),real_tensor->shape[0],real_tensor->shape[1],real_tensor->shape[0]==512); }
            }
            return failures?1:0;
        }
        int failures=codec();
        for(auto shape:shapes) { failures+=matrix("synthetic tensor FP8",shape.n,shape.k,shape.n==512); }
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

#include "ninfer/ops/vision_patch_merger.h"
#include "artifact/reader.h"
#include "ops/op_tester.h"
#include "ops/native_projection_fixture.h"
#include "targets/qwen4/native_bf16_fixture.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr ReductionCriterion kCriterion{0.02,1e-4,0.02};

struct Fixture {
    quantized_weight::PackedWeight fc1,fc2;
    std::vector<float> gamma,beta,b1,b2;
    DeviceBuffer d_fc1,d_fc2,d_gamma,d_beta,d_b1,d_b2;
    explicit Fixture(const std::string& path={}) {
        if(path.empty()) {
            fc1=native_sparse_fixture(QType::BF16_CTRL,4608,4608);
            fc2=native_sparse_fixture(QType::BF16_CTRL,2560,4608);
            for(int row=0;row<4608;++row) {
                native_sparse_set(fc1,row,(row*17+31)%4608,0.5F);
                native_sparse_set(fc1,row,(row*17+32)%4608,-1.F);
            }
            for(int row=0;row<2560;++row) {
                native_sparse_set(fc2,row,(row*7+13)%4608,1.F);
                native_sparse_set(fc2,row,(row*7+14)%4608,-1.F);
            }
            gamma.resize(1152);beta.resize(1152);b1.resize(4608);b2.resize(2560);
            fill_uniform(gamma,102,-1.2F,1.3F);fill_uniform(beta,103,-.25F,.25F);
            fill_uniform(b1,104,-.75F,.75F);fill_uniform(b2,105,-.25F,.25F);
            for(auto* values:{&gamma,&beta,&b1,&b2}) { round_to_bf16(*values); }
        } else {
            artifact::Reader reader(path);
            if(reader.identity()!=artifact::ArtifactIdentity{
                "qwen4/native-vision-merger-qualification","nvidia-bf16-source"}) {
                throw std::invalid_argument("wrong native Vision merger fixture");
            }
            const auto bits=[&](const char* role,std::vector<std::uint64_t> shape) {
                const auto* object=reader.find(std::string("model.visual.merger.")+role);
                const auto* descriptor=object?std::get_if<artifact::TensorDescriptor>(object):nullptr;
                if(!descriptor || descriptor->format!=artifact::NumericFormat::BF16 ||
                   descriptor->layout!=artifact::StorageLayout::ContiguousLeV1 || descriptor->shape!=shape) {
                    throw std::invalid_argument("invalid native Vision merger tensor");
                }
                const auto payload=reader.payload(*descriptor).data;
                std::vector<std::uint16_t> result(payload.size()/2);
                std::memcpy(result.data(),payload.data(),payload.size());return result;
            };
            const auto values=[&](const char* role,int width) {
                const auto raw=bits(role,{static_cast<unsigned>(width)});
                std::vector<float> result;for(auto value:raw) { result.push_back(bf16_to_f32(value)); }
                return result;
            };
            fc1=qwen4_native::bf16_matrix(bits("linear_fc1.weight",{4608,4608}),4608,4608);
            fc2=qwen4_native::bf16_matrix(bits("linear_fc2.weight",{2560,4608}),2560,4608);
            gamma=values("norm.weight",1152);beta=values("norm.bias",1152);
            b1=values("linear_fc1.bias",4608);b2=values("linear_fc2.bias",2560);
        }
        d_fc1=to_device(fc1.payload);d_fc2=to_device(fc2.payload);
        d_gamma=to_device_bf16(gamma);d_beta=to_device_bf16(beta);
        d_b1=to_device_bf16(b1);d_b2=to_device_bf16(b2);
    }
    ops::VisionPatchMergerWeights view() const {
        return {Tensor(d_gamma.p,DType::BF16,{1152}),Tensor(d_beta.p,DType::BF16,{1152}),
            fc1.device_weight(d_fc1.p),Tensor(d_b1.p,DType::BF16,{4608}),
            fc2.device_weight(d_fc2.p),Tensor(d_b2.p,DType::BF16,{2560})};
    }
};

// One complete independent FP64 formula for every source/synthetic/group/chunk cell.
// No private normalized-token, projection, bias or activation casts are copied here.
std::vector<double> oracle(const Fixture& w,const std::vector<float>& input,int groups) {
    std::vector<double> output;
    for(int g=0;g<groups;++g) {
        std::vector<double> merged(4608);
        for(int patch=0;patch<4;++patch) {
            const int offset=g*4608+patch*1152;
            double mean=0,variance=0;
            for(int d=0;d<1152;++d) { mean+=input[offset+d]; } mean/=1152;
            for(int d=0;d<1152;++d) { const double v=input[offset+d]-mean;variance+=v*v; }
            const double inv=1/std::sqrt(variance/1152+1e-6);
            for(int d=0;d<1152;++d) {
                merged[patch*1152+d]=(input[offset+d]-mean)*inv*w.gamma[d]+w.beta[d];
            }
        }
        auto hidden=native_projection_oracle(w.fc1,merged);
        for(int d=0;d<4608;++d) {
            const double z=hidden[d]+w.b1[d];hidden[d]=.5*z*(1+std::erf(z/std::sqrt(2.0)));
        }
        auto projected=native_projection_oracle(w.fc2,hidden);
        for(int d=0;d<2560;++d) { output.push_back(projected[d]+w.b2[d]); }
    }
    return output;
}

int run(const Fixture& fixture,int groups,bool constant=false) {
    std::vector<float> input(groups*4608);
    fill_uniform(input,71031,-.3F,.3F);
    // Different patch offsets/variance exercise per-patch normalization and merge-major order.
    for(int g=0;g<groups;++g) for(int p=0;p<4;++p) for(int d=0;d<1152;++d) {
        auto& value=input[g*4608+p*1152+d];
        value=constant ? float(p-2):value*(p+1)+float(p-2)*.125F;
    }
    round_to_bf16(input);
    const auto expected=oracle(fixture,input,groups);
    auto dx=to_device_bf16(input);
    GuardedDeviceBuffer dy(groups*2560*2),scratch(ops::vision_patch_merger_workspace_bytes(groups));
    Tensor x(dx.p,DType::BF16,{1152,4,groups}),y(dy.data(),DType::BF16,{2560,groups}),
        workspace(scratch.data(),DType::U8,{static_cast<int>(scratch.bytes())});
    const auto weights=fixture.view();
    int failures=0;
    for(bool partitioned:{false,true}) {
        if(partitioned && groups!=129) { continue; }
        for(int start=0;start<groups;) {
            const int count=partitioned ? (start==0?128:1):groups;
            auto xx=x.slice(2,start,count),yy=y.slice(1,start,count);
            ops::vision_patch_merger(xx,weights,yy,workspace,nullptr);start+=count;
        }
        cuda_synchronize();
        const auto actual=from_device_bf16(dy.data(),expected.size());
        failures+=verify_reduction("Vision merger complete FP64 oracle",actual,expected,kCriterion);
        double error2=0,norm2=0,maximum=0;
        for(std::size_t i=0;i<actual.size();++i) {
            const double e=actual[i]-expected[i];error2+=e*e;norm2+=expected[i]*expected[i];maximum=std::max(maximum,std::abs(e));
        }
        std::cout<<"VISION_MERGER G="<<groups<<" chunk="<<partitioned<<" constant="<<constant
            <<" relative_l2="<<std::sqrt(error2/norm2)<<" max_abs="<<maximum<<std::endl;
    }
    if(groups==1) {
        dy.fill(0x5a);scratch.fill(0x5a);
        auto aliased=y;aliased.data=x.data;
        try { ops::vision_patch_merger(x,weights,aliased,workspace,nullptr);++failures; }
        catch(const std::invalid_argument&) {}
        cuda_synchronize();
        std::vector<std::uint16_t> original_bits;
        for(float value:input) { original_bits.push_back(f32_to_bf16(value)); }
        failures+=verify_exact("Vision merger rejection preserves input",
            from_device<std::uint16_t>(dx.p,input.size()),original_bits);
        failures+=verify_exact("Vision merger rejection preserves output",
            from_device<std::uint8_t>(dy.data(),dy.bytes()),std::vector<std::uint8_t>(dy.bytes(),0x5a));
    }
    failures+=dy.verify_guards("Vision merger output");
    failures+=scratch.verify_guards("Vision merger workspace");
    return failures;
}
}

int main(int argc,char** argv) {
    if(require_cuda()!=0) { return 1; }
    std::string path;
    if(argc==2 && std::string_view(argv[1])=="--native-real") {
        const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");if(!root) { return 77; }
        path=std::string(root)+"/qwen4-vision-merger.ninfer";
    } else if(argc!=1) { return 1; }
    Fixture fixture(path);
    int failures=0;
    for(int groups:{1,5,28,129}) { failures+=run(fixture,groups); }
    if(path.empty()) { failures+=run(fixture,1,true); }
    return failures?1:0;
}

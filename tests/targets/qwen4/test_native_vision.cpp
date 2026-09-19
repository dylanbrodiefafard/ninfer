#include "targets/qwen4/vision.h"
#include "targets/qwen4/vision_frontend.h"
#include "ops/op_tester.h"
#include "artifact/reader.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>

using namespace ninfer;
using namespace ninfer::test;
namespace q4=ninfer::targets::qwen4;
namespace {
template<class T> std::vector<T> read(const artifact::Reader& r,const char* name) {
    const auto* object=r.find(name); if(!object) throw std::runtime_error("missing Vision reference tensor");
    const auto bytes=r.payload(*object).data;
    std::vector<T> out(bytes.size()/sizeof(T)); std::memcpy(out.data(),bytes.data(),bytes.size()); return out;
}
int controls() {
    const std::array grids{q4::VisionGrid{1,2,2},q4::VisionGrid{2,2,4}};
    const auto c=q4::prepare_vision_control(grids);
    const std::vector<int> expected_segments{0,4,12,20};
    int failures=verify_exact("Vision temporal-plane segmentation",c.segments,expected_segments);
    for(int i=0;i<c.patches;++i) {
        const int local=i<4?i:(i-4)%8;
        const int y=(local%4)/2,x=(local/4)*2+local%2;
        if(c.positions[i]!=y || c.positions[c.patches+i]!=x) ++failures;
        float sum=0; for(int j=0;j<4;++j) sum+=c.position_weights[4*i+j];
        if(std::abs(sum-1.F)>1e-6F) ++failures;
    }
    bool rejected=false;
    try { (void)q4::prepare_vision_control(std::array{q4::VisionGrid{1,3,2}}); }
    catch(const std::invalid_argument&) { rejected=true; }
    failures+=!rejected;
    const std::vector<std::uint8_t> types{0,1,1,1,1,0,2,2,0,2,2,0};
    const auto m=q4::prepare_multimodal_positions(types,std::array{q4::VisionGrid{1,4,4}},
                                                std::array{q4::VisionGrid{2,2,4}});
    const std::vector<int> expected{0,1,2,3,4,5,6,7,8,9,10,11,
        0,1,1,1,1,3,4,4,6,7,7,9,
        0,1,1,2,2,3,4,4,6,7,7,9,
        0,1,2,1,2,3,4,5,6,7,8,9};
    failures+=verify_exact("Qwen4 causal and multimodal position rows",m.values,expected);
    failures+=m.rope_delta!=-2;
    failures+=verify_exact("Qwen4 image placement",m.image_columns,std::vector<int>{1,2,3,4});
    failures+=verify_exact("Qwen4 video placement",m.video_columns,std::vector<int>{6,7,9,10});
    return failures;
}
media::decode::Image pattern(int h,int w,int frame) {
    media::decode::Image result{w,h,std::vector<std::uint8_t>(std::size_t(h)*w*3)};
    for(std::size_t i=0;i<result.rgb.size();++i) result.rgb[i]=(i*29+frame*37)%256;
    return result;
}
int frontend() {
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { std::cout<<"SKIP: Vision frontend reference absent\n"; return 77; }
    artifact::Reader r(std::string(root)+"/qwen4-vision-frontend-reference.ninfer");
    int failures=0;
    for(auto [name,frames,h,w]:std::array<std::tuple<const char*,int,int,int>,3>{{
        {"image",1,83,121},{"video",3,63,95},{"small_video",2,17,25}}}) {
        q4::VisionPixels got;
        if(frames==1) got=q4::prepare_vision_image(pattern(h,w,0));
        else {
            media::decode::Video video; video.width=w;video.height=h;video.fps=24.;
            for(int f=0;f<frames;++f) { video.frames.push_back(pattern(h,w,f));video.indices.push_back(f*12); }
            got=q4::prepare_vision_video(video);
            for(int t=0;t<got.grid.temporal;++t) {
                const double expected=(double(t*2)+std::min(t*2+1,frames-1))*.25;
                if(got.timestamps[t]!=expected) ++failures;
            }
        }
        failures+=verify_exact("Vision frontend grid",std::vector<int>{got.grid.temporal,got.grid.height,got.grid.width},
            read<int>(r,(std::string(name)+".grid").c_str()));
        const auto expected=read<float>(r,(std::string(name)+".patches").c_str());
        if(got.patches.size()!=expected.size()) { ++failures;continue; }
        double maximum=0;
        for(std::size_t i=0;i<expected.size();++i) maximum=std::max(maximum,double(std::abs(got.patches[i]-expected[i])));
        std::cout<<"Vision frontend "<<name<<" max_abs="<<maximum<<'\n';
        if(maximum>1e-6) { std::cerr<<"Vision source pixel preparation differs\n";++failures; }
    }
    return failures;
}
int real() {
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { std::cout<<"SKIP: native complete Vision fixture absent\n"; return 77; }
    artifact::Reader reference(std::string(root)+"/qwen4-vision-reference.ninfer");
    if(reference.identity()!=artifact::ArtifactIdentity{"qwen4/native-vision-reference","fp64-public-op-boundaries"})
        throw std::runtime_error("wrong native Vision reference identity");
    DeviceContext device;
    auto model=q4::LoadedVision::load(std::string(root)+"/qwen4-vision.ninfer",device);
    q4::VisionProgram program(model->weights(),12,2);
    const auto control=q4::prepare_vision_control(std::array{q4::VisionGrid{1,2,2},q4::VisionGrid{1,2,4}});
    int failures=verify_exact("Vision source positions",control.positions,read<int>(reference,"positions"));
    failures+=verify_exact("Vision source interpolation indices",control.position_indices,read<int>(reference,"indices"));
    failures+=verify_exact("Vision source interpolation weights",control.position_weights,read<float>(reference,"weights"));
    failures+=verify_exact("Vision source segments",control.segments,read<int>(reference,"segments"));
    program.configure(control,device.stream);
    const auto patches=read<std::uint16_t>(reference,"patches");
    auto dp=to_device(patches);
    Tensor input(dp.p,DType::BF16,{1536,12});
    DeviceBuffer trace_storage(28*1152*12*2);
    q4::VisionTrace trace{&trace_storage,[](void* context,int layer,const Tensor& x,cudaStream_t stream) {
        auto& buffer=*static_cast<DeviceBuffer*>(context);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(buffer.p)+(layer+1)*x.bytes(),x.data,
                                    x.bytes(),cudaMemcpyDeviceToDevice,stream));
    }};
    auto out=program.execute(input,device.stream,&trace); device.synchronize();
    if(const char* path=std::getenv("NINFER_QWEN4_VISION_TRACE")) {
        std::vector<std::uint16_t> bits(trace_storage.bytes/2);
        trace_storage.copy_to_host(bits.data(),trace_storage.bytes);
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bits.data()),trace_storage.bytes);
        if(!file) throw std::runtime_error("cannot write requested Vision diagnostic trace");
    }
    const auto got=from_device_bf16(out.data,2560*3);
    const auto encoded=from_device_bf16(program.encoder_output().data,1152*12);
    const auto expect=read<float>(reference,"output"),encoder=read<float>(reference,"encoder");
    // Predeclared full-composition gate, unchanged from native source-block composition.
    constexpr ReductionCriterion criterion{.02,.005,.02};
    failures+=verify_reduction("complete native 27-block Vision encoder",encoded,
        std::vector<double>(encoder.begin(),encoder.end()),criterion);
    failures+=verify_reduction("complete native Vision merger output",got,
        std::vector<double>(expect.begin(),expect.end()),criterion);
    for(int token=0;token<3;++token) {
        failures+=verify_reduction("Vision per-visual-token composition",
            std::span<const double>(got).subspan(token*2560,2560),
            std::vector<double>(expect.begin()+token*2560,expect.begin()+(token+1)*2560),criterion);
    }

    // All device buffers and controls are startup-fixed; execute is capture-safe.
    cudaGraph_t graph=nullptr; cudaGraphExec_t executable=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
    (void)program.execute(input,device.stream);
    CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    CUDA_CHECK(cudaGraphLaunch(executable,device.stream)); device.synchronize();
    failures+=verify_exact("Vision graph replay",from_device_bf16(out.data,2560*3),got);
    CUDA_CHECK(cudaGraphExecDestroy(executable)); CUDA_CHECK(cudaGraphDestroy(graph));

    // A second segment cannot affect the first segment, including all27 blocks and merger.
    auto changed=patches;
    for(std::size_t i=4*1536;i<changed.size();++i) changed[i]^=0x8000;
    dp.copy_from_host(changed.data(),changed.size()*2);
    (void)program.execute(input,device.stream); device.synchronize();
    const auto altered=from_device_bf16(out.data,2560);
    failures+=verify_exact("Vision full-tower segment isolation",altered,std::vector<double>(got.begin(),got.begin()+2560));

    // The exact same first image alone must preserve merge order and output coordinates.
    const auto one=q4::prepare_vision_control(std::array{q4::VisionGrid{1,2,2}});
    program.configure(one,device.stream);
    Tensor first(dp.p,DType::BF16,{1536,4});
    auto separate=program.execute(first,device.stream); device.synchronize();
    failures+=verify_exact("Vision packed versus separate image",from_device_bf16(separate.data,2560),
        std::vector<double>(got.begin(),got.begin()+2560));
    return failures;
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--native-real") return real();
        if(argc==2 && std::string(argv[1])=="--frontend-real") return frontend();
        return controls()?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

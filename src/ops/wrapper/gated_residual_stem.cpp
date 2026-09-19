#include "ninfer/ops/gated_residual_stem.h"
#include "ops/common/projection.h"
#include "ops/launcher/gated_residual_stem.h"
#include "core/layout.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops {
namespace {
struct Scratch { Tensor en, hn, ep, hp; };
template<class A> Scratch scratch(A& a, int t) {
    return {a.alloc(DType::FP32,{2560,t}), a.alloc(DType::FP32,{10240,t}),
            a.alloc(DType::BF16,{2560,t}), a.alloc(DType::BF16,{2560,4*t})};
}
void tensor(const Tensor& x, std::array<int,4> shape) {
    if(x.dtype!=DType::BF16 || !x.data || !x.is_contiguous() ||
       (reinterpret_cast<std::uintptr_t>(x.data)&15)) throw std::invalid_argument("gated_residual_stem: invalid tensor");
    for(int i=0;i<4;++i) if(x.ne[i]!=shape[i]) throw std::invalid_argument("gated_residual_stem: invalid shape");
}
}
std::size_t gated_residual_stem_workspace_capacity_bytes(std::int32_t t) {
    if(t<1 || t>4096) throw std::invalid_argument("gated_residual_stem: T outside [1,4096]");
    WorkspaceLayoutBuilder a;
    (void)scratch(a,t);
    return a.peak_bytes();
}
void gated_residual_stem(const Tensor& e, const Tensor& h, const Tensor& en, const Tensor& hn,
                         const Weight& ew, const Weight& hw, Tensor& out,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const int t=e.ne[1];
    const auto required=gated_residual_stem_workspace_capacity_bytes(t);
    tensor(e,{2560,t,1,1}); tensor(h,{2560,4,t,1}); tensor(en,{2560,1,1,1});
    tensor(hn,{10240,1,1,1}); tensor(out,{2560,4,t,1});
    for(const Weight* w:{&ew,&hw}) {
        if(w->qtype!=QType::BF16_CTRL) throw std::invalid_argument("gated_residual_stem: BF16 weights required");
        detail::validate_native_projection(*w,2560,2560,"gated_residual_stem");
    }
    if(!workspace.base() || workspace.capacity()<required ||
       workspace.used()>workspace.capacity()-required) throw std::invalid_argument("gated_residual_stem: insufficient workspace");
    const std::array<DeviceSpan,8> ranges{{{e.data,e.bytes()},{h.data,h.bytes()},
        {en.data,en.bytes()},{hn.data,hn.bytes()},{const_cast<void*>(ew.payload),ew.payload_bytes},
        {const_cast<void*>(hw.payload),hw.payload_bytes},{out.data,out.bytes()},
        {workspace.base(),workspace.capacity()}}};
    for(std::size_t i=0;i<ranges.size();++i) for(std::size_t j=0;j<i;++j) {
        auto a=reinterpret_cast<std::uintptr_t>(ranges[i].data), b=reinterpret_cast<std::uintptr_t>(ranges[j].data);
        if(a<b+ranges[j].bytes && b<a+ranges[i].bytes) throw std::invalid_argument("gated_residual_stem: overlapping storage");
    }
    auto scope=workspace.scope();
    auto s=scratch(workspace,t);
    Tensor flat(h.data,DType::BF16,{10240,t});
    detail::gated_residual_stem_normalize_launch(e,en,s.en,stream);
    detail::gated_residual_stem_normalize_launch(flat,hn,s.hn,stream);
    Tensor branches(s.hn.data,DType::FP32,{2560,4*t});
    detail::gated_residual_stem_project_launch(s.en,ew,s.ep,stream);
    detail::gated_residual_stem_project_launch(branches,hw,s.hp,stream);
    detail::gated_residual_stem_add_launch(s.ep,s.hp,out,stream);
}
}

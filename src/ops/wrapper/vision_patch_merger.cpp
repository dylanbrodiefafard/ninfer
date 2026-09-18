#include "ninfer/ops/vision_patch_merger.h"
#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ops/common/projection.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
void require_tensor(const Tensor& t,DType type,int d0,int d1,int d2=1) {
    if(t.dtype!=type || t.ne[0]!=d0 || t.ne[1]!=d1 || t.ne[2]!=d2 || t.ne[3]!=1 ||
       !t.is_contiguous() || !t.data || (reinterpret_cast<std::uintptr_t>(t.data)&15)) {
        throw std::invalid_argument("vision_patch_merger: invalid tensor");
    }
}
struct Range { std::uintptr_t begin,end; };
Range range(const void* p,std::size_t bytes) {
    const auto begin=reinterpret_cast<std::uintptr_t>(p);
    if(!p || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-begin) {
        throw std::invalid_argument("vision_patch_merger: invalid storage");
    }
    return {begin,begin+bytes};
}
void validate_groups(int groups) {
    if(groups<=0 || groups>std::numeric_limits<int>::max()/4608) {
        throw std::invalid_argument("vision_patch_merger: invalid group count");
    }
}
template<class Allocator>
auto scratch(Allocator& a,int groups) {
    return std::array{a.alloc(DType::BF16,{1152,4,groups}),a.alloc(DType::BF16,{4608,groups})};
}
}

std::size_t vision_patch_merger_workspace_bytes(std::int32_t groups) {
    validate_groups(groups);
    WorkspaceLayoutBuilder layout;
    (void)scratch(layout,groups);
    return layout.peak_bytes();
}

void vision_patch_merger(const Tensor& x,const VisionPatchMergerWeights& w,
                         Tensor& y,Tensor& workspace,cudaStream_t stream) {
    const int groups=x.ne[2];
    const auto bytes=vision_patch_merger_workspace_bytes(groups);
    require_tensor(x,DType::BF16,1152,4,groups);
    require_tensor(y,DType::BF16,2560,groups);
    require_tensor(w.norm_weight,DType::BF16,1152,1);
    require_tensor(w.norm_bias,DType::BF16,1152,1);
    require_tensor(w.fc1_bias,DType::BF16,4608,1);
    require_tensor(w.fc2_bias,DType::BF16,2560,1);
    require_tensor(workspace,DType::U8,workspace.ne[0],1);
    if(workspace.ne[0]<=0 || workspace.bytes()<bytes ||
       (reinterpret_cast<std::uintptr_t>(workspace.data)&255)) {
        throw std::invalid_argument("vision_patch_merger: insufficient or misaligned workspace");
    }
    if(w.fc1.qtype!=QType::BF16_CTRL || w.fc2.qtype!=QType::BF16_CTRL) {
        throw std::invalid_argument("vision_patch_merger: weights must be native BF16");
    }
    detail::validate_native_projection(w.fc1,4608,4608,"vision_patch_merger");
    detail::validate_native_projection(w.fc2,2560,4608,"vision_patch_merger");
    const std::array ranges{range(x.data,x.bytes()),range(y.data,y.bytes()),
        range(w.norm_weight.data,w.norm_weight.bytes()),range(w.norm_bias.data,w.norm_bias.bytes()),
        range(w.fc1.qdata,w.fc1.payload_bytes),range(w.fc1_bias.data,w.fc1_bias.bytes()),
        range(w.fc2.qdata,w.fc2.payload_bytes),range(w.fc2_bias.data,w.fc2_bias.bytes()),
        range(workspace.data,workspace.bytes())};
    for(std::size_t i=0;i<ranges.size();++i) for(std::size_t j=0;j<i;++j) {
        if(ranges[i].begin<ranges[j].end && ranges[j].begin<ranges[i].end) {
            throw std::invalid_argument("vision_patch_merger: overlapping storage");
        }
    }
    DeviceArena arena(DeviceSpan{workspace.data,workspace.bytes()});
    auto buffers=scratch(arena,groups);
    layer_norm(x,w.norm_weight,w.norm_bias,1e-6F,buffers[0],stream);
    auto merged=buffers[0].reshape({4608,groups});
    linear(merged,w.fc1,buffers[1],LinearPolicy::A16Only,arena,stream);
    add_bias(w.fc1_bias,buffers[1],stream);
    gelu(buffers[1],GeluMode::Exact,stream);
    linear(buffers[1],w.fc2,y,LinearPolicy::A16Only,arena,stream);
    add_bias(w.fc2_bias,y,stream);
}
} // namespace ninfer::ops

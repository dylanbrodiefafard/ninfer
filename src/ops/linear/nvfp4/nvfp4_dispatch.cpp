#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    cudaStream_t stream) {
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("nvfp4 linear: only A16Only is supported");
    }
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (weight.n != Nvfp4LinearDecodeGeometry::kOutputRows ||
        weight.k != Nvfp4LinearDecodeGeometry::kInputRows || x.ne[1] <= 0 ||
        x.ne[1] > kNvfp4LastSmallT) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape or T");
    }
    if (x.ne[1] == 1) {
        launch_nvfp4_decode(x, weight, out, stream);
    } else {
        launch_nvfp4_small_t(x, weight, out, stream);
    }
}

} // namespace ninfer::ops::detail

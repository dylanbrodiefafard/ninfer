// ninfer::ops - residual_rmsnorm wrapper: public API validation + launcher dispatch.
// Host-compiled; never includes the kernel header. See docs/op-development.md §2.
#include "ninfer/ops/residual_rmsnorm.h"

#include "ops/launcher/residual_rmsnorm.h" // detail::residual_rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_nonzero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] <= 0) {
            throw std::invalid_argument("residual_rmsnorm: y/x/out dimensions must be positive");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(
                std::string("residual_rmsnorm: ") + std::string(b_label) +
                " shape must match y [x/out]");
        }
    }
}

} // namespace

void residual_rmsnorm(const Tensor& y, Tensor& x, const Tensor& weight, float eps, Tensor& out,
                       cudaStream_t stream) {
    if (y.dtype != DType::BF16 || x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("residual_rmsnorm: y/x/weight/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("residual_rmsnorm: eps must be positive and finite");
    }
    require_same_shape(y, x, "x");
    require_same_shape(y, out, "out");
    if (weight.ne[0] != y.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("residual_rmsnorm: weight must be 1-D with ne[0] == y.ne[0]");
    }
    if (numel_nonzero(y) == 0) { return; }
    const std::int64_t rows = numel_nonzero(out) / out.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("residual_rmsnorm: row count exceeds CUDA grid limit");
    }
    if (!y.is_contiguous() || !x.is_contiguous() || !weight.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("residual_rmsnorm: y/x/weight/out must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr || weight.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("residual_rmsnorm: y/x/weight/out data must be non-null");
    }

    detail::residual_rmsnorm_launch(y, x, weight, eps, out, stream);
}

} // namespace ninfer::ops
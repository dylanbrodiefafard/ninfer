#include "ops/linear/w8/w8_dispatch.h"

#include "ops/linear/w8/w8_rowsplit_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena& ws, cudaStream_t stream) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        (void)ws;
        w8_rowsplit_dispatch(x, w, out, stream);
        return;
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("w8 linear: unsupported policy");
}

} // namespace ninfer::ops::detail

#include "ops/linear/q5/q5_dispatch.h"

#include "ops/linear/q5/q5_rowsplit_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena& ws, cudaStream_t stream) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        (void)ws;
        q5_rowsplit_dispatch(x, w, out, stream);
        return;
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q5 linear: unsupported policy");
}

} // namespace ninfer::ops::detail

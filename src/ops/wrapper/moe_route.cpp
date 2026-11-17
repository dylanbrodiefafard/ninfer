// ninfer::ops - moe_route wrapper: implements the public api, validates parameters, and dispatches
// to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/moe_route.h"

#include "ops/launcher/moe_route.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool valid(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return t.dtype == dtype && t.ne[0] == rows && t.ne[1] == tokens && t.ne[2] == 1 &&
           t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
           (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

} // namespace

void moe_route(const Tensor& logits, Tensor& expert_ids, Tensor& route_weights,
               Tensor& shared_scale, const Tensor& valid_columns, std::int32_t packed_width,
               cudaStream_t stream) {
    const std::int32_t tokens = logits.ne[1];
    if (tokens < 1 || !valid(logits, DType::FP32, kMoeControlRows, tokens)) {
        throw std::invalid_argument(
            "moe_route: logits must be aligned contiguous FP32 [544,T], T>0");
    }
    if (!valid(expert_ids, DType::I32, kMoeTopK, tokens) ||
        !valid(route_weights, DType::FP32, kMoeTopK, tokens) ||
        !valid(shared_scale, DType::FP32, tokens, 1)) {
        throw std::invalid_argument(
            "moe_route: outputs must be aligned contiguous I32 [10,T], FP32 [10,T], FP32 [T]");
    }
    if (overlaps(expert_ids, logits) || overlaps(route_weights, logits) ||
        overlaps(shared_scale, logits) || overlaps(expert_ids, route_weights) ||
        overlaps(expert_ids, shared_scale) || overlaps(route_weights, shared_scale)) {
        throw std::invalid_argument("moe_route: tensors must be pairwise disjoint");
    }
    const bool packed = valid_columns.data != nullptr;
    if (packed_width < 1 || tokens % packed_width != 0 || (!packed && packed_width != 1) ||
        (packed && !valid(valid_columns, DType::I32, tokens / packed_width, 1))) {
        throw std::invalid_argument(
            "moe_route: valid_columns must be aligned contiguous I32 [T/packed_width], "
            "or empty with packed_width=1; packed_width must divide T");
    }
    if (packed &&
        (overlaps(valid_columns, logits) || overlaps(valid_columns, expert_ids) ||
         overlaps(valid_columns, route_weights) || overlaps(valid_columns, shared_scale))) {
        throw std::invalid_argument("moe_route: valid_columns must be disjoint from other tensors");
    }
    detail::moe_route_launch(
        static_cast<const float*>(logits.data), static_cast<std::int32_t*>(expert_ids.data),
        static_cast<float*>(route_weights.data), static_cast<float*>(shared_scale.data), tokens,
        static_cast<const std::int32_t*>(valid_columns.data), packed_width, stream);
}

} // namespace ninfer::ops

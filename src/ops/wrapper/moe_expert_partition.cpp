// ninfer::ops - moe_expert_partition / moe_expert_partial_join wrappers: validate parameters and
// dispatch to the launcher. Host-compiled; never includes the kernel code.
#include "ninfer/ops/moe_expert_partition.h"

#include "ninfer/ops/moe_expert_record.h"
#include "ninfer/ops/moe_route.h"
#include "ops/launcher/moe_expert_partition.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

// The helper contract's routing is moe_route's: ten ids per token over 512 experts.
static_assert(kMoeHelperRoutesPerToken == kMoeTopK && kMoeHelperExperts == kMoeExperts &&
              kMoeExpertMaskWords * 32 == kMoeExperts);

namespace {

void require(bool condition, const char* op, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

bool is_tensor(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1) {
    return t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == 1 && t.ne[3] == 1 &&
           t.is_contiguous() && t.data != nullptr &&
           (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data);
    return a_begin < b_begin + b.bytes() && b_begin < a_begin + a.bytes();
}

} // namespace

void moe_expert_partition(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
                          const Tensor& local_experts, Tensor& compact_map, Tensor& compact_count,
                          Tensor& compact_x, Tensor& compact_ids, Tensor& compact_weights,
                          cudaStream_t stream) {
    constexpr const char* kOp   = "moe_expert_partition";
    const std::int32_t tokens   = x.ne[1];
    const std::int32_t capacity = compact_x.ne[1];
    require(tokens >= 1 && tokens <= kMoeHelperMaxTokens, kOp, "T must be in [1,48]");
    require(is_tensor(x, DType::BF16, kMoeHidden, tokens), kOp,
            "x must be aligned contiguous BF16 [2560,T]");
    require(is_tensor(expert_ids, DType::I32, kMoeHelperRoutesPerToken, tokens), kOp,
            "expert_ids must be aligned contiguous I32 [10,T]");
    require(is_tensor(route_weights, DType::FP32, kMoeHelperRoutesPerToken, tokens), kOp,
            "route_weights must be aligned contiguous FP32 [10,T]");
    require(is_tensor(local_experts, DType::I32, kMoeExpertMaskWords, 1), kOp,
            "local_experts must be aligned contiguous I32 [16]");
    require(is_tensor(compact_map, DType::I32, tokens, 1), kOp,
            "compact_map must be aligned contiguous I32 [T]");
    require(is_tensor(compact_count, DType::I32, 1, 1), kOp,
            "compact_count must be aligned contiguous I32 [1]");
    require(capacity >= tokens && is_tensor(compact_x, DType::BF16, kMoeHidden, capacity), kOp,
            "compact_x must be aligned contiguous BF16 [2560,C] with C >= T");
    require(is_tensor(compact_ids, DType::I32, kMoeHelperRoutesPerToken, capacity), kOp,
            "compact_ids must be aligned contiguous I32 [10,C]");
    require(is_tensor(compact_weights, DType::FP32, kMoeHelperRoutesPerToken, capacity), kOp,
            "compact_weights must be aligned contiguous FP32 [10,C]");
    const Tensor* outputs[] = {&compact_map, &compact_count, &compact_x, &compact_ids,
                               &compact_weights};
    for (std::size_t i = 0; i < std::size(outputs); ++i) {
        for (const Tensor* input : {&x, &expert_ids, &route_weights, &local_experts}) {
            require(!overlaps(*outputs[i], *input), kOp, "outputs must be disjoint from inputs");
        }
        for (std::size_t k = i + 1; k < std::size(outputs); ++k) {
            require(!overlaps(*outputs[i], *outputs[k]), kOp, "outputs must be disjoint");
        }
    }
    detail::moe_expert_partition_launch(static_cast<const std::uint16_t*>(x.data),
                                        static_cast<const std::int32_t*>(expert_ids.data),
                                        static_cast<const float*>(route_weights.data),
                                        static_cast<const std::uint32_t*>(local_experts.data),
                                        static_cast<std::int32_t*>(compact_map.data),
                                        static_cast<std::uint32_t*>(compact_count.data),
                                        static_cast<std::uint16_t*>(compact_x.data),
                                        static_cast<std::int32_t*>(compact_ids.data),
                                        static_cast<float*>(compact_weights.data), tokens, stream);
}

void moe_expert_partial_join(const Tensor& partial, const Tensor& compact_map, const Tensor& helper,
                             Tensor& y, cudaStream_t stream) {
    constexpr const char* kOp   = "moe_expert_partial_join";
    const std::int32_t tokens   = partial.ne[1];
    const std::int32_t capacity = helper.ne[1];
    require(tokens >= 1 && tokens <= kMoeHelperMaxTokens, kOp, "T must be in [1,48]");
    require(is_tensor(partial, DType::FP32, kMoeHidden, tokens), kOp,
            "partial must be aligned contiguous FP32 [2560,T]");
    require(is_tensor(compact_map, DType::I32, tokens, 1), kOp,
            "compact_map must be aligned contiguous I32 [T]");
    require(capacity >= tokens && is_tensor(helper, DType::FP32, kMoeHidden, capacity), kOp,
            "helper must be aligned contiguous FP32 [2560,C] with C >= T");
    require(is_tensor(y, DType::BF16, kMoeHidden, tokens), kOp,
            "y must be aligned contiguous BF16 [2560,T]");
    for (const Tensor* input : {&partial, &compact_map, &helper}) {
        require(!overlaps(y, *input), kOp, "y must be disjoint from every input");
    }
    detail::moe_expert_partial_join_launch(static_cast<const float*>(partial.data),
                                           static_cast<const std::int32_t*>(compact_map.data),
                                           static_cast<const float*>(helper.data), capacity,
                                           static_cast<std::uint16_t*>(y.data), tokens, stream);
}

} // namespace ninfer::ops

#include "ninfer/ops/normalized_rope_kv_append.h"

#include "ops/launcher/normalized_rope_kv_append.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& tensor, DType dtype, std::array<std::int32_t, 4> shape,
                    const char* name) {
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous() ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & 3u) != 0u) {
        throw std::invalid_argument("normalized_rope_kv_append: invalid storage for " +
                                    std::string(name));
    }
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (tensor.ne[axis] != shape[axis]) {
            throw std::invalid_argument("normalized_rope_kv_append: invalid shape for " +
                                        std::string(name));
        }
    }
}

void require_disjoint(const std::array<const Tensor*, 8>& tensors) {
    std::array<std::uintptr_t, 8> begin{};
    std::array<std::uintptr_t, 8> end{};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        begin[i]                = reinterpret_cast<std::uintptr_t>(tensors[i]->data);
        const std::size_t bytes = tensors[i]->bytes();
        if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin[i]) {
            throw std::invalid_argument(
                "normalized_rope_kv_append: tensor address range overflows");
        }
        end[i] = begin[i] + bytes;
        for (std::size_t j = 0; j < i; ++j) {
            if (begin[i] < end[j] && begin[j] < end[i]) {
                throw std::invalid_argument("normalized_rope_kv_append: tensor storage overlaps");
            }
        }
    }
}

} // namespace

void normalized_rope_kv_append(const Tensor& k, const Tensor& v, const Tensor& gamma,
                               const Tensor& positions, const Tensor& counts, const Tensor& lanes,
                               float epsilon, float theta,
                               KVCacheAppendPrefixExecutionEnvelope envelope,
                               CyclicKVCacheLayerView cache, cudaStream_t stream) {
    const std::int32_t width = k.ne[2];
    const std::int32_t batch = k.ne[3];
    if (width <= 0 || batch < 1 || batch > 6 || !(epsilon > 0.0F) || !std::isfinite(epsilon) ||
        !(theta > 0.0F) || !std::isfinite(theta) || envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) || envelope.max_count > 2048) {
        throw std::invalid_argument(
            "normalized_rope_kv_append: invalid extent, scalar, or envelope");
    }
    if (cache.head_dim != 128 || cache.num_kv_heads != 8 || cache.capacity != 2048 ||
        cache.padded_capacity < cache.capacity ||
        cache.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        cache.lane_capacity < batch || cache.lane_capacity > 6) {
        throw std::invalid_argument("normalized_rope_kv_append: invalid cyclic cache geometry");
    }
    require_tensor(k, DType::BF16, {128, 8, width, batch}, "k");
    require_tensor(v, DType::BF16, {128, 8, width, batch}, "v");
    require_tensor(gamma, DType::BF16, {128, 1, 1, 1}, "gamma");
    require_tensor(positions, DType::I32, {width, batch, 1, 1}, "positions");
    require_tensor(counts, DType::I32, {batch, 1, 1, 1}, "counts");
    require_tensor(lanes, DType::I32, {batch, 1, 1, 1}, "lanes");
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    require_tensor(cache.k, DType::BF16, {128, padded, 8, cache.lane_capacity}, "cache k");
    require_tensor(cache.v, DType::BF16, {128, padded, 8, cache.lane_capacity}, "cache v");
    require_disjoint({&k, &v, &gamma, &positions, &counts, &lanes, &cache.k, &cache.v});
    detail::normalized_rope_kv_append_launch(k, v, gamma, positions, counts, lanes, epsilon, theta,
                                             envelope, cache, stream);
}

} // namespace ninfer::ops

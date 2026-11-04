#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/nll_from_logits.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/types.h"

#include "core/device.h"
#include "ops/r9700/schedule/scalar_schedule_ops.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* operation) {
    std::int64_t result = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (tensor.ne[dimension] < 0) {
            throw std::invalid_argument(std::string(operation) + ": negative dimension");
        }
        if (tensor.ne[dimension] == 0) { return 0; }
        if (result > std::numeric_limits<std::int64_t>::max() / tensor.ne[dimension]) {
            throw std::overflow_error(std::string(operation) + ": tensor size overflows int64");
        }
        result *= tensor.ne[dimension];
    }
    return result;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_i32_vector(const Tensor& tensor, std::int32_t size, const char* operation,
                        const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != size || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": " + name +
                                    " must be a contiguous I32 vector");
    }
}

void require_i32_matrix(const Tensor& tensor, std::int32_t width, std::int32_t batch,
                        const char* operation, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != width || tensor.ne[1] != batch ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": " + name +
                                    " must be a contiguous I32 matrix");
    }
}

} // namespace

void add_bias(const Tensor& bias, Tensor& x, hipStream_t stream) {
    constexpr const char* operation = "add_bias";
    if (bias.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("add_bias: bias/x must be BF16");
    }
    const std::int64_t elements = checked_numel(x, operation);
    (void)checked_numel(bias, operation);
    if (bias.ne[0] != x.ne[0] || bias.ne[1] != 1 || bias.ne[2] != 1 || bias.ne[3] != 1) {
        throw std::invalid_argument("add_bias: bias must have shape [x.ne[0]]");
    }
    if (elements == 0) { return; }
    if (x.ne[0] <= 0 || !bias.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("add_bias: bias/x must be contiguous");
    }
    if (bias.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("add_bias: bias/x data must be non-null");
    }
    if (overlaps(bias, x)) { throw std::invalid_argument("add_bias: bias must not overlap x"); }
    const auto rows = static_cast<std::size_t>(elements / x.ne[0]);
    HIP_CHECK(r9700::schedule::add_bias_bf16(static_cast<const hip_bfloat16*>(bias.data),
                                             static_cast<hip_bfloat16*>(x.data),
                                             static_cast<std::uint32_t>(x.ne[0]), rows, stream));
}

void nll_from_logits(const Tensor& logits, const Tensor& targets, Tensor& out,
                     std::int32_t valid_rows, hipStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("nll_from_logits: logits must be BF16");
    }
    if (targets.dtype != DType::I32) {
        throw std::invalid_argument("nll_from_logits: targets must be I32");
    }
    if (out.dtype != DType::FP32) {
        throw std::invalid_argument("nll_from_logits: out must be FP32");
    }
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || targets.ne[1] != 1 || targets.ne[2] != 1 ||
        targets.ne[3] != 1 || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("nll_from_logits: expected logits [V,T], targets/out [T]");
    }
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("nll_from_logits: invalid row or column extent");
    }
    if (targets.ne[0] != logits.ne[1] || out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("nll_from_logits: targets/out shape must be [logits.ne[1]]");
    }
    if (!logits.is_contiguous() || !targets.is_contiguous() || !out.is_contiguous() ||
        logits.data == nullptr || targets.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("nll_from_logits: tensors must be contiguous and non-null");
    }
    if (overlaps(logits, targets) || overlaps(logits, out) || overlaps(targets, out)) {
        throw std::invalid_argument("nll_from_logits: tensors must not overlap");
    }
    HIP_CHECK(r9700::schedule::nll_from_logits_bf16(
        static_cast<const hip_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(targets.data), static_cast<float*>(out.data),
        static_cast<std::uint32_t>(logits.ne[0]), static_cast<std::uint32_t>(valid_rows),
        static_cast<std::uint32_t>(logits.ne[1]), stream));
}

void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          hipStream_t stream) {
    constexpr const char* operation = "prepare_masked_block";
    const std::int32_t width        = ids.ne[0];
    const std::int32_t batch        = ids.ne[1];
    if (width < 1 || width > 16 || batch < 1 ||
        batch > static_cast<std::int32_t>(kDFlashMaximumConcurrency) || mask_id < 0) {
        throw std::invalid_argument("prepare_masked_block: requires W=1..16, B=1..8, mask_id>=0");
    }
    require_i32_vector(anchors, batch, operation, "anchors");
    require_i32_vector(lengths, batch, operation, "lengths");
    require_i32_vector(valid_columns, batch, operation, "valid_columns");
    require_i32_matrix(ids, width, batch, operation, "ids");
    require_i32_matrix(positions, width, batch, operation, "positions");
    if (overlaps(ids, positions) || overlaps(anchors, ids) || overlaps(anchors, positions) ||
        overlaps(lengths, ids) || overlaps(lengths, positions) || overlaps(valid_columns, ids) ||
        overlaps(valid_columns, positions)) {
        throw std::invalid_argument("prepare_masked_block: inputs and outputs must not overlap");
    }
    HIP_CHECK(r9700::schedule::prepare_masked_block_i32(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(lengths.data),
        static_cast<const std::int32_t*>(valid_columns.data), mask_id,
        static_cast<std::int32_t*>(ids.data), static_cast<std::int32_t*>(positions.data),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(batch), stream));
}

void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, hipStream_t stream) {
    const std::int32_t width = source.ne[1];
    const std::int32_t batch = destination.ne[2];
    const auto vector_shape  = [batch](const Tensor& tensor) {
        return tensor.dtype == DType::I32 && tensor.ne[0] == batch && tensor.ne[1] == 1 &&
               tensor.ne[2] == 1 && tensor.ne[3] == 1 && tensor.is_contiguous() &&
               tensor.data != nullptr;
    };
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 || source.ne[0] <= 0 ||
        (source.ne[0] % 8) != 0 || width <= 0 || width > 16 || source.ne[2] <= 0 ||
        source.ne[3] != 1 || destination.ne[0] != source.ne[0] || destination.ne[1] != width ||
        batch <= 0 || batch > static_cast<std::int32_t>(kDFlashMaximumConcurrency) ||
        destination.ne[3] != 1 || positions.dtype != DType::I32 || positions.ne[0] != width ||
        positions.ne[1] != batch || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        !vector_shape(lanes) || !vector_shape(starts) || !vector_shape(ends) ||
        !vector_shape(counts)) {
        throw std::invalid_argument("prepare_ragged_prefix: invalid tensor geometry");
    }
    if (source.data == nullptr || destination.data == nullptr || !destination.is_contiguous() ||
        !positions.is_contiguous() || positions.data == nullptr || source.nb[0] != 2 ||
        source.nb[1] < static_cast<std::int64_t>(source.ne[0]) * 2 ||
        source.nb[2] < source.nb[1] * width ||
        (source.nb[1] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        (source.nb[2] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        ((reinterpret_cast<std::uintptr_t>(source.data) |
          reinterpret_cast<std::uintptr_t>(destination.data)) &
         15U) != 0U) {
        throw std::invalid_argument(
            "prepare_ragged_prefix: tensors must have aligned dense-column storage");
    }
    if (overlaps(source, destination) || overlaps(destination, positions) ||
        overlaps(destination, counts) || overlaps(positions, counts)) {
        throw std::invalid_argument("prepare_ragged_prefix: outputs must not overlap inputs");
    }
    HIP_CHECK(r9700::schedule::prepare_ragged_prefix_bf16(
        source.data, static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(starts.data), static_cast<const std::int32_t*>(ends.data),
        destination.data, static_cast<std::int32_t*>(positions.data),
        static_cast<std::int32_t*>(counts.data), static_cast<std::uint32_t>(source.ne[0] / 8),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(batch),
        static_cast<std::size_t>(source.nb[1] / sizeof(uint4)),
        static_cast<std::size_t>(source.nb[2] / sizeof(uint4)), stream));
}

} // namespace ninfer::ops

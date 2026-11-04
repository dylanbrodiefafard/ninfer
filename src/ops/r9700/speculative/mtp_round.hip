// Implements: include/ninfer/ops/mtp_round.h
// Match: fixed K=1..5 MTP round transition for one startup-frozen B=1..16 batch.
#include "ninfer/ops/mtp_round.h"
#include "ninfer/types.h"

#include "core/device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* op, const char* name) {
    if (tensor.dtype != dtype || rows <= 0 || columns <= 0 || tensor.ne[0] != rows ||
        tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
    require_contiguous_nonnull(tensor, op, name);
}

void require_vector(const Tensor& tensor, DType dtype, std::int32_t count, const char* op,
                    const char* name) {
    require_matrix(tensor, dtype, count, 1, op, name);
}

void require_row_pitched_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                                const char* op, const char* name) {
    constexpr std::int64_t element = sizeof(std::int32_t);
    if (rows <= 0 || columns <= 0 || tensor.dtype != DType::I32 || tensor.ne[0] != rows ||
        tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
        tensor.nb[0] != element || tensor.nb[1] < rows * element || tensor.nb[1] % element != 0 ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid row-pitched matrix for " + name);
    }
}

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context) {
    const int row      = static_cast<int>(blockIdx.y);
    const int width    = k + 1;
    int accepted_count = accepted[row];
    accepted_count     = accepted_count < 0 ? 0 : (accepted_count > k ? k : accepted_count);
    for (int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
         column < width; column += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x)) {
        alignment_ids[row * width + column] =
            column < accepted_count ? verify_ids[row * width + column + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int remaining      = remaining_budgets[row] - licensed_counts[row];
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int step = 0; step < steps; ++step) {
            const int offset          = step * ar_step_stride + row;
            const int position        = updated_frontiers[row] + step;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            ar_valid_columns[offset]  = step + 1 < next ? 1 : 0;
        }
    }
}

} // namespace

void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            hipStream_t stream) {
    constexpr const char* op = "mtp_prepare_next_round";
    const std::int32_t width = verify_ids.ne[0];
    const std::int32_t batch = verify_ids.ne[1];
    if (width < 2 || width > 6) {
        throw std::invalid_argument("mtp_prepare_next_round: T must be in [2,6]");
    }
    if (batch < 1 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("mtp_prepare_next_round: B must be in [1,16]");
    }
    if (max_context <= 0) {
        throw std::invalid_argument("mtp_prepare_next_round: max_context must be positive");
    }
    require_matrix(verify_ids, DType::I32, width, batch, op, "verify_ids");
    require_vector(next_anchors, DType::I32, batch, op, "next_anchors");
    require_vector(accepted, DType::I32, batch, op, "accepted");
    require_vector(updated_frontiers, DType::I32, batch, op, "updated_frontiers");
    require_vector(remaining_budgets, DType::I32, batch, op, "remaining_budgets");
    require_vector(licensed_counts, DType::I32, batch, op, "licensed_counts");
    require_vector(rope_deltas, DType::I32, batch, op, "rope_deltas");
    require_matrix(alignment_ids, DType::I32, width, batch, op, "alignment_ids");
    require_vector(next_extents, DType::I32, batch, op, "next_extents");
    const std::int32_t steps = std::max(width - 2, 1);
    require_row_pitched_matrix(ar_positions, batch, steps, op, "ar_positions");
    require_row_pitched_matrix(ar_rope_positions, batch, steps, op, "ar_rope_positions");
    require_row_pitched_matrix(ar_valid_columns, batch, steps, op, "ar_valid_columns");
    if (ar_rope_positions.nb[1] != ar_positions.nb[1] ||
        ar_valid_columns.nb[1] != ar_positions.nb[1]) {
        throw std::invalid_argument(
            "mtp_prepare_next_round: AR outputs must share one step stride");
    }

    constexpr int block_size = 32;
    const int k              = width - 1;
    const int ar_step_stride =
        static_cast<int>(ar_positions.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((width + block_size - 1) / block_size),
                    static_cast<unsigned int>(batch));
    mtp_prepare_next_round_kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(next_anchors.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(updated_frontiers.data),
        static_cast<const std::int32_t*>(remaining_budgets.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(alignment_ids.data),
        static_cast<std::int32_t*>(next_extents.data),
        static_cast<std::int32_t*>(ar_positions.data),
        static_cast<std::int32_t*>(ar_rope_positions.data),
        static_cast<std::int32_t*>(ar_valid_columns.data), k, ar_step_stride, max_context);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops

#include "ninfer/ops/qsa.h"

#include "ops/launcher/qsa.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& tensor, DType dtype, const char* op, const char* name) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be non-null, contiguous, and have the required dtype");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

int require_state(const QsaStateView& state, const char* op) {
    require_tensor(state.k_codes, DType::U8, op, "state.k_codes");
    require_tensor(state.v_codes, DType::U8, op, "state.v_codes");
    require_tensor(state.k_scales, DType::FP8_E4M3FN, op, "state.k_scales");
    require_tensor(state.v_scales, DType::FP8_E4M3FN, op, "state.v_scales");
    require_tensor(state.raw_index_keys, DType::BF16, op, "state.raw_index_keys");
    require_tensor(state.positions, DType::I32, op, "state.positions");
    const int capacity = state.raw_index_keys.ne[1];
    if (capacity <= 0 || capacity > kQsaMaximumTokens) {
        throw std::invalid_argument(std::string(op) + ": state capacity must be in [1,4096]");
    }
    require_shape(state.k_codes, 128, capacity, kQsaKvHeads, op, "state.k_codes");
    require_shape(state.v_codes, 128, capacity, kQsaKvHeads, op, "state.v_codes");
    require_shape(state.k_scales, 16, capacity, kQsaKvHeads, op, "state.k_scales");
    require_shape(state.v_scales, 16, capacity, kQsaKvHeads, op, "state.v_scales");
    require_shape(state.raw_index_keys, kQsaIndexHeadDim, capacity, 1, op,
                  "state.raw_index_keys");
    require_shape(state.positions, 3, capacity, 1, op, "state.positions");
    return capacity;
}

} // namespace

void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                      const Tensor& position_ids, const Tensor& append_ids, QsaStateView state,
                      cudaStream_t stream) {
    constexpr const char* op = "qsa_state_append";
    require_state(state, op);
    require_tensor(k, DType::BF16, op, "k");
    require_tensor(v, DType::BF16, op, "v");
    require_tensor(raw_index_keys, DType::BF16, op, "raw_index_keys");
    require_tensor(position_ids, DType::I32, op, "position_ids");
    require_tensor(append_ids, DType::I32, op, "append_ids");
    const int width = k.ne[2];
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument(std::string(op) + ": width must be in [1,4096]");
    }
    require_shape(k, kQsaHeadDim, kQsaKvHeads, width, op, "k");
    require_shape(v, kQsaHeadDim, kQsaKvHeads, width, op, "v");
    require_shape(raw_index_keys, kQsaIndexHeadDim, width, 1, op, "raw_index_keys");
    require_shape(position_ids, 3, width, 1, op, "position_ids");
    require_shape(append_ids, width, 1, 1, op, "append_ids");
    detail::qsa_state_append_launch(k, v, raw_index_keys, position_ids, append_ids, state, stream);
}

std::size_t qsa_index_select_workspace_bytes(std::int32_t width) {
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument("qsa_index_select_workspace_bytes: width must be in [1,4096]");
    }
    return static_cast<std::size_t>(width) * 3U * 512U * sizeof(float);
}

void qsa_index_select(const Tensor& raw_query, const QsaStateView& state,
                      const Tensor& query_ids, const Tensor& visible_ids,
                      const Tensor& visible_offsets, const Tensor& query_norm_weight,
                      const Tensor& key_norm_weight, Tensor& selected_ids,
                      Tensor& selected_count, Tensor& workspace, cudaStream_t stream) {
    constexpr const char* op = "qsa_index_select";
    require_state(state, op);
    require_tensor(raw_query, DType::BF16, op, "raw_query");
    require_tensor(query_ids, DType::I32, op, "query_ids");
    require_tensor(visible_ids, DType::I32, op, "visible_ids");
    require_tensor(visible_offsets, DType::I32, op, "visible_offsets");
    require_tensor(query_norm_weight, DType::FP32, op, "query_norm_weight");
    require_tensor(key_norm_weight, DType::FP32, op, "key_norm_weight");
    require_tensor(selected_ids, DType::I32, op, "selected_ids");
    require_tensor(selected_count, DType::I32, op, "selected_count");
    require_tensor(workspace, DType::U8, op, "workspace");
    const int width = raw_query.ne[2];
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument(std::string(op) + ": width must be in [1,4096]");
    }
    if (visible_ids.ne[0] <= 0 || visible_ids.ne[0] > kQsaMaximumTokens * width) {
        throw std::invalid_argument(std::string(op) + ": invalid visible-id extent");
    }
    require_shape(raw_query, kQsaIndexHeadDim, kQsaIndexQueryHeads, width, op, "raw_query");
    require_shape(query_ids, width, 1, 1, op, "query_ids");
    require_shape(visible_ids, visible_ids.ne[0], 1, 1, op, "visible_ids");
    require_shape(visible_offsets, width + 1, 1, 1, op, "visible_offsets");
    require_shape(query_norm_weight, kQsaIndexHeadDim, 1, 1, op, "query_norm_weight");
    require_shape(key_norm_weight, kQsaIndexHeadDim, 1, 1, op, "key_norm_weight");
    require_shape(selected_ids, kQsaSelectedCapacity, width, 1, op, "selected_ids");
    require_shape(selected_count, width, 1, 1, op, "selected_count");
    if (workspace.bytes() < qsa_index_select_workspace_bytes(width)) {
        throw std::invalid_argument(std::string(op) + ": workspace is too small");
    }
    if ((reinterpret_cast<std::uintptr_t>(workspace.data) & (alignof(float) - 1U)) != 0U) {
        throw std::invalid_argument(std::string(op) + ": workspace must be four-byte aligned");
    }
    detail::qsa_index_select_launch(raw_query, state, query_ids, visible_ids, visible_offsets,
                                    query_norm_weight, key_norm_weight, selected_ids,
                                    selected_count, workspace, stream);
}

void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const QsaStateView& state, Tensor& out,
                            cudaStream_t stream) {
    constexpr const char* op = "qsa_selected_attention";
    require_state(state, op);
    require_tensor(q, DType::BF16, op, "q");
    require_tensor(selected_ids, DType::I32, op, "selected_ids");
    require_tensor(selected_count, DType::I32, op, "selected_count");
    require_tensor(out, DType::BF16, op, "out");
    const int width = q.ne[2];
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument(std::string(op) + ": width must be in [1,4096]");
    }
    require_shape(q, kQsaHeadDim, kQsaQueryHeads, width, op, "q");
    require_shape(selected_ids, kQsaSelectedCapacity, width, 1, op, "selected_ids");
    require_shape(selected_count, width, 1, 1, op, "selected_count");
    require_shape(out, kQsaHeadDim, kQsaQueryHeads, width, op, "out");
    detail::qsa_selected_attention_launch(q, selected_ids, selected_count, state, out, stream);
}

} // namespace ninfer::ops

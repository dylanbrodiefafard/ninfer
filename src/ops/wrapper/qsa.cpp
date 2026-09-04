#include "ninfer/ops/qsa.h"

#include "ops/launcher/qsa.h"
#include "ops/wrapper/qsa_validation.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::uintptr_t scalar_alignment(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return alignof(std::uint16_t);
    case DType::FP32:
    case DType::I32:
        return alignof(std::uint32_t);
    default:
        return 1;
    }
}

void require_tensor(const Tensor& tensor, DType dtype, const char* op, const char* name,
                    std::uintptr_t alignment = 0) {
    if (alignment == 0) { alignment = scalar_alignment(dtype); }
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1U)) != 0U) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be non-null, contiguous, aligned, and have the "
                                    "required dtype");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

namespace detail {

QsaAddressRange qsa_address_range(const Tensor& tensor, const char* op, const char* name) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto bytes = static_cast<std::size_t>(tensor.bytes());
    if (tensor.data == nullptr || bytes == 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string(op) + ": invalid address range for " + name);
    }
    return {begin, begin + bytes, name};
}

QsaAddressRange qsa_address_range(const Weight& weight, const char* op, const char* name) {
    const auto begin = reinterpret_cast<std::uintptr_t>(weight.payload);
    if (weight.payload == nullptr || weight.payload_bytes == 0 ||
        weight.payload_bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string(op) + ": invalid address range for " + name);
    }
    return {begin, begin + static_cast<std::uintptr_t>(weight.payload_bytes), name};
}

void qsa_require_disjoint(std::initializer_list<QsaAddressRange> ranges, const char* op) {
    for (auto left = ranges.begin(); left != ranges.end(); ++left) {
        for (auto right = left + 1; right != ranges.end(); ++right) {
            if (left->begin < right->end && right->begin < left->end) {
                throw std::invalid_argument(std::string(op) + ": " + left->name +
                                            " overlaps " + right->name);
            }
        }
    }
}

int qsa_validate_state(const QsaStateView& state, const char* op) {
    // Codes are stored through 32-bit writes by qsa_state_append. The remaining alignments match
    // the scalar types used by append, selection, and attention.
    require_tensor(state.k_codes, DType::U8, op, "state.k_codes", alignof(std::uint32_t));
    require_tensor(state.v_codes, DType::U8, op, "state.v_codes", alignof(std::uint32_t));
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
    qsa_require_disjoint({qsa_address_range(state.k_codes, op, "state.k_codes"),
                          qsa_address_range(state.v_codes, op, "state.v_codes"),
                          qsa_address_range(state.k_scales, op, "state.k_scales"),
                          qsa_address_range(state.v_scales, op, "state.v_scales"),
                          qsa_address_range(state.raw_index_keys, op, "state.raw_index_keys"),
                          qsa_address_range(state.positions, op, "state.positions")},
                         op);
    return capacity;
}

} // namespace detail

void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                      const Tensor& position_ids, const Tensor& append_ids, QsaStateView state,
                      cudaStream_t stream) {
    constexpr const char* op = "qsa_state_append";
    detail::qsa_validate_state(state, op);
    // NVFP4 K16 quantization reads each source group through two uint4 loads.
    require_tensor(k, DType::BF16, op, "k", alignof(uint4));
    require_tensor(v, DType::BF16, op, "v", alignof(uint4));
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
    detail::qsa_require_disjoint(
        {detail::qsa_address_range(k, op, "k"), detail::qsa_address_range(v, op, "v"),
         detail::qsa_address_range(raw_index_keys, op, "raw_index_keys"),
         detail::qsa_address_range(position_ids, op, "position_ids"),
         detail::qsa_address_range(append_ids, op, "append_ids"),
         detail::qsa_address_range(state.k_codes, op, "state.k_codes"),
         detail::qsa_address_range(state.v_codes, op, "state.v_codes"),
         detail::qsa_address_range(state.k_scales, op, "state.k_scales"),
         detail::qsa_address_range(state.v_scales, op, "state.v_scales"),
         detail::qsa_address_range(state.raw_index_keys, op, "state.raw_index_keys"),
         detail::qsa_address_range(state.positions, op, "state.positions")},
        op);
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
    detail::qsa_validate_state(state, op);
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
    detail::qsa_require_disjoint(
        {detail::qsa_address_range(raw_query, op, "raw_query"),
         detail::qsa_address_range(query_ids, op, "query_ids"),
         detail::qsa_address_range(visible_ids, op, "visible_ids"),
         detail::qsa_address_range(visible_offsets, op, "visible_offsets"),
         detail::qsa_address_range(query_norm_weight, op, "query_norm_weight"),
         detail::qsa_address_range(key_norm_weight, op, "key_norm_weight"),
         detail::qsa_address_range(state.k_codes, op, "state.k_codes"),
         detail::qsa_address_range(state.v_codes, op, "state.v_codes"),
         detail::qsa_address_range(state.k_scales, op, "state.k_scales"),
         detail::qsa_address_range(state.v_scales, op, "state.v_scales"),
         detail::qsa_address_range(state.raw_index_keys, op, "state.raw_index_keys"),
         detail::qsa_address_range(state.positions, op, "state.positions"),
         detail::qsa_address_range(selected_ids, op, "selected_ids"),
         detail::qsa_address_range(selected_count, op, "selected_count"),
         detail::qsa_address_range(workspace, op, "workspace")},
        op);
    detail::qsa_index_select_launch(raw_query, state, query_ids, visible_ids, visible_offsets,
                                    query_norm_weight, key_norm_weight, selected_ids,
                                    selected_count, workspace, stream);
}

void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const QsaStateView& state, Tensor& out,
                            Tensor& workspace, cudaStream_t stream) {
    constexpr const char* op = "qsa_selected_attention";
    detail::qsa_validate_state(state, op);
    require_tensor(q, DType::BF16, op, "q");
    require_tensor(selected_ids, DType::I32, op, "selected_ids");
    require_tensor(selected_count, DType::I32, op, "selected_count");
    require_tensor(out, DType::BF16, op, "out");
    require_tensor(workspace, DType::U8, op, "workspace");
    const int width = q.ne[2];
    if (width != 1) {
        throw std::invalid_argument(std::string(op) +
                                    ": only the implemented one-token route is supported");
    }
    const int selected_bound = selected_ids.ne[0];
    if (selected_bound <= 0 || selected_bound > kQsaSelectedCapacity) {
        throw std::invalid_argument(std::string(op) + ": selected bound must be in [1,2051]");
    }
    require_shape(q, kQsaHeadDim, kQsaQueryHeads, width, op, "q");
    require_shape(selected_ids, selected_bound, width, 1, op, "selected_ids");
    require_shape(selected_count, width, 1, 1, op, "selected_count");
    require_shape(out, kQsaHeadDim, kQsaQueryHeads, width, op, "out");
    if (workspace.bytes() < qsa_selected_attention_workspace_bytes() ||
        (reinterpret_cast<std::uintptr_t>(workspace.data) & 255U) != 0U) {
        throw std::invalid_argument(std::string(op) + ": workspace is too small or misaligned");
    }
    detail::qsa_require_disjoint(
        {detail::qsa_address_range(q, op, "q"),
         detail::qsa_address_range(selected_ids, op, "selected_ids"),
         detail::qsa_address_range(selected_count, op, "selected_count"),
         detail::qsa_address_range(state.k_codes, op, "state.k_codes"),
         detail::qsa_address_range(state.v_codes, op, "state.v_codes"),
         detail::qsa_address_range(state.k_scales, op, "state.k_scales"),
         detail::qsa_address_range(state.v_scales, op, "state.v_scales"),
         detail::qsa_address_range(state.raw_index_keys, op, "state.raw_index_keys"),
         detail::qsa_address_range(state.positions, op, "state.positions"),
         detail::qsa_address_range(out, op, "out"),
         detail::qsa_address_range(workspace, op, "workspace")},
        op);
    detail::qsa_selected_attention_launch(q, selected_ids, selected_count, state, out, workspace,
                                          stream);
}

std::size_t qsa_selected_attention_workspace_bytes() {
    constexpr std::size_t score_floats =
        ((static_cast<std::size_t>(kQsaQueryHeads) * kQsaSelectedCapacity + 63U) / 64U) * 64U;
    constexpr std::size_t value_tiles =
        (static_cast<std::size_t>(kQsaSelectedCapacity) + 63U) / 64U;
    constexpr std::size_t partial_floats = value_tiles * kQsaQueryHeads * kQsaHeadDim;
    return (score_floats + partial_floats) * sizeof(float);
}

} // namespace ninfer::ops

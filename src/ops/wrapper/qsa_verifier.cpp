#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/qsa_verifier.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Scratch {
    Tensor index_query;
    Tensor index_key;
    Tensor raw_query_gate;
    Tensor raw_key;
    Tensor raw_value;
    Tensor query;
    Tensor key;
    Tensor attention;
    Tensor gated;
    Tensor select_workspace;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator) {
    return {
        allocator.alloc(DType::BF16, {128, 4, 1}),
        allocator.alloc(DType::BF16, {128, 1}),
        allocator.alloc(DType::BF16, {12288}),
        allocator.alloc(DType::BF16, {512}),
        allocator.alloc(DType::BF16, {512}),
        allocator.alloc(DType::BF16, {256, 24, 1}),
        allocator.alloc(DType::BF16, {256, 2, 1}),
        allocator.alloc(DType::BF16, {256, 24, 1}),
        allocator.alloc(DType::BF16, {6144}),
        allocator.alloc(DType::U8,
                        {static_cast<std::int32_t>(qsa_index_select_workspace_bytes(1))}),
    };
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1U)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, const char* name, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("qsa_verifier_token: invalid ") + name);
    }
}

void require_bf16_weight(const Weight& weight, int rows, const char* name) {
    const std::uint64_t expected = static_cast<std::uint64_t>(rows) * 2560U * 2U;
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2560 || weight.shape[0] != rows ||
        weight.shape[1] != 2560 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2560 || weight.group != 0 || weight.group_size != 0 ||
        weight.payload == nullptr || weight.qdata != weight.payload ||
        weight.payload_bytes != expected || weight.qhigh != nullptr || weight.scales != nullptr ||
        !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("qsa_verifier_token: invalid ") + name);
    }
}

void require_q5_weight(const Weight& weight, int rows, int columns, const char* name) {
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(columns / 256) * 176U;
    if (columns % 256 != 0 || weight.qtype != QType::GGML_Q5_K ||
        weight.layout != QuantLayout::GgmlBlockRow || weight.ndim != 2 || weight.n != rows ||
        weight.k != columns || weight.shape[0] != rows || weight.shape[1] != columns ||
        weight.padded_shape[0] != rows || weight.padded_shape[1] != columns ||
        weight.group != 256 || weight.group_size != 256 || weight.payload == nullptr ||
        weight.qdata != weight.payload || weight.payload_bytes != row_bytes * rows ||
        weight.qhigh != nullptr || weight.scales != nullptr || !aligned_to(weight.qdata, 256)) {
        throw std::invalid_argument(std::string("qsa_verifier_token: invalid ") + name);
    }
}

void validate_weights(const QsaVerifierWeights& weights) {
    require_bf16_weight(weights.index_query, 512, "index_query");
    require_bf16_weight(weights.index_key, 128, "index_key");
    require_q5_weight(weights.core_query_gate, 12288, 2560, "core_query_gate");
    require_q5_weight(weights.core_key, 512, 2560, "core_key");
    require_q5_weight(weights.core_value, 512, 2560, "core_value");
    require_q5_weight(weights.output, 2560, 6144, "output");
    require_tensor(weights.index_query_norm, DType::FP32, 128, 1, 1, "index_query_norm");
    require_tensor(weights.index_key_norm, DType::FP32, 128, 1, 1, "index_key_norm");
    require_tensor(weights.core_query_norm, DType::FP32, 256, 1, 1, "core_query_norm");
    require_tensor(weights.core_key_norm, DType::FP32, 256, 1, 1, "core_key_norm");
}

} // namespace

std::size_t qsa_verifier_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout);
    return layout.peak_bytes();
}

void qsa_verifier_token(const Tensor& x, const Tensor& token_id, const Tensor& position,
                        const Tensor& visible_ids, const Tensor& visible_offsets,
                        const QsaVerifierWeights& weights, QsaStateView state,
                        Tensor& selected_ids, Tensor& selected_count, Tensor& out,
                        Tensor& workspace, cudaStream_t stream) {
    require_tensor(x, DType::BF16, 2560, 1, 1, "x");
    require_tensor(token_id, DType::I32, 1, 1, 1, "token_id", 4);
    require_tensor(position, DType::I32, 3, 1, 1, "position", 4);
    require_tensor(visible_ids, DType::I32, visible_ids.ne[0], 1, 1, "visible_ids", 4);
    require_tensor(visible_offsets, DType::I32, 2, 1, 1, "visible_offsets", 4);
    require_tensor(selected_ids, DType::I32, kQsaSelectedCapacity, 1, 1, "selected_ids", 4);
    require_tensor(selected_count, DType::I32, 1, 1, 1, "selected_count", 4);
    require_tensor(out, DType::BF16, 2560, 1, 1, "out");
    require_tensor(workspace, DType::U8, workspace.ne[0], 1, 1, "workspace", 256);
    if (visible_ids.ne[0] <= 0 || visible_ids.ne[0] > kQsaMaximumTokens) {
        throw std::invalid_argument("qsa_verifier_token: visible extent must be in [1,4096]");
    }
    if (workspace.bytes() < qsa_verifier_workspace_bytes()) {
        throw std::invalid_argument("qsa_verifier_token: workspace is too small");
    }
    validate_weights(weights);

    DeviceArena arena(DeviceSpan{workspace.data, workspace.bytes()});
    Scratch scratch = allocate_scratch(arena);
    detail::qsa_bf16_project_launch(x, weights.index_query, scratch.index_query, stream);
    detail::qsa_bf16_project_launch(x, weights.index_key, scratch.index_key, stream);
    ggml_block_linear(x, weights.core_query_gate, scratch.raw_query_gate, stream);
    ggml_block_linear(x, weights.core_key, scratch.raw_key, stream);
    ggml_block_linear(x, weights.core_value, scratch.raw_value, stream);
    detail::qsa_core_norm_rope_launch(scratch.raw_query_gate, scratch.raw_key, position,
                                      weights.core_query_norm, weights.core_key_norm,
                                      scratch.query, scratch.key, stream);
    Tensor append_key = scratch.key.reshape({256, 2, 1});
    Tensor append_value = scratch.raw_value.reshape({256, 2, 1});
    Tensor append_index_key = scratch.index_key.reshape({128, 1});
    Tensor append_position = position.reshape({3, 1});
    qsa_state_append(append_key, append_value, append_index_key, append_position, token_id, state,
                     stream);
    qsa_index_select(scratch.index_query, state, token_id, visible_ids, visible_offsets,
                     weights.index_query_norm, weights.index_key_norm, selected_ids,
                     selected_count, scratch.select_workspace, stream);
    qsa_selected_attention(scratch.query, selected_ids, selected_count, state, scratch.attention,
                           stream);
    detail::qsa_output_gate_launch(scratch.attention, scratch.raw_query_gate, scratch.gated,
                                   stream);
    ggml_block_linear(scratch.gated, weights.output, out, stream);
}

} // namespace ninfer::ops

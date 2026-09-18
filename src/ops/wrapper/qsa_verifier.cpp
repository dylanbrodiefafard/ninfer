#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/qsa_verifier.h"
#include "ops/wrapper/qsa_validation.h"
#include "ops/common/projection.h"
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/fp8/fp8_tensor.h"

#include <algorithm>
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
    Tensor attention_workspace;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, std::int32_t width,QType key_type) {
    return {
        allocator.alloc(DType::BF16, {128, 4, width}),
        allocator.alloc(DType::BF16, {128, width}),
        allocator.alloc(DType::BF16, {12288, width}),
        allocator.alloc(key_type==QType::BF16_CTRL || key_type==QType::FP8_E4M3FN_TENSOR_F32M
            ? DType::FP32 : DType::BF16, {512, width}),
        allocator.alloc(DType::BF16, {512, width}),
        allocator.alloc(DType::BF16, {256, 24, width}),
        allocator.alloc(DType::BF16, {256, 2, width}),
        allocator.alloc(DType::BF16, {256, 24, width}),
        allocator.alloc(DType::BF16, {6144, width}),
        allocator.alloc(DType::U8,
                        {static_cast<std::int32_t>(qsa_index_select_workspace_bytes(width))}),
        allocator.alloc(DType::U8,
                        {static_cast<std::int32_t>(qsa_selected_attention_workspace_bytes())}),
    };
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1U)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, const char* name, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("qsa_verifier: invalid ") + name);
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
        throw std::invalid_argument(std::string("qsa_verifier: invalid ") + name);
    }
}

void require_q5_weight(const Weight& weight, int rows, int columns, const char* name) {
    if(weight.qtype==QType::FP8_E4M3FN_TENSOR_F32M) {
        if(weight.n!=rows || weight.k!=columns) { throw std::invalid_argument("qsa FP8 projection shape"); }
        detail::validate_fp8_tensor_weight(weight,name);return;
    }
    if (detail::is_native_projection(weight.qtype)) {
        detail::validate_native_projection(weight, rows, columns, name);
        return;
    }
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(columns / 256) * 176U;
    if (columns % 256 != 0 || weight.qtype != QType::GGML_Q5_K ||
        weight.layout != QuantLayout::GgmlBlockRow || weight.ndim != 2 || weight.n != rows ||
        weight.k != columns || weight.shape[0] != rows || weight.shape[1] != columns ||
        weight.padded_shape[0] != rows || weight.padded_shape[1] != columns ||
        weight.group != 256 || weight.group_size != 256 || weight.payload == nullptr ||
        weight.qdata != weight.payload || weight.payload_bytes != row_bytes * rows ||
        weight.qhigh != nullptr || weight.scales != nullptr || !aligned_to(weight.qdata, 256)) {
        throw std::invalid_argument(std::string("qsa_verifier: invalid ") + name);
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

std::size_t projection_capacity(QType type,int rows,int columns,int width) {
    if(type==QType::FP8_E4M3FN_TENSOR_F32M) {
        return linear_workspace_capacity_bytes(type,rows,columns,LinearPolicy::A16Only,1,width);
    }
    if(type!=QType::GGML_Q5_K && !detail::is_native_projection(type)) {
        throw std::invalid_argument("qsa: unsupported projection format");
    }
    return detail::projection_workspace_bytes(type,rows,columns,width);
}

void project(const Tensor& x,const Weight& w,Tensor& y,
    WorkspaceArena& arena,cudaStream_t stream) {
    if(w.qtype==QType::FP8_E4M3FN_TENSOR_F32M) { linear(x,w,y,LinearPolicy::A16Only,arena,stream); }
    else { detail::projection(x,w,y,arena,stream); }
}

} // namespace

std::size_t qsa_verifier_workspace_bytes(std::int32_t width, QType query_gate,
                                         QType key, QType value, QType output) {
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument("qsa_verifier_workspace_bytes: width must be in [1,4096]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, width,key);
    const auto bytes = std::max({
        projection_capacity(query_gate,12288,2560,width),
        projection_capacity(key,512,2560,width),
        projection_capacity(value,512,2560,width),
        projection_capacity(output,2560,6144,width)});
    if (bytes) { (void)layout.alloc_bytes(bytes); }
    return layout.peak_bytes();
}

void qsa_verifier(const Tensor& x, const Tensor& append_ids, const Tensor& position,
                  const Tensor& visible_ids, const Tensor& visible_offsets,
                  const QsaVerifierWeights& weights, QsaStateView state,
                  Tensor& selected_ids, Tensor& selected_count, Tensor& out,
                  Tensor& workspace, cudaStream_t stream) {
    const std::int32_t width = x.ne[1];
    if (width <= 0 || width > kQsaMaximumTokens) {
        throw std::invalid_argument("qsa_verifier: width must be in [1,4096]");
    }
    require_tensor(x, DType::BF16, 2560, width, 1, "x");
    require_tensor(append_ids, DType::I32, width, 1, 1, "append_ids", 4);
    require_tensor(position, DType::I32, 3, width, 1, "position", 4);
    require_tensor(visible_ids, DType::I32, visible_ids.ne[0], 1, 1, "visible_ids", 4);
    require_tensor(visible_offsets, DType::I32, width + 1, 1, 1, "visible_offsets", 4);
    require_tensor(selected_ids, DType::I32, kQsaSelectedCapacity, width, 1, "selected_ids", 4);
    require_tensor(selected_count, DType::I32, width, 1, 1, "selected_count", 4);
    require_tensor(out, DType::BF16, 2560, width, 1, "out");
    require_tensor(workspace, DType::U8, workspace.ne[0], 1, 1, "workspace", 256);
    const std::int64_t max_visible =
        static_cast<std::int64_t>(kQsaMaximumTokens) * width;
    if (visible_ids.ne[0] <= 0 || visible_ids.ne[0] > max_visible) {
        throw std::invalid_argument("qsa_verifier: visible extent must be in [1,4096*width]");
    }
    if (workspace.bytes() < qsa_verifier_workspace_bytes(
            width, weights.core_query_gate.qtype, weights.core_key.qtype,
            weights.core_value.qtype, weights.output.qtype)) {
        throw std::invalid_argument("qsa_verifier: workspace is too small");
    }
    validate_weights(weights);
    detail::qsa_validate_state(state, "qsa_verifier");
    detail::qsa_require_disjoint(
        {detail::qsa_address_range(x, "qsa_verifier", "x"),
         detail::qsa_address_range(append_ids, "qsa_verifier", "append_ids"),
         detail::qsa_address_range(position, "qsa_verifier", "position"),
         detail::qsa_address_range(visible_ids, "qsa_verifier", "visible_ids"),
         detail::qsa_address_range(visible_offsets, "qsa_verifier", "visible_offsets"),
         detail::qsa_address_range(weights.index_query, "qsa_verifier", "index_query"),
         detail::qsa_address_range(weights.index_key, "qsa_verifier", "index_key"),
         detail::qsa_address_range(weights.core_query_gate, "qsa_verifier",
                                   "core_query_gate"),
         detail::qsa_address_range(weights.core_key, "qsa_verifier", "core_key"),
         detail::qsa_address_range(weights.core_value, "qsa_verifier", "core_value"),
         detail::qsa_address_range(weights.output, "qsa_verifier", "output_weight"),
         detail::qsa_address_range(weights.index_query_norm, "qsa_verifier",
                                   "index_query_norm"),
         detail::qsa_address_range(weights.index_key_norm, "qsa_verifier",
                                   "index_key_norm"),
         detail::qsa_address_range(weights.core_query_norm, "qsa_verifier",
                                   "core_query_norm"),
         detail::qsa_address_range(weights.core_key_norm, "qsa_verifier",
                                   "core_key_norm"),
         detail::qsa_address_range(state.k, "qsa_verifier", "state.k"),
         detail::qsa_address_range(state.v, "qsa_verifier", "state.v"),
         detail::qsa_scale_address_range(state.k_scales, state.format, "qsa_verifier", "state.k_scales"),
         detail::qsa_scale_address_range(state.v_scales, state.format, "qsa_verifier", "state.v_scales"),
         detail::qsa_address_range(state.raw_index_keys, "qsa_verifier",
                                   "state.raw_index_keys"),
         detail::qsa_address_range(state.positions, "qsa_verifier", "state.positions"),
         detail::qsa_address_range(selected_ids, "qsa_verifier", "selected_ids"),
         detail::qsa_address_range(selected_count, "qsa_verifier", "selected_count"),
         detail::qsa_address_range(out, "qsa_verifier", "out"),
         detail::qsa_address_range(workspace, "qsa_verifier", "workspace")},
        "qsa_verifier");

    DeviceArena arena(DeviceSpan{workspace.data, workspace.bytes()});
    Scratch scratch = allocate_scratch(arena, width,weights.core_key.qtype);
    detail::qsa_bf16_project_launch(x, weights.index_query, scratch.index_query, stream);
    detail::qsa_bf16_project_launch(x, weights.index_key, scratch.index_key, stream);
    project(x,weights.core_query_gate,scratch.raw_query_gate,arena,stream);
    if(weights.core_key.qtype==QType::BF16_CTRL) {
        detail::launch_bf16_f32(x,weights.core_key,scratch.raw_key,stream);
    } else if(weights.core_key.qtype==QType::FP8_E4M3FN_TENSOR_F32M) {
        detail::fp8_tensor_key_f32(x,weights.core_key,scratch.raw_key,stream);
    } else {
        detail::projection(x, weights.core_key, scratch.raw_key, arena, stream);
    }
    project(x,weights.core_value,scratch.raw_value,arena,stream);
    detail::qsa_core_norm_rope_launch(scratch.raw_query_gate, scratch.raw_key, position,
                                      weights.core_query_norm, weights.core_key_norm,
                                      scratch.query, scratch.key, stream);
    Tensor append_key = scratch.key.reshape({256, 2, width});
    Tensor append_value = scratch.raw_value.reshape({256, 2, width});
    Tensor append_index_key = scratch.index_key.reshape({128, width});
    Tensor append_position = position.reshape({3, width});
    qsa_state_append(append_key, append_value, append_index_key, append_position, append_ids, state,
                     stream);
    qsa_index_select(scratch.index_query, state, append_ids, visible_ids, visible_offsets,
                     weights.index_query_norm, weights.index_key_norm, selected_ids,
                     selected_count, scratch.select_workspace, stream);
    qsa_selected_attention(scratch.query, selected_ids, selected_count, state,
                           scratch.attention, scratch.attention_workspace, stream);
    detail::qsa_output_gate_launch(scratch.attention, scratch.raw_query_gate, scratch.gated,
                                   stream);
    project(scratch.gated,weights.output,out,arena,stream);
}

} // namespace ninfer::ops

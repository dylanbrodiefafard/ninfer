#include "ninfer/ops/qwen4_sparse_moe.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/qwen4_sparse_moe.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

struct BlockGeometry {
    std::uint32_t values;
    std::uint32_t bytes;
};

BlockGeometry block_geometry(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q8_0:
        return {32, 34};
    case QType::GGML_Q5_K:
        return {256, 176};
    case QType::GGML_Q6_K:
        return {256, 210};
    case QType::GGML_IQ1_S:
        return {256, 50};
    case QType::GGML_IQ2_XXS:
        return {256, 66};
    case QType::GGML_IQ4_NL:
        return {32, 18};
    default:
        throw std::invalid_argument("qwen4_sparse_moe: unsupported GGML format");
    }
}

std::uint64_t matrix_bytes(QType qtype, std::int32_t rows, std::int32_t columns) {
    const BlockGeometry geometry = block_geometry(qtype);
    if (columns <= 0 || columns % static_cast<std::int32_t>(geometry.values) != 0 || rows <= 0) {
        throw std::invalid_argument("qwen4_sparse_moe: invalid GGML matrix geometry");
    }
    return static_cast<std::uint64_t>(rows) *
           (static_cast<std::uint64_t>(columns) / geometry.values) * geometry.bytes;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1U)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("qwen4_sparse_moe: invalid ") + name);
    }
}

void require_dense_router(const Weight& weight) {
    constexpr std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(kQwen4SparseMoeExperts) * kQwen4SparseMoeHidden * sizeof(float);
    if (weight.qtype != QType::FP32_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.ndim != 2 || weight.n != kQwen4SparseMoeExperts ||
        weight.k != kQwen4SparseMoeHidden || weight.shape[0] != kQwen4SparseMoeExperts ||
        weight.shape[1] != kQwen4SparseMoeHidden || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != kQwen4SparseMoeExperts ||
        weight.padded_shape[1] != kQwen4SparseMoeHidden || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.payload == nullptr ||
        weight.qdata != weight.payload || weight.payload_bytes != expected_bytes ||
        weight.qhigh != nullptr ||
        weight.scales != nullptr || weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument("qwen4_sparse_moe: invalid FP32 router");
    }
}

void require_ggml_weight(const Weight& weight, QType qtype, std::int32_t matrices,
                         std::int32_t rows, std::int32_t columns, const char* name) {
    const BlockGeometry geometry = block_geometry(qtype);
    const std::uint64_t one_matrix = matrix_bytes(qtype, rows, columns);
    const std::uint64_t expected = one_matrix * static_cast<std::uint64_t>(matrices);
    const bool shape_ok = matrices == 1
                              ? weight.ndim == 2 && weight.shape[0] == rows &&
                                    weight.shape[1] == columns && weight.shape[2] == 1
                              : weight.ndim == 3 && weight.shape[0] == matrices &&
                                    weight.shape[1] == rows && weight.shape[2] == columns;
    const bool padded_ok = matrices == 1
                               ? weight.padded_shape[0] == rows &&
                                     weight.padded_shape[1] == columns &&
                                     weight.padded_shape[2] == 1
                               : weight.padded_shape[0] == matrices &&
                                     weight.padded_shape[1] == rows &&
                                     weight.padded_shape[2] == columns;
    if (weight.qtype != qtype || weight.layout != QuantLayout::GgmlBlockRow || !shape_ok ||
        weight.shape[3] != 1 || !padded_ok || weight.padded_shape[3] != 1 || weight.n != rows ||
        weight.k != columns || weight.group_size != geometry.values ||
        weight.group != static_cast<std::int32_t>(geometry.values) || weight.payload == nullptr ||
        weight.qdata != weight.payload || weight.payload_bytes != expected ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.payload, 256)) {
        throw std::invalid_argument(std::string("qwen4_sparse_moe: invalid ") + name);
    }
}

void require_mapped_bank(const Qwen4MappedRoutedGateUp& bank) {
    if (bank.qtype != QType::GGML_IQ1_S && bank.qtype != QType::GGML_IQ2_XXS) {
        throw std::invalid_argument("qwen4_sparse_moe: routed bank must be IQ1_S or IQ2_XXS");
    }
    const std::uint64_t expected =
        matrix_bytes(bank.qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden) *
        kQwen4SparseMoeExperts;
    if (bank.gate.data() == nullptr || bank.up.data() == nullptr || bank.gate.size() != expected ||
        bank.up.size() != expected) {
        throw std::invalid_argument("qwen4_sparse_moe: invalid mapped routed bank spans");
    }
}

struct Scratch {
    Tensor logits;
    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor expert;
    Tensor routed;
    Tensor shared_gate_value;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator) {
    return {
        allocator.alloc(DType::FP32, {kQwen4SparseMoeExperts}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden}),
        allocator.alloc(DType::FP32, {kQwen4SparseMoeHidden}),
        allocator.alloc(DType::FP32, {1}),
    };
}

std::size_t required_workspace() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout);
    return layout.peak_bytes();
}

Weight matrix_view(const void* data, QType qtype, std::int32_t rows, std::int32_t columns) {
    const BlockGeometry geometry = block_geometry(qtype);
    Weight out{};
    out.payload         = data;
    out.payload_bytes   = matrix_bytes(qtype, rows, columns);
    out.qtype           = qtype;
    out.group_size      = geometry.values;
    out.qdata           = data;
    out.n               = rows;
    out.k               = columns;
    out.group           = static_cast<std::int32_t>(geometry.values);
    out.layout          = QuantLayout::GgmlBlockRow;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    std::string name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("qwen4_sparse_moe: empty " + name);
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument("qwen4_sparse_moe: address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

bool overlaps(const AddressRange& left, const AddressRange& right) {
    return left.begin < right.end && right.begin < left.end;
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument("qwen4_sparse_moe: " + ranges[i].name +
                                            " overlaps " + ranges[j].name);
            }
        }
    }
}

} // namespace

std::size_t qwen4_sparse_moe_selected_stage_bytes(QType routed_qtype) {
    if (routed_qtype != QType::GGML_IQ1_S && routed_qtype != QType::GGML_IQ2_XXS) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_selected_stage_bytes: unsupported routed format");
    }
    return static_cast<std::size_t>(2 * kQwen4SparseMoeTopK) *
           matrix_bytes(routed_qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden);
}

std::size_t qwen4_sparse_moe_workspace_capacity_bytes() { return required_workspace(); }

void qwen4_sparse_moe(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                      void* pinned_stage, std::size_t pinned_stage_bytes, Tensor& device_stage,
                      Tensor& selected_ids, Tensor& selected_weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    require_tensor(x, DType::BF16, kQwen4SparseMoeHidden, "x");
    require_tensor(weights.shared_gate, DType::FP32, kQwen4SparseMoeHidden, "shared_gate");
    require_tensor(device_stage, DType::U8, static_cast<std::int32_t>(kQwen4SparseMoeStageBytes),
                   "device_stage");
    require_tensor(selected_ids, DType::I32, kQwen4SparseMoeTopK, "selected_ids");
    require_tensor(selected_weights, DType::FP32, kQwen4SparseMoeTopK, "selected_weights");
    require_tensor(destination, DType::BF16, kQwen4SparseMoeHidden, "destination");
    require_dense_router(weights.router);
    require_mapped_bank(weights.routed_gate_up);
    require_ggml_weight(weights.routed_down, QType::GGML_IQ4_NL, kQwen4SparseMoeExperts,
                        kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate, "routed_down");
    if (weights.shared_gate_proj.qtype != weights.shared_up.qtype ||
        (weights.shared_gate_proj.qtype != QType::GGML_Q5_K &&
         weights.shared_gate_proj.qtype != QType::GGML_Q6_K)) {
        throw std::invalid_argument("qwen4_sparse_moe: shared gate/up formats differ or invalid");
    }
    require_ggml_weight(weights.shared_gate_proj, weights.shared_gate_proj.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden,
                        "shared_gate_proj");
    require_ggml_weight(weights.shared_up, weights.shared_up.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden, "shared_up");
    require_ggml_weight(weights.shared_down, QType::GGML_Q8_0, 1, kQwen4SparseMoeHidden,
                        kQwen4SparseMoeIntermediate, "shared_down");
    if (pinned_stage_bytes != kQwen4SparseMoeStageBytes || pinned_stage == nullptr) {
        throw std::invalid_argument("qwen4_sparse_moe: pinned stage has wrong capacity");
    }
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, pinned_stage);
    if (attribute_status != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (attribute_status != cudaSuccess) { (void)cudaGetLastError(); }
        throw std::invalid_argument("qwen4_sparse_moe: stage is not pinned host memory");
    }
    const std::size_t required = required_workspace();
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("qwen4_sparse_moe: insufficient workspace");
    }

    std::vector<AddressRange> ranges{
        address_range(x.data, x.bytes(), "x"),
        address_range(weights.router.payload, weights.router.payload_bytes, "router"),
        address_range(weights.routed_down.payload, weights.routed_down.payload_bytes,
                      "routed_down"),
        address_range(weights.shared_gate.data, weights.shared_gate.bytes(), "shared_gate"),
        address_range(weights.shared_gate_proj.payload, weights.shared_gate_proj.payload_bytes,
                      "shared_gate_proj"),
        address_range(weights.shared_up.payload, weights.shared_up.payload_bytes, "shared_up"),
        address_range(weights.shared_down.payload, weights.shared_down.payload_bytes,
                      "shared_down"),
        address_range(device_stage.data, device_stage.bytes(), "device_stage"),
        address_range(selected_ids.data, selected_ids.bytes(), "selected_ids"),
        address_range(selected_weights.data, selected_weights.bytes(), "selected_weights"),
        address_range(destination.data, destination.bytes(), "destination"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    require_disjoint(ranges);
    require_disjoint({address_range(weights.routed_gate_up.gate.data(),
                                    weights.routed_gate_up.gate.size(), "mapped_gate"),
                      address_range(weights.routed_gate_up.up.data(),
                                    weights.routed_gate_up.up.size(), "mapped_up"),
                      address_range(pinned_stage, pinned_stage_bytes, "pinned_stage")});

    auto scope = workspace.scope();
    Scratch scratch = allocate_scratch(workspace);
    detail::qwen4_sparse_moe_route_launch(x, weights.router, scratch.logits, selected_ids,
                                          selected_weights, stream);

    CUDA_CHECK(cudaMemcpyAsync(pinned_stage, selected_ids.data,
                               kQwen4SparseMoeTopK * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::array<std::int32_t, kQwen4SparseMoeTopK> host_ids{};
    std::memcpy(host_ids.data(), pinned_stage, sizeof(host_ids));

    const std::size_t one_matrix = static_cast<std::size_t>(matrix_bytes(
        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden));
    auto* stage_bytes = static_cast<std::byte*>(pinned_stage);
    for (std::int32_t rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
        const std::int32_t expert = host_ids[rank];
        if (expert < 0 || expert >= kQwen4SparseMoeExperts) {
            throw std::runtime_error("qwen4_sparse_moe: GPU route produced an invalid expert id");
        }
        for (std::int32_t prior = 0; prior < rank; ++prior) {
            if (host_ids[prior] == expert) {
                throw std::runtime_error("qwen4_sparse_moe: GPU route produced a duplicate id");
            }
        }
        std::memcpy(stage_bytes + static_cast<std::size_t>(2 * rank) * one_matrix,
                    weights.routed_gate_up.gate.data() +
                        static_cast<std::size_t>(expert) * one_matrix,
                    one_matrix);
        std::memcpy(stage_bytes + static_cast<std::size_t>(2 * rank + 1) * one_matrix,
                    weights.routed_gate_up.up.data() +
                        static_cast<std::size_t>(expert) * one_matrix,
                    one_matrix);
    }
    const std::size_t copied_bytes = qwen4_sparse_moe_selected_stage_bytes(
        weights.routed_gate_up.qtype);
    CUDA_CHECK(cudaMemcpyAsync(device_stage.data, pinned_stage, copied_bytes,
                               cudaMemcpyHostToDevice, stream));

    detail::qwen4_sparse_moe_zero_routed_launch(scratch.routed, stream);
    const auto* device_bytes = static_cast<const std::byte*>(device_stage.data);
    const auto* routed_down = static_cast<const std::byte*>(weights.routed_down.payload);
    const std::size_t routed_down_matrix = static_cast<std::size_t>(matrix_bytes(
        QType::GGML_IQ4_NL, kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate));
    for (std::int32_t rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
        const Weight gate =
            matrix_view(device_bytes + static_cast<std::size_t>(2 * rank) * one_matrix,
                        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate,
                        kQwen4SparseMoeHidden);
        const Weight up =
            matrix_view(device_bytes + static_cast<std::size_t>(2 * rank + 1) * one_matrix,
                        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate,
                        kQwen4SparseMoeHidden);
        const Weight down = matrix_view(
            routed_down + static_cast<std::size_t>(host_ids[rank]) * routed_down_matrix,
            QType::GGML_IQ4_NL, kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate);
        ggml_block_linear(x, gate, scratch.gate, stream);
        ggml_block_linear(x, up, scratch.up, stream);
        detail::qwen4_sparse_moe_swiglu_launch(scratch.gate, scratch.up, scratch.activated,
                                               stream);
        ggml_block_linear(scratch.activated, down, scratch.expert, stream);
        detail::qwen4_sparse_moe_accumulate_launch(scratch.expert, selected_weights, rank,
                                                   scratch.routed, stream);
    }

    detail::qwen4_sparse_moe_shared_gate_launch(x, weights.shared_gate,
                                                scratch.shared_gate_value, stream);
    ggml_block_linear(x, weights.shared_gate_proj, scratch.gate, stream);
    ggml_block_linear(x, weights.shared_up, scratch.up, stream);
    detail::qwen4_sparse_moe_swiglu_launch(scratch.gate, scratch.up, scratch.activated, stream);
    ggml_block_linear(scratch.activated, weights.shared_down, scratch.expert, stream);
    detail::qwen4_sparse_moe_finish_launch(scratch.routed, scratch.expert,
                                           scratch.shared_gate_value, destination, stream);
}

} // namespace ninfer::ops

#include "ninfer/ops/qwen4_sparse_moe.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/qwen4_sparse_moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kResidentGroupedMinWidth = 256;
// Start consuming after half a slot; this balances earlier first work against overlap of the
// following full-slot transfer on the mapped-host RTX 5090 profile.
constexpr std::size_t kPrefillBootstrapExperts = 16;

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

bool has_workspace_capacity(const WorkspaceArena& workspace, std::size_t required) {
    constexpr std::size_t alignment = 256;
    const std::size_t used = workspace.used();
    if (workspace.base() == nullptr || used > workspace.capacity()) { return false; }
    const std::size_t remainder =
        (reinterpret_cast<std::uintptr_t>(workspace.base()) + used) & (alignment - 1U);
    const std::size_t padding = remainder == 0 ? 0 : alignment - remainder;
    return padding <= workspace.capacity() - used &&
           required <= workspace.capacity() - used - padding;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("qwen4_sparse_moe: invalid ") + name);
    }
}

void require_matrix_tensor(const Tensor& tensor, DType dtype, std::int32_t n0,
                           std::int32_t width, const char* name,
                           const char* operation = "qwen4_sparse_moe_prefill") {
    if (width <= 0 || width > kQwen4SparseMoePrefillMaxWidth || tensor.dtype != dtype ||
        tensor.ne[0] != n0 || tensor.ne[1] != width || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid " + name);
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
    Tensor shared;
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
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden}),
        allocator.alloc(DType::FP32, {kQwen4SparseMoeHidden}),
        allocator.alloc(DType::FP32, {1}),
    };
}

struct ResidentScratch {
    Tensor logits;
    Tensor shared_activated;
    Tensor shared;
    Tensor shared_gate_value;
    Tensor routed_activated;
};

struct ResidentWideScratch {
    Tensor logits;
    Tensor shared_gate_value;
    Tensor shared_gate_projection;
    Tensor shared_up;
    Tensor shared;
    Tensor routed_gate;
    Tensor routed_up;
    Tensor rank_results;
    Tensor expert_counts;
    Tensor expert_offsets;
    Tensor expert_cursors;
    Tensor occurrence_slots;
};

struct PrefillScratch {
    Tensor logits;
    Tensor shared_gate_value;
    Tensor shared_gate_projection;
    Tensor shared_up;
    Tensor shared_activated;
    Tensor shared;
    Tensor gathered;
    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor expert;
    Tensor rank_results;
};

template <class Allocator>
PrefillScratch allocate_prefill_scratch(Allocator& allocator, std::int32_t width) {
    const std::int32_t occurrences = kQwen4SparseMoeTopK * width;
    return {
        allocator.alloc(DType::FP32, {kQwen4SparseMoeExperts, width}),
        allocator.alloc(DType::FP32, {width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden, kQwen4SparseMoeTopK, width}),
    };
}

template <class Allocator>
ResidentScratch allocate_resident_scratch(Allocator& allocator) {
    return {
        allocator.alloc(DType::FP32, {kQwen4SparseMoeExperts}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden}),
        allocator.alloc(DType::FP32, {1}),
        allocator.alloc(DType::BF16,
                        {kQwen4SparseMoeTopK * kQwen4SparseMoeIntermediate}),
    };
}

template <class Allocator>
ResidentWideScratch allocate_resident_wide_scratch(Allocator& allocator,
                                                   std::int32_t width) {
    const std::int32_t occurrences = kQwen4SparseMoeTopK * width;
    return {
        allocator.alloc(DType::FP32, {kQwen4SparseMoeExperts, width}),
        allocator.alloc(DType::FP32, {width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeHidden, width}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, occurrences}),
        allocator.alloc(DType::BF16, {kQwen4SparseMoeIntermediate, occurrences}),
        allocator.alloc(DType::BF16,
                        {kQwen4SparseMoeHidden, kQwen4SparseMoeTopK, width}),
        allocator.alloc(DType::I32, {kQwen4SparseMoeExperts}),
        allocator.alloc(DType::I32, {kQwen4SparseMoeExperts}),
        allocator.alloc(DType::I32, {kQwen4SparseMoeExperts}),
        allocator.alloc(DType::I32, {occurrences}),
    };
}

std::size_t required_workspace() {
    WorkspaceLayoutBuilder staged_layout;
    (void)allocate_scratch(staged_layout);
    WorkspaceLayoutBuilder resident_layout;
    (void)allocate_resident_scratch(resident_layout);
    return staged_layout.peak_bytes() > resident_layout.peak_bytes()
               ? staged_layout.peak_bytes()
               : resident_layout.peak_bytes();
}

std::size_t required_prefill_workspace(std::int32_t width) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_prefill_scratch(layout, width);
    return layout.peak_bytes();
}


std::size_t required_resident_workspace(std::int32_t width) {
    WorkspaceLayoutBuilder layout;
    if (width == 1) {
        (void)allocate_resident_scratch(layout);
    } else if (width < kResidentGroupedMinWidth) {
        // The scalar implementation reuses one token's scratch serially.
        (void)allocate_resident_scratch(layout);
    } else {
        (void)allocate_resident_wide_scratch(layout, width);
    }
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
    const char* name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, const char* name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument(std::string("qwen4_sparse_moe: empty ") + name);
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument("qwen4_sparse_moe: address range overflows");
    }
    return {begin, begin + bytes, name};
}

bool overlaps(const AddressRange& left, const AddressRange& right) {
    return left.begin < right.end && right.begin < left.end;
}

void require_disjoint(std::span<const AddressRange> ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument(std::string("qwen4_sparse_moe: ") +
                                            ranges[i].name + " overlaps " + ranges[j].name);
            }
        }
    }
}

} // namespace

std::size_t qwen4_sparse_moe_rank_stage_bytes(QType routed_qtype) {
    if (routed_qtype != QType::GGML_IQ1_S && routed_qtype != QType::GGML_IQ2_XXS) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_rank_stage_bytes: unsupported routed format");
    }
    return 2U *
           matrix_bytes(routed_qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden);
}

std::size_t qwen4_sparse_moe_workspace_capacity_bytes() { return required_workspace(); }

std::size_t qwen4_sparse_moe_prefill_workspace_capacity_bytes(std::int32_t width) {
    if (width <= 0 || width > kQwen4SparseMoePrefillMaxWidth) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill_workspace_capacity_bytes: width must be in [1,4096]");
    }
    return required_prefill_workspace(width);
}

std::size_t qwen4_sparse_moe_resident_workspace_capacity_bytes(std::int32_t width) {
    if (width <= 0 || width > kQwen4SparseMoePrefillMaxWidth) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_resident_workspace_capacity_bytes: width must be in [1,4096]");
    }
    return required_resident_workspace(width);
}

void qwen4_sparse_moe(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                      Qwen4SparseMoePipeline& pipeline,
                      Tensor& selected_ids, Tensor& selected_weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    require_tensor(x, DType::BF16, kQwen4SparseMoeHidden, "x");
    require_tensor(weights.shared_gate, DType::FP32, kQwen4SparseMoeHidden, "shared_gate");
    require_tensor(pipeline.device_stage, DType::U8,
                   static_cast<std::int32_t>(kQwen4SparseMoePipelineStageBytes),
                   "pipeline device_stage");
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
    if (pipeline.pinned_stage_bytes != kQwen4SparseMoePipelineStageBytes ||
        pipeline.pinned_stage == nullptr) {
        throw std::invalid_argument("qwen4_sparse_moe: pipeline pinned stage has wrong capacity");
    }
    if (pipeline.transfer_stream == nullptr || pipeline.transfer_stream == stream) {
        throw std::invalid_argument("qwen4_sparse_moe: transfer stream must be distinct");
    }
    if (pipeline.compute_stream != stream) {
        throw std::invalid_argument("qwen4_sparse_moe: pipeline compute stream changed");
    }
    std::array<cudaEvent_t, 2 + 2 * kQwen4SparseMoePipelineSlots> events{
        pipeline.route_ready,
        pipeline.ids_ready,
        pipeline.transfer_ready[0],
        pipeline.transfer_ready[1],
        pipeline.consumer_complete[0],
        pipeline.consumer_complete[1],
    };
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i] == nullptr) {
            throw std::invalid_argument("qwen4_sparse_moe: pipeline event is null");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (events[i] == events[j]) {
                throw std::invalid_argument("qwen4_sparse_moe: pipeline events must be distinct");
            }
        }
    }
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status =
        cudaPointerGetAttributes(&attributes, pipeline.pinned_stage);
    if (attribute_status != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (attribute_status != cudaSuccess) { (void)cudaGetLastError(); }
        throw std::invalid_argument("qwen4_sparse_moe: stage is not pinned host memory");
    }
    const std::size_t required = required_workspace();
    if (!has_workspace_capacity(workspace, required)) {
        throw std::invalid_argument("qwen4_sparse_moe: insufficient workspace");
    }

    const std::array<AddressRange, 12> ranges{
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
        address_range(pipeline.device_stage.data, pipeline.device_stage.bytes(), "device_stage"),
        address_range(selected_ids.data, selected_ids.bytes(), "selected_ids"),
        address_range(selected_weights.data, selected_weights.bytes(), "selected_weights"),
        address_range(destination.data, destination.bytes(), "destination"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    require_disjoint(ranges);
    const std::array<AddressRange, 3> host_ranges{
        address_range(weights.routed_gate_up.gate.data(),
                      weights.routed_gate_up.gate.size(), "mapped_gate"),
        address_range(weights.routed_gate_up.up.data(), weights.routed_gate_up.up.size(),
                      "mapped_up"),
        address_range(pipeline.pinned_stage, pipeline.pinned_stage_bytes, "pinned_stage")};
    require_disjoint(host_ranges);

    auto scope = workspace.scope();
    Scratch scratch = allocate_scratch(workspace);
    detail::qwen4_sparse_moe_route_launch(x, weights.router, scratch.logits, selected_ids,
                                          selected_weights, stream);

    CUDA_CHECK(cudaEventRecord(pipeline.route_ready, stream));
    CUDA_CHECK(cudaStreamWaitEvent(pipeline.transfer_stream, pipeline.route_ready, 0));
    CUDA_CHECK(cudaMemcpyAsync(pipeline.pinned_stage, selected_ids.data,
                               kQwen4SparseMoeTopK * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, pipeline.transfer_stream));
    CUDA_CHECK(cudaEventRecord(pipeline.ids_ready, pipeline.transfer_stream));

    // This branch is independent of the selected routed experts. Queue it while the route ids
    // cross to the host and preserve its result in dedicated scratch before routed scratch reuse.
    detail::qwen4_sparse_moe_zero_routed_launch(scratch.routed, stream);
    detail::qwen4_sparse_moe_shared_gate_launch(x, weights.shared_gate,
                                                scratch.shared_gate_value, stream);
    ggml_block_linear(x, weights.shared_gate_proj, scratch.gate, stream);
    ggml_block_linear(x, weights.shared_up, scratch.up, stream);
    detail::qwen4_sparse_moe_swiglu_launch(scratch.gate, scratch.up, scratch.activated, stream);
    ggml_block_linear(scratch.activated, weights.shared_down, scratch.shared, stream);

    CUDA_CHECK(cudaEventSynchronize(pipeline.ids_ready));
    std::array<std::int32_t, kQwen4SparseMoeTopK> host_ids{};
    std::memcpy(host_ids.data(), pipeline.pinned_stage, sizeof(host_ids));

    const std::size_t one_matrix = static_cast<std::size_t>(matrix_bytes(
        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden));
    const std::size_t rank_bytes = qwen4_sparse_moe_rank_stage_bytes(
        weights.routed_gate_up.qtype);
    auto* pinned_bytes = static_cast<std::byte*>(pipeline.pinned_stage);
    auto* device_bytes = static_cast<std::byte*>(pipeline.device_stage.data);
    const auto* routed_down = static_cast<const std::byte*>(weights.routed_down.payload);
    const std::size_t routed_down_matrix = static_cast<std::size_t>(matrix_bytes(
        QType::GGML_IQ4_NL, kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate));
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
        const std::int32_t slot = rank % kQwen4SparseMoePipelineSlots;
        auto* pinned_slot = pinned_bytes + static_cast<std::size_t>(slot) *
                                               kQwen4SparseMoeRankStageCapacityBytes;
        auto* device_slot = device_bytes + static_cast<std::size_t>(slot) *
                                               kQwen4SparseMoeRankStageCapacityBytes;
        if (rank >= kQwen4SparseMoePipelineSlots) {
            // transfer_ready protects the host slot from overwrite; consumer_complete protects
            // the paired device slot. Neither wait changes routed rank order on compute stream.
            CUDA_CHECK(cudaEventSynchronize(pipeline.transfer_ready[slot]));
            CUDA_CHECK(cudaStreamWaitEvent(pipeline.transfer_stream,
                                           pipeline.consumer_complete[slot], 0));
        }
        std::memcpy(pinned_slot,
                    weights.routed_gate_up.gate.data() +
                        static_cast<std::size_t>(expert) * one_matrix,
                    one_matrix);
        std::memcpy(pinned_slot + one_matrix,
                    weights.routed_gate_up.up.data() +
                        static_cast<std::size_t>(expert) * one_matrix,
                    one_matrix);
        CUDA_CHECK(cudaMemcpyAsync(device_slot, pinned_slot, rank_bytes,
                                   cudaMemcpyHostToDevice, pipeline.transfer_stream));
        CUDA_CHECK(cudaEventRecord(pipeline.transfer_ready[slot], pipeline.transfer_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline.transfer_ready[slot], 0));
        const Weight gate =
            matrix_view(device_slot,
                        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate,
                        kQwen4SparseMoeHidden);
        const Weight up =
            matrix_view(device_slot + one_matrix,
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
        CUDA_CHECK(cudaEventRecord(pipeline.consumer_complete[slot], stream));
    }

    detail::qwen4_sparse_moe_finish_launch(scratch.routed, scratch.shared,
                                           scratch.shared_gate_value, destination, stream);
}

void qwen4_sparse_moe_prefill(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                              Qwen4SparseMoePrefillPipeline& pipeline,
                              Tensor& selected_ids, Tensor& selected_weights,
                              Tensor& destination, WorkspaceArena& workspace,
                              cudaStream_t stream) {
    const std::int32_t width = x.ne[1];
    require_matrix_tensor(x, DType::BF16, kQwen4SparseMoeHidden, width, "x");
    require_tensor(weights.shared_gate, DType::FP32, kQwen4SparseMoeHidden, "shared_gate");
    require_matrix_tensor(selected_ids, DType::I32, kQwen4SparseMoeTopK, width,
                          "selected_ids");
    require_matrix_tensor(selected_weights, DType::FP32, kQwen4SparseMoeTopK, width,
                          "selected_weights");
    require_matrix_tensor(destination, DType::BF16, kQwen4SparseMoeHidden, width,
                          "destination");
    require_tensor(pipeline.device_stage, DType::U8,
                   static_cast<std::int32_t>(kQwen4SparseMoePrefillPipelineStageBytes),
                   "prefill pipeline device_stage");
    require_dense_router(weights.router);
    require_mapped_bank(weights.routed_gate_up);
    require_ggml_weight(weights.routed_down, QType::GGML_IQ4_NL, kQwen4SparseMoeExperts,
                        kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate, "routed_down");
    if (weights.shared_gate_proj.qtype != weights.shared_up.qtype ||
        (weights.shared_gate_proj.qtype != QType::GGML_Q5_K &&
         weights.shared_gate_proj.qtype != QType::GGML_Q6_K)) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: shared gate/up formats differ or invalid");
    }
    require_ggml_weight(weights.shared_gate_proj, weights.shared_gate_proj.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden,
                        "shared_gate_proj");
    require_ggml_weight(weights.shared_up, weights.shared_up.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden, "shared_up");
    require_ggml_weight(weights.shared_down, QType::GGML_Q8_0, 1,
                        kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate, "shared_down");
    if (pipeline.pinned_stage == nullptr ||
        pipeline.pinned_stage_bytes != kQwen4SparseMoePrefillPipelineStageBytes) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: pipeline pinned stage has wrong capacity");
    }
    if (!aligned_to(pipeline.host_scratch, alignof(std::int32_t)) ||
        pipeline.host_scratch_bytes != kQwen4SparseMoePrefillHostScratchBytes) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: pipeline host scratch has wrong alignment or capacity");
    }
    if (pipeline.transfer_stream == nullptr || pipeline.transfer_stream == stream) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: transfer stream must be distinct");
    }
    if (pipeline.compute_stream != stream) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: pipeline compute stream changed");
    }
    std::array<cudaEvent_t, 2 + 2 * kQwen4SparseMoePipelineSlots> events{
        pipeline.route_ready,
        pipeline.ids_ready,
        pipeline.transfer_ready[0],
        pipeline.transfer_ready[1],
        pipeline.consumer_complete[0],
        pipeline.consumer_complete[1],
    };
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i] == nullptr) {
            throw std::invalid_argument("qwen4_sparse_moe_prefill: pipeline event is null");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (events[i] == events[j]) {
                throw std::invalid_argument(
                    "qwen4_sparse_moe_prefill: pipeline events must be distinct");
            }
        }
    }
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status =
        cudaPointerGetAttributes(&attributes, pipeline.pinned_stage);
    if (attribute_status != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (attribute_status != cudaSuccess) { (void)cudaGetLastError(); }
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: stage is not pinned host memory");
    }
    const cudaError_t scratch_attribute_status =
        cudaPointerGetAttributes(&attributes, pipeline.host_scratch);
    if (scratch_attribute_status != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (scratch_attribute_status != cudaSuccess) { (void)cudaGetLastError(); }
        throw std::invalid_argument(
            "qwen4_sparse_moe_prefill: host scratch is not pinned host memory");
    }
    const std::size_t required = required_prefill_workspace(width);
    if (!has_workspace_capacity(workspace, required)) {
        throw std::invalid_argument("qwen4_sparse_moe_prefill: insufficient workspace");
    }

    const std::array<AddressRange, 12> ranges{
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
        address_range(pipeline.device_stage.data, pipeline.device_stage.bytes(), "device_stage"),
        address_range(selected_ids.data, selected_ids.bytes(), "selected_ids"),
        address_range(selected_weights.data, selected_weights.bytes(), "selected_weights"),
        address_range(destination.data, destination.bytes(), "destination"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    require_disjoint(ranges);
    const std::array<AddressRange, 4> host_ranges{
        address_range(weights.routed_gate_up.gate.data(),
                      weights.routed_gate_up.gate.size(), "mapped_gate"),
        address_range(weights.routed_gate_up.up.data(), weights.routed_gate_up.up.size(),
                      "mapped_up"),
        address_range(pipeline.pinned_stage, pipeline.pinned_stage_bytes, "pinned_stage"),
        address_range(pipeline.host_scratch, pipeline.host_scratch_bytes, "host_scratch")};
    require_disjoint(host_ranges);

    const std::size_t occurrence_count =
        static_cast<std::size_t>(kQwen4SparseMoeTopK) * width;
    auto* host_ids = static_cast<std::int32_t*>(pipeline.host_scratch);
    auto* occurrence_slots =
        host_ids + kQwen4SparseMoeTopK * kQwen4SparseMoePrefillMaxWidth;
    auto* unique_experts =
        occurrence_slots + kQwen4SparseMoeTopK * kQwen4SparseMoePrefillMaxWidth;
    auto* expert_counts = unique_experts + kQwen4SparseMoeExperts;
    auto* expert_offsets = expert_counts + kQwen4SparseMoeExperts;
    auto* expert_cursor = expert_offsets + kQwen4SparseMoeExperts + 1;
    std::fill_n(expert_counts, kQwen4SparseMoeExperts, 0);
    std::fill_n(expert_offsets, kQwen4SparseMoeExperts + 1, 0);
    std::fill_n(expert_cursor, kQwen4SparseMoeExperts, 0);
    std::size_t unique_count = 0;

    auto scope = workspace.scope();
    PrefillScratch scratch = allocate_prefill_scratch(workspace, width);
    detail::qwen4_sparse_moe_prefill_route_launch(
        x, weights.router, scratch.logits, selected_ids, selected_weights, stream);

    auto* pinned_bytes = static_cast<std::byte*>(pipeline.pinned_stage);
    auto* device_bytes = static_cast<std::byte*>(pipeline.device_stage.data);
    CUDA_CHECK(cudaEventRecord(pipeline.route_ready, stream));
    CUDA_CHECK(cudaStreamWaitEvent(pipeline.transfer_stream, pipeline.route_ready, 0));
    CUDA_CHECK(cudaMemcpyAsync(host_ids, selected_ids.data,
                               occurrence_count * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, pipeline.transfer_stream));
    CUDA_CHECK(cudaEventRecord(pipeline.ids_ready, pipeline.transfer_stream));

    detail::qwen4_sparse_moe_shared_gate_launch(x, weights.shared_gate,
                                                scratch.shared_gate_value, stream);
    ggml_block_linear(x, weights.shared_gate_proj, scratch.shared_gate_projection, stream);
    ggml_block_linear(x, weights.shared_up, scratch.shared_up, stream);
    detail::qwen4_sparse_moe_swiglu_launch(scratch.shared_gate_projection, scratch.shared_up,
                                           scratch.shared_activated, stream);
    ggml_block_linear(scratch.shared_activated, weights.shared_down, scratch.shared, stream);

    CUDA_CHECK(cudaEventSynchronize(pipeline.ids_ready));
    for (std::int32_t token = 0; token < width; ++token) {
        for (std::int32_t rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
            const std::int32_t slot = rank + kQwen4SparseMoeTopK * token;
            const std::int32_t expert = host_ids[slot];
            if (expert < 0 || expert >= kQwen4SparseMoeExperts) {
                throw std::runtime_error(
                    "qwen4_sparse_moe_prefill: GPU route produced an invalid expert id");
            }
            for (std::int32_t prior = 0; prior < rank; ++prior) {
                if (host_ids[prior + kQwen4SparseMoeTopK * token] == expert) {
                    throw std::runtime_error(
                        "qwen4_sparse_moe_prefill: GPU route produced a duplicate id");
                }
            }
            ++expert_counts[expert];
        }
    }
    for (std::int32_t expert = 0; expert < kQwen4SparseMoeExperts; ++expert) {
        expert_offsets[expert + 1] = expert_offsets[expert] + expert_counts[expert];
        expert_cursor[expert] = expert_offsets[expert];
        if (expert_counts[expert] != 0) { unique_experts[unique_count++] = expert; }
    }
    for (std::int32_t token = 0; token < width; ++token) {
        for (std::int32_t rank = 0; rank < kQwen4SparseMoeTopK; ++rank) {
            const std::int32_t rank_token = rank + kQwen4SparseMoeTopK * token;
            const std::int32_t expert = host_ids[rank_token];
            occurrence_slots[expert_cursor[expert]++] = rank_token;
        }
    }

    const std::size_t one_matrix = static_cast<std::size_t>(matrix_bytes(
        weights.routed_gate_up.qtype, kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden));
    const std::size_t rank_bytes = qwen4_sparse_moe_rank_stage_bytes(
        weights.routed_gate_up.qtype);
    const auto* routed_down = static_cast<const std::byte*>(weights.routed_down.payload);
    const std::size_t routed_down_matrix = static_cast<std::size_t>(matrix_bytes(
        QType::GGML_IQ4_NL, kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate));
    const std::size_t groups =
        unique_count <= kQwen4SparseMoePrefillGroupExperts
            ? 1
            : 1 + (unique_count - kPrefillBootstrapExperts +
                   kQwen4SparseMoePrefillGroupExperts - 1) /
                      kQwen4SparseMoePrefillGroupExperts;
    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t group_begin =
            group == 0
                ? 0
                : kPrefillBootstrapExperts +
                      (group - 1) * kQwen4SparseMoePrefillGroupExperts;
        const std::size_t group_capacity =
            group == 0 && groups != 1 ? kPrefillBootstrapExperts
                       : static_cast<std::size_t>(kQwen4SparseMoePrefillGroupExperts);
        const std::size_t group_end = std::min(
            group_begin + group_capacity, unique_count);
        const std::int32_t slot = static_cast<std::int32_t>(group %
                                                            kQwen4SparseMoePipelineSlots);
        auto* pinned_slot = pinned_bytes +
                            static_cast<std::size_t>(slot) *
                                kQwen4SparseMoePrefillSlotCapacityBytes;
        auto* device_slot = device_bytes +
                            static_cast<std::size_t>(slot) *
                                kQwen4SparseMoePrefillSlotCapacityBytes;
        if (group >= kQwen4SparseMoePipelineSlots) {
            CUDA_CHECK(cudaEventSynchronize(pipeline.transfer_ready[slot]));
            CUDA_CHECK(cudaStreamWaitEvent(pipeline.transfer_stream,
                                           pipeline.consumer_complete[slot], 0));
        }
        for (std::size_t local = 0; local < group_end - group_begin; ++local) {
            const std::int32_t expert = unique_experts[group_begin + local];
            auto* pair = pinned_slot + local * rank_bytes;
            std::memcpy(pair,
                        weights.routed_gate_up.gate.data() +
                            static_cast<std::size_t>(expert) * one_matrix,
                        one_matrix);
            std::memcpy(pair + one_matrix,
                        weights.routed_gate_up.up.data() +
                            static_cast<std::size_t>(expert) * one_matrix,
                        one_matrix);
        }
        const std::int32_t first_expert = unique_experts[group_begin];
        const std::int32_t last_expert = unique_experts[group_end - 1];
        const std::int32_t group_occurrence_begin = expert_offsets[first_expert];
        const std::int32_t group_occurrence_end = expert_offsets[last_expert + 1];
        const std::int32_t group_occurrences =
            group_occurrence_end - group_occurrence_begin;
        const std::size_t staged_matrix_bytes = (group_end - group_begin) * rank_bytes;
        auto* pinned_occurrences = pinned_slot + staged_matrix_bytes;
        auto* device_occurrences = device_slot + staged_matrix_bytes;
        std::memcpy(pinned_occurrences,
                    occurrence_slots + group_occurrence_begin,
                    static_cast<std::size_t>(group_occurrences) * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpyAsync(device_slot, pinned_slot,
                                   staged_matrix_bytes +
                                       static_cast<std::size_t>(group_occurrences) *
                                           sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, pipeline.transfer_stream));
        CUDA_CHECK(cudaEventRecord(pipeline.transfer_ready[slot], pipeline.transfer_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline.transfer_ready[slot], 0));

        Tensor occurrence_tensor(device_occurrences, DType::I32, {group_occurrences});
        for (std::size_t local = 0; local < group_end - group_begin; ++local) {
            const std::int32_t expert = unique_experts[group_begin + local];
            const std::int32_t count = expert_counts[expert];
            const std::int32_t offset = expert_offsets[expert] - group_occurrence_begin;
            Tensor gathered = scratch.gathered.slice(1, offset, count);
            Tensor gate_output = scratch.gate.slice(1, offset, count);
            Tensor up_output = scratch.up.slice(1, offset, count);
            Tensor activated = scratch.activated.slice(1, offset, count);
            Tensor expert_output = scratch.expert.slice(1, offset, count);
            detail::qwen4_sparse_moe_prefill_gather_launch(
                x, occurrence_tensor, offset, count, gathered, stream);
            const auto* pair = device_slot + local * rank_bytes;
            const Weight gate = matrix_view(pair, weights.routed_gate_up.qtype,
                                            kQwen4SparseMoeIntermediate,
                                            kQwen4SparseMoeHidden);
            const Weight up = matrix_view(pair + one_matrix, weights.routed_gate_up.qtype,
                                          kQwen4SparseMoeIntermediate,
                                          kQwen4SparseMoeHidden);
            const Weight down = matrix_view(
                routed_down + static_cast<std::size_t>(expert) * routed_down_matrix,
                QType::GGML_IQ4_NL, kQwen4SparseMoeHidden,
                kQwen4SparseMoeIntermediate);
            ggml_block_linear(gathered, gate, gate_output, stream);
            ggml_block_linear(gathered, up, up_output, stream);
            detail::qwen4_sparse_moe_swiglu_launch(gate_output, up_output, activated, stream);
            ggml_block_linear(activated, down, expert_output, stream);
            detail::qwen4_sparse_moe_prefill_scatter_launch(
                expert_output, occurrence_tensor, offset, count, scratch.rank_results, stream);
        }
        CUDA_CHECK(cudaEventRecord(pipeline.consumer_complete[slot], stream));
    }
    detail::qwen4_sparse_moe_prefill_finish_launch(
        scratch.rank_results, selected_weights, scratch.shared, scratch.shared_gate_value,
        destination, stream);
}

void qwen4_sparse_moe_resident(const Tensor& x,
                               const Qwen4ResidentSparseMoeWeights& weights,
                               Tensor& selected_ids, Tensor& selected_weights,
                               Tensor& destination, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const std::int32_t width = x.ne[1];
    require_matrix_tensor(x, DType::BF16, kQwen4SparseMoeHidden, width, "x",
                          "qwen4_sparse_moe_resident");
    require_tensor(weights.shared_gate, DType::FP32, kQwen4SparseMoeHidden, "shared_gate");
    require_matrix_tensor(selected_ids, DType::I32, kQwen4SparseMoeTopK, width,
                          "selected_ids", "qwen4_sparse_moe_resident");
    require_matrix_tensor(selected_weights, DType::FP32, kQwen4SparseMoeTopK, width,
                          "selected_weights", "qwen4_sparse_moe_resident");
    require_matrix_tensor(destination, DType::BF16, kQwen4SparseMoeHidden, width,
                          "destination", "qwen4_sparse_moe_resident");
    require_dense_router(weights.router);
    const QType routed_qtype = weights.routed_gate.qtype;
    if ((routed_qtype != QType::GGML_IQ1_S && routed_qtype != QType::GGML_IQ2_XXS) ||
        weights.routed_up.qtype != routed_qtype) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_resident: routed gate/up formats differ or are invalid");
    }
    require_ggml_weight(weights.routed_gate, routed_qtype, kQwen4SparseMoeExperts,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden, "routed_gate");
    require_ggml_weight(weights.routed_up, routed_qtype, kQwen4SparseMoeExperts,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden, "routed_up");
    require_ggml_weight(weights.routed_down, QType::GGML_IQ4_NL, kQwen4SparseMoeExperts,
                        kQwen4SparseMoeHidden, kQwen4SparseMoeIntermediate, "routed_down");
    if ((weights.shared_gate_proj.qtype != QType::GGML_Q5_K &&
         weights.shared_gate_proj.qtype != QType::GGML_Q6_K) ||
        weights.shared_up.qtype != weights.shared_gate_proj.qtype) {
        throw std::invalid_argument(
            "qwen4_sparse_moe_resident: shared gate/up formats differ or are invalid");
    }
    require_ggml_weight(weights.shared_gate_proj, weights.shared_gate_proj.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden,
                        "shared_gate_proj");
    require_ggml_weight(weights.shared_up, weights.shared_gate_proj.qtype, 1,
                        kQwen4SparseMoeIntermediate, kQwen4SparseMoeHidden, "shared_up");
    require_ggml_weight(weights.shared_down, QType::GGML_Q8_0, 1, kQwen4SparseMoeHidden,
                        kQwen4SparseMoeIntermediate, "shared_down");
    const std::size_t required = required_resident_workspace(width);
    if (!has_workspace_capacity(workspace, required)) {
        throw std::invalid_argument("qwen4_sparse_moe_resident: insufficient workspace");
    }

    const std::array<AddressRange, 13> ranges{
        address_range(x.data, x.bytes(), "x"),
        address_range(weights.router.payload, weights.router.payload_bytes, "router"),
        address_range(weights.routed_gate.payload, weights.routed_gate.payload_bytes,
                      "routed_gate"),
        address_range(weights.routed_up.payload, weights.routed_up.payload_bytes, "routed_up"),
        address_range(weights.routed_down.payload, weights.routed_down.payload_bytes,
                      "routed_down"),
        address_range(weights.shared_gate.data, weights.shared_gate.bytes(), "shared_gate"),
        address_range(weights.shared_gate_proj.payload, weights.shared_gate_proj.payload_bytes,
                      "shared_gate_proj"),
        address_range(weights.shared_up.payload, weights.shared_up.payload_bytes, "shared_up"),
        address_range(weights.shared_down.payload, weights.shared_down.payload_bytes,
                      "shared_down"),
        address_range(selected_ids.data, selected_ids.bytes(), "selected_ids"),
        address_range(selected_weights.data, selected_weights.bytes(), "selected_weights"),
        address_range(destination.data, destination.bytes(), "destination"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    require_disjoint(ranges);

    auto scope = workspace.scope();
    if (width < kResidentGroupedMinWidth) {
        ResidentScratch scratch = allocate_resident_scratch(workspace);
        for (std::int32_t token = 0; token < width; ++token) {
            auto* x_data = static_cast<std::byte*>(x.data) +
                           static_cast<std::size_t>(token) * kQwen4SparseMoeHidden *
                               sizeof(std::uint16_t);
            auto* ids_data = static_cast<std::byte*>(selected_ids.data) +
                             static_cast<std::size_t>(token) * kQwen4SparseMoeTopK *
                                 sizeof(std::int32_t);
            auto* selected_data = static_cast<std::byte*>(selected_weights.data) +
                                  static_cast<std::size_t>(token) * kQwen4SparseMoeTopK *
                                      sizeof(float);
            auto* destination_data = static_cast<std::byte*>(destination.data) +
                                     static_cast<std::size_t>(token) *
                                         kQwen4SparseMoeHidden * sizeof(std::uint16_t);
            Tensor token_x(x_data, DType::BF16, {kQwen4SparseMoeHidden});
            Tensor token_ids(ids_data, DType::I32, {kQwen4SparseMoeTopK});
            Tensor token_weights(selected_data, DType::FP32, {kQwen4SparseMoeTopK});
            Tensor token_destination(destination_data, DType::BF16,
                                     {kQwen4SparseMoeHidden});
            detail::qwen4_sparse_moe_resident_route_launch(
                token_x, weights.router, weights.shared_gate, scratch.logits, token_ids,
                token_weights, scratch.shared_gate_value, stream);
            detail::qwen4_sparse_moe_shared_gate_up_swiglu_launch(
                token_x, weights.shared_gate_proj, weights.shared_up,
                scratch.shared_activated, stream);
            ggml_block_linear(scratch.shared_activated, weights.shared_down, scratch.shared,
                              stream);
            detail::qwen4_sparse_moe_indexed_gate_up_swiglu_launch(
                token_x, weights.routed_gate, weights.routed_up, token_ids,
                scratch.routed_activated, stream);
            detail::qwen4_sparse_moe_indexed_down_finish_launch(
                scratch.routed_activated, weights.routed_down, token_ids, token_weights,
                scratch.shared, scratch.shared_gate_value, token_destination, stream);
        }
        return;
    }

    ResidentWideScratch scratch = allocate_resident_wide_scratch(workspace, width);
    detail::qwen4_sparse_moe_resident_wide_route_launch(
        x, weights.router, weights.shared_gate, scratch.logits, selected_ids,
        selected_weights, scratch.shared_gate_value, stream);
    ggml_block_linear(x, weights.shared_gate_proj, scratch.shared_gate_projection, stream);
    ggml_block_linear(x, weights.shared_up, scratch.shared_up, stream);
    detail::qwen4_sparse_moe_swiglu_launch(
        scratch.shared_gate_projection, scratch.shared_up,
        scratch.shared_gate_projection, stream);
    ggml_block_linear(scratch.shared_gate_projection, weights.shared_down, scratch.shared,
                      stream);

    detail::qwen4_sparse_moe_resident_group_launch(
        selected_ids, scratch.expert_counts, scratch.expert_offsets,
        scratch.expert_cursors, scratch.occurrence_slots, stream);
    detail::qwen4_sparse_moe_resident_grouped_gate_up_launch(
        x, weights.routed_gate, scratch.expert_counts, scratch.expert_offsets,
        scratch.occurrence_slots, scratch.routed_gate, stream);
    detail::qwen4_sparse_moe_resident_grouped_gate_up_launch(
        x, weights.routed_up, scratch.expert_counts, scratch.expert_offsets,
        scratch.occurrence_slots, scratch.routed_up, stream);
    detail::qwen4_sparse_moe_swiglu_launch(
        scratch.routed_gate, scratch.routed_up, scratch.routed_gate, stream);
    detail::qwen4_sparse_moe_resident_grouped_down_launch(
        scratch.routed_gate, weights.routed_down, scratch.expert_counts,
        scratch.expert_offsets, scratch.occurrence_slots, scratch.rank_results, stream);
    detail::qwen4_sparse_moe_prefill_finish_launch(
        scratch.rank_results, selected_weights, scratch.shared,
        scratch.shared_gate_value, destination, stream);
}

} // namespace ninfer::ops

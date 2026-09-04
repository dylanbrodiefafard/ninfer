#include "ninfer/ops/ple.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/ple.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1U)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, const char* op, const char* name,
                    std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_q8_weight(const Weight& weight, std::int32_t rows, const char* name) {
    constexpr std::int32_t columns = kPleEmbeddingWidth;
    constexpr std::uint64_t row_bytes = (columns / 32) * 34;
    const std::uint64_t expected = static_cast<std::uint64_t>(rows) * row_bytes;
    if (weight.qtype != QType::GGML_Q8_0 || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.ndim != 2 || weight.n != rows || weight.k != columns || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != rows || weight.padded_shape[1] != columns ||
        weight.padded_shape[2] != 1 || weight.padded_shape[3] != 1 || weight.group != 32 ||
        weight.group_size != 32 || weight.payload == nullptr || weight.qdata != weight.payload ||
        weight.payload_bytes != expected || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.payload, 256)) {
        throw std::invalid_argument(std::string("ple_inject: invalid ") + name);
    }
}

struct Scratch {
    Tensor key;
    Tensor value;
    Tensor gated;
    Tensor current_state;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, std::int32_t width) {
    return {
        allocator.alloc(DType::BF16, {kPleChannels, width}),
        allocator.alloc(DType::BF16, {kPleEmbeddingWidth, width}),
        allocator.alloc(DType::FP32, {kPleChannels, width}),
        allocator.alloc(DType::BF16, {kPleChannels, width}),
    };
}

std::size_t required_workspace(std::int32_t width) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, width);
    return layout.peak_bytes();
}

struct Range {
    std::uintptr_t begin;
    std::uintptr_t end;
};

Range range(const void* pointer, std::size_t bytes, const char* op) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || bytes == 0 || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string(op) + ": invalid storage range");
    }
    return {begin, begin + bytes};
}

bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

void require_disjoint(const std::vector<Range>& ranges, const char* op) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument(std::string(op) + ": storage overlap is not permitted");
            }
        }
    }
}

} // namespace

void ple_iq4_nl_stage_rows(const PleMappedIq4NlTable& table,
                           std::span<const std::int32_t, kPleHeads> row_ids, void* pinned_rows,
                           std::size_t pinned_bytes, Tensor& device_rows, cudaStream_t stream) {
    constexpr const char* op = "ple_iq4_nl_stage_rows";
    require_tensor(device_rows, DType::U8, kPleIq4NlRowBytes, kPleHeads, 1, op, "device_rows", 16);
    if (table.data == nullptr || table.rows == 0 || table.rows > UINT64_MAX / kPleIq4NlRowBytes ||
        table.bytes != table.rows * kPleIq4NlRowBytes) {
        throw std::invalid_argument("ple_iq4_nl_stage_rows: invalid mapped table span");
    }
    if (pinned_rows == nullptr || pinned_bytes < kPleStagedBytes) {
        throw std::invalid_argument("ple_iq4_nl_stage_rows: pinned slot is too small");
    }
    require_disjoint({range(table.data, static_cast<std::size_t>(table.bytes), op),
                      range(pinned_rows, kPleStagedBytes, op),
                      range(device_rows.data, device_rows.bytes(), op)},
                     op);
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, pinned_rows);
    if (attribute_status != cudaSuccess || attributes.type != cudaMemoryTypeHost) {
        if (attribute_status != cudaSuccess) { (void)cudaGetLastError(); }
        throw std::invalid_argument("ple_iq4_nl_stage_rows: staging source is not pinned host memory");
    }
    for (int head = 0; head < kPleHeads; ++head) {
        const std::int32_t row = row_ids[head];
        if (row < 0 || static_cast<std::uint64_t>(row) >= table.rows) {
            throw std::invalid_argument("ple_iq4_nl_stage_rows: row id is outside mapped table");
        }
    }
    auto* destination = static_cast<std::uint8_t*>(pinned_rows);
    for (int head = 0; head < kPleHeads; ++head) {
        const std::int32_t row = row_ids[head];
        std::memcpy(destination + static_cast<std::size_t>(head) * kPleIq4NlRowBytes,
                    table.data + static_cast<std::uint64_t>(row) * kPleIq4NlRowBytes,
                    kPleIq4NlRowBytes);
    }
    CUDA_CHECK(cudaMemcpyAsync(device_rows.data, pinned_rows, kPleStagedBytes,
                               cudaMemcpyHostToDevice, stream));
}

void ple_iq4_nl_decode_rows(const Tensor& device_rows, Tensor& embedding, cudaStream_t stream) {
    constexpr const char* op = "ple_iq4_nl_decode_rows";
    require_tensor(device_rows, DType::U8, kPleIq4NlRowBytes, kPleHeads, 1, op, "device_rows", 16);
    require_tensor(embedding, DType::BF16, kPleRowWidth, kPleHeads, 1, op, "embedding", 16);
    if (overlaps(range(device_rows.data, device_rows.bytes(), op),
                 range(embedding.data, embedding.bytes(), op))) {
        throw std::invalid_argument("ple_iq4_nl_decode_rows: input and output overlap");
    }
    detail::ple_iq4_nl_decode_rows_launch(device_rows, embedding, stream);
}

std::size_t ple_workspace_capacity_bytes(std::int32_t width) {
    if (width <= 0 || width > 4096) {
        throw std::invalid_argument("ple_workspace_capacity_bytes: width must be in [1,4096]");
    }
    return required_workspace(width);
}

void ple_inject(const Tensor& residual, const Tensor& embedding, const Weight& key_weight,
                const Weight& value_weight, const Tensor& key_norm_weight,
                const Tensor& query_norm_weight, const Tensor& conv_norm_weight,
                const Tensor& conv_weight, const Tensor& old_conv_state,
                Tensor& new_conv_state, Tensor& residual_out, WorkspaceArena& workspace,
                cudaStream_t stream) {
    constexpr const char* op = "ple_inject";
    const int width = residual.ne[2];
    if (width <= 0 || width > 4096) {
        throw std::invalid_argument("ple_inject: width must be in [1,4096]");
    }
    require_tensor(residual, DType::BF16, kPleEmbeddingWidth, kPleBranches, width, op, "residual");
    require_tensor(embedding, DType::BF16, kPleEmbeddingWidth, width, 1, op, "embedding");
    require_tensor(key_norm_weight, DType::FP32, kPleChannels, 1, 1, op, "key_norm_weight");
    require_tensor(query_norm_weight, DType::FP32, kPleChannels, 1, 1, op,
                   "query_norm_weight");
    require_tensor(conv_norm_weight, DType::FP32, kPleChannels, 1, 1, op, "conv_norm_weight");
    require_tensor(conv_weight, DType::FP32, 4, kPleChannels, 1, op, "conv_weight");
    require_tensor(old_conv_state, DType::BF16, kPleChannels, kPleConvHistory, 1, op,
                   "old_conv_state");
    require_tensor(new_conv_state, DType::BF16, kPleChannels, kPleConvHistory, 1, op,
                   "new_conv_state");
    require_tensor(residual_out, DType::BF16, kPleEmbeddingWidth, kPleBranches, width, op,
                   "residual_out");
    require_q8_weight(key_weight, kPleChannels, "key_weight");
    require_q8_weight(value_weight, kPleEmbeddingWidth, "value_weight");
    const std::size_t required = required_workspace(width);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("ple_inject: insufficient workspace capacity");
    }
    const Range residual_range = range(residual.data, residual.bytes(), op);
    const Range output_range   = range(residual_out.data, residual_out.bytes(), op);
    if (overlaps(residual_range, output_range) &&
        !(residual.data == residual_out.data && residual.bytes() == residual_out.bytes())) {
        throw std::invalid_argument("ple_inject: residual partially overlaps output");
    }
    const Range old_range = range(old_conv_state.data, old_conv_state.bytes(), op);
    const Range new_range = range(new_conv_state.data, new_conv_state.bytes(), op);
    if (overlaps(old_range, new_range) &&
        !(old_conv_state.data == new_conv_state.data &&
          old_conv_state.bytes() == new_conv_state.bytes())) {
        throw std::invalid_argument("ple_inject: convolution states partially overlap");
    }
    std::vector<Range> ranges{
        residual_range,
        range(embedding.data, embedding.bytes(), op),
        range(key_weight.payload, static_cast<std::size_t>(key_weight.payload_bytes), op),
        range(value_weight.payload, static_cast<std::size_t>(value_weight.payload_bytes), op),
        range(key_norm_weight.data, key_norm_weight.bytes(), op),
        range(query_norm_weight.data, query_norm_weight.bytes(), op),
        range(conv_norm_weight.data, conv_norm_weight.bytes(), op),
        range(conv_weight.data, conv_weight.bytes(), op),
        old_range,
        range(workspace.base(), workspace.capacity(), op),
    };
    if (residual_out.data != residual.data) { ranges.push_back(output_range); }
    if (new_conv_state.data != old_conv_state.data) { ranges.push_back(new_range); }
    require_disjoint(ranges, op);

    auto scope     = workspace.scope();
    Scratch scratch = allocate_scratch(workspace, width);
    for (int token = 0; token < width; ++token) {
        Tensor input = embedding.slice(1, token, 1).reshape({kPleEmbeddingWidth});
        Tensor key_output = scratch.key.slice(1, token, 1).reshape({kPleChannels});
        Tensor value_output = scratch.value.slice(1, token, 1).reshape({kPleEmbeddingWidth});
        ggml_block_linear(input, key_weight, key_output, stream);
        ggml_block_linear(input, value_weight, value_output, stream);
    }
    detail::ple_gate_launch(residual, scratch.key, scratch.value, key_norm_weight,
                            query_norm_weight, scratch.gated, stream);
    detail::ple_conv_input_launch(scratch.gated, conv_norm_weight, scratch.current_state, stream);
    detail::ple_conv_inject_launch(residual, scratch.gated, conv_weight, old_conv_state,
                                   scratch.current_state, residual_out, stream);
    detail::ple_state_update_launch(old_conv_state, scratch.current_state, new_conv_state, stream);
}

} // namespace ninfer::ops

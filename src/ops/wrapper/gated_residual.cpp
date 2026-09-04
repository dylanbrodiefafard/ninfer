#include "ninfer/ops/gated_residual.h"

#include "core/layout.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/gated_residual.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kHidden * kBranches;
constexpr std::int32_t kRank = 320;

struct Scratch {
    Tensor normalized;
    Tensor low_rank;
    Tensor up_logits;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator) {
    return {allocator.alloc(DType::BF16, {kFlat}), allocator.alloc(DType::BF16, {kRank}),
            allocator.alloc(DType::BF16, {kFlat})};
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::array<std::int32_t, 4> shape,
                    const char* op, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != shape[0] || tensor.ne[1] != shape[1] ||
        tensor.ne[2] != shape[2] || tensor.ne[3] != shape[3] || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_q8_weight(const Weight& weight, std::int32_t rows, std::int32_t columns,
                       const char* op, const char* name) {
    const std::uint64_t bytes = static_cast<std::uint64_t>(rows) * (columns / 32) * 34;
    if (weight.qtype != QType::GGML_Q8_0 || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.payload == nullptr || weight.qdata != weight.payload || weight.payload_bytes != bytes ||
        weight.ndim != 2 || weight.n != rows || weight.k != columns ||
        weight.shape[0] != rows || weight.shape[1] != columns || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != columns || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.group_size != 32 || weight.group != 32 ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.high_plane_bytes != 0) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    std::string name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("gated_residual: " + name + " storage must be non-empty");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("gated_residual: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

bool overlaps(const AddressRange& lhs, const AddressRange& rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (overlaps(ranges[i], ranges[j])) {
                throw std::invalid_argument("gated_residual: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::size_t required_workspace() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout);
    return layout.peak_bytes();
}

void run_read(const Tensor& residual, const Tensor& norm_weight, const Weight& down_weight,
              const Weight& up_weight, const Tensor* write_weight, Tensor& x,
              Tensor* write_scale, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "gated_residual_read";
    require_tensor(residual, DType::BF16, {kHidden, kBranches, 1, 1}, op, "residual");
    require_tensor(norm_weight, DType::FP32, {kFlat, 1, 1, 1}, op, "norm_weight");
    require_q8_weight(down_weight, kRank, kFlat, op, "down_weight");
    require_q8_weight(up_weight, kFlat, kRank, op, "up_weight");
    require_tensor(x, DType::BF16, {kHidden, 1, 1, 1}, op, "x");
    if ((write_weight == nullptr) != (write_scale == nullptr)) {
        throw std::invalid_argument("gated_residual_read: write weight/output must be paired");
    }
    if (write_weight != nullptr) {
        require_tensor(*write_weight, DType::FP32, {kFlat, kBranches, 1, 1}, op, "write_weight");
        require_tensor(*write_scale, DType::BF16, {kBranches, 1, 1, 1}, op, "write_scale");
    }

    const std::size_t required = required_workspace();
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("gated_residual_read: insufficient workspace capacity");
    }
    std::vector<AddressRange> ranges{
        address_range(residual.data, residual.bytes(), "residual"),
        address_range(norm_weight.data, norm_weight.bytes(), "norm_weight"),
        address_range(down_weight.payload, down_weight.payload_bytes, "down_weight"),
        address_range(up_weight.payload, up_weight.payload_bytes, "up_weight"),
        address_range(x.data, x.bytes(), "x"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    if (write_weight != nullptr) {
        ranges.push_back(address_range(write_weight->data, write_weight->bytes(), "write_weight"));
        ranges.push_back(address_range(write_scale->data, write_scale->bytes(), "write_scale"));
    }
    require_disjoint(ranges);

    auto scope = workspace.scope();
    Scratch scratch = allocate_scratch(workspace);
    detail::gated_residual_normalize_launch(residual, norm_weight, scratch.normalized, stream);
    ggml_block_linear(scratch.normalized, down_weight, scratch.low_rank, stream);
    detail::gated_residual_activate_launch(scratch.low_rank, stream);
    ggml_block_linear(scratch.low_rank, up_weight, scratch.up_logits, stream);
    detail::gated_residual_mix_launch(scratch.normalized, scratch.up_logits, x, stream);
    if (write_weight != nullptr) {
        detail::gated_residual_write_launch(scratch.normalized, *write_weight, *write_scale, stream);
    }
}

} // namespace

std::size_t gated_residual_workspace_capacity_bytes() { return required_workspace(); }

void gated_residual_read(const Tensor& residual, const Tensor& norm_weight,
                         const Weight& down_weight, const Weight& up_weight, Tensor& x,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    run_read(residual, norm_weight, down_weight, up_weight, nullptr, x, nullptr, workspace, stream);
}

void gated_residual_read_write(const Tensor& residual, const Tensor& norm_weight,
                               const Weight& down_weight, const Weight& up_weight,
                               const Tensor& write_weight, Tensor& x, Tensor& write_scale,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    run_read(residual, norm_weight, down_weight, up_weight, &write_weight, x, &write_scale,
             workspace, stream);
}

void gated_residual_inject(const Tensor& residual, const Tensor& block_output,
                           const Tensor& write_scale, Tensor& residual_out,
                           cudaStream_t stream) {
    constexpr const char* op = "gated_residual_inject";
    require_tensor(residual, DType::BF16, {kHidden, kBranches, 1, 1}, op, "residual");
    require_tensor(block_output, DType::BF16, {kHidden, 1, 1, 1}, op, "block_output");
    require_tensor(write_scale, DType::BF16, {kBranches, 1, 1, 1}, op, "write_scale");
    require_tensor(residual_out, DType::BF16, {kHidden, kBranches, 1, 1}, op, "residual_out");
    const AddressRange residual_range = address_range(residual.data, residual.bytes(), "residual");
    const AddressRange output_range = address_range(residual_out.data, residual_out.bytes(), "residual_out");
    const bool exact_alias = residual.data == residual_out.data && residual.bytes() == residual_out.bytes();
    if (overlaps(residual_range, output_range) && !exact_alias) {
        throw std::invalid_argument("gated_residual: residual partially overlaps residual_out");
    }
    std::vector<AddressRange> ranges{
        address_range(block_output.data, block_output.bytes(), "block_output"),
        address_range(write_scale.data, write_scale.bytes(), "write_scale"), output_range};
    require_disjoint(ranges);
    for (std::size_t i = 0; i + 1 < ranges.size(); ++i) {
        if (overlaps(residual_range, ranges[i])) {
            throw std::invalid_argument("gated_residual: residual overlaps " + ranges[i].name);
        }
    }
    detail::gated_residual_inject_launch(residual, block_output, write_scale, residual_out, stream);
}

} // namespace ninfer::ops

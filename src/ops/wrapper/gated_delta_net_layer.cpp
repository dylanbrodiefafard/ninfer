#include "ninfer/ops/gated_delta_net_layer.h"

#include "core/layout.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/gated_delta_net_layer.h"

#include <array>
#include <cmath>
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
constexpr std::int32_t kQkHeads = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQkRows = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows = kValueHeads * kHeadDim;
constexpr std::int32_t kQkvRows = 2 * kQkRows + kValueRows;

struct Scratch {
    Tensor projected_qkv;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor z;
    Tensor g;
    Tensor beta;
    Tensor recurrent;
    Tensor normalized_gated;
    DeviceSpan recurrence;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator) {
    const std::size_t recurrence_bytes =
        gated_delta_net_workspace_capacity_bytes(kValueHeads, kValueHeads, true, 1, 1);
    Scratch scratch{
        allocator.alloc(DType::BF16, {kQkvRows}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, 1}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, 1}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, 1}),
        allocator.alloc(DType::BF16, {kValueRows}),
        allocator.alloc(DType::FP32, {kValueHeads}),
        allocator.alloc(DType::FP32, {kValueHeads}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, 1}),
        allocator.alloc(DType::BF16, {kValueRows}),
        {},
    };
    if (recurrence_bytes != 0) { scratch.recurrence = allocator.alloc_bytes(recurrence_bytes); }
    return scratch;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::array<std::int32_t, 4> shape,
                    const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != shape[0] || tensor.ne[1] != shape[1] ||
        tensor.ne[2] != shape[2] || tensor.ne[3] != shape[3] || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gated_delta_net_layer: invalid ") + name);
    }
}

std::uint64_t ggml_payload_bytes(QType qtype, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t block_bytes = qtype == QType::GGML_Q5_K ? 176U : 210U;
    return static_cast<std::uint64_t>(rows) * (columns / 256) * block_bytes;
}

void require_ggml_weight(const Weight& weight, std::int32_t rows, std::int32_t columns,
                         bool allow_q5, const char* name) {
    const bool qtype_ok = weight.qtype == QType::GGML_Q6_K ||
                          (allow_q5 && weight.qtype == QType::GGML_Q5_K);
    if (!qtype_ok || weight.layout != QuantLayout::GgmlBlockRow || weight.ndim != 2 ||
        weight.n != rows || weight.k != columns || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != rows || weight.padded_shape[1] != columns ||
        weight.padded_shape[2] != 1 || weight.padded_shape[3] != 1 ||
        weight.group_size != 256 || weight.group != 256 || weight.payload == nullptr ||
        weight.qdata != weight.payload ||
        weight.payload_bytes != ggml_payload_bytes(weight.qtype, rows, columns) ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.high_plane_bytes != 0) {
        throw std::invalid_argument(std::string("gated_delta_net_layer: invalid ") + name);
    }
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
    std::string name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("gated_delta_net_layer: empty " + name);
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("gated_delta_net_layer: address range overflow");
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
                throw std::invalid_argument("gated_delta_net_layer: " + ranges[i].name +
                                            " overlaps " + ranges[j].name);
            }
        }
    }
}

bool exact_alias(const Tensor& input, const Tensor& output) {
    return input.data == output.data && input.bytes() == output.bytes();
}

std::size_t required_workspace() {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout);
    return layout.peak_bytes();
}

} // namespace

std::size_t gated_delta_net_layer_workspace_capacity_bytes() { return required_workspace(); }

void gated_delta_net_layer(const Tensor& x, const GatedDeltaNetLayerWeights& weights,
                           const Tensor& conv_state_in, Tensor& conv_state_out,
                           const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    require_tensor(x, DType::BF16, {kHidden, 1, 1, 1}, "x");
    require_ggml_weight(weights.qkv, kQkvRows, kHidden, true, "qkv weight");
    require_ggml_weight(weights.z, kValueRows, kHidden, true, "z weight");
    if (weights.z.qtype != weights.qkv.qtype) {
        throw std::invalid_argument("gated_delta_net_layer: qkv and z formats differ");
    }
    require_tensor(weights.a, DType::FP32, {kHidden, kValueHeads, 1, 1}, "a weight");
    require_tensor(weights.b, DType::FP32, {kHidden, kValueHeads, 1, 1}, "b weight");
    require_tensor(weights.conv, DType::FP32, {4, kQkvRows, 1, 1}, "conv weight");
    require_tensor(weights.ssm_a, DType::FP32, {kValueHeads, 1, 1, 1}, "ssm_a");
    require_tensor(weights.dt_bias, DType::FP32, {kValueHeads, 1, 1, 1}, "dt_bias");
    require_tensor(weights.norm, DType::FP32, {kHeadDim, 1, 1, 1}, "norm weight");
    require_ggml_weight(weights.output, kHidden, kValueRows, false, "output weight");
    require_tensor(conv_state_in, DType::BF16, {kQkvRows, 3, 1, 1}, "conv_state_in");
    require_tensor(conv_state_out, DType::BF16, {kQkvRows, 3, 1, 1}, "conv_state_out");
    require_tensor(ssm_state_in, DType::FP32, {kHeadDim, kHeadDim, kValueHeads, 1},
                   "ssm_state_in");
    require_tensor(ssm_state_out, DType::FP32, {kHeadDim, kHeadDim, kValueHeads, 1},
                   "ssm_state_out");
    require_tensor(out, DType::BF16, {kHidden, 1, 1, 1}, "out");

    const bool conv_alias = exact_alias(conv_state_in, conv_state_out);
    const bool ssm_alias = exact_alias(ssm_state_in, ssm_state_out);
    const AddressRange conv_in = address_range(conv_state_in.data, conv_state_in.bytes(), "conv_state_in");
    const AddressRange conv_out = address_range(conv_state_out.data, conv_state_out.bytes(), "conv_state_out");
    const AddressRange ssm_in = address_range(ssm_state_in.data, ssm_state_in.bytes(), "ssm_state_in");
    const AddressRange ssm_out = address_range(ssm_state_out.data, ssm_state_out.bytes(), "ssm_state_out");
    if ((overlaps(conv_in, conv_out) && !conv_alias) || (overlaps(ssm_in, ssm_out) && !ssm_alias)) {
        throw std::invalid_argument("gated_delta_net_layer: state overlap must be exact");
    }

    const std::size_t required = required_workspace();
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("gated_delta_net_layer: insufficient workspace");
    }
    std::vector<AddressRange> ranges{
        address_range(x.data, x.bytes(), "x"),
        address_range(weights.qkv.payload, weights.qkv.payload_bytes, "qkv weight"),
        address_range(weights.z.payload, weights.z.payload_bytes, "z weight"),
        address_range(weights.a.data, weights.a.bytes(), "a weight"),
        address_range(weights.b.data, weights.b.bytes(), "b weight"),
        address_range(weights.conv.data, weights.conv.bytes(), "conv weight"),
        address_range(weights.ssm_a.data, weights.ssm_a.bytes(), "ssm_a"),
        address_range(weights.dt_bias.data, weights.dt_bias.bytes(), "dt_bias"),
        address_range(weights.norm.data, weights.norm.bytes(), "norm weight"),
        address_range(weights.output.payload, weights.output.payload_bytes, "output weight"),
        conv_in,
        ssm_in,
        address_range(out.data, out.bytes(), "out"),
        address_range(workspace.base(), workspace.capacity(), "workspace"),
    };
    if (!conv_alias) { ranges.push_back(conv_out); }
    if (!ssm_alias) { ranges.push_back(ssm_out); }
    require_disjoint(ranges);

    auto scope = workspace.scope();
    Scratch scratch = allocate_scratch(workspace);
    ggml_block_linear(x, weights.qkv, scratch.projected_qkv, stream);
    ggml_block_linear(x, weights.z, scratch.z, stream);
    detail::gated_delta_net_layer_control_launch(x, weights.a, weights.b, weights.ssm_a,
                                                  weights.dt_bias, scratch.g, scratch.beta, stream);
    detail::gated_delta_net_layer_conv_launch(scratch.projected_qkv, weights.conv, conv_state_in,
                                               conv_state_out, scratch.q, scratch.k, scratch.v,
                                               stream);
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    if (scratch.recurrence.bytes == 0) {
        gated_delta_net(scratch.q, scratch.k, scratch.v, scratch.g, scratch.beta, scale, true,
                        workspace, ssm_state_in, ssm_state_out, scratch.recurrent, stream);
    } else {
        WorkspaceArena recurrence_workspace(scratch.recurrence);
        gated_delta_net(scratch.q, scratch.k, scratch.v, scratch.g, scratch.beta, scale, true,
                        recurrence_workspace, ssm_state_in, ssm_state_out, scratch.recurrent,
                        stream);
    }
    detail::gated_delta_net_layer_norm_launch(scratch.recurrent, scratch.z, weights.norm,
                                               scratch.normalized_gated, stream);
    ggml_block_linear(scratch.normalized_gated, weights.output, out, stream);
}

} // namespace ninfer::ops

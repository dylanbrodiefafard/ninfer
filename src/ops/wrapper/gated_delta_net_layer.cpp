#include "ninfer/ops/gated_delta_net_layer.h"

#include "core/layout.h"
#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ops/launcher/gated_delta_net_layer.h"
#include "ops/common/projection.h"
#include "ops/linear/fp8/fp8_tensor.h"
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kQkHeads = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQkRows = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows = kValueHeads * kHeadDim;
constexpr std::int32_t kQkvRows = 2 * kQkRows + kValueRows;

bool native_projection(QType type) {
    return detail::is_native_projection(type) || type == QType::FP8_E4M3FN_TENSOR_F32M;
}

std::size_t required_projection_bytes(QType type, int rows, int columns, int tokens,
                              LinearPolicy policy, bool z_projection = false) {
    if (policy != LinearPolicy::A16Only &&
        (policy != LinearPolicy::AllowA8 ||
         (type != QType::FP8_E4M3FN_TENSOR_F32M &&
          !(z_projection && type == QType::FP8_E4M3FN_ROW_BF16S)))) {
        throw std::invalid_argument("gated_delta_net_layer: unsupported projection precision policy");
    }
    return native_projection(type)
        ? linear_workspace_capacity_bytes(type, rows, columns, policy, 1, tokens) : 0;
}

void project(const Tensor& input, const Weight& weight, Tensor& output,
             WorkspaceArena& workspace, cudaStream_t stream, LinearPolicy policy) {
    if (weight.qtype == QType::BF16_CTRL && weight.n == kQkvRows) {
        // Shared by scalar, compact, batch and replay. The public ingress remains BF16;
        // only private accumulation avoids long-dot rounding amplified by recurrent state.
        detail::launch_bf16_gdn_qkv(input, weight, output, stream);
    } else if (native_projection(weight.qtype)) {
        linear(input, weight, output, policy, workspace, stream);
    } else {
        ggml_block_linear(input, weight, output, stream);
    }
}

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
Scratch allocate_scratch(Allocator& allocator, std::int32_t tokens, bool compact = false) {
    const std::size_t recurrence_bytes =
        compact ? 0 : gated_delta_net_workspace_capacity_bytes(kValueHeads, kValueHeads, true, 1, tokens);
    Scratch scratch{
        allocator.alloc(DType::BF16, {kQkvRows, tokens}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, tokens}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, tokens}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, tokens}),
        allocator.alloc(DType::BF16, {kValueRows, tokens}),
        allocator.alloc(DType::FP32, {kValueHeads, tokens}),
        allocator.alloc(DType::FP32, {kValueHeads, tokens}),
        allocator.alloc(DType::BF16, {kHeadDim, kValueHeads, tokens}),
        allocator.alloc(DType::BF16, {kValueRows, tokens}),
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

void require_projection_weight(const Weight& weight, std::int32_t rows, std::int32_t columns,
                         bool allow_q5, const char* name) {
    if (weight.qtype == QType::FP8_E4M3FN_TENSOR_F32M) {
        if (weight.n != rows || weight.k != columns) {
            throw std::invalid_argument("gated_delta_net_layer: calibrated projection shape");
        }
        detail::validate_fp8_tensor_weight(weight, name);
        return;
    }
    if (detail::is_native_projection(weight.qtype)) {
        detail::validate_native_projection(weight, rows, columns, name);
        return;
    }
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
    const char* name;
};

AddressRange address_range(const void* pointer, std::size_t bytes, const char* name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument(std::string("gated_delta_net_layer: empty ") + name);
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("gated_delta_net_layer: address range overflow");
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
                throw std::invalid_argument(std::string("gated_delta_net_layer: ") +
                                            ranges[i].name + " overlaps " + ranges[j].name);
            }
        }
    }
}

bool exact_alias(const Tensor& input, const Tensor& output) {
    return input.data == output.data && input.bytes() == output.bytes();
}

std::size_t required_workspace(std::int32_t tokens, QType qkv, QType z, QType output,
                               GatedDeltaNetProjectionPolicy policy, bool compact = false) {
    if (policy.qkv == LinearPolicy::AllowA8 && policy.output == LinearPolicy::AllowA8) {
        throw std::invalid_argument("gated_delta_net_layer: simultaneous QKV/output A8 is not qualified");
    }
    if (z == QType::FP8_E4M3FN_ROW_BF16S && policy.z == LinearPolicy::AllowA8 &&
        (policy.qkv != LinearPolicy::A16Only || policy.output != LinearPolicy::A16Only)) {
        throw std::invalid_argument("gated_delta_net_layer: row-FP8 Z A8 requires other projections A16");
    }
    if (!native_projection(qkv) &&
        !native_projection(z) && qkv != z) {
        throw std::invalid_argument("gated_delta_net_layer: GGML input formats differ");
    }
    for (QType type : {qkv, z}) {
        if (type != QType::GGML_Q5_K && type != QType::GGML_Q6_K &&
            !native_projection(type)) {
            throw std::invalid_argument("gated_delta_net_layer: unsupported input format");
        }
    }
    if (output != QType::GGML_Q6_K && !native_projection(output)) {
        throw std::invalid_argument("gated_delta_net_layer: unsupported output format");
    }
    WorkspaceLayoutBuilder layout;
    // Native accuracy profile keeps normalization and recurrent arithmetic in FP32.
    // The diagnostic GGML profile retains its separately qualified chunked schedule.
    (void)allocate_scratch(layout, tokens, compact || native_projection(qkv));
    const auto projection_bytes = std::max({
        required_projection_bytes(qkv, kQkvRows, kHidden, tokens, policy.qkv),
        required_projection_bytes(z, kValueRows, kHidden, tokens, policy.z, true),
        required_projection_bytes(output, kHidden, kValueRows, tokens, policy.output)});
    if (projection_bytes) { (void)layout.alloc_bytes(projection_bytes); }
    return layout.peak_bytes();
}

} // namespace

std::size_t gated_delta_net_layer_workspace_capacity_bytes(
    std::int32_t max_tokens, QType qkv, QType z, QType output,
    GatedDeltaNetProjectionPolicy policy) {
    if (max_tokens <= 0 || max_tokens > 4096) {
        throw std::invalid_argument(
            "gated_delta_net_layer_workspace_capacity_bytes: max_tokens must be in [1,4096]");
    }
    return required_workspace(max_tokens, qkv, z, output, policy);
}

void gated_delta_net_layer(const Tensor& x, const GatedDeltaNetLayerWeights& weights,
                           const Tensor& conv_state_in, Tensor& conv_state_out,
                           const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           GatedDeltaNetProjectionPolicy policy,
                           const GdnReplayRecordLayer* replay) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || tokens > 4096) {
        throw std::invalid_argument("gated_delta_net_layer: T must be in [1,4096]");
    }
    require_tensor(x, DType::BF16, {kHidden, tokens, 1, 1}, "x");
    require_projection_weight(weights.qkv, kQkvRows, kHidden, true, "qkv weight");
    require_projection_weight(weights.z, kValueRows, kHidden, true, "z weight");
    if (!native_projection(weights.z.qtype) &&
        !native_projection(weights.qkv.qtype) &&
        weights.z.qtype != weights.qkv.qtype) {
        throw std::invalid_argument("gated_delta_net_layer: qkv and z formats differ");
    }
    require_tensor(weights.a, DType::FP32, {kHidden, kValueHeads, 1, 1}, "a weight");
    require_tensor(weights.b, DType::FP32, {kHidden, kValueHeads, 1, 1}, "b weight");
    require_tensor(weights.conv, DType::FP32, {4, kQkvRows, 1, 1}, "conv weight");
    require_tensor(weights.ssm_a, DType::FP32, {kValueHeads, 1, 1, 1}, "ssm_a");
    require_tensor(weights.dt_bias, DType::FP32, {kValueHeads, 1, 1, 1}, "dt_bias");
    require_tensor(weights.norm, DType::FP32, {kHeadDim, 1, 1, 1}, "norm weight");
    require_projection_weight(weights.output, kHidden, kValueRows, false, "output weight");
    require_tensor(conv_state_in, DType::BF16, {kQkvRows, 3, 1, 1}, "conv_state_in");
    require_tensor(conv_state_out, DType::BF16, {kQkvRows, 3, 1, 1}, "conv_state_out");
    require_tensor(ssm_state_in, DType::FP32, {kHeadDim, kHeadDim, kValueHeads, 1},
                   "ssm_state_in");
    require_tensor(ssm_state_out, DType::FP32, {kHeadDim, kHeadDim, kValueHeads, 1},
                   "ssm_state_out");
    require_tensor(out, DType::BF16, {kHidden, tokens, 1, 1}, "out");
    if (replay) {
        if (tokens < 2 || tokens > 16)
            throw std::invalid_argument("gated_delta_net_layer: replay width must be 2..16");
        require_tensor(replay->conv, DType::BF16, {kQkvRows,tokens,1,1}, "conv replay");
        require_tensor(replay->key, DType::BF16, {kHeadDim,kValueHeads,tokens,1}, "key replay");
        require_tensor(replay->value, DType::BF16, {kHeadDim,kValueHeads,tokens,1}, "value replay");
        require_tensor(replay->gate, DType::FP32, {2,kValueHeads,tokens,1}, "gate replay");
    }

    const bool conv_alias = exact_alias(conv_state_in, conv_state_out);
    const bool ssm_alias = exact_alias(ssm_state_in, ssm_state_out);
    const AddressRange conv_in = address_range(conv_state_in.data, conv_state_in.bytes(), "conv_state_in");
    const AddressRange conv_out = address_range(conv_state_out.data, conv_state_out.bytes(), "conv_state_out");
    const AddressRange ssm_in = address_range(ssm_state_in.data, ssm_state_in.bytes(), "ssm_state_in");
    const AddressRange ssm_out = address_range(ssm_state_out.data, ssm_state_out.bytes(), "ssm_state_out");
    if ((overlaps(conv_in, conv_out) && !conv_alias) || (overlaps(ssm_in, ssm_out) && !ssm_alias)) {
        throw std::invalid_argument("gated_delta_net_layer: state overlap must be exact");
    }

    const std::size_t required = required_workspace(
        tokens, weights.qkv.qtype, weights.z.qtype, weights.output.qtype, policy);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("gated_delta_net_layer: insufficient workspace");
    }
    std::array<AddressRange, 20> ranges{};
    std::size_t range_count = 0;
    ranges[range_count++] = address_range(x.data, x.bytes(), "x");
    ranges[range_count++] =
        address_range(weights.qkv.payload, weights.qkv.payload_bytes, "qkv weight");
    ranges[range_count++] =
        address_range(weights.z.payload, weights.z.payload_bytes, "z weight");
    ranges[range_count++] = address_range(weights.a.data, weights.a.bytes(), "a weight");
    ranges[range_count++] = address_range(weights.b.data, weights.b.bytes(), "b weight");
    ranges[range_count++] = address_range(weights.conv.data, weights.conv.bytes(), "conv weight");
    ranges[range_count++] = address_range(weights.ssm_a.data, weights.ssm_a.bytes(), "ssm_a");
    ranges[range_count++] =
        address_range(weights.dt_bias.data, weights.dt_bias.bytes(), "dt_bias");
    ranges[range_count++] = address_range(weights.norm.data, weights.norm.bytes(), "norm weight");
    ranges[range_count++] =
        address_range(weights.output.payload, weights.output.payload_bytes, "output weight");
    ranges[range_count++] = conv_in;
    ranges[range_count++] = ssm_in;
    ranges[range_count++] = address_range(out.data, out.bytes(), "out");
    ranges[range_count++] =
        address_range(workspace.base(), workspace.capacity(), "workspace");
    if (!conv_alias) { ranges[range_count++] = conv_out; }
    if (!ssm_alias) { ranges[range_count++] = ssm_out; }
    if (replay) {
        for (const Tensor* plane : {&replay->conv,&replay->key,&replay->value,&replay->gate})
            ranges[range_count++] = address_range(plane->data,plane->bytes(),"replay record");
    }
    require_disjoint(std::span<const AddressRange>(ranges.data(), range_count));

    auto scope = workspace.scope();
    const bool native_recurrent = native_projection(weights.qkv.qtype);
    Scratch scratch = allocate_scratch(workspace, tokens, native_recurrent);
    project(x, weights.qkv, scratch.projected_qkv, workspace, stream, policy.qkv);
    project(x, weights.z, scratch.z, workspace, stream, policy.z);
    detail::gated_delta_net_layer_control_launch(x, weights.a, weights.b, weights.ssm_a,
                                                  weights.dt_bias, scratch.g, scratch.beta, stream);
    detail::gated_delta_net_layer_conv_launch(scratch.projected_qkv, weights.conv, conv_state_in,
                                               conv_state_out, scratch.q, scratch.k, scratch.v,
                                               stream);
    if (replay) {
        for (const auto& pair : {std::pair{&replay->conv,&scratch.projected_qkv},
                                std::pair{&replay->key,&scratch.k},
                                std::pair{&replay->value,&scratch.v}})
            CUDA_CHECK(cudaMemcpyAsync(pair.first->data,pair.second->data,pair.first->bytes(),
                                       cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaMemcpy2DAsync(replay->gate.data,8,scratch.g.data,4,4,kValueHeads*tokens,
                                     cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(replay->gate.data)+4,8,
                                     scratch.beta.data,4,4,kValueHeads*tokens,
                                     cudaMemcpyDeviceToDevice,stream));
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    if (native_recurrent) {
        detail::gated_delta_net::launch_recurrent_inout(scratch.q,scratch.k,scratch.v,
            scratch.g,scratch.beta,scale,true,ssm_state_in,ssm_state_out,scratch.recurrent,stream);
    } else if (scratch.recurrence.bytes == 0) {
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
    project(scratch.normalized_gated, weights.output, out, workspace, stream, policy.output);
}

std::size_t gated_delta_net_layer_batch_workspace_capacity_bytes(
    std::int32_t width, std::int32_t batch, QType qkv, QType z, QType output,
    GatedDeltaNetProjectionPolicy policy) {
    if (batch < 1 || batch > 4 || width < 1 || width > 4096 / batch ||
        !native_projection(qkv) || !native_projection(z) || !native_projection(output))
        throw std::invalid_argument("gated_delta_net_layer_batch: invalid native profile");
    return required_workspace(width * batch, qkv, z, output, policy, true);
}

void gated_delta_net_layer_batch(const Tensor& x, const GatedDeltaNetLayerWeights& w,
    const Tensor& conv_in, Tensor& conv_out, const Tensor& ssm_in, Tensor& ssm_out,
    const Tensor& slots, const Tensor& valid, Tensor& out,
    WorkspaceArena& workspace, cudaStream_t stream,
    GatedDeltaNetProjectionPolicy policy, const GdnReplayRecordLayer* replay) {
    const int width = x.ne[1], batch = x.ne[2], capacity = conv_in.ne[2];
    const auto required = gated_delta_net_layer_batch_workspace_capacity_bytes(
        width, batch, w.qkv.qtype, w.z.qtype, w.output.qtype, policy);
    const int tokens = width * batch;
    if (capacity < batch || capacity > 4)
        throw std::invalid_argument("gated_delta_net_layer_batch: state capacity");
    require_tensor(x, DType::BF16, {kHidden,width,batch,1}, "x");
    require_tensor(out, DType::BF16, {kHidden,width,batch,1}, "out");
    require_tensor(slots, DType::I32, {batch,1,1,1}, "slots");
    require_tensor(valid, DType::I32, {batch,1,1,1}, "valid_columns");
    require_projection_weight(w.qkv,kQkvRows,kHidden,false,"qkv");
    require_projection_weight(w.z,kValueRows,kHidden,false,"z");
    require_projection_weight(w.output,kHidden,kValueRows,false,"output");
    require_tensor(w.a,DType::FP32,{kHidden,kValueHeads,1,1},"a");
    require_tensor(w.b,DType::FP32,{kHidden,kValueHeads,1,1},"b");
    require_tensor(w.conv,DType::FP32,{4,kQkvRows,1,1},"conv");
    require_tensor(w.ssm_a,DType::FP32,{kValueHeads,1,1,1},"ssm_a");
    require_tensor(w.dt_bias,DType::FP32,{kValueHeads,1,1,1},"dt_bias");
    require_tensor(w.norm,DType::FP32,{kHeadDim,1,1,1},"norm");
    require_tensor(conv_in,DType::BF16,{kQkvRows,3,capacity,1},"conv in");
    require_tensor(conv_out,DType::BF16,{kQkvRows,3,capacity,1},"conv out");
    require_tensor(ssm_in,DType::FP32,{kHeadDim,kHeadDim,kValueHeads,capacity},"ssm in");
    require_tensor(ssm_out,DType::FP32,{kHeadDim,kHeadDim,kValueHeads,capacity},"ssm out");
    if (replay) {
        if (width < 2 || width > 16)
            throw std::invalid_argument("gated_delta_net_layer_batch: replay width");
        require_tensor(replay->conv,DType::BF16,{kQkvRows,width,batch,1},"conv record");
        require_tensor(replay->key,DType::BF16,{kHeadDim,kValueHeads,width,batch},"key record");
        require_tensor(replay->value,DType::BF16,{kHeadDim,kValueHeads,width,batch},"value record");
        require_tensor(replay->gate,DType::FP32,{2,kValueHeads,width,batch},"gate record");
    }
    if (!workspace.base() || workspace.used() > workspace.capacity() ||
        required > workspace.capacity() - workspace.used())
        throw std::invalid_argument("gated_delta_net_layer_batch: workspace capacity");
    std::array<AddressRange,24> ranges{};
    std::size_t count = 0;
    for (const auto* t : std::initializer_list<const Tensor*>{&x,&out,&slots,&valid,&w.a,&w.b,
                          &w.conv,&w.ssm_a,&w.dt_bias,&w.norm,&conv_in,&ssm_in})
        ranges[count++] = address_range(t->data,t->bytes(),"batch operand");
    for (const auto* weight : {&w.qkv,&w.z,&w.output})
        ranges[count++] = address_range(weight->payload,weight->payload_bytes,"batch weight");
    if (!exact_alias(conv_in,conv_out))
        ranges[count++] = address_range(conv_out.data,conv_out.bytes(),"conv output");
    if (!exact_alias(ssm_in,ssm_out))
        ranges[count++] = address_range(ssm_out.data,ssm_out.bytes(),"ssm output");
    if (replay) for (const auto* t : {&replay->conv,&replay->key,&replay->value,&replay->gate})
        ranges[count++] = address_range(t->data,t->bytes(),"batch record");
    ranges[count++] = address_range(workspace.base(),workspace.capacity(),"workspace");
    require_disjoint(std::span<const AddressRange>(ranges.data(),count));

    auto scope = workspace.scope();
    auto scratch = allocate_scratch(workspace,tokens,true);
    auto flat_x = x.reshape({kHidden,tokens});
    auto flat_out = out.reshape({kHidden,tokens});
    project(flat_x,w.qkv,scratch.projected_qkv,workspace,stream,policy.qkv);
    project(flat_x,w.z,scratch.z,workspace,stream,policy.z);
    detail::gated_delta_net_layer_control_launch(flat_x,w.a,w.b,w.ssm_a,w.dt_bias,
                                               scratch.g,scratch.beta,stream);
    detail::gated_delta_net_layer_conv_batch_launch(scratch.projected_qkv,w.conv,
        conv_in,conv_out,slots,valid,width,scratch.q,scratch.k,scratch.v,stream);
    if (replay) {
        for (const auto& pair : {std::pair{&replay->conv,&scratch.projected_qkv},
                                 std::pair{&replay->key,&scratch.k},
                                 std::pair{&replay->value,&scratch.v}})
            CUDA_CHECK(cudaMemcpyAsync(pair.first->data,pair.second->data,pair.first->bytes(),
                                       cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaMemcpy2DAsync(replay->gate.data,8,scratch.g.data,4,4,kValueHeads*tokens,
                                     cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(replay->gate.data)+4,8,
                                     scratch.beta.data,4,4,kValueHeads*tokens,
                                     cudaMemcpyDeviceToDevice,stream));
    }
    auto q=scratch.q.reshape({kHeadDim,kValueHeads,width,batch});
    auto k=scratch.k.reshape({kHeadDim,kValueHeads,width,batch});
    auto v=scratch.v.reshape({kHeadDim,kValueHeads,width,batch});
    auto g=scratch.g.reshape({kValueHeads,width,batch});
    auto beta=scratch.beta.reshape({kValueHeads,width,batch});
    auto recurrent=scratch.recurrent.reshape({kHeadDim,kValueHeads,width,batch});
    detail::gated_delta_net::launch_recurrent_batch_inout(q,k,v,g,beta,
        1.0F/std::sqrt(float(kHeadDim)),slots,valid,ssm_in,ssm_out,recurrent,stream);
    detail::gated_delta_net_layer_norm_launch(scratch.recurrent,scratch.z,w.norm,
                                             scratch.normalized_gated,stream);
    project(scratch.normalized_gated,w.output,flat_out,workspace,stream,policy.output);
}

} // namespace ninfer::ops

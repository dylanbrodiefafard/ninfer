#include "core/linear_attention_state.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_positive(std::int32_t value, const char* message) {
    if (value <= 0) { throw std::invalid_argument(message); }
}

void validate_layer_slot(const LinearAttentionStatePool& pool, std::uint32_t layer,
                         std::int32_t slot, const char* label) {
    if (layer >= pool.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

} // namespace

LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec) {
    if (spec.layers == 0) {
        throw std::invalid_argument("LinearAttentionStatePool layers must be nonzero");
    }
    if (spec.layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("LinearAttentionStatePool layer count exceeds int32");
    }
    validate_positive(spec.conv_channels,
                      "LinearAttentionStatePool conv_channels must be positive");
    validate_positive(spec.conv_width, "LinearAttentionStatePool conv_width must be positive");
    validate_positive(spec.value_heads, "LinearAttentionStatePool value_heads must be positive");
    validate_positive(spec.value_head_dim,
                      "LinearAttentionStatePool value_head_dim must be positive");
    validate_positive(spec.key_head_dim, "LinearAttentionStatePool key_head_dim must be positive");
    validate_positive(spec.slot_count, "LinearAttentionStatePool slot_count must be positive");
    if (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32) {
        throw std::invalid_argument("LinearAttentionStatePool conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count});

    LinearAttentionStatePoolLayout layout;
    layout.spec = spec;
    layout.conv.reserve(spec.layers);
    layout.recurrent.reserve(spec.layers);
    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string prefix = "Linear Attention layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        layout.recurrent.push_back(
            builder.add(recurrent_shape.bytes(), kArenaAlign, prefix + " recurrent"));
    }
    return layout;
}

LinearAttentionStatePool::LinearAttentionStatePool(DeviceSpan backing,
                                                   const LinearAttentionStatePoolLayout& layout)
    : spec(layout.spec) {
    if (layout.conv.empty() || layout.recurrent.size() != layout.conv.size() ||
        layout.conv.size() != spec.layers) {
        throw std::invalid_argument(
            "LinearAttentionStatePool layout layer counts are inconsistent");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count});
    conv.reserve(layout.conv.size());
    recurrent.reserve(layout.recurrent.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes() ||
            layout.recurrent[layer].bytes != recurrent_shape.bytes()) {
            throw std::logic_error(
                "LinearAttentionStatePool layout tensor byte size is inconsistent");
        }
        conv.emplace_back(layout.conv[layer].bind(backing).data, spec.conv_dtype,
                          std::initializer_list<std::int32_t>{spec.conv_channels, spec.conv_width,
                                                              spec.slot_count});
        recurrent.emplace_back(
            layout.recurrent[layer].bind(backing).data, DType::FP32,
            std::initializer_list<std::int32_t>{spec.key_head_dim, spec.value_head_dim,
                                                spec.value_heads, spec.slot_count});
    }
}

std::uint32_t LinearAttentionStatePool::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

std::int32_t LinearAttentionStatePool::slot_count() const noexcept { return spec.slot_count; }

std::int64_t LinearAttentionStatePool::conv_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.conv_channels) *
           static_cast<std::int64_t>(spec.conv_width);
}

std::int64_t LinearAttentionStatePool::recurrent_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.key_head_dim) *
           static_cast<std::int64_t>(spec.value_head_dim) *
           static_cast<std::int64_t>(spec.value_heads);
}

Tensor LinearAttentionStatePool::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool conv_slot");
    return conv.at(layer).slice(2, slot, 1).view({spec.conv_channels, spec.conv_width});
}

Tensor LinearAttentionStatePool::recurrent_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool recurrent_slot");
    return recurrent.at(layer)
        .slice(3, slot, 1)
        .view({spec.key_head_dim, spec.value_head_dim, spec.value_heads});
}

void LinearAttentionStatePool::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    validate_layer_slot(*this, 0, src, "LinearAttentionStatePool copy_slot source");
    validate_layer_slot(*this, 0, dst, "LinearAttentionStatePool copy_slot destination");
    if (src == dst) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = conv_slot(layer, src);
        const Tensor destination = conv_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = recurrent_slot(layer, src);
        const Tensor destination = recurrent_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LinearAttentionStatePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_layer_slot(*this, 0, slot, "LinearAttentionStatePool zero_slot");
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = conv_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = recurrent_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
}

} // namespace ninfer

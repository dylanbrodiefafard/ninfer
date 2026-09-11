#include "targets/qwen3_6/impl/runtime/tool_masks.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {
constexpr std::size_t kWords = (kTokenDomain + 31) / 32;
}

ToolMaskExchange::ToolMaskExchange(Tensor masks, Tensor sampling)
    : masks_(masks), sampling_(sampling), width_(masks.ne[1]), capacity_(masks.ne[2]),
      host_masks_(masks.bytes()), host_sampling_(sampling.bytes()),
      host_ids_(width_ * capacity_ * sizeof(TokenId)),
      host_parents_(width_ * capacity_ * sizeof(std::int32_t)),
      host_counts_(capacity_ * sizeof(std::int32_t)) {
    if (masks.dtype != DType::I32 || masks.ne[0] != kWords ||
        !masks.is_contiguous() || width_ == 0 || capacity_ == 0 ||
        sampling.bytes() < capacity_ * sizeof(ops::SamplingConfig)) {
        throw std::invalid_argument("invalid tool mask exchange storage");
    }
    outputs_.resize(capacity_, nullptr);
    configs_.resize(capacity_);
    std::fill_n(static_cast<std::uint32_t*>(host_masks_.data()), masks.bytes() / 4,
                ~std::uint32_t{0});
}

void ToolMaskExchange::bind(std::span<const OutputSession* const> outputs,
                            std::span<const ops::SamplingConfig> sampling) {
    if (outputs.empty() || outputs.size() > capacity_ || outputs.size() != sampling.size()) {
        throw std::invalid_argument("invalid tool mask batch bindings");
    }
    outputs_.assign(outputs.begin(), outputs.end());
    configs_.assign(sampling.begin(), sampling.end());
    error_ = {};
}

std::uint32_t* ToolMaskExchange::host_mask(std::size_t row) const {
    return static_cast<std::uint32_t*>(host_masks_.data()) + row * width_ * kWords;
}

const std::uint32_t* ToolMaskExchange::device_mask(std::size_t row) const {
    return static_cast<const std::uint32_t*>(masks_.data) + row * width_ * kWords;
}

ops::SamplingConfig ToolMaskExchange::root(std::size_t row, cudaStream_t stream) {
    auto config = configs_.at(row);
    config.allowed_token_words = nullptr;
    config.allowed_token_column_stride = 0;
    if (outputs_[row] && outputs_[row]->has_tool_grammar()) {
        const std::array<TokenId, 1> token{0};
        const std::array<std::int32_t, 1> parent{-1};
        outputs_[row]->fill_tool_masks(token, parent, {host_mask(row), kWords});
        CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint32_t*>(device_mask(row)), host_mask(row),
                                   kWords * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
        config.allowed_token_words = device_mask(row);
    }
    return config;
}

const ops::SamplingConfig* ToolMaskExchange::enqueue(
    const Tensor& ids, const Tensor* parents, const Tensor& valid_columns, cudaStream_t stream) {
    const auto batch = static_cast<std::size_t>(ids.ne[1]);
    const auto width = static_cast<std::size_t>(ids.ne[0]);
    if (width > width_ || batch > capacity_ || ids.dtype != DType::I32 ||
        valid_columns.dtype != DType::I32 || valid_columns.ne[0] != ids.ne[1]) {
        throw std::invalid_argument("invalid speculative tool mask inputs");
    }
    // Captured graphs must not store per-launch stack addresses. Shape-specific
    // copies use the fixed max-width pitch; counts delimit valid nodes at replay.
    CUDA_CHECK(cudaMemcpy2DAsync(host_ids_.data(), width_ * sizeof(TokenId), ids.data, ids.nb[1],
                                width * sizeof(TokenId), batch, cudaMemcpyDeviceToHost, stream));
    if (parents) {
        CUDA_CHECK(cudaMemcpy2DAsync(host_parents_.data(), width_ * sizeof(std::int32_t),
                                    parents->data, parents->nb[1], width * sizeof(std::int32_t),
                                    batch, cudaMemcpyDeviceToHost, stream));
    }
    // tree_ is engine-wide for a graph family, but cannot be assigned at capture
    // and assumed to change at replay. Encode it into the callback identity.
    const auto chain_callback = +[](void* opaque) {
        auto& self = *static_cast<ToolMaskExchange*>(opaque);
        self.tree_ = false;
        match(opaque);
    };
    const auto tree_callback = +[](void* opaque) {
        auto& self = *static_cast<ToolMaskExchange*>(opaque);
        self.tree_ = true;
        match(opaque);
    };
    CUDA_CHECK(cudaMemcpyAsync(host_counts_.data(), valid_columns.data,
                               batch * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaLaunchHostFunc(stream, parents ? tree_callback : chain_callback, this));
    CUDA_CHECK(cudaMemcpyAsync(masks_.data, host_masks_.data(), masks_.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(sampling_.data, host_sampling_.data(), sampling_.bytes(),
                               cudaMemcpyHostToDevice, stream));
    return static_cast<const ops::SamplingConfig*>(sampling_.data);
}

void CUDART_CB ToolMaskExchange::match(void* opaque) noexcept {
    auto& self = *static_cast<ToolMaskExchange*>(opaque);
    try { self.fill(); }
    catch (...) {
        self.error_ = std::current_exception();
        // CUDA cannot be aborted from a host node. Finish the round safely with
        // nonempty domains, then fail before any candidate can be published.
        auto* configs = static_cast<ops::SamplingConfig*>(self.host_sampling_.data());
        for (std::size_t row = 0; row < self.configs_.size(); ++row) {
            configs[row] = self.configs_[row];
            configs[row].allowed_token_words = nullptr;
            configs[row].allowed_token_column_stride = 0;
        }
    }
}

void ToolMaskExchange::fill() {
    auto* configs = static_cast<ops::SamplingConfig*>(host_sampling_.data());
    const auto* counts = static_cast<const std::int32_t*>(host_counts_.data());
    const auto* ids = static_cast<const TokenId*>(host_ids_.data());
    auto* parents = static_cast<std::int32_t*>(host_parents_.data());
    for (std::size_t row = 0; row < outputs_.size(); ++row) {
        configs[row] = configs_[row];
        configs[row].allowed_token_words = nullptr;
        configs[row].allowed_token_column_stride = 0;
        if (!outputs_[row] || !outputs_[row]->has_tool_grammar()) { continue; }
        if (counts[row] < 1 || static_cast<std::size_t>(counts[row]) > width_) {
            throw std::logic_error("invalid tool grammar verification node count");
        }
        const auto count = static_cast<std::size_t>(counts[row]);
        if (!tree_) {
            for (std::size_t node = 0; node < count; ++node) {
                parents[row * width_ + node] = static_cast<std::int32_t>(node) - 1;
            }
        }
        std::fill_n(host_mask(row), width_ * kWords, ~std::uint32_t{0});
        outputs_[row]->fill_tool_masks({ids + row * width_, count},
                                       {parents + row * width_, count},
                                       {host_mask(row), count * kWords});
        configs[row].allowed_token_words = device_mask(row);
        configs[row].allowed_token_column_stride = kWords;
    }
}

void ToolMaskExchange::rethrow_error() const {
    if (error_) { std::rethrow_exception(error_); }
}

} // namespace ninfer::targets::qwen3_6

// ninfer::ops - sigmoid_mul wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/sigmoid_mul.h"

#include "ops/common/a4_activation_check.h"
#include "ops/common/a8_activation_check.h"
#include "ops/launcher/sigmoid_gate_mul.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("sigmoid_mul: gate/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("sigmoid_mul: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != x.ne[d]) {
            throw std::invalid_argument("sigmoid_mul: gate/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (!gate.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be contiguous");
    }
    if (gate.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("sigmoid_mul: gate/x data must be non-null");
    }

    detail::sigmoid_gate_mul_launch(gate, x, stream); // single variant -> direct dispatch
}

void sigmoid_mul_a8(const Tensor& gate, const Tensor& x, A8Activation& activation,
                    cudaStream_t stream) {
    constexpr std::int32_t kRows = 6144;
    const std::int32_t tokens    = x.ne[1];
    const auto valid             = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == kRows && t.ne[1] == tokens && t.ne[2] == 1 &&
               t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
    };
    if (tokens <= 0 || !valid(gate) || !valid(x)) {
        throw std::invalid_argument(
            "sigmoid_mul_a8: gate/x must be aligned contiguous BF16 [6144,T] with T>0");
    }
    if (detail::validate_a8_activation(activation, kRows, "sigmoid_mul_a8") != tokens ||
        activation.codes.ne[1] != tokens) {
        throw std::invalid_argument("sigmoid_mul_a8: activation must be [6144,T]");
    }
    detail::sigmoid_gate_mul_a8_launch(gate, x, activation, stream);
}

void sigmoid_mul_a4(const Tensor& gate, const Tensor& x, A4Activation& activation,
                    cudaStream_t stream) {
    constexpr std::int32_t kRows = 6144;
    const std::int32_t tokens    = x.ne[1];
    const auto valid             = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == kRows && t.ne[1] == tokens && t.ne[2] == 1 &&
               t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
    };
    if (tokens <= 0 || !valid(gate) || !valid(x)) {
        throw std::invalid_argument(
            "sigmoid_mul_a4: gate/x must be aligned contiguous BF16 [6144,T] with T>0");
    }
    if (detail::validate_a4_activation(activation, kRows, "sigmoid_mul_a4") != tokens) {
        throw std::invalid_argument("sigmoid_mul_a4: activation must be [6144,T]");
    }
    detail::sigmoid_gate_mul_a4_launch(gate, x, activation, stream);
}

} // namespace ninfer::ops

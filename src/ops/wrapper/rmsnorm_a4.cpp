#include "ninfer/ops/rmsnorm.h"
#include "ops/common/a4_activation_check.h"
#include "ops/launcher/rmsnorm_a4.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
void rmsnorm_a4(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                A4Activation& activation, cudaStream_t stream) {
    detail::validate_a4_activation(activation);
    if (activation.rows != 5120) {
        throw std::invalid_argument("rmsnorm_a4: activation must have K5120");
    }
    const auto matrix = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == 5120 && t.ne[1] == activation.tokens &&
               t.ne[2] == 1 && t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t.data) % 16) == 0;
    };
    if (!matrix(x) || (normalized != nullptr && !matrix(*normalized)) ||
        weight.dtype != DType::BF16 || weight.ne[0] != 5120 || weight.ne[1] != 1 ||
        weight.ne[2] != 1 || weight.ne[3] != 1 || !weight.is_contiguous() ||
        weight.data == nullptr || (reinterpret_cast<std::uintptr_t>(weight.data) % 4) != 0 ||
        !(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_a4: invalid BF16 inputs or eps");
    }
    const std::array<DeviceSpan, 3> outputs{
        activation.codes,
        activation.scales,
        normalized != nullptr ? DeviceSpan{normalized->data, normalized->bytes()} : DeviceSpan{},
    };
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].data == nullptr) { continue; }
        detail::require_disjoint(x.data, x.bytes(), outputs[i].data, outputs[i].bytes);
        detail::require_disjoint(weight.data, weight.bytes(), outputs[i].data, outputs[i].bytes);
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (outputs[j].data != nullptr) {
                detail::require_disjoint(outputs[i].data, outputs[i].bytes, outputs[j].data,
                                         outputs[j].bytes);
            }
        }
    }
    detail::rmsnorm_a4_launch(x, weight, eps, normalized, activation, stream);
}
} // namespace ninfer::ops

// ninfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include "ops/common/a4_activation_check.h"
#include "ops/common/a8_activation_check.h"
#include "ops/launcher/rmsnorm.h"

#include <cmath>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

std::int64_t validate_rmsnorm(const Tensor& x, const Tensor& weight, float eps, const Tensor* z,
                              const Tensor& out) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    (void)numel_allow_zero(weight, "weight");
    require_same_shape(x, out, "out");
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "z");
        require_same_shape(x, *z, "z");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return 0; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    return n;
}

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                  const Tensor* z, Tensor& out, cudaStream_t stream) {
    if (validate_rmsnorm(x, weight, eps, z, out) == 0) { return; }
    detail::rmsnorm_launch(x, weight, eps, unit_offset, z, out, stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, out, stream);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, false, &z, out, stream);
}

namespace {

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) % 16) == 0;
}

void require_bf16_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns,
                         const char* message) {
    if (t.dtype != DType::BF16 || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr || !aligned16(t.data)) {
        throw std::invalid_argument(message);
    }
}

void require_bf16_vector(const Tensor& t, std::int32_t elements, const char* message) {
    if (t.dtype != DType::BF16 || t.numel() != elements || t.ne[0] != elements ||
        !t.is_contiguous() || t.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(t.data) % 4) != 0) {
        throw std::invalid_argument(message);
    }
}

void require_eps(float eps, const char* message) {
    if (!(eps > 0.0f) || !std::isfinite(eps)) { throw std::invalid_argument(message); }
}

} // namespace

void rmsnorm_a8(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                A8Activation& activation, cudaStream_t stream) {
    constexpr std::int32_t kD = 5120;
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("rmsnorm_a8: T must be positive"); }
    require_bf16_matrix(x, kD, tokens, "rmsnorm_a8: x must be aligned contiguous BF16 [5120,T]");
    require_bf16_vector(weight, kD, "rmsnorm_a8: weight must be contiguous BF16 [5120]");
    if (normalized != nullptr) {
        require_bf16_matrix(*normalized, kD, tokens,
                            "rmsnorm_a8: normalized must be aligned contiguous BF16 [5120,T]");
    }
    require_eps(eps, "rmsnorm_a8: eps must be positive and finite");
    if (detail::validate_a8_activation(activation, kD, "rmsnorm_a8") != tokens ||
        activation.codes.ne[1] != tokens) {
        throw std::invalid_argument("rmsnorm_a8: activation must be [5120,T]");
    }
    detail::rmsnorm_a8_launch(x, weight, eps, normalized, activation, stream);
}

void rmsnorm_a4(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                A4Activation& activation, cudaStream_t stream) {
    constexpr std::int32_t kD = 5120;
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("rmsnorm_a4: T must be positive"); }
    require_bf16_matrix(x, kD, tokens, "rmsnorm_a4: x must be aligned contiguous BF16 [5120,T]");
    require_bf16_vector(weight, kD, "rmsnorm_a4: weight must be contiguous BF16 [5120]");
    if (normalized != nullptr) {
        require_bf16_matrix(*normalized, kD, tokens,
                            "rmsnorm_a4: normalized must be aligned contiguous BF16 [5120,T]");
    }
    require_eps(eps, "rmsnorm_a4: eps must be positive and finite");
    if (detail::validate_a4_activation(activation, kD, "rmsnorm_a4") != tokens) {
        throw std::invalid_argument("rmsnorm_a4: activation must be [5120,T]");
    }
    detail::rmsnorm_a4_launch(x, weight, eps, normalized, activation, stream);
}

void gated_rmsnorm_a8(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                      A8Activation& activation, cudaStream_t stream) {
    constexpr std::int32_t kHeadDim = 128;
    constexpr std::int32_t kHeads   = 48;
    const std::int32_t tokens       = x.ne[2];
    const auto valid_heads          = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == kHeadDim && t.ne[1] == kHeads &&
               t.ne[2] == tokens && t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               aligned16(t.data);
    };
    if (tokens <= 0 || !valid_heads(x) || !valid_heads(z)) {
        throw std::invalid_argument(
            "gated_rmsnorm_a8: x/z must be aligned contiguous BF16 [128,48,T] with T>0");
    }
    require_bf16_vector(weight, kHeadDim, "gated_rmsnorm_a8: weight must be contiguous BF16 [128]");
    require_eps(eps, "gated_rmsnorm_a8: eps must be positive and finite");
    if (detail::validate_a8_activation(activation, kHeadDim * kHeads, "gated_rmsnorm_a8") !=
            tokens ||
        activation.codes.ne[1] != tokens) {
        throw std::invalid_argument("gated_rmsnorm_a8: activation must be [6144,T]");
    }
    detail::gated_rmsnorm_a8_launch(x, weight, z, eps, activation, stream);
}

void gated_rmsnorm_a4(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                      A4Activation& activation, cudaStream_t stream) {
    constexpr std::int32_t kHeadDim = 128;
    constexpr std::int32_t kHeads   = 48;
    const std::int32_t tokens       = x.ne[2];
    const auto valid_heads          = [&](const Tensor& t) {
        return t.dtype == DType::BF16 && t.ne[0] == kHeadDim && t.ne[1] == kHeads &&
               t.ne[2] == tokens && t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               aligned16(t.data);
    };
    if (tokens <= 0 || !valid_heads(x) || !valid_heads(z)) {
        throw std::invalid_argument(
            "gated_rmsnorm_a4: x/z must be aligned contiguous BF16 [128,48,T] with T>0");
    }
    require_bf16_vector(weight, kHeadDim, "gated_rmsnorm_a4: weight must be contiguous BF16 [128]");
    require_eps(eps, "gated_rmsnorm_a4: eps must be positive and finite");
    if (detail::validate_a4_activation(activation, kHeadDim * kHeads, "gated_rmsnorm_a4") !=
        tokens) {
        throw std::invalid_argument("gated_rmsnorm_a4: activation must be [6144,T]");
    }
    detail::gated_rmsnorm_a4_launch(x, weight, z, eps, activation, stream);
}

void dual_offset_rmsnorm(const Tensor& x0, const Tensor& weight0, const Tensor& x1,
                         const Tensor& weight1, float eps, Tensor& out0, Tensor& out1,
                         cudaStream_t stream) {
    if (x0.ne[0] != 5120 || x0.ne[1] <= 0 || x0.ne[2] != 1 || x0.ne[3] != 1) {
        throw std::invalid_argument("dual_offset_rmsnorm: inputs must be [5120,T], T positive");
    }
    require_same_shape(x0, x1, "second input");
    (void)validate_rmsnorm(x0, weight0, eps, nullptr, out0);
    (void)validate_rmsnorm(x1, weight1, eps, nullptr, out1);

    const std::array<const Tensor*, 6> operands{&x0, &weight0, &x1, &weight1, &out0, &out1};
    std::array<std::uintptr_t, 6> starts{};
    std::array<std::uintptr_t, 6> ends{};
    for (std::size_t index = 0; index < operands.size(); ++index) {
        starts[index]    = reinterpret_cast<std::uintptr_t>(operands[index]->data);
        const auto bytes = operands[index]->bytes();
        if ((starts[index] & (alignof(std::uint16_t) - 1)) != 0) {
            throw std::invalid_argument("dual_offset_rmsnorm: pointers must be BF16-aligned");
        }
        if (starts[index] > std::numeric_limits<std::uintptr_t>::max() - bytes) {
            throw std::overflow_error("dual_offset_rmsnorm: storage interval overflows uintptr");
        }
        ends[index] = starts[index] + bytes;
        for (std::size_t other = 0; other < index; ++other) {
            if (starts[index] < ends[other] && starts[other] < ends[index]) {
                throw std::invalid_argument("dual_offset_rmsnorm: operands must not overlap");
            }
        }
    }
    detail::dual_offset_rmsnorm_launch(x0, weight0, x1, weight1, eps, out0, out1, stream);
}

} // namespace ninfer::ops

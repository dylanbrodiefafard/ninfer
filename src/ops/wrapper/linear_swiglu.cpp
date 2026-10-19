#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm_linear_swiglu.h"
#include "ops/common/a4_activation_check.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include <cstdint>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: W8 admits only A16");
        }
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: Q4 admits only A16");
        }
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool w8_shape    = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                             gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                             gate_up_weight.padded_shape[1] == 2048;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !w8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight    = large_shape && gate_up_weight.qtype == QType::Q4G64_F16S &&
                              gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                              common_row_split;
    const bool w8_weight    = w8_shape && gate_up_weight.qtype == QType::W8G32_F16S &&
                              gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
                              gate_up_weight.qhigh == nullptr &&
                              gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16S;
    if (!q4_weight && !w8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("linear_swiglu: Q4/W8 admit only A16");
    }
    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, w8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (w8_weight) {
        detail::w8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

namespace {

constexpr std::int32_t kA4Hidden       = 5120;
constexpr std::int32_t kA4Intermediate = 17408;

// Validates the A4 input and NVFP4 [34816,5120] gate/up weight of the A4 overloads; returns T.
std::int32_t validate_a4_swiglu_input(const A4Activation& x, const Weight& gate_up_weight) {
    const std::int32_t t = detail::validate_a4_activation(x, kA4Hidden, "linear_swiglu");
    if (t < kA4MlpGateUpMinTokens) {
        throw std::invalid_argument("linear_swiglu: A4 activation must be [5120,T] with T >= 2");
    }
    if (gate_up_weight.qtype != QType::NVFP4 || gate_up_weight.n != 2 * kA4Intermediate ||
        gate_up_weight.k != kA4Hidden) {
        throw std::invalid_argument("linear_swiglu: an A4 activation requires NVFP4 [34816,5120]");
    }
    detail::validate_nvfp4_weight(gate_up_weight, "linear_swiglu");
    if (x.input_scale_divisor != gate_up_weight.input_scale_divisor) {
        throw std::invalid_argument(
            "linear_swiglu: A4 activation input scale divisor differs from the weight's");
    }
    return t;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

detail::Nvfp4W4a4Workspace a4_storage(const A4Activation& activation) {
    return {static_cast<std::uint8_t*>(activation.codes.data),
            static_cast<std::uint8_t*>(activation.scales.data)};
}

} // namespace

void linear_swiglu(const A4Activation& x, const Weight& gate_up_weight, Tensor& out,
                   cudaStream_t stream) {
    const std::int32_t t = validate_a4_swiglu_input(x, gate_up_weight);
    if (out.dtype != DType::BF16 || out.ne[0] != kA4Intermediate || out.ne[1] != t ||
        out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || !aligned_to(out.data, 16)) {
        throw std::invalid_argument(
            "linear_swiglu: out must be contiguous 16-byte aligned BF16 [17408,T]");
    }
    if (overlaps(x.codes, out) || overlaps(x.scales, out)) {
        throw std::invalid_argument("linear_swiglu: out overlaps the A4 activation");
    }
    detail::nvfp4_linear_swiglu_w4a4_project(gate_up_weight, t, a4_storage(x), out, stream);
}

void linear_swiglu(const A4Activation& x, const Weight& gate_up_weight, A4Activation& out,
                   cudaStream_t stream) {
    const std::int32_t t = validate_a4_swiglu_input(x, gate_up_weight);
    if (detail::validate_a4_activation(out, kA4Intermediate, "linear_swiglu") != t) {
        throw std::invalid_argument("linear_swiglu: A4 output must be [17408,T]");
    }
    if (overlaps(x.codes, out.codes) || overlaps(x.codes, out.scales) ||
        overlaps(x.scales, out.codes) || overlaps(x.scales, out.scales)) {
        throw std::invalid_argument("linear_swiglu: A4 output overlaps the input");
    }
    detail::nvfp4_linear_swiglu_w4a4_project_a4(gate_up_weight, t, a4_storage(x), a4_storage(out),
                                                out.input_scale_divisor, stream);
}

std::size_t rmsnorm_linear_swiglu_workspace_capacity_bytes(std::int32_t tokens) {
    return detail::fp8_a8_workspace_capacity_bytes(tokens, 5120);
}

void rmsnorm_linear_swiglu(const Tensor& x, const Tensor& norm_weight, float eps,
                           const Weight& gate_up, Tensor& out, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    (void)rmsnorm_linear_swiglu_workspace_capacity_bytes(x.ne[1]);
    if (x.dtype != DType::BF16 || norm_weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        x.ne[0] != 5120 || x.ne[2] != 1 || x.ne[3] != 1 || norm_weight.ne[0] != 5120 ||
        norm_weight.numel() != 5120 || out.ne[0] != 17408 || out.ne[1] != x.ne[1] ||
        out.ne[2] != 1 || out.ne[3] != 1 || !x.is_contiguous() || !norm_weight.is_contiguous() ||
        !out.is_contiguous() || !aligned_to(x.data, 16) || !aligned_to(out.data, 16) ||
        !aligned_to(norm_weight.data, 4) || gate_up.qtype != QType::NVFP4 || gate_up.n != 34816 ||
        gate_up.k != 5120 || !(eps > 0) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_linear_swiglu: invalid normalized NVFP4/A8 geometry");
    }
    (void)detail::validate_nvfp4_weight(gate_up, "rmsnorm_linear_swiglu");
    detail::nvfp4_rmsnorm_linear_swiglu_launch(x, norm_weight, eps, gate_up, out, workspace,
                                               stream);
}

} // namespace ninfer::ops

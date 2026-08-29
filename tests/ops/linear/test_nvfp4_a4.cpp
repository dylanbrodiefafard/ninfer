#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <span>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a4() {
    constexpr std::array invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{6, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{10, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{12, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{15, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{18, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{36, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array<std::int32_t, 1> packed_col0{2};
    int failures = 0;
    for (const auto& shape : {
             ShapeCase{14336, 5120, 719U, Comparison::Sampled, true, invocations},
             ShapeCase{16384, 5120, 721U, Comparison::Sampled, true, invocations},
             ShapeCase{34816, 5120, 722U, Comparison::Sampled, true, invocations},
             ShapeCase{5120, 6144, 723U, Comparison::Sampled, true, invocations},
             ShapeCase{5120, 17408, 725U, Comparison::Sampled, true, invocations},
             ShapeCase{5120, 10240, 727U, Comparison::Sampled, true, invocations},
         }) {
        failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight, shape);
    }
    failures += run_packed_column0_matches_decode(
        "NVFP4_A4 packed-col0 [14336,5120]", make_nvfp4_weight, 14336, 5120, 719U,
        ops::LinearPolicy::AllowA4, packed_col0);
    failures += run_packed_column0_matches_decode(
        "NVFP4_A4 packed-col0 [5120,6144]", make_nvfp4_weight, 5120, 6144, 723U,
        ops::LinearPolicy::AllowA4, packed_col0);
    failures += run_packed_column0_matches_decode(
        "NVFP4_A4 packed-col0 [5120,17408]", make_nvfp4_weight, 5120, 17408, 725U,
        ops::LinearPolicy::AllowA4, packed_col0);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    try {
        const int failures = run_nvfp4_a4();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}

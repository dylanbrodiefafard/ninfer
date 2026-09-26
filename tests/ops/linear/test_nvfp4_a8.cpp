#include "ops/linear/linear_test_common.h"
#include <array>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear;
    if (!cuda_available()) { return 1; }
    constexpr std::array invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{6, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{10, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{12, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{15, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{18, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{20, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{30, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{36, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{36, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = 0;
    for (const auto& shape : {
        ShapeCase{14336, 5120, 719U, Comparison::Sampled, true, invocations},
        ShapeCase{16384, 5120, 721U, Comparison::Sampled, true, invocations},
        ShapeCase{34816, 5120, 722U, Comparison::Sampled, true, invocations},
        ShapeCase{5120, 6144, 723U, Comparison::Sampled, true, invocations},
        ShapeCase{5120, 17408, 725U, Comparison::Sampled, true, invocations},
    }) {
        failures += run_shape("NVFP4_A8", ActivationCompute::A8, make_nvfp4_weight, shape);
    }
    std::cout << (failures ? "FAIL" : "OK") << " NVFP4_A8 Linear\n";
    return failures ? 1 : 0;
}

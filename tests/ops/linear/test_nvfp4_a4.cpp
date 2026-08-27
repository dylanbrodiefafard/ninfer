#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

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
        Invocation{18, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{36, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{4096, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 10240, 727U, Comparison::Sampled, true, invocations});
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

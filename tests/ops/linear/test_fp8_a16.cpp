#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

// Generic FP8 Linear registers only the vocabulary head, which keeps A16 under every policy.
int run_fp8_a16() {
    std::vector<Invocation> vocabulary_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{96, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{97, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{160, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{161, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    for (int t = 1; t <= 41; ++t)
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    for (int t : {7, 25, 41, 65, 128})
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    int failures =
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {248320, 5120, 823U, Comparison::Sampled, true, vocabulary_invocations});
    constexpr std::array vocabulary_policies{
        ops::LinearPolicy::A16Only,
        ops::LinearPolicy::AllowA8,
        ops::LinearPolicy::AllowA4,
    };
    for (const ops::LinearPolicy policy : vocabulary_policies) {
        try {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, 248320, 5120, policy, 1, 2048);
            if (capacity != 0) {
                std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "FP8 vocabulary policy was rejected: " << error.what() << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}

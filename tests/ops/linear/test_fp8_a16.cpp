#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{11, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                             {14336, 5120, 811U, Comparison::Sampled, true, attn_invocations});
    constexpr std::array gdn_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{10, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 817U, Comparison::Sampled, true, gdn_invocations});
    constexpr std::array mlp_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 821U, Comparison::Sampled, true, mlp_invocations});
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
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
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
    constexpr std::array residual6144_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {5120, 6144, 827U, Comparison::Sampled, true, residual6144_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures +=
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {5120, 17408, 829U, Comparison::Sampled, true, residual17408_invocations});

    struct AddedProblem {
        std::int32_t n;
        std::int32_t k;
        std::int32_t mma_first_t;
    };
    constexpr std::array added_shapes{
        AddedProblem{10240, 2560, 17}, AddedProblem{6144, 2560, 21},
        AddedProblem{12288, 2560, 21}, AddedProblem{512, 2560, 29},
        AddedProblem{2560, 6144, 37},  AddedProblem{640, 2560, 29},
        AddedProblem{1280, 2560, 29},  AddedProblem{2560, 640, 17},
        AddedProblem{320, 10240, 45},  AddedProblem{10240, 320, 9},
        AddedProblem{2560, 2560, 29},
    };
    std::uint32_t added_seed = 881U;
    for (const auto& shape : added_shapes) {
        const std::array invocations{
            Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
            Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{shape.mma_first_t - 1, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{shape.mma_first_t, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{512, CallForm::Policy, ops::LinearPolicy::A16Only},
            Invocation{4096, CallForm::Policy, ops::LinearPolicy::A16Only},
        };
        failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                              {shape.n, shape.k, added_seed, Comparison::Sampled, true,
                               invocations});
        added_seed += 2U;
    }

    constexpr std::array vocabulary2560_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {248320, 2560, 907U, Comparison::Sampled, true,
                           vocabulary2560_invocations});
    for (const ops::LinearPolicy policy : vocabulary_policies) {
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 248320, 2560, policy, 1, 2048);
        if (capacity != 0) {
            std::cerr << "FP8 K=2560 vocabulary A16 route reported nonzero workspace\n";
            ++failures;
        }
    }

    constexpr std::array full_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{6, CallForm::Policy, ops::LinearPolicy::A16Only, true},
    };
    failures += run_shape("FP8_A16_FULL", ActivationCompute::A16, make_fp8_weight,
                          {5120, 6144, 839U, Comparison::Full, true, full_invocations});
    constexpr std::array batches{1, 2, 3, 4};
    failures += run_packed_sequences_matches_panels("FP8_PACKED", make_fp8_weight,
                                                     14336, 5120, 841U, 6, batches);
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

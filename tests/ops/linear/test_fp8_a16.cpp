#include "ops/input_projection_test_common.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

// Distinct rows make a misrouted output-row range visible to the oracle.
test::quantized_weight::PackedWeight make_distinct_fp8_weight(std::int32_t n, std::int32_t k,
                                                              std::uint32_t seed) {
    return test::input_projection::with_distinct_fp8_rows(make_fp8_weight(n, k, seed));
}

// Generic FP8 Linear registers the vocabulary heads and the Flash-Next GDN/attention output, which
// keep A16 under every policy.
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
    // The [248320,2560] head: one width in every small-T tile tier (8 to 48 columns), the first
    // GEMM width, each GEMM schedule partial and full, and 128-token prefixes with tails.
    const std::vector<Invocation> vocabulary_2560_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{96, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{127, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::A16Only, true},
    };
    failures +=
        run_shape("FP8_A16 K=2560", ActivationCompute::A16, make_fp8_weight,
                  {248320, 2560, 827U, Comparison::Sampled, true, vocabulary_2560_invocations});
    // The Flash-Next GDN/attention output [2560,6144], compared in full: the T=1 GEMV, every SIMT
    // width T=2..8, the 16-, 24-, 32-, and 48-column small-T MMA tiles, the 32x64 GEMM through
    // T=384, and the 64x64 GEMM beyond, each on both sides of its boundary.
    const std::vector<Invocation> mixer_out_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{5, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{6, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{7, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{12, CallForm::Policy, ops::LinearPolicy::A16Only, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{127, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{1000, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    failures += run_shape("FP8_A16 Flash-Next mixer output", ActivationCompute::A16,
                          make_distinct_fp8_weight,
                          {2560, 6144, 829U, Comparison::Full, true, mixer_out_invocations});
    constexpr std::array vocabulary_policies{
        ops::LinearPolicy::A16Only,
        ops::LinearPolicy::AllowA8,
        ops::LinearPolicy::AllowA4,
    };
    for (const ops::LinearPolicy policy : vocabulary_policies) {
        for (const std::int32_t hidden : {5120, 2560}) {
            try {
                const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16S, 248320, hidden, policy, 1, 2048);
                if (capacity != 0) {
                    std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                    ++failures;
                }
            } catch (const std::exception& error) {
                std::cerr << "FP8 vocabulary policy was rejected: " << error.what() << '\n';
                ++failures;
            }
        }
        try {
            if (ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, 2560, 6144,
                                                     policy, 1, 4096) != 0) {
                std::cerr << "FP8 Flash-Next mixer output reported nonzero workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "FP8 Flash-Next mixer output policy was rejected: " << error.what()
                      << '\n';
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

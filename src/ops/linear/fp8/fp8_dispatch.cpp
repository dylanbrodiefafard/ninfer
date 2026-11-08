#include "ops/linear/fp8/fp8_dispatch.h"

#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Every policy is admitted; none enables a lower-precision vocabulary route.
void validate_fp8_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("fp8 linear: unsupported policy");
}

// Generic FP8 Linear serves only the vocabulary head, which retains BF16 activation compute under
// every policy; the Text projections run through their fused Ops.
void require_vocabulary_problem(std::int32_t output_rows, std::int32_t input_rows,
                                std::int32_t tokens) {
    if (tokens <= 0 || !is_fp8_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("fp8 linear: unsupported shape");
    }
    if (resolve_fp8_problem(output_rows, input_rows) != Fp8Problem::Vocabulary) {
        throw std::invalid_argument("fp8 linear: only the vocabulary problem is registered");
    }
}

void launch_vocabulary(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] >= kFp8VocabularyFirstA16GemmT) {
        launch_fp8_vocabulary_a16_gemm(x, weight, out, stream);
        return;
    }
    // Every width below the GEMM crossover is one SmallT MMA launch.
    static_assert(kFp8VocabularyLastA16SmallTMmaT >= kFp8VocabularyFirstA16GemmT - 1);
    launch_fp8_vocabulary_a16_small_t(x, weight, out, stream);
}

} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    }
    validate_fp8_policy(policy);
    require_vocabulary_problem(output_rows, input_rows, max_tokens);
    return 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    validate_fp8_policy(policy);
    require_vocabulary_problem(weight.n, weight.k, x.ne[1]);
    launch_vocabulary(x, weight, out, stream);
}

} // namespace ninfer::ops::detail

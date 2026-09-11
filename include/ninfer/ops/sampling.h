#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// L∞ logit perturbation used for the p-less membership cut. First-order softmax
// maps ||Δz||_∞ ≤ ε to a relative p factor exp(2ε/T): admit v when
// p_v ≥ L · exp(-2ε/T) ≡ L/(1+α). L itself is the unperturbed collision
// probability and is not rebuilt. ε is one BF16 ulp at logit magnitude 8–16.
inline constexpr float kPLessLogitPerturbation = 0.0625f;

#if !defined(__CUDA_ARCH__)
[[nodiscard]] inline double p_less_admission_scale(double temperature) noexcept {
    if (!(temperature > 0.0)) { return 1.0; }
    return std::exp(2.0 * static_cast<double>(kPLessLogitPerturbation) / temperature);
}
#endif

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] occurrence-count array used by both penalties. The first
// suppressed_token_count entries in suppressed_tokens are excluded from selection;
// callers must keep the count in [0, kMaximumSuppressedTokens] and leave at least one token in the
// sampling domain eligible.
struct SamplingConfig {
    static constexpr std::int32_t kMaximumSuppressedTokens = 4;

    float temperature          = 0.0f; // <= 0 => greedy argmax (bit-identical to argmax())
    std::int32_t top_k         = 0;    // clamped to 20: top_k <= 0 or top_k > 20 => 20
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    std::int32_t p_less        = 0; // != 0 => p-less; ignore top_k/top_p/min_p/penalties
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
    std::int32_t suppressed_token_count = 0;
    std::int32_t suppressed_tokens[kMaximumSuppressedTokens] = {-1, -1, -1, -1};
    // Cycle-exit continuation, or -1. Not a suppressed_tokens member: V and L are
    // computed on the eligible domain as usual. Greedy (temperature<=0) ignores it.
    std::int32_t typical_exclude = -1;
    // Optional caller-owned device eligibility bitset, one bit per vocabulary token.
    // Null means unrestricted. One means allowed, intersected with suppressed_tokens.
    // sample() reads column zero. Speculative consumers select the verification
    // column (tree node, not acceptance depth); stride is in uint32 words, with
    // zero broadcasting one mask. Every reachable column must retain an eligible
    // token. Storage stays live through the complete sampling/accept invocation.
    const std::uint32_t* allowed_token_words = nullptr;
    std::int32_t allowed_token_column_stride = 0;
};

// Caller-owned transient capacity for every parallel sampling-lane count in the inclusive
// interval. For sample(), one lane is one batch row; speculative acceptance uses the same
// workspace substrate for its verification columns. token_domain is the fixed route profile.
// Invalid profiles or intervals throw; a legal single-block route returns zero.
[[nodiscard]] std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                                            std::int32_t min_lanes,
                                                            std::int32_t max_lanes);

/**
 * Produces one token id per independent request row. `logits` is contiguous BF16
 * [physical_rows,B], `out` and `logical_positions` are contiguous I32 [B], and only vocabulary
 * rows v in [0,token_domain) participate. `configs` is a device-resident contiguous
 * SamplingConfig[B] array. Greedy and stochastic rows may coexist in one invocation.
 *
 * For row b with configs[b].temperature<=0:
 *
 *   out[b] = min argmax_v float(logits[v,b]) over v allowed by the optional
 *   eligibility bitset and not listed in the first suppressed_token_count
 *   entries of suppressed_tokens.
 *
 * Penalties, filters, RNG, and token_counts updates are skipped for that row. With positive
 * temperature and configs[b].p_less!=0, let z_v=float(logits[v,b]) over v in [0,token_domain)
 * allowed by the optional eligibility bitset and not listed in the first
 * suppressed_token_count entries of suppressed_tokens
 * (penalties, top_k, top_p, and min_p are ignored). Let p=softmax(z/temperature) over that
 * eligible domain, L=sum_v p_v^2, and V={v: p_v >= L·exp(-2ε/T)} with
 * ε=kPLessLogitPerturbation (non-empty: the eligible mode is always admitted).
 * Equivalently, with e_v=exp((z_v-m)/temperature) and m the eligible max logit, V is
 * {v: e_v * sum_i e_i >= sum_i e_i^2 · exp(-2ε/T)} so the cut is not rounded through
 * L = (sum e^2)/(sum e)^2 and a first-order softmax perturbation of the logits cannot
 * drop a token that exact-math p_v>=L would have kept. L is the unperturbed collision
 * probability; the scale only relaxes membership.
 * If typical_exclude is in V and the remaining admitted mass is strictly positive, sample from
 * the renormalized restriction of p to V without that atom. If V is exactly {typical_exclude},
 * emit the in-domain runner-up (second-max eligible logit, lower id breaking ties). That atom
 * is never added to suppressed_tokens and never rebuilds L. Sample from the renormalized
 * restriction of p to V when typical_exclude is negative or outside V. With positive temperature
 * and p_less==0, let
 * c_v=configs[b].token_counts[v] (or zero when the pointer is null):
 *
 *   adjusted_v = float(logits[v,b])
 *                - configs[b].presence_penalty * (c_v > 0)
 *                - configs[b].frequency_penalty * c_v.
 *
 * Candidates are formed from the same eligible v (suppressed tokens are omitted) and sorted by
 * adjusted_v descending with lower token id breaking ties. Per-row top_k
 * in [1,19] keeps that many candidates; top_k<=0 or top_k>=20 keeps min(20,token_domain).
 * Candidate weights are exp(adjusted_v/temperature-max). min_p removes the suffix below
 * min_p*max_weight; top_p keeps the shortest remaining prefix whose cumulative weight reaches
 * top_p times the pre-truncation candidate weight. At least the best candidate remains, the
 * support is renormalized, and one id is drawn for that row. Greedy, p-less, and truncated
 * rows may coexist in one invocation.
 *
 * Row b uses counter-based RNG key
 * (configs[b].seed,logical_positions[b],purpose), without mutable RNG state or dependence on the
 * compact row index. In the positive-temperature branch the selected token atomically increments
 * configs[b].token_counts when it is non-null. Non-null token-count arrays belonging to distinct
 * active requests must not alias. `out` must not overlap logits, configs, logical_positions, or any
 * token-count array. The Op writes all of out, uses caller-owned transient storage reported by
 * sampling_workspace_capacity_bytes(), and has no other persistent-state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops

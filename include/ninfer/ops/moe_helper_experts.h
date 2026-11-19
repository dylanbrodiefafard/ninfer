#pragma once

#include "ninfer/ops/moe_expert_record.h"
#include "ninfer/ops/moe_helper_contract.h"

#include <hip/hip_runtime_api.h> // hipStream_t

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next second-GPU expert helper (Radeon AI PRO R9700, gfx1201). The helper holds
// routed-expert records the RTX 5090 does not, and for one decode or verify round, or one prefill
// chunk, of one layer returns the weighted sum of its experts' outputs per participating token;
// the 5090 adds the shared expert and its own experts
// (docs/maintainer/qwen3.8-flash-next-expert-helper.md).

// Token capacity of one moe_helper_experts call: a prefill chunk of up to 16384 tokens.
inline constexpr std::int32_t kMoeHelperExpertsMaxTokens = 16384;

/**
 * The route entries of the compact tokens as two planes (the expert-helper wire's layout), entry
 * e = 10 t + j holding route slot j of compact token t:
 *
 *   experts  I32 [max_tokens][10], 4-byte aligned: an expert id in [0,512) or
 *            kMoeHelperInactiveExpert.
 *   weights  FP32 [max_tokens][10], 4-byte aligned: the entry's route weight.
 */
struct MoeHelperRoutes {
    const std::int32_t* experts = nullptr;
    const float* weights        = nullptr;
};

/**
 * One layer's helper-owned experts, resident in the helper's device memory.
 *
 *   expert_table   U64 [512]: for each expert the helper owns, the 256-byte aligned device address
 *                  of its record with its MoeExpertCodec in bit 0. A record holds exactly the bytes
 *                  of the artifact's moe-expert-tiles-v1 record of its codec ("Routed-expert
 *                  record" in ninfer/ops/moe_experts.h). Entries of experts no active route names
 *                  are not read.
 *   matrix_scales  FP32 [512][3]: element 3 e + m is expert e's multiplier of matrix m (0 gate,
 *                  1 up, 2 down).
 *   f3_books       the layer's F3G32 codebooks of gate, up and down (eight E2M1 nibbles each,
 *                  entry v in bits 4v..4v+3), as the artifact's f3_codebooks row holds them;
 *                  passed by value, so a captured graph keeps the books of its capture.
 *
 * Records, table and scales are read on the device when the call executes and must not change
 * until it completes.
 */
struct MoeHelperExpertBank {
    const std::uint64_t* expert_table = nullptr;
    const float* matrix_scales        = nullptr;
    std::array<std::uint32_t, 3> f3_books{};
};

/**
 * Storage type of the helper partial y. Fp32 serves every max_tokens and is the decode and
 * verification return; Bf16 serves only max_tokens > kMoeHelperMaxTokens (48), the prefill
 * return, and halves the bytes sent back over the link.
 */
enum class MoeHelperOutputDtype : std::uint8_t { Fp32, Bf16 };

/**
 * Activation profile of a call: the form of the gate/up input and the arithmetic of the grouped
 * route ("Numeric" at moe_helper_experts). A16 takes BF16 x at every max_tokens. A8 takes x as its
 * E4M3 row quantization (MoeHelperE4m3Rows) and serves only max_tokens > kMoeHelperMaxTokens
 * (48), the prefill extents; it is a private implementation profile of prefill, not a different
 * Op.
 */
enum class MoeHelperActivationProfile : std::uint8_t { A16, A8 };

/**
 * The A8 profile's gate/up input: the represented BF16 x [max_tokens][2560] quantized per token row
 * by the shared A8 activation rule of Flash-Next routed-expert prefill, which the RTX 5090 applies
 * bit for bit the same way:
 *
 *   s_t     = max_k |x[t,k]| / 448, one IEEE FP32 division of the exact FP32 maximum; +0 when the
 *             row is all zeros
 *   q[t,k]  = E4M3FN(x[t,k] / s_t): IEEE FP32 division, then round to nearest even, saturating to
 *             +-448 (no NaN); every code of a zero row is +0 (0x00)
 *
 * so decode(q[t,k]) * s_t approximates x[t,k] with relative error at most 2^-4 (to within the FP32
 * rounding of the quotient) in E4M3's normal range (|x| >= 2^-6 s_t), and absolute error at most
 * 2^-10 s_t below it. The sign of a zero or underflowing element of a nonzero row is kept (0x80
 * for negative).
 *
 *   codes   E4M3FN bytes [max_tokens][2560], 16-byte aligned; rows [0,N) hold q. Every code is
 *           finite (not 0x7F or 0xFF).
 *   scales  FP32 [max_tokens], 4-byte aligned; elements [0,N) hold s_t, finite and >= +0.
 */
struct MoeHelperE4m3Rows {
    const std::uint8_t* codes = nullptr;
    const float* scales       = nullptr;
};

/**
 * Device operands of one moe_helper_experts call, with N = min(*tokens, max_tokens) compact tokens
 * read on the device when the call executes.
 *
 *   tokens      I32 [1]: the compact token count; values below 0 count as 0.
 *   max_tokens  host value in [1, kMoeHelperExpertsMaxTokens]: the extent every buffer below is
 *               sized for.
 *   activation  the profile: A16 at every max_tokens, A8 only at max_tokens > 48.
 *   x           A16: BF16 bits [max_tokens][2560], 16-byte aligned: rows [0,N) are the tokens'
 *               expert input. A8: nullptr.
 *   x_e4m3      A8: the E4M3 rows of the tokens' BF16 expert input (MoeHelperE4m3Rows). A16: both
 *               pointers nullptr.
 *   routes      MoeHelperRoutes: rows [0,N) of both planes are the tokens' route entries in
 *               route-slot order j = 0..9. An active entry names an expert the bank owns; an
 *               inactive entry (kMoeHelperInactiveExpert) may appear anywhere in a row, and its
 *               weight is not read. A token may name an expert more than once (each entry
 *               contributes).
 *   y           output_dtype [max_tokens][2560], 16-byte aligned, disjoint from every other
 *               operand: rows [0,N) receive the output; rows [N,max_tokens) are not written. y
 *               may be host memory mapped into the device (the expert-helper mailbox's response
 *               payload): the call writes it with plain stores, complete when its kernels
 *               complete, so an observer on another agent needs a later system-scope release in
 *               stream order (a fence that writes back the device L2, then the publishing store).
 *   output_dtype y's element type: Fp32 at every max_tokens, Bf16 only at max_tokens > 48.
 */
struct MoeHelperExpertsArgs {
    const std::int32_t* tokens            = nullptr;
    std::int32_t max_tokens               = 0;
    MoeHelperActivationProfile activation = MoeHelperActivationProfile::A16;
    const std::uint16_t* x                = nullptr;
    MoeHelperE4m3Rows x_e4m3;
    MoeHelperRoutes routes;
    MoeHelperExpertBank bank;
    void* y                           = nullptr;
    MoeHelperOutputDtype output_dtype = MoeHelperOutputDtype::Fp32;
};

/**
 * Device workspace of moe_helper_experts for every call with max_tokens <= this max_tokens:
 * work-item and completion counters, route grouping, the gate/up products, the A8 profile's
 * quantized down input and the per-route contributions; device_bytes(max_tokens) bytes, about
 * 119 KiB per token above max_tokens 48 (481 MiB at 4096; the per-route FP32 contributions are
 * 100 KiB of it). Allocated on the device
 * current at construction, with the counters zeroed; every completed call leaves them zero again.
 * Calls sharing a workspace must be ordered (one stream, or explicit dependencies): it serves one
 * call at a time. A call that does not complete (an aborted launch or graph, a device fault) may
 * leave the counters of calls with max_tokens <= 48 in an arbitrary state, and such a call would
 * then misassign work or wait forever: after any failed round, its owner calls reset() before
 * the next call. Calls with max_tokens > 48 reset their own counters.
 */
class MoeHelperExpertsWorkspace {
public:
    /** Allocates and zeroes the workspace; max_tokens in [1, kMoeHelperExpertsMaxTokens]. */
    explicit MoeHelperExpertsWorkspace(std::int32_t max_tokens);

    /**
     * Device bytes a workspace for max_tokens allocates; throws std::invalid_argument outside
     * [1, kMoeHelperExpertsMaxTokens].
     */
    [[nodiscard]] static std::size_t device_bytes(std::int32_t max_tokens);
    ~MoeHelperExpertsWorkspace();

    MoeHelperExpertsWorkspace(const MoeHelperExpertsWorkspace&)            = delete;
    MoeHelperExpertsWorkspace& operator=(const MoeHelperExpertsWorkspace&) = delete;

    [[nodiscard]] std::int32_t max_tokens() const noexcept { return max_tokens_; }

    [[nodiscard]] int device() const noexcept { return device_; }

    [[nodiscard]] std::byte* base() const noexcept { return base_; }

    /**
     * Zeroes the counters on `stream` (asynchronously), restoring the state of a fresh workspace.
     * No call using this workspace may be in flight or queued ahead of it on another stream; the
     * next call must follow it in stream order. Throws std::runtime_error when the memset cannot
     * be queued.
     */
    void reset(hipStream_t stream);

private:
    std::byte* base_         = nullptr;
    std::int32_t max_tokens_ = 0;
    int device_              = 0;
};

/**
 * Op: moe_helper_experts
 *
 * Math, for each compact token t < N with x_t = x[t,:] and its active route entries
 * (e_j, w_j) = (routes.experts[t,j], routes.weights[t,j]):
 *
 *   E_e(x)  = D_e (SiLU(G_e x) * U_e x)          (SiLU(z) = z / (1 + exp(-z)), * elementwise)
 *   y[t,:]  = sum_j w_j E_{e_j}(x_t)
 *
 * where G_e, U_e, D_e are expert e's record decoded as "Routed-expert record" in
 * ninfer/ops/moe_experts.h specifies (code through the layer's F3 book or as E2M1, times its E4M3
 * block scale), each multiplied by the expert's FP32 matrix scale. A token with no active entry
 * gets y[t,:] = 0 exactly. There is no shared expert.
 *
 * Numeric: the oracle evaluates the formula in FP64 from the represented BF16 x (for A8, the BF16
 * x the caller quantized into x_e4m3), the decoded weights and the FP32 scales and route weights;
 * it does not quantize.
 *
 * Implementation profile A16: products of decoded weights and BF16 x are exact and summed in
 * FP32, with BF16 WMMA (FP32 accumulation) or, for experts with few tokens in calls with
 * max_tokens <= 48, v_dot2. Calls with max_tokens <= 48 sum each scale group and scale it by its
 * E4M3 scale in FP32; larger calls decode each weight as code times E4M3 scale, which is exact in
 * BF16, and sum the full K. Matrix scales, SiLU and the gate/up product are FP32, and the product
 * enters the down projection rounded once to BF16 (the RTX 5090 route's profile, not a semantic
 * boundary).
 *
 * Implementation profile A8 (the shared A8 activation profile of Flash-Next routed-expert
 * prefill, which the RTX 5090's prefill route applies to the same activations): the gate/up
 * products take x_e4m3's codes. Each weight code expands exactly to E4M3 (E2M1 values are E4M3
 * values), FP8 WMMA sums each scale group (16 K for NVFP4, 32 K for F3) exactly in FP32, and that
 * group partial times (E4M3 block scale * FP32 matrix scale) is accumulated in FP32; the token's
 * row scale multiplies the sum. SiLU and the gate/up product are FP32; the product is rounded once
 * to BF16 and then quantized per (token, expert) row of 640 by the rule of MoeHelperE4m3Rows, and
 * the down projection takes those codes and row scale the same way. Relative to the oracle, the
 * two E4M3 roundings (relative error at most 2^-4 per element) dominate: about 4e-2 relative L2.
 *
 * In both profiles each route's contribution is FP32, and a token's contributions are summed in
 * route-slot order j = 0..9. An Fp32 y is that sum; a Bf16 y is its explicit output cast, one
 * round-to-nearest-even conversion to BF16, so it equals the BF16 rounding of the Fp32 y of the
 * same call bit for bit. No float atomics: y is bit-identical for identical inputs, routes,
 * max_tokens, profile and output dtype; at max_tokens <= 48 the path an expert takes depends on
 * how many tokens name it, and above it a token's row does not depend on the other tokens.
 *
 * Execution: asynchronous kernel launches on `stream`, no host synchronization. N, routes,
 * tables, scales and records are read on the device at execution, so a captured HIP graph stays
 * valid while their contents change; its shape depends only on max_tokens. The workspace must
 * belong to the current device and have max_tokens() >= args.max_tokens. Invalid extents,
 * profile, input form, output dtype, alignment, aliasing or workspace throw std::invalid_argument
 * before any work is queued; a launch failure throws std::runtime_error.
 */
void moe_helper_experts(const MoeHelperExpertsArgs& args, MoeHelperExpertsWorkspace& workspace,
                        hipStream_t stream);

} // namespace ninfer::ops

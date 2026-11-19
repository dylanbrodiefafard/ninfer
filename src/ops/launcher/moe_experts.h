#pragma once

#include "ninfer/ops/moe_experts.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace ninfer::ops::detail {

// Expert-token assignments per token: the ten routed slots, then the shared expert as slot 10.
inline constexpr std::int32_t kMoeSlots        = kMoeTopK + 1;
inline constexpr std::int32_t kMoeSharedExpert = kMoeExperts;

// Tokens one job (one expert's group of assignments) holds at most: the MMA column tile. A CTA
// multiplies only the eight-token column blocks its job fills, so a wide tile costs no MMAs; it
// sets the shared-memory staging and how many jobs a popular expert splits into. An expert with
// more tokens than the tile has several jobs, and each reads its record (over PCIe when it is
// host-resident); since a token's ten ids are distinct an expert holds at most T tokens, so
// every A16 expert has a single job up to T = 16, the decode and verification range. The A8
// profile serves prefill and has only the 64-token tile.
[[nodiscard]] constexpr std::int32_t moe_experts_token_tile(std::int32_t tokens,
                                                            MoeActivationProfile profile) {
    if (profile == MoeActivationProfile::A8) { return 64; }
    if (tokens <= 256) { return 16; }
    if (tokens <= 1024) { return 32; }
    return 64;
}

// Upper bound of the job count for T tokens and token tile C: sum over experts of ceil(n_e / C),
// with at most min(512, 10T) routed experts holding the 10T routed assignments, plus ceil(T / C)
// shared jobs. Every term stays below 2^31 for T <= 16384.
[[nodiscard]] constexpr std::int32_t moe_experts_max_jobs(std::int32_t tokens, std::int32_t tile) {
    const std::int32_t experts = std::min(kMoeExperts, kMoeTopK * tokens);
    return (kMoeTopK * tokens + experts * (tile - 1)) / tile + (tokens + tile - 1) / tile;
}

// One job: up to C assignments of one expert, contiguous in the sorted assignment list.
struct alignas(16) MoeJob {
    std::int32_t expert; // [0,512) routed, 512 shared
    std::int32_t begin;  // first sorted position
    std::int32_t count;  // assignments, in [1,C]
};

// Matmul launches of one call, in stream order: gate|up and down over the jobs before the deferred
// ones (every job without a deferral), then gate|up and down over the deferred jobs.
inline constexpr std::int32_t kMoeMatmulLaunches = 4;

// Device-side schedule of one call, written by the grouping kernel before any matmul launch:
//   job_count  jobs before the deferred ones, and all jobs (equal without a deferral)
//   next_item  per matmul launch, the count of work-item claims issued so far: claim w takes item
//              w while w is below the launch's item count (zeroed by the grouping kernel; see
//              moe_expert_matmul_kernel)
struct MoeJobSchedule {
    std::int32_t job_count[2];
    std::int32_t next_item[kMoeMatmulLaunches];
};

// Workspace of one call, for T tokens and token tile C:
//   jobs           [moe_experts_max_jobs(T, C)]: the shared expert's jobs, then the routed jobs
//                  read before the deferral wait, then the deferred experts' jobs
//   schedule       the job counts and work-item counters (MoeJobSchedule)
//   sorted         I32 [11T]: assignment ids a = 11 t + slot, the shared expert's T assignments
//                  first in token order, then the routed ones grouped by ascending expert and in
//                  ascending a within an expert
//   gated          BF16 [11T][640]: SiLU(G x) * U x of the assignment at each sorted position
//   contributions  FP32 [11T][2560]: weight * E(x) of assignment a at row a
// and for the A8 profile only (null for A16), the moe_activation_quantize encodings of
//   x_codes, x_scales          E4M3 [T][2560] and FP32 [T]: x, one row per token
//   gated_codes, gated_scales  E4M3 [11T][640] and FP32 [11T]: gated, one row per sorted position
//                              (written for the positions the call's jobs cover)
struct MoeExpertsWorkspace {
    MoeJob* jobs              = nullptr;
    MoeJobSchedule* schedule  = nullptr;
    std::int32_t* sorted      = nullptr;
    std::uint16_t* gated      = nullptr;
    float* contributions      = nullptr;
    std::uint8_t* x_codes     = nullptr;
    float* x_scales           = nullptr;
    std::uint8_t* gated_codes = nullptr;
    float* gated_scales       = nullptr;
};

// Raw device operands of one call. The launcher trusts the wrapper's validation: shapes, dtypes,
// 16-byte alignment, 1 <= tokens <= 16384, and disjointness.
struct MoeExpertsOperands {
    MoeActivationProfile profile          = MoeActivationProfile::A16;
    const std::uint16_t* x                = nullptr;
    const std::int32_t* expert_ids        = nullptr;
    const float* route_weights            = nullptr;
    const float* shared_scale             = nullptr;
    const std::uint64_t* expert_table     = nullptr;
    const std::uint64_t* deferred_slots   = nullptr; // null: no deferral
    cudaEvent_t deferred_ready            = nullptr;
    const float* matrix_scales            = nullptr;
    std::array<std::uint32_t, 3> f3_books = {};
    const std::uint8_t* shared_record     = nullptr;
    const float* shared_matrix_scales     = nullptr;
    // Local-partial call: the U32 [16] local-expert bitset (assignments of other experts are
    // excluded) and the FP32 [2560,T] output; both null for an ordinary call, which writes y.
    const std::uint32_t* local_experts = nullptr;
    float* partial                     = nullptr;
    std::uint16_t* y                   = nullptr;
    std::int32_t tokens                = 0;
};

void moe_experts_launch(const MoeExpertsOperands& operands, const MoeExpertsWorkspace& workspace,
                        cudaStream_t stream);

// moe_activation_quantize of x BF16 [tokens][2560] into codes E4M3 [tokens][2560] and scales FP32
// [tokens]; 1 <= tokens <= 16384, buffers 16-byte aligned (validated by the wrapper).
void moe_activation_quantize_launch(const std::uint16_t* x, std::uint8_t* codes, float* scales,
                                    std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
